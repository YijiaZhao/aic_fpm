from typing import Optional, Any, List
from types import FunctionType
import logging
import torch
import time
import os
import importlib
import json
import builtins
import re


logger = logging.getLogger(__name__)


class BaseHook:
    HOOK_CLASS_NAME: Optional[str] = None
    HOOK_MODULE_NAME: Optional[str] = None
    REGEX: bool = False

    def __init__(self):
        pass

    @classmethod
    def hook(cls, target) -> Any:
        """
        Return a new target or simply modify the target reference.
        """
        raise NotImplementedError


_builtins_build_class_ = builtins.__build_class__


CLASS_HOOKS = []


def _custom_build_class_(func, name: str, *bases, **kwargs):
    for hook in CLASS_HOOKS:
        if (
            hook.REGEX and re.search(hook.HOOK_CLASS_NAME, name)
        ) or name == hook.HOOK_CLASS_NAME:
            module_name = None
            if isinstance(func, FunctionType):
                module_name = getattr(func, "__globals__", {}).get("__name__", "")
            if (
                hook.REGEX and re.search(hook.HOOK_MODULE_NAME, module_name)
            ) or module_name == hook.HOOK_MODULE_NAME:
                logger.debug(
                    f"Hooking Class: {hook.__name__} into {module_name}|{name}"
                    + (
                        "(Regex is enabled, which might cause unexpected behavior.)"
                        if hook.REGEX
                        else ""
                    )
                )
                target_class = _builtins_build_class_(func, name, *bases, **kwargs)
                try:
                    new_class = hook.hook(target_class)
                except Exception as e:
                    new_class = None
                    logger.warning(f"Fail to hook class[{name}], error: {e}")
                if new_class is not None:
                    return new_class
                else:
                    return target_class

    return _builtins_build_class_(func, name, *bases, **kwargs)


def install_class_hooks(hooks: List[BaseHook]):
    global CLASS_HOOKS
    CLASS_HOOKS = hooks
    builtins.__build_class__ = _custom_build_class_


class C_SglangSchedulerHook(BaseHook):
    HOOK_CLASS_NAME = "Scheduler"
    HOOK_MODULE_NAME = "sglang.srt.managers.scheduler"

    SCHEDULE_INFOS = []
    REQUEST_INFOS = []

    @classmethod
    def hook(cls, target_class):
        original_run_batch = target_class.run_batch
        original_process_batch_result = target_class.process_batch_result

        def wrapped_run_batch(self, batch):
            torch.cuda.synchronize()
            start = time.time()
            result = original_run_batch(self, batch)
            torch.cuda.synchronize()  # synchronize
            end = time.time()

            if batch is not None:
                request_infos = []
                for req in batch.reqs:
                    request_infos.append(
                        {
                            "rid": req.rid,
                            "extend_input_len": req.extend_input_len,
                            "prefix_indices_len": len(req.prefix_indices),
                            "output_ids_len": len(req.output_ids),
                        }
                    )

                C_SglangSchedulerHook.SCHEDULE_INFOS.append(
                    {
                        "start_timestamp": start,
                        "end_timestamp": end,
                        "forward_mode": int(batch.forward_mode),
                        "request_infos": request_infos,
                        "iter_latency": end - start,
                    }
                )

            return result

        def wrapped_process_batch_result(self, batch, result):
            ret = original_process_batch_result(self, batch, result)
            if batch.reqs is None:
                # dummy first batch while overlap schedule is enable.
                return ret

            for req in batch.reqs:
                if req.finished():
                    C_SglangSchedulerHook.REQUEST_INFOS.append(
                        {
                            "rid": req.rid,
                            "input_ids": req.origin_input_ids,
                            "output_ids": req.output_ids,
                        }
                    )

        def wrapped_profile(self, *args, **kwargs):
            SGL_HOOK_REQ_INFO_DIR = os.getenv(
                "SGLANG_HOOK_REQ_INFO_DIR",
                os.getenv("SGL_HOOK_REQ_INFO_DIR", os.getcwd()),
            )

            parallel_state = getattr(self, "ps", None)

            def get_parallel_attr(name: str, default: int = 0):
                if hasattr(self, name):
                    return getattr(self, name)
                return getattr(parallel_state, name, default)

            filename_prefix = f"TP{get_parallel_attr('tp_rank')}"

            if get_parallel_attr("dp_size", 1) > 1:
                filename_prefix += f"-DP{get_parallel_attr('dp_rank')}"
            if get_parallel_attr("pp_size", 1) > 1:
                filename_prefix += f"-PP{get_parallel_attr('pp_rank')}"
            if get_parallel_attr("moe_ep_size", 1) > 1:
                filename_prefix += f"-EP{get_parallel_attr('moe_ep_rank')}"

            os.makedirs(SGL_HOOK_REQ_INFO_DIR, exist_ok=True)
            with open(
                f"{SGL_HOOK_REQ_INFO_DIR}/{filename_prefix}_schedule_batch.jsonl", "w"
            ) as f:
                for batch_infos in C_SglangSchedulerHook.SCHEDULE_INFOS:
                    f.write(json.dumps(batch_infos) + "\n")

            with open(
                f"{SGL_HOOK_REQ_INFO_DIR}/{filename_prefix}.request.jsonl", "w"
            ) as f:
                for req in C_SglangSchedulerHook.REQUEST_INFOS:
                    f.write(json.dumps(req) + "\n")

            C_SglangSchedulerHook.SCHEDULE_INFOS.clear()
            C_SglangSchedulerHook.REQUEST_INFOS.clear()

            ProfileReqOutput = getattr(
                importlib.import_module("sglang.srt.managers.io_struct"),
                "ProfileReqOutput",
            )
            return ProfileReqOutput(True, "Success")

        target_class.run_batch = wrapped_run_batch
        target_class.process_batch_result = wrapped_process_batch_result
        target_class.profile = wrapped_profile
