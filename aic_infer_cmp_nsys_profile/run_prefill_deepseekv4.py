#!/usr/bin/env python3
"""Replay one DeepSeek-V4-Pro prefill batch from hooked SGLang JSONL data."""

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, fields
from typing import Any, Dict, List, Optional, Tuple


def _find_data_files(data_dir: str, tp_rank: int) -> Tuple[str, str]:
    prefix = None
    for fname in os.listdir(data_dir):
        if fname.startswith(f"TP{tp_rank}") and fname.endswith("_schedule_batch.jsonl"):
            prefix = fname.replace("_schedule_batch.jsonl", "")
            break
    if prefix is None:
        raise FileNotFoundError(f"No TP{tp_rank}_schedule_batch.jsonl in {data_dir}")
    return (
        os.path.join(data_dir, f"{prefix}_schedule_batch.jsonl"),
        os.path.join(data_dir, f"{prefix}.request.jsonl"),
    )


def _load_jsonl_record(schedule_jsonl: str, zero_based_index: int) -> Dict[str, Any]:
    with open(schedule_jsonl, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx == zero_based_index:
                rec = json.loads(line)
                rec["_jsonl_index"] = idx
                rec["_schedule_line"] = idx + 1
                return rec
    raise FileNotFoundError(f"JSONL index {zero_based_index} not found: {schedule_jsonl}")


def _load_request_ids(request_jsonl: str) -> Dict[str, Dict[str, List[int]]]:
    out = {}
    with open(request_jsonl, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            out[rec["rid"]] = {
                "input_ids": rec["input_ids"],
                "output_ids": rec.get("output_ids", []),
            }
    return out


def _target_index(args: argparse.Namespace) -> int:
    if args.schedule_line is not None:
        if args.schedule_line <= 0:
            raise ValueError("--schedule-line is 1-based and must be positive")
        return args.schedule_line - 1
    return args.schedule_index


def _normalize_request_infos(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    infos = []
    for req in record["request_infos"]:
        infos.append(
            {
                "rid": req["rid"],
                "input_length": int(float(req["extend_input_len"])),
                "past_kv_length": int(float(req["prefix_indices_len"])),
                "output_ids_len": int(float(req.get("output_ids_len", 0))),
            }
        )
    return infos


def _build_prompts(
    record: Dict[str, Any],
    request_map: Dict[str, Dict[str, List[int]]],
) -> Tuple[List[List[int]], List[List[int]], List[Dict[str, Any]]]:
    prefix_prompts = []
    full_prompts = []
    normalized = _normalize_request_infos(record)
    missing = []

    for info in normalized:
        rid = info["rid"]
        extend_len = info["input_length"]
        prefix_len = info["past_kv_length"]
        if rid not in request_map:
            missing.append(rid)
            continue

        input_ids = request_map[rid]["input_ids"]
        full_len = prefix_len + extend_len
        if len(input_ids) < full_len:
            raise ValueError(
                f"rid={rid} needs {full_len} input ids, only found {len(input_ids)}"
            )
        if prefix_len > 0:
            prefix_prompts.append(input_ids[:prefix_len])
        full_prompts.append(input_ids[:full_len])

    if missing:
        raise KeyError(f"{len(missing)} rids missing from request JSONL, first={missing[0]}")
    if not full_prompts:
        raise ValueError("No replay prompts built")

    return prefix_prompts, full_prompts, normalized


def _install_scheduler_nvtx_hook() -> None:
    import torch

    hook_dirs = [
        "/home/scratch.kimiz_gpu_2/docker_v/agent_work_space/b300_glm5_runtime/ds4_new/aic_fpm/hook_dataset_collector",
        "/raid/kimi/ds4_new/aic_fpm/hook_dataset_collector",
        os.path.join(os.path.dirname(__file__), "hook_dataset_collector"),
    ]
    for hook_dir in hook_dirs:
        if os.path.exists(os.path.join(hook_dir, "hook.py")):
            sys.path.insert(0, hook_dir)
            break
    else:
        raise FileNotFoundError("Cannot find hook_dataset_collector/hook.py")

    import hook

    class C_SglangSchedulerRunBatchAnnotationHook(hook.BaseHook):
        HOOK_CLASS_NAME = "Scheduler"
        HOOK_MODULE_NAME = "sglang.srt.managers.scheduler"

        @classmethod
        def hook(cls, target_class):
            original_run_batch = target_class.run_batch

            def wrapped_run_batch(self, batch, *args, **kwargs):
                if batch is None:
                    return original_run_batch(self, batch, *args, **kwargs)

                bs = batch.batch_size()
                seq_lens = (
                    batch.seq_lens.tolist()
                    if hasattr(batch.seq_lens, "tolist")
                    else list(batch.seq_lens)
                )
                if batch.forward_mode.is_decode_or_idle():
                    input_lengths = [1] * bs
                    past_kv_lengths = [s - 1 for s in seq_lens]
                else:
                    ext = batch.extend_num_tokens
                    if isinstance(ext, int):
                        input_lengths = [ext]
                    elif hasattr(ext, "tolist"):
                        input_lengths = ext.tolist()
                    else:
                        input_lengths = list(ext)
                    past_kv_lengths = [
                        s - input_len for s, input_len in zip(seq_lens, input_lengths)
                    ]

                extra_args = {
                    "bs": bs,
                    "forward_mode": batch.forward_mode.name,
                }
                if bs <= 8:
                    extra_args["input_length"] = input_lengths
                    extra_args["past_kv_length"] = past_kv_lengths
                else:
                    extra_args["total_input_tokens"] = sum(input_lengths)
                    extra_args["past_kv_min"] = min(past_kv_lengths)
                    extra_args["past_kv_max"] = max(past_kv_lengths)
                    extra_args["past_kv_avg"] = sum(past_kv_lengths) // bs

                torch.cuda.synchronize()
                start = time.perf_counter()
                torch.cuda.nvtx.range_push(f"Scheduler.run_batch: {extra_args}")
                out = original_run_batch(self, batch, *args, **kwargs)
                torch.cuda.synchronize()
                elapsed_ms = (time.perf_counter() - start) * 1000
                torch.cuda.nvtx.range_pop()
                print(
                    f"[replay_run_batch] {extra_args} latency_ms={elapsed_ms:.3f}",
                    flush=True,
                )
                return out

            target_class.run_batch = wrapped_run_batch
            return target_class

    hook.install_class_hooks([C_SglangSchedulerRunBatchAnnotationHook])


def _server_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    from sglang.srt.server_args import ServerArgs

    kwargs = {
        "model_path": args.model,
        "tokenizer_path": args.model,
        "tp_size": args.tp_size,
        "ep_size": args.ep_size,
        "load_format": "auto",
        "trust_remote_code": True,
        "disable_cuda_graph": True,
        "cuda_graph_max_bs": 128,
        "disable_cuda_graph_padding": True,
        "chunked_prefill_size": 8192,
        "max_running_requests": 256,
        "max_prefill_tokens": 16384,
        "mem_fraction_static": 0.93,
        "attention_backend": "compressed",
        "moe_runner_backend": "flashinfer_mxfp4",
        "speculative_moe_runner_backend": "flashinfer_mxfp4",
        "kv_cache_dtype": "fp8_e4m3",
        "tool_call_parser": "deepseekv4",
        "reasoning_parser": "deepseek-v4",
        "disable_flashinfer_autotune": True,
        "allow_auto_truncate": True,
        "disable_piecewise_cuda_graph": True,
        "disable_overlap_schedule": False,
        "enable_mixed_chunk": False,
        "skip_server_warmup": True,
    }
    if args.context_length is not None:
        kwargs["context_length"] = args.context_length
    if args.disable_custom_all_reduce:
        kwargs["disable_custom_all_reduce"] = True
    if args.enforce_disable_flashinfer_allreduce_fusion:
        kwargs["enforce_disable_flashinfer_allreduce_fusion"] = True

    valid = {field.name for field in fields(ServerArgs)}
    return {k: v for k, v in kwargs.items() if k in valid}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--tp-rank", type=int, default=0)
    parser.add_argument("--schedule-index", type=int, default=804)
    parser.add_argument("--schedule-line", type=int)
    parser.add_argument("--model", default="/DeepSeek-V4-Pro/")
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--ep-size", type=int, default=1)
    parser.add_argument("--context-length", type=int)
    parser.add_argument("--profile-max-new-tokens", type=int, default=0)
    parser.add_argument("--print-only", action="store_true")
    parser.add_argument("--disable-custom-all-reduce", action="store_true")
    parser.add_argument(
        "--enforce-disable-flashinfer-allreduce-fusion",
        action="store_true",
    )
    args = parser.parse_args()

    schedule_file, request_file = _find_data_files(args.data_dir, args.tp_rank)
    record = _load_jsonl_record(schedule_file, _target_index(args))
    if int(record["forward_mode"]) != 1:
        raise ValueError(f"Target is not prefill: forward_mode={record['forward_mode']}")

    request_map = _load_request_ids(request_file)
    prefix_prompts, full_prompts, request_infos = _build_prompts(record, request_map)

    target_latency_ms = float(record.get("iter_latency", 0.0)) * 1000
    print(f"[data] schedule={schedule_file}")
    print(f"[data] request={request_file}")
    print(
        f"[target] schedule_line={record['_schedule_line']} "
        f"jsonl_index={record['_jsonl_index']} latency_ms={target_latency_ms:.3f}"
    )
    print(f"[target] request_infos={request_infos}")
    print(f"[target] prefix_prompt_lens={[len(p) for p in prefix_prompts]}")
    print(f"[target] full_prompt_lens={[len(p) for p in full_prompts]}")

    if args.print_only:
        return

    _install_scheduler_nvtx_hook()

    from sglang.srt.entrypoints.engine import Engine
    from sglang.srt.server_args import ServerArgs
    import torch

    kwargs = _server_kwargs(args)
    print("[server_args] input:")
    for key in sorted(kwargs):
        print(f"  {key}={kwargs[key]!r}")

    server_args = ServerArgs(**kwargs)
    normalized = asdict(server_args)
    print("[server_args] normalized selected:")
    for key in (
        "model_path",
        "tokenizer_path",
        "context_length",
        "tp_size",
        "ep_size",
        "disable_cuda_graph",
        "cuda_graph_max_bs",
        "disable_cuda_graph_padding",
        "chunked_prefill_size",
        "max_running_requests",
        "max_prefill_tokens",
        "mem_fraction_static",
        "attention_backend",
        "moe_runner_backend",
        "kv_cache_dtype",
        "disable_piecewise_cuda_graph",
        "disable_overlap_schedule",
    ):
        if key in normalized:
            print(f"  {key}={normalized[key]!r}")

    llm = Engine(**normalized)
    try:
        if prefix_prompts:
            print("[warmup] priming prefix cache")
            llm.generate(
                input_ids=prefix_prompts,
                sampling_params={"temperature": 0, "top_p": 1, "max_new_tokens": 0},
            )
            print("[warmup] done")

        print("[profile] cudaProfilerStart")
        torch.cuda.cudart().cudaProfilerStart()
        outputs = llm.generate(
            input_ids=full_prompts,
            sampling_params={
                "temperature": 0,
                "top_p": 1,
                "max_new_tokens": args.profile_max_new_tokens,
            },
        )
        torch.cuda.cudart().cudaProfilerStop()
        print("[profile] cudaProfilerStop")
        print(f"[done] outputs={len(outputs)}")
    finally:
        try:
            llm.shutdown()
        except Exception as exc:
            print(f"[warn] llm.shutdown failed: {exc}", flush=True)


if __name__ == "__main__":
    main()
