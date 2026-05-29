#!/usr/bin/env python3
"""
GLM-5 decode replay tool. Real tokens, real weights.

This is the decode-only counterpart of run_prefill_glm5.py. It follows the
GLM-5 launch configuration from refactor_test_aic/config.py, keeps regular CUDA
graph enabled, and explicitly disables piecewise CUDA graph.
"""

import argparse
import csv
import json
import os
import sys
from dataclasses import asdict, fields

csv.field_size_limit(sys.maxsize)

import torch

os.environ.setdefault("SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK", "0")

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "hook_dataset_collector")
)
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
                past_kv_lengths = [s - il for s, il in zip(seq_lens, input_lengths)]

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
            torch.cuda.nvtx.range_push(f"Scheduler.run_batch: {extra_args}")
            out = original_run_batch(self, batch, *args, **kwargs)
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_pop()
            return out

        target_class.run_batch = wrapped_run_batch
        return target_class


hook.install_class_hooks([C_SglangSchedulerRunBatchAnnotationHook])


from run_prefill import (
    _find_csv_path,
    _find_data_files,
    _load_csv_row_by_case_id,
    _load_jsonl_record,
    _load_request_ids,
    _resolve_jsonl_case_id_from_csv,
)


def _set_if_supported(server_kwargs, server_arg_names, key, value):
    if key in server_arg_names:
        server_kwargs[key] = value


def _round_up(value, multiple):
    return ((value + multiple - 1) // multiple) * multiple


def _build_glm5_server_args(args, batch_size):
    from sglang.srt.server_args import ServerArgs

    server_arg_names = {field.name for field in fields(ServerArgs)}
    server_kwargs = dict(
        model_path=args.model,
        tp_size=args.tp_size,
        ep_size=args.ep_size,
        load_format="auto",
        trust_remote_code=True,
        disable_overlap_schedule=True,
        attention_backend="nsa",
        chunked_prefill_size=16384,
        disable_flashinfer_autotune=True,
        enable_cache_report=True,
        enable_dp_lm_head=True,
        kv_cache_dtype="fp8_e4m3",
        max_prefill_tokens=16384,
        cuda_graph_max_bs=256,
        disable_cuda_graph=False,
        disable_cuda_graph_padding=True,
        max_running_requests=256,
        mem_fraction_static=0.8,
        moe_dense_tp_size=1,
        quantization="modelopt_fp4",
        sampling_backend="pytorch",
        watchdog_timeout=1000000,
    )

    if args.cuda_graph_bs_list:
        graph_bs = [
            int(x.strip()) for x in args.cuda_graph_bs_list.split(",") if x.strip()
        ]
        if not graph_bs:
            raise ValueError("--cuda-graph-bs-list did not contain any batch sizes")
        server_kwargs["cuda_graph_bs"] = sorted(set(graph_bs))
    elif args.force_case_cuda_graph_bs:
        # GLM-5 decode uses TP gather in this SGLang build. CUDA graph capture
        # filters out batch sizes whose token count is not aligned to TP=8, so
        # bs=1 must replay through the single supported graph key 8.
        server_kwargs["cuda_graph_bs"] = [_round_up(batch_size, args.tp_size)]

    _set_if_supported(
        server_kwargs, server_arg_names, "disable_piecewise_cuda_graph", True
    )
    _set_if_supported(
        server_kwargs, server_arg_names, "enable_piecewise_cuda_graph", False
    )
    _set_if_supported(
        server_kwargs, server_arg_names, "enforce_piecewise_cuda_graph", False
    )
    _set_if_supported(
        server_kwargs, server_arg_names, "moe_runner_backend", "flashinfer_trtllm"
    )
    _set_if_supported(
        server_kwargs,
        server_arg_names,
        "speculative_moe_runner_backend",
        "flashinfer_trtllm",
    )
    _set_if_supported(
        server_kwargs, server_arg_names, "nsa_prefill_backend", "trtllm"
    )
    _set_if_supported(
        server_kwargs, server_arg_names, "nsa_decode_backend", "trtllm"
    )

    if args.disable_flashinfer_allreduce_fusion:
        _set_if_supported(
            server_kwargs,
            server_arg_names,
            "enforce_disable_flashinfer_allreduce_fusion",
            True,
        )
    elif "enable_flashinfer_allreduce_fusion" in server_arg_names:
        server_kwargs["enable_flashinfer_allreduce_fusion"] = True

    _set_if_supported(
        server_kwargs,
        server_arg_names,
        "disable_custom_all_reduce",
        args.disable_custom_all_reduce,
    )
    _set_if_supported(
        server_kwargs, server_arg_names, "disable_shared_experts_fusion", True
    )

    print(
        "[server_args] selected GLM5 decode config:",
        {
            k: server_kwargs.get(k)
            for k in (
                "attention_backend",
                "nsa_decode_backend",
                "moe_runner_backend",
                "enable_dp_lm_head",
                "enable_flashinfer_allreduce_fusion",
                "enforce_disable_flashinfer_allreduce_fusion",
                "disable_custom_all_reduce",
                "disable_shared_experts_fusion",
                "disable_cuda_graph",
                "cuda_graph_bs",
                "cuda_graph_max_bs",
                "disable_cuda_graph_padding",
                "enable_piecewise_cuda_graph",
                "disable_piecewise_cuda_graph",
                "enforce_piecewise_cuda_graph",
            )
            if k in server_kwargs
        },
    )
    return ServerArgs(**server_kwargs)


def main():
    ap = argparse.ArgumentParser(
        description="GLM-5 decode replay: real tokens, regular CUDA graph only"
    )
    ap.add_argument("--data-dir", type=str, required=True)
    ap.add_argument("--tp-rank", type=int, default=0)
    ap.add_argument("--csv-case-id", type=int, required=True)
    ap.add_argument("--csv-path", type=str, default="")
    ap.add_argument(
        "--model",
        type=str,
        default="/raid/kimi/ds4_new/model_configs/nvidia--GLM-5-NVFP4",
    )
    ap.add_argument("--tp-size", type=int, default=8)
    ap.add_argument("--ep-size", type=int, default=1)
    ap.add_argument(
        "--iters", type=int, default=3, help="Number of profiled generate calls."
    )
    ap.add_argument(
        "--profile-max-new-tokens",
        type=int,
        default=4,
        help="Generated tokens per profiled request. Use >=2 to capture decode.",
    )
    ap.add_argument(
        "--cuda-graph-bs-list",
        type=str,
        default="",
        help="Optional comma-separated CUDA graph batch sizes.",
    )
    ap.add_argument(
        "--force-case-cuda-graph-bs",
        action="store_true",
        help="Capture only the target case batch size in regular CUDA graph.",
    )
    ap.add_argument(
        "--disable-flashinfer-allreduce-fusion",
        action="store_true",
        help="Disable FlashInfer TRTLLM allreduce fusion for profiling experiments.",
    )
    ap.add_argument(
        "--disable-custom-all-reduce",
        action="store_true",
        help="Disable SGLang custom allreduce for profiling experiments.",
    )
    ap.add_argument(
        "--torch-profile-dir",
        type=str,
        default="",
        help="If set, use SGLang torch profiler and write chrome traces here.",
    )
    args = ap.parse_args()

    schedule_file, request_file = _find_data_files(args.data_dir, args.tp_rank)
    csv_path = _find_csv_path(args.data_dir, args.csv_path)
    print(f"[data] schedule: {schedule_file}")
    print(f"[data] request: {request_file}")
    print(f"[data] csv: {csv_path}")

    _load_csv_row_by_case_id(csv_path, args.csv_case_id)
    jsonl_line = _resolve_jsonl_case_id_from_csv(
        schedule_file, csv_path, args.csv_case_id
    )
    print(f"[mapping] csv_case_id={args.csv_case_id} -> jsonl line {jsonl_line}")

    record = _load_jsonl_record(schedule_file, jsonl_line)
    fm = int(record["forward_mode"])
    if fm != 2:
        raise ValueError(f"Not a decode batch: forward_mode={fm}")

    batch_size = len(record["request_infos"])
    target_latency = record.get("iter_latency", 0) * 1000
    print(f"[target] bs={batch_size}, latency={target_latency:.3f}ms")

    rid_map = _load_request_ids(request_file)
    print(f"[data] {len(rid_map)} requests loaded")

    warmup_prompts = []
    profile_prompts = []
    missing = []
    request_shapes = []
    for i, req_info in enumerate(record["request_infos"]):
        rid = req_info["rid"]
        output_ids_len = int(req_info["output_ids_len"])
        if rid not in rid_map:
            missing.append(rid)
            continue
        req_data = rid_map[rid]
        full = req_data["input_ids"] + req_data["output_ids"][:output_ids_len]
        if len(full) < 2:
            raise ValueError(f"Request {rid} is too short for decode replay")
        warmup_prompts.append(full)
        profile_prompts.append(full[:-1])
        request_shapes.append(
            {
                "input_length": 1,
                "past_kv_length": len(profile_prompts[-1]) - 1,
                "prefix_indices_len": int(req_info["prefix_indices_len"]),
                "output_ids_len": output_ids_len,
            }
        )
        if i < 3:
            print(
                f"  req[{i}] rid={rid[:16]}... input={len(req_data['input_ids'])} "
                f"out_used={output_ids_len} warmup={len(full)} "
                f"profile={len(full) - 1} past_kv={len(full) - 2}"
            )

    if missing:
        print(f"[warn] {len(missing)} rids missing")
    if not warmup_prompts:
        raise ValueError("No prompts")

    print(
        f"[prompts] {len(profile_prompts)} prompts, warmup_len: "
        f"min={min(len(p) for p in warmup_prompts)}, "
        f"max={max(len(p) for p in warmup_prompts)}"
    )
    print(f"[shape] {json.dumps(request_shapes, ensure_ascii=False)}")

    from sglang.srt.entrypoints.engine import Engine

    server_args = _build_glm5_server_args(args, len(warmup_prompts))
    llm = Engine(**asdict(server_args))

    print("\n[warmup] Prefill + decode to build the target cache state...")
    _ = llm.generate(
        input_ids=warmup_prompts,
        sampling_params={
            "temperature": 0,
            "top_p": 1,
            "max_new_tokens": max(1, args.profile_max_new_tokens),
            "ignore_eos": True,
        },
    )
    print("[warmup] Done.\n")

    if args.torch_profile_dir:
        print(f"[profile] torch profiler -> {args.torch_profile_dir}")
        llm.start_profile(
            output_dir=args.torch_profile_dir,
            activities=["GPU"],
            with_stack=False,
            record_shapes=False,
        )
    else:
        print("[profile] cudaProfilerStart")
        torch.cuda.cudart().cudaProfilerStart()

    outputs = []
    for i in range(args.iters):
        print(f"[profile] iter {i + 1}")
        outputs = llm.generate(
            input_ids=profile_prompts,
            sampling_params={
                "temperature": 0,
                "top_p": 1,
                "max_new_tokens": max(1, args.profile_max_new_tokens),
                "ignore_eos": True,
            },
        )

    if args.torch_profile_dir:
        llm.stop_profile()
        print("[profile] torch profiler stopped")
    else:
        torch.cuda.cudart().cudaProfilerStop()
        print("[profile] cudaProfilerStop")

    print(f"\n[done] {len(outputs)} outputs")
    try:
        llm.shutdown()
    except RuntimeError as exc:
        print(f"[warn] llm.shutdown failed after successful replay: {exc}")


if __name__ == "__main__":
    main()
