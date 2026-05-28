#!/usr/bin/env python3
"""
Decode replay tool. Real tokens, real weights.
Imports mapping utilities from run_prefill_fixed.py.
"""

import argparse
import csv
import json
import os
import sys
from dataclasses import asdict, fields

# 解除 CSV 字段大小限制
csv.field_size_limit(sys.maxsize)

import torch

# hook.py 位于兄弟目录 hook_dataset_collector
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
            if batch is not None:
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
            return original_run_batch(self, batch, *args, **kwargs)

        target_class.run_batch = wrapped_run_batch
        return target_class


hook.install_class_hooks([C_SglangSchedulerRunBatchAnnotationHook])


from run_prefill import (
    _find_data_files,
    _find_csv_path,
    _load_csv_row_by_case_id,
    _load_jsonl_record,
    _load_request_ids,
    _resolve_jsonl_case_id_from_csv,
)


def main():
    ap = argparse.ArgumentParser(
        description="Decode replay — real tokens, real weights"
    )
    ap.add_argument("--data-dir", type=str, required=True)
    ap.add_argument("--tp-rank", type=int, default=0)
    ap.add_argument("--csv-case-id", type=int, required=True)
    ap.add_argument("--csv-path", type=str, default="")
    ap.add_argument(
        "--model", type=str, default="/models/Qwen3-235B-A22B-Instruct-2507-FP8"
    )
    ap.add_argument("--tp-size", type=int, default=8)
    ap.add_argument("--ep-size", type=int, default=8)
    ap.add_argument(
        "--iters", type=int, default=3, help="Profile 迭代次数 (default: 3)"
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

    csv_row = _load_csv_row_by_case_id(csv_path, args.csv_case_id)
    jsonl_line = _resolve_jsonl_case_id_from_csv(
        schedule_file, csv_path, args.csv_case_id
    )
    print(f"[mapping] csv_case_id={args.csv_case_id} -> jsonl line {jsonl_line}")

    record = _load_jsonl_record(schedule_file, jsonl_line)
    fm = int(record["forward_mode"])
    if fm != 2:
        raise ValueError(f"Not a decode batch: forward_mode={fm}")

    bs = len(record["request_infos"])
    target_latency = record.get("iter_latency", 0) * 1000
    print(f"[target] bs={bs}, latency={target_latency:.3f}ms")

    rid_map = _load_request_ids(request_file)
    print(f"[data] {len(rid_map)} requests loaded")

    warmup_prompts = []
    profile_prompts = []
    missing = []
    for i, req_info in enumerate(record["request_infos"]):
        rid = req_info["rid"]
        output_ids_len = int(req_info["output_ids_len"])
        if rid not in rid_map:
            missing.append(rid)
            continue
        req_data = rid_map[rid]
        full = req_data["input_ids"] + req_data["output_ids"][:output_ids_len]
        warmup_prompts.append(full)
        profile_prompts.append(full[:-1])
        if i < 3:
            print(
                f"  req[{i}] rid={rid[:16]}... input={len(req_data['input_ids'])} out_used={output_ids_len} warmup={len(full)} profile={len(full) - 1}"
            )

    if missing:
        print(f"[warn] {len(missing)} rids missing")
    if not warmup_prompts:
        raise ValueError("No prompts")

    print(
        f"[prompts] {len(profile_prompts)} prompts, len: min={min(len(p) for p in warmup_prompts)}, max={max(len(p) for p in warmup_prompts)}"
    )

    max_remaining = []
    for req_info in record["request_infos"]:
        rid = req_info["rid"]
        if rid not in rid_map:
            continue
        max_remaining.append(
            len(rid_map[rid]["output_ids"]) - int(req_info["output_ids_len"])
        )
    min_remaining = min(max_remaining) if max_remaining else 0

    from sglang.srt.entrypoints.engine import Engine
    from sglang.srt.server_args import ServerArgs

    is_glm5 = "glm-5" in args.model.lower() or "glm5" in args.model.lower()
    chunked_prefill_size = 16384 if is_glm5 else 8192

    server_kwargs = dict(
        model_path=args.model,
        tp_size=args.tp_size,
        ep_size=args.ep_size,
        load_format="auto",
        trust_remote_code=True,
        disable_overlap_schedule=True,
        cuda_graph_max_bs=256 if is_glm5 else len(warmup_prompts),
        max_running_requests=256 if is_glm5 else len(warmup_prompts),
        disable_cuda_graph_padding=True,
        chunked_prefill_size=chunked_prefill_size,
    )
    if not is_glm5:
        server_kwargs["cuda_graph_bs"] = [len(warmup_prompts)]
        server_kwargs["piecewise_cuda_graph_tokens"] = [chunked_prefill_size]
    if is_glm5:
        server_kwargs.update(
            attention_backend="nsa",
            disable_flashinfer_autotune=True,
            enable_cache_report=True,
            enable_dp_lm_head=True,
            kv_cache_dtype="fp8_e4m3",
            max_prefill_tokens=16384,
            mem_fraction_static=0.8,
            moe_dense_tp_size=1,
            quantization="modelopt_fp4",
            sampling_backend="pytorch",
            watchdog_timeout=1000000,
        )
    server_arg_names = {field.name for field in fields(ServerArgs)}
    if not is_glm5:
        if "enable_piecewise_cuda_graph" in server_arg_names:
            server_kwargs["enable_piecewise_cuda_graph"] = True
        elif "disable_piecewise_cuda_graph" in server_arg_names:
            server_kwargs["disable_piecewise_cuda_graph"] = False
    server_args = ServerArgs(**server_kwargs)
    llm = Engine(**asdict(server_args))

    print("\n[warmup] Prefill + 1 decode...")
    _ = llm.generate(
        input_ids=warmup_prompts,
        sampling_params={"temperature": 0, "top_p": 1, "max_new_tokens": 4, "ignore_eos": True},
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
    for i in range(args.iters):
        print(f"[profile] iter {i + 1}")
        outputs = llm.generate(
            input_ids=profile_prompts,
            sampling_params={"temperature": 0, "top_p": 1, "max_new_tokens": 4, "ignore_eos": True},
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
