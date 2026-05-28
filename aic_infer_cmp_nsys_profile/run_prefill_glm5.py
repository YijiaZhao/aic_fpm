#!/usr/bin/env python3
"""
Prefill replay tool. Real tokens, real weights. Self-contained, no external dependencies.
"""

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import asdict, fields
from typing import Any

# 解除 CSV 字段大小限制，signed_error CSV 含 origin_input_ids 大字段
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
                    if getattr(batch, "extend_lens", None) is not None:
                        input_lengths = [int(x) for x in batch.extend_lens]
                    elif getattr(batch, "extend_seq_lens_cpu", None) is not None:
                        input_lengths = [int(x) for x in batch.extend_seq_lens_cpu]
                    elif getattr(batch, "extend_seq_lens", None) is not None:
                        input_lengths = [int(x) for x in batch.extend_seq_lens.tolist()]
                    else:
                        ext = batch.extend_num_tokens
                        if isinstance(ext, int):
                            input_lengths = [ext]
                        elif hasattr(ext, "tolist"):
                            input_lengths = ext.tolist()
                        else:
                            input_lengths = list(ext)
                    if getattr(batch, "prefix_lens", None) is not None:
                        past_kv_lengths = [int(x) for x in batch.prefix_lens]
                    else:
                        past_kv_lengths = [
                            s - il for s, il in zip(seq_lens, input_lengths)
                        ]
                extra_args = {
                    "bs": bs,
                    "forward_mode": batch.forward_mode.name,
                }
                guard_file = os.environ.get("SGLANG_REPLAY_SHAPE_GUARD_FILE", "")
                expected_shape_raw = os.environ.get("SGLANG_REPLAY_EXPECT_SHAPE", "")
                shape_guard_active = (
                    bool(guard_file)
                    and bool(expected_shape_raw)
                    and os.path.exists(guard_file)
                )
                if bs <= 8 or shape_guard_active:
                    extra_args["input_length"] = input_lengths
                    extra_args["past_kv_length"] = past_kv_lengths
                else:
                    extra_args["total_input_tokens"] = sum(input_lengths)
                    extra_args["past_kv_min"] = min(past_kv_lengths)
                    extra_args["past_kv_max"] = max(past_kv_lengths)
                    extra_args["past_kv_avg"] = sum(past_kv_lengths) // bs
                if shape_guard_active and batch.forward_mode.name == "EXTEND":
                    expected_shape = json.loads(expected_shape_raw)
                    actual_pairs = sorted(zip(input_lengths, past_kv_lengths))
                    expected_pairs = sorted(
                        zip(
                            expected_shape["input_length"],
                            expected_shape["past_kv_length"],
                        )
                    )
                    if actual_pairs != expected_pairs:
                        msg = (
                            "target shape mismatch: "
                            f"actual input={input_lengths}, past={past_kv_lengths}; "
                            f"expected input={expected_shape['input_length']}, "
                            f"past={expected_shape['past_kv_length']}"
                        )
                        print(f"[shape_mismatch] {msg}", flush=True)
                        raise RuntimeError(msg)
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
            return original_run_batch(self, batch, *args, **kwargs)

        target_class.run_batch = wrapped_run_batch
        return target_class


hook.install_class_hooks([C_SglangSchedulerRunBatchAnnotationHook])

# ============================================================================
# CSV / JSONL mapping utilities (self-contained)
# ============================================================================


def _load_csv_row_by_case_id(csv_path: str, csv_case_id: int) -> dict[str, str]:
    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                if int(float(row.get("case_id", ""))) == int(csv_case_id):
                    return row
            except (TypeError, ValueError):
                continue
    raise FileNotFoundError(f"csv_case_id={csv_case_id} not found in {csv_path}")


def _normalized_request_infos(rec: dict[str, Any]) -> list[dict[str, int]]:
    fm = int(rec["forward_mode"])
    out = []
    for req in rec["request_infos"]:
        if fm == 1:
            out.append(
                {
                    "input_length": int(float(req["extend_input_len"])),
                    "past_kv_length": int(float(req["prefix_indices_len"])),
                }
            )
        else:
            out.append(
                {
                    "input_length": 1,
                    "past_kv_length": int(float(req["prefix_indices_len"]))
                    + int(float(req["output_ids_len"])),
                }
            )
    return out


def _request_infos_signature(request_infos):
    return json.dumps(request_infos, separators=(",", ":"), sort_keys=True)


def _resolve_jsonl_case_id_from_csv(
    schedule_jsonl: str, csv_path: str, csv_case_id: int
) -> int:
    """CSV case_id 即 JSONL 0-based 行号（pipeline 已统一），直接返回。"""
    row = _load_csv_row_by_case_id(csv_path, csv_case_id)
    return int(float(row.get("case_id", csv_case_id)))


# ============================================================================
# Data loading
# ============================================================================


def _find_data_files(data_dir, tp_rank=0):
    prefix = None
    for fname in os.listdir(data_dir):
        if fname.endswith("_schedule_batch.jsonl") and fname.startswith(f"TP{tp_rank}"):
            prefix = fname.replace("_schedule_batch.jsonl", "")
            break
    if prefix is None:
        raise FileNotFoundError(
            f"No schedule_batch.jsonl for TP{tp_rank} in {data_dir}"
        )
    return (
        os.path.join(data_dir, f"{prefix}_schedule_batch.jsonl"),
        os.path.join(data_dir, f"{prefix}.request.jsonl"),
    )


def _find_csv_path(data_dir, csv_path_arg):
    if csv_path_arg:
        return csv_path_arg
    candidate = os.path.join(
        data_dir, "signed_error", "aic_vs_measured_signed_error_cases.csv"
    )
    if os.path.exists(candidate):
        return candidate
    for root, dirs, files in os.walk(data_dir):
        for f in files:
            if "signed_error" in f and f.endswith(".csv"):
                return os.path.join(root, f)
    raise FileNotFoundError(f"No signed_error CSV found in {data_dir}")


def _load_request_ids(request_jsonl):
    rid_map = {}
    with open(request_jsonl) as f:
        for line in f:
            r = json.loads(line)
            rid_map[r["rid"]] = {
                "input_ids": r["input_ids"],
                "output_ids": r["output_ids"],
            }
    return rid_map


def _load_jsonl_record(schedule_jsonl, line_idx):
    with open(schedule_jsonl, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx == int(line_idx):
                return json.loads(line)
    raise FileNotFoundError(f"line {line_idx} not found in {schedule_jsonl}")


def _repeat_or_truncate_tokens(input_ids: list[int], target_len: int) -> list[int]:
    if target_len <= 0:
        return []
    if not input_ids:
        return [100] * target_len
    if len(input_ids) >= target_len:
        return input_ids[:target_len]
    repeats = (target_len + len(input_ids) - 1) // len(input_ids)
    return (input_ids * repeats)[:target_len]


# ============================================================================
# Main
# ============================================================================


def main():
    ap = argparse.ArgumentParser(
        description="Prefill replay — real tokens, real weights"
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
        "--enforce-disable-flashinfer-allreduce-fusion",
        action="store_true",
        help="Disable FlashInfer TRTLLM allreduce fusion for profiling experiments.",
    )
    ap.add_argument(
        "--disable-custom-all-reduce",
        action="store_true",
        help="Disable SGLang custom allreduce for profiling experiments.",
    )
    ap.add_argument(
        "--piecewise-token-list",
        type=str,
        default="",
        help="Comma-separated piecewise CUDA graph token sizes for profiling smoke tests.",
    )
    ap.add_argument(
        "--avg-isl",
        type=int,
        default=0,
        help=(
            "Override every request's extend length with this AVG input length. "
            "Server args stay unchanged; only replay input tokens and shape guard are rewritten."
        ),
    )
    ap.add_argument(
        "--avg-past-kv",
        type=int,
        default=-1,
        help=(
            "Override every request's prefix length with this AVG past-KV length. "
            "Must be used together with --avg-isl for AIC AVG-shape replay."
        ),
    )
    args = ap.parse_args()

    if (args.avg_isl > 0) != (args.avg_past_kv >= 0):
        raise ValueError("--avg-isl and --avg-past-kv must be provided together")

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
    if fm != 1:
        raise ValueError(f"Not a prefill batch: forward_mode={fm}")

    bs = len(record["request_infos"])
    target_latency = record.get("iter_latency", 0) * 1000
    print(f"[target] bs={bs}, latency={target_latency:.3f}ms")

    rid_map = _load_request_ids(request_file)
    print(f"[data] {len(rid_map)} requests loaded")

    prefix_prompts = []
    full_prompts = []
    request_infos = []
    missing = []
    avg_shape = args.avg_isl > 0 and args.avg_past_kv >= 0
    if avg_shape:
        print(
            f"[avg_shape] enabled: every request uses input_length={args.avg_isl}, "
            f"past_kv_length={args.avg_past_kv}"
        )
    for i, req_info in enumerate(record["request_infos"]):
        rid = req_info["rid"]
        original_extend_len = int(float(req_info["extend_input_len"]))
        original_prefix_len = int(float(req_info["prefix_indices_len"]))
        extend_len = args.avg_isl if avg_shape else original_extend_len
        prefix_len = args.avg_past_kv if avg_shape else original_prefix_len
        if rid not in rid_map:
            missing.append(rid)
            continue
        input_ids = rid_map[rid]["input_ids"]
        prompt_len = prefix_len + extend_len
        prompt_ids = (
            _repeat_or_truncate_tokens(input_ids, prompt_len)
            if avg_shape
            else input_ids[:prompt_len]
        )
        prefix_prompts.append(prompt_ids[:prefix_len] if prefix_len > 0 else [100])
        full_prompts.append(prompt_ids)
        request_infos.append({"input_length": extend_len, "past_kv_length": prefix_len})
        if i < 3:
            print(
                f"  req[{i}] rid={rid[:16]}... prefix={prefix_len} extend={extend_len} "
                f"full={prompt_len} original_prefix={original_prefix_len} "
                f"original_extend={original_extend_len}"
            )

    if missing:
        print(f"[warn] {len(missing)} rids missing")
    if not full_prompts:
        raise ValueError("No prompts")
    if len(full_prompts) > 3:
        print(f"  ... ({len(full_prompts) - 3} more)")
    print(f"[prompts] {len(full_prompts)} prompts")

    from sglang.srt.entrypoints.engine import Engine
    from sglang.srt.server_args import ServerArgs

    total_extend_tokens = max(1, sum(ri["input_length"] for ri in request_infos))
    is_glm5 = "glm-5" in args.model.lower() or "glm5" in args.model.lower()
    if args.piecewise_token_list:
        piecewise_tokens = [
            int(x.strip()) for x in args.piecewise_token_list.split(",") if x.strip()
        ]
        piecewise_tokens = sorted(set(piecewise_tokens))
        if not piecewise_tokens:
            raise ValueError("--piecewise-token-list did not contain any token sizes")
        piecewise_max_tokens = max(piecewise_tokens)
    else:
        piecewise_capture_tokens = ((total_extend_tokens + 7) // 8) * 8
        piecewise_tokens = [piecewise_capture_tokens]
        piecewise_max_tokens = piecewise_capture_tokens
    print(f"[piecewise] actual extend tokens={total_extend_tokens}")
    if args.piecewise_token_list:
        print("[piecewise] cuda graph tokens overridden by --piecewise-token-list")
    else:
        print(f"[piecewise] cuda graph tokens use the smallest multiple of 8 greater than ISL, not prefix+ISL")
    print(f"[piecewise] cuda graph tokens={piecewise_tokens}")
    print(f"[piecewise] cuda graph max tokens={piecewise_max_tokens}")

    server_kwargs = dict(
        model_path=args.model,
        tp_size=args.tp_size,
        ep_size=args.ep_size,
        load_format="auto",
        trust_remote_code=True,
        cuda_graph_max_bs=256 if is_glm5 else len(full_prompts),
        disable_cuda_graph=True,
        disable_cuda_graph_padding=True,
        chunked_prefill_size=16384 if is_glm5 else 8192,
        max_running_requests=256 if is_glm5 else len(full_prompts),
        skip_server_warmup=not is_glm5,
        disable_overlap_schedule=True,
    )
    if not is_glm5:
        server_kwargs["cuda_graph_bs"] = [len(full_prompts)]
    server_kwargs["piecewise_cuda_graph_tokens"] = piecewise_tokens
    server_kwargs["piecewise_cuda_graph_max_tokens"] = piecewise_max_tokens
    if is_glm5:
        server_kwargs.update(
            attention_backend="nsa",
            disable_flashinfer_autotune=True,
            enable_cache_report=True,
            kv_cache_dtype="fp8_e4m3",
            max_prefill_tokens=16384,
            mem_fraction_static=0.8,
            moe_dense_tp_size=1,
            quantization="modelopt_fp4",
            sampling_backend="pytorch",
            watchdog_timeout=1000000,
        )
    server_arg_names = {field.name for field in fields(ServerArgs)}
    if is_glm5:
        # Match the GLM-5 benchmark server args recorded in b200_glm5_data/server.log.
        # The replay has to reproduce the serving path first; otherwise nsys
        # explains a different workload than the JSONL measured latency.
        for key, value in {
            "enable_dp_lm_head": False,
            "moe_runner_backend": "flashinfer_trtllm",
            "speculative_moe_runner_backend": "flashinfer_trtllm",
            "nsa_prefill_backend": "trtllm",
            "nsa_decode_backend": "trtllm",
            "enable_flashinfer_allreduce_fusion": True,
            "disable_shared_experts_fusion": True,
        }.items():
            if key in server_arg_names:
                server_kwargs[key] = value
    if "enable_piecewise_cuda_graph" in server_arg_names:
        server_kwargs["enable_piecewise_cuda_graph"] = True
    elif "disable_piecewise_cuda_graph" in server_arg_names:
        server_kwargs["disable_piecewise_cuda_graph"] = False
    if is_glm5 and "enforce_piecewise_cuda_graph" in server_arg_names:
        server_kwargs["enforce_piecewise_cuda_graph"] = True
    if args.enforce_disable_flashinfer_allreduce_fusion:
        server_kwargs["enforce_disable_flashinfer_allreduce_fusion"] = True
    if args.disable_custom_all_reduce:
        server_kwargs["disable_custom_all_reduce"] = True
    print(
        "[server_args] selected config:",
        {
            k: server_kwargs.get(k)
            for k in (
                "skip_server_warmup",
                "attention_backend",
                "nsa_prefill_backend",
                "moe_runner_backend",
                "enable_dp_lm_head",
                "enable_flashinfer_allreduce_fusion",
                "enforce_disable_flashinfer_allreduce_fusion",
                "disable_custom_all_reduce",
                "disable_shared_experts_fusion",
                "disable_cuda_graph",
                "enable_piecewise_cuda_graph",
                "disable_piecewise_cuda_graph",
                "enforce_piecewise_cuda_graph",
                "piecewise_cuda_graph_tokens",
                "piecewise_cuda_graph_max_tokens",
            )
            if k in server_kwargs
        },
    )
    expected_shape = {
        "input_length": [ri["input_length"] for ri in request_infos],
        "past_kv_length": [ri["past_kv_length"] for ri in request_infos],
    }
    guard_file = f"/tmp/sglang_replay_shape_guard_{os.getpid()}"
    os.environ["SGLANG_REPLAY_EXPECT_SHAPE"] = json.dumps(expected_shape)
    os.environ["SGLANG_REPLAY_SHAPE_GUARD_FILE"] = guard_file
    server_args = ServerArgs(**server_kwargs)
    llm = Engine(**asdict(server_args))

    if any(len(p) > 1 for p in prefix_prompts):
        print("\n[warmup] Priming prefix cache...")
        _ = llm.generate(
            input_ids=prefix_prompts,
            sampling_params={"temperature": 0, "top_p": 1, "max_new_tokens": 0},
        )
        print("[warmup] Done.\n")
    elif is_glm5:
        warmup_len = min(max(len(p) for p in full_prompts), 16384)
        first_token = full_prompts[0][0] if full_prompts and full_prompts[0] else 100
        warmup_token = 1 if first_token == 0 else 0
        print(f"\n[warmup] Running no-prefix warmup, len={warmup_len}...")
        _ = llm.generate(
            input_ids=[[warmup_token] * warmup_len],
            sampling_params={"temperature": 0, "top_p": 1, "max_new_tokens": 0},
        )
        print("[warmup] Done.\n")

    print("[profile] cudaProfilerStart")
    outputs = []
    try:
        with open(guard_file, "w", encoding="utf-8") as f:
            f.write("1\n")
        torch.cuda.cudart().cudaProfilerStart()
        outputs = llm.generate(
            input_ids=full_prompts,
            sampling_params={"temperature": 0, "top_p": 1, "max_new_tokens": 1},
        )
    finally:
        torch.cuda.cudart().cudaProfilerStop()
        if os.path.exists(guard_file):
            os.remove(guard_file)
    print("[profile] cudaProfilerStop")

    print(f"\n[done] {len(outputs)} outputs")
    llm.shutdown()


if __name__ == "__main__":
    main()
