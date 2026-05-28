#!/usr/bin/env python3
"""Synthetic mixed-prefill replay for nsys profiling.

This is a small sibling of run_prefill.py for cases that do not exist in the
captured request JSONL.  It creates synthetic token ids for each request, warms
the prefix cache with `past_kv` tokens, then profiles one full-prompt generate
so SGLang should run an EXTEND batch with the requested `(isl, past_kv)` pairs.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from dataclasses import asdict, fields

import torch

# hook.py lives next to this script in ../hook_dataset_collector when copied
# into refactor_test_aic/aic_infer_cmp_nsys_profile.
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "hook_dataset_collector")
)
import hook  # noqa: E402


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


def _parse_reqs(text: str) -> list[tuple[int, int]]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = ast.literal_eval(text)
    reqs = [(int(isl), int(past)) for isl, past in value]
    if not reqs:
        raise ValueError("reqs cannot be empty")
    for isl, past in reqs:
        if isl <= 0 or past < 0:
            raise ValueError(f"invalid req: {(isl, past)}")
    return reqs


def _make_token_ids(
    length: int, request_idx: int, token_base: int, token_span: int
) -> list[int]:
    # Different first tokens avoid accidental cross-request radix prefix sharing.
    base = token_base + request_idx * token_span
    return [base + (i % token_span) for i in range(length)]


def _load_real_token_prompts(path: str, lengths: list[int]) -> list[list[int]]:
    if not path or not os.path.exists(path):
        return []

    prompts: list[list[int]] = []
    with open(path, encoding="utf-8") as f:
        records = [json.loads(line)["input_ids"] for line in f if line.strip()]

    used_indices: set[int] = set()
    for length in lengths:
        chosen = None
        for idx, input_ids in enumerate(records):
            if idx in used_indices:
                continue
            if len(input_ids) >= length:
                chosen = idx
                break
        if chosen is None:
            return []
        used_indices.add(chosen)
        prompts.append(records[chosen][:length])
    return prompts


def main():
    ap = argparse.ArgumentParser(description="Synthetic mixed prefill replay")
    ap.add_argument(
        "--reqs",
        dest="reqs_text",
        default="[(233,5000),(333,4000),(512,2048)]",
        help="Synthetic requests as [(isl,past_kv), ...].",
    )
    ap.add_argument(
        "--reqs-json",
        dest="reqs_text",
        help="Backward-compatible alias for --reqs.",
    )
    ap.add_argument("--model", default="/raid/kimi/DeepSeek-V4-Flash-FP8/")
    ap.add_argument("--tp-size", type=int, default=8)
    ap.add_argument("--ep-size", type=int, default=1)
    ap.add_argument("--token-base", type=int, default=1000)
    ap.add_argument("--token-span", type=int, default=251)
    ap.add_argument(
        "--token-source-jsonl", default="/raid/kimi/ds4/dsv4_data/TP0.request.jsonl"
    )
    args = ap.parse_args()

    reqs = _parse_reqs(args.reqs_text)
    full_lengths = [past + isl for isl, past in reqs]
    full_prompts = _load_real_token_prompts(args.token_source_jsonl, full_lengths)
    if full_prompts:
        print(f"[tokens] using real input_ids from {args.token_source_jsonl}")
    else:
        print(
            "[tokens] real token source unavailable/insufficient; using deterministic synthetic ids"
        )
        full_prompts = [
            _make_token_ids(past + isl, i, args.token_base, args.token_span)
            for i, (isl, past) in enumerate(reqs)
        ]
    print(f"[synthetic] reqs={reqs}")
    print(
        f"[synthetic] bs={len(reqs)} total_extend_tokens={sum(isl for isl, _past in reqs)}"
    )
    print(f"[synthetic] avg_isl={sum(isl for isl, _past in reqs) / len(reqs):.6f}")
    print(f"[synthetic] avg_past_kv={sum(past for _isl, past in reqs) / len(reqs):.6f}")
    print(f"[synthetic] full_lengths={[len(p) for p in full_prompts]}")

    from sglang.srt.entrypoints.engine import Engine
    from sglang.srt.server_args import ServerArgs

    prefix_prompts = []
    request_infos = []
    for i, (full, (extend_len, prefix_len)) in enumerate(zip(full_prompts, reqs)):
        prefix_prompts.append(full[:prefix_len] if prefix_len > 0 else [100])
        request_infos.append({"input_length": extend_len, "past_kv_length": prefix_len})
        if i < 3:
            print(
                f"  req[{i}] prefix={prefix_len} extend={extend_len} full={prefix_len + extend_len}"
            )
    if len(full_prompts) > 3:
        print(f"  ... ({len(full_prompts) - 3} more)")
    print(f"[prompts] {len(full_prompts)} prompts")

    total_extend_tokens = max(1, sum(ri["input_length"] for ri in request_infos))
    piecewise_tokens = [total_extend_tokens]
    piecewise_max_tokens = max(len(p) for p in full_prompts)
    print(f"[piecewise] cuda graph tokens={piecewise_tokens}")
    print(f"[piecewise] cuda graph max seq len={piecewise_max_tokens}")

    server_kwargs = dict(
        model_path=args.model,
        tp_size=args.tp_size,
        ep_size=args.ep_size,
        load_format="auto",
        trust_remote_code=True,
        cuda_graph_max_bs=len(full_prompts),
        cuda_graph_bs=[len(full_prompts)],
        disable_cuda_graph_padding=True,
        piecewise_cuda_graph_tokens=piecewise_tokens,
        piecewise_cuda_graph_max_tokens=piecewise_max_tokens,
        skip_server_warmup=True,
        disable_overlap_schedule=True,
    )
    server_arg_names = {field.name for field in fields(ServerArgs)}
    if "enable_piecewise_cuda_graph" in server_arg_names:
        server_kwargs["enable_piecewise_cuda_graph"] = True
    elif "disable_piecewise_cuda_graph" in server_arg_names:
        server_kwargs["disable_piecewise_cuda_graph"] = False
    server_args = ServerArgs(**server_kwargs)

    llm = Engine(**asdict(server_args))

    if any(len(p) > 1 for p in prefix_prompts):
        print("\n[warmup] Priming prefix cache...")
        _ = llm.generate(
            input_ids=prefix_prompts,
            sampling_params={"temperature": 0, "top_p": 1, "max_new_tokens": 0},
        )
        print("[warmup] Done.\n")

    print("[profile] cudaProfilerStart")
    torch.cuda.cudart().cudaProfilerStart()
    outputs = llm.generate(
        input_ids=full_prompts,
        sampling_params={"temperature": 0, "top_p": 1, "max_new_tokens": 1},
    )
    torch.cuda.cudart().cudaProfilerStop()
    print("[profile] cudaProfilerStop")

    print(f"\n[done] {len(outputs)} outputs")
    llm.shutdown()


if __name__ == "__main__":
    main()
