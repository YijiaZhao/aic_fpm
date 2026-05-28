#!/usr/bin/env python3
"""
AIC per-op 预估 vs nsys 实测 kernel 时延对比工具。

从 nsys .sqlite 中提取 GPU kernel trace，按 AIC op 分类汇总，
再与 aic_infer_component.py 的 per-op 预估对齐，输出对比表。

Usage:
    # 基本用法：指定 nsys 报告和 case_id
    python compare_aic_nsys.py \
        --nsys-rep /path/to/nsys_prefill_case_1115.nsys-rep \
        --data-dir /path/to/data \
        --case-id 1115

    # 使用已有 sqlite（跳过导出）
    python compare_aic_nsys.py \
        --nsys-sqlite /path/to/nsys_prefill_case_1115.sqlite \
        --data-dir /path/to/data \
        --case-id 1115

    # 指定使用第 N 次迭代（默认取中位数迭代）
    python compare_aic_nsys.py ... --iter-index 2
"""

import argparse
import json
import os
import re
import sqlite3
import subprocess
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# 添加项目路径
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR.parent.parent / "src"))
sys.path.insert(0, str(_SCRIPT_DIR.parent))


# ============================================================================
# AIC op 分类规则 — 基于 kernel 名称模式匹配
# ============================================================================

# 注意：顺序很重要，靠前的规则优先匹配
KERNEL_CLASSIFY_RULES: List[Tuple[str, str]] = [
    # ---- Attention ----
    ("FlashAttnFwd", "attention"),
    ("FlashAttnBwd", "attention"),
    ("flash::prepare_varlen", "attention"),
    ("fused_qknorm", "attention"),
    ("fused_rope_store", "attention"),
    ("masked_fill_kernel", "attention"),
    # ---- NCCL (按位置分类，见 _classify_nccl) ----
    # ncclAllReduce / ncclAllGather 等由位置决定，不在这里匹配
    # ---- MoE ----
    ("fused_moe_kernel", "moe"),
    ("act_and_mul_kernel", "moe"),
    ("moe_sum_reduce", "moe"),
    ("moe_align_block_size", "moe"),
    ("count_and_sort_expert", "moe"),
    ("topkGatingSoftmax", "moe"),
    # ---- GEMM ----
    ("deep_gemm::sm90_fp8_gemm", "gemm"),  # deep_gemm FP8
    ("nvjet_tst", "gemm"),  # cublas JET
    ("splitKreduce_kernel", "gemm"),  # cublas splitK reduce
    ("sm90_xmma_gemm", "gemm"),  # cublas SM90 GEMM
    ("ampere_fp16", "gemm"),  # cublas FP16
    ("cutlass::Kernel", "gemm"),  # cutlass GEMM (非 flash attn)
    # ---- Norm ----
    ("RMSNormKernel", "norm"),
    ("FusedAddRMSNormKernel", "norm"),
    ("LayerNormKernel", "norm"),
    # ---- Quantization ----
    ("per_token_group_quant", "quant"),
    ("per_channel_quant", "quant"),
    # ---- Embedding ----
    ("vectorized_gather", "embedding"),
    ("write_req_to_token_pool", "embedding"),
    # ---- Misc / overhead ----
    ("compute_position_kernel", "overhead"),
    ("DeviceScanKernel", "overhead"),
    ("DeviceScanInitKernel", "overhead"),
    ("FillFunctor", "overhead"),
    ("compare_scalar_kernel", "overhead"),
    ("bitwise_not_kernel", "overhead"),
    ("BitwiseAndFunctor", "overhead"),
    ("BitwiseOrFunctor", "overhead"),
    ("CUDAFunctor_add", "overhead"),
    ("CUDAFunctorOnSelf_add", "overhead"),
    ("AUnaryFunctor", "overhead"),
    ("BinaryFunctor<long", "overhead"),
    ("direct_copy_kernel", "overhead"),
    ("index_elementwise_kernel", "overhead"),
    ("index_kernel_impl", "overhead"),
    ("reduce_kernel", "overhead"),
]


def _classify_kernel_by_name(name: str) -> str:
    """通过 kernel 名称模式匹配分类。"""
    for pattern, category in KERNEL_CLASSIFY_RULES:
        if pattern in name:
            return category
    return "unknown"


# ============================================================================
# 全局顺序分类 — 基于 NCCL 计数和 kernel 顺序
# ============================================================================


def _is_allreduce_kernel(name: str) -> bool:
    """判断是否为 AllReduce 类通信 kernel（NCCL 或 SGLang 自定义）。"""
    lower = name.lower()
    return (
        "nccl" in lower
        or "cross_device_reduce" in lower
        or "custom_all_reduce" in lower
        or "ncclallgather" in lower.replace("_", "")
    )


def _classify_all_kernels(
    kernels: List[dict], per_layer_detail: bool = False
) -> Tuple[Dict[str, float], int, Optional[List[dict]]]:
    """
    对整个迭代的 kernel 序列进行全局顺序分类。

    利用固定的执行模式 (WideEP+DeepEP 场景):
    - 模型前处理: embedding + misc → ncclAllReduce (embedding AR)
    - 每层 transformer:
        norm → quant+GEMM(QKV) → attn → quant+GEMM(proj) → AllReduce(moe_pre_dispatch)
        → FusedAddRMSNorm → router+splitK → MoE系列 → AllReduce(moe_post_dispatch)
    - 模型后处理: final norm + lm_head

    NCCL 序列: embedding_ar, (moe_pre_dispatch, moe_post_dispatch) × num_layers

    Args:
        per_layer_detail: 若为 True，额外返回逐层的 moe/moe_post_dispatch 明细。

    Returns:
        (per_op_dict, num_layers, layer_details)
        layer_details: None 或 [{"moe": ms, "moe_post_dispatch": ms, "router_gemm": ms, "add_norm_2": ms}, ...]
    """
    totals = defaultdict(float)
    layer_details = [] if per_layer_detail else None

    # Phase 1: 找到所有 AllReduce kernel 的位置
    nccl_indices = []
    for i, k in enumerate(kernels):
        if _is_allreduce_kernel(k["name"]):
            nccl_indices.append(i)

    if not nccl_indices:
        # 没有 NCCL，全部归入 overhead
        for k in kernels:
            totals["overhead"] += k["dur_ns"] / 1e6
        return dict(totals), 0, layer_details

    # 第一个 NCCL = embedding AllReduce
    # 之后每 2 个 NCCL = 一层 (moe_pre_dispatch, moe_post_dispatch)
    num_layers = (len(nccl_indices) - 1) // 2

    # Phase 2: 预处理区域 (index 0 ~ nccl_indices[0])
    embedding_nccl_idx = nccl_indices[0]
    for i in range(embedding_nccl_idx):
        k = kernels[i]
        dur_ms = k["dur_ns"] / 1e6
        cat = _classify_kernel_by_name(k["name"])
        if cat == "embedding":
            totals["embedding"] += dur_ms
        else:
            totals["overhead"] += dur_ms
    totals["embedding_ar"] += kernels[embedding_nccl_idx]["dur_ns"] / 1e6

    # Phase 3: 逐层分类
    for layer_idx in range(num_layers):
        nccl_base = 1 + layer_idx * 2  # index in nccl_indices
        pre_dispatch_nccl_idx = nccl_indices[nccl_base]  # moe_pre_dispatch AllReduce
        post_dispatch_nccl_idx = nccl_indices[
            nccl_base + 1
        ]  # moe_post_dispatch AllReduce

        # 本层开始 = 上一个 NCCL 的下一个 kernel
        if nccl_base == 1:
            layer_start = embedding_nccl_idx + 1
        else:
            layer_start = nccl_indices[nccl_base - 1] + 1

        # 逐层明细
        layer_moe_ms = 0.0
        layer_post_dispatch_ms = 0.0
        layer_router_gemm_ms = 0.0
        layer_add_norm_2_ms = 0.0

        # ---- 注意力部分: layer_start ~ pre_dispatch_nccl_idx (不含) ----
        gemm_count_attn = 0
        pending_quant = 0.0
        for i in range(layer_start, pre_dispatch_nccl_idx):
            k = kernels[i]
            name = k["name"]
            dur_ms = k["dur_ns"] / 1e6
            cat = _classify_kernel_by_name(name)

            if cat == "norm":
                totals["add_norm_1"] += dur_ms + pending_quant
                pending_quant = 0.0
            elif cat == "quant":
                pending_quant += dur_ms
            elif cat == "gemm":
                total = dur_ms + pending_quant
                pending_quant = 0.0
                if gemm_count_attn == 0:
                    totals["qkv_gemm"] += total
                else:
                    totals["proj_gemm"] += total
                gemm_count_attn += 1
            elif cat == "attention":
                totals["attention"] += dur_ms
            else:
                totals["overhead"] += dur_ms + pending_quant
                pending_quant = 0.0

        # moe_pre_dispatch AllReduce
        totals["moe_pre_dispatch"] += kernels[pre_dispatch_nccl_idx]["dur_ns"] / 1e6

        # ---- FFN 部分: pre_dispatch_nccl_idx+1 ~ post_dispatch_nccl_idx (不含) ----
        pending_quant = 0.0
        for i in range(pre_dispatch_nccl_idx + 1, post_dispatch_nccl_idx):
            k = kernels[i]
            name = k["name"]
            dur_ms = k["dur_ns"] / 1e6
            cat = _classify_kernel_by_name(name)

            if cat == "norm":
                val = dur_ms + pending_quant
                totals["add_norm_2"] += val
                layer_add_norm_2_ms += val
                pending_quant = 0.0
            elif cat == "quant":
                pending_quant += dur_ms
            elif cat == "gemm":
                val = dur_ms + pending_quant
                totals["router_gemm"] += val
                layer_router_gemm_ms += val
                pending_quant = 0.0
            elif cat == "moe":
                val = dur_ms + pending_quant
                totals["moe"] += val
                layer_moe_ms += val
                pending_quant = 0.0
            elif cat == "attention":
                totals["overhead"] += dur_ms
            else:
                totals["overhead"] += dur_ms + pending_quant
                pending_quant = 0.0

        # moe_post_dispatch AllReduce
        layer_post_dispatch_ms = kernels[post_dispatch_nccl_idx]["dur_ns"] / 1e6
        totals["moe_post_dispatch"] += layer_post_dispatch_ms

        if layer_details is not None:
            layer_details.append(
                {
                    "moe": layer_moe_ms,
                    "moe_post_dispatch": layer_post_dispatch_ms,
                    "router_gemm": layer_router_gemm_ms,
                    "add_norm_2": layer_add_norm_2_ms,
                }
            )

    # Phase 4: 后处理区域 (最后一个 NCCL 之后)
    post_start = nccl_indices[-1] + 1
    for i in range(post_start, len(kernels)):
        k = kernels[i]
        dur_ms = k["dur_ns"] / 1e6
        totals["overhead"] += dur_ms

    return dict(totals), num_layers, layer_details


# ============================================================================
# nsys SQLite 数据提取
# ============================================================================


def _ensure_sqlite(nsys_rep_path: str) -> str:
    """确保 nsys-rep 已导出为 sqlite，返回 sqlite 路径。"""
    sqlite_path = nsys_rep_path.replace(".nsys-rep", ".sqlite")
    if os.path.exists(sqlite_path):
        return sqlite_path
    # 使用 nsys stats 触发导出
    subprocess.run(
        ["nsys", "stats", "--force-export=true", "--report", "nvtx_sum", nsys_rep_path],
        capture_output=True,
    )
    if not os.path.exists(sqlite_path):
        raise FileNotFoundError(f"Failed to export sqlite from {nsys_rep_path}")
    return sqlite_path


def _load_nvtx_ranges(
    conn: sqlite3.Connection, forward_mode_filter: Optional[str] = None
) -> List[dict]:
    """加载所有 Scheduler.run_batch NVTX ranges。

    Args:
        forward_mode_filter: 可选, 'EXTEND' 或 'DECODE'，只返回匹配的 range。
    """
    cur = conn.cursor()
    cur.execute(
        "SELECT start, end, text FROM NVTX_EVENTS "
        "WHERE text LIKE '%Scheduler.run_batch%' ORDER BY start"
    )
    ranges = []
    for start, end, text in cur.fetchall():
        if end is None or end <= start:
            continue
        # 过滤 forward_mode
        if forward_mode_filter:
            if f"'forward_mode': '{forward_mode_filter}'" not in text:
                continue
        ranges.append(
            {"start": start, "end": end, "text": text, "dur_ms": (end - start) / 1e6}
        )
    return ranges


def _load_kernels_in_range(
    conn: sqlite3.Connection, range_start: int, range_end: int, device_id: int = 0
) -> List[dict]:
    """提取指定时间范围内的 GPU kernels。"""
    cur = conn.cursor()
    cur.execute(
        """
        SELECT k.start, k.end, s.value, (k.end - k.start) as dur_ns
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON k.demangledName = s.id
        WHERE k.start >= ? AND k.end <= ? AND k.deviceId = ?
        ORDER BY k.start
        """,
        (range_start, range_end, device_id),
    )
    return [
        {"start": r[0], "end": r[1], "name": r[2], "dur_ns": r[3]}
        for r in cur.fetchall()
    ]


# ============================================================================
# 主对比逻辑
# ============================================================================

# nsys 分类 key → AIC op name 的映射
NSYS_TO_AIC_OP = OrderedDict(
    [
        ("embedding", "context_embedding"),
        ("add_norm_1", "context_add_norm_1"),
        ("qkv_gemm", "context_qkv_gemm"),
        ("attention", "context_attention"),
        ("proj_gemm", "context_proj_gemm"),
        ("moe_pre_dispatch", "context_moe_pre_dispatch"),
        ("add_norm_2", "context_add_norm_2"),
        ("router_gemm", "context_router_gemm"),
        ("moe", "context_moe"),
        ("moe_post_dispatch", "context_moe_post_dispatch"),
        ("overhead", "(framework_overhead)"),
    ]
)

# decode 模式的映射
NSYS_TO_AIC_OP_DECODE = OrderedDict(
    [
        ("embedding", "generation_embedding"),
        ("add_norm_1", "generation_add_norm_1"),
        ("qkv_gemm", "generation_qkv_gemm"),
        ("attention", "generation_attention"),
        ("proj_gemm", "generation_proj_gemm"),
        ("moe_pre_dispatch", "generation_moe_pre_dispatch"),
        ("add_norm_2", "generation_add_norm_2"),
        ("router_gemm", "generation_router_gemm"),
        ("moe", "generation_moe"),
        ("moe_post_dispatch", "generation_moe_post_dispatch"),
        ("overhead", "(framework_overhead)"),
    ]
)


def _select_iteration(
    nvtx_ranges: List[dict], iter_index: Optional[int] = None
) -> Tuple[int, dict]:
    """选择要分析的迭代。默认跳过 warmup，取中位数时延的迭代。"""
    num_iters = len(nvtx_ranges)
    if iter_index is not None:
        return iter_index, nvtx_ranges[iter_index]
    if num_iters > 2:
        candidates = [(i, r["dur_ms"]) for i, r in enumerate(nvtx_ranges) if i > 0]
        candidates.sort(key=lambda x: x[1])
        idx = candidates[len(candidates) // 2][0]
        return idx, nvtx_ranges[idx]
    if num_iters > 1:
        return 1, nvtx_ranges[1]
    return 0, nvtx_ranges[0]


def extract_nsys_per_op(
    sqlite_path: str,
    iter_index: Optional[int] = None,
    is_decode: bool = False,
    multi_rank: bool = False,
) -> Tuple[Dict[str, float], dict]:
    """
    从 nsys sqlite 提取 per-op 时延汇总。

    Args:
        is_decode: 是否为 decode case，若是则过滤 DECODE forward_mode 的 NVTX range。
        multi_rank: 若为 True，跨所有 device 做 MoE 负载均衡对齐：
                    每层选 MoE 时间最长的 rank，用它的 moe 和 moe_post_dispatch。
    """
    conn = sqlite3.connect(sqlite_path)

    forward_mode = "DECODE" if is_decode else None
    nvtx_ranges = _load_nvtx_ranges(conn, forward_mode_filter=forward_mode)
    if not nvtx_ranges:
        raise ValueError("No Scheduler.run_batch NVTX ranges found")

    num_iters = len(nvtx_ranges)
    selected_idx, selected_range = _select_iteration(nvtx_ranges, iter_index)
    print(
        f"[nsys] {num_iters} iterations found, using iter #{selected_idx} "
        f"(dur={selected_range['dur_ms']:.3f}ms)"
    )
    print(f"[nsys] NVTX: {selected_range['text'][:120]}")

    # 主 device (device 0) 分类
    kernels = _load_kernels_in_range(
        conn, selected_range["start"], selected_range["end"], device_id=0
    )
    print(f"[nsys] {len(kernels)} kernels in selected iteration (device 0)")

    need_per_layer = multi_rank
    per_op_totals, num_layers, layer_details_d0 = _classify_all_kernels(
        kernels, per_layer_detail=need_per_layer
    )
    print(f"[nsys] {num_layers} transformer layers detected")

    # 多 rank MoE 对齐
    if multi_rank and num_layers > 0:
        per_op_totals = _aggregate_multi_rank_moe(
            conn, selected_range, per_op_totals, num_layers, layer_details_d0
        )

    conn.close()

    meta = {
        "num_iters": num_iters,
        "selected_iter": selected_idx,
        "total_dur_ms": selected_range["dur_ms"],
        "num_layers": num_layers,
        "num_kernels": len(kernels),
        "multi_rank": multi_rank,
    }
    return dict(per_op_totals), meta


def _aggregate_multi_rank_moe(
    conn: sqlite3.Connection,
    selected_range: dict,
    base_totals: Dict[str, float],
    num_layers: int,
    layer_details_d0: List[dict],
) -> Dict[str, float]:
    """
    跨所有 rank 做 MoE 负载均衡对齐。

    对每层：选 MoE 计算时间最长的 rank，用它的 moe 和 moe_post_dispatch 替换 device 0 的值。
    这样可以反映真实的 MoE 计算瓶颈（最慢 rank 决定整体时延），
    而非单个 rank 的偏低 MoE + 偏高 AllReduce 等待时间。
    """
    cur = conn.cursor()

    # 检测有多少个 device
    cur.execute(
        "SELECT DISTINCT deviceId FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY deviceId"
    )
    all_devices = [r[0] for r in cur.fetchall()]
    if len(all_devices) <= 1:
        print("[multi-rank] only 1 device found, skipping multi-rank aggregation")
        return base_totals

    print(
        f"[multi-rank] {len(all_devices)} devices detected, aggregating MoE across ranks..."
    )

    # 收集每个 device 的逐层 MoE 明细
    all_layer_details = {0: layer_details_d0}
    for dev in all_devices:
        if dev == 0:
            continue
        kernels_dev = _load_kernels_in_range(
            conn, selected_range["start"], selected_range["end"], device_id=dev
        )
        _, nl_dev, ld_dev = _classify_all_kernels(kernels_dev, per_layer_detail=True)
        if nl_dev == num_layers and ld_dev is not None:
            all_layer_details[dev] = ld_dev
        else:
            print(
                f"[multi-rank] device {dev}: {nl_dev} layers (expected {num_layers}), skipped"
            )

    # 逐层选 MoE 最长的 rank
    new_moe_total = 0.0
    new_post_dispatch_total = 0.0
    old_moe_total = base_totals.get("moe", 0.0)
    old_post_dispatch_total = base_totals.get("moe_post_dispatch", 0.0)

    for layer_idx in range(num_layers):
        best_dev = 0
        best_moe = layer_details_d0[layer_idx]["moe"]

        for dev, details in all_layer_details.items():
            moe_ms = details[layer_idx]["moe"]
            if moe_ms > best_moe:
                best_moe = moe_ms
                best_dev = dev

        chosen = all_layer_details[best_dev][layer_idx]
        new_moe_total += chosen["moe"]
        new_post_dispatch_total += chosen["moe_post_dispatch"]

    # 替换 base_totals 中的 moe 和 moe_post_dispatch
    result = dict(base_totals)
    result["moe"] = new_moe_total
    result["moe_post_dispatch"] = new_post_dispatch_total

    print(
        f"[multi-rank] moe: {old_moe_total:.3f}ms (dev0) → {new_moe_total:.3f}ms (max-rank)"
    )
    print(
        f"[multi-rank] moe_post_dispatch: {old_post_dispatch_total:.3f}ms (dev0) → {new_post_dispatch_total:.3f}ms (max-rank)"
    )

    return result


def get_aic_per_op(data_dir: str, case_id: int) -> Tuple[Dict[str, float], float, bool]:
    """
    调用 aic_infer_component 获取 per-op 预估。

    Returns:
        (latency_dict, total_ms, is_decode)
    """
    from config import SUBDIR_CSV
    from utils import RequestInfo

    csv_path = os.path.join(data_dir, SUBDIR_CSV, "batches_output.csv")

    # 延迟导入避免循环
    from aic_infer_component import (
        _load_csv_row_by_case_id,
        _parse_request_infos,
        estimate_batch_per_op,
    )

    row = _load_csv_row_by_case_id(csv_path, case_id)
    batch_type = row["batch_type"].strip().lower()
    is_decode = batch_type == "decode"
    reqs = _parse_request_infos(row["request_infos"])

    latency_dict, runtime_config, total_ms = estimate_batch_per_op(reqs, is_decode)
    return latency_dict, total_ms, is_decode


def _prepare_aic_dict(aic_dict: Dict[str, float], is_decode: bool) -> Dict[str, float]:
    """准备 AIC dict 用于对比（直接透传，不再合并 dispatch）。"""
    return dict(aic_dict)


def print_comparison(
    aic_dict: Dict[str, float],
    nsys_dict: Dict[str, float],
    nsys_meta: dict,
    is_decode: bool,
):
    """打印 AIC vs nsys 对比表。"""
    op_map = NSYS_TO_AIC_OP_DECODE if is_decode else NSYS_TO_AIC_OP

    # 准备 AIC dict
    aic_merged = _prepare_aic_dict(aic_dict, is_decode)

    prefix = "generation_" if is_decode else "context_"

    print()
    print("=" * 100)
    print(f"  AIC per-op 预估 vs nsys 实测 kernel 时延对比")
    print(
        f"  (nsys iter #{nsys_meta['selected_iter']}, "
        f"{nsys_meta['num_layers']} layers, "
        f"{nsys_meta['num_kernels']} kernels)"
    )
    print("=" * 100)
    print(
        f"{'AIC Op':<42s} {'AIC (ms)':>10s} {'nsys (ms)':>10s} {'Diff (ms)':>10s} {'Err%':>8s}"
    )
    print("-" * 100)

    aic_total = 0.0
    nsys_total = 0.0
    rows = []

    for nsys_key, aic_op_name in op_map.items():
        nsys_ms = nsys_dict.get(nsys_key, 0.0)
        aic_ms = aic_merged.get(aic_op_name, 0.0)

        # 特殊处理：embedding_ar 在 nsys 中单独分类
        if nsys_key == "embedding":
            # 加上 embedding_ar
            nsys_ms += nsys_dict.get("embedding_ar", 0.0)
            # AIC embedding + embedding_ar
            aic_ms = aic_merged.get(f"{prefix}embedding", 0.0) + aic_merged.get(
                f"{prefix}embedding_ar", 0.0
            )
            aic_op_name = f"{prefix}embedding(+ar)"

        # 跳过两边都是 0 的
        if nsys_ms < 0.001 and aic_ms < 0.001:
            continue

        diff = aic_ms - nsys_ms
        err_pct = (diff / nsys_ms * 100) if nsys_ms > 0.001 else float("nan")

        rows.append((aic_op_name, aic_ms, nsys_ms, diff, err_pct))
        if not aic_op_name.startswith("("):
            aic_total += aic_ms
            nsys_total += nsys_ms

    # framework overhead 单独一行
    overhead_ms = nsys_dict.get("overhead", 0.0)
    if overhead_ms > 0.001:
        rows.append(
            ("(framework_overhead)", 0.0, overhead_ms, -overhead_ms, float("nan"))
        )
        nsys_total += overhead_ms

    for aic_op_name, aic_ms, nsys_ms, diff, err_pct in rows:
        err_str = f"{err_pct:+.1f}%" if not np.isnan(err_pct) else "N/A"
        marker = ""
        if not np.isnan(err_pct) and abs(err_pct) > 20:
            marker = " ⚠️" if abs(err_pct) > 50 else " ⚡"
        print(
            f"{aic_op_name:<42s} {aic_ms:10.3f} {nsys_ms:10.3f} {diff:+10.3f} {err_str:>8s}{marker}"
        )

    print("-" * 100)
    diff_total = aic_total - nsys_total
    err_total = (diff_total / nsys_total * 100) if nsys_total > 0 else 0
    print(
        f"{'TOTAL':<42s} {aic_total:10.3f} {nsys_total:10.3f} {diff_total:+10.3f} {err_total:+.1f}%"
    )
    print(
        f"{'nsys wall-clock (NVTX range)':<42s} {'':>10s} {nsys_meta['total_dur_ms']:10.3f}"
    )
    print()

    # 占比分析
    print("=" * 70)
    print("  nsys 各 op 时延占比")
    print("=" * 70)
    all_entries = [(k, v) for k, v in nsys_dict.items() if v > 0.001]
    all_entries.sort(key=lambda x: -x[1])
    for k, v in all_entries:
        pct = v / nsys_meta["total_dur_ms"] * 100
        bar = "█" * int(pct / 2) + "▏" * (1 if pct % 2 > 0.5 else 0)
        print(f"  {k:<30s} {v:8.3f} ms  ({pct:5.1f}%)  {bar}")
    print()


# ============================================================================
# CLI
# ============================================================================


def main():
    ap = argparse.ArgumentParser(
        description="AIC per-op 预估 vs nsys 实测 kernel 时延对比",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--nsys-rep", type=str, help="nsys .nsys-rep 文件路径")
    group.add_argument(
        "--nsys-sqlite", type=str, help="nsys .sqlite 文件路径（已导出）"
    )

    ap.add_argument("--data-dir", type=str, required=True, help="数据根目录")
    ap.add_argument("--case-id", type=int, required=True, help="CSV case_id")
    ap.add_argument(
        "--iter-index",
        type=int,
        default=None,
        help="使用第 N 次迭代 (0-based)，默认自动选取中位数",
    )
    ap.add_argument("--device-id", type=int, default=0, help="GPU device ID (默认 0)")
    ap.add_argument(
        "--multi-rank",
        action="store_true",
        help="跨所有 rank 做 MoE 负载均衡对齐：每层选 MoE 最长的 rank",
    )
    ap.add_argument("--json", action="store_true", help="额外输出 JSON 格式结果")

    args = ap.parse_args()

    # 1. 获取 sqlite
    if args.nsys_sqlite:
        sqlite_path = args.nsys_sqlite
    else:
        sqlite_path = _ensure_sqlite(args.nsys_rep)
    print(f"[sqlite] {sqlite_path}")

    # 2. 获取 AIC per-op（先获取以确定 is_decode）
    print(f"\n[aic] Getting AIC per-op for case_id={args.case_id} ...")
    aic_dict, aic_total, is_decode = get_aic_per_op(args.data_dir, args.case_id)

    # 3. 提取 nsys per-op（用 is_decode 过滤 forward_mode）
    nsys_dict, nsys_meta = extract_nsys_per_op(
        sqlite_path, args.iter_index, is_decode=is_decode, multi_rank=args.multi_rank
    )

    # 4. 打印对比
    print_comparison(aic_dict, nsys_dict, nsys_meta, is_decode)

    # 5. 可选 JSON 输出
    if args.json:
        result = {
            "case_id": args.case_id,
            "is_decode": is_decode,
            "nsys_meta": nsys_meta,
            "aic_per_op": aic_dict,
            "nsys_per_op": nsys_dict,
            "aic_total_ms": aic_total,
            "nsys_total_ms": nsys_meta["total_dur_ms"],
        }
        json_path = sqlite_path.replace(".sqlite", "_aic_vs_nsys.json")
        with open(json_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"[output] JSON: {json_path}")


if __name__ == "__main__":
    main()
