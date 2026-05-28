"""
公共工具模块
============
RequestInfo 数据类、CSV 读写辅助、日志配置、通用计算函数。
"""

import csv
import json
import logging
import os
import sys
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

# 提升 CSV 字段大小限制，避免包含大 JSON 的字段触发 _csv.Error
csv.field_size_limit(sys.maxsize)


# ============================================================================
# 日志
# ============================================================================


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """配置并返回 refactor_test_aic 的根 logger。"""
    logger = logging.getLogger("refactor_test_aic")
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "[%(asctime)s][%(levelname)s] %(message)s", datefmt="%H:%M:%S"
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


logger = setup_logging()


# ============================================================================
# 数据类
# ============================================================================


@dataclass
class RequestInfo:
    input_length: int
    past_kv_length: int


# ============================================================================
# 目录 / 路径
# ============================================================================


def ensure_dir(path: str) -> str:
    """确保目录存在，返回路径本身。"""
    os.makedirs(path, exist_ok=True)
    return path


# ============================================================================
# CSV 读写
# ============================================================================


def read_csv_rows(csv_path: str) -> tuple[list[str], list[list[str]]]:
    """读取 CSV，返回 (headers, rows)。"""
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        headers = next(reader)
        rows = [row for row in reader]
    return headers, rows


def write_csv(csv_path: str, headers: list[str], rows: list[list]) -> None:
    """写出 CSV。"""
    ensure_dir(os.path.dirname(csv_path) or ".")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
    logger.info(f"已写入: {csv_path}  ({len(rows)} 行)")


# ============================================================================
# request_infos 解析
# ============================================================================


def parse_request_infos(request_infos_str: str) -> Optional[List[RequestInfo]]:
    """
    将 JSON 字符串解析为 RequestInfo 列表。
    解析失败返回 None。
    """
    try:
        req_dicts = json.loads(request_infos_str)
        if not isinstance(req_dicts, list):
            return None
        return [RequestInfo(d["input_length"], d["past_kv_length"]) for d in req_dicts]
    except Exception:
        return None


def request_infos_signature(request_infos_str: str) -> Optional[str]:
    """
    将 request_infos JSON 字符串标准化为可哈希的签名字符串。
    用于判断两个 batch 的请求组成是否完全一致。
    先按 (input_length, past_kv_length) 排序，再序列化。
    """
    try:
        req_dicts = json.loads(request_infos_str)
        if not isinstance(req_dicts, list):
            return None
        # 按 (input_length, past_kv_length) 排序后序列化
        sorted_reqs = sorted(
            req_dicts, key=lambda d: (d["input_length"], d["past_kv_length"])
        )
        return json.dumps(sorted_reqs, separators=(",", ":"), sort_keys=True)
    except Exception:
        return None


# ============================================================================
# 通用计算
# ============================================================================

GLM5_MODEL_MARKERS = ("glm-5", "glm5")

GLM5_DSA_DIMS = {
    "hidden_size": 6144,
    "q_lora_rank": 2048,
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 192,
    "qk_rope_head_dim": 64,
    "v_head_dim": 256,
    "index_topk": 2048,
    "index_head_dim": 128,
    "index_n_heads": 32,
}
GLM5_DSA_LOCAL_NUM_HEADS = 8


def is_glm5_model(model_name: str = "", model_path: str = "") -> bool:
    """Return whether the configured model is GLM5."""
    model_id = f"{model_name} {model_path}".lower().replace("_", "-")
    return any(marker in model_id for marker in GLM5_MODEL_MARKERS)


def ctx_attn_flops_ratio_with_avg(reqs: List[RequestInfo]) -> float:
    """
    计算 context attention 的实际 FLOPs 与使用均值估算的 FLOPs 之比。
    用于 prefill 阶段的 seq_imbalance_correction_scale。
    """
    if len(reqs) == 1:
        return 1.0
    mean_past = np.mean([r.past_kv_length for r in reqs])
    mean_input = np.mean([r.input_length for r in reqs])
    avg_flops = (mean_past + mean_past + mean_input) * mean_input / 2 * len(reqs)

    actual_flops = 0.0
    for r in reqs:
        actual_flops += (
            (r.past_kv_length + r.past_kv_length + r.input_length) * r.input_length / 2
        )

    return actual_flops / avg_flops if avg_flops > 0 else 1.0


def glm5_dsa_context_module_flops(
    batch_size: float,
    query_len: float,
    prefix: float,
    num_heads: int = GLM5_DSA_LOCAL_NUM_HEADS,
) -> float:
    """Compute GLM5 DSA context module work with the AIC sparse-attn formula."""
    b = float(batch_size)
    s = float(query_len)
    prefix = float(prefix)
    if b <= 0 or s <= 0:
        return 0.0

    hidden_size = GLM5_DSA_DIMS["hidden_size"]
    q_lora = GLM5_DSA_DIMS["q_lora_rank"]
    kv_lora = GLM5_DSA_DIMS["kv_lora_rank"]
    qk_nope = GLM5_DSA_DIMS["qk_nope_head_dim"]
    qk_rope = GLM5_DSA_DIMS["qk_rope_head_dim"]
    v_dim = GLM5_DSA_DIMS["v_head_dim"]
    index_topk = GLM5_DSA_DIMS["index_topk"]
    index_head_dim = GLM5_DSA_DIMS["index_head_dim"]
    index_n_heads = GLM5_DSA_DIMS["index_n_heads"]

    tokens = b * s
    full_s = prefix + s
    qk_head_dim = qk_nope + qk_rope
    attn_head_dim = kv_lora + qk_rope
    proj_out = q_lora + kv_lora + qk_rope + index_head_dim

    gemm_group_ops = (
        2 * tokens * hidden_size * proj_out
        + 2 * tokens * q_lora * (num_heads * qk_head_dim)
        + 2 * tokens * q_lora * (index_n_heads * index_head_dim)
        + 2 * tokens * hidden_size * index_n_heads
        + 2 * tokens * (num_heads * v_dim) * hidden_size
        + 2 * num_heads * tokens * qk_nope * kv_lora
        + 2 * num_heads * tokens * kv_lora * v_dim
    )

    if full_s <= index_topk:
        indexer_logits_ops = 0.0
    else:
        indexer_logits_ops = 2 * tokens * index_n_heads * index_head_dim * full_s

    if full_s <= index_topk:
        total_kv_pairs = b * (
            full_s * (full_s + 1.0) - prefix * (prefix + 1.0)
        ) / 2.0
    elif prefix >= index_topk:
        total_kv_pairs = tokens * index_topk
    else:
        ramp_pairs = b * (
            index_topk * (index_topk + 1.0) - prefix * (prefix + 1.0)
        ) / 2.0
        sat_pairs = b * (full_s - index_topk) * index_topk
        total_kv_pairs = ramp_pairs + sat_pairs

    sparse_attn_ops = 2 * num_heads * (attn_head_dim + kv_lora) * total_kv_pairs
    return float(gemm_group_ops + indexer_logits_ops + sparse_attn_ops)


def glm5_dsa_context_module_flops_ratio_with_avg(reqs: List[RequestInfo]) -> float:
    """
    Compute the seq-imbalance correction using GLM5 sparse DSA module work.

    This keeps the old correction mechanism, but replaces the full-attention
    FLOPs proxy with the same GLM5 DSA context-module work terms used by AIC:
    projection/indexer GEMMs, MQA indexer logits, and sparse top-k attention.
    """
    if len(reqs) == 1:
        return 1.0

    mean_past = np.mean([r.past_kv_length for r in reqs])
    mean_input = np.mean([r.input_length for r in reqs])
    avg_flops = glm5_dsa_context_module_flops(len(reqs), mean_input, mean_past)

    actual_flops = 0.0
    for r in reqs:
        actual_flops += glm5_dsa_context_module_flops(
            1,
            r.input_length,
            r.past_kv_length,
        )

    return actual_flops / avg_flops if avg_flops > 0 else 1.0


def prefill_seq_imbalance_correction(
    reqs: List[RequestInfo],
    model_name: str = "",
    model_path: str = "",
) -> float:
    """Compute the prefill seq imbalance correction for the configured model."""
    if is_glm5_model(model_name, model_path):
        return glm5_dsa_context_module_flops_ratio_with_avg(reqs)
    return ctx_attn_flops_ratio_with_avg(reqs)
