"""
阶段 2：AIC 时延预估
=====================
对 batches_output.csv 中的每条 batch 调用 AIConfigurator SDK 估算单步时延，
将结果按 prefill / decode 分开输出。

输入:  {data_dir}/csv/batches_output.csv
输出:  {data_dir}/estimation/batches_output_with_aic_prefill.csv
       {data_dir}/estimation/batches_output_with_aic_decode.csv
"""

import argparse
import csv
import json
import os
import traceback
from typing import List

import numpy as np

from .config import (
    AIC_BACKEND,
    AIC_SYSTEM,
    AIC_VERSION,
    BACKEND_NAME,
    DATA_DIR,
    DECODE_CORRECTION_FACTOR,
    MODEL_CONFIG_KWARGS,
    MODEL_NAME,
    MODEL_PATH,
    PREFILL_CORRECTION_FACTOR,
    SUBDIR_CSV,
    SUBDIR_ESTIMATION,
    USE_POLYNOMIAL_FIT,
    get_output_dir,
    model_supports_polynomial_fit,
)
from .utils import RequestInfo, logger, prefill_seq_imbalance_correction, write_csv


# ============================================================================
# AIC Session 初始化（延迟导入，避免无 GPU 环境下报错）
# ============================================================================

_session = None  # 单例


def _get_session():
    """延迟初始化 AIC InferenceSession 单例。"""
    global _session
    if _session is not None:
        return _session

    from aiconfigurator.sdk import config as aic_cfg, models
    from aiconfigurator.sdk.backends.factory import get_backend
    from aiconfigurator.sdk.common import DatabaseMode
    from aiconfigurator.sdk.inference_session import InferenceSession
    from aiconfigurator.sdk.perf_database import get_database, get_systems_paths

    logger.info("[Stage 2] 正在初始化 AIC session ...")

    database = get_database(
        system=AIC_SYSTEM,
        backend=AIC_BACKEND,
        version=AIC_VERSION,
        systems_paths=get_systems_paths(),
    )
    database.set_default_database_mode(DatabaseMode.SILICON)

    # 禁用 inner_only 以支持更灵活的输入
    db_nearest_1d_point_helper = database._nearest_1d_point_helper

    def wrapped_nearest_1d_point_helper(x, values, inner_only=False):
        return db_nearest_1d_point_helper(x, values, inner_only)

    database._nearest_1d_point_helper = wrapped_nearest_1d_point_helper

    model_config = aic_cfg.ModelConfig(**MODEL_CONFIG_KWARGS)
    logger.debug(f"Model config: {model_config}")
    model = models.get_model(
        model_path=MODEL_PATH, model_config=model_config, backend_name=BACKEND_NAME
    )
    logger.debug(f"Model config: {model.config}")
    _session = InferenceSession(
        model=model,
        backend=get_backend(BACKEND_NAME),
        database=database,
    )
    logger.info("[Stage 2] AIC session 初始化完成")
    return _session


# ============================================================================
# 单 batch 时延估算
# ============================================================================


def estimate_batch_latency(reqs: List[RequestInfo], is_decode: bool) -> float:
    """
    调用 AIC SDK 估算一个 batch 的单步时延 (ms)。
    """
    from aiconfigurator.sdk.config import RuntimeConfig

    session = _get_session()

    if is_decode:
        isl = int(np.mean([r.past_kv_length for r in reqs]))
        runtime_config = RuntimeConfig(batch_size=len(reqs), isl=isl, osl=2)
        summary = session.run_static(runtime_config, mode="static_gen")
        latency_dict = summary.get_generation_latency_dict()
    else:
        mean_past = np.mean([r.past_kv_length for r in reqs])
        mean_input = np.mean([r.input_length for r in reqs])
        isl = int(mean_past + mean_input)
        prefix = int(mean_past)
        correction = prefill_seq_imbalance_correction(reqs)
        use_polynomial_fit = USE_POLYNOMIAL_FIT and model_supports_polynomial_fit(
            MODEL_NAME, MODEL_PATH
        )
        polynomial_request_infos = reqs if use_polynomial_fit else None
        if correction >= 0.4 and correction != 1.0:
            runtime_config = RuntimeConfig(
                batch_size=len(reqs),
                isl=isl,
                prefix=prefix,
                osl=1,
                seq_imbalance_correction_scale=correction,
                request_infos=polynomial_request_infos,
                use_polynomial_fit=use_polynomial_fit,
            )
        else:
            runtime_config = RuntimeConfig(
                batch_size=len(reqs),
                isl=isl,
                prefix=prefix,
                osl=1,
                request_infos=polynomial_request_infos,
                use_polynomial_fit=use_polynomial_fit,
            )
        summary = session.run_static(runtime_config, mode="static_ctx")
        latency_dict = summary.get_context_latency_dict()

    total_ms = sum(latency_dict.values())

    # 校正系数
    if is_decode:
        total_ms *= DECODE_CORRECTION_FACTOR
    else:
        total_ms *= PREFILL_CORRECTION_FACTOR

    if summary.check_oom():
        logger.warning("OOM detected during estimation.")
        return float("inf")

    return total_ms


# ============================================================================
# 主逻辑
# ============================================================================


def run_aic_estimation(data_dir: str) -> tuple[str, str]:
    """
    读取 CSV，逐行调用 AIC 估算，输出 prefill / decode 两份结果 CSV。

    Returns:
        (prefill_csv_path, decode_csv_path)
    """
    csv_dir = get_output_dir(data_dir, SUBDIR_CSV)
    input_csv = os.path.join(csv_dir, "batches_output.csv")
    out_dir = get_output_dir(data_dir, SUBDIR_ESTIMATION)

    logger.info(f"[Stage 2] 开始 AIC 时延预估")
    logger.info(f"  输入: {input_csv}")
    logger.info(f"  输出目录: {out_dir}")

    prefill_items = []
    decode_items = []

    with open(input_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                latency_raw = float(row["latency"])
            except (ValueError, KeyError):
                continue

            batch_type = row["batch_type"].strip().lower()
            request_infos_str = row["request_infos"]
            case_id = row.get("case_id", "")

            try:
                req_dicts = json.loads(request_infos_str)
                reqs = [
                    RequestInfo(d["input_length"], d["past_kv_length"])
                    for d in req_dicts
                ]
            except Exception:
                reqs = None

            item = {
                "case_id": case_id,
                "batch_type": batch_type,
                "batch_size": len(reqs) if reqs else -1,
                "avg_input_length": float(np.mean([r.input_length for r in reqs]))
                if reqs
                else -1.0,
                "avg_past_kv_length": float(np.mean([r.past_kv_length for r in reqs]))
                if reqs
                else -1.0,
                "latency_raw": latency_raw,
                "reqs": reqs,
                "request_infos_str": request_infos_str,
            }

            if batch_type == "prefill":
                prefill_items.append(item)
            elif batch_type == "decode":
                decode_items.append(item)

    # 错误日志文件：记录完整堆栈和 case 信息
    error_log_path = os.path.join(out_dir, "error.log")
    error_count = 0
    error_fh = open(error_log_path, "w", encoding="utf-8")

    def _log_error(item, exc, tb_str):
        """将报错 case 的完整信息和堆栈写入 error.log。"""
        nonlocal error_count
        error_count += 1
        error_fh.write(f"{'=' * 70}\n")
        error_fh.write(f"Error #{error_count}\n")
        error_fh.write(f"  case_id        : {item['case_id']}\n")
        error_fh.write(f"  batch_type     : {item['batch_type']}\n")
        error_fh.write(f"  batch_size     : {item['batch_size']}\n")
        error_fh.write(f"  avg_input_len  : {item['avg_input_length']}\n")
        error_fh.write(f"  avg_past_kv_len: {item['avg_past_kv_length']}\n")
        error_fh.write(f"  request_infos  : {item['request_infos_str'][:200]}...\n")
        error_fh.write(f"  exception      : {exc}\n")
        error_fh.write(f"Traceback:\n{tb_str}\n\n")
        error_fh.flush()

    def _process(items, is_decode):
        processed = []
        total = len(items)
        for idx, item in enumerate(items, 1):
            if idx % 200 == 0 or idx == total:
                logger.info(f"  {'decode' if is_decode else 'prefill'}: {idx}/{total}")

            latency_ms = item["latency_raw"] * 1000.0
            if item["reqs"] is None:
                estimated = -1.0
            else:
                try:
                    estimated = estimate_batch_latency(item["reqs"], is_decode)
                    if estimated == float("inf"):
                        estimated = -1.0
                except Exception as e:
                    tb_str = traceback.format_exc()
                    logger.warning(f"估算失败 (case_id={item['case_id']}): {e}")
                    _log_error(item, e, tb_str)
                    estimated = -1.0

            ape = -1.0
            if estimated >= 0 and latency_ms > 0:
                ape = abs((estimated - latency_ms) / latency_ms) * 100.0

            processed.append(
                {
                    "ape": ape,
                    "data": [
                        item["case_id"],
                        item["batch_type"],
                        str(item["batch_size"]),
                        f"{item['avg_input_length']:.3f}"
                        if item["avg_input_length"] >= 0
                        else "-1.0",
                        f"{item['avg_past_kv_length']:.3f}"
                        if item["avg_past_kv_length"] >= 0
                        else "-1.0",
                        f"{latency_ms:.3f}",
                        f"{estimated:.3f}" if estimated >= 0 else "-1.0",
                        f"{ape:.3f}"
                        if isinstance(ape, float) and not np.isinf(ape)
                        else "inf",
                        item["request_infos_str"],
                    ],
                }
            )

        processed.sort(
            key=lambda x: (
                -(x["ape"])
                if x["ape"] >= 0 and not np.isinf(x["ape"])
                else float("inf"),
            )
        )
        return [e["data"] for e in processed]

    out_headers = [
        "case_id",
        "batch_type",
        "batch_size",
        "avg_input_length",
        "avg_past_kv_length",
        "latency_ms",
        "estimated_latency_ms",
        "APE(%)",
        "request_infos",
    ]

    prefill_csv = os.path.join(out_dir, "batches_output_with_aic_prefill.csv")
    decode_csv = os.path.join(out_dir, "batches_output_with_aic_decode.csv")

    logger.info(f"  处理 prefill ({len(prefill_items)} 条) ...")
    write_csv(prefill_csv, out_headers, _process(prefill_items, is_decode=False))

    logger.info(f"  处理 decode ({len(decode_items)} 条) ...")
    write_csv(decode_csv, out_headers, _process(decode_items, is_decode=True))

    error_fh.close()
    if error_count > 0:
        logger.warning(f"[Stage 2] 共 {error_count} 条估算报错，详见: {error_log_path}")
    else:
        # 无错误则删除空文件
        os.remove(error_log_path)

    logger.info("[Stage 2] 完成")
    return prefill_csv, decode_csv


# ============================================================================
# 命令行入口
# ============================================================================


def main():
    ap = argparse.ArgumentParser(description="阶段 2: AIC 时延预估")
    ap.add_argument("--data-dir", type=str, default=DATA_DIR, help="数据根目录")
    args = ap.parse_args()
    run_aic_estimation(args.data_dir)


if __name__ == "__main__":
    main()
