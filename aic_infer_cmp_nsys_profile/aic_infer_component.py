"""AIC per-op latency estimator — 通过 case_id 从 CSV 读取 batch 信息并输出逐 op 预估时延。

Usage:
    # 指定 case_id，自动从 batches_output.csv 读取并推断 prefill/decode
    python aic_infer_component.py --data-dir /path/to/data --case-id 42

    # 也可手动指定 phase 和参数（向后兼容）
    python aic_infer_component.py --phase prefill --batch-size 1 --isl 3786
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import aiconfigurator.sdk.operations as sdk_ops
from aiconfigurator.sdk import interpolation
from aiconfigurator.sdk.backends.factory import get_backend
from aiconfigurator.sdk.common import DatabaseMode
from aiconfigurator.sdk.config import ModelConfig, RuntimeConfig
from aiconfigurator.sdk.inference_session import InferenceSession
from aiconfigurator.sdk.models import get_model
from aiconfigurator.sdk.perf_database import get_database, get_systems_paths

# ---------------------------------------------------------------------------
# 复用 refactor_test_aic 的统一配置
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    AIC_BACKEND,
    AIC_SYSTEM,
    AIC_VERSION,
    ATTN_FLOPS_CORRELATION,
    BACKEND_NAME,
    DECODE_CORRECTION_FACTOR,
    MODEL_CONFIG_KWARGS,
    MODEL_NAME,
    MODEL_PATH,
    PREFILL_CORRECTION_FACTOR,
    SUBDIR_CSV,
)
from utils import RequestInfo, prefill_seq_imbalance_correction


# ============================================================================
# AIC Session 初始化（延迟单例）
# ============================================================================

_session: Optional[InferenceSession] = None
_model = None
_database = None


def _patch_nearest_1d_point_helper(database):
    """Keep old refactor extrapolation behavior across AIC SDK versions."""
    original = getattr(
        interpolation.nearest_1d_point_helper,
        "_refactor_original",
        interpolation.nearest_1d_point_helper,
    )

    def wrapped_nearest_1d_point_helper(x, values, inner_only=False):
        return original(x, values, inner_only)

    wrapped_nearest_1d_point_helper._refactor_original = original
    interpolation.nearest_1d_point_helper = wrapped_nearest_1d_point_helper

    database_helper = getattr(database, "_nearest_1d_point_helper", None)
    if database_helper is not None:

        def wrapped_database_nearest_1d_point_helper(x, values, inner_only=False):
            return database_helper(x, values, inner_only)

        database._nearest_1d_point_helper = wrapped_database_nearest_1d_point_helper


def _nearest_1d_point(database, x, values, inner_only=False):
    database_helper = getattr(database, "_nearest_1d_point_helper", None)
    if database_helper is not None:
        return database_helper(x, values, inner_only)
    return interpolation.nearest_1d_point_helper(x, values, inner_only)


def _get_session():
    """延迟初始化 AIC InferenceSession 单例，复用 stage2 的初始化逻辑。"""
    global _session, _model, _database
    if _session is not None:
        return _session, _model, _database

    print("[init] 正在初始化 AIC session ...")

    _database = get_database(
        system=AIC_SYSTEM,
        backend=AIC_BACKEND,
        version=AIC_VERSION,
        systems_paths=get_systems_paths(),
    )
    _database.set_default_database_mode(DatabaseMode.SILICON)

    # 禁用 inner_only 以支持更灵活的输入（与 stage2 一致）。
    _patch_nearest_1d_point_helper(_database)

    model_config = ModelConfig(**MODEL_CONFIG_KWARGS)
    _model = get_model(
        model_path=MODEL_PATH, model_config=model_config, backend_name=BACKEND_NAME
    )
    _session = InferenceSession(
        model=_model,
        backend=get_backend(BACKEND_NAME),
        database=_database,
    )
    print("[init] AIC session 初始化完成")
    return _session, _model, _database


# ============================================================================
# CSV 读取
# ============================================================================


def _load_csv_row_by_case_id(csv_path: str, case_id: int) -> dict:
    """从 batches_output.csv 按 case_id 查找行。"""
    # 提升 CSV 字段大小限制
    csv.field_size_limit(sys.maxsize)
    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                if int(float(row.get("case_id", ""))) == case_id:
                    return row
            except (TypeError, ValueError):
                continue
    raise FileNotFoundError(f"case_id={case_id} not found in {csv_path}")


def _parse_request_infos(request_infos_str: str) -> List[RequestInfo]:
    """将 CSV 中的 request_infos JSON 字符串解析为 RequestInfo 列表。"""
    req_dicts = json.loads(request_infos_str)
    return [RequestInfo(d["input_length"], d["past_kv_length"]) for d in req_dicts]


# ============================================================================
# 核心推理逻辑（复用 stage2 的 RuntimeConfig 构建方式）
# ============================================================================


def estimate_batch_per_op(
    reqs: List[RequestInfo], is_decode: bool
) -> tuple[Dict[str, float], RuntimeConfig, float]:
    """
    对一个 batch 调用 AIC SDK 估算，返回:
        - latency_dict: 每个 op 名称 -> 时延 (ms)
        - runtime_config: 实际使用的 RuntimeConfig
        - total_ms: 校正后的总时延 (ms)
    """
    session, model, database = _get_session()

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
        correction = (
            prefill_seq_imbalance_correction(reqs, MODEL_NAME, MODEL_PATH)
            if ATTN_FLOPS_CORRELATION
            else 1.0
        )
        if correction >= 0.4 and correction != 1.0:
            runtime_config = RuntimeConfig(
                batch_size=len(reqs),
                isl=isl,
                prefix=prefix,
                osl=1,
                seq_imbalance_correction_scale=correction,
            )
        else:
            runtime_config = RuntimeConfig(
                batch_size=len(reqs),
                isl=isl,
                prefix=prefix,
                osl=1,
            )
        summary = session.run_static(runtime_config, mode="static_ctx")
        latency_dict = summary.get_context_latency_dict()

    total_ms = sum(latency_dict.values())
    if is_decode:
        total_ms *= DECODE_CORRECTION_FACTOR
    else:
        total_ms *= PREFILL_CORRECTION_FACTOR

    if summary.check_oom():
        print("[warn] OOM detected during estimation.")

    return latency_dict, runtime_config, total_ms


# ============================================================================
# 辅助打印
# ============================================================================


def _safe_latency(entry):
    if isinstance(entry, dict):
        return float(entry.get("latency", 0.0))
    return float(entry)


def _safe_energy(entry):
    if isinstance(entry, dict):
        return float(entry.get("energy", 0.0))
    return 0.0


def _print_decode_moe_interp_debug(runtime_config, latency_dict):
    """打印 decode 阶段 MoE 插值的 debug 信息。"""
    _, model, database = _get_session()

    query_x = (
        runtime_config.batch_size
        * (getattr(model, "_nextn", 0) + 1)
        * runtime_config.beam_width
    )
    moe_debug_rows = []

    for op in model.generation_ops:
        if not isinstance(op, sdk_ops.MoE):
            continue

        query_tokens = int(query_x * getattr(op, "_attention_dp_size", 1))
        quant_mode = op._quant_mode
        workload_distribution = op._workload_distribution
        topk = op._topk
        num_experts = op._num_experts
        hidden_size = op._hidden_size
        inter_size = op._inter_size
        moe_tp_size = op._moe_tp_size
        moe_ep_size = op._moe_ep_size
        scale_factor = float(op._scale_factor)
        moe_backend = getattr(op, "_moe_backend", None)

        if database.backend == "sglang" and moe_backend == "deepep_moe":
            moe_data = database._wideep_generation_moe_data
        else:
            moe_data = database._moe_data
        moe_data.raise_if_not_loaded()

        used_workload_distribution = (
            workload_distribution
            if workload_distribution in moe_data[quant_mode]
            else "uniform"
        )
        moe_dict = moe_data[quant_mode][used_workload_distribution][topk][num_experts][
            hidden_size
        ][inter_size][moe_tp_size][moe_ep_size]
        token_points = sorted(moe_dict.keys())

        overflow = query_tokens > token_points[-1]
        if overflow:
            left_tok, right_tok = token_points[-2], token_points[-1]
            interp_raw = None
            interp_scaled = None
        else:
            left_tok, right_tok = _nearest_1d_point(
                database,
                query_tokens, token_points, inner_only=False
            )
            interp_result = database._interp_1d(
                [left_tok, right_tok],
                [moe_dict[left_tok], moe_dict[right_tok]],
                query_tokens,
            )
            interp_raw = _safe_latency(interp_result)
            interp_scaled = interp_raw * scale_factor

        left_entry = moe_dict[left_tok]
        right_entry = moe_dict[right_tok]
        op_total_latency = float(latency_dict.get(op._name, 0.0))

        moe_debug_rows.append(
            {
                "op_name": op._name,
                "query_num_tokens": query_tokens,
                "used_workload_distribution": used_workload_distribution,
                "left_token": left_tok,
                "right_token": right_tok,
                "left_latency_ms_raw": _safe_latency(left_entry),
                "right_latency_ms_raw": _safe_latency(right_entry),
                "left_energy_wms_raw": _safe_energy(left_entry),
                "right_energy_wms_raw": _safe_energy(right_entry),
                "interp_latency_ms_raw": interp_raw,
                "interp_latency_ms_scaled": interp_scaled,
                "op_scale_factor": scale_factor,
                "op_total_latency_ms_from_summary": op_total_latency,
                "overflow_above_max_collected_tokens": overflow,
                "max_collected_token": token_points[-1],
            }
        )

    if moe_debug_rows:
        print()
        print("=== Decode MoE interpolation debug ===")
        print(json.dumps(moe_debug_rows, indent=2, default=str))


def _print_op_table(latency_dict, is_decode, phase_label, threshold_pct=5.0):
    """对齐 AIC 结果的标准输出: 5 列 (op/module, 层数, 总ms, 占比, per-layer ms).

    - 层数        : op._scale_factor (该 op 在多少层里重复)
    - per-layer   : 总ms / 层数
    - 占比        : 总ms / sum(总ms)
    - ``★`` 行内标记: 占比 >= threshold_pct (默认 5%), 即做 timeline 对齐时要关注的 op
    - TOTAL 行    : 总时间 + 整体 per-layer (= 总时间 / num_layers)
    """
    _, model, _ = _get_session()
    ops = model.generation_ops if is_decode else model.context_ops
    sf = {op._name: float(getattr(op, "_scale_factor", 1) or 1) for op in ops}

    total = sum(_safe_latency(e) for e in latency_dict.values())
    num_layers = max(sf.values()) if sf else 1.0

    rows = []
    for name, entry in latency_dict.items():
        val = _safe_latency(entry)
        layers = sf.get(name, 1.0) or 1.0
        pct = (val / total * 100.0) if total else 0.0
        rows.append((name, layers, val, pct, val / layers if layers else 0.0))
    rows.sort(key=lambda r: -r[3])

    print("=== %s AIC per-op 标准表 (★ = 占比 >= %.0f%%) ===" % (phase_label, threshold_pct))
    print("%-32s %6s %12s %8s %12s" % ("op/module", "层数", "总(ms)", "占比", "per-layer"))
    print("-" * 76)
    for name, layers, val, pct, per_layer in rows:
        mark = "★" if pct >= threshold_pct else " "
        print("%s %-30s %6.0f %12.4f %7.2f%% %12.4f" % (mark, name, layers, val, pct, per_layer))
    print("-" * 76)
    print("  %-30s %6.0f %12.4f %7.2f%% %12.4f"
          % ("TOTAL", num_layers, total, 100.0, total / num_layers if num_layers else 0.0))


def _print_results(
    case_id: Optional[int],
    is_decode: bool,
    reqs: List[RequestInfo],
    latency_dict: Dict[str, float],
    runtime_config: RuntimeConfig,
    total_ms: float,
    measured_latency_ms: Optional[float] = None,
):
    """格式化打印单 case 的预估结果。"""
    phase_label = "Decode" if is_decode else "Prefill"
    mode = "static_gen" if is_decode else "static_ctx"

    print("=" * 70)
    if case_id is not None:
        print(f"Case ID       : {case_id}")
    print(f"Phase         : {phase_label} ({mode})")
    print(f"Batch Size    : {len(reqs)}")
    print(f"Avg Input Len : {np.mean([r.input_length for r in reqs]):.1f}")
    print(f"Avg Past KV   : {np.mean([r.past_kv_length for r in reqs]):.1f}")
    print(
        f"RuntimeConfig : bs={runtime_config.batch_size}, isl={runtime_config.isl}, "
        f"osl={runtime_config.osl}, prefix={getattr(runtime_config, 'prefix', 'N/A')}"
    )
    if hasattr(runtime_config, "seq_imbalance_correction_scale"):
        print(f"Correction    : {runtime_config.seq_imbalance_correction_scale}")
    print(f"Estimated Total: {total_ms:.3f} ms")
    if measured_latency_ms is not None and measured_latency_ms > 0:
        ape = abs((total_ms - measured_latency_ms) / measured_latency_ms) * 100.0
        print(f"Measured Total : {measured_latency_ms:.3f} ms")
        print(f"APE            : {ape:.2f}%")
    print()
    _print_op_table(latency_dict, is_decode, phase_label)

    if is_decode:
        _print_decode_moe_interp_debug(runtime_config, latency_dict)


# ============================================================================
# 主函数
# ============================================================================


def run_from_case_id(data_dir: str, case_id: int, csv_path: Optional[str] = None):
    """通过 case_id 从 CSV 读取 batch 信息，自动推断 prefill/decode 并输出逐 op 预估时延。"""
    if csv_path is None:
        csv_path = os.path.join(data_dir, SUBDIR_CSV, "batches_output.csv")

    print(f"[load] CSV: {csv_path}")
    print(f"[load] Looking for case_id={case_id}")

    row = _load_csv_row_by_case_id(csv_path, case_id)
    batch_type = row["batch_type"].strip().lower()
    is_decode = batch_type == "decode"
    reqs = _parse_request_infos(row["request_infos"])
    measured_latency_s = float(row.get("latency", 0))
    measured_latency_ms = measured_latency_s * 1000.0

    print(
        f"[load] batch_type={batch_type}, batch_size={len(reqs)}, "
        f"measured_latency={measured_latency_ms:.3f}ms"
    )

    latency_dict, runtime_config, total_ms = estimate_batch_per_op(reqs, is_decode)

    _print_results(
        case_id=case_id,
        is_decode=is_decode,
        reqs=reqs,
        latency_dict=latency_dict,
        runtime_config=runtime_config,
        total_ms=total_ms,
        measured_latency_ms=measured_latency_ms,
    )
    return latency_dict, total_ms


def run_from_manual(phase: str, batch_size: int, isl: int, osl: int, prefix: int = 0):
    """手动指定参数运行（向后兼容）。"""
    is_decode = phase == "decode"
    if is_decode:
        reqs = [
            RequestInfo(input_length=1, past_kv_length=isl) for _ in range(batch_size)
        ]
    else:
        reqs = [
            RequestInfo(input_length=isl - prefix, past_kv_length=prefix)
            for _ in range(batch_size)
        ]

    latency_dict, runtime_config, total_ms = estimate_batch_per_op(reqs, is_decode)

    _print_results(
        case_id=None,
        is_decode=is_decode,
        reqs=reqs,
        latency_dict=latency_dict,
        runtime_config=runtime_config,
        total_ms=total_ms,
    )
    return latency_dict, total_ms


def run_batch_case_ids(
    data_dir: str,
    case_ids: List[int],
    csv_path: Optional[str] = None,
):
    """批量运行多个 case_id，汇总输出。"""
    results = []
    for cid in case_ids:
        print(f"\n{'#' * 70}")
        print(f"# Processing case_id = {cid}")
        print(f"{'#' * 70}")
        try:
            latency_dict, total_ms = run_from_case_id(data_dir, cid, csv_path)
            results.append(
                {"case_id": cid, "total_ms": total_ms, "per_op": latency_dict}
            )
        except Exception as e:
            print(f"[error] case_id={cid}: {e}")
            results.append({"case_id": cid, "error": str(e)})
    return results


def main():
    ap = argparse.ArgumentParser(
        description="AIC per-op 时延预估工具 — 支持通过 case_id 或手动参数运行"
    )
    sub = ap.add_subparsers(dest="command", help="运行模式")

    # 子命令: case — 通过 case_id 从 CSV 读取
    p_case = sub.add_parser("case", help="通过 case_id 从 CSV 自动读取 batch 信息")
    p_case.add_argument("--data-dir", type=str, required=True, help="数据根目录")
    p_case.add_argument(
        "--case-id",
        type=int,
        nargs="+",
        required=True,
        help="一个或多个 case_id",
    )
    p_case.add_argument(
        "--csv-path",
        type=str,
        default=None,
        help="CSV 文件路径（默认 data-dir/csv/batches_output.csv）",
    )

    # 子命令: manual — 手动指定参数
    p_manual = sub.add_parser("manual", help="手动指定 phase/batch_size/isl 等参数")
    p_manual.add_argument(
        "--phase", type=str, choices=["prefill", "decode"], default="prefill"
    )
    p_manual.add_argument("--batch-size", type=int, default=1)
    p_manual.add_argument("--isl", type=int, default=3786)
    p_manual.add_argument("--osl", type=int, default=2)
    p_manual.add_argument("--prefix", type=int, default=0)

    args = ap.parse_args()

    if args.command == "case":
        if len(args.case_id) == 1:
            run_from_case_id(args.data_dir, args.case_id[0], args.csv_path)
        else:
            run_batch_case_ids(args.data_dir, args.case_id, args.csv_path)
    elif args.command == "manual":
        run_from_manual(args.phase, args.batch_size, args.isl, args.osl, args.prefix)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
