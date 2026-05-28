"""
阶段 3：准确度分析与可视化
===========================
合并 draw_aic_accuracy.py 和 bucket_signed_error.py 的功能:
  - 按 batch_size 聚合 MAPE 统计 + 绘图
  - 有符号误差桶分布 + 绘图

输入:  {data_dir}/estimation/batches_output_with_aic_{prefill,decode}.csv
输出:  {data_dir}/accuracy/ 下的 CSV 和 PNG 文件
"""

import argparse
import math
import os
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import (
    DATA_DIR,
    SUBDIR_ACCURACY,
    SUBDIR_ESTIMATION,
    SUBDIR_SIGNED_ERROR,
    get_output_dir,
)
from .utils import logger


# ============================================================================
# 1) MAPE 统计 + 绘图（来自 draw_aic_accuracy.py）
# ============================================================================


def analyze_mape(
    csv_path: str,
    title: str,
    plot_path: str,
    stats_path: str,
    count_threshold: int = 10,
) -> None:
    """按 batch_size 分组统计 MAPE，保存统计 CSV 和折线图。"""
    df = pd.read_csv(csv_path)
    df["APE(%)"] = df["APE(%)"].replace("inf", np.nan)
    df["APE(%)"] = pd.to_numeric(df["APE(%)"], errors="coerce")
    df = df[(df["APE(%)"] >= 0) & df["APE(%)"].notna()]
    df["batch_size"] = pd.to_numeric(df["batch_size"], errors="coerce")
    df = df[df["batch_size"].notna() & (df["batch_size"] >= 1)]
    df["batch_size"] = df["batch_size"].astype(int)

    if df.empty:
        logger.warning(f"无有效数据: {csv_path}")
        return

    overall_mean = df["APE(%)"].mean()
    overall_p90 = np.percentile(df["APE(%)"], 90)
    logger.info(f"  {title}: Mean MAPE={overall_mean:.3f}%, P90={overall_p90:.3f}%")

    stats = (
        df.groupby("batch_size")["APE(%)"]
        .agg(
            mean_mape="mean",
            min_mape="min",
            max_mape="max",
            std_mape="std",
            p90_mape=lambda x: np.percentile(x, 90),
            count="size",
        )
        .reset_index()
        .sort_values("batch_size")
        .round(3)
    )
    stats["std_mape"] = stats["std_mape"].fillna(0)

    os.makedirs(os.path.dirname(stats_path) or ".", exist_ok=True)
    stats.to_csv(stats_path, index=False)
    logger.info(f"  统计已保存: {stats_path}")

    # 找出 count >= threshold 的 top 5% worst batch_size
    sufficient = stats[stats["count"] >= count_threshold].sort_values(
        "mean_mape", ascending=False
    )
    if not sufficient.empty:
        n_top = max(1, math.ceil(len(sufficient) * 0.05))
        top = sufficient.head(n_top)
        phase = "Prefill" if "prefill" in title.lower() else "Decode"
        logger.info(
            f"  Top {n_top} worst batch sizes ({phase}, >= {count_threshold} samples):"
        )
        for _, row in top.iterrows():
            logger.info(
                f"    BS={int(row['batch_size'])}, Mean MAPE={row['mean_mape']:.3f}%, N={int(row['count'])}"
            )

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(
        stats["batch_size"],
        stats["mean_mape"],
        marker="o",
        linestyle="-",
        color="tab:blue",
        label="Mean MAPE",
    )
    plt.fill_between(
        stats["batch_size"],
        stats["min_mape"],
        stats["max_mape"],
        color="tab:blue",
        alpha=0.2,
        label="MAPE Range",
    )
    plt.xlabel("Batch Size")
    plt.ylabel("MAPE (%)")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(plot_path) or ".", exist_ok=True)
    plt.savefig(plot_path, dpi=300)
    plt.close()
    logger.info(f"  图表已保存: {plot_path}")


# ============================================================================
# 2) 有符号误差桶分析（来自 bucket_signed_error.py）
# ============================================================================


def _bucketize(err: float, step: int = 1, clip: int = 30) -> str:
    if not np.isfinite(err):
        return "nan"
    if clip and clip > 0:
        if err < -clip:
            return f"<-{clip}%"
        if err > clip:
            return f">{clip}%"
    b = int(np.floor(err / step) * step)
    return "0%" if b == 0 else f"{b}%"


def _ordered_buckets(step: int = 1, clip: int = 30) -> list[str]:
    buckets = [f"<-{clip}%"]
    for v in range(-clip, clip + 1, step):
        buckets.append("0%" if v == 0 else f"{v}%")
    buckets.append(f">{clip}%")
    return buckets


def _ordered_buckets_from_data(series: pd.Series, step: int = 1) -> list[str]:
    x = pd.to_numeric(series, errors="coerce")
    x = x[np.isfinite(x)]
    if x.empty:
        return ["0%"]
    mn = int(np.floor(float(x.min()) / step) * step)
    mx = int(np.floor(float(x.max()) / step) * step)
    return ["0%" if v == 0 else f"{v}%" for v in range(mn, mx + step, step)]


def _bucket_sign(bucket: str) -> int:
    if not isinstance(bucket, str):
        return 0
    if bucket.startswith("<-") or bucket.startswith("-"):
        return -1
    if bucket.startswith(">") or (bucket.endswith("%") and bucket[0].isdigit()):
        return 1
    return 0


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _build_diff_table(
    csv_paths: list[str], step: int = 1, clip: int = 30
) -> pd.DataFrame:
    """合并 CSV 并计算有符号百分比误差。"""
    actual_cands = ["latency_ms", "latency", "latency_raw"]
    aic_cands = ["estimated_latency_ms", "aic_latency_ms", "aic", "estimated"]

    dfs = []
    for p in csv_paths:
        if p and os.path.exists(p):
            tmp = pd.read_csv(p)
            tmp["source_csv"] = os.path.basename(p)
            dfs.append(tmp)
    if not dfs:
        raise FileNotFoundError(f"No CSV found in: {csv_paths}")

    df = pd.concat(dfs, ignore_index=True)
    actual_col = _pick_col(df, actual_cands)
    aic_col = _pick_col(df, aic_cands)
    if actual_col is None or aic_col is None:
        raise ValueError(f"Cannot find required columns. Available: {list(df.columns)}")

    actual = pd.to_numeric(df[actual_col], errors="coerce")
    if actual_col == "latency_raw":
        actual = actual * 1000.0
    aic = pd.to_numeric(df[aic_col], errors="coerce")

    out = pd.DataFrame()
    if "batch_type" in df.columns:
        out["batch_type"] = df["batch_type"].astype(str)
    else:
        out["batch_type"] = "unknown"
    for col in ["batch_size", "avg_input_length", "avg_past_kv_length"]:
        if col in df.columns:
            out[col] = pd.to_numeric(df[col], errors="coerce")
    out["latency_ms"] = actual
    out["aic_latency_ms"] = aic
    if "request_infos" in df.columns:
        out["request_infos"] = df["request_infos"].astype(str)
    out["source_csv"] = df["source_csv"].astype(str)
    out["signed_error_pct"] = (
        (out["aic_latency_ms"] - out["latency_ms"]) / out["latency_ms"] * 100.0
    )

    valid = (
        out["latency_ms"].notna()
        & np.isfinite(out["latency_ms"])
        & (out["latency_ms"] > 0)
        & out["aic_latency_ms"].notna()
        & np.isfinite(out["aic_latency_ms"])
        & (out["aic_latency_ms"] >= 0)
    )
    # 保留原始 case_id（即 JSONL 0-based 行号），不重新编号
    if "case_id" in df.columns:
        out["case_id"] = pd.to_numeric(df["case_id"], errors="coerce").astype(int)
    else:
        out["case_id"] = np.arange(len(df))
    out = out[valid].copy()
    out["bucket"] = out["signed_error_pct"].apply(
        lambda v: _bucketize(v, step=step, clip=clip)
    )
    return out


def _plot_error_buckets(
    df: pd.DataFrame,
    out_png: str,
    title: str,
    step: int = 1,
    clip: int = 30,
    annotate: bool = True,
) -> None:
    """绘制有符号误差桶柱状图。"""
    df = df.copy()
    if clip and clip > 0:
        order = _ordered_buckets(step=step, clip=clip)
        df["bucket"] = df["signed_error_pct"].apply(
            lambda v: _bucketize(v, step=step, clip=clip)
        )
    else:
        order = _ordered_buckets_from_data(df["signed_error_pct"], step=step)
        df["bucket"] = df["signed_error_pct"].apply(
            lambda v: _bucketize(v, step=step, clip=0)
        )

    df["bucket"] = pd.Categorical(df["bucket"], categories=order, ordered=True)
    counts = df.groupby("bucket", observed=False).size().reindex(order, fill_value=0)

    x = np.arange(len(order))
    y = counts.to_numpy()

    plt.figure(figsize=(14, 6))
    colors = []
    for b in order:
        s = _bucket_sign(b)
        colors.append("tab:green" if s < 0 else ("tab:red" if s > 0 else "tab:gray"))
    plt.bar(x, y, color=colors, alpha=0.85)
    plt.xlabel("Signed error bucket (%)")
    plt.ylabel("Cases (count)")
    plt.title(title)
    plt.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.xticks(x, order, rotation=45, ha="right")

    # 下采样 x 标签
    n = len(order)
    if n > 40:
        step_d = int(np.ceil(n / 40))
        keep = set(range(0, n, step_d)) | {0, n - 1}
        if "0%" in order:
            keep.add(order.index("0%"))
        new_labels = [order[i] if i in keep else "" for i in range(n)]
        plt.gca().set_xticklabels(new_labels, rotation=45, ha="right")

    handles = [
        plt.Line2D([0], [0], color="tab:green", lw=8, label="Negative: AIC faster"),
        plt.Line2D([0], [0], color="tab:red", lw=8, label="Positive: measured faster"),
        plt.Line2D([0], [0], color="tab:gray", lw=8, label="0%: tie"),
    ]
    plt.legend(handles=handles, loc="best")

    if annotate:
        y_max = float(y.max(initial=0))
        fs = 8 if n <= 80 else (7 if n <= 140 else 6)
        offset = max(y_max * 0.01, 0.8)
        for xi, yi in zip(x, y):
            if yi > 0:
                plt.text(
                    xi, yi + offset, str(int(yi)), ha="center", va="bottom", fontsize=fs
                )

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    plt.savefig(out_png, dpi=300)
    plt.close()


def analyze_signed_error(
    prefill_csv: str,
    decode_csv: str,
    out_dir: str,
    step: int = 1,
    clip: int = 30,
    clip_0: int = 0,
) -> None:
    """运行有符号误差桶分析，输出 CSV 和 PNG。"""
    csvs = [p for p in [prefill_csv, decode_csv] if p and os.path.exists(p)]
    if not csvs:
        logger.warning("无可用的 estimation CSV，跳过误差桶分析")
        return

    df = _build_diff_table(csvs, step=step, clip=clip)
    csv_out = os.path.join(out_dir, "aic_vs_measured_signed_error_cases.csv")
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(csv_out, index=False)
    logger.info(f"  误差表已保存: {csv_out} ({len(df)} 行)")

    # 总图
    _plot_error_buckets(
        df,
        os.path.join(out_dir, "aic_vs_measured_signed_error_cases.png"),
        "AIC vs Measured (all)",
        step=step,
        clip=clip,
    )

    # 按 batch_type 分图，clip=30 和 clip=0 各画一份
    for bt in ["prefill", "decode"]:
        sub = df[df["batch_type"].str.lower() == bt].copy()
        if sub.empty:
            continue
        for c in [clip, clip_0]:
            suffix = f"_{c}_{bt}.png"
            _plot_error_buckets(
                sub,
                os.path.join(out_dir, f"aic_vs_measured_signed_error_cases{suffix}"),
                f"AIC vs Measured ({bt}), clip={c}",
                step=step,
                clip=c,
            )

    logger.info(f"  误差桶图表已保存至: {out_dir}")


# ============================================================================
# 主入口：串联 MAPE + 误差桶
# ============================================================================


def run_accuracy_analysis(data_dir: str) -> None:
    """阶段 3 主函数。"""
    est_dir = get_output_dir(data_dir, SUBDIR_ESTIMATION)
    acc_dir = get_output_dir(data_dir, SUBDIR_ACCURACY)
    err_dir = get_output_dir(data_dir, SUBDIR_SIGNED_ERROR)

    prefill_csv = os.path.join(est_dir, "batches_output_with_aic_prefill.csv")
    decode_csv = os.path.join(est_dir, "batches_output_with_aic_decode.csv")

    logger.info(f"[Stage 3] 开始准确度分析")
    logger.info(f"  estimation 目录: {est_dir}")
    logger.info(f"  MAPE 输出目录: {acc_dir}")
    logger.info(f"  误差桶输出目录: {err_dir}")

    # MAPE 分析
    if os.path.exists(prefill_csv):
        analyze_mape(
            prefill_csv,
            "Prefill: Mean MAPE vs Batch Size",
            os.path.join(acc_dir, "prefill_mape_vs_bs.png"),
            os.path.join(acc_dir, "prefill_mape_stats.csv"),
        )
    if os.path.exists(decode_csv):
        analyze_mape(
            decode_csv,
            "Decode: Mean MAPE vs Batch Size",
            os.path.join(acc_dir, "decode_mape_vs_bs.png"),
            os.path.join(acc_dir, "decode_mape_stats.csv"),
        )

    # 误差桶分析（clip=30 + clip=0）—— 输出到独立的 signed_error/ 目录
    analyze_signed_error(prefill_csv, decode_csv, err_dir, step=1, clip=30, clip_0=0)

    logger.info("[Stage 3] 完成")


# ============================================================================
# 命令行入口
# ============================================================================


def main():
    ap = argparse.ArgumentParser(description="阶段 3: 准确度分析")
    ap.add_argument("--data-dir", type=str, default=DATA_DIR, help="数据根目录")
    args = ap.parse_args()
    run_accuracy_analysis(args.data_dir)


if __name__ == "__main__":
    main()
