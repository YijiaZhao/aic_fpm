"""
AIC 单步时延预估准确度评测 — 主流水线
======================================
串联 4 个阶段:
  Stage 1: JSONL -> CSV 转换
  Stage 2: AIC 时延预估
  Stage 3: 准确度分析（MAPE + 误差桶）
用法:
  # 运行全部阶段
  python3 -m refactor_test_aic.pipeline --data-dir /host/aiconfigurator/batch_info/qwen3-235B-A22B/ep_tp/

  # 跳过某些阶段（例如已有 CSV，只想重跑分析）
  python3 -m refactor_test_aic.pipeline --data-dir ... --skip 1 2

  # 只跑 stage 3
  python3 -m refactor_test_aic.pipeline --data-dir ... --only 3
"""

import argparse
import sys
import time

from .config import DATA_DIR
from .utils import logger


def run_pipeline(
    data_dir: str,
    skip_stages: set[int] | None = None,
    only_stages: set[int] | None = None,
) -> None:
    """
    按顺序执行各阶段。

    Args:
        data_dir:      数据根目录
        skip_stages:   要跳过的阶段编号集合
        only_stages:   只运行这些阶段（与 skip_stages 互斥）
    """
    all_stages = {1, 2, 3}
    if only_stages:
        stages_to_run = only_stages & all_stages
    elif skip_stages:
        stages_to_run = all_stages - skip_stages
    else:
        stages_to_run = all_stages

    banner = "=" * 60
    logger.info(banner)
    logger.info("AIC 单步时延预估准确度评测流水线")
    logger.info(f"  数据目录: {data_dir}")
    logger.info(f"  运行阶段: {sorted(stages_to_run)}")
    logger.info(banner)

    t0 = time.time()

    # ---- Stage 1: JSONL -> CSV ----
    if 1 in stages_to_run:
        logger.info(f"\n{'─' * 50}")
        logger.info("▶ Stage 1: JSONL -> CSV 转换")
        logger.info(f"{'─' * 50}")
        from .stage1_convert_batch_log import convert_batch_log

        convert_batch_log(data_dir)

    # ---- Stage 2: AIC 时延预估 ----
    if 2 in stages_to_run:
        logger.info(f"\n{'─' * 50}")
        logger.info("▶ Stage 2: AIC 时延预估")
        logger.info(f"{'─' * 50}")
        from .stage2_run_aic_estimation import run_aic_estimation

        run_aic_estimation(data_dir)

    # ---- Stage 3: 准确度分析 ----
    if 3 in stages_to_run:
        logger.info(f"\n{'─' * 50}")
        logger.info("▶ Stage 3: 准确度分析")
        logger.info(f"{'─' * 50}")
        from .stage3_analyze_accuracy import run_accuracy_analysis

        run_accuracy_analysis(data_dir)

    elapsed = time.time() - t0
    logger.info(f"\n{banner}")
    logger.info(f"全部阶段完成，耗时 {elapsed:.1f}s")
    logger.info(banner)


# ============================================================================
# 命令行入口
# ============================================================================


def main():
    ap = argparse.ArgumentParser(
        description="AIC 单步时延预估准确度评测流水线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
示例:
  # 运行全部阶段
  python3 -m refactor_test_aic.pipeline --data-dir /host/aiconfigurator/batch_info/qwen3-235B-A22B/ep_tp/

  # 跳过 Stage 1 和 2（已有 CSV + estimation）
  python3 -m refactor_test_aic.pipeline --data-dir ... --skip 1 2

  # 只运行 Stage 3
  python3 -m refactor_test_aic.pipeline --data-dir ... --only 3

阶段说明:
  1 - JSONL -> CSV 转换
  2 - AIC 时延预估（需要 GPU + aiconfigurator SDK）
  3 - 准确度分析（MAPE + 误差桶可视化）
""",
    )
    ap.add_argument("--data-dir", type=str, default=DATA_DIR, help="数据根目录")
    ap.add_argument(
        "--skip", type=int, nargs="+", default=[], help="要跳过的阶段编号 (1-3)"
    )
    ap.add_argument(
        "--only",
        type=int,
        nargs="+",
        default=[],
        help="只运行这些阶段 (与 --skip 互斥)",
    )
    args = ap.parse_args()

    skip = set(args.skip) if args.skip else None
    only = set(args.only) if args.only else None

    if skip and only:
        logger.error("--skip 和 --only 不能同时使用")
        sys.exit(1)

    run_pipeline(args.data_dir, skip_stages=skip, only_stages=only)


if __name__ == "__main__":
    main()
