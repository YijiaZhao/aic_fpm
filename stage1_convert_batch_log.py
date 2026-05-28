"""
阶段 1：JSONL -> CSV 转换
=========================
将 hook.py 采集的 schedule_batch.jsonl 转换为结构化 CSV。

输入:  {data_dir}/TP0-EP0_schedule_batch.jsonl
输出:  {data_dir}/csv/batches_output.csv

CSV 列: case_id, latency, batch_type, request_infos

其中 case_id 为 JSONL 的 0-based 行号，可直接用于 nsys_profiler --case-id
"""

import argparse
import json
import os

from .config import (
    DATA_DIR,
    SCHEDULE_JSONL_FILENAME,
    SUBDIR_CSV,
    get_output_dir,
)
from .utils import logger, write_csv


def convert_batch_log(data_dir: str) -> str:
    """
    将 schedule_batch.jsonl 转换为 batches_output.csv。

    Returns:
        生成的 CSV 文件路径。
    """
    jsonl_path = os.path.join(data_dir, SCHEDULE_JSONL_FILENAME)
    out_dir = get_output_dir(data_dir, SUBDIR_CSV)
    csv_path = os.path.join(out_dir, "batches_output.csv")

    logger.info(f"[Stage 1] 开始转换 JSONL -> CSV")
    logger.info(f"  输入: {jsonl_path}")
    logger.info(f"  输出: {csv_path}")

    headers = ["case_id", "latency", "batch_type", "request_infos"]
    rows = []
    line_idx = 0  # JSONL 0-based 行号（含空行也计数，与文件行号一致）
    valid_count = 0
    skipped = 0

    with open(jsonl_path, "r", encoding="utf-8") as fin:
        for line in fin:
            current_idx = line_idx
            line_idx += 1

            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            valid_count += 1
            latency = record["iter_latency"]
            forward_mode = record["forward_mode"]
            batch_type = "prefill" if forward_mode == 1 else "decode"

            request_infos = []
            for req in record["request_infos"]:
                if forward_mode == 1:  # prefill
                    input_length = req["extend_input_len"]
                    past_kv_length = req["prefix_indices_len"]
                else:  # decode
                    input_length = 1
                    past_kv_length = req["prefix_indices_len"] + req["output_ids_len"]

                info = {"input_length": input_length, "past_kv_length": past_kv_length}
                if "origin_input_ids" in req:
                    info["origin_input_ids"] = req["origin_input_ids"]
                request_infos.append(info)

            request_infos_str = json.dumps(request_infos, separators=(",", ":"))
            rows.append([current_idx, latency, batch_type, request_infos_str])

    write_csv(csv_path, headers, rows)

    logger.info(
        f"[Stage 1] 完成: 总行数={line_idx}, 有效={valid_count}, 跳过={skipped}"
    )
    return csv_path


# ============================================================================
# 命令行入口
# ============================================================================


def main():
    ap = argparse.ArgumentParser(description="阶段 1: JSONL -> CSV 转换")
    ap.add_argument("--data-dir", type=str, default=DATA_DIR, help="数据根目录")
    args = ap.parse_args()
    convert_batch_log(args.data_dir)


if __name__ == "__main__":
    main()
