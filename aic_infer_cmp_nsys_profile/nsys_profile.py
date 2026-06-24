#!/usr/bin/env python3
"""
nsys profiling 整合脚本 —— 在容器中执行 prefill / decode 的 nsys profiling。

用法示例:
    # prefill profiling (case_id=321)
    python nsys_profile.py --mode prefill --case-id 321

    # decode profiling (case_id=1796)
    python nsys_profile.py --mode decode --case-id 1796

    # 自定义参数
    python nsys_profile.py --mode decode --case-id 1796 --data-dir /path/to/data --container mycontainer --iters 5

    # 清理容器残留进程
    python nsys_profile.py --cleanup

    # dry-run 模式（只打印命令，不执行）
    python nsys_profile.py --mode prefill --case-id 321 --dry-run
"""

import argparse
import os
import subprocess
import sys


# ============================================================================
# Defaults
# ============================================================================
DEFAULT_CONTAINER = "mry-aic-collect"
DEFAULT_DATA_DIR = "/host/aiconfigurator/batch_info/qwen3-235B-A22B/ep_tp_new_gt"
DEFAULT_NSYS_DIR = None  # 默认 = {data_dir}/nsys
DEFAULT_MODEL = "/models/Qwen3-235B-A22B-Instruct-2507-FP8"
DEFAULT_WORKDIR = "/host/aiconfigurator/aic_fpm/aic_infer_cmp_nsys_profile"
DEFAULT_TP_SIZE = 8
DEFAULT_EP_SIZE = 8
DEFAULT_DECODE_ITERS = 3


def _docker_exec(container: str, cmd: str, dry_run: bool = False) -> int:
    """在容器中执行命令."""
    full_cmd = f'docker exec {container} bash -c "{cmd}"'
    if dry_run:
        print(f"\n[dry-run] {full_cmd}")
        return 0
    print(f"\n[exec] docker exec {container} bash -c ...")
    print(f"  {cmd[:200]}{'...' if len(cmd) > 200 else ''}")
    return subprocess.call(full_cmd, shell=True)


def cleanup(container: str, dry_run: bool = False):
    """清理容器中残留的 sglang / nsys / torch compile worker 进程."""
    print(f"[cleanup] Cleaning up stale processes in container '{container}'...")

    # 列出相关进程
    _docker_exec(
        container,
        "ps aux | grep -E 'sglang|run_prefill|run_decode|nsys|compile_worker' | grep -v grep || true",
        dry_run,
    )

    # kill 残留进程
    kill_patterns = [
        "torch._inductor.compile_worker",
        "run_prefill.py",
        "run_decode.py",
        "sglang",
    ]
    for pattern in kill_patterns:
        _docker_exec(container, f"pkill -f '{pattern}' 2>/dev/null || true", dry_run)

    if not dry_run:
        import time

        time.sleep(2)

    # 验证
    _docker_exec(
        container,
        "ps aux | grep -E 'sglang|run_prefill|run_decode|nsys|compile_worker' | grep -v grep || echo '[cleanup] All clean.'",
        dry_run,
    )
    print("[cleanup] Done.")


def build_nsys_command(
    mode: str,
    case_id: int,
    data_dir: str,
    nsys_dir: str,
    model: str,
    workdir: str,
    tp_size: int,
    ep_size: int,
    decode_iters: int,
) -> str:
    """构建完整的 nsys profile 命令."""
    # 输出文件名
    output_file = os.path.join(nsys_dir, f"nsys_{mode}_case_{case_id}")

    # nsys 参数
    nsys_args = " ".join(
        [
            "nsys profile",
            "-f true",
            "-t cuda,nvtx",
            "--cuda-graph-trace=node",
            "--capture-range=cudaProfilerApi",
            "--capture-range-end=stop",
            "--sample=none",
            "--cpuctxsw=none",
            f"-o {output_file}",
        ]
    )

    # python 脚本
    if mode == "prefill":
        script = f"python3 run_prefill.py --data-dir {data_dir} --csv-case-id {case_id} --model {model} --tp-size {tp_size} --ep-size {ep_size}"
    else:
        script = f"python3 run_decode.py --data-dir {data_dir} --csv-case-id {case_id} --model {model} --tp-size {tp_size} --ep-size {ep_size} --iters {decode_iters}"

    # 完整命令
    cmd = f"cd {workdir} && FLASHINFER_DISABLE_VERSION_CHECK=1 {nsys_args} {script}"
    return cmd


def run_profile(args):
    """执行 nsys profiling."""
    nsys_dir = args.nsys_dir or os.path.join(args.data_dir, "nsys")

    # 确保 nsys 输出目录存在
    _docker_exec(args.container, f"mkdir -p {nsys_dir}", args.dry_run)

    cmd = build_nsys_command(
        mode=args.mode,
        case_id=args.case_id,
        data_dir=args.data_dir,
        nsys_dir=nsys_dir,
        model=args.model,
        workdir=args.workdir,
        tp_size=args.tp_size,
        ep_size=args.ep_size,
        decode_iters=args.iters,
    )

    print(f"\n{'=' * 60}")
    print(f"  nsys {args.mode} profiling | case_id={args.case_id}")
    print(f"  container: {args.container}")
    print(f"  data-dir:  {args.data_dir}")
    print(f"  nsys-dir:  {nsys_dir}")
    print(f"  model:     {args.model}")
    print(f"  tp/ep:     {args.tp_size}/{args.ep_size}")
    if args.mode == "decode":
        print(f"  iters:     {args.iters}")
    print(f"{'=' * 60}")

    rc = _docker_exec(args.container, cmd, args.dry_run)

    if not args.dry_run:
        # 输出文件检查
        output_file = os.path.join(
            nsys_dir, f"nsys_{args.mode}_case_{args.case_id}.nsys-rep"
        )
        _docker_exec(
            args.container,
            f"ls -lh {output_file} 2>/dev/null || echo '[error] Output file not found: {output_file}'",
        )

    return rc


def main():
    ap = argparse.ArgumentParser(
        description="nsys profiling 整合脚本 — prefill / decode replay",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python nsys_profile.py --mode prefill --case-id 321
  python nsys_profile.py --mode decode --case-id 1796 --iters 3
  python nsys_profile.py --cleanup
  python nsys_profile.py --mode prefill --case-id 321 --dry-run
        """,
    )

    # 核心参数
    ap.add_argument(
        "--mode",
        choices=["prefill", "decode"],
        help="profiling 模式: prefill 或 decode",
    )
    ap.add_argument(
        "--case-id", type=int, help="CSV 中的 case_id (即 JSONL 0-based 行号)"
    )

    # 环境参数
    ap.add_argument(
        "--container",
        type=str,
        default=DEFAULT_CONTAINER,
        help=f"Docker 容器名 (default: {DEFAULT_CONTAINER})",
    )
    ap.add_argument(
        "--data-dir",
        type=str,
        default=DEFAULT_DATA_DIR,
        help=f"数据目录 (容器内路径, default: {DEFAULT_DATA_DIR})",
    )
    ap.add_argument(
        "--nsys-dir",
        type=str,
        default=DEFAULT_NSYS_DIR,
        help="nsys 输出目录 (容器内路径, default: {data_dir}/nsys)",
    )
    ap.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"模型路径 (default: {DEFAULT_MODEL})",
    )
    ap.add_argument(
        "--workdir",
        type=str,
        default=DEFAULT_WORKDIR,
        help=f"工作目录 (default: {DEFAULT_WORKDIR})",
    )

    # 模型参数
    ap.add_argument(
        "--tp-size",
        type=int,
        default=DEFAULT_TP_SIZE,
        help=f"TP size (default: {DEFAULT_TP_SIZE})",
    )
    ap.add_argument(
        "--ep-size",
        type=int,
        default=DEFAULT_EP_SIZE,
        help=f"EP size (default: {DEFAULT_EP_SIZE})",
    )

    # Decode 专属
    ap.add_argument(
        "--iters",
        type=int,
        default=DEFAULT_DECODE_ITERS,
        help=f"Decode profile 迭代次数 (default: {DEFAULT_DECODE_ITERS})",
    )

    # 操作
    ap.add_argument(
        "--cleanup", action="store_true", help="清理容器中残留的 sglang/nsys/torch 进程"
    )
    ap.add_argument("--dry-run", action="store_true", help="只打印命令，不执行")

    args = ap.parse_args()

    if args.cleanup:
        cleanup(args.container, args.dry_run)
        return

    if not args.mode or args.case_id is None:
        ap.error("--mode 和 --case-id 是必填参数 (除非使用 --cleanup)")

    rc = run_profile(args)
    sys.exit(rc)


if __name__ == "__main__":
    main()
