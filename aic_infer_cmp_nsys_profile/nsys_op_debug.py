import sqlite3
import argparse

parser = argparse.ArgumentParser(description="nsys kernel 序列调试工具")
parser.add_argument("--sqlite", required=True, help="nsys sqlite 文件路径")
parser.add_argument("--iter", type=int, default=3, help="选择第几次迭代 (default: 3)")
parser.add_argument("--limit", type=int, default=0, help="最多输出多少行 (0=全部)")
args = parser.parse_args()

conn = sqlite3.connect(args.sqlite)
cur = conn.cursor()

# 取指定迭代的 run_batch NVTX range
cur.execute(
    "SELECT start, end FROM NVTX_EVENTS WHERE text LIKE '%Scheduler.run_batch%' ORDER BY start LIMIT 1 OFFSET ?",
    (args.iter,),
)
row = cur.fetchone()
if row is None:
    print(f"[error] 未找到第 {args.iter} 次迭代的 Scheduler.run_batch NVTX range")
    exit(1)
s, e = row

# 查该范围内所有 kernel，按时间排序
query = """
SELECT k.start, k.end, s.value, (k.end - k.start)/1e3 as dur_us
FROM CUPTI_ACTIVITY_KIND_KERNEL k
JOIN StringIds s ON k.demangledName = s.id
WHERE k.start >= ? AND k.end <= ? AND k.deviceId = 0
ORDER BY k.start
"""
cur.execute(query, (s, e))
rows = cur.fetchall()

print(f"[info] iter #{args.iter}, {len(rows)} kernels total")
print(f"{'#':>5} | {'dur (us)':>10} | {'gap (us)':>10} | kernel name")
print("-" * 120)

prev_end = s  # NVTX range start
for i, r in enumerate(rows):
    if args.limit and i >= args.limit:
        break
    k_start, k_end, k_name, dur_us = r
    gap_us = (k_start - prev_end) / 1e3
    prev_end = k_end
    name = k_name.replace("void ", "")
    if len(name) > 80:
        name = name[:80] + "..."
    gap_str = f"{gap_us:10.1f}" if gap_us > 0.5 else f"{'':>10}"
    print(f"{i:5d} | {dur_us:10.1f} | {gap_str} | {name}")

conn.close()
