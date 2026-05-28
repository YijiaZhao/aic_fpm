# Prefill case 1115，前 55 个 kernel
python3 nsys_op_debug.py \
    --sqlite /host/aiconfigurator/batch_info/qwen3-235B-A22B/ep_tp_new_gt/nsys/nsys_prefill_case_1115.sqlite \
    --limit 55

# 看全部 2480 个 kernel（不加 --limit）
python3 nsys_op_debug.py \
    --sqlite /host/aiconfigurator/batch_info/qwen3-235B-A22B/ep_tp_new_gt/nsys/nsys_prefill_case_1115.sqlite

# 换迭代（默认 iter=3）
python3 nsys_op_debug.py --sqlite xxx.sqlite --iter 5