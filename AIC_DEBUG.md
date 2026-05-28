# AIC 单步时延预估准确度评测 — 调试与使用指南

本文档涵盖 `refactor_test_aic` 模块的完整使用流程，包括数据集采集、AIC 预估、准确度分析、nsys/torch profiling 以及 OP 级别误差定位。

---

## 目录

- [模块总览](#模块总览)
- [Phase 1: 数据集采集](#phase-1-数据集采集)
- [Phase 2: AIC 预估及准确度验证](#phase-2-aic-预估及准确度验证)
- [Phase 3: nsys Profile 对比 OP 误差](#phase-3-nsys-profile-对比-op-误差)
- [Phase 4: Torch Profiler 采集](#phase-4-torch-profiler-采集)
- [Phase 5: AIC OP 级别分解调试](#phase-5-aic-op-级别分解调试)
- [配置参考](#配置参考)
- [输出目录结构](#输出目录结构)
- [常见问题排查](#常见问题排查)

---

## 模块总览

```
refactor_test_aic/
├── config.py                        # 统一配置（模型参数、路径、AIC SDK 配置、校正系数）
├── utils.py                         # 公共工具（日志、CSV 读写、RequestInfo 数据类）
├── pipeline.py                      # 主流水线入口（串联 Stage 1-3）
├── stage1_convert_batch_log.py      # Stage 1: JSONL → CSV 转换
├── stage2_run_aic_estimation.py     # Stage 2: AIC SDK 时延估算
├── stage3_analyze_accuracy.py       # Stage 3: 准确度分析 & 可视化
├── nsys_profiler.py                 # nsys 性能分析（按 case_id / batch_size+seq_len 匹配）
├── torch_profiler.py                # Torch Profiler（Server/Engine 双模式）
├── hook_dataset_collector/
│   ├── hook.py                      # SGLang Hook 基类，monkey-patch Scheduler 采集数据
│   ├── sglang_launch_server.py      # 包装脚本，自动安装 Hook 后启动 SGLang
│   └── README.MD
└── aic_infer_cmp_nsys_profile/
    ├── nsys_profile.py              # Host 侧整合脚本（docker exec 执行 nsys）
    ├── run_prefill.py               # Prefill replay（真实 tokens + 真实权重）
    ├── run_decode.py                # Decode replay（支持多迭代 profiling）
    ├── aic_infer_component.py       # AIC SDK per-op 分解调试（支持 case_id 自动读取）
    ├── compare_aic_nsys.py          # AIC vs nsys per-op 时延对比工具
    └── NSYS_PROFILE_README.md
```

### 核心流程

```
数据采集 (Hook) → JSONL → CSV (Stage 1) → AIC 估算 (Stage 2) → 准确度分析 (Stage 3)
                                                                        ↓
                                                            nsys/torch profiling → OP 级别误差定位
```

---

## Phase 1: 数据集采集

通过 Hook 机制拦截 SGLang `Scheduler.run_batch`，采集每次 forward 的 batch 信息（时延、forward_mode、request_infos）。

### 1.1 启动服务（含 Hook）

```bash
export FLASHINFER_DISABLE_VERSION_CHECK=1

# 指定数据导出目录
export SGL_HOOK_REQ_INFO_DIR=/host/aiconfigurator/batch_info/qwen3-235B-A22B/ep_tp_new_gt

# 启用 Hook 数据采集
export SGL_HOOK_FETCH_BATCH_INFO=1

python3 /host/aiconfigurator/refactor_test_aic/hook_dataset_collector/sglang_launch_server.py \
    --model-path /models/Qwen3-235B-A22B-Instruct-2507-FP8 \
    --disable-overlap-schedule \
    --mem-fraction-static 0.8 \
    --tp-size 8 \
    --ep-size 8 \
    --cuda-graph-max-bs 256 \
    --trust-remote-code \
    --max-running-requests 256 \
    --port 30001 \
    --enable-piecewise-cuda-graph \
    --disable-cuda-graph-padding \
    --cuda-graph-max-bs 256
```

> **注意**: `--disable-overlap-schedule` 是必须的，否则 Hook 采集的时延数据不准确。

### 1.2 Warmup 后清除预热数据

```bash
# 触发一次数据导出（导出预热阶段的数据）
curl http://localhost:30001/start_profile

# 删除生成的 JSONL 文件（预热数据不需要）
rm -f $SGL_HOOK_REQ_INFO_DIR/TP*_schedule_batch.jsonl
rm -f $SGL_HOOK_REQ_INFO_DIR/TP*.request.jsonl
```

### 1.3 压测

```bash
python3 /host/sglang/benchmark/hicache/bench_serving.py \
    --backend sglang \
    --dataset-name random \
    --random-input-len 5000 \
    --random-output-len 100 \
    --num-prompts 2000 \
    --request-rate 50 \
    --host 127.0.0.1 \
    --port 30001 \
    --dataset-path /host/aiconfigurator/ShareGPT_V3_unfiltered_cleaned_split.json \
    --random-range-ratio 0.5
```

### 1.4 导出数据

```bash
# 压测完成后，触发数据导出
curl http://localhost:30001/start_profile
```

Hook 接口截获了 `start_profile`，调用后会将内存中的采集数据写入文件：
- `TP{rank}-EP{rank}_schedule_batch.jsonl` — 每行一条 batch 记录
- `TP{rank}-EP{rank}.request.jsonl` — 已完成请求的 input_ids / output_ids

> **不触发接口不会有数据导出！**

### 1.5 采集数据格式

**schedule_batch.jsonl** 每行格式：
```json
{
  "start_timestamp": 1713800000.123,
  "end_timestamp": 1713800000.456,
  "forward_mode": 1,
  "request_infos": [
    {"rid": "xxx", "extend_input_len": 128, "prefix_indices_len": 512, "output_ids_len": 0}
  ],
  "iter_latency": 0.333
}
```

其中 `forward_mode`: `1` = prefill, `2` = decode。

---

## Phase 2: AIC 预估及准确度验证

三阶段流水线：JSONL → CSV → AIC 估算 → 准确度分析。

### 2.1 修改配置

编辑 `config.py` 中的关键参数：

```python
# 模型 / 后端
MODEL_NAME = "Qwen3-235B-A22B-Instruct-2507-FP8"
AIC_SYSTEM = "h20_sxm"
AIC_BACKEND = "sglang"
AIC_VERSION = "0.5.9"

# ModelConfig 参数（调整 power law、量化模式等）
MODEL_CONFIG_KWARGS = dict(
    pp_size=1,
    tp_size=8,
    moe_tp_size=1,
    moe_ep_size=8,
    attention_dp_size=1,
    enable_wideep=False,
    workload_distribution="power_law_1.01",    # 调整 MoE 负载分布
    gemm_quant_mode=GEMMQuantMode.fp8_block,
    moe_quant_mode=MoEQuantMode.fp8_block,
    kvcache_quant_mode=KVCacheQuantMode.float16,
    fmha_quant_mode=FMHAQuantMode.float16,
    comm_quant_mode=CommQuantMode.half,
)

# 校正系数（默认 1.0，可根据误差分析结果调整）
DECODE_CORRECTION_FACTOR = 1.0
PREFILL_CORRECTION_FACTOR = 1.0
```

也可通过环境变量覆盖数据目录：
```bash
export AIC_DATA_DIR=/host/aiconfigurator/batch_info/qwen3-235B-A22B/ep_tp_new_gt/
```

### 2.2 运行全流水线

```bash
# 运行全部阶段 (Stage 1 + 2 + 3)
python3 -m refactor_test_aic.pipeline \
    --data-dir /host/aiconfigurator/batch_info/qwen3-235B-A22B/ep_tp_new_gt/
```

### 2.3 选择性运行

```bash
# 跳过 Stage 1（已有 CSV）
python3 -m refactor_test_aic.pipeline --data-dir ... --skip 1

# 跳过 Stage 1 和 2（已有 CSV + estimation，只重跑分析）
python3 -m refactor_test_aic.pipeline --data-dir ... --skip 1 2

# 只运行 Stage 3（准确度分析）
python3 -m refactor_test_aic.pipeline --data-dir ... --only 3

# 只运行 Stage 2 和 3
python3 -m refactor_test_aic.pipeline --data-dir ... --only 2 3
```

> **`--skip` 和 `--only` 不能同时使用。**

### 2.4 单独运行各阶段

```bash
# Stage 1: JSONL → CSV
python3 -m refactor_test_aic.stage1_convert_batch_log --data-dir ...

# Stage 2: AIC 时延估算（需要 GPU + aiconfigurator SDK）
python3 -m refactor_test_aic.stage2_run_aic_estimation --data-dir ...

# Stage 3: 准确度分析
python3 -m refactor_test_aic.stage3_analyze_accuracy --data-dir ...
```

### 2.5 各阶段说明

| 阶段 | 输入 | 输出 | 说明 |
|------|------|------|------|
| Stage 1 | `TP0-EP0_schedule_batch.jsonl` | `csv/batches_output.csv` | JSONL 0-based 行号作为 case_id |
| Stage 2 | `csv/batches_output.csv` | `estimation/batches_output_with_aic_prefill.csv`<br>`estimation/batches_output_with_aic_decode.csv` | 逐行调用 AIC SDK 估算，按 APE 降序排列 |
| Stage 3 | estimation CSV | `accuracy/` (MAPE 统计+图表)<br>`signed_error/` (误差桶分析+图表) | 按 batch_size 分组的 MAPE + 有符号误差桶 |

### 2.6 Stage 2 估算逻辑

- **Decode**: 取 `avg(past_kv_length)` 作为 `isl`，`osl=2`，调用 `run_static(mode="static_gen")`
- **Prefill**: 取 `avg(past_kv_length + input_length)` 作为 `isl`，`avg(past_kv_length)` 作为 `prefix`，计算 `seq_imbalance_correction_scale`（基于实际 FLOPs / 均值 FLOPs 比），调用 `run_static(mode="static_ctx")`
- 估算结果乘以校正系数 `DECODE_CORRECTION_FACTOR` / `PREFILL_CORRECTION_FACTOR`

### 2.7 Stage 3 分析输出

**MAPE 分析** (`accuracy/`):
- `prefill_mape_vs_bs.png` / `decode_mape_vs_bs.png` — 按 batch_size 的 Mean MAPE 折线图
- `prefill_mape_stats.csv` / `decode_mape_stats.csv` — 统计 Mean/Min/Max/Std/P90 MAPE

**误差桶分析** (`signed_error/`):
- `aic_vs_measured_signed_error_cases.csv` — 完整误差表
- 柱状图 (clip=30 和 clip=0 两个版本):
  - 绿色柱: AIC 估值更快（负误差）
  - 红色柱: 实测更快（正误差）
  - 灰色柱: 0% 持平

---

## Phase 3: nsys Profile 对比 OP 误差

对 Stage 2 中误差较大的 case，通过 nsys profiling 复现真实 GPU kernel 执行，定位 OP 级别误差。

### 方式一：使用 nsys_profiler.py（推荐，dummy 权重，快速验证）

```bash
# dry-run 验证 prompt 构建
python3 -m refactor_test_aic.nsys_profiler --case-id 42 --dry-run

# 基础 nsys profiling（外层包 nsys）
nsys profile -o report \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  -t cuda,nvtx,osrt \
  --cuda-graph-trace=node \
  --sample=none --cpuctxsw=none \
  python3 -m refactor_test_aic.nsys_profiler --case-id 42

# 按 batch_size + seq_len 匹配
nsys profile -o report --capture-range=cudaProfilerApi \
  python3 -m refactor_test_aic.nsys_profiler --batch-size 16 --seq-len 128

# 带 Python 调用栈（推荐，可看到 kernel 对应的 Python 代码）
nsys profile -o report \
  --capture-range=cudaProfilerApi \
  --python-backtrace=cuda \
  --python-sampling=true \
  --cpuctxsw=process-tree \
  python3 -m refactor_test_aic.nsys_profiler --case-id 42
```

**nsys_profiler.py 参数**:

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--case-id` | — | JSONL 0-based 行号 |
| `--batch-size` + `--seq-len` | — | 按 (bs, seq_len, forward_mode) 匹配 |
| `--forward-mode` | `1` | 1=prefill, 2=decode |
| `--match-index` | `0` | 多条匹配时取第 N 条 |
| `--data-dir` | config.DATA_DIR | 数据根目录 |
| `--model` | — | 覆盖 model_path |
| `--load-format` | `dummy` | 权重加载格式 |
| `--token-id` | `325` | 构建 prompt 的 dummy token id |
| `--dry-run` | — | 只打印 prompt 信息，不启动 Engine |

> nsys_profiler 自动从 `config.py` 的 `SGLANG_LAUNCH_CMD` 解析 ServerArgs 参数。profiling 时强制 `load_format="dummy"` + `disable_cuda_graph=True`。

### 方式二：真实权重 nsys profiling（精确复现）

使用 `aic_infer_cmp_nsys_profile/` 下的 replay 脚本，加载真实权重和真实 tokens。

#### Host 侧通过 docker exec（推荐）

```bash
# Prefill
python nsys_profile.py --mode prefill --case-id 321

# Decode
python nsys_profile.py --mode decode --case-id 6 --iters 3

# 清理容器残留进程
python nsys_profile.py --cleanup

# Dry-run
python nsys_profile.py --mode prefill --case-id 321 --dry-run
```

#### 容器内直接执行

```bash
# Prefill
cd /host/aiconfigurator/refactor_test_aic/aic_infer_cmp_nsys_profile/ && \
FLASHINFER_DISABLE_VERSION_CHECK=1 \
nsys profile \
  -f true \
  -t cuda,nvtx,osrt \
  --cuda-graph-trace=node \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --sample=none \
  --cpuctxsw=none \
  -o /host/aiconfigurator/batch_info/qwen3-235B-A22B/ep_tp_new_gt/nsys/nsys_prefill_case_321 \
  python3 run_prefill.py \
    --data-dir /host/aiconfigurator/batch_info/qwen3-235B-A22B/ep_tp_new_gt \
    --csv-case-id 321 \
    --model /models/Qwen3-235B-A22B-Instruct-2507-FP8 \
    --tp-size 8 \
    --ep-size 8

# Decode
cd /host/aiconfigurator/refactor_test_aic/aic_infer_cmp_nsys_profile && \
FLASHINFER_DISABLE_VERSION_CHECK=1 \
nsys profile \
  -f true \
  -t cuda,nvtx,osrt \
  --cuda-graph-trace=node \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --sample=none \
  --cpuctxsw=none \
  -o /host/aiconfigurator/batch_info/qwen3-235B-A22B/ep_tp_new_gt/nsys/nsys_decode_case_6 \
  python3 run_decode.py \
    --data-dir /host/aiconfigurator/batch_info/qwen3-235B-A22B/ep_tp_new_gt \
    --csv-case-id 6 \
    --model /models/Qwen3-235B-A22B-Instruct-2507-FP8 \
    --tp-size 8 \
    --ep-size 8 \
    --iters 3
```

### nsys 参数说明

| 参数 | 说明 |
|------|------|
| `-f true` | 覆盖已有输出文件 |
| `-t cuda,nvtx,osrt` | 采集 CUDA API、NVTX 标注、OS runtime 事件 |
| `--cuda-graph-trace=node` | 将 CUDA Graph 展开到 node 级别 |
| `--capture-range=cudaProfilerApi` | 仅在 `cudaProfilerStart/Stop` 之间采集 |
| `--capture-range-end=stop` | 遇到 `cudaProfilerStop` 立即结束 |
| `--sample=none` | 关闭 CPU 采样（减小报告体积） |
| `--cpuctxsw=none` | 关闭 CPU 上下文切换跟踪 |

### NVTX 标注内容

replay 脚本通过 Hook 在 `Scheduler.run_batch` 前后插入 NVTX range：

| 字段 | 说明 |
|------|------|
| `bs` | Batch size |
| `forward_mode` | EXTEND (prefill) / DECODE |
| `input_length` | 每个请求的新输入 token 数 (bs≤8 时显示列表) |
| `past_kv_length` | 每个请求的 KV cache 长度 (bs≤8 时显示列表) |

当 `bs > 8` 时改为摘要: `total_input_tokens`, `past_kv_min/max/avg`。

---

## Phase 4: Torch Profiler 采集

基于 SGLang `/start_profile` HTTP API 的 Torch Profiler 采集，生成 Chrome trace 文件。

### Server 模式（推荐）

```bash
# 基础用法（Server 已启动，需要先设置 SGLANG_TORCH_PROFILER_DIR）
python3 torch_profiler.py --input-len 128 --output-len 64

# 带 prefix cache
python3 torch_profiler.py --input-len 64 --prefix-len 512 --output-len 32

# 多请求并发
python3 torch_profiler.py --input-len 128 --output-len 64 --num-requests 4

# 只 profile decode 阶段
python3 torch_profiler.py --input-len 32 --output-len 128 \
    --profile-stages DECODE --num-steps 10
```

### 自动启动 Server

```bash
# 2 层快速 profile（所有 sglang 参数直接透传）
python3 torch_profiler.py --launch-server \
    --model-path /models/Qwen3-235B-A22B-Instruct-2507-FP8 \
    --num-layers 2 \
    --input-len 128 --output-len 64 \
    --tp-size 8 --ep-size 8 --trust-remote-code \
    --moe-a2a-backend deepep

# piecewise cuda graph profile
python3 torch_profiler.py --launch-server \
    --model-path /models/Qwen3-235B-A22B-Instruct-2507-FP8 \
    --num-layers 2 \
    --input-len 128 --output-len 64 \
    --tp-size 8 --ep-size 8 --trust-remote-code \
    --enable-piecewise-cuda-graph --disable-cuda-graph-padding \
    --piecewise-cuda-graph-tokens 2 3 4 5 6 7 8
```

### Engine 模式（进阶）

```bash
# 自定义 prefix + extend
python3 -m refactor_test_aic.torch_profiler --mode engine \
    --custom-prefix-len 100 --custom-extend-len 10

# 从 JSONL case 加载
python3 -m refactor_test_aic.torch_profiler --mode engine \
    --case-id 42
```

### Torch Profiler 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input-len` | 128 | 输入 token 长度 |
| `--prefix-len` | 0 | prefix cache 长度 |
| `--output-len` | 64 | 最大输出 token 数 |
| `--num-requests` | 1 | 并发请求数 |
| `--num-steps` | 5 | 每阶段采集 forward step 数 |
| `--profile-by-stage` | True | 分别采集 prefill/decode trace |
| `--profile-stages` | — | 只 profile 指定阶段 (如 `EXTEND,DECODE`) |
| `--with-stack` | True | 记录 Python 调用栈 |
| `--launch-server` | — | 自动启动 SGLang server |
| `--model-path` | — | 模型路径（配合 --launch-server） |
| `--num-layers` | — | 覆盖模型层数（加速 profiling） |
| `--output-dir` | /tmp/sglang_profile | trace 输出目录 |

trace 文件可在 [perfetto.dev](https://ui.perfetto.dev) 中可视化。

---

## Phase 5: AIC OP 级别分解调试

使用 `aic_infer_component.py` 查看单个 case 的 per-op 时延分解，用于调试 MoE 插值、power law 参数等。

脚本复用了 `config.py` 中的统一配置（`MODEL_CONFIG_KWARGS`、校正系数等）和 Stage 2 的 `RuntimeConfig` 构建逻辑，确保预估结果与 pipeline 一致。

### 5.1 通过 case_id 自动读取（推荐）

从 `batches_output.csv` 按 case_id 查找 batch 信息，自动判断 prefill/decode 并输出每个 op 的预估时延：

```bash
# 单个 case_id
python3 aic_infer_component.py case \
    --data-dir /host/aiconfigurator/batch_info/qwen3-235B-A22B/ep_tp/ \
    --case-id 42

# 批量多个 case_id（连续输出每个 case 的 per-op 分解）
python3 aic_infer_component.py case \
    --data-dir /host/aiconfigurator/batch_info/qwen3-235B-A22B/ep_tp/ \
    --case-id 42 100 200

# 指定自定义 CSV 路径（默认使用 data-dir/csv/batches_output.csv）
python3 aic_infer_component.py case \
    --data-dir /host/aiconfigurator/batch_info/qwen3-235B-A22B/ep_tp/ \
    --case-id 42 \
    --csv-path /path/to/custom.csv
```

### 5.2 手动指定参数（向后兼容）

不依赖 CSV，直接指定 phase、batch_size、isl 等参数：

```bash
# Prefill per-op 分解
python3 aic_infer_component.py manual --phase prefill --batch-size 1 --isl 3786

# Decode per-op 分解
python3 aic_infer_component.py manual --phase decode --batch-size 51 --isl 2048

# Prefill 带 prefix cache
python3 aic_infer_component.py manual --phase prefill --batch-size 4 --isl 4096 --prefix 512
```

### 5.3 参数说明

**`case` 子命令参数：**

| 参数 | 必填 | 说明 |
|------|------|------|
| `--data-dir` | 是 | 数据根目录 |
| `--case-id` | 是 | 一个或多个 case_id（JSONL 0-based 行号） |
| `--csv-path` | 否 | CSV 文件路径，默认 `{data-dir}/csv/batches_output.csv` |

**`manual` 子命令参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--phase` | `prefill` | `prefill` 或 `decode` |
| `--batch-size` | `1` | Batch size |
| `--isl` | `3786` | Input sequence length |
| `--osl` | `2` | Output sequence length |
| `--prefix` | `0` | Prefix cache 长度 |

### 5.4 输出内容

- **RuntimeConfig 详情**: 实际使用的 bs/isl/osl/prefix 和 correction 参数
- **Per-op breakdown**: 每个算子 (GEMM/MoE/Attention/Comm) 的估算时延 (ms)
- **APE 对比**: 预估总时延 vs 实测时延的绝对百分比误差（case 模式自动计算）
- **Decode MoE interpolation debug**: MoE 插值详情（query_num_tokens、采集点、插值结果、scale_factor）

### 5.5 AIC vs nsys per-op 时延对比（compare_aic_nsys.py）

将 AIC 的 per-op 预估与 nsys 实测 kernel 时延对齐，直观定位偏差最大的 op。

```bash
# Prefill case 对比
python3 compare_aic_nsys.py \
    --nsys-sqlite /host/aiconfigurator/batch_info/qwen3-235B-A22B/ep_tp_new_gt/nsys/nsys_prefill_case_1115.sqlite \
    --data-dir /host/aiconfigurator/batch_info/qwen3-235B-A22B/ep_tp_new_gt/ \
    --case-id 1115

# Decode case 对比（自动过滤 DECODE forward_mode）
python3 compare_aic_nsys.py \
    --nsys-sqlite /host/aiconfigurator/batch_info/qwen3-235B-A22B/ep_tp_new_gt/nsys/nsys_decode_case_6.sqlite \
    --data-dir /host/aiconfigurator/batch_info/qwen3-235B-A22B/ep_tp_new_gt/ \
    --case-id 6

# 使用 .nsys-rep 文件（自动导出 sqlite）
python3 compare_aic_nsys.py \
    --nsys-rep /path/to/report.nsys-rep \
    --data-dir /path/to/data \
    --case-id 42

# 指定迭代 + 输出 JSON
python3 compare_aic_nsys.py \
    --nsys-sqlite /path/to/report.sqlite \
    --data-dir /path/to/data \
    --case-id 42 --iter-index 2 --json
```

**AIC op → nsys kernel 映射关系：**

| AIC Op | nsys Kernel(s) | 说明 |
|--------|---------------|------|
| `context_add_norm_1` | `RMSNormKernel` | 层前注意力归一化 |
| `context_qkv_gemm` | `per_token_group_quant` + `deep_gemm` (QKV) | FP8 量化 + QKV 投影 |
| `context_attention` | `fused_qknorm` + `fused_rope` + `FlashAttnFwd` | Q/K norm + RoPE + Flash Attention |
| `context_proj_gemm` | `per_token_group_quant` + `deep_gemm` (proj) | FP8 量化 + O 投影 |
| `(attn_tp_allreduce)` | `ncclAllReduce` / `cross_device_reduce` | TP 通信（AIC 无显式建模） |
| `context_add_norm_2` | `FusedAddRMSNormKernel` | 残差 + FFN 前归一化 |
| `context_router_gemm` | `nvjet`/`cublas` + `splitKreduce` + `topkGatingSoftmax` | MoE 路由 |
| `context_moe_pre/post_dispatch` | `ncclAllReduce` / `cross_device_reduce` (MoE EP) | EP 调度通信 |
| `context_moe` | `fused_moe_kernel`×2 + `act_and_mul` + quant + reduce | MoE 专家计算 |
| `context_embedding_ar` | 首个 `ncclAllReduce` | 嵌入层 AllReduce |

> Decode 阶段同理，op 前缀为 `generation_` 而非 `context_`。Decode 可能使用 SGLang 自定义 `cross_device_reduce` 而非 NCCL AllReduce。

### 5.6 典型调试流程

```
1. 在 signed_error CSV 中找到误差最大的 case_id
2. 用 aic_infer_component.py case --case-id <id> 查看 per-op 分解
3. 对比 nsys profiling 的实际 kernel 时延，定位偏差最大的 op
4. 调整 config.py 中的 power law、校正系数等参数后重跑验证
```

---

## 配置参考

### config.py 关键配置项

| 配置项 | 说明 |
|--------|------|
| `DATA_DIR` | 数据根目录，可通过 `AIC_DATA_DIR` 环境变量覆盖 |
| `SCHEDULE_JSONL_FILENAME` | Hook 生成的 JSONL 文件名，默认 `TP0-EP0_schedule_batch.jsonl` |
| `MODEL_CONFIG_KWARGS` | AIC SDK 的 ModelConfig 参数（tp/ep/量化模式/power law 等） |
| `SGLANG_LAUNCH_CMD` | SGLang 启动命令，nsys_profiler 自动解析为 ServerArgs |
| `DECODE_CORRECTION_FACTOR` | Decode 校正系数（默认 1.0） |
| `PREFILL_CORRECTION_FACTOR` | Prefill 校正系数（默认 1.0） |

### 环境变量

| 环境变量 | 说明 |
|----------|------|
| `AIC_DATA_DIR` | 覆盖 config.py 中的 DATA_DIR |
| `SGL_HOOK_REQ_INFO_DIR` | Hook 数据导出目录 |
| `SGL_HOOK_FETCH_BATCH_INFO` | 设为 1 启用 Hook 数据采集 |
| `FLASHINFER_DISABLE_VERSION_CHECK` | 设为 1 绕过 flashinfer 版本检查 |
| `SGLANG_TORCH_PROFILER_DIR` | Torch Profiler trace 输出目录 |

---

## 输出目录结构

运行完 pipeline 后，数据目录结构如下：

```
{data_dir}/
├── TP0-EP0_schedule_batch.jsonl        # Hook 采集的原始数据
├── TP0-EP0.request.jsonl               # 请求的 input_ids/output_ids
├── csv/
│   └── batches_output.csv              # Stage 1 输出
├── estimation/
│   ├── batches_output_with_aic_prefill.csv   # Stage 2: prefill 估算结果
│   ├── batches_output_with_aic_decode.csv    # Stage 2: decode 估算结果
│   └── error.log                             # 估算报错详情（无报错时自动删除）
├── accuracy/
│   ├── prefill_mape_vs_bs.png          # Prefill MAPE 折线图
│   ├── decode_mape_vs_bs.png           # Decode MAPE 折线图
│   ├── prefill_mape_stats.csv          # Prefill MAPE 统计表
│   └── decode_mape_stats.csv           # Decode MAPE 统计表
├── signed_error/
│   ├── aic_vs_measured_signed_error_cases.csv   # 有符号误差表
│   ├── aic_vs_measured_signed_error_cases.png   # 总误差桶图
│   ├── aic_vs_measured_signed_error_cases_30_prefill.png  # Prefill clip=30
│   ├── aic_vs_measured_signed_error_cases_0_prefill.png   # Prefill clip=0
│   ├── aic_vs_measured_signed_error_cases_30_decode.png   # Decode clip=30
│   └── aic_vs_measured_signed_error_cases_0_decode.png    # Decode clip=0
└── nsys/
    ├── nsys_prefill_case_{id}.nsys-rep  # nsys profiling 报告
    └── nsys_decode_case_{id}.nsys-rep
```

---

## 常见问题排查

### 1. Hook 没有数据导出
- 确认启动时使用了 `sglang_launch_server.py`（而非直接 `sglang.launch_server`）
- 确认设置了 `SGL_HOOK_REQ_INFO_DIR` 环境变量
- 压测完成后必须调用 `curl http://localhost:{port}/start_profile` 触发导出

### 2. Stage 2 估算报错
- 检查 `estimation/error.log` 获取详细堆栈
- 常见原因：batch_size 超出采集范围、MoE token 数超出最大采集点
- AIC SDK 使用 `inner_only=False` 已禁用，支持外推但精度可能下降

### 3. nsys profiling 卡住或失败
- 确认容器中没有残留进程：`python nsys_profile.py --cleanup`
- 确认 `--disable-overlap-schedule` 已设置
- 检查 GPU 显存是否足够（真实权重模式需要完整加载模型）

### 4. Torch Profiler 未生成 trace
- 确认启动 server 时设置了 `SGLANG_TORCH_PROFILER_DIR` 环境变量
- 可能需要发送更多请求触发 trace 生成（脚本会自动重试最多 20 轮）
- 使用 `--launch-server` 模式时环境变量会自动设置

### 5. case_id 如何确定
- `case_id` 是 JSONL 文件的 0-based 行号，由 Stage 1 转换时保留
- 在 `signed_error/aic_vs_measured_signed_error_cases.csv` 中按误差排序可找到最差 case
- replay 脚本的 `--csv-case-id` 参数即为此 case_id
