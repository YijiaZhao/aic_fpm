# refactor_test_aic — AIC 单步时延预估准确度评测工具包

对 AIConfigurator (AIC) 的单步推理时延预估结果进行端到端的准确度评测。
包含 3 个流水线阶段 + 1 个独立的 nsys profiling 工具。

## 目录结构

```
refactor_test_aic/
├── config.py              # 统一配置（路径、模型参数、SGLang 启动命令等）
├── utils.py               # 公共工具（数据类、CSV 读写、日志、计算函数）
├── stage1_convert_batch_log.py   # 阶段 1: JSONL → CSV
├── stage2_run_aic_estimation.py  # 阶段 2: AIC SDK 时延预估
├── stage3_analyze_accuracy.py    # 阶段 3: 准确度分析 + 可视化
├── pipeline.py            # 主流水线入口（串联 Stage 1-3）
└── nsys_profiler.py       # Nsys Profile 工具（独立使用）
```

## 前置条件

- **Stage 1**: 无特殊依赖，只需 Python 标准库
- **Stage 2**: 需要 GPU + `aiconfigurator` SDK（`pip install -e .`）
- **Stage 3**: 需要 `pandas`、`matplotlib`、`numpy`
- **nsys_profiler**: 需要 `sglang`、`torch`、NVIDIA Nsight Systems

## 快速开始

```bash
# 进入容器
docker exec -it -w /host/aiconfigurator/test_aic mry-aic bash

# 运行全部阶段（Stage 2 需要 GPU）
python3 -m refactor_test_aic.pipeline

# 跳过 Stage 1、2，只跑准确度分析
python3 -m refactor_test_aic.pipeline --skip 1 2

# 只跑 Stage 1
python3 -m refactor_test_aic.pipeline --only 1

# 指定数据目录
python3 -m refactor_test_aic.pipeline --data-dir /host/aiconfigurator/batch_info/qwen3-235B-A22B/ep_tp/
```

## 流水线阶段

### Stage 1: JSONL → CSV 转换

将 SGLang hook.py 采集的 `schedule_batch.jsonl` 转换为结构化 CSV。

- **输入**: `{data_dir}/TP0-EP0_schedule_batch.jsonl`
- **输出**: `{data_dir}/csv/batches_output.csv`
- **CSV 列**: `case_id`（JSONL 行号）、`latency`、`batch_type`、`request_infos`

`case_id` 为 JSONL 的 0-based 行号，可直接用于 nsys_profiler 的 `--case-id` 参数。

```bash
# 单独运行
python3 -m refactor_test_aic.stage1_convert_batch_log
```

### Stage 2: AIC 时延预估

对 CSV 中的每条 batch 调用 AIConfigurator SDK 估算单步时延。

- **输入**: `{data_dir}/csv/batches_output.csv`
- **输出**:
  - `{data_dir}/estimation/batches_output_with_aic_prefill.csv`
  - `{data_dir}/estimation/batches_output_with_aic_decode.csv`
- **输出 CSV 列**: `case_id`、`batch_type`、`batch_size`、`avg_input_length`、`avg_past_kv_length`、`latency_ms`、`estimated_latency_ms`、`APE(%)`、`request_infos`

关键逻辑:
- Prefill 使用 `seq_imbalance_correction_scale` 修正序列长度不平衡
- Decode / Prefill 分别应用校正系数（`config.py` 中配置）

```bash
python3 -m refactor_test_aic.stage2_run_aic_estimation
```

### Stage 3: 准确度分析

分为两部分，输出到不同子目录：

**MAPE 统计** → `{data_dir}/accuracy/`
- 按 `batch_size` 分组统计 Mean/Min/Max/Std/P90 MAPE
- 生成折线图 + 统计 CSV

**有符号误差桶分析** → `{data_dir}/signed_error/`
- 计算 `(AIC预估 - 实测) / 实测 × 100%` 的有符号百分比误差
- 分桶统计并生成柱状图（绿色=AIC偏快，红色=实测偏快）
- 分别生成 clip=30% 和 clip=0（无裁剪）的图表

```bash
python3 -m refactor_test_aic.stage3_analyze_accuracy
```

## 输出目录结构

```
{data_dir}/
├── csv/                          # Stage 1 输出
│   └── batches_output.csv
├── estimation/                   # Stage 2 输出
│   ├── batches_output_with_aic_prefill.csv
│   └── batches_output_with_aic_decode.csv
├── accuracy/                     # Stage 3: MAPE 统计
│   ├── prefill_mape_stats.csv
│   ├── prefill_mape_vs_bs.png
│   ├── decode_mape_stats.csv
│   └── decode_mape_vs_bs.png
└── signed_error/                 # Stage 3: 误差桶分析
    ├── aic_vs_measured_signed_error_cases.csv
    ├── aic_vs_measured_signed_error_cases.png
    ├── aic_vs_measured_signed_error_cases_30_prefill.png
    ├── aic_vs_measured_signed_error_cases_0_prefill.png
    ├── aic_vs_measured_signed_error_cases_30_decode.png
    └── aic_vs_measured_signed_error_cases_0_decode.png
```

## Nsys Profiler

独立工具，用于对指定 batch 进行 Nvidia Nsight Systems GPU 性能分析。

### 工作原理

1. 从 `schedule_batch.jsonl` 中加载目标 batch 的 request_infos
2. 构建 prefix prompts（填充 SGLang prefix cache）和 full prompts
3. 先发 prefix prompts 预热 KV cache，再在 CUDA Profiler 范围内运行 full prompts
4. 配合 `nsys --capture-range=cudaProfilerApi` 精确捕获目标 batch 的 GPU kernel

### case_id 怎么来的

`case_id` 是 JSONL 文件的 0-based 行号。Stage 1 转换时会将它写入 CSV 的第一列。
你可以在 `batches_output.csv`（或 Stage 2 输出的 estimation CSV）中找到感兴趣的行，
直接取 `case_id` 列的值传给 nsys_profiler。

### 使用方法

```bash
# 1. 先用 --dry-run 验证 prompt 构建（不需要 GPU）
python3 -m refactor_test_aic.nsys_profiler --case-id 42 --dry-run

# 2. 基础 profiling（仅 GPU kernel + NVTX）
nsys profile -o report --capture-range=cudaProfilerApi \
    python3 -m refactor_test_aic.nsys_profiler --case-id 42

# 3. 带 Python 调用栈（推荐！可看到每个 kernel 对应的 Python 代码位置）
nsys profile -o report \
    --capture-range=cudaProfilerApi \
    --python-backtrace=cuda \
    --python-sampling=true \
    --cpuctxsw=process-tree \
    python3 -m refactor_test_aic.nsys_profiler --case-id 42

# 4. 按 (batch_size, seq_len) 匹配
nsys profile -o report --capture-range=cudaProfilerApi \
    python3 -m refactor_test_aic.nsys_profiler --batch-size 16 --seq-len 128

# 5. 覆盖模型路径（默认从 config.py 的 SGLANG_LAUNCH_CMD 解析）
python3 -m refactor_test_aic.nsys_profiler --case-id 42 --model /models/other-model
```

### nsys Python 调用栈参数说明

| 参数 | 作用 |
|------|------|
| `--python-backtrace=cuda` | 在每次 CUDA API 调用时采集 Python 调用栈，在 nsys GUI 中点击某个 kernel 就能看到它是哪行 Python 代码触发的 |
| `--python-sampling=true` | 启用 Python 采样分析器，按时间间隔采样 Python 栈帧，可以看到 CPU 上 Python 代码的时间分布 |
| `--cpuctxsw=process-tree` | 采集 CPU 上下文切换信息，帮助关联 Python 线程与 GPU 活动的时序关系 |

在 Nsight Systems GUI 中查看报告时：
- **Timeline** 视图：可以看到 NVTX range（`prime_prefix_cache` 和 `generate_full_prompts`）与 GPU kernel 的对应关系
- **Bottom-Up** 视图：点击任意 GPU kernel，右侧面板会显示完整的 Python 调用栈
- **Python Sampling** 行：时间线上会出现单独的 Python 采样行，展示 CPU 上 Python 代码的执行情况

### SGLang 服务参数配置

nsys_profiler 会自动从 `config.py` 中的 `SGLANG_LAUNCH_CMD` 解析 SGLang 的启动参数
（如 `tp_size`、`ep_size`、`moe_a2a_backend` 等），无需手动传递。

直接把部署 sglang 时的命令粘贴到 `SGLANG_LAUNCH_CMD` 即可：

```python
# config.py
SGLANG_LAUNCH_CMD = """
python3 sglang_launch_server.py \
    --model-path /models/Qwen3-235B-A22B-Instruct-2507-FP8 \
    --tp-size 8 \
    --ep-size 8 \
    --moe-a2a-backend deepep \
    ...
"""
```

Profiling 时会自动覆盖 `load_format=dummy`（不加载真实权重）和 `disable_cuda_graph=True`。

## 配置说明

所有可调参数集中在 `config.py`，切换模型/数据时只需修改此文件：

| 配置项 | 说明 |
|--------|------|
| `DATA_DIR` | 数据根目录，支持 `AIC_DATA_DIR` 环境变量覆盖 |
| `SCHEDULE_JSONL_FILENAME` | hook.py 生成的 JSONL 文件名 |
| `MODEL_PATH` / `MODEL_NAME` | 模型路径和名称 |
| `AIC_SYSTEM` / `AIC_BACKEND` / `AIC_VERSION` | AIC 性能数据库参数 |
| `MODEL_CONFIG_KWARGS` | AIC ModelConfig 参数（tp_size、moe_ep_size 等） |
| `SGLANG_LAUNCH_CMD` | SGLang 服务启动命令（nsys_profiler 自动解析） |
| `DECODE_CORRECTION_FACTOR` | Decode 估算校正系数 |
| `PREFILL_CORRECTION_FACTOR` | Prefill 估算校正系数 |
