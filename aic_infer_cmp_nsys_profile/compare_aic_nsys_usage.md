# compare_aic_nsys.py 使用说明

AIC per-op 预估 vs nsys 实测 kernel 时延对比工具。

从 nsys `.sqlite` 中提取 GPU kernel trace，按 AIC op 分类汇总，再与 `aic_infer_component.py` 的 per-op 预估对齐，输出对比表。

---

## 前置依赖

- 需在 `mry-aic-collect` 容器中执行（已安装 AIC SDK 和 nsys 工具）
- 需要已有的 nsys 报告文件（`.nsys-rep` 或已导出的 `.sqlite`）
- 需要对应的 CSV 数据目录（含 `csv/batches_output.csv`）

---

## 基本用法

### 1. 使用已有 sqlite 文件（推荐）

```bash
docker exec mry-aic-collect python3 \
    /host/aiconfigurator/refactor_test_aic/aic_infer_cmp_nsys_profile/compare_aic_nsys.py \
    --nsys-sqlite /host/aiconfigurator/batch_info/qwen3-235B-A22B/ep_tp_new_gt/nsys/nsys_prefill_case_1115.sqlite \
    --data-dir /host/aiconfigurator/batch_info/qwen3-235B-A22B/ep_tp_new_gt/ \
    --case-id 1115
```

### 2. 从 .nsys-rep 自动导出 sqlite

```bash
docker exec mry-aic-collect python3 \
    /host/aiconfigurator/refactor_test_aic/aic_infer_cmp_nsys_profile/compare_aic_nsys.py \
    --nsys-rep /host/aiconfigurator/batch_info/qwen3-235B-A22B/ep_tp_new_gt/nsys/nsys_prefill_case_1115.nsys-rep \
    --data-dir /host/aiconfigurator/batch_info/qwen3-235B-A22B/ep_tp_new_gt/ \
    --case-id 1115
```

脚本会自动调用 `nsys stats --force-export=true` 在同目录下生成 `.sqlite`。

### 3. Decode case 示例

```bash
docker exec mry-aic-collect python3 \
    /host/aiconfigurator/refactor_test_aic/aic_infer_cmp_nsys_profile/compare_aic_nsys.py \
    --nsys-sqlite /host/aiconfigurator/batch_info/qwen3-235B-A22B/ep_tp_new_gt/nsys/nsys_decode_case_6.sqlite \
    --data-dir /host/aiconfigurator/batch_info/qwen3-235B-A22B/ep_tp_new_gt/ \
    --case-id 6
```

---

## 参数说明

| 参数 | 必须 | 说明 |
|------|------|------|
| `--nsys-sqlite` | 二选一 | nsys 已导出的 `.sqlite` 文件路径 |
| `--nsys-rep` | 二选一 | nsys `.nsys-rep` 文件路径（自动导出 sqlite） |
| `--data-dir` | 是 | 数据根目录，需包含 `csv/batches_output.csv` |
| `--case-id` | 是 | CSV 中的 case_id，自动判断 prefill/decode |
| `--iter-index` | 否 | 指定第 N 次迭代（0-based），默认自动选中位数 |
| `--device-id` | 否 | GPU device ID，默认 0 |
| `--multi-rank` | 否 | 跨所有 rank 做 MoE 负载均衡对齐（见下方说明） |
| `--json` | 否 | 额外输出 JSON 格式结果文件 |

---

## 输出内容

### 对比表

```
AIC Op                                       AIC (ms)  nsys (ms)  Diff (ms)     Err%
----------------------------------------------------------------------------------------------------
context_embedding(+ar)                          0.070      3.141     -3.072   -97.8% ⚠️
context_add_norm_1                              1.310      1.271     +0.038    +3.0%
context_qkv_gemm                                6.149      6.467     -0.318    -4.9%
context_attention                              15.483     15.556     -0.073    -0.5%
context_proj_gemm                               4.360      4.742     -0.382    -8.1%
context_moe_pre_dispatch                        6.024      7.382     -1.358   -18.4%
context_add_norm_2                              1.310      1.292     +0.017    +1.3%
context_router_gemm                             1.476      1.470     +0.006    +0.4%
context_moe                                    44.551     36.025     +8.525   +23.7% ⚡
context_moe_post_dispatch                       6.024     21.823    -15.799   -72.4% ⚠️
```

- `⚡` 标记：误差 > 20%
- `⚠️` 标记：误差 > 50%

> **注意**：上述为单 rank（device 0）结果。由于 MoE 负载不均衡，单 rank 的 MoE 偏低而 AllReduce 等待时间偏高。
> 使用 `--multi-rank` 可获得更准确的结果（见下方说明）。

### 占比分析

展示 nsys 各 op 时延占比及柱状图，便于快速定位性能瓶颈。

---

## AIC Op → nsys Kernel 映射关系

在 WideEP+DeepEP 场景下（Qwen3-235B-A22B），每层 Transformer 的 kernel 执行顺序：

| AIC Op | nsys Kernel | 说明 |
|--------|-------------|------|
| `context_add_norm_1` | `RMSNormKernel` / `FusedAddRMSNormKernel` | 层前 RMSNorm（第0层为独立 RMS，后续层为 FusedAdd） |
| `context_qkv_gemm` | `per_token_group_quant` + `deep_gemm (1280→4096)` | FP8 量化 + QKV 投影 |
| `context_attention` | `fused_qknorm` + `fused_rope_store` + `FlashAttnFwdSm90` | QK Norm + RoPE + Flash Attention |
| `context_proj_gemm` | `per_token_group_quant` + `deep_gemm (4096→1024)` | FP8 量化 + Output 投影 |
| `context_moe_pre_dispatch` | `ncclAllReduce` / `cross_device_reduce` | MoE token 分发前通信 |
| `context_add_norm_2` | `FusedAddRMSNormKernel` | 残差连接 + FFN 前 RMSNorm |
| `context_router_gemm` | `nvjet (cublas)` + `splitKreduce` + `topkGatingSoftmax` | Router GEMM + Top-K |
| `context_moe` | `moe_align_block_size` + `count_and_sort_expert` + `fused_moe_kernel` ×2 + `act_and_mul` + `moe_sum_reduce` | 完整 MoE 专家计算 |
| `context_moe_post_dispatch` | `ncclAllReduce` / `cross_device_reduce` | MoE token 聚合后通信 |

**每层固定模式**：`Norm → Quant+GEMM → Attn → Quant+GEMM → AllReduce(pre) → Norm → Router → MoE → AllReduce(post)`

### 层边界识别逻辑

脚本通过全局 AllReduce 计数来识别层边界：
- 第 1 个 AllReduce = embedding AR
- 之后每 2 个 AllReduce = 一层的 `(moe_pre_dispatch, moe_post_dispatch)`
- 层数 = `(AllReduce总数 - 1) / 2`

---

## 多 Rank MoE 负载均衡对齐（`--multi-rank`）

在 WideEP+DeepEP 场景下，MoE 的 token 分配到各 EP rank 上并不均匀。
只看单个 rank 会导致：
- **MoE 计算偏低**：该 rank 分到的 token 少，计算快
- **post_dispatch AllReduce 偏高**：该 rank 提前完成计算后等待最慢 rank

加 `--multi-rank` 后，脚本会：
1. 对每个 device（rank）独立分析逐层 kernel 时延
2. 每层选 **MoE 计算时间最长的 rank**（即处理 token 最多、最慢的那个）
3. 用该 rank 的 `moe` 和 `moe_post_dispatch` 替换 device 0 的值

这样得到的数据反映了真实的 MoE 计算瓶颈，而非单 rank 的局部视角。

### 使用示例

```bash
docker exec mry-aic-collect python3 \
    /host/aiconfigurator/refactor_test_aic/aic_infer_cmp_nsys_profile/compare_aic_nsys.py \
    --nsys-sqlite /host/aiconfigurator/batch_info/qwen3-235B-A22B/ep_tp_new_gt/nsys/nsys_prefill_case_1115.sqlite \
    --data-dir /host/aiconfigurator/batch_info/qwen3-235B-A22B/ep_tp_new_gt/ \
    --case-id 1115 --multi-rank
```

### 效果对比（prefill case 1115）

| Op | 单 rank (dev0) | 多 rank (max) | 说明 |
|---|---|---|---|
| `moe` | 36.025ms | **51.579ms** | 取最慢 rank，反映真实 MoE 瓶颈 |
| `moe_post_dispatch` | 21.823ms | **7.408ms** | 最慢 rank 等待最短 |
| `moe` AIC 误差 | +23.7% | **-13.6%** | 从高估变为合理低估 |
| `post_dispatch` AIC 误差 | -72.4% | **-18.7%** | 从严重低估变为合理误差 |

---

## 迭代选择策略

- 默认跳过第 0 次迭代（warmup），从剩余中取**中位数时延**的迭代
- 使用 `--iter-index N` 可手动指定
- Decode case 自动过滤 `forward_mode='DECODE'` 的 NVTX range（排除 EXTEND 阶段）

---

## Piecewise CUDA Graph 说明

在 prefill 场景下，SGLang 使用 piecewise CUDA graph：
- **Graph 内**：embedding、Norm、GEMM（QKV/Proj）、AllReduce、Router、MoE 等
- **Graph 外（eager）**：仅 Flash Attention（因 KV cache 长度动态变化）
- 每层被 Attention 切成一段 Graph，整个 forward 约 95 个 `cudaGraphLaunch`

---

## MoE 负载不均衡原理

在 EP（Expert Parallelism）场景下，每个 batch 的 token 由 Router 动态分配到各 Expert。
由于 Expert 热度不同（token 分布不均），各 rank 实际处理的 token 数不一样：

- **最慢 rank**：分到最多 token，MoE 计算耗时最长，AllReduce 等待最短
- **最快 rank**：分到最少 token，MoE 计算耗时最短，AllReduce 等待最长
- **AllReduce 是同步点**：所有 rank 完成后才能继续，因此整体时延由最慢 rank 决定

单 rank 视角下 `moe + moe_post_dispatch ≈ 常数`（约 58-60ms），但分项差异很大。
`--multi-rank` 通过选取最慢 rank 来还原真实的 MoE 计算瓶颈。

---

## 配套工具

| 工具 | 用途 |
|------|------|
| `aic_infer_component.py` | 单独获取 AIC per-op 预估值 |
| `nsys_op_debug.py` | 查看 nsys kernel 执行序列（带编号） |
| `nsys_profile.py` | 采集 nsys profiling 报告 |
| `run_prefill.py` / `run_decode.py` | 重放指定 case 的 prefill/decode |
