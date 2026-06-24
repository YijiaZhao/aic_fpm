---
name: fpm
description: >
  This skill should be used when the user says "/fpm", "fpm 对齐", "GLM5 对齐",
  "per-layer 对齐", "AIC per-op 对齐", or wants to run the AIC-vs-silicon per-layer
  timeline alignment flow FOR nvidia/GLM-5-NVFP4 (DSA, sglang 0.5.13, B200). FPM =
  AIC per-op standard table → pick ≥5% non-comm ops → capture per-layer microbench
  timelines (nsys --cuda-graph-trace=node) → compare against real serve timeline.
  NOTE: this skill is GLM-5-NVFP4-specific; alignment is done per-model (other
  models need their own kernel map / ServerArgs / op set).
---

# /fpm — nvidia/GLM-5-NVFP4 AIC vs Silicon per-layer 对齐 skill

> **本 skill 专属 `nvidia/GLM-5-NVFP4`**(DSA / sglang 0.5.13 / B200)。里面的
> ServerArgs、kernel→op 映射表、op 集、microbench 参数都是 GLM5 特定的。
> **对齐是逐 model 做的** —— 别的模型(Qwen3 / DSV4-Pro / …)要各自的 align skill
> (方法论通用,但 Step0 的启动配置、Step3 的 collector 入口、Step4 的 kernel 映射要按该 model 重写)。

## 目标

把 **AIC 的 per-op / per-module 预测**和**真机 serve 的 timeline** 在 **per-layer 粒度**对齐;
对不齐时定位到是哪个 op/module、差在采集还是 delta 修正。

对齐单位是**单层**(serve 时间线是逐层重复的 pattern),不是 e2e。

> ⚠️ 本 skill 持续扩充(用户会不断加增量),改动请追加到对应步骤 + 末尾「增量记录」。

---

## 最终交付物(对任何模型,就这 3 样)

跑完一个 case,用户要的只有 3 样东西:

1. **8 列表格**(每个 op/module 一行,全列):
   `op/module | 层数 | 总(ms)AIC | 实测总(ms) | 占比 | AIC per-layer | 实测 per-layer | microbench per-layer`
   - `总(ms)AIC` / `AIC per-layer` = AIC 预测(Step1)。`实测总` / `实测 per-layer` = serve nsys 按 op 归类
     (Step4,**含通信** op,如 dispatch = 每层 2× nccl AllReduce)。`microbench per-layer` = **只**对
     ≥5% 非通信 compute op 抓(Step3,其余填 —)。`实测总 = 实测 per-layer × 层数`,放在第 3 列右边对标 AIC 总。
2. **实测 nsys**:serve 整模型 timeline 一份(Step0 的 static-mode replay 抓的,`--cuda-graph-trace=node`)。
3. **>5% 非通信 op 的 microbench nsys**:每个 ≥5% 非通信 compute op 一份(Step3 collector 单卡单抓)。

3 个(及以上)nsys 放进一个文件夹交付:`nsys_rep_dir/fpm_<model>_<config>_<shape>/`,用户 `scp -r` 拷走。
误差靠读表(AIC 列 vs 实测列)看,不单列 —— 这是 4286/2112 等 case 反复确认的最终形态。

---

## 环境与文件位置

**AIC 侧**(可在任意有 AIC SDK 的机器,本项目默认 `.8` = `root@10.6.131.8`,经 `ssh kimiz@computelab` 跳):
- 代码根:`/raid/kimi/ds4_new/aiconfigurator`
- 对齐工具:`/raid/kimi/ds4_new/aic_fpm/aic_infer_cmp_nsys_profile/`
  - `aic_infer_component.py` — AIC per-op 标准表(本流程第 1 步)
  - `nsys_profile.py` — 抓 serve nsys timeline
  - `compare_aic_nsys.py` — 核心对齐器(nsys kernel → AIC op,per-layer 对比)
  - `compare_aic_nsys_usage.md` / `nsys_op_debug.py` — 用法 + kernel 调试
- 统一配置:`/raid/kimi/ds4_new/aic_fpm/config.py`(MODEL_CONFIG_KWARGS、tp/cp、数据目录)
- 运行环境变量:
  ```bash
  export PYTHONPATH=/raid/kimi/ds4_new/aiconfigurator/src:/raid/kimi/ds4_new/aiconfigurator:/raid/kimi/ds4_new
  ```

**Silicon 侧**(抓 module/serve timeline 需 **B200**):
- Teleport pod `kimiz-glm5`(b200-resv-7),或 Slurm B200(computelab-sc-01)。
- module 采集器:`aiconfigurator/collector/sglang/collect_mla_module.py`(dsa_context_module 等)。

**GLM5-NVFP4 关键参数**:num_hidden_layers=78,first_k_dense_replace=3,index_topk=2048,
max_position=202752;`--kv-cache-dtype fp8_ds_mla`;DSA backend = **trtllm**(非CP默认)/ **flashmla_kv**(CP)。

---

## 流程

### Step 0 — 启动配置 + 构造/跑 silicon case(对齐的地基)

**这是整个流程的第 0 步、前提**:先用 `run_prefill_glm5.py` 把 serve 启动配置定下来 + 构造出要对齐的
那个 case(bs, isl, prefix),才有后面 —— Step1 AIC 5 列表、Step2 找 >5%、Step3 收 timeline 全部
**复用 Step 0 这同一份 (config, case shape)**。配置错(TP/CP/backend 搞混)→ 对齐全错。

`run_prefill_glm5.py` 同时是「启动配置载体」+「silicon 侧 case 构造/运行器」,**2 mode**:
- **Mode 1 `--csv-case-id`**:从 captured trace JSONL 读真实 batch(用户做过 trace 时)。
- **Mode 2 `--static-mode --bs N --avg-isl X --avg-past-kv Y`**:**不依赖 trace**,从 sglang ShareGPT
  (`download_and_cache_hf_file` 自动从 HF 下/缓存)取真实 token,**严丝合缝**构造 bs×(past_kv+isl)
  精确 shape(每条 distinct 首 token 防 radix 共享)。开发完没 trace 时走这个。
  - ⚠️ **prefix(past_kv)必须是 `page_size`(GLM5=64)的倍数**:radix cache 按页缓存,
    非对齐的 prefix(如 2099)warmup 时被向下取整(→2048),导致 extend 吃掉余数、shape 对不上
    (shape-guard 会报 `target shape mismatch` 直接 fail)。真实 captured prefix 天然对齐(serve
    本就按页缓存,如 95808=1497×64);static-mode 自己给的 prefix 要取 64 的倍数(`((p+63)//64)*64`)。
- 两 mode 都套用下面那份 GLM5 `ServerArgs`,跑真权重整模型(**tp8 = 8 GPU**),可配
  `nsys --capture-range=cudaProfilerApi --cuda-graph-trace=node` 直接抓真机 timeline(= Step 4 的源)。

**启动配置来源两级:**

对齐的一切配置必须 = **serve 实际跑的那条指令**里的启动配置,否则 AIC 配错(TP/CP/backend
搞混 → 对齐全错)。来源两级:

1. **主源:解析 sglang serve 启动指令**(实测时优先拿这条)。要抽的 flag → AIC 映射:
   | sglang flag | → AIC / 抓取 |
   |---|---|
   | `--model-path` | MODEL_NAME / MODEL_PATH |
   | `--tp-size` | `tp_size`(→ attention per-rank num_heads = num_attention_heads // tp) |
   | `--attn-cp-size` / `--context-parallel-size` | `cp_size`(>1 → context_attention 走 delta/`_query_cp`,抓 backend=flashmla_kv) |
   | `--ep-size` / `--moe-ep-size` | `moe_ep_size` |
   | `--dp-size` / 启用 attention DP | `attention_dp_size`(per-rank bs = global_bs // dp) |
   | `--kv-cache-dtype`(`fp8_ds_mla`) | kvcache_quant_mode |
   | `--nsa-prefill-backend` / `--nsa-decode-backend`(`flashmla_kv`/`trtllm`) | DSA backend(抓取末参) |
   | `--chunked-prefill-size` | chunk cap(单 step token 上限) |
   | quant(nvfp4) | gemm/fmha/moe quant mode |

2. **Fallback:内置启动配置 = 现成的 replay 脚本**(开发完、拿不到 serve 指令时用)。
   不要手攒 profile —— 直接用项目里已编码好 serve `ServerArgs` 的启动脚本作为内置:
   - **GLM5 prefill**:`aic_infer_cmp_nsys_profile/run_prefill_glm5.py`(`--attn-cp-size N` 切 CP)
   - **GLM5 decode**:`aic_infer_cmp_nsys_profile/run_decode_glm5.py`
   - 其它模型:`run_prefill_deepseekv4.py` / `run_prefill.py` / `run_decode.py`

   GLM5 内置 `ServerArgs` 关键字段(摘自 `run_prefill_glm5.py`,= b200_glm5 serve 实测路径):
   ```
   tp_size=8(默认), ep_size=1(chunked_replay: MoE-TP 不是 EP), moe_dense_tp_size=1
   attention_backend="nsa", kv_cache_dtype="fp8_e4m3"(serve CLI --kv-cache-dtype fp8_ds_mla)
   quantization="modelopt_fp4", moe_runner_backend="flashinfer_trtllm"
   chunked_prefill_size=16384, max_prefill_tokens=16384, max_running_requests=256, mem_fraction_static=0.8
   非CP: nsa_prefill_backend="trtllm", nsa_decode_backend="trtllm"
   CP(attn_cp_size>1): enable_dsa_prefill_context_parallel=True, dsa_prefill_cp_mode="round-robin-split",
                       dsa_prefill_backend="flashmla_kv", dsa_decode_backend="flashmla_kv"
   enable_flashinfer_allreduce_fusion=True, disable_shared_experts_fusion=True, enable_dp_lm_head=False
   cuda_graph_max_bs=8, disable_cuda_graph=False, disable_cuda_graph_padding=True
   enable_piecewise_cuda_graph=True, enforce_piecewise_cuda_graph=True, disable_overlap_schedule=True
   ```
   模型常量:num_attention_heads=64,num_hidden_layers=78,first_k_dense_replace=3,
   index_topk=2048,max_position=202752。

这个 config 同时驱动:**AIC 侧**(`config.py` 的 `MODEL_CONFIG_KWARGS` 必须与之一致)
和 **Step 3 抓取**(backend / per-rank num_heads)。

### Step 1 — AIC 侧标准输出(5 列表)

跑 `aic_infer_component.py`,得到**对齐标准表**:`op/module | 层数 | 总(ms) | 占比 | per-layer(ms)`,
`★` = 占比≥5%,TOTAL 行 = 总时间 + 整体 per-layer(=总/num_layers)。

```bash
cd /raid/kimi/ds4_new/aic_fpm/aic_infer_cmp_nsys_profile
export PYTHONPATH=/raid/kimi/ds4_new/aiconfigurator/src:/raid/kimi/ds4_new/aiconfigurator:/raid/kimi/ds4_new

# 手动指定(注意 --isl 是 TOTAL = isl+prefix)
python3 aic_infer_component.py manual --phase prefill --batch-size 1 --isl 112192 --prefix 95808
# 或按 case_id 从数据集读
python3 aic_infer_component.py case --data-dir /raid/kimi/ds4_new/b200_glm5_pccg_data_tp --case-id 1081
```

输出示例(case 1081,TP8 prefill bs1 isl16384 prefix95808):
```
=== Prefill AIC per-op 标准表 (★ = 占比 >= 5%) ===
op/module                            层数        总(ms)       占比    per-layer
----------------------------------------------------------------------------
★ context_attention                  78    1296.9100   86.70%      16.6271
★ context_moe                        78      78.7870    5.27%       1.0101
  context_moe_pre_dispatch           78      43.3682    2.90%       0.5560
  ...
----------------------------------------------------------------------------
  TOTAL                              78    1495.8354  100.00%      19.1774
```

实现:`_print_op_table()`(per-layer = 总/`op._scale_factor`;占比 = op总/sum;遍历
`model.context_ops`/`model.generation_ops`)。`estimate_batch_per_op` 的 AIC 计算未改动
(=`summary.get_context_latency_dict()`)。

**产出**:占比 ≥5% 的 op/module 列表(只对这些做 timeline 对齐;<5% 忽略,且占比按 case 动态算)。

⚠️ **≥5% 里通信类 op 不抓 timeline**:`context_moe_pre_dispatch` / `context_moe_post_dispatch` /
`context_p2p` / allreduce 等都是通信(dispatch/AllReduce),通信时间另算(系统/拓扑相关,非
module microbench 范畴),**即使占比 ≥5% 也跳过**。只对 **compute op**(context_attention、
context_moe 等)抓 timeline。

### Step 2 — 对每个 ≥5% op,按 AIC 建模分类:直接采集 vs delta

读 AIC 建模代码(`aiconfigurator/src/aiconfigurator/sdk/operations/` 下对应 op,如
`dsa.py` 的 `ContextDSAModule`、`moe.py` 的 `MoE`),判断这个 op 在 **SILICON 模式**下:

- **直接采集**(含网格点之间**插值** —— 插值**不算** delta):
  module 的值就是采集数据查表/插值得来。
  → **直接拿当前 case 抓这个 module 的 timeline**。
- **delta**(借助 **op 级修正**去调 module,典型 = **GLM5 CP**,cp_size>1 走
  `ContextDSAModule._query_cp`,用 delta_mqa / delta_topk / base 等 op 项修正 module):
  → **把所有 delta 相关的 op timeline 全抓下来**。

判定口径(用户定义):**delta 不包括插值**;delta 专指"借助 op 修正 module"(CP 这种)。

已查清(GLM5):
| op(≥5%) | SILICON 建模 | 分类(TP8 / cp=1) |
|---|---|---|
| context_attention | 采集 `dsa_context_module`,topk-piecewise-from-raw + prefix 插值 | 直接采集 |
| context_moe | 采集 `moe`,按 token 数 interp_1d + workload_distribution | 直接采集 |
> cp_size>1 时 context_attention 改走 `_query_cp` → 变 delta(需摸清借了哪些 op,见增量 TODO)。

### Step 3 — 抓 timeline(必须经 AIC collector,`nsys --cuda-graph-trace=node`)

**核心原则**:要抓的是 **AIC collector 把 module 单独跑出来的 microbench timeline**
——因为那才**等于 AIC 数据的来源**,跟 AIC 的预测可比。**不要**手搓一个独立 nsys 去抓,
抓出来的 kernel/形状对不上 collector 采集时的样子(本 session 追 dsa_context bug 就是靠
collector 的 eager timeline 对比 serve)。

- **MLA / dsa_context_module(及其它 module 类 op)**:走 `collector/sglang/collect_mla_module.py`
  的 `run_mla_module(...)`,**agent 自己构造对应的单个测例**喂给它,**一次单抓一个**(不批量、不从 sweep 挑)。

  **第一原则:先确认单卡能抓 —— dsa module microbench 就是单卡(`CUDA_VISIBLE_DEVICES=0`)。**
  不需要 4/8 GPU,起 1-GPU pod 即可。

  **num_heads 必须 = AIC 实际查询的 per-rank heads**(= `num_attention_heads // tp_size`),
  用 `op._num_heads` 确认。GLM5 TP8 = **8**(64/8),**不是 64**。⚠️ 历史上抓过 64 是错的。

  **构造测例**(shape 来自 Step 1 该 op + `op._num_heads`/`op._cp_size`):
  - `AIC_DSA_CONTEXT_SEQ_LENS=<s>` `AIC_DSA_CONTEXT_PREFIX_LENS=<prefix>`(各只给一个值,= scope 过滤,见 `_build_module_test_cases`)
  - `run_mla_module('dsa', <num_heads>, 'nvidia/GLM-5-NVFP4', 'fp8','bfloat16','bfloat16', True, 0, <out>, None, <bs_filter>, <tp>, <dsa_backend>)`
    (签名:attn_type, head_num, model, kv_cache_dtype, compute_dtype, gemm_type, is_prefill, gpu_id, output_path, attention_backend, batch_size_filter, target_tp_size, dsa_prefill_backend)

  **实测模板**(case 1081 TP8 prefill,bs1 isl16384 prefix95808,num_heads=8):
  ```bash
  # /workspace/run_1081.py
  import os
  from collect_mla_module import run_mla_module
  run_mla_module('dsa', 8, 'nvidia/GLM-5-NVFP4', 'fp8', 'bfloat16', 'bfloat16',
                 True, 0, '/workspace/nsys_dsa_out', None, 1, 1, 'trtllm')  # 末参换 flashmla_kv 抓 CP/对照版

  # 跑(单卡 + node 粒度 cuda graph):
  export AIC_DSA_CONTEXT_SEQ_LENS=16384 AIC_DSA_CONTEXT_PREFIX_LENS=95808 CUDA_VISIBLE_DEVICES=0
  export SGLANG_LOAD_FORMAT=dummy SGLANG_TEST_NUM_LAYERS=2   # dummy 权重,不下模型
  export PYTHONPATH=<aic>:<aic>/src:<sglang>/python
  nsys profile -o /workspace/nsys_1081 --force-overwrite true -t cuda,nvtx \
       --cuda-graph-trace=node python3 /workspace/run_1081.py
  ```
  - **单抓**:每次只跑这一个测例 + nsys,一个 op / 一个 case 地抓。
  - 环境:**B200**,镜像 `lmsysorg/sglang:v0.5.13`,挂 PVC `dsv4-pro-cache`(代码在 `/workspace/cache/aic_dot8_sync/aiconfigurator`)。
  - backend 要对:非CP=`trtllm` / CP=`flashmla_kv`;`--kv-cache-dtype fp8_ds_mla`。
  - eager(collector 在超过 piecewise max token 时本就走 eager;timeline 要 eager 的)。
  - `nsys profile --cuda-graph-trace=node`(node 粒度,cuda graph 内单 kernel 可见)。
  - 需 **B200**(数据是 b200_sxm)。
  - 本 session 已产出过该模板:`nsys_rep_dir/glm5_dsa_module_case1081_tp8_isl16384_prefix95808_{flashmla_kv,trtllm}_eager.nsys-rep`。
    ⚠️ **精确命令待真机跑通后填入**(从那批 rep 的产出命令翻出固化)。
- **delta 的 op(CP)**:把 `_query_cp` 借助的每个 op-级修正项的 timeline 都抓(同样经 collector)。
- 抓完的 `.nsys-rep` 按项目规范 copy 到 `nsys_rep_dir/` 并给下载命令。

### Step 4 — Silicon 侧:从 serve nsys 把每个 op 的实测 per-layer 拆出来(已验证)

`实测` 列的来源。方法(已在 bs3/isl2048/prefix2112 验证):

```bash
nsys export --type sqlite --force-overwrite true -o serve.sqlite <serve.nsys-rep>
# 同样导出每个 >5% op 的 microbench rep -> 它的 kernel 名集合 = 该 op 的“指纹”
```

1. **>5% compute op**:用该 op 的 **microbench rep 的 kernel 名集合**当指纹,在 serve 里认领同名/同类 kernel。
   主导 kernel 在两边都主导 = 采集正确(如 attention 的 `fmhaSm100f` 在 microbench 和 serve 都主导)。
2. **通信 op**:就是 nccl kernel —— 每层 **2× `ncclDevKernel_AllReduce`** = `context_moe_pre_dispatch` /
   `context_moe_post_dispatch`(各 ≈ 总AllReduce/2/层)。`AllGather`(非逐层、count≠层数)是另一类集合通信,
   不归 dispatch。`p2p` ≈ ReduceScatter,~0。
3. **小 op(norm/router/shared gemm)**:靠**层内 kernel 时间序**归属。取相邻两个 `fmhaSm100f`(一层一个)
   之间的窗口,按 GLM5 单层执行序认领:
   `norm1 → kv_a_proj/q_b_proj → indexer(rope/hadamard/mqa_logits/topk/get_k_and_s) → fmha → o_proj →
    AllReduce(pre) → norm2 → router_gemm → moe(routing/bmm/finalize) → AllReduce(post) → shared(gate_up/act/ffn2)`
4. 每个 op 的 kernel duration 求和 = `实测总`,÷层数 = `实测 per-layer`(per-AIC 惯例统一 /num_layers)。

**GLM5 kernel → AIC op 映射表(device 0,已验证)**:
| AIC op | serve kernel |
|---|---|
| context_attention | `fmhaSm100f` · `topk_transform_prefill` · `sm100_fp8_mqa_logits` · `fast_hadamard_transform` · `fused_rope` · `RopeQuantize` · `_get_k_and_s` · `_act_quant` · `fused_store_indexer` · `set_mla_kv_buffer` · attn 投影 nvjet gemm · q_a/kv_a `RMSNorm` |
| context_moe | `bmm_E2m1` · `bmm_Bfloat16_E2m1` · `moe::dev::finalize` · `moe::dev::routing*` · moe nvfp4 quant(75×) |
| context_moe_pre/post_dispatch | `ncclDevKernel_AllReduce`(每层 2×) |
| context_add_norm_1/2 | `FusedAddRMSNorm`(每层 2×) |
| context_router_gemm | dispatch 前的 nvjet gemm(75×)· `topkGating` |
| context_shared_gate_up/ffn2/act | cutlass `blockscaled` gemm · `act_and_mul` |
| context_logits/embedding/p2p | 一次性,~0 |

### Step 5 — 输出 8 列表格(最终交付物 ①)

| op/module | 层数 | 总(ms)AIC | 实测总(ms) | 占比 | AIC per-layer | 实测 per-layer | microbench per-layer |
|---|---|---|---|---|---|---|---|
| context_attention | 78 | 152.04 | 155.85 | 61.52% | 1.9492 | 1.998 | 2.171 |
| context_moe | 78 | 35.40 | 21.45 | 14.32% | 0.4538 | 0.275 | 0.4569 |
| context_moe_pre_dispatch | 78 | 22.79 | 23.17 | 9.22% | 0.2921 | 0.297 | — |
| context_moe_post_dispatch | 78 | 22.79 | 23.17 | 9.22% | 0.2921 | 0.297 | — |
| context_add_norm_1/2 | 78 | 3.91 ×2 | 5.08 ×2 | 1.58%×2 | 0.0502 ×2 | 0.065 ×2 | — |
| context_shared_gate_up/ffn2 | 78 | 2.22/2.05 | 1.40/1.33 | 0.9/0.83% | 0.028/0.026 | 0.018/0.017 | — |
| context_router_gemm | 78 | 1.63 | 1.64 | 0.66% | 0.0210 | 0.0211 | — |
| context_shared_act_gate | 78 | 0.35 | 0.41 | 0.14% | 0.0045 | 0.0053 | — |
| context_logits/embedding/p2p | 1 | ~0.05 | ~0 | <0.1% | ~0 | ~0 | — |
| **TOTAL** | 78 | 247.14 | ~245 | 100% | 3.1685 | — | — |

读表即对齐:AIC 列 vs 实测列。误差大的 op → 回 Step2 看是采集错(backend/pattern/kernel tile)还是 delta 错。
(本 case 结论:attention/dispatch/norm/router 全对齐;**moe AIC 高估 65%**,根因采集 bmm tile 与 serve 不一致。)

---

## Refactor 组件清单(每步用哪个、怎么用、里面要有什么)

根目录:`/raid/kimi/ds4_new/aic_fpm/`(.8;pod 上为 `/workspace/cache/aic_fpm/`)。
子目录 `aic_infer_cmp_nsys_profile/` = 对齐工具集。

| 组件 | 步骤 | 作用 / 怎么用 | 里面必须有 |
|---|---|---|---|
| `config.py` | Step0/1 | 全局配置:`MODEL_NAME/MODEL_PATH`、`MODEL_CONFIG_KWARGS`(tp/ep/dp/moe_tp)、`AIC_SYSTEM/BACKEND/VERSION`、`DATA_DIR`、`PREFILL_CORRECTION_FACTOR`。改它切模型/配置。 | `MODEL_CONFIG_KWARGS` 必须和 serve 实际配置一致(tp/cp/ep);`AIC_VERSION` 对应采集库版本(GLM5=0.5.13) |
| `utils.py` | Step1/3 | `RequestInfo(input_length, past_kv_length)`、`prefill_seq_imbalance_correction`。 | RequestInfo 数据类 |
| `aic_infer_cmp_nsys_profile/run_prefill_glm5.py` | **Step0**(实测) | silicon 侧 replay,**2 mode**:`--csv-case-id`(读 trace JSONL)/ `--static-mode --bs --avg-isl --avg-past-kv`(造合成 case,sharegpt 真 token)。套 nsys `--capture-range=cudaProfilerApi --cuda-graph-trace=node`。tp8=8GPU。 | GLM5 `ServerArgs`(nsa/fp8_e4m3/modelopt_fp4/trtllm·CP flashmla_kv);`cudaProfilerStart/Stop` 包 measured generate;`SGLANG_REPLAY_EXPECT_SHAPE` shape-guard;`--static-mode`(page64 取整 + sharegpt 构造);CP 由 `--attn-cp-size` 切 |
| `aic_infer_cmp_nsys_profile/run_decode_glm5.py` | Step0(decode) | decode 版(prefill 跑通后再做)。 | 同上,decode 路径 |
| `hook_dataset_collector/hook.py` | Step0 | `Scheduler.run_batch` 打 NVTX(标 bs/isl/past_kv)+ shape-guard 断言实际 shape。run_prefill 自动 import。 | run_batch NVTX range + EXPECT_SHAPE 断言 |
| `aic_infer_cmp_nsys_profile/aic_infer_component.py` | **Step1**(AIC 表) | AIC per-op 标准 5 列表。`manual --phase prefill --batch-size N --isl <TOTAL=extend+prefix> --prefix P` / `case --data-dir --case-id`。 | `_print_op_table()`(占比 + per-layer=总/`op._scale_factor` + ≥5% 标记 + TOTAL);`estimate_batch_per_op`(=`get_context_latency_dict`,不改 AIC 计算) |
| AIC SDK `operations/dsa.py`·`moe.py` | Step2 | 判定 op 是直接采集还是 CP-delta;读 `op._num_heads/_scale_factor`、moe `_topk/_num_experts/_hidden_size/_inter_size/_moe_tp_size/_moe_ep_size/_workload_distribution`(给 Step3 microbench 当参数)。 | — |
| `collector/sglang/collect_mla_module.py` `run_mla_module(...)` | **Step3**(attn microbench) | dsa/attention module 单卡单抓。scope env `AIC_DSA_CONTEXT_SEQ_LENS/PREFIX_LENS/BATCH_SIZES` 锁单 case;`run_mla_module('dsa', heads, model, 'fp8','bfloat16','bfloat16', True, 0, out, None, bs, tp, backend)`;套 nsys。 | per-backend(trtllm/flashmla_kv)、prefix>65536 eager 兜底、chunked_prefill cap |
| `collector/sglang/collect_moe.py` `run_moe_torch(...)` | **Step3**(moe microbench) | moe module 单卡单抓。`run_moe_torch(moe_type, num_tokens, hidden, inter, topk, num_experts, moe_tp, moe_ep, model, distributed=, power_law_alpha=, moe_backend=, perf_filename=)`;套 nsys。参数全来自 Step2 读的 AIC moe op。 | nvfp4/trtllm moe 路径;perf_filename 落 latency |
| `nsys export --type sqlite` + sqlite 查询 | **Step4**(实测拆分) | serve rep → sqlite;用 microbench 指纹 + 层内时序 + nccl 把 kernel 归到 AIC op(见上「映射表」)。 | — |
| `aic_infer_cmp_nsys_profile/compare_aic_nsys.py` | Step4(目标) | AIC vs nsys per-op 对比器(现 Qwen3 写死,GLM5 版 = 把上面映射表 + 层边界写进 `KERNEL_CLASSIFY_RULES`)。 | GLM5 `KERNEL_CLASSIFY_RULES` + `fmhaSm100f`/AllReduce 切层 + 出 8 列表 |
| `aic_infer_cmp_nsys_profile/nsys_profile.py` | Step0(备选) | 给已起的 serve 抓 nsys(替代 run_prefill replay)。 | — |
| `aic_infer_cmp_nsys_profile/run_prefill2.py` | 参考 | static-mode 的原型(合成 mixed-prefill + cudaProfiler),通用非 GLM5。 | — |

---

## 自动化状态 / GAP

整条流程已**手动端到端验证通过**(case bs3/isl2048/prefix2112):Step0 static-mode replay → Step1 AIC 表 →
Step3 microbench 单抓(attention/moe)→ Step4 sqlite + 指纹/层内时序归类(含通信)→ Step5 8 列表 + 打包交付。

未自动化的部分(现在靠 sqlite 查询 + 上面映射表手动做,要做成一键 `compare_aic_nsys.py` GLM5 版):
1. 把上面「GLM5 kernel→op 映射表」+ 层内时序归属写成代码(替代 Qwen3 写死的 `KERNEL_CLASSIFY_RULES`)。
2. 层边界:GLM5 用 `fmhaSm100f`(每层一个)或 `AllReduce` 计数切层。
3. 一键吐出 8 列表 + 自动打包 3 类 nsys 到 `nsys_rep_dir/fpm_<...>/`。

---

## 增量记录(持续追加)

- 2026-06-23 初版:Step1 AIC 标准 5 列表已落地(`aic_infer_component.py` `_print_op_table`)。
  delta 口径 = CP 借 op 修正 module(插值不算)。GLM5 的 compare_aic_nsys 适配未做。
- 2026-06-23 Step4 silicon 侧静态抓取工具就绪:`run_prefill_glm5.py` 加了 `--static-mode`。
  - 2 mode:① `--csv-case-id`(读 trace JSONL,原有);② `--static-mode --bs N --avg-isl X
    --avg-past-kv Y`(**不依赖 trace**)——从 sglang ShareGPT(`sample_sharegpt`/`download_and_cache_hf_file`
    自动从 HF 下载缓存)取真实 token,严丝合缝构造 bs×(past_kv+isl) 精确 shape,每条 distinct 首 token
    避免 radix 共享;复用 GLM5 ServerArgs + cudaProfiler + shape-guard/hook 断言。
  - 构造已验证严丝合缝(bs2/isl16384/prefix95808 → 每条 len=112192 精确)。
  - tp8 真机 trial(bs3/isl2048/prefix2099)端到端跑通:模型加载 + GLM5 ServerArgs 全生效;
    但 prefix=2099 非 64 对齐 → shape-guard 报 `target shape mismatch`(实际 extend2099/past2048)。
    **结论:static-mode prefix 必须取 page_size(64) 的倍数;shape-guard 工作正常,严丝合缝有保障。**
  - ⚠️ **完整 silicon replay = real weights 整模型 = tp8 = 8 GPU**(1 卡只够 AIC collector 单卡抓)。
    跑法:8-GPU B200 pod + `nsys --capture-range=cudaProfilerApi --cuda-graph-trace=node python3 run_prefill_glm5.py --static-mode ...`。
  - 纯离线环境 sharegpt 下不了需预 stage(PVC `dsv4-pro-cache` 已缓存)。
- 2026-06-23 Step3 验证通过(B200 单卡):case 1081 context_attention 用上面模板单抓成功。
  - 单卡足够(`CUDA_VISIBLE_DEVICES=0`,1-GPU pod `k8s_launch_dir/fpm-b200-1gpu.yaml`,
    镜像 `lmsysorg/sglang:v0.5.13`,挂 `dsv4-pro-cache`)。
  - `op._num_heads` 确认 = **8**(TP8 per-rank,不是 64)。
  - collector microbench = **17.712 ms** vs AIC 预测 per-layer **16.627 ms**(~6.5%,采集自洽)。
  - rep:`nsys_rep_dir/glm5_dsa_module_case1081_tp8_h8_isl16384_prefix95808_trtllm.nsys-rep`。
  - ⚠️ 注意:nsys 收尾的「Downloaded symbol information」很慢但非必需,rep 写完(`Generated:`)即可用/拷走;
    可 `pkill -9 -f <driver>` 杀掉符号下载收尾再拷(GPU 已空)。

- 2026-06-24 **整条流程端到端跑通 + 8 列表交付物定稿**(case tp8/bs3/isl2048/prefix2112):
  - Step0 实测:`run_prefill_glm5.py --static-mode --bs 3 --avg-isl 2048 --avg-past-kv 2112`(8-GPU pod
    `fpm-b200-8gpu`)→ `[replay_run_batch] EXTEND input=[2048×3] past=[2112×3] latency 255.2ms`,shape-guard 通过。
    static-mode 已加 page_size(64) 向下取整(`avg_past_kv // 64 * 64`,2099→2048 自动)。
  - Step1 AIC 表:total 247.14ms;≥5% = context_attention(61.5%)、context_moe(14.3%)、dispatch×2(各 9.2%,通信)。
  - Step3 microbench(1-GPU pod):attention `run_mla_module('dsa',8,...,3,1,'trtllm')` = 2.171ms/层;
    moe `run_moe_torch('nvfp4',6144,6144,2048,8,256,8,1,...,'power_law',1.01)` = 0.4569ms/层(参数从
    AIC op `_topk/_num_experts/_hidden_size/_inter_size/_moe_tp_size/_moe_ep_size/_workload_distribution` 读)。
  - Step4 实测拆分(指纹+层内时序+nccl):见上「GLM5 kernel→op 映射表」。
  - 8 列表见 Step5。**结论:attention/dispatch/norm/router/shared_act 全对齐;moe AIC 高估 65%**
    (microbench 0.457≈AIC 0.454,但 serve 实测仅 0.275 → **采集 bmm tile 与 serve 不一致**:
    microbench `bmm_Bfloat16..._tokFp32`,serve `..._t128x128x256u2`)。这是该 case 真正要回去修的点。
  - 交付:3 个 nsys 放 `nsys_rep_dir/fpm_glm5_tp8_bs3_isl2048_prefix2112/`(serve + attn microbench + moe microbench)。
