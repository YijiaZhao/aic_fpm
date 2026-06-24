"""
统一配置模块
============
集中管理路径、模型参数、AIC SDK 配置以及各阶段输出子目录。
修改此文件即可切换到不同模型 / 不同数据目录。
"""

import os

from aiconfigurator.sdk.common import (
    CommQuantMode,
    FMHAQuantMode,
    GEMMQuantMode,
    KVCacheQuantMode,
    MoEQuantMode,
)

# ============================================================================
# 数据目录（容器内路径；宿主机映射路径请自行调整）
# ============================================================================

DATA_DIR = os.environ.get(
    "AIC_DATA_DIR",
    "/raid/kimi/ds4_new/dsv4_main_data",
)

# JSONL 文件名（由 hook.py 生成）
# SCHEDULE_JSONL_FILENAME = "TP0-EP0_schedule_batch.jsonl"
SCHEDULE_JSONL_FILENAME = "TP0_schedule_batch.jsonl"

# ============================================================================
# 各阶段输出子目录
# ============================================================================
SUBDIR_CSV = "csv"  # 阶段 1 输出
SUBDIR_ESTIMATION = "estimation"  # 阶段 2 输出
SUBDIR_ACCURACY = "accuracy"  # 阶段 3: MAPE 统计输出
SUBDIR_SIGNED_ERROR = "signed_error"  # 阶段 3: 误差桶分析输出

# ============================================================================
# 模型 / 后端参数
# ============================================================================
MODEL_NAME = "sgl-project/DeepSeek-V4-Flash-FP8"
MODEL_PATH = f"/root/.cache/huggingface/hub/models--sgl-project--DeepSeek-V4-Flash-FP8/snapshots/ae01d80c06cdfe30581edfd0e1c5449dc7ed7f17/"
BACKEND_NAME = "sglang"

# AIC 性能数据库参数
AIC_SYSTEM = "h20e_sxm"
AIC_BACKEND = "sglang"
AIC_VERSION = "0.5.13"

# ============================================================================
# ModelConfig 参数（对应 aiconfigurator.sdk.config.ModelConfig）
# ============================================================================

MODEL_CONFIG_KWARGS = dict(
    pp_size=1,
    tp_size=8,
    moe_tp_size=8,
    moe_ep_size=1,
    attention_dp_size=1,
    enable_wideep=False,
    workload_distribution="power_law_1.2",
    # workload_distribution="balanced",
    gemm_quant_mode=GEMMQuantMode.fp8_block,
    moe_quant_mode=MoEQuantMode.fp8_block,
    kvcache_quant_mode=KVCacheQuantMode.fp8,  # fp8 -> fp8_e4m3
    fmha_quant_mode=FMHAQuantMode.bfloat16,
    comm_quant_mode=CommQuantMode.half,
)

# ============================================================================
# SGLang Server 启动命令（用于 nsys_profiler 自动解析为 ServerArgs）
# 直接粘贴你部署 sglang 时用的命令即可
# ============================================================================
SGLANG_LAUNCH_CMD = """
SGLANG_JIT_DEEPGEMM_PRECOMPILE=0 \
SGLANG_DSV4_FP4_EXPERTS=0 \
sglang serve \
  --trust-remote-code \
  --model-path /nfs/lvm/models/sgl-project/DeepSeek-V4-Flash-FP8/ \
  --tp 8 \
  --cuda-graph-max-bs 256 \
  --disable-cuda-graph-padding \
  --max-running-requests 256 \
  --tool-call-parser deepseekv4 \
  --reasoning-parser deepseek-v4
"""

# ============================================================================
# 估算校正系数（来自 aic_fpm/stage2_run_aic_estimation.py ）
# ============================================================================
DECODE_CORRECTION_FACTOR = 1.0
PREFILL_CORRECTION_FACTOR = 1.0

# ============================================================================
# 辅助函数：获取各阶段的绝对输出目录
# ============================================================================


def get_output_dir(data_dir: str, subdir: str) -> str:
    """返回子目录的绝对路径，不存在则创建。"""
    path = os.path.join(data_dir, subdir)
    os.makedirs(path, exist_ok=True)
    return path
