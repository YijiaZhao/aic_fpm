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
    "/raid/kimi/ds4_new/b200_glm5_pccg_data",
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
MODEL_NAME = "nvidia/GLM-5-NVFP4"
MODEL_PATH = "/raid/kimi/ds4_new/model_configs/nvidia--GLM-5-NVFP4"
BACKEND_NAME = "sglang"

# AIC 性能数据库参数
AIC_SYSTEM = "b200_sxm"
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
    workload_distribution="power_law",
    # workload_distribution="balanced",
    gemm_quant_mode=GEMMQuantMode.nvfp4,
    moe_quant_mode=MoEQuantMode.nvfp4,
    kvcache_quant_mode=KVCacheQuantMode.fp8,  # fp8 -> fp8_e4m3
    fmha_quant_mode=FMHAQuantMode.bfloat16,
    comm_quant_mode=CommQuantMode.half,
)

# ============================================================================
# SGLang Server 启动命令（用于 nsys_profiler 自动解析为 ServerArgs）
# 直接粘贴你部署 sglang 时用的命令即可
# ============================================================================
SGLANG_LAUNCH_CMD = """
SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0 \
sglang serve \
  --trust-remote-code \
  --model-path /raid/kimi/ds4_new/model_configs/nvidia--GLM-5-NVFP4 \
  --disable-overlap-schedule \
  --attention-backend nsa \
  --chunked-prefill-size 16384 \
  --disable-flashinfer-autotune \
  --enable-cache-report \
  --enable-dp-lm-head \
  --kv-cache-dtype fp8_e4m3 \
  --max-prefill-tokens 16384 \
  --cuda-graph-max-bs 256 \
  --disable-cuda-graph-padding \
  --max-running-requests 256 \
  --mem-fraction-static 0.8 \
  --moe-dense-tp-size 1 \
  --quantization modelopt_fp4 \
  --sampling-backend pytorch \
  --tp-size 8 \
  --watchdog-timeout 1000000
"""

# ============================================================================
# 估算校正系数（来自 refactor_test_aic/stage2_run_aic_estimation.py ）
# ============================================================================
DECODE_CORRECTION_FACTOR = 1.0
PREFILL_CORRECTION_FACTOR = 1.0

# 多项式拟合开关。
# 作用范围：当前只给 GLM5 的 sparse-attention 多项式拟合使用。
# 打开时 refactor 会把当前 batch 的 request_infos 传给 AIC，由 AIC
# 内部决定 GLM5 DSA op 是否使用多项式拟合；其他模型仍走原来的
# AVG 查表/修正路径。
# 设为 0 时关闭多项式拟合，回到原始 AVG (avg_isl, avg_past_kv) 路径。
USE_POLYNOMIAL_FIT = os.environ.get(
    "AIC_USE_POLYNOMIAL_FIT", "1"
).lower() not in {"0", "false", "no", "off"}
POLYNOMIAL_FIT_MODEL_MARKERS = (
    "glm-5",
    "glm5",
)


def model_supports_polynomial_fit(model_name: str, model_path: str = "") -> bool:
    model_id = f"{model_name} {model_path}".lower().replace("_", "-")
    return any(marker in model_id for marker in POLYNOMIAL_FIT_MODEL_MARKERS)

# ============================================================================
# 辅助函数：获取各阶段的绝对输出目录
# ============================================================================


def get_output_dir(data_dir: str, subdir: str) -> str:
    """返回子目录的绝对路径，不存在则创建。"""
    path = os.path.join(data_dir, subdir)
    os.makedirs(path, exist_ok=True)
    return path
