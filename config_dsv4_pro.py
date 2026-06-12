"""
统一配置模块
============
集中管理路径、模型参数、AIC SDK 配置以及各阶段输出子目录。
修改此文件即可切换到不同模型 / 不同数据目录。
"""

import os
import sys
from pathlib import Path

_AIC_SRC = Path(
    os.environ.get("AIC_REPO_SRC", "/raid/kimi/ds4_new/aiconfigurator/src")
)
if _AIC_SRC.exists():
    sys.path.insert(0, str(_AIC_SRC))

from aiconfigurator.sdk.common import (
    CommQuantMode,
    FMHAQuantMode,
    GEMMQuantMode,
    KVCacheQuantMode,
    MoEQuantMode,
)


def _enum_value(enum_cls, *names):
    for name in names:
        if hasattr(enum_cls, name):
            return getattr(enum_cls, name)
    raise AttributeError(f"{enum_cls.__name__} has none of: {', '.join(names)}")


# ============================================================================
# 数据目录（容器内路径；宿主机映射路径请自行调整）
# ============================================================================

DATA_DIR = os.environ.get(
    "AIC_DATA_DIR",
    "/raid/kimi/ds4_new/b300_deepseekv4_pro_data",
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
MODEL_NAME = "deepseek-ai/DeepSeek-V4-Pro"
MODEL_PATH = MODEL_NAME
SGLANG_MODEL_PATH = os.environ.get(
    "DSV4_PRO_MODEL_PATH",
    "/raid/kimi/huggingface_cache/hub/models--deepseek-ai--DeepSeek-V4-Pro/snapshots/89d501aed998d33fa4f4702102ec1bb2331e10f6/",
)
BACKEND_NAME = "sglang"

# AIC 性能数据库参数
AIC_SYSTEM = "b300_sxm"
AIC_BACKEND = "sglang"
AIC_VERSION = "0.5.12.post2.dev454+ge958f4561"

# ============================================================================
# ModelConfig 参数（对应 aiconfigurator.sdk.config.ModelConfig）
# ============================================================================

MODEL_CONFIG_KWARGS = dict(
    pp_size=1,
    tp_size=4,
    moe_tp_size=4,
    moe_ep_size=1,
    attention_dp_size=1,
    enable_wideep=False,
    workload_distribution="power_law_1.2",
    # workload_distribution="balanced",
    gemm_quant_mode=GEMMQuantMode.fp8_block,
    moe_quant_mode=MoEQuantMode.w4a8_mxfp4_mxfp8_trtllm,
    kvcache_quant_mode=KVCacheQuantMode.fp8,  # fp8 -> fp8_e4m3
    fmha_quant_mode=_enum_value(FMHAQuantMode, "bfloat16", "float16"),
    comm_quant_mode=CommQuantMode.half,
)

# ============================================================================
# SGLang Server 启动命令（用于 nsys_profiler 自动解析为 ServerArgs）
# 直接粘贴你部署 sglang 时用的命令即可
# ============================================================================
SGLANG_LAUNCH_CMD = f"""
SGLANG_JIT_DEEPGEMM_PRECOMPILE=0 \
python3 /raid/kimi/ds4_new/refactor_test_aic/hook_dataset_collector/sglang_launch_server.py \
  --trust-remote-code \
  --model-path {SGLANG_MODEL_PATH} \
  --tp 4 \
  --cuda-graph-max-bs 128 \
  --chunked-prefill-size 8192 \
  --disable-cuda-graph-padding \
  --max-running-requests 256 \
  --mem-fraction-static 0.93 \
  --attention-backend compressed \
  --moe-runner-backend flashinfer_mxfp4 \
  --kv-cache-dtype fp8_e4m3 \
  --tool-call-parser deepseekv4 \
  --reasoning-parser deepseek-v4 \
  --disable-flashinfer-autotune \
  --allow-auto-truncate
"""

# ============================================================================
# 估算校正系数（来自 refactor_test_aic/stage2_run_aic_estimation.py ）
# ============================================================================
DECODE_CORRECTION_FACTOR = 1.0
PREFILL_CORRECTION_FACTOR = 1.0
ATTN_FLOPS_CORRELATION = False

# ============================================================================
# 辅助函数：获取各阶段的绝对输出目录
# ============================================================================


def get_output_dir(data_dir: str, subdir: str) -> str:
    """返回子目录的绝对路径，不存在则创建。"""
    path = os.path.join(data_dir, subdir)
    os.makedirs(path, exist_ok=True)
    return path
