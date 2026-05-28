#!/usr/bin/env bash
set -euo pipefail

AIC=${AIC:-/workspace/cache/aic_from_dot8_20260525_090903/aiconfigurator}
SGL=${SGL:-/sgl-workspace/sglang}
MODEL_PATH=${MODEL_PATH:-nvidia/GLM-5-NVFP4}
GPU_NAME=${GPU_NAME:-b200_sxm}
SGLANG_COMMIT=${SGLANG_COMMIT:-bc8d64bf36c687580ea9d4dc17fed8bcd8e62395}
RUN_DIR=${RUN_DIR:-/workspace/cache/results/dsa_b200_dot8_sglang_default_pcg_$(date +%Y%m%d_%H%M%S)}

mkdir -p "$RUN_DIR"
echo "$RUN_DIR"

cd "$SGL"
git checkout "$SGLANG_COMMIT"

sha256sum "$AIC/collector/sglang/collect_mla_module.py" > "$RUN_DIR/aic_hash.txt"
git rev-parse HEAD > "$RUN_DIR/sglang_git.txt"
git branch --show-current >> "$RUN_DIR/sglang_git.txt" || true
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits > "$RUN_DIR/gpu_before.txt"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export PYTHONPATH=$AIC/src:$AIC:$SGL/python:${PYTHONPATH:-}
export SGLANG_LOAD_FORMAT=dummy
export SGLANG_TEST_NUM_LAYERS=2
export AIC_ENABLE_PIECEWISE_CUDA_GRAPH=1
export AIC_ENABLE_MODULE_PIECEWISE_REPLAY=1
export AIC_PIECEWISE_CUDA_GRAPH_TOKENS=${AIC_PIECEWISE_CUDA_GRAPH_TOKENS:-4,8,12,16,20,24,28,32,48,64,80,96,112,128,144,160,176,192,208,224,240,256,288,320,352,384,416,448,480,512,576,640,704,768,832,896,960,1024,1280,1536,1792,2048}
export AIC_PIECEWISE_CUDA_GRAPH_MAX_TOKENS=${AIC_PIECEWISE_CUDA_GRAPH_MAX_TOKENS:-2048}
export AIC_MLA_MODULE_SUBPROCESS_TIMEOUT_SEC=${AIC_MLA_MODULE_SUBPROCESS_TIMEOUT_SEC:-900}
export SGLANG_DSV4_FP4_EXPERTS=0
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=1

env | grep -E 'AIC_|SGLANG_|CUDA_VISIBLE_DEVICES|PYTHONPATH' | sort > "$RUN_DIR/collect_env.txt"

cd "$RUN_DIR"

python3 "$AIC/collector/collect.py" \
  --backend sglang \
  --model-path "$MODEL_PATH" \
  --gpu "$GPU_NAME" \
  --ops dsa_context_module dsa_generation_module \
  --checkpoint-dir "$RUN_DIR/.collector_checkpoint"
