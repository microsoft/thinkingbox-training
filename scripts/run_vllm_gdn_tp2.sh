#!/usr/bin/env bash
# Rollout policy server with runtime GDN-LoRA hot swapping.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/runtime_contract.sh

require_env MODEL_NAME
require_env VLLM_HOST VLLM_PORT VLLM_TP_SIZE VLLM_DP_SIZE
require_env VLLM_API_SERVER_COUNT VLLM_MAX_MODEL_LEN
require_env VLLM_GPU_MEMORY_UTILIZATION VLLM_MAX_NUM_SEQS
require_env VLLM_CUDA_VISIBLE_DEVICES VLLM_MAX_LORA_RANK VLLM_MAX_LORAS
require_env VLLM_RPC_TIMEOUT
require_directory_var VLLM_SERVE_VENV

export CUDA_VISIBLE_DEVICES="${VLLM_CUDA_VISIBLE_DEVICES}"
export VLLM_SERVER_DEV_MODE=1
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=1
export VLLM_RPC_TIMEOUT

source "${VLLM_SERVE_VENV}/bin/activate"
exec vllm serve "${MODEL_NAME}" \
  --served-model-name "${MODEL_NAME}" \
  --host "${VLLM_HOST}" \
  --port "${VLLM_PORT}" \
  --api-server-count "${VLLM_API_SERVER_COUNT}" \
  --tensor-parallel-size "${VLLM_TP_SIZE}" \
  --data-parallel-size "${VLLM_DP_SIZE}" \
  --max-model-len "${VLLM_MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
  --enable-prefix-caching \
  --enable-lora \
  --max-lora-rank "${VLLM_MAX_LORA_RANK}" \
  --max-loras "${VLLM_MAX_LORAS}" \
  --max-num-seqs "${VLLM_MAX_NUM_SEQS}" \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder
