#!/usr/bin/env bash
# Launch training from explicit model, data, schedule, and runtime inputs.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/runtime_contract.sh

required=(
  MODEL_NAME TRAIN_LIST PROMPT_SCHEDULE TB_CONFIG DATASET_DIR
  VLLM_BASE_URL MCP_URLS NODE_RANK NODE_NPROC NNODES RDZV_ENDPOINT RDZV_ID
  CUDA_VISIBLE_DEVICES RUN_DIR TAG MAX_STEPS START_STEP MAX_AGENT_TURNS
  AGENT N_PROMPTS PROMPT_REPEATS_PER_TASK G CONCURRENCY SP_SIZE MAX_SEQ_LEN
  ACTIVATION_OFFLOAD LOGPROB_CHUNK_SIZE
  LORA_RANK LORA_ALPHA LORA_TARGET_PROFILE LR ALGO LOSS_AGG_MODE
  CLIP_LOW CLIP_HIGH CLIP_C OVERLONG_PENALTY REWARD_FN
  DYNAMIC_SAMPLING_MAX_ROUNDS ROLLOUT_TIMEOUT SEED
  ROLLOUT_STEP_MAX_RETRIES REQUIRE_CLEAN_ROLLOUT_BATCH
  ROLLOUT_ALLOW_TIMEOUT_CENSORING MIN_ROLLOUT_COMPLETE_GROUPS
  MB_EMPTY_MAX_RETRIES MB_CAP MB_TOKEN_BUDGET
  VLLM_LORA_MODULE_PREFIX VLLM_WRITER_NODE VLLM_HEALTH_URLS
  VLLM_HEALTH_MAX_WAIT SP1_MAX_UPDATE_T SP_SYNC_CHECK
  NCCL_PG_TIMEOUT_MIN PYTORCH_CUDA_ALLOC_CONF
  PRODUCER_SOURCE_COMMIT PRODUCER_SOURCE_ARCHIVE_SHA256
  PRODUCER_METRIC_SCHEMA PRODUCER_CHECKPOINT_SCHEMA
)
require_env "${required[@]}"
require_file_var TRAIN_LIST
require_file_var PROMPT_SCHEDULE
require_file_var TB_CONFIG
require_directory_var DATASET_DIR
reject_placeholder_file "$TRAIN_LIST"
reject_placeholder_file "$PROMPT_SCHEDULE"
reject_placeholder_file "$TB_CONFIG"

if [[ -n "${INIT_LORA_DIR:-}" && -n "${RESUME_LORA_DIR:-}" ]]; then
  echo "ERROR: INIT_LORA_DIR and RESUME_LORA_DIR are mutually exclusive" >&2
  exit 2
fi
case "${SP_SYNC_CHECK}" in
  0|1) ;;
  *)
    echo "ERROR: SP_SYNC_CHECK must be 0 or 1" >&2
    exit 2
    ;;
esac
case "${ACTIVATION_OFFLOAD}" in
  0|1) ;;
  *)
    echo "ERROR: ACTIVATION_OFFLOAD must be 0 or 1" >&2
    exit 2
    ;;
esac
if [[ "${START_STEP}" != "0" ]]; then
  require_env RECOVERY_SOURCE_CHECKPOINT_STEP
  require_env RECOVERY_OPTIMIZER_CHECKPOINT_STEP
  if [[ -z "${RESUME_LORA_DIR:-}" ]] && {
    [[ "${RECOVERY_SOURCE_CHECKPOINT_STEP}" != "0" ]] ||
    [[ "${RECOVERY_OPTIMIZER_CHECKPOINT_STEP}" != "0" ]]
  }; then
    echo "ERROR: adapter-free recovery requires source/optimizer steps 0/0" >&2
    exit 2
  fi
fi
if [[ -n "${RESUME_LORA_DIR:-}" ]]; then
  require_directory_var RESUME_LORA_DIR
fi
if [[ -n "${INIT_LORA_DIR:-}" ]]; then
  require_directory_var INIT_LORA_DIR
fi

export CUDA_VISIBLE_DEVICES TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export ROLLOUT_STEP_MAX_RETRIES REQUIRE_CLEAN_ROLLOUT_BATCH
export ROLLOUT_ALLOW_TIMEOUT_CENSORING MIN_ROLLOUT_COMPLETE_GROUPS
export MB_EMPTY_MAX_RETRIES MB_CAP MB_TOKEN_BUDGET
export VLLM_LORA_MODULE_PREFIX VLLM_WRITER_NODE VLLM_HEALTH_URLS
export VLLM_HEALTH_MAX_WAIT SP1_MAX_UPDATE_T SP_SYNC_CHECK
export NCCL_PG_TIMEOUT_MIN PYTORCH_CUDA_ALLOC_CONF

read -r -a mcp_urls <<<"${MCP_URLS}"
args=(
  torchrun
  --nnodes="${NNODES}"
  --nproc-per-node="${NODE_NPROC}"
  --node-rank="${NODE_RANK}"
  --rdzv-backend=c10d
  --rdzv-endpoint="${RDZV_ENDPOINT}"
  --rdzv-id="${RDZV_ID}"
  train/driver.py
  --model-name "${MODEL_NAME}"
  --train-list "${TRAIN_LIST}"
  --prompt-schedule "${PROMPT_SCHEDULE}"
  --tb-config "${TB_CONFIG}"
  --dataset-dir "${DATASET_DIR}"
  --vllm-base-url "${VLLM_BASE_URL}"
  --mcp-urls "${mcp_urls[@]}"
  --tag "${TAG}"
  --agent "${AGENT}"
  --max-steps "${MAX_STEPS}"
  --start-step "${START_STEP}"
  --max-agent-turns "${MAX_AGENT_TURNS}"
  --n-prompts "${N_PROMPTS}"
  --prompt-repeats-per-task "${PROMPT_REPEATS_PER_TASK}"
  --g "${G}"
  --concurrency "${CONCURRENCY}"
  --sp-size "${SP_SIZE}"
  --max-seq-len "${MAX_SEQ_LEN}"
  --logprob-chunk-size "${LOGPROB_CHUNK_SIZE}"
  --lora-rank "${LORA_RANK}"
  --lora-alpha "${LORA_ALPHA}"
  --lora-target-profile "${LORA_TARGET_PROFILE}"
  --lr "${LR}"
  --algo "${ALGO}"
  --loss-agg-mode "${LOSS_AGG_MODE}"
  --clip-low "${CLIP_LOW}"
  --clip-high "${CLIP_HIGH}"
  --clip-c "${CLIP_C}"
  --overlong-penalty "${OVERLONG_PENALTY}"
  --reward-fn "${REWARD_FN}"
  --dynamic-sampling-max-rounds "${DYNAMIC_SAMPLING_MAX_ROUNDS}"
  --rollout-timeout "${ROLLOUT_TIMEOUT}"
  --seed "${SEED}"
  --lora-save-dir "${RUN_DIR}/lora"
  --metrics-out "${RUN_DIR}/metrics.jsonl"
  --save-optimizer-state
  --producer-source-commit "${PRODUCER_SOURCE_COMMIT}"
  --producer-source-archive-sha256 "${PRODUCER_SOURCE_ARCHIVE_SHA256}"
  --producer-metric-schema "${PRODUCER_METRIC_SCHEMA}"
  --producer-checkpoint-schema "${PRODUCER_CHECKPOINT_SCHEMA}"
)

if [[ "${ACTIVATION_OFFLOAD}" == "1" ]]; then
  args+=(--activation-offload)
fi
if [[ -n "${INIT_LORA_DIR:-}" ]]; then
  args+=(--init-lora-dir "${INIT_LORA_DIR}")
fi
if [[ -n "${RESUME_LORA_DIR:-}" ]]; then
  args+=(
    --resume-lora-dir "${RESUME_LORA_DIR}"
    --require-optimizer-resume
  )
fi
if [[ -n "${RECOVERY_SOURCE_CHECKPOINT_STEP:-}" ||
      -n "${RECOVERY_OPTIMIZER_CHECKPOINT_STEP:-}" ]]; then
  require_env RECOVERY_SOURCE_CHECKPOINT_STEP
  require_env RECOVERY_OPTIMIZER_CHECKPOINT_STEP
  args+=(
    --recovery-source-checkpoint-step "${RECOVERY_SOURCE_CHECKPOINT_STEP}"
    --recovery-optimizer-checkpoint-step "${RECOVERY_OPTIMIZER_CHECKPOINT_STEP}"
  )
fi

source .venv/bin/activate
mkdir -p "${RUN_DIR}/lora"
log="${RUN_DIR}/train.node${NODE_RANK}.log"
PYTHONPATH="$PWD:${PYTHONPATH:-}" "${args[@]}" 2>&1 | tee -a "${log}"
exit "${PIPESTATUS[0]}"
