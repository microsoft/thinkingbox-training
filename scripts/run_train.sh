#!/usr/bin/env bash
#
# Launch GRPO or PPO training on a ThinkingBox task list.
#
#   ./scripts/run_train.sh                                  # defaults below
#   GPUS=1,2,3,4 GROUP_SIZE=4 ./scripts/run_train.sh
#   ./scripts/run_train.sh trainer.total_training_steps=2   # extra verl overrides
#
# Anything after the script name is forwarded to verl as a Hydra override.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$HERE"

# ---------------------------------------------------------------- configuration
VENV=${VENV:-$HERE/.venv}
DATA=${THINKINGBOX_DATA:-}
MODEL=${MODEL:-}
TEST_LIST=${TEST_LIST:-}
VAL_LIST=${VAL_LIST:-}                 # defaults to TEST_LIST
AGENT=${AGENT:-think}
PROXY_URL=${PROXY_URL:-http://127.0.0.1:7112}
AGENT_LOOP_CONFIG=${AGENT_LOOP_CONFIG:-$HERE/trainer/agent_loop.yaml}
DATASET_LOADER=${THINKINGBOX_DATASET_LOADER:-}

# Assumes the node is free. Drop GPU 0 (GPUS=1,2,...) if the eval vLLM server
# is running, and lower GROUP_SIZE/BATCH_SIZE to keep the product divisible by
# the GPU count.
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
NODES=${NODES:-1}
SEQUENCE_PARALLEL_SIZE=${SEQUENCE_PARALLEL_SIZE:-1}
ROLLOUT_TP_SIZE=${ROLLOUT_TP_SIZE:-${TP:-2}}
# GRPO rollouts per prompt. Advantage is reward minus the group mean, so a group
# whose rollouts all score the same yields zero gradient -- 8 gives a reasonable
# spread. BATCH_SIZE*GROUP_SIZE must be divisible by effective data parallelism.
GROUP_SIZE=${GROUP_SIZE:-8}
BATCH_SIZE=${BATCH_SIZE:-8}            # distinct prompts per step (8x8 = 64 trajectories)
# Set generously: verl LEFT-truncates anything longer, silently dropping the
# system prompt and tool schemas off the front while still grading against the
# full task. It only warns, from inside a Ray worker, so it is easy to miss.
# This is a ceiling, not an allocation -- main_ppo_sync builds variable-length
# nested tensors rather than padding to it, and vLLM's max_model_len is a
# separate knob, so headroom is free. Tool-heavy tasks can render substantially
# longer prompts than their user query alone suggests.
MAX_PROMPT=${MAX_PROMPT:-32768}
MAX_RESPONSE=${MAX_RESPONSE:-8192}     # whole episode: generations + observations + user turns
GPU_MEM=${GPU_MEM:-0.4}                # vLLM share. Below ~0.25 a 14B model at TP=2
                                       # cannot allocate any KV cache and refuses to start.
# Offload actor params/optimizer to CPU between phases. On by default because
# verl colocates: the actor's weights, grads and Adam state share each GPU with
# vLLM's own model copy and KV cache. Leaving them resident OOMs vLLM during
# engine init ("CUDA Error: out of memory at cumem_allocator.cpp") even with the
# optimizer sharded across 6-8 GPUs. Set OFFLOAD=false only if you have verified
# both fit -- it is faster when it does.
OFFLOAD=${OFFLOAD:-true}
JUDGE_CONFIG=${JUDGE_CONFIG:-}

N_GPUS=$(awk -F, '{print NF}' <<<"$GPUS")
GPUS_PER_NODE=${GPUS_PER_NODE:-$N_GPUS}

# ------------------------------------------------------------------- preflight
fail() { echo "error: $*" >&2; exit 1; }

require_value() {
  local name=$1
  local value=${!name:-}
  [[ -n $value ]] || fail "$name is required"
}

require_value THINKINGBOX_DATA
require_value THINKINGBOX_DATASET_LOADER
require_value MODEL
require_value TEST_LIST
require_value JUDGE_CONFIG

[[ $NODES =~ ^[1-9][0-9]*$ ]] || fail "NODES must be a positive integer"
[[ $GPUS_PER_NODE =~ ^[1-9][0-9]*$ ]] || \
  fail "GPUS_PER_NODE must be a positive integer"
[[ $SEQUENCE_PARALLEL_SIZE =~ ^[1-9][0-9]*$ ]] || \
  fail "SEQUENCE_PARALLEL_SIZE must be a positive integer"
[[ $ROLLOUT_TP_SIZE =~ ^[1-9][0-9]*$ ]] || \
  fail "ROLLOUT_TP_SIZE must be a positive integer"

TOTAL_GPUS=$((NODES * GPUS_PER_NODE))

if [[ $AGENT_LOOP_CONFIG == "$HERE/trainer/agent_loop.yaml" ]]; then
  require_value TBT_USER_API_KEY
  require_value TBT_USER_ENDPOINT_URL
  require_value TBT_USER_DEPLOYMENT
fi

[[ -x $VENV/bin/python ]] || fail "no venv at $VENV -- run: uv venv -p 3.12 .venv && uv pip install -e '.[dev]'"
"$VENV/bin/python" "$HERE/scripts/verify_verl_install.py" >/dev/null || \
  fail "Verl installation does not match the required patched v0.9.0 source"
[[ -d $MODEL ]]           || fail "model not found: $MODEL"
[[ -f $TEST_LIST ]]       || fail "test list not found: $TEST_LIST"
[[ -d $DATA/dataset ]]    || fail "dataset checkout not found: $DATA/dataset"
[[ -f $AGENT_LOOP_CONFIG ]] || fail "agent-loop config not found: $AGENT_LOOP_CONFIG"

if [[ -n $VAL_LIST && ! -f $VAL_LIST ]]; then
  fail "validation test list not found: $VAL_LIST"
fi

((N_GPUS == GPUS_PER_NODE)) || \
  fail "GPUS selects $N_GPUS devices, but GPUS_PER_NODE is $GPUS_PER_NODE"
((GPUS_PER_NODE % ROLLOUT_TP_SIZE == 0)) || \
  fail "GPUS_PER_NODE ($GPUS_PER_NODE) must be divisible by ROLLOUT_TP_SIZE ($ROLLOUT_TP_SIZE)"
((TOTAL_GPUS % SEQUENCE_PARALLEL_SIZE == 0)) || \
  fail "total GPU count ($TOTAL_GPUS) must be divisible by SEQUENCE_PARALLEL_SIZE ($SEQUENCE_PARALLEL_SIZE)"

DP_SIZE=$((TOTAL_GPUS / SEQUENCE_PARALLEL_SIZE))

# verl asserts this and the message is opaque, so check it here instead.
(( (BATCH_SIZE * GROUP_SIZE) % DP_SIZE == 0 )) || \
  fail "BATCH_SIZE*GROUP_SIZE ($((BATCH_SIZE*GROUP_SIZE))) must be divisible by effective DP size ($DP_SIZE)"

# The proxy must serve the selected dataset's tool servers.
if ! curl -sf -m 5 "$PROXY_URL/health" >/dev/null; then
  cat >&2 <<EOF
MCP proxy not reachable at $PROXY_URL. Start it with:

  THINKINGBOX_DATA=/path/to/dataset \\
  tb mcp-start --servers /path/to/servers.yaml --port 7112 --host 127.0.0.1
EOF
  exit 1
fi

# Validate Azure CLI only when the selected judge configuration requests it.
if [[ $JUDGE_CONFIG == *'"type":"az-cli"'* ]]; then
  command -v az >/dev/null 2>&1 || \
    fail "judge configuration requires Azure CLI, but az is unavailable"
  az account get-access-token \
    --resource https://cognitiveservices.azure.com/ >/dev/null 2>&1 || \
    fail "judge configuration requires a valid Azure CLI session"
fi

# flash-attn is required: the sdpa fallback needs O(n^2) attention memory,
# which 32K+ contexts cannot afford on GPUs already shared with vLLM.
"$VENV/bin/python" -c "import flash_attn" >/dev/null 2>&1 || \
  fail "flash-attn not installed -- run: uv pip install -e '.[fast]'"

cat <<EOF
launching training
  model      $MODEL
  test list  $TEST_LIST  (agent=$AGENT)
  gpus       $GPUS
  topology   nodes=$NODES, gpus/node=$GPUS_PER_NODE, world=$TOTAL_GPUS, sp=$SEQUENCE_PARALLEL_SIZE, dp=$DP_SIZE
  rollout    tp=$ROLLOUT_TP_SIZE, replicas/node=$((GPUS_PER_NODE/ROLLOUT_TP_SIZE))
  batch      ${BATCH_SIZE} prompts x ${GROUP_SIZE} rollouts = $((BATCH_SIZE*GROUP_SIZE)) trajectories/step
  offload    $OFFLOAD
  proxy      $PROXY_URL
  agent loop $AGENT_LOOP_CONFIG
EOF

# ----------------------------------------------------------------------- launch
export CUDA_VISIBLE_DEVICES=$GPUS
export THINKINGBOX_DATA=$DATA
export THINKINGBOX_DATASET_LOADER=$DATASET_LOADER
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}

exec "$VENV/bin/python" -m trainer.train \
  --model "$MODEL" \
  --train-files "$TEST_LIST" \
  ${VAL_LIST:+--val-files "$VAL_LIST"} \
  --dataset-root "$DATA/dataset" \
  --dataset-loader "$DATASET_LOADER" \
  --agent "$AGENT" \
  --mcp-proxy-url "$PROXY_URL" \
  --judge-config "$JUDGE_CONFIG" \
  --agent-loop-config "$AGENT_LOOP_CONFIG" \
  --group-size "$GROUP_SIZE" \
  --train-batch-size "$BATCH_SIZE" \
  --max-prompt-length "$MAX_PROMPT" \
  --max-response-length "$MAX_RESPONSE" \
  --nodes "$NODES" \
  --gpus-per-node "$GPUS_PER_NODE" \
  -- \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size="$SEQUENCE_PARALLEL_SIZE" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="$ROLLOUT_TP_SIZE" \
  actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_MEM" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.fsdp_config.param_offload="$OFFLOAD" \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload="$OFFLOAD" \
  actor_rollout_ref.actor.ppo_mini_batch_size="$BATCH_SIZE" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  trainer.val_before_train=false \
  'trainer.logger=[console]' \
  "$@"
