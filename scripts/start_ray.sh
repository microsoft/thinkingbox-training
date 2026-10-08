#!/usr/bin/env bash
#
# Start a Ray head or worker with the complete Qwen3.8 training environment.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VENV=${VENV:-$HERE/.venv}
ROLE=${ROLE:-}
NODE_IP=${NODE_IP:-}
HEAD_ADDRESS=${HEAD_ADDRESS:-}
HEAD_PORT=${HEAD_PORT:-6379}
NUM_CPUS=${NUM_CPUS:-$(nproc)}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NODES=${NODES:-1}
DATASET_LOADER=${THINKINGBOX_DATASET_LOADER:-}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-$HERE/checkpoints}
SHARED_CHECKPOINTS_CONFIRMED=${SHARED_CHECKPOINTS_CONFIRMED:-false}

fail() { echo "error: $*" >&2; exit 1; }
require_value() {
  local name=$1
  [[ -n ${!name:-} ]] || fail "$name is required"
}

[[ $ROLE == head || $ROLE == worker ]] || fail "ROLE must be head or worker"
require_value NODE_IP
require_value THINKINGBOX_DATA
require_value THINKINGBOX_DATASET_LOADER
require_value THINKINGBOX_MCP_PROXY_URL
require_value TBT_USER_API_KEY
require_value TBT_USER_ENDPOINT_URL
require_value TBT_USER_DEPLOYMENT
JUDGE_CONFIG_ENV=${JUDGE_CONFIG_ENV:-JUDGE_CONFIG}
[[ $JUDGE_CONFIG_ENV =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || \
  fail "JUDGE_CONFIG_ENV must name an environment variable"
require_value "$JUDGE_CONFIG_ENV"
[[ $ROLE == head ]] || require_value HEAD_ADDRESS
[[ -x $VENV/bin/python ]] || fail "target environment is missing: $VENV"
[[ -x $VENV/bin/ray ]] || fail "Ray is not installed in $VENV"
"$VENV/bin/python" "$HERE/scripts/verify_verl_install.py" >/dev/null || \
  fail "Verl installation does not match the required patched v0.9.0 source"
if ((NODES > 1)); then
  [[ $CHECKPOINT_DIR == /* && -d $CHECKPOINT_DIR ]] || \
    fail "multi-node CHECKPOINT_DIR must be an existing absolute shared path"
  [[ $SHARED_CHECKPOINTS_CONFIRMED == true ]] || \
    fail "set SHARED_CHECKPOINTS_CONFIRMED=true after mounting CHECKPOINT_DIR"
fi

export PATH="$VENV/bin:$PATH"
export PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}

"$VENV/bin/python" - "$DATASET_LOADER" <<'PY'
import importlib
import sys

from accelerate import big_modeling

module_name, separator, function_name = sys.argv[1].partition(":")
if not separator or not module_name or not function_name:
    raise SystemExit("THINKINGBOX_DATASET_LOADER must be module:function")
module = importlib.import_module(module_name)
loader = getattr(module, function_name)
if not callable(loader):
    raise SystemExit("configured dataset loader is not callable")
if not getattr(big_modeling.init_on_device, "_thinkingbox_patched", False):
    raise SystemExit("Qwen3.8 runtime compatibility was not activated")
PY

common=(
  --node-ip-address="$NODE_IP"
  --num-cpus="$NUM_CPUS"
  --num-gpus="$GPUS_PER_NODE"
  --disable-usage-stats
)

if [[ $ROLE == head ]]; then
  exec "$VENV/bin/ray" start \
    --head \
    --port="$HEAD_PORT" \
    --dashboard-host=0.0.0.0 \
    "${common[@]}" \
    "$@"
fi

exec "$VENV/bin/ray" start \
  --address="$HEAD_ADDRESS" \
  "${common[@]}" \
  "$@"
