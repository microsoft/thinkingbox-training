#!/usr/bin/env bash
# Test a single PEFT LoRA adapter against a test list.
#
# Usage:
#   EVAL_BASE_CONFIG=/path/to/rendered.yaml \
#   THINKINGBOX_ROOT=/path/to/framework \
#   DATASET=/path/to/dataset \
#   scripts/eval_one_checkpoint.sh <lora_dir> <test_list> [repeat] [batch_size]
#
# The test list and rendered evaluation config are always explicit.
# Output: output/<runtag>/<runtag>.jsonl
#
# This script:
#   1. Resolves and validates the LoRA directory
#   2. POSTs the LoRA to the running vLLM (idempotent: HTTP 400 with body
#      containing "already" is treated as success)
#   3. Generates a temp eval config pointing at that LoRA name
#   4. Calls tb infer (with the train.patches monkey-patch for system-msg merge)
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/runtime_contract.sh

INPUT="${1:?usage: $0 <lora_dir> <test_list> [repeat] [batch]}"
TEST_LIST="${2:?supply a test-list path}"
REPEAT="${3:-10}"
BATCH="${4:-25}"
VLLM_URL="${VLLM_URL:-http://127.0.0.1:8000}"
require_file_var EVAL_BASE_CONFIG
require_directory_var THINKINGBOX_ROOT
require_directory_var DATASET
reject_placeholder_file "$EVAL_BASE_CONFIG"
[[ -r "$TEST_LIST" ]] || {
  echo "ERROR: test list is not readable: $TEST_LIST" >&2
  exit 2
}
reject_placeholder_file "$TEST_LIST"
TRAINING_ROOT="$PWD"
TEST_LIST_ABS="$(cd "$(dirname "$TEST_LIST")" && pwd)/$(basename "$TEST_LIST")"

# --- Resolve the LoRA directory ----------------------------------------------
[[ -d "$INPUT" ]] || {
  echo "ERROR: LoRA directory not found: $INPUT" >&2
  exit 2
}
LORA_DIR="$(cd "$INPUT" && pwd)"

for required in adapter_config.json adapter_model.safetensors; do
  [[ -f "$LORA_DIR/$required" ]] || {
    echo "ERROR: LoRA directory is missing $required: $LORA_DIR" >&2
    exit 2
  }
done

LORA_NAME="${LORA_NAME_OVERRIDE:-$(basename "$LORA_DIR")}"
RUNTAG="$LORA_NAME"
OUT_DIR="output/${RUNTAG}"
LOAD_RESPONSE="$(mktemp /tmp/lora_load_XXXX.json)"
REGISTRY_RESPONSE="$(mktemp /tmp/lora_registry_XXXX.json)"
EVAL_CFG=""
cleanup() {
  rm -f "$LOAD_RESPONSE" "$REGISTRY_RESPONSE"
  [[ -z "$EVAL_CFG" ]] || rm -f "$EVAL_CFG"
}
trap cleanup EXIT

echo "==============================================================="
echo "  LoRA dir   : $LORA_DIR"
echo "  vLLM name  : $LORA_NAME"
echo "  test-list  : $TEST_LIST"
echo "  repeat     : $REPEAT   batch: $BATCH"
echo "  output     : $OUT_DIR/${RUNTAG}.jsonl"
echo "==============================================================="

# --- 1. Tell vLLM to load this adapter ---------------------------------------
echo "==> loading LoRA into vLLM at $VLLM_URL"
HTTP_CODE=$(
  curl -sS -o "$LOAD_RESPONSE" -w '%{http_code}' \
    -X POST "$VLLM_URL/v1/load_lora_adapter" \
    -H 'Content-Type: application/json' \
    -d "{\"lora_name\":\"$LORA_NAME\",\"lora_path\":\"$LORA_DIR\"}" \
    || echo "000"
)
case "$HTTP_CODE" in
  200) echo "  loaded ok" ;;
  400)
    if grep -q "already" "$LOAD_RESPONSE" 2>/dev/null; then
      echo "  already registered; verifying exact path"
    else
      echo "  load failed (400):"; cat "$LOAD_RESPONSE"; exit 3
    fi
    ;;
  *) echo "  load failed (HTTP $HTTP_CODE):"; cat "$LOAD_RESPONSE" 2>/dev/null; exit 3 ;;
esac

# A duplicate model ID is safe only when it already points to these exact
# adapter bytes on the server's local filesystem.
curl -sS "$VLLM_URL/v1/models" >"$REGISTRY_RESPONSE"
python3 - "$REGISTRY_RESPONSE" "$LORA_NAME" "$LORA_DIR" <<'PY'
import json
import sys
from pathlib import Path

registry, name, expected = sys.argv[1:4]
cards = [
    card
    for card in json.loads(Path(registry).read_text(encoding="utf-8")).get("data", [])
    if card.get("id") == name
]
if len(cards) != 1:
    raise SystemExit(f"vLLM registry has {len(cards)} entries for {name!r}")
served = Path(cards[0].get("root", "")).resolve()
expected_path = Path(expected).resolve()
if served != expected_path:
    raise SystemExit(
        f"vLLM registry path mismatch for {name!r}: "
        f"served={served} expected={expected_path}"
    )
PY

# --- 2. Make a temp eval config ---------------------------------------------
EVAL_CFG="$(mktemp /tmp/eval_${LORA_NAME}_XXXX.yaml)"
source .venv/bin/activate
python - "$EVAL_BASE_CONFIG" "$EVAL_CFG" "$LORA_NAME" \
  "${VLLM_URL%/}/v1/chat/completions" <<'PY'
import sys
from pathlib import Path

import yaml

src, dst, deployment, endpoint = sys.argv[1:5]
config = yaml.safe_load(Path(src).read_text(encoding="utf-8"))
agent_model = config["orchestrator"]["agent_model"]
agent_model["deployment"] = deployment
agent_model["endpoint_url"] = endpoint
Path(dst).write_text(
    yaml.safe_dump(config, sort_keys=False),
    encoding="utf-8",
)
PY

# --- 3. Run tb infer ---------------------------------------------------------
mkdir -p "$OUT_DIR"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="${TRAINING_ROOT}:${PYTHONPATH:-}"
OUT_FILE="${TRAINING_ROOT}/${OUT_DIR}/${RUNTAG}.jsonl"

cd "$THINKINGBOX_ROOT"
python - "$EVAL_CFG" "$DATASET" "$TEST_LIST_ABS" "$REPEAT" "$BATCH" "$OUT_FILE" <<'PY'
import sys
from train.patches import apply
apply()
cfg, dataset, demo, repeat, batch, out = sys.argv[1:7]
sys.argv = [
    "tb", "infer", "-c", cfg,
    "--dataset", dataset, "--agent", "think",
    "--test-list", demo,
    "--repeat", repeat, "--batch-size", batch,
    "--output", out,
]
from thinkingbox.cli.main import main
main()
PY

cd "$TRAINING_ROOT"

# --- 4. Print a per-scenario summary table ----------------------------------
if [[ -s "$OUT_FILE" ]]; then
  python scripts/summarize_eval.py "$OUT_FILE" || true
fi

echo
echo "=== DONE: $LORA_NAME -> $OUT_FILE ==="
