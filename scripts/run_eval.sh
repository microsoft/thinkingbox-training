#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
#
# Run a complete ThinkingBox evaluation and aggregate only exact clean coverage.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VENV=${VENV:-$HERE/.venv}
THINKINGBOX_ROOT=${THINKINGBOX_ROOT:-}
THINKINGBOX_DATA=${THINKINGBOX_DATA:-}
EVAL_CONFIG=${EVAL_CONFIG:-}
OUTPUT_DIR=${OUTPUT_DIR:-}
TEST_LIST=${TEST_LIST:-$THINKINGBOX_DATA/releases/thinkingbox_bench_v1/testlist_thinkingbox_bench_v1.yaml}
AGENT=${AGENT:-think}
REPEAT=${REPEAT:-20}
BATCH_SIZE=${BATCH_SIZE:-16}
TIMEOUT=${TIMEOUT:-1800}
EXPECTED_TASKS=${EXPECTED_TASKS:-507}
RUN_NAME=${RUN_NAME:-q38_thinkingbox_bench_v1_${REPEAT}x}
RESULTS_FILE=${RESULTS_FILE:-}
AGG_OUTPUT=${AGG_OUTPUT:-}
PREVIOUS_RESULTS_FILE=${PREVIOUS_RESULTS_FILE:-}

fail() { echo "error: $*" >&2; exit 1; }
require_value() {
  local name=$1
  [[ -n ${!name:-} ]] || fail "$name is required"
}

require_value THINKINGBOX_ROOT
require_value THINKINGBOX_DATA
require_value EVAL_CONFIG
require_value OUTPUT_DIR
[[ -x $VENV/bin/tb ]] || fail "tb is not installed in $VENV"
[[ -x $VENV/bin/python ]] || fail "Python is not installed in $VENV"
[[ -f $EVAL_CONFIG ]] || fail "evaluation config not found: $EVAL_CONFIG"
[[ -f $TEST_LIST ]] || fail "test list not found: $TEST_LIST"
[[ -d $THINKINGBOX_DATA/dataset ]] || fail "dataset not found: $THINKINGBOX_DATA/dataset"
[[ $REPEAT =~ ^[1-9][0-9]*$ ]] || fail "REPEAT must be a positive integer"
[[ $BATCH_SIZE =~ ^[1-9][0-9]*$ ]] || fail "BATCH_SIZE must be a positive integer"
[[ $EXPECTED_TASKS =~ ^[1-9][0-9]*$ ]] || \
  fail "EXPECTED_TASKS must be a positive integer"
[[ $TIMEOUT =~ ^[1-9][0-9]*([.][0-9]+)?$ ]] || \
  fail "TIMEOUT must be a positive finite number"

OUTPUT_DIR=$("$VENV/bin/python" -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "$OUTPUT_DIR")
case "$OUTPUT_DIR/" in
  "$HERE/"*) fail "evaluation output must be outside the Git checkout" ;;
esac
mkdir -p "$OUTPUT_DIR"
RESULTS_FILE=${RESULTS_FILE:-$OUTPUT_DIR/${RUN_NAME}.jsonl}
AGG_OUTPUT=${AGG_OUTPUT:-$OUTPUT_DIR/${RUN_NAME}_agg.json}
RESULTS_FILE=$("$VENV/bin/python" -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "$RESULTS_FILE")
AGG_OUTPUT=$("$VENV/bin/python" -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "$AGG_OUTPUT")
case "$RESULTS_FILE" in
  "$OUTPUT_DIR"/*) ;;
  *) fail "RESULTS_FILE must be inside OUTPUT_DIR" ;;
esac
case "$AGG_OUTPUT" in
  "$OUTPUT_DIR"/*) ;;
  *) fail "AGG_OUTPUT must be inside OUTPUT_DIR" ;;
esac

"$VENV/bin/python" - "$TEST_LIST" "$EXPECTED_TASKS" <<'PY'
import sys
from pathlib import Path

import yaml

selectors = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = int(sys.argv[2])
if not isinstance(selectors, list) or not all(
    isinstance(selector, str) and selector for selector in selectors
):
    raise SystemExit("test list must be a non-empty YAML list")
if len(selectors) != expected or len(set(selectors)) != expected:
    raise SystemExit(f"test list must contain {expected} unique selectors")
PY

infer_args=(
  -c "$EVAL_CONFIG"
  --dataset "$THINKINGBOX_DATA/dataset"
  --agent "$AGENT"
  --test-list "$TEST_LIST"
  --repeat "$REPEAT"
  --batch-size "$BATCH_SIZE"
  --timeout "$TIMEOUT"
  --output "$RESULTS_FILE"
)
if [[ -n $PREVIOUS_RESULTS_FILE ]]; then
  [[ -f $PREVIOUS_RESULTS_FILE ]] || \
    fail "previous results not found: $PREVIOUS_RESULTS_FILE"
  infer_args+=(--previous-results-file "$PREVIOUS_RESULTS_FILE")
fi

cd "$THINKINGBOX_ROOT"
"$VENV/bin/tb" infer "${infer_args[@]}"

"$VENV/bin/python" "$HERE/scripts/validate_eval_results.py" \
  --results "$RESULTS_FILE" \
  --test-list "$TEST_LIST" \
  --repetitions "$REPEAT" \
  --expected-tasks "$EXPECTED_TASKS"

"$VENV/bin/tb" agg --output-format json "$RESULTS_FILE" | tee "$AGG_OUTPUT"
