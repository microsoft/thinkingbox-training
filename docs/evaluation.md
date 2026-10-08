# Merge and evaluate a checkpoint

This guide converts a sharded Verl actor checkpoint into a Hugging Face model, serves it with the Qwen parsers used by evaluation, and runs ThinkingBox-Bench 507 × 20.

## Create a merged inference checkpoint

Verl training checkpoints are sharded FSDP state. Merge the actor directory
into a standard Hugging Face model before serving:

```bash
source ~/thinkingbox-workspace/thinkingbox-training/.venv/bin/activate

export RUN_ROOT="$HOME/thinkingbox-runs/q38-full-rlft"
export STEP=50
export ACTOR_CHECKPOINT="$RUN_ROOT/checkpoints/global_step_$STEP/actor"
export MERGED_MODEL="$RUN_ROOT/merged/global_step_$STEP"

python scripts/verify_checkpoint.py \
  --checkpoint-root "$RUN_ROOT/checkpoints" \
  --step "$STEP" \
  --world-size 24

python -m verl.model_merger merge \
  --backend fsdp \
  --local_dir "$ACTOR_CHECKPOINT" \
  --target_dir "$MERGED_MODEL" \
  --use_cpu_initialization
```

Before evaluation, verify that the target contains the model index, every
referenced safetensors shard, tokenizer files, chat template, and model config.
The merged model contains inference weights only; it cannot resume training.

## Serve the merged Q38 checkpoint

On an 8-GPU evaluation host:

```bash
source ~/thinkingbox-workspace/thinkingbox-training/.venv/bin/activate

vllm serve "$MERGED_MODEL" \
  --served-model-name q38-rlft \
  --host 127.0.0.1 \
  --port 8000 \
  --tensor-parallel-size 8 \
  --dtype bfloat16 \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.85 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 8192 \
  --language-model-only \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml
```

Configure a ThinkingBox evaluation YAML whose agent endpoint is
`http://127.0.0.1:8000/v1/chat/completions`, deployment is `q38-rlft`, and
sampling is temperature 1.0, top-k 20, top-p 0.95. Configure the simulated
user and judge through supported public providers as described in the
[ThinkingBox LLM endpoint guide](https://github.com/microsoft/thinkingbox/blob/main/docs/llm_endpoint_config.md).

## Run ThinkingBox-Bench 507 × 20

ThinkingBox-Bench v1.0 is defined by:

```text
thinkingbox-data/releases/thinkingbox_bench_v1/
testlist_thinkingbox_bench_v1.yaml
```

With Typesense, MCP, the Q38 vLLM server, and the configured simulated user and
judge running:

```bash
cd ~/thinkingbox-workspace/thinkingbox-training
source .venv/bin/activate

export THINKINGBOX_ROOT="$HOME/thinkingbox-workspace/thinkingbox"
export THINKINGBOX_DATA="$HOME/thinkingbox-workspace/thinkingbox-data"
export EVAL_CONFIG="$HOME/thinkingbox-config/eval.yaml"
export OUTPUT_DIR="$HOME/thinkingbox-evaluations/q38"

./scripts/run_eval.sh
```

This run expects exactly:

```text
507 tasks × 20 repetitions = 10,140 UID/repetition keys
```

`run_eval.sh` uses a finite queue-inactivity timeout, verifies exact
selector/repetition coverage with no duplicates or system-error rows, and only
then invokes `tb agg`. Outputs and generated run metadata must remain outside
the Git checkout.

If infrastructure errors occur, rerun only missing or system-error keys:

```bash
PREVIOUS_RESULTS_FILE="$HOME/thinkingbox-evaluations/q38/q38_thinkingbox_bench_v1_20x.jsonl" \
RUN_NAME=q38_thinkingbox_bench_v1_20x_repaired \
./scripts/run_eval.sh
```

For a bounded smoke test, provide a five-selector YAML list and override the
expected shape:

```bash
TEST_LIST="$HOME/thinkingbox-evaluations/q38/smoke5.yaml" \
EXPECTED_TASKS=5 \
REPEAT=2 \
BATCH_SIZE=10 \
RUN_NAME=q38_smoke5_2x \
./scripts/run_eval.sh
```

Do not report a final score until the artifact has 10,140 unique expected
keys, no duplicates, and zero system-error rows.
