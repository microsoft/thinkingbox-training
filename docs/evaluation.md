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
  --port 8001 \
  --tensor-parallel-size 8 \
  --dtype bfloat16 \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.85 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 8192 \
  --override-generation-config '{"temperature":1.0,"top_k":20,"top_p":0.95}' \
  --language-model-only \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml
```

Port 8001 leaves port 8000 available for OpenEnv. The serving override makes
top-k 20 and top-p 0.95 explicit: the pinned ThinkingBox `AOAISessionConfig`
does not expose those YAML fields, so adding them to the agent entry does
not configure sampling.

## Start the Qwen system-message compatibility proxy

ThinkingBox can send consecutive leading system messages for agent and
scenario instructions. Qwen's template accepts only one leading system
message and otherwise raises `System message must be at the beginning.`
The opt-in proxy merges that leading prefix with two newlines between each
instruction, then forwards to vLLM. Other messages, tool calls, and request
parameters are preserved.

When first creating the environment, select the `eval-proxy` extra alongside
the extras you need:

```bash
uv sync --frozen --extra eval-proxy
```

Do this before installing patched VeRL. Do not resync an environment that
already contains the patched source installation.

In a separate terminal, from the ThinkingBox Training root:

```bash
source .venv/bin/activate
BACKEND=http://127.0.0.1:8001 \
PYTHON="$PWD/.venv/bin/python" \
  bash scripts/launch_sysmerge.sh
```

The launcher stays in the foreground. It defaults to loopback port 8014;
`HOST`, `PORT`, `BACKEND`, and `PYTHON` can override the launch settings.
Keep it running while evaluating. Check backend readiness before starting:

```bash
curl --fail http://127.0.0.1:8014/readyz
```

`/healthz` checks only the proxy; `/readyz` checks vLLM's `/v1/models`.
The proxy preserves upstream error responses and does not retry requests.
Connection failures return HTTP 502 and timeouts return HTTP 504.
JSON and streaming chat responses are supported, including cleanup on
completion or disconnect. An upstream failure after streaming begins
terminates the stream rather than fabricating a completion.

System messages after a non-system turn are rejected with HTTP 400 rather
than reordered. System content must be text or text-part arrays; unsupported
content is rejected rather than silently dropped. Leading system messages
with extra message-level fields are also rejected when a merge would discard
those fields. Zero or one leading system message is otherwise unchanged.
The proxy is for a trusted local vLLM endpoint, not a public authenticated
gateway. It does not log message bodies or credentials.

Update only the agent entry in your existing ThinkingBox evaluation YAML:

```yaml
orchestrator:
  agent_model:
    type: aoai
    endpoint_url: http://127.0.0.1:8014/v1/chat/completions
    deployment: q38-rlft
    credential:
      type: api-key
      api_key: Fake
    is_reasoning: true
    temperature: 1.0
    max_completion_tokens: 8192
    timeout: 1800
```

Retain the MCP, simulated-user, and judge entries. Configure the simulated
user and judge through supported public providers as described in the
[ThinkingBox LLM endpoint guide](https://github.com/microsoft/thinkingbox/blob/main/docs/llm_endpoint_config.md).
The proxy is specific to the Qwen endpoint; do not route the other model
roles through it.

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
