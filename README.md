# Qwen3.8-27B Training with ThinkingBox

This branch contains the full-parameter reinforcement-learning integration used
to train a tool-using Qwen3.8-27B policy with executable ThinkingBox rewards.
It connects ThinkingBox task hydration, multi-turn MCP rollouts, simulated-user
interaction, executable grading, and Verl's synchronous GRPO trainer.

The repository contains training code only. It does **not** contain training
tasks, probe trajectories, generated task selections, model weights,
credentials, endpoint configuration, or evaluation results.

## Scope

The documented reference topology is:

- Qwen3.8-27B full-parameter training;
- KL-free GRPO;
- 3 nodes × 8 GPUs = world size 24;
- Ulysses sequence parallelism 4 and effective data parallelism 6;
- rollout tensor parallelism 4;
- 18 prompts × 8 rollouts per update;
- 50 optimizer updates;
- executable ThinkingBox terminal reward.

The implementation also retains generic single-node, PPO, Hermes tool-format,
and alternative-topology support.

## Repository layout

```text
data/
  prepare_data.py        private-output data-curation CLI
  helpers/utils.py       aggregation, posterior, selection, and split helpers
scripts/
  run_train.sh           validated training launcher
trainer/
  train.py               Verl/Hydra entry point
  agent_loop.yaml        MCP and simulated-user configuration
  core/
    tb_dataset.py        ThinkingBox-to-Verl dataset adapter
    tb_roller.py         token-exact multi-turn rollout loop
    tb_render.py         verified Qwen delta rendering
    tb_interpret.py      Qwen reasoning and tool-call interpretation
    tb_grader.py         executable ThinkingBox reward
```

## Data boundary

Training and evaluation data have different purposes:

- **Training:** supply an authorized task list at runtime. Training selectors
  must be disjoint from the public benchmark used for final evaluation.
- **Evaluation:** ThinkingBox-Bench v1.0 contains 507 public tasks and is
  intended exclusively for evaluation. Do not train on its tasks, expected
  outcomes, golden state, or trajectories.

A task list is a YAML list of selectors:

```yaml
- example_tasks.py:test_first_workflow
- example_tasks.py:test_second_workflow
```

The selectors must resolve in the supplied `thinkingbox-data` checkout or an
authorized dataset with the same public ThinkingBox layout.

## Prerequisites

- Linux or WSL
- Python 3.12
- `uv`
- NVIDIA GPUs with a CUDA/NCCL stack compatible with the pinned packages
- a local Qwen3.8-27B checkpoint
- an OpenAI-compatible simulated-user endpoint
- a ThinkingBox-compatible judge endpoint

Install basic system tools on Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y git curl tar coreutils procps

curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
```

## Clone the public repositories

Keep the repositories side by side:

```bash
mkdir -p ~/q38-workspace
cd ~/q38-workspace

git clone https://github.com/microsoft/thinkingbox.git
git clone https://github.com/microsoft/thinkingbox-data.git
git clone https://github.com/microsoft/thinkingbox-training.git

# Until this work becomes the default branch, select its published branch.
git -C thinkingbox-training checkout '<q38-training-branch-or-release-tag>'
```

For the canonical 507-task evaluation, pin the public benchmark release:

```bash
git -C thinkingbox-data checkout thinkingbox-bench-v1.0
```

For training, use an authorized task source that is disjoint from the
benchmark. If training and evaluation need different data revisions, use
separate `thinkingbox-data` worktrees or clones and record both revisions.

## Create the training environment

Create one environment from `thinkingbox-training` and install the public
framework and MCP server packages into it:

```bash
cd ~/q38-workspace/thinkingbox-training

uv venv --python 3.12
source .venv/bin/activate

uv pip install -e ../thinkingbox
uv pip install --config-settings editable-mode=compat \
  -e ../thinkingbox-data/servers/thinkingbox_tools
uv pip install --config-settings editable-mode=compat \
  -e ../thinkingbox-data/servers/tb_business_ops_servers_202606
uv pip install -e '.[fast,tracking]'

python scripts/prepare_verl.py --dest .deps/verl --install
python scripts/verify_verl_install.py

uv pip check
python -c \
  "import torch, transformers, vllm, verl, ray, thinkingbox, trainer.train"
```

### Required Verl compatibility changes

`pyproject.toml` pins the validated upstream version, `verl==0.9.0`. The
reference 27B runtime also requires three small compatibility changes:

1. move an FSDP-offloaded `lm_head` weight to the hidden-state CUDA device
   before fused linear cross-entropy;
2. synchronize/reduce-scatter every microbatch instead of retaining
   unsharded accumulated FP32 gradients;
3. classify Qwen3.8 as multimodal only when actual multimodal inputs are
   present, so text-only inputs receive Ulysses sequence slicing.

The repository carries these changes as three reviewable git-format patches
under `patches/verl/v0.9.0/`. The preparation script clones exact upstream
commit `483b8a009ba3a97563edee3a19887e4862b8094a`, verifies source and patch
hashes, applies the series, and installs the patched checkout:

```bash
python scripts/prepare_verl.py --dest .deps/verl --install
python scripts/verify_verl_install.py
```

The operation is idempotent and fails on modified, partially patched, or
incompatible source. `run_train.sh` invokes the verifier before launch, so
vanilla or replaced Verl cannot silently start the reference recipe.

Do not run an automatic package sync after installing the patched checkout;
it may replace the source installation with the vanilla wheel. A future
upstream release containing equivalent fixes can replace this patch step after
parity validation.

## Public dataset-loader adapter

The trainer accepts a runtime `module:function` loader. The public ThinkingBox
framework already exposes strict selector hydration through
`thinkingbox.common.hydrator.iter_cases_by_names`.

Create an adapter outside the Git checkout:

```bash
mkdir -p /secure/q38
cat >/secure/q38/public_dataset_loader.py <<'PY'
from pathlib import Path

import yaml
from thinkingbox.common.hydrator import iter_cases_by_names


def load_cases(list_file, *, agent, dataset_root):
    names = yaml.safe_load(Path(list_file).read_text(encoding="utf-8"))
    if not isinstance(names, list) or not all(
        isinstance(name, str) and name for name in names
    ):
        raise ValueError("task list must be a non-empty YAML list of selectors")
    yield from iter_cases_by_names(
        names,
        base_dir=dataset_root,
        agent=agent,
        strict=True,
    )
PY

export PYTHONPATH="/secure/q38:${PYTHONPATH:-}"
export THINKINGBOX_DATASET_LOADER=public_dataset_loader:load_cases
```

Keeping this adapter outside the checkout makes the dataset location and task
selection runtime inputs rather than repository state.

## Public sample task list

`data/examples/training_tasks.example.yaml` contains 39 runnable selectors from
five public development files in `thinkingbox-data`:

- `airline_tau_bench.py`
- `banking.py`
- `banking_email.py`
- `email_system_org.py`
- `mcs_defaults.py`

The list is pinned to public data revision
`49eacaa530b07177d99acd1d0570ee117d43a20b`. None of its selectors occurs in
the canonical ThinkingBox-Bench v1.0 507-task list. One additional function in
`banking.py` is intentionally tagged `skip` upstream and is omitted because
strict task hydration rejects skipped cases.

**This sample list does not represent the actual training set used in our
paper.**

Validate all selectors before using the list:

```bash
python - <<'PY'
from pathlib import Path

import yaml
from thinkingbox.common.hydrator import iter_cases_by_names

selectors = yaml.safe_load(
    Path("data/examples/training_tasks.example.yaml").read_text(encoding="utf-8")
)
cases = list(
    iter_cases_by_names(
        selectors,
        base_dir="../thinkingbox-data/dataset",
        agent="think",
        strict=True,
    )
)
assert len(cases) == len(selectors) == 39
print("hydrated 39 public sample tasks")
PY
```

Use the sample list with the launcher:

```bash
export TEST_LIST="$PWD/data/examples/training_tasks.example.yaml"
```

`data/examples/exclusion_uids.example.yaml` demonstrates the UID-list shape
accepted by `data/prepare_data.py`. It is an interface example only; real
training and exclusion lists remain private runtime inputs outside Git.

## Install and start Typesense and MCP

Install Typesense from the public ThinkingBox repository:

```bash
cd ~/q38-workspace/thinkingbox
source ../thinkingbox-training/.venv/bin/activate

./scripts/install_typesense.sh
typesense-server --version
```

Start Typesense and the MCP Session Proxy:

```bash
export THINKINGBOX_DATA="../thinkingbox-data"
export TB_MCP_START_SERVERS_FILE="../thinkingbox-data/servers/servers.yaml"
./scripts/background_tasks.sh
```

Wait for `All processes are running`. The public benchmark server
configuration uses `TYPESENSE_API_KEY=Fake`. Keep this terminal running.

The public background script starts the MCP proxy on port 7111, while the
training launcher's generic default is 7112. Set the launch URL explicitly:

```bash
export PROXY_URL=http://127.0.0.1:7111
```

## Configure user simulation and judging

The default agent-loop configuration reads the simulated-user endpoint from
environment variables:

```bash
export TBT_USER_ENDPOINT_URL='https://provider.example/v1/chat/completions'
export TBT_USER_DEPLOYMENT='user-model'
export TBT_USER_API_KEY='<runtime-secret>'
```

Supply the judge as a ThinkingBox `LLMSessionConfigT` JSON object:

```bash
export JUDGE_CONFIG='{
  "type": "aoai",
  "credential": {"type": "api-key", "api_key": "<runtime-secret>"},
  "endpoint_url": "https://provider.example/v1/chat/completions",
  "deployment": "judge-model",
  "temperature": 0.0,
  "max_completion_tokens": 128,
  "timeout": 600.0
}'
```

Use secret injection appropriate for your environment. Do not write real
credentials into shell history, configuration committed to Git, or training
artifacts.

## Prepare a private training list

`data/prepare_data.py` can aggregate private policy probes, classify
policy-relative difficulty with Jeffreys posteriors, apply an optional stronger
reference-model solvability gate, exclude benchmark UIDs, and produce a private
selection or train/test split.

All outputs are required to live outside this repository:

```bash
cd ~/q38-workspace/thinkingbox-training
source .venv/bin/activate

python data/prepare_data.py \
  --policy /secure/q38/probes/policy.jsonl \
  --reference /secure/q38/probes/reference.jsonl \
  --exclude-uids /secure/q38/lists/evaluation_uids.yaml \
  --output-dir /secure/q38/prepared \
  --uid-field uid \
  --success-field test_result.result \
  --system-error-field is_system_error \
  --reference-uid-field uid \
  --reference-success-field test_result.result \
  --reference-system-error-field is_system_error \
  --min-clean-runs 4 \
  --retain-band hard \
  --retain-band medium \
  --retain-band easy \
  --reference-min-clean-runs 4 \
  --reference-min-successes 1 \
  --mode split \
  --test-fraction 0.2 \
  --seed 42
```

The CLI prints aggregate counts only. UID lists, statistics, and manifests
remain private runtime artifacts.

## Validate the launch without training

Set runtime paths:

```bash
cd ~/q38-workspace/thinkingbox-training
source .venv/bin/activate

export THINKINGBOX_DATA="$HOME/q38-workspace/thinkingbox-data"
export MODEL=/models/Qwen3.8-27B
export TEST_LIST=/secure/q38/prepared/train.yaml
export VAL_LIST=/secure/q38/prepared/test.yaml
export PROXY_URL=http://127.0.0.1:7111
```

Render the Verl/Hydra configuration without starting Ray or training:

```bash
python -m trainer.train \
  --dry-run \
  --tool-format qwen3_coder \
  --model "$MODEL" \
  --train-files "$TEST_LIST" \
  --val-files "$VAL_LIST" \
  --dataset-root "$THINKINGBOX_DATA/dataset" \
  --dataset-loader "$THINKINGBOX_DATASET_LOADER" \
  --mcp-proxy-url "$PROXY_URL" \
  --judge-config "$JUDGE_CONFIG" \
  --group-size 8 \
  --nodes 3 \
  --gpus-per-node 8 \
  --train-batch-size 18 \
  --max-prompt-length 26624 \
  --max-response-length 106496 \
  -- \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=4 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
  actor_rollout_ref.rollout.max_model_len=133120 \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  trainer.total_training_steps=50 \
  trainer.save_freq=5
```

Review every emitted override before running the job.

## Start a multi-node Q38 training run

The launcher configures training; it does not provision hosts. Prepare an
identical checkout, model, environment, task list, and service configuration on
all nodes. Start one Ray head and join the remaining workers according to your
cluster's networking and scheduler policy.

For a manually managed trusted network, the shape is:

```bash
# Head node
ray start --head \
  --node-ip-address="$HEAD_IP" \
  --port=6379 \
  --num-gpus=8

# Each worker node
ray start \
  --address="$HEAD_IP:6379" \
  --node-ip-address="$WORKER_IP" \
  --num-gpus=8
```

After all 24 GPUs appear in `ray status`, launch once from the head node:

```bash
cd ~/q38-workspace/thinkingbox-training
source .venv/bin/activate

export RAY_ADDRESS=auto
export GPUS=0,1,2,3,4,5,6,7
export NODES=3
export GPUS_PER_NODE=8
export SEQUENCE_PARALLEL_SIZE=4
export ROLLOUT_TP_SIZE=4
export BATCH_SIZE=18
export GROUP_SIZE=8
export MAX_PROMPT=26624
export MAX_RESPONSE=106496
export GPU_MEM=0.70
export OFFLOAD=true

./scripts/run_train.sh \
  actor_rollout_ref.rollout.multi_turn.format=qwen3_coder \
  actor_rollout_ref.rollout.max_model_len=133120 \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  trainer.project_name=thinkingbox-training \
  trainer.experiment_name=q38-full-rlft \
  trainer.total_training_steps=50 \
  trainer.save_freq=5 \
  trainer.default_local_dir=/secure/q38/runs/q38-full-rlft/checkpoints
```

The launcher fails closed when:

- selected GPUs do not match `GPUS_PER_NODE`;
- rollout TP does not divide GPUs per node;
- world size does not divide by sequence parallelism;
- prompts × rollouts does not divide by effective data parallelism;
- the model, task list, dataset, loader, agent loop, MCP proxy, or judge
  configuration is missing.

## Create a merged inference checkpoint

Verl training checkpoints are sharded FSDP state. Merge the actor directory
into a standard Hugging Face model before serving:

```bash
source ~/q38-workspace/thinkingbox-training/.venv/bin/activate

export RUN_ROOT=/secure/q38/runs/q38-full-rlft
export STEP=50
export ACTOR_CHECKPOINT="$RUN_ROOT/checkpoints/global_step_$STEP/actor"
export MERGED_MODEL="$RUN_ROOT/merged/global_step_$STEP"

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
source ~/q38-workspace/thinkingbox-training/.venv/bin/activate

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
cd ~/q38-workspace/thinkingbox
source ../thinkingbox-training/.venv/bin/activate

export EVAL_CONFIG=/secure/q38/eval.yaml

tb infer -c "$EVAL_CONFIG" \
  --dataset ../thinkingbox-data/dataset \
  --agent think \
  --test-list ../thinkingbox-data/releases/thinkingbox_bench_v1/testlist_thinkingbox_bench_v1.yaml \
  --repeat 20 \
  --batch-size 16 \
  --output /secure/q38/evaluation/q38_thinkingbox_bench_v1_20x.jsonl
```

This run expects exactly:

```text
507 tasks × 20 repetitions = 10,140 UID/repetition keys
```

If infrastructure errors occur, rerun only missing or system-error keys:

```bash
tb infer -c "$EVAL_CONFIG" \
  --dataset ../thinkingbox-data/dataset \
  --agent think \
  --test-list ../thinkingbox-data/releases/thinkingbox_bench_v1/testlist_thinkingbox_bench_v1.yaml \
  --repeat 20 \
  --batch-size 16 \
  --previous-results-file /secure/q38/evaluation/q38_thinkingbox_bench_v1_20x.jsonl \
  --output /secure/q38/evaluation/q38_thinkingbox_bench_v1_20x_repaired.jsonl
```

Aggregate the terminal artifact:

```bash
tb agg \
  /secure/q38/evaluation/q38_thinkingbox_bench_v1_20x_repaired.jsonl
```

Do not report a final score until the artifact has 10,140 unique expected
keys, no duplicates, and zero system-error rows.

## Reproducibility checklist

Record for every run:

- `thinkingbox-training`, `thinkingbox`, and `thinkingbox-data` revisions;
- model and tokenizer identity;
- Verl patch identity;
- installed package versions;
- task-list and exclusion-list hashes without publishing their contents;
- resolved Verl/Hydra overrides;
- world/SP/DP/rollout-TP topology;
- sampling, context, timeout, user-model, and judge configuration;
- checkpoint step and merge command;
- exact evaluation coverage and aggregate metrics.

Keep credentials, private task selectors, trajectories, generated selections,
checkpoints, and evaluation JSONL outside Git.
