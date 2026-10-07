# Reference Qwen3.8-27B run

This guide documents the validated 24-H100 full-parameter configuration. The launcher configures training; it does not provision hosts.

### Prerequisites

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

### Clone the public repositories

Keep the repositories side by side:

```bash
mkdir -p ~/thinkingbox-workspace
cd ~/thinkingbox-workspace

git clone https://github.com/microsoft/thinkingbox.git
git clone https://github.com/microsoft/thinkingbox-data.git
git clone https://github.com/microsoft/thinkingbox-training.git

# Use the immutable launch release once it is published.
git -C thinkingbox-training checkout v0.1.0
```

For the canonical 507-task evaluation, pin the public benchmark release:

```bash
git -C thinkingbox-data checkout thinkingbox-bench-v1.0
```

For training, use an authorized task source that is disjoint from the benchmark. If training and evaluation require different data revisions, use separate `thinkingbox-data` worktrees or clones and record both revisions.

### Create the training environment

Create one environment from `thinkingbox-training` and install the public
framework and MCP server packages into it:

```bash
cd ~/thinkingbox-workspace/thinkingbox-training

uv venv --python 3.12
source .venv/bin/activate

uv pip install -e ../thinkingbox
uv pip install --config-settings editable-mode=compat \
  -e ../thinkingbox-data/servers/thinkingbox_tools
uv pip install --config-settings editable-mode=compat \
  -e ../thinkingbox-data/servers/tb_business_ops_servers_202606
uv pip install -e '.[tracking]'

# Compiled extensions are a separate stage. Use compatible prebuilt wheels
# when available; otherwise this host must expose CUDA_HOME and nvcc.
uv pip install packaging ninja
uv pip install --no-build-isolation \
  flash-attn==2.8.3.post1 \
  causal-conv1d==1.6.2.post1

python scripts/prepare_verl.py --dest .deps/verl --install
python scripts/verify_verl_install.py

uv pip check
python -c \
  "import torch, transformers, vllm, verl, ray, thinkingbox, trainer.train, sitecustomize"
```

`prepare_verl.py --install` prefers `uv pip --python <environment-python>` and
falls back to `python -m pip` only when uv is unavailable. The target
environment therefore does not need to contain the pip module.

On restricted hosts, stage exact source archives and compatible wheels from an
internet-connected machine, verify their hashes after transfer, and replace
the editable source paths above with the extracted local paths. Do not replace
the patched Verl installation with the unmodified PyPI wheel.

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

The installed package also provides a small process-local compatibility hook
through `sitecustomize.py`. It preserves Accelerate's Hugging Face parameter
initialization marker and makes text-only Qwen3.8 agent-loop batches use
one-dimensional position IDs. The hook is loaded in the driver and in every
Ray worker; both `start_ray.sh` and `run_train.sh` fail closed when it is not
active.

Do not run an automatic package sync after installing the patched checkout;
it may replace the source installation with the vanilla wheel. A future
upstream release containing equivalent fixes can replace this patch step after
parity validation.

## Install and start Typesense and MCP

Install Typesense from the public ThinkingBox repository:

```bash
cd ~/thinkingbox-workspace/thinkingbox
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
export THINKINGBOX_MCP_PROXY_URL=http://127.0.0.1:7111
```

## Configure user simulation and judging

The default agent-loop configuration reads the simulated-user endpoint from
environment variables:

```bash
export TBT_USER_ENDPOINT_URL='https://provider.example/v1/chat/completions'
export TBT_USER_DEPLOYMENT='user-model'
export TBT_USER_API_KEY='<runtime-secret>'
```

Supply the judge as a ThinkingBox `LLMSessionConfigT` JSON object in an
environment variable:

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
artifacts. The launcher passes only the environment-variable name through
Hydra; the reward worker resolves the value immediately before constructing
the judge session. The key therefore does not appear in process arguments,
dry-run output, resolved Verl configuration logs, or tracking configuration.
Set `JUDGE_CONFIG_ENV` only when using an environment-variable name other than
`JUDGE_CONFIG`.

### Reward semantics

The judge endpoint is configured because task tests may contain narrow rubric requirements. It is not the primary reward model:

| Path | When it applies | Reward source |
|---|---|---|
| Executable check—the large majority | Required outcome appears in terminal backend state or side effects | Deterministic test returns pass/fail without an LLM judge call |
| Judge-assisted check—a minority | A narrow requirement cannot be determined cleanly from backend state | Configured judge answers a binary rubric question used by the executable test |

The simulated user is a separate environment role: it supplies task information during the conversation and does not grade the policy.

## Validate the launch without training

Set runtime paths:

```bash
cd ~/thinkingbox-workspace/thinkingbox-training
source .venv/bin/activate

export THINKINGBOX_DATA="$HOME/thinkingbox-workspace/thinkingbox-data"
export MODEL=/path/to/Qwen3.8-27B
export TEST_LIST=$TB_RUN_ROOT/prepared/train.yaml
export VAL_LIST=$TB_RUN_ROOT/prepared/test.yaml
export THINKINGBOX_MCP_PROXY_URL=http://127.0.0.1:7111
export CHECKPOINT_DIR="$HOME/thinkingbox-runs/q38-full-rlft/checkpoints"
export SHARED_CHECKPOINTS_CONFIRMED=true
```

Render the exact launcher configuration without starting Ray or training:

```bash
DRY_RUN=true \
NODES=3 \
GPUS_PER_NODE=8 \
SEQUENCE_PARALLEL_SIZE=4 \
ROLLOUT_TP_SIZE=4 \
./scripts/run_train.sh \
  trainer.total_training_steps=50 \
  trainer.save_freq=5
```

Expected output begins with the Verl entry point followed by the resolved overrides, for example:

```text
python -m verl.trainer.main_ppo
  actor_rollout_ref.model.path=...
  data.train_batch_size=18
  actor_rollout_ref.rollout.n=8
  trainer.nnodes=3
  trainer.n_gpus_per_node=8
```

The command must exit zero without starting Ray or allocating GPUs. Review every emitted override before running the job.

## Start a multi-node Q38 training run

The launcher configures training; it does not provision hosts. Prepare an
identical checkout, model, environment, task list, and service configuration on
all nodes. Ray must be started after the runtime environment variables are
set: existing daemons do not acquire later `PATH` or `PYTHONPATH` changes.

For a manually managed trusted network, the shape is:

```bash
# Set these on every node without printing secrets.
export VENV="$HOME/thinkingbox-workspace/thinkingbox-training/.venv"
export THINKINGBOX_DATA="$HOME/thinkingbox-workspace/thinkingbox-data"
export THINKINGBOX_DATASET_LOADER=public_dataset_loader:load_cases
export THINKINGBOX_MCP_PROXY_URL=http://mcp-proxy.example:7111
export TBT_USER_ENDPOINT_URL=https://provider.example/v1/chat/completions
export TBT_USER_DEPLOYMENT=user-model
export TBT_USER_API_KEY='<runtime-secret>'
export JUDGE_CONFIG_ENV=JUDGE_CONFIG
# Export JUDGE_CONFIG as shown above on every node.
export PYTHONPATH="$TB_RUN_ROOT:${PYTHONPATH:-}"
export NODES=3
export CHECKPOINT_DIR="$HOME/thinkingbox-runs/q38-full-rlft/checkpoints"
export SHARED_CHECKPOINTS_CONFIRMED=true

# Head node.
ROLE=head NODE_IP="$HEAD_IP" GPUS_PER_NODE=8 \
  ./scripts/start_ray.sh

# Each worker node.
ROLE=worker NODE_IP="$WORKER_IP" \
  HEAD_ADDRESS="$HEAD_IP:6379" GPUS_PER_NODE=8 \
  ./scripts/start_ray.sh
```

`start_ray.sh` verifies the three pinned Verl patches, validates the
dataset-loader import and runtime compatibility hook before starting Ray,
prepends the virtualenv `bin` directory to `PATH` (required for `ninja`), and
propagates the same loader, MCP, user-model, judge, and offline-model
environment to workers.

After all 24 GPUs appear in `ray status`, launch once from the head node:

```bash
cd ~/thinkingbox-workspace/thinkingbox-training
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
export AGENT_LOOP_WORKERS=6

./scripts/run_train.sh \
  trainer.project_name=thinkingbox-training \
  trainer.experiment_name=q38-full-rlft \
  trainer.total_training_steps=50 \
  trainer.save_freq=5
```

The launcher supplies the validated Q38 profile: seed 42, shuffled data,
remove-padding, gradient checkpointing, fused kernels, actor/reference SP4,
TP4 sampling at temperature 1.0/top-k 20/top-p 0.95, learning rate `1e-6`,
CPU parameter/optimizer offload, KL disabled, synchronous failed-group refill,
validation disabled, test frequency disabled, and automatic checkpoint resume.
Explicit trailing Hydra overrides can intentionally replace these defaults.

`GROUP_SIZE` controls rollouts per prompt; `AGENT_LOOP_WORKERS` controls live
MCP sessions. They are not the same setting. Six live workers was validated
with one proxy during the release review. Capacity-test the proxy before
raising concurrency.

For `NODES>1`, `CHECKPOINT_DIR` must be an existing absolute directory on one
POSIX filesystem mounted at the same path on every node. Identical node-local
paths are not sufficient: each rank writes its own model, optimizer, and
extra-state shard. The launcher requires
`SHARED_CHECKPOINTS_CONFIRMED=true`, owns `trainer.default_local_dir`, and
verifies any existing latest checkpoint before automatic resume.

The launcher fails closed when:

- selected GPUs do not match `GPUS_PER_NODE`;
- rollout TP does not divide GPUs per node;
- world size does not divide by sequence parallelism;
- prompts × rollouts does not divide by effective data parallelism;
- the model, task list, dataset, loader, agent loop, MCP proxy, or judge
  configuration is missing;
- the two MCP proxy environment variables disagree;
- multi-node shared-checkpoint storage is not explicitly confirmed;
- an existing resumable checkpoint lacks any expected rank shard or metadata;
- patched Verl or the Qwen3.8 process-local compatibility hook is inactive.
