# thinkingbox-training

RL post-training for tool-using Qwen policies on top of the
[thinkingbox](https://github.com/microsoft/thinkingbox) framework.

The project supports group-relative policy optimization, LoRA adapters,
multi-turn MCP rollouts, FSDP2, sequence parallelism, checkpoint recovery,
and runtime adapter reload into an OpenAI-compatible vLLM server.

## Release-data boundary

This source tree does not contain:

- private training or evaluation task selections;
- absolute-step schedules from historical runs;
- internal service, account, storage, cluster, or filesystem coordinates;
- raw training or evaluation trajectories;
- process, pod, checkpoint-transaction, or incident records; or
- runtime credentials.

Checked-in configuration and input files are examples. Adapt all paths,
endpoints, identities, and credentials before launch.

See:

- [`data/README.md`](data/README.md) for task-list handling;
- [`config/README.md`](config/README.md) for runtime configuration;
- [`examples/training_inputs/README.md`](examples/training_inputs/README.md)
  for the files required to configure a training job.

## Layout

```text
config/        runtime configuration examples
data/          public upstream data plus shape-only list examples
examples/      public-safe training input examples
scripts/       launch, serving, evaluation, and validation commands
train/         FSDP2 driver plus rollout, trace, reward, and optimizer code
checkpoints/   local run outputs; ignored except for .gitkeep
output/        local evaluation outputs; ignored except for .gitkeep
```

## Environment

Requirements:

- Linux with supported NVIDIA GPUs and drivers;
- Python 3.12;
- a compatible CUDA/PyTorch/vLLM stack;
- a compatible thinkingbox framework checkout; and
- an authorized dataset checkout.

`requirements.txt` pins the direct runtime versions used by the Qwen3.6 qval
smoke. Configure the platform-specific Torch, vLLM, and CUDA wheel source
externally; this repository does not hardcode a package index.

```bash
uv venv --python 3.12
source .venv/bin/activate

uv pip install -e /path/to/thinkingbox
uv pip install -r requirements.txt
uv pip install -e .

uv pip check
python scripts/verify_training_environment.py
python -c "import torch, transformers, peft, verl, fla; import train.driver"
```

Dependency versions, model/tokenizer hashes, CUDA, driver, NCCL, and framework
revisions are part of a strict experiment identity. Do not silently upgrade
them when matching an existing measurement configuration.

## Configure and launch training

Materialize the templates outside the worktree:

```bash
mkdir -p /secure/run-inputs
cp config/config_training_verl.yaml /secure/run-inputs/training.yaml
cp config/eval_base.yaml /secure/run-inputs/eval.yaml
cp examples/training_inputs/launch.env.example /secure/run-inputs/launch.env
cp examples/training_inputs/prompt_schedule.example.yaml /secure/run-inputs/prompt-schedule.yaml
```

Replace every angle-bracket identifier and point `launch.env` at
the rendered configuration, a task list from `thinkingbox-data`, and immutable
run inputs. The trainer validates the task list, schedule, and strict hydration
before the first rollout.

Set `ACTIVATION_OFFLOAD=1` for 128K/SP2 updates; use `0` at shorter context.
After the MCP and policy services below are healthy, launch once on each
training node. Node copies share `RDZV_ENDPOINT`/`RDZV_ID` and set their own
`NODE_RANK` and `CUDA_VISIBLE_DEVICES`.

```bash
set -a
source /secure/run-inputs/launch.env
set +a
scripts/launch_training.sh
```

The exact algorithm, schedule, topology, context, sampling, timeout,
user-simulator, judge, reward, and initialization values must come from the
experiment configuration being matched. Substituting any of them creates a
different experiment.

### Manual checkpoint resume

Automatic multi-node recovery is not included. To resume manually, update
`launch.env` with:

```bash
START_STEP=<next-absolute-schedule-step>
RESUME_LORA_DIR=/path/to/checkpoints/<run>/lora/<adapter>
RECOVERY_SOURCE_CHECKPOINT_STEP=<adapter-checkpoint-step>
RECOVERY_OPTIMIZER_CHECKPOINT_STEP=<optimizer-checkpoint-step>
```

The source and optimizer checkpoint steps must match. After a no-update metric,
`START_STEP` may be greater than the checkpoint step because the prompt schedule
advanced without creating a new policy checkpoint. Launch each node with the
same command shown above.

## Supporting services

Start the framework MCP and search services with explicit paths:

```bash
THINKINGBOX_ROOT=/path/to/thinkingbox \
THINKINGBOX_DATA=/path/to/data \
TYPESENSE_API_KEY=<runtime-value> \
scripts/start_servers.sh
```

Start the rollout policy with an explicit model and serving environment:

```bash
MODEL_NAME=/path/to/model \
VLLM_SERVE_VENV="$PWD/.venv" \
VLLM_HOST=<bind-host> \
VLLM_PORT=<port> \
VLLM_TP_SIZE=<tensor-parallel-size> \
VLLM_DP_SIZE=<data-parallel-size> \
VLLM_API_SERVER_COUNT=<api-server-count> \
VLLM_MAX_MODEL_LEN=<context-limit> \
VLLM_GPU_MEMORY_UTILIZATION=<memory-fraction> \
VLLM_MAX_NUM_SEQS=<maximum-sequences> \
VLLM_CUDA_VISIBLE_DEVICES=<serving-gpus> \
VLLM_MAX_LORA_RANK=<maximum-lora-rank> \
VLLM_MAX_LORAS=<maximum-loaded-adapters> \
VLLM_RPC_TIMEOUT=<milliseconds> \
scripts/run_vllm_gdn_tp2.sh
```

## Evaluation

Evaluation requires an explicit test list and rendered runtime configuration:

```bash
EVAL_BASE_CONFIG=/secure/run-inputs/eval.yaml \
THINKINGBOX_ROOT=/path/to/thinkingbox \
DATASET=/path/to/evaluation-dataset \
scripts/eval_one_checkpoint.sh \
  checkpoints/<run>/lora/<adapter> \
  /path/to/eval-list.yaml \
  20 32
```

Evaluation JSONL files are local restricted artifacts and are ignored by Git.
Only privacy- and dataset-owner-approved aggregate metrics may be included in
a release.

## Runtime checks

Validate the installed environment before launching:

```bash
uv pip check
python scripts/verify_training_environment.py
```

## Training implementation

`train/driver.py` is the only training entry point. Its supporting modules are:

```text
train/algorithm.py
train/checkpoint_consistency.py
train/data_pipeline.py
train/eval_loop.py
train/token_replay.py
train/fsdp_lora_sync.py
train/lora_sync.py
train/patches.py
train/prompt_schedule.py
train/rewards.py
train/rollout.py
train/sp_ulysses.py
train/tokenize_chat.py
train/wandb_logger.py
```

The trainer uses PyTorch FSDP2 for sharding and Ulysses sequence parallelism
for long sequences. Deprecated alternative training backends are not included.
