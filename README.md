# ThinkingBox Training

**Executable database-state rewards for multi-turn agent post-training.**

ThinkingBox Training connects stateful [ThinkingBox](https://github.com/microsoft/thinkingbox) tasks to online reinforcement learning. It selects tasks relative to the policy being trained, runs isolated multi-turn MCP episodes with a simulated user, evaluates terminal backend state and side effects, and returns the binary outcome to [Verl](https://github.com/verl-project/verl).

[Paper](https://arxiv.org/abs/2608.19741) · [ThinkingBox](https://github.com/microsoft/thinkingbox) · [ThinkingBox-Bench](https://huggingface.co/datasets/microsoft/ThinkingBox-Bench) · [Benchmark blog](https://huggingface.co/blog/microsoft/thinkingbox) · [OpenEnv docs](https://huggingface.co/docs/openenv/environments/thinkingbox)

## Results at a glance

The reference Qwen3.8-27B full-parameter run used a task pool disjoint from the 507 public benchmark tasks.

| Measure | Base Qwen3.8-27B | RL fine-tuned |
|---|---:|---:|
| Average success (pass@1) | 51.70% | **60.78%** |
| Tasks completed on all 20 attempts | 38 | **107** |
| Tasks completed at least once in 20 | 89.35% | 87.97% |

The pass@1 result is reported in the [paper](https://arxiv.org/abs/2608.19741). The repeated-trial values were aggregated after publication from the same 507 × 20 evaluation campaigns with the public `tb agg` implementation. Training tasks and evaluation tasks are UID-disjoint but use the same sandbox and workflow domains; this is held-out task transfer, not an out-of-distribution claim.

## What this repository provides

- policy-probe aggregation and policy-relative difficulty filtering;
- deterministic benchmark-UID exclusion and grouped train/validation splitting;
- runtime ThinkingBox task hydration without a flattened training snapshot;
- token-exact multi-turn MCP rollouts with a simulated user;
- Qwen reasoning/tool-call parsing aligned with evaluation semantics;
- executable terminal-state rewards with explicit system-error handling;
- synchronous GRPO and generic PPO launch support;
- a validated three-node Qwen3.8-27B reference configuration;
- three pinned Verl v0.9.0 compatibility patches;
- checkpoint merge and ThinkingBox-Bench evaluation procedures;
- 39 public development selectors for exercising the interface.

Provide an authorized ThinkingBox-compatible task source at runtime. Keep training selectors disjoint from the public benchmark used for final evaluation.

## How it works

1. **Select:** probe the base policy and retain tasks whose outcomes can provide useful contrast.
2. **Hydrate:** load the executable task, tools, fixtures, user context, and test code at runtime.
3. **Roll out:** run isolated multi-turn agent/tool/user episodes against fresh backend state.
4. **Check:** execute task-specific checks over terminal state and side effects. Most rewards are deterministic; a minority of narrow rubric requirements can invoke the configured binary judge.
5. **Update:** return the binary task verdict to Verl for the policy update.
6. **Evaluate:** run the resulting checkpoint on the disjoint ThinkingBox-Bench task list.

## Compute paths

| Path | Purpose | Hardware |
|---|---|---|
| Selector hydration | Verify the public development list and loader contract | CPU |
| Configuration dry run | Resolve trainer/Verl overrides without starting Ray | CPU; no training starts |
| Custom training | Train with an authorized task source | Depends on model, context, and rollout concurrency |
| Reference Qwen3.8 full-parameter run | Paper/reference topology | 3 nodes × 8 H100 GPUs; world 24, SP4, effective DP6, rollout TP4 |
| Merged-checkpoint inference | Serve Qwen3.8-27B BF16 | One 80 GB GPU can hold roughly 54 GB of weights; capacity depends on context and concurrency |

## Repository layout

```text
data/
  prepare_data.py                    probe aggregation and task selection CLI
  examples/
    training_tasks.example.yaml      39 public development selectors
    exclusion_uids.example.yaml      benchmark-exclusion interface example
  helpers/utils.py                   posterior, selection, and split helpers
patches/verl/v0.9.0/                 pinned Verl compatibility patches
scripts/
  prepare_verl.py                    clone, verify, patch, and install Verl
  verify_verl_install.py             validate the patched source
  run_train.sh                       fail-closed training launcher
  start_ray.sh                       validated Ray startup and worker environment
  run_eval.sh                        benchmark evaluation launcher
  validate_eval_results.py           evaluation coverage and repair validation
  verify_checkpoint.py               distributed checkpoint completeness gate
trainer/
  train.py                           CLI and Verl/Hydra entry point
  agent_loop.yaml                    MCP and simulated-user configuration
  runtime_compat.py                  Qwen3.8 runtime compatibility checks
  core/
    tb_dataset.py                    ThinkingBox-to-Verl dataset adapter
    tb_roller.py                     token-exact multi-turn rollout loop
    tb_render.py                     verified Qwen delta rendering
    tb_interpret.py                  reasoning/tool-call interpretation
    tb_grader.py                     executable ThinkingBox reward bridge
docs/
  data-preparation.md                task hydration, samples, and selection
  reference-q38-run.md               environment and 24-H100 reference run
  evaluation.md                      checkpoint merge and 507 × 20 evaluation
```

## Start here

The launch release is `v0.1.0`. Keep the public repositories side by side:

```bash
mkdir -p ~/thinkingbox-workspace
cd ~/thinkingbox-workspace

git clone https://github.com/microsoft/thinkingbox.git
git clone https://github.com/microsoft/thinkingbox-data.git
git clone https://github.com/microsoft/thinkingbox-training.git

git -C thinkingbox-training checkout v0.1.0
git -C thinkingbox checkout 892964e7226044e5463ad188da119df838f4bec1
git -C thinkingbox-data checkout thinkingbox-bench-v1.0
```

### 1. Create the validated environment

```bash
cd ~/thinkingbox-workspace/thinkingbox-training

uv venv --python 3.12
source .venv/bin/activate

uv sync --frozen --extra fast --extra tracking
uv pip install -e ../thinkingbox
uv pip install --config-settings editable-mode=compat \
  -e ../thinkingbox-data/servers/thinkingbox_tools
uv pip install --config-settings editable-mode=compat \
  -e ../thinkingbox-data/servers/tb_business_ops_servers_202606

python scripts/prepare_verl.py --dest .deps/verl --install
python scripts/verify_verl_install.py
uv pip check
```

Expected final gates include:

```text
{"status": "verified", ...}
No broken requirements found
```

Do not run an automatic package sync after installing the patched Verl checkout; it can replace the source installation with the vanilla wheel.

### 2. Validate the public development selectors

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
print("hydrated 39 public development tasks")
PY
```

Expected output:

```text
hydrated 39 public development tasks
```

### 3. Render a training configuration without launching a job

Create the runtime-loader adapter, set placeholders, then render the reference topology:

```bash
export TB_WORKSPACE="$HOME/thinkingbox-workspace"
export TB_RUN_ROOT="$HOME/thinkingbox-runs/interface-check"
mkdir -p "$TB_RUN_ROOT"

cat >"$TB_RUN_ROOT/public_dataset_loader.py" <<'PY'
from pathlib import Path
import yaml
from thinkingbox.common.hydrator import iter_cases_by_names


def load_cases(list_file, *, agent, dataset_root):
    names = yaml.safe_load(Path(list_file).read_text(encoding="utf-8"))
    yield from iter_cases_by_names(
        names,
        base_dir=dataset_root,
        agent=agent,
        strict=True,
    )
PY

export PYTHONPATH="$TB_RUN_ROOT:${PYTHONPATH:-}"
export THINKINGBOX_DATASET_LOADER=public_dataset_loader:load_cases
export JUDGE_CONFIG='{
  "type":"aoai",
  "credential":{"type":"api-key","api_key":"placeholder"},
  "endpoint_url":"https://provider.example/v1/chat/completions",
  "deployment":"judge-model"
}'

python -m trainer.train \
  --dry-run \
  --tool-format qwen3_coder \
  --model /path/to/Qwen3.8-27B \
  --train-files data/examples/training_tasks.example.yaml \
  --val-files data/examples/training_tasks.example.yaml \
  --dataset-root "$TB_WORKSPACE/thinkingbox-data/dataset" \
  --dataset-loader "$THINKINGBOX_DATASET_LOADER" \
  --mcp-proxy-url http://127.0.0.1:7111 \
  --judge-config-env JUDGE_CONFIG \
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

Expected output begins with the Verl entry point followed by the resolved overrides:

```text
python -m verl.trainer.main_ppo
  actor_rollout_ref.model.path=...
  data.train_batch_size=18
  actor_rollout_ref.rollout.n=8
  trainer.nnodes=3
  trainer.n_gpus_per_node=8
```

The command must exit zero without starting Ray or allocating GPUs. Review every emitted override before running a job.

## Choose your path

- **Prepare and select tasks:** [docs/data-preparation.md](docs/data-preparation.md)
- **Run the Qwen3.8 reference configuration:** [docs/reference-q38-run.md](docs/reference-q38-run.md)
- **Merge and evaluate a checkpoint:** [docs/evaluation.md](docs/evaluation.md)
- **Review the pinned Verl changes:** [patches/verl/v0.9.0/README.md](patches/verl/v0.9.0/README.md)

## Reward semantics

A judge endpoint is configured because task tests may contain narrow rubric requirements. It is not the primary reward model:

| Path | When it applies | Reward source |
|---|---|---|
| Executable check—the large majority | Required outcome appears in terminal backend state or side effects | Deterministic test returns pass/fail without an LLM judge call |
| Judge-assisted check—a minority | A narrow requirement cannot be determined cleanly from backend state | Configured judge answers a binary rubric question used by the executable test |

The simulated user is a separate environment role and does not grade the policy. Use secret injection appropriate for your environment; never commit credentials or put real keys in shell history.

## Reference Qwen3.8 configuration

| Setting | Value |
|---|---|
| Training paradigm | Full parameter, token-mean GRPO |
| Training pool | 187 tasks |
| Selected checkpoint | Step 50 |
| Scheduled groups × rollouts | 18 × 8 |
| Parallelism | World 24 / SP4 / effective DP6 / rollout TP4 |
| GPUs | 24 H100s, shared by training and synchronous rollouts |
| Learning rate | `1e-6` |
| KL regularization | Disabled |
| Parameter/optimizer offload | Enabled |

The launcher fails closed when the GPU count, topology, batch divisibility, model, task list, loader, agent loop, MCP proxy, judge, shared-checkpoint, or runtime-compatibility configuration is invalid.

## Run record checklist

Record for every run:

- `thinkingbox-training`, `thinkingbox`, and `thinkingbox-data` revisions;
- model and tokenizer identity;
- Verl baseline, patch manifest, and installed package lock;
- task-list and exclusion-list hashes;
- resolved trainer/Verl overrides;
- world/SP/DP/rollout-TP topology;
- sampling, context, timeout, simulator, and judge configuration;
- checkpoint step and merge command;
- evaluation coverage, repair lineage, and aggregate metrics.

## Citation

```bibtex
@article{li2026thinkingbox,
  title   = {One Success Isn't Reliability: Thinkingbox, a Sandbox and Benchmark for Agents in Stateful Business Workflows},
  author  = {Li, Zhuochun and Ko, Youngmin and Keramati, Ali and Kundu, Tuhin and
             Tsai, Liang-Chun and Ferri, Nicola and Milletari, Mirco and Liu, Jiaxiang and
             Lopez Pelaez, Susana Palmaz and Wang, Yuepeng and Smolyakov, Vadim and
             Jiang, Xiang and Olafsson, Kjartan and Guy, Tommy},
  journal = {arXiv preprint arXiv:2608.19741},
  year    = {2026},
  url     = {https://arxiv.org/abs/2608.19741}
}
```

## License

ThinkingBox Training is licensed under the [MIT License](LICENSE.TXT). The
Verl compatibility patches retain upstream file headers and attribution; see
[NOTICE](NOTICE) and
[patches/verl/v0.9.0/README.md](patches/verl/v0.9.0/README.md).

## Contributing

Contributions are subject to Microsoft's Contributor License Agreement. See
[CONTRIBUTING.md](CONTRIBUTING.md) for contribution and validation guidance.

## Code of Conduct

This project has adopted the
[Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).

## Security

Do not report security vulnerabilities through public GitHub issues. Report
them privately to the Microsoft Security Response Center through the
[MSRC reporting portal](https://msrc.microsoft.com/create-report).

## Trademarks

This project may contain trademarks or logos for projects, products, or
services. Authorized use of Microsoft trademarks or logos is subject to and
must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must
not cause confusion or imply Microsoft sponsorship. Any use of third-party
trademarks or logos is subject to those third parties' policies.
