# ThinkingBox Training

**Executable database-state rewards for multi-turn agent post-training.**

ThinkingBox Training trains tool-using agents with rewards from executable task checks. Each rollout gives the agent a fresh [ThinkingBox](https://github.com/microsoft/thinkingbox) backend, MCP tools, and a simulated user. When the episode ends, the checks inspect the resulting database state and side effects, then return a binary reward to [VeRL](https://github.com/verl-project/verl).

Before training, we probe the base policy and select a fixed pool of tasks whose outcomes can provide useful contrast. The pool remains unchanged throughout the run.

Most rewards come from deterministic checks of terminal backend state and side effects. Some tasks also use a narrow binary rubric answered by the configured judge. The simulated user is a separate environment role and does not grade the agent.

[Paper](https://arxiv.org/abs/2608.19741) · [ThinkingBox](https://github.com/microsoft/thinkingbox) · [ThinkingBox-Bench](https://huggingface.co/datasets/microsoft/ThinkingBox-Bench) · [Benchmark blog](https://huggingface.co/blog/microsoft/thinkingbox) · [OpenEnv docs](https://huggingface.co/docs/openenv/environments/thinkingbox)

## Results at a glance

| Measure | Base Qwen3.8-27B | RL fine-tuned |
|---|---:|---:|
| Average success (pass@1) | 51.70% | **60.78%** |
| Tasks completed on all 20 attempts | 38 | **107** |
| Tasks completed at least once in 20 | 89.35% | 87.97% |

The pass@1 result is reported in the [paper](https://arxiv.org/abs/2608.19741). The repeated-trial values were aggregated after publication from the same 507 × 20 evaluation campaigns with the public `tb agg` implementation. No ThinkingBox-Bench task was included in the training pool.

## What you can do

Prepare a training pool from policy probes, train agents on executable ThinkingBox tasks, and evaluate the resulting checkpoints. The repository includes a documented Qwen3.8-27B reference run and sample tasks for checking the integration.

Provide an authorized ThinkingBox-compatible task source at runtime. Keep training selectors disjoint from the public benchmark used for final evaluation.

### Training loop

Probe and select tasks → run isolated agent episodes → check terminal state → update the policy with VeRL.

Evaluate the resulting checkpoint separately on ThinkingBox-Bench.

## Getting started

Install the stack and check the sample tasks below. Then follow the guides to [prepare a training pool](docs/data-preparation.md), [run the Qwen3.8 reference configuration](docs/reference-q38-run.md), or [merge and evaluate a checkpoint](docs/evaluation.md).

### Requirements

The stack targets Linux and Python 3.12. Training requires NVIDIA GPUs, a local model checkpoint, ThinkingBox services, and endpoints for the simulated user and judge-assisted checks.

The documented Qwen3.8 full-parameter run uses three nodes with eight H100 GPUs each. Other configurations depend on model size, episode length, and rollout concurrency.

Sample hydration and configuration checks do not launch GPU training. Installing the full training stack still requires the compatible CUDA/NCCL environment described in the [reference-run guide](docs/reference-q38-run.md).

### Install

Clone the latest `main` revision of ThinkingBox Training and keep the public repositories side by side:

```bash
mkdir -p ~/thinkingbox-workspace
cd ~/thinkingbox-workspace

git clone https://github.com/microsoft/thinkingbox.git
git clone https://github.com/microsoft/thinkingbox-data.git
git clone --branch main https://github.com/microsoft/thinkingbox-training.git

git -C thinkingbox checkout 892964e7226044e5463ad188da119df838f4bec1
git -C thinkingbox-data checkout thinkingbox-bench-v1.0
```

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

The VeRL verification should report `"status": "verified"`, and `uv pip check` should report no broken requirements.

Do not run an automatic package sync after installing the patched VeRL checkout; it can replace the source installation with the vanilla wheel.

### Check the sample tasks

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

### Check the training configuration

The following command prints the resolved VeRL launch command. It does not start Ray or allocate GPUs. Placeholder model and endpoint values are for this configuration check only.

Use the [example dataset loader](data/examples/public_dataset_loader.py) and set placeholders for the model and judge:

```bash
export TB_WORKSPACE="$HOME/thinkingbox-workspace"
export PYTHONPATH="$PWD/data/examples:${PYTHONPATH:-}"
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

Expected output begins with the VeRL entry point followed by the resolved overrides:

```text
python -m verl.trainer.main_ppo
  actor_rollout_ref.model.path=...
  data.train_batch_size=18
  actor_rollout_ref.rollout.n=8
  trainer.nnodes=3
  trainer.n_gpus_per_node=8
```

The command must exit zero without starting Ray or allocating GPUs. Review every emitted override before running a job.

## Train and evaluate

- **Prepare and select tasks:** [docs/data-preparation.md](docs/data-preparation.md)
- **Run the Qwen3.8 reference configuration:** [docs/reference-q38-run.md](docs/reference-q38-run.md)
- **Merge and evaluate a checkpoint:** [docs/evaluation.md](docs/evaluation.md)

The full model, optimizer, parallelism, and launch settings are documented in the [Qwen3.8 reference-run guide](docs/reference-q38-run.md).

The launcher checks GPU allocation, batch divisibility, required services, checkpoint storage, and the patched VeRL installation before starting. See the reference-run guide for [troubleshooting](docs/reference-q38-run.md#launcher-checks) and the [information to record for each run](docs/reference-q38-run.md#run-record-checklist).

Use secret injection appropriate for your environment; never commit credentials or put real keys in shell history.

## Required VeRL patches

The reference configuration uses VeRL v0.9.0 with three pinned compatibility patches. The installation commands above apply and verify them. See the [patch documentation](patches/verl/v0.9.0/README.md) for the changes and their upstream revision.

## Repository layout

```text
data/       Task probing, selection, and splitting
trainer/    ThinkingBox rollouts and VeRL integration
scripts/    Training, evaluation, and installation helpers
patches/    Required changes to the pinned VeRL version
docs/       Preparation, training, and evaluation guides
```

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
VeRL compatibility patches retain upstream file headers and attribution; see
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
