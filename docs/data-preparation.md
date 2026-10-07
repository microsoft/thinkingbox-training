# Task preparation and runtime hydration

This guide covers the public loader contract, development selectors, benchmark exclusion and policy-probe selection.

## Public dataset-loader adapter

The trainer accepts a runtime `module:function` loader. The public ThinkingBox
framework already exposes strict selector hydration through
`thinkingbox.common.hydrator.iter_cases_by_names`.

Create an adapter in your run workspace:

```bash
export TB_RUN_ROOT="$HOME/thinkingbox-runs/q38"
mkdir -p "$TB_RUN_ROOT"
cat >"$TB_RUN_ROOT/public_dataset_loader.py" <<'PY'
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

export PYTHONPATH="$TB_RUN_ROOT:${PYTHONPATH:-}"
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

Use this development list to validate the public interface before pointing the loader at your own authorized tasks.

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

Expected output:

```text
hydrated 39 public sample tasks
```

Use the sample list with the launcher:

```bash
export TEST_LIST="$PWD/data/examples/training_tasks.example.yaml"
```

`data/examples/exclusion_uids.example.yaml` demonstrates the UID-list shape accepted by `data/prepare_data.py`.

## Prepare a training list

`data/prepare_data.py` can aggregate policy probes, classify
policy-relative difficulty with Jeffreys posteriors, apply an optional stronger
reference-model solvability gate, exclude benchmark UIDs, and produce a selection or train/test split.

All outputs are required to live outside this repository:

```bash
cd ~/thinkingbox-workspace/thinkingbox-training
source .venv/bin/activate

python data/prepare_data.py \
  --policy $TB_RUN_ROOT/probes/policy.jsonl \
  --reference $TB_RUN_ROOT/probes/reference.jsonl \
  --exclude-uids $TB_RUN_ROOT/lists/evaluation_uids.yaml \
  --output-dir $TB_RUN_ROOT/prepared \
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

The CLI records aggregate counts, generated selector lists, hashes, statistics, and a run manifest under the selected output directory. Generated selector files are top-level YAML
lists and can be passed directly to the documented dataset-loader adapter.
