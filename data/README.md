# Dataset setup

Install datasets and task lists from
[`microsoft/thinkingbox-data`](https://github.com/microsoft/thinkingbox-data)
rather than copying them here:

```bash
git clone https://github.com/microsoft/thinkingbox-data.git
git -C thinkingbox-data checkout ds-sandbox-rl-zendesk-2026-03-v1.0
```

For example, the Zendesk evaluation list is:

```text
thinkingbox-data/releases/dataset_2603_sandbox_rl_zendesk/testlist_2603_sandbox_rl_zendesk.yaml
```

Pass `thinkingbox-data/dataset` as `DATASET_DIR` and the required YAML list as
`TRAIN_LIST` or the evaluation command's test-list argument. Keep private task
selections and experiment input bundles outside this repository.
