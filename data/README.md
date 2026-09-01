# Dataset setup

Install datasets and task lists from
[`microsoft/thinkingbox-data`](https://github.com/microsoft/thinkingbox-data)
rather than copying them here:

```bash
git clone https://github.com/microsoft/thinkingbox-data.git
```

For example, the canonical public ThinkingBox-Bench evaluation list is:

```text
thinkingbox-data/releases/thinkingbox_bench_v1/testlist_thinkingbox_bench_v1.yaml
```

Pass `thinkingbox-data/dataset` as `DATASET_DIR` and the desired public YAML
list as the evaluation command's test-list argument.
