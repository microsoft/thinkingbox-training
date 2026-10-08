# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""verl dataset backed by a ThinkingBox dataset checkout.

Point ``data.train_files`` at a test list (the same ``file.py:test_name`` YAML
``tb infer --test-list`` takes) and the cases are hydrated at startup. There is
no intermediate parquet, so the dataset checkout stays the single source of
truth: edit a rubric or a world state and the next run picks it up, rather than
silently training against a stale snapshot.

Reading and hydrating cases belongs to the selected dataset integration. Set
``data.thinkingbox.loader`` or ``THINKINGBOX_DATASET_LOADER`` to a public
``module:function`` loader. What remains here is the verl adapter: the row
schema verl expects, and the ``RLHFDataset`` subclass that supplies it.

Wire it up with::

    data.custom_cls.path=<repo>/trainer/core/tb_dataset.py
    data.custom_cls.name=ThinkingBoxDataset
    data.train_files=/path/to/train.yaml
    +data.thinkingbox.agent=think
    +data.thinkingbox.loader=package.module:load_cases

``trainer/train.py`` sets the dataset adapter for you.
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Callable, Iterable
from typing import Any

import datasets
import numpy as np
from thinkingbox.common.config_types import HydratedTestCase
from verl.utils.dataset.rl_dataset import RLHFDataset

DATA_SOURCE = "thinkingbox"
DEFAULT_AGENT = "think"


def resolve_case_loader(
    specification: str | None,
) -> Callable[..., Iterable[HydratedTestCase]]:
    """Resolve a public ``module:function`` dataset loader."""
    if not specification:
        raise ValueError(
            "Set data.thinkingbox.loader or THINKINGBOX_DATASET_LOADER "
            "to a public module:function case loader"
        )
    module_name, separator, function_name = specification.partition(":")
    if not separator or not module_name or not function_name:
        raise ValueError(
            "ThinkingBox dataset loader must use module:function syntax"
        )
    module = importlib.import_module(module_name)
    loader = getattr(module, function_name, None)
    if not callable(loader):
        raise TypeError(f"ThinkingBox dataset loader is not callable: {specification}")
    return loader


def build_row(case: HydratedTestCase, index: int) -> dict[str, Any]:
    """Turn a hydrated case into the row schema verl reads.

    ``data_source`` and ``reward_model.ground_truth`` are required: verl's
    reward manager indexes both unconditionally and supplies no default, so a
    row missing them raises KeyError at the first reward computation. The
    grader ignores their values.
    """
    return {
        "data_source": DATA_SOURCE,
        # ThinkingBox builds the real prompt from the case; verl only needs a
        # well-formed chat prompt present on the row.
        "prompt": [{"role": "user", "content": case.query}],
        "ability": "agent",
        "reward_model": {"style": "rule", "ground_truth": case.uid},
        "extra_info": {
            "index": index,
            "uid": case.uid,
            # A JSON *string*, not a nested dict: scenarios have differently
            # shaped world_states with no common arrow schema. tb_roller
            # accepts either form.
            "thinkingbox_case": case.model_dump_json(),
        },
    }


class ThinkingBoxDataset(RLHFDataset):
    """Hydrates ThinkingBox test lists instead of reading parquet.

    Only the loading step differs from ``RLHFDataset``; row processing, prompt
    rendering, length filtering and checkpoint resume are inherited.
    """

    def _download(self, use_origin_parquet: bool = False) -> None:
        # data_files are local test lists; there is nothing to fetch or cache.
        return

    def _read_files_and_tokenize(self) -> None:
        # ThinkingBox cases are text-only, but AutoProcessor resolves a
        # multimodal processor for some text checkpoints (Qwen3.8 ->
        # Qwen3VLProcessor). RLHFDataset then routes prompt-length filtering
        # and __getitem__ through the multimodal path, which cannot survive
        # datasets' worker serialization of a custom_cls subclass. Force the
        # tokenizer path; tb_roller does its own encoding at rollout time.
        self.processor = None

        tb_config = self.config.get("thinkingbox", {}) or {}
        agent = tb_config.get("agent") or DEFAULT_AGENT
        dataset_root = tb_config.get("dataset_root")
        loader = resolve_case_loader(
            tb_config.get("loader") or os.getenv("THINKINGBOX_DATASET_LOADER")
        )

        rows: list[dict[str, Any]] = []
        for list_file in self.data_files:
            for case in loader(
                list_file, agent=agent, dataset_root=dataset_root
            ):
                rows.append(build_row(case, len(rows)))

        if not rows:
            raise ValueError(f"No usable ThinkingBox cases in {list(self.data_files)}")

        self.dataframe = datasets.Dataset.from_list(rows)
        total = len(self.dataframe)
        print(f"dataset len: {total} (hydrated from {list(self.data_files)})")

        # Same max_samples semantics as the parent.
        if self.max_samples > 0 and self.max_samples < total:
            if self.shuffle:
                rng_args = (self.seed,) if self.seed is not None else ()
                indices = np.random.default_rng(*rng_args).choice(
                    total, size=self.max_samples, replace=False
                )
            else:
                indices = np.arange(self.max_samples)
            self.dataframe = self.dataframe.select(indices.tolist())
            print(f"selected {self.max_samples} samples out of {total}")

        self.dataframe = self.maybe_filter_out_long_prompts(self.dataframe)
