# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Load a YAML selector list using ThinkingBox's strict task hydration."""

from collections.abc import Iterator
from pathlib import Path

import yaml
from thinkingbox.common.config_types import HydratedTestCase
from thinkingbox.common.hydrator import iter_cases_by_names


def load_cases(
    list_file: str | Path, *, agent: str, dataset_root: str | Path
) -> Iterator[HydratedTestCase]:
    names = yaml.safe_load(Path(list_file).read_text(encoding="utf-8"))
    if (
        not isinstance(names, list)
        or not names
        or not all(isinstance(name, str) and name.strip() for name in names)
    ):
        raise ValueError("task list must be a non-empty YAML list of selectors")
    yield from iter_cases_by_names(
        names,
        base_dir=dataset_root,
        agent=agent,
        strict=True,
    )
