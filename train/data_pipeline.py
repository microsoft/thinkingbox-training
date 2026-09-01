"""Load test-case lists, do seeded train/eval split, hydrate cases."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import yaml

from thinkingbox.common.config_types import HydratedTestCase
from thinkingbox.common.hydrator import iter_cases_by_names

def require_materialized_input(path: str | Path) -> Path:
    """Reject unresolved example identifiers before they reach a runtime."""
    resolved = Path(path).expanduser()
    text = resolved.read_text(encoding="utf-8")
    if re.search(r"<[A-Za-z0-9_]+>", text):
        raise ValueError(f"{resolved} contains an unresolved example identifier")
    return resolved


def load_test_list(path: str | Path) -> list[str]:
    """Load a YAML list of 'filename:testname' strings."""
    path = require_materialized_input(path)
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a YAML list, got {type(data).__name__}")
    return [str(x) for x in data]


def hydrate(
    names: Iterable[str],
    dataset_dir: str | Path,
    agent: str,
    strict: bool = True,
) -> list[HydratedTestCase]:
    """Resolve names → HydratedTestCase objects via the thinkingbox hydrator."""
    return list(
        iter_cases_by_names(
            list(names),
            base_dir=str(Path(dataset_dir).expanduser()),
            agent=agent,
            strict=strict,
        )
    )
