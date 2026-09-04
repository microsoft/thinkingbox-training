#!/usr/bin/env python3
"""Verify the installed behavior-critical training dependency versions."""

from __future__ import annotations

import importlib
import re
import sys
from importlib import metadata
from pathlib import Path
from typing import Callable


ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = ROOT / "requirements.txt"
REQUIRED_IMPORTS = (
    "torch",
    "transformers",
    "peft",
    "verl",
    "vllm",
    "fla",
    "triton",
    "numpy",
    "safetensors",
    "yaml",
    "requests",
    "train.driver",
)


def canonical_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def public_version(version: str) -> str:
    return version.split("+", 1)[0]


def load_expected_versions(path: Path = REQUIREMENTS) -> dict[str, str]:
    expected: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.count("==") != 1:
            raise ValueError(f"{path}:{line_number}: expected one exact == pin")
        name, version = (part.strip() for part in line.split("==", 1))
        if not name or not version:
            raise ValueError(f"{path}:{line_number}: invalid dependency pin")
        key = canonical_name(name)
        if key in expected:
            raise ValueError(f"{path}:{line_number}: duplicate dependency {name}")
        expected[key] = version
    return expected


def version_errors(
    expected: dict[str, str],
    version_getter: Callable[[str], str] = metadata.version,
) -> list[str]:
    errors: list[str] = []
    for name, required in expected.items():
        try:
            installed = version_getter(name)
        except metadata.PackageNotFoundError:
            errors.append(f"{name}: missing; expected {required}")
            continue
        if public_version(installed) != required:
            errors.append(f"{name}: installed {installed}; expected {required}")
    return errors


def import_errors() -> list[str]:
    errors: list[str] = []
    for module_name in REQUIRED_IMPORTS:
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            errors.append(f"{module_name}: import failed: {exc}")
    return errors


def main() -> int:
    errors: list[str] = []
    if sys.version_info[:2] != (3, 12):
        errors.append(
            "python: installed "
            f"{sys.version_info.major}.{sys.version_info.minor}; expected 3.12"
        )
    errors.extend(version_errors(load_expected_versions()))
    errors.extend(import_errors())
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"Training environment matches {REQUIREMENTS.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
