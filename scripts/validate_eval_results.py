# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def _selector(row: dict) -> str:
    metadata = row.get("metadata")
    if not isinstance(metadata, dict):
        raise TypeError("result metadata is missing")
    test_file = metadata.get("test_case_file")
    test_name = metadata.get("test_case_name")
    if not isinstance(test_file, str) or not isinstance(test_name, str):
        raise TypeError("result test selector metadata is missing")
    return f"{Path(test_file).name}:{test_name}"


def validate_results(
    results_path: Path,
    test_list_path: Path,
    repetitions: int,
    expected_tasks: int,
) -> dict[str, int]:
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    if expected_tasks < 1:
        raise ValueError("expected task count must be positive")

    test_list = yaml.safe_load(test_list_path.read_text(encoding="utf-8"))
    if not isinstance(test_list, list) or not all(
        isinstance(selector, str) and selector for selector in test_list
    ):
        raise ValueError("test list must be a non-empty YAML list")
    if len(test_list) != expected_tasks or len(set(test_list)) != expected_tasks:
        raise ValueError(
            f"test list must contain {expected_tasks} unique selectors"
        )

    expected = {
        (selector, repetition)
        for selector in test_list
        for repetition in range(repetitions)
    }
    actual: set[tuple[str, int]] = set()
    duplicate_count = 0
    system_error_count = 0

    with results_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSON on result line {line_number}"
                ) from error
            if not isinstance(row, dict):
                raise TypeError(f"result line {line_number} is not an object")

            metadata = row.get("metadata")
            repetition = (
                metadata.get("repetition") if isinstance(metadata, dict) else None
            )
            if not isinstance(repetition, int):
                raise TypeError(
                    f"result repetition is missing on line {line_number}"
                )
            key = (_selector(row), repetition)
            if key in actual:
                duplicate_count += 1
            actual.add(key)

            test_result = row.get("test_result")
            test_system_error = (
                test_result.get("is_system_error")
                if isinstance(test_result, dict)
                else True
            )
            if row.get("is_system_error") is True or test_system_error is not False:
                system_error_count += 1

    missing_count = len(expected - actual)
    unexpected_count = len(actual - expected)
    summary = {
        "duplicates": duplicate_count,
        "expected_rows": len(expected),
        "missing": missing_count,
        "rows": len(actual),
        "system_errors": system_error_count,
        "unexpected": unexpected_count,
    }
    if duplicate_count or missing_count or unexpected_count or system_error_count:
        raise ValueError(json.dumps(summary, sort_keys=True))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify exact task/repetition coverage before tb agg."
    )
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--test-list", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, required=True)
    parser.add_argument("--expected-tasks", type=int, required=True)
    args = parser.parse_args()
    try:
        summary = validate_results(
            args.results.expanduser().resolve(),
            args.test_list.expanduser().resolve(),
            args.repetitions,
            args.expected_tasks,
        )
    except (OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
