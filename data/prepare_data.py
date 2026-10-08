# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import argparse
import json
from pathlib import Path

from helpers.utils import BANDS, run_preparation


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare private training selectors from probe JSONL."
    )
    parser.add_argument("--policy", action="append", required=True)
    parser.add_argument("--reference", action="append", default=[])
    parser.add_argument("--exclude-uids")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--uid-field", required=True)
    parser.add_argument("--success-field", required=True)
    parser.add_argument("--system-error-field")
    parser.add_argument("--reference-uid-field")
    parser.add_argument("--reference-success-field")
    parser.add_argument("--reference-system-error-field")
    parser.add_argument("--min-clean-runs", type=int, default=3)
    parser.add_argument(
        "--retain-band",
        action="append",
        choices=[label for label, _, _ in BANDS],
        dest="retained_bands",
    )
    parser.add_argument("--reference-min-clean-runs", type=int, default=1)
    parser.add_argument("--reference-min-successes", type=int, default=1)
    parser.add_argument("--reference-solvability-threshold", type=float)
    parser.add_argument("--reference-min-probability-above", type=float)
    parser.add_argument("--mode", choices=("selection", "split"), default="selection")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.reference and not (
        args.reference_uid_field and args.reference_success_field
    ):
        raise SystemExit(
            "reference inputs require explicit reference UID and success fields"
        )
    try:
        result = run_preparation(
            policy_paths=args.policy,
            reference_paths=args.reference,
            exclusion_path=args.exclude_uids,
            output_directory=args.output_dir,
            repository_root=Path(__file__).resolve().parents[1],
            uid_field=args.uid_field,
            success_field=args.success_field,
            system_error_field=args.system_error_field,
            reference_uid_field=args.reference_uid_field or args.uid_field,
            reference_success_field=(
                args.reference_success_field or args.success_field
            ),
            reference_system_error_field=(
                args.reference_system_error_field or args.system_error_field
            ),
            min_clean_runs=args.min_clean_runs,
            retained_bands=set(args.retained_bands or ("hard", "medium", "easy")),
            reference_min_clean_runs=args.reference_min_clean_runs,
            reference_min_successes=args.reference_min_successes,
            reference_solvability_threshold=args.reference_solvability_threshold,
            reference_min_probability_above=args.reference_min_probability_above,
            mode=args.mode,
            test_fraction=args.test_fraction,
            seed=args.seed,
        )
    except (OSError, TypeError, ValueError, RuntimeError):
        print('{"failed":1}')
        return 2
    print(json.dumps(result["summary"], sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
