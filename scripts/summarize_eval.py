#!/usr/bin/env python
"""Print a per-scenario pass-rate table for an eval JSONL.

Used by scripts/eval_one_checkpoint.sh to give the user a one-glance result
right after a run completes. Also runnable standalone:

    python scripts/summarize_eval.py output/<runtag>/<runtag>.jsonl
    python scripts/summarize_eval.py output/<runtag>/<runtag>.jsonl --csv
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("jsonl", help="path to inference output jsonl")
    ap.add_argument("--csv", action="store_true", help="emit CSV (no formatting)")
    args = ap.parse_args()

    p = Path(args.jsonl)
    if not p.is_file():
        print(f"ERROR: not a file: {p}", file=sys.stderr)
        return 2

    by_scenario: dict[str, list[bool]] = defaultdict(list)
    n_total = 0
    n_sys_err = 0
    all_passes: list[bool] = []

    with p.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            n_total += 1
            if d.get("is_system_error"):
                n_sys_err += 1
                continue
            sc = (d.get("metadata") or {}).get("scenario") or "(unknown)"
            tr = d.get("test_result") or {}
            ok = bool(tr.get("result", False))
            by_scenario[sc].append(ok)
            all_passes.append(ok)

    if not all_passes:
        print("(no valid records)")
        return 1

    def stats(passes: list[bool]) -> tuple[float, float, float]:
        n = len(passes)
        rate = sum(passes) / n
        # Wilson 95% CI for binomial
        z = 1.96
        denom = 1 + z**2 / n
        center = (rate + z**2 / (2 * n)) / denom
        half = (z * math.sqrt(rate * (1 - rate) / n + z**2 / (4 * n**2))) / denom
        return rate, max(0.0, center - half), min(1.0, center + half)

    micro_rate, micro_lo, micro_hi = stats(all_passes)
    # Macro avg = mean of per-scenario pass rates (equal weight per scenario)
    scenarios_sorted = sorted(by_scenario)
    macro_rate = (
        sum(sum(by_scenario[s]) / len(by_scenario[s]) for s in scenarios_sorted)
        / len(scenarios_sorted)
    ) if scenarios_sorted else 0.0

    if args.csv:
        # CSV output: header + per-scenario rows + summary rows
        print("scenario,n,passed,pass_rate,ci_lo,ci_hi")
        for sc in scenarios_sorted:
            ps = by_scenario[sc]
            rate, lo, hi = stats(ps)
            print(f"{sc},{len(ps)},{sum(ps)},{rate*100:.2f},{lo*100:.2f},{hi*100:.2f}")
        n_valid = len(all_passes)
        n_passed = sum(all_passes)
        print(f"MICRO_AVG,{n_valid},{n_passed},{micro_rate*100:.2f},{micro_lo*100:.2f},{micro_hi*100:.2f}")
        print(f"MACRO_AVG,{n_valid},,{macro_rate*100:.2f},,")
        return 0

    # Pretty table
    file_label = p.parent.name or p.name
    print()
    print("=" * 78)
    print(f"  {file_label}")
    print(f"  {n_total} records  |  {n_sys_err} sys errors  |  {len(all_passes)} valid")
    print("=" * 78)
    print(f"  {'scenario':<35} {'passed':>8}  {'pass%':>7}  {'95% CI':>15}")
    print("  " + "-" * 70)
    for sc in scenarios_sorted:
        ps = by_scenario[sc]
        rate, lo, hi = stats(ps)
        n_pass = sum(ps)
        print(f"  {sc[:35]:<35} {n_pass:>4}/{len(ps):<3}  {rate*100:>6.2f}%  [{lo*100:>4.1f},{hi*100:>5.1f}]")
    print("  " + "-" * 70)
    n_valid = len(all_passes)
    n_passed = sum(all_passes)
    print(f"  {'MICRO AVG (per-case overall)':<35} {n_passed:>4}/{n_valid:<3}  {micro_rate*100:>6.2f}%  [{micro_lo*100:>4.1f},{micro_hi*100:>5.1f}]")
    print(f"  {'MACRO AVG (per-scenario mean)':<35} {'-':>8}  {macro_rate*100:>6.2f}%  {'-':>15}")
    print("=" * 78)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
