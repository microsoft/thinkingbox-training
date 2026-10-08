# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import argparse
from pathlib import Path


def _ranks(actor_directory: Path, prefix: str, world_size: int) -> set[int]:
    ranks: set[int] = set()
    pattern = f"{prefix}_world_size_{world_size}_rank_*.pt"
    for path in actor_directory.glob(pattern):
        if path.stat().st_size <= 0:
            raise ValueError(f"empty checkpoint shard: {path.name}")
        try:
            ranks.add(int(path.stem.rsplit("_rank_", 1)[1]))
        except (IndexError, ValueError) as error:
            raise ValueError(f"invalid checkpoint shard name: {path.name}") from error
    return ranks


def verify_checkpoint(root: Path, step: int, world_size: int) -> None:
    if step < 0:
        raise ValueError("step must be non-negative")
    if world_size < 1:
        raise ValueError("world size must be positive")

    step_directory = root / f"global_step_{step}"
    actor_directory = step_directory / "actor"
    expected_ranks = set(range(world_size))

    for prefix in ("model", "optim", "extra_state"):
        actual_ranks = _ranks(actor_directory, prefix, world_size)
        if actual_ranks != expected_ranks:
            missing = sorted(expected_ranks - actual_ranks)
            unexpected = sorted(actual_ranks - expected_ranks)
            raise ValueError(
                f"{prefix} shard coverage mismatch; "
                f"missing={missing}, unexpected={unexpected}"
            )

    for path in (actor_directory / "fsdp_config.json", step_directory / "data.pt"):
        if not path.is_file() or path.stat().st_size <= 0:
            raise ValueError(f"missing or empty checkpoint artifact: {path.name}")

    pointer = root / "latest_checkpointed_iteration.txt"
    if not pointer.is_file():
        raise ValueError("latest checkpoint pointer is missing")
    try:
        pointed_step = int(pointer.read_text(encoding="utf-8").strip())
    except ValueError as error:
        raise ValueError("latest checkpoint pointer is not an integer") from error
    if pointed_step != step:
        raise ValueError(
            f"latest checkpoint pointer is {pointed_step}, expected {step}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify a complete resumable distributed Verl checkpoint."
    )
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    args = parser.parse_args()
    try:
        verify_checkpoint(
            args.checkpoint_root.expanduser().resolve(),
            args.step,
            args.world_size,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(
        f"verified global_step_{args.step}: "
        f"model/optim/extra_state ranks 0-{args.world_size - 1}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
