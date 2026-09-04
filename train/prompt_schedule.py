"""Validated absolute-step prompt schedules for deterministic RL runs."""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

SCHEDULE_SCHEMA_VERSION = 1


class PromptScheduleError(ValueError):
    """Raised when a prompt schedule is malformed or incompatible with a run."""


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PromptScheduleError(
            f"{field} must be a mapping, got {type(value).__name__}"
        )
    return value


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise PromptScheduleError(f"{field} must be a positive integer")
    return value


def _required_keys(
    mapping: Mapping[str, Any],
    field: str,
    required: set[str],
    optional: set[str] | None = None,
) -> None:
    optional = optional or set()
    missing = sorted(required - set(mapping))
    unknown = sorted(set(mapping) - required - optional)
    if missing:
        raise PromptScheduleError(f"{field} is missing required keys: {missing}")
    if unknown:
        raise PromptScheduleError(f"{field} contains unsupported keys: {unknown}")


@dataclass(frozen=True)
class PromptSchedule:
    """An immutable schedule addressed by the trainer's zero-based step index."""

    path: Path
    sha256: str
    schedule_id: str
    run: Mapping[str, Any]
    coverage: Mapping[str, Any]
    steps: tuple[tuple[str, ...], ...]

    @property
    def max_steps(self) -> int:
        return int(self.run["max_steps"])

    @property
    def prompts_per_step(self) -> int:
        return int(self.run["n_prompts"])

    def prompts_for_absolute_step(self, absolute_step: int) -> tuple[str, ...]:
        """Return the prompts for the exact absolute step, including on resume."""
        if isinstance(absolute_step, bool) or not isinstance(absolute_step, int):
            raise PromptScheduleError("absolute_step must be an integer")
        if absolute_step < 0 or absolute_step >= len(self.steps):
            raise PromptScheduleError(
                f"absolute_step {absolute_step} is outside [0, {len(self.steps)})"
            )
        return self.steps[absolute_step]

    def validate_for_training(
        self,
        train_pool_uids: Iterable[object],
        *,
        max_steps: int,
        n_prompts: int,
        g: int,
        dp_world: int,
        algorithm: str,
        prompt_repeats_per_task: int,
    ) -> None:
        """Fail closed unless this schedule exactly matches the current run."""
        expected = {
            "max_steps": max_steps,
            "n_prompts": n_prompts,
            "g": g,
            "dp_world": dp_world,
            "algorithm": algorithm,
            "prompt_repeats_per_task": prompt_repeats_per_task,
        }
        for key, value in expected.items():
            if self.run[key] != value:
                raise PromptScheduleError(
                    f"schedule run.{key}={self.run[key]!r} does not match "
                    f"the current run value {value!r}"
                )

        train_pool = {str(uid) for uid in train_pool_uids}
        scheduled = {uid for prompts in self.steps for uid in prompts}
        unknown = sorted(scheduled - train_pool)
        if unknown:
            preview = ", ".join(unknown[:3])
            suffix = " ..." if len(unknown) > 3 else ""
            raise PromptScheduleError(
                f"schedule contains {len(unknown)} UID(s) absent from the "
                f"train pool: {preview}{suffix}"
            )

        if bool(self.coverage["exact_train_pool_coverage"]):
            first_cycle_steps = int(self.coverage["first_cycle_steps"])
            first_cycle = [
                uid for prompts in self.steps[:first_cycle_steps] for uid in prompts
            ]
            if len(first_cycle) != len(train_pool):
                raise PromptScheduleError(
                    "exact first-cycle coverage requires "
                    f"{len(first_cycle)} scheduled UIDs to equal the "
                    f"{len(train_pool)}-UID train pool"
                )
            if len(set(first_cycle)) != len(first_cycle):
                raise PromptScheduleError(
                    "first-cycle exact coverage contains repeated prompt UIDs"
                )
            if set(first_cycle) != train_pool:
                missing = sorted(train_pool - set(first_cycle))
                extra = sorted(set(first_cycle) - train_pool)
                raise PromptScheduleError(
                    "first-cycle exact coverage does not equal the train pool "
                    f"(missing={len(missing)}, extra={len(extra)})"
                )


def _load_document(path: Path) -> tuple[Mapping[str, Any], bytes]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise PromptScheduleError(f"could not read schedule {path}: {exc}") from exc
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PromptScheduleError(f"schedule {path} is not UTF-8") from exc

    try:
        if path.suffix.lower() == ".json":
            document = json.loads(text)
        elif path.suffix.lower() in {".yaml", ".yml"}:
            document = yaml.safe_load(text)
        else:
            raise PromptScheduleError(
                f"schedule {path} must use a .yaml, .yml, or .json extension"
            )
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        raise PromptScheduleError(f"could not parse schedule {path}: {exc}") from exc
    return _mapping(document, "schedule"), raw


def load_prompt_schedule(path: str | Path) -> PromptSchedule:
    """Load and structurally validate a YAML or JSON prompt schedule."""
    schedule_path = Path(path).expanduser().resolve()
    document, raw = _load_document(schedule_path)
    _required_keys(
        document,
        "schedule",
        {
            "schema_version",
            "schedule_id",
            "run",
            "coverage",
            "steps",
        },
        {"description", "provenance"},
    )
    if (
        isinstance(document["schema_version"], bool)
        or document["schema_version"] != SCHEDULE_SCHEMA_VERSION
    ):
        raise PromptScheduleError(
            "unsupported schedule schema_version "
            f"{document['schema_version']!r}; expected {SCHEDULE_SCHEMA_VERSION}"
        )
    schedule_id = document["schedule_id"]
    if not isinstance(schedule_id, str) or not schedule_id.strip():
        raise PromptScheduleError("schedule_id must be a non-empty string")

    run = _mapping(document["run"], "run")
    _required_keys(
        run,
        "run",
        {
            "max_steps",
            "n_prompts",
            "g",
            "dp_world",
            "algorithm",
            "prompt_repeats_per_task",
        },
    )
    normalized_run = {
        key: _positive_int(run[key], f"run.{key}")
        for key in (
            "max_steps",
            "n_prompts",
            "g",
            "dp_world",
            "prompt_repeats_per_task",
        )
    }
    if not isinstance(run["algorithm"], str) or not run["algorithm"].strip():
        raise PromptScheduleError("run.algorithm must be a non-empty string")
    normalized_run["algorithm"] = run["algorithm"]

    coverage = _mapping(document["coverage"], "coverage")
    _required_keys(
        coverage,
        "coverage",
        {"first_cycle_steps", "exact_train_pool_coverage"},
        {"second_cycle_prefix_steps"},
    )
    first_cycle_steps = _positive_int(
        coverage["first_cycle_steps"], "coverage.first_cycle_steps"
    )
    if first_cycle_steps > normalized_run["max_steps"]:
        raise PromptScheduleError(
            "coverage.first_cycle_steps cannot exceed run.max_steps"
        )
    if not isinstance(coverage["exact_train_pool_coverage"], bool):
        raise PromptScheduleError(
            "coverage.exact_train_pool_coverage must be a boolean"
        )
    normalized_coverage = {
        "first_cycle_steps": first_cycle_steps,
        "exact_train_pool_coverage": coverage["exact_train_pool_coverage"],
    }
    if "second_cycle_prefix_steps" in coverage:
        second_prefix = _positive_int(
            coverage["second_cycle_prefix_steps"],
            "coverage.second_cycle_prefix_steps",
        )
        if first_cycle_steps + second_prefix > normalized_run["max_steps"]:
            raise PromptScheduleError(
                "first_cycle_steps + second_cycle_prefix_steps exceeds run.max_steps"
            )
        normalized_coverage["second_cycle_prefix_steps"] = second_prefix

    raw_steps = document["steps"]
    if not isinstance(raw_steps, list):
        raise PromptScheduleError("steps must be a list")
    if len(raw_steps) != normalized_run["max_steps"]:
        raise PromptScheduleError(
            f"steps has {len(raw_steps)} entries but run.max_steps is "
            f"{normalized_run['max_steps']}"
        )

    steps: list[tuple[str, ...]] = []
    for expected_step, raw_step in enumerate(raw_steps):
        step = _mapping(raw_step, f"steps[{expected_step}]")
        _required_keys(
            step, f"steps[{expected_step}]", {"absolute_step", "prompt_uids"}
        )
        absolute_step = step["absolute_step"]
        if (
            isinstance(absolute_step, bool)
            or not isinstance(absolute_step, int)
            or absolute_step != expected_step
        ):
            raise PromptScheduleError(
                f"steps[{expected_step}].absolute_step must equal {expected_step}"
            )
        prompts = step["prompt_uids"]
        if not isinstance(prompts, list):
            raise PromptScheduleError(
                f"steps[{expected_step}].prompt_uids must be a list"
            )
        if len(prompts) != normalized_run["n_prompts"]:
            raise PromptScheduleError(
                f"steps[{expected_step}] has {len(prompts)} prompt UIDs; expected "
                f"{normalized_run['n_prompts']}"
            )
        if any(not isinstance(uid, str) or not uid.strip() for uid in prompts):
            raise PromptScheduleError(
                f"steps[{expected_step}].prompt_uids must contain non-empty strings"
            )
        if len(set(prompts)) != len(prompts):
            raise PromptScheduleError(
                f"steps[{expected_step}] repeats a prompt group UID"
            )
        steps.append(tuple(prompts))

    return PromptSchedule(
        path=schedule_path,
        sha256=hashlib.sha256(raw).hexdigest(),
        schedule_id=schedule_id,
        run=normalized_run,
        coverage=normalized_coverage,
        steps=tuple(steps),
    )


def select_global_prompt_uids(
    pool_uids: Iterable[object],
    rng: random.Random,
    n_prompts: int,
    *,
    schedule: PromptSchedule | None = None,
    absolute_step: int | None = None,
) -> list[str]:
    """Choose a global prompt batch, preserving legacy random sampling if omitted."""
    if schedule is not None:
        if absolute_step is None:
            raise PromptScheduleError(
                "absolute_step is required when selecting from a prompt schedule"
            )
        prompts = list(schedule.prompts_for_absolute_step(absolute_step))
        if len(prompts) != n_prompts:
            raise PromptScheduleError(
                f"scheduled step {absolute_step} has {len(prompts)} prompts; "
                f"expected {n_prompts}"
            )
        return prompts
    pool = sorted(str(uid) for uid in pool_uids)
    return rng.sample(pool, min(n_prompts, len(pool)))


def stripe_global_prompt_uids(
    global_uids: Sequence[str],
    *,
    dp_rank: int,
    dp_world: int,
) -> list[str]:
    """Return the stable DP stripe for an already-agreed global prompt batch."""
    if isinstance(dp_world, bool) or not isinstance(dp_world, int) or dp_world < 1:
        raise PromptScheduleError("dp_world must be a positive integer")
    if isinstance(dp_rank, bool) or not isinstance(dp_rank, int):
        raise PromptScheduleError("dp_rank must be an integer")
    if dp_rank < 0 or dp_rank >= dp_world:
        raise PromptScheduleError(f"dp_rank {dp_rank} is outside [0, {dp_world})")
    return list(global_uids[dp_rank::dp_world])
