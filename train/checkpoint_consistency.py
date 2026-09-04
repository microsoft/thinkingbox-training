"""Pure checkpoint consistency helpers used before distributed training resumes."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Sequence

ADAPTER_PAYLOAD_FILES = (
    "adapter_config.json",
    "adapter_model.safetensors",
)
_RESTORED_RE = re.compile(r"restored\(step=(-?\d+)\)")


def compute_adapter_payload_digest(adapter_dir: str | Path) -> str:
    """Hash semantically canonical adapter config plus exact tensor payload."""
    root = Path(adapter_dir)
    digest = hashlib.sha256()
    for name in ADAPTER_PAYLOAD_FILES:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(
                f"required adapter payload missing: {path}")
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        if name == "adapter_config.json":
            config = json.loads(path.read_text(encoding="utf-8"))
            # PEFT stores target_modules as a set and may serialize it in a
            # different order on each node even when the adapter is identical.
            targets = config.get("target_modules")
            if isinstance(targets, list):
                config["target_modules"] = sorted(targets)
            payload = json.dumps(
                config, sort_keys=True, separators=(",", ":"),
            ).encode("utf-8")
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)
        else:
            digest.update(path.stat().st_size.to_bytes(8, "big"))
            with path.open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    digest.update(chunk)
    return digest.hexdigest()


def adapter_digests_match(digests: Sequence[str]) -> bool:
    """Return whether non-empty SHA-256 hex digests are valid and identical."""
    if not digests:
        raise ValueError("at least one adapter digest is required")
    normalized = []
    for digest in digests:
        try:
            raw = bytes.fromhex(digest)
        except ValueError as exc:
            raise ValueError(f"invalid adapter SHA-256 digest: {digest!r}") from exc
        if len(raw) != hashlib.sha256().digest_size:
            raise ValueError(f"invalid adapter SHA-256 digest: {digest!r}")
        normalized.append(raw)
    return len(set(normalized)) == 1


def optimizer_status_code(status: str) -> tuple[int, int]:
    """Normalize a local optimizer status to (kind, step) integer fields."""
    if status == "missing(fresh AdamW)":
        return 0, -1
    match = _RESTORED_RE.fullmatch(status)
    if match is not None:
        return 1, int(match.group(1))
    raise ValueError(f"unrecognized optimizer resume status: {status!r}")


def validate_optimizer_resume_consensus(
    kind_min: int,
    kind_max: int,
    step_min: int,
    step_max: int,
    *,
    require: bool,
    expected_step: int,
) -> str:
    """Validate collective MIN/MAX status fields and return one status."""
    if kind_min != kind_max:
        raise RuntimeError(
            "optimizer resume status mismatch across ranks: "
            "mixed fresh and restored state")
    if kind_min == 0:
        if require:
            raise RuntimeError(
                "optimizer resume required but all ranks reported fresh state")
        return "missing(fresh AdamW)"
    if kind_min != 1:
        raise RuntimeError(
            f"invalid optimizer resume status kind: {kind_min}")
    if step_min != step_max:
        raise RuntimeError(
            "optimizer restored-step mismatch across ranks: "
            f"min={step_min} max={step_max}")
    if step_min != expected_step:
        raise RuntimeError(
            "optimizer restored step does not match requested start step: "
            f"restored={step_min} expected={expected_step}")
    return f"restored(step={step_min})"


def validate_optimizer_resume_statuses(
    statuses: Sequence[str],
    *,
    require: bool,
    expected_step: int,
) -> str:
    """Pure list-based wrapper used by tests and non-distributed callers."""
    if not statuses:
        raise ValueError("at least one optimizer status is required")
    normalized = [optimizer_status_code(status) for status in statuses]
    kinds = [kind for kind, _ in normalized]
    steps = [step for _, step in normalized]
    return validate_optimizer_resume_consensus(
        min(kinds),
        max(kinds),
        min(steps),
        max(steps),
        require=require,
        expected_step=expected_step,
    )


def resolve_recovery_resume(
    *,
    next_schedule_step: int,
    source_policy_checkpoint_step: int | None,
    optimizer_checkpoint_step: int | None,
    has_resume_adapter: bool,
    require_optimizer_resume: bool,
) -> tuple[int | None, int | None]:
    """Validate cursor-driven resume state independently from schedule position.

    A no-update schedule step advances ``next_schedule_step`` without changing
    the source policy or AdamW state.  The historical ``--start-step`` contract
    conflated those coordinates; this helper makes their relationship explicit
    before distributed initialization can strand a partial world.
    """

    if next_schedule_step < 0:
        raise ValueError("next schedule step must be nonnegative")
    source = (
        None
        if source_policy_checkpoint_step in (None, 0)
        else source_policy_checkpoint_step
    )
    optimizer = (
        None
        if optimizer_checkpoint_step in (None, 0)
        else optimizer_checkpoint_step
    )
    if source is not None and source <= 0:
        raise ValueError("source policy checkpoint step must be positive or zero/base")
    if optimizer is not None and optimizer <= 0:
        raise ValueError("optimizer checkpoint step must be positive or zero/fresh")
    if next_schedule_step == 0 and (source is not None or optimizer is not None):
        raise ValueError("schedule step zero must start from true base/fresh AdamW")
    if source is None:
        if has_resume_adapter:
            raise ValueError("base/fresh recovery must not receive --resume-lora-dir")
        if optimizer is not None or require_optimizer_resume:
            raise ValueError("base/fresh recovery must not request optimizer restore")
        return None, None
    if not has_resume_adapter:
        raise ValueError("checkpoint recovery requires --resume-lora-dir")
    if source > next_schedule_step:
        raise ValueError(
            "source policy checkpoint cannot be newer than next schedule step"
        )
    # Current producer saves policy and AdamW atomically under one checkpoint
    # directory.  Rejecting a split pair avoids silently mixing policies/moments.
    if optimizer != source or not require_optimizer_resume:
        raise ValueError(
            "checkpoint recovery requires matching source/optimizer checkpoint "
            "steps and --require-optimizer-resume"
        )
    return source, optimizer
