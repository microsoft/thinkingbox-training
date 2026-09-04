"""Thin wrapper around wandb (+ optional weave) for the training loop.

Designed so the rest of the code doesn't have to care whether logging is on,
which project to use, or whether the run is resumed. Rank-0 only.

Usage::

    log = WandbLogger.maybe_init(
        enabled=args.wandb,
        project="thinkingbox-training",
        entity=None,
        run_name=args.wandb_run_name,
        run_id=resumed_run_id,            # None for fresh, str for resume
        weave_enabled=args.weave,
        config=cfg_dict,
    )
    log.log_train(step, train_row)
    log.log_eval(step, eval_row)
    run_id_for_checkpoint = log.run_id
    log.finish()
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Keys we never want to push as scalar metrics to wandb (strings / lists / etc).
_NON_SCALAR_TRAIN_KEYS = {"lora_name", "next_lora_name"}
_NON_SCALAR_EVAL_KEYS = {"policy_id", "phase", "per_case", "skipped"}


@dataclass
class WandbLogger:
    enabled: bool
    run: Any = None        # wandb.sdk.wandb_run.Run | None
    weave_client: Any = None
    run_id: str | None = None

    @classmethod
    def maybe_init(
        cls,
        enabled: bool,
        project: str,
        entity: str | None,
        run_name: str | None,
        run_id: str | None,
        weave_enabled: bool,
        config: dict | None = None,
    ) -> "WandbLogger":
        if not enabled and not weave_enabled:
            return cls(enabled=False)

        run = None
        weave_client = None
        final_run_id = run_id

        if enabled:
            try:
                import wandb
            except ImportError:
                logger.warning("wandb requested but not installed; pip install wandb")
            else:
                init_kwargs: dict[str, Any] = {
                    "project": project,
                    "entity": entity,
                    "name": run_name,
                    "config": config,
                }
                if run_id:
                    init_kwargs["id"] = run_id
                    init_kwargs["resume"] = "allow"
                run = wandb.init(**{k: v for k, v in init_kwargs.items() if v is not None})
                final_run_id = run.id
                logger.info("wandb run started: %s (id=%s)", run.name, run.id)

        if weave_enabled:
            try:
                import weave
            except ImportError:
                logger.warning("weave requested but not installed; pip install weave")
            else:
                # weave expects 'entity/project' as a single string.
                target = f"{entity}/{project}" if entity else project
                weave_client = weave.init(target)
                logger.info("weave tracing enabled: %s", target)

        return cls(
            enabled=bool(run is not None or weave_client is not None),
            run=run,
            weave_client=weave_client,
            run_id=final_run_id,
        )

    # ------------------------------------------------------------------
    # Per-step logging
    # ------------------------------------------------------------------
    def log_train(self, step: int, row: dict) -> None:
        if self.run is None:
            return
        payload = {f"train/{k}": v for k, v in row.items()
                   if k not in _NON_SCALAR_TRAIN_KEYS and k != "step"
                   and isinstance(v, (int, float, bool))}
        # Surface adapter name as a wandb summary string (not a per-step metric).
        if "lora_name" in row:
            self.run.summary["last_train_lora"] = row["lora_name"]
        self.run.log(payload, step=step)

    def log_eval(self, step: int, row: dict) -> None:
        if self.run is None:
            return
        if row.get("skipped"):
            return
        payload = {f"eval/{k}": v for k, v in row.items()
                   if k not in _NON_SCALAR_EVAL_KEYS and k != "step"
                   and isinstance(v, (int, float, bool))}
        if "policy_id" in row:
            self.run.summary["last_eval_lora"] = row["policy_id"]
        self.run.log(payload, step=step)

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()
