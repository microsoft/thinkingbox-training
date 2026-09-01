"""Lightweight periodic eval for thinkingbox-training.

One sample per case (no grouping), runs the configured rollout + reward path,
returns aggregate metrics. Designed to be cheap: rank-0 only, parallelism set
to len(eval_cases) so the whole eval is one HTTP burst against vLLM.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from thinkingbox.common.config_types import ConfigFile, HydratedTestCase

from train.rewards import score_rollouts
from train.rollout import Rollout, RolloutRunner

logger = logging.getLogger(__name__)

# Upper bound on concurrent eval rollouts, independent of eval-set size, so a
# large --eval-list doesn't fan out one rollout per case onto MCP + user-sim.
_EVAL_MAX_CONCURRENCY = 64


def _build_runner(
    tb_cfg: ConfigFile,
    policy_id: str,
    concurrency: int,
    mcp_urls: list[str] | None = None,
) -> RolloutRunner:
    new_cfg = tb_cfg.model_copy(deep=True)
    new_cfg.orchestrator.agent_model.deployment = policy_id
    # Eval never consumes behavior logprobs, so it opts out of the fail-closed
    # exact-token trace requirement (require_exact_tokens=False) -> identical to the
    # pre-P1 eval path; only the training rollouts capture exact traces.
    return RolloutRunner(config=new_cfg, concurrency=concurrency, dump_raw=False,
                         require_exact_tokens=False, mcp_urls=mcp_urls)


async def _bounded_eval_rollouts(
    runner: RolloutRunner,
    eval_cases: list[HydratedTestCase],
    timeout_s: float,
    progress_label: str,
) -> tuple[dict[str, list[Rollout]], int]:
    """Like ``RolloutRunner.rollout_many`` but with a wall-clock budget.

    Cases that don't complete within ``timeout_s`` are cancelled and replaced
    with system-error stub Rollouts so the eval can still report metrics over
    the cases that did finish. Returns ``(by_uid, n_timed_out)``.
    """
    from train.rollout import _log_progress  # local import to avoid cycle

    triples = [
        (tc.uid, asyncio.create_task(runner._one(tc, 0)))
        for tc in eval_cases
    ]
    counter = {"done": 0, "err": 0}

    async def _track(uid: str, task: asyncio.Task) -> None:
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass
        finally:
            counter["done"] += 1

    trackers = [asyncio.create_task(_track(u, t)) for u, t in triples]
    progress_task = asyncio.create_task(
        _log_progress(progress_label, counter, len(triples), 30.0)
    )
    try:
        await asyncio.wait(trackers, timeout=timeout_s)
    finally:
        progress_task.cancel()
        try:
            await progress_task
        except (asyncio.CancelledError, Exception):
            pass
        # Cancel anything still in flight (the tracker + the underlying rollout).
        for tr in trackers:
            tr.cancel()
        for _, task in triples:
            if not task.done():
                task.cancel()
        # Drain so we don't leak warnings.
        await asyncio.gather(*trackers, return_exceptions=True)
        await asyncio.gather(*(t for _, t in triples), return_exceptions=True)

    by_uid: dict[str, list[Rollout]] = {}
    n_timed_out = 0
    for uid, task in triples:
        if task.done() and not task.cancelled() and task.exception() is None:
            by_uid.setdefault(uid, []).append(task.result())
        else:
            n_timed_out += 1
            by_uid.setdefault(uid, []).append(
                Rollout(
                    uid=uid,
                    sample_idx=0,
                    reward=0.0,
                    is_correct=False,
                    is_system_error=True,
                    finish_reason="eval_timeout",
                    messages=[],
                    raw_messages=None,
                    metadata={"eval_timeout": True},
                )
            )
    return by_uid, n_timed_out


def run_eval(
    step: int,
    tb_cfg: ConfigFile,
    eval_cases: list[HydratedTestCase],
    policy_id: str,
    reward_fn: str = "binary_test_result",
    timeout_s: float = 1500.0,
    mcp_urls: list[str] | None = None,
) -> dict[str, Any]:
    """Roll out 1 sample per eval case against vLLM, return aggregate metrics.

    Cheap on purpose: no tokenization, no policy forward, no grad. The whole
    eval pass = one parallel rollout burst + reward scoring.

    Bounded by ``timeout_s`` (default 25 min, well under the 30 min NCCL
    watchdog): unfinished cases are dropped as system errors so the post-eval
    barrier on other ranks never deadlocks.
    """
    if not eval_cases:
        return {"phase": "eval", "step": step, "n_cases": 0, "skipped": "no_cases"}

    # Cap eval fan-out: a large --eval-list (e.g. a 98/507-case OOD set) would
    # otherwise spawn one concurrent rollout per case and overload the MCP +
    # user-sim endpoints. Small val splits (<= cap) are unaffected.
    eval_concurrency = min(len(eval_cases), _EVAL_MAX_CONCURRENCY)
    runner = _build_runner(
        tb_cfg,
        policy_id,
        concurrency=eval_concurrency,
        mcp_urls=mcp_urls,
    )
    t0 = time.monotonic()
    rollouts_by_uid, n_timed_out = asyncio.run(
        _bounded_eval_rollouts(
            runner, eval_cases,
            timeout_s=timeout_s,
            progress_label=f"eval step={step:04d}",
        )
    )
    t_eval = time.monotonic() - t0
    if n_timed_out:
        logger.warning(
            "eval step=%04d: %d/%d case(s) cancelled by eval timeout (%.0fs budget)",
            step, n_timed_out, len(eval_cases), timeout_s,
        )

    flat = []
    uids = []
    for uid in sorted(rollouts_by_uid):
        for r in rollouts_by_uid[uid]:
            flat.append(r)
            uids.append(uid)
    if not flat:
        return {"phase": "eval", "step": step, "n_cases": 0, "skipped": "no_rollouts",
                "t_eval_s": round(t_eval, 2)}

    # Report both denominators and define headline pass_rate over every
    # scheduled case.
    #
    # The old metric divided only by completed (non-system-error) cases, so a step
    # with more crashes scored HIGHER and the denominator moved between steps. That
    # makes an eval curve ambiguous: a rising pass_rate can mean a better
    # policy or a less reliable harness.
    #
    #   pass_rate            passes / all scheduled cases   (stable denominator)
    #   pass_rate_completed  passes / completed cases       (the legacy number)
    #
    # A system-errored case counts as a non-pass in the headline metric: it did not
    # solve the task, whatever the reason.
    completed = [(u, r) for u, r in zip(uids, flat) if not r.is_system_error]
    n_sys_errors = sum(r.is_system_error for r in flat)
    n_scheduled = len(flat)
    if completed:
        c_uids = [u for u, _ in completed]
        c_rollouts = [r for _, r in completed]
        rewards = score_rollouts(c_rollouts, name=reward_fn)
        n = len(rewards)
        n_passed = sum(1 for r in rewards if r > 0)
        reward_sum = sum(rewards)
        mean = reward_sum / n if n else 0.0
        pass_rate_completed = n_passed / n if n else 0.0
        # scheduled-denominator versions: system errors contribute 0 reward
        mean_scheduled = reward_sum / n_scheduled if n_scheduled else 0.0
        pass_rate = n_passed / n_scheduled if n_scheduled else 0.0
        per_case = [{"uid": u, "reward": float(r)} for u, r in zip(c_uids, rewards)]
    else:
        n = 0
        mean = 0.0
        mean_scheduled = 0.0
        pass_rate = 0.0
        pass_rate_completed = 0.0
        per_case = []

    return {
        "phase": "eval",
        "step": step,
        "policy_id": policy_id,
        "n_cases": n,
        "n_scheduled": n_scheduled,
        "n_sys_errors": n_sys_errors,
        "n_timed_out": n_timed_out,
        "reward_mean": float(mean_scheduled),
        "reward_mean_completed": float(mean),
        "pass_rate": float(pass_rate),
        "pass_rate_completed": float(pass_rate_completed),
        "t_eval_s": round(t_eval, 2),
        "per_case": per_case,
    }
