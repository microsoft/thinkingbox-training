"""Async rollout runner: produces `G` trajectories per prompt for GRPO.

Reuses `thinkingbox.cli.infer.TBWorker` so the rollout matches what `tb infer`
does end-to-end (agent loop + test/judge → `TestResult.reward`).
"""

from __future__ import annotations

import asyncio
import logging
import zlib
from dataclasses import dataclass, field
from typing import Any

from pydantic_core import to_jsonable_python

from thinkingbox.cli.infer import TBWorker
from thinkingbox.common.chat_types import DecodeResult
from thinkingbox.common.config_types import ConfigFile, HydratedTestCase

from train import patches as _patches
from train.token_replay import (
    BehaviorGeneration,
    TokenReplayError,
    snapshot_generation_traces,
)

logger = logging.getLogger(__name__)


@dataclass
class Rollout:
    uid: str
    sample_idx: int
    reward: float
    is_correct: bool
    is_system_error: bool
    finish_reason: str
    messages: list[dict]
    raw_messages: list[dict] | None
    metadata: dict[str, Any] = field(default_factory=dict)
    # OAI-encoded tool schema active during this rollout (the same list tb
    # passes to vLLM on the wire). None if not captured.
    tools: list[dict] | None = None
    # Immutable exact token/logprob trace captured from the behavior (vLLM) engine.
    # Required for the P1 behavior-logprob PPO ratio; system-error/eval stubs leave
    # this None and are excluded from training.
    behavior_generations: tuple[BehaviorGeneration, ...] | None = None
    capture_integrity_error: str | None = None

    @classmethod
    def from_decode_result(cls, result: DecodeResult, sample_idx: int) -> "Rollout":
        tr = result.test_result
        reward = float(tr.reward) if tr is not None else 0.0
        is_correct = bool(tr.result) if tr is not None else False
        return cls(
            uid=result.uid,
            sample_idx=sample_idx,
            reward=reward,
            is_correct=is_correct,
            is_system_error=bool(result.is_system_error),
            finish_reason=str(result.finish_reason),
            messages=to_jsonable_python(result.messages),
            raw_messages=result.raw_messages,
            metadata=to_jsonable_python(result.metadata),
        )


class RolloutRunner:
    """Run G rollouts per test case, bounded by a global concurrency semaphore."""

    def __init__(
        self,
        config: ConfigFile,
        concurrency: int = 8,
        dump_raw: bool = True,
        mcp_urls: list[str] | None = None,
        require_exact_tokens: bool = True,
    ):
        self.config = config
        self.concurrency = concurrency
        self.dump_raw = dump_raw
        # When True (RL training), each rollout must expose an exact behavior
        # token/logprob trace; rollouts without one are marked with a
        # capture_integrity_error and excluded from the training batch. Eval sets
        # this False (it never touches behavior logprobs).
        self.require_exact_tokens = require_exact_tokens
        # Fail-closed: RL training that consumes behavior logprobs (the P1 fix)
        # is only correct if the behavior engine was actually told to record the
        # exact sampled tokens/logprobs. If capture is not enabled in the config
        # every rollout would silently produce no trace -> capture_integrity_error
        # on all of them -> an empty training batch. Refuse to start instead.
        if require_exact_tokens:
            orchestrator = getattr(config, "orchestrator", None)
            agent_model = (
                getattr(orchestrator, "agent_model", None)
                if orchestrator is not None
                else None
            )
            if getattr(agent_model, "capture_exact_tokens", False) is not True:
                raise TokenReplayError(
                    "training rollouts require orchestrator.agent_model."
                    "capture_exact_tokens=true"
                )
        self._sem = asyncio.Semaphore(concurrency)
        # MCP load-balancing: build one TBWorker per UNIQUE MCP endpoint URL.
        # A TBWorker is stateless across `.work()` calls — each `_decode` opens
        # its own MCPProxyClient session from `self.config.mcp_proxy.endpoint_url`
        # (see thinkingbox.cli.infer.TBWorker._decode) — so we can safely share a
        # worker across concurrent tasks AND round-robin across multiple workers
        # whose configs point at different MCP instances (load balancing).
        if not mcp_urls:
            # default: single endpoint exactly as configured today.
            mcp_urls = [config.mcp_proxy.endpoint_url]
        # de-dup while preserving order so `--mcp-urls A A` collapses to one worker.
        seen: set[str] = set()
        uniq_urls = [u for u in mcp_urls if not (u in seen or seen.add(u))]
        self.mcp_urls = uniq_urls
        self._workers: list[TBWorker] = []
        for url in uniq_urls:
            wcfg = config.model_copy(deep=True)
            wcfg.mcp_proxy.endpoint_url = url
            self._workers.append(
                TBWorker(
                    config=wcfg,
                    skip_test=False,
                    skip_agent=False,
                    dump_tools=False,
                    dump_testcontext=False,
                    dump_userllm=False,
                    dump_raw=dump_raw,
                    debug_test=False,
                    linger_sessions=False,
                )
            )

    async def _one(self, tc: HydratedTestCase, sample_idx: int) -> Rollout:
        async with self._sem:
            # deep-copy so per-sample metadata (e.g. repetition idx) doesn't leak
            tc_copy = tc.model_copy(deep=True)
            tc_copy.metadata["rollout_sample_idx"] = sample_idx
            # Reset the per-task tool slot before the worker runs; the patched
            # AgentSession.__init__ inside worker.work() will populate it. Each
            # _one() runs in its own asyncio task so this contextvar is task-local.
            _patches.current_tools.set(None)
            _patches.current_trace_session.set(None)
            _patches.current_trace_error.set(None)
            # MCP load balancing: pick a worker by a STABLE per-task key (uid hash +
            # sample_idx). Keying on sample_idx alone pinned every single-sample call
            # site to worker 0 (eval does _one(tc, 0) for every task -> no balancing);
            # hashing the uid spreads those across endpoints too, while +sample_idx
            # still spreads a prompt's G training samples. Each worker's `_decode` opens
            # an independent session, so concurrent use of the same worker is safe.
            worker_idx = (zlib.adler32(tc_copy.uid.encode("utf-8")) + sample_idx) % len(self._workers)
            worker = self._workers[worker_idx]
            try:
                work_result = await worker.work(tc_copy)
            except Exception as exc:
                trace_error = _patches.current_trace_error.get()
                if trace_error is None and type(exc).__name__ != "ExactTokenTraceError":
                    raise
                # An exact-token-trace failure is a disposition, not a crash: return a
                # system-error stub so the group survives and the rollout is dropped.
                return Rollout(
                    uid=tc_copy.uid,
                    sample_idx=sample_idx,
                    reward=0.0,
                    is_correct=False,
                    is_system_error=True,
                    finish_reason="exact_capture_error",
                    messages=[],
                    raw_messages=None,
                    metadata={"exact_capture_error": trace_error or str(exc)},
                    capture_integrity_error=trace_error or str(exc),
                )
            tools = _patches.current_tools.get()
            r = Rollout.from_decode_result(work_result.result, sample_idx)
            r.tools = tools
            if self.require_exact_tokens:
                trace_error = _patches.current_trace_error.get()
                if trace_error is not None:
                    r.capture_integrity_error = trace_error
                    return r
                if r.is_system_error:
                    return r
                trace_session = _patches.current_trace_session.get()
                if trace_session is None:
                    r.capture_integrity_error = (
                        "agent session did not expose exact generation traces; "
                        "the exact-capture framework support is required"
                    )
                    return r
                try:
                    r.behavior_generations = snapshot_generation_traces(
                        trace_session.get_generation_traces()
                    )
                except TokenReplayError as exc:
                    r.capture_integrity_error = str(exc)
            return r

    async def rollout(self, tc: HydratedTestCase, n_samples: int) -> list[Rollout]:
        """Produce `n_samples` rollouts for a single test case (concurrent)."""
        tasks = [self._one(tc, i) for i in range(n_samples)]
        return await asyncio.gather(*tasks)

    async def rollout_many(
        self,
        tcs: list[HydratedTestCase],
        n_samples: int,
        progress_label: str | None = None,
        progress_interval: float = 30.0,
    ) -> dict[str, list[Rollout]]:
        """Run rollouts for every tc; returns {uid: [Rollout, ...]}.

        If ``progress_label`` is set, a background task logs ``Progress: ...``
        every ``progress_interval`` seconds while rollouts are in flight.
        Pass e.g. "eval step=5" or "step 7 rollout" to label the output.
        """
        total = len(tcs) * n_samples
        if total == 0:
            return {}

        counter = {"done": 0, "err": 0}

        async def _tracked(tc: HydratedTestCase, i: int) -> Rollout:
            try:
                r = await self._one(tc, i)
            finally:
                counter["done"] += 1
            if r.is_system_error:
                counter["err"] += 1
            return r

        progress_task = None
        if progress_label:
            progress_task = asyncio.create_task(
                _log_progress(progress_label, counter, total, progress_interval)
            )
        try:
            tasks = [_tracked(tc, i) for tc in tcs for i in range(n_samples)]
            flat: list[Rollout] = await asyncio.gather(*tasks)
        finally:
            if progress_task is not None:
                progress_task.cancel()
                try:
                    await progress_task
                except (asyncio.CancelledError, Exception):
                    pass
        out: dict[str, list[Rollout]] = {}
        for r in flat:
            out.setdefault(r.uid, []).append(r)
        for uid in out:
            out[uid].sort(key=lambda r: r.sample_idx)
        return out

    async def bounded_rollout_many(
        self,
        tcs: list[HydratedTestCase],
        n_samples: int,
        timeout_s: float,
        progress_label: str | None = None,
        progress_interval: float = 30.0,
        group_ids: list[str] | None = None,
    ) -> tuple[dict[str, list[Rollout]], int]:
        """Run every sample under one wall-clock budget without losing partial work.

        Returns ``({uid: [Rollout, ...]}, n_unfinished)``. Every requested sample has
        exactly one result: completed rollouts are preserved, while timed-out,
        cancelled, or failed tasks become explicit system-error stubs. All pending
        tasks and the optional progress logger are cancelled and drained before
        this method returns.
        """
        if group_ids is None:
            group_ids = [str(tc.uid) for tc in tcs]
        elif len(group_ids) != len(tcs):
            raise ValueError("group_ids must align one-to-one with tcs")
        elif len(set(group_ids)) != len(group_ids):
            raise ValueError("group_ids must be unique")

        counter = {"done": 0, "err": 0}

        async def _tracked(tc: HydratedTestCase, sample_idx: int) -> Rollout:
            try:
                rollout = await self._one(tc, sample_idx)
                if rollout.is_system_error:
                    counter["err"] += 1
                return rollout
            finally:
                counter["done"] += 1

        triples = [
            (group_id, sample_idx,
             asyncio.create_task(_tracked(tc, sample_idx)))
            for group_id, tc in zip(group_ids, tcs)
            for sample_idx in range(n_samples)
        ]
        if not triples:
            return {}, 0

        tasks = [task for _, _, task in triples]
        progress_task: asyncio.Task[None] | None = None
        timed_out: set[asyncio.Task[Rollout]] = set()
        if progress_label:
            progress_task = asyncio.create_task(
                _log_progress(
                    progress_label, counter, len(tasks), progress_interval
                )
            )

        try:
            _, pending = await asyncio.wait(tasks, timeout=timeout_s)
            timed_out = set(pending)
        finally:
            unfinished = [task for task in tasks if not task.done()]
            for task in unfinished:
                task.cancel()
            if unfinished:
                await asyncio.gather(*unfinished, return_exceptions=True)
            if progress_task is not None:
                progress_task.cancel()
                await asyncio.gather(progress_task, return_exceptions=True)

        out: dict[str, list[Rollout]] = {}
        n_unfinished = 0
        for uid, sample_idx, task in triples:
            if task in timed_out:
                rollout = _system_error_rollout(uid, sample_idx, "timeout")
                n_unfinished += 1
            elif task.cancelled():
                rollout = _system_error_rollout(uid, sample_idx, "cancelled")
                n_unfinished += 1
            else:
                exc = task.exception()
                if exc is None:
                    rollout = task.result()
                else:
                    reason = f"exception: {type(exc).__name__}"
                    rollout = _system_error_rollout(
                        uid, sample_idx, reason, detail=str(exc)
                    )
                    n_unfinished += 1
            out.setdefault(uid, []).append(rollout)
        for uid in out:
            out[uid].sort(key=lambda rollout: rollout.sample_idx)
        return out, n_unfinished


def _system_error_rollout(
    uid: str,
    sample_idx: int,
    reason: str,
    detail: str | None = None,
) -> Rollout:
    """Create a non-trainable placeholder for an unfinished rollout task."""
    metadata: dict[str, Any] = {
        "reason": reason,
        "rollout_unfinished": True,
    }
    if reason == "timeout":
        metadata["rollout_timeout"] = True
    if detail:
        metadata["detail"] = detail
    return Rollout(
        uid=uid,
        sample_idx=sample_idx,
        reward=0.0,
        is_correct=False,
        is_system_error=True,
        finish_reason=reason,
        messages=[],
        raw_messages=None,
        metadata=metadata,
    )


async def _log_progress(label: str, counter: dict, total: int, interval: float) -> None:
    """Periodic progress logger run alongside a batch of rollouts."""
    loop = asyncio.get_running_loop()
    t0 = loop.time()
    while True:
        try:
            await asyncio.sleep(interval)
        except asyncio.CancelledError:
            return
        done = counter["done"]
        err = counter["err"]
        in_flight = total - done
        elapsed = loop.time() - t0
        logger.info(
            "[%s] Progress: t=%.0fs  done=%d/%d  err=%d  in-flight=%d",
            label, elapsed, done, total, err, in_flight,
        )
