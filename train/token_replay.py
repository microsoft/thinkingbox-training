"""Build policy inputs from exact behavior-engine token traces.

RL training must consume the *exact* tokens the behavior engine (vLLM) sampled,
together with their per-token behavior logprobs, so the PPO/DAPO importance ratio
``exp(log pi_current - log pi_behavior)`` is a real off-policy ratio rather than the
degenerate ``ratio == 1`` produced by reusing the training forward's own detached
logprob (the "P1" bug).

Each serving request becomes an independent ``prompt_ids + completion_ids`` causal
segment. Later chat prompts may reserialize earlier reasoning/tool turns and are
therefore never assumed to extend previous token streams.

This module is intentionally self-contained (stdlib only) so it can be unit-tested
on CPU without the trainer, tokenizer, or a live serving engine.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Sequence


class TokenReplayError(RuntimeError):
    """A captured token trace is incomplete or ambiguous."""


class TokenProvenance(str, Enum):
    CONTEXT = "context"
    ASSISTANT_ACTION = "assistant_action"
    ASSISTANT_STOP = "assistant_stop"

    @property
    def is_action(self) -> bool:
        return self is not TokenProvenance.CONTEXT


@dataclass(frozen=True)
class BehaviorToken:
    token_id: int
    logprob: float
    is_stop: bool = False


@dataclass(frozen=True)
class BehaviorGeneration:
    prompt_token_ids: tuple[int, ...]
    completion_tokens: tuple[BehaviorToken, ...]
    finish_reason: str | None = None
    stop_reason: int | str | None = None


@dataclass
class TokenizedRollout:
    input_ids: list[int]
    token_provenance: list[TokenProvenance]
    behavior_logprobs: list[float | None]
    assistant_spans: list[tuple[int, int]]  # (start, end_exclusive) per assistant message
    generation_index: int = 0
    prompt_reserialized: bool = False
    finish_reason: str | None = None
    stop_reason: int | str | None = None

    def __post_init__(self) -> None:
        n = len(self.input_ids)
        if len(self.token_provenance) != n or len(self.behavior_logprobs) != n:
            raise ValueError("token IDs, provenance, and behavior logprobs must align")
        for index, (source, logprob) in enumerate(
            zip(self.token_provenance, self.behavior_logprobs, strict=True)
        ):
            if source.is_action:
                if logprob is None or not math.isfinite(logprob):
                    raise ValueError(
                        f"assistant action token {index} needs a finite behavior logprob"
                    )
            elif logprob is not None:
                raise ValueError(f"context token {index} cannot have a behavior logprob")

    @property
    def assistant_mask(self) -> list[int]:
        """Policy-action mask derived exclusively from captured provenance."""
        return [int(source.is_action) for source in self.token_provenance]


@dataclass(frozen=True)
class SegmentDisposition:
    segment: TokenizedRollout | None
    invalid_reason: str | None = None

    @property
    def is_valid(self) -> bool:
        return self.segment is not None


def _strict_int_tuple(value: object, field: str) -> tuple[int, ...]:
    if not isinstance(value, (tuple, list)) or any(
        not isinstance(item, int) or isinstance(item, bool) for item in value
    ):
        raise TokenReplayError(f"{field} must be a sequence of integer token IDs")
    return tuple(value)


def snapshot_generation_traces(raw_traces: object) -> tuple[BehaviorGeneration, ...]:
    """Copy framework traces into strict, trainer-owned immutable values."""
    if not isinstance(raw_traces, (tuple, list)):
        raise TokenReplayError("generation traces must be a sequence")
    snapshots: list[BehaviorGeneration] = []
    for trace_index, raw_trace in enumerate(raw_traces):
        prompt_ids = _strict_int_tuple(
            getattr(raw_trace, "prompt_token_ids", None),
            f"trace[{trace_index}].prompt_token_ids",
        )
        raw_tokens = getattr(raw_trace, "completion_tokens", None)
        if not isinstance(raw_tokens, (tuple, list)):
            raise TokenReplayError(
                f"trace[{trace_index}].completion_tokens must be a sequence"
            )
        tokens: list[BehaviorToken] = []
        for token_index, raw_token in enumerate(raw_tokens):
            token_id = getattr(raw_token, "token_id", None)
            logprob = getattr(raw_token, "logprob", None)
            is_stop = getattr(raw_token, "is_stop", None)
            if not isinstance(token_id, int) or isinstance(token_id, bool):
                raise TokenReplayError(
                    f"trace[{trace_index}].completion_tokens[{token_index}].token_id "
                    "must be an integer"
                )
            if (
                not isinstance(logprob, (int, float))
                or isinstance(logprob, bool)
                or not math.isfinite(float(logprob))
            ):
                raise TokenReplayError(
                    f"trace[{trace_index}].completion_tokens[{token_index}].logprob "
                    "must be finite"
                )
            if not isinstance(is_stop, bool):
                raise TokenReplayError(
                    f"trace[{trace_index}].completion_tokens[{token_index}].is_stop "
                    "must be boolean"
                )
            tokens.append(
                BehaviorToken(
                    token_id=token_id,
                    logprob=float(logprob),
                    is_stop=is_stop,
                )
            )
        if not tokens:
            raise TokenReplayError(
                f"trace[{trace_index}] has no sampled completion tokens"
            )
        finish_reason = getattr(raw_trace, "finish_reason", None)
        stop_reason = getattr(raw_trace, "stop_reason", None)
        if finish_reason is not None and not isinstance(finish_reason, str):
            raise TokenReplayError(
                f"trace[{trace_index}].finish_reason must be a string or null"
            )
        if (
            stop_reason is not None
            and (
                not isinstance(stop_reason, (int, str))
                or isinstance(stop_reason, bool)
            )
        ):
            raise TokenReplayError(
                f"trace[{trace_index}].stop_reason must be an integer, string, or null"
            )
        snapshots.append(
            BehaviorGeneration(
                prompt_token_ids=prompt_ids,
                completion_tokens=tuple(tokens),
                finish_reason=finish_reason,
                stop_reason=stop_reason,
            )
        )
    if not snapshots:
        raise TokenReplayError("rollout has no captured behavior generations")
    return tuple(snapshots)


def build_token_training_segments(
    generations: Sequence[BehaviorGeneration],
) -> list[TokenizedRollout]:
    """Create one exact causal training segment per serving generation."""
    if not generations:
        raise TokenReplayError("rollout has no captured behavior generations")

    segments: list[TokenizedRollout] = []
    previous_sampled_stream: list[int] | None = None
    for generation_index, generation in enumerate(generations):
        prompt = list(generation.prompt_token_ids)
        if not prompt:
            raise TokenReplayError(
                f"generation {generation_index} has no causal prompt tokens"
            )
        completion_ids = [token.token_id for token in generation.completion_tokens]
        prompt_reserialized = (
            previous_sampled_stream is not None
            and (
                len(prompt) < len(previous_sampled_stream)
                or prompt[: len(previous_sampled_stream)] != previous_sampled_stream
            )
        )
        ids = list(prompt)
        provenance = [TokenProvenance.CONTEXT] * len(prompt)
        behavior_logprobs: list[float | None] = [None] * len(prompt)
        span_start = len(prompt)
        for token in generation.completion_tokens:
            ids.append(token.token_id)
            provenance.append(
                TokenProvenance.ASSISTANT_STOP
                if token.is_stop
                else TokenProvenance.ASSISTANT_ACTION
            )
            behavior_logprobs.append(token.logprob)
        segments.append(
            TokenizedRollout(
                input_ids=ids,
                token_provenance=provenance,
                behavior_logprobs=behavior_logprobs,
                assistant_spans=[(span_start, len(ids))],
                generation_index=generation_index,
                prompt_reserialized=prompt_reserialized,
                finish_reason=generation.finish_reason,
                stop_reason=generation.stop_reason,
            )
        )
        previous_sampled_stream = prompt + completion_ids
    return segments


def validate_segment_length(
    segment: TokenizedRollout,
    *,
    reward: float,
    max_seq_len: int,
) -> SegmentDisposition:
    """Return an invalid disposition instead of truncating sampled actions."""
    if max_seq_len < 2:
        raise ValueError("max_seq_len must leave at least one causal target")
    if len(segment.input_ids) <= max_seq_len:
        return SegmentDisposition(segment=segment)
    removed_actions = any(
        source.is_action for source in segment.token_provenance[max_seq_len:]
    )
    if removed_actions:
        return SegmentDisposition(
            segment=None,
            invalid_reason=(
                f"generation {segment.generation_index} exceeds max_seq_len="
                f"{max_seq_len}; truncation would remove sampled actions while "
                f"retaining terminal reward {reward}"
            ),
        )
    return SegmentDisposition(
        segment=None,
        invalid_reason=(
            f"generation {segment.generation_index} exceeds max_seq_len="
            f"{max_seq_len} without a complete sampled action"
        ),
    )
