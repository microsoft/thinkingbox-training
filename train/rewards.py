"""Reward functions for RL training.

Each reward function takes a `Rollout` (see train.rollout) and returns a scalar.
Register new functions here so the training loop can pick by string name.

Current functions:
  * binary_test_result : 1.0 if `tb infer` test passed, 0.0 otherwise — the
    default for outcome-based RL on thinkingbox cases. Reads
    `Rollout.reward` which is sourced from `DecodeResult.test_result.reward`.

Future ideas (not implemented):
  * judge_score      : continuous score from an LLM judge
  * reward_model     : score from a learned RM
  * length_penalty   : combine correctness with a brevity bonus
"""
from __future__ import annotations

from typing import Callable, Iterable


# Pluggable registry: name -> fn(rollout) -> float
REWARD_FNS: dict[str, Callable] = {}


def register(name: str):
    def deco(fn: Callable):
        REWARD_FNS[name] = fn
        return fn
    return deco


@register("binary_test_result")
def binary_test_result(rollout) -> float:
    """Use the raw pass/fail reward already extracted from the test harness."""
    return float(rollout.reward)


def score_rollouts(rollouts: Iterable, name: str = "binary_test_result") -> list[float]:
    """Apply the named reward function to a sequence of rollouts."""
    if name not in REWARD_FNS:
        raise KeyError(f"unknown reward fn: {name!r}; available: {sorted(REWARD_FNS)}")
    fn = REWARD_FNS[name]
    return [fn(r) for r in rollouts]


def apply_overlong_penalty(
    rollouts: list,
    rewards: list[float],
    penalty: float,
    finish_reasons: tuple[str, ...] = ("length",),
) -> list[float]:
    """DAPO-style overlong shaping for multi-turn agentic rollouts.

    Subtract `penalty` from the reward of any rollout that failed to terminate
    (finish_reason in `finish_reasons`, default "length" = hit the token cap).
    Unlike DAPO's length-graded soft punishment, we key off finish_reason so a
    legitimately long *terminated* agentic task (finish_reason="stop"/"tool_calls")
    is NOT penalized -- only non-terminating rollouts cut off by length.
    Creates a gradient toward proper termination in mixed-reward groups.

    Returns a new rewards list; only applied to TRAINING rewards (not eval).
    """
    if penalty <= 0.0:
        return rewards
    out = list(rewards)
    for i, r in enumerate(rollouts):
        if getattr(r, "finish_reason", None) in finish_reasons:
            out[i] = out[i] - penalty
    return out
