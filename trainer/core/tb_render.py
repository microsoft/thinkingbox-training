"""Anchored rendering of conversation deltas into trajectory tokens.

Observation deltas (tool responses, simulated-user turns) are appended to a
trajectory's token buffer by rendering just the new messages through the
chat template. Chat templates only render *complete* conversations, and
Qwen3.8's enforces that two ways:

- handed a fragment with no leading system message, it injects its default
  system preamble conditionally on the fragment's leading role, so verl's
  ``remove_system_prompt=True`` — a blind slice sized on a user-first probe
  — beheads tool-first observations (verl #3948/#3081, unfixed in 0.9.0);
- it REQUIRES at least one real user query (a user message not wrapped in
  ``<tool_response>``): rendering a fragment without one raises
  ``TemplateError('No user query found in messages.')``, so tool-only
  fragments cannot be rendered alone at all.

Both problems disappear by rendering every delta behind a fixed ANCHOR
(system message, user query) pair: the template takes its "caller supplied
a system prompt" branch and always sees a real user query. The pair's
rendered prefix — measured once by ``build_delta_anchor_prefix`` with the
same template kwargs the rollout renders use — is stripped exactly by
``strip_delta_anchor`` after token-by-token verification. A template that
renders the anchor context-dependently raises ``DeltaRenderError``, turning
would-be silent corruption into a loud rollout failure.

This module deliberately has no verl or thinkingbox dependencies, so its
tests run anywhere.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

DELTA_ANCHOR = {"role": "system", "content": "tbt-delta-anchor"}
DELTA_ANCHOR_QUERY = {"role": "user", "content": "tbt-delta-anchor-query"}


class DeltaRenderError(RuntimeError):
    """An anchored delta render did not start with the anchor's exact prefix."""


def anchored(messages: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """The message list to hand the template: the anchor pair, then the delta."""
    return [dict(DELTA_ANCHOR), dict(DELTA_ANCHOR_QUERY), *messages]


def is_anchored_delta(messages: Sequence[dict[str, Any]]) -> bool:
    """Return whether a render starts with the private delta anchor pair."""

    return list(messages[:2]) == [DELTA_ANCHOR, DELTA_ANCHOR_QUERY]


def _token_ids(rendered) -> list[int]:
    """Normalize ``apply_chat_template`` output to a flat id list.

    Newer transformers return a BatchEncoding dict from
    ``tokenize=True`` (its iteration yields KEYS — treating it as ids
    silently produces a 2-element "prefix"); batched output nests one
    conversation in a list.
    """
    if hasattr(rendered, "keys"):
        rendered = rendered["input_ids"]
    ids = list(rendered)
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return ids


def build_delta_anchor_prefix(tokenizer, **template_kwargs) -> list[int]:
    """Measure the anchor pair's rendered prefix with the rollout's kwargs.

    ``template_kwargs`` must match what the rollout passes on every render
    (verl forwards ``data.apply_chat_template_kwargs``); a mismatch is
    caught by the per-render verification in ``strip_delta_anchor``.
    """
    prefix = _token_ids(
        tokenizer.apply_chat_template(
            [dict(DELTA_ANCHOR), dict(DELTA_ANCHOR_QUERY)],
            tokenize=True,
            add_generation_prompt=False,
            **{"return_dict": False, **template_kwargs},
        )
    )
    if not prefix:
        raise ValueError(
            "delta anchor rendered to zero tokens; this template drops "
            "system and user messages, so anchored delta rendering cannot "
            "work with it"
        )
    return list(prefix)


def strip_delta_anchor(
    rendered_ids: Sequence[int], anchor_prefix: Sequence[int]
) -> list[int]:
    """Return the delta's tokens, verifying the anchor prefix token-by-token."""
    ids = list(rendered_ids)
    prefix = list(anchor_prefix)
    if ids[: len(prefix)] != prefix:
        raise DeltaRenderError(
            f"anchored delta render did not begin with the expected "
            f"{len(prefix)}-token anchor prefix: the chat template renders "
            "the anchor messages context-dependently, and appending this "
            f"delta would corrupt the trajectory. expected prefix ids: "
            f"{prefix} | got head ids: {ids[: len(prefix) + 8]}"
        )
    return ids[len(prefix) :]
