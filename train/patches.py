"""Monkey-patches applied before importing thinkingbox runners.

Three things happen here:

1. Qwen3.5's chat template enforces a single system message at index 0 and
   raises ``System message must be at the beginning`` otherwise. Thinkingbox's
   ``AgentSession`` builds a ``prefix_conversation`` of ``[msg_system, msg_bot]``
   whenever ``bot_instructions`` is non-empty, which fails this check both at
   vLLM-rollout time and at our local tokenization-render time. We merge them
   into a single system message at the boundary.

2. The encoded tool schema sent to vLLM during rollout is rendered into the
   prompt by Qwen3's chat template (because tb passes ``tools=[...]`` on the
   chat-completions call). Our trainer-side tokenization needs the same tool
   list, otherwise the policy is being trained on a *different* prompt than
   what produced its rollouts. We capture the encoded tool schema into a
   ``contextvars.ContextVar`` from inside ``AgentSession.__init__`` so each
   rollout's tools can be read by the rollout runner after ``worker.work()``
   returns.

3. vLLM's ``--reasoning-parser qwen3`` returns the model's extracted
   chain-of-thought under the key ``reasoning`` (newer OpenAI-style), but
   ``AOAISession._decode_message`` reads ``reasoning_content`` (DeepSeek-style).
   Without a remap the reasoning is silently dropped: no ``tag="think"`` message
   is ever produced, so the model's ``<think>`` trace never reaches training and
   we train a thinking model to stop thinking. We mirror ``reasoning`` ->
   ``reasoning_content`` on the raw response dict (the same object appended to
   ``self.conversation``, so both ``_decode_message`` and
   ``get_internal_conversation`` see it).

Import this module once, early, before any rollout begins.
"""
from __future__ import annotations

import contextvars
import json
import logging
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)

_PATCHED = False

# Per-asyncio-task slot for the OAI-encoded tool schema active in the most
# recently constructed AgentSession on this task. Read by train.rollout._one()
# right after worker.work() returns to attach tools to the Rollout.
current_tools: contextvars.ContextVar[list[dict] | None] = contextvars.ContextVar(
    "tb_train_current_tools", default=None
)


@runtime_checkable
class ExactTraceSession(Protocol):
    def get_generation_traces(self) -> tuple[object, ...]: ...


# The agent LLM session is task-local and remains alive through worker.work().
# RolloutRunner reads its immutable traces immediately after the worker returns.
current_trace_session: contextvars.ContextVar[ExactTraceSession | None] = (
    contextvars.ContextVar("tb_train_current_trace_session", default=None)
)
current_trace_error: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "tb_train_current_trace_error", default=None
)


def _merge_leading_systems(messages: list) -> list:
    """Collapse a leading run of role=='system' messages into one."""
    if not messages:
        return messages
    head = []
    i = 0
    while i < len(messages) and getattr(messages[i], "role", None) == "system":
        head.append(messages[i])
        i += 1
    if len(head) <= 1:
        return messages
    # Concatenate string contents with a blank line; non-string contents are
    # left as-is on the first system (rare for our flows).
    first = head[0]
    parts = []
    for m in head:
        c = m.content
        if isinstance(c, str) and c:
            parts.append(c)
    merged = first.model_copy(update={"content": "\n\n".join(parts)})
    return [merged] + list(messages[i:])


def apply() -> None:
    global _PATCHED
    if _PATCHED:
        return
    from thinkingbox.common.agent_session import AgentSession

    _orig_init = AgentSession.__init__

    def _patched_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        # Merge prefix_conversation systems
        new_msgs = _merge_leading_systems(list(self.prefix_conversation.messages))
        if len(new_msgs) != len(self.prefix_conversation.messages):
            logger.debug("merged %d leading system messages into 1",
                         len(self.prefix_conversation.messages) - len(new_msgs) + 1)
            self.prefix_conversation.messages = new_msgs
            # Re-seed the live conversation + llm state so the very first turn
            # already uses the merged prefix.
            self._reset_conversation()
        # Capture the encoded tool schema for this rollout. self.llm.tools is
        # populated by AgentSession.__init__ -> add_tools() and matches what tb
        # sends to vLLM on the wire. We deep-copy to keep ownership clear.
        try:
            tools = list(getattr(self.llm, "tools", None) or [])
            current_tools.set(tools)
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("failed to capture tools from AgentSession: %s", e)
            current_tools.set(None)
        current_trace_session.set(
            self.llm if isinstance(self.llm, ExactTraceSession) else None
        )

    AgentSession.__init__ = _patched_init

    # ---- (3) normalize the reasoning field name on the raw vLLM response ----
    # vLLM --reasoning-parser qwen3 returns the chain-of-thought under "reasoning";
    # AOAISession._decode_message reads "reasoning_content". Mirror the former onto
    # the latter (in place on the returned dict, which is the same object appended
    # to self.conversation) so the <think> trace is captured into a tag="think"
    # message and reaches training. Gated on is_reasoning so non-reasoning sessions
    # (judge/user-sim) are untouched. Idempotent: no-op if reasoning_content already set.
    from thinkingbox.common.aoai_session import AOAISession

    _orig_get_completion = AOAISession._get_completion

    async def _patched_get_completion(self, *args, **kwargs):
        try:
            msg = await _orig_get_completion(self, *args, **kwargs)
        except Exception as exc:
            if type(exc).__name__ == "ExactTokenTraceError":
                current_trace_error.set(f"{type(exc).__name__}: {exc}")
                raise

            error_text = str(exc)
            is_context_overflow = (
                getattr(self, "is_reasoning", False)
                and "maximum context length" in error_text
                and "input tokens" in error_text
            )
            if not is_context_overflow:
                raise

            last_exc = exc
            compacted = False
            conversation = getattr(self, "conversation", None)
            if isinstance(conversation, list):
                for index, item in enumerate(conversation):
                    if not isinstance(item, dict) or item.get("role") != "tool":
                        continue
                    content = item.get("content")
                    if not isinstance(content, str) or len(content) <= 8192:
                        continue
                    item["content"] = json.dumps(
                        {
                            "status": "truncated",
                            "note": "Earlier oversized tool result omitted to fit context.",
                            "original_chars": len(content),
                        },
                        separators=(",", ":"),
                    )
                    logger.warning(
                        "policy prompt exceeded context; compacted oldest oversized "
                        "tool result at message=%d original_chars=%d",
                        index,
                        len(content),
                    )
                    try:
                        msg = await _orig_get_completion(self, *args, **kwargs)
                        compacted = True
                        break
                    except Exception as retry_exc:
                        last_exc = retry_exc
                        retry_text = str(retry_exc)
                        if not (
                            "maximum context length" in retry_text
                            and "input tokens" in retry_text
                        ):
                            if type(retry_exc).__name__ == "ExactTokenTraceError":
                                current_trace_error.set(
                                    f"{type(retry_exc).__name__}: {retry_exc}"
                                )
                            raise

            if not compacted:
                original_limit = int(getattr(self, "max_completion_tokens", 4096))
                try:
                    for retry_limit in (2048, 1024, 512, 256, 128, 64):
                        if retry_limit >= original_limit:
                            continue
                        self.max_completion_tokens = retry_limit
                        logger.warning(
                            "policy prompt exceeded context at "
                            "max_completion_tokens=%d; retrying with %d",
                            original_limit,
                            retry_limit,
                        )
                        try:
                            msg = await _orig_get_completion(self, *args, **kwargs)
                            break
                        except Exception as retry_exc:
                            last_exc = retry_exc
                            retry_text = str(retry_exc)
                            if not (
                                "maximum context length" in retry_text
                                and "input tokens" in retry_text
                            ):
                                if type(retry_exc).__name__ == "ExactTokenTraceError":
                                    current_trace_error.set(
                                        f"{type(retry_exc).__name__}: {retry_exc}"
                                    )
                                raise
                    else:
                        raise last_exc
                finally:
                    self.max_completion_tokens = original_limit
        if getattr(self, "is_reasoning", False) and isinstance(msg, dict):
            if msg.get("reasoning") and not msg.get("reasoning_content"):
                msg["reasoning_content"] = msg["reasoning"]
        return msg

    AOAISession._get_completion = _patched_get_completion

    _PATCHED = True
    logger.info("AgentSession.__init__ patched: merge systems + capture tools; "
                "AOAISession._get_completion patched: context fallback + "
                "reasoning -> reasoning_content")
