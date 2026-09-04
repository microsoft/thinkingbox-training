"""Render conversations with the Qwen3 chat template and build an assistant mask.

For each assistant message we compute a (start, end) token span by re-rendering
the prefix with and without `add_generation_prompt=True`; the span we want to
train on is everything strictly after the `<|im_start|>assistant\\n` header up to
and including the closing `<|im_end|>` for that message.

This is incremental tokenization, which is O(N^2) but N is small (<32k tokens,
~50 messages) so it doesn't matter and the alternative (regex on the rendered
string) is much more fragile.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from transformers import AutoTokenizer, PreTrainedTokenizerBase


@dataclass
class TokenizedRollout:
    input_ids: list[int]          # full conversation token ids
    assistant_mask: list[int]     # 1 = train-on (assistant content + im_end), 0 = context
    assistant_spans: list[tuple[int, int]]  # (start, end_exclusive) per assistant message


def load_tokenizer(model_name_or_path: str) -> PreTrainedTokenizerBase:
    """Load HF tokenizer for Qwen3.5-9B (or compatible)."""
    return AutoTokenizer.from_pretrained(model_name_or_path)


def _tb_to_openai(messages: list[dict]) -> list[dict]:
    """Convert thinkingbox Message dicts (Text/ToolCall/ParallelToolCall/ToolResponse,
    discriminated by `T`) to OpenAI-format chat messages. Pass-through if `T` absent.

    Reasoning handling: thinkingbox emits the model's reasoning as a SEPARATE
    ``Text`` message tagged ``metadata.tag == "think"`` that precedes the same
    turn's output (``tag == "text"``) / tool-call. Left as two messages, the
    Qwen3 chat template renders each as its own assistant turn with an empty
    ``<think></think>`` and the reasoning leaking into the output position. We
    instead fold a ``think`` message into the FOLLOWING assistant message's
    ``reasoning_content`` field so the template renders the canonical
    ``<think>reasoning</think>output`` as a single turn.
    """
    import json as _json

    out: list[dict] = []
    pending_reasoning: str | None = None

    def _emit_assistant(msg: dict) -> None:
        nonlocal pending_reasoning
        if pending_reasoning is not None:
            msg["reasoning_content"] = pending_reasoning
            pending_reasoning = None
        out.append(msg)

    def _flush_standalone() -> None:
        # Reasoning with no following assistant output in the same turn (rare):
        # emit it as its own assistant message so it is never silently dropped.
        nonlocal pending_reasoning
        if pending_reasoning is not None:
            out.append({"role": "assistant", "content": "",
                        "reasoning_content": pending_reasoning})
            pending_reasoning = None

    for m in messages:
        t = m.get("T")
        if t in (None, "Message"):
            _flush_standalone()
            out.append(m)
            continue
        if t == "Text":
            if (m.get("metadata") or {}).get("tag") == "think":
                # Not its own turn: hold as the <think> block of the next message.
                rc = m.get("content", "") or ""
                pending_reasoning = rc if pending_reasoning is None else pending_reasoning + rc
                continue
            if m.get("role") == "assistant":
                _emit_assistant({"role": "assistant", "content": m.get("content", "")})
            else:
                _flush_standalone()
                out.append({"role": m["role"], "content": m.get("content", "")})
        elif t == "ToolCall":
            _emit_assistant({
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": m.get("id", ""),
                    "type": "function",
                    "function": {"name": m["name"], "arguments": _json.dumps(m.get("arguments") or {})},
                }],
            })
        elif t == "ParallelToolCall":
            tcs = []
            for tc in m.get("tool_calls", []):
                tcs.append({
                    "id": tc.get("id", ""),
                    "type": "function",
                    "function": {"name": tc["name"], "arguments": _json.dumps(tc.get("arguments") or {})},
                })
            _emit_assistant({"role": "assistant", "content": "", "tool_calls": tcs})
        elif t == "ToolResponse":
            _flush_standalone()
            out.append({
                "role": "tool",
                "tool_call_id": m.get("id", ""),
                "content": m.get("content", ""),
            })
        else:
            _flush_standalone()
            out.append(m)
    _flush_standalone()
    return out


def normalize_messages(messages: list[dict]) -> list[dict]:
    """Adapt OpenAI-style messages to what Qwen3's chat template accepts.

    - Convert thinkingbox-format Message dicts (T=Text|ToolCall|...) → OpenAI.
    - Merge consecutive system messages (template forbids two in a row).
    - Drop None-valued OpenAI extras (refusal, audio, function_call, annotations).
    - Coerce assistant content=None → "".
    - Parse `tool_calls[].function.arguments` from JSON string → dict
      (the template iterates it as a mapping).
    """
    import json as _json

    messages = _tb_to_openai(messages)
    out: list[dict] = []
    for m in messages:
        m = {k: v for k, v in m.items() if v is not None}
        if m.get("role") == "assistant":
            m.setdefault("content", "")
            tcs = m.get("tool_calls")
            if tcs:
                new_tcs = []
                for tc in tcs:
                    tc = dict(tc)
                    fn = dict(tc.get("function") or {})
                    args = fn.get("arguments")
                    if isinstance(args, str):
                        try:
                            fn["arguments"] = _json.loads(args) if args else {}
                        except _json.JSONDecodeError:
                            fn["arguments"] = {}
                    tc["function"] = fn
                    new_tcs.append(tc)
                m["tool_calls"] = new_tcs
        if (
            out
            and out[-1].get("role") == "system"
            and m.get("role") == "system"
        ):
            prev = out[-1]
            prev_c = prev.get("content") or ""
            cur_c = m.get("content") or ""
            prev["content"] = (prev_c + "\n\n" + cur_c).strip()
        else:
            out.append(m)
    return out


def _render_ids(
    tokenizer: PreTrainedTokenizerBase,
    messages: list[dict],
    add_generation_prompt: bool,
    tools: list[dict] | None = None,
) -> list[int]:
    """Render messages to a token-id list.

    We go via string (`tokenize=False`) then re-tokenize because newer
    transformers versions return a `BatchEncoding` from `apply_chat_template`
    with `tokenize=True`, which is awkward to consume.
    """
    text = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        add_generation_prompt=add_generation_prompt,
        tokenize=False,
    )
    return tokenizer.encode(text, add_special_tokens=False)


def render_with_assistant_mask(
    messages: list[dict],
    tokenizer: PreTrainedTokenizerBase,
    tools: list[dict] | None = None,
) -> TokenizedRollout:
    """Return ids + boolean mask covering every assistant turn's body+EOS tokens.

    Header tokens (`<|im_start|>assistant\\n`) are excluded — they're forced by
    the template and the model never decides to produce them, so they don't
    belong in the policy gradient.

    Implementation: render the full conversation once, then scan the token
    stream for occurrences of the `<|im_start|>assistant\\n` header and the
    matching `<|im_end|>` close. The Qwen3 chat template is *not* prefix-stable
    (it rewrites non-final assistant turns to strip empty `<think>` blocks),
    so per-prefix rendering would give inconsistent spans.
    """
    messages = normalize_messages(messages)
    full_ids = _render_ids(tokenizer, messages, add_generation_prompt=False, tools=tools)

    # Token sequences for the assistant header and im_end terminator
    header_ids = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end_id is None or im_end_id == tokenizer.unk_token_id:
        raise RuntimeError("tokenizer has no <|im_end|> token")

    mask = [0] * len(full_ids)
    spans: list[tuple[int, int]] = []
    i = 0
    while i <= len(full_ids) - len(header_ids):
        if full_ids[i : i + len(header_ids)] == header_ids:
            span_start = i + len(header_ids)
            # find next <|im_end|>
            end = span_start
            while end < len(full_ids) and full_ids[end] != im_end_id:
                end += 1
            if end >= len(full_ids):
                raise RuntimeError(
                    f"Unterminated assistant span starting at token {span_start}"
                )
            span_end = end + 1  # include the <|im_end|> token
            for t in range(span_start, span_end):
                mask[t] = 1
            spans.append((span_start, span_end))
            i = span_end
        else:
            i += 1

    n_assistant = sum(1 for m in messages if m.get("role") == "assistant")
    if len(spans) != n_assistant:
        raise RuntimeError(
            f"found {len(spans)} assistant spans in rendered output but "
            f"messages has {n_assistant} assistant turns"
        )

    return TokenizedRollout(
        input_ids=full_ids,
        assistant_mask=mask,
        assistant_spans=spans,
    )
