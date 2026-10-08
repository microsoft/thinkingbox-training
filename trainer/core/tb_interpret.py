"""Eval-parity interpretation of Qwen3.5/3.8 generations.

The eval campaign's vLLM endpoint interpreted every generation with vLLM's
own parsers (``--reasoning-parser qwen3 --tool-call-parser qwen3_xml``) and
typed the parsed XML parameters against the request's tool schemas before
the harness ever saw a call. verl ships independent reimplementations of
both steps; using them made training-time conversations diverge from what
the baseline was measured with. This module reproduces the server-side
interpretation for the token-in/token-out rollout path:

1. detokenize without special tokens,
2. split reasoning from content (vLLM ``qwen3`` reasoning parser),
3. extract tool calls from the remaining content (vLLM ``qwen3_xml``),
4. coerce string-typed XML parameters to their JSON-schema types — the step
   the server performed via the real ChatCompletionRequest, and the one a
   minimal request stand-in silently loses (every parameter then arrives as
   a string, so schemas with boolean/integer fields reject every call).

The XML wire format cannot carry JSON types (QwenLM/Qwen3.6#187), so step 4
is what makes typed tools callable at all.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any


@dataclass
class ParsedToolCall:
    """A tool call as the eval server would have delivered it.

    ``arguments`` is a typed dict when the XML parsed cleanly, otherwise the
    raw string so the session layer can encode an honest error.
    """

    name: str
    arguments: dict[str, Any] | str


def _request_shim(tool_schemas: list[dict[str, Any]]):
    """Minimal request stand-in carrying the tool schemas for vLLM's parsers."""
    tools = [
        SimpleNamespace(
            type="function",
            function=SimpleNamespace(
                name=schema["function"]["name"],
                parameters=schema["function"].get("parameters"),
            ),
        )
        for schema in tool_schemas
    ]
    return SimpleNamespace(tools=tools, tool_choice="auto")


def coerce_arguments(
    arguments: dict[str, Any], input_schema: dict[str, Any] | None
) -> dict[str, Any]:
    """Coerce string-typed XML parameters to their JSON-schema types.

    Parameters declared as strings are never touched, unknown parameters pass
    through unchanged, and a failed cast keeps the raw string so the tool's
    own validation reports it.
    """
    props = (input_schema or {}).get("properties", {})
    coerced: dict[str, Any] = {}
    for key, value in arguments.items():
        declared = props.get(key, {})
        types = declared.get("type")
        if types is None and "anyOf" in declared:
            types = [m.get("type") for m in declared["anyOf"] if m.get("type")]
        if isinstance(types, str):
            types = [types]
        types = types or []
        if isinstance(value, str) and types and "string" not in types:
            text = value.strip()
            try:
                if "boolean" in types and text.lower() in ("true", "false"):
                    value = text.lower() == "true"
                elif "integer" in types:
                    value = int(text)
                elif "number" in types:
                    value = float(text)
                elif "array" in types or "object" in types:
                    value = json.loads(text)
                elif "null" in types and text.lower() in ("null", "none", ""):
                    value = None
            except (ValueError, json.JSONDecodeError):
                pass
        coerced[key] = value
    return coerced


class VllmInterpretation:
    """Server-order interpretation of a generation, one call per turn."""

    def __init__(self, tokenizer):
        from vllm.reasoning import ReasoningParserManager
        from vllm.tool_parsers import ToolParserManager

        self._tokenizer = tokenizer
        self._reasoning_parser = ReasoningParserManager.get_reasoning_parser(
            "qwen3"
        )(tokenizer)
        self._tool_parser = ToolParserManager.get_tool_parser("qwen3_xml")(tokenizer)

    def interpret(
        self, generated_ids: list[int], tool_schemas: list[dict[str, Any]]
    ) -> tuple[str, str, list[ParsedToolCall]]:
        """Return (reasoning, content, tool_calls) for one generation."""
        text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)
        shim = _request_shim(tool_schemas)
        reasoning, content = self._reasoning_parser.extract_reasoning(
            text, request=shim
        )
        info = self._tool_parser.extract_tool_calls(content or "", request=shim)
        schema_by_name = {
            s["function"]["name"]: s["function"].get("parameters")
            for s in tool_schemas
        }
        calls: list[ParsedToolCall] = []
        for call in info.tool_calls:
            function = call.function
            raw = function.arguments
            try:
                arguments = raw if isinstance(raw, dict) else json.loads(raw or "{}")
            except json.JSONDecodeError:
                calls.append(ParsedToolCall(name=function.name, arguments=raw))
                continue
            if isinstance(arguments, dict):
                arguments = coerce_arguments(
                    arguments, schema_by_name.get(function.name)
                )
            calls.append(ParsedToolCall(name=function.name, arguments=arguments))
        return (reasoning or "").strip(), (info.content or "").strip(), calls
