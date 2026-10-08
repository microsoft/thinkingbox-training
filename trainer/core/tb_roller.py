import copy
import json
import logging
import re
import time
from collections.abc import Collection, Iterable
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)

from pydantic import BaseModel, TypeAdapter
from thinkingbox.common.agent_session import AgentSession
from thinkingbox.common.agent_user_loop import run_agent_user_loop
from thinkingbox.common.chat_types import (
    Message,
    ParallelToolCall,
    Text,
    ToolCall,
    ToolDef,
    ToolResponse,
)
from thinkingbox.common.config_types import (
    HydratedTestCase,
    LLMSessionConfigT,
    SessionProxyConfig,
    merge_init_config,
)
from thinkingbox.common.llm_session_base import LLMSessionBase
from thinkingbox.common.llm_session_factory import create_llm_session
from thinkingbox.common.mcp_proxy_client import MCPProxyClient
from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopMetrics,
    AgentLoopOutput,
)
from verl.experimental.agent_loop.tool_parser import ToolParser
from verl.workers.rollout.replica import TokenOutput

from trainer.core.tb_interpret import VllmInterpretation
from trainer.core.tb_render import (
    anchored,
    build_delta_anchor_prefix,
    is_anchored_delta,
    strip_delta_anchor,
)

_THINK_BLOCK = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


def response_budget_exhausted_message() -> Text:
    """Terminate a rollout cleanly when no trainable response budget remains."""

    return Text(
        role="assistant",
        content="Generation stopped at the response token limit. <DONE>",
        metadata={"response_budget_exhausted": True, "is_done": True},
    )


def make_agent_session_factory(agent_model: LLMSessionBase):
    """Adapt a verl-backed model session to ThinkingBox's session factory API."""

    def factory(**factory_kwargs):
        return AgentSession.from_config(
            agent_model=agent_model,
            **factory_kwargs,
        )

    return factory


def strip_special_tokens(text: str, special_tokens: Iterable[str]) -> str:
    """Drop control tokens the tokenizer round-trips into decoded text.

    Eval reads assistant content from an OpenAI-compatible server, which
    detokenizes with ``skip_special_tokens=True``. verl decodes the raw
    generated ids instead, so a trailing ``<|im_end|>`` would otherwise defeat
    ThinkingBox's ``<DONE>`` end-of-turn marker. Only the text handed to
    ThinkingBox is normalized; the trained token ids are untouched.
    """
    for token in special_tokens:
        if token in text:
            text = text.replace(token, "")
    return text


def split_reasoning(content: str, think_preopened: bool = False) -> tuple[str, str]:
    """Split a Qwen-style ``<think>`` block into (reasoning, visible answer).

    Mirrors vLLM's ``--reasoning-parser qwen3``, which eval relies on to keep
    reasoning out of the simulated user's transcript and ``TestContext.response``.

    ``think_preopened`` marks templates (Qwen3.5/3.8) whose generation prompt
    already opened ``<think>``: output with no marker at all is then
    *unterminated reasoning* (length-capped mid-think), not visible text.
    Leaking it into the transcript feeds raw chain-of-thought — including
    simulated ``<|im_start|>`` turns — to the user simulator, which external
    safety filters may reject.
    """
    match = _THINK_BLOCK.search(content)
    if match is not None:
        visible = content[: match.start()] + content[match.end() :]
        return match.group(1).strip(), visible.strip()
    if _THINK_OPEN in content:
        # Generation was cut off before </think>: it is all reasoning.
        return content.split(_THINK_OPEN, 1)[1].strip(), ""
    if _THINK_CLOSE in content:
        # Output starts mid-reasoning and carries only the closing tag:
        # everything before it is reasoning.
        reasoning, visible = content.split(_THINK_CLOSE, 1)
        return reasoning.strip(), visible.strip()
    if think_preopened:
        return content.strip(), ""
    return "", content


def _encode_tool(tool: ToolDef) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema,
        },
    }


def _flatten_union_properties(tool_schema: dict[str, Any]) -> dict[str, Any]:
    """Collapse JSON-Schema union parameters into the flat form verl accepts.

    Toloka tools express optional parameters as ``anyOf: [{type: X}, {type:
    "null"}]`` with no top-level ``type``, but verl's
    OpenAIFunctionPropertySchema requires ``type``. The parser only uses these
    schemas to coerce XML argument strings, so the non-null member types carry
    all the information it needs.
    """
    tool_schema = copy.deepcopy(tool_schema)
    properties = (
        tool_schema.get("function", {}).get("parameters", {}).get("properties", {})
    )
    for prop in properties.values():
        if not isinstance(prop, dict) or "type" in prop:
            continue
        union = prop.get("anyOf") or prop.get("oneOf") or []
        types = [
            member["type"]
            for member in union
            if isinstance(member, dict)
            and isinstance(member.get("type"), str)
            and member["type"] != "null"
        ]
        if len(types) == 1:
            prop["type"] = types[0]
        elif types:
            prop["type"] = types
        else:
            prop["type"] = "string"
    return tool_schema


def _encode_message(message: Message) -> dict[str, Any]:
    if isinstance(message, Text):
        return {"role": message.role, "content": message.content}
    if isinstance(message, ParallelToolCall):
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        # Qwen3.5/3.8 chat templates iterate arguments as a
                        # mapping ("Can only get item pairs from a mapping");
                        # hermes-era templates tolerate a dict too.
                        "arguments": call.arguments
                        if isinstance(call.arguments, dict)
                        else json.loads(call.arguments),
                    },
                }
                for call in message.tool_calls
            ],
        }
    if isinstance(message, ToolCall):
        return _encode_message(ParallelToolCall(tool_calls=[message]))
    if isinstance(message, ToolResponse):
        return {
            "role": "tool",
            "name": message.name,
            "tool_call_id": message.id,
            "content": message.content,
        }
    raise TypeError(f"Unsupported ThinkingBox message: {type(message)!r}")


class VerlLLMSession(LLMSessionBase):
    """ThinkingBox LLM session backed by verl's token-in/token-out client."""

    def __init__(
        self,
        agent_loop: AgentLoopBase,
        sampling_params: dict[str, Any],
        tool_parser: ToolParser,
        response_length: int,
    ):
        self.agent_loop = agent_loop
        self.sampling_params = dict(sampling_params)
        self.tool_parser = tool_parser
        self.response_length = response_length
        # Match the OpenAI server's skip_special_tokens=True semantics: strip
        # every vocab-flagged special token, not just all_special_tokens. For
        # Qwen3.8 the turn markers (<|im_start|> etc.) are special-flagged but
        # absent from all_special_tokens; leaving them in visible text feeds
        # template markup to the user simulator, which external safety filters
        # may reject.
        decoder = getattr(agent_loop.tokenizer, "added_tokens_decoder", None) or {}
        self.special_tokens = tuple(
            {t.content for t in decoder.values() if getattr(t, "special", False)}
            | set(getattr(agent_loop.tokenizer, "all_special_tokens", ()) or ())
        )
        self.request_id = uuid4().hex
        self.tools: dict[str, ToolDef] = {}
        self.conversation: list[dict[str, Any]] = []
        self._rendered_message_count = 0
        self.prompt_ids: list[int] | None = None
        self.all_token_ids: list[int] = []
        self.response_mask: list[int] = []
        self.response_logprobs: list[float] | None = None
        self.metrics = AgentLoopMetrics()
        self.extra_fields: dict[str, Any] = {}

    @property
    def tool_names(self) -> Collection[str]:
        return self.tools.keys()

    @property
    def tool_schemas(self) -> list[dict[str, Any]]:
        return [_encode_tool(tool) for tool in self.tools.values()]

    def add_tools(self, tools: list[ToolDef]) -> None:
        self.tools.update((tool.name, tool.model_copy(deep=True)) for tool in tools)

    def reset_tools(self) -> None:
        self.tools.clear()

    def _parser_tool_schemas(self):
        # verl's Qwen3XMLToolParser needs OpenAIFunctionToolSchema objects to
        # type tool parameters; hermes ignores the argument entirely.
        from verl.tools.schemas import OpenAIFunctionToolSchema

        return [
            OpenAIFunctionToolSchema.model_validate(_flatten_union_properties(s))
            for s in self.tool_schemas
        ]

    def add_messages(self, messages: list[Message]) -> None:
        self.conversation.extend(_encode_message(message) for message in messages)
        # Qwen3.5/3.8 chat templates raise if any message after index 0 has
        # role=system; the harness prefixes system_instructions AND
        # bot_instructions as two system messages. Coalesce the leading run.
        merged = 0
        while (
            len(self.conversation) > merged + 1
            and self.conversation[merged].get("role") == "system"
            and self.conversation[merged + 1].get("role") == "system"
        ):
            self.conversation[merged]["content"] = (
                f"{self.conversation[merged]['content']}\n\n"
                f"{self.conversation[merged + 1]['content']}"
            )
            del self.conversation[merged + 1]

    def reset_messages(self) -> None:
        self.conversation.clear()
        self._rendered_message_count = 0
        self.prompt_ids = None
        self.all_token_ids.clear()
        self.response_mask.clear()
        self.response_logprobs = None

    def get_internal_conversation(self) -> list[dict[str, Any]]:
        return list(self.conversation)

    async def _append_pending_observations(self) -> None:
        if self.prompt_ids is None:
            self.prompt_ids = await self.agent_loop.apply_chat_template(
                self.conversation,
                tools=self.tool_schemas,
            )
            self.all_token_ids = list(self.prompt_ids)
            self._rendered_message_count = len(self.conversation)
            return

        pending = self.conversation[self._rendered_message_count :]
        if not pending:
            return
        # Render behind the fixed anchor (system, user) pair and strip its
        # exact, verified prefix — see tbt/core/tb_render.py for why verl's
        # remove_system_prompt=True is unsound here and why the template
        # demands a real user query in every render.
        rendered_ids = await self.agent_loop.apply_chat_template(anchored(pending))
        observation_ids = strip_delta_anchor(
            rendered_ids, self.agent_loop.delta_anchor_prefix
        )
        self.all_token_ids.extend(observation_ids)
        self.response_mask.extend([0] * len(observation_ids))
        if self.response_logprobs is not None:
            self.response_logprobs.extend([0.0] * len(observation_ids))
        self._rendered_message_count = len(self.conversation)

    async def get_completion(
        self,
        stop_tag: str | None = None,
        parallel_tool_calls: bool = False,
        conversation: list[Message] | None = None,
        update_conversation: bool = True,
    ) -> list[Message]:
        del stop_tag, parallel_tool_calls
        if conversation is not None or not update_conversation:
            raise NotImplementedError("verl rollouts require the stateful session path")

        await self._append_pending_observations()
        if len(self.response_mask) >= self.response_length:
            return [response_budget_exhausted_message()]

        sampling_params = dict(self.sampling_params)
        # Per-turn generation cap, mirroring eval serving's
        # max_completion_tokens: 8192. Bounds runaway reasoning turns (temp 1.0
        # can ramble for tens of thousands of tokens without it); the global
        # response budget still governs trajectory length.
        remaining = self.response_length - len(self.response_mask)
        sampling_params["max_tokens"] = max(1, min(8192, remaining))
        # Eval's server added no tool-call stop tokens — generation ran to the
        # model's own turn end. Skip verl's parser stops for parity.
        if self.agent_loop.vllm_interp is None and self.tool_parser.stop_token_ids:
            sampling_params["stop_token_ids"] = list(
                set(
                    (sampling_params.get("stop_token_ids") or [])
                    + self.tool_parser.stop_token_ids
                )
            )

        started = time.monotonic()
        output: TokenOutput = await self.agent_loop.server_manager.generate(
            request_id=self.request_id,
            prompt_ids=self.all_token_ids,
            sampling_params=sampling_params,
        )
        self.metrics.generate_sequences += time.monotonic() - started
        if self.metrics.num_preempted == -1:
            self.metrics.num_preempted = output.num_preempted or 0
        else:
            self.metrics.num_preempted += output.num_preempted or 0
        self.extra_fields.update(output.extra_fields)

        generated_ids = list(output.token_ids)
        self.all_token_ids.extend(generated_ids)
        self.response_mask.extend([1] * len(generated_ids))
        if output.log_probs is not None:
            if self.response_logprobs is None:
                self.response_logprobs = [0.0] * (
                    len(self.response_mask) - len(generated_ids)
                )
            self.response_logprobs.extend(output.log_probs)
        elif self.response_logprobs is not None:
            self.response_logprobs.extend([0.0] * len(generated_ids))

        if self.agent_loop.vllm_interp is not None:
            # Eval-parity interpretation incl. schema-typed arguments (see
            # tbt/core/tb_interpret.py).
            reasoning, content, parsed_calls = self.agent_loop.vllm_interp.interpret(
                generated_ids, self.tool_schemas
            )
        else:
            content, parsed_calls = await self.tool_parser.extract_tool_calls(
                generated_ids, tools=self._parser_tool_schemas()
            )
            content = strip_special_tokens(content, self.special_tokens)
            reasoning, content = split_reasoning(
                content, think_preopened=self.agent_loop.think_preopened
            )
        messages: list[Message] = []
        if reasoning:
            # tag "think" makes it is_visible=False, so ThinkingBox keeps it out
            # of _should_stop, the user-sim transcript, and TestContext.response.
            messages.append(
                Text(role="assistant", content=reasoning, metadata={"tag": "think"})
            )
        if content:
            messages.append(
                Text(role="assistant", content=content, metadata={"tag": "text"})
            )
        if parsed_calls:
            calls = []
            for parsed_call in parsed_calls:
                try:
                    raw = parsed_call.arguments
                    arguments = raw if isinstance(raw, dict) else json.loads(raw)
                    if not isinstance(arguments, dict):
                        raise TypeError("tool arguments must be a JSON object")
                    metadata = {}
                except (json.JSONDecodeError, TypeError) as error:
                    arguments = {}
                    metadata = {"error": f"Error: {error}"}
                calls.append(
                    ToolCall(
                        name=parsed_call.name,
                        arguments=arguments,
                        metadata=metadata,
                    )
                )
            messages.append(ParallelToolCall(tool_calls=calls))
        if not messages:
            messages.append(
                Text(
                    role="assistant",
                    content=self.agent_loop.tokenizer.decode(
                        generated_ids, skip_special_tokens=True
                    ),
                )
            )

        self.conversation.extend(_encode_message(message) for message in messages)
        self._rendered_message_count = len(self.conversation)
        return messages

    def get_completion_sync(self, *args, **kwargs) -> list[Message]:
        raise RuntimeError("VerlLLMSession is asynchronous")

    @classmethod
    def from_config(cls, config: BaseModel) -> "VerlLLMSession":
        del config
        raise NotImplementedError("VerlLLMSession requires verl runtime objects")

    def to_agent_loop_output(self, num_turns: int) -> AgentLoopOutput:
        if self.prompt_ids is None:
            raise RuntimeError("The rollout did not request a model completion")
        response_ids = self.all_token_ids[len(self.prompt_ids) :]
        return AgentLoopOutput(
            prompt_ids=self.prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=self.response_mask[: self.response_length],
            response_logprobs=(
                self.response_logprobs[: self.response_length]
                if self.response_logprobs is not None
                else None
            ),
            num_turns=num_turns,
            metrics=self.metrics,
            extra_fields=self.extra_fields,
        )


class ThinkingBoxAgentLoop(AgentLoopBase):
    def __init__(
        self,
        *args,
        mcp_proxy: dict[str, Any],
        user_agent: dict[str, Any] | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        rollout_format = self.rollout_config.multi_turn.format
        # hermes: Qwen3-14B JSON tool calls; qwen3_coder: the Qwen3.5/3.8 XML
        # format (<function=...><parameter=...>), parsed by verl's
        # Qwen3XMLToolParser.
        if rollout_format not in ("hermes", "qwen3_coder"):
            raise ValueError(
                "ThinkingBoxAgentLoop supports multi_turn.format hermes|qwen3_coder"
            )
        if self.processor is not None:
            # Qwen3.5/3.8 text checkpoints still resolve a multimodal
            # AutoProcessor (Qwen3VLProcessor). Our session path is purely
            # tokenizer-based, so ignore it rather than refusing to start.
            logger.warning(
                "AutoProcessor %s resolved for %s; proceeding tokenizer-only",
                type(self.processor).__name__,
                getattr(self.config, "model", "<model>"),
            )
        # Does this chat template pre-open <think> in the generation prompt
        # (Qwen3.5/3.8)? Decides how unterminated output is classified.
        template_tail = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": "x"}],
            tokenize=False,
            add_generation_prompt=True,
        )
        self.think_preopened = template_tail.rstrip().endswith(_THINK_OPEN)
        self.mcp_proxy_config = SessionProxyConfig.model_validate(mcp_proxy)
        # Optional simulated-user model, same schema as user_model in the eval
        # configs. Leave unset to keep single-turn rollouts.
        self.user_agent_config: LLMSessionConfigT | None = (
            TypeAdapter(LLMSessionConfigT).validate_python(user_agent)
            if user_agent is not None
            else None
        )
        self.tool_parser = ToolParser.get_tool_parser(rollout_format, self.tokenizer)
        self.delta_anchor_prefix = build_delta_anchor_prefix(
            self.tokenizer,
            **(getattr(self, "apply_chat_template_kwargs", None) or {}),
        )
        # For Qwen3.5/3.8, interpret model output exactly as the eval
        # campaign's vLLM endpoint did (see tbt/core/tb_interpret.py).
        self.vllm_interp = (
            VllmInterpretation(self.tokenizer)
            if rollout_format == "qwen3_coder"
            else None
        )

    async def apply_chat_template(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        images=None,
        videos=None,
        audios=None,
        mm_processor_kwargs=None,
        remove_system_prompt: bool = False,
    ):
        """Render text-only delta fragments without verl's prompt truncation."""

        if not is_anchored_delta(messages):
            return await super().apply_chat_template(
                messages,
                tools=tools,
                images=images,
                videos=videos,
                audios=audios,
                mm_processor_kwargs=mm_processor_kwargs,
                remove_system_prompt=remove_system_prompt,
            )
        if images or videos or audios:
            raise ValueError("ThinkingBox anchored deltas support text-only rollouts")

        tokenized = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                messages,
                tools=tools,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=False,
                **self.apply_chat_template_kwargs,
            ),
        )
        if hasattr(tokenized, "keys"):
            tokenized = tokenized["input_ids"]
        if hasattr(tokenized, "tolist"):
            tokenized = tokenized.tolist()
        while tokenized and isinstance(tokenized[0], list):
            if len(tokenized) != 1:
                raise ValueError("anchored render produced a batched token sequence")
            tokenized = tokenized[0]
        prompt_ids = [int(token_id) for token_id in tokenized]
        if remove_system_prompt:
            prompt_ids = prompt_ids[len(self.system_prompt) :]
        return prompt_ids

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        extra_info = kwargs.get("extra_info") or {}
        case_data = kwargs.get("thinkingbox_case") or extra_info.get("thinkingbox_case")
        if case_data is None:
            raise ValueError("Dataset row must provide extra_info.thinkingbox_case")
        # Parquet stores the case as a JSON string, since scenarios have
        # differently shaped world_states and share no single arrow schema.
        test_case = (
            HydratedTestCase.model_validate_json(case_data)
            if isinstance(case_data, str)
            else HydratedTestCase.model_validate(case_data)
        )
        if not test_case.test_code.strip():
            raise ValueError("ThinkingBox reward requires test_code in every case")
        user_model: LLMSessionBase | None = (
            create_llm_session(self.user_agent_config)
            if self.user_agent_config is not None
            else None
        )
        if test_case.user_context and user_model is None:
            raise ValueError("Simulated-user rollouts require a user_agent")
        server_config = merge_init_config(
            test_case.scenario.world_state,
            test_case.init,
        )

        session = VerlLLMSession(
            agent_loop=self,
            sampling_params=sampling_params,
            tool_parser=self.tool_parser,
            response_length=self.rollout_config.response_length,
        )
        async with MCPProxyClient.session_context_from_config(
            config=self.mcp_proxy_config,
            server_config=server_config,
            available_tools=[tool.name for tool in test_case.scenario.tools],
        ) as mcp_proxy:
            result = await run_agent_user_loop(
                test_case,
                agent_session_factory=make_agent_session_factory(session),
                mcp_proxy=mcp_proxy,
                user_model=user_model,
                store_test_context=True,
            )

        if result.is_system_error:
            error = result.metadata.get("error", {})
            raise RuntimeError(error.get("message", "ThinkingBox rollout failed"))
        if result.test_context is None:
            raise RuntimeError("ThinkingBox rollout did not produce a test context")
        session.extra_fields.update(
            tb_test_context=result.test_context.model_dump_json(),
            tb_test_code=test_case.test_code,
            tb_test_uid=test_case.uid,
        )
        return session.to_agent_loop_output(num_turns=len(result.messages))
