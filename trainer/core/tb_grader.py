from __future__ import annotations

from typing import Any

from pydantic import TypeAdapter
from thinkingbox.common.chat_types import TestContext, TestResult
from thinkingbox.common.config_types import FixtureConfig, LLMSessionConfigT
from thinkingbox.common.testrunner import TestScriptSubprocess


class ThinkingBoxGradingError(RuntimeError):
	"""Raised when ThinkingBox cannot produce a valid training reward."""


async def compute_score(
	data_source: str,
	solution_str: str,
	ground_truth: Any,
	extra_info: dict[str, Any],
	*,
	judge_config: dict[str, Any],
	judge_type: str = "motivation",
	fixtures_config: dict[str, Any] | None = None,
	system_error_score: float | None = None,
) -> dict[str, Any]:
	"""Evaluate a serialized ThinkingBox test context for verl."""
	del data_source, solution_str, ground_truth

	test_context = TestContext.model_validate_json(extra_info["tb_test_context"])
	test_code = extra_info["tb_test_code"]
	test_uid = extra_info["tb_test_uid"]
	if not isinstance(test_code, str) or not test_code.strip():
		raise ValueError("ThinkingBox reward requires non-empty tb_test_code")
	if not isinstance(test_uid, str) or not test_uid:
		raise ValueError("ThinkingBox reward requires non-empty tb_test_uid")

	validated_judge_config = TypeAdapter(LLMSessionConfigT).validate_python(
		judge_config
	)
	validated_fixtures = TypeAdapter(dict[str, FixtureConfig]).validate_python(
		fixtures_config or {}
	)
	test = TestScriptSubprocess(
		code=test_code,
		judge_config=validated_judge_config,
		judge_type=judge_type,
		fixtures_config=validated_fixtures,
		test_uid=test_uid,
	)
	result: TestResult = await test.evaluate(test_context)
	if result.is_system_error:
		# Training relies on the raise: the v1 sampler marks the group failed
		# and refills it. Validation-only runs have no refill path — a raise
		# would stall the job — so val configs set
		# custom_reward_function.reward_kwargs.system_error_score to score
		# the row instead, flagged so summaries can exclude it from clean
		# statistics.
		if system_error_score is None:
			message = result.tb.strip() or "ThinkingBox grader system error"
			raise ThinkingBoxGradingError(message)
		return {
			"score": float(system_error_score),
			"tb_test_passed": False,
			"tb_system_error": True,
			"tb_test_uid": test_uid,
			"tb_test_result_json": result.model_dump_json(),
			"tb_test_context_json": extra_info["tb_test_context"],
		}

	# Values must stay scalar or string: verl copies these keys verbatim into
	# reward_extra_info, aggregates non-strings with np.mean in its validation
	# metrics, and dumps them as the extra columns of validation_data_dir/0.jsonl.
	return {
		"score": result.reward,
		"tb_test_passed": result.result,
		"tb_system_error": False,
		"tb_test_uid": test_uid,
		"tb_test_result_json": result.model_dump_json(),
		"tb_test_context_json": extra_info["tb_test_context"],
	}
