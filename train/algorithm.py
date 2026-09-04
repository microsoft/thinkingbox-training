"""Pure helpers for selecting and normalizing verl trainer objectives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

Algorithm = Literal["dapo", "grpo"]
LossAggregation = Literal["token-mean", "rollout-mean"]
MIN_GRPO_COMPLETE_GROUPS = 2


@dataclass(frozen=True)
class AlgorithmDefaults:
    clip_low: float
    clip_high: float
    clip_c: float
    overlong_penalty: float
    dynamic_sampling_max_rounds: int
    loss_agg_mode: LossAggregation


ALGORITHM_DEFAULTS: dict[Algorithm, AlgorithmDefaults] = {
    "dapo": AlgorithmDefaults(
        clip_low=0.2,
        clip_high=0.28,
        clip_c=10.0,
        overlong_penalty=1.0,
        dynamic_sampling_max_rounds=1,
        loss_agg_mode="token-mean",
    ),
    "grpo": AlgorithmDefaults(
        clip_low=0.2,
        clip_high=0.2,
        clip_c=1e9,
        overlong_penalty=0.0,
        dynamic_sampling_max_rounds=0,
        loss_agg_mode="token-mean",
    ),
}


def resolve_algorithm_defaults(args: Any) -> None:
    """Fill algorithm-dependent options that the user did not set explicitly."""
    try:
        defaults = ALGORITHM_DEFAULTS[args.algo]
    except KeyError as exc:
        raise ValueError(f"unsupported algorithm: {args.algo!r}") from exc
    for name in (
        "clip_low",
        "clip_high",
        "clip_c",
        "overlong_penalty",
        "dynamic_sampling_max_rounds",
        "loss_agg_mode",
    ):
        if getattr(args, name) is None:
            setattr(args, name, getattr(defaults, name))


def effective_algorithm_settings(args: Any) -> dict[str, object]:
    """Return the exact objective settings recorded in logs and metrics."""
    return {
        "algo": args.algo,
        "clip_low": float(args.clip_low),
        "clip_high": float(args.clip_high),
        "clip_c": float(args.clip_c),
        "overlong_penalty": float(args.overlong_penalty),
        "dynamic_sampling_max_rounds": int(
            args.dynamic_sampling_max_rounds
        ),
        "norm_adv_by_std_in_grpo": args.algo == "grpo",
        "loss_agg_mode": args.loss_agg_mode,
        "filter_zero_variance_groups": args.algo == "dapo",
        "require_complete_groups": args.algo == "grpo",
        "prompt_repeats_per_task": int(
            getattr(args, "prompt_repeats_per_task", 1)
        ),
        "kl_coef": 0.0,
    }


def validate_training_args(args: Any) -> None:
    """Validate resume and clean adapter-initialization semantics."""
    if args.start_step < 0 or args.start_step > args.max_steps:
        raise ValueError("--start-step must satisfy 0 <= start-step <= max-steps")
    if args.resume_lora_dir and args.init_lora_dir:
        raise ValueError(
            "--resume-lora-dir and --init-lora-dir are mutually exclusive"
        )
    if args.init_lora_dir and args.start_step != 0:
        raise ValueError("--init-lora-dir requires --start-step 0")
    # A cursor can advance the absolute schedule through a legitimate
    # no-update GRPO step while preserving the true base policy/fresh AdamW.
    # The driver validates that explicit (0, 0) recovery pair immediately
    # after this generic argument validation.
    explicit_base_recovery = (
        args.start_step > 0
        and getattr(args, "recovery_source_checkpoint_step", None) == 0
        and getattr(args, "recovery_optimizer_checkpoint_step", None) == 0
    )
    if (
        (args.start_step > 0) != bool(args.resume_lora_dir)
        and not explicit_base_recovery
    ):
        raise ValueError(
            "--start-step > 0 and --resume-lora-dir must be provided together"
        )
    if args.init_lora_dir and args.require_optimizer_resume:
        raise ValueError(
            "--require-optimizer-resume cannot be used with --init-lora-dir"
        )
    if (getattr(args, "algo", "dapo") == "dapo"
            and getattr(args, "loss_agg_mode", "token-mean") != "token-mean"):
        raise ValueError(
            "DAPO requires --loss-agg-mode token-mean"
        )
    if (getattr(args, "algo", "dapo") == "grpo"
            and getattr(args, "g", 1) < 2):
        raise ValueError("GRPO requires --g >= 2")
    if getattr(args, "max_agent_turns", 0) < 0:
        raise ValueError("--max-agent-turns must be non-negative")
    if getattr(args, "eval_max_agent_turns", 0) < 0:
        raise ValueError("--eval-max-agent-turns must be non-negative")
    if getattr(args, "eval_timeout", 1500.0) <= 0:
        raise ValueError("--eval-timeout must be positive")
    prompt_repeats = int(getattr(args, "prompt_repeats_per_task", 1))
    if prompt_repeats < 1:
        raise ValueError("--prompt-repeats-per-task must be >= 1")
    if (prompt_repeats > 1
            and getattr(args, "algo", "dapo") != "grpo"):
        raise ValueError(
            "--prompt-repeats-per-task > 1 is available only with --algo grpo"
        )
    if getattr(args, "n_prompts", 1) < 1:
        raise ValueError("--n-prompts must be >= 1")
    if getattr(args, "concurrency", 1) < 1:
        raise ValueError("--concurrency must be >= 1")
    if getattr(args, "eval_metrics_out", None) and (
            not getattr(args, "eval_list", None)
            or getattr(args, "eval_every", 0) < 1):
        raise ValueError(
            "--eval-metrics-out requires --eval-list and --eval-every >= 1"
        )
    if (getattr(args, "algo", "dapo") == "grpo"
            and getattr(args, "g", 2) < 2):
        raise ValueError("--algo grpo requires --g >= 2")


def build_prompt_group_pool(
    train_cases: Sequence[Any],
    repeats_per_task: int = 1,
) -> dict[str, Any]:
    """Build independently grouped prompt replicas without copying trajectories.

    GRPO compares ``g`` stochastic trajectories for one prompt group. A
    one-task overfit experiment still needs multiple independent groups per
    update, so this helper exposes virtual group IDs that all point to the same
    hydrated task. The rollout itself remains fresh for every group/sample.
    """
    if repeats_per_task < 1:
        raise ValueError("repeats_per_task must be >= 1")
    if repeats_per_task == 1:
        return {str(case.uid): case for case in train_cases}

    groups: dict[str, Any] = {}
    for case in train_cases:
        source_uid = str(case.uid)
        for repeat_idx in range(repeats_per_task):
            group_id = (
                f"{source_uid}::grpo_group_{repeat_idx:04d}"
            )
            groups[group_id] = case
    return groups


def cap_training_agent_turns(
    train_cases: Sequence[Any],
    max_agent_turns: int,
) -> int:
    """Apply a training-only upper bound while preserving stricter case limits."""
    if max_agent_turns < 0:
        raise ValueError("max_agent_turns must be non-negative")
    if max_agent_turns == 0:
        return 0
    changed = 0
    for case in train_cases:
        current = int(case.max_agent_sim_turns)
        capped = min(current, max_agent_turns)
        if capped != current:
            case.max_agent_sim_turns = capped
            changed += 1
    return changed


def rollout_response_token_totals(
    segment_response_tokens: Sequence[float],
    rollout_ids: Sequence[object],
) -> list[float]:
    """Return each segment's parent rollout response-token total."""
    if len(segment_response_tokens) != len(rollout_ids):
        raise ValueError("response-token counts and rollout IDs must align")
    totals: dict[str, float] = {}
    normalized_ids = [str(rollout_id) for rollout_id in rollout_ids]
    for tokens, rollout_id in zip(segment_response_tokens, normalized_ids):
        value = float(tokens)
        if value < 0:
            raise ValueError("response-token counts must be non-negative")
        totals[rollout_id] = totals.get(rollout_id, 0.0) + value
    return [totals[rollout_id] for rollout_id in normalized_ids]


def filter_complete_rollout_groups(
    groups: Mapping[str, Sequence[dict[str, Any]]],
    candidate_group_ids: Sequence[object],
    expected_g: int,
    drop_incomplete: bool,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    """Classify groups by distinct trainable rollouts and optionally censor them."""
    if expected_g <= 0:
        raise ValueError("expected_g must be positive")
    candidate_ids = list(dict.fromkeys(
        str(group_id) for group_id in candidate_group_ids))
    for group_id in groups:
        if str(group_id) not in candidate_ids:
            candidate_ids.append(str(group_id))

    kept: dict[str, list[dict[str, Any]]] = {}
    complete_groups = 0
    informative_groups = 0
    incomplete_groups = 0
    incomplete_samples = 0
    for group_id in candidate_ids:
        rows = list(groups.get(group_id, ()))
        rollout_rewards: dict[str, float] = {}
        for row in rows:
            if not bool(row.get("is_system_error", False)):
                rollout_rewards.setdefault(
                    str(row["rollout_id"]), float(row.get("reward", 0.0)))
        rollout_ids = set(rollout_rewards)
        is_complete = len(rollout_ids) == expected_g
        if (len(rollout_rewards) >= 2
                and len(set(rollout_rewards.values())) > 1):
            informative_groups += 1
        if is_complete:
            complete_groups += 1
        else:
            incomplete_groups += 1
            incomplete_samples += max(0, expected_g - len(rollout_ids))
        if rows and (is_complete or not drop_incomplete):
            kept[group_id] = rows
    return kept, {
        "complete_groups": complete_groups,
        "informative_groups": informative_groups,
        "incomplete_groups": incomplete_groups,
        "incomplete_samples": incomplete_samples,
    }


def has_sufficient_complete_groups(
    algo: Algorithm,
    complete_groups: int,
    minimum: int = MIN_GRPO_COMPLETE_GROUPS,
) -> bool:
    """DAPO is permissive; GRPO requires enough complete groups to update."""
    if minimum < 1:
        raise ValueError("minimum complete groups must be positive")
    return algo != "grpo" or complete_groups >= minimum


def is_clean_rollout_batch(
    *,
    sys_errors: int,
    unfinished: int,
    incomplete_groups: int,
    incomplete_samples: int,
    timed_out_samples: int,
) -> bool:
    """Return whether every scheduled rollout completed without infra loss."""
    return all(
        value == 0
        for value in (
            sys_errors,
            unfinished,
            incomplete_groups,
            incomplete_samples,
            timed_out_samples,
        )
    )


def is_timeout_censored_rollout_batch(
    *,
    sys_errors: int,
    unfinished: int,
    incomplete_samples: int,
    timed_out_samples: int,
    complete_groups: int,
    minimum_complete_groups: int,
) -> bool:
    """Accept only timeout-censored samples while rejecting other infra loss."""
    if minimum_complete_groups < 1:
        raise ValueError("minimum complete groups must be positive")
    return (
        complete_groups >= minimum_complete_groups
        and sys_errors == timed_out_samples
        and unfinished == timed_out_samples
        and incomplete_samples == timed_out_samples
    )


def step_optimizer_if_advantage(
    optimizer: Any,
    has_nonzero_global_advantage: bool,
) -> bool:
    """Avoid residual optimizer-momentum drift on a zero-advantage batch."""
    if not has_nonzero_global_advantage:
        return False
    optimizer.step()
    return True


def policy_loss_weight(
    loss_agg_mode: LossAggregation,
    segment_response_tokens: float,
    global_response_tokens: float,
    rollout_response_tokens: float,
    global_rollout_count: int,
) -> float:
    """Weight a segment token-mean into the selected global objective."""
    segment_tokens = float(segment_response_tokens)
    if segment_tokens <= 0:
        return 0.0
    if loss_agg_mode == "token-mean":
        return segment_tokens / max(float(global_response_tokens), 1.0)
    if loss_agg_mode == "rollout-mean":
        rollout_tokens = float(rollout_response_tokens)
        if rollout_tokens <= 0 or global_rollout_count <= 0:
            raise ValueError(
                "rollout-mean weighting requires positive rollout tokens and count"
            )
        return (
            segment_tokens
            / rollout_tokens
            / float(global_rollout_count)
        )
    raise ValueError(
        f"unsupported loss aggregation mode: {loss_agg_mode!r}")
