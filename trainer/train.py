from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AGENT_LOOP_CONFIG = PROJECT_ROOT / "trainer" / "agent_loop.yaml"
DEFAULT_GRADER = PROJECT_ROOT / "trainer" / "core" / "tb_grader.py"
DEFAULT_DATASET = PROJECT_ROOT / "trainer" / "core" / "tb_dataset.py"


def _json_object(value: str, field_name: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError(
            f"{field_name} must be a JSON object: {error}"
        ) from error
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError(f"{field_name} must be a JSON object")
    return parsed


def _hydra_value(value: Any) -> str:
    if isinstance(value, dict):
        items = (f"{key}:{_hydra_value(item)}" for key, item in value.items())
        return "{" + ",".join(items) + "}"
    if isinstance(value, list):
        return "[" + ",".join(_hydra_value(item) for item in value) + "]"
    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(value)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch verl training with ThinkingBox rollouts and grading.",
        epilog="Unknown arguments are forwarded as native verl/Hydra overrides.",
    )
    parser.add_argument(
        "--tool-format",
        default="hermes",
        choices=["hermes", "qwen3_coder"],
        help="verl multi_turn tool-call format (hermes for Qwen3-14B; qwen3_coder for Qwen3.5/3.8 XML calls)",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("TBT_MODEL", "/path/to/model"),
        help="Hugging Face model ID or local checkpoint path.",
    )
    parser.add_argument(
        "--train-files",
        default=os.getenv("TBT_TRAIN_FILES"),
        help="Training test list (YAML of 'file.py:test_name' entries) in a "
        "ThinkingBox dataset checkout. Cases are hydrated at startup.",
    )
    parser.add_argument(
        "--val-files",
        default=os.getenv("TBT_VAL_FILES"),
        help="Validation test list. Defaults to --train-files.",
    )
    parser.add_argument(
        "--agent",
        default=os.getenv("TBT_AGENT", "think"),
        help="ThinkingBox agent config name used to hydrate the cases.",
    )
    parser.add_argument(
        "--dataset-root",
        default=os.getenv("THINKINGBOX_DATASET_ROOT"),
        help="Dataset root holding agent/, scenario/, test_case/. "
        "Derived from the test list path when omitted.",
    )
    parser.add_argument(
        "--dataset-loader",
        default=os.getenv("THINKINGBOX_DATASET_LOADER"),
        help="Public module:function loader for ThinkingBox cases.",
    )
    parser.add_argument(
        "--mcp-proxy-url",
        default=os.getenv("THINKINGBOX_MCP_PROXY_URL", "http://localhost:8000"),
    )
    parser.add_argument(
        "--judge-config-env",
        default=os.getenv("TBT_JUDGE_CONFIG_ENV", "JUDGE_CONFIG"),
        help="Environment variable containing ThinkingBox LLMSessionConfigT JSON.",
    )
    parser.add_argument("--judge-type", default="motivation")
    parser.add_argument(
        "--fixtures-config",
        default=os.getenv("TBT_FIXTURES_CONFIG", "{}"),
        help="ThinkingBox fixture configuration encoded as JSON.",
    )
    parser.add_argument(
        "--agent-loop-config",
        type=Path,
        default=DEFAULT_AGENT_LOOP_CONFIG,
    )
    parser.add_argument("--project-name", default="thinkingbox-training")
    parser.add_argument("--experiment-name", default="tbt-grpo")
    parser.add_argument(
        "--algorithm",
        choices=("grpo", "ppo"),
        default="grpo",
        help="GRPO samples groups without a critic; PPO uses GAE and a critic.",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=8,
        help="Number of responses sampled per prompt for GRPO.",
    )
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--gpus-per-node", type=int, default=8)
    parser.add_argument("--train-batch-size", type=int, default=64)
    parser.add_argument("--max-prompt-length", type=int, default=4096)
    parser.add_argument("--max-response-length", type=int, default=4096)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the generated verl arguments without starting training.",
    )
    return parser


def _build_overrides(args: argparse.Namespace) -> list[str]:
    fixtures_config = _json_object(args.fixtures_config, "--fixtures-config")

    if not args.judge_config_env.isidentifier():
        raise ValueError("--judge-config-env must name an environment variable")
    judge_config_value = os.getenv(args.judge_config_env)
    if not judge_config_value:
        raise ValueError(
            f"judge configuration environment variable is unset: "
            f"{args.judge_config_env}"
        )
    _json_object(judge_config_value, args.judge_config_env)

    if args.group_size < 2 and args.algorithm == "grpo":
        raise ValueError("GRPO requires --group-size of at least 2")

    algorithm_overrides = (
        [
            "algorithm.adv_estimator=grpo",
            f"actor_rollout_ref.rollout.n={args.group_size}",
            "critic.enable=false",
        ]
        if args.algorithm == "grpo"
        else [
            "algorithm.adv_estimator=gae",
            "actor_rollout_ref.rollout.n=1",
            "critic.enable=true",
        ]
    )

    if not args.train_files:
        raise ValueError("--train-files is required (a ThinkingBox test list YAML)")
    if not args.dataset_loader:
        raise ValueError(
            "--dataset-loader or THINKINGBOX_DATASET_LOADER is required"
        )
    # Validation always has to point somewhere; reuse the training list when the
    # caller has not split one out.
    val_files = args.val_files or args.train_files

    dataset_overrides = [
        f"data.custom_cls.path={DEFAULT_DATASET}",
        "data.custom_cls.name=ThinkingBoxDataset",
        f"+data.thinkingbox.agent={args.agent}",
        f"+data.thinkingbox.loader={args.dataset_loader}",
    ]
    if args.dataset_root:
        dataset_overrides.append(f"+data.thinkingbox.dataset_root={args.dataset_root}")

    return [
        f"actor_rollout_ref.model.path={args.model}",
        f"data.train_files={args.train_files}",
        f"data.val_files={val_files}",
        *dataset_overrides,
        f"data.train_batch_size={args.train_batch_size}",
        f"data.max_prompt_length={args.max_prompt_length}",
        f"data.max_response_length={args.max_response_length}",
        "actor_rollout_ref.rollout.name=vllm",
        f"actor_rollout_ref.rollout.multi_turn.format={args.tool_format}",
        f"actor_rollout_ref.rollout.agent.agent_loop_config_path={args.agent_loop_config.resolve()}",
        "actor_rollout_ref.rollout.agent.default_agent_loop=thinkingbox",
        "reward.reward_model.enable=false",
        "reward.reward_manager.name=naive",
        f"reward.custom_reward_function.path={DEFAULT_GRADER}",
        "reward.custom_reward_function.name=compute_score",
        "+reward.custom_reward_function.reward_kwargs.judge_config_env="
        + _hydra_value(args.judge_config_env),
        f"+reward.custom_reward_function.reward_kwargs.judge_type={args.judge_type}",
        "+reward.custom_reward_function.reward_kwargs.fixtures_config="
        + _hydra_value(fixtures_config),
        *algorithm_overrides,
        f"trainer.project_name={args.project_name}",
        f"trainer.experiment_name={args.experiment_name}",
        f"trainer.nnodes={args.nodes}",
        f"trainer.n_gpus_per_node={args.gpus_per_node}",
    ]


def main() -> None:
    parser = _parser()
    args, verl_overrides = parser.parse_known_args()
    if verl_overrides[:1] == ["--"]:
        verl_overrides = verl_overrides[1:]

    os.environ["THINKINGBOX_MCP_PROXY_URL"] = args.mcp_proxy_url
    overrides = [*_build_overrides(args), *verl_overrides]
    if args.dry_run:
        print("python -m verl.trainer.main_ppo  (0.9+; main_ppo_sync on verl 0.8)")
        print("\n".join(f"  {override}" for override in overrides))
        return

    sys.argv = [sys.argv[0], *overrides]
    try:  # verl 0.8
        from verl.trainer.main_ppo_sync import main as verl_main
    except ImportError:  # verl >=0.9: sync/async unified into the V1 trainer
        from verl.trainer.main_ppo import main as verl_main

    verl_main()


if __name__ == "__main__":
    main()
