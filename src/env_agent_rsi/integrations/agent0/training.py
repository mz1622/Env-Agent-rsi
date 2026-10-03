"""生成或执行 Agent0 原生 ADPO 训练命令。

本模块只把 AppWorld 路径和较小的单机 Qwen3-4B 参数翻译成 Hydra overrides；真正的
数据集类、rollout、ADPO、FSDP、vLLM 与 checkpoint 逻辑均来自固定的 Agent0 submodule。
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

from env_agent_rsi.integrations.agent0.upstream import (
    EXECUTOR_TRAIN_ROOT,
    REPOSITORY_ROOT,
    validate_agent0_checkout,
)


DEFAULT_CONFIG = REPOSITORY_ROOT / "configs/agent0/appworld_minimal.json"


def _repository_path(value: str | Path) -> Path:
    """让 JSON 中的相对路径始终相对仓库，而不是调用者当前目录。"""

    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()


def build_training_command(config: Mapping[str, Any]) -> list[str]:
    """用上游 main_ppo 入口和配置中的 Hydra overrides 组成命令。"""

    data = dict(config["data"])
    trainer = dict(config["trainer"])
    agent = dict(config["agent"])
    model = str(config["model"])
    reward_path = REPOSITORY_ROOT / "src/env_agent_rsi/integrations/agent0/reward.py"
    stop_tokens = _repository_path(str(agent["action_stop_tokens_file"]))
    overrides = [
        "algorithm.adv_estimator=adpo",
        "+actor_rollout_ref.actor.policy_loss_fn=adpo",
        "+algorithm.min_score_for_scaling=0.3",
        "+algorithm.max_score_for_scaling=0.8",
        "+algorithm.min_advantage_scale=0.6",
        "+actor_rollout_ref.actor.max_epsilon_bonus=0.1",
        f"data.train_files={_repository_path(str(data['train_files']))}",
        f"data.val_files={_repository_path(str(data['val_files']))}",
        f"data.train_batch_size={int(data['train_batch_size'])}",
        f"data.val_batch_size={int(data['val_batch_size'])}",
        f"data.max_prompt_length={int(data['max_prompt_length'])}",
        f"data.max_response_length={int(data['max_response_length'])}",
        "data.truncation=right",
        "reward_model.reward_manager=naive",
        "reward_model.launch_reward_fn_async=True",
        f"custom_reward_function.path={reward_path}",
        "custom_reward_function.name=compute_score",
        f"actor_rollout_ref.model.path={model}",
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        "actor_rollout_ref.model.use_remove_padding=True",
        "actor_rollout_ref.model.trust_remote_code=True",
        f"actor_rollout_ref.actor.optim.lr={trainer['learning_rate']}",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={int(trainer['mini_batch_size'])}",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.actor.use_kl_loss=True",
        "actor_rollout_ref.actor.kl_loss_coef=0.01",
        "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
        "actor_rollout_ref.actor.strategy=fsdp",
        "actor_rollout_ref.actor.fsdp_config.param_offload=True",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True",
        "actor_rollout_ref.agent.enable_agent=True",
        f"actor_rollout_ref.agent.tool_server_url={agent['tool_server_url']}",
        f"actor_rollout_ref.agent.max_prompt_length={int(data['max_prompt_length'])}",
        f"actor_rollout_ref.agent.max_start_length={int(data['max_prompt_length'])}",
        f"actor_rollout_ref.agent.max_response_length={int(data['max_response_length'])}",
        f"actor_rollout_ref.agent.max_obs_length={int(agent['max_obs_length'])}",
        f"actor_rollout_ref.agent.max_turns={int(agent['max_turns'])}",
        "actor_rollout_ref.agent.additional_eos_token_ids=[151645]",
        f"actor_rollout_ref.agent.action_stop_tokens={stop_tokens}",
        "actor_rollout_ref.agent.mask_observations=True",
        "actor_rollout_ref.agent.enable_mtrl=True",
        "actor_rollout_ref.agent.mtrl_role=user",
        "actor_rollout_ref.agent.max_action_length=4096",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.mode=async",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1",
        f"actor_rollout_ref.rollout.n={int(trainer['rollouts_per_task'])}",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={trainer['gpu_memory_utilization']}",
        "actor_rollout_ref.rollout.temperature=1.0",
        "actor_rollout_ref.rollout.top_p=1.0",
        "actor_rollout_ref.rollout.top_k=-1",
        "actor_rollout_ref.rollout.max_num_seqs=64",
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.ref.fsdp_config.param_offload=True",
        f"critic.model.path={model}",
        "critic.strategy=fsdp",
        "critic.optim.lr=1e-5",
        "critic.ppo_micro_batch_size_per_gpu=1",
        "critic.model.fsdp_config.param_offload=True",
        "critic.model.fsdp_config.optimizer_offload=True",
        "algorithm.kl_ctrl.kl_coef=0.01",
        "trainer.logger=[console]",
        "trainer.project_name=env-agent-rsi",
        f"trainer.experiment_name={trainer['experiment_name']}",
        "trainer.val_before_train=True",
        "trainer.default_hdfs_dir=null",
        f"trainer.n_gpus_per_node={int(trainer['gpus'])}",
        "trainer.nnodes=1",
        f"trainer.save_freq={int(trainer['save_freq'])}",
        f"trainer.test_freq={int(trainer['test_freq'])}",
        f"trainer.total_training_steps={int(trainer['total_training_steps'])}",
    ]
    return [sys.executable, "-m", "verl_tool.trainer.main_ppo", *overrides]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Agent0 ADPO on AppWorld")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--execute", action="store_true", help="execute instead of only printing"
    )
    args = parser.parse_args()
    checkout = validate_agent0_checkout()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    stop_file = _repository_path(config["agent"]["action_stop_tokens_file"])
    stop_file.parent.mkdir(parents=True, exist_ok=True)
    stop_file.write_text("</tool_call>", encoding="utf-8")
    command = build_training_command(config)
    if not args.execute:
        print(
            json.dumps(
                {
                    "cwd": checkout["executor_train_root"],
                    "command": shlex.join(command),
                    "note": "start env-agent-rsi-agent0-server before executing",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    environment = dict(os.environ)
    source_root = str(REPOSITORY_ROOT / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        value
        for value in (
            source_root,
            str(EXECUTOR_TRAIN_ROOT),
            environment.get("PYTHONPATH", ""),
        )
        if value
    )
    subprocess.run(command, cwd=EXECUTOR_TRAIN_ROOT, env=environment, check=True)


if __name__ == "__main__":
    main()
