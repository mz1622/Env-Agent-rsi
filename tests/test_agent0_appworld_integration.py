"""验证 Agent0 × AppWorld 薄适配，不启动 GPU 训练或真实工具服务器。"""

from __future__ import annotations

import json
from pathlib import Path

from env_agent_rsi.core.protocol import EvaluationResult
from env_agent_rsi.harness.factory import available_components
from env_agent_rsi.integrations.agent0.dataset import (
    build_agent0_record,
    write_parquet_splits,
)
from env_agent_rsi.integrations.agent0.reward import (
    compute_score,
    decode_evaluation_marker,
    make_evaluation_marker,
)
from env_agent_rsi.integrations.agent0.tool import AppWorldAgent0Tool
from env_agent_rsi.integrations.agent0.training import build_training_command
from env_agent_rsi.integrations.agent0.upstream import (
    AGENT0_COMMIT,
    validate_agent0_checkout,
)


ROOT = Path(__file__).resolve().parents[1]


class FakeBackend:
    """只为工具桥单测保留的 AppWorld backend 替身。"""

    instances: list["FakeBackend"] = []

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.closed = False
        self.actions: list[tuple[str, dict[str, object]]] = []
        self.instances.append(self)

    def step(self, tool: str, arguments: dict[str, object]) -> dict[str, object]:
        self.actions.append((tool, arguments))
        return {"observation": {"ok": True}, "terminated": tool == "finish"}

    def evaluate(self) -> EvaluationResult:
        return EvaluationResult(
            success=True,
            reason="fake official evaluator",
            metrics={"passed": 2, "total": 2},
        )

    def close(self) -> None:
        self.closed = True


def test_agent0_submodule_is_pinned() -> None:
    checkout = validate_agent0_checkout()
    assert checkout["revision"] == AGENT0_COMMIT
    assert checkout["pinned"] is True


def test_only_appworld_environment_is_registered() -> None:
    assert available_components()["environments"] == ("appworld_process",)


def test_signed_reward_rejects_forgery_and_wrong_task() -> None:
    payload = {"task_id": "task-1", "success": True, "passed": 3, "total": 3}
    marker = make_evaluation_marker(payload, "secret-key")
    assert decode_evaluation_marker(marker, "secret-key") == payload
    score = compute_score(
        "appworld", marker, {"task_id": "task-1"}, {"reward_key": "secret-key"}
    )
    assert score["score"] == 1.0
    assert (
        compute_score(
            "appworld",
            marker.replace('signature="', 'signature="0'),
            {"task_id": "task-1"},
            {"reward_key": "secret-key"},
        )["score"]
        == 0.0
    )
    assert (
        compute_score(
            "appworld", marker, {"task_id": "task-2"}, {"reward_key": "secret-key"}
        )["score"]
        == 0.0
    )


def test_agent0_tool_routes_finish_and_closes_worker() -> None:
    FakeBackend.instances.clear()
    tool = AppWorldAgent0Tool(backend_factory=FakeBackend)
    action = '<tool_call>{"name":"finish","arguments":{}}</tool_call>'
    observation, done, valid = tool.conduct_action(
        "trajectory-1",
        action,
        {"task_id": "task-1", "reward_key": "secret-key"},
    )
    assert done is False
    assert valid is True
    assert isinstance(observation, dict)
    assert observation["reward"] == 1.0
    assert (
        compute_score(
            "appworld",
            observation["obs"],
            {"task_id": "task-1"},
            {"reward_key": "secret-key"},
        )["score"]
        == 1.0
    )
    backend = FakeBackend.instances[0]
    assert observation["obs"].startswith("<tool_response>")
    assert observation["obs"].endswith("</tool_response>")
    tool.delete_env("trajectory-1")
    assert backend.closed is True


def test_agent0_records_write_expected_parquet_schema(tmp_path: Path) -> None:
    records = [
        build_agent0_record(
            task_id=f"task-{index}",
            instruction=f"instruction {index}",
            split="train",
            appworld_root=tmp_path / "AppWorld",
            python_executable=tmp_path / "python",
            reward_key=f"key-{index}",
        )
        for index in range(2)
    ]
    manifest = write_parquet_splits(records, tmp_path / "output", val_count=1)
    assert Path(manifest["train_path"]).is_file()
    assert Path(manifest["validation_path"]).is_file()
    assert manifest["train_task_ids"] == ["task-0"]
    assert records[0]["reward_model"]["ground_truth"] == {"task_id": "task-0"}
    system = records[0]["prompt"][0]["content"]
    assert "[Skill Register]" in system
    assert "# Tools" in system
    assert "<tools>" in system
    assert "</tools>" in system
    assert "[Tool Register]" not in system


def test_training_command_reuses_agent0_main_ppo() -> None:
    config = json.loads(
        (ROOT / "configs/agent0/appworld_minimal.json").read_text(encoding="utf-8")
    )
    command = build_training_command(config)
    assert command[1:3] == ["-m", "verl_tool.trainer.main_ppo"]
    assert "actor_rollout_ref.model.path=Qwen/Qwen3-4B-Base" in command
    assert "actor_rollout_ref.agent.enable_mtrl=True" in command
    assert "actor_rollout_ref.agent.mtrl_role=user" in command
    assert any(value.startswith("custom_reward_function.path=") for value in command)
