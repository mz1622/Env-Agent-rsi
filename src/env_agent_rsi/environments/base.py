"""状态型本地环境的公共骨架。

子类只负责初始业务状态和工具 handler；本类统一处理 descriptor、reset、action
路由、审计、结束、快照与 verifier 调用。这样不同任务共享生命周期语义，同时不
共享业务数据结构。
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Mapping

from env_agent_rsi.core.protocol import (
    Action,
    EnvDescriptor,
    EnvResponse,
    EvaluationResult,
    JsonObject,
)
from env_agent_rsi.core.verifier import StateVerifier
from env_agent_rsi.core.tooling import (
    ActionValidationError,
    make_descriptor,
    tool_schema,
    validate_action,
)


class StatefulTaskEnv:
    """Reusable state/snapshot shell for deterministic API task environments."""

    SNAPSHOT_VERSION = 1

    def __init__(
        self,
        *,
        verifier: StateVerifier,
        task_id: str,
        instruction: str,
        tool_schemas: list[JsonObject],
    ) -> None:
        self.verifier = verifier
        self.task_id = task_id
        self.instruction = instruction
        self.tool_schemas = deepcopy(tool_schemas)
        self.reset()

    def initial_state(self, seed: int, options: Mapping[str, Any]) -> JsonObject:
        raise NotImplementedError

    def handlers(self) -> Mapping[str, Callable[[JsonObject], EnvResponse]]:
        raise NotImplementedError

    def describe(self) -> EnvDescriptor:
        return make_descriptor(
            task_id=self.task_id,
            instruction=self.instruction,
            tools=self.tool_schemas,
            metadata={"environment": type(self).__name__},
        )

    def reset(
        self, seed: int = 0, options: Mapping[str, Any] | None = None
    ) -> EnvResponse:
        self.seed = seed
        self.state = self.initial_state(seed, dict(options or {}))
        self.initial_business_state = deepcopy(self.state)
        self.audit_log: list[JsonObject] = []
        self.step_count = 0
        self.terminated = False
        descriptor = self.describe()
        return EnvResponse(
            observation={"ok": True, **descriptor.to_dict()},
            info={"event": "reset", "seed": seed},
        )

    def step(self, action: Action) -> EnvResponse:
        if self.terminated:
            return self.error("EPISODE_TERMINATED", "finish has already been called")
        try:
            validate_action(action, self.describe())
        except ActionValidationError as exc:
            return self.error(exc.code, exc.message)
        self.step_count += 1
        handler = self.handlers().get(action.tool)
        if handler is None:
            return self.error("UNKNOWN_TOOL", f"unknown tool: {action.tool}")
        return handler(dict(action.arguments))

    def finish(self, arguments: JsonObject) -> EnvResponse:
        del arguments
        self.terminated = True
        evaluation = self.evaluate()
        return EnvResponse(
            observation={
                "ok": True,
                "status": "finished",
                "task_success": evaluation.success,
                "reason": evaluation.reason,
            },
            reward=1.0 if evaluation.success else 0.0,
            terminated=True,
            info={"event": "finish", "evaluation": evaluation.to_dict()},
        )

    def success(
        self,
        observation: JsonObject,
        *,
        event: str,
        state_changed: bool = False,
    ) -> EnvResponse:
        self.audit_log.append(
            {"step": self.step_count, "event": event, "state_changed": state_changed}
        )
        return EnvResponse(
            observation={"ok": True, **deepcopy(observation)},
            info={"event": event, "state_changed": state_changed},
        )

    def error(self, code: str, message: str) -> EnvResponse:
        self.audit_log.append(
            {"step": self.step_count, "event": "error", "error_code": code}
        )
        return EnvResponse(
            observation={"ok": False, "error": {"code": code, "message": message}},
            info={"event": "error", "state_changed": False, "error_code": code},
        )

    def observe(self) -> JsonObject:
        return {
            "task_id": self.task_id,
            "state": deepcopy(self.state),
            "terminated": self.terminated,
            "step_count": self.step_count,
        }

    def evaluate(self) -> EvaluationResult:
        return self.verifier.evaluate(self.save_state())

    def get_env_state(self) -> JsonObject:
        return deepcopy(self.save_state())

    def save_state(self) -> JsonObject:
        return {
            "snapshot_version": self.SNAPSHOT_VERSION,
            "seed": self.seed,
            "task_id": self.task_id,
            "instruction": self.instruction,
            "state": deepcopy(self.state),
            "initial_business_state": deepcopy(self.initial_business_state),
            "audit_log": deepcopy(self.audit_log),
            "step_count": self.step_count,
            "terminated": self.terminated,
        }

    def load_state(self, snapshot: Mapping[str, Any]) -> None:
        if snapshot.get("snapshot_version") != self.SNAPSHOT_VERSION:
            raise ValueError(f"unsupported {type(self).__name__} snapshot version")
        if snapshot.get("task_id") != self.task_id:
            raise ValueError(
                f"task mismatch: expected {self.task_id!r}, got {snapshot.get('task_id')!r}"
            )
        self.seed = int(snapshot["seed"])
        self.instruction = str(snapshot["instruction"])
        self.state = deepcopy(snapshot["state"])
        self.initial_business_state = deepcopy(snapshot["initial_business_state"])
        self.audit_log = deepcopy(snapshot["audit_log"])
        self.step_count = int(snapshot["step_count"])
        self.terminated = bool(snapshot["terminated"])


tool = tool_schema
"""向后兼容的简写；具体 schema 构造逻辑集中在 ``core.tooling``。"""
