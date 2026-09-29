"""把异构 benchmark 后端适配为统一 ActionableEnv。

后端继续拥有原生工具执行、状态和评分逻辑；BenchmarkAdapter 只规范化任务描述、动作
校验、响应 envelope、快照以及 mutation surface，供同一 Harness 和 AgentRunner 使用。
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from env_agent_rsi.core.protocol import (
    Action,
    EnvDescriptor,
    EnvResponse,
    EvaluationResult,
    JsonObject,
)
from env_agent_rsi.core.tooling import ActionValidationError, make_descriptor, validate_action
from env_agent_rsi.evolution.surface import MutationSurface


@dataclass(frozen=True)
class BenchmarkTask:
    """一个 benchmark case 对 Agent 可见的稳定描述。"""

    task_id: str
    instruction: str
    tools: tuple[JsonObject, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class BenchmarkBackend(Protocol):
    """adapter 要求原生 benchmark 暴露的最小边界。"""

    def task(self) -> BenchmarkTask: ...

    def reset(self, seed: int, options: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def step(self, tool: str, arguments: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def observe(self) -> Mapping[str, Any]: ...

    def evaluate(self) -> EvaluationResult: ...

    def save_state(self) -> Mapping[str, Any]: ...

    def load_state(self, snapshot: Mapping[str, Any]) -> None: ...

    def mutation_surface(self) -> MutationSurface: ...


class BenchmarkAdapter:
    """将符合 BenchmarkBackend 的对象变成 ActionableEnv。"""

    def __init__(self, backend: BenchmarkBackend) -> None:
        self.backend = backend

    def describe(self) -> EnvDescriptor:
        task = self.backend.task()
        return make_descriptor(
            task_id=task.task_id,
            instruction=task.instruction,
            tools=task.tools,
            metadata=task.metadata,
        )

    def reset(
        self, seed: int = 0, options: Mapping[str, Any] | None = None
    ) -> EnvResponse:
        observation = deepcopy(dict(self.backend.reset(seed, dict(options or {}))))
        observation.update(self.describe().to_dict())
        return EnvResponse(
            observation=observation,
            info={"event": "benchmark_reset", "seed": seed},
        )

    def step(self, action: Action) -> EnvResponse:
        try:
            validate_action(action, self.describe())
        except ActionValidationError as exc:
            return EnvResponse(
                observation={
                    "ok": False,
                    "error": {"code": exc.code, "message": exc.message},
                },
                info={"event": "schema_validation_blocked", "error_code": exc.code},
            )
        native = dict(self.backend.step(action.tool, dict(action.arguments)))
        if "observation" in native:
            observation = dict(native.pop("observation"))
        else:
            observation = native
            native = {}
        return EnvResponse(
            observation=deepcopy(observation),
            reward=float(native.get("reward", 0.0)),
            terminated=bool(native.get("terminated", False)),
            truncated=bool(native.get("truncated", False)),
            info=deepcopy(dict(native.get("info", {}))),
        )

    def observe(self) -> JsonObject:
        return deepcopy(dict(self.backend.observe()))

    def evaluate(self) -> EvaluationResult:
        return self.backend.evaluate()

    def get_env_state(self) -> JsonObject:
        return deepcopy(dict(self.backend.save_state()))

    def save_state(self) -> JsonObject:
        return deepcopy(dict(self.backend.save_state()))

    def load_state(self, snapshot: Mapping[str, Any]) -> None:
        self.backend.load_state(deepcopy(dict(snapshot)))

    def mutation_surface(self) -> MutationSurface:
        return self.backend.mutation_surface()
