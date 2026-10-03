"""核心交互协议设计。

本文件只定义跨模块共享的数据边界：Agent 提交 ``Action``，环境返回
``EnvResponse``，``EnvDescriptor`` 描述当前任务与工具契约。这里不依赖任何
具体场景，确保 Agent runner、环境适配器和 verifier 可以独立替换。
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable


JsonObject = dict[str, Any]


@dataclass(frozen=True)
class Action:
    """A tool call submitted by an agent."""

    tool: str
    arguments: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> JsonObject:
        return {"tool": self.tool, "arguments": dict(self.arguments)}


@dataclass
class EnvResponse:
    """One environment step.

    ``observation`` is agent-visible. ``info`` is harness-only metadata used for
    debugging, validation, and fault-trigger accounting.
    """

    observation: JsonObject
    reward: float = 0.0
    terminated: bool = False
    truncated: bool = False
    info: JsonObject = field(default_factory=dict)

    def to_dict(self) -> JsonObject:
        return asdict(self)


@dataclass(frozen=True)
class EvaluationResult:
    success: bool
    reason: str
    metrics: Mapping[str, Any]

    def to_dict(self) -> JsonObject:
        return asdict(self)


@dataclass(frozen=True)
class EnvDescriptor:
    """Agent 可见的环境契约，不包含 verifier 或真实业务状态。"""

    task_id: str
    instruction: str
    tools: tuple[JsonObject, ...]
    contract_version: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> JsonObject:
        return {
            "task_id": self.task_id,
            "task": self.instruction,
            "tools": deepcopy(list(self.tools)),
            "contract_version": self.contract_version,
            "metadata": deepcopy(dict(self.metadata)),
        }


@runtime_checkable
class ActionableEnv(Protocol):
    """Small EnvHarness-style protocol used by the MVP."""

    def describe(self) -> EnvDescriptor:
        ...

    def reset(
        self, seed: int = 0, options: Mapping[str, Any] | None = None
    ) -> EnvResponse:
        ...

    def step(self, action: Action) -> EnvResponse:
        ...

    def observe(self) -> JsonObject:
        ...

    def evaluate(self) -> EvaluationResult:
        ...

    def get_env_state(self) -> JsonObject:
        ...

    def save_state(self) -> JsonObject:
        ...

    def load_state(self, snapshot: Mapping[str, Any]) -> None:
        ...

    def notify_replay_complete(self) -> None:
        """通知环境 Setup 已结束，应保留世界状态并重置 episode 计数。"""

        ...
