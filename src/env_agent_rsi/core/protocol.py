from __future__ import annotations

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


@runtime_checkable
class ActionableEnv(Protocol):
    """Small EnvHarness-style protocol used by the MVP."""

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
