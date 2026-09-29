"""环境可变表面的显式清单。

MutationSurface 把 benchmark 的工具、观察通道以及允许变化的阶段公布给环境搜索器，
避免搜索器通过读取私有 handler 猜测动作空间，也避免产生 adapter 无法执行的变体。
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from env_agent_rsi.core.protocol import EnvDescriptor, JsonObject
from env_agent_rsi.evolution.mutation import MUTATION_PHASES


@dataclass(frozen=True)
class ToolSemantics:
    """一个工具可否被 Setup 使用以及其副作用类别。"""

    name: str
    side_effect: str = "unknown"
    setup_allowed: bool = False
    tags: tuple[str, ...] = ()

    def to_dict(self) -> JsonObject:
        return {
            "name": self.name,
            "side_effect": self.side_effect,
            "setup_allowed": self.setup_allowed,
            "tags": list(self.tags),
        }


@dataclass(frozen=True)
class ObservationChannel:
    """一个可被 f_O 改写的观察通道。"""

    name: str
    mutable: bool = True
    description: str = ""

    def to_dict(self) -> JsonObject:
        return {
            "name": self.name,
            "mutable": self.mutable,
            "description": self.description,
        }


@dataclass(frozen=True)
class MutationSurface:
    """一个 adapter 明确承诺支持的环境变化边界。"""

    tools: tuple[ToolSemantics, ...]
    observations: tuple[ObservationChannel, ...] = ()
    supported_phases: tuple[str, ...] = MUTATION_PHASES
    budget_dimensions: tuple[str, ...] = ("steps",)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        unknown = set(self.supported_phases) - set(MUTATION_PHASES)
        if unknown:
            raise ValueError(f"unsupported mutation phases: {sorted(unknown)!r}")

    @classmethod
    def from_descriptor(
        cls,
        descriptor: EnvDescriptor,
        *,
        tool_semantics: Mapping[str, Mapping[str, Any]] | None = None,
        observation_channels: Sequence[str] = (),
    ) -> "MutationSurface":
        semantics = tool_semantics or {}
        tools: list[ToolSemantics] = []
        for schema in descriptor.tools:
            name = str(schema.get("function", {}).get("name", ""))
            if not name:
                continue
            options = semantics.get(name, {})
            tools.append(
                ToolSemantics(
                    name=name,
                    side_effect=str(options.get("side_effect", "unknown")),
                    setup_allowed=bool(options.get("setup_allowed", False)),
                    tags=tuple(str(tag) for tag in options.get("tags", ())),
                )
            )
        return cls(
            tools=tuple(tools),
            observations=tuple(
                ObservationChannel(name=str(name)) for name in observation_channels
            ),
            metadata={"task_id": descriptor.task_id},
        )

    def to_dict(self) -> JsonObject:
        return {
            "tools": [tool.to_dict() for tool in self.tools],
            "observations": [channel.to_dict() for channel in self.observations],
            "supported_phases": list(self.supported_phases),
            "budget_dimensions": list(self.budget_dimensions),
            "metadata": deepcopy(dict(self.metadata)),
        }
