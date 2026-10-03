"""环境可变表面的显式清单。

MutationSurface 把 benchmark 的工具、观察通道以及允许变化的阶段公布给环境搜索器，
避免搜索器通过读取私有 handler 猜测动作空间，也避免产生 adapter 无法执行的变体。
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from env_agent_rsi.core.protocol import EnvDescriptor, JsonObject
from env_agent_rsi.evolution.mutation import (
    CONTRACT_AXES,
    ENVHARNESS_COMPONENT_TYPES,
    MUTATION_PHASES,
)

if TYPE_CHECKING:
    from env_agent_rsi.evolution.catalog import MutationCatalog
    from env_agent_rsi.evolution.mutation import MutationSpec


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
    supported_components: tuple[str, ...] = ENVHARNESS_COMPONENT_TYPES
    supported_contract_axes: tuple[str, ...] = CONTRACT_AXES
    supported_implementations: Mapping[str, Sequence[str]] = field(default_factory=dict)
    budget_dimensions: tuple[str, ...] = ("steps",)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        unknown = set(self.supported_phases) - set(MUTATION_PHASES)
        if unknown:
            raise ValueError(f"unsupported mutation phases: {sorted(unknown)!r}")
        unknown_components = set(self.supported_components) - set(
            ENVHARNESS_COMPONENT_TYPES
        )
        if unknown_components:
            raise ValueError(
                f"unsupported component types: {sorted(unknown_components)!r}"
            )
        unknown_axes = set(self.supported_contract_axes) - set(CONTRACT_AXES)
        if unknown_axes:
            raise ValueError(f"unsupported contract axes: {sorted(unknown_axes)!r}")
        unknown_implementation_phases = set(self.supported_implementations) - set(
            MUTATION_PHASES
        )
        if unknown_implementation_phases:
            raise ValueError(
                "implementation allowlist contains unknown phases: "
                f"{sorted(unknown_implementation_phases)!r}"
            )

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
            "supported_components": list(self.supported_components),
            "supported_contract_axes": list(self.supported_contract_axes),
            "supported_implementations": {
                phase: list(names)
                for phase, names in self.supported_implementations.items()
            },
            "budget_dimensions": list(self.budget_dimensions),
            "metadata": deepcopy(dict(self.metadata)),
        }

    def validate(
        self, mutation: "MutationSpec", catalog: "MutationCatalog"
    ) -> None:
        """确认变化属于 adapter 能执行的相位、实现、工具和预算边界。"""

        if mutation.phase not in self.supported_phases:
            raise ValueError(
                f"mutation phase {mutation.phase!r} is not supported by this environment"
            )
        if mutation.component_type not in self.supported_components:
            raise ValueError(
                f"component {mutation.component_type!r} is not supported by this environment"
            )
        if (
            mutation.primary_axis is not None
            and mutation.primary_axis not in self.supported_contract_axes
        ):
            raise ValueError(
                f"contract axis {mutation.primary_axis!r} is not supported"
            )
        catalog.validate_parameters(mutation)
        if self.supported_implementations:
            allowed = tuple(self.supported_implementations.get(mutation.phase, ()))
            if mutation.implementation not in allowed:
                raise ValueError(
                    f"implementation {mutation.implementation!r} is not allowed in "
                    f"phase {mutation.phase!r}; available: {list(allowed)!r}"
                )
        tool_names = {tool.name for tool in self.tools}
        parameters = dict(mutation.parameters)
        tool = parameters.get("tool")
        if tool is not None and tool not in tool_names:
            raise ValueError(f"mutation references unknown tool: {tool!r}")
        write_tools = parameters.get("write_tools", [])
        unknown_write_tools = set(write_tools) - tool_names
        if unknown_write_tools:
            raise ValueError(
                f"mutation references unknown write tools: {sorted(unknown_write_tools)!r}"
            )
        if mutation.phase == "setup" and mutation.operation != "remove":
            semantics = {tool.name: tool for tool in self.tools}
            for action in parameters.get("actions", []):
                name = str(action["tool"])
                if name not in semantics:
                    raise ValueError(f"setup references unknown tool: {name!r}")
                if not semantics[name].setup_allowed:
                    raise ValueError(f"tool {name!r} is not allowed during setup")
        if mutation.phase == "budget" and mutation.operation != "remove":
            if "max_steps" in parameters and "steps" not in self.budget_dimensions:
                raise ValueError("environment does not expose a steps budget")
            if "max_writes" in parameters and "writes" not in self.budget_dimensions:
                raise ValueError("environment does not expose a writes budget")
