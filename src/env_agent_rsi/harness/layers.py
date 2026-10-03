"""环境 Harness 的独立层绑定协议。

RuleHarness 需要非对称顺序：执行前是 Contract→f_A→Budget，执行后是 f_T→f_O→Budget。
因此本项目不用递归 decorator 猜测顺序，而以显式 pipeline 执行；每条规则仍通过
RuleLayer 成为拥有稳定身份、独立状态和加载边界的可组合层。
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, runtime_checkable

from env_agent_rsi.core.protocol import JsonObject
from env_agent_rsi.evolution.mutation import PHASE_CLASSIFICATION
from env_agent_rsi.transforms.protocols import StatefulRule


@runtime_checkable
class HarnessLayer(Protocol):
    """checkpoint 与编排器可依赖的最小独立层协议。"""

    component_type: str
    primary_axis: str | None
    execution_phase: str
    implementation: str
    ordinal: int

    def reset(self) -> None: ...

    def save_state(self) -> JsonObject: ...

    def load_state(self, state: Mapping[str, Any]) -> None: ...

    def to_snapshot(self) -> JsonObject: ...


@dataclass
class RuleLayer:
    """把现有 StatefulRule 绑定到 EnvHarness 组件分类。"""

    component_type: str
    primary_axis: str | None
    execution_phase: str
    implementation: str
    ordinal: int
    rule: StatefulRule

    def __post_init__(self) -> None:
        expected = PHASE_CLASSIFICATION.get(self.execution_phase)
        if expected != (self.component_type, self.primary_axis):
            raise ValueError(
                f"layer classification mismatch for {self.execution_phase!r}: "
                f"expected {expected!r}, got "
                f"{(self.component_type, self.primary_axis)!r}"
            )
        if self.implementation != self.rule.name:
            raise ValueError("layer implementation must match rule.name")

    def reset(self) -> None:
        self.rule.reset()

    def save_state(self) -> JsonObject:
        return deepcopy(self.rule.save_state())

    def load_state(self, state: Mapping[str, Any]) -> None:
        self.rule.load_state(state)

    def to_snapshot(self) -> JsonObject:
        return {
            "component_type": self.component_type,
            "primary_axis": self.primary_axis,
            "execution_phase": self.execution_phase,
            "implementation": self.implementation,
            "ordinal": self.ordinal,
            "state": self.save_state(),
        }

    @property
    def identity(self) -> tuple[str, str | None, str, str, int]:
        return (
            self.component_type,
            self.primary_axis,
            self.execution_phase,
            self.implementation,
            self.ordinal,
        )
