"""EnvHarness 组件分类下的声明式环境变化与稳定标识。

论文层使用 Stage、Contract、Chain 三类组件；本项目再加入 Extension。执行层仍保留
六个兼容 phase，其中 Contract 被展开为可见契约、f_A、f_T、f_O，Budget 属于扩展。
这样既能与既有配置兼容，又不会再把组件类型和 Contract 内部轴误写成同一层级。
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Mapping

from env_agent_rsi.core.protocol import JsonObject


MUTATION_PHASES = (
    "setup",
    "contract",
    "action",
    "transition",
    "observation",
    "budget",
)

ENVHARNESS_COMPONENT_TYPES = ("stage", "contract", "chain", "extension")
CONTRACT_AXES = ("f_A", "f_T", "f_O")

PHASE_CLASSIFICATION: dict[str, tuple[str, str | None]] = {
    "setup": ("stage", None),
    "contract": ("contract", "f_A"),
    "action": ("contract", "f_A"),
    "transition": ("contract", "f_T"),
    "observation": ("contract", "f_O"),
    "budget": ("extension", None),
}

MUTATION_OPERATIONS = ("add", "replace", "remove")


@dataclass(frozen=True)
class MutationSpec:
    """一次最小环境变化，且只能属于一个执行阶段。"""

    phase: str
    implementation: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    rationale: str = ""
    operation: str = "add"
    target_implementation: str | None = None

    def __post_init__(self) -> None:
        if self.phase not in MUTATION_PHASES:
            raise ValueError(
                f"unknown mutation phase {self.phase!r}; expected {MUTATION_PHASES!r}"
            )
        if not self.implementation:
            raise ValueError("mutation implementation must not be empty")
        if self.operation not in MUTATION_OPERATIONS:
            raise ValueError(
                f"unknown mutation operation {self.operation!r}; "
                f"expected {MUTATION_OPERATIONS!r}"
            )
        if self.operation == "replace" and not self.target_implementation:
            raise ValueError("replace mutation requires target_implementation")
        if self.operation == "add" and self.target_implementation is not None:
            raise ValueError("add mutation requires target_implementation=null")
        if self.operation == "remove" and self.parameters:
            raise ValueError("remove mutation must not contain parameters")
        try:
            json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)
        except TypeError as exc:
            raise ValueError("mutation parameters must be JSON serializable") from exc

    @property
    def component_type(self) -> str:
        """返回论文层组件类别。"""

        return PHASE_CLASSIFICATION[self.phase][0]

    @property
    def primary_axis(self) -> str | None:
        """返回 Contract 的唯一主要轴；非 Contract 组件返回空。"""

        return PHASE_CLASSIFICATION[self.phase][1]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MutationSpec":
        parameters = value.get("parameters", {})
        if not isinstance(parameters, Mapping):
            raise ValueError("mutation parameters must be an object")
        phase = value.get("execution_phase", value.get("phase"))
        if phase is None:
            phase = _phase_from_component(
                str(value.get("component_type", "")),
                value.get("primary_axis"),
            )
        mutation = cls(
            phase=str(phase),
            implementation=str(value["implementation"]),
            parameters=deepcopy(dict(parameters)),
            rationale=str(value.get("rationale", "")),
            operation=str(value.get("operation", "add")),
            target_implementation=(
                str(value["target_implementation"])
                if value.get("target_implementation")
                else None
            ),
        )
        component_type = value.get("component_type")
        if component_type is not None and str(component_type) != mutation.component_type:
            raise ValueError(
                f"component_type {component_type!r} conflicts with "
                f"execution_phase {mutation.phase!r}"
            )
        primary_axis = value.get("primary_axis")
        if "primary_axis" in value and primary_axis != mutation.primary_axis:
            raise ValueError(
                f"primary_axis {primary_axis!r} conflicts with "
                f"execution_phase {mutation.phase!r}"
            )
        return mutation

    def to_dict(self) -> JsonObject:
        return {
            "component_type": self.component_type,
            "primary_axis": self.primary_axis,
            "execution_phase": self.phase,
            # phase 保留一个兼容周期，旧的日志与外部脚本仍能读取。
            "phase": self.phase,
            "implementation": self.implementation,
            "parameters": deepcopy(dict(self.parameters)),
            "rationale": self.rationale,
            "operation": self.operation,
            "target_implementation": self.target_implementation,
        }

    @property
    def digest(self) -> str:
        payload = json.dumps(
            self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:16]


def _phase_from_component(component_type: str, primary_axis: Any) -> str:
    """把新的组件级 JSON 映射到现有确定性执行槽。"""

    if component_type == "stage":
        return "setup"
    if component_type == "contract":
        axis = str(primary_axis or "")
        mapping = {"f_A": "contract", "f_T": "transition", "f_O": "observation"}
        if axis not in mapping:
            raise ValueError(f"contract mutation requires primary_axis in {CONTRACT_AXES!r}")
        return mapping[axis]
    if component_type == "extension":
        return "budget"
    if component_type == "chain":
        raise ValueError("chain mutations are classified but not executable in the MVP")
    raise ValueError(
        f"unknown component_type {component_type!r}; "
        f"expected {ENVHARNESS_COMPONENT_TYPES!r}"
    )
