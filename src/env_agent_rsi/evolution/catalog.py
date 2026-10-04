"""可执行环境变化的实现目录与参数 schema。

目录是 LLM 输出和 Harness 注册实现之间的白名单：每个实现声明所属相位、必填与
可选参数及参数类型。诊断或修改 Agent 不能通过自由文本发明未注册的环境能力。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from env_agent_rsi.core.protocol import JsonObject
from env_agent_rsi.evolution.mutation import MUTATION_PHASES, MutationSpec


PARAMETER_TYPES = {
    "string",
    "positive_integer",
    "non_negative_integer",
    "string_list",
    "action_list",
}


@dataclass(frozen=True)
class MutationImplementation:
    """一个已注册变化实现及其可接受参数。"""

    phase: str
    name: str
    required: Mapping[str, str]
    optional: Mapping[str, str]
    description: str = ""

    def __post_init__(self) -> None:
        if self.phase not in MUTATION_PHASES:
            raise ValueError(f"unknown mutation phase: {self.phase}")
        unknown_types = (
            set(self.required.values()) | set(self.optional.values())
        ) - PARAMETER_TYPES
        if unknown_types:
            raise ValueError(f"unknown parameter types: {sorted(unknown_types)!r}")
        overlap = set(self.required) & set(self.optional)
        if overlap:
            raise ValueError(f"parameters cannot be required and optional: {overlap!r}")

    def to_dict(self) -> JsonObject:
        probe = MutationSpec(self.phase, self.name)
        return {
            "component_type": probe.component_type,
            "primary_axis": probe.primary_axis,
            "execution_phase": self.phase,
            "phase": self.phase,
            "implementation": self.name,
            "required_parameters": dict(self.required),
            "optional_parameters": dict(self.optional),
            "description": self.description,
        }


class MutationCatalog:
    """按相位和实现名解析、校验变化实现。"""

    def __init__(self) -> None:
        self._items: dict[tuple[str, str], MutationImplementation] = {}

    def register(self, item: MutationImplementation) -> None:
        key = (item.phase, item.name)
        if key in self._items:
            raise ValueError(f"mutation implementation already registered: {key!r}")
        self._items[key] = item

    def resolve(self, phase: str, name: str) -> MutationImplementation:
        try:
            return self._items[(phase, name)]
        except KeyError as exc:
            available = sorted(
                item.name for item in self._items.values() if item.phase == phase
            )
            raise ValueError(
                f"unknown {phase} mutation implementation {name!r}; "
                f"available: {available!r}"
            ) from exc

    def validate_parameters(self, mutation: MutationSpec) -> None:
        item = self.resolve(mutation.phase, mutation.implementation)
        if mutation.operation == "remove":
            return
        parameters = dict(mutation.parameters)
        missing = set(item.required) - set(parameters)
        unknown = set(parameters) - set(item.required) - set(item.optional)
        if missing:
            raise ValueError(
                f"missing parameters for {mutation.implementation}: {sorted(missing)!r}"
            )
        if unknown:
            raise ValueError(
                f"unknown parameters for {mutation.implementation}: {sorted(unknown)!r}"
            )
        for name, kind in {**item.required, **item.optional}.items():
            if name in parameters:
                _validate_parameter(name, parameters[name], kind)

    def to_prompt_value(
        self,
        *,
        phases: tuple[str, ...] | None = None,
        implementations: Mapping[str, Any] | None = None,
    ) -> list[JsonObject]:
        allowed = set(phases) if phases is not None else None
        return [
            self._items[key].to_dict()
            for key in sorted(self._items)
            if allowed is None or self._items[key].phase in allowed
            if implementations is None
            or not implementations
            or self._items[key].name
            in set(implementations.get(self._items[key].phase, ()))
        ]


def default_mutation_catalog() -> MutationCatalog:
    """返回与 Harness 内置注册实现一致的默认白名单。"""

    catalog = MutationCatalog()
    catalog.register(
        MutationImplementation(
            "setup",
            "replay",
            {"actions": "action_list"},
            {},
            "在 reset 后重放合法动作以构造可达初始状态。",
        )
    )
    for phase in ("contract", "action"):
        catalog.register(
            MutationImplementation(
                phase,
                "require_argument",
                {"tool": "string", "argument": "string"},
                {"error_code": "string", "message": "string"},
                "在契约或动作接收阶段要求指定参数。",
            )
        )
    catalog.register(
        MutationImplementation(
            "contract",
            "add_tool_guidance",
            {"tool": "string", "guidance": "string"},
            {},
            (
                "向一个现有工具的 Agent 可见说明追加简短操作引导；不得包含任务答案，"
                "也不改变参数、执行语义或 verifier。"
            ),
        )
    )
    catalog.register(
        MutationImplementation(
            "transition",
            "post_commit_timeout",
            {},
            {"tool": "string", "trigger_on_nth": "positive_integer"},
            "在指定提交后模拟一次超时。",
        )
    )
    catalog.register(
        MutationImplementation(
            "observation",
            "stale_read_after_write",
            {},
            {"stale_reads": "positive_integer"},
            "在写入后模拟有限次数的陈旧读取。",
        )
    )
    catalog.register(
        MutationImplementation(
            "observation",
            "stale_field_after_action",
            {
                "trigger_tool": "string",
                "state_key": "string",
                "observation_key": "string",
            },
            {"stale_reads": "positive_integer"},
            "在指定有状态动作后有限次数返回动作前的字段值。",
        )
    )
    catalog.register(
        MutationImplementation(
            "budget",
            "step_budget",
            {"max_steps": "positive_integer"},
            {
                "max_writes": "non_negative_integer",
                "write_tools": "string_list",
            },
            "限制 episode 步数和写调用次数。",
        )
    )
    return catalog


def _validate_parameter(name: str, value: Any, kind: str) -> None:
    if kind == "string":
        valid = isinstance(value, str) and bool(value.strip())
    elif kind == "positive_integer":
        valid = isinstance(value, int) and not isinstance(value, bool) and value > 0
    elif kind == "non_negative_integer":
        valid = isinstance(value, int) and not isinstance(value, bool) and value >= 0
    elif kind == "string_list":
        valid = isinstance(value, list) and all(
            isinstance(item, str) and bool(item) for item in value
        )
    elif kind == "action_list":
        valid = isinstance(value, list) and all(
            isinstance(item, Mapping)
            and isinstance(item.get("tool"), str)
            and bool(item.get("tool"))
            and isinstance(item.get("arguments", {}), Mapping)
            for item in value
        )
    else:
        valid = False
    if not valid:
        raise ValueError(f"parameter {name!r} must have type {kind}")
