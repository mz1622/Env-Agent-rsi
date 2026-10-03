"""把单一 MutationSpec 物化为新的完整环境 JSON spec。

物化器把旧 rules 无损升级为 schema v2 components，并且只修改组件列表；任务、
verifier 与基础环境保持不变。所有变化在写入前经过 Surface 和目录白名单校验。
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from env_agent_rsi.evolution.catalog import MutationCatalog
from env_agent_rsi.evolution.mutation import (
    MUTATION_PHASES,
    PHASE_CLASSIFICATION,
    MutationSpec,
)
from env_agent_rsi.evolution.surface import MutationSurface


def materialize_environment_spec(
    parent_spec: Mapping[str, Any],
    mutation: MutationSpec,
    *,
    surface: MutationSurface,
    catalog: MutationCatalog,
) -> dict[str, Any]:
    """返回应用一次 add/replace/remove 后的独立子环境配置。"""

    surface.validate(mutation, catalog)
    child = canonicalize_environment_spec(parent_spec)
    components = child["components"]
    assert isinstance(components, list)

    if mutation.operation == "add":
        components.append(_rule_value(mutation))
        return child

    target = mutation.target_implementation or mutation.implementation
    matches = [
        index
        for index, value in enumerate(components)
        if isinstance(value, Mapping)
        and value.get("type") == target
        and value.get("execution_phase") == mutation.phase
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{mutation.operation} requires exactly one {target!r} rule "
            f"in phase {mutation.phase!r}; found {len(matches)}"
        )
    index = matches[0]
    if mutation.operation == "remove":
        del components[index]
    else:
        components[index] = _rule_value(mutation)
    return child


def _rule_value(mutation: MutationSpec) -> dict[str, Any]:
    return {
        "component_type": mutation.component_type,
        "primary_axis": mutation.primary_axis,
        "execution_phase": mutation.phase,
        "type": mutation.implementation,
        **deepcopy(dict(mutation.parameters)),
    }


def canonicalize_environment_spec(
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    """把旧 ``rules`` 配置无损提升为组件级 ``components`` 配置。"""

    result = deepcopy(dict(spec))
    if "components" in result:
        components = result["components"]
        if not isinstance(components, list):
            raise ValueError("environment components must be a list")
        for index, component in enumerate(components):
            if not isinstance(component, Mapping):
                raise ValueError(f"environment component {index} must be an object")
        result["schema_version"] = 2
        return result

    raw_rules = result.pop("rules", {})
    if not isinstance(raw_rules, Mapping):
        raise ValueError("environment rules must be an object")
    components: list[dict[str, Any]] = []
    for phase in MUTATION_PHASES:
        entries = raw_rules.get(phase, [])
        if not isinstance(entries, list):
            raise ValueError(f"environment rule group {phase!r} must be a list")
        component_type, primary_axis = PHASE_CLASSIFICATION[phase]
        for entry in entries:
            if not isinstance(entry, Mapping):
                raise ValueError(f"environment rule in {phase!r} must be an object")
            components.append(
                {
                    "component_type": component_type,
                    "primary_axis": primary_axis,
                    "execution_phase": phase,
                    **deepcopy(dict(entry)),
                }
            )
    result["components"] = components
    result["schema_version"] = 2
    return result
