"""AppWorld 配置到环境 Harness 的装配工厂。

仓库只保留 AppWorld 数据接口。六类环境变化仍由独立注册表装配，因此诊断与修改
模块不依赖具体 benchmark；AppWorld 官方 evaluator 则留在进程隔离的 backend 内。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from env_agent_rsi.core.protocol import Action, ActionableEnv
from env_agent_rsi.core.registry import Builder, ComponentRegistry
from env_agent_rsi.benchmarks import AppWorldProcessBackend, BenchmarkAdapter
from env_agent_rsi.evolution.materializer import canonicalize_environment_spec
from env_agent_rsi.evolution.mutation import PHASE_CLASSIFICATION
from env_agent_rsi.harness.wrapper import RuleHarness
from env_agent_rsi.transforms import (
    ActionRule,
    BudgetRule,
    ContractRule,
    ObservationRule,
    PostCommitTimeoutRule,
    ReplaySetupRule,
    RequireArgumentContractRule,
    RequireArgumentRule,
    SetupRule,
    StaleFieldAfterActionRule,
    StaleReadAfterWriteRule,
    StepBudgetRule,
    TransitionRule,
)


ENVIRONMENTS = ComponentRegistry[ActionableEnv]("environment")
SETUP_RULES = ComponentRegistry[SetupRule]("setup rule")
ACTION_RULES = ComponentRegistry[ActionRule]("action rule")
CONTRACT_RULES = ComponentRegistry[ContractRule]("contract rule")
TRANSITION_RULES = ComponentRegistry[TransitionRule]("transition rule")
OBSERVATION_RULES = ComponentRegistry[ObservationRule]("observation rule")
BUDGET_RULES = ComponentRegistry[BudgetRule]("budget rule")


def register_environment(name: str, builder: Builder[ActionableEnv]) -> None:
    ENVIRONMENTS.register(name, builder)


def register_setup_rule(name: str, builder: Builder[SetupRule]) -> None:
    SETUP_RULES.register(name, builder)


def register_action_rule(name: str, builder: Builder[ActionRule]) -> None:
    ACTION_RULES.register(name, builder)


def register_contract_rule(name: str, builder: Builder[ContractRule]) -> None:
    CONTRACT_RULES.register(name, builder)


def register_transition_rule(name: str, builder: Builder[TransitionRule]) -> None:
    TRANSITION_RULES.register(name, builder)


def register_observation_rule(name: str, builder: Builder[ObservationRule]) -> None:
    OBSERVATION_RULES.register(name, builder)


def register_budget_rule(name: str, builder: Builder[BudgetRule]) -> None:
    BUDGET_RULES.register(name, builder)


def load_spec(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        spec = json.load(handle)
    if not isinstance(spec, dict):
        raise ValueError("environment spec must be a JSON object")
    return spec


def build_environment(spec: Mapping[str, Any]) -> RuleHarness:
    environment_spec = dict(spec.get("environment", {}))
    environment_type = str(environment_spec.get("type", "appworld_process"))
    base = ENVIRONMENTS.build(environment_type, spec)

    rules = _rules_by_execution_phase(spec)
    setup_rules = _build_rules(SETUP_RULES, rules.get("setup", []))
    contract_rules = _build_rules(CONTRACT_RULES, rules.get("contract", []))
    action_rules = _build_rules(ACTION_RULES, rules.get("action", []))
    transition_rules = _build_rules(TRANSITION_RULES, rules.get("transition", []))
    observation_rules = _build_rules(OBSERVATION_RULES, rules.get("observation", []))
    budget_rules = _build_rules(BUDGET_RULES, rules.get("budget", []))
    return RuleHarness(
        base,
        setup_rules=setup_rules,
        contract_rules=contract_rules,
        action_rules=action_rules,
        transition_rules=transition_rules,
        observation_rules=observation_rules,
        budget_rules=budget_rules,
        environment_spec=canonicalize_environment_spec(spec),
    )


def _rules_by_execution_phase(spec: Mapping[str, Any]) -> dict[str, list[Any]]:
    """统一读取新 components 格式和旧 rules 格式。"""

    if "components" not in spec:
        raw = spec.get("rules", {})
        if not isinstance(raw, Mapping):
            raise ValueError("environment rules must be an object")
        return {phase: list(raw.get(phase, [])) for phase in (
            "setup", "contract", "action", "transition", "observation", "budget"
        )}

    components = spec.get("components")
    if not isinstance(components, list):
        raise ValueError("environment components must be a list")
    grouped = {
        phase: []
        for phase in (
            "setup", "contract", "action", "transition", "observation", "budget"
        )
    }
    for index, component in enumerate(components):
        if not isinstance(component, Mapping):
            raise ValueError(f"environment component {index} must be an object")
        phase = str(component.get("execution_phase", ""))
        if phase not in grouped:
            raise ValueError(
                f"environment component {index} has unsupported execution_phase {phase!r}"
            )
        expected_type, expected_axis = PHASE_CLASSIFICATION[phase]
        actual_type = component.get("component_type")
        actual_axis = component.get("primary_axis")
        if actual_type != expected_type or actual_axis != expected_axis:
            raise ValueError(
                f"environment component {index} classification conflicts with "
                f"execution_phase {phase!r}: expected "
                f"{(expected_type, expected_axis)!r}"
            )
        grouped[phase].append(dict(component))
    return grouped


def available_components() -> dict[str, tuple[str, ...]]:
    return {
        "environments": ENVIRONMENTS.names(),
        "setup_rules": SETUP_RULES.names(),
        "contract_rules": CONTRACT_RULES.names(),
        "action_rules": ACTION_RULES.names(),
        "transition_rules": TRANSITION_RULES.names(),
        "observation_rules": OBSERVATION_RULES.names(),
        "budget_rules": BUDGET_RULES.names(),
    }


def _build_rules(registry: ComponentRegistry[Any], specs: Any) -> list[Any]:
    if not isinstance(specs, list):
        raise ValueError(f"{registry.component_kind} specs must be a list")
    built = []
    for rule_spec in specs:
        if not isinstance(rule_spec, Mapping):
            raise ValueError(f"{registry.component_kind} spec must be an object")
        rule_type = str(rule_spec.get("type", ""))
        built.append(registry.build(rule_type, rule_spec))
    return built


def _build_appworld(spec: Mapping[str, Any]) -> BenchmarkAdapter:
    """从官方本地 bundle 构建进程隔离的 AppWorld Train 环境。"""

    task = dict(spec.get("task", {}))
    environment = dict(spec.get("environment", {}))
    parameters = dict(environment.get("parameters", {}))
    task_id = str(task.get("id", ""))
    return BenchmarkAdapter(
        AppWorldProcessBackend(
            task_id=task_id,
            appworld_root=parameters.get("appworld_root"),
            python_executable=parameters.get("python_executable"),
            experiment_name=str(
                parameters.get("experiment_name", "env_agent_rsi_appworld")
            ),
            max_interactions=int(parameters.get("max_interactions", 40)),
            request_timeout=float(parameters.get("request_timeout", 120.0)),
        )
    )


def _build_require_argument(config: Mapping[str, Any]) -> RequireArgumentRule:
    return RequireArgumentRule(
        tool=str(config["tool"]),
        argument=str(config["argument"]),
        error_code=str(config.get("error_code", "REQUIRED_ARGUMENT_MISSING")),
        message=str(config["message"]) if "message" in config else None,
    )


def _build_replay_setup(config: Mapping[str, Any]) -> ReplaySetupRule:
    action_specs = config.get("actions", [])
    if not isinstance(action_specs, list):
        raise ValueError("setup replay actions must be a list")
    actions: list[Action] = []
    for index, action_spec in enumerate(action_specs):
        if not isinstance(action_spec, Mapping) or "tool" not in action_spec:
            raise ValueError(f"setup action {index} must contain tool")
        actions.append(
            Action(
                str(action_spec["tool"]),
                dict(action_spec.get("arguments", {})),
            )
        )
    return ReplaySetupRule(actions)


def _build_require_argument_contract(
    config: Mapping[str, Any],
) -> RequireArgumentContractRule:
    return RequireArgumentContractRule(
        tool=str(config["tool"]),
        argument=str(config["argument"]),
        error_code=str(config.get("error_code", "REQUIRED_ARGUMENT_MISSING")),
        message=str(config["message"]) if "message" in config else None,
    )


def _build_post_commit_timeout(
    config: Mapping[str, Any],
) -> PostCommitTimeoutRule:
    return PostCommitTimeoutRule(
        tool=str(config.get("tool", "append_item")),
        trigger_on_nth=int(config.get("trigger_on_nth", 1)),
    )


def _build_stale_read(config: Mapping[str, Any]) -> StaleReadAfterWriteRule:
    return StaleReadAfterWriteRule(stale_reads=int(config.get("stale_reads", 1)))


def _build_stale_field(config: Mapping[str, Any]) -> StaleFieldAfterActionRule:
    return StaleFieldAfterActionRule(
        trigger_tool=str(config["trigger_tool"]),
        state_key=str(config["state_key"]),
        observation_key=str(config["observation_key"]),
        stale_reads=int(config.get("stale_reads", 1)),
    )


def _build_step_budget(config: Mapping[str, Any]) -> StepBudgetRule:
    write_tools = config.get("write_tools", [])
    if not isinstance(write_tools, list):
        raise ValueError("write_tools must be a list")
    max_writes = config.get("max_writes")
    return StepBudgetRule(
        max_steps=int(config["max_steps"]),
        max_writes=None if max_writes is None else int(max_writes),
        write_tools=[str(tool) for tool in write_tools],
    )


register_environment("appworld_process", _build_appworld)
register_setup_rule("replay", _build_replay_setup)
register_contract_rule("require_argument", _build_require_argument_contract)
register_action_rule("require_argument", _build_require_argument)
register_transition_rule("post_commit_timeout", _build_post_commit_timeout)
register_observation_rule("stale_read_after_write", _build_stale_read)
register_observation_rule("stale_field_after_action", _build_stale_field)
register_budget_rule("step_budget", _build_step_budget)
