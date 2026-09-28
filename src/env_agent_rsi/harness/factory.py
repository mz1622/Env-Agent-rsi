"""配置到运行环境的装配工厂。

本模块维护环境、verifier、contract 与 f_A/f_T/f_O 的显式注册表，根据 task.json
构建完整 RuleHarness。新增组件通过注册函数接入，runner 不需要出现业务分支。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from env_agent_rsi.core.protocol import ActionableEnv
from env_agent_rsi.core.registry import Builder, ComponentRegistry
from env_agent_rsi.core.verifier import StateVerifier
from env_agent_rsi.environments import (
    CalendarEmailEnv,
    CodeRepairEnv,
    IssueWorkflowEnv,
    OrderLifecycleEnv,
)
from env_agent_rsi.harness.wrapper import RuleHarness
from env_agent_rsi.micro_api.item_env import ItemEnv
from env_agent_rsi.transforms import (
    ActionRule,
    ContractRule,
    ObservationRule,
    PostCommitTimeoutRule,
    RequireArgumentContractRule,
    RequireArgumentRule,
    StaleReadAfterWriteRule,
    TransitionRule,
)
from env_agent_rsi.verifiers import (
    CalendarEmailGoalVerifier,
    CodeRepairGoalVerifier,
    ExactlyOnceVerifier,
    IssueGoalVerifier,
    OrderGoalVerifier,
)


ENVIRONMENTS = ComponentRegistry[ActionableEnv]("environment")
VERIFIERS = ComponentRegistry[StateVerifier]("verifier")
ACTION_RULES = ComponentRegistry[ActionRule]("action rule")
CONTRACT_RULES = ComponentRegistry[ContractRule]("contract rule")
TRANSITION_RULES = ComponentRegistry[TransitionRule]("transition rule")
OBSERVATION_RULES = ComponentRegistry[ObservationRule]("observation rule")


def register_environment(name: str, builder: Builder[ActionableEnv]) -> None:
    ENVIRONMENTS.register(name, builder)


def register_verifier(name: str, builder: Builder[StateVerifier]) -> None:
    VERIFIERS.register(name, builder)


def register_action_rule(name: str, builder: Builder[ActionRule]) -> None:
    ACTION_RULES.register(name, builder)


def register_contract_rule(name: str, builder: Builder[ContractRule]) -> None:
    CONTRACT_RULES.register(name, builder)


def register_transition_rule(name: str, builder: Builder[TransitionRule]) -> None:
    TRANSITION_RULES.register(name, builder)


def register_observation_rule(name: str, builder: Builder[ObservationRule]) -> None:
    OBSERVATION_RULES.register(name, builder)


def load_spec(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        spec = json.load(handle)
    if not isinstance(spec, dict):
        raise ValueError("environment spec must be a JSON object")
    return spec


def build_environment(spec: Mapping[str, Any]) -> RuleHarness:
    environment_spec = dict(spec.get("environment", {"type": "item"}))
    environment_type = str(environment_spec.get("type", "item"))
    base = ENVIRONMENTS.build(environment_type, spec)

    rules = dict(spec.get("rules", {}))
    contract_rules = _build_rules(CONTRACT_RULES, rules.get("contract", []))
    action_rules = _build_rules(ACTION_RULES, rules.get("action", []))
    transition_rules = _build_rules(TRANSITION_RULES, rules.get("transition", []))
    observation_rules = _build_rules(OBSERVATION_RULES, rules.get("observation", []))
    return RuleHarness(
        base,
        contract_rules=contract_rules,
        action_rules=action_rules,
        transition_rules=transition_rules,
        observation_rules=observation_rules,
    )


def available_components() -> dict[str, tuple[str, ...]]:
    return {
        "environments": ENVIRONMENTS.names(),
        "verifiers": VERIFIERS.names(),
        "contract_rules": CONTRACT_RULES.names(),
        "action_rules": ACTION_RULES.names(),
        "transition_rules": TRANSITION_RULES.names(),
        "observation_rules": OBSERVATION_RULES.names(),
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


def _build_item(spec: Mapping[str, Any]) -> ItemEnv:
    task = dict(spec.get("task", {}))
    environment = dict(spec.get("environment", {}))
    parameters = dict(environment.get("parameters", {}))
    verifier = dict(spec.get("verifier", {"type": "exactly_once"}))
    verifier_type = str(verifier.get("type", "exactly_once"))
    return ItemEnv(
        target_value=str(task.get("target_value", "target-item")),
        page_size=int(parameters.get("page_size", task.get("page_size", 2))),
        verifier=VERIFIERS.build(verifier_type, verifier),
    )


def _task_and_verifier(
    spec: Mapping[str, Any], default_task_id: str, default_verifier: str
) -> tuple[dict[str, Any], StateVerifier]:
    task = dict(spec.get("task", {}))
    verifier_spec = dict(spec.get("verifier", {"type": default_verifier}))
    verifier_type = str(verifier_spec.get("type", default_verifier))
    task.setdefault("id", default_task_id)
    return task, VERIFIERS.build(verifier_type, verifier_spec)


def _build_order_lifecycle(spec: Mapping[str, Any]) -> OrderLifecycleEnv:
    task, verifier = _task_and_verifier(
        spec, "tau_retail_adapted_66", "order_goal_state"
    )
    kwargs: dict[str, Any] = {"verifier": verifier, "task_id": str(task["id"])}
    if "instruction" in task:
        kwargs["instruction"] = str(task["instruction"])
    return OrderLifecycleEnv(**kwargs)


def _build_issue_workflow(spec: Mapping[str, Any]) -> IssueWorkflowEnv:
    task, verifier = _task_and_verifier(
        spec, "webarena_verified_adapted_446", "issue_goal_state"
    )
    kwargs: dict[str, Any] = {"verifier": verifier, "task_id": str(task["id"])}
    if "instruction" in task:
        kwargs["instruction"] = str(task["instruction"])
    return IssueWorkflowEnv(**kwargs)


def _build_calendar_email(spec: Mapping[str, Any]) -> CalendarEmailEnv:
    task, verifier = _task_and_verifier(
        spec, "workbench_multidomain_adapted_151", "calendar_email_goal_state"
    )
    kwargs: dict[str, Any] = {"verifier": verifier, "task_id": str(task["id"])}
    if "instruction" in task:
        kwargs["instruction"] = str(task["instruction"])
    return CalendarEmailEnv(**kwargs)


def _build_code_repair(spec: Mapping[str, Any]) -> CodeRepairEnv:
    task, verifier = _task_and_verifier(
        spec,
        "swebench_lite_adapted_astropy_14365",
        "test_patch_verifier",
    )
    kwargs: dict[str, Any] = {"verifier": verifier, "task_id": str(task["id"])}
    if "instruction" in task:
        kwargs["instruction"] = str(task["instruction"])
    return CodeRepairEnv(**kwargs)


def _build_require_argument(config: Mapping[str, Any]) -> RequireArgumentRule:
    return RequireArgumentRule(
        tool=str(config["tool"]),
        argument=str(config["argument"]),
        error_code=str(config.get("error_code", "REQUIRED_ARGUMENT_MISSING")),
        message=str(config["message"]) if "message" in config else None,
    )


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


register_environment("item", _build_item)
register_environment("order_api", _build_order_lifecycle)
register_environment("issue_tracker", _build_issue_workflow)
register_environment("workplace_apps", _build_calendar_email)
register_environment("code_repository", _build_code_repair)
register_verifier("exactly_once", lambda _: ExactlyOnceVerifier())
register_verifier("order_goal_state", lambda _: OrderGoalVerifier())
register_verifier("issue_goal_state", lambda _: IssueGoalVerifier())
register_verifier("calendar_email_goal_state", lambda _: CalendarEmailGoalVerifier())
register_verifier("test_patch_verifier", lambda _: CodeRepairGoalVerifier())
register_contract_rule("require_argument", _build_require_argument_contract)
register_action_rule("require_argument", _build_require_argument)
register_transition_rule("post_commit_timeout", _build_post_commit_timeout)
register_observation_rule("stale_read_after_write", _build_stale_read)
