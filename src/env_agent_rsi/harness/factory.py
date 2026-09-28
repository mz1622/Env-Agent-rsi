from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from env_agent_rsi.core.protocol import ActionableEnv
from env_agent_rsi.core.registry import Builder, ComponentRegistry
from env_agent_rsi.core.verifier import StateVerifier
from env_agent_rsi.harness.wrapper import RuleHarness
from env_agent_rsi.micro_api.item_env import ItemEnv
from env_agent_rsi.transforms import (
    ActionRule,
    ObservationRule,
    PostCommitTimeoutRule,
    RequireArgumentRule,
    StaleReadAfterWriteRule,
    TransitionRule,
)
from env_agent_rsi.verifiers import ExactlyOnceVerifier


ENVIRONMENTS = ComponentRegistry[ActionableEnv]("environment")
VERIFIERS = ComponentRegistry[StateVerifier]("verifier")
ACTION_RULES = ComponentRegistry[ActionRule]("action rule")
TRANSITION_RULES = ComponentRegistry[TransitionRule]("transition rule")
OBSERVATION_RULES = ComponentRegistry[ObservationRule]("observation rule")


def register_environment(name: str, builder: Builder[ActionableEnv]) -> None:
    ENVIRONMENTS.register(name, builder)


def register_verifier(name: str, builder: Builder[StateVerifier]) -> None:
    VERIFIERS.register(name, builder)


def register_action_rule(name: str, builder: Builder[ActionRule]) -> None:
    ACTION_RULES.register(name, builder)


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
    action_rules = _build_rules(ACTION_RULES, rules.get("action", []))
    transition_rules = _build_rules(TRANSITION_RULES, rules.get("transition", []))
    observation_rules = _build_rules(OBSERVATION_RULES, rules.get("observation", []))
    return RuleHarness(
        base,
        action_rules=action_rules,
        transition_rules=transition_rules,
        observation_rules=observation_rules,
    )


def available_components() -> dict[str, tuple[str, ...]]:
    return {
        "environments": ENVIRONMENTS.names(),
        "verifiers": VERIFIERS.names(),
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


def _build_require_argument(config: Mapping[str, Any]) -> RequireArgumentRule:
    return RequireArgumentRule(
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
register_verifier("exactly_once", lambda _: ExactlyOnceVerifier())
register_action_rule("require_argument", _build_require_argument)
register_transition_rule("post_commit_timeout", _build_post_commit_timeout)
register_observation_rule("stale_read_after_write", _build_stale_read)
