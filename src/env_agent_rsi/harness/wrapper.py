"""环境与规则的组合执行器。

RuleHarness 先生成有效工具契约，再按 Contract validation → schema validation → f_A
→ Base → f_T → f_O 顺序执行；它也负责规则快照和 contract version 变化通知。
Agent 只能收到 observation，info 与真实状态保留给诊断和 verifier。
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import Any, Mapping, Sequence

from env_agent_rsi.core.protocol import (
    Action,
    ActionableEnv,
    EnvDescriptor,
    EnvResponse,
    EvaluationResult,
    JsonObject,
)
from env_agent_rsi.core.tooling import ActionValidationError, validate_action
from env_agent_rsi.transforms.protocols import (
    ActionRule,
    ContractRule,
    ObservationRule,
    TransitionRule,
)


class RuleHarness:
    """Compose f_A, f_T and f_O rules around a base ActionableEnv."""

    SNAPSHOT_VERSION = 1

    def __init__(
        self,
        base: ActionableEnv,
        transition_rules: Sequence[TransitionRule] = (),
        observation_rules: Sequence[ObservationRule] = (),
        action_rules: Sequence[ActionRule] = (),
        contract_rules: Sequence[ContractRule] = (),
    ):
        self.base = base
        self.contract_rules = list(contract_rules)
        self.action_rules = list(action_rules)
        self.transition_rules = list(transition_rules)
        self.observation_rules = list(observation_rules)

    def reset(
        self, seed: int = 0, options: Mapping[str, Any] | None = None
    ) -> EnvResponse:
        for rule in [
            *self.contract_rules,
            *self.action_rules,
            *self.transition_rules,
            *self.observation_rules,
        ]:
            rule.reset()
        response = self.base.reset(seed=seed, options=options)
        descriptor = self.describe()
        observation = deepcopy(response.observation)
        observation.update(descriptor.to_dict())
        info = deepcopy(response.info)
        info["contract_version"] = descriptor.contract_version
        return replace(response, observation=observation, info=info)

    def describe(self) -> EnvDescriptor:
        descriptor = self.base.describe()
        state = self.base.get_env_state()
        for rule in self.contract_rules:
            descriptor = rule.transform_descriptor(descriptor, state)
        return descriptor

    def step(self, action: Action) -> EnvResponse:
        previous_state = self.base.get_env_state()
        descriptor_before = self.describe()
        effective_action = action
        blocked_response: EnvResponse | None = None
        for rule in self.contract_rules:
            blocked_response = rule.validate_action(previous_state, effective_action)
            if blocked_response is not None:
                break
        if blocked_response is None:
            try:
                validate_action(effective_action, descriptor_before)
            except ActionValidationError as exc:
                blocked_response = EnvResponse(
                    observation={
                        "ok": False,
                        "error": {"code": exc.code, "message": exc.message},
                    },
                    info={
                        "event": "schema_validation_blocked",
                        "state_changed": False,
                        "error_code": exc.code,
                    },
                )
        if blocked_response is None:
            for rule in self.action_rules:
                decision = rule.before_step(previous_state, effective_action)
                effective_action = decision.action
                if decision.response is not None:
                    blocked_response = decision.response
                    break

        raw_response = (
            blocked_response
            if blocked_response is not None
            else self.base.step(effective_action)
        )
        current_state = self.base.get_env_state()

        response = raw_response
        if blocked_response is None:
            for rule in self.transition_rules:
                response = rule.after_step(
                    previous_state, effective_action, response, current_state
                )

        for rule in self.observation_rules:
            rule.on_transition(
                previous_state, effective_action, raw_response, current_state
            )
        for rule in self.observation_rules:
            response = rule.transform(effective_action, response, current_state)

        info = deepcopy(response.info)
        info["rule_order"] = {
            "contract": [rule.name for rule in self.contract_rules],
            "action": [rule.name for rule in self.action_rules],
            "transition": [rule.name for rule in self.transition_rules],
            "observation": [rule.name for rule in self.observation_rules],
        }
        if effective_action != action:
            info["effective_action"] = effective_action.to_dict()
        descriptor_after = self.describe()
        info["contract_version"] = descriptor_after.contract_version
        observation = deepcopy(response.observation)
        if descriptor_after.contract_version != descriptor_before.contract_version:
            observation["contract_update"] = descriptor_after.to_dict()
        return replace(response, observation=observation, info=info)

    def observe(self) -> JsonObject:
        current_state = self.base.get_env_state()
        response = EnvResponse(
            observation={"ok": True, **self.base.observe()},
            info={"event": "observe", "state_changed": False},
        )
        action = Action("observe")
        for rule in self.observation_rules:
            response = rule.transform(action, response, current_state)
        return response.observation

    def evaluate(self) -> EvaluationResult:
        return self.base.evaluate()

    def get_env_state(self) -> JsonObject:
        return self.base.get_env_state()

    def save_state(self) -> JsonObject:
        return {
            "snapshot_version": self.SNAPSHOT_VERSION,
            "base": self.base.save_state(),
            "contract_rules": [
                {"name": rule.name, "state": rule.save_state()}
                for rule in self.contract_rules
            ],
            "action_rules": [
                {"name": rule.name, "state": rule.save_state()}
                for rule in self.action_rules
            ],
            "transition_rules": [
                {"name": rule.name, "state": rule.save_state()}
                for rule in self.transition_rules
            ],
            "observation_rules": [
                {"name": rule.name, "state": rule.save_state()}
                for rule in self.observation_rules
            ],
        }

    def load_state(self, snapshot: Mapping[str, Any]) -> None:
        if snapshot.get("snapshot_version") != self.SNAPSHOT_VERSION:
            raise ValueError("unsupported RuleHarness snapshot version")
        self.base.load_state(snapshot["base"])
        self._load_rule_group(
            self.contract_rules, snapshot.get("contract_rules", []), "contract"
        )
        self._load_rule_group(
            self.action_rules, snapshot.get("action_rules", []), "action"
        )
        self._load_rule_group(
            self.transition_rules, snapshot["transition_rules"], "transition"
        )
        self._load_rule_group(
            self.observation_rules, snapshot["observation_rules"], "observation"
        )

    @staticmethod
    def _load_rule_group(
        rules: Sequence[ContractRule | ActionRule | TransitionRule | ObservationRule],
        states: Sequence[Mapping[str, Any]],
        group: str,
    ) -> None:
        expected = [rule.name for rule in rules]
        actual = [entry["name"] for entry in states]
        if expected != actual:
            raise ValueError(
                f"{group} rule mismatch: expected {expected!r}, got {actual!r}"
            )
        for rule, entry in zip(rules, states):
            rule.load_state(entry["state"])
