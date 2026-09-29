"""环境与规则的组合执行器。

RuleHarness 在 reset 后先执行 Setup，再按 Contract validation → schema validation → f_A
→ Budget pre-check → Base → f_T → f_O → Budget accounting 顺序执行；它也负责六类
规则快照和 contract version 变化通知。Agent 只能收到 observation，info 与真实状态
保留给诊断和 verifier。
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
    BudgetRule,
    ContractRule,
    ObservationRule,
    SetupRule,
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
        setup_rules: Sequence[SetupRule] = (),
        budget_rules: Sequence[BudgetRule] = (),
    ):
        self.base = base
        self.setup_rules = list(setup_rules)
        self.contract_rules = list(contract_rules)
        self.action_rules = list(action_rules)
        self.transition_rules = list(transition_rules)
        self.observation_rules = list(observation_rules)
        self.budget_rules = list(budget_rules)

    def reset(
        self, seed: int = 0, options: Mapping[str, Any] | None = None
    ) -> EnvResponse:
        all_rules = [
            *self.setup_rules,
            *self.contract_rules,
            *self.action_rules,
            *self.transition_rules,
            *self.observation_rules,
            *self.budget_rules,
        ]
        for rule in all_rules:
            rule.reset()
        response = self.base.reset(seed=seed, options=options)
        setup_trace: list[JsonObject] = []
        setup_observation = deepcopy(response.observation)
        for rule in self.setup_rules:
            actions = rule.initial_actions(self.base.get_env_state())
            for index, action in enumerate(actions, start=1):
                try:
                    validate_action(action, self.base.describe())
                except ActionValidationError as exc:
                    raise ValueError(
                        f"invalid setup action {rule.name}[{index}]: {exc.message}"
                    ) from exc
                setup_response = self.base.step(action)
                setup_trace.append(
                    {
                        "rule": rule.name,
                        "index": index,
                        "action": action.to_dict(),
                        "response": setup_response.to_dict(),
                    }
                )
                setup_observation = deepcopy(setup_response.observation)
                if not setup_response.observation.get("ok", False):
                    raise ValueError(
                        f"setup action {rule.name}[{index}] failed: "
                        f"{setup_response.observation!r}"
                    )
                if setup_response.terminated or setup_response.truncated:
                    raise ValueError(
                        f"setup action {rule.name}[{index}] ended the episode"
                    )

        # Setup 只塑造基础状态，不应消耗随后交互规则的内部计数和预算。
        for rule in [
            *self.contract_rules,
            *self.action_rules,
            *self.transition_rules,
            *self.observation_rules,
            *self.budget_rules,
        ]:
            rule.reset()
        descriptor = self.describe()
        observation = setup_observation
        observation.update(descriptor.to_dict())
        info = deepcopy(response.info)
        info["contract_version"] = descriptor.contract_version
        info["setup_trace"] = setup_trace
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
        if blocked_response is None:
            for rule in self.budget_rules:
                blocked_response = rule.before_step(previous_state, effective_action)
                if blocked_response is not None:
                    break

        executed = blocked_response is None
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
        if executed:
            for rule in self.budget_rules:
                response = rule.after_step(effective_action, response)

        info = deepcopy(response.info)
        info["rule_order"] = {
            "setup": [rule.name for rule in self.setup_rules],
            "contract": [rule.name for rule in self.contract_rules],
            "action": [rule.name for rule in self.action_rules],
            "transition": [rule.name for rule in self.transition_rules],
            "observation": [rule.name for rule in self.observation_rules],
            "budget": [rule.name for rule in self.budget_rules],
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
            "setup_rules": [
                {"name": rule.name, "state": rule.save_state()}
                for rule in self.setup_rules
            ],
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
            "budget_rules": [
                {"name": rule.name, "state": rule.save_state()}
                for rule in self.budget_rules
            ],
        }

    def load_state(self, snapshot: Mapping[str, Any]) -> None:
        if snapshot.get("snapshot_version") != self.SNAPSHOT_VERSION:
            raise ValueError("unsupported RuleHarness snapshot version")
        self.base.load_state(snapshot["base"])
        self._load_rule_group(
            self.setup_rules, snapshot.get("setup_rules", []), "setup"
        )
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
        self._load_rule_group(
            self.budget_rules, snapshot.get("budget_rules", []), "budget"
        )

    @staticmethod
    def _load_rule_group(
        rules: Sequence[
            SetupRule
            | ContractRule
            | ActionRule
            | TransitionRule
            | ObservationRule
            | BudgetRule
        ],
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
