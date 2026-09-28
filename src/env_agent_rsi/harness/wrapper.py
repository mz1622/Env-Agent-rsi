from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import Any, Mapping, Sequence

from env_agent_rsi.core.protocol import (
    Action,
    ActionableEnv,
    EnvResponse,
    EvaluationResult,
    JsonObject,
)
from env_agent_rsi.transforms.protocols import (
    ActionRule,
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
    ):
        self.base = base
        self.action_rules = list(action_rules)
        self.transition_rules = list(transition_rules)
        self.observation_rules = list(observation_rules)

    def reset(
        self, seed: int = 0, options: Mapping[str, Any] | None = None
    ) -> EnvResponse:
        for rule in [
            *self.action_rules,
            *self.transition_rules,
            *self.observation_rules,
        ]:
            rule.reset()
        return self.base.reset(seed=seed, options=options)

    def step(self, action: Action) -> EnvResponse:
        previous_state = self.base.get_env_state()
        effective_action = action
        blocked_response: EnvResponse | None = None
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
            "action": [rule.name for rule in self.action_rules],
            "transition": [rule.name for rule in self.transition_rules],
            "observation": [rule.name for rule in self.observation_rules],
        }
        if effective_action != action:
            info["effective_action"] = effective_action.to_dict()
        return replace(response, info=info)

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
        rules: Sequence[ActionRule | TransitionRule | ObservationRule],
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
