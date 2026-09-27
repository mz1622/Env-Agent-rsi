from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import Any, Mapping, Sequence

from env_agent_rsi.core.protocol import Action, ActionableEnv, EnvResponse, EvaluationResult, JsonObject
from env_agent_rsi.harness.rules import ObservationRule, TransitionRule


class RuleHarness:
    """Compose f_T and f_O rules around a base ActionableEnv."""

    SNAPSHOT_VERSION = 1

    def __init__(
        self,
        base: ActionableEnv,
        transition_rules: Sequence[TransitionRule] = (),
        observation_rules: Sequence[ObservationRule] = (),
    ):
        self.base = base
        self.transition_rules = list(transition_rules)
        self.observation_rules = list(observation_rules)

    def reset(
        self, seed: int = 0, options: Mapping[str, Any] | None = None
    ) -> EnvResponse:
        for rule in [*self.transition_rules, *self.observation_rules]:
            rule.reset()
        return self.base.reset(seed=seed, options=options)

    def step(self, action: Action) -> EnvResponse:
        previous_state = self.base.get_env_state()
        raw_response = self.base.step(action)
        current_state = self.base.get_env_state()

        response = raw_response
        for rule in self.transition_rules:
            response = rule.after_step(
                previous_state, action, response, current_state
            )

        for rule in self.observation_rules:
            rule.on_transition(
                previous_state, action, raw_response, current_state
            )
        for rule in self.observation_rules:
            response = rule.transform(action, response, current_state)

        info = deepcopy(response.info)
        info["rule_order"] = {
            "transition": [rule.name for rule in self.transition_rules],
            "observation": [rule.name for rule in self.observation_rules],
        }
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
            self.transition_rules, snapshot["transition_rules"], "transition"
        )
        self._load_rule_group(
            self.observation_rules, snapshot["observation_rules"], "observation"
        )

    @staticmethod
    def _load_rule_group(
        rules: Sequence[TransitionRule | ObservationRule],
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
