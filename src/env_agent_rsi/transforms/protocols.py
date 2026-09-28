from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from env_agent_rsi.core.protocol import Action, EnvResponse, JsonObject


@dataclass(frozen=True)
class ActionDecision:
    """Result of an f_A rule.

    A rule may rewrite the action, or return a response to block execution before
    the base environment mutates.
    """

    action: Action
    response: EnvResponse | None = None


class StatefulRule(Protocol):
    name: str

    def reset(self) -> None:
        ...

    def save_state(self) -> JsonObject:
        ...

    def load_state(self, state: Mapping[str, Any]) -> None:
        ...


class ActionRule(StatefulRule, Protocol):
    def before_step(self, state: Mapping[str, Any], action: Action) -> ActionDecision:
        ...


class TransitionRule(StatefulRule, Protocol):
    def after_step(
        self,
        previous_state: Mapping[str, Any],
        action: Action,
        response: EnvResponse,
        current_state: Mapping[str, Any],
    ) -> EnvResponse:
        ...


class ObservationRule(StatefulRule, Protocol):
    def on_transition(
        self,
        previous_state: Mapping[str, Any],
        action: Action,
        raw_response: EnvResponse,
        current_state: Mapping[str, Any],
    ) -> None:
        ...

    def transform(
        self,
        action: Action,
        response: EnvResponse,
        current_state: Mapping[str, Any],
    ) -> EnvResponse:
        ...
