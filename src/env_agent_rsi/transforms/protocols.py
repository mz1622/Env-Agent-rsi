"""环境变换扩展点的协议定义。

SetupRule 通过合法动作重放改变初始状态；ContractRule 改变 Agent 可见工具契约并
同步执行校验；Action/Transition/ObservationRule 分别实现 f_A、f_T、f_O；
BudgetRule 管理 episode 资源。所有有状态规则必须支持 reset 与快照，保证环境搜索
节点可复放。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

from env_agent_rsi.core.protocol import Action, EnvDescriptor, EnvResponse, JsonObject


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


class SetupRule(StatefulRule, Protocol):
    def initial_actions(self, state: Mapping[str, Any]) -> Sequence[Action]:
        ...


class ActionRule(StatefulRule, Protocol):
    def before_step(self, state: Mapping[str, Any], action: Action) -> ActionDecision:
        ...


class ContractRule(StatefulRule, Protocol):
    def transform_descriptor(
        self, descriptor: EnvDescriptor, state: Mapping[str, Any]
    ) -> EnvDescriptor:
        ...

    def validate_action(
        self, state: Mapping[str, Any], action: Action
    ) -> EnvResponse | None:
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


class BudgetRule(StatefulRule, Protocol):
    def before_step(
        self, state: Mapping[str, Any], action: Action
    ) -> EnvResponse | None:
        ...

    def after_step(self, action: Action, response: EnvResponse) -> EnvResponse:
        ...
