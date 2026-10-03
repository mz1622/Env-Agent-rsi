"""通用的动作后陈旧字段 f_O 规则。

规则在指定有状态动作后，把一次响应中的 observation 字段替换为动作前 state 字段；
它不改变真实状态，主要用于验证 Modifier 能删除导致错误决策的观察故障。
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import Any, Mapping

from env_agent_rsi.core.protocol import Action, EnvResponse, JsonObject


class StaleFieldAfterActionRule:
    """在指定动作后有限次数返回动作前的字段值。"""

    name = "stale_field_after_action"

    def __init__(
        self,
        *,
        trigger_tool: str,
        state_key: str,
        observation_key: str,
        stale_reads: int = 1,
    ) -> None:
        if stale_reads <= 0:
            raise ValueError("stale_reads must be positive")
        self.trigger_tool = trigger_tool
        self.state_key = state_key
        self.observation_key = observation_key
        self.stale_reads = stale_reads
        self.reset()

    def reset(self) -> None:
        self.remaining = 0
        self.stale_value: Any = None

    def on_transition(
        self,
        previous_state: Mapping[str, Any],
        action: Action,
        raw_response: EnvResponse,
        current_state: Mapping[str, Any],
    ) -> None:
        del current_state
        found, previous_value = _lookup_path(previous_state, self.state_key)
        if (
            action.tool == self.trigger_tool
            and raw_response.info.get("state_changed")
            and found
        ):
            self.stale_value = deepcopy(previous_value)
            self.remaining = self.stale_reads

    def transform(
        self,
        action: Action,
        response: EnvResponse,
        current_state: Mapping[str, Any],
    ) -> EnvResponse:
        del current_state
        if (
            action.tool != self.trigger_tool
            or self.remaining <= 0
            or self.observation_key not in response.observation
        ):
            return response
        self.remaining -= 1
        observation = deepcopy(response.observation)
        observation[self.observation_key] = deepcopy(self.stale_value)
        info = deepcopy(response.info)
        info["fault_events"] = [
            *info.get("fault_events", []),
            {
                "rule": self.name,
                "phase": "observation",
                "field": self.observation_key,
                "remaining_after_this_read": self.remaining,
            },
        ]
        return replace(response, observation=observation, info=info)

    def save_state(self) -> JsonObject:
        return {
            "remaining": self.remaining,
            "stale_value": deepcopy(self.stale_value),
        }

    def load_state(self, state: Mapping[str, Any]) -> None:
        self.remaining = int(state["remaining"])
        self.stale_value = deepcopy(state["stale_value"])


def _lookup_path(value: Mapping[str, Any], path: str) -> tuple[bool, Any]:
    """读取 ``state.current_numbers`` 这类点分隔快照路径。"""

    current: Any = value
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return False, None
        current = current[part]
    return True, current
