from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import Any, Mapping

from env_agent_rsi.core.protocol import Action, EnvResponse, JsonObject


class PostCommitTimeoutRule:
    """Replace the Nth successful commit response with a timeout."""

    name = "post_commit_timeout"

    def __init__(self, tool: str = "append_item", trigger_on_nth: int = 1):
        if trigger_on_nth <= 0:
            raise ValueError("trigger_on_nth must be positive")
        self.tool = tool
        self.trigger_on_nth = trigger_on_nth
        self.reset()

    def reset(self) -> None:
        self.commit_count = 0
        self.fired = False

    def after_step(
        self,
        previous_state: Mapping[str, Any],
        action: Action,
        response: EnvResponse,
        current_state: Mapping[str, Any],
    ) -> EnvResponse:
        del previous_state, current_state
        if action.tool != self.tool or not response.info.get("committed"):
            return response
        self.commit_count += 1
        if self.fired or self.commit_count != self.trigger_on_nth:
            return response
        self.fired = True
        info = deepcopy(response.info)
        info["fault_events"] = [
            *info.get("fault_events", []),
            {
                "rule": self.name,
                "phase": "after_commit",
                "true_state_changed": True,
            },
        ]
        return replace(
            response,
            observation={
                "ok": False,
                "error": {
                    "code": "TIMEOUT",
                    "message": "request timed out before a response was received",
                },
            },
            info=info,
        )

    def save_state(self) -> JsonObject:
        return {"commit_count": self.commit_count, "fired": self.fired}

    def load_state(self, state: Mapping[str, Any]) -> None:
        self.commit_count = int(state["commit_count"])
        self.fired = bool(state["fired"])
