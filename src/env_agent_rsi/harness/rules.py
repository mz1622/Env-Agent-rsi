from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import Any, Mapping, Protocol

from env_agent_rsi.core.protocol import Action, EnvResponse, JsonObject
from env_agent_rsi.micro_api.item_env import ItemEnv


class TransitionRule(Protocol):
    name: str

    def reset(self) -> None: ...

    def after_step(
        self,
        previous_state: Mapping[str, Any],
        action: Action,
        response: EnvResponse,
        current_state: Mapping[str, Any],
    ) -> EnvResponse: ...

    def save_state(self) -> JsonObject: ...

    def load_state(self, state: Mapping[str, Any]) -> None: ...


class ObservationRule(Protocol):
    name: str

    def reset(self) -> None: ...

    def on_transition(
        self,
        previous_state: Mapping[str, Any],
        action: Action,
        raw_response: EnvResponse,
        current_state: Mapping[str, Any],
    ) -> None: ...

    def transform(
        self,
        action: Action,
        response: EnvResponse,
        current_state: Mapping[str, Any],
    ) -> EnvResponse: ...

    def save_state(self) -> JsonObject: ...

    def load_state(self, state: Mapping[str, Any]) -> None: ...


class PostCommitTimeoutRule:
    """Replace the Nth successful commit response with a timeout.

    The base transition is not rolled back. The true state therefore contains
    the write even though the agent-visible observation reports failure.
    """

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


class StaleReadAfterWriteRule:
    """Return the pre-write item view for a bounded number of list calls."""

    name = "stale_read_after_write"

    def __init__(self, stale_reads: int = 1):
        if stale_reads <= 0:
            raise ValueError("stale_reads must be positive")
        self.stale_reads = stale_reads
        self.reset()

    def reset(self) -> None:
        self.remaining = 0
        self.stale_items: list[JsonObject] | None = None

    def on_transition(
        self,
        previous_state: Mapping[str, Any],
        action: Action,
        raw_response: EnvResponse,
        current_state: Mapping[str, Any],
    ) -> None:
        del current_state
        if action.tool == "append_item" and raw_response.info.get("committed"):
            self.stale_items = deepcopy(previous_state["items"])
            self.remaining = self.stale_reads

    def transform(
        self,
        action: Action,
        response: EnvResponse,
        current_state: Mapping[str, Any],
    ) -> EnvResponse:
        if (
            action.tool not in {"list_items", "observe"}
            or self.remaining <= 0
            or self.stale_items is None
            or not response.observation.get("ok")
        ):
            return response
        try:
            cursor = int(action.arguments.get("cursor", 0))
            limit = int(action.arguments.get("limit", 2))
        except (TypeError, ValueError):
            return response
        self.remaining -= 1
        info = deepcopy(response.info)
        info["fault_events"] = [
            *info.get("fault_events", []),
            {
                "rule": self.name,
                "phase": "observation",
                "remaining_after_this_read": self.remaining,
            },
        ]
        if action.tool == "observe":
            observation = {
                "ok": True,
                "items": deepcopy(self.stale_items),
                "terminated": bool(current_state["terminated"]),
                "step_count": int(current_state["step_count"]),
            }
        else:
            observation = ItemEnv.render_item_page(self.stale_items, cursor, limit)
        return replace(response, observation=observation, info=info)

    def save_state(self) -> JsonObject:
        return {
            "remaining": self.remaining,
            "stale_items": deepcopy(self.stale_items),
        }

    def load_state(self, state: Mapping[str, Any]) -> None:
        self.remaining = int(state["remaining"])
        self.stale_items = deepcopy(state["stale_items"])
