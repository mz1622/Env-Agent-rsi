from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import Any, Mapping

from env_agent_rsi.core.protocol import Action, EnvResponse, JsonObject


class StaleReadAfterWriteRule:
    """Return the pre-write item view for a bounded number of reads."""

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
            page = deepcopy(self.stale_items[cursor : cursor + limit])
            next_cursor = (
                cursor + limit if cursor + limit < len(self.stale_items) else None
            )
            observation = {
                "ok": True,
                "items": page,
                "next_cursor": next_cursor,
            }
        return replace(response, observation=observation, info=info)

    def save_state(self) -> JsonObject:
        return {
            "remaining": self.remaining,
            "stale_items": deepcopy(self.stale_items),
        }

    def load_state(self, state: Mapping[str, Any]) -> None:
        self.remaining = int(state["remaining"])
        self.stale_items = deepcopy(state["stale_items"])
