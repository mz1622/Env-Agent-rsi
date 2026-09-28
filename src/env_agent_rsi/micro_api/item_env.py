from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from env_agent_rsi.core.protocol import (
    Action,
    EnvResponse,
    EvaluationResult,
    JsonObject,
)
from env_agent_rsi.core.verifier import StateVerifier
from env_agent_rsi.environments.base import tool


ITEM_TOOLS = [
    tool(
        "list_items",
        "List items using cursor pagination.",
        {"cursor": {"type": "integer"}, "limit": {"type": "integer"}},
    ),
    tool(
        "get_item",
        "Read one item by numeric id.",
        {"id": {"type": "integer"}},
        ["id"],
    ),
    tool(
        "append_item",
        "Append one value, optionally with a stable idempotency key.",
        {
            "value": {"type": "string"},
            "idempotency_key": {"type": "string"},
        },
        ["value"],
    ),
    tool("finish", "Finish the task and run the independent verifier.", {}),
]


class ItemEnv:
    """Deterministic stateful API environment with an exactly-once task."""

    SNAPSHOT_VERSION = 1

    def __init__(
        self,
        verifier: StateVerifier,
        target_value: str = "target-item",
        page_size: int = 2,
    ):
        self.target_value = target_value
        self.page_size = page_size
        self.verifier = verifier
        self.reset()

    def reset(
        self, seed: int = 0, options: Mapping[str, Any] | None = None
    ) -> EnvResponse:
        options = dict(options or {})
        self.seed = seed
        self.target_value = str(options.get("target_value", self.target_value))
        self.items: list[JsonObject] = [
            {"id": 1, "value": "existing-item", "idempotency_key": "initial-1"}
        ]
        self.initial_items = deepcopy(self.items)
        self.audit_log: list[JsonObject] = []
        self.next_id = 2
        self.step_count = 0
        self.terminated = False
        return EnvResponse(
            observation={
                "ok": True,
                "task": f"Append value {self.target_value!r} exactly once, then finish.",
                "tools": deepcopy(ITEM_TOOLS),
            },
            info={"event": "reset", "seed": seed},
        )

    def step(self, action: Action) -> EnvResponse:
        if self.terminated:
            return self._error("EPISODE_TERMINATED", "finish has already been called")

        self.step_count += 1
        handlers = {
            "list_items": self._list_items,
            "get_item": self._get_item,
            "append_item": self._append_item,
            "finish": self._finish,
        }
        handler = handlers.get(action.tool)
        if handler is None:
            return self._error("UNKNOWN_TOOL", f"unknown tool: {action.tool}")
        return handler(dict(action.arguments))

    def _list_items(self, arguments: JsonObject) -> EnvResponse:
        try:
            cursor = int(arguments.get("cursor", 0))
            limit = int(arguments.get("limit", self.page_size))
        except (TypeError, ValueError):
            return self._error("INVALID_ARGUMENT", "cursor and limit must be integers")
        if cursor < 0 or limit <= 0:
            return self._error("INVALID_ARGUMENT", "cursor must be >= 0 and limit > 0")
        return EnvResponse(
            observation=self.render_item_page(self.items, cursor, limit),
            info={"event": "list_items", "state_changed": False},
        )

    @staticmethod
    def render_item_page(
        items: list[JsonObject], cursor: int, limit: int
    ) -> JsonObject:
        page = deepcopy(items[cursor : cursor + limit])
        next_cursor = cursor + limit if cursor + limit < len(items) else None
        return {"ok": True, "items": page, "next_cursor": next_cursor}

    def _get_item(self, arguments: JsonObject) -> EnvResponse:
        try:
            item_id = int(arguments["id"])
        except (KeyError, TypeError, ValueError):
            return self._error("INVALID_ARGUMENT", "id must be an integer")
        for item in self.items:
            if item["id"] == item_id:
                return EnvResponse(
                    observation={"ok": True, "item": deepcopy(item)},
                    info={"event": "get_item", "state_changed": False},
                )
        return self._error("NOT_FOUND", f"item {item_id} was not found")

    def _append_item(self, arguments: JsonObject) -> EnvResponse:
        value = arguments.get("value")
        if not isinstance(value, str) or not value:
            return self._error("INVALID_ARGUMENT", "value must be a non-empty string")

        key = arguments.get("idempotency_key")
        if key is not None and (not isinstance(key, str) or not key):
            return self._error(
                "INVALID_ARGUMENT", "idempotency_key must be a non-empty string"
            )

        if key is not None:
            for item in self.items:
                if item["idempotency_key"] == key:
                    self.audit_log.append(
                        {
                            "step": self.step_count,
                            "event": "idempotent_replay",
                            "item_id": item["id"],
                            "idempotency_key": key,
                        }
                    )
                    return EnvResponse(
                        observation={
                            "ok": True,
                            "status": "already_exists",
                            "item": deepcopy(item),
                        },
                        info={
                            "event": "idempotent_replay",
                            "state_changed": False,
                            "committed": False,
                        },
                    )

        item = {"id": self.next_id, "value": value, "idempotency_key": key}
        self.next_id += 1
        self.items.append(item)
        self.audit_log.append(
            {
                "step": self.step_count,
                "event": "append",
                "item_id": item["id"],
                "idempotency_key": key,
            }
        )
        return EnvResponse(
            observation={"ok": True, "status": "created", "item": deepcopy(item)},
            info={"event": "append", "state_changed": True, "committed": True},
        )

    def _finish(self, arguments: JsonObject) -> EnvResponse:
        del arguments
        self.terminated = True
        evaluation = self.evaluate()
        return EnvResponse(
            observation={
                "ok": True,
                "status": "finished",
                "task_success": evaluation.success,
            },
            reward=1.0 if evaluation.success else 0.0,
            terminated=True,
            info={"event": "finish", "evaluation": evaluation.to_dict()},
        )

    def _error(self, code: str, message: str) -> EnvResponse:
        return EnvResponse(
            observation={"ok": False, "error": {"code": code, "message": message}},
            info={"event": "error", "state_changed": False, "error_code": code},
        )

    def observe(self) -> JsonObject:
        return {
            "items": deepcopy(self.items),
            "terminated": self.terminated,
            "step_count": self.step_count,
        }

    def evaluate(self) -> EvaluationResult:
        return self.verifier.evaluate(self.save_state())

    def get_env_state(self) -> JsonObject:
        return deepcopy(self.save_state())

    def save_state(self) -> JsonObject:
        return {
            "snapshot_version": self.SNAPSHOT_VERSION,
            "seed": self.seed,
            "target_value": self.target_value,
            "page_size": self.page_size,
            "items": deepcopy(self.items),
            "initial_items": deepcopy(self.initial_items),
            "audit_log": deepcopy(self.audit_log),
            "next_id": self.next_id,
            "step_count": self.step_count,
            "terminated": self.terminated,
        }

    def load_state(self, snapshot: Mapping[str, Any]) -> None:
        if snapshot.get("snapshot_version") != self.SNAPSHOT_VERSION:
            raise ValueError("unsupported ItemEnv snapshot version")
        self.seed = int(snapshot["seed"])
        self.target_value = str(snapshot["target_value"])
        self.page_size = int(snapshot["page_size"])
        self.items = deepcopy(snapshot["items"])
        self.initial_items = deepcopy(snapshot["initial_items"])
        self.audit_log = deepcopy(snapshot["audit_log"])
        self.next_id = int(snapshot["next_id"])
        self.step_count = int(snapshot["step_count"])
        self.terminated = bool(snapshot["terminated"])
