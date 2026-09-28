"""执行前 f_A 动作规则。

本文件的 guard 只在调用发生时阻止危险 action，因此约束对 Agent 默认是隐藏的；若
辅助环境需要在调用前公开 required 参数，应使用 ``transforms.contract``。
"""

from __future__ import annotations

from typing import Any, Mapping

from env_agent_rsi.core.protocol import Action, EnvResponse, JsonObject
from env_agent_rsi.transforms.protocols import ActionDecision


class RequireArgumentRule:
    """Assistive f_A guard that blocks unsafe writes before side effects."""

    name = "require_argument"

    def __init__(
        self,
        tool: str,
        argument: str,
        error_code: str = "REQUIRED_ARGUMENT_MISSING",
        message: str | None = None,
    ):
        self.tool = tool
        self.argument = argument
        self.error_code = error_code
        self.message = message or f"{argument} is required for {tool}"
        self.block_count = 0

    def reset(self) -> None:
        self.block_count = 0

    def before_step(self, state: Mapping[str, Any], action: Action) -> ActionDecision:
        del state
        value = action.arguments.get(self.argument)
        if action.tool != self.tool or value not in (None, ""):
            return ActionDecision(action=action)
        self.block_count += 1
        return ActionDecision(
            action=action,
            response=EnvResponse(
                observation={
                    "ok": False,
                    "error": {
                        "code": self.error_code,
                        "message": self.message,
                        "required_argument": self.argument,
                    },
                },
                info={
                    "event": "action_blocked",
                    "state_changed": False,
                    "fault_events": [
                        {
                            "rule": self.name,
                            "phase": "before_step",
                            "tool": self.tool,
                        }
                    ],
                },
            ),
        )

    def save_state(self) -> JsonObject:
        return {"block_count": self.block_count}

    def load_state(self, state: Mapping[str, Any]) -> None:
        self.block_count = int(state["block_count"])
