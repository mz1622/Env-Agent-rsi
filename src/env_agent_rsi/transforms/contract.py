"""Agent 可见工具契约的变换实现。

这里的规则同时修改 tool schema 和执行前校验，解决“模型看到 optional、环境实际
要求 required”的不一致。Contract 规则属于显式环境辅助；若希望约束保持隐藏，
应使用 ``transforms.action`` 中只在调用后反馈错误的 f_A 规则。
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from env_agent_rsi.core.protocol import Action, EnvDescriptor, EnvResponse, JsonObject
from env_agent_rsi.core.tooling import replace_tools


class RequireArgumentContractRule:
    """把已有工具参数标记为必填，并在执行时使用同一约束。"""

    name = "require_argument"

    def __init__(
        self,
        tool: str,
        argument: str,
        error_code: str = "REQUIRED_ARGUMENT_MISSING",
        message: str | None = None,
    ) -> None:
        self.tool = tool
        self.argument = argument
        self.error_code = error_code
        self.message = message or f"{argument} is required for {tool}"
        self.block_count = 0

    def reset(self) -> None:
        self.block_count = 0

    def transform_descriptor(
        self, descriptor: EnvDescriptor, state: Mapping[str, Any]
    ) -> EnvDescriptor:
        del state
        tools = deepcopy(list(descriptor.tools))
        matched = False
        for schema in tools:
            function = schema.get("function", {})
            if function.get("name") != self.tool:
                continue
            parameters = function.setdefault("parameters", {})
            properties = parameters.setdefault("properties", {})
            if self.argument not in properties:
                raise ValueError(
                    f"contract argument {self.argument!r} is absent from tool {self.tool!r}"
                )
            required = list(parameters.get("required", []))
            if self.argument not in required:
                required.append(self.argument)
            parameters["required"] = required
            property_schema = properties[self.argument]
            property_schema.setdefault("description", self.message)
            matched = True
        if not matched:
            raise ValueError(
                f"contract tool {self.tool!r} is not in the base descriptor"
            )
        return replace_tools(descriptor, tools)

    def validate_action(
        self, state: Mapping[str, Any], action: Action
    ) -> EnvResponse | None:
        del state
        value = action.arguments.get(self.argument)
        if action.tool != self.tool or value not in (None, ""):
            return None
        self.block_count += 1
        return EnvResponse(
            observation={
                "ok": False,
                "error": {
                    "code": self.error_code,
                    "message": self.message,
                    "required_argument": self.argument,
                },
            },
            info={
                "event": "contract_blocked",
                "state_changed": False,
                "fault_events": [
                    {
                        "rule": self.name,
                        "phase": "contract_validation",
                        "tool": self.tool,
                    }
                ],
            },
        )

    def save_state(self) -> JsonObject:
        return {"block_count": self.block_count}

    def load_state(self, state: Mapping[str, Any]) -> None:
        self.block_count = int(state["block_count"])
