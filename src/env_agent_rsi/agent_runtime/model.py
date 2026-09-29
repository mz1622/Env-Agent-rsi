"""模型调用边界与可复放实现。

ModelClient 接收标准 messages 和当前环境 tools，返回一个规范化 ModelOutput；具体
OpenAI、Anthropic 或本地模型 SDK 只需实现这一接口。ScriptedModelClient 用于测试
完整链路，CallableModelClient 用于零依赖接入外部 SDK。
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from env_agent_rsi.core.protocol import Action, JsonObject


Message = Mapping[str, Any]


@dataclass(frozen=True)
class ModelOutput:
    """一次模型响应中规范化的单个工具调用。"""

    tool_name: str | None
    arguments: Mapping[str, Any] = field(default_factory=dict)
    content: str = ""
    call_id: str = "tool-call"

    @classmethod
    def from_action(cls, action: Action, call_id: str = "tool-call") -> "ModelOutput":
        return cls(
            tool_name=action.tool,
            arguments=dict(action.arguments),
            call_id=call_id,
        )

    def to_assistant_message(self) -> JsonObject:
        message: JsonObject = {"role": "assistant", "content": self.content}
        if self.tool_name is not None:
            message["tool_calls"] = [
                {
                    "id": self.call_id,
                    "type": "function",
                    "function": {
                        "name": self.tool_name,
                        "arguments": json.dumps(
                            dict(self.arguments), ensure_ascii=False, sort_keys=True
                        ),
                    },
                }
            ]
        return message


@runtime_checkable
class ModelClient(Protocol):
    def generate(
        self,
        messages: Sequence[Message],
        tools: Sequence[JsonObject],
    ) -> ModelOutput:
        ...


class ScriptedModelClient:
    """按预定 action 顺序响应，同时记录每轮实际收到的工具契约。"""

    def __init__(self, actions: Sequence[Action]) -> None:
        self.actions = list(actions)
        self.index = 0
        self.requests: list[JsonObject] = []

    def generate(
        self, messages: Sequence[Message], tools: Sequence[JsonObject]
    ) -> ModelOutput:
        self.requests.append(
            {"messages": deepcopy(list(messages)), "tools": deepcopy(list(tools))}
        )
        if self.index >= len(self.actions):
            return ModelOutput(tool_name=None, content="script exhausted")
        action = self.actions[self.index]
        self.index += 1
        return ModelOutput.from_action(action, call_id=f"script-{self.index}")


class CallableModelClient:
    """把任意 SDK adapter 函数包装成 ModelClient。"""

    def __init__(
        self,
        generate_fn: Callable[[Sequence[Message], Sequence[JsonObject]], ModelOutput],
    ) -> None:
        self.generate_fn = generate_fn

    def generate(
        self, messages: Sequence[Message], tools: Sequence[JsonObject]
    ) -> ModelOutput:
        output = self.generate_fn(messages, tools)
        if not isinstance(output, ModelOutput):
            raise TypeError("model adapter must return ModelOutput")
        return output
