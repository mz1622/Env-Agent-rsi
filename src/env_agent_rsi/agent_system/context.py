"""支持连续多轮工具调用的 Agent 上下文。

上下文始终保留 system、skills 和初始任务；后续按 assistant tool-call 与 tool result
成对追加。超过上限时只从最旧的完整交互轮开始裁剪，避免留下孤立 tool 消息。
"""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, Mapping, Sequence

from env_agent_rsi.agent_runtime.model import Message, ModelOutput
from env_agent_rsi.core.protocol import JsonObject


class ConversationContext:
    """可裁剪但保持工具调用配对关系的消息容器。"""

    def __init__(
        self,
        system_prompt: str,
        skill_texts: Sequence[str] = (),
        *,
        max_messages: int = 80,
    ) -> None:
        if max_messages < 4:
            raise ValueError("max_messages must be at least 4")
        sections = [system_prompt.strip()]
        sections.extend(text.strip() for text in skill_texts if text.strip())
        self.max_messages = max_messages
        self._messages: list[JsonObject] = [
            {"role": "system", "content": "\n\n".join(sections)}
        ]

    def start_task(self, payload: Mapping[str, Any] | str) -> None:
        if len(self._messages) != 1:
            raise RuntimeError("task context has already been started")
        content = (
            payload
            if isinstance(payload, str)
            else json.dumps(payload, ensure_ascii=False, sort_keys=True)
        )
        self._messages.append({"role": "user", "content": content})

    def append_assistant(self, output: ModelOutput) -> None:
        self._messages.append(output.to_assistant_message())
        self._trim_complete_turns()

    def append_tool_result(
        self, call_id: str, tool_name: str, observation: Mapping[str, Any]
    ) -> None:
        self._messages.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "name": tool_name,
                "content": json.dumps(
                    dict(observation), ensure_ascii=False, sort_keys=True
                ),
            }
        )
        self._trim_complete_turns()

    @property
    def messages(self) -> tuple[JsonObject, ...]:
        return tuple(deepcopy(self._messages))

    def for_model(self) -> Sequence[Message]:
        return self.messages

    def _trim_complete_turns(self) -> None:
        while len(self._messages) > self.max_messages and len(self._messages) > 4:
            # 前两个位置固定为 system 与原始任务，其后每轮通常为 assistant/tool。
            if self._messages[2].get("role") == "assistant":
                del self._messages[2]
                if len(self._messages) > 2 and self._messages[2].get("role") == "tool":
                    del self._messages[2]
            else:
                del self._messages[2]
