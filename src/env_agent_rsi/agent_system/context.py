"""按 Agent0/Qwen3 Hermes 协议维护 Target Agent 多轮上下文。

上下文始终保留 system、可信 Skill Register、动态 ``<tools>`` 注册和初始任务。后续
按 assistant ``<tool_call>`` 与 user ``<tool_response>`` 成对追加，裁剪时不留下孤立
环境返回；工具契约变化时只重建 system 消息中的注册段。
"""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, Mapping, Sequence

from env_agent_rsi.agent_runtime.agent0_protocol import (
    render_tool_register,
    render_tool_response,
)
from env_agent_rsi.agent_runtime.model import Message, ModelOutput
from env_agent_rsi.core.protocol import JsonObject


class ConversationContext:
    """可裁剪但保持工具调用配对关系的消息容器。"""

    def __init__(
        self,
        system_prompt: str,
        skill_texts: Sequence[str] = (),
        tools: Sequence[Mapping[str, Any]] | None = None,
        *,
        max_messages: int = 80,
    ) -> None:
        if max_messages < 4:
            raise ValueError("max_messages must be at least 4")
        self._base_sections = [system_prompt.strip()]
        rendered_skills = [text.strip() for text in skill_texts if text.strip()]
        if rendered_skills:
            self._base_sections.append(
                "[Skill Register]\n" + "\n\n".join(rendered_skills)
            )
        self.max_messages = max_messages
        self._tools: tuple[JsonObject, ...] = ()
        self._messages: list[JsonObject] = [
            {"role": "system", "content": "\n\n".join(self._base_sections)}
        ]
        self._tool_protocol_enabled = tools is not None
        if tools is not None:
            self.bind_tools(tools)

    def bind_tools(self, tools: Sequence[Mapping[str, Any]]) -> None:
        """把当前环境契约写入 system；内容未变化时不重复构造。"""

        normalized = tuple(deepcopy(dict(tool)) for tool in tools)
        if self._tool_protocol_enabled and normalized == self._tools:
            return
        self._tool_protocol_enabled = True
        self._tools = normalized
        sections = [*self._base_sections, render_tool_register(normalized)]
        self._messages[0] = {"role": "system", "content": "\n\n".join(sections)}

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

    def append_tool_result(self, observation: Mapping[str, Any] | str) -> None:
        self._messages.append(
            {
                "role": "user",
                "content": render_tool_response(observation),
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
            # 前两个位置固定为 system 与原始任务，其后为 assistant/user 工具交互对。
            if self._messages[2].get("role") == "assistant":
                del self._messages[2]
                if len(self._messages) > 2 and self._messages[2].get("role") == "user":
                    del self._messages[2]
            else:
                del self._messages[2]
