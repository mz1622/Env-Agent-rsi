"""统一 Target Agent 与 Agent0 使用的 Qwen3 Hermes 工具协议。

本模块只负责把动态 JSON Schema 渲染进 ``<tools>``、把单个动作编码为
``<tool_call>``、解析模型动作，并把环境输出编码为 user-role 的
``<tool_response>``。内部环境仍只接收结构化 ``Action``，不会依赖 XML 文本。
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Sequence

from env_agent_rsi.core.protocol import JsonObject


TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def render_tool_register(tools: Sequence[Mapping[str, Any]]) -> str:
    """按 Qwen3 官方 Hermes 模板渲染工具定义和动作格式。"""

    schemas = "\n".join(
        json.dumps(dict(tool), ensure_ascii=False, sort_keys=True) for tool in tools
    )
    return (
        "# Tools\n\n"
        "You may call one function per turn to assist with the task.\n\n"
        "You are provided with function signatures within <tools></tools> XML tags:\n"
        f"<tools>\n{schemas}\n</tools>\n\n"
        "For each function call, return a JSON object with function name and "
        "arguments within <tool_call></tool_call> XML tags:\n"
        "<tool_call>\n"
        '{"name": <function-name>, "arguments": <args-json-object>}\n'
        "</tool_call>"
    )


def render_tool_call(name: str, arguments: Mapping[str, Any]) -> str:
    """把一个结构化动作编码为 Agent0/Qwen3 的单工具调用。"""

    payload = {"name": name, "arguments": dict(arguments)}
    return (
        "<tool_call>\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        + "\n</tool_call>"
    )


def parse_tool_call(
    content: str,
    *,
    supported_tools: Sequence[str] | set[str] | frozenset[str] | None = None,
) -> tuple[JsonObject, bool]:
    """解析恰好一个 Hermes 工具调用；思考文本可位于标签之前。"""

    matches = list(TOOL_CALL_PATTERN.finditer(content or ""))
    if len(matches) != 1:
        return {}, False
    try:
        value = json.loads(matches[0].group(1))
    except json.JSONDecodeError:
        return {}, False
    if not isinstance(value, dict):
        return {}, False
    name = value.get("name")
    arguments = value.get("arguments", {})
    if not isinstance(name, str) or not name or not isinstance(arguments, dict):
        return {}, False
    if supported_tools is not None and name not in supported_tools:
        return {}, False
    return {"name": name, "arguments": dict(arguments)}, True


def render_tool_response(observation: Any) -> str:
    """把环境结果编码为 Qwen3 下一轮所需的 ``<tool_response>``。"""

    if isinstance(observation, str):
        content = observation
    elif isinstance(observation, Mapping):
        content = json.dumps(dict(observation), ensure_ascii=False, sort_keys=True)
    else:
        content = json.dumps(observation, ensure_ascii=False, sort_keys=True)
    return f"<tool_response>\n{content}\n</tool_response>"
