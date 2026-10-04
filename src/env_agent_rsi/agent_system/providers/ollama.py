"""本地 Ollama Chat API 的 ModelClient 实现。

该 provider 保留 Target Agent 的 Agent0/Hermes 文本协议：工具注册已经位于 system
消息中，因此请求不再额外发送原生 tools；返回中的 ``<tool_call>`` 被规范化为统一
ModelOutput。它只用于本地 Target，Diagnostic 与 Modifier 仍由 DeepSeek API 执行。
"""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from env_agent_rsi.agent_runtime.agent0_protocol import (
    parse_tool_call,
    render_tool_call,
    render_tool_register,
)
from env_agent_rsi.agent_runtime.model import Message, ModelOutput
from env_agent_rsi.core.protocol import JsonObject


class OllamaModelClient:
    """调用本地 Ollama ``/api/chat`` 并解析单个 Hermes 工具动作。"""

    native_tool_transport = True

    def __init__(
        self,
        *,
        model: str,
        base_url: str = "http://127.0.0.1:11434",
        args: Mapping[str, Any] | None = None,
    ) -> None:
        if not model:
            raise ValueError("Ollama provider requires a model")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.args = deepcopy(dict(args or {}))

    def generate(
        self, messages: Sequence[Message], tools: Sequence[JsonObject]
    ) -> ModelOutput:
        timeout = float(self.args.get("timeout", 300.0))
        use_native_tools = bool(self.args.get("use_native_tools", True))
        request_messages = deepcopy(list(messages))
        if use_native_tools and tools:
            # ConversationContext 仍保存唯一 Agent0 注册；发给 Ollama 时去掉该后缀，
            # 让 Qwen 官方模板从 native tools 精确渲染一次，避免重复注册。
            registry = render_tool_register(tools)
            first = dict(request_messages[0])
            content = str(first.get("content", ""))
            suffix = "\n\n" + registry
            if content.endswith(suffix):
                first["content"] = content[: -len(suffix)]
                request_messages[0] = first
        payload: JsonObject = {
            "model": self.model,
            "messages": request_messages,
            "stream": False,
        }
        if use_native_tools and tools:
            payload["tools"] = deepcopy(list(tools))
        for key, value in self.args.items():
            if key not in {"timeout", "use_native_tools"}:
                payload[key] = deepcopy(value)
        request = Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Ollama returned HTTP {exc.code}: {detail}"
            ) from exc
        except URLError as exc:
            raise RuntimeError(f"Ollama request failed: {exc.reason}") from exc
        except TimeoutError as exc:
            raise RuntimeError(
                f"Ollama response timed out after {timeout:g} seconds"
            ) from exc
        return _parse_ollama_chat(result)


def _parse_ollama_chat(result: Mapping[str, Any]) -> ModelOutput:
    message = result.get("message")
    if not isinstance(message, Mapping):
        raise ValueError("Ollama response does not contain message")
    content = str(message.get("content") or "")
    thinking = str(message.get("thinking") or "").strip()
    if thinking and "<think>" not in content:
        content = f"<think>\n{thinking}\n</think>\n{content}"
    parsed, valid = parse_tool_call(content)
    if valid:
        return ModelOutput(
            tool_name=str(parsed["name"]),
            arguments=dict(parsed["arguments"]),
            content=content,
            call_id="ollama-tool-call",
            serialized_action=True,
        )
    calls = message.get("tool_calls") or []
    if not calls:
        return ModelOutput(tool_name=None, content=content)
    function = calls[0].get("function", {})
    arguments = function.get("arguments", {})
    if isinstance(arguments, str):
        arguments = json.loads(arguments or "{}")
    if not isinstance(arguments, Mapping):
        raise ValueError("Ollama tool-call arguments must be an object")
    name = str(function["name"])
    normalized = render_tool_call(name, arguments)
    return ModelOutput(
        tool_name=name,
        arguments=dict(arguments),
        content=f"<think>\n{thinking}\n</think>\n{normalized}" if thinking else normalized,
        call_id="ollama-tool-call",
        serialized_action=True,
    )
