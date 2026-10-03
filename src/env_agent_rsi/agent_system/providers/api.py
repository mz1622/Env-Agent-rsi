"""OpenAI-compatible HTTP API 的 ModelClient 实现。

该文件只处理网络协议和返回值规范化，不知道 Target/Diagnostic/Modifier 角色；使用标准库
HTTP，避免把某一家 SDK 变成项目的强制依赖。
"""

from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from env_agent_rsi.agent_runtime.model import Message, ModelOutput
from env_agent_rsi.agent_runtime.agent0_protocol import (
    parse_tool_call,
    render_tool_call,
)
from env_agent_rsi.core.protocol import JsonObject


class APIModelClient:
    """调用 OpenAI-compatible `/chat/completions` API。"""

    def __init__(
        self,
        *,
        model: str,
        base_url: str = "https://api.deepseek.com",
        api_key_env: str = "DEEPSEEK_API_KEY",
        api_key_file: str | Path | None = "api.txt",
        args: Mapping[str, Any] | None = None,
    ) -> None:
        if not model:
            raise ValueError("API provider requires a model")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.api_key_file = Path(api_key_file).resolve() if api_key_file else None
        self.args = deepcopy(dict(args or {}))

    def generate(
        self, messages: Sequence[Message], tools: Sequence[JsonObject]
    ) -> ModelOutput:
        api_key = self._load_api_key()
        timeout = float(self.args.get("timeout", 60.0))
        extra = {key: value for key, value in self.args.items() if key != "timeout"}
        payload: JsonObject = {
            "model": self.model,
            "messages": deepcopy(list(messages)),
            **extra,
        }
        if tools:
            payload["tools"] = deepcopy(list(tools))
        request = Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"model API returned HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"model API request failed: {exc.reason}") from exc
        except TimeoutError as exc:
            raise RuntimeError(
                f"model API response timed out after {timeout:g} seconds"
            ) from exc
        return _parse_chat_completion(result)

    def _load_api_key(self) -> str:
        """优先从受忽略的本地文件读取密钥，兼容环境变量回退。"""

        if self.api_key_file is not None:
            try:
                value = self.api_key_file.read_text(encoding="utf-8").strip()
            except FileNotFoundError:
                value = ""
            except OSError as exc:
                raise RuntimeError(
                    f"cannot read API key file: {self.api_key_file}"
                ) from exc
            if value:
                return value
        api_key = os.environ.get(self.api_key_env, "").strip()
        if api_key:
            return api_key
        raise RuntimeError(
            f"missing API key in {self.api_key_file} and environment variable "
            f"{self.api_key_env}"
        )


def _parse_chat_completion(result: Mapping[str, Any]) -> ModelOutput:
    try:
        message = result["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("API response does not contain choices[0].message") from exc
    content = str(message.get("content") or "")
    parsed, valid = parse_tool_call(content)
    if valid:
        return ModelOutput(
            tool_name=str(parsed["name"]),
            arguments=dict(parsed["arguments"]),
            content=content,
            call_id=str(message.get("id", "agent0-tool-call")),
            serialized_action=True,
        )
    calls = message.get("tool_calls") or []
    if not calls:
        return ModelOutput(tool_name=None, content=content)
    call = calls[0]
    function = call.get("function", {})
    arguments = function.get("arguments", {})
    if isinstance(arguments, str):
        arguments = json.loads(arguments or "{}")
    if not isinstance(arguments, Mapping):
        raise ValueError("tool call arguments must decode to an object")
    name = str(function["name"])
    normalized_content = render_tool_call(name, arguments)
    return ModelOutput(
        tool_name=name,
        arguments=dict(arguments),
        content=normalized_content,
        call_id=str(call.get("id", "tool-call")),
        serialized_action=True,
    )
