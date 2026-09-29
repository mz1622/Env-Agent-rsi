"""ADK 执行器到 ModelClient 的轻量桥接。

不同 ADK 的 session/run 方式差异很大，因此配置给出一个 `模块:函数` executor；桥接
层负责动态载入、传递统一参数和规范化返回值，而不会让 ADK 依赖渗入 Agent 结构。
"""

from __future__ import annotations

import importlib
import json
from copy import deepcopy
from typing import Any, Callable, Mapping, Sequence

from env_agent_rsi.agent_runtime.model import Message, ModelOutput
from env_agent_rsi.core.protocol import JsonObject


class ADKModelClient:
    """调用用户配置的 ADK executor。"""

    def __init__(
        self,
        *,
        model: str,
        executor: str | Callable[..., Any],
        args: Mapping[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.executor = _load_executor(executor) if isinstance(executor, str) else executor
        self.args = deepcopy(dict(args or {}))

    def generate(
        self, messages: Sequence[Message], tools: Sequence[JsonObject]
    ) -> ModelOutput:
        result = self.executor(
            model=self.model,
            messages=deepcopy(list(messages)),
            tools=deepcopy(list(tools)),
            **deepcopy(self.args),
        )
        if isinstance(result, ModelOutput):
            return result
        if not isinstance(result, Mapping):
            raise TypeError("ADK executor must return ModelOutput or an object")
        arguments = result.get("arguments", {})
        if isinstance(arguments, str):
            arguments = json.loads(arguments or "{}")
        return ModelOutput(
            tool_name=(
                str(result["tool_name"]) if result.get("tool_name") is not None else None
            ),
            arguments=dict(arguments),
            content=str(result.get("content", "")),
            call_id=str(result.get("call_id", "adk-tool-call")),
        )


def _load_executor(path: str) -> Callable[..., Any]:
    if ":" not in path:
        raise ValueError("ADK executor must use 'module:function' format")
    module_name, function_name = path.split(":", 1)
    value = getattr(importlib.import_module(module_name), function_name)
    if not callable(value):
        raise TypeError(f"ADK executor {path!r} is not callable")
    return value
