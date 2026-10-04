"""根据统一 ProviderSettings 创建模型客户端。

Agent 只传入同一份配置对象；本工厂是 API 与 ADK 分支唯一出现的位置，并允许测试
通过 client_override 完全绕过真实网络调用。
"""

from __future__ import annotations

from typing import Any, Mapping

from env_agent_rsi.agent_runtime.model import ModelClient
from env_agent_rsi.agent_system.config import ProviderSettings
from env_agent_rsi.agent_system.providers.adk import ADKModelClient
from env_agent_rsi.agent_system.providers.api import APIModelClient
from env_agent_rsi.agent_system.providers.ollama import OllamaModelClient


def build_model_client(
    settings: ProviderSettings,
    *,
    args_override: Mapping[str, Any] | None = None,
) -> ModelClient:
    args = dict(settings.args)
    args.update(dict(args_override or {}))
    if settings.type == "api":
        return APIModelClient(
            model=settings.model,
            base_url=settings.base_url or "https://api.deepseek.com",
            api_key_env=settings.api_key_env or "DEEPSEEK_API_KEY",
            api_key_file=settings.api_key_file or "api.txt",
            args=args,
        )
    if settings.type == "adk":
        if not settings.executor:
            raise ValueError("ADK provider requires executor")
        return ADKModelClient(
            model=settings.model,
            executor=settings.executor,
            args=args,
        )
    if settings.type == "ollama":
        return OllamaModelClient(
            model=settings.model,
            base_url=settings.base_url or "http://127.0.0.1:11434",
            args=args,
        )
    raise ValueError(f"unknown provider type: {settings.type!r}")
