"""由同一配置结构装配不同角色 Agent 的公共基类。

该类加载 JSON prompt、skills 和 provider 参数；角色子类只负责 episode 或诊断行为，
从而确保 API、ADK、脚本模型都走同一 ModelClient 边界。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from env_agent_rsi.agent_runtime.model import ModelClient
from env_agent_rsi.agent_system.config import AgentConfig, load_agent_config
from env_agent_rsi.agent_system.context import ConversationContext
from env_agent_rsi.agent_system.memory import MemoryRetriever, build_memory_retriever
from env_agent_rsi.agent_system.prompts import PromptSpec, SkillSpec, load_prompt, load_skill
from env_agent_rsi.agent_system.providers.factory import build_model_client


class ConfiguredAgent:
    """Target、Diagnostic 与 Modifier 共享的资源和 provider 装配。"""

    def __init__(
        self,
        config: AgentConfig,
        model: ModelClient | None = None,
        memory: MemoryRetriever | None = None,
    ) -> None:
        self.config = config
        self.prompt: PromptSpec = load_prompt(config.system_prompt)
        self.skills: tuple[SkillSpec, ...] = tuple(
            load_skill(path) for path in config.skills
        )
        self.model = model or build_model_client(config.provider)
        self.memory = memory or build_memory_retriever(config.memory)

    @classmethod
    def from_config(
        cls,
        path: str | Path,
        *,
        model: ModelClient | None = None,
        memory: MemoryRetriever | None = None,
        provider_overrides: Mapping[str, Any] | None = None,
    ) -> "ConfiguredAgent":
        return cls(
            load_agent_config(path, provider_overrides=provider_overrides),
            model=model,
            memory=memory,
        )

    def new_context(self) -> ConversationContext:
        return ConversationContext(
            self.prompt.content,
            [skill.render() for skill in self.skills],
            max_messages=self.config.max_context_messages,
        )
