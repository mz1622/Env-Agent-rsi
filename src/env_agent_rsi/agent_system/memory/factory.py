"""根据统一 Agent 配置构建 MemoryRetriever。"""

from __future__ import annotations

from env_agent_rsi.agent_system.config import MemorySettings
from env_agent_rsi.agent_system.memory.json_store import JsonMemoryStore
from env_agent_rsi.agent_system.memory.protocol import MemoryRetriever, NullMemoryRetriever


def build_memory_retriever(settings: MemorySettings) -> MemoryRetriever:
    """把 null/json 配置转换为 Target 可用的只读检索器。"""

    if settings.type == "null":
        return NullMemoryRetriever()
    if settings.type == "json":
        if settings.path is None:
            raise ValueError("json memory requires a path")
        return JsonMemoryStore(settings.path)
    raise ValueError(f"unsupported memory type: {settings.type!r}")
