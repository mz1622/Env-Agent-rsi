"""定义 Agent 长期 Memory 的只读检索边界。

Memory 位于模型上下文之外；Target 在 episode 开始时用任务和初始观察构造查询，只把
返回的 JSON 记录注入当前任务。这里暂不定义写入、自我总结或在线更新接口。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, runtime_checkable

from env_agent_rsi.core.protocol import JsonObject


@dataclass(frozen=True)
class MemoryQuery:
    """一次只读检索所需的最小任务上下文。"""

    task_id: str
    instruction: str
    initial_observation: Mapping[str, object]

    def to_dict(self) -> JsonObject:
        return {
            "task_id": self.task_id,
            "instruction": self.instruction,
            "initial_observation": dict(self.initial_observation),
        }


@runtime_checkable
class MemoryRetriever(Protocol):
    """可替换的只读 Memory 检索器。"""

    def retrieve(self, query: MemoryQuery, *, limit: int) -> tuple[JsonObject, ...]:
        ...


class NullMemoryRetriever:
    """未配置 Memory 时使用的空实现。"""

    def retrieve(self, query: MemoryQuery, *, limit: int) -> tuple[JsonObject, ...]:
        del query, limit
        return ()
