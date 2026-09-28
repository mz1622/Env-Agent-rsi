"""通用 Agent 执行层的公共导出。

本包把模型 provider 与环境解耦：ModelClient 只产生标准 tool call，AgentRunner
负责加载当前 descriptor、维护消息、执行 action、刷新动态契约并记录轨迹。
"""

from .model import CallableModelClient, ModelClient, ModelOutput, ScriptedModelClient
from .runner import AgentRunner, EpisodeResult

__all__ = [
    "AgentRunner",
    "CallableModelClient",
    "EpisodeResult",
    "ModelClient",
    "ModelOutput",
    "ScriptedModelClient",
]
