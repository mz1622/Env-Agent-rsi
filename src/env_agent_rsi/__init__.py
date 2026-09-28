"""Env-Agent-RSI 顶层公共 API。

顶层只暴露最常用的环境协议、Agent runner 与 factory；实验实现仍保留在各自子包，
避免使用者依赖内部细节。
"""

from .agent_runtime import AgentRunner, ModelClient, ModelOutput
from .core.protocol import (
    Action,
    ActionableEnv,
    EnvDescriptor,
    EnvResponse,
    EvaluationResult,
)
from .harness.factory import build_environment, load_spec

__all__ = [
    "Action",
    "ActionableEnv",
    "AgentRunner",
    "EnvDescriptor",
    "EnvResponse",
    "EvaluationResult",
    "ModelClient",
    "ModelOutput",
    "build_environment",
    "load_spec",
]
