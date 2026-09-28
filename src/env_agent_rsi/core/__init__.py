"""核心公共 API 聚合模块。

这里只重导出环境协议、注册表和工具契约等稳定类型，业务环境和 Agent runtime
不得反向写入本层，从而维持依赖方向清晰。
"""

from .protocol import (
    Action,
    ActionableEnv,
    EnvDescriptor,
    EnvResponse,
    EvaluationResult,
)
from .registry import ComponentRegistry
from .tooling import (
    ActionValidationError,
    make_descriptor,
    tool_schema,
    validate_action,
)
from .verifier import StateVerifier

__all__ = [
    "Action",
    "ActionableEnv",
    "ComponentRegistry",
    "EnvDescriptor",
    "EnvResponse",
    "EvaluationResult",
    "StateVerifier",
    "ActionValidationError",
    "make_descriptor",
    "tool_schema",
    "validate_action",
]
