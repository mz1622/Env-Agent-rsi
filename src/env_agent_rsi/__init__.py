"""Env-Agent-RSI 顶层公共 API。

顶层只暴露最常用的环境协议、Agent runner 与 factory；实验实现仍保留在各自子包，
避免使用者依赖内部细节。
"""

from .agent_runtime import AgentRunner, ModelClient, ModelOutput
from .agent_system.diagnostic import DiagnosticAgent
from .agent_system.modifier import EnvironmentModificationAgent
from .agent_system.target import TargetAgent
from .benchmarks import BenchmarkAdapter, BenchmarkTask
from .core.protocol import (
    Action,
    ActionableEnv,
    EnvDescriptor,
    EnvResponse,
    EvaluationResult,
)
from .harness.factory import build_environment, load_spec
from .evolution import (
    BestFirstEnvironmentSearch,
    EnvironmentBucket,
    EnvironmentDAG,
    EvaluationBatch,
    FailureSignature,
    MutationSpec,
    MutationSurface,
)
from .orchestration import PairedRolloutEvaluator, TraceStore

__all__ = [
    "Action",
    "ActionableEnv",
    "AgentRunner",
    "BenchmarkAdapter",
    "BenchmarkTask",
    "DiagnosticAgent",
    "EnvDescriptor",
    "EnvResponse",
    "EvaluationResult",
    "EnvironmentDAG",
    "EnvironmentBucket",
    "EvaluationBatch",
    "BestFirstEnvironmentSearch",
    "EnvironmentModificationAgent",
    "FailureSignature",
    "ModelClient",
    "ModelOutput",
    "MutationSpec",
    "MutationSurface",
    "PairedRolloutEvaluator",
    "TraceStore",
    "TargetAgent",
    "build_environment",
    "load_spec",
]
