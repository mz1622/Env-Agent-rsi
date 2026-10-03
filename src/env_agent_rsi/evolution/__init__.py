"""环境演化对象的公共入口。

该包只保存“环境哪里可变、一次改了什么、失败属于哪一类、节点如何组成 DAG”这
些可审计数据，不负责直接调用模型或执行 benchmark。
"""

from env_agent_rsi.evolution.catalog import (
    MutationCatalog,
    MutationImplementation,
    default_mutation_catalog,
)
from env_agent_rsi.evolution.failure import FailureSignature, infer_failure_signature
from env_agent_rsi.evolution.lineage import EnvironmentDAG, EnvironmentNode
from env_agent_rsi.evolution.materializer import (
    canonicalize_environment_spec,
    materialize_environment_spec,
)
from env_agent_rsi.evolution.mutation import (
    CONTRACT_AXES,
    ENVHARNESS_COMPONENT_TYPES,
    MUTATION_OPERATIONS,
    MUTATION_PHASES,
    PHASE_CLASSIFICATION,
    MutationSpec,
)
from env_agent_rsi.evolution.surface import MutationSurface, ObservationChannel, ToolSemantics

__all__ = [
    "EnvironmentDAG",
    "EnvironmentNode",
    "FailureSignature",
    "CONTRACT_AXES",
    "ENVHARNESS_COMPONENT_TYPES",
    "MUTATION_OPERATIONS",
    "MUTATION_PHASES",
    "PHASE_CLASSIFICATION",
    "MutationCatalog",
    "MutationImplementation",
    "MutationSpec",
    "MutationSurface",
    "ObservationChannel",
    "ToolSemantics",
    "default_mutation_catalog",
    "canonicalize_environment_spec",
    "infer_failure_signature",
    "materialize_environment_spec",
]
