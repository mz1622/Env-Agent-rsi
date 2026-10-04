"""从持久化 bucket 当前最佳节点继续展开环境树。

BestFirstEnvironmentSearch 不负责让 LLM 生成 mutation，也不负责执行 rollout；它只保证
每次 expansion 的父节点来自 bucket 的当前 best，调用现有 materializer 得到唯一子
环境，并在评测回写后由 bucket 决定 best 是否移动。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from env_agent_rsi.core.protocol import JsonObject
from env_agent_rsi.evolution.bucket import EnvironmentBucket, EvaluationBatch
from env_agent_rsi.evolution.catalog import MutationCatalog
from env_agent_rsi.evolution.lineage import EnvironmentNode
from env_agent_rsi.evolution.materializer import materialize_environment_spec
from env_agent_rsi.evolution.mutation import MutationSpec
from env_agent_rsi.evolution.surface import MutationSurface


@dataclass(frozen=True)
class EnvironmentExpansion:
    """一次从 best 父节点生成、尚待评测的环境树边。"""

    parent_node_id: str
    child_node_id: str
    mutation: MutationSpec
    environment_spec: JsonObject


class BestFirstEnvironmentSearch:
    """以当前最优已评测环境为 expansion parent 的搜索控制器。"""

    def __init__(
        self,
        bucket: EnvironmentBucket,
        *,
        surface: MutationSurface,
        catalog: MutationCatalog,
    ) -> None:
        self.bucket = bucket
        self.surface = surface
        self.catalog = catalog

    def initialize(
        self,
        environment_spec: Mapping[str, Any],
        evaluation: EvaluationBatch,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> EnvironmentNode:
        """只在空 bucket 中加入并评测 root；已有树必须显式恢复。"""

        if self.bucket.dag.nodes:
            raise RuntimeError("environment search bucket is already initialized")
        root = self.bucket.add_root(environment_spec, metadata=metadata)
        self.bucket.record_evaluation(root.node_id, evaluation)
        return root

    def expand_best(
        self,
        mutation: MutationSpec,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> EnvironmentExpansion:
        """从当前 best 物化一个 child；未评测 child 不会改变 best。"""

        parent = self.bucket.best_node()
        parent_spec = self.bucket.get_spec(parent.node_id)
        child_spec = materialize_environment_spec(
            parent_spec,
            mutation,
            surface=self.surface,
            catalog=self.catalog,
        )
        child = self.bucket.add_candidate(
            [parent.node_id],
            mutation,
            child_spec,
            metadata=metadata,
        )
        return EnvironmentExpansion(
            parent_node_id=parent.node_id,
            child_node_id=child.node_id,
            mutation=mutation,
            environment_spec=child_spec,
        )

    def record_result(
        self, expansion: EnvironmentExpansion, evaluation: EvaluationBatch
    ) -> EnvironmentNode:
        """回写 child 评测；只有严格更优时 best 才移动。"""

        return self.bucket.record_evaluation(expansion.child_node_id, evaluation)
