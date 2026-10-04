"""环境版本的多父节点谱系 DAG。

同一个物化环境可能由不同变化顺序得到，因此结构使用内容哈希去重并允许多条父边；
边保存具体 MutationSpec，节点只保存可复现的环境身份与评估摘要。
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from env_agent_rsi.core.protocol import JsonObject
from env_agent_rsi.evolution.mutation import MutationSpec


@dataclass
class EnvironmentNode:
    """一个已物化、可运行的环境版本。"""

    node_id: str
    environment_hash: str
    parents: set[str] = field(default_factory=set)
    metadata: JsonObject = field(default_factory=dict)

    def to_dict(self) -> JsonObject:
        return {
            "node_id": self.node_id,
            "environment_hash": self.environment_hash,
            "parents": sorted(self.parents),
            "metadata": deepcopy(self.metadata),
        }


class EnvironmentDAG:
    """支持内容去重、环检测和多父边的环境谱系容器。"""

    def __init__(self) -> None:
        self.nodes: dict[str, EnvironmentNode] = {}
        self.edges: list[JsonObject] = []

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EnvironmentDAG":
        """从持久化 JSON 恢复 DAG，并重新执行边与环校验。"""

        raw_nodes = value.get("nodes", [])
        raw_edges = value.get("edges", [])
        if not isinstance(raw_nodes, list) or not isinstance(raw_edges, list):
            raise ValueError("environment DAG nodes and edges must be arrays")
        dag = cls()
        declared_parents: dict[str, set[str]] = {}
        for raw_node in raw_nodes:
            if not isinstance(raw_node, Mapping):
                raise ValueError("environment DAG node must be an object")
            environment_hash = str(raw_node["environment_hash"])
            node = dag.add_root(environment_hash, raw_node.get("metadata", {}))
            if str(raw_node.get("node_id", "")) != node.node_id:
                raise ValueError("environment DAG node id does not match hash")
            parents = raw_node.get("parents", [])
            if not isinstance(parents, list):
                raise ValueError("environment DAG parents must be an array")
            declared_parents[node.node_id] = {str(parent) for parent in parents}
        for raw_edge in raw_edges:
            if not isinstance(raw_edge, Mapping):
                raise ValueError("environment DAG edge must be an object")
            child_id = str(raw_edge["child"])
            child = dag.nodes.get(child_id)
            if child is None:
                raise KeyError(f"unknown child node in DAG edge: {child_id!r}")
            mutation = MutationSpec.from_dict(dict(raw_edge["mutation"]))
            if raw_edge.get("mutation_id") != mutation.digest:
                raise ValueError("environment DAG mutation id mismatch")
            dag.add_child(
                [str(raw_edge["parent"])],
                mutation,
                child.environment_hash,
            )
        for node_id, parents in declared_parents.items():
            if dag.nodes[node_id].parents != parents:
                raise ValueError("environment DAG declared parents disagree with edges")
        return dag

    def add_root(
        self, environment_hash: str, metadata: Mapping[str, Any] | None = None
    ) -> EnvironmentNode:
        return self._get_or_create(environment_hash, metadata)

    def add_child(
        self,
        parent_ids: Sequence[str],
        mutation: MutationSpec,
        environment_hash: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> EnvironmentNode:
        if not parent_ids:
            raise ValueError("a child environment needs at least one parent")
        missing = [parent for parent in parent_ids if parent not in self.nodes]
        if missing:
            raise KeyError(f"unknown parent nodes: {missing!r}")
        node = self._get_or_create(environment_hash, metadata)
        if node.node_id in parent_ids:
            raise ValueError("an environment node cannot be its own parent")
        for parent in parent_ids:
            if self._can_reach(node.node_id, parent):
                raise ValueError("edge would create a cycle")
            node.parents.add(parent)
            edge = {
                "parent": parent,
                "child": node.node_id,
                "mutation": mutation.to_dict(),
                "mutation_id": mutation.digest,
            }
            if edge not in self.edges:
                self.edges.append(edge)
        return node

    def _get_or_create(
        self, environment_hash: str, metadata: Mapping[str, Any] | None
    ) -> EnvironmentNode:
        if not environment_hash:
            raise ValueError("environment_hash must not be empty")
        node_id = environment_hash[:16]
        existing = self.nodes.get(node_id)
        if existing is not None:
            if existing.environment_hash != environment_hash:
                raise ValueError("environment hash prefix collision")
            if metadata:
                existing.metadata.update(deepcopy(dict(metadata)))
            return existing
        node = EnvironmentNode(
            node_id=node_id,
            environment_hash=environment_hash,
            metadata=deepcopy(dict(metadata or {})),
        )
        self.nodes[node_id] = node
        return node

    def _can_reach(self, start: str, target: str) -> bool:
        if start == target:
            return True
        children: dict[str, set[str]] = {}
        for edge in self.edges:
            children.setdefault(str(edge["parent"]), set()).add(str(edge["child"]))
        pending = list(children.get(start, ()))
        seen: set[str] = set()
        while pending:
            current = pending.pop()
            if current == target:
                return True
            if current not in seen:
                seen.add(current)
                pending.extend(children.get(current, ()))
        return False

    def to_dict(self) -> JsonObject:
        return {
            "nodes": [self.nodes[key].to_dict() for key in sorted(self.nodes)],
            "edges": deepcopy(self.edges),
        }
