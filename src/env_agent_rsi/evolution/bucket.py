"""持久化环境树、完整环境配置与评测 archive。

EnvironmentBucket 把内容寻址的环境 spec、EnvironmentDAG、逐批 rollout 统计和当前
best 指针原子写入目录。失败候选不会被删除；后续 best-first 搜索从已验证的最佳节点
继续展开，同时保留所有父子边用于复现、回退和分析无效方向。
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Mapping, Sequence

from env_agent_rsi.core.protocol import JsonObject
from env_agent_rsi.evolution.lineage import EnvironmentDAG, EnvironmentNode
from env_agent_rsi.evolution.mutation import MutationSpec


BUCKET_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class EvaluationBatch:
    """同一环境版本的一批 rollout 聚合，不保存受保护的完整轨迹。"""

    success_rate: float
    verifier_score: float
    rollout_count: int
    mean_steps: float | None = None
    seeds: tuple[int, ...] = ()
    source: str = "evaluation"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.success_rate <= 1.0:
            raise ValueError("success_rate must be between zero and one")
        if not 0.0 <= self.verifier_score <= 1.0:
            raise ValueError("verifier_score must be between zero and one")
        if self.rollout_count <= 0:
            raise ValueError("rollout_count must be positive")
        if self.mean_steps is not None and self.mean_steps < 0:
            raise ValueError("mean_steps must be non-negative")
        if self.seeds and len(self.seeds) != self.rollout_count:
            raise ValueError("seeds length must match rollout_count")

    @classmethod
    def from_episodes(
        cls,
        episodes: Sequence[Any],
        *,
        seeds: Sequence[int] = (),
        source: str = "episode",
        metadata: Mapping[str, Any] | None = None,
    ) -> "EvaluationBatch":
        """从 EpisodeResult 或等价 JSON 中提取官方 success、partial score 与步数。"""

        if not episodes:
            raise ValueError("at least one episode is required")
        successes: list[float] = []
        verifier_scores: list[float] = []
        steps: list[float] = []
        for episode in episodes:
            value = episode.to_dict() if hasattr(episode, "to_dict") else episode
            if not isinstance(value, Mapping):
                raise TypeError("episode must be a mapping or expose to_dict")
            evaluation = value.get("evaluation", {})
            if not isinstance(evaluation, Mapping):
                raise ValueError("episode evaluation must be an object")
            success = bool(evaluation.get("success", False))
            metrics = evaluation.get("metrics", {})
            metrics = metrics if isinstance(metrics, Mapping) else {}
            passed = metrics.get("passed")
            total = metrics.get("total")
            if (
                isinstance(passed, (int, float))
                and not isinstance(passed, bool)
                and isinstance(total, (int, float))
                and not isinstance(total, bool)
                and total > 0
            ):
                partial = max(0.0, min(1.0, float(passed) / float(total)))
            else:
                partial = 1.0 if success else 0.0
            raw_steps = value.get("steps", 0)
            if not isinstance(raw_steps, (int, float)) or isinstance(raw_steps, bool):
                raise ValueError("episode steps must be numeric")
            successes.append(1.0 if success else 0.0)
            verifier_scores.append(partial)
            steps.append(float(raw_steps))
        count = len(episodes)
        return cls(
            success_rate=sum(successes) / count,
            verifier_score=sum(verifier_scores) / count,
            rollout_count=count,
            mean_steps=sum(steps) / count,
            seeds=tuple(int(seed) for seed in seeds),
            source=source,
            metadata=deepcopy(dict(metadata or {})),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvaluationBatch":
        seeds = value.get("seeds", [])
        metadata = value.get("metadata", {})
        if not isinstance(seeds, list) or not isinstance(metadata, Mapping):
            raise ValueError("invalid evaluation batch arrays or metadata")
        mean_steps = value.get("mean_steps")
        return cls(
            success_rate=float(value["success_rate"]),
            verifier_score=float(value["verifier_score"]),
            rollout_count=int(value["rollout_count"]),
            mean_steps=None if mean_steps is None else float(mean_steps),
            seeds=tuple(int(seed) for seed in seeds),
            source=str(value.get("source", "evaluation")),
            metadata=deepcopy(dict(metadata)),
        )

    def to_dict(self) -> JsonObject:
        return {
            "success_rate": self.success_rate,
            "verifier_score": self.verifier_score,
            "rollout_count": self.rollout_count,
            "mean_steps": self.mean_steps,
            "seeds": list(self.seeds),
            "source": self.source,
            "metadata": deepcopy(dict(self.metadata)),
        }


class EnvironmentBucket:
    """可恢复的环境 DAG archive，以及稳定的 best-node 选择器。"""

    MANIFEST_NAME = "bucket.json"

    def __init__(self, root: str | Path, *, scope: str | None = None) -> None:
        self.root = Path(root).expanduser().resolve()
        self.specs_dir = self.root / "specs"
        self.manifest_path = self.root / self.MANIFEST_NAME
        self._lock = Lock()
        self.scope = scope or "default"
        self.dag = EnvironmentDAG()
        self.evaluations: dict[str, list[EvaluationBatch]] = {}
        self.best_node_id: str | None = None
        self.best_history: list[JsonObject] = []
        self.root.mkdir(parents=True, exist_ok=True)
        self.specs_dir.mkdir(parents=True, exist_ok=True)
        if self.manifest_path.exists():
            self._load(scope)

    @property
    def has_best(self) -> bool:
        return self.best_node_id is not None

    def add_root(
        self,
        environment_spec: Mapping[str, Any],
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> EnvironmentNode:
        spec = deepcopy(dict(environment_spec))
        environment_hash = stable_environment_hash(spec)
        node = self.dag.add_root(environment_hash, metadata)
        self.evaluations.setdefault(node.node_id, [])
        self._write_spec(environment_hash, spec)
        self._persist()
        return node

    def add_candidate(
        self,
        parent_ids: Sequence[str],
        mutation: MutationSpec,
        environment_spec: Mapping[str, Any],
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> EnvironmentNode:
        spec = deepcopy(dict(environment_spec))
        environment_hash = stable_environment_hash(spec)
        node = self.dag.add_child(
            parent_ids,
            mutation,
            environment_hash,
            metadata,
        )
        self.evaluations.setdefault(node.node_id, [])
        self._write_spec(environment_hash, spec)
        self._persist()
        return node

    def record_evaluation(
        self, node_id: str, evaluation: EvaluationBatch
    ) -> EnvironmentNode:
        if node_id not in self.dag.nodes:
            raise KeyError(f"unknown environment node: {node_id!r}")
        self.evaluations.setdefault(node_id, []).append(evaluation)
        self._recompute_best()
        self._persist()
        return self.dag.nodes[node_id]

    def best_node(self) -> EnvironmentNode:
        if self.best_node_id is None:
            raise RuntimeError("bucket has no evaluated environment")
        return self.dag.nodes[self.best_node_id]

    def best_spec(self) -> JsonObject:
        return self.get_spec(self.best_node().node_id)

    def get_spec(self, node_id: str) -> JsonObject:
        try:
            node = self.dag.nodes[node_id]
        except KeyError as exc:
            raise KeyError(f"unknown environment node: {node_id!r}") from exc
        path = self._spec_path(node.environment_hash)
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise RuntimeError(f"missing environment spec for node {node_id}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"environment spec for node {node_id} must be an object")
        if stable_environment_hash(value) != node.environment_hash:
            raise ValueError(f"environment spec hash mismatch for node {node_id}")
        return value

    def aggregate(self, node_id: str) -> JsonObject | None:
        batches = self.evaluations.get(node_id, [])
        if not batches:
            return None
        rollouts = sum(batch.rollout_count for batch in batches)
        success_rate = sum(
            batch.success_rate * batch.rollout_count for batch in batches
        ) / rollouts
        verifier_score = sum(
            batch.verifier_score * batch.rollout_count for batch in batches
        ) / rollouts
        step_batches = [batch for batch in batches if batch.mean_steps is not None]
        step_rollouts = sum(batch.rollout_count for batch in step_batches)
        mean_steps = (
            sum(
                float(batch.mean_steps) * batch.rollout_count
                for batch in step_batches
            )
            / step_rollouts
            if step_rollouts
            else None
        )
        return {
            "success_rate": success_rate,
            "verifier_score": verifier_score,
            "mean_steps": mean_steps,
            "rollout_count": rollouts,
            "evaluation_batches": len(batches),
        }

    def to_dict(self) -> JsonObject:
        return {
            "schema_version": BUCKET_SCHEMA_VERSION,
            "scope": self.scope,
            "selection_policy": (
                "lexicographic(success_rate, verifier_score, successful_efficiency)"
            ),
            "best_node_id": self.best_node_id,
            "best_history": deepcopy(self.best_history),
            "dag": self.dag.to_dict(),
            "evaluations": {
                node_id: [batch.to_dict() for batch in batches]
                for node_id, batches in sorted(self.evaluations.items())
            },
            "aggregates": {
                node_id: self.aggregate(node_id)
                for node_id in sorted(self.dag.nodes)
            },
        }

    def _selection_key(self, node_id: str) -> tuple[float, float, float]:
        aggregate = self.aggregate(node_id)
        if aggregate is None:
            raise ValueError("unevaluated node has no selection key")
        # 步数只在已经成功时作为效率 tie-break；失败时避免奖励过早停止。
        efficiency = (
            -float(aggregate["mean_steps"])
            if aggregate["success_rate"] > 0
            and aggregate["mean_steps"] is not None
            else 0.0
        )
        return (
            float(aggregate["success_rate"]),
            float(aggregate["verifier_score"]),
            efficiency,
        )

    def _recompute_best(self) -> None:
        eligible = [node_id for node_id in self.dag.nodes if self.aggregate(node_id)]
        if not eligible:
            return
        best_key = max(self._selection_key(node_id) for node_id in eligible)
        tied = [node_id for node_id in eligible if self._selection_key(node_id) == best_key]
        previous = self.best_node_id
        selected = previous if previous in tied else sorted(tied)[0]
        self.best_node_id = selected
        if selected != previous:
            self.best_history.append(
                {
                    "changed_at": datetime.now(timezone.utc).isoformat(),
                    "previous_node_id": previous,
                    "best_node_id": selected,
                    "selection_key": list(best_key),
                }
            )

    def _load(self, requested_scope: str | None) -> None:
        value = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise ValueError("environment bucket manifest must be an object")
        if int(value.get("schema_version", 0)) != BUCKET_SCHEMA_VERSION:
            raise ValueError("unsupported environment bucket schema version")
        stored_scope = str(value.get("scope", "default"))
        if requested_scope is not None and requested_scope != stored_scope:
            raise ValueError(
                f"bucket scope mismatch: expected {requested_scope!r}, "
                f"found {stored_scope!r}"
            )
        self.scope = stored_scope
        raw_dag = value.get("dag", {})
        if not isinstance(raw_dag, Mapping):
            raise ValueError("bucket dag must be an object")
        self.dag = EnvironmentDAG.from_dict(raw_dag)
        raw_evaluations = value.get("evaluations", {})
        if not isinstance(raw_evaluations, Mapping):
            raise ValueError("bucket evaluations must be an object")
        self.evaluations = {}
        for node_id in self.dag.nodes:
            batches = raw_evaluations.get(node_id, [])
            if not isinstance(batches, list):
                raise ValueError("bucket node evaluations must be an array")
            if any(not isinstance(batch, Mapping) for batch in batches):
                raise ValueError("bucket evaluation batch must be an object")
            self.evaluations[node_id] = [
                EvaluationBatch.from_dict(batch) for batch in batches
            ]
            self.get_spec(node_id)
        raw_best = value.get("best_node_id")
        self.best_node_id = str(raw_best) if raw_best is not None else None
        if self.best_node_id is not None and self.best_node_id not in self.dag.nodes:
            raise ValueError("bucket best_node_id references an unknown node")
        history = value.get("best_history", [])
        if not isinstance(history, list) or any(
            not isinstance(item, Mapping) for item in history
        ):
            raise ValueError("bucket best_history must be an array of objects")
        self.best_history = [deepcopy(dict(item)) for item in history]
        expected = self.best_node_id
        self._recompute_best()
        if self.best_node_id != expected:
            raise ValueError("stored best_node_id disagrees with bucket evaluations")

    def _spec_path(self, environment_hash: str) -> Path:
        return self.specs_dir / f"{environment_hash}.json"

    def _write_spec(self, environment_hash: str, spec: Mapping[str, Any]) -> None:
        path = self._spec_path(environment_hash)
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if stable_environment_hash(existing) != environment_hash:
                raise ValueError("existing environment spec does not match its hash")
            return
        _atomic_json_write(path, dict(spec))

    def _persist(self) -> None:
        with self._lock:
            _atomic_json_write(self.manifest_path, self.to_dict())


def stable_environment_hash(spec: Mapping[str, Any]) -> str:
    """计算环境 spec 的稳定内容身份。"""

    payload = json.dumps(
        dict(spec), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_json_write(path: Path, value: Mapping[str, Any]) -> None:
    """先写同目录临时文件再原子替换，避免中断留下半份 manifest。"""

    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
