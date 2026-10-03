"""Baseline 与 candidate 使用相同 seed 的成对多次评估。

本模块不决定候选是否接受，只保证比较协议：两个环境使用完全相同的 seed 列表，每次
执行产生一条完整记录，并可选写入 TraceStore。最终 acceptance gate 可以独立消费结果。
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from typing import Any, Callable, Mapping, Sequence

from env_agent_rsi.core.protocol import JsonObject
from env_agent_rsi.orchestration.storage import TraceStore


EpisodeExecutor = Callable[[Mapping[str, Any], int], Any]


@dataclass(frozen=True)
class RolloutRecord:
    """一条带有环境身份和随机种子的完整 episode。"""

    environment_role: str
    environment_hash: str
    seed: int
    rollout_index: int
    result: JsonObject

    def to_dict(self) -> JsonObject:
        return {
            "environment_role": self.environment_role,
            "environment_hash": self.environment_hash,
            "seed": self.seed,
            "rollout_index": self.rollout_index,
            "result": deepcopy(self.result),
        }


@dataclass(frozen=True)
class PairedRolloutResult:
    """顺序对齐的 baseline/candidate 评估结果。"""

    baseline: tuple[RolloutRecord, ...]
    candidate: tuple[RolloutRecord, ...]

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(record.seed for record in self.baseline)

    def to_dict(self) -> JsonObject:
        return {
            "seeds": list(self.seeds),
            "baseline": [record.to_dict() for record in self.baseline],
            "candidate": [record.to_dict() for record in self.candidate],
        }


class PairedRolloutEvaluator:
    """以相同 seed 集运行两个环境版本。"""

    def __init__(
        self,
        executor: EpisodeExecutor,
        *,
        trace_store: TraceStore | None = None,
    ) -> None:
        self.executor = executor
        self.trace_store = trace_store

    def compare(
        self,
        baseline_spec: Mapping[str, Any],
        candidate_spec: Mapping[str, Any],
        *,
        seeds: Sequence[int],
    ) -> PairedRolloutResult:
        if not seeds:
            raise ValueError("paired rollout evaluation requires at least one seed")
        baseline = self._run_role("baseline", baseline_spec, seeds)
        candidate = self._run_role("candidate", candidate_spec, seeds)
        if tuple(item.seed for item in baseline) != tuple(item.seed for item in candidate):
            raise RuntimeError("paired rollout seeds diverged")
        return PairedRolloutResult(tuple(baseline), tuple(candidate))

    def _run_role(
        self,
        role: str,
        spec: Mapping[str, Any],
        seeds: Sequence[int],
    ) -> list[RolloutRecord]:
        environment_hash = _stable_hash(spec)
        records: list[RolloutRecord] = []
        for rollout_index, seed in enumerate(seeds):
            raw = self.executor(deepcopy(dict(spec)), int(seed))
            if hasattr(raw, "to_dict"):
                raw = raw.to_dict()
            if not isinstance(raw, Mapping):
                raise TypeError("episode executor must return a mapping or to_dict object")
            record = RolloutRecord(
                environment_role=role,
                environment_hash=environment_hash,
                seed=int(seed),
                rollout_index=rollout_index,
                result=deepcopy(dict(raw)),
            )
            records.append(record)
            if self.trace_store is not None:
                self.trace_store.append(record.to_dict())
        return records


def _stable_hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        dict(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
