"""六类环境变化的声明式描述与稳定标识。

MutationSpec 是搜索器与执行器之间的边界：搜索器只能从固定的六个 phase 中选择，
参数保持 JSON 可序列化，稳定摘要用于复现、缓存和环境 DAG 去重。
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Mapping

from env_agent_rsi.core.protocol import JsonObject


MUTATION_PHASES = (
    "setup",
    "contract",
    "action",
    "transition",
    "observation",
    "budget",
)


@dataclass(frozen=True)
class MutationSpec:
    """一次最小环境变化，且只能属于一个执行阶段。"""

    phase: str
    implementation: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    rationale: str = ""

    def __post_init__(self) -> None:
        if self.phase not in MUTATION_PHASES:
            raise ValueError(
                f"unknown mutation phase {self.phase!r}; expected {MUTATION_PHASES!r}"
            )
        if not self.implementation:
            raise ValueError("mutation implementation must not be empty")
        try:
            json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)
        except TypeError as exc:
            raise ValueError("mutation parameters must be JSON serializable") from exc

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MutationSpec":
        parameters = value.get("parameters", {})
        if not isinstance(parameters, Mapping):
            raise ValueError("mutation parameters must be an object")
        return cls(
            phase=str(value["phase"]),
            implementation=str(value["implementation"]),
            parameters=deepcopy(dict(parameters)),
            rationale=str(value.get("rationale", "")),
        )

    def to_dict(self) -> JsonObject:
        return {
            "phase": self.phase,
            "implementation": self.implementation,
            "parameters": deepcopy(dict(self.parameters)),
            "rationale": self.rationale,
        }

    @property
    def digest(self) -> str:
        payload = json.dumps(
            self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:16]
