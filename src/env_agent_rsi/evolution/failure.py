"""从 episode 轨迹提取可用于环境搜索的失败签名。

这里先提供确定性归因，保证没有 Diagnostic Agent 时系统仍可运行；模型诊断只负责
在同一结构上补充解释，不能修改 verifier 给出的成败事实。
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from env_agent_rsi.core.protocol import EvaluationResult, JsonObject
from env_agent_rsi.evolution.mutation import MUTATION_PHASES


@dataclass(frozen=True)
class FailureSignature:
    """诊断输出的稳定结构，可直接作为环境搜索条件。"""

    category: str
    phase: str
    summary: str
    evidence: tuple[JsonObject, ...] = ()
    confidence: float = 0.0
    candidate_changes: tuple[JsonObject, ...] = ()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FailureSignature":
        phase = str(value.get("phase", "observation"))
        if phase not in {"agent", "none", *MUTATION_PHASES}:
            raise ValueError(f"unknown failure phase: {phase}")
        return cls(
            category=str(value.get("category", "unknown")),
            phase=phase,
            summary=str(value.get("summary", "")),
            evidence=tuple(deepcopy(list(value.get("evidence", ())))),
            confidence=float(value.get("confidence", 0.0)),
            candidate_changes=tuple(
                deepcopy(list(value.get("candidate_changes", ())))
            ),
        )

    def to_dict(self) -> JsonObject:
        return {
            "category": self.category,
            "phase": self.phase,
            "summary": self.summary,
            "evidence": deepcopy(list(self.evidence)),
            "confidence": self.confidence,
            "candidate_changes": deepcopy(list(self.candidate_changes)),
        }


def infer_failure_signature(
    trace: Sequence[Mapping[str, Any]], evaluation: EvaluationResult
) -> FailureSignature:
    """依据稳定错误码和真实评估结果生成保守的规则诊断。"""

    if evaluation.success:
        return FailureSignature(
            category="success",
            phase="none",
            summary="verifier reported success",
            confidence=1.0,
        )

    errors: list[JsonObject] = []
    writes: list[JsonObject] = []
    for item in trace:
        action = dict(item.get("action", {}))
        response = dict(item.get("response", {}))
        observation = dict(response.get("observation", {}))
        info = dict(response.get("info", {}))
        if "error" in observation or info.get("error_code"):
            errors.append(
                {
                    "step": item.get("step"),
                    "tool": action.get("tool"),
                    "error": deepcopy(observation.get("error")),
                    "event": info.get("event"),
                }
            )
        if info.get("state_changed") or info.get("committed"):
            writes.append(
                {"step": item.get("step"), "tool": action.get("tool")}
            )

    if len(writes) > 1:
        return FailureSignature(
            category="repeated_side_effect",
            phase="transition",
            summary="multiple state-changing actions preceded verifier failure",
            evidence=tuple(writes),
            confidence=0.85,
            candidate_changes=(
                {
                    "phase": "contract",
                    "implementation": "require_argument",
                    "parameters": {"argument": "idempotency_key"},
                },
                {
                    "phase": "observation",
                    "implementation": "read_after_write_confirmation",
                    "parameters": {},
                },
            ),
        )
    if errors:
        return FailureSignature(
            category="tool_error_unrecovered",
            phase="action",
            summary="the episode ended after one or more tool errors",
            evidence=tuple(errors[-3:]),
            confidence=0.7,
            candidate_changes=(
                {
                    "phase": "contract",
                    "implementation": "clarify_tool_contract",
                    "parameters": {},
                },
            ),
        )
    return FailureSignature(
        category="goal_not_reached",
        phase="agent",
        summary=evaluation.reason,
        evidence=({"metrics": deepcopy(dict(evaluation.metrics))},),
        confidence=0.4,
    )
