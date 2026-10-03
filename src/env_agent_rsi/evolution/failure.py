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
    """诊断输出的唯一最高优先级根因与环境改动方向。"""

    category: str
    phase: str
    summary: str
    evidence: tuple[JsonObject, ...] = ()
    confidence: float = 0.0
    environment_actionable: bool = False
    primary_change: JsonObject | None = None
    priority_reason: str = ""
    causal_chain: tuple[str, ...] = ()
    expected_effect: str = ""
    falsification_condition: str = ""
    regression_guards: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FailureSignature":
        phase = str(value.get("phase", "observation"))
        if phase not in {"agent", "none", *MUTATION_PHASES}:
            raise ValueError(f"unknown failure phase: {phase}")
        raw_actionable = value.get(
            "environment_actionable", phase in MUTATION_PHASES
        )
        if not isinstance(raw_actionable, bool):
            raise ValueError("environment_actionable must be a boolean")
        actionable = raw_actionable
        raw_change = value.get("primary_change")
        if raw_change is not None and not isinstance(raw_change, Mapping):
            raise ValueError("primary_change must be an object or null")
        if actionable and raw_change is None:
            raise ValueError(
                "environment-actionable diagnosis requires exactly one primary_change"
            )
        raw_evidence = value.get("evidence", ())
        if not isinstance(raw_evidence, (list, tuple)) or any(
            not isinstance(item, Mapping) for item in raw_evidence
        ):
            raise ValueError("evidence must be an array of JSON objects")
        confidence = float(value.get("confidence", 0.0))
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        causal_chain = _string_tuple(value.get("causal_chain", ()), "causal_chain")
        regression_guards = _string_tuple(
            value.get("regression_guards", ()), "regression_guards"
        )
        return cls(
            category=str(value.get("category", "unknown")),
            phase=phase,
            summary=str(value.get("summary", "")),
            evidence=tuple(deepcopy(list(raw_evidence))),
            confidence=confidence,
            environment_actionable=actionable,
            primary_change=(
                deepcopy(dict(raw_change)) if raw_change is not None else None
            ),
            priority_reason=str(value.get("priority_reason", "")),
            causal_chain=causal_chain,
            expected_effect=str(value.get("expected_effect", "")),
            falsification_condition=str(value.get("falsification_condition", "")),
            regression_guards=regression_guards,
        )

    def to_dict(self) -> JsonObject:
        return {
            "category": self.category,
            "phase": self.phase,
            "summary": self.summary,
            "evidence": deepcopy(list(self.evidence)),
            "confidence": self.confidence,
            "environment_actionable": self.environment_actionable,
            "primary_change": deepcopy(self.primary_change),
            "priority_reason": self.priority_reason,
            "causal_chain": list(self.causal_chain),
            "expected_effect": self.expected_effect,
            "falsification_condition": self.falsification_condition,
            "regression_guards": list(self.regression_guards),
        }


def _string_tuple(value: Any, field_name: str) -> tuple[str, ...]:
    """严格读取字符串数组，避免把单个字符串拆成逐字符证据。"""

    if not isinstance(value, (list, tuple)) or any(
        not isinstance(item, str) for item in value
    ):
        raise ValueError(f"{field_name} must be an array of strings")
    return tuple(value)


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
            environment_actionable=False,
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
        # ``state_changed`` 也可能只是导航、光标或测试缓存变化，不能据此
        # 宣称发生了重复业务写入；只有环境显式标记 committed 才采用该规则。
        if info.get("committed"):
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
            environment_actionable=True,
            primary_change={
                "component_type": "contract",
                "primary_axis": "f_A",
                "execution_phase": "contract",
                "phase": "contract",
                "implementation": "require_argument",
                "operation": "add",
                "parameters": {
                    "tool": str(writes[-1].get("tool", "append_item")),
                    "argument": "idempotency_key",
                },
                "rationale": "make the retry identity explicit before another write",
            },
            priority_reason="repeated state changes directly violate the verifier",
            causal_chain=(
                "the first write changed state",
                "the agent issued another state-changing request",
                "the verifier observed a repeated side effect",
            ),
            expected_effect="reduce repeated writes after an ambiguous result",
            falsification_condition=(
                "the same agent still repeats the write after seeing the required contract"
            ),
            regression_guards=("do not alter the verifier or target value",),
        )
    if errors:
        return FailureSignature(
            category="tool_error_unrecovered",
            phase="action",
            summary="the episode ended after one or more tool errors",
            evidence=tuple(errors[-3:]),
            confidence=0.7,
            environment_actionable=False,
            priority_reason=(
                "the trace proves an unrecovered error but not which environment "
                "mechanism can be changed safely"
            ),
        )
    return FailureSignature(
        category="goal_not_reached",
        phase="agent",
        summary=evaluation.reason,
        evidence=({"metrics": deepcopy(dict(evaluation.metrics))},),
        confidence=0.4,
        environment_actionable=False,
    )
