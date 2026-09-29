"""分析失败轨迹并提出结构化环境变化的 Diagnostic Agent。

诊断器只读取只读诊断 trace 与 verifier 结果，不执行环境 action，也不能改 success；
模型输出解析失败时回退到确定性 failure inference，保证实验流程不会因诊断中断。
"""

from __future__ import annotations

import json
from typing import Any, Mapping

from env_agent_rsi.agent_runtime.runner import EpisodeResult
from env_agent_rsi.agent_system.agent import ConfiguredAgent
from env_agent_rsi.core.protocol import EvaluationResult
from env_agent_rsi.evolution.failure import FailureSignature, infer_failure_signature


class DiagnosticAgent(ConfiguredAgent):
    """把一次 episode 压缩成失败签名与候选环境变化。"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if self.config.role != "diagnostic":
            raise ValueError(
                f"DiagnosticAgent requires role='diagnostic', got {self.config.role!r}"
            )

    def diagnose(self, episode: EpisodeResult | Mapping[str, Any]) -> FailureSignature:
        if isinstance(episode, EpisodeResult):
            trace = episode.trace
            evaluation = episode.evaluation
        else:
            trace = tuple(episode.get("trace", ()))
            raw = dict(episode.get("evaluation", {}))
            evaluation = EvaluationResult(
                success=bool(raw.get("success", False)),
                reason=str(raw.get("reason", "")),
                metrics=dict(raw.get("metrics", {})),
            )
        if evaluation.success:
            return infer_failure_signature(trace, evaluation)
        context = self.new_context()
        context.start_task(
            {
                "evaluation": evaluation.to_dict(),
                "trace": list(trace),
                "allowed_environment_phases": [
                    "setup",
                    "contract",
                    "action",
                    "transition",
                    "observation",
                    "budget",
                ],
            }
        )
        try:
            output = self.model.generate(context.for_model(), ())
            value = _extract_json(output.content)
            return FailureSignature.from_dict(value)
        except (KeyError, RuntimeError, TypeError, ValueError, json.JSONDecodeError):
            return infer_failure_signature(trace, evaluation)


def _extract_json(content: str) -> Mapping[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]).strip()
        if text.startswith("json"):
            text = text[4:].lstrip()
    value = json.loads(text)
    if not isinstance(value, Mapping):
        raise ValueError("diagnostic output must be a JSON object")
    return value
