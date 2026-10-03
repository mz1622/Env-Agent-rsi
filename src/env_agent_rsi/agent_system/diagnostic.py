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
from env_agent_rsi.evolution.catalog import MutationCatalog
from env_agent_rsi.evolution.mutation import MutationSpec
from env_agent_rsi.evolution.surface import MutationSurface


class DiagnosticAgent(ConfiguredAgent):
    """把一次 episode 压缩成失败签名与候选环境变化。"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.last_diagnosis_source = "not_run"
        if self.config.role != "diagnostic":
            raise ValueError(
                f"DiagnosticAgent requires role='diagnostic', got {self.config.role!r}"
            )

    def diagnose(
        self,
        episode: EpisodeResult | Mapping[str, Any],
        *,
        environment_spec: Mapping[str, Any] | None = None,
        surface: MutationSurface | None = None,
        catalog: MutationCatalog | None = None,
    ) -> FailureSignature:
        if isinstance(episode, EpisodeResult):
            trace = episode.trace
            evaluation = episode.evaluation
            descriptor = episode.final_descriptor.to_dict()
            stopped_reason = episode.stopped_reason
            steps = episode.steps
        else:
            trace = tuple(episode.get("trace", ()))
            raw = dict(episode.get("evaluation", {}))
            evaluation = EvaluationResult(
                success=bool(raw.get("success", False)),
                reason=str(raw.get("reason", "")),
                metrics=dict(raw.get("metrics", {})),
            )
            raw_descriptor = episode.get("final_descriptor", {})
            descriptor = (
                dict(raw_descriptor) if isinstance(raw_descriptor, Mapping) else {}
            )
            stopped_reason = str(episode.get("stopped_reason", "unknown"))
            steps = int(episode.get("steps", len(trace)))
        if evaluation.success:
            self.last_diagnosis_source = "verifier"
            return infer_failure_signature(trace, evaluation)
        context = self.new_context()
        diagnostic_input: dict[str, Any] = {
            "task": {
                "task_id": descriptor.get("task_id"),
                "instruction": descriptor.get("task"),
            },
            "episode": {"stopped_reason": stopped_reason, "steps": steps},
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
        if environment_spec is not None:
            diagnostic_input["environment_spec"] = dict(environment_spec)
        if surface is not None:
            diagnostic_input["mutation_surface"] = surface.to_dict()
        if catalog is not None:
            diagnostic_input["registered_implementations"] = catalog.to_prompt_value(
                phases=surface.supported_phases if surface is not None else None,
                implementations=(
                    surface.supported_implementations if surface is not None else None
                ),
            )
        context.start_task(diagnostic_input)
        messages = list(context.for_model())
        for attempt in range(2):
            try:
                output = self.model.generate(messages, ())
                value = _extract_json(output.content)
                diagnosis = FailureSignature.from_dict(value)
                if diagnosis.environment_actionable and surface is not None and catalog is not None:
                    mutation = MutationSpec.from_dict(diagnosis.primary_change or {})
                    surface.validate(mutation, catalog)
                self.last_diagnosis_source = (
                    "model" if attempt == 0 else "model_schema_repair"
                )
                return diagnosis
            except (KeyError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as exc:
                if attempt == 1:
                    break
                messages.extend(
                    [
                        output.to_assistant_message(),
                        {
                            "role": "user",
                            "content": (
                                "Your previous JSON violated the required schema or the "
                                f"environment allowlist: {exc}. Return one corrected JSON "
                                "object only. Arrays must remain arrays; replay actions must "
                                "use {\"tool\": string, \"arguments\": object}."
                            ),
                        },
                    ]
                )
        self.last_diagnosis_source = "rule_fallback"
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
