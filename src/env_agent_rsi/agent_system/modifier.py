"""把唯一诊断结论翻译成可执行环境变化的 Modification Agent。

该角色不重新排列问题，也不生成候选列表；它只依据 Diagnostic Agent 的最高优先级
根因、当前环境配置、MutationSurface 和注册目录，返回一个经过白名单校验的变化。
"""

from __future__ import annotations

import json
from typing import Any, Mapping

from env_agent_rsi.agent_system.agent import ConfiguredAgent
from env_agent_rsi.evolution.catalog import MutationCatalog
from env_agent_rsi.evolution.failure import FailureSignature
from env_agent_rsi.evolution.mutation import MutationSpec
from env_agent_rsi.evolution.surface import MutationSurface


class EnvironmentModificationAgent(ConfiguredAgent):
    """为一个环境根因生成一个且仅一个 MutationSpec。"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.last_proposal_source = "not_run"
        if self.config.role != "modifier":
            raise ValueError(
                "EnvironmentModificationAgent requires role='modifier', "
                f"got {self.config.role!r}"
            )

    def propose(
        self,
        diagnosis: FailureSignature,
        *,
        environment_spec: Mapping[str, Any],
        surface: MutationSurface,
        catalog: MutationCatalog,
    ) -> MutationSpec:
        if not diagnosis.environment_actionable:
            raise ValueError("diagnosis is not environment-actionable")
        context = self.new_context()
        context.start_task(
            {
                "diagnosis": diagnosis.to_dict(),
                "environment_spec": dict(environment_spec),
                "mutation_surface": surface.to_dict(),
                "registered_implementations": catalog.to_prompt_value(
                    phases=surface.supported_phases,
                    implementations=surface.supported_implementations,
                ),
            }
        )
        messages = list(context.for_model())
        last_error: Exception | None = None
        for attempt in range(2):
            output = self.model.generate(messages, ())
            try:
                value = _extract_json(output.content)
                mutation = MutationSpec.from_dict(value)
                surface.validate(mutation, catalog)
                self.last_proposal_source = (
                    "model" if attempt == 0 else "model_schema_repair"
                )
                return mutation
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                last_error = exc
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
                                "object only. For action_list, use an array of objects shaped "
                                "as {\"tool\": string, \"arguments\": object}; never use "
                                "plain command strings."
                            ),
                        },
                    ]
                )
        raise ValueError(f"modifier failed schema repair: {last_error}") from last_error


def _extract_json(content: str) -> Mapping[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]).strip()
        if text.startswith("json"):
            text = text[4:].lstrip()
    value = json.loads(text)
    if not isinstance(value, Mapping):
        raise ValueError("modifier output must be a JSON object")
    return value
