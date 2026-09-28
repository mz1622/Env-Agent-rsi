"""模型驱动的通用 episode runner。

Runner 在 reset 后通过 ``env.describe()`` 获取任务与有效工具契约，把 tools 与消息
分开传给 ModelClient；每轮只把 Agent 可见 observation 写回模型历史，privileged
info 仅进入诊断轨迹。若 contract version 改变，下一轮自动重新绑定工具。
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping

from env_agent_rsi.agent_runtime.model import ModelClient
from env_agent_rsi.core.protocol import (
    Action,
    ActionableEnv,
    EnvDescriptor,
    EvaluationResult,
    JsonObject,
)


DEFAULT_SYSTEM_PROMPT = (
    "Complete the task by calling only the supplied tools. "
    "Use observations to decide the next action and call the terminal tool when done."
)


@dataclass(frozen=True)
class EpisodeResult:
    evaluation: EvaluationResult
    trace: tuple[JsonObject, ...]
    messages: tuple[JsonObject, ...]
    stopped_reason: str
    steps: int
    final_descriptor: EnvDescriptor

    def to_dict(self) -> JsonObject:
        return {
            "evaluation": self.evaluation.to_dict(),
            "trace": deepcopy(list(self.trace)),
            "messages": deepcopy(list(self.messages)),
            "stopped_reason": self.stopped_reason,
            "steps": self.steps,
            "final_descriptor": self.final_descriptor.to_dict(),
        }


class AgentRunner:
    """对任意 ActionableEnv 执行一个单工具调用 Agent。"""

    def __init__(
        self,
        env: ActionableEnv,
        model: ModelClient,
        *,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_steps: int = 30,
    ) -> None:
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        self.env = env
        self.model = model
        self.system_prompt = system_prompt
        self.max_steps = max_steps

    def run(
        self, seed: int = 0, options: Mapping[str, Any] | None = None
    ) -> EpisodeResult:
        reset_response = self.env.reset(seed=seed, options=options)
        descriptor = self.env.describe()
        messages: list[JsonObject] = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "task_id": descriptor.task_id,
                        "task": descriptor.instruction,
                        "initial_observation": _without_contract(
                            reset_response.observation
                        ),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            },
        ]
        trace: list[JsonObject] = []
        stopped_reason = "max_steps"

        for step_index in range(1, self.max_steps + 1):
            output = self.model.generate(messages, descriptor.tools)
            if output.tool_name is None:
                stopped_reason = "model_returned_no_tool"
                break

            action = Action(output.tool_name, dict(output.arguments))
            response = self.env.step(action)
            trace.append(
                {
                    "step": step_index,
                    "contract_version": descriptor.contract_version,
                    "action": action.to_dict(),
                    "response": response.to_dict(),
                }
            )
            messages.append(output.to_assistant_message())
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": output.call_id,
                    "name": output.tool_name,
                    "content": json.dumps(
                        response.observation, ensure_ascii=False, sort_keys=True
                    ),
                }
            )
            descriptor = self.env.describe()
            if response.terminated or response.truncated:
                stopped_reason = "terminated" if response.terminated else "truncated"
                break

        return EpisodeResult(
            evaluation=self.env.evaluate(),
            trace=tuple(trace),
            messages=tuple(messages),
            stopped_reason=stopped_reason,
            steps=len(trace),
            final_descriptor=descriptor,
        )


def _without_contract(observation: Mapping[str, Any]) -> JsonObject:
    """初始用户消息不重复嵌入较大的 tool schema。"""

    return {
        key: deepcopy(value)
        for key, value in observation.items()
        if key not in {"tools", "task", "task_id", "contract_version", "metadata"}
    }
