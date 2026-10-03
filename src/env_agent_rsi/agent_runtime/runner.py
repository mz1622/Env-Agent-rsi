"""模型驱动的通用 episode runner。

Runner 在 reset 后通过 ``env.describe()`` 获取任务与有效工具契约，把 tools 与消息
分开传给 ModelClient；每轮只把 Agent 可见 observation 写回模型历史，privileged
info 仅进入诊断轨迹。若 contract version 改变，下一轮自动重新绑定工具。
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping

from env_agent_rsi.agent_runtime.model import ModelClient
from env_agent_rsi.agent_system.context import ConversationContext
from env_agent_rsi.agent_system.memory import (
    MemoryQuery,
    MemoryRetriever,
    NullMemoryRetriever,
)
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
        skill_texts: tuple[str, ...] | list[str] = (),
        max_steps: int = 30,
        max_context_messages: int = 80,
        memory_retriever: MemoryRetriever | None = None,
        memory_top_k: int = 0,
    ) -> None:
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        self.env = env
        self.model = model
        self.system_prompt = system_prompt
        self.skill_texts = tuple(skill_texts)
        self.max_steps = max_steps
        self.max_context_messages = max_context_messages
        self.memory_retriever = memory_retriever or NullMemoryRetriever()
        if memory_top_k < 0:
            raise ValueError("memory_top_k must be non-negative")
        self.memory_top_k = memory_top_k

    def run(
        self, seed: int = 0, options: Mapping[str, Any] | None = None
    ) -> EpisodeResult:
        reset_response = self.env.reset(seed=seed, options=options)
        descriptor = self.env.describe()
        context = ConversationContext(
            self.system_prompt,
            self.skill_texts,
            max_messages=self.max_context_messages,
        )
        initial_observation = _without_contract(reset_response.observation)
        retrieved_memory = self.memory_retriever.retrieve(
            MemoryQuery(
                task_id=descriptor.task_id,
                instruction=descriptor.instruction,
                initial_observation=initial_observation,
            ),
            limit=self.memory_top_k,
        )
        context.start_task(
            {
                "task_id": descriptor.task_id,
                "task": descriptor.instruction,
                "retrieved_memory": list(retrieved_memory),
                "initial_observation": initial_observation,
            }
        )
        trace: list[JsonObject] = []
        stopped_reason = "max_steps"

        for step_index in range(1, self.max_steps + 1):
            output = self.model.generate(context.for_model(), descriptor.tools)
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
            context.append_assistant(output)
            context.append_tool_result(
                output.call_id, output.tool_name, response.observation
            )
            descriptor = self.env.describe()
            if response.terminated or response.truncated:
                stopped_reason = "terminated" if response.terminated else "truncated"
                break

        return EpisodeResult(
            evaluation=self.env.evaluate(),
            trace=tuple(trace),
            messages=context.messages,
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
