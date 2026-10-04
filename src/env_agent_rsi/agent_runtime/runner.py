"""使用 Agent0/Qwen3 工具协议执行环境任务的通用 episode runner。

Runner 在 reset 后通过 ``env.describe()`` 获取任务与有效工具契约，把 tools 与消息
渲染进 system，并只把 Agent 可见 observation 作为 ``<tool_response>`` 写回历史；
privileged info 仅进入诊断轨迹。若 contract version 改变，下一轮自动重建工具注册。
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping

from env_agent_rsi.agent_runtime.agent0_protocol import render_tool_call
from env_agent_rsi.agent_runtime.model import ModelClient, ModelOutput
from env_agent_rsi.agent_system.context import ConversationContext
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
    ) -> None:
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        self.env = env
        self.model = model
        self.system_prompt = system_prompt
        self.skill_texts = tuple(skill_texts)
        self.max_steps = max_steps
        self.max_context_messages = max_context_messages

    def run(
        self, seed: int = 0, options: Mapping[str, Any] | None = None
    ) -> EpisodeResult:
        reset_response = self.env.reset(seed=seed, options=options)
        descriptor = self.env.describe()
        context = ConversationContext(
            self.system_prompt,
            self.skill_texts,
            descriptor.tools,
            max_messages=self.max_context_messages,
        )
        initial_observation = _without_contract(reset_response.observation)
        task_content = descriptor.instruction
        if initial_observation:
            task_content += (
                "\n\n<initial_observation>\n"
                + _json(initial_observation)
                + "\n</initial_observation>"
            )
        context.start_task(task_content)
        trace: list[JsonObject] = []
        stopped_reason = "max_steps"

        for step_index in range(1, self.max_steps + 1):
            context.bind_tools(descriptor.tools)
            provider_tools = (
                descriptor.tools
                if getattr(self.model, "native_tool_transport", False)
                else ()
            )
            output = self.model.generate(context.for_model(), provider_tools)
            if output.tool_name is None:
                # 无合法动作时的最后输出是诊断“为什么停止”的关键证据。
                context.append_assistant(output)
                stopped_reason = "model_finished"
                break
            output = _as_agent0_action(output)

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
            context.append_tool_result(response.observation)
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


def _json(value: Mapping[str, Any]) -> str:
    """稳定序列化初始观察，避免任务上下文因字典顺序漂移。"""

    import json

    return json.dumps(dict(value), ensure_ascii=False, sort_keys=True)


def _as_agent0_action(output: ModelOutput) -> ModelOutput:
    """把 provider 的结构化动作规范化为上下文唯一接受的 Hermes 文本。"""

    if output.serialized_action or output.tool_name is None:
        return output
    call = render_tool_call(output.tool_name, output.arguments)
    prefix = output.content.strip()
    return ModelOutput(
        tool_name=output.tool_name,
        arguments=dict(output.arguments),
        content=f"{prefix}\n{call}" if prefix else call,
        call_id=output.call_id,
        serialized_action=True,
    )
