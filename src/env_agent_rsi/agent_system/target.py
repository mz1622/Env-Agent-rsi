"""执行环境任务的 Target Agent。

TargetAgent 不含环境专用逻辑；它把 JSON 配置加载的 prompt/skills、统一 ModelClient
与 ActionableEnv 交给 AgentRunner，因此可连续完成任意数量的工具调用。
"""

from __future__ import annotations

from typing import Any, Mapping

from env_agent_rsi.agent_runtime.runner import AgentRunner, EpisodeResult
from env_agent_rsi.agent_system.agent import ConfiguredAgent
from env_agent_rsi.core.protocol import ActionableEnv


class TargetAgent(ConfiguredAgent):
    """被环境变化评估和辅助的任务执行 Agent。"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if self.config.role != "target":
            raise ValueError(f"TargetAgent requires role='target', got {self.config.role!r}")

    def run(
        self,
        env: ActionableEnv,
        *,
        seed: int = 0,
        options: Mapping[str, Any] | None = None,
    ) -> EpisodeResult:
        runner = AgentRunner(
            env,
            self.model,
            system_prompt=self.prompt.content,
            skill_texts=[skill.render() for skill in self.skills],
            max_steps=self.config.max_steps,
            max_context_messages=self.config.max_context_messages,
            memory_retriever=self.memory,
            memory_top_k=self.config.memory.top_k,
        )
        return runner.run(seed=seed, options=options)
