"""独立的 Budget 环境变化。

StepBudgetRule 在真实 action 前检查总步数和写操作额度，并在执行后更新计数。预算耗尽
通过标准 EnvResponse 截断 episode，计数随 Harness 快照保存，避免恢复后重新获得额度。
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import Any, Mapping, Sequence

from env_agent_rsi.core.protocol import Action, EnvResponse, JsonObject


class StepBudgetRule:
    """限制 Agent action 总数以及指定写工具的调用次数。"""

    name = "step_budget"

    def __init__(
        self,
        max_steps: int,
        *,
        max_writes: int | None = None,
        write_tools: Sequence[str] = (),
    ) -> None:
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if max_writes is not None and max_writes < 0:
            raise ValueError("max_writes must be non-negative")
        self.max_steps = max_steps
        self.max_writes = max_writes
        self.write_tools = frozenset(write_tools)
        self.reset()

    def reset(self) -> None:
        self.steps_used = 0
        self.writes_used = 0

    def before_step(
        self, state: Mapping[str, Any], action: Action
    ) -> EnvResponse | None:
        del state
        if self.steps_used >= self.max_steps:
            return self._blocked("STEP_BUDGET_EXHAUSTED")
        if (
            action.tool in self.write_tools
            and self.max_writes is not None
            and self.writes_used >= self.max_writes
        ):
            return self._blocked("WRITE_BUDGET_EXHAUSTED")
        return None

    def after_step(self, action: Action, response: EnvResponse) -> EnvResponse:
        self.steps_used += 1
        if action.tool in self.write_tools:
            self.writes_used += 1
        observation = deepcopy(response.observation)
        observation["budget"] = self._budget_view()
        info = deepcopy(response.info)
        info["budget"] = self._budget_view()
        exhausted = self.steps_used >= self.max_steps
        return replace(
            response,
            observation=observation,
            info=info,
            truncated=bool(response.truncated or (exhausted and not response.terminated)),
        )

    def _blocked(self, code: str) -> EnvResponse:
        return EnvResponse(
            observation={
                "ok": False,
                "error": {"code": code, "message": "environment budget exhausted"},
                "budget": self._budget_view(),
            },
            truncated=True,
            info={
                "event": "budget_blocked",
                "state_changed": False,
                "fault_events": [
                    {"rule": self.name, "phase": "budget", "code": code}
                ],
            },
        )

    def _budget_view(self) -> JsonObject:
        writes_remaining = (
            None
            if self.max_writes is None
            else max(0, self.max_writes - self.writes_used)
        )
        return {
            "steps_used": self.steps_used,
            "steps_remaining": max(0, self.max_steps - self.steps_used),
            "writes_used": self.writes_used,
            "writes_remaining": writes_remaining,
        }

    def save_state(self) -> JsonObject:
        return {"steps_used": self.steps_used, "writes_used": self.writes_used}

    def load_state(self, state: Mapping[str, Any]) -> None:
        self.steps_used = int(state["steps_used"])
        self.writes_used = int(state["writes_used"])
