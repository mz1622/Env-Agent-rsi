"""重置后的 Setup 环境变化。

ReplaySetupRule 保存一组标准 Action，并在每个 episode 重置后由 RuleHarness 对基础环境
重放。动作必须经过基础环境原生 schema 校验，因而得到的初始状态保持可达和可复现；
规则本身只记录应用次数，便于快照和谱系审计。
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence

from env_agent_rsi.core.protocol import Action, JsonObject


class ReplaySetupRule:
    """把配置中的合法动作前缀重放为新的可达初始状态。"""

    name = "replay"

    def __init__(self, actions: Sequence[Action]) -> None:
        self.actions = tuple(actions)
        self.applied_count = 0

    def reset(self) -> None:
        self.applied_count = 0

    def initial_actions(self, state: Mapping[str, Any]) -> tuple[Action, ...]:
        del state
        self.applied_count += len(self.actions)
        return tuple(
            Action(action.tool, deepcopy(dict(action.arguments)))
            for action in self.actions
        )

    def save_state(self) -> JsonObject:
        return {"applied_count": self.applied_count}

    def load_state(self, state: Mapping[str, Any]) -> None:
        self.applied_count = int(state["applied_count"])
