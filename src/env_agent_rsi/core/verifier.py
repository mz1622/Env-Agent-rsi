"""独立状态 verifier 的最小协议。

Verifier 只读取环境真实快照并返回结构化结果，不读取 Agent observation，也不改变
状态；这一边界保证环境辅助不能偷偷修改任务成功标准。
"""

from __future__ import annotations

from typing import Any, Mapping, Protocol, runtime_checkable

from .protocol import EvaluationResult


@runtime_checkable
class StateVerifier(Protocol):
    """Read-only task success function over true environment state."""

    def evaluate(self, state: Mapping[str, Any]) -> EvaluationResult:
        ...
