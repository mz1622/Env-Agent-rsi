from __future__ import annotations

from typing import Any, Mapping, Protocol, runtime_checkable

from .protocol import EvaluationResult


@runtime_checkable
class StateVerifier(Protocol):
    """Read-only task success function over true environment state."""

    def evaluate(self, state: Mapping[str, Any]) -> EvaluationResult:
        ...
