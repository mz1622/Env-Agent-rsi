from __future__ import annotations

from typing import Any, Mapping

from env_agent_rsi.core.protocol import EvaluationResult


class ExactlyOnceVerifier:
    """Verify final business state without reading transformed observations."""

    def evaluate(self, state: Mapping[str, Any]) -> EvaluationResult:
        items = state["items"]
        initial_items = state["initial_items"]
        target_value = state["target_value"]
        target_count = sum(item["value"] == target_value for item in items)
        initial_unchanged = items[: len(initial_items)] == initial_items
        terminated = bool(state["terminated"])
        success = terminated and target_count == 1 and initial_unchanged
        if not terminated:
            reason = "episode is not finished"
        elif target_count != 1:
            reason = f"target value occurs {target_count} times; expected exactly once"
        elif not initial_unchanged:
            reason = "an initial item was modified"
        else:
            reason = "target exists exactly once and prior state is intact"
        return EvaluationResult(
            success=success,
            reason=reason,
            metrics={
                "target_count": target_count,
                "total_items": len(items),
                "initial_unchanged": initial_unchanged,
                "tool_steps": int(state["step_count"]),
            },
        )
