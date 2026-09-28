from __future__ import annotations

from typing import Any, Mapping

from env_agent_rsi.core.protocol import EvaluationResult


class OrderGoalVerifier:
    """Verify the cancellation/refund outcome and absence of collateral writes."""

    def __init__(self, target_order_id: str = "O-1007") -> None:
        self.target_order_id = target_order_id

    def evaluate(self, snapshot: Mapping[str, Any]) -> EvaluationResult:
        state = snapshot["state"]
        initial = snapshot["initial_business_state"]
        order = state["orders"].get(self.target_order_id, {})
        refund = state["refunds"].get(self.target_order_id, {})
        target_ok = (
            order.get("status") == "cancelled"
            and order.get("cancellation_reason") == "no longer needed"
        )
        refund_ok = (
            refund.get("amount") == order.get("total")
            and refund.get("payment_method_id") == order.get("payment_method_id")
            and refund.get("status") == "completed"
        )
        other_orders_ok = all(
            current == initial["orders"].get(order_id)
            for order_id, current in state["orders"].items()
            if order_id != self.target_order_id
        )
        reference_ok = (
            state["users"] == initial["users"]
            and state["payment_methods"] == initial["payment_methods"]
        )
        terminated = bool(snapshot["terminated"])
        success = (
            terminated and target_ok and refund_ok and other_orders_ok and reference_ok
        )
        failed = [
            name
            for name, passed in (
                ("episode_finished", terminated),
                ("target_cancelled", target_ok),
                ("refund_correct", refund_ok),
                ("other_orders_unchanged", other_orders_ok),
                ("reference_data_unchanged", reference_ok),
            )
            if not passed
        ]
        return EvaluationResult(
            success=success,
            reason="all order goals satisfied"
            if success
            else f"failed checks: {', '.join(failed)}",
            metrics={
                "target_cancelled": target_ok,
                "refund_correct": refund_ok,
                "collateral_unchanged": other_orders_ok and reference_ok,
                "tool_steps": snapshot["step_count"],
            },
        )
