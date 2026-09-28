from __future__ import annotations

from typing import Any, Mapping

from env_agent_rsi.core.protocol import EvaluationResult


class IssueGoalVerifier:
    def __init__(
        self,
        target_issue_id: str = "I-404",
        assignee_id: str = "U-17",
        comment: str = "Reproduced on the current documentation build.",
        status: str = "In Progress",
    ) -> None:
        self.target_issue_id = target_issue_id
        self.assignee_id = assignee_id
        self.comment = comment
        self.status = status

    def evaluate(self, snapshot: Mapping[str, Any]) -> EvaluationResult:
        state = snapshot["state"]
        initial = snapshot["initial_business_state"]
        issue = state["issues"].get(self.target_issue_id, {})
        comments = issue.get("comments", [])
        target_ok = (
            issue.get("assignee_id") == self.assignee_id
            and issue.get("status") == self.status
            and [entry.get("body") for entry in comments] == [self.comment]
        )
        others_ok = all(
            current == initial["issues"].get(issue_id)
            for issue_id, current in state["issues"].items()
            if issue_id != self.target_issue_id
        )
        users_ok = state["users"] == initial["users"]
        terminated = bool(snapshot["terminated"])
        success = terminated and target_ok and others_ok and users_ok
        failed = [
            name
            for name, passed in (
                ("episode_finished", terminated),
                ("target_issue_exact", target_ok),
                ("other_issues_unchanged", others_ok),
                ("users_unchanged", users_ok),
            )
            if not passed
        ]
        return EvaluationResult(
            success=success,
            reason="all issue workflow goals satisfied"
            if success
            else f"failed checks: {', '.join(failed)}",
            metrics={
                "assignee_correct": issue.get("assignee_id") == self.assignee_id,
                "status_correct": issue.get("status") == self.status,
                "comment_count": len(comments),
                "collateral_unchanged": others_ok and users_ok,
                "tool_steps": snapshot["step_count"],
            },
        )
