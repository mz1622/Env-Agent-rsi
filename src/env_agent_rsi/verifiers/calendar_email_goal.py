"""日历与邮件任务的结果 verifier。

直接读取真实事件、邮件和参考数据，检查唯一性、最早时间、精确内容与旁路状态，不
依赖 Agent 是否调用过某个查询工具。
"""

from __future__ import annotations

from typing import Any, Mapping

from env_agent_rsi.core.protocol import EvaluationResult


class CalendarEmailGoalVerifier:
    def evaluate(self, snapshot: Mapping[str, Any]) -> EvaluationResult:
        state = snapshot["state"]
        events = list(state["events"].values())
        emails = state["emails"]
        event_ok = len(events) == 1 and events[0] == {
            "id": "E-1",
            "title": "Catch up on overdue tasks",
            "attendee_emails": ["leila.azizi@atlas.com"],
            "start": "2023-12-01T13:00:00",
            "end": "2023-12-01T13:30:00",
            "duration_minutes": 30,
        }
        email_ok = len(emails) == 1 and emails[0] == {
            "id": "M-1",
            "recipient": "leila.azizi@atlas.com",
            "subject": "Discuss overdue tasks",
            "body": "I noticed you have a few overdue tasks - let's catch up tomorrow.",
        }
        initial = snapshot["initial_business_state"]
        reference_ok = (
            state["contacts"] == initial["contacts"]
            and state["busy"] == initial["busy"]
            and state["owner_email"] == initial["owner_email"]
        )
        terminated = bool(snapshot["terminated"])
        success = terminated and event_ok and email_ok and reference_ok
        failed = [
            name
            for name, passed in (
                ("episode_finished", terminated),
                ("event_exact", event_ok),
                ("email_exact", email_ok),
                ("reference_data_unchanged", reference_ok),
            )
            if not passed
        ]
        return EvaluationResult(
            success=success,
            reason="calendar and email goals satisfied"
            if success
            else f"failed checks: {', '.join(failed)}",
            metrics={
                "event_count": len(events),
                "email_count": len(emails),
                "event_exact": event_ok,
                "email_exact": email_ok,
                "tool_steps": snapshot["step_count"],
            },
        )
