from __future__ import annotations

from typing import Callable

from env_agent_rsi.code_tasks import INITIAL_FILES
from env_agent_rsi.core.protocol import (
    Action,
    ActionableEnv,
    EvaluationResult,
    JsonObject,
)


def _act(
    env: ActionableEnv, trace: list[JsonObject], tool: str, **arguments: object
) -> JsonObject:
    action = Action(tool, arguments)
    response = env.step(action)
    trace.append({"action": action.to_dict(), "response": response.to_dict()})
    return response.observation


def order_oracle(env: ActionableEnv) -> tuple[EvaluationResult, list[JsonObject]]:
    trace: list[JsonObject] = []
    _act(
        env,
        trace,
        "authenticate_user",
        first_name="Aarav",
        last_name="Lee",
        zip="85025",
    )
    orders = _act(env, trace, "search_orders", status="pending")["orders"]
    order_id = next(
        order["id"] for order in orders if order["items"][0]["product"] == "luggage set"
    )
    order = _act(env, trace, "get_order", order_id=order_id)["order"]
    if not order["items"][0]["allowed_replacements"]:
        _act(env, trace, "cancel_order", order_id=order_id, reason="no longer needed")
    _act(env, trace, "get_refund", order_id=order_id)
    _act(env, trace, "finish")
    return env.evaluate(), trace


def issue_oracle(env: ActionableEnv) -> tuple[EvaluationResult, list[JsonObject]]:
    trace: list[JsonObject] = []
    issues = _act(env, trace, "search_issues", query="404 errors", status="Open")[
        "issues"
    ]
    issue_id = next(
        issue["id"]
        for issue in issues
        if issue["project"] == "a11yproject/a11yproject.com"
    )
    users = _act(env, trace, "search_users", query="Roshanjossey")["users"]
    user_id = next(user["id"] for user in users if user["username"] == "Roshanjossey")
    _act(env, trace, "assign_issue", issue_id=issue_id, user_id=user_id)
    _act(
        env,
        trace,
        "add_comment",
        issue_id=issue_id,
        body="Reproduced on the current documentation build.",
    )
    _act(env, trace, "change_status", issue_id=issue_id, status="In Progress")
    _act(env, trace, "get_issue", issue_id=issue_id)
    _act(env, trace, "finish")
    return env.evaluate(), trace


def calendar_email_oracle(
    env: ActionableEnv,
) -> tuple[EvaluationResult, list[JsonObject]]:
    trace: list[JsonObject] = []
    contacts = _act(env, trace, "search_contacts", query="Leila Azizi")["contacts"]
    email = next(x["email"] for x in contacts if x["name"] == "Leila Azizi")
    slots = _act(
        env,
        trace,
        "find_free_slots",
        emails=["me@atlas.com", email],
        date="2023-12-01",
        duration_minutes=30,
    )["slots"]
    event = _act(
        env,
        trace,
        "create_event",
        title="Catch up on overdue tasks",
        attendee_emails=[email],
        start=slots[0]["start"],
        duration_minutes=30,
    )["event"]
    _act(env, trace, "get_event", event_id=event["id"])
    _act(
        env,
        trace,
        "send_email",
        recipient=email,
        subject="Discuss overdue tasks",
        body="I noticed you have a few overdue tasks - let's catch up tomorrow.",
    )
    _act(env, trace, "finish")
    return env.evaluate(), trace


def code_repair_oracle(env: ActionableEnv) -> tuple[EvaluationResult, list[JsonObject]]:
    trace: list[JsonObject] = []
    _act(env, trace, "list_files")
    source = INITIAL_FILES["qdp_parser.py"]
    source = source.replace(
        're.compile(r"^(READ|SKIP)(?:\\s|$)")',
        're.compile(r"^(READ|SKIP)(?:\\s|$)", re.IGNORECASE)',
    ).replace('if token == "NO":', 'if token.upper() == "NO":')
    _act(env, trace, "edit_file", path="qdp_parser.py", content=source)
    _act(env, trace, "run_tests")
    _act(env, trace, "submit")
    return env.evaluate(), trace


SCENARIO_ORACLES: dict[
    str, Callable[[ActionableEnv], tuple[EvaluationResult, list[JsonObject]]]
] = {
    "order_api": order_oracle,
    "issue_tracker": issue_oracle,
    "workplace_apps": calendar_email_oracle,
    "code_repository": code_repair_oracle,
}
