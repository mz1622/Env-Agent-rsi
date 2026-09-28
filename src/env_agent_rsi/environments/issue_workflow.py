"""多步骤 issue 工作流环境。

环境提供相似 issue 与用户作为检索干扰，写操作覆盖分配、评论和状态推进；状态结构
与工具契约局部封装，verifier 精确检查目标 issue 并保护其他记录。
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from env_agent_rsi.core.protocol import EnvResponse, JsonObject
from env_agent_rsi.core.verifier import StateVerifier
from env_agent_rsi.environments.base import StatefulTaskEnv, tool


ISSUE_TOOLS = [
    tool(
        "search_issues",
        "Search issues by words in the title.",
        {"query": {"type": "string"}, "status": {"type": "string"}},
        ["query"],
    ),
    tool(
        "get_issue",
        "Read an issue and its current assignee, comments, and status.",
        {"issue_id": {"type": "string"}},
        ["issue_id"],
    ),
    tool(
        "search_users",
        "Search project users by username or display name.",
        {"query": {"type": "string"}},
        ["query"],
    ),
    tool(
        "assign_issue",
        "Assign an issue to one project user.",
        {"issue_id": {"type": "string"}, "user_id": {"type": "string"}},
        ["issue_id", "user_id"],
    ),
    tool(
        "add_comment",
        "Add a comment to an issue.",
        {"issue_id": {"type": "string"}, "body": {"type": "string"}},
        ["issue_id", "body"],
    ),
    tool(
        "change_status",
        "Change issue status.",
        {
            "issue_id": {"type": "string"},
            "status": {"type": "string", "enum": ["Open", "In Progress", "Closed"]},
        },
        ["issue_id", "status"],
    ),
    tool("finish", "Finish the task and run the independent verifier.", {}),
]


class IssueWorkflowEnv(StatefulTaskEnv):
    def __init__(
        self,
        *,
        verifier: StateVerifier,
        task_id: str = "webarena_verified_adapted_446",
        instruction: str = (
            "In project a11yproject/a11yproject.com, find the open issue about 404 "
            "errors. Assign it to Roshanjossey, add exactly one comment "
            "'Reproduced on the current documentation build.', change its status to "
            "'In Progress', and finish. Do not modify the similarly named issue."
        ),
    ) -> None:
        super().__init__(
            verifier=verifier,
            task_id=task_id,
            instruction=instruction,
            tool_schemas=ISSUE_TOOLS,
        )

    def initial_state(self, seed: int, options: Mapping[str, Any]) -> JsonObject:
        del seed, options
        return {
            "users": [
                {"id": "U-17", "username": "Roshanjossey", "name": "Roshan Jossey"},
                {"id": "U-18", "username": "Rohan", "name": "Rohan Kumar"},
                {"id": "U-19", "username": "roshan-bot", "name": "Roshan Bot"},
            ],
            "issues": {
                "I-404": {
                    "id": "I-404",
                    "project": "a11yproject/a11yproject.com",
                    "title": "404 errors on documentation links",
                    "status": "Open",
                    "assignee_id": None,
                    "comments": [],
                },
                "I-405": {
                    "id": "I-405",
                    "project": "a11yproject/a11yproject.com",
                    "title": "404 design proposal",
                    "status": "Open",
                    "assignee_id": None,
                    "comments": [],
                },
                "I-318": {
                    "id": "I-318",
                    "project": "primer/design",
                    "title": "404 errors in theme preview",
                    "status": "Open",
                    "assignee_id": "U-18",
                    "comments": [],
                },
            },
        }

    def handlers(self):
        return {
            "search_issues": self._search_issues,
            "get_issue": self._get_issue,
            "search_users": self._search_users,
            "assign_issue": self._assign_issue,
            "add_comment": self._add_comment,
            "change_status": self._change_status,
            "finish": self.finish,
        }

    def _search_issues(self, arguments: JsonObject) -> EnvResponse:
        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            return self.error("INVALID_ARGUMENT", "query must be a non-empty string")
        status = arguments.get("status")
        words = query.casefold().split()
        results = [
            deepcopy(issue)
            for issue in self.state["issues"].values()
            if all(word in issue["title"].casefold() for word in words)
            and (status is None or issue["status"] == status)
        ]
        return self.success({"issues": results}, event="search_issues")

    def _issue(self, issue_id: Any) -> JsonObject | None:
        return self.state["issues"].get(str(issue_id))

    def _get_issue(self, arguments: JsonObject) -> EnvResponse:
        issue = self._issue(arguments.get("issue_id"))
        if not issue:
            return self.error("NOT_FOUND", "issue was not found")
        return self.success({"issue": deepcopy(issue)}, event="get_issue")

    def _search_users(self, arguments: JsonObject) -> EnvResponse:
        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            return self.error("INVALID_ARGUMENT", "query must be a non-empty string")
        needle = query.casefold()
        users = [
            deepcopy(user)
            for user in self.state["users"]
            if needle in user["username"].casefold()
            or needle in user["name"].casefold()
        ]
        return self.success({"users": users}, event="search_users")

    def _assign_issue(self, arguments: JsonObject) -> EnvResponse:
        issue = self._issue(arguments.get("issue_id"))
        if not issue:
            return self.error("NOT_FOUND", "issue was not found")
        user_id = arguments.get("user_id")
        if not any(user["id"] == user_id for user in self.state["users"]):
            return self.error("NOT_FOUND", "user was not found")
        issue["assignee_id"] = user_id
        return self.success(
            {"issue": deepcopy(issue)}, event="assign_issue", state_changed=True
        )

    def _add_comment(self, arguments: JsonObject) -> EnvResponse:
        issue = self._issue(arguments.get("issue_id"))
        if not issue:
            return self.error("NOT_FOUND", "issue was not found")
        body = arguments.get("body")
        if not isinstance(body, str) or not body.strip():
            return self.error("INVALID_ARGUMENT", "body must be a non-empty string")
        comment = {"id": f"C-{len(issue['comments']) + 1}", "body": body}
        issue["comments"].append(comment)
        return self.success(
            {"comment": deepcopy(comment)}, event="add_comment", state_changed=True
        )

    def _change_status(self, arguments: JsonObject) -> EnvResponse:
        issue = self._issue(arguments.get("issue_id"))
        if not issue:
            return self.error("NOT_FOUND", "issue was not found")
        status = arguments.get("status")
        if status not in {"Open", "In Progress", "Closed"}:
            return self.error("INVALID_STATUS", "status is not allowed")
        issue["status"] = status
        return self.success(
            {"issue": deepcopy(issue)}, event="change_status", state_changed=True
        )
