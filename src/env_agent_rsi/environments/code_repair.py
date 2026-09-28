from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from env_agent_rsi.code_tasks import INITIAL_FILES, run_qdp_tests
from env_agent_rsi.core.protocol import EnvResponse, JsonObject
from env_agent_rsi.core.verifier import StateVerifier
from env_agent_rsi.environments.base import StatefulTaskEnv, tool


CODE_TOOLS = [
    tool("list_files", "List files in the task repository.", {}),
    tool(
        "search_code",
        "Search repository text and return matching lines.",
        {"query": {"type": "string"}},
        ["query"],
    ),
    tool(
        "read_file",
        "Read a UTF-8 repository file.",
        {"path": {"type": "string"}},
        ["path"],
    ),
    tool(
        "edit_file",
        "Replace the full contents of an editable source file.",
        {"path": {"type": "string"}, "content": {"type": "string"}},
        ["path", "content"],
    ),
    tool("run_tests", "Run FAIL_TO_PASS and PASS_TO_PASS tests.", {}),
    tool("submit", "Submit the current patch and run the independent verifier.", {}),
]


class CodeRepairEnv(StatefulTaskEnv):
    def __init__(
        self,
        *,
        verifier: StateVerifier,
        task_id: str = "swebench_lite_adapted_astropy_14365",
        instruction: str = (
            "QDP commands are case-insensitive, but qdp_parser.py assumes commands "
            "and missing-value tokens are uppercase. Make lower-case input such as "
            "'read serr 1 2' and 'no 1.0' work without regressing uppercase input. "
            "Inspect the repository, edit qdp_parser.py, run tests, and submit."
        ),
    ) -> None:
        super().__init__(
            verifier=verifier,
            task_id=task_id,
            instruction=instruction,
            tool_schemas=CODE_TOOLS,
        )

    def initial_state(self, seed: int, options: Mapping[str, Any]) -> JsonObject:
        del seed, options
        return {
            "files": deepcopy(INITIAL_FILES),
            "editable_files": ["qdp_parser.py"],
            "last_test_result": None,
            "submitted": False,
        }

    def handlers(self):
        return {
            "list_files": self._list_files,
            "search_code": self._search_code,
            "read_file": self._read_file,
            "edit_file": self._edit_file,
            "run_tests": self._run_tests,
            "submit": self._submit,
        }

    def _list_files(self, arguments: JsonObject) -> EnvResponse:
        del arguments
        return self.success({"files": sorted(self.state["files"])}, event="list_files")

    def _search_code(self, arguments: JsonObject) -> EnvResponse:
        query = arguments.get("query")
        if not isinstance(query, str) or not query:
            return self.error("INVALID_ARGUMENT", "query must be a non-empty string")
        matches = []
        for path, content in self.state["files"].items():
            for line_number, line in enumerate(content.splitlines(), start=1):
                if query.casefold() in line.casefold():
                    matches.append({"path": path, "line": line_number, "text": line})
        return self.success({"matches": matches}, event="search_code")

    def _read_file(self, arguments: JsonObject) -> EnvResponse:
        path = arguments.get("path")
        if path not in self.state["files"]:
            return self.error("NOT_FOUND", "file was not found")
        return self.success(
            {"path": path, "content": self.state["files"][path]}, event="read_file"
        )

    def _edit_file(self, arguments: JsonObject) -> EnvResponse:
        path = arguments.get("path")
        content = arguments.get("content")
        if path not in self.state["editable_files"]:
            return self.error(
                "EDIT_NOT_ALLOWED", "only listed source files are editable"
            )
        if not isinstance(content, str):
            return self.error("INVALID_ARGUMENT", "content must be a string")
        self.state["files"][path] = content
        self.state["last_test_result"] = None
        return self.success(
            {"path": path, "bytes": len(content.encode("utf-8"))},
            event="edit_file",
            state_changed=True,
        )

    def _run_tests(self, arguments: JsonObject) -> EnvResponse:
        del arguments
        result = run_qdp_tests(self.state["files"])
        self.state["last_test_result"] = deepcopy(result)
        return self.success(
            {"test_result": result}, event="run_tests", state_changed=True
        )

    def _submit(self, arguments: JsonObject) -> EnvResponse:
        del arguments
        self.state["submitted"] = True
        self.terminated = True
        evaluation = self.evaluate()
        return EnvResponse(
            observation={
                "ok": True,
                "status": "submitted",
                "task_success": evaluation.success,
                "reason": evaluation.reason,
            },
            reward=1.0 if evaluation.success else 0.0,
            terminated=True,
            info={"event": "submit", "evaluation": evaluation.to_dict()},
        )
