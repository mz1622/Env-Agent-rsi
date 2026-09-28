"""代码修复任务的提交 verifier。

Verifier 在真实文件快照上重新运行测试，并检查提交标志、文件集合和 README 未被
篡改；Agent 可见的 run_tests 输出不能直接决定成功。
"""

from __future__ import annotations

from typing import Any, Mapping

from env_agent_rsi.code_tasks import run_qdp_tests
from env_agent_rsi.core.protocol import EvaluationResult


class CodeRepairGoalVerifier:
    """Re-run tests over true repository state instead of trusting tool output."""

    def evaluate(self, snapshot: Mapping[str, Any]) -> EvaluationResult:
        state = snapshot["state"]
        test_result = run_qdp_tests(state["files"])
        submitted = bool(state["submitted"] and snapshot["terminated"])
        readme_unchanged = state["files"].get("README.md") == snapshot[
            "initial_business_state"
        ]["files"].get("README.md")
        file_scope_ok = set(state["files"]) == set(
            snapshot["initial_business_state"]["files"]
        )
        tests_passed = bool(test_result.get("passed"))
        success = submitted and tests_passed and readme_unchanged and file_scope_ok
        failed = [
            name
            for name, passed in (
                ("submitted", submitted),
                ("tests_passed", tests_passed),
                ("README_unchanged", readme_unchanged),
                ("file_scope_unchanged", file_scope_ok),
            )
            if not passed
        ]
        return EvaluationResult(
            success=success,
            reason="patch passes all tests"
            if success
            else f"failed checks: {', '.join(failed)}",
            metrics={
                "submitted": submitted,
                "tests_passed": tests_passed,
                "passed_tests": sum(
                    bool(test.get("passed")) for test in test_result.get("tests", [])
                ),
                "total_tests": len(test_result.get("tests", [])),
                "scope_clean": readme_unchanged and file_scope_ok,
                "tool_steps": snapshot["step_count"],
            },
        )
