"""五个可执行 scenario 的端到端测试。

检查 manifest/tool schema 对齐、Oracle 可解性、确定性快照、collateral-damage 拒绝
以及代码任务从 FAIL_TO_PASS 到通过的真实过程。
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from env_agent_rsi.code_tasks import INITIAL_FILES
from env_agent_rsi.core.protocol import Action
from env_agent_rsi.harness.factory import build_environment, load_spec
from env_agent_rsi.scenario_agents import SCENARIO_ORACLES


ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ROOT / "scenarios"


def scenario_paths() -> list[Path]:
    return sorted(SCENARIOS.glob("*/task.json"))


class RunnableScenarioTests(unittest.TestCase):
    def test_every_manifest_action_has_an_executable_tool_schema(self) -> None:
        for task_path in scenario_paths():
            manifest = json.loads((task_path.parent / "scenario.json").read_text())
            env = build_environment(load_spec(task_path))
            reset = env.reset(seed=7)
            tool_names = {
                schema["function"]["name"] for schema in reset.observation["tools"]
            }
            self.assertEqual(
                set(manifest["actions"]), tool_names, task_path.parent.name
            )
            self.assertEqual(manifest["status"], "implemented")

    def test_non_item_oracles_solve_all_catalog_tasks(self) -> None:
        for task_path in scenario_paths()[1:]:
            spec = load_spec(task_path)
            env = build_environment(spec)
            env.reset(seed=0)
            environment_type = spec["environment"]["type"]
            result, trace = SCENARIO_ORACLES[environment_type](env)
            self.assertTrue(result.success, (task_path, result))
            self.assertTrue(trace)

    def test_reset_and_snapshot_are_deterministic(self) -> None:
        for task_path in scenario_paths():
            spec = load_spec(task_path)
            env = build_environment(spec)
            env.reset(seed=11)
            first = env.save_state()
            env.reset(seed=11)
            self.assertEqual(first, env.save_state(), task_path)
            restored = build_environment(spec)
            restored.load_state(first)
            self.assertEqual(first, restored.save_state(), task_path)

    def test_order_verifier_rejects_wrong_order_cancellation(self) -> None:
        env = build_environment(load_spec(SCENARIOS / "02_order_lifecycle/task.json"))
        env.reset()
        env.step(
            Action(
                "authenticate_user",
                {"first_name": "Aarav", "last_name": "Lee", "zip": "85025"},
            )
        )
        response = env.step(
            Action(
                "cancel_order",
                {"order_id": "O-1008", "reason": "no longer needed"},
            )
        )
        self.assertFalse(response.observation["ok"])
        env.step(Action("finish"))
        self.assertFalse(env.evaluate().success)

    def test_issue_verifier_rejects_collateral_change(self) -> None:
        env = build_environment(load_spec(SCENARIOS / "03_issue_workflow/task.json"))
        env.reset()
        env.step(Action("assign_issue", {"issue_id": "I-405", "user_id": "U-17"}))
        env.step(Action("finish"))
        result = env.evaluate()
        self.assertFalse(result.success)
        self.assertFalse(result.metrics["collateral_unchanged"])

    def test_calendar_rejects_conflicting_event(self) -> None:
        env = build_environment(load_spec(SCENARIOS / "04_calendar_email/task.json"))
        env.reset()
        response = env.step(
            Action(
                "create_event",
                {
                    "title": "Catch up on overdue tasks",
                    "attendee_emails": ["leila.azizi@atlas.com"],
                    "start": "2023-12-01T12:30:00",
                    "duration_minutes": 30,
                },
            )
        )
        self.assertEqual(response.observation["error"]["code"], "CALENDAR_CONFLICT")

    def test_code_task_starts_failing_and_oracle_patch_passes(self) -> None:
        task_path = SCENARIOS / "05_code_repair/task.json"
        env = build_environment(load_spec(task_path))
        env.reset()
        initial = env.step(Action("run_tests"))
        self.assertFalse(initial.observation["test_result"]["passed"])
        self.assertEqual(
            sum(
                not test["passed"]
                for test in initial.observation["test_result"]["tests"]
            ),
            2,
        )
        env = build_environment(load_spec(task_path))
        env.reset()
        result, _ = SCENARIO_ORACLES["code_repository"](env)
        self.assertTrue(result.success)
        self.assertNotEqual(
            env.get_env_state()["state"]["files"]["qdp_parser.py"],
            INITIAL_FILES["qdp_parser.py"],
        )


if __name__ == "__main__":
    unittest.main()
