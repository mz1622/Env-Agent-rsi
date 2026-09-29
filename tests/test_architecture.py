"""组件注册、规则组合和 scenario 元数据的架构测试。

这些测试约束插件发现、重复注册、contract guard、快照恢复和目录清单，防止扩展新
实现时破坏公共装配边界。
"""

from __future__ import annotations

import json
import ast
import unittest
from pathlib import Path

from env_agent_rsi.core.protocol import Action
from env_agent_rsi.core.registry import ComponentRegistry
from env_agent_rsi.harness.factory import (
    available_components,
    build_environment,
    load_spec,
)


ROOT = Path(__file__).resolve().parents[1]
ASSISTIVE = ROOT / "configs/micro_api/assistive_idempotency.json"
SCENARIOS = ROOT / "scenarios"


class RegistryTests(unittest.TestCase):
    def test_registry_builds_registered_component(self) -> None:
        registry = ComponentRegistry[dict]("test")
        registry.register("example", lambda config: {"value": config["value"]})
        self.assertEqual(registry.build("example", {"value": 3}), {"value": 3})

    def test_registry_rejects_duplicate_names(self) -> None:
        registry = ComponentRegistry[object]("test")
        registry.register("example", lambda config: object())
        with self.assertRaisesRegex(ValueError, "already registered"):
            registry.register("example", lambda config: object())

    def test_builtin_components_are_discoverable(self) -> None:
        components = available_components()
        self.assertIn("item", components["environments"])
        self.assertIn("order_api", components["environments"])
        self.assertIn("issue_tracker", components["environments"])
        self.assertIn("workplace_apps", components["environments"])
        self.assertIn("code_repository", components["environments"])
        self.assertIn("exactly_once", components["verifiers"])
        self.assertIn("order_goal_state", components["verifiers"])
        self.assertIn("issue_goal_state", components["verifiers"])
        self.assertIn("calendar_email_goal_state", components["verifiers"])
        self.assertIn("test_patch_verifier", components["verifiers"])
        self.assertIn("replay", components["setup_rules"])
        self.assertIn("require_argument", components["contract_rules"])
        self.assertIn("require_argument", components["action_rules"])
        self.assertIn("post_commit_timeout", components["transition_rules"])
        self.assertIn("stale_read_after_write", components["observation_rules"])
        self.assertIn("step_budget", components["budget_rules"])


class ActionRuleTests(unittest.TestCase):
    def test_guard_blocks_unsafe_write_before_state_change(self) -> None:
        env = build_environment(load_spec(ASSISTIVE))
        env.reset(seed=0)
        response = env.step(Action("append_item", {"value": "target-item"}))
        self.assertEqual(
            response.observation["error"]["code"], "IDEMPOTENCY_KEY_REQUIRED"
        )
        self.assertEqual(env.get_env_state()["step_count"], 0)
        self.assertFalse(
            any(item["value"] == "target-item" for item in env.get_env_state()["items"])
        )

    def test_guard_allows_safe_write_and_round_trips_snapshot(self) -> None:
        spec = load_spec(ASSISTIVE)
        env = build_environment(spec)
        env.reset(seed=0)
        env.step(Action("append_item", {"value": "target-item"}))
        response = env.step(
            Action(
                "append_item",
                {"value": "target-item", "idempotency_key": "task:target-item"},
            )
        )
        self.assertTrue(response.observation["ok"])

        snapshot = env.save_state()
        restored = build_environment(spec)
        restored.load_state(snapshot)
        self.assertEqual(restored.save_state(), snapshot)


class ScenarioCatalogTests(unittest.TestCase):
    def test_first_five_scenarios_have_docs_and_manifests(self) -> None:
        directories = sorted(path for path in SCENARIOS.iterdir() if path.is_dir())
        self.assertEqual(len(directories), 5)
        required_headings = (
            "## 任务是什么",
            "## 来源是什么",
            "## 哪些工作用了这个问题",
            "## 哪些 benchmark 与它有关",
        )
        ids: set[str] = set()
        for directory in directories:
            readme = (directory / "README.md").read_text(encoding="utf-8")
            for heading in required_headings:
                self.assertIn(heading, readme, directory.name)
            manifest = json.loads(
                (directory / "scenario.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["schema_version"], 1)
            self.assertNotIn(manifest["id"], ids)
            ids.add(manifest["id"])
            self.assertTrue(manifest["actions"])
            self.assertTrue(manifest["related_benchmarks"])
            self.assertEqual(manifest["status"], "implemented")
            self.assertTrue((directory / "task.json").is_file())
            task = json.loads((directory / "task.json").read_text(encoding="utf-8"))
            self.assertEqual(task["schema_version"], 1)
            self.assertIn("provenance", task["task"])


class ModuleDocumentationTests(unittest.TestCase):
    def test_every_python_file_starts_with_chinese_design_docstring(self) -> None:
        python_files = sorted((ROOT / "src").rglob("*.py")) + sorted(
            (ROOT / "tests").rglob("*.py")
        )
        self.assertTrue(python_files)
        for path in python_files:
            module = ast.parse(path.read_text(encoding="utf-8"))
            docstring = ast.get_docstring(module, clean=False)
            self.assertTrue(docstring, path)
            self.assertTrue(
                any("\u4e00" <= character <= "\u9fff" for character in docstring),
                path,
            )


if __name__ == "__main__":
    unittest.main()
