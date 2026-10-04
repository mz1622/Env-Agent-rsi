"""AppWorld-only 组件注册与模块中文设计文档的架构测试。

这些测试约束插件发现、重复注册和源码文档，防止再次把已移除的数据集接口悄然接回。
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

from env_agent_rsi.core.registry import ComponentRegistry
from env_agent_rsi.harness.factory import (
    available_components,
)


ROOT = Path(__file__).resolve().parents[1]
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
        self.assertEqual(components["environments"], ("appworld_process",))
        self.assertNotIn("verifiers", components)
        self.assertIn("replay", components["setup_rules"])
        self.assertIn("require_argument", components["contract_rules"])
        self.assertIn("add_tool_guidance", components["contract_rules"])
        self.assertIn("require_argument", components["action_rules"])
        self.assertIn("post_commit_timeout", components["transition_rules"])
        self.assertIn("stale_read_after_write", components["observation_rules"])
        self.assertIn("step_budget", components["budget_rules"])

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
