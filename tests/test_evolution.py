"""六类变化、环境 DAG、Setup/Budget 和 benchmark adapter 测试。

这些测试约束环境搜索的数据边界：变化必须属于固定阶段，DAG 可合并等价节点，外部
benchmark 可通过 adapter 运行，Setup 不消耗正式预算。
"""

from __future__ import annotations

import unittest
from typing import Any, Mapping

from env_agent_rsi.benchmarks import BenchmarkAdapter, BenchmarkTask
from env_agent_rsi.core.protocol import Action, EvaluationResult
from env_agent_rsi.core.tooling import tool_schema
from env_agent_rsi.evolution import (
    EnvironmentDAG,
    MUTATION_PHASES,
    MutationSpec,
    MutationSurface,
    ToolSemantics,
    default_mutation_catalog,
    materialize_environment_spec,
)
from env_agent_rsi.harness.wrapper import RuleHarness
from env_agent_rsi.transforms import ReplaySetupRule, StepBudgetRule
from env_agent_rsi.transforms import AddToolGuidanceContractRule


class FakeBenchmarkBackend:
    """用于验证 adapter 协议的内存 benchmark。"""

    def __init__(self) -> None:
        self.value = 0

    def task(self) -> BenchmarkTask:
        return BenchmarkTask(
            task_id="fake-1",
            instruction="increment once",
            tools=(
                tool_schema("increment", "Increment value.", {}, []),
                tool_schema("finish", "Finish.", {}, []),
            ),
        )

    def reset(self, seed: int, options: Mapping[str, Any]) -> Mapping[str, Any]:
        del seed, options
        self.value = 0
        return {"ok": True, "value": self.value}

    def step(self, tool: str, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        del arguments
        if tool == "increment":
            self.value += 1
            return {"observation": {"ok": True, "value": self.value}}
        return {
            "observation": {"ok": True, "value": self.value},
            "terminated": True,
        }

    def observe(self) -> Mapping[str, Any]:
        return {"value": self.value}

    def evaluate(self) -> EvaluationResult:
        return EvaluationResult(self.value == 1, "value must be one", {"value": self.value})

    def save_state(self) -> Mapping[str, Any]:
        return {"value": self.value}

    def load_state(self, snapshot: Mapping[str, Any]) -> None:
        self.value = int(snapshot["value"])

    def mutation_surface(self) -> MutationSurface:
        return MutationSurface(
            tools=(ToolSemantics("increment", "write", True),),
            budget_dimensions=("steps", "writes"),
        )


class MutationAndLineageTests(unittest.TestCase):
    def test_six_mutation_phases_have_stable_ids(self) -> None:
        for phase in MUTATION_PHASES:
            first = MutationSpec(phase, "example", {"amount": 1})
            second = MutationSpec.from_dict(first.to_dict())
            self.assertEqual(first.digest, second.digest)
        with self.assertRaises(ValueError):
            MutationSpec("agent", "rewrite-policy")

    def test_surface_rejects_unknown_implementation_and_parameters(self) -> None:
        surface = MutationSurface(
            tools=(ToolSemantics("append_item", "write"),),
        )
        catalog = default_mutation_catalog()
        with self.assertRaises(ValueError):
            surface.validate(
                MutationSpec("observation", "confirm_commit"),
                catalog,
            )
        with self.assertRaises(ValueError):
            surface.validate(
                MutationSpec(
                    "contract",
                    "require_argument",
                    {"tool": "missing_tool", "argument": "key"},
                ),
                catalog,
            )

    def test_materializer_add_replace_remove_preserves_appworld_task(self) -> None:
        parent = {
            "schema_version": 1,
            "environment": {"type": "appworld_process"},
            "task": {"id": "22cc237_3"},
            "rules": {"contract": []},
        }
        surface = MutationSurface(
            tools=(ToolSemantics("execute_python", "world_state"),),
        )
        catalog = default_mutation_catalog()
        added = materialize_environment_spec(
            parent,
            MutationSpec(
                "contract",
                "require_argument",
                {"tool": "execute_python", "argument": "code"},
            ),
            surface=surface,
            catalog=catalog,
        )
        self.assertEqual(parent["rules"]["contract"], [])
        self.assertEqual(added["task"], parent["task"])
        self.assertNotIn("rules", added)
        self.assertEqual(added["components"][0]["component_type"], "contract")
        self.assertEqual(added["components"][0]["primary_axis"], "f_A")
        self.assertEqual(added["components"][0]["type"], "require_argument")

        replaced = materialize_environment_spec(
            added,
            MutationSpec(
                "contract",
                "require_argument",
                {"tool": "execute_python", "argument": "timeout"},
                operation="replace",
                target_implementation="require_argument",
            ),
            surface=surface,
            catalog=catalog,
        )
        self.assertEqual(replaced["components"][0]["argument"], "timeout")

        removed = materialize_environment_spec(
            replaced,
            MutationSpec(
                "contract",
                "require_argument",
                {},
                operation="remove",
            ),
            surface=surface,
            catalog=catalog,
        )
        self.assertEqual(removed["components"], [])

    def test_dag_merges_equivalent_child_from_multiple_parents(self) -> None:
        dag = EnvironmentDAG()
        first = dag.add_root("a" * 64)
        second = dag.add_root("b" * 64)
        mutation = MutationSpec("contract", "require_argument", {"argument": "key"})
        child = dag.add_child([first.node_id], mutation, "c" * 64)
        same_child = dag.add_child([second.node_id], mutation, "c" * 64)
        self.assertIs(child, same_child)
        self.assertEqual(child.parents, {first.node_id, second.node_id})
        self.assertEqual(len(dag.nodes), 3)
        self.assertEqual(len(dag.edges), 2)


class HarnessAndAdapterTests(unittest.TestCase):
    def test_tool_guidance_changes_visible_contract_without_blocking_action(self) -> None:
        backend = FakeBenchmarkBackend()
        env = RuleHarness(
            BenchmarkAdapter(backend),
            contract_rules=[
                AddToolGuidanceContractRule(
                    "increment", "Inspect the current value before finishing."
                )
            ],
        )
        descriptor = env.describe()
        description = descriptor.tools[0]["function"]["description"]
        self.assertIn("Inspect the current value", description)
        response = env.step(Action("increment", {}))
        self.assertTrue(response.observation["ok"])

    def test_setup_runs_before_agent_and_does_not_consume_budget(self) -> None:
        backend = FakeBenchmarkBackend()
        env = RuleHarness(
            BenchmarkAdapter(backend),
            setup_rules=[ReplaySetupRule([Action("increment", {})])],
            budget_rules=[StepBudgetRule(max_steps=1)],
        )
        reset = env.reset(seed=0)
        self.assertEqual(len(reset.info["setup_trace"]), 1)
        self.assertEqual(backend.value, 1)
        response = env.step(Action("increment", {}))
        self.assertTrue(response.truncated)
        self.assertEqual(response.observation["budget"]["steps_used"], 1)
        self.assertEqual(
            tuple(response.info["rule_order"]),
            ("setup", "contract", "action", "transition", "observation", "budget"),
        )

    def test_benchmark_adapter_normalizes_external_backend(self) -> None:
        env = BenchmarkAdapter(FakeBenchmarkBackend())
        reset = env.reset(seed=7)
        self.assertEqual(reset.observation["task_id"], "fake-1")
        self.assertEqual(env.mutation_surface().budget_dimensions, ("steps", "writes"))
        env.step(Action("increment", {}))
        snapshot = env.save_state()
        env.step(Action("increment", {}))
        self.assertFalse(env.evaluate().success)
        env.load_state(snapshot)
        self.assertTrue(env.evaluate().success)


if __name__ == "__main__":
    unittest.main()
