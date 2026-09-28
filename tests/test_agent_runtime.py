"""通用 Agent 链路的集成测试。

覆盖 descriptor 加载、可见 contract、schema 校验、ModelClient 工具绑定、消息隐私
边界和完整 episode；这些测试保证新增环境无需修改 AgentRunner。
"""

from __future__ import annotations

import json
import unittest
from copy import deepcopy
from pathlib import Path

from env_agent_rsi.agent_runtime import AgentRunner, ScriptedModelClient
from env_agent_rsi.core.protocol import Action
from env_agent_rsi.core.tooling import replace_tools
from env_agent_rsi.harness.factory import build_environment, load_spec
from env_agent_rsi.harness.wrapper import RuleHarness
from env_agent_rsi.micro_api.item_env import ItemEnv
from env_agent_rsi.verifiers import ExactlyOnceVerifier


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "configs/micro_api/baseline.json"
ASSISTIVE = ROOT / "configs/micro_api/assistive_idempotency.json"


def required_arguments(tools, tool_name: str) -> list[str]:
    schema = next(x for x in tools if x["function"]["name"] == tool_name)
    return schema["function"]["parameters"]["required"]


class DescriptorAndContractTests(unittest.TestCase):
    def test_contract_rule_changes_visible_schema_and_version(self) -> None:
        baseline = build_environment(load_spec(BASELINE))
        baseline.reset()
        assisted = build_environment(load_spec(ASSISTIVE))
        reset = assisted.reset()

        self.assertNotIn(
            "idempotency_key",
            required_arguments(baseline.describe().tools, "append_item"),
        )
        self.assertIn(
            "idempotency_key",
            required_arguments(assisted.describe().tools, "append_item"),
        )
        self.assertEqual(
            reset.observation["contract_version"],
            assisted.describe().contract_version,
        )
        self.assertNotEqual(
            baseline.describe().contract_version,
            assisted.describe().contract_version,
        )

    def test_contract_uses_custom_error_and_does_not_mutate_base_state(self) -> None:
        env = build_environment(load_spec(ASSISTIVE))
        env.reset()
        response = env.step(Action("append_item", {"value": "target-item"}))
        self.assertEqual(
            response.observation["error"]["code"], "IDEMPOTENCY_KEY_REQUIRED"
        )
        self.assertEqual(env.get_env_state()["step_count"], 0)

    def test_schema_validation_rejects_type_and_extra_argument(self) -> None:
        env = build_environment(load_spec(BASELINE))
        env.reset()
        wrong_type = env.step(Action("list_items", {"limit": "many"}))
        extra = env.step(Action("finish", {"unexpected": True}))
        unknown = env.step(Action("delete_everything"))
        self.assertEqual(
            wrong_type.observation["error"]["code"], "INVALID_ARGUMENT_TYPE"
        )
        self.assertEqual(extra.observation["error"]["code"], "UNEXPECTED_ARGUMENT")
        self.assertEqual(unknown.observation["error"]["code"], "UNKNOWN_TOOL")
        self.assertEqual(env.get_env_state()["step_count"], 0)


class AgentRunnerTests(unittest.TestCase):
    def test_runner_loads_tools_executes_actions_and_finishes(self) -> None:
        env = build_environment(load_spec(ASSISTIVE))
        model = ScriptedModelClient(
            [
                Action(
                    "append_item",
                    {
                        "value": "target-item",
                        "idempotency_key": "runner-target-item",
                    },
                ),
                Action("finish"),
            ]
        )
        result = AgentRunner(env, model, max_steps=4).run(seed=3)

        self.assertTrue(result.evaluation.success)
        self.assertEqual(result.stopped_reason, "terminated")
        self.assertEqual(result.steps, 2)
        self.assertEqual(len(model.requests), 2)
        for request in model.requests:
            self.assertIn(
                "idempotency_key",
                required_arguments(request["tools"], "append_item"),
            )

    def test_model_history_contains_observation_but_never_privileged_info(self) -> None:
        env = build_environment(load_spec(BASELINE))
        model = ScriptedModelClient(
            [Action("list_items", {"limit": 10}), Action("finish")]
        )
        result = AgentRunner(env, model, max_steps=3).run()
        tool_messages = [m for m in result.messages if m["role"] == "tool"]
        self.assertTrue(tool_messages)
        for message in tool_messages:
            content = json.loads(message["content"])
            self.assertNotIn("info", content)
            self.assertNotIn("rule_order", content)
        self.assertIn("rule_order", result.trace[0]["response"]["info"])

    def test_runner_stops_cleanly_when_model_returns_no_tool(self) -> None:
        env = build_environment(load_spec(BASELINE))
        result = AgentRunner(env, ScriptedModelClient([]), max_steps=2).run()
        self.assertEqual(result.stopped_reason, "model_returned_no_tool")
        self.assertEqual(result.steps, 0)
        self.assertFalse(result.evaluation.success)

    def test_runner_rebinds_tools_when_contract_version_changes(self) -> None:
        class RequireKeyAfterFirstWrite:
            name = "require_key_after_first_write"

            def reset(self) -> None:
                pass

            def transform_descriptor(self, descriptor, state):
                if len(state["items"]) == 1:
                    return descriptor
                tools = deepcopy(list(descriptor.tools))
                append_schema = next(
                    tool for tool in tools if tool["function"]["name"] == "append_item"
                )
                append_schema["function"]["parameters"]["required"].append(
                    "idempotency_key"
                )
                return replace_tools(descriptor, tools)

            def validate_action(self, state, action):
                del state, action
                return None

            def save_state(self):
                return {}

            def load_state(self, state):
                del state

        env = RuleHarness(
            ItemEnv(verifier=ExactlyOnceVerifier()),
            contract_rules=[RequireKeyAfterFirstWrite()],
        )
        model = ScriptedModelClient(
            [Action("append_item", {"value": "target-item"}), Action("finish")]
        )
        result = AgentRunner(env, model).run()

        self.assertTrue(result.evaluation.success)
        self.assertNotIn(
            "idempotency_key",
            required_arguments(model.requests[0]["tools"], "append_item"),
        )
        self.assertIn(
            "idempotency_key",
            required_arguments(model.requests[1]["tools"], "append_item"),
        )
        self.assertIn("contract_update", result.trace[0]["response"]["observation"])


if __name__ == "__main__":
    unittest.main()
