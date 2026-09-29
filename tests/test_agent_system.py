"""配置驱动 Agent、JSON 资源与多轮工具上下文测试。

测试使用 Scripted/Callable 模型，不访问外部 API；它验证 Target 与 Diagnostic 的角色
边界、skill 注入、OpenAI 标准 tool message 以及 provider 参数的统一覆盖方式。
"""

from __future__ import annotations

import unittest
from pathlib import Path

from env_agent_rsi.agent_runtime import CallableModelClient, ScriptedModelClient
from env_agent_rsi.agent_runtime.model import ModelOutput
from env_agent_rsi.agent_system.config import load_agent_config
from env_agent_rsi.agent_system.diagnostic import DiagnosticAgent
from env_agent_rsi.agent_system.providers.adk import ADKModelClient
from env_agent_rsi.agent_system.providers.api import APIModelClient
from env_agent_rsi.agent_system.providers.factory import build_model_client
from env_agent_rsi.agent_system.target import TargetAgent
from env_agent_rsi.core.protocol import Action
from env_agent_rsi.harness.factory import build_environment, load_spec


ROOT = Path(__file__).resolve().parents[1]
TARGET_CONFIG = ROOT / "configs/agents/target_agent.json"
DIAGNOSTIC_CONFIG = ROOT / "configs/agents/diagnostic_agent.json"
BASELINE = ROOT / "configs/micro_api/baseline.json"


class AgentConfigurationTests(unittest.TestCase):
    def test_config_resolves_prompt_skills_and_provider_overrides(self) -> None:
        config = load_agent_config(
            TARGET_CONFIG,
            provider_overrides={"model": "test-model", "args": {"temperature": 1}},
        )
        self.assertEqual(config.role, "target")
        self.assertTrue(config.system_prompt.is_file())
        self.assertTrue(all(path.is_file() for path in config.skills))
        self.assertEqual(config.provider.model, "test-model")
        self.assertEqual(config.provider.args["temperature"], 1)
        self.assertEqual(config.provider.args["timeout"], 60)
        client = build_model_client(config.provider)
        self.assertIsInstance(client, APIModelClient)

    def test_adk_bridge_passes_unified_arguments(self) -> None:
        received = {}

        def executor(**kwargs):
            received.update(kwargs)
            return {
                "tool_name": "finish",
                "arguments": {},
                "call_id": "adk-1",
            }

        client = ADKModelClient(
            model="adk-test-model",
            executor=executor,
            args={"session_name": "experiment-1"},
        )
        output = client.generate([{"role": "user", "content": "run"}], [])
        self.assertEqual(output.tool_name, "finish")
        self.assertEqual(received["model"], "adk-test-model")
        self.assertEqual(received["session_name"], "experiment-1")

    def test_target_agent_executes_multiple_tool_turns_with_loaded_skill(self) -> None:
        model = ScriptedModelClient(
            [
                Action(
                    "append_item",
                    {
                        "value": "target-item",
                        "idempotency_key": "task:target-item",
                    },
                ),
                Action("list_items", {"cursor": 0, "limit": 10}),
                Action("finish", {}),
            ]
        )
        agent = TargetAgent.from_config(TARGET_CONFIG, model=model)
        result = agent.run(build_environment(load_spec(BASELINE)))
        self.assertTrue(result.evaluation.success)
        self.assertEqual(result.steps, 3)
        self.assertIn("[Skill: ambiguous-write-recovery]", result.messages[0]["content"])
        assistants = [item for item in result.messages if item["role"] == "assistant"]
        tools = [item for item in result.messages if item["role"] == "tool"]
        self.assertEqual(len(assistants), 3)
        self.assertEqual(len(tools), 3)
        self.assertIn("tool_calls", assistants[0])
        self.assertIn("tool_call_id", tools[0])
        self.assertEqual(len(model.requests), 3)
        self.assertTrue(
            any(message["role"] == "tool" for message in model.requests[1]["messages"])
        )

    def test_diagnostic_agent_parses_structured_failure(self) -> None:
        content = (
            '{"category":"ambiguous_commit","phase":"observation",'
            '"summary":"write status was hidden","evidence":[],"confidence":0.9,'
            '"candidate_changes":[{"phase":"observation",'
            '"implementation":"confirm_commit","parameters":{}}]}'
        )
        model = CallableModelClient(
            lambda messages, tools: ModelOutput(tool_name=None, content=content)
        )
        agent = DiagnosticAgent.from_config(DIAGNOSTIC_CONFIG, model=model)
        signature = agent.diagnose(
            {
                "evaluation": {
                    "success": False,
                    "reason": "target missing",
                    "metrics": {},
                },
                "trace": [],
            }
        )
        self.assertEqual(signature.category, "ambiguous_commit")
        self.assertEqual(signature.phase, "observation")
        self.assertEqual(signature.candidate_changes[0]["implementation"], "confirm_commit")


if __name__ == "__main__":
    unittest.main()
