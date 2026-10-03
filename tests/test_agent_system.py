"""配置驱动诊断/修改 Agent、JSON 资源与结构化输出修复测试。

Target 的训练和多轮工具循环已交给 Agent0；这里继续用 Callable 模型验证环境进化侧的
Diagnostic、Modifier 以及 provider 参数覆盖，不访问外部 API。
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from env_agent_rsi.agent_runtime import CallableModelClient
from env_agent_rsi.agent_runtime.model import ModelOutput
from env_agent_rsi.agent_system.config import load_agent_config
from env_agent_rsi.agent_system.diagnostic import DiagnosticAgent
from env_agent_rsi.agent_system.modifier import EnvironmentModificationAgent
from env_agent_rsi.agent_system.providers.adk import ADKModelClient
from env_agent_rsi.agent_system.providers.api import APIModelClient
from env_agent_rsi.agent_system.providers.factory import build_model_client
from env_agent_rsi.evolution import (
    FailureSignature,
    MutationSurface,
    ToolSemantics,
    default_mutation_catalog,
)


ROOT = Path(__file__).resolve().parents[1]
DIAGNOSTIC_CONFIG = ROOT / "configs/agents/diagnostic_agent.json"
MODIFIER_CONFIG = ROOT / "configs/agents/environment_modifier_agent.json"


class AgentConfigurationTests(unittest.TestCase):
    def test_environment_side_llm_roles_share_deepseek_flash(self) -> None:
        configs = [
            load_agent_config(DIAGNOSTIC_CONFIG),
            load_agent_config(MODIFIER_CONFIG),
        ]
        self.assertEqual({item.provider.type for item in configs}, {"api"})
        self.assertEqual({item.provider.model for item in configs}, {"deepseek-flash"})
        self.assertEqual(
            {item.provider.base_url for item in configs},
            {"https://api.deepseek.com"},
        )
        self.assertEqual(
            {item.provider.api_key_file for item in configs},
            {ROOT / "api.txt"},
        )

    def test_config_resolves_prompt_skills_and_provider_overrides(self) -> None:
        config = load_agent_config(
            DIAGNOSTIC_CONFIG,
            provider_overrides={"model": "test-model", "args": {"temperature": 1}},
        )
        self.assertEqual(config.role, "diagnostic")
        self.assertTrue(config.system_prompt.is_file())
        self.assertTrue(all(path.is_file() for path in config.skills))
        self.assertEqual(config.provider.model, "test-model")
        self.assertEqual(config.provider.args["temperature"], 1)
        self.assertEqual(config.provider.args["timeout"], 180)
        self.assertEqual(config.provider.base_url, "https://api.deepseek.com")
        self.assertEqual(config.provider.api_key_file, ROOT / "api.txt")
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

    def test_diagnostic_agent_parses_structured_failure(self) -> None:
        content = (
            '{"category":"ambiguous_commit","phase":"observation",'
            '"summary":"write status was hidden","evidence":[],"confidence":0.9,'
            '"environment_actionable":true,'
            '"primary_change":{"phase":"contract","implementation":"require_argument",'
            '"operation":"add","parameters":{"tool":"append_item",'
            '"argument":"idempotency_key"}},'
            '"priority_reason":"prevents ambiguous retry first",'
            '"causal_chain":["commit hidden","write repeated"],'
            '"expected_effect":"fewer repeated writes",'
            '"falsification_condition":"agent still repeats",'
            '"regression_guards":["verifier unchanged"]}'
        )
        requests = []

        def generate(messages, tools):
            requests.append(list(messages))
            return ModelOutput(tool_name=None, content=content)

        model = CallableModelClient(generate)
        agent = DiagnosticAgent.from_config(DIAGNOSTIC_CONFIG, model=model)
        signature = agent.diagnose(
            {
                "evaluation": {
                    "success": False,
                    "reason": "target missing",
                    "metrics": {},
                },
                "trace": [],
                "stopped_reason": "terminated",
                "steps": 3,
                "final_descriptor": {
                    "task_id": "diagnostic-task",
                    "task": "complete the diagnostic task",
                },
            }
        )
        self.assertEqual(signature.category, "ambiguous_commit")
        self.assertEqual(signature.phase, "observation")
        self.assertTrue(signature.environment_actionable)
        self.assertEqual(signature.primary_change["implementation"], "require_argument")
        diagnostic_payload = json.loads(requests[0][1]["content"])
        self.assertEqual(diagnostic_payload["task"]["task_id"], "diagnostic-task")
        self.assertEqual(
            diagnostic_payload["task"]["instruction"],
            "complete the diagnostic task",
        )
        self.assertEqual(diagnostic_payload["episode"]["steps"], 3)

    def test_modifier_returns_one_registered_executable_mutation(self) -> None:
        content = (
            '{"phase":"contract","implementation":"require_argument",'
            '"operation":"add","target_implementation":null,'
            '"parameters":{"tool":"append_item","argument":"idempotency_key"},'
            '"rationale":"make retry identity explicit"}'
        )
        model = CallableModelClient(
            lambda messages, tools: ModelOutput(tool_name=None, content=content)
        )
        agent = EnvironmentModificationAgent.from_config(MODIFIER_CONFIG, model=model)
        diagnosis = FailureSignature.from_dict(
            {
                "category": "ambiguous_commit",
                "phase": "observation",
                "summary": "commit status hidden",
                "environment_actionable": True,
                "primary_change": {
                    "phase": "contract",
                    "implementation": "require_argument",
                    "operation": "add",
                    "parameters": {
                        "tool": "append_item",
                        "argument": "idempotency_key",
                    },
                },
            }
        )
        surface = MutationSurface(
            tools=(ToolSemantics("append_item", "write"),),
        )
        mutation = agent.propose(
            diagnosis,
            environment_spec={"rules": {}},
            surface=surface,
            catalog=default_mutation_catalog(),
        )
        self.assertEqual(mutation.phase, "contract")
        self.assertEqual(mutation.implementation, "require_argument")
        self.assertEqual(mutation.operation, "add")

    def test_diagnostic_repairs_string_arrays_before_parsing(self) -> None:
        outputs = iter(
            [
                ModelOutput(
                    tool_name=None,
                    content=(
                        '{"category":"bad-shape","phase":"agent",'
                        '"summary":"bad","evidence":"not-an-array",'
                        '"confidence":0.5,"environment_actionable":false,'
                        '"primary_change":null,"priority_reason":"bad",'
                        '"causal_chain":"not-an-array","expected_effect":"",'
                        '"falsification_condition":"","regression_guards":[]}'
                    ),
                ),
                ModelOutput(
                    tool_name=None,
                    content=(
                        '{"category":"agent_policy","phase":"agent",'
                        '"summary":"missed prerequisite","evidence":[{"step":2}],'
                        '"confidence":0.8,"environment_actionable":false,'
                        '"primary_change":null,"priority_reason":"earliest",'
                        '"causal_chain":["missed prerequisite"],'
                        '"expected_effect":"none","falsification_condition":"none",'
                        '"regression_guards":["verifier unchanged"]}'
                    ),
                ),
            ]
        )
        requests = []

        def generate(messages, tools):
            requests.append(list(messages))
            return next(outputs)

        agent = DiagnosticAgent.from_config(
            DIAGNOSTIC_CONFIG, model=CallableModelClient(generate)
        )
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
        self.assertEqual(signature.evidence, ({"step": 2},))
        self.assertEqual(signature.causal_chain, ("missed prerequisite",))
        self.assertEqual(len(requests), 2)
        self.assertIn("violated the required schema", requests[1][-1]["content"])

    def test_modifier_repairs_invalid_action_list_once(self) -> None:
        outputs = iter(
            [
                ModelOutput(
                    tool_name=None,
                    content=(
                        '{"component_type":"stage","primary_axis":null,'
                        '"execution_phase":"setup","implementation":"replay",'
                        '"operation":"add","target_implementation":null,'
                        '"parameters":{"actions":["open cabinet"]},'
                        '"rationale":"open prerequisite"}'
                    ),
                ),
                ModelOutput(
                    tool_name=None,
                    content=(
                        '{"component_type":"stage","primary_axis":null,'
                        '"execution_phase":"setup","implementation":"replay",'
                        '"operation":"add","target_implementation":null,'
                        '"parameters":{"actions":[{"tool":"do","arguments":'
                        '{"command":"open cabinet"}}]},'
                        '"rationale":"open prerequisite"}'
                    ),
                ),
            ]
        )
        requests = []

        def generate(messages, tools):
            requests.append(list(messages))
            return next(outputs)

        agent = EnvironmentModificationAgent.from_config(
            MODIFIER_CONFIG, model=CallableModelClient(generate)
        )
        diagnosis = FailureSignature.from_dict(
            {
                "category": "missed_precondition",
                "phase": "setup",
                "summary": "cabinet stayed closed",
                "evidence": [{"step": 2}],
                "confidence": 0.9,
                "environment_actionable": True,
                "primary_change": {
                    "component_type": "stage",
                    "primary_axis": None,
                    "execution_phase": "setup",
                    "implementation": "replay",
                    "operation": "add",
                    "parameters": {
                        "actions": [
                            {
                                "tool": "do",
                                "arguments": {"command": "open cabinet"},
                            }
                        ]
                    },
                },
                "causal_chain": ["cabinet stayed closed"],
                "regression_guards": ["verifier unchanged"],
            }
        )
        surface = MutationSurface(
            tools=(ToolSemantics("do", "world_state", setup_allowed=True),),
            supported_phases=("setup",),
            supported_components=("stage",),
            supported_contract_axes=(),
            supported_implementations={"setup": ("replay",)},
        )
        mutation = agent.propose(
            diagnosis,
            environment_spec={"components": []},
            surface=surface,
            catalog=default_mutation_catalog(),
        )
        self.assertEqual(mutation.parameters["actions"][0]["tool"], "do")
        self.assertEqual(len(requests), 2)
        self.assertIn("action_list", requests[1][-1]["content"])


if __name__ == "__main__":
    unittest.main()
