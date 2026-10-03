"""验证 Target 与 Agent0 共用的 Qwen3 Hermes 工具上下文协议。"""

from __future__ import annotations

from env_agent_rsi.agent_runtime.agent0_protocol import (
    parse_tool_call,
    render_tool_call,
    render_tool_register,
    render_tool_response,
)
from env_agent_rsi.agent_runtime.model import CallableModelClient, ModelOutput
from env_agent_rsi.agent_runtime.runner import AgentRunner
from env_agent_rsi.agent_system.context import ConversationContext
from env_agent_rsi.agent_system.providers.api import _parse_chat_completion
from env_agent_rsi.core.protocol import (
    Action,
    EnvDescriptor,
    EnvResponse,
    EvaluationResult,
)
from env_agent_rsi.core.tooling import tool_schema


TOOLS = (
    tool_schema(
        "lookup",
        "Look up one item.",
        {"item_id": {"type": "string"}},
        ["item_id"],
    ),
)


def test_tool_register_uses_qwen3_hermes_template() -> None:
    rendered = render_tool_register(TOOLS)
    assert rendered.startswith("# Tools")
    assert "<tools>" in rendered
    assert '"name": "lookup"' in rendered
    assert "<tool_call>" in rendered


def test_tool_call_round_trip_allows_thinking_prefix() -> None:
    call = render_tool_call("lookup", {"item_id": "item-1"})
    parsed, valid = parse_tool_call(
        "<think>inspect first</think>\n" + call,
        supported_tools={"lookup"},
    )
    assert valid is True
    assert parsed == {
        "name": "lookup",
        "arguments": {"item_id": "item-1"},
    }


def test_protocol_rejects_parallel_calls_for_single_action_environment() -> None:
    content = render_tool_call("lookup", {"item_id": "a"}) + render_tool_call(
        "lookup", {"item_id": "b"}
    )
    assert parse_tool_call(content, supported_tools={"lookup"}) == ({}, False)


def test_context_uses_assistant_call_and_user_tool_response() -> None:
    context = ConversationContext("system", ["skill"], TOOLS)
    context.start_task("complete the task")
    context.append_assistant(
        ModelOutput.from_action(Action("lookup", {"item_id": "item-1"}))
    )
    context.append_tool_result({"found": True})
    messages = context.messages
    assert "<tools>" in messages[0]["content"]
    assert messages[2]["role"] == "assistant"
    assert "<tool_call>" in messages[2]["content"]
    assert messages[3]["role"] == "user"
    assert messages[3]["content"] == render_tool_response({"found": True})


def test_runner_hides_native_tools_and_normalizes_provider_action() -> None:
    requests = []

    class FakeEnv:
        def reset(self, seed=0, options=None):
            return EnvResponse({"ready": True})

        def describe(self):
            return EnvDescriptor("task-1", "look up the item", TOOLS, "v1")

        def step(self, action):
            assert action == Action("lookup", {"item_id": "item-1"})
            return EnvResponse({"found": True}, terminated=True)

        def evaluate(self):
            return EvaluationResult(True, "done", {"passed": 1, "total": 1})

    def generate(messages, tools):
        requests.append((list(messages), list(tools)))
        return ModelOutput("lookup", {"item_id": "item-1"})

    result = AgentRunner(FakeEnv(), CallableModelClient(generate), max_steps=1).run()
    request_messages, request_tools = requests[0]
    assert request_tools == []
    assert "<tools>" in request_messages[0]["content"]
    assert "retrieved_memory" not in request_messages[1]["content"]
    assert result.messages[2]["role"] == "assistant"
    assert "<tool_call>" in result.messages[2]["content"]
    assert result.messages[3]["role"] == "user"
    assert "<tool_response>" in result.messages[3]["content"]


def test_api_provider_parses_raw_agent0_action() -> None:
    content = "<think>inspect</think>\n" + render_tool_call(
        "lookup", {"item_id": "item-1"}
    )
    output = _parse_chat_completion(
        {"choices": [{"message": {"role": "assistant", "content": content}}]}
    )
    assert output.tool_name == "lookup"
    assert output.arguments == {"item_id": "item-1"}
    assert output.serialized_action is True
    assert output.content == content


def test_api_provider_normalizes_native_call_to_agent0_text() -> None:
    output = _parse_chat_completion(
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {
                                    "name": "lookup",
                                    "arguments": '{"item_id":"item-1"}',
                                },
                            }
                        ],
                    }
                }
            ]
        }
    )
    assert output.tool_name == "lookup"
    assert output.serialized_action is True
    assert output.content == render_tool_call("lookup", {"item_id": "item-1"})
