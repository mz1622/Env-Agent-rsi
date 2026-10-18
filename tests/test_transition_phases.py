"""The T axis surrounds native execution while legacy T stays compatible."""

from envharness.agents.harness_agent import (
    HarnessAgentContext, LLMHarnessAgent, PROPOSE_TOOL,
    _contract_slots_from_rules_code, _is_transition_code,
    _rules_code_from_contract,
)
from envharness.bridges.alfworld.bridge import AlfworldEnv
from envharness.bridges.toy24.bridge import Toy24Env
from envharness.core.code_loader import load_rules_instance
from envharness.core.types import (
    Action, Candidate, EnvResponse, Observation, TransitionPlan,
)
from envharness.harnesses.rules import Rules
from envharness.orchestration.runner import (
    EnvSpec, EpisodeSpec, InProcessRunner, PolicySpec,
)


def _board() -> Toy24Env:
    env = Toy24Env()
    env.reset(options={"numbers": [3, 3, 7, 7], "target": 24})
    return env


def test_two_phase_transition_expands_one_visible_step_and_preserves_results():
    class Expand(Rules):
        def prepare_transition(self, action, env_state):
            assert env_state.step_count == 0
            return TransitionPlan(actions=[
                action,
                Action(name="combine", kwargs={"i": 2, "j": 0, "op": "add"}),
            ], context={"before": list(env_state.current_numbers)})

        def finalize_transition(self, action, plan, native_responses, env_state):
            assert plan.context["before"] == [3.0, 3.0, 7.0, 7.0]
            assert len(native_responses) == 2
            assert env_state.current_numbers == [7.0, 24.0]
            return super().finalize_transition(
                action, plan, native_responses, env_state,
            )

    inner = _board()
    rules = Expand(inner=inner)
    response = rules.step(Action(
        name="combine", kwargs={"i": 2, "j": 0, "op": "mul"},
    ))
    assert inner.state.step_count == 2
    assert response.observation.data["numbers"] == [7.0, 24.0]
    assert len(rules._last_step_hooks["transition_plan_actions"]) == 2
    assert len(rules._last_step_hooks["transition_native_responses"]) == 2


def test_zero_native_action_can_advance_virtual_state_without_engine_step():
    class VirtualTick(Rules):
        def prepare_transition(self, action, env_state):
            return TransitionPlan(actions=[], context={"tick": 1})

        def finalize_transition(self, action, plan, native_responses, env_state):
            assert native_responses == []
            env_state.extras["ticks"] = env_state.extras.get("ticks", 0) + 1
            current = super().finalize_transition(
                action, plan, native_responses, env_state,
            )
            return EnvResponse(
                observation=Observation(
                    text=current.observation.text + "\nWaiting tick completed.",
                    data=dict(current.observation.data),
                ),
                reward=current.reward, terminated=current.terminated,
                truncated=current.truncated, info=dict(current.info),
            )

    inner = _board()
    rules = VirtualTick(inner=inner)
    result = rules.step(Action(name="combine", kwargs={"i": 2, "j": 0, "op": "mul"}))
    assert inner.state.step_count == 0
    assert inner.state.extras["ticks"] == 1
    assert rules._last_step_hooks["transition_plan_actions"] == []
    assert rules._last_step_hooks["transition_native_responses"] == []
    assert result.reward == 0.0 and not result.terminated
    assert "Waiting tick completed." in result.observation.text
    assert "Waiting tick completed." in rules.observe().text


def test_native_macro_stops_at_terminal_result():
    class StopThenCombine(Rules):
        def prepare_transition(self, action, env_state):
            return TransitionPlan(actions=[
                Action(name="stop"),
                Action(name="combine", kwargs={"i": 2, "j": 0, "op": "mul"}),
            ])

    inner = _board()
    rules = StopThenCombine(inner=inner)
    result = rules.step(Action(name="stop"))
    assert result.terminated
    assert inner.state.step_count == 1
    assert len(rules._last_step_hooks["transition_native_responses"]) == 1


def test_two_phase_source_compiles_and_round_trips_as_one_f_t_slot():
    paired = (
        "def prepare_transition(self, action, env_state):\n"
        "    return TransitionPlan(actions=[action])\n\n"
        "def finalize_transition(self, action, plan, native_responses, env_state):\n"
        "    return super().finalize_transition(action, plan, native_responses, env_state)"
    )
    legacy = (
        "def modify_transition(self, action, raw_response, env_state):\n"
        "    return raw_response"
    )
    assert _is_transition_code(paired)
    assert _is_transition_code(legacy)
    assert not _is_transition_code(paired + "\n\n" + legacy)
    code = _rules_code_from_contract({"f_T": paired})
    assert _contract_slots_from_rules_code(code)["f_T"] == paired
    rules = load_rules_instance(code, inner=_board())
    result = rules.step(Action(name="combine", kwargs={"i": 2, "j": 0, "op": "mul"}))
    assert result.observation.data["numbers"] == [3.0, 7.0, 21.0]


def test_runner_trace_retains_each_native_action_of_a_two_phase_transition():
    paired = (
        "def prepare_transition(self, action, env_state):\n"
        "    return TransitionPlan(actions=[action, Action(name='combine', "
        "kwargs={'i': 2, 'j': 0, 'op': 'add'})])\n\n"
        "def finalize_transition(self, action, plan, native_responses, env_state):\n"
        "    return super().finalize_transition(action, plan, native_responses, env_state)"
    )
    candidate = Candidate(
        f_T=paired,
        rules_code=_rules_code_from_contract({"f_T": paired}),
    )
    trace = InProcessRunner().run(EpisodeSpec(
        env=EnvSpec(
            import_path="envharness.bridges.toy24:Toy24Env",
            reset_options={"numbers": [3, 3, 7, 7], "target": 24},
        ),
        candidate=candidate,
        policy=PolicySpec(
            client_factory="envharness.infra.llm:ScriptedClient",
            client_kwargs={"model_id": "scripted/seq", "script": [{
                "name": "combine", "kwargs": {"i": 2, "j": 0, "op": "mul"},
            }]},
            task_prompt="Make 24.", action_format="function_calling",
        ),
        iteration_id="two-phase", max_steps=1,
    ))
    assert trace.error is None
    assert len(trace.steps) == 1
    step = trace.steps[0]
    assert [action.kwargs["op"] for action in step.transition_plan_actions] == [
        "mul", "add",
    ]
    assert len(step.transition_native_responses) == 2
    assert step.filtered_observation.data["numbers"] == [7.0, 24.0]


def test_alfworld_prompt_and_tool_expose_two_phase_f_t_contract():
    agent = LLMHarnessAgent(client=None)
    context = HarnessAgentContext(
        env_state_schema=AlfworldEnv.env_state_schema(),
        task_description="cool an object",
    )
    system = agent._base_messages(context, intro="Propose now.")[0].content
    assert "prepare_transition" in system
    assert "finalize_transition" in system
    assert "CURRENT held object's type" in system
    schema = PROPOSE_TOOL["function"]["parameters"]["properties"]
    description = schema["contract"]["properties"]["f_T"]["properties"]
    assert "prepare_transition" in description["code"]["description"]
