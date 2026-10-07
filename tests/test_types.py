# Copyright 2026 The EnvHarness Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the core data contracts in envharness.core.types.

When you add a new field to one of these models, add a test below so that
breaking changes to consumers (Bridges, Mutators, Objectives) get caught
at the type layer instead of inside a 10-min Docker rollout.
"""
import pytest
from pydantic import ValidationError

from envharness.agents.harness_agent import PROPOSE_TOOL, _candidate_from_args
from envharness.core.code_loader import load_rules_subclass
from envharness.core.types import (
    Action, Blocked, Observation, EnvResponse, Candidate, Decision,
    DecideResult, FailureAnalysis, Trace, EvaluationResult,
)


def test_action_extra_fields_forbidden():
    """Action uses extra='forbid' — typos in kwargs surface as ValidationError."""
    Action(name="combine", kwargs={"i": 0})
    with pytest.raises(ValidationError):
        Action(name="combine", kwargs={"i": 0}, oops="typo")


def test_blocked_default_kind():
    b = Blocked(reason="too risky")
    assert b.kind == "blocked"
    assert b.reason == "too risky"


def test_observation_text_and_extras():
    obs = Observation(text="see room", data={"key": "value"})
    assert obs.text == "see room"
    assert obs.data["key"] == "value"


def test_env_response_round_trip_json():
    """EnvResponse must be JSON-serializable (subprocess workers exchange JSON)."""
    r = EnvResponse(
        observation=Observation(text="hello"),
        reward=1.5, terminated=True, truncated=False,
        info={"won": True},
    )
    s = r.model_dump_json()
    r2 = EnvResponse.model_validate_json(s)
    assert r2.reward == 1.5
    assert r2.observation.text == "hello"
    assert r2.info == {"won": True}


def test_candidate_defaults():
    c = Candidate()
    assert c.rules_code == ""
    assert c.in_env_actions == []
    assert c.f_A == c.f_T == c.f_O == ""
    assert c.stage_rationale == ""
    assert c.f_A_rationale == c.f_T_rationale == c.f_O_rationale == ""
    assert c.rationale == ""
    assert c.change_type == "none"


def test_candidate_supports_stage_contract_and_combined_changes():
    stage = Candidate(in_env_actions=[Action(name="look")])
    contract = Candidate(rules_code="class _Rules(Rules):\n    pass")
    combined = Candidate(
        rules_code="class _Rules(Rules):\n    pass",
        in_env_actions=[Action(name="look")],
    )
    assert stage.change_type == "stage"
    assert contract.change_type == "contract"
    assert combined.change_type == "combined"


def test_propose_protocol_parses_four_independent_slots():
    params = PROPOSE_TOOL["function"]["parameters"]
    required = params["required"]
    assert set(required) == {"stage", "contract"}
    assert set(params["properties"]["stage"]["required"]) == {
        "in_env_actions", "rationale",
    }
    contract_schema = params["properties"]["contract"]
    assert set(contract_schema["required"]) == {"f_A", "f_T", "f_O"}
    for slot in ("f_A", "f_T", "f_O"):
        assert set(contract_schema["properties"][slot]["required"]) == {
            "code", "rationale",
        }

    candidate = _candidate_from_args({
        "stage": {
            "in_env_actions": [{"name": "look", "kwargs_json": "{}"}],
            "rationale": "start after inspecting the room",
        },
        "contract": {
            "f_A": {"code": (
                    "def filter_action(self, action, env_state):\n"
                    "    return action"
                ), "rationale": "keep the action boundary explicit"},
            "f_T": {"code": (
                    "def modify_transition(self, action, raw_response, env_state):\n"
                    "    return raw_response"
                ), "rationale": "preserve the transition result"},
            "f_O": {"code": (
                    "def filter_observation(self, obs, env_state):\n"
                    "    return obs"
                ), "rationale": "preserve the visible observation"},
        },
    })
    assert candidate.change_type == "combined"
    assert [action.name for action in candidate.in_env_actions] == ["look"]
    assert "def filter_action" in candidate.rules_code
    assert "def modify_transition" in candidate.rules_code
    assert "def filter_observation" in candidate.rules_code
    assert candidate.f_A.startswith("def filter_action")
    assert candidate.f_T.startswith("def modify_transition")
    assert candidate.f_O.startswith("def filter_observation")
    assert candidate.stage_rationale == "start after inspecting the room"
    assert candidate.f_A_rationale == "keep the action boundary explicit"
    assert candidate.f_T_rationale == "preserve the transition result"
    assert candidate.f_O_rationale == "preserve the visible observation"
    serialized = candidate.model_dump()
    assert serialized["stage_rationale"] == "start after inspecting the room"
    assert serialized["f_A_rationale"] == "keep the action boundary explicit"
    assert load_rules_subclass(candidate.rules_code).__name__ == "_Rules"


def test_propose_protocol_accepts_empty_slots_with_empty_rationales():
    candidate = _candidate_from_args({
        "stage": {"in_env_actions": [], "rationale": ""},
        "contract": {
            "f_A": {"code": "", "rationale": ""},
            "f_T": {"code": "", "rationale": ""},
            "f_O": {"code": "", "rationale": ""},
        },
    })
    assert candidate.change_type == "none"
    assert candidate.stage_rationale == ""
    assert candidate.f_A_rationale == ""
    assert candidate.f_T_rationale == ""
    assert candidate.f_O_rationale == ""


def test_decision_enum_strings():
    """Decision is a str-Enum: code can compare to either the enum or the string."""
    assert Decision.ACCEPT == "accept"
    assert Decision.REJECT.value == "reject"
    assert Decision("refine") is Decision.REFINE


def test_decide_result_with_failure_analysis():
    dr = DecideResult(
        decision=Decision.REFINE,
        failure_analysis=FailureAnalysis(primary_axis="A", label="too_strict"),
        rationale="A axis too narrow",
    )
    assert dr.decision is Decision.REFINE
    assert dr.failure_analysis.primary_axis == "A"


def test_failure_analysis_primary_axis_enum_constraint():
    """primary_axis is Stage S0, Contract A/T/O, or a diagnosis label."""
    FailureAnalysis(primary_axis="S0")
    FailureAnalysis(primary_axis="none")
    with pytest.raises(ValidationError):
        FailureAnalysis(primary_axis="Z")


def test_trace_minimal_shape():
    """A Trace must carry a Candidate; everything else has defaults."""
    t = Trace(episode_id="ep1", iteration_id="it1", task_id="t1",
              candidate=Candidate())
    assert t.kind == "accepted"           # default
    assert t.success is False
    assert t.steps == []


def test_evaluation_result_default_score():
    r = EvaluationResult(success=True)
    assert r.score == 0.0
    assert r.metrics == {}
