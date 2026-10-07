"""Unit tests for applying selected MCTS proposals to complete Candidates."""
from __future__ import annotations

from envharness.core.types import Action, Candidate, Trace
from envharness.orchestration.tree_runtime import (
    OrchestratorEvolutionRuntime,
    merge_candidate_change,
    validation_summary,
)
from envharness.orchestration.tree_search import EnvironmentChangeProposal


def test_merge_one_contract_change_preserves_other_parent_slots():
    parent = Candidate(
        in_env_actions=[Action(name="look")],
        stage_rationale="existing stage",
        f_A="def filter_action(self, action, env_state):\n    return action",
        f_A_rationale="existing A",
    )
    proposal = EnvironmentChangeProposal(
        proposal_id="root-f_O",
        environment_id="root",
        kind="f_O",
        change=Candidate(
            f_O="def filter_observation(self, obs, env_state):\n    return obs",
            f_O_rationale="new O",
        ),
    )

    child = merge_candidate_change(parent, proposal)
    assert child.in_env_actions == parent.in_env_actions
    assert child.f_A == parent.f_A
    assert child.f_O == proposal.change.f_O
    assert child.f_A_rationale == "existing A"
    assert child.f_O_rationale == "new O"
    assert "def filter_action" in child.rules_code
    assert "def filter_observation" in child.rules_code


def test_validation_summary_uses_fixed_rollout_group():
    traces = [
        Trace(
            episode_id=f"ep-{index}", iteration_id="it", task_id="task",
            candidate=Candidate(), success=index < 3,
            final_reward=float(index < 3), duration_steps=index + 1,
            error="boom" if index == 3 else None,
        )
        for index in range(4)
    ]
    summary = validation_summary(traces)
    assert summary.n_rollouts == 4
    assert summary.n_success == 3
    assert summary.success_rate == 0.75
    assert summary.n_errors == 1
    assert summary.objective_signal is not None
    assert summary.objective_signal.score == 0.75


def test_preloaded_root_requires_exact_rollout_group(tmp_path):
    class _Config:
        explicit_task_ids = [7]
        k_per_candidate = 4

    class _Orchestrator:
        config = _Config()

    runtime = OrchestratorEvolutionRuntime(
        orchestrator=_Orchestrator(),
        task_idx=0,
        output_dir=tmp_path,
        root_traces=[Trace(
            episode_id="one", iteration_id="old", task_id="task",
            candidate=Candidate(),
        )],
    )
    import pytest
    with pytest.raises(ValueError, match="must equal k_per_candidate"):
        runtime.create_root()
