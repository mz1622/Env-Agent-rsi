"""Protocol tests for the external environment-tree search scaffold.

Fake ports verify ordering and schemas without implementing MCTS or invoking a
real model/environment.
"""
from __future__ import annotations

import json

import pytest

from envharness.core.types import Action, Candidate, Trace
from envharness.orchestration.tree_search import (
    EnvironmentChangeProposal,
    EnvironmentNode,
    EnvironmentTree,
    ExternalSearchCoordinator,
    NodeSubmission,
    ProposalBatch,
    SearchCommand,
    UniformPUCTPolicy,
    ValidationSummary,
    split_candidate_envelope,
)


def _trace(node_id: str, candidate: Candidate, success: bool = True) -> Trace:
    return Trace(
        episode_id=f"episode-{node_id}",
        iteration_id="tree-test",
        task_id="7",
        candidate=candidate,
        candidate_id=node_id,
        success=success,
        kind="exploration",
    )


def _submission(
    node_id: str,
    parent_node_id: str | None,
    depth: int,
    candidate: Candidate | None = None,
    incoming_proposal_id: str | None = None,
    success_rate: float = 1.0,
) -> NodeSubmission:
    candidate = candidate or Candidate()
    trace = _trace(node_id, candidate)
    return NodeSubmission(
        node=EnvironmentNode(
            node_id=node_id,
            parent_node_id=parent_node_id,
            incoming_proposal_id=incoming_proposal_id,
            depth=depth,
            task_id=7,
            candidate=candidate,
            checkpoint={
                "schema_version": 1,
                "env": {"type": "toy24", "state": {"reset_seed": 7}},
                "harnesses": [],
                "metadata": {"node_id": node_id},
            },
            validation=ValidationSummary(
                n_rollouts=1,
                n_success=int(success_rate == 1.0),
                success_rate=success_rate,
                trace_ids=[trace.episode_id],
            ),
        ),
        validation_traces=[trace],
    )


def _proposal(environment_id: str, proposal_id: str,
              kind: str) -> EnvironmentChangeProposal:
    if kind == "stage":
        change = Candidate(
            in_env_actions=[Action(name="look")],
            stage_rationale="change the initial state",
        )
    else:
        methods = {
            "f_A": "def filter_action(self, action, env_state):\n    return action",
            "f_T": (
                "def modify_transition(self, action, raw_response, env_state):\n"
                "    return raw_response"
            ),
            "f_O": "def filter_observation(self, obs, env_state):\n    return obs",
        }
        change = Candidate(**{
            kind: methods[kind],
            f"{kind}_rationale": f"independent {kind} change",
        })
    return EnvironmentChangeProposal(
        proposal_id=proposal_id,
        environment_id=environment_id,
        kind=kind,
        change=change,
    )


def test_four_slot_envelope_splits_into_four_independent_proposals():
    envelope = Candidate(
        in_env_actions=[Action(name="look")],
        stage_rationale="stage reason",
        f_A="def filter_action(self, action, env_state):\n    return action",
        f_A_rationale="A reason",
        f_T=("def modify_transition(self, action, raw_response, env_state):\n"
             "    return raw_response"),
        f_T_rationale="T reason",
        f_O="def filter_observation(self, obs, env_state):\n    return obs",
        f_O_rationale="O reason",
    )
    batch = split_candidate_envelope("root", envelope, "root-turn-1")

    assert [p.kind for p in batch.proposals] == ["stage", "f_A", "f_T", "f_O"]
    assert all(p.environment_id == "root" for p in batch.proposals)
    assert all(sum([
        bool(p.change.in_env_actions), bool(p.change.f_A),
        bool(p.change.f_T), bool(p.change.f_O),
    ]) == 1 for p in batch.proposals)


def test_empty_slots_do_not_become_proposals():
    envelope = Candidate(
        f_O="def filter_observation(self, obs, env_state):\n    return obs",
        f_O_rationale="only observation applies",
    )
    batch = split_candidate_envelope("env-1", envelope, "env-1-turn-1")
    assert len(batch.proposals) == 1
    assert batch.proposals[0].kind == "f_O"


def test_uniform_puct_assigns_priors_only_to_nonempty_slots():
    tree = EnvironmentTree()
    root = _submission("root", None, 0)
    tree.add(root)
    policy = UniformPUCTPolicy(c_puct=1.0)
    policy.observe_node(root, tree.snapshot())

    envelope = Candidate(
        f_A="def filter_action(self, action, env_state):\n    return action",
        f_A_rationale="A change",
        f_O="def filter_observation(self, obs, env_state):\n    return obs",
        f_O_rationale="O change",
    )
    batch = split_candidate_envelope("root", envelope, "root-turn-1")
    policy.observe_proposals(batch, tree.snapshot())

    assert [proposal.kind for proposal in batch.proposals] == ["f_A", "f_O"]
    assert policy.prior_for("root-turn-1-f_A") == pytest.approx(0.5)
    assert policy.prior_for("root-turn-1-f_O") == pytest.approx(0.5)

    command = policy.next_command(tree.snapshot())
    assert command.action == "apply"
    assert command.environment_id == "root"
    assert command.proposal == batch.proposals[0]


def test_uniform_puct_tries_cached_sibling_after_low_value_branch():
    tree = EnvironmentTree()
    root = _submission("root", None, 0, success_rate=1.0)
    tree.add(root)
    policy = UniformPUCTPolicy(c_puct=1.0)
    policy.observe_node(root, tree.snapshot())

    batch = ProposalBatch(
        environment_id="root",
        proposals=[
            _proposal("root", "root-stage", "stage"),
            _proposal("root", "root-f_A", "f_A"),
        ],
    )
    policy.observe_proposals(batch, tree.snapshot())
    first = policy.next_command(tree.snapshot())
    assert first.proposal is not None
    assert first.proposal.proposal_id == "root-stage"

    low_child = _submission(
        "node-1", "root", 1,
        candidate=first.proposal.change,
        incoming_proposal_id=first.proposal.proposal_id,
        success_rate=0.0,
    )
    tree.add(low_child)
    policy.observe_node(low_child, tree.snapshot())

    # The direct edge return is -1.0.  With its one visit, its PUCT score
    # is below the still-untried cached sibling, so selection switches back.
    assert policy.puct_score_for("root-stage") == pytest.approx(-0.75)
    assert policy.puct_score_for("root-f_A") == pytest.approx(0.5)
    second = policy.next_command(tree.snapshot())
    assert second.action == "apply"
    assert second.environment_id == "root"
    assert second.proposal is not None
    assert second.proposal.proposal_id == "root-f_A"


def test_uniform_puct_emits_stop_when_no_proposal_is_available():
    tree = EnvironmentTree()
    root = _submission("root", None, 0)
    tree.add(root)
    policy = UniformPUCTPolicy()
    policy.observe_node(root, tree.snapshot())
    policy.observe_proposals(
        ProposalBatch(environment_id="root", proposals=[]),
        tree.snapshot(),
    )

    command = policy.next_command(tree.snapshot())
    assert command.action == "stop"
    assert command.best_node_id == "root"
    assert command.rationale == "no pending proposal is reachable"


def test_external_protocol_payload_is_json_serializable():
    payload = _submission("root", None, 0)
    decoded = json.loads(payload.model_dump_json())
    assert decoded["node"]["node_id"] == "root"
    assert decoded["node"]["validation"]["success_rate"] == 1.0


def test_tree_enforces_parent_depth_and_task_invariants():
    tree = EnvironmentTree()
    tree.add(_submission("root", None, 0))
    tree.add(_submission("child", "root", 1,
                         incoming_proposal_id="root-stage"))
    assert tree.snapshot().expansion_count == 1

    with pytest.raises(ValueError, match="depth"):
        tree.add(_submission("bad-depth", "root", 3,
                             incoming_proposal_id="root-f_A"))


class _FakeRuntime:
    def __init__(self):
        self.generated_for: list[str] = []
        self.applied_to: list[tuple[str, str]] = []

    def create_root(self):
        return _submission("root", None, 0)

    def generate_proposals(self, environment, validation_traces, snapshot):
        self.generated_for.append(environment.node_id)
        assert validation_traces
        if environment.node_id == "root":
            proposals = [
                _proposal("root", "root-stage", "stage"),
                _proposal("root", "root-f_A", "f_A"),
            ]
        elif environment.node_id == "node-1":
            proposals = [_proposal("node-1", "node-1-f_O", "f_O")]
        else:
            proposals = []
        return ProposalBatch(
            environment_id=environment.node_id,
            proposals=proposals,
        )

    def apply_and_validate(self, environment, proposal,
                           validation_traces, snapshot):
        self.applied_to.append((environment.node_id, proposal.proposal_id))
        node_number = len(self.applied_to)
        return _submission(
            node_id=f"node-{node_number}",
            parent_node_id=environment.node_id,
            depth=environment.depth + 1,
            candidate=proposal.change,
            incoming_proposal_id=proposal.proposal_id,
        )


class _FakeExternalPolicy:
    """Second choice deliberately selects an older root proposal."""

    def __init__(self):
        self.nodes: list[str] = []
        self.batches: list[str] = []
        self.pending: dict[str, EnvironmentChangeProposal] = {}

    def observe_node(self, submission, snapshot):
        self.nodes.append(submission.node.node_id)

    def observe_proposals(self, batch, snapshot):
        self.batches.append(batch.environment_id)
        self.pending.update({p.proposal_id: p for p in batch.proposals})

    def next_command(self, snapshot):
        if snapshot.expansion_count == 0:
            proposal = self.pending["root-stage"]
            return SearchCommand(
                action="apply", environment_id="root", proposal=proposal,
            )
        if snapshot.expansion_count == 1:
            # The newest batch belongs to node-1, but MCTS chooses a pending
            # proposal generated for root in the previous turn.
            proposal = self.pending["root-f_A"]
            return SearchCommand(
                action="apply", environment_id="root", proposal=proposal,
            )
        return SearchCommand(
            action="stop",
            best_node_id="node-2",
            rationale="external selector chose the second root branch",
        )


def test_coordinator_can_apply_historical_proposal_to_its_source_environment():
    runtime = _FakeRuntime()
    policy = _FakeExternalPolicy()
    result = ExternalSearchCoordinator(runtime, policy).run(max_expansions=3)

    assert runtime.generated_for == ["root", "node-1", "node-2"]
    assert runtime.applied_to == [
        ("root", "root-stage"),
        ("root", "root-f_A"),
    ]
    assert policy.nodes == ["root", "node-1", "node-2"]
    assert policy.batches == ["root", "node-1", "node-2"]
    assert result.best_node_id == "node-2"
    assert result.expansion_count == 2


def test_coordinator_defaults_to_uniform_puct_and_uses_budget_boundary():
    runtime = _FakeRuntime()
    result = ExternalSearchCoordinator(runtime).run(max_expansions=2)

    # The uniform root prior ties initially, so the first generated edge wins
    # deterministically.  After validating it, PUCT returns to the untried
    # cached root sibling rather than consuming the new child proposal first.
    assert runtime.generated_for == ["root", "node-1"]
    assert runtime.applied_to == [
        ("root", "root-stage"),
        ("root", "root-f_A"),
    ]
    assert result.expansion_count == 2
    assert result.stop_rationale == "max_expansions reached"


def test_external_cannot_retarget_or_modify_a_proposal():
    proposal = _proposal("root", "root-f_A", "f_A")
    with pytest.raises(ValueError, match="not generated"):
        SearchCommand(
            action="apply", environment_id="another-env", proposal=proposal,
        )


def test_search_command_has_no_refine_or_reject_action():
    with pytest.raises(ValueError):
        SearchCommand(action="refine", environment_id="root")
    with pytest.raises(ValueError):
        SearchCommand(action="reject", best_node_id="root")
