# Copyright 2026 The EnvHarness Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tree-search contracts and a local PUCT baseline for environment evolution.

The JSON-safe boundary lets EnvRigger/EnvHarness send independent candidate
changes to either a remote selector or :class:`UniformPUCTPolicy` below:

1. EnvRigger generates up to four independent changes for one environment.
2. The whole proposal batch is sent to external MCTS and retained there.
3. MCTS may select a proposal from this batch or an older batch, returning the
   proposal together with the environment id it was generated for.
4. EnvRigger/EnvHarness applies that selected change to that environment.
5. The resulting environment is validated before becoming a child node.

The selector only selects an already-generated edge; it never writes or
materializes environment code itself.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from envharness.core.types import Candidate, ObjectiveSignal, Trace


ChangeKind = Literal["stage", "f_A", "f_T", "f_O"]


class ValidationSummary(BaseModel):
    """Compact result of the mandatory K-rollout validation of one node."""

    model_config = ConfigDict(extra="forbid")
    n_rollouts: int = 0
    n_success: int = 0
    success_rate: float = 0.0
    avg_reward: float = 0.0
    avg_steps: float = 0.0
    n_errors: int = 0
    objective_signal: ObjectiveSignal | None = None
    trace_ids: list[str] = Field(default_factory=list)


class EnvironmentNode(BaseModel):
    """One fully specified and already validated environment tree node.

    ``candidate`` is the complete Stage + Contract state of this environment.
    ``parent_node_id`` records which environment the selected change was
    applied to. The root uses an empty Candidate.
    """

    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[1] = 1
    node_id: str
    parent_node_id: str | None = None
    incoming_proposal_id: str | None = None
    depth: int = 0
    task_id: int
    candidate: Candidate = Field(default_factory=Candidate)
    checkpoint: dict[str, Any]
    validation: ValidationSummary
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _root_and_depth_are_consistent(self):
        if self.parent_node_id is None:
            if self.depth != 0:
                raise ValueError("root node must have depth=0")
            if self.incoming_proposal_id is not None:
                raise ValueError("root node cannot have incoming_proposal_id")
        else:
            if self.depth < 1:
                raise ValueError("non-root node must have depth>=1")
            if not self.incoming_proposal_id:
                raise ValueError("non-root node requires incoming_proposal_id")
        return self


class NodeSubmission(BaseModel):
    """A node pushed externally only after its environment was validated."""

    model_config = ConfigDict(extra="forbid")
    node: EnvironmentNode
    validation_traces: list[Trace] = Field(default_factory=list)


class EnvironmentChangeProposal(BaseModel):
    """One independent, not-yet-applied environment change.

    A proposal is bound to ``environment_id``. It may be selected later, but
    only for that same source environment. ``change`` contains exactly one of
    Stage, f_A, f_T or f_O; the other three slots must be empty.
    """

    model_config = ConfigDict(extra="forbid")
    proposal_id: str
    environment_id: str
    kind: ChangeKind
    change: Candidate
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _candidate_contains_exactly_declared_change(self):
        populated = {
            "stage": bool(self.change.in_env_actions),
            "f_A": bool(self.change.f_A.strip()),
            "f_T": bool(self.change.f_T.strip()),
            "f_O": bool(self.change.f_O.strip()),
        }
        active = [name for name, present in populated.items() if present]
        if active != [self.kind]:
            raise ValueError(
                "proposal change must populate exactly its declared slot; "
                f"kind={self.kind!r}, populated={active!r}"
            )
        rationale = (
            self.change.stage_rationale if self.kind == "stage"
            else getattr(self.change, f"{self.kind}_rationale")
        )
        if not rationale.strip():
            raise ValueError("a populated proposal requires its own rationale")
        return self


class ProposalBatch(BaseModel):
    """The zero-to-four independent proposals emitted in one EnvRigger turn."""

    model_config = ConfigDict(extra="forbid")
    environment_id: str
    proposals: list[EnvironmentChangeProposal] = Field(
        default_factory=list, max_length=4,
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _proposals_are_bound_and_unique(self):
        ids: set[str] = set()
        kinds: set[str] = set()
        for proposal in self.proposals:
            if proposal.environment_id != self.environment_id:
                raise ValueError("all proposals must target batch environment_id")
            if proposal.proposal_id in ids:
                raise ValueError(f"duplicate proposal_id: {proposal.proposal_id}")
            if proposal.kind in kinds:
                raise ValueError(f"duplicate proposal kind: {proposal.kind}")
            ids.add(proposal.proposal_id)
            kinds.add(proposal.kind)
        return self


def split_candidate_envelope(
    environment_id: str,
    envelope: Candidate,
    proposal_id_prefix: str,
) -> ProposalBatch:
    """Split one four-slot EnvRigger response into independent proposals.

    ``proposal_id_prefix`` is supplied by the runtime so ids remain stable
    across process/RPC boundaries (for example ``env-3-turn-2``). Empty slots
    are omitted. The function does not apply or validate any change.
    """
    proposals: list[EnvironmentChangeProposal] = []
    if envelope.in_env_actions:
        proposals.append(EnvironmentChangeProposal(
            proposal_id=f"{proposal_id_prefix}-stage",
            environment_id=environment_id,
            kind="stage",
            change=Candidate(
                in_env_actions=list(envelope.in_env_actions),
                stage_rationale=envelope.stage_rationale,
            ),
        ))
    for kind in ("f_A", "f_T", "f_O"):
        code = getattr(envelope, kind)
        if not code.strip():
            continue
        proposals.append(EnvironmentChangeProposal(
            proposal_id=f"{proposal_id_prefix}-{kind}",
            environment_id=environment_id,
            kind=kind,
            change=Candidate(**{
                kind: code,
                f"{kind}_rationale": getattr(envelope, f"{kind}_rationale"),
            }),
        ))
    return ProposalBatch(environment_id=environment_id, proposals=proposals)


class TreeSnapshot(BaseModel):
    """Read-only environment tree structure sent with external events."""

    model_config = ConfigDict(extra="forbid")
    root_node_id: str
    nodes: list[EnvironmentNode]
    expansion_count: int = 0


class SearchCommand(BaseModel):
    """Selection returned by external MCTS after it receives a proposal batch.

    ``apply`` returns an environment id plus a proposal. That proposal may come
    from any previously submitted batch, not necessarily the newest one.
    ``stop`` selects the best already validated environment node.
    """

    model_config = ConfigDict(extra="forbid")
    action: Literal["apply", "stop"]
    environment_id: str | None = None
    proposal: EnvironmentChangeProposal | None = None
    best_node_id: str | None = None
    rationale: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _fields_match_action(self):
        if self.action == "apply":
            if not self.environment_id or self.proposal is None:
                raise ValueError("apply requires environment_id and proposal")
            if self.proposal.environment_id != self.environment_id:
                raise ValueError("proposal was not generated for environment_id")
            if self.best_node_id is not None:
                raise ValueError("apply must not set best_node_id")
        else:
            if not self.best_node_id:
                raise ValueError("stop requires best_node_id")
            if self.environment_id is not None or self.proposal is not None:
                raise ValueError("stop must not include an environment/proposal")
        return self


class SearchRunResult(BaseModel):
    """Protocol-level result; it contains no MCTS-internal statistics."""

    model_config = ConfigDict(extra="forbid")
    best_node_id: str
    root_node_id: str
    expansion_count: int
    stop_rationale: str = ""
    snapshot: TreeSnapshot


class EnvironmentEvolutionRuntime(Protocol):
    """EnvRigger/EnvHarness port; concrete rollout integration comes later."""

    def create_root(self) -> NodeSubmission:
        """Validate the unmodified environment and return the root node."""
        ...

    def generate_proposals(
        self,
        environment: EnvironmentNode,
        validation_traces: list[Trace],
        snapshot: TreeSnapshot,
    ) -> ProposalBatch:
        """Use EnvRigger to emit up to four independent changes."""
        ...

    def apply_and_validate(
        self,
        environment: EnvironmentNode,
        proposal: EnvironmentChangeProposal,
        validation_traces: list[Trace],
        snapshot: TreeSnapshot,
    ) -> NodeSubmission:
        """Apply the chosen change in EnvRigger/EnvHarness, then validate it."""
        ...


class ExternalTreePolicy(Protocol):
    """Transport-neutral port to a future external MCTS implementation."""

    def observe_node(self, submission: NodeSubmission,
                     snapshot: TreeSnapshot) -> None:
        """Receive a validated root or child node."""
        ...

    def observe_proposals(self, batch: ProposalBatch,
                          snapshot: TreeSnapshot) -> None:
        """Receive and retain the newest zero-to-four candidate edges."""
        ...

    def next_command(self, snapshot: TreeSnapshot) -> SearchCommand:
        """Select any retained proposal for its source environment, or stop."""
        ...


class EnvironmentTree:
    """Minimal validated node store; contains no selection/return algorithm."""

    def __init__(self):
        self._nodes: dict[str, EnvironmentNode] = {}
        self._traces: dict[str, list[Trace]] = {}
        self._root_node_id: str | None = None

    def add(self, submission: NodeSubmission) -> None:
        node = submission.node
        if node.node_id in self._nodes:
            raise ValueError(f"duplicate node_id: {node.node_id}")
        if node.parent_node_id is None:
            if self._root_node_id is not None:
                raise ValueError("tree already has a root")
            self._root_node_id = node.node_id
        else:
            parent = self._nodes.get(node.parent_node_id)
            if parent is None:
                raise ValueError(f"unknown parent_node_id: {node.parent_node_id}")
            if node.depth != parent.depth + 1:
                raise ValueError("child depth must equal parent depth + 1")
            if node.task_id != parent.task_id:
                raise ValueError("child and parent must target the same task_id")
        self._nodes[node.node_id] = node
        self._traces[node.node_id] = list(submission.validation_traces)

    def get(self, node_id: str) -> EnvironmentNode:
        try:
            return self._nodes[node_id]
        except KeyError as exc:
            raise ValueError(f"unknown node_id: {node_id}") from exc

    def traces_for(self, node_id: str) -> list[Trace]:
        self.get(node_id)
        return list(self._traces[node_id])

    def snapshot(self) -> TreeSnapshot:
        if self._root_node_id is None:
            raise ValueError("tree has no root")
        nodes = sorted(self._nodes.values(), key=lambda n: (n.depth, n.node_id))
        return TreeSnapshot(
            root_node_id=self._root_node_id,
            nodes=nodes,
            expansion_count=max(0, len(nodes) - 1),
        )

    def best_validated_node_id(self) -> str:
        """Return the best node under the current success-rate objective.

        This is a budget-boundary fallback, not a search-policy ``stop``
        decision.  A shallower node wins exact ties so the root remains the
        conservative choice when no candidate has demonstrated an improvement.
        """
        if self._root_node_id is None:
            raise ValueError("tree has no root")
        return min(
            self._nodes.values(),
            key=lambda node: (
                -node.validation.success_rate,
                -node.validation.avg_reward,
                node.depth,
                node.node_id,
            ),
        ).node_id


class ProposalCatalog:
    """Pending EnvRigger proposals, including proposals from older batches."""

    def __init__(self):
        self._pending: dict[str, EnvironmentChangeProposal] = {}
        self._seen: set[str] = set()

    def add_batch(self, batch: ProposalBatch) -> None:
        for proposal in batch.proposals:
            if proposal.proposal_id in self._seen:
                raise ValueError(f"duplicate proposal_id: {proposal.proposal_id}")
            self._pending[proposal.proposal_id] = proposal
            self._seen.add(proposal.proposal_id)

    def consume_selected(
        self,
        environment_id: str,
        selected: EnvironmentChangeProposal,
    ) -> EnvironmentChangeProposal:
        stored = self._pending.get(selected.proposal_id)
        if stored is None:
            raise ValueError(f"unknown or consumed proposal_id: {selected.proposal_id}")
        if stored.environment_id != environment_id:
            raise ValueError("selected proposal belongs to another environment")
        if stored.model_dump() != selected.model_dump():
            raise ValueError("external search must return the proposal unchanged")
        del self._pending[selected.proposal_id]
        return stored


@dataclass
class PUCTNodeStats:
    """Visit/accumulated-return (``G``) statistics for one environment node."""

    visit_count: int = 0
    g_sum: float = 0.0

    @property
    def mean_g(self) -> float:
        return self.g_sum / self.visit_count if self.visit_count else 0.0


@dataclass
class PUCTEdgeStats:
    """One EnvRigger proposal as a deferred, one-shot MCTS edge."""

    proposal: EnvironmentChangeProposal
    prior: float
    order: int
    visit_count: int = 0
    g_sum: float = 0.0
    child_node_id: str | None = None
    in_flight: bool = False

    @property
    def mean_g(self) -> float:
        return self.g_sum / self.visit_count if self.visit_count else 0.0


class NoSelectableProposalError(RuntimeError):
    """Backward-compatible error type from the pre-stop PUCT policy."""


class UniformPUCTPolicy:
    """PUCT policy whose local prior is uniform over non-empty proposals.

    EnvRigger emits a four-slot envelope.  ``split_candidate_envelope`` has
    already removed empty slots by the time this policy receives a
    :class:`ProposalBatch`, so every proposal in a batch is a real candidate.
    The policy assigns all candidates for the same parent an equal prior; no
    JEV or other model score is queried.

    Each proposal is retained as a deferred edge.  It becomes a child only
    after the coordinator applies it and returns a validated
    :class:`EnvironmentNode`.  A newly validated leaf backs up its cumulative
    success-rate change from the root.  That return is the sum of the direct
    parent/child success-rate deltas along the path, so PUCT can prefer a
    multi-step improvement rather than only a locally positive first change.

    ``next_command`` returns ``action='stop'`` when no deferred proposal is
    reachable, selecting the best already validated environment.
    """

    def __init__(self, c_puct: float = 1.0):
        if not isfinite(c_puct) or c_puct < 0:
            raise ValueError("c_puct must be a finite value >= 0")
        self.c_puct = float(c_puct)
        self._root_node_id: str | None = None
        self._nodes: dict[str, EnvironmentNode] = {}
        self._node_stats: dict[str, PUCTNodeStats] = {}
        self._edges: dict[str, PUCTEdgeStats] = {}
        self._edge_ids_by_parent: dict[str, list[str]] = {}
        self._next_edge_order = 0

    def observe_node(self, submission: NodeSubmission,
                     snapshot: TreeSnapshot) -> None:
        """Register a validated node and back up its observed ``G`` return once."""
        node = submission.node
        if node.node_id in self._nodes:
            raise ValueError(f"duplicate observed node_id: {node.node_id}")

        self._nodes[node.node_id] = node
        self._node_stats[node.node_id] = PUCTNodeStats()

        if node.parent_node_id is None:
            if self._root_node_id is not None:
                raise ValueError("PUCT policy already has a root node")
            if snapshot.root_node_id != node.node_id:
                raise ValueError("root submission does not match tree snapshot")
            self._root_node_id = node.node_id
            return

        parent_id = node.parent_node_id
        if parent_id not in self._nodes:
            raise ValueError(f"unknown PUCT parent node: {parent_id}")
        proposal_id = node.incoming_proposal_id
        if proposal_id is None:  # narrowed by EnvironmentNode validation
            raise ValueError("non-root node omitted incoming_proposal_id")
        edge = self._edges.get(proposal_id)
        if edge is None:
            raise ValueError(
                "child references a proposal that was not observed by PUCT: "
                f"{proposal_id}"
            )
        if edge.proposal.environment_id != parent_id:
            raise ValueError("child proposal belongs to another parent")
        if edge.child_node_id is not None:
            raise ValueError(f"proposal already has a child: {proposal_id}")

        edge.child_node_id = node.node_id
        edge.in_flight = False
        self._backpropagate(node.node_id)

    def observe_proposals(self, batch: ProposalBatch,
                          snapshot: TreeSnapshot) -> None:
        """Cache a proposal batch and assign a uniform local prior.

        The production coordinator generates one batch per node.  Repeated
        batches are nevertheless safe: all candidates currently known for the
        parent are re-normalized to one uniform distribution.
        """
        parent_id = batch.environment_id
        if parent_id not in self._nodes:
            raise ValueError(f"proposal batch has unknown environment_id: {parent_id}")
        if parent_id not in {node.node_id for node in snapshot.nodes}:
            raise ValueError("proposal batch parent is absent from tree snapshot")

        siblings = self._edge_ids_by_parent.setdefault(parent_id, [])
        for proposal in batch.proposals:
            if proposal.proposal_id in self._edges:
                raise ValueError(f"duplicate observed proposal_id: {proposal.proposal_id}")
            if proposal.environment_id != parent_id:
                raise ValueError("proposal batch contains another environment's proposal")
            self._edges[proposal.proposal_id] = PUCTEdgeStats(
                proposal=proposal,
                prior=0.0,
                order=self._next_edge_order,
            )
            self._next_edge_order += 1
            siblings.append(proposal.proposal_id)

        if siblings:
            prior = 1.0 / len(siblings)
            for proposal_id in siblings:
                self._edges[proposal_id].prior = prior

    def next_command(self, snapshot: TreeSnapshot) -> SearchCommand:
        """Return the highest-scoring deferred proposal as an ``apply`` command."""
        if self._root_node_id is None:
            raise ValueError("PUCT policy has not observed a root node")
        if snapshot.root_node_id != self._root_node_id:
            raise ValueError("tree snapshot belongs to another root")

        edge = self._select_deferred_edge()
        if edge is None:
            best = min(
                snapshot.nodes,
                key=lambda node: (
                    -node.validation.success_rate,
                    -node.validation.avg_reward,
                    node.depth,
                    node.node_id,
                ),
            )
            return SearchCommand(
                action="stop",
                best_node_id=best.node_id,
                rationale="no pending proposal is reachable",
            )
        edge.in_flight = True
        return SearchCommand(
            action="apply",
            environment_id=edge.proposal.environment_id,
            proposal=edge.proposal,
        )

    def prior_for(self, proposal_id: str) -> float:
        """Expose a cached local prior for telemetry and tests."""
        try:
            return self._edges[proposal_id].prior
        except KeyError as exc:
            raise ValueError(f"unknown proposal_id: {proposal_id}") from exc

    def puct_score_for(self, proposal_id: str) -> float:
        """Expose the current PUCT score for telemetry and tests."""
        try:
            edge = self._edges[proposal_id]
        except KeyError as exc:
            raise ValueError(f"unknown proposal_id: {proposal_id}") from exc
        return self._puct_score(edge)

    def _backpropagate(self, leaf_node_id: str) -> None:
        if self._root_node_id is None:
            raise ValueError("cannot backpropagate without a root")
        root_rate = self._nodes[self._root_node_id].validation.success_rate
        leaf_rate = self._nodes[leaf_node_id].validation.success_rate
        g = leaf_rate - root_rate

        current_id = leaf_node_id
        while True:
            node_stats = self._node_stats[current_id]
            node_stats.visit_count += 1
            node_stats.g_sum += g
            if current_id == self._root_node_id:
                return

            current = self._nodes[current_id]
            proposal_id = current.incoming_proposal_id
            if proposal_id is None or current.parent_node_id is None:
                raise ValueError("invalid non-root node during PUCT backpropagation")
            edge = self._edges[proposal_id]
            edge.visit_count += 1
            edge.g_sum += g
            current_id = current.parent_node_id

    def _select_deferred_edge(self) -> PUCTEdgeStats | None:
        """Follow PUCT choices until reaching an unmaterialized edge."""
        if self._root_node_id is None:
            return None

        current_id = self._root_node_id
        visited_nodes: set[str] = set()
        while current_id not in visited_nodes:
            visited_nodes.add(current_id)
            candidate_edges = [
                self._edges[proposal_id]
                for proposal_id in self._edge_ids_by_parent.get(current_id, [])
                if self._edge_can_reach_deferred_proposal(
                    self._edges[proposal_id], set(),
                )
            ]
            if not candidate_edges:
                return None

            # Lower ``order`` wins an exact tie.  The deterministic behavior
            # makes a uniform-prior baseline reproducible.
            selected = max(
                candidate_edges,
                key=lambda edge: (self._puct_score(edge), -edge.order),
            )
            if selected.child_node_id is None:
                return selected
            current_id = selected.child_node_id
        raise ValueError("PUCT edge graph contains a cycle")

    def _edge_can_reach_deferred_proposal(
        self,
        edge: PUCTEdgeStats,
        seen_nodes: set[str],
    ) -> bool:
        if edge.child_node_id is None:
            return not edge.in_flight
        return self._node_has_deferred_proposal(edge.child_node_id, seen_nodes)

    def _node_has_deferred_proposal(
        self,
        node_id: str,
        seen_nodes: set[str],
    ) -> bool:
        if node_id in seen_nodes:
            return False
        seen_nodes.add(node_id)
        for proposal_id in self._edge_ids_by_parent.get(node_id, []):
            edge = self._edges[proposal_id]
            if self._edge_can_reach_deferred_proposal(edge, seen_nodes):
                return True
        return False

    def _puct_score(self, edge: PUCTEdgeStats) -> float:
        parent_visits = self._node_stats[edge.proposal.environment_id].visit_count
        exploration = (
            self.c_puct * edge.prior * sqrt(parent_visits)
            / (1 + edge.visit_count)
        )
        return edge.mean_g + exploration


class ExternalSearchCoordinator:
    """Wire proposal generation, external selection and validation in order.

    Passing no policy selects the built-in :class:`UniformPUCTPolicy` baseline.
    The coordinator still owns only lifecycle ordering; PUCT state remains in
    the policy.  ``max_expansions`` is an unconditional safety boundary and
    does not require the policy to emit a ``stop`` command.
    """

    def __init__(
        self,
        runtime: EnvironmentEvolutionRuntime,
        external_policy: ExternalTreePolicy | None = None,
    ):
        self.runtime = runtime
        self.external_policy = external_policy or UniformPUCTPolicy()
        self.tree = EnvironmentTree()
        self.proposals = ProposalCatalog()

    def run(self, max_expansions: int) -> SearchRunResult:
        if max_expansions < 0:
            raise ValueError("max_expansions must be >= 0")

        root = self.runtime.create_root()
        self.tree.add(root)
        self.external_policy.observe_node(root, self.tree.snapshot())
        proposal_source_id = root.node.node_id

        while True:
            snapshot = self.tree.snapshot()
            if snapshot.expansion_count >= max_expansions:
                return SearchRunResult(
                    best_node_id=self.tree.best_validated_node_id(),
                    root_node_id=snapshot.root_node_id,
                    expansion_count=snapshot.expansion_count,
                    stop_rationale="max_expansions reached",
                    snapshot=snapshot,
                )

            proposal_source = self.tree.get(proposal_source_id)
            batch = self.runtime.generate_proposals(
                environment=proposal_source,
                validation_traces=self.tree.traces_for(proposal_source_id),
                snapshot=snapshot,
            )
            if batch.environment_id != proposal_source_id:
                raise ValueError("proposal batch does not match EnvRigger source")
            self.proposals.add_batch(batch)
            self.external_policy.observe_proposals(batch, snapshot)

            command = self.external_policy.next_command(snapshot)
            if command.action == "stop":
                self.tree.get(command.best_node_id or "")
                return SearchRunResult(
                    best_node_id=command.best_node_id or "",
                    root_node_id=snapshot.root_node_id,
                    expansion_count=snapshot.expansion_count,
                    stop_rationale=command.rationale,
                    snapshot=snapshot,
                )

            environment_id = command.environment_id or ""
            environment = self.tree.get(environment_id)
            if command.proposal is None:  # narrowed by SearchCommand validation
                raise ValueError("apply command omitted proposal")
            proposal = self.proposals.consume_selected(
                environment_id=environment_id,
                selected=command.proposal,
            )
            child = self.runtime.apply_and_validate(
                environment=environment,
                proposal=proposal,
                validation_traces=self.tree.traces_for(environment_id),
                snapshot=snapshot,
            )
            if child.node.parent_node_id != environment_id:
                raise ValueError("validated child does not reference selected environment")
            if child.node.incoming_proposal_id != proposal.proposal_id:
                raise ValueError("validated child does not reference selected proposal")
            self.tree.add(child)
            self.external_policy.observe_node(child, self.tree.snapshot())

            # Next EnvRigger turn diagnoses the newly validated child. MCTS may
            # still select a pending proposal from any older environment.
            proposal_source_id = child.node.node_id
