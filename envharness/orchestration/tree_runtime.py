# Copyright 2026 The EnvHarness Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Concrete EnvHarness adapter for the external environment-tree protocol."""
from __future__ import annotations

import uuid
from pathlib import Path

from envharness.agents.harness_agent import (
    HarnessAgentContext,
    _rules_code_from_contract,
)
from envharness.core.types import (
    BaselineRolloutSummary,
    BaselineSnapshot,
    Candidate,
    ObjectiveSignal,
    Trace,
)
from envharness.orchestration.orchestrator import (
    Orchestrator,
    _checkpoint_from_candidate,
)
from envharness.orchestration.tree_search import (
    EnvironmentChangeProposal,
    EnvironmentNode,
    NodeSubmission,
    ProposalBatch,
    TreeSnapshot,
    ValidationSummary,
    split_candidate_envelope,
)


class OrchestratorEvolutionRuntime:
    """Reuse an Orchestrator's runner, specs, agent, storage and logging.

    One instance runs one benchmark task id. Parent nodes store complete
    Candidates. Applying a proposal replaces only its selected slot and keeps
    the parent's other slots, then rebuilds native Rules code and validates the
    complete child with the existing K-rollout runner.
    """

    def __init__(self, orchestrator: Orchestrator, task_idx: int,
                 output_dir: str | Path,
                 root_traces: list[Trace] | None = None):
        self.orchestrator = orchestrator
        self.task_idx = int(task_idx)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.nodes_dir = self.output_dir / "nodes"
        self.nodes_dir.mkdir(parents=True, exist_ok=True)
        self.iteration_id = f"mcts-{uuid.uuid4().hex[:8]}"
        self.task_id = self._resolve_task_id()
        self.attempt = 0
        self.turn = 0
        self.all_traces: list[Trace] = []
        self.baseline: BaselineSnapshot | None = None
        self._preloaded_root_traces = list(root_traces or [])

    def _resolve_task_id(self) -> int:
        cfg = self.orchestrator.config
        if cfg.explicit_task_ids:
            return int(cfg.explicit_task_ids[self.task_idx])
        return int(
            cfg.base_seed
            + self.task_idx * cfg.task_id_stride
            + cfg.task_id_base_offset
        )

    def create_root(self) -> NodeSubmission:
        candidate = Candidate()
        if self._preloaded_root_traces:
            traces = list(self._preloaded_root_traces)
            expected = self.orchestrator.config.k_per_candidate
            if len(traces) != expected:
                raise ValueError(
                    "preloaded root trace count must equal k_per_candidate; "
                    f"got {len(traces)}, expected {expected}"
                )
            self.all_traces.extend(traces)
        else:
            traces = self._validate(candidate, node_id="root")
        for trace in traces:
            trace.kind = "baseline"
            trace.candidate_id = "root"
        self.baseline = self._baseline_from_traces(traces)
        return self._build_node_submission(
            node_id="root",
            parent_node_id=None,
            incoming_proposal_id=None,
            depth=0,
            candidate=candidate,
            traces=traces,
        )

    def generate_proposals(
        self,
        environment: EnvironmentNode,
        validation_traces: list[Trace],
        snapshot: TreeSnapshot,
    ) -> ProposalBatch:
        self.turn += 1
        ctx = HarnessAgentContext(
            history_traces=list(self.all_traces),
            tool_schemas=self.orchestrator.tool_schemas,
            env_state_schema=self.orchestrator.env_state_schema,
            objective=self.orchestrator.objective,
            iteration_id=self.iteration_id,
            task_description=self.orchestrator.config.task_description,
            task_id=self.task_id,
            baseline=self.baseline,
            baseline_traces=(
                self.all_traces[:self.orchestrator.config.k_per_candidate]
            ),
        )
        envelope = self.orchestrator.harness_agent.propose_changes(
            environment.candidate,
            validation_traces,
            ctx,
        )
        batch = split_candidate_envelope(
            environment_id=environment.node_id,
            envelope=envelope,
            proposal_id_prefix=f"{environment.node_id}-turn-{self.turn}",
        )
        self.orchestrator.log.event(
            "mcts_proposals_generated",
            iteration_id=self.iteration_id,
            environment_id=environment.node_id,
            n_proposals=len(batch.proposals),
            kinds=[proposal.kind for proposal in batch.proposals],
        )
        return batch

    def apply_and_validate(
        self,
        environment: EnvironmentNode,
        proposal: EnvironmentChangeProposal,
        validation_traces: list[Trace],
        snapshot: TreeSnapshot,
    ) -> NodeSubmission:
        child_candidate = merge_candidate_change(
            environment.candidate,
            proposal,
        )
        child_id = f"env-{snapshot.expansion_count + 1:02d}"
        traces = self._validate(child_candidate, node_id=child_id)
        submission = self._build_node_submission(
            node_id=child_id,
            parent_node_id=environment.node_id,
            incoming_proposal_id=proposal.proposal_id,
            depth=environment.depth + 1,
            candidate=child_candidate,
            traces=traces,
        )
        self.orchestrator.log.event(
            "mcts_node_validated",
            iteration_id=self.iteration_id,
            node_id=child_id,
            parent_node_id=environment.node_id,
            proposal_id=proposal.proposal_id,
            proposal_kind=proposal.kind,
            success_rate=submission.node.validation.success_rate,
        )
        return submission

    def _validate(self, candidate: Candidate, node_id: str) -> list[Trace]:
        self.attempt += 1
        traces = self.orchestrator._rollout_k(
            candidate=candidate,
            candidate_id=node_id,
            iteration_id=self.iteration_id,
            task_idx=self.task_idx,
            attempt=self.attempt,
            ctx_task_id=self.task_id,
        )
        self.all_traces.extend(traces)
        for trace in traces:
            self.orchestrator.trace_store.add(trace)
        return traces

    def _build_node_submission(
        self,
        *,
        node_id: str,
        parent_node_id: str | None,
        incoming_proposal_id: str | None,
        depth: int,
        candidate: Candidate,
        traces: list[Trace],
    ) -> NodeSubmission:
        summary = validation_summary(traces)
        checkpoint = _checkpoint_from_candidate(
            env_spec=self.orchestrator.env_spec,
            candidate=candidate,
            task_id=self.task_id,
            metadata={
                "search": "external_mcts",
                "node_id": node_id,
                "parent_node_id": parent_node_id,
                "incoming_proposal_id": incoming_proposal_id,
                "task_idx": self.task_idx,
                "task_id": self.task_id,
                "validation": summary.model_dump(),
            },
        )
        checkpoint.save(self.nodes_dir / f"{node_id}.json")
        node = EnvironmentNode(
            node_id=node_id,
            parent_node_id=parent_node_id,
            incoming_proposal_id=incoming_proposal_id,
            depth=depth,
            task_id=self.task_id,
            candidate=candidate,
            checkpoint=checkpoint.to_dict(),
            validation=summary,
        )
        return NodeSubmission(node=node, validation_traces=traces)

    @staticmethod
    def _baseline_from_traces(traces: list[Trace]) -> BaselineSnapshot:
        successes = [trace for trace in traces if trace.success]
        success_steps = [trace.duration_steps for trace in successes]
        sample = min(successes, key=lambda trace: trace.duration_steps,
                     default=None)
        return BaselineSnapshot(
            n=len(traces),
            n_success=len(successes),
            sr=len(successes) / max(1, len(traces)),
            avg_success_steps=(
                sum(success_steps) / len(success_steps)
                if success_steps else None
            ),
            per_rollout=[BaselineRolloutSummary(
                success=trace.success,
                steps=trace.duration_steps,
            ) for trace in traces],
            sample_success_actions=(
                [step.raw_action for step in sample.steps] if sample else []
            ),
        )


def validation_summary(traces: list[Trace]) -> ValidationSummary:
    """Build the external MCTS value payload from K validation traces."""
    n = len(traces)
    n_success = sum(1 for trace in traces if trace.success)
    success_rate = n_success / max(1, n)
    avg_reward = sum(trace.final_reward for trace in traces) / max(1, n)
    avg_steps = sum(trace.duration_steps for trace in traces) / max(1, n)
    n_errors = sum(1 for trace in traces if trace.error)
    return ValidationSummary(
        n_rollouts=n,
        n_success=n_success,
        success_rate=success_rate,
        avg_reward=avg_reward,
        avg_steps=avg_steps,
        n_errors=n_errors,
        objective_signal=ObjectiveSignal(
            score=success_rate,
            diagnostic=(
                f"Validated {n} rollouts: success_rate={success_rate:.3f}, "
                f"errors={n_errors}."
            ),
            suggestion_prompt="External MCTS maximizes validation success rate.",
        ),
        trace_ids=[trace.episode_id for trace in traces],
    )


def merge_candidate_change(
    parent: Candidate,
    proposal: EnvironmentChangeProposal,
) -> Candidate:
    """Apply one selected proposal while preserving the parent's other slots."""
    values = parent.model_dump()
    change = proposal.change
    if proposal.kind == "stage":
        values["in_env_actions"] = list(change.in_env_actions)
        values["stage_rationale"] = change.stage_rationale
    else:
        values[proposal.kind] = getattr(change, proposal.kind)
        values[f"{proposal.kind}_rationale"] = getattr(
            change, f"{proposal.kind}_rationale",
        )
    values["rules_code"] = _rules_code_from_contract({
        "f_A": values["f_A"],
        "f_T": values["f_T"],
        "f_O": values["f_O"],
    })
    # This compatibility-only aggregate rationale must not imply that several
    # independent changes share one explanation.
    values["rationale"] = ""
    return Candidate(**values)
