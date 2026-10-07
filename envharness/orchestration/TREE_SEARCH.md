# MCTS environment-search flow

This document defines the boundary between EnvRigger, EnvHarness and MCTS. It
also documents the built-in `UniformPUCTPolicy` baseline. The same event
contracts remain usable by a remote MCTS implementation later.

## Correct ordering

EnvRigger proposes changes before MCTS selects an edge:

```text
Validated environment E_current
          |
          | EnvRigger diagnoses E_current trajectories
          v
Generate zero-to-four independent proposals
  [Stage] [f_A] [f_T] [f_O]
          |
          | send the whole ProposalBatch
          v
MCTS stores the proposals as pending edges
          |
          | may select a current or historical pending edge
          v
Return (environment_id, selected proposal)
          |
          | EnvRigger/EnvHarness applies it to that environment
          v
Run K validation rollouts
          |
          v
Validated child node -> MCTS
```

The returned `environment_id` does not have to be the environment used in the
latest EnvRigger turn. MCTS may return a proposal generated earlier for another
tree node. The proposal must still be applied to the same environment for which
it was generated.

For example:

```text
EnvRigger just generated proposals for E7

MCTS may return:
  environment_id = E3
  proposal        = proposal_f_T_previously_generated_for_E3

EnvRigger/EnvHarness then:
  loads E3
  applies that f_T proposal to E3
  validates the result
  creates child E8 under E3
```

## From one model output to zero-to-four proposals

EnvRigger still emits one structured response:

```json
{
  "stage": {
    "in_env_actions": [],
    "rationale": ""
  },
  "contract": {
    "f_A": {"code": "", "rationale": ""},
    "f_T": {"code": "", "rationale": ""},
    "f_O": {"code": "", "rationale": ""}
  }
}
```

This response is split before it is sent to MCTS:

- non-empty Stage becomes one `EnvironmentChangeProposal(kind="stage")`;
- non-empty `f_A` becomes one proposal;
- non-empty `f_T` becomes one proposal;
- non-empty `f_O` becomes one proposal;
- empty slots produce no proposal.

Therefore one EnvRigger response produces zero to four alternative candidate
edges. They are not applied together and are not one combined child node.

Each proposal has:

- a globally unique `proposal_id`;
- the `environment_id` for which EnvRigger generated it;
- exactly one change kind;
- that change's payload/code and rationale.

## Responsibility split

EnvRigger/EnvHarness owns:

- observing and diagnosing one environment's validation trajectories;
- generating Stage/f_A/f_T/f_O proposals;
- splitting the response into independent proposals;
- applying the selected proposal to its specified environment;
- producing the resulting complete Candidate/checkpoint;
- running K validation rollouts;
- persisting the node and traces.

MCTS owns:

- storing pending proposals from every environment node;
- selecting a pending proposal from any current or historical batch;
- maintaining visits, accumulated return $G$, priors, children and backpropagation;
- optionally deciding when search stops;
- selecting the final best validated environment.

MCTS does not write Python hook code and does not mutate an environment. It
only selects an EnvRigger-generated proposal.

## Node and proposal are different objects

A proposal is an unvalidated possible edge. It is not yet an environment node.

An `EnvironmentNode` is created only after:

1. MCTS selects `(environment_id, proposal)`;
2. EnvRigger/EnvHarness applies the proposal to that environment;
3. the resulting environment finishes K validation rollouts;
4. its complete Candidate and checkpoint are available.

Weak or failed validated environments remain nodes. MCTS can assign them low
accumulated return and stop selecting their branches; no `REJECT` operation is
necessary.

## Candidate semantics

Each node stores a complete environment Candidate. Its `parent_node_id` points
to the selected `environment_id`, and `incoming_proposal_id` records the edge
used to create it.

The individual proposal is a single-axis change, but the resulting child
Candidate may preserve the other Stage/Contract fields already present in its
parent. The actual merge/replacement operation belongs to EnvRigger/EnvHarness,
not MCTS. This is the later `apply_and_validate` adapter's responsibility.

This design does not introduce a generic EnvHarness Chain. It only defines how
one selected change produces one complete child environment version.

## External protocol

EnvHarness sends two event types:

- `NodeSubmission`: a newly validated environment plus validation evidence;
- `ProposalBatch`: zero to four pending changes generated for one environment.

The external side returns one of:

```json
{
  "action": "apply",
  "environment_id": "env-3",
  "proposal": {
    "proposal_id": "env-3-f_T-1",
    "environment_id": "env-3",
    "kind": "f_T",
    "change": "..."
  }
}
```

or:

```json
{
  "action": "stop",
  "best_node_id": "env-8",
  "rationale": "search finished"
}
```

The coordinator verifies that the returned proposal was previously generated,
has not already been consumed, belongs to the returned environment, and was not
modified by the external process.

## Existing EnvHarness mapping

| Existing capability | New role |
|---|---|
| baseline rollout/cache | root-node validation |
| EnvRigger structured response | proposal batch source |
| Stage/f_A/f_T/f_O parser | split into zero-to-four independent edges |
| `HarnessAgent.propose_changes` | generate proposals for a validated environment |
| Candidate/checkpoint loader | apply one selected proposal and materialize the child |
| `_rollout_k` and evaluator | mandatory child validation |
| TraceStore | validation evidence storage |
| `HarnessAgent.decide` | not used in external-search path |
| ACCEPT/REFINE/REJECT budget loop | replaced by external selection/stop plus a safety cap |

The root must retain or recompute full validation trajectories when EnvRigger
needs to diagnose it. A summary-only baseline cache is sufficient for metrics,
but not for generating grounded proposals.

## Built-in uniform PUCT baseline

`envharness.orchestration.tree_search.UniformPUCTPolicy` is the current local
baseline. It replaces JEV with a uniform policy prior:

- the `ProposalBatch` already omits empty Stage / f_A / f_T / f_O slots;
- if a parent has `k` non-empty cached proposals, each has `P = 1/k`;
- proposals remain cached as deferred edges until selected, so a poor branch
  naturally sends PUCT back to untried siblings from an older batch.

For each selectable edge, it uses:

```text
G(edge) / N(edge)
  + c_puct * P(edge) * sqrt(N(parent)) / (1 + N(edge))
```

When a child is validated, the backed-up return is its success-rate difference
from the root. This equals the sum of the direct parent/child success-rate
deltas along its path, while still allowing deeper outcomes to update ancestor
edges.

The policy emits only:

```json
{
  "action": "apply",
  "environment_id": "env-3",
  "proposal": {
    "proposal_id": "env-3-f_T-1",
    "environment_id": "env-3",
    "kind": "f_T",
    "change": {}
  }
}
```

`stop` remains part of the protocol but is not selected by this baseline. If
no pending proposal is reachable, it raises `NoSelectableProposalError`; the
coordinator still enforces `max_expansions` as a safety boundary.

## Components currently provided

`tree_search.py` provides:

- schemas for nodes, proposals, batches, validation, commands and results;
- invariants enforcing at most four unique proposal kinds per batch;
- invariants enforcing exactly one changed slot per proposal;
- a proposal catalog that retains proposals from older batches;
- a minimal environment tree store;
- uniform priors, PUCT selection, visit counts, and success-rate
  backpropagation;
- transport-neutral runtime and external-MCTS interfaces;
- a coordinator enforcing generate -> submit -> select -> apply -> validate.

Still intentionally out of scope are HTTP/RPC transport, the concrete adapter
to current Orchestrator internals, and the concrete parent-Candidate plus
selected-change merge operation.
