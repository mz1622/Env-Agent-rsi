# Minimal environment implementation plan

## Decision

Start with a local, deterministic, stateful API environment rather than a browser, desktop, or codebase benchmark. The first task is “append one value exactly once” under different failure timings. This isolates environment evolution from Agent self-modification and gives us a reliable verifier, cheap snapshots, and short validation cycles.

## Scope boundary

- The frozen Agent may be observed, but the search process cannot edit prompts, skills, memory, tools, or Agent code.
- A candidate may only change an `EnvironmentMutationSpec`.
- The verifier is read-only and stored outside the candidate’s writable area.
- Phase 1 implements Setup/Stage and Rules/Contract. Link/Chain is postponed.

## Minimal API

```text
reset(seed) -> ResetResponse
step(Action) -> EnvResponse
observe() -> Observation
evaluate() -> EvaluationResult
save_state() -> JSON
from_state(JSON) -> Environment
```

`Action` contains a tool name and JSON arguments. `EnvResponse` contains an observation, reward, terminated/truncated flags, and structured event metadata.

## First task

State: `items(id, value, idempotency_key)` plus an append-only audit log.

Tools:

- `list_items(cursor)`
- `get_item(id)`
- `append_item(value, idempotency_key)`
- `finish()`

Success means the target value exists exactly once, prior rows are unchanged, and no forbidden side effect occurred.

## First mutation set

1. Baseline success.
2. Pre-commit timeout: no state change.
3. Post-commit timeout: state changed, error returned.
4. Lost response followed by confirmable read.
5. One stale read after a successful write.
6. Idempotency key supported.
7. Idempotency key ignored; read-before-retry is required.
8. Target row beyond the first page.
9. One-write budget.
10. Truncated error message with stable error code.

## Candidate validation gates

Every candidate must pass all gates before it enters the archive:

1. Schema and static safety checks.
2. Reset determinism for a fixed seed.
3. Snapshot/restore round trip.
4. Verifier hash unchanged.
5. Oracle success on every validation seed.
6. Named fault event triggered at the intended transition.
7. No state leakage across episodes.
8. Frozen-Agent success rate is neither trivial nor zero unless the candidate is kept as a boundary case.

## Search representation

Use a lineage DAG, not a binary tree. A node is a normalized environment specification plus validation results. Edges are typed mutations. Identical normalized specs share one node. A node can have multiple parents when compatible mutations are combined.

Keep a Pareto archive over:

- distance to target Agent success rate;
- Oracle solvability;
- fault trigger coverage;
- behavioral novelty;
- runtime cost;
- reset variance;
- collateral state changes.

## Proposed repository layout

```text
src/env_agent_rsi/
  core/
  micro_api/
  harness/
  search/
tests/
configs/micro_api/
artifacts/runs/
research/
```

## Milestones

### M0: Contract and threat model (1–2 days)

- Freeze the environment interface and JSON schemas.
- Define verifier isolation and mutation allowlist.
- Add run manifest and lineage formats.

Exit: schemas validate representative actions, responses, states, and mutations.

### M1: Deterministic base environment (2–3 days)

- Implement the state store, tools, task factory, verifier, and Oracle.
- Add random and scripted baseline agents.
- Add determinism, snapshot, and verifier tests.

Exit: 100 repeated same-seed runs produce the same state hash and verifier result.

### M2: Environment wrappers (3–5 days)

- Implement action-replay Setup.
- Implement before-action, after-transition, and observation Rules hooks.
- Materialize the ten hand-authored variants.
- Save trajectories and named fault events.

Exit: all variants pass Oracle and trigger-coverage gates.

### M3: Bounded environment search (3–5 days)

- Start with a rule-based proposer; optionally add an LLM proposer constrained to the schema.
- Generate at most 4–8 candidates per archive expansion.
- Allow at most two mutations per initial candidate.
- Validate, deduplicate, score, and archive candidates.

Exit: one command creates a reproducible lineage with accepted and rejected candidates and reasons.

### M4: External benchmark bridge (1–2 weeks)

Preferred order:

1. AppWorld subset for stateful multi-app APIs and strong state-based verification.
2. τ-bench subset for database state plus policy-constrained tool use.
3. ALFWorld for cheap cross-domain Stage/Contract testing.
4. WebArena or SWE-bench only after runtime isolation is stable.

Exit: the same mutation and validation pipeline runs without benchmark-specific changes outside the Bridge.

## First experiment

Freeze one Agent configuration. Run 20 seeds per environment variant. Report task success, duplicate-write rate, read-before-retry rate, unconditional retry rate, average tool steps, fault trigger coverage, and collateral changes. Compare targeted variants against random action/observation perturbations with the same run budget.

The experiment succeeds if the targeted environment variants reliably expose distinct recovery failures while Oracle solvability and reset reproducibility remain intact.
