# Architecture

## Design goal

The repository separates task meaning from environment dynamics and from curriculum assistance. A scenario may have multiple environment, verifier, transformation, policy, and curriculum implementations without requiring a central `if/elif` chain.

```text
scenario README + manifest + JSON config
                    │
                    ▼
          component registries/factory
                    │
          ┌─────────┼──────────────────┐
          ▼         ▼                  ▼
      base env    verifier   six environment changes
          └─────────┼──────────────────┘
                    ▼
                RuleHarness
                    │
          EnvDescriptor + tool schemas
                    │
          ModelClient / AgentRunner
                    │
          trajectory + evaluation
```

## Dependency rules

1. `core/` contains data contracts, protocols and the generic component registry. It must not import a concrete scenario.
2. `micro_api/` and future environment packages own state and tool dynamics. They receive a verifier rather than defining task success inline.
3. `verifiers/` are read-only functions over true state. They never read an `f_O`-transformed observation.
4. `transforms/` contains six independent extension points:
   - `SetupRule` replays legal actions after reset to construct the initial state.
   - `ContractRule` changes Agent-visible tool schemas and enforces the same constraint before execution.
   - `ActionRule` implements `f_A` before the base step and may rewrite or block an action.
   - `TransitionRule` implements `f_T` after a real base transition.
   - `ObservationRule` implements `f_O` over agent-visible results.
   - `BudgetRule` checks and accounts for episode resources.
5. `harness/` only composes components and owns ordering, dynamic contract versions, snapshots, and config construction.
6. `agent_runtime/` owns the provider-neutral model boundary and episode loop. It consumes only `ActionableEnv` and never imports concrete environments.
7. `agent_system/` loads Agent, prompt, skill and provider JSON; Target executes tasks while Diagnostic is read-only.
8. `benchmarks/` adapts external environments and publishes their legal mutation surface.
9. `evolution/` owns mutation specs, failure signatures and the environment lineage DAG; it does not execute hidden source edits.
10. `scenarios/` contains research provenance and machine-readable manifests, not duplicate runtime implementations.
11. Curriculum search selects configurations; it does not mutate verifier code.

## Step order

```text
reset
  -> Setup action replay
agent Action
  -> Contract rules
  -> current EnvDescriptor schema validation
  -> f_A action rules
  -> Budget pre-check
  -> base environment transition (unless f_A blocked it)
  -> f_T transition rules
  -> f_O observation rules
  -> Budget accounting
  -> agent-visible observation
```

`EnvResponse.observation` is agent-visible. `EnvResponse.info` is privileged harness metadata for replay, diagnostics and trigger coverage. Production policies should not receive `info`.

## Configuration contract

```json
{
  "schema_version": 1,
  "environment": {
    "type": "item",
    "parameters": {"page_size": 2}
  },
  "task": {"target_value": "target-item"},
  "verifier": {"type": "exactly_once"},
  "rules": {
    "setup": [],
    "contract": [],
    "action": [],
    "transition": [],
    "observation": [],
    "budget": []
  }
}
```

The generic factory resolves every `type` through a registry. A benchmark Bridge can register its own environment and rule builders during package initialization; it does not edit the central factory.

## Adding an environment implementation

1. Implement `ActionableEnv` from `core/protocol.py`, including an Agent-visible `describe()` method.
2. Accept a read-only verifier where practical.
3. Provide deterministic `reset`, `save_state` and `load_state` behavior.
4. Register a builder with `register_environment(name, builder)` and register interchangeable verifiers with `register_verifier(name, builder)`.
5. Add determinism, snapshot round-trip, Oracle and state-isolation tests.

## Adding a transformation

1. Choose exactly one phase: setup, contract, action, transition, observation or budget.
2. Implement the corresponding protocol from `transforms/protocols.py`.
3. Make all internal counters snapshotable.
4. Emit a named `fault_events` entry when the rule triggers.
5. Register its builder through the corresponding registration function.
6. Test both trigger and non-trigger paths.

If a component needs to change more than one phase, prefer two explicit rules sharing a serializable coordination token over one opaque wrapper.

## Adding a scenario

Each scenario folder contains:

- `README.md`: task, provenance, related work, related benchmarks, action space, verifier and curriculum;
- `scenario.json`: stable ID, status and component names;
- optional config files or links to shared configs.

A scenario becomes runnable by registering a base environment and verifier, then referencing them from configs. Scenario documentation is intentionally independent from the implementation so that SQLite, AppWorld, τ-bench or browser-backed versions can implement the same task family.

## Curriculum boundary

The target environment is the root. Search adds the smallest assistance needed to find a node the current Agent can solve. A successful trajectory may update a skill or policy; the runner then removes assistance and retests the parent. When a parent succeeds, its easier descendants are pruned.

Candidate selection should preserve:

- the same semantic goal;
- an unchanged verifier implementation or verifier hash;
- Oracle solvability;
- deterministic reset and replay;
- explicit distance from the target environment;
- no direct exposure of privileged ground truth.
