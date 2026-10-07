# Tests

GPU-free, docker-free, API-key-free. All tests run against the toy24
`ActionableEnv` (pure stdlib) and `ScriptedClient` (canned LLM responses).
The whole suite (115 tests) finishes in well under a second.

## Run

```bash
pip install -e ".[dev]"
pytest                                  # full suite
pytest tests/test_objectives.py -v      # one file
pytest -k difficulty_zone               # by name pattern
```

## What's covered

| File | Surface tested |
|---|---|
| `test_actionable_env_toy24.py` | Full `ActionableEnv` interface walk-through on `Toy24Env`: `reset` / `step` / `evaluate` / `observe` / `tool_schemas` / `env_state_schema` / `save_state` / `from_state` round-trip. **Template for new-Bridge tests.** |
| `test_envharness_composition.py` | Stage (`Setup`) replay and Contract (`Rules`) hook behavior over an `ActionableEnv`. |
| `test_code_loader.py` | Agent-emitted Python code lifecycle: empty body → pass-through; syntax error → `RulesCodeError`; wrong base class → typed error. |
| `test_objectives.py` | `DifficultyZone`: band validation, in-band / too-easy / too-hard branches, axis-weights output. |
| `test_budget.py` | All `BudgetPolicy` implementations: stop conditions and edge cases. |
| `test_baseline_cache.py` | Cache key determinism + load/save round-trip + corrupt-file resilience. |

## Patterns to copy when adding tests

### Adding a Bridge (`ActionableEnv`) for a new benchmark

Model your test on `test_actionable_env_toy24.py`. The `ActionableEnv` ABC
contract is:

- `reset(seed, options) → EnvResetResponse` produces an `Observation`
- `step(Action) → EnvResponse` with `reward / terminated / info`
- `evaluate() → EvaluationResult` (the authoritative success signal)
- `observe() → Observation` works BEFORE the first `step`
- `tool_schemas()` and `env_state_schema()` are non-empty
- `save_state() → dict` / `from_state(dict) → ActionableEnv` round-trip

Pin a known-solvable task via `reset(options=...)` and walk it through to
success in the test.

### Testing Stage and Contract

Model tests on `test_envharness_composition.py`: Stage must replay only normal
environment actions, while Contract may change only the A/T/O hooks. Candidate
tests must cover independent Stage/A/T/O payloads and per-change rationales.

### Adding a new Harness Agent implementation

Model on `test_envharness_composition.py` for the scripted agent. The
`HarnessAgent` ABC contract is:

- `propose(ctx) → Candidate` (independent Stage plus Contract f_A/f_T/f_O slots,
  each with its own rationale)
- `propose_changes(environment, traces, ctx) → Candidate` generates a four-slot
  envelope that is split into zero-to-four independent MCTS proposals; it does
  not select or apply a proposal
- `decide(candidate, traces, ctx) → DecideResult` (ACCEPT / REFINE / REJECT)
- `refine(candidate, traces, ctx) → Candidate`

The last two methods belong to the legacy internal-decision orchestrator. The
external tree-search protocol generates proposals first, lets the external
controller select any current or historical proposal, then applies and
validates that proposal on its source environment.

`HarnessAgentContext` exposes `tool_schemas`, `env_state_schema`,
`objective`, `task_id`, and (when computed) `baseline: BaselineSnapshot`.

### Adding a new `MutationObjective`

Model on `test_objectives.py`. The contract is:

- `evaluate(recent_traces) → ObjectiveSignal`
- The signal carries `score`, `diagnostic`, `suggestion_prompt`, optional `weights`

Test the branches: empty history, "too easy", "too hard", "in band".

### Adding a new `BudgetPolicy`

Model on `test_budget.py`:

- `should_stop(attempts, last_decision, objective_signal) → bool`
- Test each terminating condition independently.

## Why no LLMHarnessAgent / SubprocessRunner / real-bench tests?

These need either an API key, docker daemon, or a long-running runtime
(textworld, playwright). They belong in **integration** tests run via
`scripts/run_harness.py` against a real config — see the per-benchmark
READMEs under `experiments/<bench>/`. The unit tests above pin the typed
interfaces; the integration runs validate behavior on real envs.
