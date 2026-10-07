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

"""HarnessAgent ABC + LLM-driven implementation.

HarnessAgent independently fills Stage plus Contract f_A/f_T/f_O slots. The
Orchestrator materializes all non-empty slots for each episode.

The legacy orchestrator uses decide/refine internally. The external tree-search
path does not call decide: EnvRigger first uses propose_changes to generate up
to four independent edges, then external MCTS selects one for application.
"""
from __future__ import annotations

import ast
import json
import textwrap
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from envharness.infra.llm import LLMClient, Message
from envharness.orchestration.objectives import MutationObjective
from envharness.core.types import (
    Action, BaselineSnapshot, Candidate, Decision, DecideResult,
    FailureAnalysis, TaskSummary, Trace,
)


@dataclass
class HarnessAgentContext:
    history_traces: list[Trace] = field(default_factory=list)
    tool_schemas: list[dict] = field(default_factory=list)
    env_state_schema: str = ""
    objective: MutationObjective | None = None
    iteration_id: str = ""
    task_description: str = ""
    task_id: int = 0
    """Deterministic identifier of which underlying task instance this
    iteration's rollouts target. We pass this as the `seed` arg to
    `Bridge.reset` because Gymnasium's ABC requires it, but it is NOT a
    randomness source -- in alfworld/toy24/etc. it simply indexes into the
    benchmark's prebuilt task library.

    ExploringHarnessAgent resets its scratch env with this task_id (on
    refine, the previous trajectory is replayed on top via a `Setup`
    wrapper) so the recorded trajectory is valid on the same task the K
    rollouts will replay it onto. When the resulting Candidate carries
    in_env_actions, the orchestrator locks all K rollouts to this task_id
    (fully deterministic replay; variance only from sampling temperature)."""
    baseline: BaselineSnapshot | None = None
    """Per-task baseline: K unmutated rollouts of the Policy on the same
    task_id, computed BEFORE the HarnessAgent's first propose(). Lets the
    HarnessAgent calibrate perturbation magnitude (high-baseline-SR tasks need
    hard perturbations; near-zero-SR tasks should be made EASIER or
    skipped, since 'add difficulty' is nonsensical when the Policy
    already fails). Cached in runs/_baseline_cache."""
    baseline_traces: list[Trace] = field(default_factory=list)
    """Full per-step trajectories of the K baseline rollouts (think + action +
    obs, truncated per-step). Empty when baseline came from cache (cache only
    stores summary stats). Lets the HarnessAgent diagnose WHY the Policy failed or
    succeeded by reading the actual reasoning + commands + observations."""
    candidate_tasks: list[TaskSummary] = field(default_factory=list)
    """When `OrchestratorConfig.mutator_selects_tasks=True`: the
    orchestrator presents a shortlist of candidate task_idx values; the
    HarnessAgent picks one via `select_task(ctx)`. Empty list when the flag
    is off (default) -- in that case the orchestrator's deterministic
    stride formula chooses task_id and `select_task` is never called."""


class HarnessAgent(ABC):
    @abstractmethod
    def propose(self, ctx: HarnessAgentContext) -> Candidate: ...

    @abstractmethod
    def decide(self, candidate: Candidate, traces: list[Trace],
               ctx: HarnessAgentContext) -> DecideResult: ...

    @abstractmethod
    def refine(self, candidate: Candidate, traces: list[Trace],
               ctx: HarnessAgentContext) -> Candidate: ...

    def propose_changes(self, environment: Candidate,
                        validation_traces: list[Trace],
                        ctx: HarnessAgentContext) -> Candidate:
        """Generate the four-slot proposal envelope for one environment.

        The external-search adapter splits each non-empty Stage/A/T/O slot into
        an independent proposal before MCTS sees it. This method does not apply
        a change, select an edge, or make an accept/reject decision. The default
        delegates to the existing refine prompt so current EnvRiggers remain
        reusable while the concrete runtime adapter is developed.
        """
        return self.refine(environment, validation_traces, ctx)

    # ---- optional HarnessAgent-driven task selection ---------------------
    def select_task(self, ctx: HarnessAgentContext) -> int:
        """Given the shortlist `ctx.candidate_tasks`, return the
        `task_idx` to run this iteration. Default: first candidate
        (preserves the orchestrator's deterministic order). LLMHarnessAgent
        overrides to actually consult the model.

        Only called when `OrchestratorConfig.mutator_selects_tasks=True`.
        """
        if not ctx.candidate_tasks:
            raise ValueError("select_task called with empty candidate_tasks; "
                             "orchestrator should pre-populate.")
        return ctx.candidate_tasks[0].task_idx


# ---------------------------------------------------------------------------
# Function-call schemas -- HarnessAgent output is constrained via these
# ---------------------------------------------------------------------------

PROPOSE_TOOL = {
    "type": "function",
    "function": {
        "name": "propose_candidate",
        "description": (
            "Evaluate and return all four independent environment-change "
            "slots: Stage plus Contract f_A, f_T, and f_O. Fill every "
            "applicable slot and leave an inapplicable slot empty."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "stage": {
                    "type": "object",
                    "description": (
                        "Initial-state change. Use an empty action list when "
                        "Stage is not applicable."
                    ),
                    "properties": {
                        "in_env_actions": {
                            "type": "array",
                            "description": (
                                "Actions replayed before Policy starts. Each "
                                "name must be a registered tool name and "
                                "kwargs_json must encode that tool's args."
                            ),
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "kwargs_json": {
                                        "type": "string",
                                        "description": (
                                            "JSON-encoded argument object; "
                                            "use '{}' when there are no args."
                                        ),
                                    },
                                },
                                "required": ["name", "kwargs_json"],
                            },
                        },
                        "rationale": {
                            "type": "string",
                            "description": (
                                "Why this Stage change is appropriate. Use '' "
                                "when in_env_actions is empty."
                            ),
                        },
                    },
                    "required": ["in_env_actions", "rationale"],
                },
                "contract": {
                    "type": "object",
                    "description": (
                        "Three independent A/T/O changes. Each has its own "
                        "code and rationale. A non-empty code must be one complete "
                        "Python method definition without a class wrapper. Leave "
                        "both fields empty for an unused change."
                    ),
                    "properties": {
                        "f_A": {
                            "type": "object",
                            "description": (
                                "Independent action-boundary change."
                            ),
                            "properties": {
                                "code": {"type": "string", "description": (
                                    "Complete `def filter_action(self, action, "
                                    "env_state): ...` method, or ''.")},
                                "rationale": {"type": "string", "description": (
                                    "Why this f_A change is appropriate. Use '' "
                                    "when code is empty.")},
                            },
                            "required": ["code", "rationale"],
                        },
                        "f_T": {
                            "type": "object",
                            "description": "Independent transition change.",
                            "properties": {
                                "code": {"type": "string", "description": (
                                    "Complete `def modify_transition(self, action, "
                                    "raw_response, env_state): ...` method, or ''.")},
                                "rationale": {"type": "string", "description": (
                                    "Why this f_T change is appropriate. Use '' "
                                    "when code is empty.")},
                            },
                            "required": ["code", "rationale"],
                        },
                        "f_O": {
                            "type": "object",
                            "description": "Independent observation change.",
                            "properties": {
                                "code": {"type": "string", "description": (
                                    "Complete `def filter_observation(self, obs, "
                                    "env_state): ...` method, or ''.")},
                                "rationale": {"type": "string", "description": (
                                    "Why this f_O change is appropriate. Use '' "
                                    "when code is empty.")},
                            },
                            "required": ["code", "rationale"],
                        },
                    },
                    "required": ["f_A", "f_T", "f_O"],
                },
            },
            "required": ["stage", "contract"],
        },
    },
}


DECIDE_TOOL = {
    "type": "function",
    "function": {
        "name": "decide_on_traces",
        "description": (
            "After K rollouts of one candidate, decide whether to ACCEPT "
            "(commit this candidate to the trace store as the iteration's "
            "result), REFINE (you've learned something, try a tweaked "
            "candidate), or REJECT (start fresh with a different design). "
            "ALWAYS fill failure_analysis when most rollouts failed. The "
            "rationale should reference the rollout statistics."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "decision": {"type": "string",
                              "enum": ["accept", "refine", "reject"]},
                "rationale": {"type": "string"},
                "failure_analysis": {
                    "type": "object",
                    "properties": {
                        "primary_axis": {"type": "string",
                            "enum": ["S0", "A", "O", "T",
                                      "task_understanding", "none"]},
                        "label": {"type": "string"},
                        "description": {"type": "string"},
                    },
                },
            },
            "required": ["decision", "rationale"],
        },
    },
}


SELECT_TASK_TOOL = {
    "type": "function",
    "function": {
        "name": "select_task",
        "description": (
            "Pick ONE task_idx from the shortlist of candidates "
            "shown in the prompt. The picked task_idx MUST be one of the "
            "presented values. Provide a short rationale for why this "
            "task is the right one to target next."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "task_idx": {"type": "integer",
                              "description": "the picked task_idx (must be in shortlist)"},
                "rationale": {"type": "string",
                              "description": "1-2 sentences on why."},
            },
            "required": ["task_idx", "rationale"],
        },
    },
}


# ---------------------------------------------------------------------------
# LLM-backed default
# ---------------------------------------------------------------------------

_DEFAULT_SYSTEM = """You are the Environment HarnessAgent for an agent benchmark.

Your job is to reshape the environment so the Policy agent gets the right
training/evaluation signal. Every Candidate returns four independent slots.
Inspect each slot separately, populate every applicable change, and leave an
inapplicable slot empty. Every slot has its own rationale; explain a populated
change in that slot's rationale and use an empty rationale when the slot is
empty:

1. `stage.in_env_actions` changes the starting state (S0) by replaying normal
   environment actions before the Policy starts. Explain it in
   `stage.rationale`. Use `[]` and `""` when not applicable.

2. `contract.f_A.code` changes the action boundary. Return one complete method
   `def filter_action(self, action, env_state): ...`, or `""`; explain it in
   `contract.f_A.rationale`.

3. `contract.f_T.code` changes the transition response. Return one complete
   method `def modify_transition(self, action, raw_response, env_state): ...`,
   or `""`; explain it in `contract.f_T.rationale`.

4. `contract.f_O.code` changes the observation shown to the Policy. Return one
   complete method `def filter_observation(self, obs, env_state): ...`, or
   `""`; explain it in `contract.f_O.rationale`.

The four slots are independent. A Candidate may contain zero to four changes.
The framework combines non-empty Contract methods into `_Rules(Rules)`, applies
Stage first, then applies Contract during Policy interaction.

The `in_env_actions` mechanism (S0):

  `in_env_actions` is a list[Action]. The Setup harness's reset is
  literally::

      # Setup.reset
      inner.reset(seed, options)
      for a in self.actions:
          inner.step(a)        # exactly the same dispatch as a
                                # policy action

  Each entry MUST be a valid Tool call (e.g. for ALFWorld every item
  is `{"name": "do", "kwargs_json": "{\"text\": \"<cmd>\"}"}`). After
  replay, the bridge's state is whatever those actions would have
  produced -- no different from the policy having taken them. That's
  the entire S0 mechanism.

================================================================
PYTHON SOURCE -- DO NOT OVER-ESCAPE. WRITE THE CODE AS YOU WOULD TYPE IT.
================================================================

The `f_A.code`, `f_T.code`, and `f_O.code` fields are JSON-encoded by the tool-call wrapper.
You do NOT need to escape quotes or backslashes for JSON transport. Write
each method exactly as you would type it inside a Python class, but without the
class wrapper and without leading class indentation. The framework combines and
compiles the already-decoded method strings.

WRONG -- the model emits these literal characters, Python rejects them:
    <BS><Q><Q><Q>This docstring is broken.<BS><Q><Q><Q>
    cmd = <BS><Q>git checkout<BS><Q>
    sep = <BS><Q><BS><BS>n<BS><Q>
(where <BS> is a literal backslash and <Q> is a literal double quote;
the LLM is over-escaping as if for JSON transport, then Python sees the
backslash-before-quote and raises "SyntaxError: unexpected character
after line continuation character".)

CORRECT -- emit the source exactly as you would type it in an editor:
  - Triple-quoted docstrings: open and close with three plain double
    quotes. NO backslashes before them.
  - Regular string literals: a regular double quote. NO backslash before it.
  - For an actual newline-character string, write the two-character escape
    sequence (backslash + n) ONCE inside a regular string. The framework
    does NOT need it doubled.
  - If you have a string that itself contains backslashes (e.g. Windows
    paths, regex patterns), use a raw-string prefix (lowercase r) before
    the opening quote so you can write backslashes literally.

The rule: write the code as if you were typing it into a file. Do not
manually escape for JSON. The wrapper handles JSON transport.

================================================================
TYPES IN SCOPE -- ALL Pydantic models. USE KEYWORD ARGS, NEVER POSITIONAL.
================================================================

These five names are pre-bound in your code's namespace; you do NOT import
them. Beyond them, import only from the Python standard library -- a
third-party package is not guaranteed to be installed, and an import inside a
hook body raises at the first step that reaches it, killing every rollout of
that candidate. Do not import a name you are not going to use.

    Action
        Fields:  name: str,  kwargs: dict[str, Any]
        Usage:   action.name, action.kwargs.get("op")

    Blocked
        Fields:  reason: str
        Usage:   return Blocked(reason="why this action was rejected")
        WRONG:   Blocked("why")           ← Pydantic forbids positional args

    Observation
        Fields:  text: str,  data: dict[str, Any]
        Usage:   obs.text, obs.data["numbers"], obs.data.get("history", [])
        WRONG:   obs.history              ← Observation has no .history attr
        WRONG:   obs.history = []         ← cannot assign arbitrary attrs
        To CHANGE observation:
                 return Observation(text=obs.text,
                                    data={**obs.data, "history": []})

    EnvResponse
        Fields:  observation: Observation,
                 reward: float,
                 terminated: bool,
                 truncated: bool,
                 info: dict[str, Any]
        WRONG:   raw_response.error       ← EnvResponse has NO .error field
        Bridge-side errors live in .info: raw_response.info.get("error")
        Tool result for the step is in:   raw_response.info.get("result")
        To CHANGE response:
                 return EnvResponse(observation=..., reward=...,
                                    terminated=..., truncated=..., info=...)
        Mutating raw_response.info in place is OK (it's a dict).
        Do not mutate env_state directly; express changes through the
        selected function's return value.

================================================================
WORKED EXAMPLES (idiomatic patterns)
================================================================

NOTE: the tool names ("combine", "do", ...) and env_state field names
("current_numbers", "obs_text", ...) in these examples are illustrative
ONLY. For YOUR Bridge, look up the actual tool names in `tool_schemas`
and the actual field names in `env_state schema` — both in the user
message — and substitute them in. The PATTERN (how to inspect action,
how to construct a return value, where info.result lives) is what
transfers across benchmarks; the names do not.

# Pattern 1 -- contract.f_A.code: block one verb the Policy might overuse
# (Replace the names with whatever your Bridge's tool registry exposes.)
def filter_action(self, action, env_state):
    # action.name comes from tool_schemas; action.kwargs is the dict the
    # Policy passed in. Inspect both, then return either an Action or Blocked.
    if action.name == "combine" and action.kwargs.get("op") == "div":
        return Blocked(reason="division disabled to stress A axis")
    return action

# Pattern 2 -- contract.f_T.code: perturb the Bridge's per-step result before it
# reaches Policy. The convention across Bridges is that the tool's payload
# lives at raw_response.info["result"] (a dict). Read it, build a NEW
# EnvResponse with the mutated payload.
def modify_transition(self, action, raw_response, env_state):
    result = raw_response.info.get("result")
    if not isinstance(result, dict) or "result" not in result:
        return raw_response
    new_payload = {**result, "result": result["result"] + 1}
    new_info = {**raw_response.info, "result": new_payload}
    return EnvResponse(
        observation=raw_response.observation,
        reward=raw_response.reward,
        terminated=raw_response.terminated,
        truncated=raw_response.truncated,
        info=new_info,
    )

# Pattern 3 -- contract.f_O.code: rewrite what the Policy sees. Always return
# a NEW Observation (Pydantic models are not mutated in place); copy
# obs.text + obs.data, edit keys, then construct.
def filter_observation(self, obs, env_state):
    # Example: drop a key from obs.data ("history" is toy24-specific).
    new_data = {k: v for k, v in obs.data.items() if k != "history"}
    return Observation(text=obs.text, data=new_data)

# Pattern 4 -- Stage via in_env_actions: walk the env into a non-default
# starting state BEFORE the Policy gets the first observation. You don't
# write code for this -- you put Actions in `in_env_actions`. Each item
# is replayed through env.step() in order, so its `name` must be a real
# tool and `kwargs_json` must be the JSON-encoded args for that tool.
# (Below is stage.in_env_actions, not Python code.)
# stage.in_env_actions = [
#   {"name": "combine", "kwargs_json": "{\"i\": 0, \"j\": 1, \"op\": \"mul\"}"},
# ]
# After replay: current_numbers has been mutated by the operation; the
# Policy starts from there with two fewer numbers on the board.

================================================================

`env_state` is the bench-exposed data view your hooks receive. It
contains NO runtime handles -- only data. Use the env_state schema in
the user prompt to know its fields. Hooks may READ all env_state fields;
treat them as effectively read-only (mutations have no path back into
the bench's authoritative state).

================================================================
PITFALL -- DO NOT MAKE THE TASK MATHEMATICALLY UNSOLVABLE
================================================================

A mutation that makes Policy success IMPOSSIBLE is not a difficulty
increase -- it's a broken mutation. Difficulty=0 because the task is
literally unreachable is exactly as bad as difficulty=0 because the task
is trivial: the objective wants SR in band, not SR=0 from impossibility.

Examples to AVOID on toy24:
  - Banning all 4 arithmetic operators (no path can reach target)
  - Banning stop() (Policy can never declare success)
  - filter_action blocking every viable Policy move
  - in_env_actions trajectory that leaves only impossible-from-here state

Signals in the rollout history that a PRIOR mutation was unsolvable:
  - Most rollouts end with `err=subprocess timeout` -> Policy was forced
    to grind to the runner timeout because no legal sequence wins.
  - success_rate = 0.0 across K rollouts AND failure_axis points at A.
  - The same axis (A in particular) being labeled "unsolvable_mutation"
    across multiple recent traces.

When you see these signals, your NEXT propose() must REVERSE the offending
restriction (or loosen it). Stacking more A-axis bans on top of a broken
mutation does not climb the difficulty band -- it floors the SR at 0.

Prefer subtle, narrow perturbations (one op or one observation key) over
sweeping bans. Difficulty in [0.3, 0.7] usually lives where the
task is solvable BUT requires the Policy to notice something.

================================================================
INFO DICT SCHEMA -- DO NOT BREAK BRIDGE-SET KEYS
================================================================

In modify_transition, if you build a new EnvResponse the `info` dict
must be spread-extended from the original (`{**raw_response.info, ...}`),
never replaced. Do not add a "result" key. Do not remove or change the
type of bridge-set keys: "won" (bool), "extra.gamefile" (str),
"admissible_commands" (list[str]), "max_score" (float),
"goal_condition_success_rate" (float). If you want a narration string,
put it in observation.text, not in info.

================================================================
BASELINE BLOCK (when present)
================================================================

Before your first propose(), the orchestrator runs K unmutated rollouts of
the Policy on THIS task_id and shows you the result in a BASELINE block:
its success_rate, avg_success_steps, per-rollout outcomes, and the
shortest winning trajectory's action sequence (if any).

Read the baseline first. It tells you:
  - WHETHER the Policy can solve this task at all. If baseline_sr is 0 or
    near 0, "make it harder" is nonsensical -- the Policy is already
    failing. Adding more restrictions will keep SR at 0 and waste the
    attempt budget. In that case the right move is either to make the
    task EASIER (the objective may still grade you on landing in band),
    or to write a trivial/no-op mutation and burn this task.
  - HOW the Policy solves it when it does. The sample trajectory shows
    you the exact action sequence used. A 4-step solution leaves much
    less room to "add difficulty" than a 30-step one -- a tiny
    perturbation can flip a fast-path Policy off course. Match
    perturbation magnitude to the headroom the trajectory implies.
  - WHICH parts of the env the Policy actually relies on. If the sample
    trajectory uses only `go to drawer 1`, `open drawer 1`, `take X`,
    then perturbing `examine`/`look` is irrelevant -- those commands
    were never used.

Treat the baseline as raw data, not as a hint. Decide direction and
magnitude yourself from what you see.

================================================================
REFINE STRATEGY: PRESERVE WORKING CODE
================================================================

When you refine after K rollouts, you receive the previous Candidate and
the per-rollout stats. Re-evaluate all four slots independently: preserve a
working slot, adjust a miscalibrated slot, clear an unhelpful slot, and fill a
previously empty slot only when the evidence supports it. Before deciding what
to write, ask: did this Candidate move SR toward the target band compared to
the baseline?

  * If yes (the mutation has useful structure but the MAGNITUDE is wrong
    -- either it overshot or undershot the band), COPY the previous
    working payload in your new candidate and adjust it: loosen if you
    overshot, tighten if you undershot. Do not discard a change that paid
    K rollouts of signal to establish the right perturbation type.

  * If no (the mutation had no measurable effect, or moved SR the wrong
    way), start over with a different perturbation type.

Example of the trap to avoid: baseline_sr=0.0, previous mutation =
"filter admissible_commands to 15%", previous_sr=0.8. The mutation moved
SR from 0.0 toward 0.5 (it crossed the band, overshooting). DO NOT
throw away that filter; the task is solvable ONLY because of it. KEEP
the filter and ADD a small extra restriction to pull SR from 0.8 down
toward 0.5.

================================================================

Your operating mechanism is FIXED. The OBJECTIVE that tells you what to
optimize is provided each turn -- read it carefully and let it drive your
decisions about which axes to mutate.

For every populated change, provide a short, clear rationale in that change's
own slot. Do not use one global rationale for several changes. When deciding, reference the
ROLLOUT STATISTICS (success_rate over K runs, distribution of failures,
COUNT of subprocess timeouts) rather than a single trace.
"""


class LLMHarnessAgent(HarnessAgent):
    def __init__(self, client: LLMClient, system_prompt: str = "",
                 extra_instructions: str = ""):
        self.client = client
        # system_prompt fully overrides the default if non-empty; otherwise
        # use _DEFAULT_SYSTEM. Then optionally append benchmark-specific
        # constraints.
        base = system_prompt or _DEFAULT_SYSTEM
        if extra_instructions:
            base = base + "\n\n--- experiment-specific instructions ---\n" + extra_instructions
        self.system_prompt = base

    def propose(self, ctx):
        msgs = self._base_messages(ctx, intro="Propose a Candidate now.")
        resp = self.client.chat(
            messages=msgs, tools=[PROPOSE_TOOL],
            tool_choice={"type": "function",
                          "function": {"name": "propose_candidate"}},
        )
        if not resp.tool_calls:
            return Candidate(rationale="LLM returned no tool call; pass-through.")
        return _candidate_from_args(resp.tool_calls[0].arguments)

    def decide(self, candidate, traces, ctx):
        msgs = self._base_messages(
            ctx,
            intro=("A candidate's environment change was applied and the Policy "
                   "ran K rollouts. Inspect the rollouts and decide."),
        )
        msgs.append(Message(role="user", content=self._format_candidate_block(candidate, traces)))
        resp = self.client.chat(
            messages=msgs, tools=[DECIDE_TOOL],
            tool_choice={"type": "function",
                          "function": {"name": "decide_on_traces"}},
        )
        if not resp.tool_calls:
            return DecideResult(decision=Decision.ACCEPT,
                                rationale="LLM returned no tool call; default ACCEPT.")
        args = resp.tool_calls[0].arguments or {}
        # Defensive parse: providers that don't enforce the schema's
        # `required` may omit "decision"; fall back to ACCEPT rather than
        # propagating a KeyError out of decide() (which would abort the
        # entire run on a single HarnessAgent hiccup).
        decision_value = args.get("decision")
        try:
            decision = Decision(decision_value) if decision_value else Decision.ACCEPT
        except (ValueError, KeyError):
            decision = Decision.ACCEPT
        return DecideResult(
            decision=decision,
            failure_analysis=_failure_analysis_from_args(args.get("failure_analysis")),
            rationale=args.get("rationale", ""),
        )

    def refine(self, candidate, traces, ctx):
        msgs = self._base_messages(
            ctx, intro="Refine the previous candidate based on the K rollouts.",
        )
        msgs.append(Message(role="user", content=self._format_candidate_block(candidate, traces)))
        msgs.append(Message(role="user", content=(
            "Refine all four independent slots. Preserve, adjust, add, or clear "
            "each slot based on the rollout evidence, and write a separate "
            "rationale for each populated slot. Use the same "
            "propose_candidate schema."
        )))
        resp = self.client.chat(
            messages=msgs, tools=[PROPOSE_TOOL],
            tool_choice={"type": "function",
                          "function": {"name": "propose_candidate"}},
        )
        if not resp.tool_calls:
            return candidate
        return _candidate_from_args(resp.tool_calls[0].arguments)

    def propose_changes(self, environment, validation_traces, ctx):
        """Generate an external-search proposal envelope before MCTS selects."""
        msgs = self._base_messages(
            ctx,
            intro=(
                "Generate possible environment changes for this validated "
                "environment before external MCTS selection."
            ),
        )
        msgs.append(Message(
            role="user",
            content=self._format_candidate_block(
                environment, validation_traces,
            ),
        ))
        msgs.append(Message(role="user", content=(
            "Return the usual four-slot propose_candidate structure. In this "
            "external-search turn, each non-empty slot is an ALTERNATIVE, "
            "independent edge candidate: Stage, f_A, f_T, or f_O. They will "
            "be split into up to four proposals and will NOT be applied "
            "together. Fill every individually useful alternative and give "
            "each populated slot its own rationale; leave inapplicable slots "
            "empty. Do not choose which proposal should be applied."
        )))
        resp = self.client.chat(
            messages=msgs,
            tools=[PROPOSE_TOOL],
            tool_choice={
                "type": "function",
                "function": {"name": "propose_candidate"},
            },
        )
        if not resp.tool_calls:
            return Candidate()
        return _candidate_from_args(resp.tool_calls[0].arguments)

    # ---- HarnessAgent-driven task selection ------------------------------
    def select_task(self, ctx: HarnessAgentContext) -> int:
        """Ask the LLM to pick one task_idx from `ctx.candidate_tasks`.
        Falls back to the first candidate on parse failure / unknown
        task_idx (logged via the LLM logger; no exception)."""
        if not ctx.candidate_tasks:
            raise ValueError("LLMHarnessAgent.select_task: empty candidate_tasks")
        valid_idxs = {t.task_idx for t in ctx.candidate_tasks}
        # Build a focused prompt: candidate shortlist + recent history.
        shortlist_block = self._format_candidate_tasks_block(ctx.candidate_tasks)
        recent = ctx.history_traces[-10:]
        history_text = "\n".join(
            f"  - cand={t.candidate_id} success={t.success} kind={t.kind}"
            for t in recent
        ) or "  (none)"
        msgs = [
            Message(role="system", content=self.system_prompt),
            Message(role="user", content=(
                f"Task description (project-level):\n{ctx.task_description}\n\n"
                f"Iteration: {ctx.iteration_id}\n\n"
                f"{shortlist_block}\n"
                f"Recent traces from earlier iterations (last {len(recent)}):\n"
                f"{history_text}\n\n"
                "Pick exactly ONE task_idx from the shortlist above. The "
                "framework will then run the standard propose/decide/refine "
                "loop on that task. You will see the BASELINE for the picked "
                "task in the next prompt."
            )),
        ]
        resp = self.client.chat(
            messages=msgs, tools=[SELECT_TASK_TOOL],
            tool_choice={"type": "function",
                          "function": {"name": "select_task"}},
        )
        if not resp.tool_calls:
            return ctx.candidate_tasks[0].task_idx
        args = resp.tool_calls[0].arguments or {}
        picked = args.get("task_idx")
        try:
            picked = int(picked)
        except (TypeError, ValueError):
            return ctx.candidate_tasks[0].task_idx
        if picked not in valid_idxs:
            # LLM hallucinated a task_idx not in the shortlist; fall back.
            return ctx.candidate_tasks[0].task_idx
        return picked

    def _format_candidate_tasks_block(self, tasks: list[TaskSummary]) -> str:
        """Render the shortlist for the HarnessAgent. Just data; the framework
        does NOT prescribe a selection heuristic ([[feedback-no-prescriptive-hints-to-mutator]])."""
        lines = ["CANDIDATE TASKS (pick exactly one task_idx):"]
        for t in tasks:
            extra = ""
            if t.metadata:
                # Only show metadata keys the experiment opted in (e.g.
                # repo, problem_excerpt). baseline_sr is excluded so the
                # selection prompt cannot leak eval signal.
                extra = f"  [metadata: {t.metadata}]"
            lines.append(f"  task_idx={t.task_idx}  instance_id={t.instance_id}{extra}")
            lines.append(f"      brief: {t.brief[:200]}")
        return "\n".join(lines) + "\n"

    # -----

    def _base_messages(self, ctx: HarnessAgentContext, intro: str) -> list[Message]:
        sig_text = ""
        if ctx.objective is not None:
            sig = ctx.objective.evaluate(ctx.history_traces)
            sig_text = (
                f"\nOBJECTIVE: {ctx.objective.name}\n"
                f"  score: {sig.score:.3f}\n"
                f"  diagnostic: {sig.diagnostic}\n"
                f"  suggestion: {sig.suggestion_prompt}\n"
                f"  axis_weights: {sig.weights}\n"
            )
        baseline_text = self._format_baseline_block(ctx.baseline, ctx.baseline_traces)
        recent = ctx.history_traces[-10:]
        history_text = "\n".join(
            f"  - ep={t.episode_id} cand={t.candidate_id} roll={t.rollout_idx} "
            f"success={t.success} reward={t.final_reward:.2f} kind={t.kind} "
            f"fail={t.failure_analysis.label if t.failure_analysis else '-'} "
            f"err={(t.error or '-')[:60]}"
            for t in recent
        ) or "  (none)"
        tools_text = json.dumps(ctx.tool_schemas, indent=2)[:3000]
        return [
            Message(role="system", content=self.system_prompt),
            Message(role="user", content=(
                f"Task: {ctx.task_description}\n"
                f"Iteration: {ctx.iteration_id}\n\n"
                f"{baseline_text}"
                f"Tool registry visible to Policy (full JSON schema):\n"
                f"```json\n{tools_text}\n```\n\n"
                f"env_state schema (what your hooks can read/write):\n"
                f"```\n{ctx.env_state_schema}\n```\n\n"
                f"Recent traces (last {len(recent)}):\n{history_text}\n"
                f"{sig_text}\n{intro}"
            )),
        ]

    def _format_baseline_block(self, baseline: BaselineSnapshot | None,
                                  baseline_traces: list[Trace] | None = None) -> str:
        """Render the per-task BASELINE (unmutated) block for the HarnessAgent
        prompt. Raw data only -- no prescriptive direction like "now try X
        axis"; the HarnessAgent must read the numbers and decide. Empty string
        if no baseline is available (compute_baseline=False).

        If baseline_traces is non-empty, also embeds full per-step content
        (think + action + obs) for one successful and one failed rollout so
        the HarnessAgent can diagnose WHY the Policy succeeded or failed."""
        if baseline is None:
            return ""
        per_rollout_str = ", ".join(
            f"({'✓' if r.success else '✗'}, {r.steps}stp)" for r in baseline.per_rollout
        )
        avg = ("n/a (no successes)" if baseline.avg_success_steps is None
                else f"{baseline.avg_success_steps:.1f}")

        # Pick representative success + failure trace, render full content
        full_trajectory_block = ""
        if baseline_traces:
            succ_trace = next((t for t in baseline_traces if t.success), None)
            fail_trace = next((t for t in baseline_traces if not t.success and t.steps), None)
            if succ_trace and fail_trace:
                # If both available, prefer SHORTEST success and LONGEST failure (most signal)
                succ_candidates = [t for t in baseline_traces if t.success and t.steps]
                if succ_candidates:
                    succ_trace = min(succ_candidates, key=lambda t: len(t.steps))
                fail_candidates = [t for t in baseline_traces if not t.success and t.steps]
                if fail_candidates:
                    fail_trace = max(fail_candidates, key=lambda t: len(t.steps))
            pieces = []
            if succ_trace and succ_trace.steps:
                pieces.append("  SAMPLE SUCCESSFUL TRAJECTORY (full think + action + obs):")
                pieces.append(self._render_steps_for_mutator(succ_trace.steps, max_steps=12))
            if fail_trace and fail_trace.steps:
                pieces.append("\n  SAMPLE FAILED TRAJECTORY (full think + action + obs):")
                pieces.append(self._render_steps_for_mutator(fail_trace.steps, max_steps=12))
            if pieces:
                full_trajectory_block = "\n".join(pieces) + "\n"
        # Legacy fallback: action-only success trajectory from BaselineSnapshot
        elif baseline.sample_success_actions:
            act_lines = []
            for i, a in enumerate(baseline.sample_success_actions):
                kw = ", ".join(f"{k}={v!r}" for k, v in (a.kwargs or {}).items())
                act_lines.append(f"    [{i+1:2d}] {a.name}({kw})")
            full_trajectory_block = (
                "  sample successful trajectory (action-only, shortest winning rollout, "
                f"{len(baseline.sample_success_actions)} steps):\n"
                + "\n".join(act_lines) + "\n"
            )
        else:
            full_trajectory_block = "  (no trajectories available)\n"

        return (
            "BASELINE (unmutated Policy on this exact task_id, run BEFORE your propose):\n"
            f"  success_rate:        {baseline.sr:.2f}  ({baseline.n_success}/{baseline.n} rollouts)\n"
            f"  avg_success_steps:   {avg}\n"
            f"  per_rollout:         [{per_rollout_str}]\n"
            f"{full_trajectory_block}\n"
        )

    @staticmethod
    def _render_steps_for_mutator(steps: list, max_steps: int = 12,
                                    max_think_chars: int = 500,
                                    max_obs_chars: int = 400,
                                    max_cmd_chars: int = 250) -> str:
        """Render a Trace's steps as think/action/obs for the HarnessAgent prompt.

        Truncates per-step content to keep prompt size bounded. If the trace
        has more than `max_steps`, shows the LAST `max_steps` (most relevant
        for diagnosing where the policy got stuck / pivoted).
        """
        import re
        THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
        ACTION_RE = re.compile(r"<action>.*?</action>", re.DOTALL)
        shown = steps[-max_steps:] if len(steps) > max_steps else steps
        start = len(steps) - len(shown)
        out = []
        for i, s in enumerate(shown):
            idx = start + i + 1
            raw = getattr(s, 'policy_raw_response', None) or ""
            think_m = THINK_RE.search(raw) if raw else None
            think_block = think_m.group(0) if think_m else ""
            if len(think_block) > max_think_chars:
                think_block = think_block[:max_think_chars] + " ...[truncated]</think>"
            ra = getattr(s, 'raw_action', None) or {}
            if hasattr(ra, 'model_dump'):
                ra = ra.model_dump()
            if isinstance(ra, dict):
                action_name = str(ra.get('name') or '')
                cmd_kwargs = ra.get('kwargs') or {}
                # Open-vocabulary bridges use different argument names:
                # SWE-bench uses ``command``, ALFWorld uses ``text``, and
                # WebArena uses ``action_str``.  Structured tools such as
                # Toy24 have no text field, so render the complete call.
                cmd = str(
                    cmd_kwargs.get('command')
                    or cmd_kwargs.get('text')
                    or cmd_kwargs.get('action_str')
                    or ''
                ).strip()
                if not cmd and action_name:
                    args = ", ".join(
                        f"{key}={value!r}" for key, value in cmd_kwargs.items()
                    )
                    cmd = f"{action_name}({args})"
            else:
                cmd = ""
            if len(cmd) > max_cmd_chars:
                cmd = cmd[:max_cmd_chars] + " ...[truncated]"
            fo = getattr(s, 'filtered_observation', None) or getattr(s, 'raw_observation', None) or {}
            if hasattr(fo, 'model_dump'):
                fo = fo.model_dump()
            obs = (fo.get('text', '') if isinstance(fo, dict) else '') or ""
            obs = obs.strip()
            if len(obs) > max_obs_chars:
                obs = obs[:max_obs_chars] + " ...[truncated]"
            info = getattr(s, 'info', None) or {}
            rc = info.get('returncode') if isinstance(info, dict) else None
            sub = " [SUBMITTED]" if (isinstance(info, dict) and info.get('submitted')) else ""
            block = [f"    Step {idx}:"]
            if think_block:
                block.append(f"      {think_block}")
            block.append(f"      <action>{cmd}</action>")
            if obs:
                block.append(f"      Observation: {obs}")
            if rc is not None:
                block.append(f"      rc={rc}{sub}")
            out.append("\n".join(block))
        omitted = len(steps) - len(shown)
        if omitted > 0:
            out.insert(0, f"    [showing last {len(shown)} of {len(steps)} steps; first {omitted} omitted]")
        return "\n\n".join(out)

    def _format_candidate_block(self, candidate: Candidate,
                                  traces: list[Trace]) -> str:
        n = len(traces)
        succ = sum(1 for t in traces if t.success)
        sr = succ / n if n else 0.0
        success_steps = [t.duration_steps for t in traces if t.success]
        avg_success_steps = (sum(success_steps) / len(success_steps)) if success_steps else float("nan")
        avg_reward = sum(t.final_reward for t in traces) / max(n, 1)

        rollout_lines = []
        for t in traces:
            actions = [s.raw_action.name for s in t.steps][:8]
            blocks = sum(1 for s in t.steps if s.blocked_reason)
            rollout_lines.append(
                f"    seed={t.rollout_seed} success={t.success} "
                f"reward={t.final_reward:.2f} steps={t.duration_steps} "
                f"blocked_steps={blocks} actions={actions} "
                f"err={(t.error or '-')[:60]}"
            )

        slots = {
            "f_A": candidate.f_A,
            "f_T": candidate.f_T,
            "f_O": candidate.f_O,
        }
        if not any(value.strip() for value in slots.values()):
            slots = _contract_slots_from_rules_code(candidate.rules_code)
        stage_lines = "\n".join(
            f"  - name={a.name!r} kwargs={a.kwargs!r}"
            for a in candidate.in_env_actions
        ) or "  (empty)"
        contract_lines = []
        for slot in ("f_A", "f_T", "f_O"):
            code = slots[slot]
            slot_rationale = getattr(candidate, f"{slot}_rationale")
            contract_lines.append(
                f"contract.{slot}.code:\n```python\n{code or '(empty)'}\n```\n"
                f"contract.{slot}.rationale: {slot_rationale or '(empty)'}"
            )
        payload = (
            f"stage.in_env_actions:\n{stage_lines}\n"
            f"stage.rationale: {candidate.stage_rationale or '(empty)'}\n"
            + "\n".join(contract_lines)
        )

        return (
            f"--- candidate ({n} rollouts) ---\n"
            + (f"legacy overall rationale: {candidate.rationale}\n"
               if candidate.rationale else "")
            + f"{payload}\n\n"
            f"ROLLOUT STATS:\n"
            f"  success_rate:      {sr:.2f} ({succ}/{n})\n"
            f"  avg_reward:        {avg_reward:.3f}\n"
            f"  avg_success_steps: {('n/a (no successes)' if succ == 0 else f'{avg_success_steps:.1f} (over {succ} winning rollouts)')}\n\n"
            f"per-rollout:\n" + "\n".join(rollout_lines)
        )


_VALID_FAILURE_AXES = {"S0", "A", "O", "T", "task_understanding", "none"}


def _failure_analysis_from_args(fa: Any) -> FailureAnalysis | None:
    """Defensive parse of the decide tool-call's failure_analysis blob.

    FailureAnalysis is extra='forbid' with an enum-constrained primary_axis,
    so `FailureAnalysis(**fa)` raises on a stray provider-added key or an
    off-enum axis -- and an exception out of decide() aborts the whole task,
    dropping its already-paid rollout traces. Degrade to a partial (or None)
    analysis instead: keep only the known fields, null an invalid axis."""
    if not isinstance(fa, dict):
        return None
    axis = fa.get("primary_axis")
    try:
        return FailureAnalysis(
            primary_axis=axis if axis in _VALID_FAILURE_AXES else None,
            label=str(fa.get("label") or ""),
            description=str(fa.get("description") or ""),
        )
    except Exception:
        return None


_CONTRACT_SLOT_METHODS = {
    "f_A": "filter_action",
    "f_T": "modify_transition",
    "f_O": "filter_observation",
}


def _clean_method_source(value: Any) -> str:
    """Normalize one tool-returned Contract method without changing code."""
    source = str(value or "").strip()
    if source.startswith("```") and source.endswith("```"):
        lines = source.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        source = "\n".join(lines)
    return textwrap.dedent(source).strip()


def _contract_change_from_value(value: Any) -> tuple[str, str]:
    """Parse one Contract slot, accepting the new object and legacy string."""
    if isinstance(value, dict):
        return (
            _clean_method_source(value.get("code")),
            str(value.get("rationale") or "").strip(),
        )
    return _clean_method_source(value), ""


def _rules_code_from_contract(contract: Any) -> str:
    """Combine independent f_A/f_T/f_O method strings into native Rules code."""
    if not isinstance(contract, dict):
        return ""
    methods = [
        _contract_change_from_value(contract.get(slot))[0]
        for slot in ("f_A", "f_T", "f_O")
    ]
    methods = [method for method in methods if method]
    if not methods:
        return ""
    return "class _Rules(Rules):\n" + "\n\n".join(
        textwrap.indent(method, "    ") for method in methods
    )


def _contract_slots_from_rules_code(code: str) -> dict[str, str]:
    """Recover f_A/f_T/f_O method strings for decide/refine prompts."""
    slots = {slot: "" for slot in _CONTRACT_SLOT_METHODS}
    if not (code or "").strip():
        return slots
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return slots
    rules_class = next(
        (node for node in tree.body
         if isinstance(node, ast.ClassDef) and node.name == "_Rules"),
        None,
    )
    if rules_class is None:
        return slots
    by_method = {name: slot for slot, name in _CONTRACT_SLOT_METHODS.items()}
    for node in rules_class.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        slot = by_method.get(node.name)
        if slot:
            slots[slot] = textwrap.dedent(
                ast.get_source_segment(code, node) or ""
            ).strip()
    return slots


def _candidate_from_args(args: dict[str, Any]) -> Candidate:
    args = args or {}
    # Skip malformed actions that omit "name" instead of raising KeyError --
    # a single missing-field response from a permissive provider should not
    # abort the whole run from inside _candidate_from_args.
    stage = args.get("stage") if isinstance(args.get("stage"), dict) else {}
    raw_actions = stage.get("in_env_actions")
    if raw_actions is None:
        raw_actions = args.get("in_env_actions") or []
    actions: list[Action] = []
    for a in raw_actions:
        a = a or {}
        name = a.get("name")
        if not name:
            continue
        kw = a.get("kwargs")
        if kw is None:
            raw = a.get("kwargs_json")
            if isinstance(raw, str) and raw.strip():
                try:
                    kw = json.loads(raw)
                except Exception:
                    kw = {}
            elif isinstance(raw, dict):
                kw = raw
        if not isinstance(kw, dict):
            kw = {}
        actions.append(Action(name=name, kwargs=kw))
    contract = args.get("contract")
    parsed_contract = {
        slot: _contract_change_from_value(
            contract.get(slot) if isinstance(contract, dict) else ""
        ) for slot in ("f_A", "f_T", "f_O")
    }
    contract_slots = {
        slot: parsed_contract[slot][0] for slot in ("f_A", "f_T", "f_O")
    }
    contract_rationales = {
        f"{slot}_rationale": parsed_contract[slot][1]
        for slot in ("f_A", "f_T", "f_O")
    }
    rules_code = _rules_code_from_contract(contract_slots)
    if not rules_code:
        # Backward-compatible ingestion for existing saved provider outputs.
        rules_code = args.get("rules_code", "") or ""
        contract_slots = _contract_slots_from_rules_code(rules_code)

    return Candidate(
        rules_code=rules_code,
        in_env_actions=actions,
        **contract_slots,
        stage_rationale=str(stage.get("rationale") or "").strip(),
        **contract_rationales,
        rationale=args.get("rationale", ""),
    )


# ---------------------------------------------------------------------------
# Test agents -- offline plumbing tests of the orchestrator / runner. They
# emit fixed Candidates instead of calling an LLM.
# ---------------------------------------------------------------------------

class ScriptedHarnessAgent(HarnessAgent):
    def __init__(self, fn):
        self.fn = fn

    def propose(self, ctx):
        return self.fn("propose", {"ctx": ctx})

    def decide(self, candidate, traces, ctx):
        return self.fn("decide", {"candidate": candidate, "traces": traces, "ctx": ctx})

    def refine(self, candidate, traces, ctx):
        return self.fn("refine", {"candidate": candidate, "traces": traces, "ctx": ctx})


class NoopHarnessAgent(HarnessAgent):
    """Always pass-through, always ACCEPTs. Open-loop baseline."""
    def propose(self, ctx):
        return Candidate(rationale="noop")

    def decide(self, candidate, traces, ctx):
        return DecideResult(decision=Decision.ACCEPT,
                             rationale="noop accepts everything")

    def refine(self, candidate, traces, ctx):
        return candidate


# ---------------------------------------------------------------------------
# ExploringHarnessAgent -- code-free mutation via interactive env exploration
# ---------------------------------------------------------------------------

_EXPLORE_DECIDE_SYSTEM = """You are an Environment Mutation Explorer reviewing the outcome of a perturbation you just produced.

The framework will show you (1) the perturbation trajectory you emitted, (2) what the downstream Policy agent did on each of the K rollouts, and (3) the target Policy success-rate band (as numbers).

The target band [lo, hi] IS the user's goal. The decision rule has exactly TWO cases:

CASE 1 -- the observed Policy SR is INSIDE the band (lo <= SR <= hi):
  Your perturbation achieved what the user asked for. ACCEPT.
  (You may still refine if you have a specific concrete reason -- e.g. the in-band SR is the result of a side effect rather than the intended perturbation -- but the default for in-band is ACCEPT.)

CASE 2 -- the observed Policy SR is OUTSIDE the band (SR < lo OR SR > hi):
  Your perturbation did NOT achieve the goal. You MUST refine. There are NO exceptions to this rule, even if the perturbation "did something" or "made the task trivial". Specifically:
    - SR = 100% means the Policy was unaffected by your perturbation. This is a failure (the user wanted [lo, hi], not 100%) and you MUST refine. Do not rationalize ACCEPT on the grounds that "the task is now trivial" or "the perturbation made some change" -- the only measure of success is whether SR is in the band.
    - SR = 0% means you broke the task (Policy cannot solve at all). This is also outside the band; you MUST refine.

When refining, the environment's actions are mostly irreversible, so you also choose HOW to refine:

  REFINE_CONTINUE -- keep the current perturbation as the starting state and ADD more actions on top. Use when your current trajectory is on the right track but needs more steps.

  REFINE_RESTART -- drop the current trajectory entirely and start from a fresh Bridge.reset, building a new perturbation. Use when the current trajectory took the env in a wrong direction that you cannot undo (e.g. you cleaned an object you should not have, you took an irrelevant item, you got the agent stuck somewhere unhelpful).

Output your decision on the FIRST LINE as exactly one of:
  ACCEPT
  REFINE_CONTINUE
  REFINE_RESTART

You may add a short rationale on subsequent lines. The framework will use only the first line to route the loop; the rationale is for logs.

The framework provides measurement (SR vs band) and the protocol above. Which perturbations to make, whether an in-band attempt is worth refining further, and continue-vs-restart are your decisions."""


_EXPLORE_SYSTEM_DEFAULT = """You are an Environment Mutation Explorer.

Your job: take actions on the environment to perturb its starting state for a downstream Policy agent. You are NOT solving the task yourself.

Each turn you receive an observation + admissible commands. Pick ONE command from the admissible list and output it on a single line, with NO additional explanation, no quotes, no leading bullet, no JSON. When you decide the perturbation is complete, output the single word:

STOP

After your trajectory is recorded, the framework will replay it as the starting state for a Policy agent and report back what happened on the rollouts. On any refinement turn, prior trajectories and their Policy outcomes will be shown to you as raw context; decide what to change based on that data."""


class ExploringHarnessAgent(HarnessAgent):
    """Code-free HarnessAgent. Generates Candidate.in_env_actions by driving a
    scratch Bridge with the same single-tool text-completion protocol the
    Policy uses (see PolicyAgent.text_complete).

    Output: `Candidate(in_env_actions=<trajectory>, rules_code="")`. The
    trajectory is JSON-serializable -- any future run rehydrates the mutated
    initial state by wrapping the env in a `Setup` harness whose reset()
    replays this list via inner.step() (the same save/load mechanism the
    persistence layer uses for the "setup" checkpoint entry).

    Why not extend LLMHarnessAgent? LLMHarnessAgent generates Python source for
    the 3 A/T/O Rules hooks (per-step transformations). This class operates on the
    S0 axis only and never writes code. Cleanest as a sibling, not a subclass.

    Closed-loop refinement: `decide()` reads the K-rollout SR and compares it
    to the active objective's target band (DifficultyZone.lo / .hi). If the
    perturbation was too easy (SR > hi) or too hard / unsolvable (SR < lo),
    decide() returns REFINE; the orchestrator then calls `refine()`, which
    re-runs the exploration with a feedback message summarizing what just
    happened so the LLM can revise its perturbation. The orchestrator loops
    on REFINE until ACCEPT or `budget.max_k` is exhausted.
    """

    def __init__(self, client: LLMClient, env_spec,
                 max_explore_steps: int = 10,
                 explore_temperature: float = 0.7,
                 system_prompt: str | None = None,
                 stop_token: str = "STOP",
                 max_trace_steps_in_feedback: int = 30):
        self.client = client
        self.env_spec = env_spec
        self.max_explore_steps = max_explore_steps
        self.explore_temperature = explore_temperature
        self.system_prompt = system_prompt or _EXPLORE_SYSTEM_DEFAULT
        self.stop_token = stop_token
        self._max_trace_steps_in_feedback = int(max_trace_steps_in_feedback)
        # Set by decide() when the LLM picks REFINE_CONTINUE vs REFINE_RESTART;
        # consumed by the next refine() call. Default "restart" preserves the
        # pre-3-way behavior on first refine if decide() didn't write it (e.g.
        # decide returned REFINE on an unparseable response).
        self._next_refine_mode: str = "restart"

    def propose(self, ctx: HarnessAgentContext) -> Candidate:
        return self._explore(ctx, feedback="", seed_actions=None)

    def refine(self, candidate: Candidate, traces: list[Trace],
                ctx: HarnessAgentContext) -> Candidate:
        """Refine using the mode chosen by the most recent decide() call.

          - "continue": replay the previous in_env_actions through the
            Bridge as the new starting state, then add more actions on
            top. Final in_env_actions = previous + new.
          - "restart" (default): drop the previous trajectory; the HarnessAgent
            drives a fresh Bridge.reset from scratch. Useful when the
            previous trajectory took the env in an irreversible wrong
            direction.

        The HarnessAgent LLM picks continue-vs-restart inside decide() based on
        the rollout outcomes; the framework just routes the loop."""
        feedback = self._format_refine_feedback(candidate, traces, ctx)
        if self._next_refine_mode == "continue":
            seed = list(candidate.in_env_actions)
        else:
            seed = None
        return self._explore(ctx, feedback=feedback, seed_actions=seed)

    def decide(self, candidate: Candidate, traces: list[Trace],
                ctx: HarnessAgentContext) -> DecideResult:
        """Ask the HarnessAgent LLM to decide ACCEPT or REFINE given the raw
        K-rollout outcomes + the configured target band. No hardcoded
        if/else: the framework only does numeric measurement (SR, band);
        the categorization is the HarnessAgent's call.

        Falls back to deterministic ACCEPT only when there is no objective
        configured -- nothing to compare against, no decision to make."""
        from envharness.infra.llm import Message
        n = max(len(traces), 1)
        n_succ = sum(1 for t in traces if t.success)
        sr = n_succ / n
        lo, hi = self._get_band(ctx)
        if lo is None:
            return DecideResult(
                decision=Decision.ACCEPT,
                rationale=f"no target band configured; SR={sr:.0%}; accept.",
            )

        user_msg = self._format_decide_context(candidate, traces, ctx, sr, lo, hi)
        history = [
            Message(role="system", content=_EXPLORE_DECIDE_SYSTEM),
            Message(role="user", content=user_msg),
        ]
        resp = self.client.chat(
            messages=history, tools=None,
            temperature=self.explore_temperature,
        )
        raw = (resp.content or "").strip()
        first = next((ln.strip() for ln in raw.splitlines() if ln.strip()), "").upper()
        # 3-way: ACCEPT / REFINE_CONTINUE / REFINE_RESTART. Map the latter two
        # to Decision.REFINE and stash the mode for refine() to consume.
        refine_mode = "restart"  # default if the LLM said "REFINE" without suffix
        if first.startswith("ACCEPT"):
            decision = Decision.ACCEPT
        elif first.startswith("REFINE_CONTINUE") or first.startswith("REFINE-CONTINUE") or first == "CONTINUE":
            decision = Decision.REFINE
            refine_mode = "continue"
        elif first.startswith("REFINE_RESTART") or first.startswith("REFINE-RESTART") or first == "RESTART":
            decision = Decision.REFINE
            refine_mode = "restart"
        elif first.startswith("REFINE"):
            # Bare "REFINE" without continue/restart suffix -- default to
            # restart (the more conservative choice).
            decision = Decision.REFINE
            refine_mode = "restart"
        else:
            # Unparseable -> default to REFINE_RESTART so the loop continues
            # but doesn't get stuck extending a broken trajectory.
            decision = Decision.REFINE
            refine_mode = "restart"
        # Stash the refine mode for the next refine() call.
        self._next_refine_mode = refine_mode
        return DecideResult(
            decision=decision,
            failure_analysis=None,
            rationale=(f"SR={sr:.0%}  band=[{lo:.0%},{hi:.0%}]  "
                       f"mode={refine_mode}  mutator: {raw[:180]}"),
        )

    def _format_decide_context(self, candidate: Candidate,
                                 traces: list[Trace],
                                 ctx: HarnessAgentContext,
                                 sr: float, lo: float, hi: float) -> str:
        """Same raw-data shape as the refine feedback (so the HarnessAgent sees
        the same evidence in both decide and refine), plus the explicit
        target band as numbers."""
        feedback = self._format_refine_feedback(candidate, traces, ctx)
        return (
            f"Target SR band (from objective): [{lo:.0%}, {hi:.0%}]\n\n"
            + feedback
            + "\n\nDecide whether to ACCEPT this perturbation as-is, or "
              "REFINE it (you will then write a new trajectory)."
        )

    # ----- internals ----------------------------------------------------

    def _explore(self, ctx: HarnessAgentContext, feedback: str = "",
                   seed_actions: list | None = None) -> Candidate:
        """The env-driving exploration loop.

        propose() calls with seed_actions=None: fresh env.reset, then the
        HarnessAgent LLM drives up to max_explore_steps actions.

        refine() calls with seed_actions = prev_candidate.in_env_actions:
        the base env is wrapped in `Setup(actions=seed_actions)`, whose
        reset() replays the previous trajectory via inner.step() (no LLM
        calls -- a deterministic replay of where the HarnessAgent left off,
        identical to how the K rollouts and checkpoint loads rehydrate the
        Candidate). The LLM then adds up to max_explore_steps MORE actions
        on top. Final Candidate carries seed_actions + new_actions, so the
        in_env_actions trajectory grows monotonically across refinements.
        """
        from envharness.infra.llm import Message
        from envharness.agents.policy import _normalize_command
        from envharness.core.types import Action
        from envharness.harnesses.setup import Setup
        from envharness.infra.utils import import_symbol

        # Resolve the single Bridge tool's name+arg (text_complete contract).
        if len(ctx.tool_schemas) != 1:
            raise ValueError(
                "ExploringHarnessAgent requires exactly one Bridge tool; got "
                f"{len(ctx.tool_schemas)}"
            )
        fn = ctx.tool_schemas[0]["function"]
        tool_name = fn["name"]
        props = fn.get("parameters", {}).get("properties", {})
        if len(props) != 1:
            raise ValueError(
                "ExploringHarnessAgent: Bridge tool must have exactly one "
                f"argument; got {list(props)}"
            )
        arg_name = next(iter(props))

        seed_actions = list(seed_actions or [])
        env_cls = import_symbol(self.env_spec.import_path)
        env = env_cls()
        if seed_actions:
            # Refine-continue path: replay the previous trajectory through
            # the SAME mechanism every other consumer of this Candidate uses
            # to rehydrate it -- a Setup layer whose reset() replays each
            # action via inner.step(). The explorer therefore continues from
            # exactly the state the K rollouts (and a checkpoint load) will
            # reconstruct from `in_env_actions`.
            env = Setup(inner=env, actions=seed_actions)
        new_actions: list[Action] = []
        new_normalized: list[str] = []
        try:
            reset = env.reset(
                seed=ctx.task_id,
                options=dict(self.env_spec.reset_options or {}),
            )
            # Setup.reset already returns the POST-replay observation.
            obs = reset.observation

            sys_content = self.system_prompt + f"\n\nTask context: {ctx.task_description}"
            if seed_actions:
                # Tell the LLM the env state already has the previous
                # trajectory applied; it is now ADDING actions, not replacing.
                sys_content += (
                    "\n\nNOTE: This is a REFINEMENT. The Bridge state already "
                    "has your previous perturbation trajectory applied "
                    f"({len(seed_actions)} action(s)). The observation below "
                    "reflects the post-replay state. Output STOP if no "
                    "further actions are needed."
                )
            if feedback:
                sys_content += "\n\n" + feedback
            history: list[Message] = [Message(role="system", content=sys_content)]
            for step in range(self.max_explore_steps):
                history.append(Message(role="user",
                                         content=self._format_obs(obs, step)))
                # No silent except: rate-limit / credential errors should
                # surface as a hard failure, not as "0 actions recorded".
                resp = self.client.chat(
                    messages=history, tools=None,
                    temperature=self.explore_temperature,
                )
                raw = (resp.content or "").strip()
                history.append(Message(role="assistant", content=raw))
                if self._is_stop(raw):
                    break
                admissible = list((obs.data or {}).get("admissible_commands") or [])
                normalized = _normalize_command(raw, admissible)
                if not normalized:
                    continue
                new_normalized.append(normalized)
                act = Action(name=tool_name, kwargs={arg_name: normalized})
                new_actions.append(act)
                env_resp = env.step(act)
                obs = env_resp.observation
        finally:
            try:
                env.close()
            except Exception:
                pass
        final_actions = seed_actions + new_actions
        rationale = (
            f"Explorer: {len(seed_actions)} replayed + {len(new_actions)} new "
            f"= {len(final_actions)} action(s). New: {new_normalized[:6]}"
            + (" ..." if len(new_normalized) > 6 else "")
            + (" [refined: extend]" if seed_actions else "")
        )
        return Candidate(
            rules_code="",
            in_env_actions=final_actions,
            rationale=rationale,
        )

    @staticmethod
    def _get_band(ctx: HarnessAgentContext) -> tuple[float | None, float | None]:
        """Pull (lo, hi) from the configured objective (DifficultyZone uses
        .lo and .hi). Returns (None, None) if no band is configured."""
        obj = ctx.objective
        if obj is None:
            return None, None
        lo = getattr(obj, "lo", None)
        hi = getattr(obj, "hi", None)
        if lo is None or hi is None:
            return None, None
        return float(lo), float(hi)

    def _format_refine_feedback(self, prev_candidate: Candidate,
                                  traces: list[Trace],
                                  ctx: HarnessAgentContext) -> str:
        """Raw observations from the previous attempt -- no prescriptive
        direction hints. The HarnessAgent LLM reads the trajectories and decides
        for itself what (if anything) to change. The framework supplies
        measurements only, never a suggested direction."""
        lo, hi = self._get_band(ctx)
        n = max(len(traces), 1)
        n_succ = sum(1 for t in traces if t.success)
        sr = n_succ / n

        # 1) The previous perturbation trajectory the HarnessAgent emitted.
        prior_actions = []
        for a in (prev_candidate.in_env_actions or []):
            v = next(iter(a.kwargs.values()), "") if a.kwargs else ""
            prior_actions.append(f"  - {v}")
        if not prior_actions:
            prior_actions.append("  (no actions applied last time)")

        # 2) Per-rollout summary: how the Policy actually behaved on the
        #    perturbed env, step by step. Truncated to keep prompts bounded.
        rollout_blocks: list[str] = []
        for i, t in enumerate(traces):
            verdict = "WON" if t.success else "lost"
            steps = (t.steps or [])[: self._max_trace_steps_in_feedback]
            lines = [f"  Rollout {i}: {verdict}  ({t.duration_steps} steps"
                      + (f", terminated by error: {t.error}" if t.error else "") + ")"]
            for j, s in enumerate(steps):
                act = getattr(s.raw_action, "name", "?")
                kwargs = getattr(s.raw_action, "kwargs", {}) or {}
                act_str = next(iter(kwargs.values()), "") if kwargs else act
                obs_txt = ""
                if s.filtered_observation is not None:
                    raw_obs = getattr(s.filtered_observation, "text", "")
                    # First non-empty line of the Bridge observation -- enough
                    # for the LLM to see effect vs. \"Nothing happens\".
                    obs_txt = next((ln.strip() for ln in (raw_obs or "").splitlines()
                                     if ln.strip()), "")[:180]
                lines.append(f"    step {j}: action={act_str!r}  obs={obs_txt!r}")
            if t.duration_steps > self._max_trace_steps_in_feedback:
                lines.append(f"    ... ({t.duration_steps - self._max_trace_steps_in_feedback} more steps truncated)")
            rollout_blocks.append("\n".join(lines))

        # 3) Objective numbers (raw, no verdict).
        if lo is not None and hi is not None:
            band_str = f"  target SR band: [{lo:.0%}, {hi:.0%}]   observed SR: {sr:.0%}  ({n_succ}/{n})"
        else:
            band_str = f"  observed SR: {sr:.0%}  ({n_succ}/{n})  (no target band configured)"

        return (
            "PRIOR ATTEMPT OUTCOME (raw -- decide for yourself what to change):\n"
            f"  Perturbation trajectory you emitted last time ({len(prev_candidate.in_env_actions)} actions):\n"
            + "\n".join(prior_actions) + "\n"
            + band_str + "\n"
            "  Policy K-rollouts on the perturbed env:\n"
            + "\n\n".join(rollout_blocks) + "\n"
            "Produce your next perturbation trajectory now."
        )

    @staticmethod
    def _format_obs(obs, step: int) -> str:
        return f"--- explorer step {step} ---\n{obs.text}\n\nYour command:"

    def _is_stop(self, raw: str) -> bool:
        token = self.stop_token.strip().lower()
        first = next((ln.strip().lower() for ln in (raw or "").splitlines()
                       if ln.strip()), "")
        return first == token
