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

"""Rules -- the EnvHarness with A / T / O transformation hooks.

`Rules` wraps an inner ActionableEnv and applies up to three per-step
hooks:

  - filter_action(action, env_state)               -- A axis
  - prepare_transition(action, env_state)          -- T, before native execution
  - finalize_transition(action, plan, responses,
                        env_state)                  -- T, after native execution
  - modify_transition(action, response, env_state) -- legacy post-only T
  - filter_observation(obs, env_state)             -- O axis

Initial-state changes use the separate Stage protocol implemented by
`Setup.actions`: Setup replays each action through `inner.step(action)`
before the Policy starts.

All hook defaults are pass-through. A useful `Rules` is a SUBCLASS that
overrides one or more hooks -- which is what the HarnessAgent
emits as Python source. The framework loads that source, finds the
`_Rules` subclass, and instantiates it with `inner=<the wrapped env>`.

Example of LLM-emitted code::

    class _Rules(Rules):
        def filter_action(self, action, env_state):
            if action.name == "combine" and action.kwargs.get("op") == "div":
                return Blocked(reason="division disabled")
            return action

Saved state is the Python source string; loading recompiles it.
"""
from __future__ import annotations

import copy
from typing import Any

from envharness.core.actionable_env import ActionableEnv
from envharness.core.envharness import EnvHarness
from envharness.core.registry import register_harness
from envharness.core.types import (
    Action, Blocked, EnvResetResponse, EnvResponse, Observation, TransitionPlan,
)


@register_harness("rules")
class Rules(EnvHarness):
    """A/T/O code-as-mutation EnvHarness. Subclass to override hooks.

    Initial-state changes live in the `Setup` harness (action-list replay).

    The instance carries the source code it was compiled from in
    `self.rules_code` so save_state can round-trip.
    """

    rules_code: str = ""

    def _uses_two_phase_transition(self) -> bool:
        return (
            type(self).prepare_transition is not Rules.prepare_transition
            or type(self).finalize_transition is not Rules.finalize_transition
        )

    # ------------------------------------------------------------------
    # Hooks (defaults: pass-through). Override in subclasses.
    # ------------------------------------------------------------------

    def filter_action(self, action: Action, env_state: Any) -> Action | Blocked:
        """A: transform or reject the agent's action before inner.step."""
        return action

    def prepare_transition(self, action: Action,
                           env_state: Any) -> TransitionPlan:
        """T/pre: plan zero or more native actions for one visible action.

        The default preserves the original one-action transition. A custom
        implementation should keep pre-step facts in ``plan.context`` and
        commit virtual state only after checking the native responses.
        """
        return TransitionPlan(actions=[action])

    def modify_transition(self, action: Action, raw_response: EnvResponse,
                          env_state: Any) -> EnvResponse:
        """Legacy T/post hook, retained for existing candidate code."""
        return raw_response

    def finalize_transition(self, action: Action, plan: TransitionPlan,
                            native_responses: list[EnvResponse],
                            env_state: Any) -> EnvResponse:
        """T/post: inspect every native result and return the visible result.

        The default is backward compatible with ``modify_transition``.
        Native execution stops at termination/truncation, so the final
        response is the last result actually produced. A zero-action plan
        starts from the current observation and zero reward.
        """
        if native_responses:
            last = native_responses[-1].model_copy(deep=True)
            last.reward = sum(response.reward for response in native_responses)
        else:
            last = EnvResponse(
                observation=self.inner.observe(), reward=0.0,
                terminated=False, truncated=False,
                info={"transition_no_native_action": True},
            )
        return self.modify_transition(action, last, env_state)

    def filter_observation(self, obs: Observation, env_state: Any) -> Observation:
        """O: transform the Observation the agent sees."""
        return obs

    # ------------------------------------------------------------------
    # ActionableEnv overrides -- the A / T / O step loop. No S0 / R.
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None,
              options: dict[str, Any] | None = None) -> EnvResetResponse:
        self._last_step_hooks = None
        self._current_transition_observation = None
        reset_resp = self.inner.reset(seed, options)
        # O hook fires on the initial observation so the agent's first
        # sight of the env is consistent with what it'll see going
        # forward. We use inner.observe() to refresh in case an outer
        # Setup layer changed state between reset and now.
        env_state = self.inner.get_env_state()
        fresh = self.inner.observe()
        raw_initial = fresh.model_copy(deep=True)
        self._current_transition_observation = raw_initial.model_copy(deep=True)
        filtered = self.filter_observation(fresh, env_state)
        self._last_reset_observation = raw_initial
        return EnvResetResponse(
            observation=filtered,
            info=dict(reset_resp.info),
        )

    def step(self, action: Action) -> EnvResponse:
        self._last_step_hooks = None
        env_state = self.inner.get_env_state()
        filtered = self.filter_action(action, env_state)
        if isinstance(filtered, Blocked):
            # A blocked action leaves the env unchanged, so re-observe and
            # give the agent the CURRENT state alongside the block reason.
            # A bare "[blocked]" Observation would strip env-critical
            # obs.data (webarena's browsergym_raw, alfworld's
            # admissible_commands) and strand or crash format-specific
            # policies on their next act(). The O hook applies as usual.
            raw_obs = self.inner.observe()
            raw_obs_copy = raw_obs.model_copy(deep=True)
            cached = (getattr(self, "_current_transition_observation", None)
                      if self._uses_two_phase_transition() else None)
            transition_obs = (cached or raw_obs_copy).model_copy(deep=True)
            fresh = self.filter_observation(transition_obs, env_state)
            response = EnvResponse(
                observation=Observation(
                    text=f"[blocked] {filtered.reason}\n\n{fresh.text}",
                    data={**(fresh.data or {}),
                          "blocked": True,
                          "blocked_reason": filtered.reason},
                ),
                reward=0.0, terminated=False, truncated=False,
                info={"blocked_reason": filtered.reason},
            )
            self._last_step_hooks = {
                "filtered_action": None,
                "raw_observation": raw_obs_copy,
                "transition_observation": None,
                "filtered_observation": response.observation,
            }
            return response
        plan = self.prepare_transition(filtered, env_state)
        if not isinstance(plan, TransitionPlan):
            raise TypeError("prepare_transition must return TransitionPlan")
        native_responses: list[EnvResponse] = []
        for native_action in plan.actions:
            response = self.inner.step(native_action)
            native_responses.append(response)
            if response.terminated or response.truncated:
                break
        if native_responses:
            raw_response = native_responses[-1]
        else:
            raw_response = EnvResponse(
                observation=self.inner.observe(), reward=0.0,
                terminated=False, truncated=False,
                info={"transition_no_native_action": True},
            )
        raw_obs_copy = raw_response.observation.model_copy(deep=True)
        raw_info_copy = copy.deepcopy(raw_response.info)
        native_response_copies = [response.model_copy(deep=True)
                                  for response in native_responses]
        # Re-fetch env_state for the post-transition hooks: envs that REPLACE
        # their state object on step (webarena's _build_env_state) would
        # otherwise hand T/O hooks the PRE-step snapshot -- url/axtree one
        # step stale, every step. Envs that mutate state in place (toy24,
        # alfworld) return the same object, so this is a no-op for them.
        # filter_action above deliberately keeps the pre-action state.
        post_state = self.inner.get_env_state()
        transformed = self.finalize_transition(
            filtered, plan, native_responses, post_state,
        )
        if not isinstance(transformed, EnvResponse):
            raise TypeError("finalize_transition must return EnvResponse")
        transition_obs_copy = transformed.observation.model_copy(deep=True)
        self._current_transition_observation = transition_obs_copy.model_copy(deep=True)
        new_obs = self.filter_observation(transformed.observation, post_state)
        self._last_step_hooks = {
            "filtered_action": filtered.model_copy(deep=True),
            "raw_observation": raw_obs_copy,
            "raw_reward": sum(response.reward for response in native_response_copies),
            "raw_info": raw_info_copy,
            "transition_observation": transition_obs_copy,
            "filtered_observation": new_obs.model_copy(deep=True),
        }
        if self._uses_two_phase_transition():
            self._last_step_hooks["transition_plan_actions"] = [
                native_action.model_copy(deep=True)
                for native_action in plan.actions
            ]
            self._last_step_hooks["transition_native_responses"] = (
                native_response_copies
            )
        return EnvResponse(
            observation=new_obs,
            reward=transformed.reward,                  # R untouched by Rules
            terminated=transformed.terminated,
            truncated=transformed.truncated,
            info=transformed.info,
        )

    def observe(self) -> Observation:
        env_state = self.inner.get_env_state()
        source = (getattr(self, "_current_transition_observation", None)
                  if self._uses_two_phase_transition() else None)
        if source is None:
            source = self.inner.observe()
        return self.filter_observation(source.model_copy(deep=True), env_state)

    # evaluate / get_env_state / step_reward -- inherited delegation.

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save_state(self) -> dict:
        return {"rules_code": self.rules_code or ""}

    @classmethod
    def from_state(cls, state: dict,  # type: ignore[override]
                   inner: ActionableEnv | None = None) -> "Rules":
        code = (state.get("rules_code") or "").strip()
        if not code:
            # Empty code = pass-through Rules (all hook defaults).
            return cls(inner=inner)
        # Late import: code_loader depends on Rules (this module).
        from envharness.core.code_loader import load_rules_subclass
        subclass = load_rules_subclass(code)
        instance = subclass(inner=inner)
        instance.rules_code = code
        return instance
