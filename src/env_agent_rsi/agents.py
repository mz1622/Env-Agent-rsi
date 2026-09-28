"""Exactly-once 场景的确定性基线策略。

本文件保留 naive 与 oracle 两条可解释策略，用于验证故障语义和 verifier，而不承担
通用模型执行；真实 ModelClient episode 由 ``agent_runtime`` 负责。
"""

from __future__ import annotations

from typing import Callable

from env_agent_rsi.core.protocol import (
    Action,
    ActionableEnv,
    EvaluationResult,
    JsonObject,
)


def _record(env: ActionableEnv, action: Action, trace: list[JsonObject]) -> JsonObject:
    response = env.step(action)
    trace.append({"action": action.to_dict(), "response": response.to_dict()})
    return response.observation


def run_naive_agent(
    env: ActionableEnv, target: str
) -> tuple[EvaluationResult, list[JsonObject]]:
    """Retry any failed append without an idempotency key."""

    trace: list[JsonObject] = []
    observation = _record(env, Action("append_item", {"value": target}), trace)
    if not observation.get("ok"):
        _record(env, Action("append_item", {"value": target}), trace)
    _record(env, Action("finish"), trace)
    return env.evaluate(), trace


def run_oracle_agent(
    env: ActionableEnv, target: str
) -> tuple[EvaluationResult, list[JsonObject]]:
    """Use a stable key and verify uncertain writes before retrying."""

    trace: list[JsonObject] = []
    key = f"task:{target}"
    observation = _record(
        env,
        Action("append_item", {"value": target, "idempotency_key": key}),
        trace,
    )
    if not observation.get("ok"):
        found = False
        for _ in range(3):
            listed = _record(
                env, Action("list_items", {"cursor": 0, "limit": 100}), trace
            )
            found = any(item.get("value") == target for item in listed.get("items", []))
            if found:
                break
        if not found:
            _record(
                env,
                Action("append_item", {"value": target, "idempotency_key": key}),
                trace,
            )
    _record(env, Action("finish"), trace)
    return env.evaluate(), trace


AGENTS: dict[
    str,
    Callable[[ActionableEnv, str], tuple[EvaluationResult, list[JsonObject]]],
] = {"naive": run_naive_agent, "oracle": run_oracle_agent}
