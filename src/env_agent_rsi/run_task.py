"""场景统一命令行入口。

``oracle`` 用确定性策略校准环境，``manual`` 提供 JSON Lines 外部桥，``replay``
通过 ScriptedModelClient 和 AgentRunner 走完整模型工具链；三种模式共享同一个
factory、RuleHarness 和 verifier。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from env_agent_rsi.agent_runtime import AgentRunner, ScriptedModelClient
from env_agent_rsi.agents import run_oracle_agent
from env_agent_rsi.core.protocol import Action
from env_agent_rsi.harness.factory import build_environment, load_spec
from env_agent_rsi.scenario_agents import SCENARIO_ORACLES


def _print(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True), flush=True)


def _agent_response(response) -> dict[str, object]:
    """过滤 privileged info，只保留外部 Agent 可见字段。"""

    return {
        "observation": response.observation,
        "reward": response.reward,
        "terminated": response.terminated,
        "truncated": response.truncated,
    }


def _run_manual(env) -> None:
    """JSON-lines bridge usable by any external model/tool-calling loop."""

    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            action = Action(str(payload["tool"]), dict(payload.get("arguments", {})))
            response = env.step(action)
            _print(_agent_response(response))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            _print(
                {
                    "observation": {
                        "ok": False,
                        "error": {"code": "INVALID_ACTION", "message": str(exc)},
                    }
                }
            )
        if env.get_env_state()["terminated"]:
            break


def _load_replay(path: Path) -> list[Action]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("replay file must contain a JSON array")
    actions = []
    for index, entry in enumerate(payload):
        if not isinstance(entry, dict) or "tool" not in entry:
            raise ValueError(f"replay action {index} must be an object with tool")
        actions.append(Action(str(entry["tool"]), dict(entry.get("arguments", {}))))
    return actions


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one catalog scenario")
    parser.add_argument("scenario", type=Path)
    parser.add_argument(
        "--agent", choices=("oracle", "manual", "replay"), default="oracle"
    )
    parser.add_argument(
        "--replay",
        type=Path,
        help="JSON action array used by --agent replay through the generic AgentRunner",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    spec = load_spec(args.scenario)
    env = build_environment(spec)
    environment_type = str(spec.get("environment", {}).get("type", "item"))
    if args.agent == "manual":
        reset = env.reset(seed=args.seed)
        _print(_agent_response(reset))
        _run_manual(env)
        return
    if args.agent == "replay":
        if args.replay is None:
            parser.error("--agent replay requires --replay")
        result = AgentRunner(env, ScriptedModelClient(_load_replay(args.replay))).run(
            seed=args.seed
        )
        _print({"scenario": str(args.scenario), **result.to_dict()})
        return
    reset = env.reset(seed=args.seed)
    if environment_type == "item":
        target = str(spec.get("task", {}).get("target_value", "target-item"))
        evaluation, trace = run_oracle_agent(env, target)
    else:
        evaluation, trace = SCENARIO_ORACLES[environment_type](env)
    _print(
        {
            "scenario": str(args.scenario),
            "task": reset.observation,
            "evaluation": evaluation.to_dict(),
            "trace": trace,
        }
    )


if __name__ == "__main__":
    main()
