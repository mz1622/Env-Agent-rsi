from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from env_agent_rsi.agents import run_oracle_agent
from env_agent_rsi.core.protocol import Action
from env_agent_rsi.harness.factory import build_environment, load_spec
from env_agent_rsi.scenario_agents import SCENARIO_ORACLES


def _print(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True), flush=True)


def _run_manual(env) -> None:
    """JSON-lines bridge usable by any external model/tool-calling loop."""

    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            action = Action(str(payload["tool"]), dict(payload.get("arguments", {})))
            response = env.step(action)
            _print(response.to_dict())
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one catalog scenario")
    parser.add_argument("scenario", type=Path)
    parser.add_argument("--agent", choices=("oracle", "manual"), default="oracle")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    spec = load_spec(args.scenario)
    env = build_environment(spec)
    reset = env.reset(seed=args.seed)
    environment_type = str(spec.get("environment", {}).get("type", "item"))
    if args.agent == "manual":
        _print(reset.to_dict())
        _run_manual(env)
        return
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
