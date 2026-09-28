"""Exactly-once 微环境的旧版演示入口。

该命令只比较 naive 与 scripted oracle，便于快速展示 post-commit timeout 的重复写入
风险；统一场景、外部 Agent 和 replay 应使用 ``run_task.py``。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from env_agent_rsi.agents import AGENTS
from env_agent_rsi.harness.factory import build_environment, load_spec


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the minimal environment MVP")
    parser.add_argument(
        "--scenario",
        type=Path,
        default=Path("configs/micro_api/postcommit_stale.json"),
    )
    parser.add_argument("--agent", choices=sorted(AGENTS), default="oracle")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    spec = load_spec(args.scenario)
    env = build_environment(spec)
    target = str(spec.get("task", {}).get("target_value", "target-item"))
    env.reset(seed=args.seed)
    evaluation, trace = AGENTS[args.agent](env, target)
    print(
        json.dumps(
            {
                "scenario": str(args.scenario),
                "agent": args.agent,
                "evaluation": evaluation.to_dict(),
                "trace": trace,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
