"""验证官方 AppWorld Train 到本项目 BenchmarkAdapter 的最小真实链路。

该脚本只做只读文档查询、打印常量、数据库 checkpoint 和官方评分，不运行模型，不修改
任务定义，也不把官方受保护内容复制到报告。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from env_agent_rsi.core.protocol import Action
from env_agent_rsi.harness.factory import build_environment


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selection",
        type=Path,
        default=ROOT / "artifacts/appworld_train/selection.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/appworld_train/adapter_smoke.json",
    )
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    task_id = str(selection["tasks"][0]["task_id"])

    environment = build_environment(
        {
            "schema_version": 2,
            "environment": {
                "type": "appworld_process",
                "parameters": {
                    "experiment_name": "env_agent_rsi_adapter_smoke",
                    "max_interactions": 8,
                },
            },
            "task": {"id": task_id},
            "components": [],
        }
    )
    try:
        reset = environment.reset(seed=0)
        descriptor = environment.describe()
        docs = environment.step(Action("get_api_docs", {"app_name": "api_docs"}))
        execution = environment.step(
            Action("execute_python", {"code": "print(1 + 2)"})
        )
        snapshot = environment.save_state()
        environment.load_state(snapshot)
        evaluation = environment.evaluate()
        checks = {
            "instruction_loaded": bool(descriptor.instruction),
            "tools_loaded": {
                item["function"]["name"] for item in descriptor.tools
            }
            == {"get_api_docs", "execute_python", "finish"},
            "initial_observation_loaded": bool(
                reset.observation.get("available_apps")
            ),
            "api_docs_readable": bool(docs.observation.get("ok")),
            "stateful_shell_executed": execution.observation.get("output", "").strip()
            == "3",
            "checkpoint_scope_explicit": snapshot["base"].get("snapshot_scope")
            == "official_database_only",
            "official_evaluator_ran": evaluation.metrics.get("total", 0) > 0,
        }
        result = {
            "task_id": task_id,
            "success": all(checks.values()),
            "checks": checks,
            "evaluation_expected_to_fail_without_task_actions": {
                "success": evaluation.success,
                "passed": evaluation.metrics["passed"],
                "total": evaluation.metrics["total"],
            },
        }
    finally:
        environment.base.backend.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output), **result}, ensure_ascii=False, indent=2))
    if not result["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
