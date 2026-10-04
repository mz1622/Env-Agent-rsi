"""运行三个 AppWorld 失败任务的一步环境进化配对实验。

本脚本冻结本地 Qwen Target，在原始环境运行后调用 DeepSeek Diagnostic 与 Modifier，
物化一个白名单变化，再从干净的官方 AppWorld 状态重跑。完整本地记录与不含受保护
内容的摘要分开保存，便于复核真实 evaluator、诊断来源和实际环境配置差异。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Mapping

from env_agent_rsi.agent_system.diagnostic import DiagnosticAgent
from env_agent_rsi.agent_system.modifier import EnvironmentModificationAgent
from env_agent_rsi.agent_system.target import TargetAgent
from env_agent_rsi.evolution import (
    default_mutation_catalog,
    materialize_environment_spec,
)
from env_agent_rsi.harness.factory import build_environment


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TASKS = ("e85d92a_1", "e85d92a_2", "e85d92a_3")
DEFAULT_OUTPUT = ROOT / "artifacts/appworld_train/env_evolution_three_tasks.json"


def _base_spec(task_id: str) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "environment": {
            "type": "appworld_process",
            "parameters": {
                "experiment_name": f"env_agent_rsi_evolve_{task_id}",
                "max_interactions": 40,
                "request_timeout": 120,
            },
        },
        "task": {"id": task_id},
        "components": [],
    }


def _close_environment(env: Any) -> None:
    backend = getattr(getattr(env, "base", None), "backend", None)
    close = getattr(backend, "close", None)
    if callable(close):
        close()


def _episode_summary(episode: Any) -> dict[str, Any]:
    error_codes: list[str] = []
    tools: list[str] = []
    for item in episode.trace:
        action = item.get("action", {})
        response = item.get("response", {})
        tools.append(str(action.get("tool", "unknown")))
        observation = response.get("observation", {})
        error = observation.get("error", {}) if isinstance(observation, Mapping) else {}
        if isinstance(error, Mapping) and error.get("code"):
            error_codes.append(str(error["code"]))
    return {
        "success": episode.evaluation.success,
        "reason": episode.evaluation.reason,
        "metrics": dict(episode.evaluation.metrics),
        "stopped_reason": episode.stopped_reason,
        "steps": episode.steps,
        "tool_sequence": tools,
        "error_codes": error_codes,
    }


def _run_target(target: TargetAgent, spec: Mapping[str, Any]) -> tuple[Any, Any]:
    env = build_environment(spec)
    try:
        episode = target.run(env, seed=0)
        surface = env.base.mutation_surface()
        return episode, surface
    finally:
        _close_environment(env)


def run(tasks: tuple[str, ...], output: Path) -> dict[str, Any]:
    target = TargetAgent.from_config(ROOT / "configs/agents/target_qwen3_4b.json")
    diagnostic = DiagnosticAgent.from_config(
        ROOT / "configs/agents/diagnostic_agent.json"
    )
    modifier = EnvironmentModificationAgent.from_config(
        ROOT / "configs/agents/environment_modifier_agent.json"
    )
    catalog = default_mutation_catalog()
    full_records: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    for index, task_id in enumerate(tasks, start=1):
        print(f"[{index}/{len(tasks)}] {task_id}: baseline target", flush=True)
        started = time.monotonic()
        parent_spec = _base_spec(task_id)
        baseline, surface = _run_target(target, parent_spec)
        baseline_summary = _episode_summary(baseline)
        print(
            f"[{index}/{len(tasks)}] {task_id}: baseline "
            f"{baseline_summary['metrics']} ({baseline_summary['steps']} steps)",
            flush=True,
        )

        print(f"[{index}/{len(tasks)}] {task_id}: DeepSeek diagnosis", flush=True)
        diagnosis = diagnostic.diagnose(
            baseline,
            environment_spec=parent_spec,
            surface=surface,
            catalog=catalog,
        )
        mutation = None
        candidate_spec = None
        candidate = None
        if diagnosis.environment_actionable:
            print(f"[{index}/{len(tasks)}] {task_id}: DeepSeek modifier", flush=True)
            mutation = modifier.propose(
                diagnosis,
                environment_spec=parent_spec,
                surface=surface,
                catalog=catalog,
            )
            candidate_spec = materialize_environment_spec(
                parent_spec,
                mutation,
                surface=surface,
                catalog=catalog,
            )
            print(f"[{index}/{len(tasks)}] {task_id}: modified target", flush=True)
            candidate, _ = _run_target(target, candidate_spec)

        candidate_summary = _episode_summary(candidate) if candidate else None
        summary = {
            "task_id": task_id,
            "baseline": baseline_summary,
            "diagnosis": {
                "source": diagnostic.last_diagnosis_source,
                "category": diagnosis.category,
                "phase": diagnosis.phase,
                "summary": diagnosis.summary,
                "confidence": diagnosis.confidence,
                "environment_actionable": diagnosis.environment_actionable,
                "priority_reason": diagnosis.priority_reason,
            },
            "modifier": (
                {
                    "source": modifier.last_proposal_source,
                    "mutation": mutation.to_dict(),
                }
                if mutation
                else {"source": "modifier_skipped", "mutation": None}
            ),
            "candidate": candidate_summary,
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
        summaries.append(summary)
        full_records.append(
            {
                **summary,
                "parent_environment_spec": parent_spec,
                "candidate_environment_spec": candidate_spec,
                "baseline_episode": baseline.to_dict(),
                "candidate_episode": candidate.to_dict() if candidate else None,
                "diagnosis_full": diagnosis.to_dict(),
            }
        )
        print(
            f"[{index}/{len(tasks)}] {task_id}: done; "
            f"candidate={candidate_summary and candidate_summary['metrics']}",
            flush=True,
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "benchmark": "AppWorld Train",
        "target": "qwen3:4b-direct (same local Qwen3 4B weights)",
        "diagnostic_model": "deepseek-flash",
        "modifier_model": "deepseek-flash",
        "seed": 0,
        "protected_content_exported": True,
        "records": full_records,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path = output.with_name(output.stem + "_summary.json")
    summary_path.write_text(
        json.dumps(
            {
                "benchmark": "AppWorld Train",
                "protected_content_exported": False,
                "records": summaries,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"full": str(output), "summary": str(summary_path), "records": summaries}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("task_ids", nargs="*", default=list(DEFAULT_TASKS))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run(tuple(args.task_ids), args.output.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
