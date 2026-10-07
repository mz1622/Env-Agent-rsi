#!/usr/bin/env python3
"""在 ALFWorld 任务上运行 EnvRigger + 外部 Uniform-PUCT 环境搜索。

每个 task seed 对应一个固定的 ALFWorld train game。根环境与每个被选中的
子环境都使用同一个 seed 做 K 次 rollout，从而让 MCTS 比较环境变化，而不是
混入不同任务的难度差异。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_harness import build_from_config  # noqa: E402
from envharness.orchestration.tree_runtime import (  # noqa: E402
    OrchestratorEvolutionRuntime,
)
from envharness.orchestration.tree_search import (  # noqa: E402
    ExternalSearchCoordinator,
    UniformPUCTPolicy,
)
from envharness.core.types import Trace  # noqa: E402


def run_task(
    *,
    seed: int,
    run_root: Path,
    max_expansions: int,
    rollouts: int,
    rollout_concurrency: int,
    c_puct: float,
    game_file: str | None = None,
    timeout_seconds: float = 900.0,
    repetition_threshold: int = 3,
    root_traces: list[Trace] | None = None,
) -> dict:
    """运行一棵环境树并保存所有节点、轨迹和模型调用。"""
    task_name = f"task-{seed}"
    run_name = f"{run_root.name}/{task_name}"
    orchestrator = build_from_config(
        REPO_ROOT / "experiments/alfworld/corpus_smoke.yaml",
        run_name=run_name,
        overrides={
            "n_tasks": 1,
            "n_iterations": 1,
            "explicit_task_ids": [seed],
            "k_per_candidate": rollouts,
            "rollout_concurrency": rollout_concurrency,
            "compute_baseline": False,
            "skip_passthrough_candidates": False,
        },
    )
    if game_file:
        # ALFWorld 的 seed 会先打乱整个 split；实验复现时可直接固定论文
        # 数据中的 game file，同时保留整数 seed 作为 rollout 标识。
        orchestrator.env_spec.reset_options["task_id"] = game_file
    orchestrator.env_spec.reset_options["repetition_threshold"] = int(
        repetition_threshold
    )
    if hasattr(orchestrator.runner, "timeout"):
        orchestrator.runner.timeout = float(timeout_seconds)
    output_dir = REPO_ROOT / "runs" / run_name
    runtime = OrchestratorEvolutionRuntime(
        orchestrator=orchestrator,
        task_idx=0,
        output_dir=output_dir,
        root_traces=root_traces,
    )
    started = time.time()
    result = ExternalSearchCoordinator(
        runtime=runtime,
        external_policy=UniformPUCTPolicy(c_puct=c_puct),
    ).run(max_expansions=max_expansions)

    summary = {
        "task_seed": seed,
        "game_file": game_file,
        "max_expansions": max_expansions,
        "rollouts_per_node": rollouts,
        "rollout_concurrency": rollout_concurrency,
        "timeout_seconds": timeout_seconds,
        "repetition_threshold": repetition_threshold,
        "reused_root_rollouts": bool(root_traces),
        "actual_expansions": result.expansion_count,
        "best_node_id": result.best_node_id,
        "stop_rationale": result.stop_rationale,
        "elapsed_seconds": round(time.time() - started, 3),
        "nodes": [
            {
                "node_id": node.node_id,
                "parent_node_id": node.parent_node_id,
                "incoming_proposal_id": node.incoming_proposal_id,
                "depth": node.depth,
                "change_type": node.candidate.change_type,
                "success_rate": node.validation.success_rate,
                "n_success": node.validation.n_success,
                "n_rollouts": node.validation.n_rollouts,
                "n_errors": node.validation.n_errors,
            }
            for node in result.snapshot.nodes
        ],
    }
    (output_dir / "mcts_result.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--max-expansions", type=int, default=10)
    parser.add_argument("--rollouts", type=int, default=4)
    parser.add_argument("--rollout-concurrency", type=int, default=1)
    parser.add_argument("--c-puct", type=float, default=1.0)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--repetition-threshold", type=int, default=3)
    parser.add_argument(
        "--resume-root-traces",
        default=None,
        help="Reuse exactly K validated root traces from a previous JSONL run.",
    )
    parser.add_argument(
        "--game-file",
        default=None,
        help="Pin one ALFWorld game-file path or unique suffix.",
    )
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_name = args.run_name or f"alfworld-mcts-{stamp}"
    run_root = REPO_ROOT / "runs" / run_name
    run_root.mkdir(parents=True, exist_ok=True)

    results = []
    root_traces = None
    if args.resume_root_traces:
        trace_path = Path(args.resume_root_traces)
        root_traces = [
            Trace.model_validate_json(line)
            for line in trace_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        root_traces = [trace for trace in root_traces
                       if trace.candidate_id == "root"]
    for seed in args.seeds:
        print(f"[mcts] task_seed={seed} starting", flush=True)
        try:
            result = run_task(
                seed=seed,
                run_root=run_root,
                max_expansions=args.max_expansions,
                rollouts=args.rollouts,
                rollout_concurrency=args.rollout_concurrency,
                c_puct=args.c_puct,
                game_file=args.game_file,
                timeout_seconds=args.timeout_seconds,
                repetition_threshold=args.repetition_threshold,
                root_traces=root_traces,
            )
            results.append(result)
            print(
                f"[mcts] task_seed={seed} expansions="
                f"{result['actual_expansions']} best={result['best_node_id']}",
                flush=True,
            )
        except Exception as exc:
            failure = {
                "task_seed": seed,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            results.append(failure)
            print(f"[mcts] task_seed={seed} failed: {failure}", flush=True)

    aggregate = {
        "run_name": run_name,
        "max_expansions": args.max_expansions,
        "rollouts_per_node": args.rollouts,
        "rollout_concurrency": args.rollout_concurrency,
        "tasks": results,
    }
    (run_root / "summary.json").write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(aggregate, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
