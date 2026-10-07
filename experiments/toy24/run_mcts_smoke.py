#!/usr/bin/env python3
"""Run several Toy24 tasks through EnvRigger + external uniform-PUCT MCTS."""
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


def run_task(*, seed: int, run_root: Path, max_expansions: int,
             rollouts: int, c_puct: float) -> dict:
    task_name = f"task-{seed}"
    run_name = f"{run_root.name}/{task_name}"
    orchestrator = build_from_config(
        REPO_ROOT / "experiments/toy24/mutated_smoke.yaml",
        run_name=run_name,
        overrides={
            "n_tasks": 1,
            "n_iterations": 1,
            "explicit_task_ids": [seed],
            "k_per_candidate": rollouts,
            "rollout_concurrency": 1,
            "compute_baseline": False,
            "skip_passthrough_candidates": False,
        },
    )
    output_dir = REPO_ROOT / "runs" / run_name
    runtime = OrchestratorEvolutionRuntime(
        orchestrator=orchestrator,
        task_idx=0,
        output_dir=output_dir,
    )
    policy = UniformPUCTPolicy(c_puct=c_puct)
    started = time.time()
    result = ExternalSearchCoordinator(
        runtime=runtime,
        external_policy=policy,
    ).run(max_expansions=max_expansions)

    nodes = result.snapshot.nodes
    summary = {
        "task_seed": seed,
        "max_expansions": max_expansions,
        "rollouts_per_node": rollouts,
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
            for node in nodes
        ],
    }
    (output_dir / "mcts_result.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--max-expansions", type=int, default=10)
    parser.add_argument("--rollouts", type=int, default=4)
    parser.add_argument("--c-puct", type=float, default=1.0)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_name = args.run_name or f"toy24-mcts-{stamp}"
    run_root = REPO_ROOT / "runs" / run_name
    run_root.mkdir(parents=True, exist_ok=True)

    results = []
    for seed in args.seeds:
        print(f"[mcts] task_seed={seed} starting", flush=True)
        try:
            result = run_task(
                seed=seed,
                run_root=run_root,
                max_expansions=args.max_expansions,
                rollouts=args.rollouts,
                c_puct=args.c_puct,
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
        "tasks": results,
    }
    (run_root / "summary.json").write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(aggregate, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
