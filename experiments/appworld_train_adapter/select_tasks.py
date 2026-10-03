"""从官方 AppWorld Train 中确定性选择五个复杂且不同模板的任务。

脚本只输出 task ID 和复杂度统计到被 Git 忽略的 artifacts；不导出 instruction、数据库、
参考解、API 调用序列或 evaluator 内容。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = ROOT.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--appworld-root",
        type=Path,
        default=WORKSPACE_ROOT / "data/external/AppWorld",
    )
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument(
        "--allow-same-generator",
        action="store_true",
        help="rank individual tasks instead of keeping one task per generator",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/appworld_train/selection.json",
    )
    args = parser.parse_args()
    appworld_root = args.appworld_root.resolve()
    output_path = (
        args.output.resolve()
        if args.output.is_absolute()
        else (ROOT / args.output).resolve()
    )
    os.environ["APPWORLD_ROOT"] = str(appworld_root)
    os.environ["APPWORLD_CACHE"] = str(appworld_root / "cache")
    os.chdir(appworld_root)

    from appworld import load_task_ids
    from appworld.task import Task

    records: list[dict[str, Any]] = []
    for order, task_id in enumerate(load_task_ids("train")):
        task = Task.load(task_id=task_id, ground_truth_mode="full")
        assert task.ground_truth is not None
        metadata = task.ground_truth.metadata
        records.append(
            {
                "task_id": task_id,
                "generator_id": task.generator_id,
                "official_order": order,
                "difficulty": int(metadata["difficulty"]),
                "num_apps": int(metadata["num_apps"]),
                "num_apis": int(metadata["num_apis"]),
                "num_solution_code_lines": int(metadata["num_solution_code_lines"]),
            }
        )
        task.close()
    records.sort(
        key=lambda item: (
            -item["difficulty"],
            -item["num_apps"],
            -item["num_apis"],
            -item["num_solution_code_lines"],
            item["official_order"],
        )
    )
    selected: list[dict[str, Any]] = []
    seen_generators: set[str] = set()
    for record in records:
        if (
            not args.allow_same_generator
            and record["generator_id"] in seen_generators
        ):
            continue
        selected.append(record)
        seen_generators.add(record["generator_id"])
        if len(selected) == args.count:
            break
    if len(selected) != args.count:
        raise RuntimeError(
            f"requested {args.count} distinct generators, found {len(selected)}"
        )
    version = (appworld_root / "data/version.txt").read_text(encoding="utf-8").strip()
    result = {
        "benchmark": "AppWorld",
        "data_version": version,
        "split": "train",
        "strategy": (
            "difficulty desc, num_apps desc, num_apis desc, solution lines desc; "
            + (
                "individual task ranking"
                if args.allow_same_generator
                else "one task per official generator"
            )
        ),
        "protected_content_exported": False,
        "tasks": selected,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"output": str(output_path), "count": len(selected), "tasks": selected},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
