"""将本地 AppWorld task 转为 Agent0/veRL 的 parquet 记录。

任务正文只从已安装的官方 AppWorld bundle 读取，并写入被 Git 忽略的 artifacts；仓库不
重新分发 instruction、数据库、ground truth 或 evaluator。输出字段沿用 Agent0 的
VerlToolRLHFDataset，而不是定义新的训练数据类。
"""

from __future__ import annotations

import argparse
import json
import secrets
from pathlib import Path
from typing import Any, Iterable, Mapping

from env_agent_rsi.agent_system.prompts import load_prompt, load_skill
from env_agent_rsi.agent_runtime.agent0_protocol import render_tool_register
from env_agent_rsi.benchmarks.appworld import APPWORLD_TOOLS, AppWorldProcessBackend


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
WORKSPACE_ROOT = REPOSITORY_ROOT.parent
DEFAULT_SELECTION = REPOSITORY_ROOT / "artifacts/appworld_train/selection.json"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "artifacts/appworld_train/agent0"
DEFAULT_PROMPT = REPOSITORY_ROOT / "configs/prompts/target_system.json"
DEFAULT_SKILLS = (REPOSITORY_ROOT / "configs/skills/appworld_interactive_code.json",)


def build_agent0_system_prompt(
    prompt_path: str | Path = DEFAULT_PROMPT,
    skill_paths: Iterable[str | Path] = DEFAULT_SKILLS,
) -> str:
    """从现有 JSON prompt/skill 与动态 AppWorld schema 构造 Agent0 system 消息。"""

    prompt = load_prompt(prompt_path)
    skills = [load_skill(path).render() for path in skill_paths]
    return "\n\n".join(
        (
            prompt.content.strip(),
            "[Skill Register]\n" + "\n\n".join(skills),
            render_tool_register(APPWORLD_TOOLS),
            "After the environment reports official evaluation, end with a normal "
            "response and no tool call.",
        )
    )


def build_agent0_record(
    *,
    task_id: str,
    instruction: str,
    split: str,
    appworld_root: str | Path,
    python_executable: str | Path,
    score: float = 0.5,
    reward_key: str | None = None,
    max_interactions: int = 40,
    system_prompt: str | None = None,
) -> dict[str, Any]:
    """构造一条 Agent0 VerlToolRLHFDataset 兼容记录。"""

    return {
        "data_source": "appworld",
        "ability": "stateful_tool_use",
        "prompt": [
            {
                "role": "system",
                "content": system_prompt or build_agent0_system_prompt(),
            },
            {"role": "user", "content": instruction},
        ],
        "reward_model": {
            "style": "rule",
            "ground_truth": {"task_id": task_id},
        },
        "extra_info": {
            "id": task_id,
            "task_id": task_id,
            "split": split,
            "score": float(score),
            "reward_key": reward_key or secrets.token_hex(32),
            "appworld_root": str(Path(appworld_root).expanduser().resolve()),
            "python_executable": str(Path(python_executable).expanduser().absolute()),
            "experiment_name": "agent0_appworld",
            "max_interactions": int(max_interactions),
        },
    }


def _load_task_ids(path: Path, limit: int | None) -> list[str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    tasks = value.get("tasks", [])
    task_ids = [str(item["task_id"]) for item in tasks]
    return task_ids if limit is None else task_ids[:limit]


def build_records(
    task_ids: Iterable[str],
    *,
    appworld_root: Path,
    python_executable: Path,
) -> list[dict[str, Any]]:
    """通过现有 AppWorldProcessBackend 读取公开给 Agent 的 task instruction。"""

    records: list[dict[str, Any]] = []
    system_prompt = build_agent0_system_prompt()
    for task_id in task_ids:
        backend = AppWorldProcessBackend(
            task_id=task_id,
            appworld_root=appworld_root,
            python_executable=python_executable,
            experiment_name="agent0_dataset_export",
        )
        try:
            task = backend.task()
            records.append(
                build_agent0_record(
                    task_id=task.task_id,
                    instruction=task.instruction,
                    split="train",
                    appworld_root=appworld_root,
                    python_executable=python_executable,
                    system_prompt=system_prompt,
                )
            )
        finally:
            backend.close()
    return records


def write_parquet_splits(
    records: list[Mapping[str, Any]], output: Path, *, val_count: int = 1
) -> dict[str, Any]:
    """使用 Agent0 已依赖的 Hugging Face datasets 写 train/validation parquet。"""

    if len(records) <= val_count:
        raise ValueError(
            "records must contain at least one train item beyond val_count"
        )
    from datasets import Dataset

    output.mkdir(parents=True, exist_ok=True)
    train = [dict(value) for value in records[:-val_count]]
    validation = [dict(value) for value in records[-val_count:]]
    for item in validation:
        item["extra_info"] = {**item["extra_info"], "split": "validation"}
    train_path = output / "train.parquet"
    validation_path = output / "validation.parquet"
    Dataset.from_list(train).to_parquet(str(train_path))
    Dataset.from_list(validation).to_parquet(str(validation_path))
    manifest = {
        "benchmark": "AppWorld",
        "purpose": "Agent0 integration smoke split; not final paper evaluation",
        "train_path": str(train_path),
        "validation_path": str(validation_path),
        "train_task_ids": [item["extra_info"]["task_id"] for item in train],
        "validation_task_ids": [item["extra_info"]["task_id"] for item in validation],
        "protected_content_committed": False,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare AppWorld data for Agent0")
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--appworld-root", default=WORKSPACE_ROOT / "data/external/AppWorld", type=Path
    )
    parser.add_argument(
        "--python-executable",
        default=WORKSPACE_ROOT / ".venv-appworld/bin/python",
        type=Path,
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--val-count", type=int, default=1)
    args = parser.parse_args()
    task_ids = _load_task_ids(args.selection.resolve(), args.limit)
    records = build_records(
        task_ids,
        appworld_root=args.appworld_root.resolve(),
        python_executable=args.python_executable.absolute(),
    )
    manifest = write_parquet_splits(
        records, args.output.resolve(), val_count=args.val_count
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
