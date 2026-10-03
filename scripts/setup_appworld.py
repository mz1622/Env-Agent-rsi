"""幂等准备固定版本的官方 AppWorld 源码、独立环境和 Train 数据。

脚本拒绝覆盖已有的不同版本数据，也不把解包后的受保护内容复制进 Env-Agent-rsi；默认
布局与研究工作区一致：external/appworld、.venv-appworld、data/external/AppWorld。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path


SOURCE_URL = "https://github.com/StonyBrookNLP/appworld.git"
SOURCE_REVISION = "42b5bcf3cd334fee33f0c37c02070a9f5807add5"
DATA_VERSION = "0.2.0"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPOSITORY_ROOT.parent


def _run(arguments: list[str], *, cwd: Path, environment: dict[str, str] | None = None) -> None:
    subprocess.run(arguments, cwd=cwd, env=environment, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root", type=Path, default=WORKSPACE_ROOT / "external/appworld"
    )
    parser.add_argument(
        "--data-root", type=Path, default=WORKSPACE_ROOT / "data/external/AppWorld"
    )
    parser.add_argument(
        "--venv-root", type=Path, default=WORKSPACE_ROOT / ".venv-appworld"
    )
    parser.add_argument("--verify-tasks", type=int, default=4)
    args = parser.parse_args()
    source_root = args.source_root.resolve()
    data_root = args.data_root.resolve()
    venv_root = args.venv_root.resolve()

    if not source_root.exists():
        source_root.parent.mkdir(parents=True, exist_ok=True)
        _run(
            ["git", "clone", "--filter=blob:none", SOURCE_URL, str(source_root)],
            cwd=source_root.parent,
        )
        _run(["git", "checkout", "--detach", SOURCE_REVISION], cwd=source_root)
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source_root, text=True
    ).strip()
    if revision != SOURCE_REVISION:
        raise RuntimeError(
            f"existing AppWorld revision is {revision}; expected {SOURCE_REVISION}. "
            "Refusing to change an existing checkout automatically."
        )

    python = venv_root / "bin/python"
    if not python.exists():
        uv = shutil.which("uv")
        if uv is None:
            raise RuntimeError("uv is required to create the isolated AppWorld runtime")
        _run([uv, "venv", "--python", "3.12", str(venv_root)], cwd=WORKSPACE_ROOT)
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is required to install AppWorld dependencies")
    _run(
        [uv, "pip", "install", "--python", str(python), "-e", str(source_root)],
        cwd=WORKSPACE_ROOT,
    )

    cli = venv_root / "bin/appworld"
    data_root.mkdir(parents=True, exist_ok=True)
    environment = {
        **os.environ,
        "APPWORLD_ROOT": str(data_root),
        "APPWORLD_CACHE": str(data_root / "cache"),
    }
    _run([str(cli), "install", "--repo"], cwd=source_root, environment=environment)
    version_file = data_root / "data/version.txt"
    if version_file.exists():
        existing_version = version_file.read_text(encoding="utf-8").strip()
        if existing_version != DATA_VERSION:
            raise RuntimeError(
                f"existing AppWorld data is {existing_version}; expected {DATA_VERSION}. "
                "Refusing to replace it automatically."
            )
    else:
        _run(
            [
                str(cli),
                "download",
                "data",
                "--version",
                DATA_VERSION,
                "--mode",
                "minimal",
                "--root",
                str(data_root),
            ],
            cwd=data_root,
            environment=environment,
        )
    if args.verify_tasks:
        _run(
            [
                str(cli),
                "verify",
                "tasks",
                "--root",
                str(data_root),
                "--num-processes",
                "1",
                "--include-only-first-n-tasks",
                str(args.verify_tasks),
            ],
            cwd=data_root,
            environment=environment,
        )
    counts = {
        path.stem: len(
            [line for line in path.read_text(encoding="utf-8").splitlines() if line]
        )
        for path in (data_root / "data/datasets").glob("*.txt")
    }
    print(
        json.dumps(
            {
                "source_root": str(source_root),
                "source_revision": revision,
                "data_root": str(data_root),
                "data_version": DATA_VERSION,
                "python": str(python),
                "split_counts": counts,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
