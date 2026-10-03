"""定位并校验仓库中固定版本的 Agent0 上游代码。

Agent0 以 Git submodule 保存。本模块只把其 executor_train 加入导入路径并核对 commit，
不复制、不修改 PPO/ADPO、veRL、工具服务器或 rollout manager。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
AGENT0_SUBMODULE_ROOT = REPOSITORY_ROOT / "third_party/Agent0"
AGENT0_ROOT = AGENT0_SUBMODULE_ROOT / "Agent0"
EXECUTOR_TRAIN_ROOT = AGENT0_ROOT / "executor_train"
AGENT0_COMMIT = "f775b5101e62fe92976831adf4a21a38fcc0a767"


def ensure_agent0_executor_importable() -> Path:
    """将上游 executor_train 放入当前进程的模块搜索路径。"""

    expected = EXECUTOR_TRAIN_ROOT / "verl_tool/trainer/main_ppo.py"
    if not expected.is_file():
        raise FileNotFoundError(
            "Agent0 submodule is incomplete; run "
            "`git submodule update --init --recursive third_party/Agent0`."
        )
    value = str(EXECUTOR_TRAIN_ROOT)
    if value not in sys.path:
        sys.path.insert(0, value)
    return EXECUTOR_TRAIN_ROOT


def validate_agent0_checkout(*, require_pinned_commit: bool = True) -> dict[str, Any]:
    """返回上游位置与版本；默认拒绝悄然漂移的 submodule。"""

    ensure_agent0_executor_importable()
    revision = subprocess.check_output(
        ["git", "-C", str(AGENT0_SUBMODULE_ROOT), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if require_pinned_commit and revision != AGENT0_COMMIT:
        raise RuntimeError(
            f"Agent0 checkout is {revision}, expected pinned {AGENT0_COMMIT}"
        )
    return {
        "root": str(AGENT0_SUBMODULE_ROOT),
        "executor_train_root": str(EXECUTOR_TRAIN_ROOT),
        "revision": revision,
        "pinned": revision == AGENT0_COMMIT,
    }

