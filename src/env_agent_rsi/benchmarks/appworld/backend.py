"""把独立 Python 3.12 AppWorld 运行时接到 BenchmarkBackend。

主项目不导入 AppWorld 及其大量依赖，而是维持一个 JSON Lines 子进程。子进程持有唯一
官方 AppWorld 实例；任务 instruction、数据库变化、checkpoint 和 evaluator 都来自官方包。
"""

from __future__ import annotations

import json
import os
import selectors
import subprocess
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from env_agent_rsi.benchmarks.adapter import BenchmarkTask
from env_agent_rsi.core.protocol import EvaluationResult, JsonObject
from env_agent_rsi.core.tooling import tool_schema
from env_agent_rsi.evolution.surface import (
    MutationSurface,
    ObservationChannel,
    ToolSemantics,
)


APPWORLD_TOOLS: tuple[JsonObject, ...] = (
    tool_schema(
        "get_api_docs",
        (
            "Read official AppWorld API documentation. With only app_name, list that "
            "app's public APIs; with api_name, return the complete API contract."
        ),
        {
            "app_name": {"type": "string"},
            "api_name": {"type": "string"},
        },
        ["app_name"],
    ),
    tool_schema(
        "execute_python",
        (
            "Execute one stateful Python cell in the official AppWorld shell. Call app "
            "APIs as apis.<app>.<api>(...). Print values needed for the next turn."
        ),
        {"code": {"type": "string"}},
        ["code"],
    ),
    tool_schema(
        "finish",
        (
            "Call the official supervisor.complete_task API. Supply answer only for "
            "answer-seeking tasks."
        ),
        {"answer": {"type": "string"}},
        [],
    ),
)


class AppWorldProcessBackend:
    """通过长驻子进程执行一个固定的官方 AppWorld 任务。"""

    def __init__(
        self,
        *,
        task_id: str,
        appworld_root: str | Path | None = None,
        python_executable: str | Path | None = None,
        experiment_name: str = "env_agent_rsi",
        max_interactions: int = 40,
        request_timeout: float = 120.0,
    ) -> None:
        if not task_id:
            raise ValueError("AppWorld task_id must not be empty")
        if max_interactions <= 0:
            raise ValueError("max_interactions must be positive")
        self.task_id = task_id
        self.appworld_root, self.python_executable = _resolve_runtime(
            appworld_root, python_executable
        )
        self.experiment_name = experiment_name
        self.max_interactions = max_interactions
        self.request_timeout = request_timeout
        self._request_number = 0
        self._process = self._start_worker()
        self._descriptor = self._request("reset", seed=0, options={})

    def _start_worker(self) -> subprocess.Popen[str]:
        src_root = Path(__file__).resolve().parents[3]
        environment = dict(os.environ)
        existing = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            str(src_root) if not existing else os.pathsep.join((str(src_root), existing))
        )
        environment["APPWORLD_ROOT"] = str(self.appworld_root)
        environment["APPWORLD_CACHE"] = str(self.appworld_root / "cache")
        command = [
            str(self.python_executable),
            "-m",
            "env_agent_rsi.benchmarks.appworld.worker",
            "--root",
            str(self.appworld_root),
            "--task-id",
            self.task_id,
            "--experiment-name",
            self.experiment_name,
            "--max-interactions",
            str(self.max_interactions),
        ]
        return subprocess.Popen(
            command,
            cwd=self.appworld_root,
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

    def _request(self, command: str, **payload: Any) -> JsonObject:
        process = self._process
        if process.poll() is not None:
            raise RuntimeError(self._worker_failure("AppWorld worker is not running"))
        assert process.stdin is not None and process.stdout is not None
        self._request_number += 1
        request_id = f"request-{self._request_number}"
        message = {"id": request_id, "command": command, **payload}
        process.stdin.write(json.dumps(message, ensure_ascii=False) + "\n")
        process.stdin.flush()

        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        try:
            ready = selector.select(self.request_timeout)
        finally:
            selector.close()
        if not ready:
            raise TimeoutError(
                f"AppWorld worker did not answer {command!r} within "
                f"{self.request_timeout:g} seconds"
            )
        line = process.stdout.readline()
        if not line:
            raise RuntimeError(self._worker_failure("AppWorld worker closed stdout"))
        response = json.loads(line)
        if response.get("id") != request_id:
            raise RuntimeError(
                f"AppWorld worker protocol mismatch: expected {request_id!r}, "
                f"got {response.get('id')!r}"
            )
        if not response.get("ok"):
            error = response.get("error", {})
            raise RuntimeError(
                f"AppWorld worker {error.get('type', 'Error')}: "
                f"{error.get('message', 'unknown failure')}"
            )
        result = response.get("result", {})
        if not isinstance(result, Mapping):
            raise RuntimeError("AppWorld worker result must be an object")
        return dict(result)

    def _worker_failure(self, prefix: str) -> str:
        process = self._process
        detail = ""
        if process.stderr is not None and process.poll() is not None:
            detail = process.stderr.read().strip()
        return prefix if not detail else f"{prefix}: {detail[-4000:]}"

    def task(self) -> BenchmarkTask:
        return BenchmarkTask(
            task_id=self.task_id,
            instruction=str(self._descriptor["instruction"]),
            tools=APPWORLD_TOOLS,
            metadata={
                "benchmark": "AppWorld",
                "split": "train",
                "data_version": self._descriptor.get("data_version"),
                "available_apps": self._descriptor.get("available_apps", []),
                "runtime": "isolated-python-process",
            },
        )

    def reset(self, seed: int, options: Mapping[str, Any]) -> Mapping[str, Any]:
        self._descriptor = self._request(
            "reset", seed=seed, options=dict(options)
        )
        return dict(self._descriptor["observation"])

    def step(self, tool: str, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        return self._request("step", tool=tool, arguments=dict(arguments))

    def observe(self) -> Mapping[str, Any]:
        return self._request("observe")

    def evaluate(self) -> EvaluationResult:
        value = self._request("evaluate")
        passed = int(value["passed"])
        total = int(value["total"])
        success = bool(value["success"])
        return EvaluationResult(
            success=success,
            reason=f"official AppWorld evaluator passed {passed}/{total} requirements",
            metrics={
                "passed": passed,
                "failed": int(value["failed"]),
                "total": total,
                "difficulty": value.get("difficulty"),
            },
        )

    def save_state(self) -> Mapping[str, Any]:
        state_id = "env-agent-rsi-" + uuid.uuid4().hex
        value = self._request("save_state", state_id=state_id)
        return {
            "task_id": self.task_id,
            "state_id": value["state_id"],
            "snapshot_scope": "official_database_only",
            "warning": (
                "AppWorld checkpoints restore application databases but not arbitrary "
                "Python shell variables or Agent context."
            ),
        }

    def load_state(self, snapshot: Mapping[str, Any]) -> None:
        if snapshot.get("task_id") != self.task_id:
            raise ValueError("AppWorld snapshot task_id does not match this backend")
        if snapshot.get("snapshot_scope") != "official_database_only":
            raise ValueError("unsupported AppWorld snapshot scope")
        self._request("load_state", state_id=str(snapshot["state_id"]))

    def mutation_surface(self) -> MutationSurface:
        """首版只开放能由通用 Harness 可靠验证的四个执行槽。"""

        return MutationSurface(
            tools=(
                ToolSemantics("get_api_docs", "read", setup_allowed=True),
                ToolSemantics("execute_python", "world_state", setup_allowed=True),
                ToolSemantics("finish", "terminal", setup_allowed=False),
            ),
            observations=(
                ObservationChannel("output", mutable=True),
                ObservationChannel("available_apps", mutable=False),
            ),
            supported_phases=("setup", "contract", "action", "budget"),
            supported_components=("stage", "contract", "extension"),
            supported_contract_axes=("f_A",),
            supported_implementations={
                "setup": ("replay",),
                "contract": ("require_argument",),
                "action": ("require_argument",),
                "budget": ("step_budget",),
            },
            budget_dimensions=("steps",),
            metadata={
                "benchmark": "AppWorld",
                "snapshot_scope": "official_database_only",
                "deferred_native_phases": ["transition", "observation"],
            },
        )

    def close(self) -> None:
        process = getattr(self, "_process", None)
        if process is None or process.poll() is not None:
            return
        try:
            self._request("close")
        except (BrokenPipeError, RuntimeError, TimeoutError):
            process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)

    def __enter__(self) -> "AppWorldProcessBackend":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def _resolve_runtime(
    appworld_root: str | Path | None,
    python_executable: str | Path | None,
) -> tuple[Path, Path]:
    """按显式参数、环境变量、工作区约定顺序解析外部运行时。"""

    repository_root = Path(__file__).resolve().parents[4]
    workspace_root = repository_root.parent
    root = Path(
        appworld_root
        or os.environ.get("ENV_AGENT_RSI_APPWORLD_ROOT", "")
        or workspace_root / "data/external/AppWorld"
    ).expanduser().resolve()
    # 不解析 venv 中的 python 符号链接；解析后会绕过该 venv 的 site-packages。
    python = Path(
        python_executable
        or os.environ.get("ENV_AGENT_RSI_APPWORLD_PYTHON", "")
        or workspace_root / ".venv-appworld/bin/python"
    ).expanduser().absolute()
    if not (root / "data/datasets/train.txt").is_file():
        raise FileNotFoundError(
            f"AppWorld Train data not found under {root}; run scripts/setup_appworld.py"
        )
    if not python.is_file():
        raise FileNotFoundError(
            f"AppWorld Python runtime not found at {python}; run scripts/setup_appworld.py"
        )
    return root, python
