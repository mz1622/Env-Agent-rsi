"""AppWorld 依赖环境中的长驻 JSON Lines worker。

该模块只向父进程返回 Agent 可见结果或聚合评分，不返回 evaluator 细节、参考解或数据库
内容。所有官方 AppWorld import 都在设置独立 APPWORLD_ROOT 后延迟发生。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping


class AppWorldWorker:
    """持有一个 task world，并处理父进程的顺序请求。"""

    def __init__(
        self,
        *,
        root: Path,
        task_id: str,
        experiment_name: str,
        max_interactions: int,
    ) -> None:
        os.environ["APPWORLD_ROOT"] = str(root)
        os.environ["APPWORLD_CACHE"] = str(root / "cache")
        os.chdir(root)
        from appworld import AppWorld

        self.AppWorld = AppWorld
        self.root = root
        self.task_id = task_id
        self.experiment_name = experiment_name
        self.max_interactions = max_interactions
        self.world: Any = None
        self.last_observation: dict[str, Any] = {}

    def reset(self, seed: int, options: Mapping[str, Any]) -> dict[str, Any]:
        del options
        if self.world is not None:
            self.AppWorld.close_all()
        self.world = self.AppWorld(
            task_id=self.task_id,
            experiment_name=self.experiment_name,
            max_interactions=self.max_interactions,
            random_seed=seed,
            ground_truth_mode="minimal",
            raise_on_failure=False,
            raise_on_unsafe_syntax=True,
        )
        self.last_observation = {
            "ok": True,
            "available_apps": sorted(self.world.task.app_descriptions),
            "app_descriptions": dict(self.world.task.app_descriptions),
            "interaction_count": 0,
            "completed": False,
            "guidance": (
                "Use get_api_docs before calling unfamiliar APIs. In execute_python, "
                "call apis.<app>.<api>(...) and print information needed later."
            ),
        }
        version_path = self.root / "data/version.txt"
        return {
            "instruction": self.world.task.instruction,
            "available_apps": sorted(self.world.task.app_descriptions),
            "data_version": version_path.read_text(encoding="utf-8").strip(),
            "observation": self.last_observation,
        }

    def step(self, tool: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        world = self._world()
        if tool == "get_api_docs":
            observation = self._get_api_docs(arguments)
            return {"observation": observation, "info": {"event": "api_docs_read"}}
        if tool == "execute_python":
            code = str(arguments["code"])
            output = world.execute(code)
            completed = bool(world.task_completed())
            observation = {
                "ok": not output.startswith("Execution failed."),
                "output": output,
                "interaction_count": int(world.num_interactions),
                "completed": completed,
            }
            self.last_observation = observation
            return {
                "observation": observation,
                "terminated": completed,
                "info": {"event": "execute_python"},
            }
        if tool == "finish":
            answer = arguments.get("answer")
            code = (
                "apis.supervisor.complete_task()"
                if answer is None
                else f"apis.supervisor.complete_task(answer={answer!r})"
            )
            output = world.execute(code)
            completed = bool(world.task_completed())
            observation = {
                "ok": not output.startswith("Execution failed."),
                "output": output,
                "interaction_count": int(world.num_interactions),
                "completed": completed,
            }
            self.last_observation = observation
            return {
                "observation": observation,
                "terminated": completed,
                "info": {"event": "finish"},
            }
        raise ValueError(f"unknown AppWorld adapter tool: {tool}")

    def _get_api_docs(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        docs = self._world().task.api_docs
        app_name = str(arguments["app_name"])
        if app_name not in docs:
            return {
                "ok": False,
                "error": {"code": "UNKNOWN_APP", "message": app_name},
            }
        app_docs = docs[app_name]
        api_name = arguments.get("api_name")
        if api_name is None:
            return {
                "ok": True,
                "app_name": app_name,
                "apis": [
                    {
                        "api_name": name,
                        "description": value.get("description", ""),
                    }
                    for name, value in app_docs.items()
                ],
            }
        api_name = str(api_name)
        if api_name not in app_docs:
            return {
                "ok": False,
                "error": {"code": "UNKNOWN_API", "message": f"{app_name}.{api_name}"},
            }
        return {
            "ok": True,
            "app_name": app_name,
            "api_name": api_name,
            "contract": dict(app_docs[api_name]),
        }

    def evaluate(self) -> dict[str, Any]:
        tracker = self._world().evaluate()
        return {
            "success": bool(tracker.success),
            "passed": int(tracker.pass_count),
            "failed": int(tracker.fail_count),
            "total": int(tracker.num_tests),
            "difficulty": tracker.difficulty,
        }

    def save_state(self, state_id: str) -> dict[str, Any]:
        self._world().save_state(state_id=state_id)
        return {"state_id": state_id}

    def load_state(self, state_id: str) -> dict[str, Any]:
        self._world().load_state(state_id)
        self.last_observation = {
            "ok": True,
            "output": "Official AppWorld database checkpoint restored.",
            "interaction_count": int(self.world.num_interactions),
            "completed": bool(self.world.task_completed()),
            "snapshot_scope": "official_database_only",
        }
        return dict(self.last_observation)

    def handle(self, request: Mapping[str, Any]) -> dict[str, Any]:
        command = str(request["command"])
        if command == "reset":
            return self.reset(int(request.get("seed", 0)), request.get("options", {}))
        if command == "step":
            return self.step(str(request["tool"]), request.get("arguments", {}))
        if command == "observe":
            return dict(self.last_observation)
        if command == "evaluate":
            return self.evaluate()
        if command == "save_state":
            return self.save_state(str(request["state_id"]))
        if command == "load_state":
            return self.load_state(str(request["state_id"]))
        if command == "close":
            self.AppWorld.close_all()
            return {"closed": True}
        raise ValueError(f"unknown worker command: {command}")

    def _world(self) -> Any:
        if self.world is None:
            raise RuntimeError("AppWorld worker must be reset before use")
        return self.world


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--max-interactions", type=int, required=True)
    args = parser.parse_args()
    worker = AppWorldWorker(
        root=args.root.resolve(),
        task_id=args.task_id,
        experiment_name=args.experiment_name,
        max_interactions=args.max_interactions,
    )
    for line in sys.stdin:
        request_id: Any = None
        request: dict[str, Any] = {}
        try:
            request = json.loads(line)
            request_id = request.get("id")
            result = worker.handle(request)
            response = {"id": request_id, "ok": True, "result": result}
        except Exception as exc:
            response = {
                "id": request_id,
                "ok": False,
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
        sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
        sys.stdout.flush()
        if request.get("command") == "close":
            break


if __name__ == "__main__":
    main()
