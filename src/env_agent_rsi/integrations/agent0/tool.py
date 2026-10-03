"""用 Agent0 原生 BaseTool 驱动 AppWorld backend。

本文件只实现 task 初始化、三种 AppWorld action 的路由和官方 evaluator 回传；批处理、
并发、trajectory 生命周期与 HTTP 协议全部复用 Agent0 的 BaseTool/AsyncToolServer。
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

from env_agent_rsi.agent_runtime.agent0_protocol import (
    parse_tool_call,
    render_tool_response,
)
from env_agent_rsi.benchmarks.appworld import AppWorldProcessBackend
from env_agent_rsi.integrations.agent0.reward import make_evaluation_marker
from env_agent_rsi.integrations.agent0.upstream import ensure_agent0_executor_importable


ensure_agent0_executor_importable()

from verl_tool.servers.tools.base import BaseTool, register_tool  # noqa: E402


SUPPORTED_TOOLS = frozenset({"get_api_docs", "execute_python", "finish"})


@register_tool
class AppWorldAgent0Tool(BaseTool):
    """Agent0 trajectory 到独立 AppWorld worker 的一对一映射。"""

    tool_type = "appworld"

    def __init__(
        self,
        num_workers: int = 1,
        *,
        backend_factory: Callable[..., AppWorldProcessBackend] = AppWorldProcessBackend,
    ) -> None:
        super().__init__(num_workers=num_workers)
        self.backend_factory = backend_factory

    def get_usage_inst(self) -> str:
        return (
            'Use exactly one action per turn as <tool_call>{"name": '
            '"get_api_docs|execute_python|finish", "arguments": {...}}'
            "</tool_call>."
        )

    def parse_action(self, action: str) -> tuple[dict[str, Any], bool]:
        """从 Agent0 的 action stop token 输出中读取单个 JSON 工具调用。"""

        return parse_tool_call(action, supported_tools=SUPPORTED_TOOLS)

    def _new_env(
        self, trajectory_id: str, extra_field: Mapping[str, Any]
    ) -> dict[str, Any]:
        task_id = str(extra_field.get("task_id", ""))
        if not task_id:
            raise ValueError("Agent0 extra_info requires task_id")
        backend = self.backend_factory(
            task_id=task_id,
            appworld_root=extra_field.get("appworld_root"),
            python_executable=extra_field.get("python_executable"),
            experiment_name=str(extra_field.get("experiment_name", "agent0_appworld")),
            max_interactions=int(extra_field.get("max_interactions", 40)),
            request_timeout=float(extra_field.get("request_timeout", 120.0)),
        )
        return {
            "trajectory_id": trajectory_id,
            "metadata": {"turns": 0},
            "previous_obs": [],
            "backend": backend,
            "task_id": task_id,
            "finalized": False,
        }

    def _evaluation_observation(
        self, env: Mapping[str, Any], reward_key: str
    ) -> dict[str, Any]:
        evaluation = env["backend"].evaluate()
        payload = {
            "task_id": env["task_id"],
            "success": bool(evaluation.success),
            "passed": int(evaluation.metrics.get("passed", 0)),
            "total": int(evaluation.metrics.get("total", 0)),
        }
        marker = make_evaluation_marker(payload, reward_key)
        return {
            "obs": render_tool_response(
                "Official AppWorld evaluation completed. End the trajectory without "
                f"another tool call.\n{marker}"
            ),
            "reward": float(evaluation.success),
            "appworld_evaluation": payload,
        }

    def conduct_action(
        self, trajectory_id: str, action: str, extra_field: Mapping[str, Any]
    ) -> tuple[str | dict[str, Any], bool, bool]:
        """执行一步；终态标记先进入轨迹，再由下一次普通 EOS 结束。"""

        parsed, valid = self.parse_action(action)
        if not valid:
            return render_tool_response(self.get_usage_inst()), False, False
        env = self.env_cache.get(trajectory_id)
        try:
            if env is None:
                env = self._new_env(trajectory_id, extra_field)
            if env["finalized"]:
                observation: str | dict[str, Any] = self._evaluation_observation(
                    env, str(extra_field.get("reward_key", ""))
                )
            else:
                native = env["backend"].step(parsed["name"], parsed["arguments"])
                visible = native.get("observation", native)
                observation = render_tool_response(visible)
                if parsed["name"] == "finish" or bool(native.get("terminated")):
                    env["finalized"] = True
                    observation = self._evaluation_observation(
                        env, str(extra_field.get("reward_key", ""))
                    )
        except Exception as exc:
            if env is not None:
                self.save_env(trajectory_id, env)
            return (
                render_tool_response(
                    f"AppWorld tool error: {type(exc).__name__}: {exc}"
                ),
                False,
                False,
            )
        self.update_env(
            trajectory_id,
            env,
            parsed,
            True,
            {"task_id": extra_field.get("task_id")},
            observation,
        )
        self.save_env(trajectory_id, env)
        return observation, False, True

    def delete_env(self, trajectory_id: str) -> None:
        """让 Agent0 finish tool 同时关闭对应的 AppWorld 子进程。"""

        env = self.env_cache.pop(trajectory_id, None)
        if env is not None:
            env["backend"].close()
