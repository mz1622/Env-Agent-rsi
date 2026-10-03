"""把不可信候选环境执行隔离到一次性子进程。

候选崩溃、退出或超时只产生结构化错误，不会污染主进化进程。worker 必须是模块顶层
可序列化函数，输入和输出必须为 JSON 对象；这是一层故障隔离，不等同于权限沙箱。
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import multiprocessing
from queue import Empty
import traceback
from typing import Any, Callable, Mapping

from env_agent_rsi.core.protocol import JsonObject


IsolatedWorker = Callable[[Mapping[str, Any]], Mapping[str, Any]]


class SubprocessExecutionError(RuntimeError):
    """候选子进程失败、超时或没有返回结果。"""


def _worker_entry(
    worker: IsolatedWorker,
    payload: JsonObject,
    queue: Any,
) -> None:
    try:
        result = dict(worker(payload))
        json.dumps(result, ensure_ascii=False, sort_keys=True)
        queue.put({"status": "ok", "result": result})
    except BaseException as exc:  # 子进程边界必须捕获候选的所有失败。
        queue.put(
            {
                "status": "error",
                "error_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
        )


def run_in_subprocess(
    worker: IsolatedWorker,
    payload: Mapping[str, Any],
    *,
    timeout_seconds: float = 120.0,
    start_method: str = "spawn",
) -> JsonObject:
    """运行一次候选并返回 JSON；任何故障都转换为主进程异常。"""

    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    context = multiprocessing.get_context(start_method)
    queue = context.Queue(maxsize=1)
    process = context.Process(
        target=_worker_entry,
        args=(worker, dict(payload), queue),
        daemon=True,
    )
    process.start()
    process.join(timeout_seconds)
    if process.is_alive():
        process.terminate()
        process.join(5)
        raise SubprocessExecutionError(
            f"candidate subprocess exceeded {timeout_seconds:g} seconds"
        )
    try:
        envelope = queue.get(timeout=1)
    except Empty as exc:
        raise SubprocessExecutionError(
            f"candidate subprocess exited with code {process.exitcode} without a result"
        ) from exc
    finally:
        queue.close()
    if envelope.get("status") != "ok":
        raise SubprocessExecutionError(
            f"{envelope.get('error_type', 'Error')}: {envelope.get('message', '')}\n"
            f"{envelope.get('traceback', '')}"
        )
    return dict(envelope["result"])


@dataclass(frozen=True)
class IsolatedExecutor:
    """把配对评估器的 ``(spec, seed)`` 接口适配到子进程 worker。"""

    worker: IsolatedWorker
    timeout_seconds: float = 120.0
    start_method: str = "spawn"

    def __call__(self, spec: Mapping[str, Any], seed: int) -> JsonObject:
        return run_in_subprocess(
            self.worker,
            {"environment_spec": dict(spec), "seed": int(seed)},
            timeout_seconds=self.timeout_seconds,
            start_method=self.start_method,
        )
