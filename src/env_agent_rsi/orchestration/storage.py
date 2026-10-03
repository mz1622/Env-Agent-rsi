"""逐条追加、可恢复读取的 JSONL episode 轨迹仓库。

每一行都是完整 rollout 记录，包含环境角色、环境哈希、seed、rollout 序号、Agent
消息、逐步动作/观察、最终 verifier 结果和停止原因，便于诊断及复现实验。
"""

from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Any, Iterator, Mapping

from env_agent_rsi.core.protocol import JsonObject


class TraceStore:
    """最小 JSONL 持久化层。"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def append(self, record: Mapping[str, Any]) -> None:
        line = json.dumps(dict(record), ensure_ascii=False, sort_keys=True)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
                handle.flush()

    def records(self) -> Iterator[JsonObject]:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(
                        f"trace line {line_number} must contain a JSON object"
                    )
                yield value
