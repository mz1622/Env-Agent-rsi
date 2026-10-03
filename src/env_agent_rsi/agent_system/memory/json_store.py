"""从版本化 JSON 文件加载并确定性检索 Agent Memory。

当前实现只支持读取：记录可按 task_ids 限定作用域，并用 keywords、priority 做一个透明的
轻量排序。后续可以替换向量检索器，但不改变 Target/Runner 的 MemoryRetriever 接口。
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from env_agent_rsi.agent_system.memory.protocol import MemoryQuery
from env_agent_rsi.core.protocol import JsonObject


class JsonMemoryStore:
    """只读 JSON Memory Store；默认文件可以包含空 records。"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        value = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise ValueError("memory file must be a JSON object")
        if int(value.get("schema_version", 1)) != 1:
            raise ValueError("unsupported memory schema version")
        records = value.get("records", [])
        if not isinstance(records, list):
            raise ValueError("memory records must be an array")
        self._records = tuple(_validate_record(item) for item in records)

    def retrieve(self, query: MemoryQuery, *, limit: int) -> tuple[JsonObject, ...]:
        """按显式作用域、关键词和优先级返回最多 ``limit`` 条记录。"""

        if limit < 0:
            raise ValueError("memory retrieval limit must be non-negative")
        if limit == 0:
            return ()
        query_text = json.dumps(
            query.to_dict(), ensure_ascii=False, sort_keys=True
        ).lower()
        ranked: list[tuple[float, str, JsonObject]] = []
        for record in self._records:
            task_ids = record.get("task_ids", [])
            if task_ids and query.task_id not in task_ids:
                continue
            score = float(record.get("priority", 0.0))
            if query.task_id in task_ids:
                score += 100.0
            score += sum(
                1.0 for keyword in record.get("keywords", []) if keyword.lower() in query_text
            )
            ranked.append((score, str(record["id"]), record))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return tuple(deepcopy(item[2]) for item in ranked[:limit])


def _validate_record(value: Any) -> JsonObject:
    """检查 Memory 记录保持为可审计的 JSON 对象。"""

    if not isinstance(value, Mapping):
        raise ValueError("each memory record must be an object")
    record = deepcopy(dict(value))
    if not isinstance(record.get("id"), str) or not record["id"].strip():
        raise ValueError("each memory record requires a non-empty string id")
    if "content" not in record:
        raise ValueError(f"memory record {record['id']!r} requires content")
    for field_name in ("task_ids", "keywords"):
        items = record.get(field_name, [])
        if not isinstance(items, list) or any(
            not isinstance(item, str) or not item.strip() for item in items
        ):
            raise ValueError(f"memory record {record['id']!r} {field_name} must be strings")
    priority = record.get("priority", 0.0)
    if not isinstance(priority, (int, float)) or isinstance(priority, bool):
        raise ValueError(f"memory record {record['id']!r} priority must be numeric")
    json.dumps(record, ensure_ascii=False, sort_keys=True)
    return record
