"""把 AppWorld 官方 evaluator 结果转换为 Agent0 可加载的规则奖励。

工具环境生成带逐样本 HMAC 的终态标记；奖励函数只接受签名正确且 task_id 一致的标记，
避免模型在自然语言输出中自行伪造成功。文件保持无第三方依赖，可被 Agent0 的自定义
reward loader 直接按路径加载。
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import re
from typing import Any, Mapping


MARKER_PATTERN = re.compile(
    r'<appworld_evaluation payload="(?P<payload>[A-Za-z0-9_-]+)" '
    r'signature="(?P<signature>[0-9a-f]{64})"\s*/>'
)


def make_evaluation_marker(payload: Mapping[str, Any], reward_key: str) -> str:
    """序列化并签名一次官方 evaluator 结果。"""

    if not reward_key:
        raise ValueError("reward_key must not be empty")
    raw = json.dumps(
        dict(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    encoded = base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
    signature = hmac.new(
        reward_key.encode("utf-8"), encoded.encode("ascii"), hashlib.sha256
    ).hexdigest()
    return (
        f'<appworld_evaluation payload="{encoded}" '
        f'signature="{signature}"/>'
    )


def decode_evaluation_marker(text: str, reward_key: str) -> dict[str, Any] | None:
    """读取最后一个签名正确的环境标记。"""

    matches = list(MARKER_PATTERN.finditer(text or ""))
    for match in reversed(matches):
        encoded = match.group("payload")
        expected = hmac.new(
            reward_key.encode("utf-8"), encoded.encode("ascii"), hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(expected, match.group("signature")):
            continue
        padding = "=" * (-len(encoded) % 4)
        try:
            value = json.loads(
                base64.urlsafe_b64decode(encoded + padding).decode("utf-8")
            )
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            return value
    return None


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Mapping[str, Any] | None,
) -> dict[str, float]:
    """Agent0/veRL 规则奖励入口，成功为 1，其他情况为 0。"""

    info = dict(extra_info or {})
    key = str(info.get("reward_key", ""))
    payload = decode_evaluation_marker(solution_str, key) if key else None
    expected_task = (
        str(ground_truth.get("task_id", ""))
        if isinstance(ground_truth, Mapping)
        else str(ground_truth or info.get("task_id", ""))
    )
    valid = bool(
        data_source == "appworld"
        and payload
        and str(payload.get("task_id", "")) == expected_task
    )
    success = bool(valid and payload and payload.get("success") is True)
    return {
        "score": float(success),
        "appworld_success": float(success),
        "signed_evaluation_found": float(valid),
    }

