"""工具契约的构造、版本化与运行时校验。

环境把工具统一表示为 OpenAI-compatible function schema；本模块负责生成稳定的
contract version，并在 action 进入环境前执行一小组确定性的 JSON Schema 校验。
它刻意不依赖模型 SDK，使所有 provider 和 benchmark adapter 共享同一契约语义。
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any, Mapping, Sequence

from env_agent_rsi.core.protocol import Action, EnvDescriptor, JsonObject


class ActionValidationError(ValueError):
    """携带稳定错误码的 action/schema 校验错误。"""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def tool_schema(
    name: str,
    description: str,
    properties: JsonObject,
    required: list[str] | None = None,
) -> JsonObject:
    """构造项目统一使用的 function tool schema。"""

    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": deepcopy(properties),
                "required": list(required or []),
                "additionalProperties": False,
            },
        },
    }


def contract_version(
    task_id: str, instruction: str, tools: Sequence[Mapping[str, Any]]
) -> str:
    """用 Agent 可见契约的规范 JSON 计算稳定版本号。"""

    payload = json.dumps(
        {"task_id": task_id, "instruction": instruction, "tools": list(tools)},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def make_descriptor(
    *,
    task_id: str,
    instruction: str,
    tools: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any] | None = None,
) -> EnvDescriptor:
    """创建不可变的 Agent 可见环境描述。"""

    copied_tools = tuple(deepcopy(dict(schema)) for schema in tools)
    return EnvDescriptor(
        task_id=task_id,
        instruction=instruction,
        tools=copied_tools,
        contract_version=contract_version(task_id, instruction, copied_tools),
        metadata=deepcopy(dict(metadata or {})),
    )


def replace_tools(
    descriptor: EnvDescriptor, tools: Sequence[Mapping[str, Any]]
) -> EnvDescriptor:
    """替换 descriptor 的工具并重新计算版本。"""

    return make_descriptor(
        task_id=descriptor.task_id,
        instruction=descriptor.instruction,
        tools=tools,
        metadata=descriptor.metadata,
    )


def validate_action(action: Action, descriptor: EnvDescriptor) -> None:
    """根据当前 descriptor 校验工具名和参数，失败时抛出稳定错误。"""

    schemas = {
        str(schema.get("function", {}).get("name")): schema
        for schema in descriptor.tools
    }
    schema = schemas.get(action.tool)
    if schema is None:
        raise ActionValidationError(
            "UNKNOWN_TOOL", f"tool {action.tool!r} is not available in this contract"
        )
    parameters = schema.get("function", {}).get("parameters", {})
    if not isinstance(action.arguments, Mapping):
        raise ActionValidationError(
            "INVALID_ARGUMENTS", "tool arguments must be an object"
        )
    _validate_value(dict(action.arguments), parameters, path="arguments")


def _validate_value(value: Any, schema: Mapping[str, Any], *, path: str) -> None:
    if "enum" in schema and value not in schema["enum"]:
        raise ActionValidationError(
            "INVALID_ARGUMENT",
            f"{path} must be one of {schema['enum']!r}; got {value!r}",
        )

    expected = schema.get("type")
    if expected == "object":
        if not isinstance(value, Mapping):
            _wrong_type(path, "object", value)
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        for name in required:
            if name not in value:
                raise ActionValidationError(
                    "MISSING_REQUIRED_ARGUMENT", f"{path}.{name} is required"
                )
        additional = schema.get("additionalProperties", True)
        for name, child in value.items():
            child_schema = properties.get(name)
            if child_schema is not None:
                _validate_value(child, child_schema, path=f"{path}.{name}")
            elif additional is False:
                raise ActionValidationError(
                    "UNEXPECTED_ARGUMENT", f"{path}.{name} is not allowed"
                )
            elif isinstance(additional, Mapping):
                _validate_value(child, additional, path=f"{path}.{name}")
        return
    if expected == "array":
        if not isinstance(value, list):
            _wrong_type(path, "array", value)
        item_schema = schema.get("items", {})
        for index, child in enumerate(value):
            _validate_value(child, item_schema, path=f"{path}[{index}]")
        return
    if expected == "string" and not isinstance(value, str):
        _wrong_type(path, "string", value)
    if expected == "integer" and (
        not isinstance(value, int) or isinstance(value, bool)
    ):
        _wrong_type(path, "integer", value)
    if expected == "number" and (
        not isinstance(value, (int, float)) or isinstance(value, bool)
    ):
        _wrong_type(path, "number", value)
    if expected == "boolean" and not isinstance(value, bool):
        _wrong_type(path, "boolean", value)


def _wrong_type(path: str, expected: str, value: Any) -> None:
    raise ActionValidationError(
        "INVALID_ARGUMENT_TYPE",
        f"{path} must be {expected}; got {type(value).__name__}",
    )
