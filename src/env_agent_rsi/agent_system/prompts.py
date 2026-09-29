"""从 JSON 载入 system prompt 与 skill。

提示词和技能不写在 Agent 类中；加载器检查最小 schema，并把 skill 渲染为明确分隔的
系统上下文段落，从而支持配置替换、版本记录和实验复现。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class PromptSpec:
    name: str
    content: str
    schema_version: int = 1


@dataclass(frozen=True)
class SkillSpec:
    name: str
    description: str
    instructions: tuple[str, ...]
    schema_version: int = 1

    def render(self) -> str:
        lines = [f"[Skill: {self.name}]", self.description]
        lines.extend(f"{index}. {item}" for index, item in enumerate(self.instructions, 1))
        return "\n".join(lines)


def load_prompt(path: str | Path) -> PromptSpec:
    value = _load_json_object(path)
    content = value.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError(f"prompt {path} must contain non-empty content")
    return PromptSpec(
        name=str(value.get("name", Path(path).stem)),
        content=content,
        schema_version=int(value.get("schema_version", 1)),
    )


def load_skill(path: str | Path) -> SkillSpec:
    value = _load_json_object(path)
    instructions = value.get("instructions")
    if not isinstance(instructions, list) or not all(
        isinstance(item, str) and item.strip() for item in instructions
    ):
        raise ValueError(f"skill {path} instructions must be non-empty strings")
    return SkillSpec(
        name=str(value.get("name", Path(path).stem)),
        description=str(value.get("description", "")),
        instructions=tuple(instructions),
        schema_version=int(value.get("schema_version", 1)),
    )


def _load_json_object(path: str | Path) -> Mapping[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return value
