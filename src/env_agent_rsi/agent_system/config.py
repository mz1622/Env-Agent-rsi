"""Target/Diagnostic/Modifier Agent 的统一 JSON 配置模型。

Agent 结构只读取同一组参数：角色、提示词、skills、只读 memory、上下文上限和
provider；provider 自身的差异全部进入 args，并由独立工厂解释。
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class ProviderSettings:
    type: str
    model: str
    args: Mapping[str, Any] = field(default_factory=dict)
    base_url: str | None = None
    api_key_env: str | None = None
    api_key_file: Path | None = None
    executor: str | None = None


@dataclass(frozen=True)
class MemorySettings:
    """Agent 长期记忆的只读检索配置。"""

    type: str = "null"
    path: Path | None = None
    top_k: int = 0


@dataclass(frozen=True)
class AgentConfig:
    name: str
    role: str
    system_prompt: Path
    skills: tuple[Path, ...]
    memory: MemorySettings
    provider: ProviderSettings
    max_steps: int = 30
    max_context_messages: int = 80


def load_agent_config(
    path: str | Path, *, provider_overrides: Mapping[str, Any] | None = None
) -> AgentConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, Mapping):
        raise ValueError("agent config must be a JSON object")
    provider_value = value.get("provider", {})
    if not isinstance(provider_value, Mapping):
        raise ValueError("agent provider must be a JSON object")
    merged = deepcopy(dict(provider_value))
    overrides = dict(provider_overrides or {})
    override_args = overrides.pop("args", {})
    merged.update(overrides)
    args = dict(merged.get("args", {}))
    if not isinstance(override_args, Mapping):
        raise ValueError("provider override args must be an object")
    args.update(dict(override_args))
    root = config_path.parent
    prompt_value = value.get("system_prompt")
    if not isinstance(prompt_value, str):
        raise ValueError("agent config requires system_prompt path")
    skill_values = value.get("skills", [])
    if not isinstance(skill_values, list) or not all(
        isinstance(item, str) for item in skill_values
    ):
        raise ValueError("agent skills must be a list of paths")
    memory_value = value.get("memory", {"type": "null", "top_k": 0})
    if not isinstance(memory_value, Mapping):
        raise ValueError("agent memory must be a JSON object")
    memory_type = str(memory_value.get("type", "null"))
    if memory_type not in {"null", "json"}:
        raise ValueError("agent memory type must be 'null' or 'json'")
    memory_path_value = memory_value.get("path")
    if memory_type == "json" and not isinstance(memory_path_value, str):
        raise ValueError("json agent memory requires a path")
    memory_top_k = int(memory_value.get("top_k", 0))
    if memory_top_k < 0:
        raise ValueError("agent memory top_k must be non-negative")
    return AgentConfig(
        name=str(value.get("name", config_path.stem)),
        role=str(value.get("role", "target")),
        system_prompt=(root / prompt_value).resolve(),
        skills=tuple((root / item).resolve() for item in skill_values),
        memory=MemorySettings(
            type=memory_type,
            path=(
                (root / memory_path_value).resolve()
                if isinstance(memory_path_value, str)
                else None
            ),
            top_k=memory_top_k,
        ),
        provider=ProviderSettings(
            type=str(merged.get("type", "api")),
            model=str(merged.get("model", "")),
            args=args,
            base_url=(str(merged["base_url"]) if merged.get("base_url") else None),
            api_key_env=(
                str(merged["api_key_env"]) if merged.get("api_key_env") else None
            ),
            api_key_file=(
                (root / str(merged["api_key_file"])).resolve()
                if merged.get("api_key_file")
                else None
            ),
            executor=(str(merged["executor"]) if merged.get("executor") else None),
        ),
        max_steps=int(value.get("max_steps", 30)),
        max_context_messages=int(value.get("max_context_messages", 80)),
    )
