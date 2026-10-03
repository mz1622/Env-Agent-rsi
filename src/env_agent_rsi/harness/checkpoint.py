"""基础环境与 Harness 层分离的版本化 checkpoint。

文件只保存 JSON 数据：完整环境配置负责重建组件，base_state 保存真实环境状态，layers
按执行层分别保存自己的运行时状态。稳定标签代替 Python 导入路径，便于迁移和审计。
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping

from env_agent_rsi.core.protocol import JsonObject
from env_agent_rsi.harness.wrapper import RuleHarness


CHECKPOINT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class EnvironmentCheckpoint:
    """一个可以独立重建的环境栈快照。"""

    environment_spec: JsonObject
    environment_type: str
    base_state: JsonObject
    layers: tuple[JsonObject, ...]
    metadata: JsonObject = field(default_factory=dict)
    schema_version: int = CHECKPOINT_SCHEMA_VERSION

    def to_dict(self) -> JsonObject:
        return {
            "schema_version": self.schema_version,
            "environment": {
                "type": self.environment_type,
                "state": deepcopy(self.base_state),
            },
            "layers": deepcopy(list(self.layers)),
            "environment_spec": deepcopy(self.environment_spec),
            "metadata": deepcopy(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EnvironmentCheckpoint":
        version = int(value.get("schema_version", 0))
        if version != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(f"unsupported checkpoint schema version: {version}")
        environment = value.get("environment")
        if not isinstance(environment, Mapping):
            raise ValueError("checkpoint environment must be an object")
        spec = value.get("environment_spec")
        layers = value.get("layers")
        metadata = value.get("metadata", {})
        if not isinstance(spec, Mapping):
            raise ValueError("checkpoint environment_spec must be an object")
        state = environment.get("state", {})
        if not isinstance(state, Mapping):
            raise ValueError("checkpoint environment state must be an object")
        if not isinstance(layers, list) or not all(
            isinstance(layer, Mapping) for layer in layers
        ):
            raise ValueError("checkpoint layers must be a list of objects")
        if not isinstance(metadata, Mapping):
            raise ValueError("checkpoint metadata must be an object")
        checkpoint = cls(
            environment_spec=deepcopy(dict(spec)),
            environment_type=str(environment.get("type", "")),
            base_state=deepcopy(dict(state)),
            layers=tuple(deepcopy(dict(layer)) for layer in layers),
            metadata=deepcopy(dict(metadata)),
        )
        # 强制经过 JSON，拒绝隐含的 Python 对象。
        json.dumps(checkpoint.to_dict(), ensure_ascii=False, sort_keys=True)
        return checkpoint

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        return target

    @classmethod
    def load(cls, path: str | Path) -> "EnvironmentCheckpoint":
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise ValueError("checkpoint root must be an object")
        return cls.from_dict(value)


def dump_stack(
    env: RuleHarness, *, metadata: Mapping[str, Any] | None = None
) -> EnvironmentCheckpoint:
    """把当前 base 和每个规则层分别导出。"""

    if not env.environment_spec:
        raise ValueError("environment was not built from a reconstructable spec")
    environment = env.environment_spec.get("environment", {})
    if not isinstance(environment, Mapping):
        raise ValueError("environment spec is missing environment object")
    environment_type = str(environment.get("type", ""))
    if not environment_type:
        raise ValueError("environment spec is missing environment type")
    return EnvironmentCheckpoint(
        environment_spec=deepcopy(env.environment_spec),
        environment_type=environment_type,
        base_state=deepcopy(env.base.save_state()),
        layers=tuple(env.layer_snapshots()),
        metadata=deepcopy(dict(metadata or {})),
    )


def build_stack(checkpoint: EnvironmentCheckpoint) -> RuleHarness:
    """按配置重建组件，再严格恢复 base 与各层状态。"""

    from env_agent_rsi.harness.factory import build_environment

    env = build_environment(checkpoint.environment_spec)
    configured_environment = checkpoint.environment_spec.get("environment", {})
    if not isinstance(configured_environment, Mapping):
        raise ValueError("checkpoint environment_spec is missing environment object")
    configured_type = str(configured_environment.get("type", ""))
    if configured_type != checkpoint.environment_type:
        raise ValueError(
            f"environment type mismatch: {configured_type!r} != "
            f"{checkpoint.environment_type!r}"
        )
    env.base.load_state(checkpoint.base_state)
    env.load_layer_snapshots(checkpoint.layers)
    return env


def save_checkpoint(
    env: RuleHarness,
    path: str | Path,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """导出并保存一个环境栈。"""

    return dump_stack(env, metadata=metadata).save(path)


def load_checkpoint(path: str | Path) -> RuleHarness:
    """从磁盘加载并重建环境栈，不自动 reset。"""

    return build_stack(EnvironmentCheckpoint.load(path))
