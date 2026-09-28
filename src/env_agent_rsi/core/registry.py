from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Generic, TypeVar


T = TypeVar("T")
Builder = Callable[[Mapping[str, Any]], T]


class ComponentRegistry(Generic[T]):
    """Small explicit registry for pluggable environment components."""

    def __init__(self, component_kind: str):
        self.component_kind = component_kind
        self._builders: dict[str, Builder[T]] = {}

    def register(
        self, name: str, builder: Builder[T], *, replace: bool = False
    ) -> None:
        if not name:
            raise ValueError("component name must be non-empty")
        if name in self._builders and not replace:
            raise ValueError(
                f"{self.component_kind} component {name!r} is already registered"
            )
        self._builders[name] = builder

    def build(self, name: str, config: Mapping[str, Any]) -> T:
        try:
            builder = self._builders[name]
        except KeyError as exc:
            available = ", ".join(sorted(self._builders)) or "<none>"
            raise ValueError(
                f"unknown {self.component_kind} component {name!r}; "
                f"available: {available}"
            ) from exc
        return builder(config)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._builders))
