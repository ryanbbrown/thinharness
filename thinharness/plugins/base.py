"""Plugin composition contracts."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from ..tools.base import ToolOrigin

if TYPE_CHECKING:
    from ..hooks import Hook
    from ..providers import Model
    from ..tools.base import ToolSpec


@dataclass(frozen=True)
class PluginContext:
    """Stable core context available while binding one plugin."""

    root: Path
    model: Model


@dataclass(frozen=True)
class PluginContribution:
    """Tools, instructions, and hooks contributed by one plugin."""

    tools: tuple[ToolSpec, ...] = ()
    instructions: tuple[str, ...] = ()
    hooks: tuple[Hook, ...] = ()


PluginConnector = Callable[[], AbstractAsyncContextManager[PluginContribution]]


@dataclass(frozen=True)
class PluginBinding:
    """Static plugin state and its optional connected contribution."""

    static: PluginContribution = field(default_factory=PluginContribution)
    connect: PluginConnector | None = None


@runtime_checkable
class Plugin(Protocol):
    """Configured plugin that can bind independently to a harness."""

    name: str

    def bind(self, context: PluginContext) -> PluginBinding:
        """Bind this plugin without file or network I/O."""
        ...


__all__ = [
    "Plugin",
    "PluginBinding",
    "PluginConnector",
    "PluginContext",
    "PluginContribution",
    "ToolOrigin",
]
