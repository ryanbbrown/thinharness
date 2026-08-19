"""Plugin composition contracts."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from ..tools.base import ToolOrigin

if TYPE_CHECKING:
    from ..children import ChildHarnessHost
    from ..hooks import Hook
    from ..providers import Model
    from ..tools.base import ToolSpec


@dataclass(frozen=True)
class PluginContext:
    """Stable core context available while binding one plugin."""

    root: Path
    model: Model
    child_harnesses: ChildHarnessHost


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
    agent_names: tuple[str, ...] = ()


@runtime_checkable
class Plugin(Protocol):
    """Configured plugin that can bind independently to a harness."""

    name: str

    def bind(self, context: PluginContext) -> PluginBinding:
        """Bind this plugin without file or network I/O."""
        ...


@runtime_checkable
class ChildInheritablePlugin(Protocol):
    """Plugin that explicitly supports rebinding against a child context."""

    def for_child(self) -> Plugin:
        """Return the configured plugin object to bind to one child."""
        ...


__all__ = [
    "ChildInheritablePlugin",
    "Plugin",
    "PluginBinding",
    "PluginConnector",
    "PluginContext",
    "PluginContribution",
    "ToolOrigin",
]
