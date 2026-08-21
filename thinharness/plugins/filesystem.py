"""Filesystem plugin."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from ..tools.base import ToolOrigin
from ..tools.filesystem import FileTools
from ._builtin import _FrozenBuiltinPlugin
from .base import PluginBinding, PluginContext, PluginContribution

_DEFAULT_TOOLS = ("read", "write", "edit", "search", "list", "glob")


@dataclass(frozen=True)
class _FilesystemConfig:
    selected: tuple[str, ...]
    output_dir: str | Path | None
    max_read_chars: int
    max_read_bytes: int
    max_image_bytes: int
    max_tool_chars: int
    max_search_line_chars: int
    rg_timeout: int
    search_exclude_globs: tuple[str, ...] | None
    read_paths: tuple[str | Path, ...] | None
    write_paths: tuple[str | Path, ...] | None


class FilesystemPlugin(_FrozenBuiltinPlugin, fixed_name="filesystem"):
    """Provide root-scoped filesystem tools to one harness."""

    _config: _FilesystemConfig
    _frozen: bool

    def __init__(
        self,
        *,
        tools: Sequence[str] | None = None,
        output_dir: str | Path | None = None,
        max_read_chars: int = 40_000,
        max_read_bytes: int = 1_000_000,
        max_image_bytes: int = 5_000_000,
        max_tool_chars: int = 40_000,
        max_search_line_chars: int = 180,
        rg_timeout: int = 30,
        search_exclude_globs: list[str] | None = None,
        read_paths: Sequence[str | Path] | None = None,
        write_paths: Sequence[str | Path] | None = None,
    ) -> None:
        if not isinstance(max_image_bytes, int) or isinstance(max_image_bytes, bool) or max_image_bytes <= 0:
            raise ValueError("max_image_bytes must be a positive integer")
        if isinstance(tools, (set, frozenset)):
            raise TypeError("FilesystemPlugin tools must be an ordered sequence, not a set")
        selected = tuple(_DEFAULT_TOOLS if tools is None else tools)
        if len(set(selected)) != len(selected):
            raise ValueError("FilesystemPlugin tools contains a duplicate name")
        object.__setattr__(self, "_config", _FilesystemConfig(
            selected=selected,
            output_dir=output_dir,
            max_read_chars=max_read_chars,
            max_read_bytes=max_read_bytes,
            max_image_bytes=max_image_bytes,
            max_tool_chars=max_tool_chars,
            max_search_line_chars=max_search_line_chars,
            rg_timeout=rg_timeout,
            search_exclude_globs=tuple(search_exclude_globs) if search_exclude_globs is not None else None,
            read_paths=tuple(read_paths) if read_paths is not None else None,
            write_paths=tuple(write_paths) if write_paths is not None else None,
        ))
        object.__setattr__(self, "_frozen", True)

    def for_child(self) -> FilesystemPlugin:
        """Reuse the frozen constructor configuration for a child binding."""
        return self

    def bind(self, context: PluginContext) -> PluginBinding:
        """Build static tool specifications without filesystem I/O."""
        config = self._config
        collection = FileTools(
            context.root,
            output_dir=config.output_dir,
            max_read_chars=config.max_read_chars,
            max_read_bytes=config.max_read_bytes,
            max_image_bytes=config.max_image_bytes,
            max_tool_chars=config.max_tool_chars,
            max_search_line_chars=config.max_search_line_chars,
            rg_timeout=config.rg_timeout,
            search_exclude_globs=list(config.search_exclude_globs) if config.search_exclude_globs is not None else None,
            read_paths=config.read_paths,
            write_paths=config.write_paths,
            _root_is_resolved=True,
        )
        by_name = {tool.name: tool for tool in collection.specs()}
        unknown = [name for name in config.selected if name not in by_name]
        if unknown:
            available = ", ".join(by_name)
            raise ValueError(f"unknown FilesystemPlugin tool: {unknown[0]}; available: {available}")
        specs = tuple(replace(by_name[name], origin=ToolOrigin(plugin=self.name, source=name)) for name in config.selected)
        return PluginBinding(static=PluginContribution(
            tools=specs,
            instructions=(f"Workspace root: {context.root}",),
        ))
