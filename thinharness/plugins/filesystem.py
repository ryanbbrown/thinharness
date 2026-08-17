"""Filesystem plugin."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from ..tools.base import ToolOrigin
from ..tools.filesystem import FileTools
from .base import PluginBinding, PluginContext, PluginContribution

_DEFAULT_TOOLS = ("read", "write", "edit", "search", "list", "glob")


class FilesystemPlugin:
    """Provide root-scoped filesystem tools to one harness."""

    name = "filesystem"

    def __init__(
        self,
        *,
        tools: Sequence[str] | None = None,
        output_dir: str | Path | None = None,
        max_read_chars: int = 40_000,
        max_read_bytes: int = 1_000_000,
        max_tool_chars: int = 40_000,
        max_search_line_chars: int = 180,
        rg_timeout: int = 30,
        search_exclude_globs: list[str] | None = None,
        read_paths: Sequence[str | Path] | None = None,
        write_paths: Sequence[str | Path] | None = None,
    ) -> None:
        if isinstance(tools, (set, frozenset)):
            raise TypeError("FilesystemPlugin tools must be an ordered sequence, not a set")
        selected = tuple(_DEFAULT_TOOLS if tools is None else tools)
        if len(set(selected)) != len(selected):
            raise ValueError("FilesystemPlugin tools contains a duplicate name")
        self._selected = selected
        self._options = {
            "output_dir": output_dir,
            "max_read_chars": max_read_chars,
            "max_read_bytes": max_read_bytes,
            "max_tool_chars": max_tool_chars,
            "max_search_line_chars": max_search_line_chars,
            "rg_timeout": rg_timeout,
            "search_exclude_globs": search_exclude_globs,
            "read_paths": read_paths,
            "write_paths": write_paths,
        }

    def bind(self, context: PluginContext) -> PluginBinding:
        """Build static tool specifications without filesystem I/O."""
        collection = FileTools(context.root, **self._options)
        by_name = {tool.name: tool for tool in collection.specs()}
        unknown = [name for name in self._selected if name not in by_name]
        if unknown:
            available = ", ".join(by_name)
            raise ValueError(f"unknown FilesystemPlugin tool: {unknown[0]}; available: {available}")
        specs = tuple(
            replace(by_name[name], origin=ToolOrigin(plugin=self.name, source=name))
            for name in self._selected
        )
        return PluginBinding(static=PluginContribution(
            tools=specs,
            instructions=(f"Workspace root: {context.root}",),
        ))
