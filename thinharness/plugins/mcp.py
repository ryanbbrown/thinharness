"""Model Context Protocol plugin composition."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass

from ..tools.base import ToolSpec
from ..tools.mcp import MCPServer
from ._builtin import _BuiltinPlugin
from .base import PluginBinding, PluginContext, PluginContribution


@dataclass(frozen=True)
class _BoundServer:
    """One server wrapper with identity local to a plugin binding."""

    server: MCPServer
    resolved_id: str

    async def list_tools(self) -> list[ToolSpec]:
        """Discover tools with handlers and attribution fixed to this binding."""
        return await self.server.list_tools(server_id=self.resolved_id)


class MCPPlugin(_BuiltinPlugin, fixed_name="mcp"):
    """Expose one ordered group of MCP servers through a harness plugin."""

    def __init__(self, *, servers: Sequence[MCPServer]) -> None:
        if isinstance(servers, (set, frozenset)):
            raise TypeError("MCPPlugin servers must be an ordered sequence, not a set")
        self.servers = tuple(servers)
        if any(not isinstance(server, MCPServer) for server in self.servers):
            raise TypeError("MCPPlugin servers must contain only MCPServer values")
        if len({id(server) for server in self.servers}) != len(self.servers):
            raise ValueError("MCPPlugin servers must not contain the same server object more than once")

    def bind(self, context: PluginContext) -> PluginBinding:
        """Resolve binding-local server ids without opening a connection."""
        del context
        counts: dict[str, int] = {}
        bound_servers: list[_BoundServer] = []
        for server in self.servers:
            base_id = server.id
            counts[base_id] = counts.get(base_id, 0) + 1
            resolved_id = base_id if counts[base_id] == 1 else f"{base_id}-{counts[base_id]}"
            bound_servers.append(_BoundServer(server, resolved_id))
        snapshot = tuple(bound_servers)

        @asynccontextmanager
        async def connect() -> AsyncIterator[PluginContribution]:
            stack = AsyncExitStack()
            try:
                tools: list[ToolSpec] = []
                for bound in snapshot:
                    await stack.enter_async_context(bound.server)
                    tools.extend(await bound.list_tools())
                yield PluginContribution(tools=tuple(tools))
            except BaseException as exc:
                cleanup = asyncio.create_task(stack.aclose())
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError as cancellation:
                    try:
                        await cleanup
                    except BaseException as cleanup_error:
                        cancellation.add_note(f"cleanup also failed: {type(cleanup_error).__name__}: {cleanup_error}")
                    raise cancellation
                except BaseException as cleanup_error:
                    exc.add_note(f"cleanup also failed: {type(cleanup_error).__name__}: {cleanup_error}")
                raise
            else:
                await stack.aclose()

        return PluginBinding(connect=connect)


__all__ = ["MCPPlugin"]
