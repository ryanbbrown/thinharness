"""Optional Model Context Protocol client support."""

from __future__ import annotations

import asyncio
import copy
import json
import re
from abc import ABC, abstractmethod
from contextlib import AbstractAsyncContextManager, AsyncExitStack
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

from .tools import Json, ToolResult, ToolSpec

_SHUTDOWN_GRACE_SECONDS = 3
_INSTALL_HINT = "Install MCP support with: pip install thinharness[mcp]"
_MCP_NAME_RE = re.compile(r"[^a-zA-Z0-9_-]")


class MCPError(RuntimeError):
    """Raised when MCP setup or tool discovery fails."""


class MCPDependencyError(MCPError):
    """Raised when MCP support is used without the optional dependency."""

    def __init__(self, cause: ImportError) -> None:
        super().__init__(f"{cause}. {_INSTALL_HINT}")
        self.__cause__ = cause


@dataclass
class _SessionState:
    """Connection state owned by one background session task."""

    session_task: asyncio.Task[None] | None = None
    ready_event: asyncio.Event | None = None
    stop_event: asyncio.Event | None = None
    nesting_counter: int = 0
    client: Any | None = None
    connect_error: BaseException | None = None

    async def force_close(self, task: asyncio.Task[None]) -> None:
        """Cancel a session task and bound cleanup time."""
        task.cancel()
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=_SHUTDOWN_GRACE_SECONDS)
        except TimeoutError:
            return
        except asyncio.CancelledError:
            if current_task := asyncio.current_task():
                if current_task.cancelling():
                    raise
            return


class MCPServer(ABC):
    """One MCP server connection that contributes ToolSpecs to a harness."""

    def __init__(
        self,
        *,
        tool_prefix: str | None = None,
        timeout: float = 5.0,
        read_timeout: float = 300.0,
        include_tools: list[str] | None = None,
        exclude_tools: list[str] | None = None,
        id: str | None = None,
    ) -> None:
        self.tool_prefix = tool_prefix
        self.timeout = timeout
        self.read_timeout = read_timeout
        self.include_tools = list(include_tools) if include_tools is not None else None
        self.exclude_tools = list(exclude_tools) if exclude_tools is not None else None
        self._id = id
        self._resolved_id: str | None = None
        self._session_state = _SessionState()
        self._enter_lock = asyncio.Lock()

    @property
    def id(self) -> str:
        """Stable readable identifier; falls back to a derived default."""
        return self._resolved_id or self._id or self._default_id()

    @abstractmethod
    def _default_id(self) -> str:
        """Return a readable default id for this transport."""

    @abstractmethod
    def _client_streams(self) -> AbstractAsyncContextManager[Any]:
        """Return the transport streams context manager."""

    async def __aenter__(self) -> MCPServer:
        """Open or share the MCP session."""
        async with self._enter_lock:
            state = self._session_state
            if state.session_task is None or state.session_task.done():
                state.stop_event = asyncio.Event()
                state.ready_event = asyncio.Event()
                state.connect_error = None
                state.client = None
                state.session_task = asyncio.create_task(self._session_runner())
                try:
                    await state.ready_event.wait()
                except BaseException:
                    task = state.session_task
                    if state.stop_event is not None:
                        state.stop_event.set()
                    await state.force_close(task)
                    state.session_task = None
                    state.client = None
                    raise
                if state.connect_error is not None:
                    state.session_task = None
                    err = state.connect_error
                    state.connect_error = None
                    raise err
            state.nesting_counter += 1
        return self

    async def __aexit__(self, *exc: object) -> None:
        """Release one MCP session reference and close on the last exit."""
        task: asyncio.Task[None] | None = None
        async with self._enter_lock:
            state = self._session_state
            if state.nesting_counter == 0:
                raise ValueError("MCPServer.__aexit__ called more times than __aenter__")
            state.nesting_counter -= 1
            if state.nesting_counter > 0 or state.session_task is None:
                return
            if state.stop_event is not None:
                state.stop_event.set()
            task = state.session_task
            state.session_task = None
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=_SHUTDOWN_GRACE_SECONDS)
        except TimeoutError:
            await self._session_state.force_close(task)
        except asyncio.CancelledError:
            await self._session_state.force_close(task)
            raise

    async def list_tools(self) -> list[ToolSpec]:
        """Discover and convert the MCP server's current tool snapshot."""
        async with self:
            session = self._session()
            result = await session.list_tools()
        seen: dict[str, str] = {}
        specs: list[ToolSpec] = []
        for tool in result.tools:
            original_name = str(tool.name)
            if self.exclude_tools is not None and original_name in self.exclude_tools:
                continue
            if self.include_tools is not None and original_name not in self.include_tools:
                continue
            public_name = _normalize_mcp_name(f"{self.tool_prefix}_{original_name}" if self.tool_prefix else original_name)
            if public_name in seen:
                raise MCPError(
                    f"MCP tool name collision after sanitization on server {self.id!r}: "
                    f"{seen[public_name]!r} and {original_name!r} both map to {public_name!r}"
                )
            seen[public_name] = original_name
            specs.append(ToolSpec(
                public_name,
                str(tool.description or ""),
                _clean_mcp_schema(tool.inputSchema, original_name),
                _make_tool_handler(self, original_name),
                sequential=False,
                metadata={"source": "mcp", "mcp_server_id": self.id, "mcp_tool_name": original_name},
                max_retries=None,
            ))
        return specs

    async def call_tool(self, name: str, arguments: Json) -> ToolResult:
        """Call one MCP tool and normalize its result."""
        base_metadata = {"source": "mcp", "mcp_server_id": self.id, "mcp_tool_name": name}
        try:
            async with self:
                result = await self._session().call_tool(name, arguments=arguments)
        except MCPDependencyError:
            raise
        except _mcp_error_type() as exc:
            return ToolResult(False, str(exc), {**base_metadata, "error_type": "MCPError"})
        except _mcp_transport_error_types() as exc:
            return ToolResult(False, str(exc), {**base_metadata, "error_type": "MCPError"})
        if result.isError is True:
            return ToolResult(
                False,
                _content_to_text(result.content),
                {**base_metadata, "error_type": "MCPToolError", "retry": True},
            )
        structured_content = getattr(result, "structuredContent", None)
        if structured_content is not None:
            content = json.dumps(structured_content, ensure_ascii=False)
        else:
            content = _content_to_text(result.content)
        return ToolResult(True, content, base_metadata)

    async def _session_runner(self) -> None:
        """Own transport and ClientSession lifecycle from a single task."""
        state = self._session_state
        ready_event = state.ready_event
        stop_event = state.stop_event
        assert ready_event is not None
        assert stop_event is not None
        client = None
        try:
            ClientSession = _client_session_type()
            async with AsyncExitStack() as stack:
                streams = await stack.enter_async_context(self._client_streams())
                read_stream, write_stream = streams[0], streams[1]
                session = ClientSession(
                    read_stream=read_stream,
                    write_stream=write_stream,
                    read_timeout_seconds=timedelta(seconds=self.read_timeout),
                )
                client = await stack.enter_async_context(session)
                async with asyncio.timeout(self.timeout):
                    await client.initialize()
                state.client = client
                ready_event.set()
                await stop_event.wait()
        except BaseException as exc:
            # Post-ready shutdown errors are harmless; the next first-entry path resets connect_error before respawning.
            if state.session_task is asyncio.current_task():
                state.connect_error = exc
        finally:
            if state.client is client:
                state.client = None
            ready_event.set()

    def _session(self) -> Any:
        """Return the active ClientSession or raise if disconnected."""
        client = self._session_state.client
        if client is None:
            raise MCPError(f"{type(self).__name__} is not connected")
        return client


class MCPServerStdio(MCPServer):
    """MCP server reached through a stdio subprocess."""

    def __init__(
        self,
        command: str,
        args: list[str] | None = None,
        *,
        env: dict[str, str] | None = None,
        cwd: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.command = command
        self.args = list(args or [])
        self.env = dict(env) if env is not None else None
        self.cwd = cwd

    def _default_id(self) -> str:
        """Return command and args as the default id."""
        return " ".join([self.command, *self.args])

    def _client_streams(self) -> AbstractAsyncContextManager[Any]:
        """Create stdio streams for this server."""
        try:
            from mcp.client.stdio import StdioServerParameters, stdio_client
        except ImportError as exc:
            raise MCPDependencyError(exc) from exc
        return stdio_client(StdioServerParameters(command=self.command, args=self.args, env=self.env, cwd=self.cwd))


class MCPServerSSE(MCPServer):
    """MCP server reached through SSE."""

    def __init__(self, url: str, *, headers: dict[str, str] | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.url = url
        self.headers = dict(headers) if headers is not None else None

    def _default_id(self) -> str:
        """Return the URL as the default id."""
        return self.url

    def _client_streams(self) -> AbstractAsyncContextManager[Any]:
        """Create SSE streams for this server."""
        try:
            from mcp.client.sse import sse_client
        except ImportError as exc:
            raise MCPDependencyError(exc) from exc
        return sse_client(self.url, headers=self.headers, timeout=self.timeout, sse_read_timeout=self.read_timeout)


class MCPServerStreamableHTTP(MCPServer):
    """MCP server reached through streamable HTTP."""

    def __init__(self, url: str, *, headers: dict[str, str] | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.url = url
        self.headers = dict(headers) if headers is not None else None

    def _default_id(self) -> str:
        """Return the URL as the default id."""
        return self.url

    def _client_streams(self) -> AbstractAsyncContextManager[Any]:
        """Create streamable HTTP streams for this server."""
        try:
            from mcp.client.streamable_http import streamablehttp_client
        except ImportError as exc:
            raise MCPDependencyError(exc) from exc
        return streamablehttp_client(self.url, headers=self.headers, timeout=self.timeout, sse_read_timeout=self.read_timeout)


def _client_session_type() -> type[Any]:
    """Import ClientSession lazily."""
    try:
        from mcp import ClientSession
    except ImportError as exc:
        raise MCPDependencyError(exc) from exc
    return ClientSession


def _mcp_error_type() -> type[BaseException]:
    """Import McpError lazily."""
    try:
        from mcp.shared.exceptions import McpError
    except ImportError as exc:
        raise MCPDependencyError(exc) from exc
    return McpError


def _mcp_transport_error_types() -> tuple[type[BaseException], ...]:
    """Return transport exceptions that should become MCP tool errors."""
    errors: list[type[BaseException]] = [ConnectionError, TimeoutError]
    try:
        from anyio import BrokenResourceError, ClosedResourceError, EndOfStream
    except ImportError:
        return tuple(errors)
    errors.extend([BrokenResourceError, ClosedResourceError, EndOfStream])
    return tuple(errors)


def _make_tool_handler(server: MCPServer, tool_name: str) -> Any:
    """Build an async ToolSpec handler for one MCP tool."""
    async def handler(args: Json) -> ToolResult:
        """Call the backing MCP tool."""
        return await server.call_tool(tool_name, args)

    return handler


def _normalize_mcp_name(name: str) -> str:
    """Normalize an MCP tool name for function-tool APIs."""
    normalized = _MCP_NAME_RE.sub("_", name)
    return normalized or "mcp_tool"


def _clean_mcp_schema(schema: Any, tool_name: str) -> Json:
    """Copy and minimally normalize an MCP input schema."""
    if not isinstance(schema, dict):
        raise MCPError(f"MCP tool {tool_name!r} inputSchema must be an object")
    cleaned = copy.deepcopy(schema)
    cleaned.pop("$schema", None)
    cleaned.pop("title", None)
    if cleaned.get("type") == "object" and "additionalProperties" not in cleaned:
        cleaned["additionalProperties"] = False
    return cleaned


def _content_to_text(blocks: list[Any]) -> str:
    """Convert MCP content blocks to model-visible text."""
    parts: list[str] = []
    for block in blocks:
        block_type = getattr(block, "type", "")
        if block_type == "text":
            parts.append(str(getattr(block, "text", "")))
        elif block_type == "image":
            parts.append(f"[image: {getattr(block, 'mimeType', 'unknown')}]")
        elif block_type == "audio":
            parts.append(f"[audio: {getattr(block, 'mimeType', 'unknown')}]")
        elif block_type in {"resource", "resource_link"}:
            uri = getattr(block, "uri", None)
            resource = getattr(block, "resource", None)
            if uri is None and resource is not None:
                uri = getattr(resource, "uri", None)
            parts.append(f"[resource: {uri or 'unknown'}]")
        else:
            parts.append(str(block))
    return "\n".join(parts)
