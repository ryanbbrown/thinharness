"""Optional Model Context Protocol client support."""

from __future__ import annotations

import asyncio
import copy
import json
import re
from pathlib import Path
from typing import Any

import httpx

from .base import Json, McpToolInfo, ToolResult, ToolSpec

_INSTALL_HINT = "Install MCP support with: pip install thinharness[mcp]"
_MCP_NAME_RE = re.compile(r"[^a-zA-Z0-9_-]")


class MCPError(RuntimeError):
    """Raised when MCP setup or tool discovery fails."""


class MCPDependencyError(MCPError):
    """Raised when MCP support is used without the optional dependency."""

    def __init__(self, cause: ImportError) -> None:
        super().__init__(f"{cause}. {_INSTALL_HINT}")
        self.__cause__ = cause


class MCPServer:
    """One MCP server connection that contributes ToolSpecs to a harness.

    Accepts a FastMCP ``ClientTransport`` and owns the FastMCP client built on
    it: connections open lazily, nested and concurrent entries share one
    session, and the final context exit closes the transport. Do not reuse one
    stateful transport object across several ``MCPServer`` wrappers; reuse the
    same wrapper when a session should be shared.
    """

    def __init__(
        self,
        transport: Any,
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
        self._transport = transport
        self._client: Any | None = None
        if type(self) is MCPServer:
            _validate_transport(transport)

    @property
    def id(self) -> str:
        """Stable readable identifier; falls back to a derived default."""
        return self._resolved_id or self._id or self._default_id()

    def resolve_id(self, existing_counts: dict[str, int]) -> None:
        """Resolve this server's final id using duplicate counts owned by the caller."""
        base_id = self._id or self._default_id()
        existing_counts[base_id] = existing_counts.get(base_id, 0) + 1
        self._resolved_id = base_id if existing_counts[base_id] == 1 else f"{base_id}-{existing_counts[base_id]}"

    def _default_id(self) -> str:
        """Derive a readable default id from the transport class name."""
        return _MCP_NAME_RE.sub("_", type(self._transport).__name__) or "mcp"

    def _resolve_transport(self) -> Any:
        """Return the FastMCP transport backing this server."""
        _validate_transport(self._transport)
        return self._transport

    def _ensure_client(self) -> Any:
        """Build the FastMCP client on first use and reuse it afterwards."""
        if self._client is None:
            client_type = _import_fastmcp_client()
            self._client = client_type(self._resolve_transport(), timeout=self.read_timeout, init_timeout=self.timeout)
        return self._client

    async def __aenter__(self) -> MCPServer:
        """Open or share the FastMCP client connection."""
        await self._ensure_client().__aenter__()
        return self

    async def __aexit__(self, exc_type: object = None, exc_val: object = None, exc_tb: object = None) -> None:
        """Release one connection reference; the final exit closes the transport.

        The final-close time bound lives in FastMCP's ``client_disconnect_timeout``
        setting (default 5s); re-check it when bumping the fastmcp pin.
        """
        client = self._client
        if client is None:
            return
        task = asyncio.current_task()
        pending_cancels = task.cancelling() if task is not None else 0
        await client.__aexit__(exc_type, exc_val, exc_tb)
        # FastMCP suppresses a CancelledError delivered while it awaits shutdown;
        # detect the new cancel request after cleanup and honor it.
        if task is not None and task.cancelling() > pending_cancels:
            raise asyncio.CancelledError

    async def list_tools(self) -> list[ToolSpec]:
        """Discover and convert the MCP server's current tool snapshot."""
        async with self:
            tools = await self._ensure_client().list_tools()
        seen: dict[str, str] = {}
        specs: list[ToolSpec] = []
        for tool in tools:
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
                kind="mcp",
                mcp=McpToolInfo(server_id=self.id, tool_name=original_name),
                max_retries=None,
            ))
        return specs

    async def call_tool(self, name: str, arguments: Json) -> ToolResult:
        """Call one MCP tool and normalize its result."""
        base_metadata = {"source": "mcp", "mcp_server_id": self.id, "mcp_tool_name": name}
        failure_types = _mcp_failure_types()
        try:
            async with self:
                result = await self._ensure_client().call_tool_mcp(name, arguments)
        except failure_types as exc:
            return ToolResult(False, str(exc), {**base_metadata, "error_type": "MCPError"})
        except (RuntimeError, ExceptionGroup) as exc:
            # FastMCP wraps transport failures; normalize only when the cause
            # chain contains a known failure so programming bugs still propagate.
            # ExceptionGroup (not BaseExceptionGroup) keeps groups that carry
            # cancellation or other BaseExceptions propagating structurally.
            if _find_known_failure(exc, failure_types) is None:
                raise
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


class MCPServerStdio(MCPServer):
    """MCP server reached through a stdio subprocess.

    The final context exit terminates the child process; entering again
    starts a new one.
    """

    def __init__(
        self,
        command: str,
        args: list[str] | None = None,
        *,
        env: dict[str, str] | None = None,
        cwd: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(None, **kwargs)
        self.command = command
        self.args = list(args or [])
        self.env = dict(env) if env is not None else None
        self.cwd = cwd

    def _default_id(self) -> str:
        """Return command and args as the default id."""
        return " ".join([self.command, *self.args])

    def _resolve_transport(self) -> Any:
        """Build a stdio transport whose final exit terminates the child."""
        transports = _import_fastmcp_transports()
        cwd = str(self.cwd) if self.cwd is not None else None
        return transports.StdioTransport(self.command, self.args, env=self.env, cwd=cwd, keep_alive=False)


class MCPServerSSE(MCPServer):
    """MCP server reached through SSE."""

    def __init__(self, url: str, *, headers: dict[str, str] | None = None, **kwargs: Any) -> None:
        super().__init__(None, **kwargs)
        self.url = url
        self.headers = dict(headers) if headers is not None else None

    def _default_id(self) -> str:
        """Return the URL as the default id."""
        return self.url

    def _resolve_transport(self) -> Any:
        """Build an SSE transport with separate connect and read limits."""
        transports = _import_fastmcp_transports()
        return transports.SSETransport(
            self.url,
            headers=self.headers,
            sse_read_timeout=self.read_timeout,
            httpx_client_factory=_httpx_factory(self.timeout, self.read_timeout),
        )


class MCPServerStreamableHTTP(MCPServer):
    """MCP server reached through streamable HTTP."""

    def __init__(self, url: str, *, headers: dict[str, str] | None = None, **kwargs: Any) -> None:
        super().__init__(None, **kwargs)
        self.url = url
        self.headers = dict(headers) if headers is not None else None

    def _default_id(self) -> str:
        """Return the URL as the default id."""
        return self.url

    def _resolve_transport(self) -> Any:
        """Build a streamable HTTP transport with separate connect and read limits."""
        # Streamable HTTP read limits come from the httpx factory and the
        # client-level MCP request timeout; the transport has no read parameter.
        transports = _import_fastmcp_transports()
        return transports.StreamableHttpTransport(
            self.url,
            headers=self.headers,
            httpx_client_factory=_httpx_factory(self.timeout, self.read_timeout),
        )


def _validate_transport(transport: Any) -> None:
    """Reject non-transport inputs once the optional dependency is available."""
    try:
        from fastmcp.client.transports import ClientTransport
    except ImportError:
        return
    if not isinstance(transport, ClientTransport):
        raise TypeError(
            f"MCPServer requires a fastmcp ClientTransport, got {type(transport).__name__}; "
            "wrap an in-process server with fastmcp.client.transports.FastMCPTransport"
        )


def _import_fastmcp_client() -> type[Any]:
    """Import the FastMCP Client class lazily."""
    try:
        from fastmcp import Client
    except ImportError as exc:
        raise MCPDependencyError(exc) from exc
    return Client


def _import_fastmcp_transports() -> Any:
    """Import the FastMCP client transports module lazily."""
    try:
        from fastmcp.client import transports
    except ImportError as exc:
        raise MCPDependencyError(exc) from exc
    return transports


def _httpx_factory(connect_timeout: float, read_timeout: float) -> Any:
    """Build an httpx client factory carrying separate connect and read limits."""

    def factory(
        headers: dict[str, str] | None = None,
        timeout: Any = None,
        auth: Any = None,
        follow_redirects: bool = True,
    ) -> httpx.AsyncClient:
        """Create the transport HTTP client with this server's timeouts."""
        return httpx.AsyncClient(
            headers=headers,
            auth=auth,
            follow_redirects=follow_redirects,
            timeout=httpx.Timeout(connect_timeout, read=read_timeout),
        )

    return factory


def _mcp_failure_types() -> tuple[type[BaseException], ...]:
    """Return the MCP protocol and transport failure types to normalize."""
    try:
        from anyio import BrokenResourceError, ClosedResourceError, EndOfStream
        from mcp.shared.exceptions import McpError
    except ImportError as exc:
        raise MCPDependencyError(exc) from exc
    return (McpError, ConnectionError, TimeoutError, BrokenResourceError, ClosedResourceError, EndOfStream, httpx.HTTPError)


def _find_known_failure(exc: BaseException, failure_types: tuple[type[BaseException], ...]) -> BaseException | None:
    """Return the first known MCP failure in an exception's group and cause tree.

    Deliberately walks explicit ``__cause__`` links only, not ``__context__``:
    a bug raised while handling a transport failure keeps that failure as
    implicit context, and normalizing it would hide the programming error.
    """
    stack: list[BaseException] = [exc]
    seen: set[int] = set()
    while stack:
        current = stack.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, failure_types):
            return current
        if isinstance(current, BaseExceptionGroup):
            stack.extend(current.exceptions)
        if current.__cause__ is not None:
            stack.append(current.__cause__)
    return None


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
