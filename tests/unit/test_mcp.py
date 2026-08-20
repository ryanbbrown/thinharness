from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any

import pytest
from fakes import FakeTracer, MultiCallClient, ScriptedModel, ScriptedSession, _fake_openai
from pydantic import BaseModel

from thinharness import (
    ApprovalDecision,
    FilesystemPlugin,
    Harness,
    HarnessConfig,
    HarnessError,
    Hook,
    ImageBlock,
    MCPError,
    MCPPlugin,
    MCPServer,
    MCPServerSSE,
    MCPServerStdio,
    MCPServerStreamableHTTP,
    ModelTurn,
    PluginBinding,
    PluginContribution,
    SubAgentConfig,
    SubagentsPlugin,
    TextBlock,
    ToolOrigin,
    TracingOptions,
)
from thinharness.providers import ModelToolCall, ToolOutput
from thinharness.tools.base import Json, ToolResult, ToolSpec

pytest.importorskip("mcp")
pytest.importorskip("fastmcp")
pytest.importorskip("fastmcp.server")

import httpx  # noqa: E402
from fastmcp import Client, FastMCP  # noqa: E402
from fastmcp.client.transports import FastMCPTransport  # noqa: E402

from thinharness.tools.mcp import _httpx_factory  # noqa: E402


class ObservedMCPServer(MCPServer):
    """MCPServer that counts its public lifecycle calls for assertions."""

    def __init__(self, transport: Any, **kwargs: Any) -> None:
        super().__init__(transport, **kwargs)
        self.entered = 0
        self.exited = 0
        self.list_calls = 0
        self.call_records: list[tuple[str, Json]] = []
        self.backend_log: dict[str, int] = {}

    async def __aenter__(self) -> MCPServer:
        """Count public context entries."""
        self.entered += 1
        return await super().__aenter__()

    async def __aexit__(self, *exc: object) -> None:
        """Count public context exits."""
        self.exited += 1
        return await super().__aexit__(*exc)

    async def list_tools(self, *, server_id: str | None = None) -> list[ToolSpec]:
        """Count discovery calls."""
        self.list_calls += 1
        return await super().list_tools(server_id=server_id)


def _lifespan_tracker() -> tuple[Any, dict[str, int]]:
    """Return a FastMCP lifespan and a log counting connection starts and stops."""
    log = {"starts": 0, "stops": 0}

    @asynccontextmanager
    async def lifespan(server: Any):
        log["starts"] += 1
        try:
            yield {}
        finally:
            log["stops"] += 1

    return lifespan, log


def _echo_handler(tool_name: str, records: list[tuple[str, Json]]) -> Any:
    """Build a recording echo tool function for an in-process backend."""

    def handler(value: str = "") -> str:
        records.append((tool_name, {"value": value}))
        return f"{tool_name}:{value}"

    return handler


def observed_server(*tool_names: str, id: str = "fake", **kwargs: Any) -> ObservedMCPServer:
    """Build an MCPServer over a real in-process FastMCP backend with echo tools."""
    records: list[tuple[str, Json]] = []
    lifespan, log = _lifespan_tracker()
    backend = FastMCP("fake-backend", lifespan=lifespan)
    for tool_name in tool_names or ("remote",):
        backend.tool(_echo_handler(tool_name, records), name=tool_name)
    server = ObservedMCPServer(FastMCPTransport(backend), id=id, **kwargs)
    server.call_records = records
    server.backend_log = log
    return server


_SCRIPTS: dict[int, dict[str, Any]] = {}


def _script_key(backend: Any) -> int:
    return id(backend)


async def _scripted_list_tools(self: Any, max_pages: int = 250) -> list[Any]:
    """Return the scripted tool declarations for this client's backend."""
    script = _SCRIPTS[_script_key(self.transport.server)]
    return [SimpleNamespace(name=name, description=f"{name} tool", inputSchema=schema) for name, schema in script["schemas"].items()]


async def _scripted_call_tool_mcp(self: Any, name: str, arguments: dict[str, Any], **_kwargs: Any) -> Any:
    """Return or raise the scripted outcome for one tool call."""
    from mcp import types

    script = _SCRIPTS[_script_key(self.transport.server)]
    script["records"].append((name, arguments))
    outcome = script["results"].get(name)
    if isinstance(outcome, BaseException):
        raise outcome
    if outcome is not None:
        return outcome
    return types.CallToolResult(content=[types.TextContent(type="text", text=f"{name}:{arguments}")], isError=False)


def scripted_server(
    monkeypatch: pytest.MonkeyPatch,
    tool_schemas: dict[str, Any],
    results: dict[str, Any] | None = None,
    *,
    id: str = "fake",
    **kwargs: Any,
) -> ObservedMCPServer:
    """Build an MCPServer whose protocol responses are scripted at the FastMCP client API."""
    backend = FastMCP("scripted-backend")
    monkeypatch.setattr(Client, "list_tools", _scripted_list_tools)
    monkeypatch.setattr(Client, "call_tool_mcp", _scripted_call_tool_mcp)
    server = ObservedMCPServer(FastMCPTransport(backend), id=id, **kwargs)
    records: list[tuple[str, Json]] = []
    server.call_records = records
    monkeypatch.setitem(
        _SCRIPTS,
        _script_key(backend),
        {"schemas": dict(tool_schemas), "results": dict(results or {}), "records": records},
    )
    return server


class FailingListServer(ObservedMCPServer):
    """Server double whose discovery fails after entering the context."""

    async def list_tools(self, *, server_id: str | None = None) -> list[ToolSpec]:
        """Fail discovery after entering the context."""
        del server_id
        async with self:
            raise MCPError("list failed")


class SequenceSession:
    """Script a start turn followed by tool continuation turns."""

    def __init__(self, start_turn: ModelTurn, *continue_turns: ModelTurn, dump_state: Json | None = None) -> None:
        self.start_turn = start_turn
        self.continue_turns = list(continue_turns)
        self.tool_outputs: list[list[ToolOutput]] = []
        self.requests_made = 0
        self._dump_state = dump_state if dump_state is not None else {"kind": "scripted", "version": 1, "model": "scripted-model"}

    async def start(self, prompt, constants, *, previous_response_id=None, notices=None):
        """Return the scripted first turn."""
        self.requests_made += 1
        return self.start_turn

    async def continue_with_tools(self, outputs, constants, *, notices=None):
        """Record tool outputs and return the next scripted turn."""
        self.requests_made += 1
        self.tool_outputs.append(outputs)
        if not self.continue_turns:
            raise AssertionError("unexpected tool continuation")
        return self.continue_turns.pop(0)

    async def continue_with_user_content(self, text, constants, *, notices=None):
        """Return the scripted turn for a resumed prompt; no tests expect corrections."""
        if self.requests_made:
            raise AssertionError("unexpected user-text correction")
        self.requests_made += 1
        return self.start_turn

    def dump_state(self):
        """Return scripted resume state."""
        return dict(self._dump_state)


def _schema() -> Json:
    """Return a minimal MCP input schema."""
    return {"type": "object", "properties": {"value": {"type": "string"}}, "$schema": "https://json-schema.org/draft/2020-12/schema"}


async def test_mcp_tool_discovery_and_result_metadata(monkeypatch) -> None:
    """MCP tools are converted, cleaned, and return stable metadata."""
    original_schema = _schema()
    server = scripted_server(monkeypatch, {"hello.world": original_schema}, tool_prefix="git")

    tools = await server.list_tools()
    assert tools[0].name == "git_hello_world"
    assert tools[0].parameters is not original_schema
    assert "$schema" not in tools[0].parameters
    assert tools[0].parameters["additionalProperties"] is False
    assert original_schema["$schema"]

    result = await server.call_tool("hello.world", {"value": "ok"})
    assert result == ToolResult(
        True,
        "hello.world:{'value': 'ok'}",
        {"source": "mcp", "mcp_server_id": "fake", "mcp_tool_name": "hello.world"},
    )


async def test_tool_filters_apply_before_conversion(monkeypatch) -> None:
    """include_tools and exclude_tools filter original MCP names."""
    server = scripted_server(
        monkeypatch,
        {"keep": _schema(), "drop": _schema(), "skip": _schema()},
        tool_prefix="remote",
        include_tools=["keep", "drop"],
        exclude_tools=["drop"],
    )

    tools = await server.list_tools()

    assert [tool.name for tool in tools] == ["remote_keep"]


async def test_stdio_smoke() -> None:
    """Real stdio transport starts, lists tools, calls a tool, and shuts down."""
    server_code = """
from mcp.server.fastmcp import FastMCP
mcp = FastMCP("echo")
@mcp.tool()
def echo(value: str) -> str:
    return value
if __name__ == "__main__":
    mcp.run()
"""
    server = MCPServerStdio(sys.executable, ["-c", server_code])

    async with server:
        tools = await server.list_tools()
        result = await server.call_tool("echo", {"value": "ok"})

    assert [tool.name for tool in tools] == ["echo"]
    assert result.ok is True
    assert result.metadata["mcp_tool_name"] == "echo"


async def test_mcp_tool_error_is_retryable(monkeypatch) -> None:
    """MCP isError results use the harness retry metadata path."""
    from mcp import types

    error_result = types.CallToolResult(content=[types.TextContent(type="text", text="try again")], isError=True)
    server = scripted_server(monkeypatch, {"error": _schema()}, {"error": error_result})

    result = await server.call_tool("error", {})

    assert result.ok is False
    assert result.metadata == {
        "source": "mcp",
        "mcp_server_id": "fake",
        "mcp_tool_name": "error",
        "error_type": "MCPToolError",
        "retry": True,
    }


async def test_mcp_transport_exception_returns_metadata(monkeypatch) -> None:
    """Unexpected transport/session errors still return MCP metadata."""
    server = scripted_server(monkeypatch, {"transport": _schema()}, {"transport": ConnectionError("transport closed")})

    result = await server.call_tool("transport", {})

    assert result.ok is False
    assert result.content == "transport closed"
    assert result.metadata == {
        "source": "mcp",
        "mcp_server_id": "fake",
        "mcp_tool_name": "transport",
        "error_type": "MCPError",
    }


async def test_wrapped_transport_failure_is_normalized(monkeypatch) -> None:
    """Wrapper errors whose cause chain holds a transport failure are normalized.

    Sibling exception noise in the group does not defeat normalization: after
    a transport drop, anyio teardown commonly raises secondary errors alongside
    the root failure, and the retryable envelope is the useful outcome.
    """
    wrapper = RuntimeError("Client failed to connect: stream closed")
    wrapper.__cause__ = ConnectionError("stream closed")
    grouped = BaseExceptionGroup("boom", [ValueError("noise"), wrapper])
    server = scripted_server(monkeypatch, {"wrapped": _schema()}, {"wrapped": grouped})

    result = await server.call_tool("wrapped", {})

    assert result.ok is False
    assert result.metadata["error_type"] == "MCPError"


async def test_group_of_known_failures_is_normalized(monkeypatch) -> None:
    """A group holding only transport failures becomes an MCPError envelope."""
    grouped = BaseExceptionGroup("boom", [ConnectionError("dropped"), TimeoutError("late")])
    server = scripted_server(monkeypatch, {"grouped": _schema()}, {"grouped": grouped})

    result = await server.call_tool("grouped", {})

    assert result.ok is False
    assert result.metadata["error_type"] == "MCPError"


async def test_group_with_cancellation_propagates(monkeypatch) -> None:
    """A group carrying cancellation is never normalized to an MCPError result."""
    grouped = BaseExceptionGroup("boom", [asyncio.CancelledError(), ConnectionError("dropped")])
    server = scripted_server(monkeypatch, {"cancelled": _schema()}, {"cancelled": grouped})

    with pytest.raises(BaseExceptionGroup):
        await server.call_tool("cancelled", {})


async def test_mcp_programming_bug_propagates(monkeypatch) -> None:
    """Local bugs are not hidden as normal MCP tool failures."""
    server = scripted_server(monkeypatch, {"bug": _schema()}, {"bug": RuntimeError("programming bug")})

    with pytest.raises(RuntimeError, match="programming bug"):
        await server.call_tool("bug", {})


async def test_mcp_protocol_error_returns_metadata(monkeypatch) -> None:
    """MCP protocol errors become failed results with MCPError metadata."""
    from mcp.shared.exceptions import McpError
    from mcp.types import ErrorData

    server = scripted_server(monkeypatch, {"proto": _schema()}, {"proto": McpError(ErrorData(code=-32603, message="proto boom"))})

    result = await server.call_tool("proto", {})

    assert result.ok is False
    assert result.content == "proto boom"
    assert result.metadata == {
        "source": "mcp",
        "mcp_server_id": "fake",
        "mcp_tool_name": "proto",
        "error_type": "MCPError",
    }


async def test_structured_content_wins_over_blocks(monkeypatch) -> None:
    """Successful structuredContent is returned as a JSON string."""
    from mcp import types

    scripted = types.CallToolResult(
        content=[types.TextContent(type="text", text="ignored")],
        structuredContent={"answer": 42, "note": "café"},
        isError=False,
    )
    server = scripted_server(monkeypatch, {"calc": _schema()}, {"calc": scripted})

    result = await server.call_tool("calc", {})

    assert result.ok is True
    assert result.content == '{"answer": 42, "note": "café"}'
    assert json.loads(result.content) == {"answer": 42, "note": "café"}


@pytest.mark.parametrize(
    ("block_builder", "expected"),
    [
        pytest.param(lambda types: types.TextContent(type="text", text="plain"), "plain", id="text"),
        pytest.param(lambda types: types.ImageContent(type="image", data="aGk=", mimeType="image/png"), (ImageBlock(b"hi", "image/png"),), id="image"),
        pytest.param(lambda types: types.AudioContent(type="audio", data="aGk=", mimeType="audio/wav"), "[audio: audio/wav]", id="audio"),
        pytest.param(
            lambda types: types.EmbeddedResource(
                type="resource",
                resource=types.TextResourceContents(uri="file:///notes.txt", text="hello"),
            ),
            "[resource: file:///notes.txt]",
            id="embedded-resource",
        ),
        pytest.param(
            lambda types: types.ResourceLink(type="resource_link", uri="https://example.com/doc", name="doc"),
            "[resource: https://example.com/doc]",
            id="resource-link",
        ),
    ],
)
async def test_content_block_conversion(monkeypatch, block_builder, expected: str | tuple[ImageBlock, ...]) -> None:
    """Each supported MCP content block keeps its text conversion."""
    from mcp import types

    scripted = types.CallToolResult(content=[block_builder(types)], isError=False)
    server = scripted_server(monkeypatch, {"blocks": _schema()}, {"blocks": scripted})

    result = await server.call_tool("blocks", {})

    assert result.ok is True
    assert result.content == expected


async def test_mixed_content_blocks_preserve_order(monkeypatch) -> None:
    """Mixed content blocks convert in order, joined by newlines."""
    from mcp import types

    scripted = types.CallToolResult(
        content=[
            types.TextContent(type="text", text="first"),
            types.ImageContent(type="image", data="aGk=", mimeType="image/png"),
            types.EmbeddedResource(
                type="resource",
                resource=types.BlobResourceContents(uri="file:///data.bin", blob="aGk="),
            ),
            types.TextContent(type="text", text="last"),
        ],
        isError=False,
    )
    server = scripted_server(monkeypatch, {"mixed": _schema()}, {"mixed": scripted})

    result = await server.call_tool("mixed", {})

    assert result.content == (
        TextBlock("first"),
        ImageBlock(b"hi", "image/png"),
        TextBlock("[resource: file:///data.bin]"),
        TextBlock("last"),
    )


async def test_tool_error_with_structured_content_stays_error_text(monkeypatch) -> None:
    """isError results ignore structuredContent and keep the retry envelope."""
    from mcp import types

    scripted = types.CallToolResult(
        content=[types.TextContent(type="text", text="tool blew up")],
        structuredContent={"partial": True},
        isError=True,
    )
    server = scripted_server(monkeypatch, {"errored": _schema()}, {"errored": scripted})

    result = await server.call_tool("errored", {})

    assert result.ok is False
    assert result.content == "tool blew up"
    assert result.metadata["error_type"] == "MCPToolError"
    assert result.metadata["retry"] is True


async def test_invalid_input_schema_names_tool(monkeypatch) -> None:
    """Invalid MCP schemas fail with a clear tool-specific error."""
    server = scripted_server(monkeypatch, {"bad": None})

    with pytest.raises(MCPError, match="bad.*inputSchema"):
        await server.list_tools()


async def test_sanitized_name_collision_raises(monkeypatch) -> None:
    """Sanitizer collisions include both original MCP names."""
    server = scripted_server(monkeypatch, {"foo.bar": _schema(), "foo/bar": _schema()})

    with pytest.raises(MCPError, match="foo\\.bar.*foo/bar"):
        await server.list_tools()


async def test_sanitizer_preserves_valid_edge_characters(monkeypatch) -> None:
    """Leading underscores and dashes are valid function-tool name characters."""
    server = scripted_server(monkeypatch, {"_foo": _schema(), "foo": _schema(), "-": _schema()})

    tools = await server.list_tools()

    assert [tool.name for tool in tools] == ["_foo", "foo", "-"]


def test_generic_transport_rejects_invalid_input() -> None:
    """The generic MCPServer accepts only FastMCP client transports."""
    with pytest.raises(TypeError, match="ClientTransport"):
        MCPServer("http://localhost/mcp")
    with pytest.raises(TypeError, match="FastMCPTransport"):
        MCPServer(FastMCP("bare-server"))


def test_generic_default_id_derivation() -> None:
    """Generic default ids come from the transport class name and stay non-empty."""
    backend = FastMCP("id-backend")
    assert MCPServer(FastMCPTransport(backend)).id == "FastMCPTransport"

    punctuation_transport = type("???", (FastMCPTransport,), {})(backend)
    assert MCPServer(punctuation_transport).id == "___"

    unnamed_transport = type("", (FastMCPTransport,), {})(backend)
    assert MCPServer(unnamed_transport).id == "mcp"


async def test_generic_default_id_collision_suffix(tmp_path) -> None:
    """Two generic servers with the same transport class get stable suffixes."""

    def make_backend(tool_name: str) -> FastMCP:
        backend = FastMCP(f"{tool_name}-backend")
        backend.tool(_echo_handler(tool_name, []), name=tool_name)
        return backend

    first = MCPServer(FastMCPTransport(make_backend("one")))
    second = MCPServer(FastMCPTransport(make_backend("two")))
    harness = Harness(HarnessConfig(root=tmp_path), plugins=[MCPPlugin(servers=[first, second])], model=_fake_openai(MultiCallClient([])))

    await harness.connect()
    await harness.aclose()

    metadata = {tool.name: tool.origin.source for tool in harness.tools if tool.origin is not None}
    assert metadata == {"one": "FastMCPTransport", "two": "FastMCPTransport-2"}


async def test_modern_inprocess_discovery_and_call() -> None:
    """A modern fastmcp.FastMCP server works through FastMCPTransport."""
    server = observed_server("remote")

    async with server:
        tools = await server.list_tools()
        result = await server.call_tool("remote", {"value": "ok"})

    assert [tool.name for tool in tools] == ["remote"]
    assert result.ok is True
    assert json.loads(result.content) == {"result": "remote:ok"}
    assert server.call_records == [("remote", {"value": "ok"})]


async def test_legacy_inprocess_discovery_and_call() -> None:
    """The official SDK's legacy FastMCP server works through FastMCPTransport."""
    from mcp.server.fastmcp import FastMCP as LegacyFastMCP

    legacy = LegacyFastMCP("legacy")

    @legacy.tool()
    def ping() -> str:
        return "pong"

    server = MCPServer(FastMCPTransport(legacy), id="legacy")

    async with server:
        tools = await server.list_tools()
        result = await server.call_tool("ping", {})

    assert [tool.name for tool in tools] == ["ping"]
    assert result.ok is True
    assert json.loads(result.content) == {"result": "pong"}


async def test_semley_shaped_inprocess_harness_integration(tmp_path) -> None:
    """An in-process governed server exposes only included tools through the harness."""
    backend = FastMCP("semley")
    steps: list[str] = []

    def step(action: str) -> str:
        steps.append(action)
        return f"stepped:{action}"

    def reset_session() -> str:
        return "reset"

    def hidden() -> str:
        return "nope"

    backend.tool(step)
    backend.tool(reset_session)
    backend.tool(hidden)
    server = MCPServer(FastMCPTransport(backend), id="semley", include_tools=["step", "reset_session"])
    harness = Harness(
        HarnessConfig(root=tmp_path),
        plugins=[MCPPlugin(servers=[server])],
        model=_fake_openai(MultiCallClient([("step", '{"action":"go"}')])),
    )

    result = await harness.run("go")
    await harness.aclose()

    assert sorted(tool.name for tool in harness.tools) == ["reset_session", "step"]
    assert steps == ["go"]
    assert result.text == "done"


async def test_lifecycle_reference_counted() -> None:
    """Nested entries share one connection, close on the last exit, and can reconnect."""
    server = observed_server("remote")

    await server.__aenter__()
    await server.__aenter__()
    await server.__aexit__(None, None, None)

    assert server.backend_log == {"starts": 1, "stops": 0}

    await server.__aexit__(None, None, None)

    assert server.backend_log == {"starts": 1, "stops": 1}

    async with server:
        result = await server.call_tool("remote", {"value": "again"})

    assert result.ok is True
    assert server.backend_log == {"starts": 2, "stops": 2}


async def test_concurrent_entry_shares_one_connection() -> None:
    """Concurrent entries of one wrapper share a single server connection."""
    server = observed_server("remote")

    async def use() -> None:
        async with server:
            result = await server.call_tool("remote", {"value": "x"})
            assert result.ok is True

    await asyncio.gather(*(use() for _ in range(5)))

    assert server.backend_log == {"starts": 1, "stops": 1}


async def test_first_connect_cancellation_is_bounded_and_propagates() -> None:
    """Cancelling the first connection propagates and leaves the wrapper reusable."""
    starts = {"count": 0}

    @asynccontextmanager
    async def slow_first_start(server: Any):
        starts["count"] += 1
        if starts["count"] == 1:
            await asyncio.sleep(30)
        yield {}

    backend = FastMCP("slow-start", lifespan=slow_first_start)
    backend.tool(_echo_handler("t", []), name="t")
    server = MCPServer(FastMCPTransport(backend))
    enter_task = asyncio.create_task(server.__aenter__())
    await asyncio.sleep(0.05)
    started = time.monotonic()

    enter_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await enter_task

    assert time.monotonic() - started < 5

    async with server:
        result = await server.call_tool("t", {"value": "again"})
    assert result.ok is True
    assert result.content == '{"result": "t:again"}'


async def test_final_exit_cancellation_propagates() -> None:
    """A cancel delivered during final close is honored after bounded cleanup."""

    @asynccontextmanager
    async def slow_stop(server: Any):
        try:
            yield {}
        finally:
            await asyncio.sleep(1)

    backend = FastMCP("slow-stop", lifespan=slow_stop)
    backend.tool(_echo_handler("t", []), name="t")
    server = MCPServer(FastMCPTransport(backend))
    await server.__aenter__()

    exit_task = asyncio.create_task(server.__aexit__(None, None, None))
    await asyncio.sleep(0.1)
    started = time.monotonic()
    exit_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await exit_task

    assert time.monotonic() - started < 5

    async with server:
        result = await server.call_tool("t", {"value": "back"})
    assert result.ok is True
    assert result.content == '{"result": "t:back"}'


async def test_body_cancellation_propagates_through_final_close() -> None:
    """A cancellation raised inside the context body survives the final close."""
    lifespan, log = _lifespan_tracker()
    backend = FastMCP("body-cancel", lifespan=lifespan)
    server = MCPServer(FastMCPTransport(backend))

    async def body() -> None:
        async with server:
            await asyncio.sleep(30)

    task = asyncio.create_task(body())
    await asyncio.sleep(0.1)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert log == {"starts": 1, "stops": 1}


async def test_body_exception_propagates_and_closes() -> None:
    """A body exception passes through the final close unchanged and unsuppressed."""
    lifespan, log = _lifespan_tracker()
    backend = FastMCP("body-error", lifespan=lifespan)
    server = MCPServer(FastMCPTransport(backend))

    with pytest.raises(ValueError, match="body boom"):
        async with server:
            raise ValueError("body boom")

    assert log == {"starts": 1, "stops": 1}

    async with server:
        pass
    assert log == {"starts": 2, "stops": 2}


async def test_final_close_is_bounded_without_cancellation(monkeypatch) -> None:
    """A hanging server teardown cannot wedge an uncancelled final close."""
    import fastmcp

    monkeypatch.setattr(fastmcp.settings, "client_disconnect_timeout", 0.5)

    @asynccontextmanager
    async def hanging_stop(server: Any):
        try:
            yield {}
        finally:
            await asyncio.sleep(60)

    backend = FastMCP("hang-stop", lifespan=hanging_stop)
    server = MCPServer(FastMCPTransport(backend))
    await server.__aenter__()
    started = time.monotonic()

    await server.__aexit__(None, None, None)

    assert time.monotonic() - started < 3


async def test_stdio_child_exits_on_final_close_and_reconnects() -> None:
    """Final close terminates the stdio child; re-entering starts a new one."""
    server_code = """
from mcp.server.fastmcp import FastMCP
import os
mcp = FastMCP("pid")
@mcp.tool()
def pid() -> int:
    return os.getpid()
if __name__ == "__main__":
    mcp.run()
"""
    server = MCPServerStdio(sys.executable, ["-c", server_code])

    async with server:
        first_pid = json.loads((await server.call_tool("pid", {})).content)["result"]

    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            os.kill(first_pid, 0)
        except ProcessLookupError:
            break
        await asyncio.sleep(0.05)
    else:
        pytest.fail(f"stdio child {first_pid} still alive after final close")

    async with server:
        second_pid = json.loads((await server.call_tool("pid", {})).content)["result"]

    assert second_pid != first_pid


async def test_stdio_transport_drop_returns_mcp_error() -> None:
    """A server process dying mid-call yields a failed MCPError result."""
    server_code = """
from mcp.server.fastmcp import FastMCP
import os
mcp = FastMCP("dying")
@mcp.tool()
def die() -> str:
    os._exit(1)
if __name__ == "__main__":
    mcp.run()
"""
    server = MCPServerStdio(sys.executable, ["-c", server_code])

    async with server:
        result = await server.call_tool("die", {})

    assert result.ok is False
    assert result.metadata["error_type"] == "MCPError"


async def test_stdio_read_timeout_bounds_tool_calls() -> None:
    """read_timeout bounds a stalled tool call and returns an MCPError result."""
    server_code = """
from mcp.server.fastmcp import FastMCP
import time
mcp = FastMCP("slow")
@mcp.tool()
def slow() -> str:
    time.sleep(30)
    return "late"
if __name__ == "__main__":
    mcp.run()
"""
    # The session read timeout also bounds the MCP initialize request, so it
    # must leave room for child startup on slow CI runners.
    server = MCPServerStdio(sys.executable, ["-c", server_code], read_timeout=3.0)

    async with server:
        started = time.monotonic()
        result = await server.call_tool("slow", {})
        elapsed = time.monotonic() - started

    assert result.ok is False
    assert result.metadata["error_type"] == "MCPError"
    assert elapsed < 15


async def test_stdio_init_stall_is_bounded() -> None:
    """timeout bounds initialization against a server that never speaks MCP."""
    server = MCPServerStdio(sys.executable, ["-c", "import time; time.sleep(60)"], timeout=0.5)
    started = time.monotonic()

    with pytest.raises(Exception):  # noqa: B017 - only boundedness is contractual
        await server.__aenter__()

    assert time.monotonic() - started < 15


async def _assert_stalled_http_enter_is_read_bounded(server: MCPServer) -> None:
    """Enter a wrapper against a never-responding HTTP server and expect a fast failure."""
    started = time.monotonic()

    with pytest.raises(Exception):  # noqa: B017 - only the read bound is contractual
        await server.__aenter__()

    # Well under the wrappers' 30s connect timeout: only the 0.5s read
    # timeout can produce a failure this fast against a stalled server.
    assert time.monotonic() - started < 5


@asynccontextmanager
async def _stalled_http_server():
    """Serve a TCP endpoint that accepts connections and never responds."""
    writers: list[asyncio.StreamWriter] = []

    async def stall(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        writers.append(writer)
        await reader.read(-1)

    tcp_server = await asyncio.start_server(stall, "127.0.0.1", 0)
    try:
        yield tcp_server.sockets[0].getsockname()[1]
    finally:
        tcp_server.close()
        for writer in writers:
            writer.close()


async def test_streamable_http_read_timeout_applies() -> None:
    """The streamable HTTP wrapper's read_timeout reaches the HTTP client."""
    async with _stalled_http_server() as port:
        server = MCPServerStreamableHTTP(f"http://127.0.0.1:{port}/mcp", timeout=30.0, read_timeout=0.5)
        await _assert_stalled_http_enter_is_read_bounded(server)


async def test_sse_read_timeout_applies() -> None:
    """The SSE wrapper's read_timeout reaches the HTTP client."""
    async with _stalled_http_server() as port:
        server = MCPServerSSE(f"http://127.0.0.1:{port}/sse", timeout=30.0, read_timeout=0.5)
        await _assert_stalled_http_enter_is_read_bounded(server)


def test_httpx_factory_maps_connect_and_read_timeouts() -> None:
    """The HTTP client factory keeps connect and read limits separate."""
    factory = _httpx_factory(5.0, 300.0)

    client = factory(headers={"x": "y"}, timeout=httpx.Timeout(9.0))

    assert client.timeout == httpx.Timeout(5.0, read=300.0)


async def test_harness_connects_mcp_once_across_async_runs(tmp_path, monkeypatch) -> None:
    """Harness runs reuse the discovered MCP tools until aclose."""
    server = scripted_server(monkeypatch, {"remote": _schema()})
    client = MultiCallClient([("remote", '{"value":"ok"}')])
    harness = Harness(HarnessConfig(root=tmp_path), plugins=[MCPPlugin(servers=[server])], model=_fake_openai(client))

    assert harness.tools == []
    result = await harness.run("go")
    second = await harness.run("done")
    await harness.aclose()

    assert result.text == "done"
    assert second.text == "done"
    assert server.list_calls == 1
    assert server.entered == 3
    assert server.exited == 3
    assert server.call_records == [("remote", {"value": "ok"})]


def test_mcp_plugin_validates_server_collection() -> None:
    """MCP server configuration is ordered, typed, and unique by identity."""
    server = MCPServerStdio("unused")

    with pytest.raises(TypeError, match="ordered sequence"):
        MCPPlugin(servers={server})  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="only MCPServer"):
        MCPPlugin(servers=[object()])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="same server object"):
        MCPPlugin(servers=[server, server])


def test_mcp_plugin_name_is_fixed() -> None:
    """Instances and classes cannot replace or remove the MCP plugin name."""
    plugin = MCPPlugin(servers=[])

    assert plugin.name == "mcp"
    with pytest.raises(AttributeError, match="fixed"):
        plugin.name = "renamed"
    assert plugin.name == "mcp"

    with pytest.raises(AttributeError, match="fixed"):
        MCPPlugin.name = "renamed"
    assert plugin.name == "mcp"

    with pytest.raises(AttributeError, match="fixed"):
        del MCPPlugin.name
    assert plugin.name == "mcp"

    with pytest.raises(TypeError, match="cannot override"):
        type("RenamedMCPPlugin", (MCPPlugin,), {"name": "renamed"})
    assert plugin.name == "mcp"

    class CustomMCPPlugin(MCPPlugin):
        pass

    custom_plugin = CustomMCPPlugin(servers=[])
    with pytest.raises(AttributeError, match="fixed"):
        CustomMCPPlugin.name = "renamed"
    assert custom_plugin.name == "mcp"

    with pytest.raises(AttributeError, match="fixed"):
        del CustomMCPPlugin.name
    assert custom_plugin.name == "mcp"


def test_mcp_plugin_name_is_unique_and_config_path_is_removed(tmp_path, monkeypatch) -> None:
    """One fixed-name MCP plugin is allowed and the old config path is rejected."""
    first = scripted_server(monkeypatch, {"first": _schema()})
    second = scripted_server(monkeypatch, {"second": _schema()})

    with pytest.raises(ValueError, match="duplicate plugin name: mcp"):
        Harness(
            HarnessConfig(root=tmp_path),
            plugins=[MCPPlugin(servers=[first]), MCPPlugin(servers=[second])],
            model=ScriptedModel([]),
        )
    assert "mcp_servers" not in HarnessConfig.model_fields


async def test_empty_mcp_plugin_connects_without_tools(tmp_path) -> None:
    """An empty MCP plugin is a valid connected contribution."""
    harness = Harness(
        HarnessConfig(root=tmp_path),
        plugins=[MCPPlugin(servers=[])],
        model=ScriptedModel([]),
    )

    await harness.connect()
    assert harness.tools == []
    await harness.aclose()


async def test_explicit_connect_does_not_reconnect_on_run(tmp_path, monkeypatch) -> None:
    """Explicit connect discovers MCP tools once before run."""
    server = scripted_server(monkeypatch, {"remote": _schema()})
    harness = Harness(HarnessConfig(root=tmp_path), plugins=[MCPPlugin(servers=[server])], model=_fake_openai(MultiCallClient([])))

    await harness.connect()
    await harness.connect()
    result = await harness.run("go")
    await harness.aclose()

    assert result.text == ""
    assert server.list_calls == 1


async def test_is_error_drives_harness_retry(tmp_path, monkeypatch) -> None:
    """MCP isError envelopes feed the harness tool retry accounting."""
    from mcp import types

    error_result = types.CallToolResult(content=[types.TextContent(type="text", text="try again")], isError=True)
    server = scripted_server(monkeypatch, {"error": _schema()}, {"error": error_result})
    run_end = []
    session = SequenceSession(
        ModelTurn(tool_calls=[ModelToolCall(id="call_1", name="error", arguments="{}")], raw={"id": "start"}),
        ModelTurn(tool_calls=[ModelToolCall(id="call_2", name="error", arguments="{}")], raw={"id": "retry"}),
    )
    harness = Harness(
        HarnessConfig(root=tmp_path, tool_retries=1),
        plugins=[MCPPlugin(servers=[server])],
        model=ScriptedModel([session]),
        hooks=[Hook("run_end", lambda ctx: run_end.append((ctx.stop_reason, dict(ctx.usage.tool_retries))))],
    )

    with pytest.raises(HarnessError, match="exceeded max_retries=1"):
        await harness.run("go")
    await harness.aclose()

    assert len(session.tool_outputs) == 1
    assert session.tool_outputs[0][0].output
    assert run_end == [("tool_retries_exceeded", {"error": 2})]


async def test_failed_second_server_entry_rolls_back_and_retries(tmp_path, monkeypatch) -> None:
    """A second-server entry failure closes the first and leaves no contribution."""
    first = scripted_server(monkeypatch, {"first": _schema()}, id="first")

    class FailOnceEnterServer(ObservedMCPServer):
        async def __aenter__(self) -> MCPServer:
            """Fail the first outer entry, then use the normal wrapper."""
            self.entered += 1
            if self.entered == 1:
                raise MCPError("entry failed")
            return await MCPServer.__aenter__(self)

    scripted_second = scripted_server(monkeypatch, {"second": _schema()}, id="second")
    second = FailOnceEnterServer(scripted_second._transport, id="second")
    harness = Harness(
        HarnessConfig(root=tmp_path),
        plugins=[MCPPlugin(servers=[first, second])],
        model=ScriptedModel([]),
    )

    with pytest.raises(MCPError, match="entry failed"):
        await harness.connect()
    assert harness.tools == []
    assert first.exited == 2

    await harness.connect()
    assert [tool.name for tool in harness.tools] == ["first", "second"]
    await harness.aclose()


async def test_partial_connect_failure_cleans_up(tmp_path) -> None:
    """A later discovery failure closes earlier servers and permits retry."""
    first = observed_server("ok")

    class FailOnceListServer(ObservedMCPServer):
        def __init__(self, transport: Any, **kwargs: Any) -> None:
            super().__init__(transport, **kwargs)
            self.attempts = 0

        async def list_tools(self, *, server_id: str | None = None) -> list[ToolSpec]:
            self.attempts += 1
            if self.attempts == 1:
                async with self:
                    raise MCPError("list failed")
            return await super().list_tools(server_id=server_id)

    backend = FastMCP("failing-backend")
    backend.tool(_echo_handler("recovered", []), name="recovered")
    second = FailOnceListServer(FastMCPTransport(backend), id="failing")
    harness = Harness(HarnessConfig(root=tmp_path), plugins=[MCPPlugin(servers=[first, second])], model=_fake_openai(MultiCallClient([])))

    with pytest.raises(MCPError, match="list failed"):
        await harness.connect()

    assert harness.tools == []
    assert first.exited == 2
    assert second.exited == 2

    await harness.connect()
    assert [tool.name for tool in harness.tools] == ["ok", "recovered"]
    await harness.aclose()


async def test_cancellation_during_failed_discovery_cleanup_propagates_and_retries(tmp_path) -> None:
    """Caller cancellation during failed discovery cleanup wins after cleanup."""
    cleanup_started = asyncio.Event()
    release_cleanup = asyncio.Event()

    class FailingDiscoverySlowCleanupServer(ObservedMCPServer):
        def __init__(self, transport: Any, **kwargs: Any) -> None:
            super().__init__(transport, **kwargs)
            self.attempts = 0
            self.block_cleanup = True

        async def list_tools(self, *, server_id: str | None = None) -> list[ToolSpec]:
            self.attempts += 1
            if self.attempts == 1:
                raise MCPError("discovery failed")
            return await super().list_tools(server_id=server_id)

        async def __aexit__(self, *exc: object) -> None:
            if self.block_cleanup:
                self.block_cleanup = False
                cleanup_started.set()
                await release_cleanup.wait()
            await super().__aexit__(*exc)

    backend = FastMCP("failed-discovery-slow-cleanup")
    backend.tool(_echo_handler("recovered", []), name="recovered")
    server = FailingDiscoverySlowCleanupServer(FastMCPTransport(backend), id="slow-cleanup")
    harness = Harness(
        HarnessConfig(root=tmp_path),
        plugins=[MCPPlugin(servers=[server])],
        model=ScriptedModel([]),
    )
    connecting = asyncio.create_task(harness.connect())
    await cleanup_started.wait()

    connecting.cancel()
    await asyncio.sleep(0)
    assert not connecting.done()
    release_cleanup.set()
    with pytest.raises(asyncio.CancelledError):
        await connecting
    assert harness.tools == []

    await harness.connect()
    assert [tool.name for tool in harness.tools] == ["recovered"]
    await harness.aclose()


async def test_direct_tool_collision_rolls_back_mcp(tmp_path, monkeypatch) -> None:
    """Discovered MCP tools collide atomically with direct tools."""
    server = scripted_server(monkeypatch, {"shared": _schema()})
    direct = ToolSpec("shared", "Direct", _schema(), lambda _args: "direct")
    harness = Harness(
        HarnessConfig(root=tmp_path),
        plugins=[MCPPlugin(servers=[server])],
        tools=[direct],
        model=ScriptedModel([]),
    )

    with pytest.raises(ValueError, match=r"duplicate tool name: shared \(direct and mcp\)"):
        await harness.connect()

    assert harness.tools == [direct]
    assert server.exited == 2


async def test_mcp_server_tool_collision_rolls_back_all_servers(tmp_path, monkeypatch) -> None:
    """Duplicate tools from different MCP servers install no contribution."""
    first = scripted_server(monkeypatch, {"shared": _schema()}, id="first")
    second = scripted_server(monkeypatch, {"shared": _schema()}, id="second")
    harness = Harness(
        HarnessConfig(root=tmp_path),
        plugins=[MCPPlugin(servers=[first, second])],
        model=ScriptedModel([]),
    )

    with pytest.raises(ValueError, match=r"duplicate tool name: shared \(mcp and mcp\)"):
        await harness.connect()

    assert harness.tools == []
    assert first.exited == 2
    assert second.exited == 2


async def test_mcp_collision_detected_before_model_request(tmp_path, monkeypatch) -> None:
    """MCP names collide with existing tools during connect."""
    server = scripted_server(monkeypatch, {"read": _schema()})
    client = MultiCallClient([])
    harness = Harness(
        HarnessConfig(root=tmp_path),
        model=_fake_openai(client),
        plugins=[FilesystemPlugin(tools=["read"]), MCPPlugin(servers=[server])],
    )

    with pytest.raises(ValueError, match="duplicate tool name: read"):
        await harness.run("go")
    assert client.payloads == []
    assert server.exited == 2


async def test_final_result_mcp_collision_detected(tmp_path, monkeypatch) -> None:
    """MCP tools also collide with synthetic structured-output tools."""

    class Answer(BaseModel):
        """Structured output type."""

        value: str

    server = scripted_server(monkeypatch, {"final_result": _schema()})
    harness = Harness(
        HarnessConfig(root=tmp_path, output_type=Answer, output_mode="tool"),
        plugins=[MCPPlugin(servers=[server])],
        model=_fake_openai(MultiCallClient([])),
    )

    with pytest.raises(ValueError, match="reserved for structured output"):
        await harness.connect()


async def test_duplicate_derived_id_disambiguated(tmp_path, monkeypatch) -> None:
    """Duplicate MCP server ids get readable suffixes."""
    first = scripted_server(monkeypatch, {"one": _schema()}, id="same")
    second = scripted_server(monkeypatch, {"two": _schema()}, id="same")
    harness = Harness(HarnessConfig(root=tmp_path), plugins=[MCPPlugin(servers=[first, second])], model=_fake_openai(MultiCallClient([])))

    await harness.connect()
    await harness.aclose()

    metadata = {tool.name: tool.origin for tool in harness.tools if tool.origin is not None}
    assert metadata == {
        "one": ToolOrigin(plugin="mcp", source="same", attributes={"tool_name": "one"}),
        "two": ToolOrigin(plugin="mcp", source="same-2", attributes={"tool_name": "two"}),
    }


async def test_binding_local_ids_stay_stable_for_shared_server(tmp_path, monkeypatch) -> None:
    """One shared wrapper gets independent ids, handlers, and trace attribution."""
    shared = scripted_server(monkeypatch, {"shared": _schema()}, id="same")
    first_neighbor = scripted_server(monkeypatch, {"first_neighbor": _schema()}, id="same")
    second_neighbor = scripted_server(monkeypatch, {"second_neighbor": _schema()}, id="same")
    first_tracer = FakeTracer()
    second_tracer = FakeTracer()
    first = Harness(
        HarnessConfig(root=tmp_path),
        plugins=[MCPPlugin(servers=[shared, first_neighbor])],
        model=_fake_openai(MultiCallClient([("shared", '{"value":"first"}')])),
        tracing=[TracingOptions(tracer=first_tracer)],
    )
    second = Harness(
        HarnessConfig(root=tmp_path),
        plugins=[MCPPlugin(servers=[second_neighbor, shared])],
        model=_fake_openai(MultiCallClient([("shared", '{"value":"second"}')])),
        tracing=[TracingOptions(tracer=second_tracer)],
    )

    assert shared.id == "same"
    await first.connect()
    await second.connect()
    first_tool = next(tool for tool in first.tools if tool.name == "shared")
    second_tool = next(tool for tool in second.tools if tool.name == "shared")
    first_result = await first_tool.handler({"value": "direct-first"})
    second_result = await second_tool.handler({"value": "direct-second"})

    assert first_tool.origin == ToolOrigin(plugin="mcp", source="same", attributes={"tool_name": "shared"})
    assert second_tool.origin == ToolOrigin(plugin="mcp", source="same-2", attributes={"tool_name": "shared"})
    assert first_result.metadata["mcp_server_id"] == "same"
    assert second_result.metadata["mcp_server_id"] == "same-2"
    assert shared.id == "same"

    await first.run("first")
    await second.run("second")
    first_span = next(span for span in first_tracer.spans if span.name == "execute_tool shared")
    second_span = next(span for span in second_tracer.spans if span.name == "execute_tool shared")
    assert first_span.attributes["mcp.server.id"] == "same"
    assert second_span.attributes["mcp.server.id"] == "same-2"
    await second.aclose()
    await first.aclose()


async def test_mcp_connect_cancellation_rolls_back_and_retries(tmp_path) -> None:
    """Cancellation during discovery closes the server and leaves a clean retry."""
    entered_discovery = asyncio.Event()

    class CancelOnceListServer(ObservedMCPServer):
        def __init__(self, transport: Any, **kwargs: Any) -> None:
            super().__init__(transport, **kwargs)
            self.attempts = 0

        async def list_tools(self, *, server_id: str | None = None) -> list[ToolSpec]:
            self.attempts += 1
            if self.attempts == 1:
                entered_discovery.set()
                await asyncio.Event().wait()
            return await MCPServer.list_tools(self, server_id=server_id)

    backend = FastMCP("cancel-discovery")
    backend.tool(_echo_handler("remote", []), name="remote")
    server = CancelOnceListServer(FastMCPTransport(backend), id="cancel")
    harness = Harness(
        HarnessConfig(root=tmp_path),
        plugins=[MCPPlugin(servers=[server])],
        model=ScriptedModel([]),
    )
    connecting = asyncio.create_task(harness.connect())
    await entered_discovery.wait()

    connecting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await connecting
    assert harness.tools == []
    assert server.exited == 1

    await harness.connect()
    assert [tool.name for tool in harness.tools] == ["remote"]
    await harness.aclose()


async def test_mcp_plugin_close_cancellation_propagates(tmp_path) -> None:
    """Cancellation during harness close propagates after MCP cleanup."""

    @asynccontextmanager
    async def slow_stop(server: Any):
        try:
            yield {}
        finally:
            await asyncio.sleep(1)

    backend = FastMCP("plugin-slow-stop", lifespan=slow_stop)
    backend.tool(_echo_handler("remote", []), name="remote")
    server = MCPServer(FastMCPTransport(backend), id="slow-stop")
    harness = Harness(
        HarnessConfig(root=tmp_path),
        plugins=[MCPPlugin(servers=[server])],
        model=ScriptedModel([]),
    )
    await harness.connect()
    closing = asyncio.create_task(harness.aclose())
    await asyncio.sleep(0.1)

    closing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await closing

    assert harness._closed is True
    async with server:
        result = await server.call_tool("remote", {"value": "reused"})
    assert result.ok is True


async def test_mcp_and_other_plugins_close_in_reverse_order(tmp_path) -> None:
    """Generic plugin order controls MCP cleanup with no core special case."""
    events: list[str] = []

    class LoggingServer(ObservedMCPServer):
        async def __aexit__(self, *exc: object) -> None:
            events.append("mcp")
            await super().__aexit__(*exc)

    class OtherPlugin:
        name = "other"

        def bind(self, context) -> PluginBinding:
            @asynccontextmanager
            async def connect():
                try:
                    yield PluginContribution()
                finally:
                    events.append("other")

            return PluginBinding(connect=connect)

    backend = FastMCP("close-order")
    backend.tool(_echo_handler("remote", []), name="remote")
    server = LoggingServer(FastMCPTransport(backend), id="close-order")
    harness = Harness(
        HarnessConfig(root=tmp_path),
        plugins=[MCPPlugin(servers=[server]), OtherPlugin()],
        model=ScriptedModel([]),
    )

    await harness.connect()
    events.clear()
    await harness.aclose()

    assert events == ["other", "mcp"]


async def test_closed_harness_rejects_run_and_connect_but_keeps_schema(tmp_path, monkeypatch) -> None:
    """Closed harnesses are terminal but still inspectable."""
    server = scripted_server(monkeypatch, {"remote": _schema()})
    harness = Harness(HarnessConfig(root=tmp_path), plugins=[MCPPlugin(servers=[server])], model=_fake_openai(MultiCallClient([])))

    await harness.connect()
    await harness.aclose()
    harness.add_tool(ToolSpec("late", "Late tool", {"type": "object", "properties": {}}, lambda args: "ok"))

    assert [tool["name"] for tool in harness.tool_schemas()] == ["remote", "late"]
    with pytest.raises(HarnessError, match="harness is closed"):
        await harness.connect()
    with pytest.raises(HarnessError, match="harness is closed"):
        await harness.run("go")


def test_run_sync_is_one_shot(tmp_path) -> None:
    """run_sync closes the harness after one call."""
    harness = Harness(
        HarnessConfig(root=tmp_path),
        model=ScriptedModel([ScriptedSession(start_turn=ModelTurn(text="done", raw={"id": "done"}))]),
    )

    assert harness.run_sync("go").text == "done"
    with pytest.raises(HarnessError, match="harness is closed"):
        harness.run_sync("again")


async def test_aclose_with_injected_model_closes_mcp(tmp_path, monkeypatch) -> None:
    """Harness-owned MCP resources close even when the model is injected."""
    server = scripted_server(monkeypatch, {"remote": _schema()})
    model = _fake_openai(MultiCallClient([]))
    harness = Harness(HarnessConfig(root=tmp_path), plugins=[MCPPlugin(servers=[server])], model=model)

    await harness.connect()
    await harness.aclose()

    assert server.exited == 2


async def test_unknown_tool_hook_filter_is_allowed_and_never_fires(tmp_path) -> None:
    """Tool hook filters are passive when a tool name is never registered."""
    seen = []
    harness = Harness(
        HarnessConfig(root=tmp_path),
        model=_fake_openai(MultiCallClient([])),
        hooks=[Hook("before_tool_call", lambda ctx: seen.append(ctx.tool_name), tools=["missing"])],
    )

    await harness.run("go")

    assert seen == []


async def test_default_child_does_not_inherit_parent_mcp(tmp_path, monkeypatch) -> None:
    """A default child receives no parent MCP binding."""
    server = scripted_server(monkeypatch, {"remote": _schema()})
    seen = {}

    def child_start(_prompt, _instructions, tools, _metadata, _previous_response_id):
        seen["tools"] = [tool["name"] for tool in tools]

    parent_session = ScriptedSession(
        start_turn=ModelTurn(
            tool_calls=[ModelToolCall(id="delegate", name="subagent", arguments='{"task":"help"}')],
            raw={},
        ),
        continue_turn=ModelTurn(text="done", raw={}),
    )
    child_session = ScriptedSession(
        start_turn=ModelTurn(text="child", raw={}),
        on_start=child_start,
    )
    parent = Harness(
        HarnessConfig(root=tmp_path),
        plugins=[MCPPlugin(servers=[server]), SubagentsPlugin()],
        model=ScriptedModel([parent_session, child_session]),
    )

    assert (await parent.run("delegate")).text == "done"
    await parent.aclose()
    assert seen["tools"] == []


async def test_named_child_uses_explicit_mcp_plugin(tmp_path, monkeypatch) -> None:
    """A named child connects and closes its explicit MCP binding."""
    parent_server = scripted_server(monkeypatch, {"parent": _schema()})
    child_server = scripted_server(monkeypatch, {"child": _schema()})
    seen_tools: list[list[str]] = []
    config = SubAgentConfig(
        name="mcp",
        description="MCP helper.",
        plugins=[MCPPlugin(servers=[child_server])],
    )
    parent_session = ScriptedSession(
        start_turn=ModelTurn(
            tool_calls=[ModelToolCall(id="delegate", name="subagent", arguments='{"task":"help","agent":"mcp"}')],
            raw={},
        ),
        continue_turn=ModelTurn(text="done", raw={}),
    )
    child_session = ScriptedSession(start_turn=ModelTurn(text="child", raw={}))
    parent = Harness(
        HarnessConfig(root=tmp_path),
        plugins=[
            MCPPlugin(servers=[parent_server]),
            SubagentsPlugin(agents=[config]),
        ],
        model=ScriptedModel([parent_session, child_session]),
        hooks=[Hook("after_subagent_run", lambda ctx: seen_tools.append(ctx.tools), agents=["mcp"])],
    )

    assert (await parent.run("delegate")).text == "done"
    await parent.aclose()
    assert seen_tools == [["child"]]
    assert child_server.exited == 2


async def test_run_teardown_after_tool_exception_closes_mcp(tmp_path, monkeypatch) -> None:
    """MCP resources still close after a tool raises during the run."""
    server = scripted_server(monkeypatch, {"remote": _schema()})

    def boom(_args):
        raise RuntimeError("boom")

    harness = Harness(
        HarnessConfig(root=tmp_path),
        plugins=[MCPPlugin(servers=[server])],
        model=_fake_openai(MultiCallClient([("boom", "{}")])),
        tools=[ToolSpec("boom", "Boom", {"type": "object", "properties": {}}, boom)],
    )

    await harness.run("go")
    await harness.aclose()

    assert server.exited == 2


async def test_run_teardown_after_cancellation_closes_mcp(tmp_path, monkeypatch) -> None:
    """MCP resources close cleanly after an outer run cancellation."""
    started = asyncio.Event()
    server = scripted_server(monkeypatch, {"remote": _schema()})

    async def slow(_args):
        started.set()
        await asyncio.sleep(60)

    harness = Harness(
        HarnessConfig(root=tmp_path),
        plugins=[MCPPlugin(servers=[server])],
        model=_fake_openai(MultiCallClient([("slow", "{}")])),
        tools=[ToolSpec("slow", "Slow", {"type": "object", "properties": {}}, slow)],
    )
    task = asyncio.create_task(harness.run("go"))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await harness.aclose()

    assert server.exited == 2


async def test_resume_with_mcp_reuses_connection_and_keeps_state_clean(tmp_path, monkeypatch) -> None:
    """MCP tools are harness-local and not serialized into resume state."""
    server = scripted_server(monkeypatch, {"remote": _schema()})
    first_session = SequenceSession(ModelTurn(text="first", raw={"id": "first"}))
    second_session = SequenceSession(ModelTurn(text="second", raw={"id": "second"}))
    harness = Harness(
        HarnessConfig(root=tmp_path), plugins=[MCPPlugin(servers=[server])], model=ScriptedModel([first_session, second_session])
    )

    first = await harness.run("first")
    second = await harness.run("second", resume_from=first.resume_state)
    await harness.aclose()

    assert second.text == "second"
    assert "remote" not in str(first.resume_state)
    assert "fake" not in str(first.resume_state)
    assert server.list_calls == 1


async def test_approval_resume_connects_mcp_before_validating_and_preserves_unknown_sibling(tmp_path, monkeypatch) -> None:
    """Approval resume waits for MCP discovery and keeps unknown sibling calls model-visible."""
    server = scripted_server(monkeypatch, {"remote": _schema()})
    approval_called: list[Json] = []
    first_session = SequenceSession(
        ModelTurn(
            tool_calls=[
                ModelToolCall(id="call_approval", name="deploy", arguments='{"env":"prod"}'),
                ModelToolCall(id="call_remote", name="remote", arguments='{"value":"ok"}'),
                ModelToolCall(id="call_unknown", name="missing", arguments="{}"),
            ],
            raw={"id": "approval"},
        )
    )
    resumed_session = SequenceSession(ModelTurn(raw={"unused": True}), ModelTurn(text="done", raw={"id": "done"}))
    model = ScriptedModel([first_session, resumed_session])
    approval_tool = ToolSpec(
        "deploy",
        "Deploy something.",
        {"type": "object", "properties": {"env": {"type": "string"}}, "required": ["env"]},
        lambda args: approval_called.append(args) or "deployed",
        requires_approval=True,
    )
    first_harness = Harness(
        HarnessConfig(root=tmp_path),
        plugins=[MCPPlugin(servers=[server])],
        model=model,
        tools=[approval_tool],
    )
    paused = await first_harness.run("go")
    await first_harness.aclose()

    second_harness = Harness(
        HarnessConfig(root=tmp_path),
        plugins=[MCPPlugin(servers=[server])],
        model=model,
        tools=[approval_tool],
    )
    result = await second_harness.resume_approvals(
        paused.resume_state,
        [ApprovalDecision(call_id="call_approval", approved=True)],
    )
    await second_harness.aclose()

    assert result.text == "done"
    assert approval_called == [{"env": "prod"}]
    assert server.call_records == [("remote", {"value": "ok"})]
    outputs = {output.call_id: json.loads(output.output) for output in resumed_session.tool_outputs[0]}
    assert outputs["call_remote"]["content"] == "remote:{'value': 'ok'}"
    assert outputs["call_unknown"] == {"ok": False, "content": "unknown tool missing", "metadata": {"tool": "missing"}}


async def test_trace_attribution_survives_after_tool_hook(tmp_path, monkeypatch) -> None:
    """MCP tracing attributes come from ToolSpec metadata, not result metadata."""
    tracer = FakeTracer()
    server = scripted_server(monkeypatch, {"remote": _schema()})

    def rewrite(ctx) -> None:
        ctx.output = ToolResult(True, "rewritten", {}).as_json()

    harness = Harness(
        HarnessConfig(root=tmp_path),
        plugins=[MCPPlugin(servers=[server])],
        model=_fake_openai(MultiCallClient([("remote", '{"value":"ok"}')])),
        hooks=[Hook("after_tool_call", rewrite, tools=["remote"])],
        tracing=[TracingOptions(tracer=tracer)],
    )

    await harness.run("go")
    await harness.aclose()

    tool_span = next(span for span in tracer.spans if span.name == "execute_tool remote")
    assert tool_span.attributes["mcp.server.id"] == "fake"
    assert tool_span.attributes["mcp.tool.name"] == "remote"


async def test_connection_failure_happens_before_run_hooks(tmp_path) -> None:
    """Connection failures happen before the normal run lifecycle."""
    events = []
    tracer = FakeTracer()

    class FailingConnectServer(ObservedMCPServer):
        async def list_tools(self, *, server_id: str | None = None) -> list[ToolSpec]:
            """Fail MCP setup."""
            del server_id
            raise MCPError("connect failed")

    failing = FailingConnectServer(FastMCPTransport(FastMCP("failing-connect")), id="failing")
    harness = Harness(
        HarnessConfig(root=tmp_path),
        plugins=[MCPPlugin(servers=[failing])],
        model=_fake_openai(MultiCallClient([])),
        hooks=[
            Hook("run_start", lambda ctx: events.append("start")),
            Hook("run_end", lambda ctx: events.append(ctx.stop_reason)),
        ],
        tracing=[TracingOptions(tracer=tracer)],
    )

    with pytest.raises(MCPError, match="connect failed"):
        await harness.run("go")

    assert events == []
    assert tracer.spans == []
