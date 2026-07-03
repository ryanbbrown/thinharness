from __future__ import annotations

import asyncio
import json
import sys
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any

import pytest
from fakes import FakeTracer, MultiCallClient, ScriptedModel, ScriptedSession, _fake_openai
from pydantic import BaseModel

from thinharness import (
    ApprovalDecision,
    Harness,
    HarnessConfig,
    HarnessError,
    Hook,
    MCPError,
    MCPServer,
    MCPServerStdio,
    ModelTurn,
    SubAgentConfig,
    TracingOptions,
    build_child_harness,
)
from thinharness.providers import ModelToolCall, ToolOutput
from thinharness.tools.base import Json, ToolResult, ToolSpec

pytest.importorskip("mcp")


class FakeMCPClient:
    """Tiny MCP session double for server conversion tests."""

    def __init__(self, server: FakeMCPServer) -> None:
        self.server = server

    async def list_tools(self):
        """Return fake MCP tool declarations."""
        self.server.list_calls += 1
        return SimpleNamespace(tools=[
            SimpleNamespace(name=name, description=f"{name} tool", inputSchema=schema)
            for name, schema in self.server.tool_schemas.items()
        ])

    async def call_tool(self, name: str, arguments: Json):
        """Return fake MCP call results."""
        self.server.call_records.append((name, arguments))
        if name == "transport":
            raise ConnectionError("transport closed")
        if name == "bug":
            raise RuntimeError("programming bug")
        if name == "error":
            return SimpleNamespace(isError=True, content=[SimpleNamespace(type="text", text="try again")], structuredContent=None)
        return SimpleNamespace(isError=False, content=[SimpleNamespace(type="text", text=f"{name}:{arguments}")], structuredContent=None)


class FakeMCPServer(MCPServer):
    """MCP server double with ref-count-visible enter and exit calls."""

    def __init__(self, tool_schemas: dict[str, Json], *, id: str = "fake", **kwargs: Any) -> None:
        super().__init__(id=id, **kwargs)
        self.tool_schemas = tool_schemas
        self.list_calls = 0
        self.entered = 0
        self.exited = 0
        self.call_records: list[tuple[str, Json]] = []
        self.client = FakeMCPClient(self)

    def _default_id(self) -> str:
        """Return the fake server id."""
        return "fake"

    @asynccontextmanager
    async def _client_streams(self):
        """Unused stream context for the fake server."""
        yield None, None

    async def __aenter__(self):
        """Track fake context entry."""
        self.entered += 1
        return self

    async def __aexit__(self, *exc: object) -> None:
        """Track fake context exit."""
        self.exited += 1

    def _session(self) -> FakeMCPClient:
        """Return the fake client."""
        return self.client


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

    async def continue_with_user_text(self, text, constants, *, notices=None):
        """Return the scripted turn for a resumed prompt; no tests expect corrections."""
        if self.requests_made:
            raise AssertionError("unexpected user-text correction")
        self.requests_made += 1
        return self.start_turn

    def dump_state(self):
        """Return scripted resume state."""
        return dict(self._dump_state)


class LifecycleMCPServer(MCPServer):
    """MCP server double that exercises the base ref-count lifecycle."""

    def __init__(self) -> None:
        super().__init__(id="lifecycle")
        self.starts = 0
        self.stops = 0

    def _default_id(self) -> str:
        """Return the fake lifecycle id."""
        return "lifecycle"

    @asynccontextmanager
    async def _client_streams(self):
        """Unused by the overridden runner."""
        yield None, None

    async def _session_runner(self) -> None:
        """Install a fake client and wait for the stop signal."""
        state = self._session_state
        ready_event = state.ready_event
        stop_event = state.stop_event
        assert ready_event is not None
        assert stop_event is not None
        self.starts += 1
        state.client = object()
        ready_event.set()
        try:
            await stop_event.wait()
        finally:
            state.client = None
            self.stops += 1
            ready_event.set()


def _schema() -> Json:
    """Return a minimal MCP input schema."""
    return {"type": "object", "properties": {"value": {"type": "string"}}, "$schema": "https://json-schema.org/draft/2020-12/schema"}


async def test_mcp_tool_discovery_and_result_metadata() -> None:
    """MCP tools are converted, cleaned, and return stable metadata."""
    original_schema = _schema()
    server = FakeMCPServer({"hello.world": original_schema}, tool_prefix="git")

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


async def test_tool_filters_apply_before_conversion() -> None:
    """include_tools and exclude_tools filter original MCP names."""
    server = FakeMCPServer(
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


async def test_mcp_tool_error_is_retryable() -> None:
    """MCP isError results use the harness retry metadata path."""
    server = FakeMCPServer({"error": _schema()})

    result = await server.call_tool("error", {})

    assert result.ok is False
    assert result.metadata == {
        "source": "mcp",
        "mcp_server_id": "fake",
        "mcp_tool_name": "error",
        "error_type": "MCPToolError",
        "retry": True,
    }


async def test_mcp_transport_exception_returns_metadata() -> None:
    """Unexpected transport/session errors still return MCP metadata."""
    server = FakeMCPServer({"transport": _schema()})

    result = await server.call_tool("transport", {})

    assert result.ok is False
    assert result.content == "transport closed"
    assert result.metadata == {
        "source": "mcp",
        "mcp_server_id": "fake",
        "mcp_tool_name": "transport",
        "error_type": "MCPError",
    }


async def test_mcp_programming_bug_propagates() -> None:
    """Local bugs are not hidden as normal MCP tool failures."""
    server = FakeMCPServer({"bug": _schema()})

    with pytest.raises(RuntimeError, match="programming bug"):
        await server.call_tool("bug", {})


async def test_invalid_input_schema_names_tool() -> None:
    """Invalid MCP schemas fail with a clear tool-specific error."""
    server = FakeMCPServer({"bad": None})  # type: ignore[arg-type]

    with pytest.raises(MCPError, match="bad.*inputSchema"):
        await server.list_tools()


async def test_sanitized_name_collision_raises() -> None:
    """Sanitizer collisions include both original MCP names."""
    server = FakeMCPServer({"foo.bar": _schema(), "foo/bar": _schema()})

    with pytest.raises(MCPError, match="foo\\.bar.*foo/bar"):
        await server.list_tools()


async def test_sanitizer_preserves_valid_edge_characters() -> None:
    """Leading underscores and dashes are valid function-tool name characters."""
    server = FakeMCPServer({"_foo": _schema(), "foo": _schema(), "-": _schema()})

    tools = await server.list_tools()

    assert [tool.name for tool in tools] == ["_foo", "foo", "-"]


async def test_lifecycle_reference_counted() -> None:
    """Nested entries share one session and close on the last exit."""
    server = LifecycleMCPServer()

    await server.__aenter__()
    await server.__aenter__()
    await server.__aexit__(None, None, None)

    assert server.starts == 1
    assert server.stops == 0

    await server.__aexit__(None, None, None)

    assert server.starts == 1
    assert server.stops == 1


async def test_force_close_propagates_parent_cancellation() -> None:
    """Session cleanup must not swallow the caller's cancellation."""
    server = LifecycleMCPServer()
    inner_task = asyncio.create_task(asyncio.sleep(60))
    parent_task = asyncio.current_task()
    assert parent_task is not None

    asyncio.get_running_loop().call_soon(parent_task.cancel)
    with pytest.raises(asyncio.CancelledError):
        await server._session_state.force_close(inner_task)

    parent_task.uncancel()
    await asyncio.gather(inner_task, return_exceptions=True)


async def test_aexit_propagates_parent_cancellation() -> None:
    """The __aexit__ timeout wrapper must preserve caller cancellation."""
    server = LifecycleMCPServer()
    await server.__aenter__()
    parent_task = asyncio.current_task()
    assert parent_task is not None

    asyncio.get_running_loop().call_soon(parent_task.cancel)
    with pytest.raises(asyncio.CancelledError):
        await server.__aexit__(None, None, None)

    parent_task.uncancel()


async def test_harness_connects_mcp_once_across_async_runs(tmp_path) -> None:
    """Harness runs reuse the discovered MCP tools until aclose."""
    server = FakeMCPServer({"remote": _schema()})
    client = MultiCallClient([("remote", '{"value":"ok"}')])
    harness = Harness(HarnessConfig(root=tmp_path, builtin_tools=[], mcp_servers=[server]), model=_fake_openai(client))

    result = await harness.run("go")
    second = await harness.run("done")
    await harness.aclose()

    assert result.text == "done"
    assert second.text == "done"
    assert server.list_calls == 1
    assert server.entered == 3
    assert server.exited == 3
    assert server.call_records == [("remote", {"value": "ok"})]


async def test_explicit_connect_does_not_reconnect_on_run(tmp_path) -> None:
    """Explicit connect discovers MCP tools once before run."""
    server = FakeMCPServer({"remote": _schema()})
    harness = Harness(HarnessConfig(root=tmp_path, builtin_tools=[], mcp_servers=[server]), model=_fake_openai(MultiCallClient([])))

    await harness.connect()
    result = await harness.run("go")

    assert result.text == ""
    assert server.list_calls == 1


async def test_is_error_drives_harness_retry(tmp_path) -> None:
    """MCP isError envelopes feed the harness tool retry accounting."""
    server = FakeMCPServer({"error": _schema()})
    run_end = []
    session = SequenceSession(
        ModelTurn(tool_calls=[ModelToolCall(id="call_1", name="error", arguments="{}")], raw={"id": "start"}),
        ModelTurn(tool_calls=[ModelToolCall(id="call_2", name="error", arguments="{}")], raw={"id": "retry"}),
    )
    harness = Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[], mcp_servers=[server], tool_retries=1),
        model=ScriptedModel([session]),
        hooks=[Hook("run_end", lambda ctx: run_end.append((ctx.stop_reason, dict(ctx.usage.tool_retries))))],
    )

    with pytest.raises(HarnessError, match="exceeded max_retries=1"):
        await harness.run("go")

    assert len(session.tool_outputs) == 1
    assert session.tool_outputs[0][0].output
    assert run_end == [("tool_retries_exceeded", {"error": 2})]


async def test_partial_connect_failure_cleans_up(tmp_path) -> None:
    """A later MCP discovery failure closes earlier entered servers."""
    first = FakeMCPServer({"ok": _schema()})

    class FailingListServer(FakeMCPServer):
        async def list_tools(self) -> list[ToolSpec]:
            """Fail discovery after entering the context."""
            async with self:
                raise MCPError("list failed")

    second = FailingListServer({"bad": _schema()})
    harness = Harness(HarnessConfig(root=tmp_path, builtin_tools=[], mcp_servers=[first, second]), model=_fake_openai(MultiCallClient([])))

    with pytest.raises(MCPError, match="list failed"):
        await harness.connect()

    assert first.exited == 2
    assert second.exited == 2


async def test_mcp_collision_detected_before_model_request(tmp_path) -> None:
    """MCP names collide with existing tools during connect."""
    server = FakeMCPServer({"read": _schema()})
    client = MultiCallClient([])
    harness = Harness(HarnessConfig(root=tmp_path, mcp_servers=[server]), model=_fake_openai(client))

    with pytest.raises(HarnessError, match="tool name collision"):
        await harness.run("go")
    assert client.payloads == []
    assert server.exited == 2


async def test_final_result_mcp_collision_detected(tmp_path) -> None:
    """MCP tools also collide with synthetic structured-output tools."""
    class Answer(BaseModel):
        """Structured output type."""

        value: str

    server = FakeMCPServer({"final_result": _schema()})
    harness = Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[], output_type=Answer, output_mode="tool", mcp_servers=[server]),
        model=_fake_openai(MultiCallClient([])),
    )

    with pytest.raises(HarnessError, match="tool name collision"):
        await harness.connect()


async def test_duplicate_derived_id_disambiguated(tmp_path) -> None:
    """Duplicate MCP server ids get readable suffixes."""
    first = FakeMCPServer({"one": _schema()}, id="same")
    second = FakeMCPServer({"two": _schema()}, id="same")
    harness = Harness(HarnessConfig(root=tmp_path, builtin_tools=[], mcp_servers=[first, second]), model=_fake_openai(MultiCallClient([])))

    await harness.connect()

    metadata = {tool.name: tool.mcp.server_id for tool in harness.tools if tool.mcp is not None}
    assert metadata == {"one": "same", "two": "same-2"}


async def test_closed_harness_rejects_run_and_connect_but_keeps_schema(tmp_path) -> None:
    """Closed harnesses are terminal but still inspectable."""
    server = FakeMCPServer({"remote": _schema()})
    harness = Harness(HarnessConfig(root=tmp_path, builtin_tools=[], mcp_servers=[server]), model=_fake_openai(MultiCallClient([])))

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
        HarnessConfig(root=tmp_path, builtin_tools=[]),
        model=ScriptedModel([ScriptedSession(start_turn=ModelTurn(text="done", raw={"id": "done"}))]),
    )

    assert harness.run_sync("go").text == "done"
    with pytest.raises(HarnessError, match="harness is closed"):
        harness.run_sync("again")


async def test_aclose_with_injected_model_closes_mcp(tmp_path) -> None:
    """Harness-owned MCP resources close even when the model is injected."""
    server = FakeMCPServer({"remote": _schema()})
    model = _fake_openai(MultiCallClient([]))
    harness = Harness(HarnessConfig(root=tmp_path, builtin_tools=[], mcp_servers=[server]), model=model)

    await harness.connect()
    await harness.aclose()

    assert server.exited == 2


async def test_unknown_tool_hook_filter_is_allowed_and_never_fires(tmp_path) -> None:
    """Tool hook filters are passive when a tool name is never registered."""
    seen = []
    harness = Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[]),
        model=_fake_openai(MultiCallClient([])),
        hooks=[Hook("before_tool_call", lambda ctx: seen.append(ctx.tool_name), tools=["missing"])],
    )

    await harness.run("go")

    assert seen == []


def test_default_subagent_does_not_implicitly_inherit_mcp(tmp_path) -> None:
    """MCP inheritance for child harnesses is explicit."""
    server = FakeMCPServer({"remote": _schema()})
    parent = Harness(HarnessConfig(root=tmp_path, builtin_tools=[], mcp_servers=[server]), model=ScriptedModel([]))

    child = build_child_harness(parent, None)

    assert child.config.mcp_servers == []


def test_subagent_empty_fails_validation() -> None:
    """Named subagents must still expose some tool source."""
    with pytest.raises(ValueError, match="named subagents"):
        SubAgentConfig(name="empty", description="Empty helper.")


def test_subagent_mcp_override_and_union_config(tmp_path) -> None:
    """Child config encodes MCP override and union semantics."""
    parent_server = FakeMCPServer({"parent": _schema()})
    child_server = FakeMCPServer({"child": _schema()})
    parent = Harness(HarnessConfig(root=tmp_path, builtin_tools=[], mcp_servers=[parent_server]), model=ScriptedModel([]))

    override = build_child_harness(parent, SubAgentConfig(name="override", description="Override helper.", mcp_servers=[child_server]))
    union = build_child_harness(parent, SubAgentConfig(
        name="union",
        description="Union helper.",
        inherit_mcp_servers=True,
        mcp_servers=[parent_server, child_server],
    ))

    assert override.config.mcp_servers == [child_server]
    assert union.config.mcp_servers == [parent_server, child_server]


async def test_subagent_overrides_mcp_only_runtime(tmp_path) -> None:
    """An override-only child sees explicit MCP servers but not parent MCP servers."""
    parent_server = FakeMCPServer({"parent": _schema()})
    child_server = FakeMCPServer({"child": _schema()})
    parent = Harness(HarnessConfig(root=tmp_path, builtin_tools=[], mcp_servers=[parent_server]), model=ScriptedModel([]))

    child = build_child_harness(parent, SubAgentConfig(name="override", description="Override helper.", mcp_servers=[child_server]))
    await child.connect()

    assert [tool.name for tool in child.tools] == ["child"]


async def test_subagent_unions_inherit_plus_override_runtime(tmp_path) -> None:
    """An inherited-plus-override child sees both MCP tool sets."""
    parent_server = FakeMCPServer({"parent": _schema()})
    child_server = FakeMCPServer({"child": _schema()})
    parent = Harness(HarnessConfig(root=tmp_path, builtin_tools=[], mcp_servers=[parent_server]), model=ScriptedModel([]))

    child = build_child_harness(parent, SubAgentConfig(
        name="union",
        description="Union helper.",
        inherit_mcp_servers=True,
        mcp_servers=[child_server],
    ))
    await child.connect()

    assert [tool.name for tool in child.tools] == ["parent", "child"]


async def test_subagent_identity_dedup_runtime(tmp_path) -> None:
    """The same inherited and explicit MCP object is entered once in a child."""
    server = FakeMCPServer({"remote": _schema()})
    parent = Harness(HarnessConfig(root=tmp_path, builtin_tools=[], mcp_servers=[server]), model=ScriptedModel([]))

    child = build_child_harness(parent, SubAgentConfig(
        name="dedup",
        description="Dedup helper.",
        inherit_mcp_servers=True,
        mcp_servers=[server],
    ))
    await child.connect()

    assert [tool.name for tool in child.tools] == ["remote"]
    assert child.config.mcp_servers == [server]


async def test_subagent_id_equal_but_distinct_collides(tmp_path) -> None:
    """Distinct MCP objects with identical tools collide in child connect."""
    first = FakeMCPServer({"remote": _schema()}, id="same")
    second = FakeMCPServer({"remote": _schema()}, id="same")
    parent = Harness(HarnessConfig(root=tmp_path, builtin_tools=[], mcp_servers=[first]), model=ScriptedModel([]))
    child = build_child_harness(parent, SubAgentConfig(
        name="child",
        description="Child helper.",
        inherit_mcp_servers=True,
        mcp_servers=[second],
    ))

    with pytest.raises(HarnessError, match="tool name collision"):
        await child.connect()


async def test_subagent_inherited_parent_tools_skip_mcp_duplicates(tmp_path) -> None:
    """Parent MCP tools are not copied as custom tools when also inherited as MCP."""
    server = FakeMCPServer({"remote": _schema()})
    parent = Harness(HarnessConfig(root=tmp_path, builtin_tools=[], mcp_servers=[server]), model=ScriptedModel([]))
    await parent.connect()

    child = build_child_harness(parent, SubAgentConfig(
        name="child",
        description="Child helper.",
        inherit_parent_tools=True,
        inherit_mcp_servers=True,
    ))
    await child.connect()

    assert [tool.name for tool in child.tools] == ["remote"]


async def test_subagent_mcp_only_validates_and_inherits(tmp_path) -> None:
    """A named subagent can get its only tools from inherited MCP servers."""
    server = FakeMCPServer({"remote": _schema()})
    config = SubAgentConfig(name="mcp", description="MCP helper.", inherit_mcp_servers=True)
    parent_client = MultiCallClient([("subagent", '{"task":"use remote","agent":"mcp"}')])
    parent = Harness(
        HarnessConfig(root=tmp_path, builtin_tools=["subagent"], mcp_servers=[server], subagents=[config]),
        model=_fake_openai(parent_client),
    )

    result = await parent.run("delegate")
    await parent.aclose()

    assert result.text == "done"
    assert server.list_calls == 2


async def test_run_teardown_after_tool_exception_closes_mcp(tmp_path) -> None:
    """MCP resources still close after a tool raises during the run."""
    server = FakeMCPServer({"remote": _schema()})

    def boom(_args):
        raise RuntimeError("boom")

    harness = Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[], mcp_servers=[server]),
        model=_fake_openai(MultiCallClient([("boom", "{}")])),
        tools=[ToolSpec("boom", "Boom", {"type": "object", "properties": {}}, boom)],
    )

    await harness.run("go")
    await harness.aclose()

    assert server.exited == 2


async def test_run_teardown_after_cancellation_closes_mcp(tmp_path) -> None:
    """MCP resources close cleanly after an outer run cancellation."""
    started = asyncio.Event()
    server = FakeMCPServer({"remote": _schema()})

    async def slow(_args):
        started.set()
        await asyncio.sleep(60)

    harness = Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[], mcp_servers=[server]),
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


async def test_subagent_effective_tools_include_mcp(tmp_path) -> None:
    """After-subagent hooks observe MCP-discovered child tools."""
    seen_tools: list[list[str]] = []
    server = FakeMCPServer({"remote": _schema()})
    config = SubAgentConfig(name="mcp", description="MCP helper.", inherit_mcp_servers=True)
    parent = Harness(
        HarnessConfig(root=tmp_path, builtin_tools=["subagent"], mcp_servers=[server], subagents=[config]),
        model=_fake_openai(MultiCallClient([("subagent", '{"task":"use remote","agent":"mcp"}')])),
        hooks=[Hook("after_subagent_run", lambda ctx: seen_tools.append(ctx.tools), agents=["mcp"])],
    )

    await parent.run("delegate")

    assert seen_tools == [["remote"]]


async def test_resume_with_mcp_reuses_connection_and_keeps_state_clean(tmp_path) -> None:
    """MCP tools are harness-local and not serialized into resume state."""
    server = FakeMCPServer({"remote": _schema()})
    first_session = SequenceSession(ModelTurn(text="first", raw={"id": "first"}))
    second_session = SequenceSession(ModelTurn(text="second", raw={"id": "second"}))
    harness = Harness(HarnessConfig(root=tmp_path, builtin_tools=[], mcp_servers=[server]), model=ScriptedModel([first_session, second_session]))

    first = await harness.run("first")
    second = await harness.run("second", resume_from=first.resume_state)

    assert second.text == "second"
    assert "remote" not in str(first.resume_state)
    assert "fake" not in str(first.resume_state)
    assert server.list_calls == 1


async def test_approval_resume_connects_mcp_before_validating_and_preserves_unknown_sibling(tmp_path) -> None:
    """Approval resume waits for MCP discovery and keeps unknown sibling calls model-visible."""
    server = FakeMCPServer({"remote": _schema()})
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
    paused = await Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[], mcp_servers=[server]),
        model=model,
        tools=[approval_tool],
    ).run("go")

    result = await Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[], mcp_servers=[server]),
        model=model,
        tools=[approval_tool],
    ).resume_approvals(
        paused.resume_state,
        [ApprovalDecision(call_id="call_approval", approved=True)],
    )

    assert result.text == "done"
    assert approval_called == [{"env": "prod"}]
    assert server.call_records == [("remote", {"value": "ok"})]
    outputs = {output.call_id: json.loads(output.output) for output in resumed_session.tool_outputs[0]}
    assert outputs["call_remote"]["content"] == "remote:{'value': 'ok'}"
    assert outputs["call_unknown"] == {"ok": False, "content": "unknown tool missing", "metadata": {"tool": "missing"}}


async def test_trace_attribution_survives_after_tool_hook(tmp_path) -> None:
    """MCP tracing attributes come from ToolSpec metadata, not result metadata."""
    tracer = FakeTracer()
    server = FakeMCPServer({"remote": _schema()})

    def rewrite(ctx) -> None:
        ctx.output = ToolResult(True, "rewritten", {}).as_json()

    harness = Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[], mcp_servers=[server]),
        model=_fake_openai(MultiCallClient([("remote", '{"value":"ok"}')])),
        hooks=[Hook("after_tool_call", rewrite, tools=["remote"])],
        tracing=[TracingOptions(tracer=tracer)],
    )

    await harness.run("go")

    tool_span = next(span for span in tracer.spans if span.name == "execute_tool remote")
    assert tool_span.attributes["mcp.server.id"] == "fake"
    assert tool_span.attributes["mcp.tool.name"] == "remote"


async def test_connection_failure_in_run_fires_run_hooks(tmp_path) -> None:
    """MCP connection failures happen inside the normal run lifecycle."""
    events = []
    tracer = FakeTracer()

    class FailingConnectServer(FakeMCPServer):
        async def list_tools(self) -> list[ToolSpec]:
            """Fail MCP setup."""
            raise MCPError("connect failed")

    harness = Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[], mcp_servers=[FailingConnectServer({"remote": _schema()})]),
        model=_fake_openai(MultiCallClient([])),
        hooks=[
            Hook("run_start", lambda ctx: events.append("start")),
            Hook("run_end", lambda ctx: events.append(ctx.stop_reason)),
        ],
        tracing=[TracingOptions(tracer=tracer)],
    )

    with pytest.raises(MCPError, match="connect failed"):
        await harness.run("go")

    assert events == ["start", "error"]
    assert tracer.spans[0].exceptions
