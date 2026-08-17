from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from fakes import ScriptedModel, ScriptedSession, echo_tool

from thinharness import (
    FilesystemPlugin,
    Harness,
    HarnessConfig,
    Hook,
    HookRegistry,
    ModelTurn,
    PluginBinding,
    PluginContext,
    PluginContribution,
    ToolOrigin,
    ToolSpec,
)


def _tool(name: str) -> ToolSpec:
    return ToolSpec(name, name, {"type": "object", "properties": {}}, lambda _args: "ok")


class StaticPlugin:
    def __init__(self, name: str, contribution: PluginContribution | None = None) -> None:
        self.name = name
        self.contribution = contribution or PluginContribution()
        self.bindings = 0

    def bind(self, context: PluginContext) -> PluginBinding:
        self.bindings += 1
        return PluginBinding(static=self.contribution)


class InvalidBindingPlugin:
    name = "invalid"

    def bind(self, context: PluginContext) -> object:
        return object()


class ConnectedPlugin:
    def __init__(self, name: str, events: list[str], contribution: PluginContribution, *, fail_first: bool = False) -> None:
        self.name = name
        self.events = events
        self.contribution = contribution
        self.fail_first = fail_first
        self.attempts = 0

    def bind(self, context: PluginContext) -> PluginBinding:
        @asynccontextmanager
        async def connect():
            self.attempts += 1
            self.events.append(f"enter:{self.name}")
            if self.fail_first and self.attempts == 1:
                raise RuntimeError(f"failed:{self.name}")
            try:
                yield self.contribution
            finally:
                self.events.append(f"exit:{self.name}")

        return PluginBinding(connect=connect)


def test_filesystem_plugin_is_explicit_static_and_ordered(tmp_path: Path) -> None:
    harness = Harness(
        HarnessConfig(root=tmp_path),
        model=ScriptedModel([]),
        plugins=[FilesystemPlugin(tools=["search", "read", "jsonl_search"])],
        tools=[echo_tool()],
    )

    assert [tool.name for tool in harness.tools] == ["search", "read", "jsonl_search", "echo"]
    assert [tool.origin.plugin if tool.origin else None for tool in harness.tools] == ["filesystem", "filesystem", "filesystem", None]
    assert f"Workspace root: {tmp_path.resolve()}" in harness.system_instructions()


def test_no_filesystem_plugin_has_no_tools_instruction_or_root_side_effect(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    harness = Harness(HarnessConfig(root=missing), model=ScriptedModel([]))

    assert harness.tools == []
    assert "Workspace root:" not in harness.system_instructions()
    assert not missing.exists()

    with_plugin = Harness(
        HarnessConfig(root=missing),
        model=ScriptedModel([]),
        plugins=[FilesystemPlugin()],
    )
    assert [tool.name for tool in with_plugin.tools] == ["read", "write", "edit", "search", "list", "glob"]
    read = next(tool for tool in with_plugin.tools if tool.name == "read")
    search = next(tool for tool in with_plugin.tools if tool.name == "search")
    read_result = read.handler(read.parse_args({"path": "missing.txt"}))
    search_result = search.handler(search.parse_args({"query": "missing"}))
    assert read_result.ok is False
    assert search_result.ok is True
    assert not missing.exists()


def test_filesystem_plugin_selection_validation(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="ordered sequence"):
        FilesystemPlugin(tools={"read"})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="duplicate"):
        FilesystemPlugin(tools=["read", "read"])
    with pytest.raises(ValueError, match="unknown FilesystemPlugin tool"):
        Harness(HarnessConfig(root=tmp_path), model=ScriptedModel([]), plugins=[FilesystemPlugin(tools=["missing"])])


def test_plugin_names_are_nonempty_and_unique(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        Harness(HarnessConfig(root=tmp_path), model=ScriptedModel([]), plugins=[StaticPlugin("")])
    with pytest.raises(ValueError, match="duplicate plugin name: same"):
        Harness(
            HarnessConfig(root=tmp_path),
            model=ScriptedModel([]),
            plugins=[StaticPlugin("same"), StaticPlugin("same")],
        )
    with pytest.raises(TypeError, match="returned an invalid binding"):
        Harness(
            HarnessConfig(root=tmp_path),
            model=ScriptedModel([]),
            plugins=[InvalidBindingPlugin()],  # type: ignore[list-item]
        )


def test_plugin_origin_cannot_spoof_another_plugin(tmp_path: Path) -> None:
    tool = ToolSpec(
        "owned",
        "owned",
        {"type": "object", "properties": {}},
        lambda _args: "ok",
        origin=ToolOrigin(plugin="spoofed", source="remote", attributes={"key": "value"}),
    )
    harness = Harness(
        HarnessConfig(root=tmp_path),
        model=ScriptedModel([]),
        plugins=[StaticPlugin("actual", PluginContribution(tools=(tool,)))],
    )

    assert harness.tools[0].origin == ToolOrigin(plugin="actual", source="remote", attributes={"key": "value"})


def test_duplicate_tool_error_names_both_origins(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"duplicate tool name: shared \(first and second\)"):
        Harness(
            HarnessConfig(root=tmp_path),
            model=ScriptedModel([]),
            plugins=[
                StaticPlugin("first", PluginContribution(tools=(_tool("shared"),))),
                StaticPlugin("second", PluginContribution(tools=(_tool("shared"),))),
            ],
        )


async def test_plugins_connect_before_first_run_hook_and_close_in_reverse(tmp_path: Path) -> None:
    events: list[str] = []
    connected_hook = Hook("run_start", lambda _ctx: events.append("hook:connected"))
    first = ConnectedPlugin("first", events, PluginContribution(hooks=(connected_hook,)))
    second = ConnectedPlugin("second", events, PluginContribution(tools=(_tool("dynamic"),)))
    session = ScriptedSession(start_turn=ModelTurn(text="done", raw={"id": "done"}))
    harness = Harness(
        HarnessConfig(root=tmp_path),
        model=ScriptedModel([session]),
        plugins=[first, second],
        hooks=[Hook("run_start", lambda _ctx: events.append("hook:direct"))],
    )

    assert "dynamic" not in [tool.name for tool in harness.tools]
    assert (await harness.run("go")).text == "done"
    assert events == ["enter:first", "enter:second", "hook:direct", "hook:connected"]
    assert "dynamic" in [tool.name for tool in harness.tools]

    await harness.aclose()
    assert events[-2:] == ["exit:second", "exit:first"]


async def test_concurrent_connect_enters_each_binding_once(tmp_path: Path) -> None:
    events: list[str] = []
    plugin = ConnectedPlugin("connected", events, PluginContribution())
    harness = Harness(HarnessConfig(root=tmp_path), model=ScriptedModel([]), plugins=[plugin])

    await asyncio.gather(harness.connect(), harness.connect(), harness.connect())

    assert plugin.attempts == 1
    await harness.aclose()


async def test_connection_failure_rolls_back_and_retry_is_clean(tmp_path: Path) -> None:
    events: list[str] = []
    first = ConnectedPlugin("first", events, PluginContribution(tools=(_tool("first_tool"),)))
    failing = ConnectedPlugin("failing", events, PluginContribution(), fail_first=True)
    harness = Harness(HarnessConfig(root=tmp_path), model=ScriptedModel([]), plugins=[first, failing])

    with pytest.raises(RuntimeError, match="failed:failing"):
        await harness.connect()
    assert events == ["enter:first", "enter:failing", "exit:first"]
    assert harness.tools == []

    await harness.connect()
    assert [tool.name for tool in harness.tools] == ["first_tool"]
    assert first.attempts == 2
    await harness.aclose()


async def test_connection_cancellation_rolls_back_without_run_hooks(tmp_path: Path) -> None:
    events: list[str] = []
    first = ConnectedPlugin("first", events, PluginContribution(tools=(_tool("first_tool"),)))

    class CancellingPlugin:
        name = "cancel"

        def bind(self, context: PluginContext) -> PluginBinding:
            @asynccontextmanager
            async def connect():
                events.append("enter:cancel")
                raise asyncio.CancelledError
                yield PluginContribution()  # pragma: no cover

            return PluginBinding(connect=connect)

    harness = Harness(
        HarnessConfig(root=tmp_path),
        model=ScriptedModel([]),
        plugins=[first, CancellingPlugin()],
        hooks=[Hook("run_start", lambda _ctx: events.append("hook:run_start"))],
    )

    with pytest.raises(asyncio.CancelledError):
        await harness.run("go")

    assert events == ["enter:first", "enter:cancel", "exit:first"]
    assert harness.tools == []


async def test_invalid_dynamic_contribution_is_not_committed(tmp_path: Path) -> None:
    events: list[str] = []
    plugin = ConnectedPlugin("bad", events, PluginContribution(tools=(_tool("subagent"),)))
    harness = Harness(HarnessConfig(root=tmp_path), model=ScriptedModel([]), plugins=[plugin])

    with pytest.raises(ValueError, match="reserved tool name"):
        await harness.connect()

    assert harness.tools == []
    assert events == ["enter:bad", "exit:bad"]


async def test_invalid_dynamic_hook_filter_is_not_committed(tmp_path: Path) -> None:
    events: list[str] = []
    hook = Hook("before_subagent_run", lambda _ctx: None, agents=["missing"])
    plugin = ConnectedPlugin("bad-hook", events, PluginContribution(hooks=(hook,)))
    harness = Harness(HarnessConfig(root=tmp_path), model=ScriptedModel([]), plugins=[plugin])

    with pytest.raises(ValueError, match="unknown subagent name"):
        await harness.connect()

    assert harness.hooks.hooks == []
    assert events == ["enter:bad-hook", "exit:bad-hook"]


def test_caller_hook_registry_is_copied(tmp_path: Path) -> None:
    direct = Hook("run_start", lambda _ctx: None)
    contributed = Hook("run_start", lambda _ctx: None)
    registry = HookRegistry([direct], strict_hooks=True)
    harness = Harness(
        HarnessConfig(root=tmp_path),
        model=ScriptedModel([]),
        plugins=[StaticPlugin("hooks", PluginContribution(hooks=(contributed,)))],
        hooks=registry,
    )

    assert registry.hooks == [direct]
    assert harness.hooks.hooks == [contributed, direct]
    assert harness.hooks.strict_hooks is True


def test_one_plugin_object_binds_independently_to_two_harnesses(tmp_path: Path) -> None:
    plugin = StaticPlugin("shared", PluginContribution(tools=(_tool("shared_tool"),)))
    first = Harness(HarnessConfig(root=tmp_path / "one"), model=ScriptedModel([]), plugins=[plugin])
    second = Harness(HarnessConfig(root=tmp_path / "two"), model=ScriptedModel([]), plugins=[plugin])

    assert plugin.bindings == 2
    assert first.tools[0] is not second.tools[0]


def test_core_does_not_import_filesystem_implementation() -> None:
    source = Path("thinharness/core.py").read_text(encoding="utf-8")

    assert "tools.filesystem" not in source
    assert "plugins.filesystem" not in source
