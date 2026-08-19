from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from fakes import ScriptedModel, ScriptedSession, echo_tool
from pydantic import BaseModel

import thinharness.core as core_module
from thinharness import (
    FilesystemPlugin,
    Harness,
    HarnessConfig,
    HarnessError,
    Hook,
    HookRegistry,
    ModelToolCall,
    ModelTurn,
    PluginBinding,
    PluginContext,
    PluginContribution,
    ToolOrigin,
    ToolSpec,
)
from thinharness.plugins import ToolOrigin as PluginToolOrigin
from thinharness.tools.base import ToolOrigin as DefinedToolOrigin


def _tool(name: str) -> ToolSpec:
    return ToolSpec(name, name, {"type": "object", "properties": {}}, lambda _args: "ok")


class StaticPlugin:
    def __init__(self, name: str, contribution: PluginContribution | None = None) -> None:
        self.name = name
        self.contribution = contribution or PluginContribution()
        self.bindings = 0
        self.contexts: list[PluginContext] = []

    def bind(self, context: PluginContext) -> PluginBinding:
        self.bindings += 1
        self.contexts.append(context)
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
        self.contexts: list[PluginContext] = []

    def bind(self, context: PluginContext) -> PluginBinding:
        self.contexts.append(context)

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
    assert first.contexts[0].root == tmp_path.resolve()
    assert first.contexts[0].model is harness.model
    assert second.contexts[0].model is harness.model
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


def test_one_plugin_object_receives_each_harness_root_and_model(tmp_path: Path) -> None:
    plugin = StaticPlugin("shared", PluginContribution(tools=(_tool("shared_tool"),)))
    first_model = ScriptedModel([])
    second_model = ScriptedModel([])
    first = Harness(HarnessConfig(root=tmp_path / "one"), model=first_model, plugins=[plugin])
    second = Harness(HarnessConfig(root=tmp_path / "two"), model=second_model, plugins=[plugin])

    assert plugin.bindings == 2
    assert [context.root for context in plugin.contexts] == [(tmp_path / "one").resolve(), (tmp_path / "two").resolve()]
    assert [context.model for context in plugin.contexts] == [first_model, second_model]
    assert first.tools[0] is not second.tools[0]


def test_filesystem_bind_performs_no_metadata_io(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path.resolve()

    def fail(*_args, **_kwargs):
        raise AssertionError("filesystem metadata used during bind")

    monkeypatch.setattr(Path, "resolve", fail)
    monkeypatch.setattr(Path, "exists", fail)
    monkeypatch.setattr(Path, "is_file", fail)

    binding = FilesystemPlugin(read_paths=["future"], write_paths=["outputs"]).bind(
        PluginContext(root=root, model=ScriptedModel([]))
    )

    assert [tool.name for tool in binding.static.tools] == ["read", "write", "edit", "search", "list", "glob"]


def test_tool_origin_is_exported_from_plugin_contract() -> None:
    assert PluginToolOrigin is ToolOrigin
    assert DefinedToolOrigin is ToolOrigin


def test_plugin_name_must_be_string(tmp_path: Path) -> None:
    plugin = StaticPlugin("valid")
    plugin.name = 1  # type: ignore[assignment]

    with pytest.raises(ValueError, match="non-empty string"):
        Harness(HarnessConfig(root=tmp_path), model=ScriptedModel([]), plugins=[plugin])


async def test_sequential_runs_reuse_one_connection(tmp_path: Path) -> None:
    events: list[str] = []
    plugin = ConnectedPlugin("connected", events, PluginContribution())
    model = ScriptedModel(
        [
            ScriptedSession(start_turn=ModelTurn(text="first", raw={"id": "first"})),
            ScriptedSession(start_turn=ModelTurn(text="second", raw={"id": "second"})),
        ]
    )
    harness = Harness(HarnessConfig(root=tmp_path), model=model, plugins=[plugin])

    assert (await harness.run("one")).text == "first"
    assert (await harness.run("two")).text == "second"
    assert plugin.attempts == 1

    await harness.aclose()


async def test_concurrent_failed_connection_is_shared_then_retryable(tmp_path: Path) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    class GatedPlugin:
        name = "gated"

        def __init__(self) -> None:
            self.attempts = 0

        def bind(self, context: PluginContext) -> PluginBinding:
            @asynccontextmanager
            async def connect():
                self.attempts += 1
                if self.attempts == 1:
                    entered.set()
                    await release.wait()
                    raise RuntimeError("shared failure")
                yield PluginContribution()

            return PluginBinding(connect=connect)

    plugin = GatedPlugin()
    harness = Harness(HarnessConfig(root=tmp_path), model=ScriptedModel([]), plugins=[plugin])
    calls = [asyncio.create_task(harness.connect()) for _ in range(3)]
    await entered.wait()
    await asyncio.sleep(0)
    release.set()
    results = await asyncio.gather(*calls, return_exceptions=True)

    assert plugin.attempts == 1
    assert all(isinstance(result, RuntimeError) and str(result) == "shared failure" for result in results)

    await harness.connect()
    assert plugin.attempts == 2
    await harness.aclose()


async def test_concurrent_cancelled_connection_is_shared_then_retryable(tmp_path: Path) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    class CancellingPlugin:
        name = "cancel"

        def __init__(self) -> None:
            self.attempts = 0

        def bind(self, context: PluginContext) -> PluginBinding:
            @asynccontextmanager
            async def connect():
                self.attempts += 1
                if self.attempts == 1:
                    entered.set()
                    await release.wait()
                    raise asyncio.CancelledError
                yield PluginContribution()

            return PluginBinding(connect=connect)

    plugin = CancellingPlugin()
    harness = Harness(HarnessConfig(root=tmp_path), model=ScriptedModel([]), plugins=[plugin])
    calls = [asyncio.create_task(harness.connect()) for _ in range(3)]
    await entered.wait()
    await asyncio.sleep(0)
    release.set()
    results = await asyncio.gather(*calls, return_exceptions=True)

    assert plugin.attempts == 1
    assert all(isinstance(result, asyncio.CancelledError) for result in results)

    await harness.connect()
    assert plugin.attempts == 2
    await harness.aclose()


async def test_aclose_cancels_inflight_connection_without_leak(tmp_path: Path) -> None:
    entered = asyncio.Event()
    exited = asyncio.Event()

    class SlowPlugin:
        name = "slow"

        def bind(self, context: PluginContext) -> PluginBinding:
            @asynccontextmanager
            async def connect():
                entered.set()
                try:
                    await asyncio.Event().wait()
                    yield PluginContribution()  # pragma: no cover
                finally:
                    exited.set()

            return PluginBinding(connect=connect)

    harness = Harness(HarnessConfig(root=tmp_path), model=ScriptedModel([]), plugins=[SlowPlugin()])
    connection = asyncio.create_task(harness.connect())
    await entered.wait()

    await harness.aclose()

    with pytest.raises(asyncio.CancelledError):
        await connection
    assert exited.is_set()
    assert harness._plugin_stack is None
    with pytest.raises(HarnessError, match="harness is closed"):
        await harness.connect()


async def test_aclose_propagates_caller_cancellation_after_connection_cleanup(tmp_path: Path) -> None:
    entered = asyncio.Event()
    cleanup_started = asyncio.Event()
    exited = asyncio.Event()

    class SlowCleanupPlugin:
        name = "slow-cleanup"

        def bind(self, context: PluginContext) -> PluginBinding:
            @asynccontextmanager
            async def connect():
                entered.set()
                try:
                    await asyncio.Event().wait()
                    yield PluginContribution()  # pragma: no cover
                finally:
                    cleanup_started.set()
                    try:
                        await asyncio.Event().wait()
                    finally:
                        exited.set()

            return PluginBinding(connect=connect)

    harness = Harness(HarnessConfig(root=tmp_path), model=ScriptedModel([]), plugins=[SlowCleanupPlugin()])
    connection = asyncio.create_task(harness.connect())
    await entered.wait()
    closing = asyncio.create_task(harness.aclose())
    await cleanup_started.wait()

    closing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await closing
    with pytest.raises(asyncio.CancelledError):
        await connection
    assert exited.is_set()
    assert harness._plugin_stack is None


async def test_failed_connection_attempts_every_plugin_cleanup(tmp_path: Path) -> None:
    events: list[str] = []

    class CleanupPlugin:
        def __init__(self, name: str, contribution: PluginContribution, *, fail_close: bool = False) -> None:
            self.name = name
            self.contribution = contribution
            self.fail_close = fail_close

        def bind(self, context: PluginContext) -> PluginBinding:
            @asynccontextmanager
            async def connect():
                try:
                    yield self.contribution
                finally:
                    events.append(self.name)
                    if self.fail_close:
                        raise RuntimeError(f"{self.name} close failed")

            return PluginBinding(connect=connect)

    first = CleanupPlugin("first", PluginContribution(), fail_close=True)
    second = CleanupPlugin("second", PluginContribution(tools=(_tool("subagent"),)))
    harness = Harness(HarnessConfig(root=tmp_path), model=ScriptedModel([]), plugins=[first, second])

    with pytest.raises(ValueError, match="reserved tool name") as raised:
        await harness.connect()

    assert events == ["second", "first"]
    assert any("cleanup also failed" in note for note in raised.value.__notes__)
    assert harness.tools == []


async def test_close_attempts_model_after_plugin_failure(tmp_path: Path) -> None:
    events: list[str] = []

    class FailingClosePlugin:
        name = "plugin"

        def bind(self, context: PluginContext) -> PluginBinding:
            @asynccontextmanager
            async def connect():
                try:
                    yield PluginContribution()
                finally:
                    events.append("plugin")
                    raise RuntimeError("plugin close failed")

            return PluginBinding(connect=connect)

    class ClosingProvider:
        name = "OpenAI"

        async def aclose(self) -> None:
            events.append("model")
            raise RuntimeError("model close failed")

    model = ScriptedModel([])
    model.provider = ClosingProvider()
    harness = Harness(
        HarnessConfig(root=tmp_path),
        model=model,
        plugins=[FailingClosePlugin()],
        _owns_model=True,
    )
    await harness.connect()

    with pytest.raises(RuntimeError, match="plugin close failed"):
        await harness.aclose()

    assert events == ["plugin", "model"]
    assert harness._plugin_stack is None


class _Answer(BaseModel):
    value: str


async def test_dynamic_structured_output_collision_rolls_back(tmp_path: Path) -> None:
    plugin = ConnectedPlugin("bad", [], PluginContribution(tools=(_tool("final_result"),)))
    harness = Harness(
        HarnessConfig(root=tmp_path, output_type=_Answer, output_mode="tool"),
        model=ScriptedModel([]),
        plugins=[plugin],
    )

    with pytest.raises(ValueError, match="reserved for structured output"):
        await harness.connect()
    assert harness.tools == []


async def test_dynamic_approval_tool_requires_resumable_model(tmp_path: Path) -> None:
    class NonResumableModel:
        model = "non-resumable"
        provider = type("Provider", (), {"name": "OpenAI"})()

        def new_session(self):
            raise AssertionError("not used")

    approval = ToolSpec("approve", "approve", {"type": "object", "properties": {}}, lambda _args: "ok", requires_approval=True)
    plugin = ConnectedPlugin("bad", [], PluginContribution(tools=(approval,)))
    harness = Harness(HarnessConfig(root=tmp_path), model=NonResumableModel(), plugins=[plugin])

    with pytest.raises(ValueError, match="resumable model"):
        await harness.connect()
    assert harness.tools == []


async def test_dynamic_non_callable_handler_rolls_back(tmp_path: Path) -> None:
    invalid = ToolSpec("invalid", "invalid", {"type": "object", "properties": {}}, None)  # type: ignore[arg-type]
    plugin = ConnectedPlugin("bad", [], PluginContribution(tools=(invalid,)))
    harness = Harness(HarnessConfig(root=tmp_path), model=ScriptedModel([]), plugins=[plugin])

    with pytest.raises(TypeError, match="not callable"):
        await harness.connect()
    assert harness.tools == []


async def test_connected_plugin_toolset_is_frozen_during_run(tmp_path: Path) -> None:
    outputs: list[str] = []

    class FreezeSession:
        def __init__(self) -> None:
            self.continues = 0

        async def start(self, prompt, constants, **_kwargs):
            assert [tool["name"] for tool in constants.tools] == ["register"]
            return ModelTurn(tool_calls=[ModelToolCall(id="call_1", name="register", arguments="{}")], raw={"id": "start"})

        async def continue_with_tools(self, tool_outputs, constants, **_kwargs):
            outputs.extend(output.output for output in tool_outputs)
            self.continues += 1
            if self.continues == 1:
                assert [tool["name"] for tool in constants.tools] == ["register"]
                return ModelTurn(tool_calls=[ModelToolCall(id="call_2", name="late", arguments="{}")], raw={"id": "late"})
            return ModelTurn(text="done", raw={"id": "done"})

        async def continue_with_user_text(self, text, constants, **_kwargs):
            raise AssertionError("not used")

        def dump_state(self):
            return {"kind": "scripted", "version": 1, "model": "scripted-model"}

    harness: Harness

    def register(_args):
        harness.add_tool(_tool("late"))
        return "registered"

    plugin = ConnectedPlugin(
        "dynamic",
        [],
        PluginContribution(
            tools=(
                ToolSpec(
                    "register",
                    "register",
                    {"type": "object", "properties": {}},
                    register,
                ),
            )
        ),
    )
    harness = Harness(HarnessConfig(root=tmp_path), model=ScriptedModel([FreezeSession()]), plugins=[plugin])

    assert (await harness.run("go")).text == "done"
    assert "late" in [tool.name for tool in harness.tools]
    late = json.loads(outputs[-1])
    assert late["ok"] is False
    assert "unknown tool late" in late["content"]

    await harness.aclose()


async def test_added_tool_survives_failed_connection_and_retry(tmp_path: Path) -> None:
    plugin = ConnectedPlugin("failing", [], PluginContribution(tools=(_tool("dynamic"),)), fail_first=True)
    harness = Harness(HarnessConfig(root=tmp_path), model=ScriptedModel([]), plugins=[plugin])
    harness.add_tool(_tool("direct"))

    with pytest.raises(RuntimeError, match="failed:failing"):
        await harness.connect()
    assert [tool.name for tool in harness.tools] == ["direct"]

    await harness.connect()
    assert [tool.name for tool in harness.tools] == ["direct", "dynamic"]
    await harness.aclose()


def test_missing_root_filesystem_tools_and_write_creation(tmp_path: Path) -> None:
    root = tmp_path / "missing"
    harness = Harness(
        HarnessConfig(root=root),
        model=ScriptedModel([]),
        plugins=[FilesystemPlugin(tools=["list", "glob", "jsonl_search", "write"])],
    )
    by_name = {tool.name: tool for tool in harness.tools}

    listed = by_name["list"].handler(by_name["list"].parse_args({"path": "."}))
    globbed = by_name["glob"].handler(by_name["glob"].parse_args({"pattern": "**/*"}))
    jsonl = by_name["jsonl_search"].handler(by_name["jsonl_search"].parse_args({"path": "."}))

    assert listed.ok is False
    assert globbed.ok is True
    assert jsonl.ok is True
    assert not root.exists()

    written = by_name["write"].handler(by_name["write"].parse_args({"path": "nested/out.txt", "content": "ok"}))
    assert written.ok is True
    assert (root / "nested/out.txt").read_text() == "ok"


def test_shared_filesystem_plugin_has_independent_binding_state(tmp_path: Path) -> None:
    plugin = FilesystemPlugin(tools=["write"])
    first = Harness(HarnessConfig(root=tmp_path / "first"), model=ScriptedModel([]), plugins=[plugin])
    second = Harness(HarnessConfig(root=tmp_path / "second"), model=ScriptedModel([]), plugins=[plugin])

    first.tools[0].handler(first.tools[0].parse_args({"path": "same.txt", "content": "first"}))
    second.tools[0].handler(second.tools[0].parse_args({"path": "same.txt", "content": "second"}))

    assert (tmp_path / "first/same.txt").read_text() == "first"
    assert (tmp_path / "second/same.txt").read_text() == "second"


def test_harness_plugins_preserve_order_and_empty_filesystem_instruction(tmp_path: Path) -> None:
    first = StaticPlugin("first")
    filesystem = FilesystemPlugin(tools=[])
    harness = Harness(
        HarnessConfig(root=tmp_path),
        model=ScriptedModel([]),
        plugins=[first, filesystem],
    )

    assert harness.plugins == (first, filesystem)
    assert harness.tools == []
    assert harness.system_instructions().count(f"Workspace root: {tmp_path.resolve()}") == 1


def test_core_does_not_import_filesystem_implementation() -> None:
    source = Path(core_module.__file__).read_text(encoding="utf-8")

    assert "tools.filesystem" not in source
    assert "plugins.filesystem" not in source
