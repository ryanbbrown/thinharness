from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from fakes import (
    FailingSession,
    MultiCallClient,
    ScriptedModel,
    ScriptedSession,
    _fake_openai,
    echo_tool,
    slow_tool,
    tool_output,
)

from thinharness import (
    AfterToolCallContext,
    BeforeToolCallContext,
    Harness,
    HarnessConfig,
    HarnessError,
    Hook,
    HookRegistry,
    LimitReachedContext,
    RunEndContext,
    RunStartContext,
    ToolSpec,
    UserPromptSubmitContext,
    create_subagent_tool,
)
from thinharness.hooks import current_tool_runtime_context
from thinharness.providers import ModelToolCall, ModelTurn


def test_current_tool_runtime_context_is_unset_outside_tool_call() -> None:
    assert current_tool_runtime_context() is None

def test_hook_registry_rejects_invalid_filters() -> None:
    with pytest.raises(ValueError, match="tools filter"):
        Hook("run_start", lambda ctx: None, tools=["read"])
    with pytest.raises(ValueError, match="agents filter"):
        Hook("limit_reached", lambda ctx: None, agents=["research"])
    with pytest.raises(ValueError, match="cannot be empty"):
        Hook("before_tool_call", lambda ctx: None, tools=[])
    with pytest.raises(ValueError, match="unknown hook event"):
        Hook("unknown", lambda ctx: None)  # type: ignore[arg-type]

def test_hook_filter_warnings_wait_for_constructor_tools(tmp_path: Path, caplog) -> None:
    hook = Hook("before_tool_call", lambda ctx: None, tools=["second"])

    Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[]),
        model=ScriptedModel([]),
        tools=[
            ToolSpec("first", "first", {"type": "object", "properties": {}}, lambda args: "first"),
            ToolSpec("second", "second", {"type": "object", "properties": {}}, lambda args: "second"),
        ],
        hooks=[hook],
    )

    assert "unknown tool name" not in caplog.text

def test_run_end_fires_when_new_session_fails(tmp_path: Path) -> None:
    events = []

    class BrokenModel(ScriptedModel):
        def new_session(self):
            raise RuntimeError("no session")

    def on_end(ctx):
        assert isinstance(ctx, RunEndContext)
        events.append((ctx.stop_reason, type(ctx.error).__name__, ctx.usage.model_requests))

    harness = Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[]),
        model=BrokenModel([]),
        hooks=[Hook("run_end", on_end)],
    )

    with pytest.raises(RuntimeError, match="no session"):
        harness.run_sync("go")

    assert events == [("error", "RuntimeError", 0)]

def test_run_end_fires_for_provider_and_unexpected_errors(tmp_path: Path) -> None:
    events = []
    provider_harness = Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[]),
        model=ScriptedModel([FailingSession()]),
        hooks=[Hook("run_end", lambda ctx: events.append((ctx.stop_reason, type(ctx.error).__name__)))],
    )

    with pytest.raises(HarnessError, match="child failed"):
        provider_harness.run_sync("go")

    unexpected = ScriptedSession(start_turn=ModelTurn(), on_start=lambda *_args: (_ for _ in ()).throw(ValueError("boom")))
    error_harness = Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[]),
        model=ScriptedModel([unexpected]),
        hooks=[Hook("run_end", lambda ctx: events.append((ctx.stop_reason, type(ctx.error).__name__)))],
    )

    with pytest.raises(ValueError, match="boom"):
        error_harness.run_sync("go")

    assert events == [("provider_error", "HarnessError"), ("error", "ValueError")]

async def test_strict_run_end_hook_resets_running_flag(tmp_path: Path) -> None:
    calls = 0

    def fail_once(ctx):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("end hook failed")

    harness = Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[], strict_hooks=True),
        model=ScriptedModel([
            ScriptedSession(start_turn=ModelTurn(text="first", raw={"id": "first"})),
            ScriptedSession(start_turn=ModelTurn(text="second", raw={"id": "second"})),
        ]),
        hooks=[Hook("run_end", fail_once)],
    )

    with pytest.raises(RuntimeError, match="end hook failed"):
        await harness.run("first")

    assert (await harness.run("second")).text == "second"

def test_run_hooks_append_prompt_context_and_report_usage(tmp_path: Path) -> None:
    captured = {}

    def on_start(prompt, _instructions, _tools, _metadata, _previous_response_id):
        captured["prompt"] = prompt

    def add_context(ctx):
        assert isinstance(ctx, UserPromptSubmitContext)
        ctx.additional_context.append("policy: keep it short")

    events = []
    hooks = [
        Hook("run_start", lambda ctx: events.append((ctx.event, isinstance(ctx, RunStartContext)))),
        Hook("user_prompt_submit", add_context),
        Hook("run_end", lambda ctx: events.append((ctx.event, isinstance(ctx, RunEndContext), ctx.result.usage.model_requests))),
    ]
    session = ScriptedSession(start_turn=ModelTurn(text="done", raw={"id": "done"}), on_start=on_start)
    harness = Harness(HarnessConfig(root=tmp_path, builtin_tools=[]), model=ScriptedModel([session]), hooks=hooks)

    result = harness.run_sync("summarize")

    assert captured["prompt"] == "summarize\n\n<hook_context>\npolicy: keep it short\n</hook_context>"
    assert result.usage.model_requests == 1
    assert result.usage.tool_calls == 0
    assert result.stop_reason == "end_turn"
    assert events == [("run_start", True), ("run_end", True, 1)]

def test_user_prompt_hook_can_cancel_before_model_request(tmp_path: Path) -> None:
    events = []

    def cancel(ctx):
        assert isinstance(ctx, UserPromptSubmitContext)
        events.append(ctx.event)
        ctx.cancelled = True
        ctx.cancel_reason = "blocked"

    def on_run_end(ctx):
        assert isinstance(ctx, RunEndContext)
        events.append((ctx.event, ctx.stop_reason, type(ctx.error).__name__))

    session = ScriptedSession(
        start_turn=ModelTurn(text="should not run", raw={}),
        on_start=lambda *_args: pytest.fail("model should not be called"),
    )
    harness = Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[]),
        model=ScriptedModel([session]),
        hooks=[Hook("user_prompt_submit", cancel), Hook("run_end", on_run_end)],
    )

    with pytest.raises(HarnessError, match="run blocked by hook: blocked"):
        harness.run_sync("nope")

    assert events == ["user_prompt_submit", ("run_end", "cancelled_by_hook", "HarnessError")]

def test_same_harness_reentrant_run_is_rejected(tmp_path: Path) -> None:
    captured = []
    harness = Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[]),
        model=ScriptedModel([ScriptedSession(start_turn=ModelTurn(text="done", raw={"id": "done"}))]),
    )

    def reenter(ctx):
        with pytest.raises(HarnessError, match="not re-entrant") as exc_info:
            harness.run_sync("nested")
        captured.append(str(exc_info.value))

    harness.hooks.hooks.append(Hook("user_prompt_submit", reenter))

    assert harness.run_sync("outer").text == "done"
    assert captured == ["Harness.run is not re-entrant"]

def test_tool_hooks_filter_cancel_mutate_and_preserve_tool_index(tmp_path: Path) -> None:
    client = MultiCallClient([("block", "{}"), ("ok", "{}")])
    indexes = []

    def before(ctx):
        assert isinstance(ctx, BeforeToolCallContext)
        indexes.append((ctx.tool_name, ctx.tool_index))
        if ctx.tool_name == "block":
            ctx.cancelled = True
            ctx.cancel_reason = "no"

    def after(ctx):
        assert isinstance(ctx, AfterToolCallContext)
        if ctx.tool_name == "ok":
            ctx.output = json.dumps({"ok": True, "content": "rewritten", "metadata": {}})

    harness = Harness(
        HarnessConfig(root=tmp_path, model="openai:test-model", builtin_tools=[]),
        model=_fake_openai(client),
        tools=[
            ToolSpec("block", "blocked", {"type": "object", "properties": {}}, lambda args: "bad"),
            echo_tool(),
            ToolSpec("ok", "ok", {"type": "object", "properties": {}}, lambda args: "original"),
        ],
        hooks=[
            Hook("before_tool_call", before),
            Hook("after_tool_call", after, tools=["ok"]),
        ],
    )

    result = harness.run_sync("go")

    outputs = [tool_output(item["output"]) for item in client.payloads[1]["input"]]
    assert [name_index for name_index in indexes] == [("block", 0), ("ok", 1)]
    assert outputs[0]["metadata"]["error_type"] == "ToolCallCancelled"
    assert outputs[1]["content"] == "rewritten"
    assert result.usage.tool_calls == 2
    assert result.usage.cancelled_tool_calls == 1
    assert len(result.tool_call_records) == 2

def test_tool_hook_metadata_is_copied_between_before_and_after_hooks(tmp_path: Path) -> None:
    client = MultiCallClient([("ok", "{}")])
    seen = []

    def before(ctx):
        assert isinstance(ctx, BeforeToolCallContext)
        seen.append(("before", dict(ctx.metadata)))
        ctx.metadata["conversation_id"] = "mutated"
        ctx.metadata["new"] = "ignored"

    def after(ctx):
        assert isinstance(ctx, AfterToolCallContext)
        seen.append(("after", dict(ctx.metadata)))

    harness = Harness(
        HarnessConfig(root=tmp_path, model="openai:test-model", builtin_tools=[]),
        model=_fake_openai(client),
        tools=[ToolSpec("ok", "ok", {"type": "object", "properties": {}}, lambda args: "ok")],
        hooks=[Hook("before_tool_call", before), Hook("after_tool_call", after)],
    )

    harness.run_sync("go", metadata={"conversation_id": "conv-1", "extra": "hook-only"})

    assert seen == [
        ("before", {"conversation_id": "conv-1", "extra": "hook-only"}),
        ("after", {"conversation_id": "conv-1", "extra": "hook-only"}),
    ]

def test_after_tool_hooks_see_refreshed_parsed_output(tmp_path: Path) -> None:
    client = MultiCallClient([("ok", "{}")])
    seen = []

    def rewrite(ctx):
        ctx.output = json.dumps({"ok": True, "content": "changed", "metadata": {"stage": 1}})

    def observe(ctx):
        assert isinstance(ctx, AfterToolCallContext)
        seen.append(ctx.parsed_output)

    harness = Harness(
        HarnessConfig(root=tmp_path, model="openai:test-model", builtin_tools=[]),
        model=_fake_openai(client),
        tools=[ToolSpec("ok", "ok", {"type": "object", "properties": {}}, lambda args: "original")],
        hooks=[Hook("after_tool_call", rewrite), Hook("after_tool_call", observe)],
    )

    harness.run_sync("go")

    assert seen == [{"ok": True, "content": "changed", "metadata": {"stage": 1}}]

def test_after_tool_hook_strict_exception_preserves_original_error(tmp_path: Path) -> None:
    client = MultiCallClient([("ok", "{}")])

    def fail(ctx):
        raise RuntimeError("after failed")

    harness = Harness(
        HarnessConfig(root=tmp_path, model="openai:test-model", builtin_tools=[], strict_hooks=True),
        model=_fake_openai(client),
        tools=[ToolSpec("ok", "ok", {"type": "object", "properties": {}}, lambda args: "ok")],
        hooks=[Hook("after_tool_call", fail)],
    )

    with pytest.raises(RuntimeError, match="after failed"):
        harness.run_sync("go")

def test_after_tool_hook_parsed_output_uses_normalized_invalid_output() -> None:
    seen = []
    registry = HookRegistry([
        Hook("after_tool_call", lambda ctx: seen.append(ctx.parsed_output)),
    ])
    ctx = AfterToolCallContext(
        harness=None,  # type: ignore[arg-type]
        call_id="call_1",
        tool_name="raw",
        arguments="{}",
        original_output="not json",
        output="not json",
        duration_ms=0,
    )

    registry.fire_after_tool_call(ctx)

    assert seen == [{"ok": False, "content": "not json", "metadata": {"error_type": "InvalidToolOutput"}}]

def test_strict_tool_hook_exception_surfaces_from_parallel_worker(tmp_path: Path) -> None:
    client = MultiCallClient([("a", "{}"), ("b", "{}")])

    def fail_for_b(ctx):
        if ctx.tool_name == "b":
            raise RuntimeError("strict hook failed")

    harness = Harness(
        HarnessConfig(root=tmp_path, model="openai:test-model", builtin_tools=[], strict_hooks=True),
        model=_fake_openai(client),
        tools=[slow_tool("a", 0.01), slow_tool("b", 0.01)],
        hooks=[Hook("before_tool_call", fail_for_b)],
    )

    with pytest.raises(RuntimeError, match="strict hook failed"):
        harness.run_sync("go")

def test_strict_tool_hook_exception_counts_attempted_calls_in_run_end_usage(tmp_path: Path) -> None:
    client = MultiCallClient([("a", "{}"), ("b", "{}")])
    run_end_usage = []

    def fail_for_a(ctx):
        if ctx.tool_name == "a":
            raise RuntimeError("strict hook failed")

    def on_run_end(ctx):
        assert isinstance(ctx, RunEndContext)
        assert ctx.usage is not None
        run_end_usage.append((ctx.stop_reason, ctx.usage.tool_calls, ctx.usage.cancelled_tool_calls))

    harness = Harness(
        HarnessConfig(root=tmp_path, model="openai:test-model", builtin_tools=[], strict_hooks=True),
        model=_fake_openai(client),
        tools=[slow_tool("a", 0.01), slow_tool("b", 0.01)],
        hooks=[Hook("before_tool_call", fail_for_a), Hook("run_end", on_run_end)],
    )

    with pytest.raises(RuntimeError, match="strict hook failed"):
        harness.run_sync("go")

    assert run_end_usage == [("error", 2, 0)]

async def test_strict_tool_hook_cancels_async_sibling_before_completion(tmp_path: Path) -> None:
    client = MultiCallClient([("fail", "{}"), ("wait", "{}")])
    cancelled = asyncio.Event()

    async def wait(_args):
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    def fail_for_fail(ctx):
        if ctx.tool_name == "fail":
            raise RuntimeError("strict hook failed")

    harness = Harness(
        HarnessConfig(root=tmp_path, model="openai:test-model", builtin_tools=[], strict_hooks=True),
        model=_fake_openai(client),
        tools=[
            ToolSpec("wait", "wait", {"type": "object", "properties": {}}, wait),
            ToolSpec("fail", "fail", {"type": "object", "properties": {}}, lambda args: "should not run"),
        ],
        hooks=[Hook("before_tool_call", fail_for_fail)],
    )

    task = asyncio.create_task(harness.run("go"))
    with pytest.raises(RuntimeError, match="strict hook failed"):
        await asyncio.wait_for(task, timeout=1)

    assert cancelled.is_set()

def test_explicit_hook_registry_strict_mode_is_preserved(tmp_path: Path) -> None:
    registry = HookRegistry([Hook("user_prompt_submit", lambda ctx: (_ for _ in ()).throw(RuntimeError("strict registry")))], strict_hooks=True)
    harness = Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[], strict_hooks=False),
        model=ScriptedModel([ScriptedSession(start_turn=ModelTurn(text="done", raw={"id": "done"}))]),
        hooks=registry,
    )

    with pytest.raises(RuntimeError, match="strict registry"):
        harness.run_sync("go")

def test_bare_harness_error_reports_error_stop_reason(tmp_path: Path) -> None:
    events = []

    class BareHarnessErrorSession:
        async def start(self, *, prompt, instructions, tools, metadata=None, previous_response_id=None, structured_output=None, notices=None):
            return ModelTurn(tool_calls=[ModelToolCall(id="call_1", name="ok", arguments="{}")], raw={"id": "start"})

        async def continue_with_tools(self, outputs, *, instructions=None, tools, metadata=None, structured_output=None, notices=None):
            raise HarnessError("bare harness error")

        async def continue_with_user_message(self, message, *, instructions=None, tools, metadata=None, structured_output=None, notices=None):
            raise HarnessError("bare harness error")

    harness = Harness(
        HarnessConfig(root=tmp_path, model="openai:test-model", builtin_tools=[]),
        model=ScriptedModel([BareHarnessErrorSession()]),
        tools=[ToolSpec("ok", "ok", {"type": "object", "properties": {}}, lambda args: "ok")],
        hooks=[Hook("run_end", lambda ctx: events.append((ctx.stop_reason, type(ctx.error).__name__)))],
    )

    with pytest.raises(HarnessError, match="bare harness error"):
        harness.run_sync("go")

    assert events == [("error", "HarnessError")]

def test_explicit_limits_fire_limit_hook_and_run_end(tmp_path: Path) -> None:
    client = MultiCallClient([("a", "{}"), ("b", "{}"), ("c", "{}")])
    events = []

    def on_limit(ctx):
        assert isinstance(ctx, LimitReachedContext)
        events.append((ctx.event, ctx.limit_kind, ctx.limit_value, ctx.current_count))

    def on_end(ctx):
        assert isinstance(ctx, RunEndContext)
        events.append((ctx.event, ctx.stop_reason, ctx.usage.tool_calls))

    harness = Harness(
        HarnessConfig(root=tmp_path, model="openai:test-model", builtin_tools=[], max_tool_calls=2),
        model=_fake_openai(client),
        tools=[slow_tool("a", 0), slow_tool("b", 0), slow_tool("c", 0)],
        hooks=[Hook("limit_reached", on_limit), Hook("run_end", on_end)],
    )

    with pytest.raises(HarnessError, match="tool calls would exceed max_tool_calls=2"):
        harness.run_sync("go")

    assert events == [("limit_reached", "tool_calls", 2, 3), ("run_end", "limit_reached", 0)]
    assert client.invocations == 1

def test_max_model_requests_limits_provider_continuations(tmp_path: Path) -> None:
    immediate = Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[], max_model_requests=1),
        model=ScriptedModel([ScriptedSession(start_turn=ModelTurn(text="done", raw={"id": "done"}))]),
    )
    assert immediate.run_sync("go").usage.model_requests == 1

    client = MultiCallClient([("ok", "{}")])
    limited = Harness(
        HarnessConfig(root=tmp_path, model="openai:test-model", builtin_tools=[], max_model_requests=1),
        model=_fake_openai(client),
        tools=[ToolSpec("ok", "ok", {"type": "object", "properties": {}}, lambda args: "ok")],
    )
    with pytest.raises(HarnessError, match="max_model_requests=1"):
        limited.run_sync("go")

    allowed_client = MultiCallClient([("ok", "{}")])
    allowed = Harness(
        HarnessConfig(root=tmp_path, model="openai:test-model", builtin_tools=[], max_model_requests=2),
        model=_fake_openai(allowed_client),
        tools=[ToolSpec("ok", "ok", {"type": "object", "properties": {}}, lambda args: "ok")],
    )
    assert allowed.run_sync("go").usage.model_requests == 2

def test_strict_subagent_hook_exception_surfaces_to_parent_run(tmp_path: Path) -> None:
    parent_call = ModelTurn(
        tool_calls=[ModelToolCall(id="call_1", name="subagent", arguments='{"task":"help"}')],
        raw={"id": "parent-start"},
    )
    parent = ScriptedSession(start_turn=parent_call, continue_turn=ModelTurn(text="should not continue", raw={}))

    def fail(ctx):
        raise RuntimeError("strict subagent hook failed")

    harness = Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[], strict_hooks=True),
        model=ScriptedModel([parent]),
        hooks=[Hook("before_subagent_run", fail)],
    )
    harness.add_tool(create_subagent_tool(harness, []))

    with pytest.raises(RuntimeError, match="strict subagent hook failed"):
        harness.run_sync("delegate")
