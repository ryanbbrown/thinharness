from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import pytest
from fakes import FakeTracer, ScriptedModel, ScriptedSession, tool_output
from pydantic import BaseModel

from thinharness import (
    AfterToolCallContext,
    BeforeToolCallContext,
    Harness,
    HarnessConfig,
    HarnessError,
    Hook,
    ModelRetry,
    SubAgentConfig,
    ToolSpec,
    ToolStructuredOutput,
    TracingOptions,
)
from thinharness.hooks import current_tool_call_context, current_tool_runtime_context
from thinharness.providers import ModelToolCall, ModelTurn
from thinharness.tool_execution import BackgroundToolManager, BackgroundToolStart
from thinharness.tracing import RunTracer


class SequenceSession:
    """Script one start turn and a sequence of continuations."""

    def __init__(self, start_turn: ModelTurn, *continue_turns: ModelTurn) -> None:
        self.start_turn = start_turn
        self.continue_turns = list(continue_turns)
        self.tool_outputs = []
        self.user_messages = []
        self.continuation_tools = []
        self.notice_calls = []

    async def start(self, *, prompt, instructions, tools, metadata=None, previous_response_id=None, structured_output=None, notices=None):
        """Return the scripted first turn."""
        self.start_tools = tools
        self.notice_calls.append(("start", list(notices or [])))
        return self.start_turn

    async def continue_with_tools(self, outputs, *, instructions=None, tools, metadata=None, structured_output=None, notices=None):
        """Record tool outputs and return the next scripted turn."""
        self.tool_outputs.append(outputs)
        self.continuation_tools.append(tools)
        self.notice_calls.append(("continue_with_tools", list(notices or [])))
        return self._next_turn()

    async def continue_with_user_message(self, message, *, instructions=None, tools, metadata=None, structured_output=None, notices=None):
        """Record user-message continuations and return the next scripted turn."""
        self.user_messages.append(message)
        self.notice_calls.append(("continue_with_user_message", list(notices or [])))
        return self._next_turn()

    async def continue_with_user_prompt(self, *, prompt, instructions, tools, metadata=None, structured_output=None, notices=None):
        """No tests in this file expect resumed prompts."""
        raise AssertionError("unexpected resumed prompt")

    def dump_state(self):
        """Return no resume state for these scripted sessions."""
        return None

    def _next_turn(self) -> ModelTurn:
        if not self.continue_turns:
            raise AssertionError("unexpected continuation")
        return self.continue_turns.pop(0)


class Person(BaseModel):
    name: str


def _call(name: str, args: str, call_id: str = "call_1") -> ModelToolCall:
    """Build a normalized model tool call."""
    return ModelToolCall(id=call_id, name=name, arguments=args)


async def test_drain_ready_harvests_completed_tasks_in_completion_order() -> None:
    async def delayed(label: str, delay: float):
        await asyncio.sleep(delay)
        return label

    manager = BackgroundToolManager(run_tracer=RunTracer([]))
    starts = [
        BackgroundToolStart("bg_1", "call_1", "slow", "{}", ToolSpec("slow", "Slow", {}, lambda _args: delayed("slow", 0.03)), {}),
        BackgroundToolStart("bg_2", "call_2", "fast", "{}", ToolSpec("fast", "Fast", {}, lambda _args: delayed("fast", 0.01)), {}),
        BackgroundToolStart("bg_3", "call_3", "mid", "{}", ToolSpec("mid", "Mid", {}, lambda _args: delayed("mid", 0.02)), {}),
    ]

    manager.start_many(starts)
    await asyncio.sleep(0.06)

    assert [completion.task_id for completion in manager.drain_ready()] == ["bg_2", "bg_3", "bg_1"]


async def test_wait_next_ready_returns_ready_completion_without_waiting() -> None:
    manager = BackgroundToolManager(run_tracer=RunTracer([]))
    manager.start(BackgroundToolStart("bg_1", "call_1", "fast", "{}", ToolSpec("fast", "Fast", {}, lambda _args: "fast"), {}))
    await asyncio.sleep(0)

    completion = await asyncio.wait_for(manager.wait_next_ready(), timeout=0.01)

    assert completion.task_id == "bg_1"


async def test_cancel_and_drain_returns_ready_completion_before_cancelling_pending() -> None:
    fast_done = asyncio.Event()
    release = asyncio.Event()

    async def fast(_args):
        fast_done.set()
        return "ready"

    async def slow(_args):
        await release.wait()
        return "late"

    manager = BackgroundToolManager(run_tracer=RunTracer([]))
    manager.start(BackgroundToolStart("bg_1", "call_1", "fast", "{}", ToolSpec("fast", "Fast", {}, fast), {}))
    manager.start(BackgroundToolStart(
        "bg_2",
        "call_2",
        "slow",
        "{}",
        ToolSpec("slow", "Slow", {}, slow),
        {},
    ))
    await asyncio.wait_for(fast_done.wait(), timeout=0.5)
    await asyncio.sleep(0)

    completions = await manager.cancel_and_drain()

    assert [(completion.task_id, completion.event) for completion in completions] == [("bg_1", "completed"), ("bg_2", "cancelled")]


def test_tool_spec_background_schema_and_validation() -> None:
    spec = ToolSpec("echo", "Echo", {"type": "object", "properties": {}, "additionalProperties": False}, lambda args: "ok", background="model")
    schema = spec.response_tool(include_background=True)["parameters"]

    assert spec.background == "model"
    assert schema["properties"]["_background"]["type"] == "boolean"
    assert schema["additionalProperties"] is False
    assert "_background" not in schema.get("required", [])
    assert "_background" not in spec.parameters["properties"]
    with pytest.raises(ValueError, match="sequential tools"):
        ToolSpec("bad", "Bad", {"type": "object", "properties": {}}, lambda args: "ok", sequential=True, background="model")
    with pytest.raises(ValueError, match="already defines"):
        ToolSpec("collision", "Bad", {"type": "object", "properties": {"_background": {"type": "boolean"}}}, lambda args: "ok").response_tool(
            include_background=True
        )


def test_harness_schema_exposes_only_model_choice_background(tmp_path: Path) -> None:
    model_tool = ToolSpec("model_bg", "Model", {"type": "object", "properties": {}}, lambda args: "ok", background="model")
    always_tool = ToolSpec("always_bg", "Always", {"type": "object", "properties": {}}, lambda args: "ok", background="always")
    never_tool = ToolSpec("never_bg", "Never", {"type": "object", "properties": {}}, lambda args: "ok")
    harness = Harness(HarnessConfig(root=tmp_path, builtin_tools=[]), model=ScriptedModel([]), tools=[model_tool, always_tool, never_tool])

    schemas = {tool["name"]: tool["parameters"] for tool in harness.tool_schemas()}

    assert "_background" in schemas["model_bg"]["properties"]
    assert "_background" not in schemas["always_bg"].get("properties", {})
    assert "_background" not in schemas["never_bg"].get("properties", {})
    sequential = Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[], tool_execution="sequential"),
        model=ScriptedModel([]),
        tools=[model_tool],
    )
    assert "_background" not in sequential.tool_schemas()[0]["parameters"].get("properties", {})
    with pytest.raises(ValueError, match="background='always'"):
        Harness(HarnessConfig(root=tmp_path, builtin_tools=[], tool_execution="sequential"), model=ScriptedModel([]), tools=[always_tool])


def test_sequential_mode_does_not_advertise_background(tmp_path: Path) -> None:
    harness = Harness(
        HarnessConfig(root=tmp_path, builtin_tools=["parallel_llm", "subagent"], tool_execution="sequential"),
        model=ScriptedModel([]),
    )

    model_facing = json.dumps(harness.tool_schemas(), ensure_ascii=False) + "\n" + harness.system_instructions()

    assert "_background" not in model_facing
    assert "background mode" not in model_facing
    assert "background execution" not in model_facing


def test_background_model_choice_returns_start_notice_before_tool_finishes(tmp_path: Path) -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    async def slow(_args):
        started.set()
        await release.wait()
        return "late"

    session = SequenceSession(
        ModelTurn(tool_calls=[_call("slow", '{"_background":true}')], raw={"id": "start"}),
        ModelTurn(text="intermediate", raw={"id": "intermediate"}),
        ModelTurn(text="done", raw={"id": "done"}),
    )
    harness = Harness(HarnessConfig(root=tmp_path, builtin_tools=[]), model=ScriptedModel([session]), tools=[
        ToolSpec("slow", "Slow", {"type": "object", "properties": {}, "additionalProperties": False}, slow, background="model"),
    ])

    async def run() -> None:
        task = asyncio.create_task(harness.run("go"))
        await asyncio.wait_for(started.wait(), timeout=1)
        notice = tool_output(session.tool_outputs[0][0].output)
        assert notice["metadata"] == {"background_task_id": "bg_1", "tool_name": "slow", "status": "running"}
        assert session.user_messages == []
        release.set()
        result = await asyncio.wait_for(task, timeout=1)
        assert result.text == "done"

    asyncio.run(run())
    assert session.user_messages and "Background task bg_1 completed." in session.user_messages[0]


def test_background_always_starts_without_schema_argument(tmp_path: Path) -> None:
    session = SequenceSession(
        ModelTurn(tool_calls=[_call("always", "{}")], raw={"id": "start"}),
        ModelTurn(text="first final", raw={"id": "first"}),
        ModelTurn(text="done", raw={"id": "done"}),
    )
    harness = Harness(HarnessConfig(root=tmp_path, builtin_tools=[]), model=ScriptedModel([session]), tools=[
        ToolSpec("always", "Always", {"type": "object", "properties": {}, "additionalProperties": False}, lambda args: "late", background="always"),
    ])

    result = harness.run_sync("go")

    assert result.text == "done"
    assert tool_output(session.tool_outputs[0][0].output)["metadata"]["background_task_id"] == "bg_1"


def test_background_strips_private_argument_before_validation_and_manual_invocation(tmp_path: Path) -> None:
    class Args(BaseModel):
        value: str

    seen = []
    session = SequenceSession(
        ModelTurn(tool_calls=[
            _call("typed", '{"value":"a","_background":false}', "call_1"),
            _call("manual", '{"value":"b","_background":false}', "call_2"),
        ], raw={"id": "start"}),
        ModelTurn(text="done", raw={"id": "done"}),
    )
    harness = Harness(HarnessConfig(root=tmp_path, builtin_tools=[]), model=ScriptedModel([session]), tools=[
        ToolSpec("typed", "Typed", Args, lambda args: seen.append(args.value) or "typed", background="model"),
        ToolSpec(
            "manual",
            "Manual",
            {"type": "object", "properties": {"value": {"type": "string"}}},
            lambda args: seen.append(sorted(args)) or "manual",
            background="model",
        ),
    ])

    assert harness.run_sync("go").text == "done"

    assert seen == ["a", ["value"]]


def test_final_text_waits_for_background_completion_and_continues(tmp_path: Path) -> None:
    session = SequenceSession(
        ModelTurn(tool_calls=[_call("slow", '{"_background":true}')], raw={"id": "start"}),
        ModelTurn(text="too early", raw={"id": "early"}),
        ModelTurn(text="final", raw={"id": "final"}),
    )
    harness = Harness(HarnessConfig(root=tmp_path, builtin_tools=[]), model=ScriptedModel([session]), tools=[
        ToolSpec("slow", "Slow", {"type": "object", "properties": {}}, lambda args: "late", background="model"),
    ])

    result = harness.run_sync("go")

    assert result.text == "final"
    assert "Output:" in session.user_messages[0]
    assert result.usage.tool_calls == 1
    assert result.tool_call_records[1]["background"]["event"] == "completed"


def test_final_result_tool_is_paired_when_background_completion_defers_final(tmp_path: Path) -> None:
    session = SequenceSession(
        ModelTurn(tool_calls=[_call("slow", '{"_background":true}')], raw={"id": "start"}),
        ModelTurn(tool_calls=[_call("final_result", '{"name":"early"}', "final_1")], raw={"id": "early"}),
        ModelTurn(tool_calls=[_call("final_result", '{"name":"done"}', "final_2")], raw={"id": "final"}),
    )
    harness = Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[], output_type=ToolStructuredOutput(Person)),
        model=ScriptedModel([session]),
        tools=[ToolSpec("slow", "Slow", {"type": "object", "properties": {}}, lambda args: "late", background="model")],
    )

    result = harness.run_sync("go")

    assert result.output == Person(name="done")
    deferred_output = session.tool_outputs[1][0]
    assert deferred_output.call_id == "final_1"
    assert "Final answer deferred" in deferred_output.output
    assert "Produce the final answer again now." in deferred_output.output


async def test_ready_background_completion_is_coalesced_with_foreground_tool_outputs(tmp_path: Path) -> None:
    background_finished = asyncio.Event()

    async def background(_args):
        background_finished.set()
        return "background"

    async def foreground(_args):
        await asyncio.wait_for(background_finished.wait(), timeout=0.5)
        await asyncio.sleep(0.01)
        return "foreground"

    session = SequenceSession(
        ModelTurn(tool_calls=[
            _call("background", '{"_background":true}', "call_1"),
            _call("foreground", "{}", "call_2"),
        ], raw={"id": "start"}),
        ModelTurn(text="done", raw={"id": "done"}),
    )
    harness = Harness(HarnessConfig(root=tmp_path, builtin_tools=[], max_tool_calls=2), model=ScriptedModel([session]), tools=[
        ToolSpec("background", "Background", {"type": "object", "properties": {}}, background, background="model"),
        ToolSpec("foreground", "Foreground", {"type": "object", "properties": {}}, foreground),
    ])

    result = await harness.run("go")

    assert result.text == "done"
    assert result.usage.tool_calls == 2
    assert session.user_messages == []
    assert [[output.call_id for output in batch] for batch in session.tool_outputs] == [["call_1", "call_2"]]
    notices = [notice for notice in session.notice_calls[1][1] if notice.kind == "background_completion"]
    assert [(notice.kind, "Tool: background" in notice.content) for notice in notices] == [("background_completion", True)]


async def test_background_completion_during_provider_turn_waits_until_next_tool_outputs(tmp_path: Path) -> None:
    background_finished = asyncio.Event()

    async def background(_args):
        background_finished.set()
        return "background"

    class WaitingSession(SequenceSession):
        async def continue_with_tools(self, outputs, *, instructions=None, tools, metadata=None, structured_output=None, notices=None):
            if not self.tool_outputs:
                await asyncio.wait_for(background_finished.wait(), timeout=0.5)
            return await super().continue_with_tools(
                outputs,
                instructions=instructions,
                tools=tools,
                metadata=metadata,
                structured_output=structured_output,
                notices=notices,
            )

    session = WaitingSession(
        ModelTurn(tool_calls=[_call("background", '{"_background":true}', "call_1")], raw={"id": "start"}),
        ModelTurn(tool_calls=[_call("foreground", "{}", "call_2")], raw={"id": "foreground"}),
        ModelTurn(text="done", raw={"id": "done"}),
    )
    harness = Harness(HarnessConfig(root=tmp_path, builtin_tools=[]), model=ScriptedModel([session]), tools=[
        ToolSpec("background", "Background", {"type": "object", "properties": {}}, background, background="model"),
        ToolSpec("foreground", "Foreground", {"type": "object", "properties": {}}, lambda _args: "foreground"),
    ])

    result = await harness.run("go")

    assert result.text == "done"
    assert session.notice_calls[1][1] == []
    notices = session.notice_calls[2][1]
    assert [(notice.kind, "Tool: background" in notice.content) for notice in notices] == [("background_completion", True)]


async def test_final_text_drains_multiple_ready_background_completions(tmp_path: Path) -> None:
    release_background = asyncio.Event()
    completed: set[str] = set()
    all_background_finished = asyncio.Event()

    async def finish(label: str) -> str:
        await release_background.wait()
        completed.add(label)
        if len(completed) == 2:
            all_background_finished.set()
        return label

    async def one(_args):
        return await finish("one")

    async def two(_args):
        return await finish("two")

    class WaitingSession(SequenceSession):
        async def continue_with_tools(self, outputs, *, instructions=None, tools, metadata=None, structured_output=None, notices=None):
            release_background.set()
            await asyncio.wait_for(all_background_finished.wait(), timeout=0.5)
            await asyncio.sleep(0)
            return await super().continue_with_tools(
                outputs,
                instructions=instructions,
                tools=tools,
                metadata=metadata,
                structured_output=structured_output,
                notices=notices,
            )

    session = WaitingSession(
        ModelTurn(tool_calls=[
            _call("one", '{"_background":true}', "call_1"),
            _call("two", '{"_background":true}', "call_2"),
        ], raw={"id": "start"}),
        ModelTurn(text="too early", raw={"id": "early"}),
        ModelTurn(text="done", raw={"id": "done"}),
    )
    harness = Harness(HarnessConfig(root=tmp_path, builtin_tools=[]), model=ScriptedModel([session]), tools=[
        ToolSpec("one", "One", {"type": "object", "properties": {}}, one, background="model"),
        ToolSpec("two", "Two", {"type": "object", "properties": {}}, two, background="model"),
    ])

    result = await harness.run("go")

    assert result.text == "done"
    assert len(session.user_messages) == 1
    assert "Tool: one" in session.user_messages[0]
    assert "Tool: two" in session.user_messages[0]
    assert "\n\n---\n\n" in session.user_messages[0]


async def test_final_answer_defers_when_completion_is_ready_but_not_pending(tmp_path: Path) -> None:
    background_finished = asyncio.Event()

    async def background(_args):
        background_finished.set()
        return "ready"

    class WaitingSession(SequenceSession):
        async def continue_with_tools(self, outputs, *, instructions=None, tools, metadata=None, structured_output=None, notices=None):
            await asyncio.wait_for(background_finished.wait(), timeout=0.5)
            return await super().continue_with_tools(
                outputs,
                instructions=instructions,
                tools=tools,
                metadata=metadata,
                structured_output=structured_output,
                notices=notices,
            )

    session = WaitingSession(
        ModelTurn(tool_calls=[_call("background", '{"_background":true}', "call_1")], raw={"id": "start"}),
        ModelTurn(text="too early", raw={"id": "early"}),
        ModelTurn(text="done", raw={"id": "done"}),
    )
    harness = Harness(HarnessConfig(root=tmp_path, builtin_tools=[]), model=ScriptedModel([session]), tools=[
        ToolSpec("background", "Background", {"type": "object", "properties": {}}, background, background="model"),
    ])

    result = await harness.run("go")

    assert result.text == "done"
    assert len(session.user_messages) == 1
    assert "Background task bg_1 completed." in session.user_messages[0]


def test_multiple_background_completions_are_delivered_in_completion_order(tmp_path: Path) -> None:
    async def slow(_args):
        await asyncio.sleep(0.05)
        return "slow"

    async def fast(_args):
        await asyncio.sleep(0.01)
        return "fast"

    session = SequenceSession(
        ModelTurn(tool_calls=[
            _call("slow", '{"_background":true}', "call_1"),
            _call("fast", '{"_background":true}', "call_2"),
        ], raw={"id": "start"}),
        ModelTurn(text="early", raw={"id": "early"}),
        ModelTurn(text="again", raw={"id": "again"}),
        ModelTurn(text="done", raw={"id": "done"}),
    )
    harness = Harness(HarnessConfig(root=tmp_path, builtin_tools=[]), model=ScriptedModel([session]), tools=[
        ToolSpec("slow", "Slow", {"type": "object", "properties": {}}, slow, background="model"),
        ToolSpec("fast", "Fast", {"type": "object", "properties": {}}, fast, background="model"),
    ])

    result = harness.run_sync("go")

    assert result.text == "done"
    assert "Tool: fast" in session.user_messages[0]
    assert "Tool: slow" in session.user_messages[1]


def test_background_model_retry_is_delivered_without_retry_budget_accounting(tmp_path: Path) -> None:
    session = SequenceSession(
        ModelTurn(tool_calls=[_call("retry", '{"_background":true}')], raw={"id": "start"}),
        ModelTurn(text="early", raw={"id": "early"}),
        ModelTurn(text="done", raw={"id": "done"}),
    )
    harness = Harness(HarnessConfig(root=tmp_path, builtin_tools=[]), model=ScriptedModel([session]), tools=[
        ToolSpec("retry", "Retry", {"type": "object", "properties": {}}, lambda args: (_ for _ in ()).throw(ModelRetry("try again")), background="model"),
    ])

    result = harness.run_sync("go")

    assert result.usage.tool_retries == {}
    assert "ModelRetry" in session.user_messages[0]


def test_background_subagent_preserves_parent_context_and_named_policy(tmp_path: Path) -> None:
    parent_contexts = []

    def capture_context(_args):
        parent_contexts.append((current_tool_call_context(), current_tool_runtime_context()))
        return "child tool"

    child_call = ModelTurn(tool_calls=[_call("capture", "{}")], raw={"id": "child-start"})
    child = ScriptedSession(start_turn=child_call, continue_turn=ModelTurn(text="child done", raw={"id": "child-done"}))
    parent = SequenceSession(
        ModelTurn(tool_calls=[_call("subagent", '{"task":"help","_background":true}')], raw={"id": "parent-start"}),
        ModelTurn(text="early", raw={"id": "early"}),
        ModelTurn(text="parent done", raw={"id": "parent-done"}),
    )
    harness = Harness(
        HarnessConfig(root=tmp_path, builtin_tools=["subagent"]),
        model=ScriptedModel([parent, child]),
        tools=[ToolSpec("capture", "Capture", {"type": "object", "properties": {}}, capture_context)],
    )

    assert harness.run_sync("delegate", metadata={"conversation_id": "conv"}).text == "parent done"
    assert parent_contexts == [
        (
            {"call_id": "call_1", "name": "capture"},
            {"run_metadata": {"conversation_id": "conv", "parent_call_id": "call_1"}},
        )
    ]

    named_parent = SequenceSession(
        ModelTurn(tool_calls=[_call("subagent", '{"task":"x","agent":"sync","_background":true}')], raw={"id": "start"}),
        ModelTurn(text="done", raw={"id": "done"}),
    )
    named = Harness(
        HarnessConfig(
            root=tmp_path,
            builtin_tools=["subagent"],
            subagents=[
                SubAgentConfig(
                    name="sync",
                    description="Sync helper.",
                    tools=[ToolSpec("x", "X", {"type": "object", "properties": {}}, lambda args: "x")],
                )
            ],
        ),
        model=ScriptedModel([named_parent]),
    )

    result = named.run_sync("delegate")
    assert result.usage.tool_retries == {"subagent": 1}
    output = tool_output(named_parent.tool_outputs[0][0].output)
    assert output["content"] == "selected subagent does not support background execution"
    assert output["metadata"] == {"error_type": "InvalidArguments", "retry": True}


def test_named_subagent_always_background_strips_private_argument(tmp_path: Path) -> None:
    parent = SequenceSession(
        ModelTurn(tool_calls=[_call("subagent", '{"task":"x","agent":"async","_background":true}')], raw={"id": "start"}),
        ModelTurn(text="early", raw={"id": "early"}),
        ModelTurn(text="done", raw={"id": "done"}),
    )
    child = ScriptedSession(start_turn=ModelTurn(text="child done", raw={"id": "child"}))
    harness = Harness(
        HarnessConfig(
            root=tmp_path,
            builtin_tools=["subagent"],
            subagents=[
                SubAgentConfig(
                    name="async",
                    description="Async helper.",
                    background="always",
                    tools=[ToolSpec("x", "X", {"type": "object", "properties": {}}, lambda args: "x")],
                )
            ],
        ),
        model=ScriptedModel([parent, child]),
    )

    result = harness.run_sync("delegate")

    assert result.text == "done"
    assert "child done" in parent.user_messages[0]


def test_named_subagent_never_background_strips_false_and_unknown_agent_reports_unknown(tmp_path: Path) -> None:
    sync_child = ScriptedSession(start_turn=ModelTurn(text="sync child", raw={"id": "child"}))
    sync_parent = SequenceSession(
        ModelTurn(tool_calls=[_call("subagent", '{"task":"x","agent":"sync","_background":false}')], raw={"id": "start"}),
        ModelTurn(text="done", raw={"id": "done"}),
    )
    harness = Harness(
        HarnessConfig(
            root=tmp_path,
            builtin_tools=["subagent"],
            subagents=[
                SubAgentConfig(
                    name="sync",
                    description="Sync helper.",
                    tools=[ToolSpec("x", "X", {"type": "object", "properties": {}}, lambda args: "x")],
                )
            ],
        ),
        model=ScriptedModel([sync_parent, sync_child]),
    )

    assert harness.run_sync("delegate").text == "done"
    assert tool_output(sync_parent.tool_outputs[0][0].output)["metadata"]["agent"] == "sync"

    unknown_parent = SequenceSession(
        ModelTurn(tool_calls=[_call("subagent", '{"task":"x","agent":"missing","_background":true}')], raw={"id": "start"}),
        ModelTurn(text="done", raw={"id": "done"}),
    )
    unknown = Harness(
        HarnessConfig(root=tmp_path, builtin_tools=["subagent"]),
        model=ScriptedModel([unknown_parent]),
    )

    assert unknown.run_sync("delegate").text == "done"
    assert tool_output(unknown_parent.tool_outputs[0][0].output)["metadata"]["error_type"] == "UnknownSubAgent"


def test_sequential_harness_rejects_named_always_background_subagent(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="background='always' subagents"):
        HarnessConfig(
            root=tmp_path,
            builtin_tools=["subagent"],
            tool_execution="sequential",
            subagents=[
                SubAgentConfig(
                    name="async",
                    description="Async helper.",
                    background="always",
                    tools=[ToolSpec("x", "X", {"type": "object", "properties": {}}, lambda args: "x")],
                )
            ],
        )


def test_background_tool_hooks_fire_once_and_after_sees_start_notice(tmp_path: Path) -> None:
    events = []

    def before(ctx):
        assert isinstance(ctx, BeforeToolCallContext)
        events.append(("before", ctx.tool_name))

    def after(ctx):
        assert isinstance(ctx, AfterToolCallContext)
        events.append(("after", tool_output(ctx.output)["metadata"]["status"]))

    session = SequenceSession(
        ModelTurn(tool_calls=[_call("slow", '{"_background":true}')], raw={"id": "start"}),
        ModelTurn(text="early", raw={"id": "early"}),
        ModelTurn(text="done", raw={"id": "done"}),
    )
    harness = Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[]),
        model=ScriptedModel([session]),
        tools=[ToolSpec("slow", "Slow", {"type": "object", "properties": {}}, lambda args: "late", background="model")],
        hooks=[Hook("before_tool_call", before), Hook("after_tool_call", after)],
    )

    assert harness.run_sync("go").text == "done"
    assert events == [("before", "slow"), ("after", "running")]


def test_background_tracing_has_start_and_execution_spans(tmp_path: Path) -> None:
    tracer = FakeTracer()
    session = SequenceSession(
        ModelTurn(tool_calls=[_call("slow", '{"_background":true}')], raw={"id": "start"}),
        ModelTurn(text="early", raw={"id": "early"}),
        ModelTurn(text="done", raw={"id": "done"}),
    )
    harness = Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[]),
        model=ScriptedModel([session]),
        tools=[ToolSpec("slow", "Slow", {"type": "object", "properties": {}}, lambda args: "late", background="model")],
        tracing=[TracingOptions(tracer=tracer, capture_messages=True, capture_tool_results=True)],
    )

    harness.run_sync("go")

    tool_spans = [span for span in tracer.spans if span.name == "execute_tool slow"]
    bg_chat = next(span for span in tracer.spans if span.attributes.get("thinharness.model.request.kind") == "background_completion")
    assert len(tool_spans) == 2
    assert tool_spans[1].attributes["thinharness.background.task_id"] == "bg_1"
    assert tool_spans[1].attributes["thinharness.background.phase"] == "execution"
    assert json.loads(bg_chat.attributes["gen_ai.input.messages"])[0]["parts"][0]["content"].startswith("Background task bg_1 completed.")


def test_background_limit_exhaustion_drains_pending_tasks(tmp_path: Path) -> None:
    session = SequenceSession(
        ModelTurn(tool_calls=[_call("wait", '{"_background":true}')], raw={"id": "start"}),
        ModelTurn(text="early", raw={"id": "early"}),
    )
    harness = Harness(HarnessConfig(root=tmp_path, builtin_tools=[], max_model_requests=2), model=ScriptedModel([session]), tools=[
        ToolSpec("wait", "Wait", {"type": "object", "properties": {}}, lambda args: (time.sleep(0.05), "done")[1], background="model"),
    ])

    with pytest.raises(HarnessError, match="max_model_requests=2"):
        harness.run_sync("go")


def test_max_tool_calls_does_not_block_pending_background_completion(tmp_path: Path) -> None:
    session = SequenceSession(
        ModelTurn(tool_calls=[_call("slow", '{"_background":true}', "call_1")], raw={"id": "start"}),
        ModelTurn(tool_calls=[_call("extra", "{}", "call_2")], raw={"id": "extra"}),
        ModelTurn(text="done", raw={"id": "done"}),
    )
    harness = Harness(HarnessConfig(root=tmp_path, builtin_tools=[], max_tool_calls=1), model=ScriptedModel([session]), tools=[
        ToolSpec("slow", "Slow", {"type": "object", "properties": {}}, lambda args: "late", background="model"),
        ToolSpec("extra", "Extra", {"type": "object", "properties": {}}, lambda args: "should not run"),
    ])

    result = harness.run_sync("go")

    assert result.text == "done"
    assert result.usage.tool_calls == 1
    deferred = tool_output(session.tool_outputs[1][0].output)
    assert deferred["metadata"]["error_type"] == "ToolCallsExceeded"
    notices = session.notice_calls[2][1]
    assert deferred["content"] == "Tool call was not executed because max_tool_calls=1 is exhausted."
    assert [(notice.kind, "Background task bg_1 completed." in notice.content) for notice in notices] == [("background_completion", True)]


def test_background_completion_notice_tracing_is_coalesced(tmp_path: Path) -> None:
    tracer = FakeTracer()
    background_finished = asyncio.Event()

    async def background(_args):
        background_finished.set()
        return "background"

    async def foreground(_args):
        await asyncio.wait_for(background_finished.wait(), timeout=0.5)
        await asyncio.sleep(0.01)
        return "foreground"

    session = SequenceSession(
        ModelTurn(tool_calls=[
            _call("background", '{"_background":true}', "call_1"),
            _call("foreground", "{}", "call_2"),
        ], raw={"id": "start"}),
        ModelTurn(text="done", raw={"id": "done"}),
    )
    harness = Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[]),
        model=ScriptedModel([session]),
        tools=[
            ToolSpec("background", "Background", {"type": "object", "properties": {}}, background, background="model"),
            ToolSpec("foreground", "Foreground", {"type": "object", "properties": {}}, foreground),
        ],
        tracing=[TracingOptions(tracer=tracer, capture_messages=True, capture_tool_results=True)],
    )

    harness.run_sync("go")

    coalesced_chat = next(
        span
        for span in tracer.spans
        if span.name == "chat scripted-model" and span.attributes.get("thinharness.model.request.kind") == "tool_outputs"
    )
    input_messages = json.loads(coalesced_chat.attributes["gen_ai.input.messages"])
    notices = json.loads(coalesced_chat.attributes["thinharness.model.notices"])
    contents = [part["content"] for message in input_messages for part in message["parts"]]
    assert contents == [session.tool_outputs[0][0].output, session.tool_outputs[0][1].output]
    assert notices[0]["kind"] == "background_completion"
    assert "Tool: background" in notices[0]["content"]


async def test_background_task_starts_before_slow_sibling_finishes(tmp_path: Path) -> None:
    background_started = asyncio.Event()

    async def background(_args):
        background_started.set()
        return "background"

    async def foreground(_args):
        await asyncio.wait_for(background_started.wait(), timeout=0.5)
        return "foreground"

    session = SequenceSession(
        ModelTurn(tool_calls=[
            _call("background", '{"_background":true}', "call_1"),
            _call("foreground", "{}", "call_2"),
        ], raw={"id": "start"}),
        ModelTurn(text="early", raw={"id": "early"}),
        ModelTurn(text="done", raw={"id": "done"}),
    )
    harness = Harness(HarnessConfig(root=tmp_path, builtin_tools=[]), model=ScriptedModel([session]), tools=[
        ToolSpec("background", "Background", {"type": "object", "properties": {}}, background, background="model"),
        ToolSpec("foreground", "Foreground", {"type": "object", "properties": {}}, foreground),
    ])

    result = await harness.run("go")

    assert result.text == "early"
