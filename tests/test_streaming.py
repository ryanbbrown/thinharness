from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from fakes import FailingSession, ScriptedModel, ScriptedSession, echo_tool

from thinharness import (
    BackgroundTaskCompletedEvent,
    BackgroundTaskStartedEvent,
    Harness,
    HarnessConfig,
    HarnessError,
    Hook,
    ModelMessageEvent,
    ModelRequestStartedEvent,
    ModelRetry,
    ModelRetryEvent,
    RunCompletedEvent,
    RunFailedEvent,
    RunStartedEvent,
    StreamOptions,
    SubAgentConfig,
    ToolCallCompletedEvent,
    ToolCallStartedEvent,
    ToolSpec,
)
from thinharness.providers import ModelToolCall, ModelTurn
from thinharness.tools.base import ToolResult


class SequenceSession:
    """Script one start turn and a sequence of continuations."""

    def __init__(self, start_turn: ModelTurn, *continue_turns: ModelTurn) -> None:
        self.start_turn = start_turn
        self.continue_turns = list(continue_turns)
        self.tool_outputs = []
        self.user_messages = []

    async def start(self, *, prompt, instructions, tools, metadata=None, previous_response_id=None, structured_output=None, notices=None):
        """Return the scripted first turn."""
        return self.start_turn

    async def continue_with_tools(self, outputs, *, instructions=None, tools, metadata=None, structured_output=None, notices=None):
        """Record tool outputs and return the next scripted turn."""
        self.tool_outputs.append(outputs)
        return self._next_turn()

    async def continue_with_user_message(self, message, *, instructions=None, tools, metadata=None, structured_output=None, notices=None):
        """Record user-message continuations and return the next scripted turn."""
        self.user_messages.append(message)
        return self._next_turn()

    async def continue_with_user_prompt(self, *, prompt, instructions, tools, metadata=None, structured_output=None, notices=None):
        """Return the scripted first turn for resumed prompts."""
        return self.start_turn

    def dump_state(self):
        """Return scripted resume state."""
        return {"kind": "scripted", "version": 1, "model": "scripted-model"}

    def _next_turn(self) -> ModelTurn:
        if not self.continue_turns:
            raise AssertionError("unexpected continuation")
        return self.continue_turns.pop(0)


async def _collect_events(harness: Harness, prompt: str, **kwargs):
    events = []
    async for event in harness.stream(prompt, **kwargs):
        events.append(event)
    return events


async def test_stream_returns_final_harness_result(tmp_path: Path) -> None:
    session = ScriptedSession(start_turn=ModelTurn(text="done", raw={"id": "done"}))
    harness = Harness(HarnessConfig(root=tmp_path, builtin_tools=[]), model=ScriptedModel([session]))

    events = await _collect_events(harness, "go")

    completed = [event for event in events if isinstance(event, RunCompletedEvent)]
    assert len(completed) == 1
    assert completed[0].result.text == "done"
    assert completed[0].result.usage.model_requests == 1
    assert completed[0].result.responses == [{"id": "done"}]


async def test_run_consumes_stream_and_returns_result(tmp_path: Path) -> None:
    run_session = ScriptedSession(start_turn=ModelTurn(text="done", raw={"id": "done"}))
    stream_session = ScriptedSession(start_turn=ModelTurn(text="done", raw={"id": "done"}))
    run_harness = Harness(HarnessConfig(root=tmp_path / "run", builtin_tools=[]), model=ScriptedModel([run_session]))
    stream_harness = Harness(HarnessConfig(root=tmp_path / "stream", builtin_tools=[]), model=ScriptedModel([stream_session]))

    run_result = await run_harness.run("go")
    events = await _collect_events(stream_harness, "go")
    stream_result = next(event.result for event in events if isinstance(event, RunCompletedEvent))

    assert run_result.text == stream_result.text
    assert run_result.usage.model_requests == stream_result.usage.model_requests
    assert run_result.stop_reason == stream_result.stop_reason


async def test_stream_emits_model_and_tool_lifecycle(tmp_path: Path) -> None:
    session = ScriptedSession(
        start_turn=ModelTurn(
            tool_calls=[ModelToolCall(id="call_1", name="echo", arguments='{"value":"ok"}')],
            raw={"id": "start"},
        ),
        continue_turn=ModelTurn(text="done", raw={"id": "done"}),
    )
    harness = Harness(HarnessConfig(root=tmp_path, builtin_tools=[]), model=ScriptedModel([session]), tools=[echo_tool()])

    events = await _collect_events(harness, "go")

    assert [event.kind for event in events] == [
        "run_started",
        "model_request_started",
        "model_message",
        "tool_call_started",
        "tool_call_completed",
        "model_request_started",
        "model_message",
        "run_completed",
    ]
    started = next(event for event in events if isinstance(event, ToolCallStartedEvent))
    completed = next(event for event in events if isinstance(event, ToolCallCompletedEvent))
    assert started.arguments == '{"value":"ok"}'
    assert completed.output is not None
    assert "ok" in completed.output
    assert completed.ok is True


async def test_stream_payloads_are_high_level_without_raw_provider_payloads(tmp_path: Path) -> None:
    session = ScriptedSession(
        start_turn=ModelTurn(
            text="thinking",
            tool_calls=[ModelToolCall(id="call_1", name="echo", arguments='{"value":"ok"}')],
            raw={"id": "start", "raw": True},
        ),
        continue_turn=ModelTurn(text="done", raw={"id": "done"}),
    )
    harness = Harness(HarnessConfig(root=tmp_path, builtin_tools=[]), model=ScriptedModel([session]), tools=[echo_tool()])

    events = await _collect_events(harness, "go")

    assert isinstance(events[0], RunStartedEvent)
    assert events[0].prompt == "go"
    message = next(event for event in events if isinstance(event, ModelMessageEvent) and event.text == "thinking")
    assert message.text == "thinking"
    assert not hasattr(message, "raw")
    started = next(event for event in events if isinstance(event, ToolCallStartedEvent))
    completed = next(event for event in events if isinstance(event, ToolCallCompletedEvent))
    assert started.arguments == '{"value":"ok"}'
    assert completed.output is not None
    assert "ok" in completed.output


async def test_stream_limit_warning_events(tmp_path: Path) -> None:
    session = ScriptedSession(start_turn=ModelTurn(text="done", raw={"id": "done"}))
    harness = Harness(HarnessConfig(root=tmp_path, builtin_tools=[], max_model_requests=1), model=ScriptedModel([session]))

    events = await _collect_events(harness, "go")

    warnings = [event for event in events if event.kind == "limit_warning"]
    assert [(event.limit_kind, event.remaining) for event in warnings] == [("model_requests", 1)]
    assert warnings[0].content == "Final request: produce the answer now; do not request tools."


async def test_stream_failed_tool_content_is_included_by_default(tmp_path: Path) -> None:
    error_message = "tool failed"
    session = ScriptedSession(
        start_turn=ModelTurn(tool_calls=[ModelToolCall(id="call_1", name="fail", arguments="{}")], raw={"id": "start"}),
        continue_turn=ModelTurn(text="done", raw={"id": "done"}),
    )
    harness = Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[]),
        model=ScriptedModel([session]),
        tools=[ToolSpec("fail", "Fail", {"type": "object", "properties": {}}, lambda args: ToolResult(False, error_message).as_json())],
    )

    events = await _collect_events(harness, "go")
    completed = next(event for event in events if isinstance(event, ToolCallCompletedEvent))

    assert completed.output is not None
    assert error_message in completed.output


async def test_stream_failure_yields_failed_event_then_raises(tmp_path: Path) -> None:
    harness = Harness(HarnessConfig(root=tmp_path, builtin_tools=[]), model=ScriptedModel([FailingSession()]))
    events = []
    stream = harness.stream("go")

    with pytest.raises(HarnessError, match="child failed"):
        while True:
            events.append(await stream.__anext__())

    failed = [event for event in events if isinstance(event, RunFailedEvent)]
    assert len(failed) == 1
    assert failed[0].stop_reason == "provider_error"
    assert failed[0].error_type == "HarnessError"


async def test_stream_subagent_events_include_parent_ids(tmp_path: Path) -> None:
    child = ScriptedSession(start_turn=ModelTurn(text="child done", raw={"id": "child"}))
    parent = ScriptedSession(
        start_turn=ModelTurn(tool_calls=[ModelToolCall(id="call_1", name="subagent", arguments='{"task":"help","agent":"helper"}')], raw={"id": "parent"}),
        continue_turn=ModelTurn(text="parent done", raw={"id": "parent-done"}),
    )
    harness = Harness(
        HarnessConfig(
            root=tmp_path,
            builtin_tools=["subagent"],
            subagents=[SubAgentConfig(name="helper", description="Helper.", tools=[ToolSpec("x", "X", {"type": "object"}, lambda args: "x")])],
        ),
        model=ScriptedModel([parent, child]),
    )

    events = await _collect_events(harness, "delegate")
    parent_run_id = events[0].run_id
    child_events = [event for event in events if event.run_id != parent_run_id]

    assert child_events
    assert {event.parent_run_id for event in child_events} == {parent_run_id}
    assert {event.parent_tool_call_id for event in child_events} == {"call_1"}
    assert {event.agent_name for event in child_events} == {"helper"}
    assert all(event.sequence == index for index, event in enumerate(events, start=1))


async def test_stream_options_can_hide_subagent_events(tmp_path: Path) -> None:
    child = ScriptedSession(start_turn=ModelTurn(text="child done", raw={"id": "child"}))
    parent = ScriptedSession(
        start_turn=ModelTurn(tool_calls=[ModelToolCall(id="call_1", name="subagent", arguments='{"task":"help"}')], raw={"id": "parent"}),
        continue_turn=ModelTurn(text="parent done", raw={"id": "parent-done"}),
    )
    harness = Harness(HarnessConfig(root=tmp_path, builtin_tools=["subagent"]), model=ScriptedModel([parent, child]))

    events = await _collect_events(harness, "delegate", stream_options=StreamOptions(include_subagents=False))

    assert {event.run_id for event in events} == {events[0].run_id}
    assert next(event.result for event in events if isinstance(event, RunCompletedEvent)).text == "parent done"


async def test_stream_close_after_early_break_cleans_up(tmp_path: Path) -> None:
    release = asyncio.Event()

    async def slow_start(*_args, **_kwargs):
        await release.wait()
        return ModelTurn(text="done", raw={"id": "done"})

    session = ScriptedSession(start_turn=ModelTurn(text="unused", raw={"id": "unused"}))
    session.start = slow_start
    harness = Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[]),
        model=ScriptedModel([session, ScriptedSession(start_turn=ModelTurn(text="again", raw={}))]),
    )

    async with harness.stream("go") as events:
        first = await events.__anext__()
        assert first.kind == "run_started"

    release.set()
    assert (await harness.run("again")).text == "again"


async def test_stream_background_events(tmp_path: Path) -> None:
    session = SequenceSession(
        ModelTurn(tool_calls=[ModelToolCall(id="call_1", name="slow", arguments='{"_background":true}')], raw={"id": "start"}),
        ModelTurn(text="early", raw={"id": "early"}),
        ModelTurn(text="done", raw={"id": "done"}),
    )
    harness = Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[]),
        model=ScriptedModel([session]),
        tools=[ToolSpec("slow", "Slow", {"type": "object", "properties": {}}, lambda args: "late", background="model")],
    )

    events = await _collect_events(harness, "go")

    tool_done = next(event for event in events if isinstance(event, ToolCallCompletedEvent))
    assert tool_done.background_task_id == "bg_1"
    assert tool_done.background_status == "running"
    background_done = next(event for event in events if isinstance(event, BackgroundTaskCompletedEvent))
    assert background_done.status == "completed"
    assert background_done.output is not None
    assert "late" in background_done.output
    assert any(isinstance(event, BackgroundTaskStartedEvent) and event.background_task_id == "bg_1" for event in events)
    assert any(isinstance(event, ModelRequestStartedEvent) and event.request_kind == "background_completion" for event in events)


async def test_stream_structured_output_retry_event(tmp_path: Path) -> None:
    from pydantic import BaseModel

    class Person(BaseModel):
        name: str
        age: int

    session = ScriptedSession(
        start_turn=ModelTurn(tool_calls=[ModelToolCall(id="call_final", name="final_result", arguments='{"name":"Ada"}')], raw={"id": "bad"}),
        continue_turn=ModelTurn(
            tool_calls=[ModelToolCall(id="call_final_2", name="final_result", arguments='{"name":"Ada","age":37}')],
            raw={"id": "good"},
        ),
    )
    harness = Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[], output_type=Person, output_mode="tool", output_retries=1),
        model=ScriptedModel([session]),
    )

    events = await _collect_events(harness, "go")
    retry_index = next(index for index, event in enumerate(events) if isinstance(event, ModelRetryEvent))
    next_model_index = next(
        index
        for index, event in enumerate(events[retry_index + 1 :], start=retry_index + 1)
        if isinstance(event, ModelRequestStartedEvent)
    )

    assert events[retry_index].retry_kind == "structured_output"
    assert events[retry_index].call_id == "call_final"
    assert events[next_model_index].request_kind == "output_retry_tool"


async def test_stream_tool_retry_event(tmp_path: Path) -> None:
    session = SequenceSession(
        ModelTurn(tool_calls=[ModelToolCall(id="call_1", name="flaky", arguments="{}")], raw={"id": "start"}),
        ModelTurn(tool_calls=[ModelToolCall(id="call_2", name="flaky", arguments="{}")], raw={"id": "retry"}),
        ModelTurn(text="done", raw={"id": "done"}),
    )
    calls = 0

    def flaky(_args):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ModelRetry("try again")
        return "ok"

    harness = Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[]),
        model=ScriptedModel([session]),
        tools=[ToolSpec("flaky", "Flaky", {"type": "object", "properties": {}}, flaky)],
    )

    events = await _collect_events(harness, "go")
    retry = next(event for event in events if isinstance(event, ModelRetryEvent))

    assert retry.retry_kind == "tool_retry"
    assert retry.call_id == "call_1"


async def test_stream_resume_from_emits_resume_kind(tmp_path: Path) -> None:
    session = SequenceSession(ModelTurn(text="done", raw={"id": "done"}))
    harness = Harness(HarnessConfig(root=tmp_path, builtin_tools=[]), model=ScriptedModel([session]))

    events = await _collect_events(harness, "go", resume_from={"kind": "scripted", "version": 1, "model": "scripted-model"})

    request = next(event for event in events if isinstance(event, ModelRequestStartedEvent))
    assert request.request_kind == "resume"


def test_stream_without_running_loop_does_not_brick_harness(tmp_path: Path) -> None:
    harness = Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[]),
        model=ScriptedModel([ScriptedSession(start_turn=ModelTurn(text="done", raw={"id": "done"}))]),
    )

    with pytest.raises(RuntimeError):
        harness.stream("go")

    assert harness.run_sync("go").text == "done"


def test_stream_run_sync_still_works(tmp_path: Path) -> None:
    harness = Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[]),
        model=ScriptedModel([ScriptedSession(start_turn=ModelTurn(text="done", raw={"id": "done"}))]),
    )

    assert harness.run_sync("go").text == "done"


async def test_stream_strict_after_tool_hook_failure_completes_started_tool(tmp_path: Path) -> None:
    session = ScriptedSession(
        start_turn=ModelTurn(tool_calls=[ModelToolCall(id="call_1", name="echo", arguments='{"value":"ok"}')], raw={"id": "start"}),
        continue_turn=ModelTurn(text="done", raw={"id": "done"}),
    )
    harness = Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[], strict_hooks=True),
        model=ScriptedModel([session]),
        tools=[echo_tool()],
        hooks=[Hook("after_tool_call", lambda ctx: (_ for _ in ()).throw(RuntimeError("after failed")))],
    )
    events = []
    stream = harness.stream("go")

    with pytest.raises(RuntimeError, match="after failed"):
        while True:
            events.append(await stream.__anext__())

    assert [event.kind for event in events] == [
        "run_started",
        "model_request_started",
        "model_message",
        "tool_call_started",
        "tool_call_completed",
        "run_failed",
    ]
    completed = next(event for event in events if isinstance(event, ToolCallCompletedEvent))
    assert completed.ok is False
    assert completed.error_type == "RuntimeError"
    assert completed.message == "after failed"
