from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from fakes import FakeTracer, MultiCallClient, ScriptedModel, _fake_openai, echo_tool, tool_output
from pydantic import BaseModel, model_validator

from thinharness import (
    AfterToolCallContext,
    Harness,
    HarnessConfig,
    HarnessError,
    Hook,
    ModelRetry,
    SubAgentConfig,
    ToolSpec,
    TracingOptions,
    build_child_harness,
    builtin_tools,
    call_tool,
)
from thinharness.providers import ModelToolCall, ModelTurn


class SequenceSession:
    """Script a start turn followed by each tool continuation turn."""

    def __init__(self, start_turn: ModelTurn, *continue_turns: ModelTurn) -> None:
        self.start_turn = start_turn
        self.continue_turns = list(continue_turns)
        self.tool_outputs = []

    async def start(self, prompt, constants, *, previous_response_id=None, notices=None):
        """Return the scripted first turn."""
        return self.start_turn

    async def continue_with_tools(self, outputs, constants, *, notices=None):
        """Record tool outputs and return the next scripted turn."""
        self.tool_outputs.append(outputs)
        if not self.continue_turns:
            raise AssertionError("unexpected tool continuation")
        return self.continue_turns.pop(0)

    async def continue_with_user_text(self, text, constants, *, notices=None):
        """No tests in this file expect user-text continuations."""
        raise AssertionError("unexpected user-text continuation")

    def dump_state(self):
        """Return no resume state for retry-specific scripted sessions."""
        return None


def _call(name: str, args: str, call_id: str = "call_1") -> ModelToolCall:
    """Build a normalized model tool call."""
    return ModelToolCall(id=call_id, name=name, arguments=args)


def test_model_retry_public_import_and_successful_retry(tmp_path: Path) -> None:
    calls = 0

    def flaky(_args):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ModelRetry("try again with X")
        return "ok"

    session = SequenceSession(
        ModelTurn(tool_calls=[_call("flaky", "{}")], raw={"id": "start"}),
        ModelTurn(tool_calls=[_call("flaky", "{}", "call_2")], raw={"id": "retry"}),
        ModelTurn(text="done", raw={"id": "done"}),
    )
    harness = Harness(HarnessConfig(root=tmp_path, builtin_tools=[]), model=ScriptedModel([session]), tools=[
        ToolSpec("flaky", "Flaky", {"type": "object", "properties": {}}, flaky),
    ])

    result = harness.run_sync("go")

    retry = tool_output(session.tool_outputs[0][0].output)
    assert result.text == "done"
    assert retry["metadata"] == {"error_type": "ModelRetry", "retry": True}
    assert retry["content"] == "try again with X"
    assert result.usage.tool_retries == {"flaky": 1}


def test_validation_failure_retries_then_succeeds(tmp_path: Path) -> None:
    class AgeArgs(BaseModel):
        age: int

    seen = []
    session = SequenceSession(
        ModelTurn(tool_calls=[_call("age", '{"age":"five"}')], raw={"id": "start"}),
        ModelTurn(tool_calls=[_call("age", '{"age":5}', "call_2")], raw={"id": "retry"}),
        ModelTurn(text="done", raw={"id": "done"}),
    )
    harness = Harness(HarnessConfig(root=tmp_path, builtin_tools=[]), model=ScriptedModel([session]), tools=[
        ToolSpec("age", "Age", AgeArgs, lambda args: seen.append(args.age) or "ok"),
    ])

    result = harness.run_sync("go")

    retry = tool_output(session.tool_outputs[0][0].output)
    assert result.text == "done"
    assert retry["metadata"]["error_type"] == "ValidationError"
    assert retry["metadata"]["retry"] is True
    assert "age" in retry["content"]
    assert seen == [5]


def test_handler_internal_validation_error_is_not_retry(tmp_path: Path) -> None:
    class OuterArgs(BaseModel):
        value: str

    class InnerArgs(BaseModel):
        value: int

    def handler(args):
        InnerArgs.model_validate({"value": args.value})
        return "never"

    client = MultiCallClient([("inner", '{"value":"bad"}')])
    harness = Harness(HarnessConfig(root=tmp_path, model="openai:test-model", builtin_tools=[]), model=_fake_openai(client), tools=[
        ToolSpec("inner", "Inner", OuterArgs, handler),
    ])

    result = harness.run_sync("go")
    envelope = tool_output(client.payloads[1]["input"][0]["output"])

    assert envelope["metadata"]["error_type"] == "ValidationError"
    assert envelope["metadata"].get("retry") is None
    assert result.usage.tool_retries == {}


def test_builtin_validation_failure_is_retryable(tmp_path: Path) -> None:
    read = next(tool for tool in builtin_tools(tmp_path) if tool.name == "read")

    output = tool_output(call_tool(read, '{"path":"missing.txt","limit":0}'))

    assert output["ok"] is False
    assert output["metadata"]["error_type"] == "ValidationError"
    assert output["metadata"]["retry"] is True


def test_malformed_json_retries_then_succeeds(tmp_path: Path) -> None:
    session = SequenceSession(
        ModelTurn(tool_calls=[_call("echo", "{not json")], raw={"id": "start"}),
        ModelTurn(tool_calls=[_call("echo", '{"value":"ok"}', "call_2")], raw={"id": "retry"}),
        ModelTurn(text="done", raw={"id": "done"}),
    )
    harness = Harness(HarnessConfig(root=tmp_path, builtin_tools=[]), model=ScriptedModel([session]), tools=[echo_tool()])

    result = harness.run_sync("go")
    retry = tool_output(session.tool_outputs[0][0].output)

    assert result.text == "done"
    assert retry["metadata"]["error_type"] == "InvalidArguments"
    assert retry["metadata"]["retry"] is True
    assert result.usage.tool_retries == {"echo": 1}


def test_nested_and_root_validation_error_formatting() -> None:
    class FilterArgs(BaseModel):
        name: str

    class SearchArgs(BaseModel):
        filters: list[FilterArgs]

    class RootArgs(BaseModel):
        age: int
        name: str

        @model_validator(mode="after")
        def reject(self):
            """Reject every model instance at root level."""
            raise ValueError("age must be greater than name length")

    nested = tool_output(call_tool(ToolSpec("search", "Search", SearchArgs, lambda args: "ok"), '{"filters":[{}]}'))
    root = tool_output(call_tool(ToolSpec("root", "Root", RootArgs, lambda args: "ok"), '{"age":1,"name":"Ada"}'))

    assert "filters.0.name" in nested["content"]
    assert "got dict" not in nested["content"]
    assert "<root>" in root["content"]


def test_tool_retries_exceeded_counts_over_budget_failure(tmp_path: Path) -> None:
    events = []
    session = SequenceSession(
        ModelTurn(tool_calls=[_call("flaky", "{}")], raw={"id": "start"}),
        ModelTurn(tool_calls=[_call("flaky", "{}", "call_2")], raw={"id": "retry"}),
    )
    harness = Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[], tool_retries=1),
        model=ScriptedModel([session]),
        tools=[ToolSpec("flaky", "Flaky", {"type": "object", "properties": {}}, lambda args: (_ for _ in ()).throw(ModelRetry("again")))],
        hooks=[
            Hook("limit_reached", lambda ctx: events.append((ctx.limit_kind, ctx.limit_value, ctx.current_count))),
            Hook("run_end", lambda ctx: events.append((ctx.stop_reason, dict(ctx.usage.tool_retries), ctx.result))),
        ],
    )

    with pytest.raises(HarnessError, match="exceeded max_retries=1"):
        harness.run_sync("go")

    assert len(session.tool_outputs) == 1
    assert events == [
        ("tool_retries", 1, 2),
        ("tool_retries_exceeded", {"flaky": 2}, None),
    ]


def test_tool_max_retries_zero_blocks_first_retry_continuation(tmp_path: Path) -> None:
    session = SequenceSession(ModelTurn(tool_calls=[_call("flaky", "{}")], raw={"id": "start"}))
    harness = Harness(HarnessConfig(root=tmp_path, builtin_tools=[], tool_retries=3), model=ScriptedModel([session]), tools=[
        ToolSpec("flaky", "Flaky", {"type": "object", "properties": {}}, lambda args: (_ for _ in ()).throw(ModelRetry("no")), max_retries=0),
    ])

    with pytest.raises(HarnessError, match="max_retries=0"):
        harness.run_sync("go")

    assert session.tool_outputs == []


def test_tool_max_retries_override_wins_over_config_default(tmp_path: Path) -> None:
    session = SequenceSession(
        ModelTurn(tool_calls=[_call("flaky", "{}")], raw={"id": "start"}),
        ModelTurn(tool_calls=[_call("flaky", "{}", "call_2")], raw={"id": "retry"}),
    )
    events = []
    harness = Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[], tool_retries=3),
        model=ScriptedModel([session]),
        tools=[
            ToolSpec(
                "flaky",
                "Flaky",
                {"type": "object", "properties": {}},
                lambda args: (_ for _ in ()).throw(ModelRetry("again")),
                max_retries=1,
            )
        ],
        hooks=[Hook("limit_reached", lambda ctx: events.append((ctx.limit_value, ctx.current_count)))],
    )

    with pytest.raises(HarnessError, match="max_retries=1"):
        harness.run_sync("go")

    assert len(session.tool_outputs) == 1
    assert events == [(1, 2)]


def test_two_calls_same_tool_share_budget_and_skip_batch_continuation(tmp_path: Path) -> None:
    session = SequenceSession(ModelTurn(tool_calls=[
        _call("flaky", "{}", "call_1"),
        _call("flaky", "{}", "call_2"),
    ], raw={"id": "start"}))
    events = []
    harness = Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[], tool_retries=1),
        model=ScriptedModel([session]),
        tools=[ToolSpec("flaky", "Flaky", {"type": "object", "properties": {}}, lambda args: (_ for _ in ()).throw(ModelRetry("again")))],
        hooks=[Hook("run_end", lambda ctx: events.append(dict(ctx.usage.tool_retries)))],
    )

    with pytest.raises(HarnessError):
        harness.run_sync("go")

    assert session.tool_outputs == []
    assert events == [{"flaky": 2}]


def test_unknown_tool_and_cancellation_do_not_consume_retry_budget(tmp_path: Path) -> None:
    client = MultiCallClient([("missing", "{}"), ("block", "{}")])

    def cancel(ctx):
        if ctx.tool_name == "block":
            ctx.cancelled = True
            ctx.cancel_reason = "blocked"

    harness = Harness(
        HarnessConfig(root=tmp_path, model="openai:test-model", builtin_tools=[]),
        model=_fake_openai(client),
        tools=[ToolSpec("block", "Block", {"type": "object", "properties": {}}, lambda args: "bad")],
        hooks=[Hook("before_tool_call", cancel)],
    )

    result = harness.run_sync("go")

    assert result.usage.tool_retries == {}
    assert result.usage.cancelled_tool_calls == 1


def test_parallel_retry_and_success_outputs_preserve_model_order(tmp_path: Path) -> None:
    client = MultiCallClient([("retry", "{}"), ("ok", "{}")])
    harness = Harness(
        HarnessConfig(root=tmp_path, model="openai:test-model", builtin_tools=[]),
        model=_fake_openai(client),
        tools=[
            ToolSpec("retry", "Retry", {"type": "object", "properties": {}}, lambda args: (_ for _ in ()).throw(ModelRetry("try again"))),
            ToolSpec("ok", "Ok", {"type": "object", "properties": {}}, lambda args: "done"),
        ],
    )

    result = harness.run_sync("go")
    outputs = client.payloads[1]["input"]

    assert result.usage.tool_retries == {"retry": 1}
    assert [item["call_id"] for item in outputs] == ["call_1", "call_2"]
    assert tool_output(outputs[0]["output"])["metadata"]["error_type"] == "ModelRetry"
    assert tool_output(outputs[1]["output"])["content"] == "done"


async def test_async_handler_model_retry_is_captured(tmp_path: Path) -> None:
    async def retry(_args):
        await asyncio.sleep(0)
        raise ModelRetry("async retry")

    client = MultiCallClient([("async_retry", "{}")])
    harness = Harness(HarnessConfig(root=tmp_path, model="openai:test-model", builtin_tools=[]), model=_fake_openai(client), tools=[
        ToolSpec("async_retry", "Async retry", {"type": "object", "properties": {}}, retry),
    ])

    result = await harness.run("go")
    envelope = tool_output(client.payloads[1]["input"][0]["output"])

    assert result.usage.tool_retries == {"async_retry": 1}
    assert envelope["metadata"]["error_type"] == "ModelRetry"


async def test_async_handler_internal_validation_error_is_not_retry(tmp_path: Path) -> None:
    class InnerArgs(BaseModel):
        value: int

    async def handler(args):
        await asyncio.sleep(0)
        InnerArgs.model_validate({"value": args["value"]})
        return "never"

    client = MultiCallClient([("inner", '{"value":"bad"}')])
    harness = Harness(HarnessConfig(root=tmp_path, model="openai:test-model", builtin_tools=[]), model=_fake_openai(client), tools=[
        ToolSpec("inner", "Inner", {"type": "object", "properties": {}}, handler),
    ])

    result = await harness.run("go")
    envelope = tool_output(client.payloads[1]["input"][0]["output"])

    assert envelope["metadata"]["error_type"] == "ValidationError"
    assert envelope["metadata"].get("retry") is None
    assert result.usage.tool_retries == {}


def test_negative_retry_limits_are_rejected() -> None:
    with pytest.raises(ValueError):
        HarnessConfig(tool_retries=-1)
    with pytest.raises(ValueError, match="max_retries"):
        ToolSpec("bad", "Bad", {"type": "object", "properties": {}}, lambda args: "ok", max_retries=-1)
    with pytest.raises(ValueError):
        SubAgentConfig(name="bad", description="Bad helper.", tools=[echo_tool()], tool_retries=-1)


def test_after_tool_hook_sees_retry_envelope_and_cannot_break_budget(tmp_path: Path) -> None:
    seen = []

    def after(ctx):
        assert isinstance(ctx, AfterToolCallContext)
        seen.append(ctx.envelope.metadata["error_type"])
        ctx.output = "not json"

    session = SequenceSession(ModelTurn(tool_calls=[_call("flaky", "{}")], raw={"id": "start"}))
    harness = Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[], tool_retries=0),
        model=ScriptedModel([session]),
        tools=[ToolSpec("flaky", "Flaky", {"type": "object", "properties": {}}, lambda args: (_ for _ in ()).throw(ModelRetry("again")))],
        hooks=[Hook("after_tool_call", after)],
    )

    with pytest.raises(HarnessError):
        harness.run_sync("go")

    assert seen == ["ModelRetry"]
    assert session.tool_outputs == []


def test_after_tool_hook_sees_validation_retry_envelope(tmp_path: Path) -> None:
    class AgeArgs(BaseModel):
        age: int

    seen = []
    client = MultiCallClient([("age", '{"age":"five"}')])
    harness = Harness(
        HarnessConfig(root=tmp_path, model="openai:test-model", builtin_tools=[]),
        model=_fake_openai(client),
        tools=[ToolSpec("age", "Age", AgeArgs, lambda args: "ok")],
        hooks=[Hook("after_tool_call", lambda ctx: seen.append(ctx.envelope.metadata))],
    )

    harness.run_sync("go")

    assert seen[0]["error_type"] == "ValidationError"
    assert seen[0]["retry"] is True


def test_tracing_uses_pre_hook_retry_kind(tmp_path: Path) -> None:
    tracer = FakeTracer()

    def rewrite(ctx):
        ctx.envelope.metadata["error_type"] = "Rewritten"
        ctx.output = ctx.envelope.to_json()

    client = MultiCallClient([("flaky", "{}")])
    harness = Harness(
        HarnessConfig(root=tmp_path, model="openai:test-model", builtin_tools=[]),
        model=_fake_openai(client),
        tools=[ToolSpec("flaky", "Flaky", {"type": "object", "properties": {}}, lambda args: (_ for _ in ()).throw(ModelRetry("again")))],
        hooks=[Hook("after_tool_call", rewrite)],
        tracing=[TracingOptions(tracer=tracer)],
    )

    harness.run_sync("go")

    span = next(span for span in tracer.spans if span.name == "execute_tool flaky")
    assert span.attributes["error.type"] == "ModelRetry"


def test_subagent_tool_retry_budget_inheritance(tmp_path: Path) -> None:
    parent = Harness(HarnessConfig(root=tmp_path, builtin_tools=[], tool_retries=4), model=ScriptedModel([]), tools=[echo_tool()])

    default_child = build_child_harness(parent, None)
    named_default = build_child_harness(parent, SubAgentConfig(name="named", description="Named helper.", tools=[echo_tool()]))
    named_custom = build_child_harness(parent, SubAgentConfig(name="custom", description="Custom helper.", tools=[echo_tool()], tool_retries=2))

    assert default_child.config.tool_retries == 4
    assert named_default.config.tool_retries == 1
    assert named_custom.config.tool_retries == 2
