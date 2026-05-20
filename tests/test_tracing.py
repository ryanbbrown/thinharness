from __future__ import annotations

import json
from pathlib import Path

import pytest
from fakes import (
    ContextFakeTracer,
    FailingSession,
    FakeClient,
    FakeSpan,
    FakeTracer,
    MultiCallClient,
    ScriptedModel,
    ScriptedSession,
    _fake_openai,
    echo_tool,
    tool_output,
)
from pydantic import BaseModel

from thinharness import (
    Harness,
    HarnessConfig,
    HarnessError,
    SubAgentConfig,
    ToolResult,
    ToolSpec,
    TracingOptions,
    build_child_harness,
    create_subagent_tool,
)
from thinharness.providers import ModelNotice, ModelToolCall, ModelTurn
from thinharness.tracing import ModelTraceSnapshot, _SpanAdapter, annotate_model_request, create_local_tracing_options, serialize_attribute_value


class Person(BaseModel):
    """Test structured-output type."""

    name: str
    age: int


def test_harness_tracing_records_agent_model_and_tool_spans(tmp_path: Path) -> None:
    (tmp_path / "hello.txt").write_text("hello", encoding="utf-8")
    tracer = FakeTracer()
    harness = Harness(
        HarnessConfig(root=tmp_path, model="openai:test-model"),
        model=_fake_openai(FakeClient()),
        tracing=[TracingOptions(
            tracer=tracer,
            agent_name="test-agent",
            capture_messages=True,
            capture_tool_args=True,
            capture_tool_results=True,
        )],
    )

    result = harness.run_sync("read hello", metadata={"conversation_id": "conv-1"})

    assert result.text == "done"
    assert [span.name for span in tracer.spans] == [
        "invoke_agent test-agent",
        "chat test-model",
        "execute_tool read",
        "chat test-model",
    ]
    root, first_chat, tool, second_chat = tracer.spans
    assert first_chat.parent is root
    assert tool.parent is root
    assert second_chat.parent is root
    assert root.attributes["gen_ai.operation.name"] == "invoke_agent"
    assert root.attributes["gen_ai.conversation.id"] == "conv-1"
    assert root.attributes["gen_ai.completion"] == "done"
    assert root.attributes["langfuse.trace.input"] == "read hello"
    assert "gen_ai.system_instructions" in root.attributes
    assert first_chat.attributes["gen_ai.provider.name"] == "OpenAI"
    assert first_chat.attributes["gen_ai.request.model"] == "test-model"
    assert json.loads(first_chat.attributes["gen_ai.input.messages"])[0]["parts"][0]["content"] == "read hello"
    assert json.loads(first_chat.attributes["langfuse.observation.input"]) == {"prompt": "read hello"}
    assert "gen_ai.system_instructions" not in first_chat.attributes
    first_output = json.loads(first_chat.attributes["gen_ai.output.messages"])
    assert first_output[0]["parts"][0]["type"] == "tool_call"
    assert first_output[0]["parts"][0]["name"] == "read"
    assert "gen_ai.completion" not in first_chat.attributes
    assert tool.attributes["gen_ai.tool.name"] == "read"
    assert tool.attributes["gen_ai.tool.call.id"] == "call_1"
    assert tool.attributes["gen_ai.tool.call.arguments"] == '{"path":"hello.txt"}'
    assert "hello" in tool.attributes["gen_ai.tool.call.result"]
    assert second_chat.attributes["gen_ai.completion"] == "done"
    assert json.loads(second_chat.attributes["gen_ai.output.messages"])[0]["parts"][0]["content"] == "done"
    assert json.loads(second_chat.attributes["langfuse.observation.output"]) == {"text": "done"}

def test_serialize_attribute_value_keeps_none_absent() -> None:
    assert serialize_attribute_value(None) is None

def test_model_request_snapshot_keeps_tool_output_separate_from_notices() -> None:
    span = FakeSpan("chat", {})
    snapshot = ModelTraceSnapshot(
        kind="tool_outputs",
        tool_outputs=[{"call_id": "call_1", "output": '{"ok":true,"content":"real output"}'}],
    ).with_notices([ModelNotice(kind="limit_warning", content="notice text", limit_kind="model_requests", remaining=1)])

    annotate_model_request(_SpanAdapter(span), snapshot, capture_messages=True)

    input_messages = json.loads(span.attributes["gen_ai.input.messages"])
    notices = json.loads(span.attributes["thinharness.model.notices"])
    assert input_messages[0]["parts"][0]["content"] == '{"ok":true,"content":"real output"}'
    assert "notice text" not in span.attributes["gen_ai.input.messages"]
    assert notices[0]["content"] == "notice text"

def test_local_tracing_writes_full_jsonl_trace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("THINHARNESS_DISABLE_LOCAL_TRACING", raising=False)
    (tmp_path / "hello.txt").write_text("hello", encoding="utf-8")
    trace_dir = tmp_path / "traces"
    harness = Harness(
        HarnessConfig(
            root=tmp_path,
            model="openai:test-model",
            local_tracing=True,
            local_trace_dir=trace_dir,
        ),
        model=_fake_openai(FakeClient()),
    )

    result = harness.run_sync("read hello")

    trace_files = list(trace_dir.rglob("*.jsonl"))
    assert result.text == "done"
    assert len(trace_files) == 1
    assert trace_files[0].parent != trace_dir
    records = [json.loads(line) for line in trace_files[0].read_text(encoding="utf-8").splitlines()]
    assert {record["name"] for record in records} == {
        "invoke_agent thinharness",
        "chat test-model",
        "execute_tool read",
    }
    root = next(record for record in records if record["name"] == "invoke_agent thinharness")
    first_chat = next(record for record in records if record["name"] == "chat test-model" and "gen_ai.output.messages" in record["attributes"])
    tool = next(record for record in records if record["name"] == "execute_tool read")
    assert root["attributes"]["langfuse.trace.input"] == "read hello"
    assert json.loads(first_chat["attributes"]["gen_ai.input.messages"])[0]["parts"][0]["content"] == "read hello"
    assert json.loads(first_chat["attributes"]["gen_ai.output.messages"])[0]["parts"][0]["type"] == "tool_call"
    assert tool["attributes"]["gen_ai.tool.call.arguments"] == '{"path":"hello.txt"}'
    assert "hello" in tool["attributes"]["gen_ai.tool.call.result"]

def test_local_tracing_nests_subagent_spans(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("THINHARNESS_DISABLE_LOCAL_TRACING", raising=False)
    trace_dir = tmp_path / "traces"
    parent_call = ModelTurn(
        tool_calls=[ModelToolCall(id="call_1", name="subagent", arguments='{"task":"help"}')],
        raw={"id": "parent-start"},
    )
    child = ScriptedSession(start_turn=ModelTurn(text="child done", raw={"id": "child"}))
    parent = ScriptedSession(start_turn=parent_call, continue_turn=ModelTurn(text="parent done", raw={"id": "parent-done"}))
    harness = Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[], local_tracing=True, local_trace_dir=trace_dir),
        model=ScriptedModel([parent, child]),
        tools=[echo_tool()],
    )
    harness.add_tool(create_subagent_tool(harness, []))

    harness.run_sync("delegate")

    records = [
        json.loads(line)
        for path in trace_dir.rglob("*.jsonl")
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    subagent_tool = next(record for record in records if record["name"] == "execute_tool subagent")
    child_agent = next(record for record in records if record["name"] == "invoke_agent subagent.default")
    assert child_agent["parent_id"] == subagent_tool["span_id"]
    assert child_agent["attributes"]["langfuse.observation.input"] == "help"
    assert "child done" in child_agent["attributes"]["langfuse.observation.output"]

def test_local_tracing_does_not_change_remote_capture_policy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("THINHARNESS_DISABLE_LOCAL_TRACING", raising=False)
    (tmp_path / "hello.txt").write_text("hello", encoding="utf-8")
    remote = FakeTracer()
    trace_dir = tmp_path / "traces"
    harness = Harness(
        HarnessConfig(root=tmp_path, model="openai:test-model", local_trace_dir=trace_dir),
        model=_fake_openai(FakeClient()),
        tracing=[TracingOptions(tracer=remote, capture_messages=False, capture_tool_args=False, capture_tool_results=False)],
    )

    harness.run_sync("read hello")

    forbidden = {
        "langfuse.observation.input",
        "langfuse.observation.output",
        "langfuse.trace.input",
        "langfuse.trace.output",
        "gen_ai.input.messages",
        "gen_ai.output.messages",
        "gen_ai.system_instructions",
        "gen_ai.prompt",
        "gen_ai.completion",
        "gen_ai.tool.call.arguments",
        "gen_ai.tool.call.result",
    }
    for span in remote.spans:
        assert forbidden.isdisjoint(span.attributes)
    local_text = "\n".join(path.read_text(encoding="utf-8") for path in trace_dir.rglob("*.jsonl"))
    assert "read hello" in local_text
    assert "hello" in local_text

def test_create_local_tracing_options_is_full_capture_and_project_scoped(tmp_path: Path) -> None:
    options = create_local_tracing_options(tmp_path / "traces", project_root=tmp_path)

    assert options.capture_messages is True
    assert options.capture_tool_args is True
    assert options.capture_tool_results is True
    assert options.tracer.trace_dir.parent == tmp_path / "traces"
    assert options.tracer.trace_dir != tmp_path / "traces"

def test_capture_messages_false_omits_content_attributes(tmp_path: Path) -> None:
    tracer = FakeTracer()
    harness = Harness(
        HarnessConfig(root=tmp_path, model="openai:test-model"),
        model=_fake_openai(FakeClient()),
        tracing=[TracingOptions(tracer=tracer, capture_messages=False, capture_tool_args=True, capture_tool_results=True)],
    )

    harness.run_sync("read hello")

    forbidden = {
        "langfuse.observation.input",
        "langfuse.observation.output",
        "langfuse.trace.input",
        "langfuse.trace.output",
        "gen_ai.input.messages",
        "gen_ai.output.messages",
        "gen_ai.system_instructions",
        "gen_ai.prompt",
        "gen_ai.completion",
    }
    for span in tracer.spans:
        assert forbidden.isdisjoint(span.attributes)

def test_tool_tracing_marks_normalized_failures(tmp_path: Path) -> None:
    failing = ToolSpec("fail", "Returns failure.", {"type": "object", "properties": {}}, lambda args: ToolResult(False, "nope"))
    client = MultiCallClient([("fail", "{}")])
    tracer = FakeTracer()
    harness = Harness(
        HarnessConfig(root=tmp_path, model="openai:test-model", builtin_tools=[]),
        model=_fake_openai(client),
        tools=[failing],
        tracing=[TracingOptions(tracer=tracer, capture_messages=True)],
    )

    harness.run_sync("go")

    tool = next(span for span in tracer.spans if span.name == "execute_tool fail")
    assert tool.status is not None
    assert tool.attributes["error.type"] == "ToolExecutionError"

def test_subagent_tracing_nests_child_under_parent_tool_span(tmp_path: Path) -> None:
    parent_call = ModelTurn(
        tool_calls=[ModelToolCall(id="call_1", name="subagent", arguments='{"task":"help"}')],
        raw={"id": "parent-start"},
    )
    child = ScriptedSession(start_turn=ModelTurn(text="child done", raw={"id": "child"}))
    parent = ScriptedSession(start_turn=parent_call, continue_turn=ModelTurn(text="parent done", raw={"id": "parent-done"}))
    tracer = FakeTracer()
    harness = Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[]),
        model=ScriptedModel([parent, child]),
        tools=[echo_tool()],
        tracing=[TracingOptions(tracer=tracer, capture_messages=True)],
    )
    harness.add_tool(create_subagent_tool(harness, []))

    harness.run_sync("delegate")

    assert [span.name for span in tracer.spans] == [
        "invoke_agent thinharness",
        "chat scripted-model",
        "execute_tool subagent",
        "invoke_agent subagent.default",
        "chat scripted-model",
        "chat scripted-model",
    ]
    root, first_chat, subagent_tool, child_agent, child_chat, final_chat = tracer.spans
    assert first_chat.parent is root
    assert subagent_tool.parent is root
    assert child_agent.parent is subagent_tool
    assert child_chat.parent is child_agent
    assert final_chat.parent is root
    assert child_agent.attributes["gen_ai.agent.name"] == "subagent.default"
    assert child_agent.attributes["langfuse.observation.input"] == "help"
    assert child_agent.attributes["langfuse.observation.output"]
    assert "langfuse.trace.input" not in child_agent.attributes
    assert "langfuse.trace.output" not in child_agent.attributes
    assert subagent_tool.attributes["subagent.name"] == "default"
    assert subagent_tool.attributes["subagent.tool_mode"] == "inherited"
    assert subagent_tool.attributes["subagent.tools"] == ["echo"]

def test_subagent_runs_with_tracing_disabled(tmp_path: Path) -> None:
    parent_call = ModelTurn(
        tool_calls=[ModelToolCall(id="call_1", name="subagent", arguments='{"task":"help"}')],
        raw={"id": "parent-start"},
    )
    child = ScriptedSession(start_turn=ModelTurn(text="child done", raw={"id": "child"}))
    parent = ScriptedSession(start_turn=parent_call, continue_turn=ModelTurn(text="parent done", raw={"id": "parent-done"}))
    harness = Harness(HarnessConfig(root=tmp_path, builtin_tools=[]), model=ScriptedModel([parent, child]), tools=[echo_tool()])
    harness.add_tool(create_subagent_tool(harness, []))

    assert build_child_harness(harness, None).tracing == []
    assert harness.run_sync("delegate").text == "parent done"

def test_concurrent_subagent_fanout_keeps_each_child_under_own_tool_span(tmp_path: Path) -> None:
    parent_call = ModelTurn(
        tool_calls=[
            ModelToolCall(id="call_1", name="subagent", arguments='{"task":"first"}'),
            ModelToolCall(id="call_2", name="subagent", arguments='{"task":"second"}'),
        ],
        raw={"id": "parent-start"},
    )
    child_a = ScriptedSession(start_turn=ModelTurn(text="child a", raw={"id": "child-a"}))
    child_b = ScriptedSession(start_turn=ModelTurn(text="child b", raw={"id": "child-b"}))
    parent = ScriptedSession(start_turn=parent_call, continue_turn=ModelTurn(text="parent done", raw={"id": "parent-done"}))
    tracer = ContextFakeTracer()
    harness = Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[]),
        model=ScriptedModel([parent, child_a, child_b]),
        tools=[echo_tool()],
        tracing=[TracingOptions(tracer=tracer, capture_messages=True)],
    )
    harness.add_tool(create_subagent_tool(harness, []))

    assert harness.run_sync("delegate").text == "parent done"

    root = next(span for span in tracer.spans if span.name == "invoke_agent thinharness")
    subagent_tools = [span for span in tracer.spans if span.name == "execute_tool subagent"]
    child_agents = [span for span in tracer.spans if span.name == "invoke_agent subagent.default"]
    child_model_spans = [span for span in tracer.spans if span.name == "chat scripted-model" and span.parent in child_agents]
    assert len(subagent_tools) == 2
    assert len(child_agents) == 2
    assert len(child_model_spans) == 2
    assert all(span.parent is root for span in subagent_tools)
    assert {id(span.parent) for span in child_agents} == {id(span) for span in subagent_tools}
    assert {id(span.parent) for span in child_model_spans} == {id(span) for span in child_agents}
    assert {span.attributes["langfuse.observation.input"] for span in child_agents} == {"first", "second"}
    assert all(span.attributes["langfuse.observation.input"] != "delegate" for span in child_agents)

def test_trace_request_kinds_for_resume_and_output_retries(tmp_path: Path) -> None:
    retry_session = ScriptedSession(
        start_turn=ModelTurn(tool_calls=[ModelToolCall(id="call_final", name="final_result", arguments='{"name":"Ada"}')], raw={"id": "bad"}),
        continue_turn=ModelTurn(tool_calls=[ModelToolCall(id="call_final_2", name="final_result", arguments='{"name":"Ada","age":37}')], raw={"id": "good"}),
    )
    correction_session = ScriptedSession(
        start_turn=ModelTurn(text="not json", raw={"id": "bad-text"}),
        continue_turn=ModelTurn(tool_calls=[ModelToolCall(id="call_final", name="final_result", arguments='{"name":"Ada","age":37}')], raw={"id": "good-tool"}),
    )
    first_session = ScriptedSession(
        start_turn=ModelTurn(text="ready", raw={"id": "first"}),
        dump_state={"kind": "scripted", "version": 1, "model": "scripted-model"},
    )
    resumed_session = ScriptedSession(start_turn=ModelTurn(text="done", raw={"id": "second"}))
    model = ScriptedModel([retry_session, correction_session, first_session, resumed_session])
    tracer = FakeTracer()

    Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[], output_type=Person, output_mode="tool", max_model_requests=2),
        model=model,
        tracing=[TracingOptions(tracer=tracer, capture_messages=True)],
    ).run_sync("make a person")
    Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[], output_type=Person, output_mode="tool"),
        model=model,
        tracing=[TracingOptions(tracer=tracer, capture_messages=True)],
    ).run_sync("make another")
    first = Harness(HarnessConfig(root=tmp_path, builtin_tools=[]), model=model).run_sync("first")
    Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[]),
        model=model,
        tracing=[TracingOptions(tracer=tracer, capture_messages=True)],
    ).run_sync("follow-up", resume_from=first.resume_state)

    chats = [span for span in tracer.spans if span.name == "chat scripted-model"]
    kinds = [span.attributes.get("thinharness.model.request.kind") for span in chats]
    assert "output_retry_tool" in kinds
    assert "correction" in kinds
    assert "resume" in kinds
    retry_chat = next(span for span in chats if span.attributes.get("thinharness.model.request.kind") == "output_retry_tool")
    assert json.loads(retry_chat.attributes["gen_ai.input.messages"])[0]["parts"][0]["content"].startswith("The previous response failed")
    assert "Final request" not in retry_chat.attributes["gen_ai.input.messages"]
    assert "Final request" in retry_chat.attributes["thinharness.model.notices"]
    correction_chat = next(span for span in chats if span.attributes.get("thinharness.model.request.kind") == "correction")
    assert json.loads(correction_chat.attributes["langfuse.observation.input"])["correction"].startswith("The previous response failed")
    resume_chat = next(span for span in chats if span.attributes.get("thinharness.model.request.kind") == "resume")
    assert json.loads(resume_chat.attributes["langfuse.observation.input"]) == {"prompt": "follow-up"}

def test_provider_error_keeps_trace_input_without_output(tmp_path: Path) -> None:
    tracer = FakeTracer()
    harness = Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[]),
        model=ScriptedModel([FailingSession()]),
        tracing=[TracingOptions(tracer=tracer, capture_messages=True)],
    )

    with pytest.raises(HarnessError, match="child failed"):
        harness.run_sync("fail please")

    root = tracer.spans[0]
    assert root.attributes["langfuse.trace.input"] == "fail please"
    assert "gen_ai.system_instructions" in root.attributes
    assert "langfuse.trace.output" not in root.attributes
    assert root.status is not None

def test_unknown_named_subagent_trace_marks_failed_without_child_tool_mode(tmp_path: Path) -> None:
    parent_call = ModelTurn(
        tool_calls=[ModelToolCall(id="call_1", name="subagent", arguments='{"task":"help","agent":"missing"}')],
        raw={"id": "parent-start"},
    )

    def on_parent_continue(outputs, _tools, _metadata):
        envelope = tool_output(outputs[0].output)
        assert envelope["ok"] is False
        assert envelope["metadata"]["agent"] == "missing"
        assert envelope["metadata"]["error_type"] == "UnknownSubAgent"
        assert "tool_mode" not in envelope["metadata"]
        assert "tools" not in envelope["metadata"]

    tracer = FakeTracer()
    parent = ScriptedSession(start_turn=parent_call, continue_turn=ModelTurn(text="parent done", raw={"id": "parent-done"}), on_continue=on_parent_continue)
    harness = Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[]),
        model=ScriptedModel([parent]),
        tracing=[TracingOptions(tracer=tracer)],
    )
    harness.add_tool(create_subagent_tool(harness, [SubAgentConfig(name="research", description="Research helper.", builtin_tools=["read"])]))

    assert harness.run_sync("delegate").text == "parent done"
    subagent_tool = next(span for span in tracer.spans if span.name == "execute_tool subagent")
    assert subagent_tool.attributes["subagent.name"] == "missing"
    assert "subagent.tool_mode" not in subagent_tool.attributes
    assert "subagent.tools" not in subagent_tool.attributes
    assert subagent_tool.status is not None
