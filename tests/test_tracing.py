from __future__ import annotations

from pathlib import Path

from fakes import (
    ContextFakeTracer,
    FakeClient,
    FakeTracer,
    MultiCallClient,
    ScriptedModel,
    ScriptedSession,
    _fake_openai,
    echo_tool,
    tool_output,
)

from thinharness import (
    Harness,
    HarnessConfig,
    SubAgentConfig,
    ToolResult,
    ToolSpec,
    TracingOptions,
    build_child_harness,
    create_subagent_tool,
)
from thinharness.providers import ModelToolCall, ModelTurn


def test_harness_tracing_records_agent_model_and_tool_spans(tmp_path: Path) -> None:
    (tmp_path / "hello.txt").write_text("hello", encoding="utf-8")
    tracer = FakeTracer()
    harness = Harness(
        HarnessConfig(root=tmp_path, model="openai:test-model"),
        model=_fake_openai(FakeClient()),
        tracing=TracingOptions(
            tracer=tracer,
            agent_name="test-agent",
            capture_messages=True,
            capture_tool_args=True,
            capture_tool_results=True,
        ),
    )

    result = harness.run("read hello", metadata={"conversation_id": "conv-1"})

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
    assert first_chat.attributes["gen_ai.provider.name"] == "OpenAI"
    assert first_chat.attributes["gen_ai.request.model"] == "test-model"
    assert tool.attributes["gen_ai.tool.name"] == "read"
    assert tool.attributes["gen_ai.tool.call.id"] == "call_1"
    assert tool.attributes["gen_ai.tool.call.arguments"] == '{"path":"hello.txt"}'
    assert "hello" in tool.attributes["gen_ai.tool.call.result"]
    assert second_chat.attributes["gen_ai.completion"] == "done"

def test_tool_tracing_marks_normalized_failures(tmp_path: Path) -> None:
    failing = ToolSpec("fail", "Returns failure.", {"type": "object", "properties": {}}, lambda args: ToolResult(False, "nope"))
    client = MultiCallClient([("fail", "{}")])
    tracer = FakeTracer()
    harness = Harness(
        HarnessConfig(root=tmp_path, model="openai:test-model", builtin_tools=[]),
        model=_fake_openai(client),
        tools=[failing],
        tracing=TracingOptions(tracer=tracer),
    )

    harness.run("go")

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
        tracing=TracingOptions(tracer=tracer),
    )
    harness.add_tool(create_subagent_tool(harness, []))

    harness.run("delegate")

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

    assert build_child_harness(harness, None).tracing is None
    assert harness.run("delegate").text == "parent done"

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
        tracing=TracingOptions(tracer=tracer),
    )
    harness.add_tool(create_subagent_tool(harness, []))

    assert harness.run("delegate").text == "parent done"

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
        tracing=TracingOptions(tracer=tracer),
    )
    harness.add_tool(create_subagent_tool(harness, [SubAgentConfig(name="research", description="Research helper.", builtin_tools=["read"])]))

    assert harness.run("delegate").text == "parent done"
    subagent_tool = next(span for span in tracer.spans if span.name == "execute_tool subagent")
    assert subagent_tool.attributes["subagent.name"] == "missing"
    assert "subagent.tool_mode" not in subagent_tool.attributes
    assert "subagent.tools" not in subagent_tool.attributes
    assert subagent_tool.status is not None
