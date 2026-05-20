from __future__ import annotations

import json
from dataclasses import dataclass

import pytest
from fakes import FakeAnthropicProvider, FakeTracer, ScriptedModel, ScriptedProvider, ScriptedSession
from pydantic import BaseModel
from typing_extensions import TypedDict

from thinharness import (
    AnthropicMessagesModel,
    Harness,
    HarnessConfig,
    HarnessError,
    Hook,
    ModelCapabilities,
    ModelToolCall,
    ModelTurn,
    NativeOutput,
    OpenRouterModel,
    RunUsage,
    SubAgentConfig,
    TextOutput,
    ToolSpec,
    TracingOptions,
    UnexpectedModelBehavior,
    build_child_harness,
    create_subagent_tool,
)
from thinharness.core import _compute_limit_notices
from thinharness.providers import ProviderError
from thinharness.subagents import SubAgentArgs, run_subagent_tool


class Person(BaseModel):
    name: str
    age: int


class Address(BaseModel):
    city: str
    zip_code: str


class PersonWithAddress(BaseModel):
    name: str
    address: Address


class Item(TypedDict):
    name: str
    count: int


@dataclass
class City:
    name: str
    country: str


def test_base_model_output_via_tool_mode(tmp_path) -> None:
    session = ScriptedSession(
        start_turn=ModelTurn(
            text="Done",
            tool_calls=[ModelToolCall(id="call_final", name="final_result", arguments='{"name":"Ada","age":37}')],
            raw={"id": "resp_1"},
        )
    )
    harness = Harness(HarnessConfig(root=tmp_path, builtin_tools=[], output_type=Person, output_mode="tool"), model=ScriptedModel([session]))

    result = harness.run_sync("make a person")

    assert result.text == "Done"
    assert result.output == Person(name="Ada", age=37)
    assert result.usage.tool_calls == 0
    assert result.tool_call_records == []


def test_base_model_output_via_native_mode(tmp_path) -> None:
    captured = {}
    session = ScriptedSession(
        start_turn=ModelTurn(text='{"name":"Ada","age":37}', raw={"id": "resp_1"}),
        on_start=lambda _prompt, _instructions, _tools, _metadata, _previous: captured.setdefault("tools", _tools),
    )
    model = ScriptedModel([session])
    model.capabilities = ModelCapabilities(supports_json_schema_output=True, default_structured_output_mode="native")
    harness = Harness(HarnessConfig(root=tmp_path, builtin_tools=[], output_type=Person), model=model)

    result = harness.run_sync("make a person")

    assert result.output == Person(name="Ada", age=37)
    assert captured["tools"] == []


def test_prompted_mode_strips_json_fence(tmp_path) -> None:
    session = ScriptedSession(start_turn=ModelTurn(text='```json\n{"name":"Ada","age":37}\n```', raw={"id": "resp_1"}))
    harness = Harness(HarnessConfig(root=tmp_path, builtin_tools=[], output_type=Person, output_mode="prompted"), model=ScriptedModel([session]))

    result = harness.run_sync("make a person")

    assert result.output == Person(name="Ada", age=37)


def test_typed_dict_dataclass_and_list_outputs(tmp_path) -> None:
    typed = Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[], output_type=Item, output_mode="tool"),
        model=ScriptedModel([ScriptedSession(start_turn=ModelTurn(tool_calls=[
            ModelToolCall(id="call_final", name="final_result", arguments='{"name":"bolt","count":2}')
        ], raw={"id": "typed"}))]),
    )
    data_class = Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[], output_type=City, output_mode="prompted"),
        model=ScriptedModel([ScriptedSession(start_turn=ModelTurn(text='{"name":"Paris","country":"FR"}', raw={"id": "dataclass"}))]),
    )
    people = Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[], output_type=list[Person], output_mode="tool"),
        model=ScriptedModel([ScriptedSession(start_turn=ModelTurn(tool_calls=[
            ModelToolCall(id="call_final", name="final_result", arguments='{"value":[{"name":"Ada","age":37}]}')
        ], raw={"id": "list"}))]),
    )

    assert typed.run_sync("item").output == {"name": "bolt", "count": 2}
    assert data_class.run_sync("city").output == City(name="Paris", country="FR")
    assert people.run_sync("people").output == [Person(name="Ada", age=37)]


def test_union_output_uses_wrapped_value_schema(tmp_path) -> None:
    harness = Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[], output_type=Person | City, output_mode="tool"),
        model=ScriptedModel([ScriptedSession(start_turn=ModelTurn(tool_calls=[
            ModelToolCall(id="call_final", name="final_result", arguments='{"value":{"name":"Paris","country":"FR"}}')
        ], raw={"id": "union"}))]),
    )

    result = harness.run_sync("union")

    assert result.output == City(name="Paris", country="FR")
    schema = harness.output_schema.synthetic_tools()[0]["parameters"]
    assert "$defs" not in json.dumps(schema)


def test_validation_failure_retries_and_succeeds(tmp_path) -> None:
    seen_outputs = []
    session = ScriptedSession(
        start_turn=ModelTurn(tool_calls=[ModelToolCall(id="call_final", name="final_result", arguments='{"name":"Ada"}')], raw={"id": "bad"}),
        continue_turn=ModelTurn(tool_calls=[
            ModelToolCall(id="call_final_2", name="final_result", arguments='{"name":"Ada","age":37}')
        ], raw={"id": "good"}),
        on_continue=lambda outputs, _tools, _metadata: seen_outputs.extend(outputs),
    )
    harness = Harness(HarnessConfig(root=tmp_path, builtin_tools=[], output_type=Person, output_mode="tool", output_retries=1), model=ScriptedModel([session]))

    result = harness.run_sync("make a person")

    assert result.output == Person(name="Ada", age=37)
    assert result.usage.model_requests == 2
    assert result.usage.output_retries == 1
    assert "failed structured output validation" in seen_outputs[0].output


def test_tool_mode_invalid_args_retry_uses_tool_output(tmp_path) -> None:
    seen_outputs = []
    seen_messages = []

    class RecordingSession(ScriptedSession):
        async def continue_with_user_message(self, message, *, instructions=None, tools, metadata=None, structured_output=None, notices=None):
            seen_messages.append(message)
            return await super().continue_with_user_message(
                message,
                instructions=instructions,
                tools=tools,
                metadata=metadata,
                structured_output=structured_output,
                notices=notices,
            )

    session = RecordingSession(
        start_turn=ModelTurn(tool_calls=[ModelToolCall(id="call_final", name="final_result", arguments='{"name":"Ada"}')], raw={"id": "bad"}),
        continue_turn=ModelTurn(tool_calls=[
            ModelToolCall(id="call_final_2", name="final_result", arguments='{"name":"Ada","age":37}')
        ], raw={"id": "good"}),
        on_continue=lambda outputs, _tools, _metadata: seen_outputs.extend(outputs),
    )
    harness = Harness(HarnessConfig(root=tmp_path, builtin_tools=[], output_type=Person, output_mode="tool"), model=ScriptedModel([session]))

    result = harness.run_sync("make a person")

    assert result.output == Person(name="Ada", age=37)
    assert seen_messages == []
    assert seen_outputs[0].call_id == "call_final"
    assert "failed structured output validation" in seen_outputs[0].output


def test_tool_mode_text_only_end_turn_retries_and_succeeds(tmp_path) -> None:
    messages = []
    session = ScriptedSession(
        start_turn=ModelTurn(text="not structured", raw={"id": "bad"}),
        continue_turn=ModelTurn(tool_calls=[
            ModelToolCall(id="call_final", name="final_result", arguments='{"name":"Ada","age":37}')
        ], raw={"id": "good"}),
        on_continue=lambda message, _tools, _metadata: messages.append(message),
    )
    harness = Harness(HarnessConfig(root=tmp_path, builtin_tools=[], output_type=Person, output_mode="tool"), model=ScriptedModel([session]))

    result = harness.run_sync("make a person")

    assert result.output == Person(name="Ada", age=37)
    assert result.usage.model_requests == 2
    assert result.usage.output_retries == 1
    assert "Call final_result" in messages[0]


def test_validation_failure_exhausts_retries_and_reports_run_end(tmp_path) -> None:
    events = []
    session = ScriptedSession(
        start_turn=ModelTurn(tool_calls=[ModelToolCall(id="call_final", name="final_result", arguments='{"name":"Ada"}')], raw={"id": "bad"})
    )
    harness = Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[], output_type=Person, output_mode="tool", output_retries=0),
        model=ScriptedModel([session]),
        hooks=[Hook("run_end", lambda ctx: events.append((ctx.stop_reason, ctx.usage.output_retries)))],
    )

    with pytest.raises(HarnessError, match="output validation exceeded"):
        harness.run_sync("make a person")

    assert events == [("output_validation_failed", 0)]


def test_retry_not_counted_when_model_limit_blocks_corrective_request(tmp_path) -> None:
    events = []
    session = ScriptedSession(
        start_turn=ModelTurn(tool_calls=[ModelToolCall(id="call_final", name="final_result", arguments='{"name":"Ada"}')], raw={"id": "bad"})
    )
    harness = Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[], output_type=Person, output_mode="tool", output_retries=1, max_model_requests=1),
        model=ScriptedModel([session]),
        hooks=[Hook("run_end", lambda ctx: events.append((ctx.stop_reason, ctx.usage.output_retries)))],
    )

    with pytest.raises(HarnessError, match="max_model_requests"):
        harness.run_sync("make a person")

    assert events == [("limit_reached", 0)]

def test_limit_notice_dedupes_across_structured_output_retries(tmp_path) -> None:
    session = ScriptedSession(
        start_turn=ModelTurn(text="not structured", raw={"id": "bad"}),
        continue_turn=ModelTurn(text="still not structured", raw={"id": "bad-again"}),
    )
    harness = Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[], output_type=Person, output_mode="tool", output_retries=2, max_tool_calls=0),
        model=ScriptedModel([session]),
    )

    with pytest.raises(HarnessError, match="output validation exceeded"):
        harness.run_sync("make a person")

    assert [(method, [(notice.limit_kind, notice.remaining) for notice in notices]) for method, notices in session.notice_calls] == [
        ("start", [("tool_calls", 0)]),
        ("continue_with_user_message", []),
        ("continue_with_user_message", []),
    ]
    assert session.notice_calls[0][1][0].content == "Tool calls are not available on this run; produce the answer with final_result."

def test_invalid_final_result_correction_receives_near_limit_notice(tmp_path) -> None:
    session = ScriptedSession(
        start_turn=ModelTurn(tool_calls=[ModelToolCall(id="call_final", name="final_result", arguments='{"name":"Ada"}')], raw={"id": "bad"}),
        continue_turn=ModelTurn(tool_calls=[
            ModelToolCall(id="call_final_2", name="final_result", arguments='{"name":"Ada","age":37}')
        ], raw={"id": "good"}),
    )
    harness = Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[], output_type=Person, output_mode="tool", max_model_requests=2),
        model=ScriptedModel([session]),
    )

    result = harness.run_sync("make a person")

    assert result.output == Person(name="Ada", age=37)

    assert [(method, [(notice.limit_kind, notice.remaining) for notice in notices]) for method, notices in session.notice_calls] == [
        ("start", []),
        ("continue_with_tools", [("model_requests", 1)]),
    ]
    assert result.usage.output_retries == 1
    assert result.usage.tool_calls == 0
    assert session.notice_calls[1][1][0].content == "Final request: produce the answer now with final_result."
    assert "final_result" in session.notice_calls[1][1][0].content

def test_limit_notices_ignore_output_retry_usage() -> None:
    low_retry_usage = RunUsage(model_requests=1, tool_calls=0, output_retries=0)
    high_retry_usage = RunUsage(model_requests=1, tool_calls=0, output_retries=99)
    config = HarnessConfig(max_model_requests=2, max_tool_calls=1)

    low_retry_notices = _compute_limit_notices(config, low_retry_usage, set(), final_result_tool_available=True)
    high_retry_notices = _compute_limit_notices(config, high_retry_usage, set(), final_result_tool_available=True)

    assert low_retry_notices == high_retry_notices

@pytest.mark.parametrize(
    ("output_mode", "output_type", "turn_text", "mentions_final_result"),
    [
        ("tool", Person, "", True),
        ("native", Person, '{"name":"Ada","age":37}', False),
        ("prompted", Person, '{"name":"Ada","age":37}', False),
        ("auto", TextOutput(), "plain", False),
        ("auto", None, "plain", False),
    ],
)
def test_limit_notices_only_mention_final_result_for_tool_output_mode(tmp_path, output_mode, output_type, turn_text, mentions_final_result) -> None:
    if output_mode == "tool":
        turn = ModelTurn(tool_calls=[ModelToolCall(id="call_final", name="final_result", arguments='{"name":"Ada","age":37}')], raw={"id": "done"})
    else:
        turn = ModelTurn(text=turn_text, raw={"id": "done"})
    session = ScriptedSession(start_turn=turn)
    model = ScriptedModel([session])
    model.capabilities = ModelCapabilities(supports_json_schema_output=True, default_structured_output_mode="native")
    harness = Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[], output_type=output_type, output_mode=output_mode, max_model_requests=1, max_tool_calls=0),
        model=model,
    )

    harness.run_sync("go")

    content = "\n".join(notice.content for notice in session.notice_calls[0][1])
    assert ("final_result" in content) is mentions_final_result


def test_final_result_mixed_with_tool_calls_is_unexpected_and_dispatches_no_tools(tmp_path) -> None:
    called = []
    session = ScriptedSession(
        start_turn=ModelTurn(tool_calls=[
            ModelToolCall(id="call_final", name="final_result", arguments='{"name":"Ada","age":37}'),
            ModelToolCall(id="call_boom", name="boom", arguments="{}"),
        ], raw={"id": "bad"})
    )
    harness = Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[], output_type=Person, output_mode="tool"),
        model=ScriptedModel([session]),
        tools=[ToolSpec("boom", "boom", {"type": "object", "properties": {}}, lambda args: called.append(True))],
        hooks=[Hook("before_tool_call", lambda ctx: called.append(True))],
    )

    with pytest.raises(UnexpectedModelBehavior):
        harness.run_sync("make a person")

    assert called == []


def test_repeated_final_result_calls_are_unexpected(tmp_path) -> None:
    session = ScriptedSession(
        start_turn=ModelTurn(tool_calls=[
            ModelToolCall(id="call_final_1", name="final_result", arguments='{"name":"Ada","age":37}'),
            ModelToolCall(id="call_final_2", name="final_result", arguments='{"name":"Grace","age":85}'),
        ], raw={"id": "bad"})
    )
    harness = Harness(HarnessConfig(root=tmp_path, builtin_tools=[], output_type=Person, output_mode="tool"), model=ScriptedModel([session]))

    with pytest.raises(UnexpectedModelBehavior):
        harness.run_sync("make a person")


def test_text_output_uses_text_path_without_synthetic_tool(tmp_path) -> None:
    captured = {}

    def on_start(_prompt, _instructions, tools, _metadata, _previous):
        captured["tools"] = tools

    session = ScriptedSession(start_turn=ModelTurn(text="plain", raw={"id": "resp_1"}), on_start=on_start)
    harness = Harness(HarnessConfig(root=tmp_path, builtin_tools=[], output_type=TextOutput()), model=ScriptedModel([session]))

    result = harness.run_sync("say hi")

    assert result.text == "plain"
    assert result.output == "plain"
    assert [tool["name"] for tool in captured["tools"]] == []


def test_final_result_tool_name_collision_is_rejected(tmp_path) -> None:
    with pytest.raises(ValueError, match="reserved"):
        Harness(
            HarnessConfig(root=tmp_path, builtin_tools=[], output_type=Person, output_mode="tool"),
            model=ScriptedModel([]),
            tools=[ToolSpec("final_result", "reserved", {"type": "object", "properties": {}}, lambda args: "bad")],
        )


def test_late_final_result_tool_name_collision_is_rejected(tmp_path) -> None:
    harness = Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[], output_type=Person, output_mode="tool"),
        model=ScriptedModel([]),
    )

    with pytest.raises(ValueError, match="reserved"):
        harness.add_tool(ToolSpec("final_result", "reserved", {"type": "object", "properties": {}}, lambda args: "bad"))


def test_final_result_hook_filter_is_allowed_but_never_fires(tmp_path) -> None:
    seen = []
    session = ScriptedSession(
        start_turn=ModelTurn(
            text="Done",
            tool_calls=[ModelToolCall(id="call_final", name="final_result", arguments='{"name":"Ada","age":37}')],
            raw={"id": "resp_1"},
        )
    )
    harness = Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[], output_type=Person, output_mode="tool"),
        model=ScriptedModel([session]),
        hooks=[Hook("before_tool_call", lambda ctx: seen.append(ctx.tool_name), tools=["final_result"])],
    )

    assert harness.run_sync("make a person").output == Person(name="Ada", age=37)
    assert seen == []


def test_anthropic_native_mode_is_rejected(tmp_path) -> None:
    model = AnthropicMessagesModel("claude-test", provider=FakeAnthropicProvider())

    with pytest.raises(ValueError, match="does not support native structured output"):
        Harness(HarnessConfig(root=tmp_path, builtin_tools=[], output_type=Person, output_mode="native"), model=model)


def test_native_output_marker_respects_provider_capabilities(tmp_path) -> None:
    model = AnthropicMessagesModel("claude-test", provider=FakeAnthropicProvider())

    with pytest.raises(ValueError, match="does not support native structured output"):
        Harness(HarnessConfig(root=tmp_path, builtin_tools=[], output_type=NativeOutput(Person)), model=model)


def test_native_schema_is_strict_normalized_for_openai(tmp_path) -> None:
    model = ScriptedModel([ScriptedSession(start_turn=ModelTurn(text='{"name":"Ada","age":37}', raw={"id": "native"}))])
    model.capabilities = ModelCapabilities(supports_json_schema_output=True, default_structured_output_mode="native")
    harness = Harness(HarnessConfig(root=tmp_path, builtin_tools=[], output_type=Person), model=model)

    request = harness.output_schema.structured_output_request()

    assert request.strict is True
    assert request.schema["additionalProperties"] is False


def test_native_nested_schema_is_strict_normalized_for_openai(tmp_path) -> None:
    model = ScriptedModel([ScriptedSession(start_turn=ModelTurn(text='{"name":"Ada","address":{"city":"London","zip_code":"NW1"}}', raw={"id": "native"}))])
    model.capabilities = ModelCapabilities(supports_json_schema_output=True, default_structured_output_mode="native")
    harness = Harness(HarnessConfig(root=tmp_path, builtin_tools=[], output_type=PersonWithAddress), model=model)

    request = harness.output_schema.structured_output_request()
    address_schema = request.schema["properties"]["address"]

    assert request.strict is True
    assert request.schema["additionalProperties"] is False
    assert address_schema["additionalProperties"] is False


def test_openrouter_explicit_native_provider_rejection_surfaces_as_provider_error(tmp_path) -> None:
    class RejectingProvider(ScriptedProvider):
        name = "OpenRouter"

        async def create_chat_completion(self, payload):
            raise ProviderError("native rejected")

    model = OpenRouterModel("openai/test", provider=RejectingProvider())
    harness = Harness(HarnessConfig(root=tmp_path, builtin_tools=[], output_type=Person, output_mode="native"), model=model)

    with pytest.raises(HarnessError, match="native rejected"):
        harness.run_sync("make a person")


@pytest.mark.parametrize(
    ("mode", "turn"),
    [
        ("tool", ModelTurn(tool_calls=[ModelToolCall(id="call_final", name="final_result", arguments='{"name":"Ada","age":37}')], raw={"id": "tool"})),
        ("native", ModelTurn(text='{"name":"Ada","age":37}', raw={"id": "native"})),
        ("prompted", ModelTurn(text='{"name":"Ada","age":37}', raw={"id": "prompted"})),
    ],
)
def test_structured_finalization_marks_model_span(tmp_path, mode: str, turn: ModelTurn) -> None:
    tracer = FakeTracer()
    model = ScriptedModel([ScriptedSession(start_turn=turn)])
    if mode == "native":
        model.capabilities = ModelCapabilities(supports_json_schema_output=True, default_structured_output_mode="native")
    harness = Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[], output_type=Person, output_mode=mode, tracing=[TracingOptions(tracer=tracer)]),
        model=model,
    )

    harness.run_sync("make a person")

    model_spans = [span for span in tracer.spans if span.name.startswith("chat ")]
    assert model_spans[-1].attributes["thinharness.output.mode"] == mode
    assert model_spans[-1].attributes["gen_ai.output.finalized"] is True
    agent_spans = [span for span in tracer.spans if span.name.startswith("invoke_agent")]
    assert "thinharness.output.mode" not in agent_spans[-1].attributes


async def test_named_subagent_structured_output_is_serialized_for_parent(tmp_path) -> None:
    session = ScriptedSession(
        start_turn=ModelTurn(tool_calls=[ModelToolCall(id="call_final", name="final_result", arguments='{"name":"Ada","age":37}')], raw={"id": "child"})
    )
    child_model = ScriptedModel([session])
    config = SubAgentConfig(name="typed", description="Typed helper.", inherit_parent_tools=True, output_type=Person, output_mode="tool")
    parent = Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[], subagents=[config]),
        model=child_model,
    )
    parent.add_tool(create_subagent_tool(parent, [config]))

    result = await run_subagent_tool(parent, [config], SubAgentArgs(task="make a person", agent="typed"))

    assert result.ok is True
    assert json.loads(result.content) == {"name": "Ada", "age": 37}
    assert result.metadata["structured_output"] is True


async def test_named_subagent_without_output_type_returns_text(tmp_path) -> None:
    config = SubAgentConfig(name="plain", description="Plain helper.", inherit_parent_tools=True)
    parent = Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[], subagents=[config]),
        model=ScriptedModel([ScriptedSession(start_turn=ModelTurn(text="child text", raw={"id": "child"}))]),
    )
    parent.add_tool(create_subagent_tool(parent, [config]))

    result = await run_subagent_tool(parent, [config], SubAgentArgs(task="plain work", agent="plain"))

    assert result.ok is True
    assert result.content == "child text"
    assert result.metadata["structured_output"] is False


def test_parent_output_type_is_not_inherited_by_subagents(tmp_path) -> None:
    config = SubAgentConfig(name="plain", description="Plain helper.", inherit_parent_tools=True)
    parent_model = ScriptedModel([])
    parent_model.capabilities = ModelCapabilities(supports_json_schema_output=True, default_structured_output_mode="native")
    parent = Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[], output_type=Person, subagents=[config]),
        model=parent_model,
    )

    assert parent.output_schema.mode == "native"
    assert build_child_harness(parent, None).output_schema is None
    assert build_child_harness(parent, config).output_schema is None


async def test_parent_native_mode_and_child_tool_mode_do_not_bleed_config(tmp_path) -> None:
    config = SubAgentConfig(name="typed", description="Typed helper.", inherit_parent_tools=True, output_type=Person, output_mode="tool")
    model = ScriptedModel([
        ScriptedSession(start_turn=ModelTurn(tool_calls=[
            ModelToolCall(id="call_final", name="final_result", arguments='{"name":"Ada","age":37}')
        ], raw={"id": "child"}))
    ])
    model.capabilities = ModelCapabilities(supports_json_schema_output=True, default_structured_output_mode="native")
    parent = Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[], output_type=Person, subagents=[config]),
        model=model,
    )
    parent.add_tool(create_subagent_tool(parent, [config]))

    result = await run_subagent_tool(parent, [config], SubAgentArgs(task="make a person", agent="typed"))

    assert parent.output_schema.mode == "native"
    assert json.loads(result.content) == {"name": "Ada", "age": 37}
    assert result.metadata["structured_output"] is True
