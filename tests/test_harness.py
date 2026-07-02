from __future__ import annotations

import asyncio
import json
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fakes import (
    FakeAnthropicProvider,
    FakeClient,
    FakeOpenRouterProvider,
    MultiCallClient,
    ScriptedModel,
    ScriptedSession,
    _fake_openai,
    echo_tool,
    tool_output,
)
from pydantic import BaseModel, Field

from thinharness import (
    AfterToolCallContext,
    AnthropicMessagesModel,
    Harness,
    HarnessConfig,
    HarnessError,
    Hook,
    OpenAIProvider,
    OpenAIResponsesModel,
    OpenRouterModel,
    RequestConstants,
    SubAgentConfig,
    ToolSpec,
    UnexpectedModelBehavior,
    build_child_harness,
    call_tool,
    create_subagent_tool,
)
from thinharness.core import _classify_run_failure
from thinharness.defaults import DEFAULT_PARALLEL_LLM_INSTRUCTIONS, DEFAULT_SEARCH_INSTRUCTIONS
from thinharness.hooks import current_tool_call_context
from thinharness.providers import ModelToolCall, ModelTurn, ProviderError
from thinharness.tools.base import _invoke_tool


def test_harness_tool_loop_with_custom_client(tmp_path: Path) -> None:
    (tmp_path / "hello.txt").write_text("hello", encoding="utf-8")
    client = FakeClient()
    harness = Harness(HarnessConfig(root=tmp_path, model="openai:test-model"), model=_fake_openai(client))
    result = harness.run_sync("read hello", metadata={"case": "test"})
    assert result.text == "done"
    assert client.payloads[0]["tools"]
    assert client.payloads[0]["metadata"] == {"case": "test"}
    assert client.payloads[1]["previous_response_id"] == "resp_1"
    assert client.payloads[1]["input"][0]["type"] == "function_call_output"
    assert "hello" in client.payloads[1]["input"][0]["output"]

def test_session_receives_falsy_metadata_when_run_has_no_metadata(tmp_path: Path) -> None:
    captured = {}
    session = ScriptedSession(
        start_turn=ModelTurn(text="done", raw={"id": "done"}),
        on_start=lambda _prompt, _instructions, _tools, metadata, _previous: captured.setdefault("metadata", metadata),
    )
    harness = Harness(HarnessConfig(root=tmp_path, builtin_tools=[]), model=ScriptedModel([session]))

    assert harness.run_sync("go").text == "done"

    # Providers guard payloads with `if metadata:`, so the normalized empty dict is
    # equivalent to None and never reaches the provider payload.
    assert not captured["metadata"]


class _FailureSpan:
    """Minimal span recorder for run-failure classification tests."""

    def __init__(self) -> None:
        self.exceptions = []
        self.error = None

    def record_exception(self, exc: Exception) -> None:
        """Record the exception passed to the span."""
        self.exceptions.append(exc)

    def set_error(self, message: str, error_type: str) -> None:
        """Record the span error status."""
        self.error = (message, error_type)


def test_classify_run_failure_preserves_exception_ladder_semantics() -> None:
    cases = [
        (ProviderError("provider failed"), "provider_error", HarnessError, False),
        (UnexpectedModelBehavior("bad turn"), "unexpected_model_behavior", UnexpectedModelBehavior, True),
        (HarnessError("harness failed"), "error", HarnessError, True),
        (ValueError("plain failed"), "error", ValueError, True),
    ]
    for exc, stop_reason, raised_type, same_exception in cases:
        run_ctx = SimpleNamespace(stop_reason="end_turn", terminal_error=None)
        span = _FailureSpan()

        raised = _classify_run_failure(run_ctx, span, exc)

        assert run_ctx.stop_reason == stop_reason
        assert isinstance(raised, raised_type)
        assert (raised is exc) is same_exception
        assert run_ctx.terminal_error is raised
        assert span.exceptions == [exc]
        assert span.error == (str(exc), type(exc).__name__)


def test_classify_run_failure_preserves_existing_harness_stop_reason() -> None:
    existing = HarnessError("blocked by hook")
    exc = HarnessError("strict hook failure")
    run_ctx = SimpleNamespace(stop_reason="cancelled_by_hook", terminal_error=existing)
    span = _FailureSpan()

    raised = _classify_run_failure(run_ctx, span, exc)

    assert raised is exc
    assert run_ctx.stop_reason == "cancelled_by_hook"
    assert run_ctx.terminal_error is existing

def test_max_model_requests_zero_blocks_before_provider_request(tmp_path: Path) -> None:
    session = ScriptedSession(start_turn=ModelTurn(text="done", raw={"id": "done"}))
    harness = Harness(HarnessConfig(root=tmp_path, builtin_tools=[], max_model_requests=0), model=ScriptedModel([session]))

    with pytest.raises(HarnessError, match="max_model_requests=0"):
        harness.run_sync("go")

    assert session.notice_calls == []

def test_final_model_request_notice_is_sent_on_initial_request(tmp_path: Path) -> None:
    session = ScriptedSession(start_turn=ModelTurn(text="done", raw={"id": "done"}))
    harness = Harness(HarnessConfig(root=tmp_path, builtin_tools=[], max_model_requests=1), model=ScriptedModel([session]))

    assert harness.run_sync("go").text == "done"

    assert session.notice_calls[0][0] == "start"
    assert [(notice.limit_kind, notice.remaining) for notice in session.notice_calls[0][1]] == [("model_requests", 1)]
    assert session.notice_calls[0][1][0].content == "Final request: produce the answer now; do not request tools."

def test_warning_only_run_does_not_fire_limit_reached(tmp_path: Path) -> None:
    events = []
    session = ScriptedSession(start_turn=ModelTurn(text="done", raw={"id": "done"}))
    harness = Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[], max_model_requests=1, max_tool_calls=0),
        model=ScriptedModel([session]),
        hooks=[Hook("limit_reached", lambda ctx: events.append(ctx.limit_kind))],
    )

    assert harness.run_sync("go").text == "done"

    assert events == []

def test_harness_config_has_no_notice_toggle() -> None:
    assert "limit_warnings" not in HarnessConfig.model_fields
    assert "model_notices" not in HarnessConfig.model_fields

def test_limit_notices_are_sent_on_tool_continuation(tmp_path: Path) -> None:
    session = ScriptedSession(
        start_turn=ModelTurn(tool_calls=[ModelToolCall(id="call_1", name="echo", arguments='{"value":"ok"}')], raw={"id": "start"}),
        continue_turn=ModelTurn(text="done", raw={"id": "done"}),
    )
    harness = Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[], max_model_requests=2, max_tool_calls=1),
        model=ScriptedModel([session]),
        tools=[echo_tool()],
    )

    assert harness.run_sync("go").text == "done"

    assert [(method, [(notice.limit_kind, notice.remaining) for notice in notices]) for method, notices in session.notice_calls] == [
        ("start", [("tool_calls", 1)]),
        ("continue_with_tools", [("model_requests", 1), ("tool_calls", 0)]),
    ]

def test_exhausted_tool_budget_notice_is_sent_on_initial_request(tmp_path: Path) -> None:
    session = ScriptedSession(start_turn=ModelTurn(text="done", raw={"id": "done"}))
    harness = Harness(HarnessConfig(root=tmp_path, builtin_tools=[], max_tool_calls=0), model=ScriptedModel([session]))

    assert harness.run_sync("go").text == "done"

    assert [(notice.limit_kind, notice.remaining) for notice in session.notice_calls[0][1]] == [("tool_calls", 0)]
    assert session.notice_calls[0][1][0].content == "Tool calls are not available on this run; answer without tools."

@pytest.mark.parametrize("max_tool_calls, expected_remaining", [(1, 1), (0, 0)])
def test_combined_limit_notices_use_stable_order(tmp_path: Path, max_tool_calls: int, expected_remaining: int) -> None:
    session = ScriptedSession(start_turn=ModelTurn(text="done", raw={"id": "done"}))
    harness = Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[], max_model_requests=1, max_tool_calls=max_tool_calls),
        model=ScriptedModel([session]),
    )

    assert harness.run_sync("go").text == "done"

    assert [(notice.limit_kind, notice.remaining) for notice in session.notice_calls[0][1]] == [
        ("model_requests", 1),
        ("tool_calls", expected_remaining),
    ]

def test_same_turn_tool_overage_does_not_send_continuation_notice(tmp_path: Path) -> None:
    session = ScriptedSession(
        start_turn=ModelTurn(tool_calls=[
            ModelToolCall(id="call_1", name="echo", arguments='{"value":"one"}'),
            ModelToolCall(id="call_2", name="echo", arguments='{"value":"two"}'),
        ], raw={"id": "start"})
    )
    harness = Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[], max_tool_calls=1),
        model=ScriptedModel([session]),
        tools=[echo_tool()],
    )

    with pytest.raises(HarnessError, match="max_tool_calls=1"):
        harness.run_sync("go")

    assert [(method, len(notices)) for method, notices in session.notice_calls] == [("start", 1)]

async def test_anthropic_harness_reuses_model_without_message_leak(tmp_path: Path) -> None:
    provider = FakeAnthropicProvider()
    model = AnthropicMessagesModel("claude-test", provider=provider)
    harness = Harness(HarnessConfig(root=tmp_path, builtin_tools=[]), model=model, tools=[echo_tool()])

    assert (await harness.run("first")).text == "done"
    assert (await harness.run("second")).text == "done"

    assert provider.payloads[0]["messages"] == [{"role": "user", "content": "first"}]
    assert provider.payloads[2]["messages"] == [{"role": "user", "content": "second"}]

async def test_openrouter_harness_reuses_model_without_message_leak(tmp_path: Path) -> None:
    provider = FakeOpenRouterProvider()
    model = OpenRouterModel("openai/test", provider=provider)
    harness = Harness(HarnessConfig(root=tmp_path, builtin_tools=[]), model=model, tools=[echo_tool()])

    assert (await harness.run("first")).text == "done"
    assert (await harness.run("second")).text == "done"

    assert provider.payloads[0]["messages"] == [
        {"role": "system", "content": harness.system_instructions()},
        {"role": "user", "content": "first"},
    ]
    assert provider.payloads[2]["messages"] == [
        {"role": "system", "content": harness.system_instructions()},
        {"role": "user", "content": "second"},
    ]

def test_custom_tool_is_exposed_and_callable(tmp_path: Path) -> None:
    custom = ToolSpec(
        "echo_json",
        "Echo input",
        {"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]},
        lambda args: {"echo": args["value"]},
    )
    harness = Harness(HarnessConfig(root=tmp_path), model=_fake_openai(FakeClient()), tools=[custom])
    assert any(tool["name"] == "echo_json" for tool in harness.tool_schemas())
    output = tool_output(call_tool(custom, '{"value":"ok"}'))
    assert output["ok"] is True
    assert json.loads(output["content"]) == {"echo": "ok"}

def test_custom_tool_can_use_pydantic_args_model(tmp_path: Path) -> None:
    class EchoArgs(BaseModel):
        value: str
        count: int = Field(default=1, ge=1)

    custom = ToolSpec("echo_typed", "Echo typed input", EchoArgs, lambda args: {"echo": args.value * args.count})
    harness = Harness(HarnessConfig(root=tmp_path), model=_fake_openai(FakeClient()), tools=[custom])

    schema = next(tool for tool in harness.tool_schemas() if tool["name"] == "echo_typed")["parameters"]
    assert schema["properties"]["count"]["minimum"] == 1
    output = tool_output(call_tool(custom, '{"value":"ok","count":2}'))
    assert json.loads(output["content"]) == {"echo": "okok"}
    invalid = tool_output(call_tool(custom, '{"value":"ok","count":0}'))
    assert invalid["ok"] is False
    assert invalid["metadata"]["error_type"] == "ValidationError"
    assert invalid["metadata"]["retry"] is True
    assert "Invalid arguments" in invalid["content"]

def test_custom_tool_invalid_json_is_structured(tmp_path: Path) -> None:
    custom = ToolSpec("echo", "Echo input", {"type": "object", "properties": {}}, lambda args: "ok")

    output = json.loads(call_tool(custom, "{bad json"))

    assert output["ok"] is False
    assert output["metadata"]["error_type"] == "InvalidArguments"
    assert output["metadata"]["retry"] is True
    assert "invalid JSON arguments" in output["content"]

def test_builtin_tool_selection_is_explicit(tmp_path: Path) -> None:
    harness = Harness(HarnessConfig(root=tmp_path, builtin_tools=["read", "search"]), model=_fake_openai(FakeClient()))
    assert [tool["name"] for tool in harness.tool_schemas()] == ["read", "search"]

def test_default_builtin_tools_are_minimal_filesystem_surface(tmp_path: Path) -> None:
    harness = Harness(HarnessConfig(root=tmp_path), model=_fake_openai(FakeClient()))
    assert [tool["name"] for tool in harness.tool_schemas()] == ["read", "write", "edit", "search", "list", "glob"]

def test_specialized_builtin_tools_are_explicit_opt_ins(tmp_path: Path) -> None:
    harness = Harness(HarnessConfig(root=tmp_path, builtin_tools=["jsonl_search", "subagent"]), model=_fake_openai(FakeClient()))
    assert [tool["name"] for tool in harness.tool_schemas()] == ["jsonl_search", "subagent"]

def test_enabled_tool_instructions_are_appended_after_base_instructions(tmp_path: Path) -> None:
    harness = Harness(
        HarnessConfig(root=tmp_path, builtin_tools=["parallel_llm"], system_prompt="Caller instructions."),
        model=ScriptedModel([]),
    )

    instructions = harness.system_instructions()

    assert instructions.startswith(f"Caller instructions.\n\nWorkspace root: {tmp_path}")
    assert instructions.endswith(DEFAULT_PARALLEL_LLM_INSTRUCTIONS)
    assert "It does not inherit the parent system prompt" in instructions

def test_disabled_tool_instructions_are_omitted(tmp_path: Path) -> None:
    harness = Harness(HarnessConfig(root=tmp_path, builtin_tools=["read"]), model=ScriptedModel([]))

    assert "parallel_llm usage:" not in harness.system_instructions()

def test_tool_instructions_follow_skill_summary(tmp_path: Path) -> None:
    demo = tmp_path / "skills" / "demo"
    demo.mkdir(parents=True)
    (demo / "SKILL.md").write_text("---\nname: demo\ndescription: Demo skill\n---\nDemo", encoding="utf-8")
    harness = Harness(
        HarnessConfig(
            root=tmp_path,
            skills_dir=tmp_path / "skills",
            builtin_tools=["skill_read", "parallel_llm"],
        ),
        model=ScriptedModel([]),
    )

    instructions = harness.system_instructions()

    assert instructions.index("demo - Demo skill") < instructions.index(DEFAULT_PARALLEL_LLM_INSTRUCTIONS)

def test_builtin_tool_instructions_are_appended(tmp_path: Path) -> None:
    harness = Harness(
        HarnessConfig(root=tmp_path, builtin_tools=["search"]),
        model=ScriptedModel([]),
    )

    assert DEFAULT_SEARCH_INSTRUCTIONS in harness.system_instructions()

def test_blank_tool_instructions_are_omitted(tmp_path: Path) -> None:
    custom = ToolSpec(
        "blank",
        "Blank instructions",
        {"type": "object", "properties": {}},
        lambda args: "ok",
        instructions="   ",
    )
    harness = Harness(HarnessConfig(root=tmp_path, builtin_tools=[]), model=ScriptedModel([]), tools=[custom])

    assert harness.system_instructions() == f"{harness.config.system_prompt}\n\nWorkspace root: {tmp_path}"

def test_tool_instructions_do_not_change_tool_schema(tmp_path: Path) -> None:
    custom = ToolSpec(
        "echo_json",
        "Echo input",
        {"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]},
        lambda args: {"echo": args["value"]},
        instructions="Use echo_json only when echoing JSON.",
    )
    harness = Harness(HarnessConfig(root=tmp_path, builtin_tools=[]), model=ScriptedModel([]), tools=[custom])

    schema = harness.tool_schemas()[0]

    assert schema == {
        "type": "function",
        "name": "echo_json",
        "description": "Echo input",
        "parameters": {"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]},
    }
    assert "Use echo_json only when echoing JSON." in harness.system_instructions()

def test_skill_dirs_require_selected_skill_tools(tmp_path: Path) -> None:
    skill = tmp_path / "skills" / "demo"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: demo\n---\nDemo", encoding="utf-8")

    harness = Harness(
        HarnessConfig(root=tmp_path, skills_dir=tmp_path / "skills", builtin_tools=["skill_read"]),
        model=_fake_openai(FakeClient()),
    )
    assert "skill_read" in [tool["name"] for tool in harness.tool_schemas()]
    with pytest.raises(ValueError, match="skill_read or skill_run"):
        Harness(HarnessConfig(root=tmp_path, skills_dir=tmp_path / "skills", builtin_tools=["read"]), model=_fake_openai(FakeClient()))

def test_skills_are_not_discovered_without_explicit_skills_dir(tmp_path: Path) -> None:
    skill = tmp_path / ".agents" / "skills" / "demo"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: demo\n---\nDemo", encoding="utf-8")

    harness = Harness(HarnessConfig(root=tmp_path), model=_fake_openai(FakeClient()))

    assert "skill_read" not in [tool["name"] for tool in harness.tool_schemas()]
    assert "No skills are configured." not in harness.system_instructions()

def test_selected_skills_are_exposed_when_skill_tool_is_selected(tmp_path: Path) -> None:
    demo = tmp_path / "skills" / "demo"
    other = tmp_path / "skills" / "other"
    demo.mkdir(parents=True)
    other.mkdir(parents=True)
    (demo / "SKILL.md").write_text("---\nname: demo\ndescription: Demo skill\n---\nDemo", encoding="utf-8")
    (other / "SKILL.md").write_text("---\nname: other\ndescription: Other skill\n---\nOther", encoding="utf-8")

    harness = Harness(
        HarnessConfig(
            root=tmp_path,
            skills_dir=tmp_path / "skills",
            selected_skills=["demo"],
            builtin_tools=["read", "skill_read"],
        ),
        model=_fake_openai(FakeClient()),
    )

    assert [tool["name"] for tool in harness.tool_schemas()] == ["read", "skill_read"]
    assert "demo - Demo skill" in harness.system_instructions()
    assert "other - Other skill" not in harness.system_instructions()

def test_selected_skills_without_skills_dir_fails() -> None:
    with pytest.raises(ValueError, match="selected_skills requires skills_dir"):
        HarnessConfig(selected_skills=["demo"])

def test_child_harness_tool_surfaces_follow_subagent_policy(tmp_path: Path) -> None:
    parent_echo = echo_tool()
    explicit_tool = ToolSpec("explicit", "Explicit sequential tool", {"type": "object", "properties": {}}, lambda args: "ok", sequential=True)
    parent = Harness(HarnessConfig(root=tmp_path, builtin_tools=[]), model=ScriptedModel([]), tools=[parent_echo])
    parent.add_tool(create_subagent_tool(parent, []))

    default_child = build_child_harness(parent, None)
    explicit_child = build_child_harness(parent, SubAgentConfig(name="special", description="Special helper.", tools=[explicit_tool]))

    assert default_child.tools == [parent_echo]
    assert default_child.config.subagents == []
    assert [tool.name for tool in explicit_child.tools] == ["explicit"]
    assert explicit_child.tools[0].sequential is True
    assert explicit_child.config.subagents == []
    assert build_child_harness(parent, None).model is parent.model

def test_duplicate_tool_names_are_rejected(tmp_path: Path) -> None:
    duplicate = ToolSpec("read", "Duplicate read", {"type": "object", "properties": {}}, lambda args: "ok")
    with pytest.raises(ValueError, match="duplicate tool name: read"):
        Harness(HarnessConfig(root=tmp_path), model=_fake_openai(FakeClient()), tools=[duplicate])

def test_after_tool_call_fires_for_handler_exception(tmp_path: Path) -> None:
    client = MultiCallClient([("boom", "{}")])
    seen = []

    def boom(_args):
        raise RuntimeError("nope")

    def after(ctx):
        assert isinstance(ctx, AfterToolCallContext)
        seen.append(ctx.envelope.metadata["error_type"])

    harness = Harness(
        HarnessConfig(root=tmp_path, model="openai:test-model", builtin_tools=[]),
        model=_fake_openai(client),
        tools=[ToolSpec("boom", "boom", {"type": "object", "properties": {}}, boom)],
        hooks=[Hook("after_tool_call", after)],
    )

    assert harness.run_sync("go").text == "done"
    assert seen == ["RuntimeError"]

async def test_async_run_supports_async_tool_handlers(tmp_path: Path) -> None:
    client = MultiCallClient([("async_echo", '{"value":"ok"}')])

    async def async_echo(args):
        await asyncio.sleep(0)
        return args["value"]

    harness = Harness(
        HarnessConfig(root=tmp_path, model="openai:test-model", builtin_tools=[]),
        model=_fake_openai(client),
        tools=[
            ToolSpec(
                "async_echo",
                "Async echo",
                {"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]},
                async_echo,
            )
        ],
    )

    result = await harness.run("go")

    assert result.text == "done"
    assert tool_output(client.payloads[1]["input"][0]["output"])["content"] == "ok"

async def test_async_tool_handlers_run_without_thread_hop_and_partial_works(tmp_path: Path) -> None:
    client = MultiCallClient([("async_partial", "{}")])
    loop_thread = None
    handler_thread = None

    import threading

    async def async_partial(args, *, value):
        nonlocal handler_thread
        await asyncio.sleep(0)
        handler_thread = threading.get_ident()
        return value

    loop_thread = threading.get_ident()
    harness = Harness(
        HarnessConfig(root=tmp_path, model="openai:test-model", builtin_tools=[]),
        model=_fake_openai(client),
        tools=[ToolSpec("async_partial", "Async partial", {"type": "object", "properties": {}}, partial(async_partial, value="ok"))],
    )

    result = await harness.run("go")

    assert result.text == "done"
    assert handler_thread == loop_thread
    assert tool_output(client.payloads[1]["input"][0]["output"])["content"] == "ok"

async def test_callable_object_async_handler_runs_directly(tmp_path: Path) -> None:
    client = MultiCallClient([("callable_async", "{}")])
    calls = 0

    class CallableAsync:
        async def __call__(self, args):
            nonlocal calls
            calls += 1
            await asyncio.sleep(0)
            return "ok"

    harness = Harness(
        HarnessConfig(root=tmp_path, model="openai:test-model", builtin_tools=[]),
        model=_fake_openai(client),
        tools=[ToolSpec("callable_async", "Callable async", {"type": "object", "properties": {}}, CallableAsync())],
    )

    assert (await harness.run("go")).text == "done"
    assert calls == 1
    assert tool_output(client.payloads[1]["input"][0]["output"])["content"] == "ok"

async def test_invoke_tool_calls_sync_handler_once() -> None:
    calls = 0

    def handler(args):
        nonlocal calls
        calls += 1
        return "ok"

    output = tool_output(await _invoke_tool(ToolSpec("once", "Once", {"type": "object", "properties": {}}, handler), "{}"))

    assert calls == 1
    assert output["content"] == "ok"

async def test_async_tool_exceptions_become_structured_outputs() -> None:
    async def fail(args):
        raise ValueError("bad")

    output = tool_output(await _invoke_tool(ToolSpec("fail", "Fail", {"type": "object", "properties": {}}, fail), "{}"))

    assert output["ok"] is False
    assert output["metadata"]["error_type"] == "ValueError"

def test_call_tool_with_async_handler_returns_structured_error() -> None:
    async def async_handler(args):
        return "ok"

    output = tool_output(call_tool(ToolSpec("async", "Async", {"type": "object", "properties": {}}, async_handler), "{}"))

    assert output["ok"] is False
    assert output["metadata"]["error_type"] == "AsyncHandlerInSyncContext"

async def test_tool_call_context_visible_in_async_and_threaded_handlers(tmp_path: Path) -> None:
    client = MultiCallClient([("async_ctx", "{}"), ("sync_ctx", "{}")])
    seen = {}

    async def async_ctx(args):
        seen["async"] = current_tool_call_context()
        return "async"

    def sync_ctx(args):
        seen["sync"] = current_tool_call_context()
        return "sync"

    harness = Harness(
        HarnessConfig(root=tmp_path, model="openai:test-model", builtin_tools=[]),
        model=_fake_openai(client),
        tools=[
            ToolSpec("async_ctx", "Async ctx", {"type": "object", "properties": {}}, async_ctx),
            ToolSpec("sync_ctx", "Sync ctx", {"type": "object", "properties": {}}, sync_ctx),
        ],
    )

    assert (await harness.run("go")).text == "done"
    assert seen["async"] == {"call_id": "call_1", "name": "async_ctx"}
    assert seen["sync"] == {"call_id": "call_2", "name": "sync_ctx"}

async def test_run_sync_inside_running_loop_raises(tmp_path: Path) -> None:
    harness = Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[]),
        model=ScriptedModel([ScriptedSession(start_turn=ModelTurn(text="done", raw={"id": "done"}))]),
    )

    with pytest.raises(HarnessError, match="running event loop"):
        harness.run_sync("go")

async def test_async_context_manager_closes_owned_provider_once(tmp_path: Path) -> None:
    class ClosingProvider:
        name = "OpenAI"

        def __init__(self) -> None:
            self.closed = 0

        async def aclose(self) -> None:
            self.closed += 1

    provider = ClosingProvider()
    model = ScriptedModel([ScriptedSession(start_turn=ModelTurn(text="done", raw={"id": "done"}))])
    model.provider = provider

    async with Harness(HarnessConfig(root=tmp_path, builtin_tools=[]), model=model, _owns_model=True) as harness:
        assert (await harness.run("go")).text == "done"

    await harness.aclose()
    assert provider.closed == 1

async def test_injected_http_client_is_not_closed_by_harness(tmp_path: Path) -> None:
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"id": "resp", "output_text": "done"})))
    provider = OpenAIProvider(api_key="key", http_client=client)
    model = OpenAIResponsesModel("gpt-test", provider=provider)
    harness = Harness(HarnessConfig(root=tmp_path, builtin_tools=[]), model=model, _owns_model=True)

    assert (await harness.run("go")).text == "done"
    await harness.aclose()

    assert not client.is_closed
    await client.aclose()

async def test_external_cancellation_records_run_end_and_allows_rerun(tmp_path: Path) -> None:
    started = asyncio.Event()
    events = []

    class WaitingSession:
        async def start(self, prompt, constants, *, previous_response_id=None, notices=None):
            started.set()
            await asyncio.Event().wait()

        async def continue_with_tools(self, outputs, constants, *, notices=None):
            raise AssertionError("should not continue")

        async def continue_with_user_text(self, text, constants, *, notices=None):
            raise AssertionError("should not continue")

    model = ScriptedModel([
        WaitingSession(),
        ScriptedSession(start_turn=ModelTurn(text="done", raw={"id": "done"})),
    ])
    harness = Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[]),
        model=model,
        hooks=[Hook("run_end", lambda ctx: events.append((ctx.stop_reason, type(ctx.error).__name__)))],
    )

    task = asyncio.create_task(harness.run("go"))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert events == [("cancelled", "CancelledError")]
    assert (await harness.run("again")).text == "done"

async def test_after_tool_call_does_not_fire_on_external_tool_cancellation(tmp_path: Path) -> None:
    client = MultiCallClient([("wait", "{}")])
    started = asyncio.Event()
    after_calls = []

    async def wait(_args):
        started.set()
        await asyncio.Event().wait()

    harness = Harness(
        HarnessConfig(root=tmp_path, model="openai:test-model", builtin_tools=[]),
        model=_fake_openai(client),
        tools=[ToolSpec("wait", "Wait", {"type": "object", "properties": {}}, wait)],
        hooks=[Hook("after_tool_call", lambda ctx: after_calls.append(ctx.tool_name))],
    )

    task = asyncio.create_task(harness.run("go"))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert after_calls == []

def test_harness_config_defaults_to_auto_tool_execution() -> None:
    assert HarnessConfig().tool_execution == "auto"
    assert HarnessConfig(tool_execution="sequential").tool_execution == "sequential"

def test_request_constants_is_exported() -> None:
    import thinharness

    assert "RequestConstants" in thinharness.__all__
    assert thinharness.RequestConstants is RequestConstants

def test_toolset_is_frozen_at_run_start(tmp_path: Path) -> None:
    seen_tools: list[list[str]] = []
    session = ScriptedSession(
        start_turn=ModelTurn(tool_calls=[ModelToolCall(id="call_1", name="register", arguments="{}")], raw={"id": "start"}),
        continue_turn=ModelTurn(text="done", raw={"id": "done"}),
        on_start=lambda _prompt, _instructions, tools, _metadata, _previous: seen_tools.append([tool["name"] for tool in tools]),
        on_continue=lambda _outputs, tools, _metadata: seen_tools.append([tool["name"] for tool in tools]),
    )

    def register(_args):
        harness.add_tool(ToolSpec("late", "Late", {"type": "object", "properties": {}}, lambda args: "late"))
        return "registered"

    harness = Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[]),
        model=ScriptedModel([session]),
        tools=[ToolSpec("register", "Register a tool mid-run", {"type": "object", "properties": {}}, register)],
    )

    assert harness.run_sync("go").text == "done"

    assert seen_tools == [["register"], ["register"]]
    assert any(tool.name == "late" for tool in harness.tools)
