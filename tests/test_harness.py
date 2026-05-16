from __future__ import annotations

import asyncio
import json
from functools import partial
from pathlib import Path

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
    SubAgentConfig,
    ToolSpec,
    build_child_harness,
    call_tool,
    create_subagent_tool,
)
from thinharness.hooks import current_tool_call_context
from thinharness.providers import ModelTurn
from thinharness.tools import _invoke_tool


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

def test_anthropic_harness_reuses_model_without_message_leak(tmp_path: Path) -> None:
    provider = FakeAnthropicProvider()
    model = AnthropicMessagesModel("claude-test", provider=provider)
    harness = Harness(HarnessConfig(root=tmp_path, builtin_tools=[]), model=model, tools=[echo_tool()])

    assert harness.run_sync("first").text == "done"
    assert harness.run_sync("second").text == "done"

    assert provider.payloads[0]["messages"] == [{"role": "user", "content": "first"}]
    assert provider.payloads[2]["messages"] == [{"role": "user", "content": "second"}]

def test_openrouter_harness_reuses_model_without_message_leak(tmp_path: Path) -> None:
    provider = FakeOpenRouterProvider()
    model = OpenRouterModel("openai/test", provider=provider)
    harness = Harness(HarnessConfig(root=tmp_path, builtin_tools=[]), model=model, tools=[echo_tool()])

    assert harness.run_sync("first").text == "done"
    assert harness.run_sync("second").text == "done"

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
    assert "No skills are configured." in harness.system_instructions()

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
        seen.append(ctx.parsed_output["metadata"]["error_type"])

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
        async def start(self, *, prompt, instructions, tools, metadata=None, previous_response_id=None, structured_output=None):
            started.set()
            await asyncio.Event().wait()

        async def continue_with_tools(self, outputs, *, tools, metadata=None, structured_output=None):
            raise AssertionError("should not continue")

        async def continue_with_user_message(self, message, *, tools, metadata=None, structured_output=None):
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
