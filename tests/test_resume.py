from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path

import pytest
from fakes import FakeAnthropicProvider, FakeClient, FakeOpenRouterProvider, ScriptedProvider, ScriptedSession, echo_tool
from pydantic import BaseModel

from thinharness import (
    AnthropicMessagesModel,
    Harness,
    HarnessConfig,
    HarnessError,
    Hook,
    ModelRetry,
    ModelToolCall,
    ModelTurn,
    OpenAIProvider,
    OpenAIResponsesModel,
    OpenRouterModel,
    ToolSpec,
)
from thinharness.hooks import RunEndContext
from thinharness.providers import ModelSession, ProviderError


class _TerminalOpenAIProvider(OpenAIProvider):
    def __init__(self) -> None:
        super().__init__(api_key="fake")
        self.payloads = []

    async def create_response(self, payload):
        self.payloads.append(payload)
        return {"id": f"resp_{len(self.payloads)}", "output_text": "done"}


class _MultiToolAnthropicProvider(FakeAnthropicProvider):
    async def create_message(self, payload):
        """Capture payloads and request two echo tool calls on the first user turn."""
        self.payloads.append(json.loads(json.dumps(payload)))
        last = payload["messages"][-1]
        if isinstance(last["content"], str):
            return {
                "content": [
                    {"type": "tool_use", "id": "toolu_a", "name": "echo", "input": {"value": "a"}},
                    {"type": "tool_use", "id": "toolu_b", "name": "echo", "input": {"value": "b"}},
                ],
                "stop_reason": "tool_use",
            }
        return {"content": [{"type": "text", "text": "done"}], "stop_reason": "end_turn"}


async def test_openai_resume_full_replays_transcript_for_followup(tmp_path: Path) -> None:
    (tmp_path / "hello.txt").write_text("hello", encoding="utf-8")
    client = FakeClient()
    harness = Harness(HarnessConfig(root=tmp_path, builtin_tools=["read"]), model=OpenAIResponsesModel("gpt-test", provider=client))

    first = await harness.run("first")
    state = json.loads(json.dumps(first.resume_state))
    second = await harness.run("follow-up", resume_from=state)

    assert first.resume_state["kind"] == "transcript"
    assert first.resume_state["version"] == 3
    assert first.resume_state["origin_provider"] == "openai"
    assert first.resume_state["origin_model"] == "gpt-test"
    assert [entry["role"] for entry in first.resume_state["entries"]] == ["user", "assistant", "tool", "assistant"]
    assert second.text == "done"
    assert client.payloads[1]["previous_response_id"] == "resp_1"
    assert client.payloads[1]["instructions"] == harness.system_instructions()
    assert "previous_response_id" not in client.payloads[2]
    assert [item["type"] for item in client.payloads[2]["input"]] == [
        "message",
        "function_call",
        "function_call_output",
        "message",
        "message",
    ]
    assert client.payloads[2]["input"][0]["content"][0]["text"] == "first"
    assert client.payloads[2]["input"][-1]["content"][0]["text"] == "follow-up"
    assert client.payloads[2]["instructions"] == harness.system_instructions()
    assert "first" in json.dumps(client.payloads[2])


async def test_anthropic_resume_replays_transcript_and_appends_new_user_turn(tmp_path: Path) -> None:
    provider = FakeAnthropicProvider()
    model = AnthropicMessagesModel("claude-test", provider=provider)
    harness = Harness(HarnessConfig(root=tmp_path, builtin_tools=[]), model=model, tools=[echo_tool()])

    first = await harness.run("first")
    state = json.loads(json.dumps(first.resume_state))
    second = await harness.run("follow-up", resume_from=state)

    assert second.text == "done"
    assert provider.payloads[2]["messages"][0] == {"role": "user", "content": "first"}
    assert provider.payloads[2]["messages"][-1] == {"role": "user", "content": "follow-up"}
    assistant_tool_use = provider.payloads[2]["messages"][1]["content"][0]
    user_tool_result = provider.payloads[2]["messages"][2]["content"][0]
    assert assistant_tool_use["type"] == "tool_use"
    assert user_tool_result["type"] == "tool_result"
    assert user_tool_result["tool_use_id"] == assistant_tool_use["id"]


async def test_openrouter_resume_replays_transcript_and_appends_new_user_turn(tmp_path: Path) -> None:
    provider = FakeOpenRouterProvider()
    model = OpenRouterModel("openai/test", provider=provider)
    harness = Harness(HarnessConfig(root=tmp_path, builtin_tools=[]), model=model, tools=[echo_tool()])

    first = await harness.run("first")
    state = json.loads(json.dumps(first.resume_state))
    second = await harness.run("follow-up", resume_from=state)

    assert second.text == "done"
    assert provider.payloads[2]["messages"][0] == {"role": "system", "content": harness.system_instructions()}
    assert provider.payloads[2]["messages"][1] == {"role": "user", "content": "first"}
    assert provider.payloads[2]["messages"][-1] == {"role": "user", "content": "follow-up"}
    assistant_tool_call = provider.payloads[2]["messages"][2]["tool_calls"][0]
    tool_message = provider.payloads[2]["messages"][3]
    assert assistant_tool_call["type"] == "function"
    assert tool_message["role"] == "tool"
    assert tool_message["tool_call_id"] == assistant_tool_call["id"]
    assert provider.payloads[2]["messages"][4] == {"role": "assistant", "content": "done"}


async def test_cross_provider_resume_after_tool_round_trip(tmp_path: Path) -> None:
    source_provider = FakeAnthropicProvider()
    source = Harness(HarnessConfig(root=tmp_path, builtin_tools=[]), model=AnthropicMessagesModel("claude-test", provider=source_provider), tools=[echo_tool()])
    state = json.loads(json.dumps((await source.run("first")).resume_state))

    openai_provider = _TerminalOpenAIProvider()
    openai_result = await Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[]),
        model=OpenAIResponsesModel("gpt-test", provider=openai_provider),
        tools=[echo_tool()],
    ).run("follow-up", resume_from=state)
    openai_input = openai_provider.payloads[0]["input"]
    assert openai_result.text == "done"
    assert [item["type"] for item in openai_input[:3]] == ["message", "function_call", "function_call_output"]
    assert openai_input[1]["call_id"] == "toolu_1"
    assert openai_input[2]["call_id"] == "toolu_1"

    openrouter_provider = FakeOpenRouterProvider()
    openrouter_result = await Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[]),
        model=OpenRouterModel("openai/test", provider=openrouter_provider),
        tools=[echo_tool()],
    ).run("follow-up", resume_from=state)
    replay = openrouter_provider.payloads[0]["messages"]
    assert openrouter_result.text == "done"
    assert replay[2]["tool_calls"][0]["id"] == "toolu_1"
    assert replay[3]["tool_call_id"] == "toolu_1"


async def test_multi_tool_batch_replay_shapes(tmp_path: Path) -> None:
    source_provider = _MultiToolAnthropicProvider()
    source = Harness(HarnessConfig(root=tmp_path, builtin_tools=[]), model=AnthropicMessagesModel("claude-test", provider=source_provider), tools=[echo_tool()])
    state = json.loads(json.dumps((await source.run("first")).resume_state))
    assert json.loads(json.dumps(state)) == state

    anthropic_provider = FakeAnthropicProvider()
    await Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[]),
        model=AnthropicMessagesModel("claude-test", provider=anthropic_provider),
        tools=[echo_tool()],
    ).run("follow-up", resume_from=state)
    tool_result_blocks = anthropic_provider.payloads[0]["messages"][2]["content"]
    assert [block["type"] for block in tool_result_blocks] == ["tool_result", "tool_result"]
    assert [block["tool_use_id"] for block in tool_result_blocks] == ["toolu_a", "toolu_b"]

    openrouter_provider = FakeOpenRouterProvider()
    await Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[]),
        model=OpenRouterModel("openai/test", provider=openrouter_provider),
        tools=[echo_tool()],
    ).run("follow-up", resume_from=state)
    roles = [message["role"] for message in openrouter_provider.payloads[0]["messages"]]
    assert roles[:6] == ["system", "user", "assistant", "tool", "tool", "assistant"]


async def test_resume_rederives_live_system_prompt(tmp_path: Path) -> None:
    anthropic_source = Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[], system_prompt="old system"),
        model=AnthropicMessagesModel("claude-test", provider=FakeAnthropicProvider()),
        tools=[echo_tool()],
    )
    anthropic_state = json.loads(json.dumps((await anthropic_source.run("first")).resume_state))
    anthropic_provider = FakeAnthropicProvider()
    await Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[], system_prompt="new system"),
        model=AnthropicMessagesModel("claude-test", provider=anthropic_provider),
        tools=[echo_tool()],
    ).run("follow-up", resume_from=anthropic_state)
    assert anthropic_provider.payloads[0]["system"].startswith("new system")
    assert "old system" not in anthropic_provider.payloads[0]["system"]

    openrouter_source = Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[], system_prompt="old system"),
        model=OpenRouterModel("openai/test", provider=FakeOpenRouterProvider()),
        tools=[echo_tool()],
    )
    openrouter_state = json.loads(json.dumps((await openrouter_source.run("first")).resume_state))
    openrouter_provider = FakeOpenRouterProvider()
    await Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[], system_prompt="new system"),
        model=OpenRouterModel("openai/test", provider=openrouter_provider),
        tools=[echo_tool()],
    ).run("follow-up", resume_from=openrouter_state)
    assert openrouter_provider.payloads[0]["messages"][0]["content"].startswith("new system")
    assert "old system" not in openrouter_provider.payloads[0]["messages"][0]["content"]

    openai_source = Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[], system_prompt="old system"),
        model=OpenAIResponsesModel("gpt-test", provider=_TerminalOpenAIProvider()),
    )
    openai_state = json.loads(json.dumps((await openai_source.run("first")).resume_state))
    openai_provider = _TerminalOpenAIProvider()
    await Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[], system_prompt="new system"),
        model=OpenAIResponsesModel("gpt-test", provider=openai_provider),
    ).run("follow-up", resume_from=openai_state)
    assert openai_provider.payloads[0]["instructions"].startswith("new system")
    assert "old system" not in openai_provider.payloads[0]["instructions"]


def test_resume_allows_provider_model_mismatches_and_rejects_bad_versions_and_keys(tmp_path: Path) -> None:
    class NoToolOpenAIProvider(OpenAIProvider):
        def __init__(self) -> None:
            super().__init__(api_key="fake")

        async def create_response(self, payload):
            return {"id": "resp_text", "output_text": "done"}

    def openai_harness(model_name: str = "gpt-test") -> Harness:
        """Create a fresh OpenAI resume harness."""
        return Harness(HarnessConfig(root=tmp_path, builtin_tools=[]), model=OpenAIResponsesModel(model_name, provider=NoToolOpenAIProvider()))

    state = openai_harness().run_sync("first").resume_state
    anthropic = Harness(HarnessConfig(root=tmp_path, builtin_tools=[]), model=AnthropicMessagesModel("claude-test", provider=FakeAnthropicProvider()))

    assert anthropic.run_sync("follow-up", resume_from=state).text == "done"
    assert openai_harness("other").run_sync("follow-up", resume_from=state).text == "done"

    wrong_version = {**state, "version": 1}
    with pytest.raises(HarnessError, match="resume_from version 1 is not supported"):
        openai_harness().run_sync("follow-up", resume_from=wrong_version)

    missing_version = {key: value for key, value in state.items() if key != "version"}
    with pytest.raises(HarnessError, match="resume_from version None is not supported"):
        openai_harness().run_sync("follow-up", resume_from=missing_version)

    unknown = {**state, "foo": "bar"}
    with pytest.raises(HarnessError, match="unknown keys.*foo"):
        openai_harness().run_sync("follow-up", resume_from=unknown)

    with pytest.raises(HarnessError, match="resume_from kind 'openai' is not supported"):
        openai_harness().run_sync("follow-up", resume_from={"kind": "openai", "version": 1, "model": "gpt-test", "previous_response_id": "resp_1"})


def test_resume_rejects_malformed_shapes_before_hooks_fire(tmp_path: Path) -> None:
    events: list[str] = []

    def harness() -> Harness:
        """Create a fresh harness for one run_sync validation case."""
        return Harness(
            HarnessConfig(root=tmp_path, builtin_tools=[]),
            model=AnthropicMessagesModel("claude-test", provider=FakeAnthropicProvider()),
            hooks=[
                Hook("run_start", lambda ctx: events.append(type(ctx).__name__)),
                Hook("user_prompt_submit", lambda ctx: events.append(type(ctx).__name__)),
            ],
        )

    with pytest.raises(HarnessError, match="resume_from must be a dict"):
        harness().run_sync("follow-up", resume_from="resp_abc")  # type: ignore[arg-type]
    base_state = {"kind": "transcript", "version": 3, "origin_provider": "anthropic", "origin_model": "claude-test"}
    with pytest.raises(HarnessError, match="resume_from kind None is not supported"):
        harness().run_sync("follow-up", resume_from={"version": 2, "origin_provider": "anthropic", "origin_model": "claude-test", "entries": []})
    with pytest.raises(HarnessError, match="missing required field: 'entries'"):
        harness().run_sync("follow-up", resume_from=base_state)
    with pytest.raises(HarnessError, match="field 'entries' has wrong type"):
        harness().run_sync("follow-up", resume_from={**base_state, "entries": "bad"})
    with pytest.raises(HarnessError, match="entry 'user' has wrong keys"):
        harness().run_sync("follow-up", resume_from={**base_state, "entries": [{"role": "user", "content": "hi"}]})
    with pytest.raises(HarnessError, match="assistant tool call has wrong type"):
        bad_assistant = {"role": "assistant", "text": "", "tool_calls": [{"id": 1, "name": "x", "arguments": "{}"}], "reasoning": []}
        harness().run_sync(
            "follow-up",
            resume_from={**base_state, "entries": [bad_assistant]},
        )
    with pytest.raises(HarnessError, match="JSON-serializable"):
        harness().run_sync(
            "follow-up",
            resume_from={**base_state, "entries": [{"role": "user", "content": datetime.now(), "notice": False}]},
        )

    assert events == []


def test_anthropic_resume_rejects_non_json_tool_arguments() -> None:
    model = AnthropicMessagesModel("claude-test", provider=FakeAnthropicProvider())

    with pytest.raises(HarnessError, match="resume_from assistant tool call arguments must be JSON"):
        model.resume_session({
            "kind": "transcript",
            "version": 3,
            "origin_provider": "openrouter",
            "origin_model": "openai/test",
            "entries": [{
                "role": "assistant",
                "text": "",
                "tool_calls": [{"id": "call_1", "name": "echo", "arguments": "{bad"}],
                "reasoning": [],
            }],
        })


def test_structured_output_final_result_omits_resume_state(tmp_path: Path) -> None:
    class Answer(BaseModel):
        value: str

    session = ScriptedSession(
        start_turn=ModelTurn(
            text="Done",
            tool_calls=[ModelToolCall(id="call_final", name="final_result", arguments='{"value":"ok"}')],
            raw={"id": "resp_1"},
        )
    )
    model = _ScriptedResumeModel([session])
    harness = Harness(HarnessConfig(root=tmp_path, builtin_tools=[], output_type=Answer, output_mode="tool"), model=model)

    result = harness.run_sync("make output")

    assert result.stop_reason == "end_turn"
    assert result.output == Answer(value="ok")
    assert result.resume_state is None


def test_resumed_run_can_use_structured_output_and_still_omits_resume_state(tmp_path: Path) -> None:
    class Answer(BaseModel):
        value: str

    first_session = ScriptedSession(
        start_turn=ModelTurn(text="ready", raw={"id": "first"}),
        dump_state={"kind": "scripted", "version": 1, "model": "scripted"},
    )
    resumed_session = ScriptedSession(
        start_turn=ModelTurn(
            text="Done",
            tool_calls=[ModelToolCall(id="call_final", name="final_result", arguments='{"value":"ok"}')],
            raw={"id": "second"},
        )
    )
    model = _ScriptedResumeModel([first_session, resumed_session])
    first = Harness(HarnessConfig(root=tmp_path, builtin_tools=[]), model=model).run_sync("first")
    resumed = Harness(HarnessConfig(root=tmp_path, builtin_tools=[], output_type=Answer, output_mode="tool"), model=model).run_sync(
        "structured follow-up",
        resume_from=first.resume_state,
    )

    assert resumed.output == Answer(value="ok")
    assert resumed.resume_state is None

def test_resumed_user_prompt_receives_limit_notice(tmp_path: Path) -> None:
    first_session = ScriptedSession(
        start_turn=ModelTurn(text="ready", raw={"id": "first"}),
        dump_state={"kind": "scripted", "version": 1, "model": "scripted"},
    )
    resumed_session = ScriptedSession(start_turn=ModelTurn(text="done", raw={"id": "second"}))
    model = _ScriptedResumeModel([first_session, resumed_session])
    first = Harness(HarnessConfig(root=tmp_path, builtin_tools=[], max_model_requests=1), model=model).run_sync("first")

    resumed = Harness(HarnessConfig(root=tmp_path, builtin_tools=[], max_model_requests=1), model=model).run_sync(
        "follow-up",
        resume_from=first.resume_state,
    )

    assert resumed.text == "done"

    assert [(method, [(notice.limit_kind, notice.remaining) for notice in notices]) for method, notices in resumed_session.notice_calls] == [
        ("continue_with_user_prompt", [("model_requests", 1)])
    ]

def test_resumed_user_prompt_runs_prompt_submit_hooks_before_notices(tmp_path: Path) -> None:
    captured: dict[str, str] = {}
    events: list[str] = []
    first_session = ScriptedSession(
        start_turn=ModelTurn(text="ready", raw={"id": "first"}),
        dump_state={"kind": "scripted", "version": 1, "model": "scripted"},
    )
    resumed_session = ScriptedSession(
        start_turn=ModelTurn(text="done", raw={"id": "second"}),
        on_start=lambda prompt, _instructions, _tools, _metadata, _previous: captured.setdefault("prompt", prompt),
    )
    model = _ScriptedResumeModel([first_session, resumed_session])
    first = Harness(HarnessConfig(root=tmp_path, builtin_tools=[]), model=model).run_sync("first")

    def add_context(ctx) -> None:
        events.append("user_prompt_submit")
        ctx.additional_context.append("resume policy")

    resumed = Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[], max_model_requests=1),
        model=model,
        hooks=[Hook("user_prompt_submit", add_context)],
    ).run_sync("follow-up", resume_from=first.resume_state)

    assert resumed.text == "done"
    assert events == ["user_prompt_submit"]
    assert captured["prompt"] == "follow-up\n\n<hook_context>\nresume policy\n</hook_context>"
    assert resumed_session.notice_calls[0][1][0].content == "Final request: produce the answer now; do not request tools."


def test_no_openai_response_id_still_produces_resume_state(tmp_path: Path) -> None:
    class NoIdProvider(OpenAIProvider):
        def __init__(self) -> None:
            super().__init__(api_key="fake")
            self.payloads = []

        async def create_response(self, payload):
            self.payloads.append(payload)
            return {"output_text": "done"}

    provider = NoIdProvider()
    harness = Harness(HarnessConfig(root=tmp_path, builtin_tools=[]), model=OpenAIResponsesModel("gpt-test", provider=provider))

    first = harness.run_sync("first")
    assert first.resume_state is not None
    assert first.resume_state["kind"] == "transcript"
    resumed = Harness(HarnessConfig(root=tmp_path, builtin_tools=[]), model=OpenAIResponsesModel("gpt-test", provider=provider)).run_sync(
        "follow-up",
        resume_from=json.loads(json.dumps(first.resume_state)),
    )
    assert resumed.text == "done"
    assert "previous_response_id" not in provider.payloads[1]
    assert [item["type"] for item in provider.payloads[1]["input"]] == ["message", "message", "message"]


def test_non_clean_exits_omit_resume_state(tmp_path: Path) -> None:
    session = ScriptedSession(
        start_turn=ModelTurn(tool_calls=[ModelToolCall(id="call_1", name="missing", arguments="{}")], raw={"id": "start"}),
        dump_state={"kind": "scripted", "version": 1, "model": "scripted"},
    )
    harness = Harness(HarnessConfig(root=tmp_path, builtin_tools=[], max_tool_calls=0), model=_ScriptedResumeModel([session]))

    with pytest.raises(HarnessError, match="max_tool_calls"):
        harness.run_sync("go")

    captured: list[RunEndContext] = []
    harness = Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[], max_tool_calls=0),
        model=_ScriptedResumeModel([session]),
        hooks=[Hook("run_end", lambda ctx: captured.append(ctx))],
    )
    with pytest.raises(HarnessError):
        harness.run_sync("go")
    assert captured[-1].result is None
    assert captured[-1].stop_reason == "limit_reached"


def test_provider_error_omits_resume_state(tmp_path: Path) -> None:
    class FailingProviderSession(_NoResumeSession):
        async def start(self, *, prompt, instructions, tools, metadata=None, previous_response_id=None, structured_output=None, notices=None) -> ModelTurn:
            """Fail the provider request."""
            raise ProviderError("provider failed")

    captured: list[RunEndContext] = []
    harness = Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[]),
        model=_ScriptedResumeModel([FailingProviderSession()]),
        hooks=[Hook("run_end", lambda ctx: captured.append(ctx))],
    )

    with pytest.raises(HarnessError, match="provider failed"):
        harness.run_sync("go")
    assert captured[-1].result is None
    assert captured[-1].stop_reason == "provider_error"


def test_tool_retries_exceeded_omits_resume_state(tmp_path: Path) -> None:
    session = ScriptedSession(
        start_turn=ModelTurn(tool_calls=[ModelToolCall(id="call_1", name="flaky", arguments="{}")], raw={"id": "start"}),
        dump_state={"kind": "scripted", "version": 1, "model": "scripted"},
    )
    captured: list[RunEndContext] = []
    harness = Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[], tool_retries=0),
        model=_ScriptedResumeModel([session]),
        tools=[ToolSpec("flaky", "Flaky", {"type": "object", "properties": {}}, lambda args: (_ for _ in ()).throw(ModelRetry("again")))],
        hooks=[Hook("run_end", lambda ctx: captured.append(ctx))],
    )

    with pytest.raises(HarnessError, match="exceeded max_retries=0"):
        harness.run_sync("go")
    assert captured[-1].result is None
    assert captured[-1].stop_reason == "tool_retries_exceeded"


def test_cancelled_by_hook_omits_resume_state(tmp_path: Path) -> None:
    captured: list[RunEndContext] = []

    def cancel(ctx) -> None:
        ctx.cancelled = True
        ctx.cancel_reason = "blocked"

    harness = Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[]),
        model=_ScriptedResumeModel([_NoResumeSession()]),
        hooks=[Hook("user_prompt_submit", cancel), Hook("run_end", lambda ctx: captured.append(ctx))],
    )

    with pytest.raises(HarnessError, match="run blocked by hook: blocked"):
        harness.run_sync("go")
    assert captured[-1].result is None
    assert captured[-1].stop_reason == "cancelled_by_hook"


def test_output_validation_failed_omits_resume_state(tmp_path: Path) -> None:
    class Answer(BaseModel):
        value: str

    session = ScriptedSession(
        start_turn=ModelTurn(tool_calls=[ModelToolCall(id="call_final", name="final_result", arguments='{"missing":"ok"}')], raw={"id": "bad"}),
        dump_state={"kind": "scripted", "version": 1, "model": "scripted"},
    )
    captured: list[RunEndContext] = []
    harness = Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[], output_type=Answer, output_mode="tool", output_retries=0),
        model=_ScriptedResumeModel([session]),
        hooks=[Hook("run_end", lambda ctx: captured.append(ctx))],
    )

    with pytest.raises(HarnessError, match="output validation exceeded"):
        harness.run_sync("go")
    assert captured[-1].result is None
    assert captured[-1].stop_reason == "output_validation_failed"


async def test_resume_state_is_detached_outbound_and_inbound(tmp_path: Path) -> None:
    provider = FakeOpenRouterProvider()
    model = OpenRouterModel("openai/test", provider=provider)
    harness = Harness(HarnessConfig(root=tmp_path, builtin_tools=[]), model=model, tools=[echo_tool()])
    first = await harness.run("first")
    stashed = json.loads(json.dumps(first.resume_state))

    first.resume_state["entries"][0]["content"] = "mutated"
    inbound = json.loads(json.dumps(stashed))
    result = await harness.run("follow-up", resume_from=inbound)
    inbound["entries"][0]["content"] = "mutated after call"

    assert result.text == "done"
    assert provider.payloads[2]["messages"][1]["content"] == "first"


async def test_fresh_harness_persistence_and_sequential_branching(tmp_path: Path) -> None:
    provider = FakeOpenRouterProvider()
    first_harness = Harness(HarnessConfig(root=tmp_path, builtin_tools=[]), model=OpenRouterModel("openai/test", provider=provider), tools=[echo_tool()])
    state = json.loads(json.dumps((await first_harness.run("first")).resume_state))

    second_harness = Harness(HarnessConfig(root=tmp_path, builtin_tools=[]), model=OpenRouterModel("openai/test", provider=provider), tools=[echo_tool()])
    first_branch = await second_harness.run("branch one", resume_from=state)
    second_branch = await second_harness.run("branch two", resume_from=state)

    assert first_branch.text == "done"
    assert second_branch.text == "done"
    assert provider.payloads[2]["messages"][-1]["content"] == "branch one"
    assert provider.payloads[4]["messages"][-1]["content"] == "branch two"


def test_adapter_validation_does_not_mutate_state_on_failure() -> None:
    model = AnthropicMessagesModel("claude-test", provider=FakeAnthropicProvider())

    with pytest.raises(HarnessError):
        model.resume_session({
            "kind": "transcript",
            "version": 3,
            "origin_provider": "anthropic",
            "origin_model": "claude-test",
            "entries": [{"role": "user", "content": "missing notice flag"}],
        })

    session = model.new_session()
    assert session.messages == []


def test_adapter_non_json_dump_propagates_type_error(tmp_path: Path) -> None:
    session = ScriptedSession(
        start_turn=ModelTurn(text="done", raw={"id": "done"}),
        dump_state={"kind": "scripted", "version": 1, "model": "scripted", "bad": object()},
    )
    harness = Harness(HarnessConfig(root=tmp_path, builtin_tools=[]), model=_ScriptedResumeModel([session]))

    with pytest.raises(TypeError):
        harness.run_sync("go")


def test_custom_model_without_resume_support_can_run_but_cannot_resume(tmp_path: Path) -> None:
    events: list[str] = []
    model = _NoResumeModel([_NoResumeSession(), _NoResumeSession()])
    harness = Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[]),
        model=model,
        hooks=[
            Hook("run_start", lambda ctx: events.append(type(ctx).__name__)),
            Hook("user_prompt_submit", lambda ctx: events.append(type(ctx).__name__)),
        ],
    )

    assert harness.run_sync("go").text == "done"
    events.clear()
    with pytest.raises(HarnessError, match="does not support resume"):
        Harness(
            HarnessConfig(root=tmp_path, builtin_tools=[]),
            model=model,
            hooks=[
                Hook("run_start", lambda ctx: events.append(type(ctx).__name__)),
                Hook("user_prompt_submit", lambda ctx: events.append(type(ctx).__name__)),
            ],
        ).run_sync("again", resume_from={"kind": "none"})
    assert events == []


def test_resumable_model_session_missing_dump_state_raises(tmp_path: Path) -> None:
    class MissingDumpStateSession(_NoResumeSession):
        """Session that omits dump_state despite a resumable model."""

        dump_state = None

    harness = Harness(HarnessConfig(root=tmp_path, builtin_tools=[]), model=_ScriptedResumeModel([MissingDumpStateSession()]))

    with pytest.raises(HarnessError, match="resumable model session is missing dump_state"):
        harness.run_sync("go")


def test_run_end_context_sees_resume_state(tmp_path: Path) -> None:
    captured: list[dict | None] = []
    harness = Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[]),
        model=OpenAIResponsesModel("gpt-test", provider=FakeClient()),
        hooks=[Hook("run_end", lambda ctx: captured.append(ctx.result.resume_state if ctx.result else None))],
    )

    result = harness.run_sync("first")

    assert captured == [result.resume_state]
    assert captured[0] is not None


async def test_reentrancy_beats_resume_validation(tmp_path: Path) -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    class SlowSession(_NoResumeSession):
        async def start(self, *, prompt, instructions, tools, metadata=None, previous_response_id=None, structured_output=None, notices=None):
            started.set()
            await release.wait()
            return ModelTurn(text="done", raw={"id": "done"})

    harness = Harness(HarnessConfig(root=tmp_path, builtin_tools=[]), model=_NoResumeModel([SlowSession()]))
    task = asyncio.create_task(harness.run("go"))
    await started.wait()
    with pytest.raises(HarnessError, match="not re-entrant"):
        await harness.run("again", resume_from="bad")  # type: ignore[arg-type]
    release.set()
    await task


class _ScriptedResumeModel:
    """Small resumable model for core-level resume lifecycle tests."""

    resume_kind = "scripted"

    def __init__(self, sessions: list[ModelSession], *, model: str = "scripted") -> None:
        self.model = model
        self.provider = ScriptedProvider()
        self.api_key = "key"
        self.sessions = list(sessions)

    def new_session(self) -> ModelSession:
        """Return the next scripted session."""
        return self.sessions.pop(0)

    def resume_session(self, state: dict) -> ModelSession:
        """Return the next scripted session for a resumed run."""
        return self.sessions.pop(0)


class _NoResumeModel:
    """Custom model that intentionally does not implement resume."""

    def __init__(self, sessions: list[ModelSession]) -> None:
        self.model = "no-resume"
        self.provider = ScriptedProvider()
        self.api_key = "key"
        self.sessions = list(sessions)

    def new_session(self) -> ModelSession:
        """Return the next custom session."""
        return self.sessions.pop(0)


class _NoResumeSession:
    """Minimal session implementing the non-resume run contract."""

    async def start(self, *, prompt, instructions, tools, metadata=None, previous_response_id=None, structured_output=None, notices=None) -> ModelTurn:
        """Return a terminal text turn."""
        return ModelTurn(text="done", raw={"id": "done"})

    async def continue_with_tools(self, outputs, *, instructions=None, tools, metadata=None, structured_output=None, notices=None) -> ModelTurn:
        """Reject unexpected tool continuation."""
        raise AssertionError("unexpected tool continuation")

    async def continue_with_user_message(self, message, *, instructions=None, tools, metadata=None, structured_output=None, notices=None) -> ModelTurn:
        """Reject unexpected corrective continuation."""
        raise AssertionError("unexpected user-message continuation")

    async def continue_with_user_prompt(self, *, prompt, instructions, tools, metadata=None, structured_output=None, notices=None) -> ModelTurn:
        """Reject unexpected resumed continuation."""
        raise AssertionError("unexpected resumed prompt")
