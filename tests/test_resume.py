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


def test_openai_resume_uses_previous_response_id_only_for_followup(tmp_path: Path) -> None:
    (tmp_path / "hello.txt").write_text("hello", encoding="utf-8")
    client = FakeClient()
    harness = Harness(HarnessConfig(root=tmp_path, builtin_tools=["read"]), model=OpenAIResponsesModel("gpt-test", provider=client))

    first = harness.run_sync("first")
    state = json.loads(json.dumps(first.resume_state))
    second = harness.run_sync("follow-up", resume_from=state)

    assert first.resume_state == {"kind": "openai", "version": 1, "model": "gpt-test", "previous_response_id": "resp_2"}
    assert second.text == "done"
    assert client.payloads[2]["input"] == "follow-up"
    assert client.payloads[2]["previous_response_id"] == "resp_2"
    assert "first" not in json.dumps(client.payloads[2])


def test_anthropic_resume_replays_transcript_and_appends_new_user_turn(tmp_path: Path) -> None:
    provider = FakeAnthropicProvider()
    model = AnthropicMessagesModel("claude-test", provider=provider)
    harness = Harness(HarnessConfig(root=tmp_path, builtin_tools=[]), model=model, tools=[echo_tool()])

    first = harness.run_sync("first")
    state = json.loads(json.dumps(first.resume_state))
    second = harness.run_sync("follow-up", resume_from=state)

    assert second.text == "done"
    assert provider.payloads[2]["messages"][0] == {"role": "user", "content": "first"}
    assert provider.payloads[2]["messages"][-1] == {"role": "user", "content": "follow-up"}
    assistant_tool_use = provider.payloads[2]["messages"][1]["content"][0]
    user_tool_result = provider.payloads[2]["messages"][2]["content"][0]
    assert assistant_tool_use["type"] == "tool_use"
    assert user_tool_result["type"] == "tool_result"
    assert user_tool_result["tool_use_id"] == assistant_tool_use["id"]


def test_openrouter_resume_replays_transcript_and_appends_new_user_turn(tmp_path: Path) -> None:
    provider = FakeOpenRouterProvider()
    model = OpenRouterModel("openai/test", provider=provider)
    harness = Harness(HarnessConfig(root=tmp_path, builtin_tools=[]), model=model, tools=[echo_tool()])

    first = harness.run_sync("first")
    state = json.loads(json.dumps(first.resume_state))
    second = harness.run_sync("follow-up", resume_from=state)

    assert second.text == "done"
    assert provider.payloads[2]["messages"][0] == {"role": "system", "content": harness.system_instructions()}
    assert provider.payloads[2]["messages"][1] == {"role": "user", "content": "first"}
    assert provider.payloads[2]["messages"][-1] == {"role": "user", "content": "follow-up"}
    assistant_tool_call = provider.payloads[2]["messages"][2]["tool_calls"][0]
    tool_message = provider.payloads[2]["messages"][3]
    assert assistant_tool_call["type"] == "function"
    assert tool_message["role"] == "tool"
    assert tool_message["tool_call_id"] == assistant_tool_call["id"]


def test_resume_rejects_provider_model_version_and_unknown_key_mismatches(tmp_path: Path) -> None:
    openai = Harness(HarnessConfig(root=tmp_path, builtin_tools=[]), model=OpenAIResponsesModel("gpt-test", provider=FakeClient()))
    state = openai.run_sync("first").resume_state
    anthropic = Harness(HarnessConfig(root=tmp_path, builtin_tools=[]), model=AnthropicMessagesModel("claude-test", provider=FakeAnthropicProvider()))

    with pytest.raises(HarnessError, match="resume_from kind"):
        anthropic.run_sync("follow-up", resume_from=state)

    wrong_model = {**state, "model": "other"}
    with pytest.raises(HarnessError, match="resume_from model"):
        openai.run_sync("follow-up", resume_from=wrong_model)

    wrong_version = {**state, "version": 2}
    with pytest.raises(HarnessError, match="resume_from version 2 is not supported"):
        openai.run_sync("follow-up", resume_from=wrong_version)

    missing_version = {key: value for key, value in state.items() if key != "version"}
    with pytest.raises(HarnessError, match="resume_from version None is not supported"):
        openai.run_sync("follow-up", resume_from=missing_version)

    unknown = {**state, "foo": "bar"}
    with pytest.raises(HarnessError, match="unknown keys.*foo"):
        openai.run_sync("follow-up", resume_from=unknown)


def test_resume_rejects_malformed_shapes_before_hooks_fire(tmp_path: Path) -> None:
    events: list[str] = []
    harness = Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[]),
        model=AnthropicMessagesModel("claude-test", provider=FakeAnthropicProvider()),
        hooks=[
            Hook("run_start", lambda ctx: events.append(type(ctx).__name__)),
            Hook("user_prompt_submit", lambda ctx: events.append(type(ctx).__name__)),
        ],
    )

    with pytest.raises(HarnessError, match="resume_from must be a dict"):
        harness.run_sync("follow-up", resume_from="resp_abc")  # type: ignore[arg-type]
    with pytest.raises(HarnessError, match="resume_from kind None does not match 'anthropic'"):
        harness.run_sync("follow-up", resume_from={"version": 1, "model": "claude-test", "system": "", "messages": []})
    with pytest.raises(HarnessError, match="field 'previous_response_id' has wrong type"):
        Harness(HarnessConfig(root=tmp_path, builtin_tools=[]), model=OpenAIResponsesModel("gpt-test", provider=FakeClient())).run_sync(
            "follow-up",
            resume_from={"kind": "openai", "version": 1, "model": "gpt-test", "previous_response_id": 123},
        )
    with pytest.raises(HarnessError, match="missing required field: 'system'"):
        harness.run_sync("follow-up", resume_from={"kind": "anthropic", "version": 1, "model": "claude-test"})
    with pytest.raises(HarnessError, match="field 'messages' has wrong type"):
        harness.run_sync("follow-up", resume_from={"kind": "anthropic", "version": 1, "model": "claude-test", "system": "", "messages": ["bad"]})
    with pytest.raises(HarnessError, match="JSON-serializable"):
        harness.run_sync(
            "follow-up",
            resume_from={"kind": "anthropic", "version": 1, "model": "claude-test", "system": "", "messages": [{"bad": datetime.now()}]},
        )

    assert events == []


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


def test_no_openai_response_id_omits_resume_state(tmp_path: Path) -> None:
    class NoIdProvider(OpenAIProvider):
        def __init__(self) -> None:
            super().__init__(api_key="fake")

        async def create_response(self, payload):
            return {"output_text": "done"}

    harness = Harness(HarnessConfig(root=tmp_path, builtin_tools=[]), model=OpenAIResponsesModel("gpt-test", provider=NoIdProvider()))

    assert harness.run_sync("first").resume_state is None


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
        async def start(self, *, prompt, instructions, tools, metadata=None, previous_response_id=None, structured_output=None) -> ModelTurn:
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


def test_resume_state_is_detached_outbound_and_inbound(tmp_path: Path) -> None:
    provider = FakeOpenRouterProvider()
    model = OpenRouterModel("openai/test", provider=provider)
    harness = Harness(HarnessConfig(root=tmp_path, builtin_tools=[]), model=model, tools=[echo_tool()])
    first = harness.run_sync("first")
    stashed = json.loads(json.dumps(first.resume_state))

    first.resume_state["messages"][1]["content"] = "mutated"
    inbound = json.loads(json.dumps(stashed))
    result = harness.run_sync("follow-up", resume_from=inbound)
    inbound["messages"][1]["content"] = "mutated after call"

    assert result.text == "done"
    assert provider.payloads[2]["messages"][1]["content"] == "first"


def test_fresh_harness_persistence_and_sequential_branching(tmp_path: Path) -> None:
    provider = FakeOpenRouterProvider()
    first_harness = Harness(HarnessConfig(root=tmp_path, builtin_tools=[]), model=OpenRouterModel("openai/test", provider=provider), tools=[echo_tool()])
    state = json.loads(json.dumps(first_harness.run_sync("first").resume_state))

    second_harness = Harness(HarnessConfig(root=tmp_path, builtin_tools=[]), model=OpenRouterModel("openai/test", provider=provider), tools=[echo_tool()])
    first_branch = second_harness.run_sync("branch one", resume_from=state)
    second_branch = second_harness.run_sync("branch two", resume_from=state)

    assert first_branch.text == "done"
    assert second_branch.text == "done"
    assert provider.payloads[2]["messages"][-1]["content"] == "branch one"
    assert provider.payloads[4]["messages"][-1]["content"] == "branch two"


def test_adapter_validation_does_not_mutate_state_on_failure() -> None:
    model = AnthropicMessagesModel("claude-test", provider=FakeAnthropicProvider())

    with pytest.raises(HarnessError):
        model.resume_session({"kind": "anthropic", "version": 1, "model": "claude-test", "system": "", "messages": ["bad"]})

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
    model = _NoResumeModel([_NoResumeSession()])
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
        harness.run_sync("again", resume_from={"kind": "none"})
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
        async def start(self, *, prompt, instructions, tools, metadata=None, previous_response_id=None, structured_output=None):
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

    async def start(self, *, prompt, instructions, tools, metadata=None, previous_response_id=None, structured_output=None) -> ModelTurn:
        """Return a terminal text turn."""
        return ModelTurn(text="done", raw={"id": "done"})

    async def continue_with_tools(self, outputs, *, tools, metadata=None, structured_output=None) -> ModelTurn:
        """Reject unexpected tool continuation."""
        raise AssertionError("unexpected tool continuation")

    async def continue_with_user_message(self, message, *, tools, metadata=None, structured_output=None) -> ModelTurn:
        """Reject unexpected corrective continuation."""
        raise AssertionError("unexpected user-message continuation")

    async def continue_with_user_prompt(self, *, prompt, instructions, tools, metadata=None, structured_output=None) -> ModelTurn:
        """Reject unexpected resumed continuation."""
        raise AssertionError("unexpected resumed prompt")
