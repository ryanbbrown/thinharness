from __future__ import annotations

import copy
import json
from typing import Any

import pytest
from fakes import FakeAnthropicProvider, FakeClient, FakeOpenRouterProvider, ReplayOpenAIProvider
from provider_test_helpers import ECHO_TOOLS
from provider_test_helpers import constants as _constants
from provider_test_helpers import notice as _notice
from provider_test_helpers import notice_text as _notice_text
from provider_test_helpers import tool_output as _tool_output

from thinharness import (
    AnthropicMessagesModel,
    HarnessError,
    OpenAIProvider,
    OpenAIResponsesModel,
    OpenRouterModel,
    ToolResult,
)
from thinharness.providers import ModelSettings, ProviderError, RequestConstants, ToolOutput


def _output(call_id: str, text: str, *, wire_output: str | None = None) -> ToolOutput:
    return ToolOutput(call_id, ToolResult(True, text), wire_output=wire_output)


def _state(items: list[Any]) -> dict[str, Any]:
    return {
        "kind": "transcript",
        "version": 5,
        "origin_provider": "openai",
        "origin_model": "gpt-5-mini",
        "entries": [],
        "openai_items": items,
    }


async def test_replay_sends_complete_exact_item_history() -> None:
    provider = ReplayOpenAIProvider()
    constants = RequestConstants(
        instructions="system",
        tools=ECHO_TOOLS,
        metadata={"conversation": "fixture"},
    )
    session = OpenAIResponsesModel("gpt-5-mini", provider=provider).new_session()

    first = await session.start("begin", constants)
    second = await session.continue_with_tools(
        [_output(first.tool_calls[0].id, "normalized", wire_output="wire-output-byte-for-byte")],
        constants,
        notices=[_notice()],
    )
    await session.continue_with_tools([_tool_output(second.tool_calls[0].id, "second result")], constants)
    await session.continue_with_user_content("follow-up", constants)

    user = {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "begin"}]}
    first_output = ReplayOpenAIProvider.responses[0]["output"]
    first_tool_result = {
        "type": "function_call_output",
        "call_id": "call_1",
        "output": "wire-output-byte-for-byte",
    }
    notice = {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": _notice_text()}],
    }
    second_output = ReplayOpenAIProvider.responses[1]["output"]
    second_tool_result = {
        "type": "function_call_output",
        "call_id": "call_2",
        "output": '{"ok": true, "content": "second result", "metadata": {}}',
    }
    final_output = ReplayOpenAIProvider.responses[2]["output"]
    follow_up = {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "follow-up"}]}
    expected_inputs = [
        [user],
        [user, *first_output, first_tool_result, notice],
        [user, *first_output, first_tool_result, notice, *second_output, second_tool_result],
        [user, *first_output, first_tool_result, notice, *second_output, second_tool_result, *final_output, follow_up],
    ]

    assert [payload["input"] for payload in provider.payloads] == expected_inputs
    for payload in provider.payloads:
        assert payload["store"] is False
        assert "previous_response_id" not in payload
        assert payload["instructions"] == "system"
        assert payload["tools"] == ECHO_TOOLS
        assert payload["metadata"] == {"conversation": "fixture"}
        assert payload["include"] == ["reasoning.encrypted_content"]
    assert provider.payloads[2]["input"][1:4] == first_output
    assert provider.payloads[2]["input"][1]["encrypted_content"] == "encrypted-1"
    assert provider.payloads[2]["input"][2]["phase"] == "commentary"
    assert provider.payloads[2]["input"][3]["id"] == "fc_1"


async def test_replay_non_reasoning_model_omits_include() -> None:
    provider = ReplayOpenAIProvider(responses=[{"id": "done", "output": []}])
    await OpenAIResponsesModel("gpt-4.1-mini", provider=provider).new_session().start("hi", _constants())

    assert provider.payloads[0]["store"] is False
    assert "include" not in provider.payloads[0]


async def test_replay_rejects_seeded_previous_response_id_without_call() -> None:
    provider = ReplayOpenAIProvider()
    session = OpenAIResponsesModel("gpt-5-mini", provider=provider).new_session()

    with pytest.raises(ProviderError, match="state_mode=\"continuation\""):
        await session.start("hi", _constants(), previous_response_id="resp_existing")

    assert provider.payloads == []


async def test_continuation_mode_keeps_response_id_chain_and_resume_replay() -> None:
    provider = FakeClient()
    model = OpenAIResponsesModel("gpt-test", provider=provider, state_mode="continuation")
    session = model.new_session()

    first = await session.start("hi", _constants(), previous_response_id="resp_existing")
    await session.continue_with_tools([_tool_output(first.tool_calls[0].id, "ok")], _constants())
    state = session.dump_state()

    assert provider.payloads[0]["input"] == "hi"
    assert provider.payloads[0]["previous_response_id"] == "resp_existing"
    assert provider.payloads[1]["previous_response_id"] == "resp_1"
    assert [item["type"] for item in provider.payloads[1]["input"]] == ["function_call_output"]
    assert all("store" not in payload for payload in provider.payloads)
    assert "openai_items" not in state

    resumed_provider = FakeClient()
    resumed = OpenAIResponsesModel("gpt-test", provider=resumed_provider, state_mode="continuation").resume_session(
        json.loads(json.dumps(state))
    )
    await resumed.continue_with_user_content("follow-up", _constants())
    assert [item["type"] for item in resumed_provider.payloads[0]["input"]] == [
        "message",
        "function_call",
        "function_call_output",
        "message",
        "message",
    ]
    assert "previous_response_id" not in resumed_provider.payloads[0]
    assert "store" not in resumed_provider.payloads[0]

    replay_provider = FakeClient()
    replay_resumed = OpenAIResponsesModel("gpt-test", provider=replay_provider).resume_session(json.loads(json.dumps(state)))
    await replay_resumed.continue_with_user_content("replayed", _constants())
    assert replay_provider.payloads[0]["store"] is False
    assert replay_provider.payloads[0]["input"][-1]["content"][0]["text"] == "replayed"


@pytest.mark.parametrize("key", ["input", "previous_response_id", "store"])
def test_replay_reserves_state_extra_body_keys(key: str) -> None:
    settings = ModelSettings(extra_body={key: "override"})

    with pytest.raises(ValueError, match="reserves extra_body keys"):
        OpenAIResponsesModel("gpt-test", provider=OpenAIProvider(api_key="fake"), settings=settings)

    OpenAIResponsesModel(
        "gpt-test",
        provider=OpenAIProvider(api_key="fake"),
        settings=settings,
        state_mode="continuation",
    )


def test_invalid_state_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="state_mode"):
        OpenAIResponsesModel("gpt-test", provider=OpenAIProvider(api_key="fake"), state_mode="invalid")  # type: ignore[arg-type]


def test_build_payload_does_not_add_store_by_default() -> None:
    payload = OpenAIResponsesModel("gpt-test", provider=OpenAIProvider(api_key="fake")).build_payload(
        input_payload="hi",
        tools=[],
    )

    assert "store" not in payload


async def test_only_function_call_items_are_extracted_as_tools() -> None:
    provider = ReplayOpenAIProvider(responses=[{
        "id": "resp_1",
        "output": [{"type": "tool_call", "id": "legacy", "name": "read", "arguments": "{}"}],
    }])

    turn = await OpenAIResponsesModel("gpt-test", provider=provider).new_session().start("hi", _constants())

    assert turn.tool_calls == []


async def test_missing_encrypted_reasoning_fails_without_retry() -> None:
    provider = ReplayOpenAIProvider(responses=[{
        "id": "resp_1",
        "output": [{"type": "reasoning", "id": "rs_1", "summary": []}],
    }])
    session = OpenAIResponsesModel("future-reasoning-model", provider=provider).new_session()

    with pytest.raises(ProviderError, match="_openai_supports_encrypted_reasoning.*state_mode=\"continuation\""):
        await session.start("hi", _constants())

    assert len(provider.payloads) == 1


async def test_failed_request_keeps_transcript_and_raw_inputs_aligned() -> None:
    provider = ReplayOpenAIProvider(fail_at=2)
    session = OpenAIResponsesModel("gpt-5-mini", provider=provider).new_session()
    first = await session.start("hi", _constants())

    with pytest.raises(ProviderError, match="scripted failure"):
        await session.continue_with_tools([_tool_output(first.tool_calls[0].id, "result")], _constants())

    state = session.dump_state()
    assert [entry["role"] for entry in state["entries"]] == ["user", "assistant", "tool"]
    assert [item["type"] for item in state["openai_items"]][-2:] == ["function_call", "function_call_output"]
    assert state["openai_items"][-1]["call_id"] == state["entries"][-1]["call_id"]


async def test_crash_resume_matches_uninterrupted_next_request() -> None:
    constants = _constants(ECHO_TOOLS)
    uninterrupted_provider = ReplayOpenAIProvider()
    uninterrupted = OpenAIResponsesModel("gpt-5-mini", provider=uninterrupted_provider).new_session()
    first = await uninterrupted.start("hi", constants)
    second = await uninterrupted.continue_with_tools([_tool_output(first.tool_calls[0].id, "one")], constants)
    state = json.loads(json.dumps(uninterrupted.dump_state()))
    await uninterrupted.continue_with_tools([_tool_output(second.tool_calls[0].id, "two")], constants)

    resumed_provider = ReplayOpenAIProvider(responses=[ReplayOpenAIProvider.responses[2]])
    resumed = OpenAIResponsesModel("gpt-5-mini", provider=resumed_provider).resume_session(state)
    await resumed.continue_with_tools([_tool_output(second.tool_calls[0].id, "two")], constants)

    assert resumed_provider.payloads[0] == uninterrupted_provider.payloads[2]


async def test_approval_shaped_resume_matches_uninterrupted_request() -> None:
    constants = _constants(ECHO_TOOLS)
    uninterrupted_provider = ReplayOpenAIProvider()
    uninterrupted = OpenAIResponsesModel("gpt-5-mini", provider=uninterrupted_provider).new_session()
    first = await uninterrupted.start("hi", constants)
    state = json.loads(json.dumps(uninterrupted.dump_state()))
    assert state["openai_items"][-1]["type"] == "function_call"
    await uninterrupted.continue_with_tools([_tool_output(first.tool_calls[0].id, "approved")], constants)

    resumed_provider = ReplayOpenAIProvider(responses=[ReplayOpenAIProvider.responses[1]])
    resumed = OpenAIResponsesModel("gpt-5-mini", provider=resumed_provider).resume_session(state)
    await resumed.continue_with_tools([_tool_output(first.tool_calls[0].id, "approved")], constants)

    assert resumed_provider.payloads[0] == uninterrupted_provider.payloads[1]


async def test_follow_up_resume_replays_final_raw_message() -> None:
    constants = _constants(ECHO_TOOLS)
    provider = ReplayOpenAIProvider()
    session = OpenAIResponsesModel("gpt-5-mini", provider=provider).new_session()
    first = await session.start("hi", constants)
    second = await session.continue_with_tools([_tool_output(first.tool_calls[0].id, "one")], constants)
    await session.continue_with_tools([_tool_output(second.tool_calls[0].id, "two")], constants)
    state = json.loads(json.dumps(session.dump_state()))

    resumed_provider = ReplayOpenAIProvider(responses=[ReplayOpenAIProvider.responses[3]])
    resumed = OpenAIResponsesModel("gpt-5-mini", provider=resumed_provider).resume_session(state)
    await resumed.continue_with_user_content("next", constants)

    assert resumed_provider.payloads[0]["input"][:-1] == state["openai_items"]
    assert resumed_provider.payloads[0]["input"][-2] == ReplayOpenAIProvider.responses[2]["output"][0]
    assert resumed_provider.payloads[0]["input"][-1]["content"][0]["text"] == "next"


async def test_resume_forks_do_not_share_or_mutate_item_history() -> None:
    source_provider = ReplayOpenAIProvider()
    source = OpenAIResponsesModel("gpt-5-mini", provider=source_provider).new_session()
    first = await source.start("hi", _constants())
    state = json.loads(json.dumps(source.dump_state()))
    original = copy.deepcopy(state)
    left_provider = ReplayOpenAIProvider(responses=[{"id": "left", "output": []}])
    right_provider = ReplayOpenAIProvider(responses=[{"id": "right", "output": []}])
    left = OpenAIResponsesModel("gpt-5-mini", provider=left_provider).resume_session(state)
    right = OpenAIResponsesModel("gpt-5-mini", provider=right_provider).resume_session(state)

    await left.continue_with_tools([_tool_output(first.tool_calls[0].id, "left")], _constants())
    await right.continue_with_tools([_tool_output(first.tool_calls[0].id, "right")], _constants())

    assert left_provider.payloads[0]["input"][:-1] == right_provider.payloads[0]["input"][:-1] == original["openai_items"]
    assert left_provider.payloads[0]["input"][-1] != right_provider.payloads[0]["input"][-1]
    assert state == original


@pytest.mark.parametrize(
    "items",
    [
        [{"type": "function_call_output", "call_id": "call_1", "output": "x"}],
        [{"type": "function_call", "call_id": "call_1"}, {"type": "function_call_output", "call_id": "call_2", "output": "x"}],
        [
            {"type": "function_call", "call_id": "call_1"},
            {"type": "function_call_output", "call_id": "call_1", "output": "x"},
            {"type": "function_call_output", "call_id": "call_1", "output": "x"},
        ],
        [
            {"type": "function_call", "call_id": "call_1"},
            {"type": "reasoning", "encrypted_content": "enc"},
            {"type": "message", "role": "assistant", "content": []},
        ],
        [{"type": "function_call", "call_id": "call_1"}, {"type": "message", "role": "user", "content": []}],
        [{"type": "reasoning", "encrypted_content": "enc"}],
        [
            {"type": "reasoning", "encrypted_content": "enc"},
            {"type": "message", "role": "user", "content": []},
        ],
        [
            {"type": "reasoning", "encrypted_content": "enc"},
            {"type": "function_call_output", "call_id": "call_1", "output": "x"},
        ],
        ["not-a-dict"],
        [{"role": "user"}],
    ],
    ids=[
        "output-before-call",
        "dropped-call",
        "duplicate-output",
        "unanswered-before-reasoning",
        "unanswered-before-user",
        "reasoning-last",
        "reasoning-before-user",
        "reasoning-before-output",
        "non-dict",
        "missing-type",
    ],
)
def test_malformed_raw_histories_fail_before_provider_call(items: list[Any]) -> None:
    provider = ReplayOpenAIProvider()
    model = OpenAIResponsesModel("gpt-5-mini", provider=provider)

    with pytest.raises(HarnessError, match="resume_from openai_items"):
        model.resume_session(_state(items))

    assert provider.payloads == []


def test_openai_items_must_be_a_list() -> None:
    with pytest.raises(HarnessError, match="field 'openai_items' has wrong type"):
        OpenAIResponsesModel("gpt-test", provider=FakeClient()).resume_session(_state(None))  # type: ignore[arg-type]


def test_other_providers_accept_valid_openai_items_and_use_entries() -> None:
    state = {
        "kind": "transcript",
        "version": 5,
        "origin_provider": "openai",
        "origin_model": "gpt-test",
        "entries": [{"role": "user", "content": [{"type": "text", "text": "prior"}], "notice": False}],
        "openai_items": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "raw"}]}],
    }

    anthropic = FakeAnthropicProvider()
    openrouter = FakeOpenRouterProvider()
    assert AnthropicMessagesModel("claude-test", provider=anthropic).resume_session(state).transcript[0].content[0].text == "prior"  # type: ignore[attr-defined]
    assert OpenRouterModel("openai/test", provider=openrouter).resume_session(state).transcript[0].content[0].text == "prior"  # type: ignore[attr-defined]

