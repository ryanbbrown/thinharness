from __future__ import annotations

import json

from fakes import FakeAnthropicProvider, FakeClient, FakeOpenRouterProvider
from provider_test_helpers import ECHO_TOOLS
from provider_test_helpers import constants as _constants
from provider_test_helpers import notice as _notice
from provider_test_helpers import notice_text as _notice_text
from provider_test_helpers import tool_output as _tool_output

from thinharness import (
    AnthropicMessagesModel,
    OpenAIResponsesModel,
    OpenRouterModel,
)


async def test_model_sessions_advance_independently() -> None:
    provider = FakeAnthropicProvider()
    model = AnthropicMessagesModel("claude-test", provider=provider)
    constants = _constants(ECHO_TOOLS)
    first = model.new_session()
    second = model.new_session()

    first_turn = await first.start("first", constants)
    second_turn = await second.start("second", constants)
    await first.continue_with_tools([_tool_output(first_turn.tool_calls[0].id, "first result")], constants)
    await second.continue_with_tools([_tool_output(second_turn.tool_calls[0].id, "second result")], constants)

    assert provider.payloads[2]["messages"][0] == {"role": "user", "content": "first"}
    assert provider.payloads[2]["messages"][-1]["content"][0]["content"] == '{"ok": true, "content": "first result", "metadata": {}}'
    assert provider.payloads[3]["messages"][0] == {"role": "user", "content": "second"}
    assert provider.payloads[3]["messages"][-1]["content"][0]["content"] == '{"ok": true, "content": "second result", "metadata": {}}'


async def test_non_replay_envelopes_omit_openai_items() -> None:
    constants = _constants(ECHO_TOOLS)

    anthropic = AnthropicMessagesModel("claude-test", provider=FakeAnthropicProvider()).new_session()
    await anthropic.start("hi", constants)
    openrouter = OpenRouterModel("openai/test", provider=FakeOpenRouterProvider()).new_session()
    await openrouter.start("hi", constants)
    continuation = OpenAIResponsesModel(
        "gpt-test",
        provider=FakeClient(),
        state_mode="continuation",
    ).new_session()
    await continuation.start("hi", constants)

    states = [anthropic.dump_state(), openrouter.dump_state(), continuation.dump_state()]
    assert all(state is not None and "openai_items" not in state for state in states)


async def test_resume_replays_preserved_tool_notices() -> None:
    constants = _constants(ECHO_TOOLS)
    notice = _notice()

    anthropic_provider = FakeAnthropicProvider()
    anthropic_session = AnthropicMessagesModel("claude-test", provider=anthropic_provider).new_session()
    anthropic_first = await anthropic_session.start("hi", constants)
    await anthropic_session.continue_with_tools([_tool_output(anthropic_first.tool_calls[0].id, "ok")], constants, notices=[notice])
    anthropic_state = json.loads(json.dumps(anthropic_session.dump_state()))
    assert anthropic_state == json.loads(json.dumps(anthropic_state))
    anthropic_resumed = AnthropicMessagesModel("claude-test", provider=anthropic_provider).resume_session(anthropic_state)
    await anthropic_resumed.continue_with_user_content("next", constants)
    assert anthropic_provider.payloads[2]["messages"][2]["content"][-1] == {"type": "text", "text": _notice_text()}

    openai_capture = FakeClient()
    openai_session = OpenAIResponsesModel("gpt-test", provider=openai_capture).new_session()
    openai_first = await openai_session.start("hi", constants)
    await openai_session.continue_with_tools([_tool_output(openai_first.tool_calls[0].id, "ok")], constants, notices=[notice])
    openai_state = json.loads(json.dumps(openai_session.dump_state()))
    openai_replay = FakeClient()
    openai_resumed = OpenAIResponsesModel("gpt-test", provider=openai_replay).resume_session(openai_state)
    await openai_resumed.continue_with_user_content("next", constants)
    assert {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": _notice_text()}],
    } in openai_replay.payloads[0]["input"]

    openrouter_provider = FakeOpenRouterProvider()
    openrouter_session = OpenRouterModel("openai/test", provider=openrouter_provider).new_session()
    openrouter_first = await openrouter_session.start("hi", constants)
    await openrouter_session.continue_with_tools([_tool_output(openrouter_first.tool_calls[0].id, "ok")], constants, notices=[notice])
    openrouter_state = json.loads(json.dumps(openrouter_session.dump_state()))
    openrouter_resumed = OpenRouterModel("openai/test", provider=openrouter_provider).resume_session(openrouter_state)
    await openrouter_resumed.continue_with_user_content("next", constants)
    assert {"role": "user", "content": _notice_text()} in openrouter_provider.payloads[2]["messages"]


async def test_resume_replays_preserved_user_notices() -> None:
    constants = _constants(ECHO_TOOLS)
    notice = _notice()

    anthropic_capture = FakeAnthropicProvider()
    anthropic_session = AnthropicMessagesModel("claude-test", provider=anthropic_capture).new_session()
    anthropic_first = await anthropic_session.start("hi", constants, notices=[notice])
    await anthropic_session.continue_with_tools([_tool_output(anthropic_first.tool_calls[0].id, "ok")], constants)
    anthropic_state = json.loads(json.dumps(anthropic_session.dump_state()))
    anthropic_replay = FakeAnthropicProvider()
    anthropic_resumed = AnthropicMessagesModel("claude-test", provider=anthropic_replay).resume_session(anthropic_state)
    await anthropic_resumed.continue_with_user_content("next", constants)
    assert anthropic_replay.payloads[0]["messages"][0]["content"] == f"hi\n\n{_notice_text()}"

    openai_capture = FakeClient()
    openai_session = OpenAIResponsesModel("gpt-test", provider=openai_capture).new_session()
    openai_first = await openai_session.start("hi", constants, notices=[notice])
    await openai_session.continue_with_tools([_tool_output(openai_first.tool_calls[0].id, "ok")], constants)
    openai_state = json.loads(json.dumps(openai_session.dump_state()))
    openai_replay = FakeClient()
    openai_resumed = OpenAIResponsesModel("gpt-test", provider=openai_replay).resume_session(openai_state)
    await openai_resumed.continue_with_user_content("next", constants)
    assert openai_replay.payloads[0]["input"][0]["content"][0]["text"] == f"hi\n\n{_notice_text()}"

    openrouter_capture = FakeOpenRouterProvider()
    openrouter_session = OpenRouterModel("openai/test", provider=openrouter_capture).new_session()
    openrouter_first = await openrouter_session.start("hi", constants, notices=[notice])
    await openrouter_session.continue_with_tools([_tool_output(openrouter_first.tool_calls[0].id, "ok")], constants)
    openrouter_state = json.loads(json.dumps(openrouter_session.dump_state()))
    openrouter_replay = FakeOpenRouterProvider()
    openrouter_resumed = OpenRouterModel("openai/test", provider=openrouter_replay).resume_session(openrouter_state)
    await openrouter_resumed.continue_with_user_content("next", constants)
    assert openrouter_replay.payloads[0]["messages"][1]["content"] == f"hi\n\n{_notice_text()}"
