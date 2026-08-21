from __future__ import annotations

import json
import os

import httpx
import pytest
from fakes import FakeAnthropicProvider
from provider_test_helpers import ECHO_TOOLS
from provider_test_helpers import constants as _constants
from provider_test_helpers import notice as _notice
from provider_test_helpers import notice_text as _notice_text
from provider_test_helpers import tool_output as _tool_output

from thinharness import (
    AnthropicMessagesModel,
    AnthropicProvider,
    ModelNotice,
    RequestConstants,
)
from thinharness.providers import (
    ModelSettings,
    StructuredOutputRequest,
    TokenUsage,
)


async def test_anthropic_appends_notices_to_messages() -> None:
    provider = FakeAnthropicProvider()
    model = AnthropicMessagesModel("claude-test", provider=provider)
    session = model.new_session()
    constants = _constants()
    notice = _notice()

    first = await session.start("hi\n\n<hook_context>\npolicy\n</hook_context>", constants, notices=[notice])
    await session.continue_with_tools(
        [_tool_output(first.tool_calls[0].id, "ok"), _tool_output("toolu_2", "second")],
        constants,
        notices=[notice],
    )
    await session.continue_with_user_content("fix this", constants, notices=[notice])
    await session.continue_with_user_content("follow-up", constants, notices=[notice])

    assert provider.payloads[0]["messages"][0]["content"] == f"hi\n\n<hook_context>\npolicy\n</hook_context>\n\n{_notice_text()}"
    assert [block["type"] for block in provider.payloads[1]["messages"][-1]["content"][:-1]] == ["tool_result", "tool_result"]
    assert provider.payloads[1]["messages"][-1]["content"][-1] == {
        "type": "text",
        "text": _notice_text(),
    }
    assert provider.payloads[2]["messages"][-1]["content"] == f"fix this\n\n{_notice_text()}"
    assert provider.payloads[3]["messages"][-1]["content"] == f"follow-up\n\n{_notice_text()}"


async def test_anthropic_payload_translates_max_tokens_effort_and_overrides() -> None:
    constants = _constants(ECHO_TOOLS)

    default_provider = FakeAnthropicProvider()
    await AnthropicMessagesModel("claude-test", provider=default_provider).new_session().start("hi", constants)
    assert default_provider.payloads[0]["max_tokens"] == 16384

    settings_provider = FakeAnthropicProvider()
    settings_model = AnthropicMessagesModel(
        "claude-test",
        provider=settings_provider,
        settings=ModelSettings(max_tokens=4096, effort="high", temperature=0.1),
    )
    await settings_model.new_session().start("hi", constants)
    assert settings_provider.payloads[0]["max_tokens"] == 4096
    assert settings_provider.payloads[0]["temperature"] == 0.1
    assert settings_provider.payloads[0]["output_config"] == {"effort": "high"}
    assert settings_provider.payloads[0]["thinking"] == {"type": "adaptive"}

    ctor_provider = FakeAnthropicProvider()
    ctor_model = AnthropicMessagesModel(
        "claude-test",
        provider=ctor_provider,
        settings=ModelSettings(max_tokens=4096),
        max_tokens=2048,
    )
    await ctor_model.new_session().start("hi", constants)
    assert ctor_provider.payloads[0]["max_tokens"] == 2048

    extra_provider = FakeAnthropicProvider()
    extra_model = AnthropicMessagesModel(
        "claude-test",
        provider=extra_provider,
        settings=ModelSettings(
            max_tokens=4096,
            effort="high",
            extra_body={"max_tokens": 8192, "output_config": {"effort": "low"}, "thinking": {"type": "disabled"}},
        ),
    )
    await extra_model.new_session().start("hi", constants)
    assert extra_provider.payloads[0]["max_tokens"] == 8192
    assert extra_provider.payloads[0]["output_config"] == {"effort": "low"}
    assert extra_provider.payloads[0]["thinking"] == {"type": "disabled"}


async def test_anthropic_provider_model_tool_loop() -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        calls.append((str(request.url), payload, dict(request.headers)))
        if len(calls) == 1:
            return httpx.Response(200, json={
                "content": [{"type": "tool_use", "id": "toolu_1", "name": "echo", "input": {"value": "hi"}}],
                "stop_reason": "tool_use",
            })
        assert payload["messages"][-1]["content"][0]["type"] == "tool_result"
        return httpx.Response(200, json={"content": [{"type": "text", "text": "done"}], "stop_reason": "end_turn"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = AnthropicProvider(api_key="key", http_client=client)
        model = AnthropicMessagesModel("claude-test", provider=provider)
        session = model.new_session()
        constants = _constants(ECHO_TOOLS)

        first = await session.start("hi", constants)
        assert first.tool_calls[0].name == "echo"
        second = await session.continue_with_tools([_tool_output(first.tool_calls[0].id, "ok")], constants)
        assert second.text == "done"
    assert calls[0][1]["tools"][0]["input_schema"]["type"] == "object"


async def test_anthropic_projects_only_supported_request_metadata() -> None:
    provider = FakeAnthropicProvider()
    session = AnthropicMessagesModel("claude-test", provider=provider).new_session()
    constants = RequestConstants(
        instructions="system",
        tools=[],
        metadata={"user_id": "user-1", "conversation_id": "conv-1", "parent_call_id": "call-1"},
    )

    await session.start("hi", constants)

    assert provider.payloads[0]["metadata"] == {"user_id": "user-1"}


async def test_anthropic_requests_opt_into_prompt_caching() -> None:
    provider = FakeAnthropicProvider()
    session = AnthropicMessagesModel("claude-test", provider=provider).new_session()
    constants = _constants(ECHO_TOOLS)

    first = await session.start("hi", constants)
    await session.continue_with_tools([_tool_output(first.tool_calls[0].id, "ok")], constants)

    assert [payload["cache_control"] for payload in provider.payloads] == [{"type": "ephemeral"}] * 2

    override_provider = FakeAnthropicProvider()
    override_model = AnthropicMessagesModel(
        "claude-test",
        provider=override_provider,
        settings=ModelSettings(extra_body={"cache_control": {"type": "ephemeral", "ttl": "1h"}}),
    )
    await override_model.new_session().start("hi", constants)
    assert override_provider.payloads[0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}


async def test_anthropic_native_structured_output_merges_with_output_config_extra_body() -> None:
    provider = FakeAnthropicProvider()
    model = AnthropicMessagesModel(
        "claude-test",
        provider=provider,
        settings=ModelSettings(extra_body={"output_config": {"effort": "high"}}),
    )
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}}

    await model.new_session().start("hi", _constants(structured_output=StructuredOutputRequest(name="final_result", schema=schema)))

    assert provider.payloads[0]["output_config"] == {
        "effort": "high",
        "format": {"type": "json_schema", "schema": schema},
    }


async def test_anthropic_native_structured_output_merges_with_settings_effort() -> None:
    provider = FakeAnthropicProvider()
    model = AnthropicMessagesModel("claude-test", provider=provider, settings=ModelSettings(effort="high"))
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}}

    await model.new_session().start("hi", _constants(structured_output=StructuredOutputRequest(name="final_result", schema=schema)))

    assert provider.payloads[0]["output_config"] == {
        "effort": "high",
        "format": {"type": "json_schema", "schema": schema},
    }
    assert provider.payloads[0]["thinking"] == {"type": "adaptive"}


@pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY is not set")
async def test_anthropic_notice_payload_live() -> None:
    provider = AnthropicProvider()
    model = AnthropicMessagesModel(os.getenv("THINHARNESS_LIVE_ANTHROPIC_MODEL", "claude-haiku-4-5"), provider=provider)
    session = model.new_session()
    sentinel_notice = ModelNotice(
        kind="limit_warning",
        content="If you can read this notice, reply with NOTICE-SEEN only.",
        limit_kind="model_requests",
        remaining=1,
    )
    session.system = "You are concise. Follow the latest user instruction exactly."
    session.messages = [
        {"role": "user", "content": "Use echo."},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_live", "name": "echo", "input": {"value": "hi"}}]},
    ]
    tools = [{"type": "function", "name": "echo", "description": "Echo", "parameters": {"type": "object", "properties": {"value": {"type": "string"}}}}]
    try:
        turn = await session.continue_with_tools([_tool_output("toolu_live", "ok")], _constants(tools), notices=[sentinel_notice])
    finally:
        await provider.aclose()

    assert "NOTICE-SEEN" in turn.text


async def test_anthropic_session_normalizes_usage_finish_and_model() -> None:
    class Provider(AnthropicProvider):
        def __init__(self) -> None:
            super().__init__(api_key="key")

        async def create_message(self, payload):
            return {
                "model": "claude-live",
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 10, "output_tokens": 5},
                "content": [{"type": "text", "text": "done"}],
            }

    session = AnthropicMessagesModel("claude-test", provider=Provider()).new_session()
    turn = await session.start("hi", _constants())

    assert turn.usage == TokenUsage(input_tokens=10, output_tokens=5)
    assert turn.finish_reason == "end_turn"
    assert turn.response_model == "claude-live"
