from __future__ import annotations

import json
import os

import httpx
import pytest
from fakes import FakeOpenRouterProvider
from provider_test_helpers import ECHO_TOOLS
from provider_test_helpers import constants as _constants
from provider_test_helpers import notice as _notice
from provider_test_helpers import notice_text as _notice_text
from provider_test_helpers import tool_output as _tool_output

from thinharness import (
    OpenRouterModel,
    OpenRouterProvider,
)
from thinharness.providers import (
    ModelSettings,
    StructuredOutputRequest,
    TokenUsage,
)


async def test_openrouter_appends_notices_to_messages() -> None:
    provider = FakeOpenRouterProvider()
    model = OpenRouterModel("openai/test", provider=provider)
    session = model.new_session()
    constants = _constants()
    notice = _notice()

    first = await session.start("hi\n\n<hook_context>\npolicy\n</hook_context>", constants, notices=[notice])
    await session.continue_with_tools(
        [_tool_output(first.tool_calls[0].id, "ok"), _tool_output("call_2", "second")],
        constants,
        notices=[notice],
    )
    await session.continue_with_user_content("fix this", constants, notices=[notice])
    await session.continue_with_user_content("follow-up", constants, notices=[notice])

    assert provider.payloads[0]["messages"][1]["content"] == f"hi\n\n<hook_context>\npolicy\n</hook_context>\n\n{_notice_text()}"
    continuation_messages = provider.payloads[1]["messages"]
    assert [message["role"] for message in continuation_messages[-3:]] == ["tool", "tool", "user"]
    assert continuation_messages[-1]["content"] == _notice_text()
    assert provider.payloads[2]["messages"][-1]["content"] == f"fix this\n\n{_notice_text()}"
    assert provider.payloads[3]["messages"][-1]["content"] == f"follow-up\n\n{_notice_text()}"


async def test_openrouter_provider_model_tool_loop() -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        calls.append((str(request.url), payload, dict(request.headers)))
        if len(calls) == 1:
            return httpx.Response(200, json={
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "echo", "arguments": '{"value":"hi"}'},
                        }],
                    }
                }]
            })
        assert payload["messages"][-1]["role"] == "tool"
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "done"}}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenRouterProvider(api_key="key", http_client=client)
        model = OpenRouterModel("openai/test", provider=provider)
        session = model.new_session()
        constants = _constants(ECHO_TOOLS)

        first = await session.start("hi", constants)
        assert first.tool_calls[0].id == "call_1"
        second = await session.continue_with_tools([_tool_output("call_1", "ok")], constants)
        assert second.text == "done"
    assert calls[0][1]["tools"][0]["function"]["name"] == "echo"


async def test_openrouter_native_structured_output_overrides_extra_body_response_format() -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        calls.append(payload)
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": '{"ok":true}'}}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenRouterProvider(api_key="key", http_client=client)
        model = OpenRouterModel(
            "openai/test",
            provider=provider,
            settings=ModelSettings(extra_body={"response_format": {"type": "text"}}),
        )
        session = model.new_session()

        await session.start(
            "hi",
            _constants(structured_output=StructuredOutputRequest(name="final_result", schema={"type": "object", "properties": {}})),
        )

    assert calls[0]["response_format"]["type"] == "json_schema"
    assert calls[0]["response_format"]["json_schema"]["name"] == "final_result"


async def test_openrouter_payload_translates_max_tokens_and_effort() -> None:
    provider = FakeOpenRouterProvider()
    model = OpenRouterModel("openai/test", provider=provider, settings=ModelSettings(max_tokens=2000, effort="medium", temperature=0.2))

    await model.new_session().start("hi", _constants(ECHO_TOOLS))

    assert provider.payloads[0]["max_tokens"] == 2000
    assert provider.payloads[0]["reasoning"] == {"effort": "medium"}
    assert provider.payloads[0]["temperature"] == 0.2


@pytest.mark.skipif(not os.getenv("OPENROUTER_API_KEY"), reason="OPENROUTER_API_KEY is not set")
async def test_openrouter_notice_payload_live() -> None:
    provider = OpenRouterProvider()
    model = OpenRouterModel(os.getenv("THINHARNESS_LIVE_OPENROUTER_MODEL", "openai/gpt-4o-mini"), provider=provider)
    session = model.new_session()
    session.messages = [
        {"role": "system", "content": "You are concise."},
        {"role": "user", "content": "Use echo."},
        {
            "role": "assistant",
            "tool_calls": [{
                "id": "call_live",
                "type": "function",
                "function": {"name": "echo", "arguments": '{"value":"hi"}'},
            }],
        },
    ]
    tools = [{"type": "function", "name": "echo", "description": "Echo", "parameters": {"type": "object", "properties": {"value": {"type": "string"}}}}]
    try:
        turn = await session.continue_with_tools([_tool_output("call_live", "ok")], _constants(tools), notices=[_notice()])
    finally:
        await provider.aclose()

    assert turn.text or turn.tool_calls


async def test_openrouter_session_normalizes_chat_keys_and_prefers_top_level_model() -> None:
    class Provider(OpenRouterProvider):
        def __init__(self) -> None:
            super().__init__(api_key="key")

        async def create_chat_completion(self, payload):
            return {
                "model": "top-model",
                "usage": {"prompt_tokens": 4, "completion_tokens": 2},
                "choices": [{"model": "choice-model", "finish_reason": "stop", "message": {"role": "assistant", "content": "done"}}],
            }

    session = OpenRouterModel("openai/test", provider=Provider()).new_session()
    turn = await session.start("hi", _constants())

    assert turn.usage == TokenUsage(input_tokens=4, output_tokens=2)
    assert turn.finish_reason == "stop"
    assert turn.response_model == "top-model"
