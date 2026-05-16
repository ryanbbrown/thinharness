from __future__ import annotations

import json

import httpx
import pytest
from fakes import (
    FakeAnthropicProvider,
    FakeClient,
)

from thinharness import (
    AnthropicMessagesModel,
    AnthropicProvider,
    OpenAIProvider,
    OpenAIResponsesModel,
    OpenRouterModel,
    OpenRouterProvider,
    parse_model_ref,
)
from thinharness.providers import ProviderError, ToolOutput


def test_model_refs_require_provider_prefix() -> None:
    assert parse_model_ref("openai:gpt-4.1-mini") == ("openai", "gpt-4.1-mini")
    assert parse_model_ref("anthropic:claude-3-5-haiku-latest") == ("anthropic", "claude-3-5-haiku-latest")
    with pytest.raises(ValueError):
        parse_model_ref("gpt-4.1-mini")

async def test_model_sessions_advance_independently() -> None:
    provider = FakeAnthropicProvider()
    model = AnthropicMessagesModel("claude-test", provider=provider)
    tools = [{"type": "function", "name": "echo", "description": "Echo", "parameters": {"type": "object", "properties": {}}}]
    first = model.new_session()
    second = model.new_session()

    first_turn = await first.start(prompt="first", instructions="system", tools=tools)
    second_turn = await second.start(prompt="second", instructions="system", tools=tools)
    await first.continue_with_tools([ToolOutput(first_turn.tool_calls[0].id, "first result")], tools=tools)
    await second.continue_with_tools([ToolOutput(second_turn.tool_calls[0].id, "second result")], tools=tools)

    assert provider.payloads[2]["messages"][0] == {"role": "user", "content": "first"}
    assert provider.payloads[2]["messages"][-1]["content"][0]["content"] == "first result"
    assert provider.payloads[3]["messages"][0] == {"role": "user", "content": "second"}
    assert provider.payloads[3]["messages"][-1]["content"][0]["content"] == "second result"

async def test_openai_previous_response_id_is_session_scoped() -> None:
    client = FakeClient()
    model = OpenAIResponsesModel("gpt-test", provider=client)
    tools = [{"type": "function", "name": "echo", "description": "Echo", "parameters": {"type": "object", "properties": {}}}]
    first = model.new_session()
    second = model.new_session()

    await first.start(prompt="first", instructions="system", tools=tools, previous_response_id="existing")
    await first.continue_with_tools([ToolOutput("call_1", "ok")], tools=tools)
    await second.start(prompt="second", instructions="system", tools=tools)

    assert client.payloads[0]["previous_response_id"] == "existing"
    assert client.payloads[1]["previous_response_id"] == "resp_1"
    assert "previous_response_id" not in client.payloads[2]

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
        tools = [{"type": "function", "name": "echo", "description": "Echo", "parameters": {"type": "object", "properties": {}}}]

        first = await session.start(prompt="hi", instructions="system", tools=tools)
        assert first.tool_calls[0].name == "echo"
        second = await session.continue_with_tools([ToolOutput(first.tool_calls[0].id, "ok")], tools=tools)
        assert second.text == "done"
    assert calls[0][1]["tools"][0]["input_schema"]["type"] == "object"

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
        tools = [{"type": "function", "name": "echo", "description": "Echo", "parameters": {"type": "object", "properties": {}}}]

        first = await session.start(prompt="hi", instructions="system", tools=tools)
        assert first.tool_calls[0].id == "call_1"
        second = await session.continue_with_tools([ToolOutput("call_1", "ok")], tools=tools)
        assert second.text == "done"
    assert calls[0][1]["tools"][0]["function"]["name"] == "echo"

async def test_provider_wraps_transport_errors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAIProvider(api_key="key", base_url="http://example.invalid", http_client=client)
        with pytest.raises(ProviderError, match="provider request failed"):
            await provider.post_json("/responses", {})

async def test_provider_wraps_http_status_errors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text="rate limited", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAIProvider(api_key="key", base_url="http://example.invalid", http_client=client)
        with pytest.raises(ProviderError, match="provider error 429: rate limited"):
            await provider.post_json("/responses", {})

async def test_provider_wraps_invalid_json() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not json")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAIProvider(api_key="key", base_url="http://example.invalid", http_client=client)
        with pytest.raises(ProviderError, match="invalid JSON"):
            await provider.post_json("/responses", {})
