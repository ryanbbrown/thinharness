from __future__ import annotations

import urllib.error

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

def test_model_sessions_advance_independently() -> None:
    provider = FakeAnthropicProvider()
    model = AnthropicMessagesModel("claude-test", provider=provider)
    tools = [{"type": "function", "name": "echo", "description": "Echo", "parameters": {"type": "object", "properties": {}}}]
    first = model.new_session()
    second = model.new_session()

    first_turn = first.start(prompt="first", instructions="system", tools=tools)
    second_turn = second.start(prompt="second", instructions="system", tools=tools)
    first.continue_with_tools([ToolOutput(first_turn.tool_calls[0].id, "first result")], tools=tools)
    second.continue_with_tools([ToolOutput(second_turn.tool_calls[0].id, "second result")], tools=tools)

    assert provider.payloads[2]["messages"][0] == {"role": "user", "content": "first"}
    assert provider.payloads[2]["messages"][-1]["content"][0]["content"] == "first result"
    assert provider.payloads[3]["messages"][0] == {"role": "user", "content": "second"}
    assert provider.payloads[3]["messages"][-1]["content"][0]["content"] == "second result"

def test_openai_previous_response_id_is_session_scoped() -> None:
    client = FakeClient()
    model = OpenAIResponsesModel("gpt-test", provider=client)
    tools = [{"type": "function", "name": "echo", "description": "Echo", "parameters": {"type": "object", "properties": {}}}]
    first = model.new_session()
    second = model.new_session()

    first.start(prompt="first", instructions="system", tools=tools, previous_response_id="existing")
    first.continue_with_tools([ToolOutput("call_1", "ok")], tools=tools)
    second.start(prompt="second", instructions="system", tools=tools)

    assert client.payloads[0]["previous_response_id"] == "existing"
    assert client.payloads[1]["previous_response_id"] == "resp_1"
    assert "previous_response_id" not in client.payloads[2]

def test_anthropic_provider_model_tool_loop(monkeypatch) -> None:
    calls = []

    def fake_post(url, payload, headers, timeout):
        calls.append((url, payload, headers, timeout))
        if len(calls) == 1:
            return {
                "content": [{"type": "tool_use", "id": "toolu_1", "name": "echo", "input": {"value": "hi"}}],
                "stop_reason": "tool_use",
            }
        assert payload["messages"][-1]["content"][0]["type"] == "tool_result"
        return {"content": [{"type": "text", "text": "done"}], "stop_reason": "end_turn"}

    monkeypatch.setattr("thinharness.providers._post_json", fake_post)
    provider = AnthropicProvider(api_key="key")
    model = AnthropicMessagesModel("claude-test", provider=provider)
    session = model.new_session()
    tools = [{"type": "function", "name": "echo", "description": "Echo", "parameters": {"type": "object", "properties": {}}}]

    first = session.start(prompt="hi", instructions="system", tools=tools)
    assert first.tool_calls[0].name == "echo"
    second = session.continue_with_tools([ToolOutput(first.tool_calls[0].id, "ok")], tools=tools)
    assert second.text == "done"
    assert calls[0][1]["tools"][0]["input_schema"]["type"] == "object"

def test_openrouter_provider_model_tool_loop(monkeypatch) -> None:
    calls = []

    def fake_post(url, payload, headers, timeout):
        calls.append((url, payload, headers, timeout))
        if len(calls) == 1:
            return {
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
            }
        assert payload["messages"][-1]["role"] == "tool"
        return {"choices": [{"message": {"role": "assistant", "content": "done"}}]}

    monkeypatch.setattr("thinharness.providers._post_json", fake_post)
    provider = OpenRouterProvider(api_key="key")
    model = OpenRouterModel("openai/test", provider=provider)
    session = model.new_session()
    tools = [{"type": "function", "name": "echo", "description": "Echo", "parameters": {"type": "object", "properties": {}}}]

    first = session.start(prompt="hi", instructions="system", tools=tools)
    assert first.tool_calls[0].id == "call_1"
    second = session.continue_with_tools([ToolOutput("call_1", "ok")], tools=tools)
    assert second.text == "done"
    assert calls[0][1]["tools"][0]["function"]["name"] == "echo"

def test_provider_wraps_transport_errors(monkeypatch) -> None:
    def fail_urlopen(request, timeout):
        raise urllib.error.URLError("offline")

    monkeypatch.setattr("urllib.request.urlopen", fail_urlopen)
    provider = OpenAIProvider(api_key="key", base_url="http://example.invalid")
    with pytest.raises(ProviderError, match="provider request failed"):
        provider.post_json("/responses", {})

def test_provider_wraps_invalid_json(monkeypatch) -> None:
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def read(self):
            return b"not json"

    def fake_urlopen(request, timeout):
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    provider = OpenAIProvider(api_key="key", base_url="http://example.invalid")
    with pytest.raises(ProviderError, match="invalid JSON"):
        provider.post_json("/responses", {})
