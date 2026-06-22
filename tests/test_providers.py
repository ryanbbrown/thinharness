from __future__ import annotations

import json
import os

import httpx
import pytest
from fakes import (
    FakeAnthropicProvider,
    FakeClient,
    FakeOpenRouterProvider,
)

from thinharness import (
    AnthropicMessagesModel,
    AnthropicProvider,
    ModelNotice,
    OpenAIProvider,
    OpenAIResponsesModel,
    OpenRouterModel,
    OpenRouterProvider,
    parse_model_ref,
)
from thinharness.providers import ModelSettings, ProviderError, StructuredOutputRequest, ToolOutput, append_notices_to_text, render_model_notices


def _notice() -> ModelNotice:
    """Return a reusable test notice."""
    return ModelNotice(kind="limit_warning", content="Final request.", limit_kind="model_requests", remaining=1)


def _notice_text() -> str:
    """Return rendered text for the reusable test notice."""
    return '<harness_notice kind="limit_warning">\nFinal request.\n</harness_notice>'


def test_model_refs_require_provider_prefix() -> None:
    assert parse_model_ref("openai:gpt-4.1-mini") == ("openai", "gpt-4.1-mini")
    assert parse_model_ref("anthropic:claude-3-5-haiku-latest") == ("anthropic", "claude-3-5-haiku-latest")
    with pytest.raises(ValueError):
        parse_model_ref("gpt-4.1-mini")

def test_model_notice_rendering_is_deterministic() -> None:
    first = ModelNotice(kind="limit_warning", content="Final request.", limit_kind="model_requests", remaining=1)
    second = ModelNotice(kind="limit_warning", content="One tool call remains.", limit_kind="tool_calls", remaining=1)
    background = ModelNotice(kind="background_completion", content="Background task bg_1 completed.")

    assert render_model_notices(None) == ""
    assert append_notices_to_text("hi", None) == "hi"
    assert render_model_notices([first]) == '<harness_notice kind="limit_warning">\nFinal request.\n</harness_notice>'
    assert render_model_notices([first, second]) == (
        '<harness_notice kind="limit_warning">\nFinal request.\n</harness_notice>'
        "\n\n"
        '<harness_notice kind="limit_warning">\nOne tool call remains.\n</harness_notice>'
    )
    assert append_notices_to_text("hi", [first]) == 'hi\n\n<harness_notice kind="limit_warning">\nFinal request.\n</harness_notice>'
    assert render_model_notices([background]) == '<harness_notice kind="background_completion">\nBackground task bg_1 completed.\n</harness_notice>'

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
    await first.continue_with_tools([ToolOutput("call_1", "ok")], instructions="system", tools=tools)
    await second.start(prompt="second", instructions="system", tools=tools)

    assert client.payloads[0]["previous_response_id"] == "existing"
    assert client.payloads[1]["previous_response_id"] == "resp_1"
    assert client.payloads[1]["instructions"] == "system"
    assert "previous_response_id" not in client.payloads[2]

async def test_openai_appends_notices_to_string_and_tool_inputs() -> None:
    client = FakeClient()
    model = OpenAIResponsesModel("gpt-test", provider=client)
    session = model.new_session()
    notice = _notice()

    first = await session.start(prompt="hi", instructions="system", tools=[], notices=[notice])
    await session.continue_with_tools(
        [ToolOutput(first.tool_calls[0].id, "ok"), ToolOutput("call_2", "second")],
        instructions="system",
        tools=[],
        notices=[notice],
    )
    await session.continue_with_user_message("fix this", instructions="system", tools=[], notices=[notice])
    resumed = model.resume_session({
        "kind": "transcript",
        "version": 2,
        "origin_provider": "openai",
        "origin_model": "gpt-test",
        "entries": [{"role": "user", "content": "prior", "notice": False}],
    })
    await resumed.continue_with_user_prompt("follow-up", instructions="system", tools=[], notices=[notice])

    assert client.payloads[0]["input"].endswith("<harness_notice kind=\"limit_warning\">\nFinal request.\n</harness_notice>")
    assert [item["type"] for item in client.payloads[1]["input"][:-1]] == ["function_call_output", "function_call_output"]
    assert client.payloads[1]["input"][-1] == {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": _notice_text()}],
    }
    assert client.payloads[2]["input"] == f"fix this\n\n{_notice_text()}"
    assert client.payloads[1]["instructions"] == "system"
    assert client.payloads[2]["instructions"] == "system"
    assert client.payloads[3]["input"] == [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "prior"}],
        },
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": f"follow-up\n\n{_notice_text()}"}],
        },
    ]
    assert "previous_response_id" not in client.payloads[3]

async def test_openai_no_notice_payloads_are_unchanged() -> None:
    client = FakeClient()
    model = OpenAIResponsesModel("gpt-test", provider=client)
    session = model.new_session()

    await session.start(prompt="hi", instructions="system", tools=[])
    await session.continue_with_tools([ToolOutput("call_1", "ok")], tools=[])

    assert client.payloads[0]["input"] == "hi"
    assert client.payloads[1]["input"] == [{"type": "function_call_output", "call_id": "call_1", "output": "ok"}]

async def test_anthropic_appends_notices_to_messages() -> None:
    provider = FakeAnthropicProvider()
    model = AnthropicMessagesModel("claude-test", provider=provider)
    session = model.new_session()
    notice = _notice()

    first = await session.start(prompt="hi\n\n<hook_context>\npolicy\n</hook_context>", instructions="system", tools=[], notices=[notice])
    await session.continue_with_tools(
        [ToolOutput(first.tool_calls[0].id, "ok"), ToolOutput("toolu_2", "second")],
        tools=[],
        notices=[notice],
    )
    await session.continue_with_user_message("fix this", tools=[], notices=[notice])
    await session.continue_with_user_prompt("follow-up", instructions="system", tools=[], notices=[notice])

    assert provider.payloads[0]["messages"][0]["content"] == f"hi\n\n<hook_context>\npolicy\n</hook_context>\n\n{_notice_text()}"
    assert [block["type"] for block in provider.payloads[1]["messages"][-1]["content"][:-1]] == ["tool_result", "tool_result"]
    assert provider.payloads[1]["messages"][-1]["content"][-1] == {
        "type": "text",
        "text": _notice_text(),
    }
    assert provider.payloads[2]["messages"][-1]["content"] == f"fix this\n\n{_notice_text()}"
    assert provider.payloads[3]["messages"][-1]["content"] == f"follow-up\n\n{_notice_text()}"

async def test_openrouter_appends_notices_to_messages() -> None:
    provider = FakeOpenRouterProvider()
    model = OpenRouterModel("openai/test", provider=provider)
    session = model.new_session()
    notice = _notice()

    first = await session.start(prompt="hi\n\n<hook_context>\npolicy\n</hook_context>", instructions="system", tools=[], notices=[notice])
    await session.continue_with_tools(
        [ToolOutput(first.tool_calls[0].id, "ok"), ToolOutput("call_2", "second")],
        tools=[],
        notices=[notice],
    )
    await session.continue_with_user_message("fix this", tools=[], notices=[notice])
    await session.continue_with_user_prompt("follow-up", instructions="system", tools=[], notices=[notice])

    assert provider.payloads[0]["messages"][1]["content"] == f"hi\n\n<hook_context>\npolicy\n</hook_context>\n\n{_notice_text()}"
    continuation_messages = provider.payloads[1]["messages"]
    assert [message["role"] for message in continuation_messages[-3:]] == ["tool", "tool", "user"]
    assert continuation_messages[-1]["content"] == _notice_text()
    assert provider.payloads[2]["messages"][-1]["content"] == f"fix this\n\n{_notice_text()}"
    assert provider.payloads[3]["messages"][-1]["content"] == f"follow-up\n\n{_notice_text()}"

async def test_resume_replays_preserved_tool_notices() -> None:
    tools = [{"type": "function", "name": "echo", "description": "Echo", "parameters": {"type": "object", "properties": {}}}]
    notice = _notice()

    anthropic_provider = FakeAnthropicProvider()
    anthropic_session = AnthropicMessagesModel("claude-test", provider=anthropic_provider).new_session()
    anthropic_first = await anthropic_session.start(prompt="hi", instructions="system", tools=tools)
    await anthropic_session.continue_with_tools([ToolOutput(anthropic_first.tool_calls[0].id, "ok")], tools=tools, notices=[notice])
    anthropic_state = json.loads(json.dumps(anthropic_session.dump_state()))
    assert anthropic_state == json.loads(json.dumps(anthropic_state))
    anthropic_resumed = AnthropicMessagesModel("claude-test", provider=anthropic_provider).resume_session(anthropic_state)
    await anthropic_resumed.continue_with_user_prompt("next", instructions="system", tools=tools)
    assert anthropic_provider.payloads[2]["messages"][2]["content"][-1] == {"type": "text", "text": _notice_text()}

    openai_capture = FakeClient()
    openai_session = OpenAIResponsesModel("gpt-test", provider=openai_capture).new_session()
    openai_first = await openai_session.start(prompt="hi", instructions="system", tools=tools)
    await openai_session.continue_with_tools([ToolOutput(openai_first.tool_calls[0].id, "ok")], instructions="system", tools=tools, notices=[notice])
    openai_state = json.loads(json.dumps(openai_session.dump_state()))
    openai_replay = FakeClient()
    openai_resumed = OpenAIResponsesModel("gpt-test", provider=openai_replay).resume_session(openai_state)
    await openai_resumed.continue_with_user_prompt("next", instructions="system", tools=tools)
    assert {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": _notice_text()}],
    } in openai_replay.payloads[0]["input"]

    openrouter_provider = FakeOpenRouterProvider()
    openrouter_session = OpenRouterModel("openai/test", provider=openrouter_provider).new_session()
    openrouter_first = await openrouter_session.start(prompt="hi", instructions="system", tools=tools)
    await openrouter_session.continue_with_tools([ToolOutput(openrouter_first.tool_calls[0].id, "ok")], tools=tools, notices=[notice])
    openrouter_state = json.loads(json.dumps(openrouter_session.dump_state()))
    openrouter_resumed = OpenRouterModel("openai/test", provider=openrouter_provider).resume_session(openrouter_state)
    await openrouter_resumed.continue_with_user_prompt("next", instructions="system", tools=tools)
    assert {"role": "user", "content": _notice_text()} in openrouter_provider.payloads[2]["messages"]

async def test_resume_replays_preserved_user_notices() -> None:
    tools = [{"type": "function", "name": "echo", "description": "Echo", "parameters": {"type": "object", "properties": {}}}]
    notice = _notice()

    anthropic_capture = FakeAnthropicProvider()
    anthropic_session = AnthropicMessagesModel("claude-test", provider=anthropic_capture).new_session()
    anthropic_first = await anthropic_session.start(prompt="hi", instructions="system", tools=tools, notices=[notice])
    await anthropic_session.continue_with_tools([ToolOutput(anthropic_first.tool_calls[0].id, "ok")], tools=tools)
    anthropic_state = json.loads(json.dumps(anthropic_session.dump_state()))
    anthropic_replay = FakeAnthropicProvider()
    anthropic_resumed = AnthropicMessagesModel("claude-test", provider=anthropic_replay).resume_session(anthropic_state)
    await anthropic_resumed.continue_with_user_prompt("next", instructions="system", tools=tools)
    assert anthropic_replay.payloads[0]["messages"][0]["content"] == f"hi\n\n{_notice_text()}"

    openai_capture = FakeClient()
    openai_session = OpenAIResponsesModel("gpt-test", provider=openai_capture).new_session()
    openai_first = await openai_session.start(prompt="hi", instructions="system", tools=tools, notices=[notice])
    await openai_session.continue_with_tools([ToolOutput(openai_first.tool_calls[0].id, "ok")], instructions="system", tools=tools)
    openai_state = json.loads(json.dumps(openai_session.dump_state()))
    openai_replay = FakeClient()
    openai_resumed = OpenAIResponsesModel("gpt-test", provider=openai_replay).resume_session(openai_state)
    await openai_resumed.continue_with_user_prompt("next", instructions="system", tools=tools)
    assert openai_replay.payloads[0]["input"][0]["content"][0]["text"] == f"hi\n\n{_notice_text()}"

    openrouter_capture = FakeOpenRouterProvider()
    openrouter_session = OpenRouterModel("openai/test", provider=openrouter_capture).new_session()
    openrouter_first = await openrouter_session.start(prompt="hi", instructions="system", tools=tools, notices=[notice])
    await openrouter_session.continue_with_tools([ToolOutput(openrouter_first.tool_calls[0].id, "ok")], tools=tools)
    openrouter_state = json.loads(json.dumps(openrouter_session.dump_state()))
    openrouter_replay = FakeOpenRouterProvider()
    openrouter_resumed = OpenRouterModel("openai/test", provider=openrouter_replay).resume_session(openrouter_state)
    await openrouter_resumed.continue_with_user_prompt("next", instructions="system", tools=tools)
    assert openrouter_replay.payloads[0]["messages"][1]["content"] == f"hi\n\n{_notice_text()}"

def test_openai_native_structured_output_overrides_extra_body_text() -> None:
    model = OpenAIResponsesModel(
        "gpt-test",
        provider=OpenAIProvider(api_key="key"),
        settings=ModelSettings(extra_body={"text": {"format": {"type": "text"}}}),
    )

    payload = model.build_payload(
        input_payload="hi",
        tools=[],
        structured_output=StructuredOutputRequest(name="final_result", schema={"type": "object", "properties": {}}),
    )

    assert payload["text"]["format"]["type"] == "json_schema"
    assert payload["text"]["format"]["name"] == "final_result"

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
            prompt="hi",
            instructions="system",
            tools=[],
            structured_output=StructuredOutputRequest(name="final_result", schema={"type": "object", "properties": {}}),
        )

    assert calls[0]["response_format"]["type"] == "json_schema"
    assert calls[0]["response_format"]["json_schema"]["name"] == "final_result"

@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY is not set")
async def test_openai_notice_payload_live() -> None:
    provider = OpenAIProvider()
    model = OpenAIResponsesModel(os.getenv("THINHARNESS_LIVE_OPENAI_MODEL", "gpt-4.1-mini"), provider=provider)
    session = model.new_session()
    try:
        turn = await session.start(
            prompt="Reply with OK only.",
            instructions="You are concise.",
            tools=[],
            notices=[_notice()],
        )
    finally:
        await provider.aclose()

    assert turn.text
    assert not turn.tool_calls

@pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY is not set")
async def test_anthropic_notice_payload_live() -> None:
    provider = AnthropicProvider()
    model = AnthropicMessagesModel(os.getenv("THINHARNESS_LIVE_ANTHROPIC_MODEL", "claude-3-5-haiku-latest"), provider=provider)
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
        turn = await session.continue_with_tools([ToolOutput("toolu_live", "ok")], tools=tools, notices=[sentinel_notice])
    finally:
        await provider.aclose()

    assert "NOTICE-SEEN" in turn.text

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
        turn = await session.continue_with_tools([ToolOutput("call_live", "ok")], tools=tools, notices=[_notice()])
    finally:
        await provider.aclose()

    assert turn.text or turn.tool_calls

async def test_provider_wraps_transport_errors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAIProvider(api_key="key", base_url="http://example.invalid", http_client=client)
        with pytest.raises(ProviderError, match="provider request failed") as exc_info:
            await provider.post_json("/responses", {})
    assert exc_info.value.status_code is None

async def test_provider_wraps_http_status_errors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text="rate limited", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAIProvider(api_key="key", base_url="http://example.invalid", http_client=client)
        with pytest.raises(ProviderError, match="provider error 429: rate limited") as exc_info:
            await provider.post_json("/responses", {})
    assert exc_info.value.status_code == 429

async def test_provider_wraps_invalid_json() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not json")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAIProvider(api_key="key", base_url="http://example.invalid", http_client=client)
        with pytest.raises(ProviderError, match="invalid JSON") as exc_info:
            await provider.post_json("/responses", {})
    assert exc_info.value.status_code is None
