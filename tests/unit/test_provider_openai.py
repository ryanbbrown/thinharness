from __future__ import annotations

import os

import pytest
from fakes import FakeClient
from provider_test_helpers import ECHO_TOOLS
from provider_test_helpers import constants as _constants
from provider_test_helpers import notice as _notice
from provider_test_helpers import notice_text as _notice_text
from provider_test_helpers import tool_output as _tool_output

from thinharness import (
    OpenAIProvider,
    OpenAIResponsesModel,
)
from thinharness.providers import (
    ModelSettings,
    StructuredOutputRequest,
    TokenUsage,
)


async def test_openai_previous_response_id_is_session_scoped() -> None:
    client = FakeClient()
    model = OpenAIResponsesModel("gpt-test", provider=client)
    constants = _constants(ECHO_TOOLS)
    first = model.new_session()
    second = model.new_session()

    await first.start("first", constants, previous_response_id="existing")
    await first.continue_with_tools([_tool_output("call_1", "ok")], constants)
    await second.start("second", constants)

    assert client.payloads[0]["previous_response_id"] == "existing"
    assert client.payloads[1]["previous_response_id"] == "resp_1"
    assert client.payloads[1]["instructions"] == "system"
    assert "previous_response_id" not in client.payloads[2]


async def test_openai_appends_notices_to_string_and_tool_inputs() -> None:
    client = FakeClient()
    model = OpenAIResponsesModel("gpt-test", provider=client)
    session = model.new_session()
    constants = _constants()
    notice = _notice()

    first = await session.start("hi", constants, notices=[notice])
    await session.continue_with_tools(
        [_tool_output(first.tool_calls[0].id, "ok"), _tool_output("call_2", "second")],
        constants,
        notices=[notice],
    )
    await session.continue_with_user_content("fix this", constants, notices=[notice])
    resumed = model.resume_session({
        "kind": "transcript",
        "version": 4,
        "origin_provider": "openai",
        "origin_model": "gpt-test",
        "entries": [{"role": "user", "content": [{"type": "text", "text": "prior"}], "notice": False}],
    })
    await resumed.continue_with_user_content("follow-up", constants, notices=[notice])

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
    constants = _constants()

    await session.start("hi", constants)
    await session.continue_with_tools([_tool_output("call_1", "ok")], constants)

    assert client.payloads[0]["input"] == "hi"
    assert client.payloads[1]["input"] == [{
        "type": "function_call_output",
        "call_id": "call_1",
        "output": '{"ok": true, "content": "ok", "metadata": {}}',
    }]


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


def test_openai_payload_translates_max_tokens_and_effort() -> None:
    model = OpenAIResponsesModel(
        "gpt-test",
        provider=OpenAIProvider(api_key="key"),
        settings=ModelSettings(temperature=0.4, max_tokens=2000, effort="low"),
    )

    payload = model.build_payload(input_payload="hi", tools=[])

    assert payload["temperature"] == 0.4
    assert payload["max_output_tokens"] == 2000
    assert payload["reasoning"] == {"effort": "low"}


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY is not set")
async def test_openai_notice_payload_live() -> None:
    provider = OpenAIProvider()
    model = OpenAIResponsesModel(os.getenv("THINHARNESS_LIVE_OPENAI_MODEL", "gpt-4.1-mini"), provider=provider)
    session = model.new_session()
    try:
        turn = await session.start(
            "Reply with OK only.",
            _constants(instructions="You are concise."),
            notices=[_notice()],
        )
    finally:
        await provider.aclose()

    assert turn.text
    assert not turn.tool_calls


async def test_openai_session_normalizes_usage_and_model() -> None:
    class Provider(OpenAIProvider):
        def __init__(self) -> None:
            super().__init__(api_key="key")

        async def create_response(self, payload):
            return {
                "id": "resp_1",
                "model": "gpt-live",
                "output_text": "done",
                "usage": {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10},
            }

    session = OpenAIResponsesModel("gpt-test", provider=Provider()).new_session()
    turn = await session.start("hi", _constants())

    assert turn.usage == TokenUsage(input_tokens=7, output_tokens=3)
    assert turn.finish_reason is None
    assert turn.response_model == "gpt-live"
