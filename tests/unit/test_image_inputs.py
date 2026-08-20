from __future__ import annotations

import base64
import copy
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from fakes import FakeTracer
from pydantic import BaseModel

from thinharness import (
    AnthropicMessagesModel,
    ApprovalDecision,
    BashPlugin,
    FilesystemPlugin,
    Harness,
    HarnessConfig,
    HarnessError,
    Hook,
    ImageBlock,
    OpenAIResponsesModel,
    OpenRouterModel,
    RunFailedEvent,
    RunStartedEvent,
    TextBlock,
    ToolCallCompletedEvent,
    ToolResult,
    ToolSpec,
    TracingOptions,
    call_tool,
)
from thinharness.content import content_from_json, content_to_json, normalize_content
from thinharness.providers import ProviderError

PNG = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + b"\x00" * 17
PNG_B64 = base64.b64encode(PNG).decode("ascii")
PNG_URL = f"data:image/png;base64,{PNG_B64}"


class _Provider:
    def __init__(self, name: str, responses: list[dict[str, Any]]) -> None:
        self.name = name
        self.api_key = "test"
        self.payloads: list[dict[str, Any]] = []
        self.responses = list(responses)

    async def aclose(self) -> None:
        return None

    async def create_response(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.payloads.append(copy.deepcopy(payload))
        return self.responses.pop(0)

    async def create_message(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.payloads.append(copy.deepcopy(payload))
        return self.responses.pop(0)

    async def create_chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.payloads.append(copy.deepcopy(payload))
        return self.responses.pop(0)


def _config(tmp_path: Path) -> HarnessConfig:
    return HarnessConfig(root=tmp_path, system_prompt="sys", local_tracing=False)


@pytest.mark.parametrize(
    "bad",
    [
        "",
        [],
        [TextBlock("")],
        [ImageBlock(b"", "image/png")],
        [ImageBlock(b"x", "image/svg+xml")],
        ["text"],
    ],
)
def test_content_validation_rejects_invalid_public_values(bad: Any) -> None:
    with pytest.raises((TypeError, ValueError)):
        normalize_content(bad)


def test_content_contract_detaches_round_trips_and_redacts() -> None:
    caller = [TextBlock("before"), ImageBlock(PNG, "image/png"), TextBlock("after")]
    normalized = normalize_content(caller)
    caller.clear()

    assert normalized == (TextBlock("before"), ImageBlock(PNG, "image/png"), TextBlock("after"))
    assert content_from_json(content_to_json(normalized)) == normalized
    assert PNG_B64 not in repr(normalized[1])
    assert "size_bytes=33" in repr(normalized[1])


async def test_public_initial_image_payloads_are_literal(tmp_path: Path) -> None:
    prompt = [TextBlock("before"), ImageBlock(PNG, "image/png"), TextBlock("after")]

    openai_provider = _Provider("OpenAI", [{"id": "r1", "output_text": "done"}])
    await Harness(_config(tmp_path), model=OpenAIResponsesModel("gpt-test", provider=openai_provider)).run(prompt)  # type: ignore[arg-type]
    assert openai_provider.payloads == [{
        "model": "gpt-test",
        "input": [{
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": "before"},
                {"type": "input_image", "image_url": PNG_URL},
                {"type": "input_text", "text": "after"},
            ],
        }],
        "tools": [],
        "instructions": "sys",
    }]

    anthropic_provider = _Provider("Anthropic", [{"content": [{"type": "text", "text": "done"}]}])
    await Harness(_config(tmp_path), model=AnthropicMessagesModel("claude-test", provider=anthropic_provider)).run(prompt)  # type: ignore[arg-type]
    assert anthropic_provider.payloads == [{
        "model": "claude-test",
        "max_tokens": 16384,
        "system": "sys",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "before"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": PNG_B64}},
                {"type": "text", "text": "after"},
            ],
        }],
        "tools": [],
        "cache_control": {"type": "ephemeral"},
    }]

    openrouter_provider = _Provider("OpenRouter", [{"choices": [{"message": {"role": "assistant", "content": "done"}}]}])
    await Harness(_config(tmp_path), model=OpenRouterModel("vendor/model", provider=openrouter_provider)).run(prompt)  # type: ignore[arg-type]
    assert openrouter_provider.payloads == [{
        "model": "vendor/model",
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": [
                {"type": "text", "text": "before"},
                {"type": "image_url", "image_url": {"url": PNG_URL}},
                {"type": "text", "text": "after"},
            ]},
        ],
        "tools": [],
    }]


async def test_openai_image_tool_result_payload_is_literal(tmp_path: Path) -> None:
    provider = _Provider("OpenAI", [
        {"id": "r1", "output": [{"type": "function_call", "call_id": "call_1", "name": "look", "arguments": "{}"}]},
        {"id": "r2", "output_text": "done"},
    ])
    tool = ToolSpec(
        "look",
        "Return an image.",
        {"type": "object", "properties": {}},
        lambda _args: ToolResult(True, [TextBlock("caption"), ImageBlock(PNG, "image/png")], {"source": "fixture"}),
    )

    result = await Harness(_config(tmp_path), model=OpenAIResponsesModel("gpt-test", provider=provider), tools=[tool]).run("go")  # type: ignore[arg-type]

    assert provider.payloads[1]["input"] == [{
        "type": "function_call_output",
        "call_id": "call_1",
        "output": [
            {"type": "input_text", "text": '{"ok":true,"metadata":{"source":"fixture"}}'},
            {"type": "input_text", "text": "caption"},
            {"type": "input_image", "image_url": PNG_URL},
        ],
    }]
    assert result.resume_state["version"] == 4
    tool_entry = next(entry for entry in result.resume_state["entries"] if entry["role"] == "tool")
    assert tool_entry == {
        "role": "tool",
        "call_id": "call_1",
        "ok": True,
        "content": [
            {"type": "text", "text": "caption"},
            {"type": "image", "media_type": "image/png", "data": PNG_B64},
        ],
        "metadata": {"source": "fixture"},
    }


async def test_anthropic_image_tool_result_payload_is_literal(tmp_path: Path) -> None:
    provider = _Provider("Anthropic", [
        {"content": [{"type": "tool_use", "id": "call_1", "name": "look", "input": {}}]},
        {"content": [{"type": "text", "text": "done"}]},
    ])
    tool = ToolSpec(
        "look",
        "Return an image.",
        {"type": "object", "properties": {}},
        lambda _args: ToolResult(False, [TextBlock("try another"), ImageBlock(PNG, "image/png")], {"retry": True, "error_type": "Fixture"}),
        max_retries=1,
    )

    await Harness(_config(tmp_path), model=AnthropicMessagesModel("claude-test", provider=provider), tools=[tool]).run("go")  # type: ignore[arg-type]

    assert provider.payloads[1]["messages"][-1]["content"] == [{
        "type": "tool_result",
        "tool_use_id": "call_1",
        "content": [
            {"type": "text", "text": '{"ok":false,"metadata":{"retry":true,"error_type":"Fixture"}}'},
            {"type": "text", "text": "try another"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": PNG_B64}},
        ],
    }]


async def test_openrouter_image_tool_projection_is_labelled_and_literal(tmp_path: Path) -> None:
    provider = _Provider("OpenRouter", [
        {"choices": [{"message": {"role": "assistant", "content": "", "tool_calls": [{
            "id": "call_1",
            "type": "function",
            "function": {"name": "look", "arguments": "{}"},
        }]}}]},
        {"choices": [{"message": {"role": "assistant", "content": "done"}}]},
    ])
    tool = ToolSpec(
        "look",
        "Return an image.",
        {"type": "object", "properties": {}},
        lambda _args: ToolResult(
            False,
            [TextBlock("caption"), ImageBlock(PNG, "image/png")],
            {"retry": True, "error_type": "Fixture"},
        ),
        max_retries=1,
    )

    await Harness(_config(tmp_path), model=OpenRouterModel("vendor/model", provider=provider), tools=[tool]).run("go")  # type: ignore[arg-type]

    assert provider.payloads[1]["messages"][-2:] == [
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": json.dumps({
                "ok": False,
                "content": [
                    {"type": "text", "text": "caption"},
                    {"type": "image", "media_type": "image/png", "size_bytes": 33, "block_index": 1},
                ],
                "metadata": {"retry": True, "error_type": "Fixture"},
            }, ensure_ascii=False),
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "[tool image call_id=call_1 block=1]"},
                {"type": "image_url", "image_url": {"url": PNG_URL}},
            ],
        },
    ]


def test_filesystem_read_image_is_opt_in_and_bounded(tmp_path: Path) -> None:
    image = tmp_path / "sample.png"
    image.write_bytes(PNG)
    default = FilesystemPlugin().bind(type("Context", (), {"root": tmp_path})())  # type: ignore[arg-type]
    assert "read_image" not in [tool.name for tool in default.static.tools]

    binding = FilesystemPlugin(tools=["read_image"], max_image_bytes=len(PNG)).bind(type("Context", (), {"root": tmp_path})())  # type: ignore[arg-type]
    result = binding.static.tools[0].handler({"path": "sample.png"})
    assert isinstance(result, ToolResult)
    assert result.content == (
        TextBlock('{"path":"sample.png","media_type":"image/png","size_bytes":33}'),
        ImageBlock(PNG, "image/png"),
    )

    image.write_bytes(PNG + b"x")
    oversized = binding.static.tools[0].handler({"path": "sample.png"})
    assert isinstance(oversized, ToolResult)
    assert oversized.ok is False
    assert "over max_image_bytes=33" in oversized.message_text()


def _resume_state(
    origin: str,
    *,
    include_tool_image: bool = True,
    include_user_image: bool = True,
) -> dict[str, Any]:
    user_content: list[dict[str, Any]] = [{"type": "text", "text": "prior user"}]
    if include_user_image:
        user_content.append({"type": "image", "media_type": "image/png", "data": PNG_B64})
    entries: list[dict[str, Any]] = [
        {"role": "user", "content": user_content, "notice": False},
    ]
    if include_tool_image:
        entries.extend([
            {
                "role": "assistant",
                "text": "",
                "tool_calls": [{"id": "call_foreign", "name": "look", "arguments": "{}"}],
                "reasoning": [],
            },
            {
                "role": "tool",
                "call_id": "call_foreign",
                "ok": True,
                "content": [
                    {"type": "text", "text": "prior tool"},
                    {"type": "image", "media_type": "image/png", "data": PNG_B64},
                ],
                "metadata": {"source": "resume"},
            },
        ])
    entries.append({"role": "assistant", "text": "prior done", "tool_calls": [], "reasoning": []})
    return {
        "kind": "transcript",
        "version": 4,
        "origin_provider": origin,
        "origin_model": "source-model",
        "entries": entries,
    }


def _model_for(provider_name: str, provider: _Provider):
    if provider_name == "openai":
        return OpenAIResponsesModel("target", provider=provider)  # type: ignore[arg-type]
    if provider_name == "anthropic":
        return AnthropicMessagesModel("target", provider=provider)  # type: ignore[arg-type]
    return OpenRouterModel("target", provider=provider)  # type: ignore[arg-type]


def _final_response(provider_name: str) -> dict[str, Any]:
    if provider_name == "openai":
        return {"id": "done", "output_text": "done"}
    if provider_name == "anthropic":
        return {"content": [{"type": "text", "text": "done"}]}
    return {"choices": [{"message": {"role": "assistant", "content": "done"}}]}


@pytest.mark.parametrize("origin", ["openai", "anthropic", "openrouter"])
@pytest.mark.parametrize("target", ["openai", "anthropic", "openrouter"])
async def test_every_provider_pair_replays_user_and_tool_images(tmp_path: Path, origin: str, target: str) -> None:
    provider = _Provider(target, [_final_response(target)])
    result = await Harness(_config(tmp_path), model=_model_for(target, provider)).run(
        "follow-up",
        resume_from=_resume_state(origin),
    )

    assert result.text == "done"
    rendered = json.dumps(provider.payloads[0], ensure_ascii=False)
    assert PNG_B64 in rendered
    if target == "openrouter":
        assert "[tool image call_id=call_foreign block=1]" in rendered


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda state: state["entries"][0]["content"][1].update(data="***"), "invalid base64"),
        (lambda state: state["entries"][0]["content"][1].update(media_type="image/svg+xml"), "unsupported media type"),
        (lambda state: state["entries"][0]["content"][1].update(extra=True), "wrong shape"),
        (lambda state: state.update(version=3), "version 3"),
    ],
)
def test_malformed_v4_image_state_fails_clearly(tmp_path: Path, mutate, message: str) -> None:
    state = _resume_state("openai", include_tool_image=False)
    mutate(state)
    provider = _Provider("OpenAI", [])
    harness = Harness(_config(tmp_path), model=OpenAIResponsesModel("target", provider=provider))  # type: ignore[arg-type]

    with pytest.raises(Exception, match=message):
        harness.run_sync("follow-up", resume_from=state)
    assert provider.payloads == []


@pytest.mark.parametrize(
    "value, expected",
    [
        ("", ""),
        ([], "[]"),
        (None, "null"),
        ([1, "two"], '[\n  1,\n  "two"\n]'),
        ({"answer": 42}, '{\n  "answer": 42\n}'),
    ],
)
def test_sync_tool_result_normalization_preserves_json_values(value: Any, expected: str) -> None:
    spec = ToolSpec("value", "Return value.", {"type": "object", "properties": {}}, lambda _args: value)
    result = ToolResult.from_json(call_tool(spec, {}), strict=True)

    assert result.ok is True
    assert result.content == expected


@pytest.mark.parametrize("value", [Path("not-json"), datetime(2026, 1, 1)])
def test_nonserializable_sync_tool_results_become_failed_envelopes(value: Any) -> None:
    spec = ToolSpec("bad", "Return invalid value.", {"type": "object", "properties": {}}, lambda _args: value)
    result = ToolResult.from_json(call_tool(spec, {}), strict=True)

    assert result.ok is False
    assert result.metadata["error_type"] == "InvalidToolResult"


async def test_async_invalid_metadata_and_empty_content_stay_at_tool_boundary(tmp_path: Path) -> None:
    async def invalid(_args):
        return ToolResult(True, "", {"bad": Path("not-json")})

    provider = _Provider("OpenAI", [
        {"id": "r1", "output": [{"type": "function_call", "call_id": "call_1", "name": "bad", "arguments": "{}"}]},
        {"id": "r2", "output_text": "done"},
    ])
    result = await Harness(
        _config(tmp_path),
        model=OpenAIResponsesModel("target", provider=provider),  # type: ignore[arg-type]
        tools=[ToolSpec("bad", "Bad metadata.", {"type": "object", "properties": {}}, invalid)],
    ).run("go")

    envelope = result.tool_call_records[0]["result"]
    assert envelope["ok"] is False
    assert envelope["metadata"]["error_type"] == "InvalidToolResult"


def test_image_and_text_tool_records_have_stable_keys(tmp_path: Path) -> None:
    provider = _Provider("OpenAI", [
        {"id": "r1", "output": [
            {"type": "function_call", "call_id": "text", "name": "text", "arguments": "{}"},
            {"type": "function_call", "call_id": "image", "name": "image", "arguments": "{}"},
        ]},
        {"id": "r2", "output_text": "done"},
    ])
    harness = Harness(
        _config(tmp_path),
        model=OpenAIResponsesModel("target", provider=provider),  # type: ignore[arg-type]
        tools=[
            ToolSpec("text", "Text.", {"type": "object", "properties": {}}, lambda _args: "ok"),
            ToolSpec("image", "Image.", {"type": "object", "properties": {}}, lambda _args: [ImageBlock(PNG, "image/png")]),
        ],
    )
    result = harness.run_sync("go")
    text_record, image_record = result.tool_call_records

    assert set(text_record) == set(image_record) == {"call", "result", "output"}
    assert PNG_B64 not in image_record["output"]
    assert "size_bytes" in image_record["output"]


async def test_prompt_hooks_replace_content_and_append_context_after_images(tmp_path: Path) -> None:
    def start(ctx):
        ctx.prompt = "start replacement"

    def submit(ctx):
        assert ctx.prompt == (TextBlock("start replacement"),)
        ctx.prompt = (TextBlock("hook text"), ImageBlock(PNG, "image/png"))
        ctx.additional_context.append("policy")

    provider = _Provider("OpenAI", [{"id": "done", "output_text": "done"}])
    await Harness(
        _config(tmp_path),
        model=OpenAIResponsesModel("target", provider=provider),  # type: ignore[arg-type]
        hooks=[Hook("run_start", start), Hook("user_prompt_submit", submit)],
    ).run("caller")

    content = provider.payloads[0]["input"][0]["content"]
    assert content == [
        {"type": "input_text", "text": "hook text"},
        {"type": "input_image", "image_url": PNG_URL},
        {"type": "input_text", "text": "<hook_context>\npolicy\n</hook_context>"},
    ]


async def test_image_events_and_tool_trace_are_redacted(tmp_path: Path) -> None:
    tracer = FakeTracer()
    provider = _Provider("OpenAI", [
        {"id": "r1", "output": [{"type": "function_call", "call_id": "image", "name": "image", "arguments": "{}"}]},
        {"id": "r2", "output_text": "done"},
    ])
    harness = Harness(
        _config(tmp_path),
        model=OpenAIResponsesModel("target", provider=provider),  # type: ignore[arg-type]
        tools=[ToolSpec("image", "Image.", {"type": "object", "properties": {}}, lambda _args: [ImageBlock(PNG, "image/png")])],
        tracing=[TracingOptions(tracer=tracer, capture_messages=True, capture_tool_results=True)],
    )
    events = []
    async for event in harness.stream((TextBlock("go"), ImageBlock(PNG, "image/png"))):
        events.append(event)

    started = next(event for event in events if isinstance(event, RunStartedEvent))
    completed = next(event for event in events if isinstance(event, ToolCallCompletedEvent))
    assert PNG_B64 not in started.prompt
    assert PNG_B64 not in completed.output
    trace_text = json.dumps([span.attributes for span in tracer.spans], ensure_ascii=False)
    assert PNG_B64 not in trace_text
    assert PNG_URL not in trace_text


class _FailingProvider(_Provider):
    def __init__(self, name: str = "OpenAI", *, leak: bool = True) -> None:
        super().__init__(name, [])
        self.leak = leak

    async def create_response(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.payloads.append(copy.deepcopy(payload))
        message = f"provider echoed {PNG_B64} and {PNG_URL}" if self.leak else "plain provider failure"
        raise ProviderError(message)


@pytest.mark.parametrize(
    "state",
    [
        _resume_state("openai", include_tool_image=False),
        _resume_state("openai", include_user_image=False),
    ],
)
async def test_resumed_provider_failures_redact_user_and_tool_images(tmp_path: Path, state: dict[str, Any]) -> None:
    tracer = FakeTracer()
    provider = _FailingProvider()
    harness = Harness(
        _config(tmp_path),
        model=OpenAIResponsesModel("target", provider=provider),
        tracing=[TracingOptions(tracer=tracer, capture_messages=True)],
    )
    events = []
    with pytest.raises(HarnessError):
        async for event in harness.stream("follow-up", resume_from=copy.deepcopy(state)):
            events.append(event)

    failed = next(event for event in events if isinstance(event, RunFailedEvent))
    assert PNG_B64 not in failed.message
    assert PNG_URL not in failed.message
    recorded = " ".join(str(exc) for span in tracer.spans for exc in span.exceptions)
    attributes = json.dumps([span.attributes for span in tracer.spans], ensure_ascii=False)
    assert PNG_B64 not in recorded + attributes
    assert PNG_URL not in recorded + attributes


async def test_text_only_provider_failure_preserves_recorded_exception_type(tmp_path: Path) -> None:
    tracer = FakeTracer()
    provider = _FailingProvider(leak=False)
    harness = Harness(
        _config(tmp_path),
        model=OpenAIResponsesModel("target", provider=provider),
        tracing=[TracingOptions(tracer=tracer)],
    )

    with pytest.raises(HarnessError):
        await harness.run("go")
    model_span = next(span for span in tracer.spans if span.name.startswith("chat "))
    assert any(isinstance(exc, ProviderError) for exc in model_span.exceptions)


class _ApprovalFailingProvider(_Provider):
    def __init__(self) -> None:
        super().__init__("OpenAI", [])
        self.calls = 0

    async def create_response(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.payloads.append(copy.deepcopy(payload))
        self.calls += 1
        if self.calls == 1:
            return {
                "id": "pause",
                "output": [{"type": "function_call", "call_id": "approve", "name": "approved", "arguments": "{}"}],
            }
        raise ProviderError(f"approval provider echoed {PNG_B64} and {PNG_URL}")


async def test_image_approval_resume_with_capture_redacts_failure(tmp_path: Path) -> None:
    tracer = FakeTracer()
    provider = _ApprovalFailingProvider()
    tool = ToolSpec(
        "approved",
        "Approval tool.",
        {"type": "object", "properties": {}},
        lambda _args: "approved",
        requires_approval=True,
    )
    harness = Harness(
        _config(tmp_path),
        model=OpenAIResponsesModel("target", provider=provider),  # type: ignore[arg-type]
        tools=[tool],
        tracing=[TracingOptions(tracer=tracer, capture_messages=True)],
    )
    paused = await harness.run((TextBlock("approve"), ImageBlock(PNG, "image/png")))
    assert paused.stop_reason == "approval_required"

    old_state = copy.deepcopy(paused.resume_state)
    old_state["provider_state"]["version"] = 3
    with pytest.raises(Exception, match="approval state provider_state version 3 is not supported"):
        await Harness(
            _config(tmp_path),
            model=OpenAIResponsesModel("target", provider=_Provider("OpenAI", [])),  # type: ignore[arg-type]
            tools=[tool],
        ).resume_approvals(old_state, [ApprovalDecision("approve", True)])

    events = []
    with pytest.raises(HarnessError):
        async for event in harness.stream_approvals(
            paused.resume_state,
            [ApprovalDecision("approve", True)],
        ):
            events.append(event)
    failed = next(event for event in events if isinstance(event, RunFailedEvent))
    trace_text = json.dumps([span.attributes for span in tracer.spans], ensure_ascii=False)
    exceptions = " ".join(str(exc) for span in tracer.spans for exc in span.exceptions)
    assert PNG_B64 not in failed.message + trace_text + exceptions
    assert PNG_URL not in failed.message + trace_text + exceptions


@pytest.mark.parametrize("field", ["output", "envelope"])
async def test_after_tool_hook_mutates_image_through_both_fields(tmp_path: Path, field: str) -> None:
    def mutate(ctx):
        replacement = ToolResult(True, (TextBlock("mutated"), ImageBlock(PNG + b"m", "image/png")), {"hook": field})
        if field == "output":
            ctx.output = replacement.to_json()
        else:
            ctx.envelope = replacement

    provider = _Provider("OpenAI", [
        {"id": "r1", "output": [{"type": "function_call", "call_id": "image", "name": "image", "arguments": "{}"}]},
        {"id": "r2", "output_text": "done"},
    ])
    await Harness(
        _config(tmp_path),
        model=OpenAIResponsesModel("target", provider=provider),  # type: ignore[arg-type]
        tools=[ToolSpec("image", "Image.", {"type": "object", "properties": {}}, lambda _args: [ImageBlock(PNG, "image/png")])],
        hooks=[Hook("after_tool_call", mutate)],
    ).run("go")

    output = provider.payloads[1]["input"][0]["output"]
    assert output[0] == {"type": "input_text", "text": f'{{"ok":true,"metadata":{{"hook":"{field}"}}}}'}
    assert output[1] == {"type": "input_text", "text": "mutated"}
    assert output[2]["image_url"].endswith(base64.b64encode(PNG + b"m").decode("ascii"))


async def test_parallel_async_image_tools_preserve_model_order(tmp_path: Path) -> None:
    async def first(_args):
        return [TextBlock("first"), ImageBlock(PNG + b"1", "image/png")]

    async def second(_args):
        return [TextBlock("second"), ImageBlock(PNG + b"2", "image/png")]

    provider = _Provider("OpenRouter", [
        {"choices": [{"message": {"role": "assistant", "content": "", "tool_calls": [
            {"id": "one", "type": "function", "function": {"name": "first", "arguments": "{}"}},
            {"id": "two", "type": "function", "function": {"name": "second", "arguments": "{}"}},
        ]}}]},
        {"choices": [{"message": {"role": "assistant", "content": "done"}}]},
    ])
    await Harness(
        _config(tmp_path),
        model=OpenRouterModel("target", provider=provider),  # type: ignore[arg-type]
        tools=[
            ToolSpec("first", "First.", {"type": "object", "properties": {}}, first),
            ToolSpec("second", "Second.", {"type": "object", "properties": {}}, second),
        ],
    ).run("go")

    labels = [
        part["text"]
        for part in provider.payloads[1]["messages"][-1]["content"]
        if part["type"] == "text"
    ]
    assert labels == ["[tool image call_id=one block=1]", "[tool image call_id=two block=1]"]


async def test_notice_follows_image_content(tmp_path: Path) -> None:
    provider = _Provider("OpenRouter", [_final_response("openrouter")])
    await Harness(
        HarnessConfig(root=tmp_path, system_prompt="sys", local_tracing=False, max_model_requests=1),
        model=OpenRouterModel("target", provider=provider),  # type: ignore[arg-type]
    ).run((TextBlock("go"), ImageBlock(PNG, "image/png")))

    parts = provider.payloads[0]["messages"][-1]["content"]
    assert parts[-2]["type"] == "image_url"
    assert parts[-1]["type"] == "text"
    assert parts[-1]["text"].startswith('<harness_notice kind="limit_warning">')


@pytest.mark.parametrize(
    ("media_type", "data"),
    [
        ("image/png", PNG),
        ("image/jpeg", b"\xff\xd8\xff\xe0body\xff\xd9trailing"),
        ("image/gif", b"GIF89a" + b"\x00" * 7),
        ("image/webp", b"RIFF" + (12).to_bytes(4, "little") + b"WEBPVP8 " + b"\x00" * 4 + b"trailing"),
    ],
)
def test_read_image_accepts_supported_signatures_and_trailing_data(
    tmp_path: Path,
    media_type: str,
    data: bytes,
) -> None:
    path = tmp_path / "image.bin"
    path.write_bytes(data)
    binding = FilesystemPlugin(tools=["read_image"], max_image_bytes=len(data)).bind(
        type("Context", (), {"root": tmp_path})(),  # type: ignore[arg-type]
    )

    result = binding.static.tools[0].handler({"path": "image.bin"})

    assert isinstance(result, ToolResult)
    assert result.ok is True
    assert result.metadata["media_type"] == media_type


@pytest.mark.parametrize(
    "data",
    [
        b"\x89PNG\r\n\x1a\n",
        b"\xff\xd8\xffleading only",
        b"GIF89a",
        b"RIFF\x04\x00\x00\x00WEBP",
    ],
)
def test_read_image_rejects_truncated_or_spoofed_headers(tmp_path: Path, data: bytes) -> None:
    (tmp_path / "bad.bin").write_bytes(data)
    binding = FilesystemPlugin(tools=["read_image"]).bind(
        type("Context", (), {"root": tmp_path})(),  # type: ignore[arg-type]
    )

    result = binding.static.tools[0].handler({"path": "bad.bin"})

    assert isinstance(result, ToolResult)
    assert result.ok is False
    assert "unsupported or invalid" in result.message_text()


def test_read_image_rejects_non_regular_file_without_opening(tmp_path: Path) -> None:
    directory = tmp_path / "directory"
    directory.mkdir()
    binding = FilesystemPlugin(tools=["read_image"]).bind(
        type("Context", (), {"root": tmp_path})(),  # type: ignore[arg-type]
    )

    result = binding.static.tools[0].handler({"path": "directory"})

    assert isinstance(result, ToolResult)
    assert result.ok is False
    assert "not a regular file" in result.message_text()


def test_filesystem_image_settings_validate_without_binding_io(tmp_path: Path, monkeypatch) -> None:
    for invalid in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="positive integer"):
            FilesystemPlugin(max_image_bytes=invalid)  # type: ignore[arg-type]

    plugin = FilesystemPlugin(tools=["read_image"], max_image_bytes=7)
    monkeypatch.setattr(Path, "open", lambda *_args, **_kwargs: pytest.fail("binding performed image I/O"))
    binding = plugin.bind(type("Context", (), {"root": tmp_path})())  # type: ignore[arg-type]
    assert binding.static.tools[0].name == "read_image"


class _Answer(BaseModel):
    value: str


def _structured_responses(provider_name: str) -> list[dict[str, Any]]:
    if provider_name == "openai":
        return [
            {"id": "bad", "output": [{"type": "function_call", "call_id": "final_bad", "name": "final_result", "arguments": "{}"}]},
            {"id": "good", "output": [{"type": "function_call", "call_id": "final_good", "name": "final_result", "arguments": '{"value":"ok"}'}]},
        ]
    if provider_name == "anthropic":
        return [
            {"content": [{"type": "tool_use", "id": "final_bad", "name": "final_result", "input": {}}]},
            {"content": [{"type": "tool_use", "id": "final_good", "name": "final_result", "input": {"value": "ok"}}]},
        ]
    return [
        {"choices": [{"message": {"role": "assistant", "content": "", "tool_calls": [{
            "id": "final_bad",
            "type": "function",
            "function": {"name": "final_result", "arguments": "{}"},
        }]}}]},
        {"choices": [{"message": {"role": "assistant", "content": "", "tool_calls": [{
            "id": "final_good",
            "type": "function",
            "function": {"name": "final_result", "arguments": '{"value":"ok"}'},
        }]}}]},
    ]


@pytest.mark.parametrize("provider_name", ["openai", "anthropic", "openrouter"])
async def test_structured_output_retry_keeps_plain_wire_text(tmp_path: Path, provider_name: str) -> None:
    provider = _Provider(provider_name, _structured_responses(provider_name))
    config = HarnessConfig(
        root=tmp_path,
        system_prompt="sys",
        local_tracing=False,
        output_type=_Answer,
        output_mode="tool",
    )
    result = await Harness(config, model=_model_for(provider_name, provider)).run("answer")
    assert result.output == _Answer(value="ok")

    if provider_name == "openai":
        wire = provider.payloads[1]["input"][0]["output"]
    elif provider_name == "anthropic":
        wire = provider.payloads[1]["messages"][-1]["content"][0]["content"]
    else:
        wire = provider.payloads[1]["messages"][-1]["content"]
    assert isinstance(wire, str)
    assert wire.startswith("The previous response failed structured output validation.")
    assert not wire.startswith("{")


async def test_initial_image_provider_failure_is_redacted_everywhere(tmp_path: Path) -> None:
    tracer = FakeTracer()
    provider = _FailingProvider()
    harness = Harness(
        _config(tmp_path),
        model=OpenAIResponsesModel("target", provider=provider),
        tracing=[TracingOptions(tracer=tracer, capture_messages=True)],
    )
    events = []
    with pytest.raises(HarnessError):
        async for event in harness.stream((TextBlock("go"), ImageBlock(PNG, "image/png"))):
            events.append(event)

    failed = next(event for event in events if isinstance(event, RunFailedEvent))
    trace_text = json.dumps([span.attributes for span in tracer.spans], ensure_ascii=False)
    exceptions = " ".join(str(exc) for span in tracer.spans for exc in span.exceptions)
    assert PNG_B64 not in failed.message + trace_text + exceptions
    assert PNG_URL not in failed.message + trace_text + exceptions


async def test_async_empty_string_tool_result_succeeds(tmp_path: Path) -> None:
    async def empty(_args):
        return ""

    provider = _Provider("OpenAI", [
        {"id": "r1", "output": [{"type": "function_call", "call_id": "empty", "name": "empty", "arguments": "{}"}]},
        {"id": "r2", "output_text": "done"},
    ])
    result = await Harness(
        _config(tmp_path),
        model=OpenAIResponsesModel("target", provider=provider),  # type: ignore[arg-type]
        tools=[ToolSpec("empty", "Empty.", {"type": "object", "properties": {}}, empty)],
    ).run("go")

    assert result.tool_call_records[0]["result"] == {"ok": True, "content": "", "metadata": {}}


async def test_bash_printed_image_path_stays_text_only(tmp_path: Path) -> None:
    (tmp_path / "sample.png").write_bytes(PNG)
    provider = _Provider("OpenAI", [
        {
            "id": "r1",
            "output": [{
                "type": "function_call",
                "call_id": "bash",
                "name": "bash",
                "arguments": '{"command":"printf sample.png"}',
            }],
        },
        {"id": "r2", "output_text": "done"},
    ])
    result = await Harness(
        _config(tmp_path),
        model=OpenAIResponsesModel("target", provider=provider),  # type: ignore[arg-type]
        plugins=[BashPlugin()],
    ).run("print path")

    content = result.tool_call_records[0]["result"]["content"]
    assert isinstance(content, str)
    assert "sample.png" in content
    assert PNG_B64 not in result.tool_call_records[0]["output"]


def test_read_image_rejects_symlink_escape(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside.png"
    outside.write_bytes(PNG)
    (tmp_path / "escape.png").symlink_to(outside)
    binding = FilesystemPlugin(tools=["read_image"]).bind(
        type("Context", (), {"root": tmp_path})(),  # type: ignore[arg-type]
    )

    result = binding.static.tools[0].handler({"path": "escape.png"})

    assert isinstance(result, ToolResult)
    assert result.ok is False
    assert "escapes root" in result.message_text()


def test_read_image_reports_unreadable_file(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "image.png"
    path.write_bytes(PNG)
    binding = FilesystemPlugin(tools=["read_image"]).bind(
        type("Context", (), {"root": tmp_path})(),  # type: ignore[arg-type]
    )
    original_open = Path.open

    def denied(self, *args, **kwargs):
        if self == path:
            raise PermissionError("denied")
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", denied)
    result = binding.static.tools[0].handler({"path": "image.png"})

    assert isinstance(result, ToolResult)
    assert result.ok is False
    assert result.metadata["error_type"] == "PermissionError"


async def test_image_approval_resume_succeeds_with_message_capture(tmp_path: Path) -> None:
    tracer = FakeTracer()
    provider = _Provider("OpenAI", [
        {
            "id": "pause",
            "output": [{"type": "function_call", "call_id": "approve", "name": "approved", "arguments": "{}"}],
        },
        {"id": "done", "output_text": "done"},
    ])
    tool = ToolSpec(
        "approved",
        "Approval tool.",
        {"type": "object", "properties": {}},
        lambda _args: "approved",
        requires_approval=True,
    )
    harness = Harness(
        _config(tmp_path),
        model=OpenAIResponsesModel("target", provider=provider),  # type: ignore[arg-type]
        tools=[tool],
        tracing=[TracingOptions(tracer=tracer, capture_messages=True)],
    )
    paused = await harness.run((TextBlock("approve"), ImageBlock(PNG, "image/png")))

    result = await harness.resume_approvals(
        paused.resume_state,
        [ApprovalDecision("approve", True)],
    )

    assert result.text == "done"
    assert any(span.name.startswith("chat ") for span in tracer.spans)
