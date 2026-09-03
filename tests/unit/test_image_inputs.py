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


@pytest.mark.parametrize("bad", ["", [], [TextBlock("")], [TextBlock("ok"), object()]])
async def test_public_async_prompt_validation_raises_harness_error_and_emits_failure(tmp_path: Path, bad: Any) -> None:
    provider = _Provider("OpenAI", [])
    harness = Harness(_config(tmp_path), model=OpenAIResponsesModel("target", provider=provider))  # type: ignore[arg-type]
    events = []

    with pytest.raises(HarnessError):
        async for event in harness.stream(bad):
            events.append(event)

    failed = next(event for event in events if isinstance(event, RunFailedEvent))
    assert failed.error_type == "HarnessError"
    assert failed.stop_reason == "error"
    assert "prompt" in failed.message
    assert provider.payloads == []

    direct_provider = _Provider("OpenAI", [])
    direct = Harness(_config(tmp_path), model=OpenAIResponsesModel("target", provider=direct_provider))  # type: ignore[arg-type]
    with pytest.raises(HarnessError, match="prompt"):
        await direct.run(bad)
    assert direct_provider.payloads == []


@pytest.mark.parametrize("bad", ["", [], [TextBlock("")], [TextBlock("ok"), object()]])
def test_public_run_sync_prompt_validation_raises_harness_error(tmp_path: Path, bad: Any) -> None:
    provider = _Provider("OpenAI", [])
    harness = Harness(_config(tmp_path), model=OpenAIResponsesModel("target", provider=provider))  # type: ignore[arg-type]

    with pytest.raises(HarnessError, match="prompt"):
        harness.run_sync(bad)

    assert provider.payloads == []


@pytest.mark.parametrize("event", ["run_start", "user_prompt_submit"])
@pytest.mark.parametrize("strict", [False, True])
async def test_invalid_prompt_hook_replacement_is_harness_error(
    tmp_path: Path,
    event: str,
    strict: bool,
) -> None:
    def invalidate(ctx) -> None:
        ctx.prompt = []

    provider = _Provider("OpenAI", [])
    harness = Harness(
        HarnessConfig(root=tmp_path, system_prompt="sys", local_tracing=False, strict_hooks=strict),
        model=OpenAIResponsesModel("target", provider=provider),  # type: ignore[arg-type]
        hooks=[Hook(event, invalidate)],  # type: ignore[arg-type]
    )

    with pytest.raises(HarnessError, match=f"{event} prompt must not be empty"):
        await harness.run("caller")

    assert provider.payloads == []


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
        "store": False,
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

    assert provider.payloads[1]["input"][-1] == {
        "type": "function_call_output",
        "call_id": "call_1",
        "output": [
            {"type": "input_text", "text": '{"ok":true,"metadata":{"source":"fixture"}}'},
            {"type": "input_text", "text": "caption"},
            {"type": "input_image", "image_url": PNG_URL},
        ],
    }
    assert result.resume_state["version"] == 5
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
        "wire_output": None,
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
                "wire_output": None,
            },
        ])
    entries.append({"role": "assistant", "text": "prior done", "tool_calls": [], "reasoning": []})
    return {
        "kind": "transcript",
        "version": 5,
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


async def test_same_provider_openai_resume_combines_native_reasoning_and_image(tmp_path: Path) -> None:
    state = {
        "kind": "transcript",
        "version": 5,
        "origin_provider": "openai",
        "origin_model": "o3-source",
        "entries": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "inspect"},
                    {"type": "image", "media_type": "image/png", "data": PNG_B64},
                ],
                "notice": False,
            },
            {
                "role": "assistant",
                "text": "prior",
                "tool_calls": [],
                "reasoning": [{
                    "text": "",
                    "signature": "encrypted-reasoning",
                    "id": "rs_1",
                    "provider_name": "openai",
                }],
            },
        ],
    }
    provider = _Provider("OpenAI", [{"id": "done", "output_text": "done"}])

    await Harness(_config(tmp_path), model=OpenAIResponsesModel("o3-target", provider=provider)).run(  # type: ignore[arg-type]
        "follow-up",
        resume_from=state,
    )

    payload = provider.payloads[0]
    assert any(item.get("type") == "reasoning" and item.get("encrypted_content") == "encrypted-reasoning" for item in payload["input"])
    assert PNG_URL in json.dumps(payload, ensure_ascii=False)


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


async def test_agent_trace_keeps_raw_image_prompt_and_model_trace_uses_hook_context(tmp_path: Path) -> None:
    tracer = FakeTracer()

    def add_context(ctx) -> None:
        ctx.additional_context.append("effective-only policy")

    provider = _Provider("OpenAI", [{"id": "done", "output_text": "done"}])
    prompt = (TextBlock("caller"), ImageBlock(PNG, "image/png"))
    await Harness(
        _config(tmp_path),
        model=OpenAIResponsesModel("target", provider=provider),  # type: ignore[arg-type]
        hooks=[Hook("user_prompt_submit", add_context)],
        tracing=[TracingOptions(tracer=tracer, capture_messages=True)],
    ).run(prompt)

    agent = next(span for span in tracer.spans if span.name.startswith("invoke_agent "))
    model = next(span for span in tracer.spans if span.name.startswith("chat "))
    assert "effective-only policy" not in agent.attributes["gen_ai.prompt"]
    assert "effective-only policy" in model.attributes["gen_ai.input.messages"]
    combined = json.dumps([agent.attributes, model.attributes], ensure_ascii=False)
    assert PNG_B64 not in combined
    assert PNG_URL not in combined


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


@pytest.mark.parametrize("event", ["run_start", "user_prompt_submit"])
async def test_strict_prompt_hook_failure_redacts_known_image_data(tmp_path: Path, event: str) -> None:
    tracer = FakeTracer()

    def fail(_ctx) -> None:
        raise RuntimeError(f"hook echoed {PNG_B64} and {PNG_URL}")

    provider = _Provider("OpenAI", [])
    harness = Harness(
        HarnessConfig(root=tmp_path, system_prompt="sys", local_tracing=False, strict_hooks=True),
        model=OpenAIResponsesModel("target", provider=provider),  # type: ignore[arg-type]
        hooks=[Hook(event, fail)],  # type: ignore[arg-type]
        tracing=[TracingOptions(tracer=tracer, capture_messages=True)],
    )
    events = []

    with pytest.raises(HarnessError):
        async for stream_event in harness.stream((TextBlock("go"), ImageBlock(PNG, "image/png"))):
            events.append(stream_event)

    failed = next(item for item in events if isinstance(item, RunFailedEvent))
    observed = failed.message + json.dumps([span.attributes for span in tracer.spans], ensure_ascii=False)
    observed += " ".join(str(exc) for span in tracer.spans for exc in span.exceptions)
    assert failed.error_type == "RuntimeError"
    assert PNG_B64 not in observed
    assert PNG_URL not in observed


async def test_strict_after_tool_hook_failure_registers_and_redacts_original_and_replacement_images(tmp_path: Path) -> None:
    replacement_data = PNG + b"replacement"
    replacement_b64 = base64.b64encode(replacement_data).decode("ascii")
    replacement_url = f"data:image/png;base64,{replacement_b64}"
    tracer = FakeTracer()

    def fail(ctx) -> None:
        ctx.output = ToolResult(True, (ImageBlock(replacement_data, "image/png"),), {}).to_json()
        raise RuntimeError(f"hook echoed {PNG_B64} {PNG_URL} {replacement_b64} {replacement_url}")

    provider = _Provider("OpenAI", [{
        "id": "tool",
        "output": [{"type": "function_call", "call_id": "image", "name": "image", "arguments": "{}"}],
    }])
    harness = Harness(
        HarnessConfig(root=tmp_path, system_prompt="sys", local_tracing=False, strict_hooks=True),
        model=OpenAIResponsesModel("target", provider=provider),  # type: ignore[arg-type]
        tools=[ToolSpec("image", "Image.", {"type": "object", "properties": {}}, lambda _args: (ImageBlock(PNG, "image/png"),))],
        hooks=[Hook("after_tool_call", fail)],
        tracing=[TracingOptions(tracer=tracer, capture_tool_results=True)],
    )
    events = []

    with pytest.raises(HarnessError):
        async for event in harness.stream("go"):
            events.append(event)

    completed = next(item for item in events if isinstance(item, ToolCallCompletedEvent))
    failed = next(item for item in events if isinstance(item, RunFailedEvent))
    observed = json.dumps([completed.output, completed.message, failed.message, [span.attributes for span in tracer.spans]], ensure_ascii=False)
    observed += " ".join(str(exc) for span in tracer.spans for exc in span.exceptions)
    for secret in (PNG_B64, PNG_URL, replacement_b64, replacement_url):
        assert secret not in observed


async def test_redacted_tool_projection_removes_repeated_image_data_from_text_and_metadata(tmp_path: Path) -> None:
    tracer = FakeTracer()
    repeated = f"repeated {PNG_B64} and {PNG_URL}"
    provider = _Provider("OpenAI", [
        {"id": "tool", "output": [{"type": "function_call", "call_id": "image", "name": "image", "arguments": "{}"}]},
        {"id": "done", "output_text": "done"},
    ])
    harness = Harness(
        _config(tmp_path),
        model=OpenAIResponsesModel("target", provider=provider),  # type: ignore[arg-type]
        tools=[ToolSpec(
            "image",
            "Image.",
            {"type": "object", "properties": {}},
            lambda _args: ToolResult(True, (TextBlock(repeated), ImageBlock(PNG, "image/png")), {"echo": repeated}),
        )],
        tracing=[TracingOptions(tracer=tracer, capture_tool_results=True)],
    )
    events = [event async for event in harness.stream("go")]

    completed = next(event for event in events if isinstance(event, ToolCallCompletedEvent))
    tool_span = next(span for span in tracer.spans if span.name == "execute_tool image")
    observed = completed.output + str(tool_span.attributes.get("gen_ai.tool.call.result", ""))
    assert PNG_B64 not in observed
    assert PNG_URL not in observed
    assert observed.count("[image data redacted]") >= 2


class _FailingProvider(_Provider):
    def __init__(self, name: str = "OpenAI", *, leak: bool = True) -> None:
        super().__init__(name, [])
        self.leak = leak

    async def create_response(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.payloads.append(copy.deepcopy(payload))
        message = f"provider echoed {PNG_B64} and {PNG_URL}" if self.leak else "plain provider failure"
        raise ProviderError(message, status_code=422)


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


@pytest.mark.parametrize("leak", [False, True])
async def test_provider_failure_redaction_keeps_public_classification_and_status(tmp_path: Path, leak: bool) -> None:
    tracer = FakeTracer()
    provider = _FailingProvider(leak=leak)
    harness = Harness(
        _config(tmp_path),
        model=OpenAIResponsesModel("target", provider=provider),
        tracing=[TracingOptions(tracer=tracer, capture_messages=True)],
    )
    events = []

    with pytest.raises(HarnessError) as exc_info:
        async for event in harness.stream((TextBlock("go"), ImageBlock(PNG, "image/png"))):
            events.append(event)

    failed = next(event for event in events if isinstance(event, RunFailedEvent))
    model_span = next(span for span in tracer.spans if span.name.startswith("chat "))
    assert failed.error_type == "ProviderError"
    assert model_span.attributes["error.type"] == "ProviderError"
    assert exc_info.value.__dict__["status_code"] == 422
    if leak:
        assert failed.message == "provider echoed [image data redacted] and [image data redacted]"
    else:
        assert failed.message == "plain provider failure"


async def test_non_vision_provider_error_keeps_provider_classification(tmp_path: Path) -> None:
    class NonVisionProvider(_Provider):
        async def create_response(self, payload: dict[str, Any]) -> dict[str, Any]:
            self.payloads.append(copy.deepcopy(payload))
            raise ProviderError("selected model does not support image inputs", status_code=400)

    provider = NonVisionProvider("OpenAI", [])
    harness = Harness(_config(tmp_path), model=OpenAIResponsesModel("text-only", provider=provider))  # type: ignore[arg-type]
    events = []

    with pytest.raises(HarnessError) as exc_info:
        async for event in harness.stream((TextBlock("look"), ImageBlock(PNG, "image/png"))):
            events.append(event)

    failed = next(event for event in events if isinstance(event, RunFailedEvent))
    assert failed.error_type == "ProviderError"
    assert failed.message == "selected model does not support image inputs"
    assert exc_info.value.__dict__["status_code"] == 400


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

    output = next(item["output"] for item in provider.payloads[1]["input"] if item.get("type") == "function_call_output")
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
        wire = next(item["output"] for item in provider.payloads[1]["input"] if item.get("type") == "function_call_output")
    elif provider_name == "anthropic":
        wire = provider.payloads[1]["messages"][-1]["content"][0]["content"]
    else:
        wire = provider.payloads[1]["messages"][-1]["content"]
    assert isinstance(wire, str)
    assert wire.startswith("The previous response failed structured output validation.")
    assert not wire.startswith("{")


@pytest.mark.parametrize("provider_name", ["openai", "anthropic", "openrouter"])
async def test_structured_output_retry_wire_text_persists_and_replays_exactly(tmp_path: Path, provider_name: str) -> None:
    live_provider = _Provider(provider_name, _structured_responses(provider_name))
    model = _model_for(provider_name, live_provider)
    session = model.new_session()
    model.new_session = lambda: session  # type: ignore[method-assign]
    config = HarnessConfig(
        root=tmp_path,
        system_prompt="sys",
        local_tracing=False,
        output_type=_Answer,
        output_mode="tool",
    )
    result = await Harness(config, model=model).run("answer")
    assert result.output == _Answer(value="ok")
    state = session.dump_state()
    assert state is not None
    retry_entry = next(entry for entry in state["entries"] if entry["role"] == "tool" and entry["call_id"] == "final_bad")
    live_wire = retry_entry["wire_output"]
    assert isinstance(live_wire, str)
    assert live_wire.startswith("The previous response failed structured output validation.")

    replay_provider = _Provider(provider_name, [_final_response(provider_name)])
    await Harness(_config(tmp_path), model=_model_for(provider_name, replay_provider)).run("follow-up", resume_from=state)
    payload = replay_provider.payloads[0]
    if provider_name == "openai":
        replay_wire = next(item["output"] for item in payload["input"] if item.get("type") == "function_call_output" and item.get("call_id") == "final_bad")
    elif provider_name == "anthropic":
        replay_wire = next(
            block["content"]
            for message in payload["messages"]
            if isinstance(message.get("content"), list)
            for block in message["content"]
            if block.get("type") == "tool_result" and block.get("tool_use_id") == "final_bad"
        )
    else:
        replay_wire = next(
            message["content"]
            for message in payload["messages"]
            if message.get("role") == "tool" and message.get("tool_call_id") == "final_bad"
        )
    assert replay_wire == live_wire


@pytest.mark.parametrize("mutation, message", [("wrong_type", "wrong type"), ("missing", "wrong keys")])
def test_resume_tool_wire_output_decodes_strictly(tmp_path: Path, mutation: str, message: str) -> None:
    state = _resume_state("openai")
    tool_entry = next(entry for entry in state["entries"] if entry["role"] == "tool")
    if mutation == "wrong_type":
        tool_entry["wire_output"] = 3
    else:
        del tool_entry["wire_output"]
    harness = Harness(_config(tmp_path), model=OpenAIResponsesModel("target", provider=_Provider("OpenAI", [])))  # type: ignore[arg-type]

    with pytest.raises(HarnessError, match=message):
        harness.run_sync("follow-up", resume_from=state)


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
