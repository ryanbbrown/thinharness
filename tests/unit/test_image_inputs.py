from __future__ import annotations

import base64
import copy
import json
from pathlib import Path
from typing import Any

import pytest

from thinharness import (
    AnthropicMessagesModel,
    FilesystemPlugin,
    Harness,
    HarnessConfig,
    ImageBlock,
    OpenAIResponsesModel,
    OpenRouterModel,
    TextBlock,
    ToolResult,
    ToolSpec,
)
from thinharness.content import content_from_json, content_to_json, normalize_content

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
        lambda _args: [TextBlock("caption"), ImageBlock(PNG, "image/png")],
    )

    await Harness(_config(tmp_path), model=OpenRouterModel("vendor/model", provider=provider), tools=[tool]).run("go")  # type: ignore[arg-type]

    assert provider.payloads[1]["messages"][-2:] == [
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": json.dumps({
                "ok": True,
                "content": [
                    {"type": "text", "text": "caption"},
                    {"type": "image", "media_type": "image/png", "size_bytes": 33, "block_index": 1},
                ],
                "metadata": {},
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
