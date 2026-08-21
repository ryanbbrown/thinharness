from __future__ import annotations

import asyncio
import copy
import os
import struct
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from thinharness import (
    AnthropicMessagesModel,
    AnthropicProvider,
    Harness,
    HarnessConfig,
    ImageBlock,
    OpenAIProvider,
    OpenAIResponsesModel,
    OpenRouterModel,
    OpenRouterProvider,
    TextBlock,
    ToolResult,
    ToolSpec,
)

ROOT = Path(__file__).resolve().parents[2]


def _known_image(left: bytes, right: bytes) -> bytes:
    """Return a 64x32 PNG with two known solid-color halves."""
    width, height = 64, 32
    row = b"\x00" + (left * (width // 2)) + (right * (width // 2))
    raw = row * height

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


IMAGE = _known_image(b"\xff\x00\x00", b"\x00\x00\xff")
TOOL_IMAGE = _known_image(b"\x00\xff\x00", b"\xff\xff\x00")


class RecordingOpenAI(OpenAIProvider):
    def __init__(self) -> None:
        super().__init__()
        self.payloads: list[dict] = []

    async def create_response(self, payload):
        self.payloads.append(copy.deepcopy(payload))
        return await super().create_response(payload)


class RecordingAnthropic(AnthropicProvider):
    def __init__(self) -> None:
        super().__init__()
        self.payloads: list[dict] = []

    async def create_message(self, payload):
        self.payloads.append(copy.deepcopy(payload))
        return await super().create_message(payload)


class RecordingOpenRouter(OpenRouterProvider):
    def __init__(self) -> None:
        super().__init__()
        self.payloads: list[dict] = []

    async def create_chat_completion(self, payload):
        self.payloads.append(copy.deepcopy(payload))
        return await super().create_chat_completion(payload)


def image_tool() -> ToolSpec:
    return ToolSpec(
        "inspect_fixture",
        "Return the comparison image. Call this exactly once.",
        {"type": "object", "properties": {}, "additionalProperties": False},
        lambda _args: ToolResult(
            True,
            (TextBlock("comparison fixture"), ImageBlock(TOOL_IMAGE, "image/png")),
            {"fixture": "green-yellow.png"},
        ),
    )


async def run_provider(label: str, model, payloads: list[dict]) -> None:
    harness = Harness(
        HarnessConfig(root=ROOT, max_model_requests=5, max_tool_calls=1, local_tracing=False),
        model=model,
        tools=[image_tool()],
    )
    first = await harness.run((
        TextBlock(
            "For the supplied image, name the left and right colors. Then call inspect_fixture and name its left and right colors. "
            "Report all four facts."
        ),
        ImageBlock(IMAGE, "image/png"),
    ))
    assert "red" in first.text.lower()
    assert "blue" in first.text.lower()
    assert "green" in first.text.lower()
    assert "yellow" in first.text.lower()
    assert any(record["call"]["name"] == "inspect_fixture" for record in first.tool_call_records)
    assert first.resume_state is not None
    resumed = await harness.run("In one sentence, restate the comparison.", resume_from=first.resume_state)
    assert resumed.text

    if label == "openai":
        assert any(
            isinstance(item.get("output"), list)
            for payload in payloads
            for item in payload.get("input", [])
            if isinstance(item, dict) and item.get("type") == "function_call_output"
        )
    if label == "anthropic":
        assert any(
            block.get("type") == "image"
            for payload in payloads
            for message in payload.get("messages", [])
            for item in message.get("content", []) if isinstance(message.get("content"), list)
            if isinstance(item, dict) and item.get("type") == "tool_result" and isinstance(item.get("content"), list)
            for block in item["content"]
            if isinstance(block, dict)
        )
    if label == "openrouter":
        assert any(
            part.get("text", "").startswith("[tool image call_id=")
            for payload in payloads
            for message in payload.get("messages", [])
            if isinstance(message.get("content"), list)
            for part in message["content"]
            if isinstance(part, dict) and part.get("type") == "text"
        )
    await harness.aclose()
    print(f"PASS image_inputs_journey provider={label}")


async def main() -> None:
    runs = []
    if os.getenv("OPENAI_API_KEY"):
        provider = RecordingOpenAI()
        runs.append(run_provider(
            "openai",
            OpenAIResponsesModel(os.getenv("E2E_OPENAI_IMAGE_MODEL", "gpt-5.5"), provider=provider),
            provider.payloads,
        ))
    if os.getenv("ANTHROPIC_API_KEY"):
        provider = RecordingAnthropic()
        runs.append(run_provider(
            "anthropic",
            AnthropicMessagesModel(os.getenv("E2E_ANTHROPIC_IMAGE_MODEL", "claude-sonnet-4-5"), provider=provider),
            provider.payloads,
        ))
    if os.getenv("OPENROUTER_API_KEY"):
        provider = RecordingOpenRouter()
        runs.append(run_provider(
            "openrouter",
            OpenRouterModel(os.getenv("E2E_OPENROUTER_IMAGE_MODEL", "openai/gpt-4o-mini"), provider=provider),
            provider.payloads,
        ))
    if not runs:
        print("SKIP image_inputs_journey: no provider credentials")
        return
    for run in runs:
        await run


if __name__ == "__main__":
    asyncio.run(main())
