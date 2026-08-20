from __future__ import annotations

import asyncio
import copy
import os
import sys
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
IMAGE = (ROOT / "assets" / "logo-circle.png").read_bytes()


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
            (TextBlock("comparison fixture"), ImageBlock(IMAGE, "image/png")),
            {"fixture": "logo-circle.png"},
        ),
    )


async def run_provider(label: str, model, payloads: list[dict]) -> None:
    harness = Harness(
        HarnessConfig(root=ROOT, max_model_requests=5, max_tool_calls=1, local_tracing=False),
        model=model,
        tools=[image_tool()],
    )
    first = await harness.run((
        TextBlock("Describe this image, then call inspect_fixture and compare the two images."),
        ImageBlock(IMAGE, "image/png"),
    ))
    assert first.text
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
    if label == "openrouter":
        assert any(
            part.get("text", "").startswith("[tool image call_id=")
            for payload in payloads
            for message in payload.get("messages", [])
            for part in message.get("content", []) if isinstance(message.get("content"), list)
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
