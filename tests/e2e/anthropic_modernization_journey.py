from __future__ import annotations

import asyncio
import copy
import json
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from thinharness import AnthropicMessagesModel, AnthropicProvider, FilesystemPlugin, Harness, HarnessConfig, ModelSettings, ToolSpec

MODEL = os.getenv("E2E_ANTHROPIC_MODERNIZATION_MODEL", "anthropic:claude-sonnet-5")


class InventoryAnswer(BaseModel):
    sku: str
    count: int
    status: str


class RecordingAnthropicProvider(AnthropicProvider):
    def __init__(self) -> None:
        super().__init__()
        self.payloads: list[dict] = []

    async def create_message(self, payload):
        self.payloads.append(copy.deepcopy(payload))
        return await super().create_message(payload)


def multiply_tool() -> ToolSpec:
    return ToolSpec(
        "multiply",
        "Multiply two integers and return the product.",
        {
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
            "additionalProperties": False,
        },
        lambda args: str(int(args["a"]) * int(args["b"])),
    )


async def main() -> None:
    if _should_skip(MODEL):
        return

    model_name = MODEL.split(":", 1)[1]
    with TemporaryDirectory(prefix="thinharness-e2e-anthropic-modernization-") as raw_root:
        root = Path(raw_root)
        (root / "inventory.txt").write_text("sku=TH-001\ncount=7\nstatus=ready\n", encoding="utf-8")

        await _assert_native_structured_output_and_defaults(root, model_name)
        await _assert_effort_merges_with_native_structured_output(root, model_name)
        await _assert_default_on_thinking_resume(root, model_name)

    print(f"PASS anthropic_modernization_journey model={MODEL}")


async def _assert_native_structured_output_and_defaults(root: Path, model_name: str) -> None:
    provider = RecordingAnthropicProvider()
    try:
        harness = Harness(
            HarnessConfig(
                root=root,
                output_type=InventoryAnswer,
                max_model_requests=4,
                max_tool_calls=2,
            ),
            model=AnthropicMessagesModel(model_name, provider=provider),
            plugins=[FilesystemPlugin(tools=["read"])],
        )

        result = await harness.run(
            "Read inventory.txt, then return sku, count, and status exactly as structured output."
        )
    finally:
        await provider.aclose()

    assert harness.output_schema is not None
    assert harness.output_schema.mode == "native"
    assert result.output == InventoryAnswer(sku="TH-001", count=7, status="ready")
    assert provider.payloads, "no Anthropic payloads recorded"
    first_payload = provider.payloads[0]
    assert first_payload["max_tokens"] == 16384
    assert first_payload["output_config"]["format"]["type"] == "json_schema"
    assert first_payload["output_config"]["format"]["schema"]["properties"]["sku"]["type"] == "string"


async def _assert_effort_merges_with_native_structured_output(root: Path, model_name: str) -> None:
    provider = RecordingAnthropicProvider()
    try:
        harness = Harness(
            HarnessConfig(
                root=root,
                output_type=InventoryAnswer,
                max_model_requests=2,
            ),
            model=AnthropicMessagesModel(model_name, provider=provider, settings=ModelSettings(effort="low")),
        )

        result = await harness.run(
            "Return this exact structured object: sku TH-001, count 7, status ready."
        )
    finally:
        await provider.aclose()

    assert result.output == InventoryAnswer(sku="TH-001", count=7, status="ready")
    first_payload = provider.payloads[0]
    assert first_payload["thinking"] == {"type": "adaptive"}
    assert first_payload["output_config"]["effort"] == "low"
    assert first_payload["output_config"]["format"]["type"] == "json_schema"


async def _assert_default_on_thinking_resume(root: Path, model_name: str) -> None:
    first_provider = RecordingAnthropicProvider()
    try:
        first = await Harness(
            HarnessConfig(root=root, max_model_requests=4, max_tool_calls=2),
            model=AnthropicMessagesModel(model_name, provider=first_provider),
            tools=[multiply_tool()],
        ).run("Use the multiply tool to compute 37 times 29, then state the product.")
    finally:
        await first_provider.aclose()

    state = json.loads(json.dumps(first.resume_state))
    assert _has_signed_reasoning(state), "no signed Anthropic reasoning captured in resume_state"

    second_provider = RecordingAnthropicProvider()
    try:
        second = await Harness(
            HarnessConfig(root=root, max_model_requests=3, max_tool_calls=1),
            model=AnthropicMessagesModel(model_name, provider=second_provider),
            tools=[multiply_tool()],
        ).run("Add 11 to that product and answer with the new number.", resume_from=state)
    finally:
        await second_provider.aclose()

    assert second.text
    replay_payload = second_provider.payloads[0]
    assert "thinking" not in replay_payload
    assert _payload_replays_native_thinking(replay_payload), replay_payload["messages"]


def _has_signed_reasoning(state: dict) -> bool:
    return any(
        part.get("signature")
        for entry in state["entries"]
        if entry["role"] == "assistant"
        for part in entry["reasoning"]
    )


def _payload_replays_native_thinking(payload: dict) -> bool:
    for message in payload["messages"]:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        if any(block.get("type") in {"thinking", "redacted_thinking"} for block in content if isinstance(block, dict)):
            return True
    return False


def _should_skip(model: str) -> bool:
    if os.getenv("CI"):
        print("SKIP anthropic_modernization_journey: CI is set")
        return True
    if not model.startswith("anthropic:"):
        raise ValueError("E2E_ANTHROPIC_MODERNIZATION_MODEL must use the anthropic: provider prefix")
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("SKIP anthropic_modernization_journey: ANTHROPIC_API_KEY is not set")
        return True
    return False


if __name__ == "__main__":
    asyncio.run(main())
