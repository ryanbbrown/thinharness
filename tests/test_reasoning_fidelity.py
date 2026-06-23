"""Same-provider reasoning fidelity in the neutral transcript (plan-31)."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import pytest
from fakes import FakeAnthropicProvider, FakeOpenRouterProvider, echo_tool

from thinharness import (
    AnthropicMessagesModel,
    AnthropicProvider,
    Harness,
    HarnessConfig,
    HarnessError,
    OpenAIProvider,
    OpenAIResponsesModel,
    OpenRouterModel,
    OpenRouterProvider,
    ToolSpec,
)
from thinharness.projections import trace_input_messages_from_entries, trace_output_messages_from_assistant
from thinharness.providers import AssistantEntry, ModelSettings, ModelToolCall, ReasoningPart, UserEntry, _openai_supports_encrypted_reasoning

REASONING_OPENAI_MODEL = "gpt-5-mini"
THINKING_SETTINGS = ModelSettings(extra_body={"thinking": {"type": "enabled", "budget_tokens": 1024}})


# --- reasoning-emitting fakes (real-provider-backed, per plan §Tests) -----------------------------


class ReasoningOpenAIProvider(OpenAIProvider):
    """Responses fake whose first turn carries a native reasoning item + tool call."""

    def __init__(self) -> None:
        super().__init__(api_key="fake")
        self.payloads: list = []
        self.calls = 0

    async def create_response(self, payload):
        self.calls += 1
        self.payloads.append(copy.deepcopy(payload))
        if self.calls == 1:
            return {
                "id": "resp_1",
                "output": [
                    {"type": "reasoning", "id": "rs_1", "summary": [{"type": "summary_text", "text": "thinking about it"}], "encrypted_content": "enc-blob-1"},
                    {"type": "function_call", "call_id": "call_1", "name": "echo", "arguments": '{"value":"hi"}'},
                ],
            }
        return {"id": "resp_2", "output_text": "done"}


class TerminalOpenAIProvider(OpenAIProvider):
    """Responses fake that records payloads and terminates with text."""

    def __init__(self) -> None:
        super().__init__(api_key="fake")
        self.payloads: list = []

    async def create_response(self, payload):
        self.payloads.append(copy.deepcopy(payload))
        return {"id": f"resp_{len(self.payloads)}", "output_text": "done"}


class ReasoningTextOpenAIProvider(OpenAIProvider):
    """Responses fake whose first turn carries reasoning + assistant text + a tool call."""

    def __init__(self) -> None:
        super().__init__(api_key="fake")
        self.payloads: list = []
        self.calls = 0

    async def create_response(self, payload):
        self.calls += 1
        self.payloads.append(copy.deepcopy(payload))
        if self.calls == 1:
            return {
                "id": "resp_1",
                "output": [
                    {"type": "reasoning", "id": "rs_1", "summary": [{"type": "summary_text", "text": "considering"}], "encrypted_content": "enc-blob-1"},
                    {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "let me call echo"}]},
                    {"type": "function_call", "call_id": "call_1", "name": "echo", "arguments": '{"value":"hi"}'},
                ],
            }
        return {"id": "resp_2", "output_text": "done"}


class ReasoningAnthropicProvider(AnthropicProvider):
    """Messages fake whose first turn carries configurable reasoning blocks + a tool call."""

    def __init__(self, reasoning_blocks: list) -> None:
        super().__init__(api_key="key")
        self.reasoning_blocks = reasoning_blocks
        self.payloads: list = []

    async def create_message(self, payload):
        self.payloads.append(copy.deepcopy(payload))
        last = payload["messages"][-1]
        if isinstance(last["content"], str):
            return {
                "content": [
                    *copy.deepcopy(self.reasoning_blocks),
                    {"type": "tool_use", "id": "toolu_1", "name": "echo", "input": {"value": last["content"]}},
                ],
                "stop_reason": "tool_use",
            }
        return {"content": [{"type": "text", "text": "done"}], "stop_reason": "end_turn"}


class ReasoningOpenRouterProvider(OpenRouterProvider):
    """OpenRouter fake whose first turn carries reasoning_details + a tool call."""

    def __init__(self) -> None:
        super().__init__(api_key="key")
        self.payloads: list = []

    async def create_chat_completion(self, payload):
        self.payloads.append(copy.deepcopy(payload))
        last = payload["messages"][-1]
        if last["role"] == "user":
            return {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "reasoning_details": [{"type": "reasoning.encrypted", "data": "or-enc-1", "id": "rd_1"}],
                        "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "echo", "arguments": json.dumps({"value": last["content"]})}}],
                    }
                }]
            }
        return {"choices": [{"message": {"role": "assistant", "content": "done"}}]}


def _harness(tmp_path: Path, model, **config) -> Harness:
    return Harness(HarnessConfig(root=tmp_path, builtin_tools=[], **config), model=model, tools=[echo_tool()])


async def _capture_state(tmp_path: Path, model) -> dict:
    """Run one capturing turn and return its JSON-round-tripped resume state."""
    result = await _harness(tmp_path, model).run("first")
    return json.loads(json.dumps(result.resume_state))


def _assistant_reasoning(state: dict) -> list[dict]:
    """Return the reasoning list of the first assistant entry carrying reasoning."""
    for entry in state["entries"]:
        if entry["role"] == "assistant" and entry["reasoning"]:
            return entry["reasoning"]
    raise AssertionError("no assistant entry carried reasoning")


# --- 1. capture --------------------------------------------------------------------------------


async def test_openai_capture_populates_reasoning_and_include(tmp_path: Path) -> None:
    provider = ReasoningOpenAIProvider()
    result = await _harness(tmp_path, OpenAIResponsesModel(REASONING_OPENAI_MODEL, provider=provider)).run("first")

    assert provider.payloads[0]["include"] == ["reasoning.encrypted_content"]
    reasoning = _assistant_reasoning(result.resume_state)
    assert reasoning == [{"text": "thinking about it", "signature": "enc-blob-1", "id": "rs_1", "provider_name": "openai"}]


async def test_anthropic_capture_populates_reasoning(tmp_path: Path) -> None:
    provider = ReasoningAnthropicProvider([{"type": "thinking", "thinking": "let me think", "signature": "sig-1"}])
    result = await _harness(tmp_path, AnthropicMessagesModel("claude-test", provider=provider)).run("first")

    assert _assistant_reasoning(result.resume_state) == [{"text": "let me think", "signature": "sig-1", "provider_name": "anthropic"}]


async def test_openrouter_capture_keeps_raw_reasoning_details(tmp_path: Path) -> None:
    provider = ReasoningOpenRouterProvider()
    result = await _harness(tmp_path, OpenRouterModel("openai/test", provider=provider)).run("first")

    assert _assistant_reasoning(result.resume_state) == [{
        "text": "",
        "signature": "or-enc-1",
        "id": "rd_1",
        "provider_name": "openrouter",
        "provider_details": {"type": "reasoning.encrypted", "data": "or-enc-1", "id": "rd_1"},
    }]


# --- 2. same-provider native re-emit -----------------------------------------------------------


async def test_openai_same_provider_reemits_reasoning_item(tmp_path: Path) -> None:
    state = await _capture_state(tmp_path, OpenAIResponsesModel(REASONING_OPENAI_MODEL, provider=ReasoningOpenAIProvider()))

    provider = TerminalOpenAIProvider()
    await _harness(tmp_path, OpenAIResponsesModel(REASONING_OPENAI_MODEL, provider=provider)).run("follow-up", resume_from=state)

    items = provider.payloads[0]["input"]
    # Trailing "message" pair = the source run's "done" assistant turn + the new follow-up prompt.
    assert [item["type"] for item in items] == ["message", "reasoning", "function_call", "function_call_output", "message", "message"]
    assert items[1] == {"type": "reasoning", "id": "rs_1", "encrypted_content": "enc-blob-1", "summary": []}
    assert "previous_response_id" not in provider.payloads[0]


async def test_anthropic_same_provider_reemits_thinking_first(tmp_path: Path) -> None:
    source = ReasoningAnthropicProvider([{"type": "thinking", "thinking": "let me think", "signature": "sig-1"}])
    state = await _capture_state(tmp_path, AnthropicMessagesModel("claude-test", provider=source))

    provider = FakeAnthropicProvider()
    await _harness(tmp_path, AnthropicMessagesModel("claude-test", provider=provider, settings=THINKING_SETTINGS)).run("follow-up", resume_from=state)

    content = provider.payloads[0]["messages"][1]["content"]
    assert content[0] == {"type": "thinking", "thinking": "let me think", "signature": "sig-1"}
    assert content[1]["type"] == "tool_use"


async def test_openrouter_same_provider_reattaches_reasoning_details(tmp_path: Path) -> None:
    state = await _capture_state(tmp_path, OpenRouterModel("openai/test", provider=ReasoningOpenRouterProvider()))

    provider = FakeOpenRouterProvider()
    await _harness(tmp_path, OpenRouterModel("openai/test", provider=provider)).run("follow-up", resume_from=state)

    assistant = provider.payloads[0]["messages"][2]
    assert assistant["reasoning_details"] == [{"type": "reasoning.encrypted", "data": "or-enc-1", "id": "rd_1"}]
    assert assistant["tool_calls"][0]["id"] == "call_1"


# --- 2b. cross-model same-provider degrades (resuming model can't accept native reasoning) -----


async def test_openai_cross_model_falls_back_to_text(tmp_path: Path) -> None:
    state = await _capture_state(tmp_path, OpenAIResponsesModel(REASONING_OPENAI_MODEL, provider=ReasoningOpenAIProvider()))

    provider = TerminalOpenAIProvider()
    await _harness(tmp_path, OpenAIResponsesModel("gpt-4.1-mini", provider=provider)).run("follow-up", resume_from=state)

    items = provider.payloads[0]["input"]
    assert not any(item["type"] == "reasoning" for item in items)
    fallback = next(item for item in items if item["type"] == "message" and item["role"] == "assistant")
    assert fallback["content"][0]["text"] == "<thinking>\nthinking about it\n</thinking>"
    assert "enc-blob" not in json.dumps(items)


async def test_openai_reasoning_text_toolcall_render_in_order(tmp_path: Path) -> None:
    state = await _capture_state(tmp_path, OpenAIResponsesModel(REASONING_OPENAI_MODEL, provider=ReasoningTextOpenAIProvider()))

    provider = TerminalOpenAIProvider()
    await _harness(tmp_path, OpenAIResponsesModel(REASONING_OPENAI_MODEL, provider=provider)).run("follow-up", resume_from=state)

    items = provider.payloads[0]["input"]
    types = [item["type"] for item in items]
    start = types.index("reasoning")
    assert types[start:start + 3] == ["reasoning", "message", "function_call"]
    assert items[start + 1]["content"][0]["text"] == "let me call echo"


# --- 3. cross-provider text fallback -----------------------------------------------------------


async def test_cross_provider_fallback_to_openai_text(tmp_path: Path) -> None:
    source = ReasoningAnthropicProvider([{"type": "thinking", "thinking": "let me think", "signature": "sig-1"}])
    state = await _capture_state(tmp_path, AnthropicMessagesModel("claude-test", provider=source))

    provider = TerminalOpenAIProvider()
    await _harness(tmp_path, OpenAIResponsesModel(REASONING_OPENAI_MODEL, provider=provider)).run("follow-up", resume_from=state)

    items = provider.payloads[0]["input"]
    assert not any(item["type"] == "reasoning" for item in items)
    fallback = next(item for item in items if item["type"] == "message" and item["role"] == "assistant")
    assert fallback["content"][0]["text"] == "<thinking>\nlet me think\n</thinking>"
    assert "enc-blob" not in json.dumps(items) and "sig-1" not in json.dumps(items)


async def test_cross_provider_fallback_to_openrouter_text(tmp_path: Path) -> None:
    source = ReasoningAnthropicProvider([{"type": "thinking", "thinking": "let me think", "signature": "sig-1"}])
    state = await _capture_state(tmp_path, AnthropicMessagesModel("claude-test", provider=source))

    provider = FakeOpenRouterProvider()
    await _harness(tmp_path, OpenRouterModel("openai/test", provider=provider)).run("follow-up", resume_from=state)

    assistant = provider.payloads[0]["messages"][2]
    assert assistant["content"].startswith("<thinking>\nlet me think\n</thinking>")
    assert "reasoning_details" not in assistant
    assert "sig-1" not in json.dumps(provider.payloads[0]["messages"])


# --- 4. anthropic thinking-disabled fallback ---------------------------------------------------


async def test_anthropic_same_provider_falls_back_when_thinking_disabled(tmp_path: Path) -> None:
    source = ReasoningAnthropicProvider([{"type": "thinking", "thinking": "let me think", "signature": "sig-1"}])
    state = await _capture_state(tmp_path, AnthropicMessagesModel("claude-test", provider=source))

    provider = FakeAnthropicProvider()
    await _harness(tmp_path, AnthropicMessagesModel("claude-test", provider=provider)).run("follow-up", resume_from=state)

    content = provider.payloads[0]["messages"][1]["content"]
    assert content[0] == {"type": "text", "text": "<thinking>\nlet me think\n</thinking>"}
    assert content[1]["type"] == "tool_use"
    assert not any(block["type"] == "thinking" for block in content)


# --- 5. redacted_thinking ----------------------------------------------------------------------


async def test_redacted_thinking_round_trips_natively(tmp_path: Path) -> None:
    source = ReasoningAnthropicProvider([{"type": "redacted_thinking", "data": "redacted-blob"}])
    state = await _capture_state(tmp_path, AnthropicMessagesModel("claude-test", provider=source))
    assert _assistant_reasoning(state) == [{"text": "", "signature": "redacted-blob", "id": "redacted_thinking", "provider_name": "anthropic"}]

    provider = FakeAnthropicProvider()
    await _harness(tmp_path, AnthropicMessagesModel("claude-test", provider=provider, settings=THINKING_SETTINGS)).run("follow-up", resume_from=state)
    content = provider.payloads[0]["messages"][1]["content"]
    assert content[0] == {"type": "redacted_thinking", "data": "redacted-blob"}
    assert content[1]["type"] == "tool_use"


async def test_redacted_thinking_dropped_when_thinking_disabled(tmp_path: Path) -> None:
    source = ReasoningAnthropicProvider([{"type": "redacted_thinking", "data": "redacted-blob"}])
    state = await _capture_state(tmp_path, AnthropicMessagesModel("claude-test", provider=source))

    provider = FakeAnthropicProvider()
    await _harness(tmp_path, AnthropicMessagesModel("claude-test", provider=provider)).run("follow-up", resume_from=state)
    # redacted_thinking has no text, so a disabled-thinking resume drops it entirely (no native block, no fallback).
    content = provider.payloads[0]["messages"][1]["content"]
    assert [block["type"] for block in content] == ["tool_use"]


# --- 6. multi-part turn ------------------------------------------------------------------------


async def test_multi_part_reasoning_renders_in_order(tmp_path: Path) -> None:
    source = ReasoningAnthropicProvider([
        {"type": "thinking", "thinking": "step one", "signature": "sig-A"},
        {"type": "redacted_thinking", "data": "redacted-blob"},
    ])
    state = await _capture_state(tmp_path, AnthropicMessagesModel("claude-test", provider=source))
    assert len(_assistant_reasoning(state)) == 2

    provider = FakeAnthropicProvider()
    await _harness(tmp_path, AnthropicMessagesModel("claude-test", provider=provider, settings=THINKING_SETTINGS)).run("follow-up", resume_from=state)
    content = provider.payloads[0]["messages"][1]["content"]
    assert [block["type"] for block in content] == ["thinking", "redacted_thinking", "tool_use"]


# --- 7. round-trip serialization ---------------------------------------------------------------


async def test_reasoning_state_round_trips(tmp_path: Path) -> None:
    state = (await _harness(tmp_path, OpenAIResponsesModel(REASONING_OPENAI_MODEL, provider=ReasoningOpenAIProvider())).run("first")).resume_state

    assert state["version"] == 3
    assert json.loads(json.dumps(state)) == state
    assert _assistant_reasoning(state)[0]["signature"] == "enc-blob-1"


# --- 8. version rejection ----------------------------------------------------------------------


@pytest.mark.parametrize("version", [1, 2])
def test_old_transcript_versions_are_rejected(version: int) -> None:
    state = {"kind": "transcript", "version": version, "origin_provider": "openai", "origin_model": "gpt-test", "entries": []}
    with pytest.raises(HarnessError, match=f"resume_from version {version} is not supported"):
        OpenAIResponsesModel("gpt-test", provider=OpenAIProvider(api_key="fake")).resume_session(state)


# --- 9. in-run guard ---------------------------------------------------------------------------


def test_openai_include_is_the_only_in_run_payload_delta() -> None:
    reasoning = OpenAIResponsesModel(REASONING_OPENAI_MODEL, provider=OpenAIProvider(api_key="fake")).build_payload(input_payload="x", tools=[])
    plain = OpenAIResponsesModel("gpt-4.1-mini", provider=OpenAIProvider(api_key="fake")).build_payload(input_payload="x", tools=[])

    assert set(reasoning) == {"model", "input", "tools", "include"}
    assert set(plain) == {"model", "input", "tools"}
    assert reasoning["include"] == ["reasoning.encrypted_content"]
    assert {k: reasoning[k] for k in ("input", "tools")} == {k: plain[k] for k in ("input", "tools")}


async def test_anthropic_and_openrouter_in_run_payloads_carry_no_reasoning_keys(tmp_path: Path) -> None:
    anthropic = FakeAnthropicProvider()
    await _harness(tmp_path, AnthropicMessagesModel("claude-test", provider=anthropic)).run("first")
    assert all(set(payload) == {"model", "max_tokens", "system", "messages", "tools"} for payload in anthropic.payloads)

    openrouter = FakeOpenRouterProvider()
    await _harness(tmp_path, OpenRouterModel("openai/test", provider=openrouter)).run("first")
    assert all(set(payload) == {"model", "messages", "tools"} for payload in openrouter.payloads)


# --- OTel projection (spec `thinking` part) ----------------------------------------------------


def test_reasoning_projects_to_otel_thinking_part() -> None:
    entry = AssistantEntry(
        text="answer",
        tool_calls=[ModelToolCall(id="call_1", name="echo", arguments="{}")],
        reasoning=[
            ReasoningPart(text="let me think", signature="sig-secret", provider_name="anthropic"),
            ReasoningPart(text="", signature="enc-secret", id="redacted_thinking", provider_name="anthropic"),
        ],
    )

    parts = trace_output_messages_from_assistant(entry)[0]["parts"]
    # thinking first, then text, then tool_call; the text-less (redacted) part is skipped.
    assert [part["type"] for part in parts] == ["thinking", "text", "tool_call"]
    assert parts[0] == {"type": "thinking", "content": "let me think"}
    # opaque signatures/blobs never reach traces
    assert "sig-secret" not in json.dumps(parts) and "enc-secret" not in json.dumps(parts)

    # input-history projection delegates to the same function, so it carries thinking too
    input_messages = trace_input_messages_from_entries([UserEntry(content="hi"), entry])
    assert input_messages[1]["parts"][0] == {"type": "thinking", "content": "let me think"}


# --- supporting: model-name reasoning detection ------------------------------------------------


@pytest.mark.parametrize(
    ("model_name", "expected"),
    [
        ("gpt-5-mini", True),
        ("o3-mini", True),
        ("o1", True),
        ("gpt-5.1", True),
        ("gpt-4.1-mini", False),
        ("gpt-4o", False),
        ("gpt-5-chat", False),
        ("gpt-5.3-chat-latest", False),
    ],
)
def test_openai_reasoning_detection(model_name: str, expected: bool) -> None:
    assert _openai_supports_encrypted_reasoning(model_name) is expected


# --- guarded live suite (mirrors the plan §Tests smoke verification) ----------------------------
# Gated behind real API keys; model names are env-overridable and default to reasoning-capable
# models. Each case captures native reasoning on a one-tool round-trip and resumes on the same
# provider/model, asserting the re-emitted native reasoning is accepted (the run completes).


def _multiply_tool() -> ToolSpec:
    """A small arithmetic tool that nudges a reasoning model to actually reason."""
    return ToolSpec(
        "multiply",
        "Multiply two integers and return the product.",
        {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}, "required": ["a", "b"], "additionalProperties": False},
        lambda args: str(int(args["a"]) * int(args["b"])),
    )


def _has_signed_reasoning(state: dict) -> bool:
    return any(
        part.get("signature")
        for entry in state["entries"]
        if entry["role"] == "assistant"
        for part in entry["reasoning"]
    )


async def _run_reasoning_resume_live(tmp_path: Path, make_model) -> None:
    """Capture native reasoning, then resume on the same provider/model and assert acceptance."""
    first = await Harness(HarnessConfig(root=tmp_path, builtin_tools=[]), model=make_model(), tools=[_multiply_tool()]).run(
        "Use the multiply tool to compute 21 times 19, then state the product."
    )
    state = json.loads(json.dumps(first.resume_state))
    assert _has_signed_reasoning(state), "no signed native reasoning captured in resume_state"

    second = await Harness(HarnessConfig(root=tmp_path, builtin_tools=[]), model=make_model(), tools=[_multiply_tool()]).run(
        "Add 100 to that product.", resume_from=state
    )
    assert second.text


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY is not set")
async def test_openai_reasoning_resume_live(tmp_path: Path) -> None:
    provider = OpenAIProvider()
    name = os.getenv("THINHARNESS_LIVE_OPENAI_REASONING_MODEL", "gpt-5-mini")
    settings = ModelSettings(extra_body={"reasoning": {"effort": "low"}})
    try:
        await _run_reasoning_resume_live(tmp_path, lambda: OpenAIResponsesModel(name, provider=provider, settings=settings))
    finally:
        await provider.aclose()


@pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY is not set")
async def test_anthropic_reasoning_resume_live(tmp_path: Path) -> None:
    provider = AnthropicProvider()
    name = os.getenv("THINHARNESS_LIVE_ANTHROPIC_REASONING_MODEL", "claude-sonnet-4-5")
    settings = ModelSettings(extra_body={"thinking": {"type": "enabled", "budget_tokens": 1024}})
    try:
        await _run_reasoning_resume_live(tmp_path, lambda: AnthropicMessagesModel(name, provider=provider, settings=settings, max_tokens=2048))
    finally:
        await provider.aclose()


@pytest.mark.skipif(not os.getenv("OPENROUTER_API_KEY"), reason="OPENROUTER_API_KEY is not set")
@pytest.mark.parametrize(
    ("env_var", "default"),
    [
        ("THINHARNESS_LIVE_OPENROUTER_REASONING_MODEL", "openai/gpt-5-mini"),
        ("THINHARNESS_LIVE_OPENROUTER_REASONING_TEXT_MODEL", "anthropic/claude-sonnet-4.5"),
    ],
)
async def test_openrouter_reasoning_resume_live(tmp_path: Path, env_var: str, default: str) -> None:
    provider = OpenRouterProvider()
    name = os.getenv(env_var, default)
    settings = ModelSettings(extra_body={"reasoning": {"effort": "low"}})
    try:
        await _run_reasoning_resume_live(tmp_path, lambda: OpenRouterModel(name, provider=provider, settings=settings))
    finally:
        await provider.aclose()
