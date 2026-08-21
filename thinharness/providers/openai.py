"""OpenAI Responses provider, model, session, and wire dialect."""

from __future__ import annotations

import copy
import json
import os
from typing import Any

import httpx

from ..content import NormalizedContent, Prompt, TextBlock, normalize_content, text_only_value
from ..tools.base import Json, ToolResult
from .base import (
    ModelCapabilities,
    ModelNotice,
    ModelSession,
    ModelSettings,
    ModelToolCall,
    ModelTurn,
    ReasoningPart,
    RequestConstants,
    StructuredOutputRequest,
    ToolOutput,
    _data_url,
    append_notices_to_content,
    extract_finish_reason,
    extract_response_model,
    extract_token_usage,
    render_model_notices,
)
from .transcript import (
    ToolResultEntry,
    TranscriptEntry,
    UserEntry,
    _append_assistant_turn,
    _append_tool_results,
    _thinking_fallback,
    _transcript_state,
    _validate_resume_state,
)
from .transport import Provider


class OpenAIProvider(Provider):
    """Provider for OpenAI-compatible Responses endpoints."""

    name = "OpenAI"
    api_key_env = "OPENAI_API_KEY"
    default_base_url = "https://api.openai.com/v1"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: int = 120,
        request_retries: int = 3,
        request_retry_backoff: float = 1.0,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__(
            api_key=api_key,
            base_url=base_url or os.getenv("OPENAI_BASE_URL"),
            timeout=timeout,
            request_retries=request_retries,
            request_retry_backoff=request_retry_backoff,
            http_client=http_client,
        )

    async def create_response(self, payload: Json) -> Json:
        """Create a Responses API response."""
        return await self.post_json("/responses", payload)


def _openai_supports_encrypted_reasoning(model_name: str) -> bool:
    """Whether an OpenAI model returns encrypted reasoning content.

    Mirrors pydantic-ai's profile detection (``profiles/openai.py``): only reasoning
    models accept ``include=["reasoning.encrypted_content"]``; non-reasoning models 400.
    Like pydantic-ai, this enumerates known families explicitly, so a future reasoning
    family must be added here (until then it degrades to text on resume rather than 400).
    """
    is_gpt_5_1_plus = model_name.startswith(("gpt-5.1", "gpt-5.2", "gpt-5.3", "gpt-5.4", "gpt-5.5"))
    is_gpt_5 = model_name.startswith("gpt-5") and not is_gpt_5_1_plus
    is_o_series = model_name.startswith("o")
    is_gpt_5_3_chat = model_name.startswith("gpt-5.3-chat")
    thinking_always_enabled = is_o_series or (is_gpt_5 and "-chat" not in model_name)
    return (thinking_always_enabled or is_gpt_5_1_plus) and not is_gpt_5_3_chat


class OpenAIResponsesModel:
    """Responses-like model implemented with OpenAI Responses."""

    capabilities = ModelCapabilities(supports_json_schema_output=True, default_structured_output_mode="native")
    resume_kind = "openai"

    def __init__(self, model: str, *, provider: OpenAIProvider | None = None, settings: ModelSettings | None = None) -> None:
        self.model = model
        self.provider = provider or OpenAIProvider()
        self.settings = settings or ModelSettings()

    @property
    def api_key(self) -> str | None:
        """Return the provider API key."""
        return self.provider.api_key

    def new_session(self) -> ModelSession:
        """Create an isolated Responses API session."""
        return OpenAIResponsesSession(self)

    def resume_session(self, state: dict[str, Any]) -> ModelSession:
        """Create an isolated Responses API session from resume state."""
        entries = _validate_resume_state(state)
        session = OpenAIResponsesSession(self)
        session.transcript = copy.deepcopy(entries)
        session._pending_replay = copy.deepcopy(entries)
        return session

    def build_payload(
        self,
        *,
        input_payload: Any,
        tools: list[Json],
        instructions: str | None = None,
        metadata: Json | None = None,
        structured_output: StructuredOutputRequest | None = None,
    ) -> Json:
        """Build a Responses API payload.

        ``extra_body`` is applied after tuning settings, so callers can override
        provider request knobs while native structured output remains enforced.
        """
        payload: Json = {"model": self.model, "input": input_payload, "tools": tools}
        if _openai_supports_encrypted_reasoning(self.model):
            payload["include"] = ["reasoning.encrypted_content"]
        if instructions:
            payload["instructions"] = instructions
        if metadata:
            payload["metadata"] = metadata
        if self.settings.temperature is not None:
            payload["temperature"] = self.settings.temperature
        if self.settings.max_tokens is not None:
            payload["max_output_tokens"] = self.settings.max_tokens
        if self.settings.effort is not None:
            payload["reasoning"] = {"effort": self.settings.effort}
        payload.update(self.settings.extra_body)
        if structured_output is not None:
            payload["text"] = _structured_output_to_openai_text_format(structured_output)
        return payload


class OpenAIResponsesSession:
    """Per-run OpenAI Responses state."""

    def __init__(self, model: OpenAIResponsesModel) -> None:
        self.model = model
        self.previous_response_id: str | None = None
        self.transcript: list[TranscriptEntry] = []
        self._pending_replay: list[TranscriptEntry] | None = None

    async def start(
        self,
        prompt: Prompt,
        constants: RequestConstants,
        *,
        previous_response_id: str | None = None,
        notices: list[ModelNotice] | None = None,
    ) -> ModelTurn:
        """Start a Responses API run."""
        self.previous_response_id = previous_response_id
        content = append_notices_to_content(normalize_content(prompt, label="prompt"), notices)
        self.transcript = [UserEntry(content=content)]
        input_payload = _openai_user_input(content)
        payload = self.model.build_payload(
            input_payload=input_payload,
            instructions=constants.instructions,
            tools=constants.tools,
            metadata=constants.metadata,
            structured_output=constants.structured_output,
        )
        if self.previous_response_id:
            payload["previous_response_id"] = self.previous_response_id
        return await self._complete(payload)

    async def continue_with_tools(
        self,
        outputs: list[ToolOutput],
        constants: RequestConstants,
        *,
        notices: list[ModelNotice] | None = None,
    ) -> ModelTurn:
        """Continue a Responses API run with function_call_output items."""
        input_payload: list[Json] = [
            {
                "type": "function_call_output",
                "call_id": output.call_id,
                "output": output.wire_output if output.wire_output is not None else _openai_tool_output(output.result),
            }
            for output in outputs
        ]
        notice_text = render_model_notices(notices)
        if notice_text:
            input_payload.append({
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": notice_text}],
            })
        _append_tool_results(self.transcript, outputs, notice_text)
        replay_input = self._prepend_replay(input_payload)
        payload = self.model.build_payload(
            input_payload=replay_input,
            instructions=constants.instructions,
            tools=constants.tools,
            metadata=constants.metadata,
            structured_output=constants.structured_output,
        )
        if self.previous_response_id:
            payload["previous_response_id"] = self.previous_response_id
        return await self._complete(payload)

    async def continue_with_user_content(
        self,
        content: Prompt,
        constants: RequestConstants,
        *,
        notices: list[ModelNotice] | None = None,
    ) -> ModelTurn:
        """Continue a Responses API run with user content."""
        normalized = append_notices_to_content(normalize_content(content), notices)
        self.transcript.append(UserEntry(content=normalized))
        input_payload = _openai_user_input(normalized)
        payload = self.model.build_payload(
            input_payload=self._prepend_replay(input_payload),
            instructions=constants.instructions,
            tools=constants.tools,
            metadata=constants.metadata,
            structured_output=constants.structured_output,
        )
        if self.previous_response_id:
            payload["previous_response_id"] = self.previous_response_id
        return await self._complete(payload)

    def dump_state(self) -> dict[str, Any] | None:
        """Serialize the neutral transcript for resume."""
        from . import provider_prefix

        return _transcript_state(
            model=self.model,
            origin_provider=provider_prefix(self.model.provider.name),
            entries=self.transcript,
        )

    async def _complete(self, payload: Json) -> ModelTurn:
        """Send a Responses API payload and normalize the response."""
        response = await self.model.provider.create_response(payload)
        self.previous_response_id = response.get("id") or self.previous_response_id
        turn = ModelTurn(
            text=_extract_responses_text(response),
            tool_calls=_extract_responses_tool_calls(response),
            reasoning=_extract_responses_reasoning(response),
            raw=response,
            usage=extract_token_usage(response),
            finish_reason=extract_finish_reason(response),
            response_model=extract_response_model(response),
        )
        _append_assistant_turn(self.transcript, turn)
        return turn

    def _prepend_replay(self, input_payload: str | list[Json]) -> str | list[Json]:
        if self._pending_replay is None:
            return input_payload
        replay = _render_openai_transcript(self._pending_replay, encrypted_reasoning_ok=_openai_supports_encrypted_reasoning(self.model.model))
        self._pending_replay = None
        if isinstance(input_payload, str):
            return [*replay, _openai_user_item((TextBlock(input_payload),))]
        return [*replay, *input_payload]


def _openai_content_parts(content: NormalizedContent) -> list[Json]:
    """Map neutral content to OpenAI Responses content parts."""
    return [
        {"type": "input_text", "text": block.text}
        if isinstance(block, TextBlock)
        else {"type": "input_image", "image_url": _data_url(block)}
        for block in content
    ]


def _openai_user_input(content: NormalizedContent) -> str | list[Json]:
    """Keep all-text input scalar and use a message item for images."""
    text = text_only_value(content)
    return text if text is not None else [_openai_user_item(content)]


def _openai_tool_output(result: ToolResult) -> str | list[Json]:
    """Map one canonical result to Responses function-call output."""
    if not result.has_image:
        return result.to_json()
    header = json.dumps({"ok": result.ok, "metadata": result.metadata}, ensure_ascii=False, separators=(",", ":"))
    return [{"type": "input_text", "text": header}, *_openai_content_parts(result.blocks)]


def _openai_user_item(content: NormalizedContent) -> Json:
    """Render one Responses API user message item."""
    text = text_only_value(content)
    parts = [{"type": "input_text", "text": text}] if text is not None else _openai_content_parts(content)
    return {"type": "message", "role": "user", "content": parts}


def _render_openai_transcript(entries: list[TranscriptEntry], *, encrypted_reasoning_ok: bool = False) -> list[Json]:
    """Render neutral transcript entries as Responses API input items."""
    items: list[Json] = []
    for entry in entries:
        if isinstance(entry, UserEntry):
            items.append(_openai_user_item(entry.content))
        elif isinstance(entry, ToolResultEntry):
            items.append({
                "type": "function_call_output",
                "call_id": entry.call_id,
                "output": entry.wire_output if entry.wire_output is not None else _openai_tool_output(entry.result),
            })
        else:
            for part in entry.reasoning:
                if encrypted_reasoning_ok and part.provider_name == "openai" and part.signature and part.id:
                    items.append({"type": "reasoning", "id": part.id, "encrypted_content": part.signature, "summary": []})
                elif part.text:
                    items.append({"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": _thinking_fallback(part.text)}]})
            if entry.text:
                items.append({"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": entry.text}]})
            items.extend({
                "type": "function_call",
                "call_id": call.id,
                "name": call.name,
                "arguments": call.arguments,
            } for call in entry.tool_calls)
    return items


def _structured_output_to_openai_text_format(request: StructuredOutputRequest) -> Json:
    """Convert neutral structured-output metadata to Responses text.format."""
    json_schema: Json = {"name": request.name, "schema": request.schema, "strict": request.strict}
    if request.description:
        json_schema["description"] = request.description
    return {"format": {"type": "json_schema", **json_schema}}


def _extract_responses_tool_calls(response: Json) -> list[ModelToolCall]:
    """Extract normalized tool calls from a Responses API response."""
    calls = []
    for item in response.get("output", []) or []:
        if item.get("type") in {"function_call", "tool_call"}:
            calls.append(ModelToolCall(id=str(item.get("call_id") or item.get("id")), name=str(item.get("name")), arguments=item.get("arguments") or "{}"))
    return calls


def _extract_responses_text(response: Json) -> str:
    """Extract text from a Responses API response."""
    if isinstance(response.get("output_text"), str):
        return response["output_text"]
    chunks: list[str] = []
    for item in response.get("output", []) or []:
        for content in item.get("content", []) or []:
            if isinstance(content, dict) and content.get("type") in {"output_text", "text"}:
                chunks.append(str(content.get("text", "")))
    return "".join(chunks)


def _extract_responses_reasoning(response: Json) -> list[ReasoningPart]:
    """Extract native reasoning items from a Responses API response."""
    parts: list[ReasoningPart] = []
    for item in response.get("output", []) or []:
        if not isinstance(item, dict) or item.get("type") != "reasoning":
            continue
        summary = item.get("summary") or []
        text = "".join(str(chunk.get("text", "")) for chunk in summary if isinstance(chunk, dict))
        content = item.get("content")
        parts.append(ReasoningPart(
            text=text,
            signature=item.get("encrypted_content"),
            id=item.get("id"),
            provider_name="openai",
            provider_details={"raw_content": content} if content is not None else None,
        ))
    return parts
