"""OpenRouter provider, model, session, and wire dialect."""

from __future__ import annotations

import copy
import json
import os
from typing import Any

import httpx

from ..content import ImageBlock, NormalizedContent, Prompt, TextBlock, normalize_content, text_only_value
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
from .transport import Provider, ProviderError


class OpenRouterProvider(Provider):
    """Provider for OpenRouter's OpenAI-compatible gateway."""

    name = "OpenRouter"
    api_key_env = "OPENROUTER_API_KEY"
    default_base_url = "https://openrouter.ai/api/v1"

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
            base_url=base_url or os.getenv("OPENROUTER_BASE_URL"),
            timeout=timeout,
            request_retries=request_retries,
            request_retry_backoff=request_retry_backoff,
            http_client=http_client,
        )

    def headers(self) -> Json:
        """Return OpenRouter auth and attribution headers."""
        headers = super().headers()
        if app_url := os.getenv("OPENROUTER_APP_URL"):
            headers["HTTP-Referer"] = app_url
        if app_title := os.getenv("OPENROUTER_APP_TITLE"):
            headers["X-Title"] = app_title
        return headers

    async def create_chat_completion(self, payload: Json) -> Json:
        """Create an OpenRouter chat completion."""
        return await self.post_json("/chat/completions", payload)


class OpenRouterModel:
    """Responses-like model implemented through OpenRouter chat completions."""

    capabilities = ModelCapabilities(
        supports_json_schema_output=False,
        supports_tools=True,
        permissive_native_override=True,
        default_structured_output_mode="tool",
    )
    resume_kind = "openrouter"

    def __init__(self, model: str, *, provider: OpenRouterProvider | None = None, settings: ModelSettings | None = None) -> None:
        self.model = model
        self.provider = provider or OpenRouterProvider()
        self.settings = settings or ModelSettings()

    @property
    def api_key(self) -> str | None:
        """Return the provider API key."""
        return self.provider.api_key

    def new_session(self) -> ModelSession:
        """Create an isolated OpenRouter session."""
        return OpenRouterSession(self)

    def resume_session(self, state: dict[str, Any]) -> ModelSession:
        """Create an isolated OpenRouter session from resume state."""
        entries = _validate_resume_state(state)
        session = OpenRouterSession(self)
        session.transcript = copy.deepcopy(entries)
        session._resume_entries = copy.deepcopy(entries)
        return session


class OpenRouterSession:
    """Per-run OpenRouter chat completion state."""

    def __init__(self, model: OpenRouterModel) -> None:
        self.model = model
        self.messages: list[Json] = []
        self.transcript: list[TranscriptEntry] = []
        self._resume_entries: list[TranscriptEntry] | None = None

    async def start(
        self,
        prompt: Prompt,
        constants: RequestConstants,
        *,
        previous_response_id: str | None = None,
        notices: list[ModelNotice] | None = None,
    ) -> ModelTurn:
        """Start an OpenRouter run."""
        if previous_response_id:
            raise ProviderError("previous_response_id is only supported by OpenAI Responses")
        content = append_notices_to_content(normalize_content(prompt, label="prompt"), notices)
        self.messages = [
            {"role": "system", "content": constants.instructions},
            {"role": "user", "content": _openrouter_user_content(content)},
        ]
        self.transcript = [UserEntry(content=content)]
        return await self._complete(tools=constants.tools, metadata=constants.metadata, structured_output=constants.structured_output)

    async def continue_with_tools(
        self,
        outputs: list[ToolOutput],
        constants: RequestConstants,
        *,
        notices: list[ModelNotice] | None = None,
    ) -> ModelTurn:
        """Continue an OpenRouter run with tool messages."""
        notice_text = render_model_notices(notices)
        _append_tool_results(self.transcript, outputs, notice_text)
        self._apply_resume(constants.instructions)
        for output in outputs:
            self.messages.append({
                "role": "tool",
                "tool_call_id": output.call_id,
                "content": output.wire_output if output.wire_output is not None else _openrouter_tool_output_json(output.result),
            })
        image_parts = _openrouter_tool_image_parts(outputs)
        if image_parts:
            if notice_text:
                image_parts.append({"type": "text", "text": notice_text})
            self.messages.append({"role": "user", "content": image_parts})
        elif notice_text:
            self.messages.append({"role": "user", "content": notice_text})
        return await self._complete(tools=constants.tools, metadata=constants.metadata, structured_output=constants.structured_output)

    async def continue_with_user_content(
        self,
        content: Prompt,
        constants: RequestConstants,
        *,
        notices: list[ModelNotice] | None = None,
    ) -> ModelTurn:
        """Continue an OpenRouter run with user content."""
        normalized = append_notices_to_content(normalize_content(content), notices)
        self.transcript.append(UserEntry(content=normalized))
        self._apply_resume(constants.instructions)
        self.messages.append({"role": "user", "content": _openrouter_user_content(normalized)})
        return await self._complete(tools=constants.tools, metadata=constants.metadata, structured_output=constants.structured_output)

    def dump_state(self) -> dict[str, Any] | None:
        """Serialize the neutral transcript for resume."""
        from . import provider_prefix

        return _transcript_state(
            model=self.model,
            origin_provider=provider_prefix(self.model.provider.name),
            entries=self.transcript,
        )

    async def _complete(
        self,
        *,
        tools: list[Json],
        metadata: Json | None = None,
        structured_output: StructuredOutputRequest | None = None,
    ) -> ModelTurn:
        """Send an OpenRouter request and normalize the response.

        ``extra_body`` is applied after tuning settings, so callers can override
        provider request knobs while native structured output remains enforced.
        """
        payload: Json = {
            "model": self.model.model,
            "messages": self.messages,
            "tools": [_responses_tool_to_chat(tool) for tool in tools],
        }
        if metadata:
            payload["metadata"] = metadata
        if self.model.settings.temperature is not None:
            payload["temperature"] = self.model.settings.temperature
        if self.model.settings.max_tokens is not None:
            payload["max_tokens"] = self.model.settings.max_tokens
        if self.model.settings.effort is not None:
            payload["reasoning"] = {"effort": self.model.settings.effort}
        payload.update(self.model.settings.extra_body)
        if structured_output is not None:
            payload["response_format"] = _structured_output_to_openrouter_response_format(structured_output)
        response = await self.model.provider.create_chat_completion(payload)
        message = ((response.get("choices") or [{}])[0].get("message") or {})
        self.messages.append(message)
        turn = ModelTurn(
            text=str(message.get("content") or ""),
            tool_calls=_extract_chat_tool_calls(message),
            reasoning=_extract_openrouter_reasoning(message),
            raw=response,
            usage=extract_token_usage(response),
            finish_reason=extract_finish_reason(response),
            response_model=extract_response_model(response),
        )
        _append_assistant_turn(self.transcript, turn)
        return turn

    def _apply_resume(self, instructions: str | None) -> None:
        if self._resume_entries is None:
            return
        self.messages = [{"role": "system", "content": instructions or ""}, *_render_openrouter_transcript(self._resume_entries)]
        self._resume_entries = None


def _openrouter_content_parts(content: NormalizedContent) -> list[Json]:
    """Map neutral content to OpenRouter chat parts."""
    return [
        {"type": "text", "text": block.text}
        if isinstance(block, TextBlock)
        else {"type": "image_url", "image_url": {"url": _data_url(block)}}
        for block in content
    ]


def _openrouter_user_content(content: NormalizedContent) -> str | list[Json]:
    """Keep all-text messages scalar and use parts for images."""
    return text_only_value(content) or _openrouter_content_parts(content)


def _openrouter_tool_output_json(result: ToolResult) -> str:
    """Serialize a tool result with image descriptors instead of image bytes."""
    if not result.has_image:
        return result.to_json()
    projected: list[Json] = []
    for index, block in enumerate(result.blocks):
        if isinstance(block, TextBlock):
            projected.append({"type": "text", "text": block.text})
        else:
            projected.append({
                "type": "image",
                "media_type": block.media_type,
                "size_bytes": len(block.data),
                "block_index": index,
            })
    return json.dumps({"ok": result.ok, "content": projected, "metadata": result.metadata}, ensure_ascii=False)


def _openrouter_tool_image_parts(outputs: list[ToolOutput]) -> list[Json]:
    """Build labelled user-message parts for an ordered tool batch."""
    parts: list[Json] = []
    for output in outputs:
        for index, block in enumerate(output.result.blocks):
            if not isinstance(block, ImageBlock):
                continue
            parts.extend([
                {"type": "text", "text": f"[tool image call_id={output.call_id} block={index}]"},
                {"type": "image_url", "image_url": {"url": _data_url(block)}},
            ])
    return parts


def _render_openrouter_transcript(entries: list[TranscriptEntry]) -> list[Json]:
    """Render neutral transcript entries as OpenRouter chat history."""
    messages: list[Json] = []
    index = 0
    while index < len(entries):
        entry = entries[index]
        if isinstance(entry, UserEntry):
            messages.append({"role": "user", "content": _openrouter_user_content(entry.content)})
            index += 1
            continue
        if isinstance(entry, ToolResultEntry):
            outputs: list[ToolOutput] = []
            while index < len(entries) and isinstance(entries[index], ToolResultEntry):
                tool_entry = entries[index]
                assert isinstance(tool_entry, ToolResultEntry)
                output = ToolOutput(tool_entry.call_id, tool_entry.result, wire_output=tool_entry.wire_output)
                outputs.append(output)
                messages.append({
                    "role": "tool",
                    "tool_call_id": output.call_id,
                    "content": output.wire_output if output.wire_output is not None else _openrouter_tool_output_json(output.result),
                })
                index += 1
            notice: UserEntry | None = None
            if index < len(entries):
                candidate = entries[index]
                if isinstance(candidate, UserEntry) and candidate.notice:
                    notice = candidate
                    index += 1
            image_parts = _openrouter_tool_image_parts(outputs)
            if image_parts:
                if notice is not None:
                    image_parts.extend(_openrouter_content_parts(notice.content))
                messages.append({"role": "user", "content": image_parts})
            elif notice is not None:
                messages.append({"role": "user", "content": _openrouter_user_content(notice.content)})
            continue
        message: Json = {"role": "assistant"}
        reasoning_details = [
            part.provider_details
            for part in entry.reasoning
            if part.provider_name == "openrouter" and part.provider_details is not None
        ]
        fallback_blocks = [
            _thinking_fallback(part.text)
            for part in entry.reasoning
            if not (part.provider_name == "openrouter" and part.provider_details is not None) and part.text
        ]
        if reasoning_details:
            message["reasoning_details"] = reasoning_details
        text = "\n\n".join([*fallback_blocks, *([entry.text] if entry.text else [])])
        if text:
            message["content"] = text
        if entry.tool_calls:
            message["tool_calls"] = [
                {"id": call.id, "type": "function", "function": {"name": call.name, "arguments": call.arguments}}
                for call in entry.tool_calls
            ]
        if not text and not entry.tool_calls:
            message["content"] = ""
        messages.append(message)
        index += 1
    return messages


def _responses_tool_to_chat(tool: Json) -> Json:
    """Convert a Responses API function tool to Chat Completions format."""
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool.get("parameters", {"type": "object", "properties": {}}),
        },
    }


def _structured_output_to_openrouter_response_format(request: StructuredOutputRequest) -> Json:
    """Convert neutral structured-output metadata to OpenRouter response_format."""
    json_schema: Json = {"name": request.name, "schema": request.schema, "strict": request.strict}
    if request.description:
        json_schema["description"] = request.description
    return {"type": "json_schema", "json_schema": json_schema}


def _extract_chat_tool_calls(message: Json) -> list[ModelToolCall]:
    """Extract normalized tool calls from a chat completion message."""
    calls = []
    for call in message.get("tool_calls", []) or []:
        function = call.get("function") or {}
        calls.append(ModelToolCall(id=str(call.get("id")), name=str(function.get("name")), arguments=function.get("arguments") or "{}"))
    return calls


def _extract_openrouter_reasoning(message: Json) -> list[ReasoningPart]:
    """Extract native reasoning_details from an OpenRouter chat message."""
    parts: list[ReasoningPart] = []
    for entry in message.get("reasoning_details") or []:
        if not isinstance(entry, dict):
            continue
        parts.append(ReasoningPart(
            text=str(entry.get("text") or ""),
            signature=entry.get("signature") or entry.get("data"),
            id=entry.get("id"),
            provider_name="openrouter",
            provider_details=entry,
        ))
    return parts
