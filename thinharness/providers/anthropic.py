"""Anthropic Messages provider, model, session, and wire dialect."""

from __future__ import annotations

import base64
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
    append_notices_to_content,
    extract_finish_reason,
    extract_response_model,
    extract_token_usage,
    render_model_notices,
)
from .transcript import (
    AssistantEntry,
    ToolResultEntry,
    TranscriptEntry,
    UserEntry,
    _append_assistant_turn,
    _append_tool_results,
    _thinking_fallback,
    _transcript_state,
    _validate_anthropic_tool_arguments,
    _validate_resume_state,
)
from .transport import Provider, ProviderError

DEFAULT_ANTHROPIC_MAX_TOKENS = 16384
_ANTHROPIC_THINKING_DEFAULT_OFF_PREFIXES = ("claude-opus-4", "claude-sonnet-4", "claude-haiku-4", "claude-3")


class AnthropicProvider(Provider):
    """Provider for Anthropic Messages endpoints."""

    name = "Anthropic"
    api_key_env = "ANTHROPIC_API_KEY"
    default_base_url = "https://api.anthropic.com/v1"

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
            base_url=base_url or os.getenv("ANTHROPIC_BASE_URL"),
            timeout=timeout,
            request_retries=request_retries,
            request_retry_backoff=request_retry_backoff,
            http_client=http_client,
        )

    def headers(self) -> Json:
        """Return Anthropic auth headers."""
        if not self.api_key:
            raise ProviderError("ANTHROPIC_API_KEY is required for Anthropic")
        return {"x-api-key": self.api_key, "anthropic-version": "2023-06-01", "Content-Type": "application/json"}

    async def create_message(self, payload: Json) -> Json:
        """Create an Anthropic Messages response."""
        return await self.post_json("/messages", payload)


class AnthropicMessagesModel:
    """Responses-like model implemented with Anthropic Messages."""

    capabilities = ModelCapabilities(supports_json_schema_output=True, default_structured_output_mode="native")
    resume_kind = "anthropic"

    def __init__(
        self,
        model: str,
        *,
        provider: AnthropicProvider | None = None,
        settings: ModelSettings | None = None,
        max_tokens: int | None = None,
    ) -> None:
        self.model = model
        self.provider = provider or AnthropicProvider()
        self.settings = settings or ModelSettings()
        self.max_tokens = max_tokens

    @property
    def api_key(self) -> str | None:
        """Return the provider API key."""
        return self.provider.api_key

    def new_session(self) -> ModelSession:
        """Create an isolated Anthropic Messages session."""
        return AnthropicMessagesSession(self)

    def resume_session(self, state: dict[str, Any]) -> ModelSession:
        """Create an isolated Anthropic Messages session from resume state."""
        entries = _validate_resume_state(state)
        _validate_anthropic_tool_arguments(entries)
        session = AnthropicMessagesSession(self)
        session.transcript = copy.deepcopy(entries)
        session._resume_entries = copy.deepcopy(entries)
        return session


class AnthropicMessagesSession:
    """Per-run Anthropic Messages state."""

    def __init__(self, model: AnthropicMessagesModel) -> None:
        self.model = model
        self.messages: list[Json] = []
        self.system = ""
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
        """Start an Anthropic Messages run."""
        if previous_response_id:
            raise ProviderError("previous_response_id is only supported by OpenAI Responses")
        self.system = constants.instructions
        content = append_notices_to_content(normalize_content(prompt, label="prompt"), notices)
        self.messages = [{"role": "user", "content": _anthropic_user_content(content)}]
        self.transcript = [UserEntry(content=content)]
        return await self._complete(tools=constants.tools, metadata=constants.metadata, structured_output=constants.structured_output)

    async def continue_with_tools(
        self,
        outputs: list[ToolOutput],
        constants: RequestConstants,
        *,
        notices: list[ModelNotice] | None = None,
    ) -> ModelTurn:
        """Continue an Anthropic Messages run with tool_result blocks."""
        content = [
            {
                "type": "tool_result",
                "tool_use_id": output.call_id,
                "content": output.wire_output if output.wire_output is not None else _anthropic_tool_output(output.result),
            }
            for output in outputs
        ]
        notice_text = render_model_notices(notices)
        if notice_text:
            content.append({"type": "text", "text": notice_text})
        _append_tool_results(self.transcript, outputs, notice_text)
        self._apply_resume(constants.instructions)
        self.messages.append({
            "role": "user",
            "content": content,
        })
        return await self._complete(tools=constants.tools, metadata=constants.metadata, structured_output=constants.structured_output)

    async def continue_with_user_content(
        self,
        content: Prompt,
        constants: RequestConstants,
        *,
        notices: list[ModelNotice] | None = None,
    ) -> ModelTurn:
        """Continue an Anthropic Messages run with user content."""
        normalized = append_notices_to_content(normalize_content(content), notices)
        self.transcript.append(UserEntry(content=normalized))
        self._apply_resume(constants.instructions)
        self.messages.append({"role": "user", "content": _anthropic_user_content(normalized)})
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
        """Send a Messages API request and normalize the response.

        ``extra_body`` overrides tuning keys. Native structured-output format is
        deep-set afterward so the harness never validates against an omitted schema.
        """
        payload: Json = {
            "model": self.model.model,
            "max_tokens": self.model.max_tokens
            if self.model.max_tokens is not None
            else self.model.settings.max_tokens
            if self.model.settings.max_tokens is not None
            else DEFAULT_ANTHROPIC_MAX_TOKENS,
            "system": self.system,
            "messages": self.messages,
            "tools": [_responses_tool_to_anthropic(tool) for tool in tools],
            # Top-level auto-caching: the API places the prompt-cache breakpoint
            # on the last cacheable block, so the growing prefix is reused.
            "cache_control": {"type": "ephemeral"},
        }
        # Anthropic accepts only user_id in request metadata.
        anthropic_user_id = metadata.get("user_id") if metadata is not None else None
        if isinstance(anthropic_user_id, str):
            payload["metadata"] = {"user_id": anthropic_user_id}
        if self.model.settings.temperature is not None:
            payload["temperature"] = self.model.settings.temperature
        if self.model.settings.effort is not None:
            payload["output_config"] = {"effort": self.model.settings.effort}
            payload["thinking"] = {"type": "adaptive"}
        payload.update(self.model.settings.extra_body)
        if structured_output is not None:
            output_config = payload.setdefault("output_config", {})
            if not isinstance(output_config, dict):
                output_config = {}
                payload["output_config"] = output_config
            output_config["format"] = _structured_output_to_anthropic_format(structured_output)
        response = await self.model.provider.create_message(payload)
        self.messages.append({"role": "assistant", "content": response.get("content", [])})
        turn = ModelTurn(
            text=_extract_anthropic_text(response),
            tool_calls=_extract_anthropic_tool_calls(response),
            reasoning=_extract_anthropic_reasoning(response),
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
        self.system = instructions or ""
        self.messages = _render_anthropic_transcript(
            self._resume_entries,
            thinking_enabled=_anthropic_thinking_enabled(self.model),
        )
        self._resume_entries = None


def _anthropic_content_parts(content: NormalizedContent) -> list[Json]:
    """Map neutral content to Anthropic content blocks."""
    return [
        {"type": "text", "text": block.text}
        if isinstance(block, TextBlock)
        else {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": block.media_type,
                "data": base64.b64encode(block.data).decode("ascii"),
            },
        }
        for block in content
    ]


def _anthropic_user_content(content: NormalizedContent) -> str | list[Json]:
    """Keep all-text messages scalar and use blocks for images."""
    return text_only_value(content) or _anthropic_content_parts(content)


def _anthropic_tool_output(result: ToolResult) -> str | list[Json]:
    """Map one canonical result to an Anthropic tool-result value."""
    if not result.has_image:
        return result.to_json()
    header = json.dumps({"ok": result.ok, "metadata": result.metadata}, ensure_ascii=False, separators=(",", ":"))
    return [{"type": "text", "text": header}, *_anthropic_content_parts(result.blocks)]


def _anthropic_thinking_on_by_default(model_name: str) -> bool:
    """Return whether this model runs thinking when the thinking key is omitted."""
    return not model_name.startswith(_ANTHROPIC_THINKING_DEFAULT_OFF_PREFIXES)


def _anthropic_thinking_enabled(model: AnthropicMessagesModel) -> bool:
    """Return whether the resuming Anthropic request runs extended thinking."""
    if "thinking" in model.settings.extra_body:
        thinking = model.settings.extra_body["thinking"]
        return isinstance(thinking, dict) and thinking.get("type") in ("enabled", "adaptive")
    return model.settings.effort is not None or _anthropic_thinking_on_by_default(model.model)


def _render_anthropic_transcript(entries: list[TranscriptEntry], *, thinking_enabled: bool = False) -> list[Json]:
    """Render neutral transcript entries as Anthropic Messages history."""
    messages: list[Json] = []
    index = 0
    while index < len(entries):
        entry = entries[index]
        if isinstance(entry, UserEntry):
            user_content = _anthropic_content_parts(entry.content) if entry.notice else _anthropic_user_content(entry.content)
            messages.append({"role": "user", "content": user_content})
            index += 1
            continue
        if isinstance(entry, AssistantEntry):
            content: list[Json] = []
            for part in entry.reasoning:
                if thinking_enabled and part.provider_name == "anthropic" and part.signature:
                    if part.id == "redacted_thinking":
                        content.append({"type": "redacted_thinking", "data": part.signature})
                    else:
                        content.append({"type": "thinking", "thinking": part.text, "signature": part.signature})
                elif part.text:
                    content.append({"type": "text", "text": _thinking_fallback(part.text)})
            if entry.text:
                content.append({"type": "text", "text": entry.text})
            content.extend({
                "type": "tool_use",
                "id": call.id,
                "name": call.name,
                "input": json.loads(call.arguments),
            } for call in entry.tool_calls)
            messages.append({"role": "assistant", "content": content})
            index += 1
            continue
        content = []
        while index < len(entries) and isinstance(entries[index], ToolResultEntry):
            tool_entry = entries[index]
            assert isinstance(tool_entry, ToolResultEntry)
            content.append({
                "type": "tool_result",
                "tool_use_id": tool_entry.call_id,
                "content": tool_entry.wire_output if tool_entry.wire_output is not None else _anthropic_tool_output(tool_entry.result),
            })
            index += 1
        if index < len(entries):
            notice_entry = entries[index]
            if isinstance(notice_entry, UserEntry) and notice_entry.notice:
                content.extend(_anthropic_content_parts(notice_entry.content))
                index += 1
        messages.append({"role": "user", "content": content})
    return messages


def _responses_tool_to_anthropic(tool: Json) -> Json:
    """Convert a Responses API function tool to Anthropic format."""
    return {
        "name": tool["name"],
        "description": tool.get("description", ""),
        "input_schema": tool.get("parameters", {"type": "object", "properties": {}}),
    }


def _structured_output_to_anthropic_format(request: StructuredOutputRequest) -> Json:
    """Convert neutral structured-output metadata to Anthropic output_config.format."""
    return {"type": "json_schema", "schema": request.schema}


def _extract_anthropic_tool_calls(response: Json) -> list[ModelToolCall]:
    """Extract normalized tool calls from an Anthropic response."""
    calls = []
    for block in response.get("content", []) or []:
        if block.get("type") == "tool_use":
            calls.append(ModelToolCall(id=str(block.get("id")), name=str(block.get("name")), arguments=json.dumps(block.get("input") or {})))
    return calls


def _extract_anthropic_text(response: Json) -> str:
    """Extract text from an Anthropic response."""
    return "".join(str(block.get("text", "")) for block in response.get("content", []) or [] if block.get("type") == "text")


def _extract_anthropic_reasoning(response: Json) -> list[ReasoningPart]:
    """Extract native thinking blocks from an Anthropic response."""
    parts: list[ReasoningPart] = []
    for block in response.get("content", []) or []:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "thinking":
            parts.append(ReasoningPart(text=str(block.get("thinking", "")), signature=block.get("signature"), provider_name="anthropic"))
        elif block.get("type") == "redacted_thinking":
            parts.append(ReasoningPart(text="", signature=block.get("data"), id="redacted_thinking", provider_name="anthropic"))
    return parts
