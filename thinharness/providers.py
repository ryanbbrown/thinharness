"""Provider and model implementations for normalized agent turns."""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import httpx
from pydantic import BaseModel, Field

from .tools.base import Json
from .types import HarnessError

# =============================================================================
# Normalized model types
# =============================================================================


@dataclass
class ModelToolCall:
    """A normalized model tool call."""

    id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class TokenUsage:
    """Normalized provider token usage; fields are None when unreported."""

    input_tokens: int | None = None
    output_tokens: int | None = None


@dataclass
class ReasoningPart:
    """Provider-neutral carrier for one native reasoning block.

    The opaque blob is re-emitted natively only when ``provider_name`` matches the
    resuming provider; otherwise ``text`` is replayed as a leading ``<thinking>`` block.
    """

    text: str = ""                       # plain reasoning text — always kept; cross-provider fallback
    signature: str | None = None         # opaque blob: Anthropic signature / redacted data,
    #                                      OpenAI encrypted_content, OpenRouter signature|data
    id: str | None = None                # provider reasoning-item id (OpenAI rs_…; "redacted_thinking" marker)
    provider_name: str | None = None     # origin provider prefix; native re-emit only when this matches
    provider_details: Json | None = None  # spillover: OpenAI summary raw_content; OpenRouter raw reasoning_details entry


@dataclass
class ModelTurn:
    """A normalized model response turn."""

    text: str = ""
    tool_calls: list[ModelToolCall] = field(default_factory=list)
    raw: Json = field(default_factory=dict)
    reasoning: list[ReasoningPart] = field(default_factory=list)
    usage: TokenUsage | None = None
    finish_reason: str | None = None
    response_model: str | None = None


@dataclass
class AssistantEntry:
    """Provider-neutral assistant transcript entry."""

    text: str
    tool_calls: list[ModelToolCall]
    reasoning: list[ReasoningPart] = field(default_factory=list)


@dataclass
class UserEntry:
    """Provider-neutral user transcript entry."""

    content: str
    notice: bool = False


@dataclass
class ToolResultEntry:
    """Provider-neutral tool-result transcript entry."""

    call_id: str
    output: str


TranscriptEntry = AssistantEntry | UserEntry | ToolResultEntry


@dataclass
class ToolOutput:
    """A normalized local tool output."""

    call_id: str
    output: str


@dataclass(frozen=True)
class ModelNotice:
    """Provider-neutral notice appended to model input."""

    kind: Literal["limit_warning"]
    content: str
    limit_kind: Literal["model_requests", "tool_calls"] | None = None
    remaining: int | None = None


@dataclass(frozen=True)
class StructuredOutputRequest:
    """Provider-neutral structured-output request metadata."""

    name: str
    schema: Json
    strict: bool = True
    description: str | None = None


@dataclass(frozen=True)
class RequestConstants:
    """Per-run request constants passed to every ModelSession request.

    Built once per run after run-start hooks and MCP connection, so the
    toolset and instructions are frozen for the run.
    """

    instructions: str
    tools: list[Json]
    metadata: Json | None = None
    structured_output: StructuredOutputRequest | None = None


class ModelCapabilities(BaseModel):
    """Provider capability flags used by the harness."""

    supports_json_schema_output: bool = False
    supports_tools: bool = True
    permissive_native_override: bool = False
    default_structured_output_mode: Literal["native", "tool", "prompted"] = "tool"


class ModelSettings(BaseModel):
    """Common request settings shared across models."""

    temperature: float | None = None
    extra_body: Json = Field(default_factory=dict)


class Model(Protocol):
    """Responses-like model contract consumed by the harness."""

    model: str

    @property
    def provider(self) -> Provider:
        """Return the model provider."""
        ...

    @property
    def api_key(self) -> str | None:
        """Return the model provider API key."""
        ...

    def new_session(self) -> ModelSession:
        """Create isolated state for one model run."""
        ...


class ResumableModel(Model, Protocol):
    """A Model that supports resume_from on Harness.run()."""

    resume_kind: str

    def resume_session(self, state: dict[str, Any]) -> ModelSession:
        """Create isolated state from a prior run's resume_state."""
        ...


class ModelSession(Protocol):
    """Per-run model state consumed by the harness."""

    async def start(
        self,
        prompt: str,
        constants: RequestConstants,
        *,
        previous_response_id: str | None = None,
        notices: list[ModelNotice] | None = None,
    ) -> ModelTurn:
        """Start a model run."""
        ...

    async def continue_with_tools(
        self,
        outputs: list[ToolOutput],
        constants: RequestConstants,
        *,
        notices: list[ModelNotice] | None = None,
    ) -> ModelTurn:
        """Continue a model run with tool outputs."""
        ...

    async def continue_with_user_text(
        self,
        text: str,
        constants: RequestConstants,
        *,
        notices: list[ModelNotice] | None = None,
    ) -> ModelTurn:
        """Continue a model run with user text (a correction or a resumed prompt)."""
        ...

    def dump_state(self) -> dict[str, Any] | None:
        """Serialize session state for resume, or None if unavailable."""
        ...


class ProviderError(RuntimeError):
    """Raised when a provider request fails."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


_TRANSCRIPT_RESUME_KEYS = frozenset({"kind", "version", "origin_provider", "origin_model", "entries"})
_TRANSCRIPT_ENTRY_KEYS = {
    "assistant": frozenset({"role", "text", "tool_calls", "reasoning"}),
    "user": frozenset({"role", "content", "notice"}),
    "tool": frozenset({"role", "call_id", "output"}),
}
_REASONING_PART_KEYS = frozenset({"text", "signature", "id", "provider_name", "provider_details"})
_TRANSCRIPT_VERSION = 3


def _validate_resume_state(state: dict[str, Any]) -> list[TranscriptEntry]:
    """Validate built-in provider resume state shape before any session mutation."""
    if not isinstance(state, dict):
        raise HarnessError("resume_from must be a dict")
    try:
        json.dumps(state)
    except (TypeError, ValueError) as exc:
        raise HarnessError("resume_from must be JSON-serializable") from exc
    if state.get("kind") != "transcript":
        raise HarnessError(f"resume_from kind {state.get('kind')!r} is not supported; regenerate resume_state")
    if state.get("version") != _TRANSCRIPT_VERSION:
        raise HarnessError(f"resume_from version {state.get('version')!r} is not supported; regenerate resume_state")
    unknown = set(state) - _TRANSCRIPT_RESUME_KEYS
    if unknown:
        raise HarnessError(f"resume_from has unknown keys: {sorted(unknown)!r}")
    for field_name, expected_type in {"origin_provider": str, "origin_model": str, "entries": list}.items():
        if field_name not in state:
            raise HarnessError(f"resume_from missing required field: {field_name!r}")
        if not isinstance(state[field_name], expected_type):
            raise HarnessError(f"resume_from field {field_name!r} has wrong type")
    return [_transcript_entry_from_dict(entry) for entry in state["entries"]]


def _transcript_state(*, model: Model, entries: list[TranscriptEntry]) -> dict[str, Any]:
    """Return the neutral transcript resume envelope."""
    return {
        "kind": "transcript",
        "version": _TRANSCRIPT_VERSION,
        "origin_provider": provider_prefix(model.provider.name),
        "origin_model": model.model,
        "entries": [_transcript_entry_to_dict(entry) for entry in entries],
    }


def _transcript_entry_to_dict(entry: TranscriptEntry) -> Json:
    if isinstance(entry, UserEntry):
        return {"role": "user", "content": entry.content, "notice": entry.notice}
    if isinstance(entry, ToolResultEntry):
        return {"role": "tool", "call_id": entry.call_id, "output": entry.output}
    return {
        "role": "assistant",
        "text": entry.text,
        "tool_calls": [
            {"id": call.id, "name": call.name, "arguments": call.arguments}
            for call in entry.tool_calls
        ],
        "reasoning": [_reasoning_part_to_dict(part) for part in entry.reasoning],
    }


def _reasoning_part_to_dict(part: ReasoningPart) -> Json:
    data: Json = {"text": part.text}
    if part.signature is not None:
        data["signature"] = part.signature
    if part.id is not None:
        data["id"] = part.id
    if part.provider_name is not None:
        data["provider_name"] = part.provider_name
    if part.provider_details is not None:
        data["provider_details"] = part.provider_details
    return data


def _transcript_entry_from_dict(value: Any) -> TranscriptEntry:
    if not isinstance(value, dict):
        raise HarnessError("resume_from entries must be dicts")
    role = value.get("role")
    if role not in _TRANSCRIPT_ENTRY_KEYS:
        raise HarnessError(f"resume_from entry role {role!r} is not supported")
    if set(value) != _TRANSCRIPT_ENTRY_KEYS[role]:
        raise HarnessError(f"resume_from entry {role!r} has wrong keys")
    if role == "user":
        if not isinstance(value["content"], str) or type(value["notice"]) is not bool:
            raise HarnessError("resume_from user entry has wrong type")
        return UserEntry(content=value["content"], notice=value["notice"])
    if role == "tool":
        if not isinstance(value["call_id"], str) or not isinstance(value["output"], str):
            raise HarnessError("resume_from tool entry has wrong type")
        return ToolResultEntry(call_id=value["call_id"], output=value["output"])
    if not isinstance(value["text"], str) or not isinstance(value["tool_calls"], list) or not isinstance(value["reasoning"], list):
        raise HarnessError("resume_from assistant entry has wrong type")
    return AssistantEntry(
        text=value["text"],
        tool_calls=[_model_tool_call_from_dict(call) for call in value["tool_calls"]],
        reasoning=[_reasoning_part_from_dict(part) for part in value["reasoning"]],
    )


def _reasoning_part_from_dict(value: Any) -> ReasoningPart:
    if not isinstance(value, dict):
        raise HarnessError("resume_from reasoning part must be a dict")
    unknown = set(value) - _REASONING_PART_KEYS
    if unknown:
        raise HarnessError(f"resume_from reasoning part has unknown keys: {sorted(unknown)!r}")
    if not isinstance(value.get("text"), str):
        raise HarnessError("resume_from reasoning part text must be a string")
    for key in ("signature", "id", "provider_name"):
        if key in value and not isinstance(value[key], str):
            raise HarnessError(f"resume_from reasoning part {key!r} must be a string")
    return ReasoningPart(
        text=value["text"],
        signature=value.get("signature"),
        id=value.get("id"),
        provider_name=value.get("provider_name"),
        provider_details=value.get("provider_details"),
    )


def _model_tool_call_from_dict(value: Any) -> ModelToolCall:
    if not isinstance(value, dict) or set(value) != {"id", "name", "arguments"}:
        raise HarnessError("resume_from assistant tool call has wrong shape")
    if not isinstance(value["id"], str) or not isinstance(value["name"], str) or not isinstance(value["arguments"], str):
        raise HarnessError("resume_from assistant tool call has wrong type")
    return ModelToolCall(id=value["id"], name=value["name"], arguments=value["arguments"])


def _append_tool_results(transcript: list[TranscriptEntry], outputs: list[ToolOutput], notice_text: str) -> None:
    transcript.extend(ToolResultEntry(call_id=output.call_id, output=output.output) for output in outputs)
    if notice_text:
        transcript.append(UserEntry(content=notice_text, notice=True))


def _append_assistant_turn(transcript: list[TranscriptEntry], turn: ModelTurn) -> None:
    transcript.append(
        AssistantEntry(
            text=turn.text,
            tool_calls=copy.deepcopy(turn.tool_calls),
            reasoning=copy.deepcopy(turn.reasoning),
        )
    )


def _validate_anthropic_tool_arguments(entries: list[TranscriptEntry]) -> None:
    for entry in entries:
        if isinstance(entry, AssistantEntry):
            for call in entry.tool_calls:
                try:
                    json.loads(call.arguments)
                except ValueError as exc:
                    raise HarnessError("resume_from assistant tool call arguments must be JSON for Anthropic resume") from exc


# =============================================================================
# Provider transports
# =============================================================================


class Provider:
    """Provider transport, auth, and endpoint configuration."""

    name = "provider"
    api_key_env = ""
    default_base_url = ""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: int = 120,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.api_key = api_key or (os.getenv(self.api_key_env) if self.api_key_env else None)
        self.base_url = (base_url or self.default_base_url).rstrip("/")
        self.timeout = timeout
        self._http_client = http_client
        self._owns_client = http_client is None

    def headers(self) -> Json:
        """Return auth headers for this provider."""
        if not self.api_key:
            raise ProviderError(f"{self.api_key_env} is required for {self.name}")
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def _client(self) -> httpx.AsyncClient:
        """Return the shared async HTTP client for this provider."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=self.timeout)
        return self._http_client

    async def aclose(self) -> None:
        """Close this provider's owned HTTP client."""
        if self._owns_client and self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None

    async def post_json(self, path: str, payload: Json) -> Json:
        """POST JSON to this provider."""
        try:
            response = await self._client().post(f"{self.base_url}{path}", json=payload, headers=self.headers())
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise ProviderError(f"provider error {exc.response.status_code}: {exc.response.text}", status_code=exc.response.status_code) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"provider request failed: {exc}") from exc
        try:
            return response.json()
        except ValueError as exc:
            raise ProviderError(f"provider returned invalid JSON: {exc}") from exc


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
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__(api_key=api_key, base_url=base_url or os.getenv("OPENAI_BASE_URL"), timeout=timeout, http_client=http_client)

    async def create_response(self, payload: Json) -> Json:
        """Create a Responses API response."""
        return await self.post_json("/responses", payload)


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
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__(api_key=api_key, base_url=base_url or os.getenv("ANTHROPIC_BASE_URL"), timeout=timeout, http_client=http_client)

    def headers(self) -> Json:
        """Return Anthropic auth headers."""
        if not self.api_key:
            raise ProviderError("ANTHROPIC_API_KEY is required for Anthropic")
        return {"x-api-key": self.api_key, "anthropic-version": "2023-06-01", "Content-Type": "application/json"}

    async def create_message(self, payload: Json) -> Json:
        """Create an Anthropic Messages response."""
        return await self.post_json("/messages", payload)


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
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__(api_key=api_key, base_url=base_url or os.getenv("OPENROUTER_BASE_URL"), timeout=timeout, http_client=http_client)

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


# =============================================================================
# Model adapters
# =============================================================================


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
        """Build a Responses API payload."""
        payload: Json = {"model": self.model, "input": input_payload, "tools": tools}
        if _openai_supports_encrypted_reasoning(self.model):
            payload["include"] = ["reasoning.encrypted_content"]
        if instructions:
            payload["instructions"] = instructions
        if metadata:
            payload["metadata"] = metadata
        if self.settings.temperature is not None:
            payload["temperature"] = self.settings.temperature
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
        prompt: str,
        constants: RequestConstants,
        *,
        previous_response_id: str | None = None,
        notices: list[ModelNotice] | None = None,
    ) -> ModelTurn:
        """Start a Responses API run."""
        self.previous_response_id = previous_response_id
        input_text = append_notices_to_text(prompt, notices)
        self.transcript = [UserEntry(content=input_text)]
        payload = self.model.build_payload(
            input_payload=input_text,
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
            {"type": "function_call_output", "call_id": output.call_id, "output": output.output}
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

    async def continue_with_user_text(
        self,
        text: str,
        constants: RequestConstants,
        *,
        notices: list[ModelNotice] | None = None,
    ) -> ModelTurn:
        """Continue a Responses API run with user text."""
        input_text = append_notices_to_text(text, notices)
        self.transcript.append(UserEntry(content=input_text))
        payload = self.model.build_payload(
            input_payload=self._prepend_replay(input_text),
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
        return _transcript_state(model=self.model, entries=self.transcript)

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
            return [*replay, _openai_user_item(input_payload)]
        return [*replay, *input_payload]


class AnthropicMessagesModel:
    """Responses-like model implemented with Anthropic Messages."""

    capabilities = ModelCapabilities(supports_json_schema_output=False, default_structured_output_mode="tool")
    resume_kind = "anthropic"

    def __init__(
        self,
        model: str,
        *,
        provider: AnthropicProvider | None = None,
        settings: ModelSettings | None = None,
        max_tokens: int = 1024,
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
        prompt: str,
        constants: RequestConstants,
        *,
        previous_response_id: str | None = None,
        notices: list[ModelNotice] | None = None,
    ) -> ModelTurn:
        """Start an Anthropic Messages run."""
        if constants.structured_output is not None:
            raise ProviderError("Anthropic does not support native structured output")
        if previous_response_id:
            raise ProviderError("previous_response_id is only supported by OpenAI Responses")
        self.system = constants.instructions
        content = append_notices_to_text(prompt, notices)
        self.messages = [{"role": "user", "content": content}]
        self.transcript = [UserEntry(content=content)]
        return await self._complete(tools=constants.tools, metadata=constants.metadata)

    async def continue_with_tools(
        self,
        outputs: list[ToolOutput],
        constants: RequestConstants,
        *,
        notices: list[ModelNotice] | None = None,
    ) -> ModelTurn:
        """Continue an Anthropic Messages run with tool_result blocks."""
        if constants.structured_output is not None:
            raise ProviderError("Anthropic does not support native structured output")
        content = [{"type": "tool_result", "tool_use_id": output.call_id, "content": output.output} for output in outputs]
        notice_text = render_model_notices(notices)
        if notice_text:
            content.append({"type": "text", "text": notice_text})
        _append_tool_results(self.transcript, outputs, notice_text)
        self._apply_resume(constants.instructions)
        self.messages.append({
            "role": "user",
            "content": content,
        })
        return await self._complete(tools=constants.tools, metadata=constants.metadata)

    async def continue_with_user_text(
        self,
        text: str,
        constants: RequestConstants,
        *,
        notices: list[ModelNotice] | None = None,
    ) -> ModelTurn:
        """Continue an Anthropic Messages run with user text."""
        if constants.structured_output is not None:
            raise ProviderError("Anthropic does not support native structured output")
        content = append_notices_to_text(text, notices)
        self.transcript.append(UserEntry(content=content))
        self._apply_resume(constants.instructions)
        self.messages.append({"role": "user", "content": content})
        return await self._complete(tools=constants.tools, metadata=constants.metadata)

    def dump_state(self) -> dict[str, Any] | None:
        """Serialize the neutral transcript for resume."""
        return _transcript_state(model=self.model, entries=self.transcript)

    async def _complete(self, *, tools: list[Json], metadata: Json | None = None) -> ModelTurn:
        """Send a Messages API request and normalize the response."""
        payload: Json = {
            "model": self.model.model,
            "max_tokens": self.model.max_tokens,
            "system": self.system,
            "messages": self.messages,
            "tools": [_responses_tool_to_anthropic(tool) for tool in tools],
        }
        if metadata:
            payload["metadata"] = metadata
        if self.model.settings.temperature is not None:
            payload["temperature"] = self.model.settings.temperature
        payload.update(self.model.settings.extra_body)
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
            thinking_enabled=_anthropic_thinking_enabled(self.model.settings),
        )
        self._resume_entries = None


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
        prompt: str,
        constants: RequestConstants,
        *,
        previous_response_id: str | None = None,
        notices: list[ModelNotice] | None = None,
    ) -> ModelTurn:
        """Start an OpenRouter run."""
        if previous_response_id:
            raise ProviderError("previous_response_id is only supported by OpenAI Responses")
        content = append_notices_to_text(prompt, notices)
        self.messages = [
            {"role": "system", "content": constants.instructions},
            {"role": "user", "content": content},
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
            self.messages.append({"role": "tool", "tool_call_id": output.call_id, "content": output.output})
        if notice_text:
            self.messages.append({"role": "user", "content": notice_text})
        return await self._complete(tools=constants.tools, metadata=constants.metadata, structured_output=constants.structured_output)

    async def continue_with_user_text(
        self,
        text: str,
        constants: RequestConstants,
        *,
        notices: list[ModelNotice] | None = None,
    ) -> ModelTurn:
        """Continue an OpenRouter run with user text."""
        content = append_notices_to_text(text, notices)
        self.transcript.append(UserEntry(content=content))
        self._apply_resume(constants.instructions)
        self.messages.append({"role": "user", "content": content})
        return await self._complete(tools=constants.tools, metadata=constants.metadata, structured_output=constants.structured_output)

    def dump_state(self) -> dict[str, Any] | None:
        """Serialize the neutral transcript for resume."""
        return _transcript_state(model=self.model, entries=self.transcript)

    async def _complete(
        self,
        *,
        tools: list[Json],
        metadata: Json | None = None,
        structured_output: StructuredOutputRequest | None = None,
    ) -> ModelTurn:
        """Send an OpenRouter request and normalize the response."""
        payload: Json = {
            "model": self.model.model,
            "messages": self.messages,
            "tools": [_responses_tool_to_chat(tool) for tool in tools],
        }
        if metadata:
            payload["metadata"] = metadata
        if self.model.settings.temperature is not None:
            payload["temperature"] = self.model.settings.temperature
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


# =============================================================================
# Model selection
# =============================================================================


def infer_model(
    model_ref: str,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: int = 120,
    temperature: float | None = None,
    extra_body: Json | None = None,
) -> Model:
    """Create a model from a provider:model reference."""
    provider_name, model_name = parse_model_ref(model_ref)
    settings = ModelSettings(temperature=temperature, extra_body=extra_body or {})
    if provider_name == "openai":
        provider = OpenAIProvider(api_key=api_key, base_url=base_url, timeout=timeout)
        return OpenAIResponsesModel(model_name, provider=provider, settings=settings)
    if provider_name == "anthropic":
        provider = AnthropicProvider(api_key=api_key, base_url=base_url, timeout=timeout)
        return AnthropicMessagesModel(model_name, provider=provider, settings=settings)
    if provider_name == "openrouter":
        provider = OpenRouterProvider(api_key=api_key, base_url=base_url, timeout=timeout)
        return OpenRouterModel(model_name, provider=provider, settings=settings)
    raise ValueError(f"unknown model provider {provider_name!r}")


def parse_model_ref(model_ref: str) -> tuple[str, str]:
    """Parse a provider:model reference."""
    if ":" not in model_ref:
        raise ValueError(f"model reference must include a provider prefix: {model_ref}")
    provider, model = model_ref.split(":", 1)
    if not provider or not model:
        raise ValueError(f"invalid model reference: {model_ref}")
    return provider, model


def provider_prefix(name: str) -> str:
    """Normalize provider display names to model-ref prefixes."""
    normalized = name.lower().replace(" ", "")
    return {
        "openai": "openai",
        "anthropic": "anthropic",
        "openrouter": "openrouter",
    }.get(normalized, normalized)


def model_capabilities(model: Model) -> ModelCapabilities:
    """Return declared model capabilities with the custom-model default."""
    return getattr(model, "capabilities", ModelCapabilities())


def same_provider_model_ref(model: Model, model_ref: str) -> bool:
    """Return whether a model reference uses the same provider as a model instance."""
    child_provider, _ = parse_model_ref(model_ref)
    parent_provider = provider_prefix(getattr(getattr(model, "provider", None), "name", ""))
    return child_provider == parent_provider


# =============================================================================
# Provider format helpers
# =============================================================================


def render_model_notices(notices: list[ModelNotice] | None) -> str:
    """Render provider-neutral notices as deterministic text."""
    if not notices:
        return ""
    return "\n\n".join(
        f'<harness_notice kind="{notice.kind}">\n{notice.content}\n</harness_notice>'
        for notice in notices
    )


def append_notices_to_text(text: str, notices: list[ModelNotice] | None) -> str:
    """Append rendered notices to provider text input."""
    notice_text = render_model_notices(notices)
    return text if not notice_text else f"{text}\n\n{notice_text}"


def _thinking_fallback(text: str) -> str:
    """Render reasoning text as a degraded cross-provider thinking block."""
    return f"<thinking>\n{text}\n</thinking>"


def _anthropic_thinking_enabled(settings: ModelSettings) -> bool:
    """Return whether the resuming Anthropic request enables extended thinking."""
    thinking = settings.extra_body.get("thinking")
    return isinstance(thinking, dict) and thinking.get("type") == "enabled"


def _render_anthropic_transcript(entries: list[TranscriptEntry], *, thinking_enabled: bool = False) -> list[Json]:
    """Render neutral transcript entries as Anthropic Messages history."""
    messages: list[Json] = []
    index = 0
    while index < len(entries):
        entry = entries[index]
        if isinstance(entry, UserEntry):
            if entry.notice:
                messages.append({"role": "user", "content": [{"type": "text", "text": entry.content}]})
            else:
                messages.append({"role": "user", "content": entry.content})
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
            content.append({"type": "tool_result", "tool_use_id": tool_entry.call_id, "content": tool_entry.output})
            index += 1
        if index < len(entries):
            notice_entry = entries[index]
            if isinstance(notice_entry, UserEntry) and notice_entry.notice:
                content.append({"type": "text", "text": notice_entry.content})
                index += 1
        messages.append({"role": "user", "content": content})
    return messages


def _render_openrouter_transcript(entries: list[TranscriptEntry]) -> list[Json]:
    """Render neutral transcript entries as OpenRouter chat history."""
    messages: list[Json] = []
    for entry in entries:
        if isinstance(entry, UserEntry):
            messages.append({"role": "user", "content": entry.content})
        elif isinstance(entry, ToolResultEntry):
            messages.append({"role": "tool", "tool_call_id": entry.call_id, "content": entry.output})
        else:
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
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {"name": call.name, "arguments": call.arguments},
                    }
                    for call in entry.tool_calls
                ]
            if not text and not entry.tool_calls:
                message["content"] = ""
            messages.append(message)
    return messages


def _openai_user_item(text: str) -> Json:
    """Render one Responses API user message item."""
    return {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}


def _render_openai_transcript(entries: list[TranscriptEntry], *, encrypted_reasoning_ok: bool = False) -> list[Json]:
    """Render neutral transcript entries as Responses API input items."""
    items: list[Json] = []
    for entry in entries:
        if isinstance(entry, UserEntry):
            items.append(_openai_user_item(entry.content))
        elif isinstance(entry, ToolResultEntry):
            items.append({"type": "function_call_output", "call_id": entry.call_id, "output": entry.output})
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


def _responses_tool_to_anthropic(tool: Json) -> Json:
    """Convert a Responses API function tool to Anthropic format."""
    return {
        "name": tool["name"],
        "description": tool.get("description", ""),
        "input_schema": tool.get("parameters", {"type": "object", "properties": {}}),
    }


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


def _structured_output_to_openai_text_format(request: StructuredOutputRequest) -> Json:
    """Convert neutral structured-output metadata to Responses text.format."""
    json_schema: Json = {"name": request.name, "schema": request.schema, "strict": request.strict}
    if request.description:
        json_schema["description"] = request.description
    return {"format": {"type": "json_schema", **json_schema}}


def _structured_output_to_openrouter_response_format(request: StructuredOutputRequest) -> Json:
    """Convert neutral structured-output metadata to OpenRouter response_format."""
    json_schema: Json = {"name": request.name, "schema": request.schema, "strict": request.strict}
    if request.description:
        json_schema["description"] = request.description
    return {"type": "json_schema", "json_schema": json_schema}


def extract_token_usage(raw: Json) -> TokenUsage | None:
    """Best-effort normalized token usage from a raw provider response.

    Handles both key styles (input_tokens/output_tokens and
    prompt_tokens/completion_tokens); missing keys yield None fields.
    """
    usage = raw.get("usage")
    if not isinstance(usage, dict):
        return None
    input_tokens = usage.get("input_tokens", usage.get("prompt_tokens"))
    output_tokens = usage.get("output_tokens", usage.get("completion_tokens"))
    return TokenUsage(
        input_tokens=input_tokens if isinstance(input_tokens, int) else None,
        output_tokens=output_tokens if isinstance(output_tokens, int) else None,
    )


def extract_finish_reason(raw: Json) -> str | None:
    """Best-effort normalized finish reason from a raw provider response.

    Precedence: stop_reason, then top-level finish_reason, then
    choices[0].finish_reason; only the first choice is normalized.
    """
    if isinstance(raw.get("stop_reason"), str):
        return raw["stop_reason"]
    if isinstance(raw.get("finish_reason"), str):
        return raw["finish_reason"]
    choices = raw.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        reason = choices[0].get("finish_reason")
        if isinstance(reason, str):
            return reason
    return None


def extract_response_model(raw: Json) -> str | None:
    """Best-effort normalized response model from a raw provider response.

    Precedence: top-level model, then choices[0].model.
    """
    if isinstance(raw.get("model"), str):
        return raw["model"]
    choices = raw.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        model = choices[0].get("model")
        if isinstance(model, str):
            return model
    return None


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


def _extract_chat_tool_calls(message: Json) -> list[ModelToolCall]:
    """Extract normalized tool calls from a chat completion message."""
    calls = []
    for call in message.get("tool_calls", []) or []:
        function = call.get("function") or {}
        calls.append(ModelToolCall(id=str(call.get("id")), name=str(function.get("name")), arguments=function.get("arguments") or "{}"))
    return calls


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
