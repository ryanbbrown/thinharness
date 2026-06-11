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


@dataclass
class ModelTurn:
    """A normalized model response turn."""

    text: str = ""
    tool_calls: list[ModelToolCall] = field(default_factory=list)
    raw: Json = field(default_factory=dict)
    finalized_output_mode: str | None = None


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
        *,
        prompt: str,
        instructions: str,
        tools: list[Json],
        metadata: Json | None = None,
        previous_response_id: str | None = None,
        structured_output: StructuredOutputRequest | None = None,
        notices: list[ModelNotice] | None = None,
    ) -> ModelTurn:
        """Start a model run."""
        ...

    async def continue_with_tools(
        self,
        outputs: list[ToolOutput],
        *,
        instructions: str | None = None,
        tools: list[Json],
        metadata: Json | None = None,
        structured_output: StructuredOutputRequest | None = None,
        notices: list[ModelNotice] | None = None,
    ) -> ModelTurn:
        """Continue a model run with tool outputs."""
        ...

    async def continue_with_user_message(
        self,
        message: str,
        *,
        instructions: str | None = None,
        tools: list[Json],
        metadata: Json | None = None,
        structured_output: StructuredOutputRequest | None = None,
        notices: list[ModelNotice] | None = None,
    ) -> ModelTurn:
        """Continue a model run with a corrective user message."""
        ...

    async def continue_with_user_prompt(
        self,
        prompt: str,
        *,
        instructions: str,
        tools: list[Json],
        metadata: Json | None = None,
        structured_output: StructuredOutputRequest | None = None,
        notices: list[ModelNotice] | None = None,
    ) -> ModelTurn:
        """Continue a resumed model run with a new user prompt."""
        ...

    def dump_state(self) -> dict[str, Any] | None:
        """Serialize session state for resume, or None if unavailable."""
        ...


class ProviderError(RuntimeError):
    """Raised when a provider request fails."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


_BASE_RESUME_KEYS = frozenset({"kind", "version", "model"})


def _validate_resume_state(
    state: dict[str, Any],
    *,
    expected_kind: str,
    expected_model: str,
    required_fields: dict[str, type | tuple[type, ...]],
) -> None:
    """Validate resume state shape before any session mutation."""
    if not isinstance(state, dict):
        raise HarnessError("resume_from must be a dict")
    if state.get("kind") != expected_kind:
        raise HarnessError(f"resume_from kind {state.get('kind')!r} does not match {expected_kind!r}")
    if state.get("version") != 1:
        raise HarnessError(f"resume_from version {state.get('version')!r} is not supported")
    if state.get("model") != expected_model:
        raise HarnessError(f"resume_from model {state.get('model')!r} does not match current model {expected_model!r}")
    for field_name, expected_type in required_fields.items():
        if field_name not in state:
            raise HarnessError(f"resume_from missing required field: {field_name!r}")
        if not isinstance(state[field_name], expected_type):
            raise HarnessError(f"resume_from field {field_name!r} has wrong type")
    unknown = set(state) - (_BASE_RESUME_KEYS | required_fields.keys())
    if unknown:
        raise HarnessError(f"resume_from has unknown keys: {sorted(unknown)!r}")
    try:
        json.dumps(state)
    except (TypeError, ValueError) as exc:
        raise HarnessError("resume_from must be JSON-serializable") from exc


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
        _validate_resume_state(
            state,
            expected_kind=self.resume_kind,
            expected_model=self.model,
            required_fields={"previous_response_id": str},
        )
        if not state["previous_response_id"]:
            raise HarnessError("resume_from field 'previous_response_id' must be non-empty")
        session = OpenAIResponsesSession(self)
        session.previous_response_id = state["previous_response_id"]
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

    async def start(
        self,
        *,
        prompt: str,
        instructions: str,
        tools: list[Json],
        metadata: Json | None = None,
        previous_response_id: str | None = None,
        structured_output: StructuredOutputRequest | None = None,
        notices: list[ModelNotice] | None = None,
    ) -> ModelTurn:
        """Start a Responses API run."""
        self.previous_response_id = previous_response_id
        payload = self.model.build_payload(
            input_payload=append_notices_to_text(prompt, notices),
            instructions=instructions,
            tools=tools,
            metadata=metadata,
            structured_output=structured_output,
        )
        if self.previous_response_id:
            payload["previous_response_id"] = self.previous_response_id
        return await self._complete(payload)

    async def continue_with_tools(
        self,
        outputs: list[ToolOutput],
        *,
        instructions: str | None = None,
        tools: list[Json],
        metadata: Json | None = None,
        structured_output: StructuredOutputRequest | None = None,
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
        payload = self.model.build_payload(
            input_payload=input_payload,
            instructions=instructions,
            tools=tools,
            metadata=metadata,
            structured_output=structured_output,
        )
        if self.previous_response_id:
            payload["previous_response_id"] = self.previous_response_id
        return await self._complete(payload)

    async def continue_with_user_message(
        self,
        message: str,
        *,
        instructions: str | None = None,
        tools: list[Json],
        metadata: Json | None = None,
        structured_output: StructuredOutputRequest | None = None,
        notices: list[ModelNotice] | None = None,
    ) -> ModelTurn:
        """Continue a Responses API run with a corrective user message."""
        payload = self.model.build_payload(
            input_payload=append_notices_to_text(message, notices),
            instructions=instructions,
            tools=tools,
            metadata=metadata,
            structured_output=structured_output,
        )
        if self.previous_response_id:
            payload["previous_response_id"] = self.previous_response_id
        return await self._complete(payload)

    async def continue_with_user_prompt(
        self,
        prompt: str,
        *,
        instructions: str,
        tools: list[Json],
        metadata: Json | None = None,
        structured_output: StructuredOutputRequest | None = None,
        notices: list[ModelNotice] | None = None,
    ) -> ModelTurn:
        """Continue a resumed Responses API run with a new user prompt."""
        payload = self.model.build_payload(
            input_payload=append_notices_to_text(prompt, notices),
            instructions=instructions,
            tools=tools,
            metadata=metadata,
            structured_output=structured_output,
        )
        if self.previous_response_id:
            payload["previous_response_id"] = self.previous_response_id
        return await self._complete(payload)

    def dump_state(self) -> dict[str, Any] | None:
        """Serialize the latest Responses API continuation token."""
        if not self.previous_response_id:
            return None
        return {
            "kind": self.model.resume_kind,
            "version": 1,
            "model": self.model.model,
            "previous_response_id": self.previous_response_id,
        }

    async def _complete(self, payload: Json) -> ModelTurn:
        """Send a Responses API payload and normalize the response."""
        response = await self.model.provider.create_response(payload)
        self.previous_response_id = response.get("id") or self.previous_response_id
        return ModelTurn(text=_extract_responses_text(response), tool_calls=_extract_responses_tool_calls(response), raw=response)


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
        _validate_resume_state(
            state,
            expected_kind=self.resume_kind,
            expected_model=self.model,
            required_fields={"system": (str, list), "messages": list},
        )
        if not all(isinstance(message, dict) for message in state["messages"]):
            raise HarnessError("resume_from field 'messages' has wrong type")
        session = AnthropicMessagesSession(self)
        session.system = state["system"]
        session.messages = copy.deepcopy(state["messages"])
        return session


class AnthropicMessagesSession:
    """Per-run Anthropic Messages state."""

    def __init__(self, model: AnthropicMessagesModel) -> None:
        self.model = model
        self.messages: list[Json] = []
        self.system = ""

    async def start(
        self,
        *,
        prompt: str,
        instructions: str,
        tools: list[Json],
        metadata: Json | None = None,
        previous_response_id: str | None = None,
        structured_output: StructuredOutputRequest | None = None,
        notices: list[ModelNotice] | None = None,
    ) -> ModelTurn:
        """Start an Anthropic Messages run."""
        if structured_output is not None:
            raise ProviderError("Anthropic does not support native structured output")
        if previous_response_id:
            raise ProviderError("previous_response_id is only supported by OpenAI Responses")
        self.system = instructions
        self.messages = [{"role": "user", "content": append_notices_to_text(prompt, notices)}]
        return await self._complete(tools=tools, metadata=metadata)

    async def continue_with_tools(
        self,
        outputs: list[ToolOutput],
        *,
        instructions: str | None = None,
        tools: list[Json],
        metadata: Json | None = None,
        structured_output: StructuredOutputRequest | None = None,
        notices: list[ModelNotice] | None = None,
    ) -> ModelTurn:
        """Continue an Anthropic Messages run with tool_result blocks."""
        if structured_output is not None:
            raise ProviderError("Anthropic does not support native structured output")
        content = [{"type": "tool_result", "tool_use_id": output.call_id, "content": output.output} for output in outputs]
        notice_text = render_model_notices(notices)
        if notice_text:
            content.append({"type": "text", "text": notice_text})
        self.messages.append({
            "role": "user",
            "content": content,
        })
        return await self._complete(tools=tools, metadata=metadata)

    async def continue_with_user_message(
        self,
        message: str,
        *,
        instructions: str | None = None,
        tools: list[Json],
        metadata: Json | None = None,
        structured_output: StructuredOutputRequest | None = None,
        notices: list[ModelNotice] | None = None,
    ) -> ModelTurn:
        """Continue an Anthropic Messages run with a corrective user message."""
        if structured_output is not None:
            raise ProviderError("Anthropic does not support native structured output")
        self.messages.append({"role": "user", "content": append_notices_to_text(message, notices)})
        return await self._complete(tools=tools, metadata=metadata)

    async def continue_with_user_prompt(
        self,
        prompt: str,
        *,
        instructions: str,
        tools: list[Json],
        metadata: Json | None = None,
        structured_output: StructuredOutputRequest | None = None,
        notices: list[ModelNotice] | None = None,
    ) -> ModelTurn:
        """Continue a resumed Anthropic Messages run with a new user prompt."""
        if structured_output is not None:
            raise ProviderError("Anthropic does not support native structured output")
        self.messages.append({"role": "user", "content": append_notices_to_text(prompt, notices)})
        return await self._complete(tools=tools, metadata=metadata)

    def dump_state(self) -> dict[str, Any] | None:
        """Serialize the Anthropic transcript for resume."""
        return {
            "kind": self.model.resume_kind,
            "version": 1,
            "model": self.model.model,
            "system": self.system,
            "messages": copy.deepcopy(self.messages),
        }

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
        return ModelTurn(text=_extract_anthropic_text(response), tool_calls=_extract_anthropic_tool_calls(response), raw=response)


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
        _validate_resume_state(
            state,
            expected_kind=self.resume_kind,
            expected_model=self.model,
            required_fields={"messages": list},
        )
        if not all(isinstance(message, dict) for message in state["messages"]):
            raise HarnessError("resume_from field 'messages' has wrong type")
        session = OpenRouterSession(self)
        session.messages = copy.deepcopy(state["messages"])
        return session


class OpenRouterSession:
    """Per-run OpenRouter chat completion state."""

    def __init__(self, model: OpenRouterModel) -> None:
        self.model = model
        self.messages: list[Json] = []

    async def start(
        self,
        *,
        prompt: str,
        instructions: str,
        tools: list[Json],
        metadata: Json | None = None,
        previous_response_id: str | None = None,
        structured_output: StructuredOutputRequest | None = None,
        notices: list[ModelNotice] | None = None,
    ) -> ModelTurn:
        """Start an OpenRouter run."""
        if previous_response_id:
            raise ProviderError("previous_response_id is only supported by OpenAI Responses")
        self.messages = [
            {"role": "system", "content": instructions},
            {"role": "user", "content": append_notices_to_text(prompt, notices)},
        ]
        return await self._complete(tools=tools, metadata=metadata, structured_output=structured_output)

    async def continue_with_tools(
        self,
        outputs: list[ToolOutput],
        *,
        instructions: str | None = None,
        tools: list[Json],
        metadata: Json | None = None,
        structured_output: StructuredOutputRequest | None = None,
        notices: list[ModelNotice] | None = None,
    ) -> ModelTurn:
        """Continue an OpenRouter run with tool messages."""
        for output in outputs:
            self.messages.append({"role": "tool", "tool_call_id": output.call_id, "content": output.output})
        notice_text = render_model_notices(notices)
        if notice_text:
            self.messages.append({"role": "user", "content": notice_text})
        return await self._complete(tools=tools, metadata=metadata, structured_output=structured_output)

    async def continue_with_user_message(
        self,
        message: str,
        *,
        instructions: str | None = None,
        tools: list[Json],
        metadata: Json | None = None,
        structured_output: StructuredOutputRequest | None = None,
        notices: list[ModelNotice] | None = None,
    ) -> ModelTurn:
        """Continue an OpenRouter run with a corrective user message."""
        self.messages.append({"role": "user", "content": append_notices_to_text(message, notices)})
        return await self._complete(tools=tools, metadata=metadata, structured_output=structured_output)

    async def continue_with_user_prompt(
        self,
        prompt: str,
        *,
        instructions: str,
        tools: list[Json],
        metadata: Json | None = None,
        structured_output: StructuredOutputRequest | None = None,
        notices: list[ModelNotice] | None = None,
    ) -> ModelTurn:
        """Continue a resumed OpenRouter run with a new user prompt."""
        self.messages.append({"role": "user", "content": append_notices_to_text(prompt, notices)})
        return await self._complete(tools=tools, metadata=metadata, structured_output=structured_output)

    def dump_state(self) -> dict[str, Any] | None:
        """Serialize the OpenRouter transcript for resume."""
        return {
            "kind": self.model.resume_kind,
            "version": 1,
            "model": self.model.model,
            "messages": copy.deepcopy(self.messages),
        }

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
        return ModelTurn(text=str(message.get("content") or ""), tool_calls=_extract_chat_tool_calls(message), raw=response)


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
