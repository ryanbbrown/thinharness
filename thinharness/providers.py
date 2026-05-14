"""Provider and model implementations for normalized agent turns."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Protocol

from pydantic import Field
from pydantic.dataclasses import dataclass

from .tools import Json


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
    tool_calls: list[ModelToolCall] = Field(default_factory=list)
    raw: Json = Field(default_factory=dict)


@dataclass
class ToolOutput:
    """A normalized local tool output."""

    call_id: str
    output: str


@dataclass
class ModelSettings:
    """Common request settings shared across models."""

    temperature: float | None = None
    extra_body: Json = Field(default_factory=dict)


class Model(Protocol):
    """Responses-like model contract consumed by the harness."""

    model: str

    @property
    def provider(self) -> "Provider":
        """Return the model provider."""
        ...

    @property
    def api_key(self) -> str | None:
        """Return the model provider API key."""
        ...

    def new_session(self) -> "ModelSession":
        """Create isolated state for one model run."""
        ...


class ModelSession(Protocol):
    """Per-run model state consumed by the harness."""

    def start(
        self,
        *,
        prompt: str,
        instructions: str,
        tools: list[Json],
        metadata: Json | None = None,
        previous_response_id: str | None = None,
    ) -> ModelTurn:
        """Start a model run."""
        ...

    def continue_with_tools(self, outputs: list[ToolOutput], *, tools: list[Json], metadata: Json | None = None) -> ModelTurn:
        """Continue a model run with tool outputs."""
        ...


class ProviderError(RuntimeError):
    """Raised when a provider request fails."""


# =============================================================================
# Provider transports
# =============================================================================


class ResponsesClient:
    """Tiny stdlib client for OpenAI-compatible /responses endpoints."""

    def __init__(self, *, api_key: str | None = None, base_url: str | None = None, timeout: int = 120) -> None:
        self.provider = OpenAIProvider(api_key=api_key, base_url=base_url, timeout=timeout)
        self.api_key = self.provider.api_key

    def create(self, payload: Json) -> Json:
        """Create a response through the configured endpoint."""
        return self.provider.post_json("/responses", payload)


class Provider:
    """Provider transport, auth, and endpoint configuration."""

    name = "provider"
    api_key_env = ""
    default_base_url = ""

    def __init__(self, *, api_key: str | None = None, base_url: str | None = None, timeout: int = 120) -> None:
        self.api_key = api_key or (os.getenv(self.api_key_env) if self.api_key_env else None)
        self.base_url = (base_url or self.default_base_url).rstrip("/")
        self.timeout = timeout

    def headers(self) -> Json:
        """Return auth headers for this provider."""
        if not self.api_key:
            raise ProviderError(f"{self.api_key_env} is required for {self.name}")
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def post_json(self, path: str, payload: Json) -> Json:
        """POST JSON to this provider."""
        return _post_json(f"{self.base_url}{path}", payload, self.headers(), self.timeout)


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
        client: Any | None = None,
    ) -> None:
        super().__init__(api_key=api_key, base_url=base_url or os.getenv("OPENAI_BASE_URL"), timeout=timeout)
        self.client = client
        if not self.api_key and client:
            self.api_key = getattr(client, "api_key", None)

    def create_response(self, payload: Json) -> Json:
        """Create a Responses API response."""
        if self.client:
            return self.client.create(payload)
        return self.post_json("/responses", payload)


class AnthropicProvider(Provider):
    """Provider for Anthropic Messages endpoints."""

    name = "Anthropic"
    api_key_env = "ANTHROPIC_API_KEY"
    default_base_url = "https://api.anthropic.com/v1"

    def __init__(self, *, api_key: str | None = None, base_url: str | None = None, timeout: int = 120) -> None:
        super().__init__(api_key=api_key, base_url=base_url or os.getenv("ANTHROPIC_BASE_URL"), timeout=timeout)

    def headers(self) -> Json:
        """Return Anthropic auth headers."""
        if not self.api_key:
            raise ProviderError("ANTHROPIC_API_KEY is required for Anthropic")
        return {"x-api-key": self.api_key, "anthropic-version": "2023-06-01", "Content-Type": "application/json"}

    def create_message(self, payload: Json) -> Json:
        """Create an Anthropic Messages response."""
        return self.post_json("/messages", payload)


class OpenRouterProvider(Provider):
    """Provider for OpenRouter's OpenAI-compatible gateway."""

    name = "OpenRouter"
    api_key_env = "OPENROUTER_API_KEY"
    default_base_url = "https://openrouter.ai/api/v1"

    def __init__(self, *, api_key: str | None = None, base_url: str | None = None, timeout: int = 120) -> None:
        super().__init__(api_key=api_key, base_url=base_url or os.getenv("OPENROUTER_BASE_URL"), timeout=timeout)

    def headers(self) -> Json:
        """Return OpenRouter auth and attribution headers."""
        headers = super().headers()
        if app_url := os.getenv("OPENROUTER_APP_URL"):
            headers["HTTP-Referer"] = app_url
        if app_title := os.getenv("OPENROUTER_APP_TITLE"):
            headers["X-Title"] = app_title
        return headers

    def create_chat_completion(self, payload: Json) -> Json:
        """Create an OpenRouter chat completion."""
        return self.post_json("/chat/completions", payload)


# =============================================================================
# Model adapters
# =============================================================================


class OpenAIResponsesModel:
    """Responses-like model implemented with OpenAI Responses."""

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

    def _payload(self, *, input_payload: Any, tools: list[Json], instructions: str | None = None, metadata: Json | None = None) -> Json:
        """Build a Responses API payload."""
        payload: Json = {"model": self.model, "input": input_payload, "tools": tools}
        if instructions:
            payload["instructions"] = instructions
        if metadata:
            payload["metadata"] = metadata
        if self.settings.temperature is not None:
            payload["temperature"] = self.settings.temperature
        payload.update(self.settings.extra_body)
        return payload


class OpenAIResponsesSession:
    """Per-run OpenAI Responses state."""

    def __init__(self, model: OpenAIResponsesModel) -> None:
        self.model = model
        self.previous_response_id: str | None = None

    def start(
        self,
        *,
        prompt: str,
        instructions: str,
        tools: list[Json],
        metadata: Json | None = None,
        previous_response_id: str | None = None,
    ) -> ModelTurn:
        """Start a Responses API run."""
        self.previous_response_id = previous_response_id
        payload = self.model._payload(input_payload=prompt, instructions=instructions, tools=tools, metadata=metadata)
        if self.previous_response_id:
            payload["previous_response_id"] = self.previous_response_id
        return self._complete(payload)

    def continue_with_tools(self, outputs: list[ToolOutput], *, tools: list[Json], metadata: Json | None = None) -> ModelTurn:
        """Continue a Responses API run with function_call_output items."""
        input_payload = [
            {"type": "function_call_output", "call_id": output.call_id, "output": output.output}
            for output in outputs
        ]
        payload = self.model._payload(input_payload=input_payload, tools=tools, metadata=metadata)
        if self.previous_response_id:
            payload["previous_response_id"] = self.previous_response_id
        return self._complete(payload)

    def _complete(self, payload: Json) -> ModelTurn:
        """Send a Responses API payload and normalize the response."""
        response = self.model.provider.create_response(payload)
        self.previous_response_id = response.get("id") or self.previous_response_id
        return ModelTurn(text=_extract_responses_text(response), tool_calls=_extract_responses_tool_calls(response), raw=response)


class AnthropicMessagesModel:
    """Responses-like model implemented with Anthropic Messages."""

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


class AnthropicMessagesSession:
    """Per-run Anthropic Messages state."""

    def __init__(self, model: AnthropicMessagesModel) -> None:
        self.model = model
        self.messages: list[Json] = []
        self.system = ""

    def start(
        self,
        *,
        prompt: str,
        instructions: str,
        tools: list[Json],
        metadata: Json | None = None,
        previous_response_id: str | None = None,
    ) -> ModelTurn:
        """Start an Anthropic Messages run."""
        if previous_response_id:
            raise ProviderError("previous_response_id is only supported by OpenAI Responses")
        self.system = instructions
        self.messages = [{"role": "user", "content": prompt}]
        return self._complete(tools=tools, metadata=metadata)

    def continue_with_tools(self, outputs: list[ToolOutput], *, tools: list[Json], metadata: Json | None = None) -> ModelTurn:
        """Continue an Anthropic Messages run with tool_result blocks."""
        self.messages.append({
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": output.call_id, "content": output.output} for output in outputs],
        })
        return self._complete(tools=tools, metadata=metadata)

    def _complete(self, *, tools: list[Json], metadata: Json | None = None) -> ModelTurn:
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
        response = self.model.provider.create_message(payload)
        self.messages.append({"role": "assistant", "content": response.get("content", [])})
        return ModelTurn(text=_extract_anthropic_text(response), tool_calls=_extract_anthropic_tool_calls(response), raw=response)


class OpenRouterModel:
    """Responses-like model implemented through OpenRouter chat completions."""

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


class OpenRouterSession:
    """Per-run OpenRouter chat completion state."""

    def __init__(self, model: OpenRouterModel) -> None:
        self.model = model
        self.messages: list[Json] = []

    def start(
        self,
        *,
        prompt: str,
        instructions: str,
        tools: list[Json],
        metadata: Json | None = None,
        previous_response_id: str | None = None,
    ) -> ModelTurn:
        """Start an OpenRouter run."""
        if previous_response_id:
            raise ProviderError("previous_response_id is only supported by OpenAI Responses")
        self.messages = [
            {"role": "system", "content": instructions},
            {"role": "user", "content": prompt},
        ]
        return self._complete(tools=tools, metadata=metadata)

    def continue_with_tools(self, outputs: list[ToolOutput], *, tools: list[Json], metadata: Json | None = None) -> ModelTurn:
        """Continue an OpenRouter run with tool messages."""
        for output in outputs:
            self.messages.append({"role": "tool", "tool_call_id": output.call_id, "content": output.output})
        return self._complete(tools=tools, metadata=metadata)

    def _complete(self, *, tools: list[Json], metadata: Json | None = None) -> ModelTurn:
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
        response = self.model.provider.create_chat_completion(payload)
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
    client: Any | None = None,
) -> Model:
    """Create a model from a provider:model reference."""
    provider_name, model_name = parse_model_ref(model_ref)
    settings = ModelSettings(temperature=temperature, extra_body=extra_body or {})
    if provider_name == "openai":
        provider = OpenAIProvider(api_key=api_key, base_url=base_url, timeout=timeout, client=client)
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


# =============================================================================
# HTTP and provider format helpers
# =============================================================================


def _post_json(url: str, payload: Json, headers: Json, timeout: int) -> Json:
    """POST JSON with urllib and decode JSON response."""
    request = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise ProviderError(f"provider error {exc.code}: {body}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise ProviderError(f"provider request failed: {exc}") from exc
    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise ProviderError(f"provider returned invalid JSON: {exc}") from exc


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
