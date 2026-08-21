"""Provider models, transports, and model-reference selection."""

from __future__ import annotations

from ..tools.base import Json
from .anthropic import (
    DEFAULT_ANTHROPIC_MAX_TOKENS,
    AnthropicMessagesModel,
    AnthropicMessagesSession,
    AnthropicProvider,
)
from .base import (
    Model,
    ModelCapabilities,
    ModelNotice,
    ModelSession,
    ModelSettings,
    ModelToolCall,
    ModelTurn,
    ReasoningPart,
    RequestConstants,
    ResumableModel,
    StructuredOutputRequest,
    TokenUsage,
    ToolOutput,
    append_notices_to_content,
    extract_finish_reason,
    extract_response_model,
    extract_token_usage,
    render_model_notices,
)
from .openai import OpenAIProvider, OpenAIResponsesModel, OpenAIResponsesSession
from .openrouter import OpenRouterModel, OpenRouterProvider, OpenRouterSession
from .transcript import (
    AssistantEntry,
    ToolResultEntry,
    TranscriptEntry,
    UserEntry,
    session_image_blocks,
)
from .transport import Provider, ProviderError


def infer_model(
    model_ref: str,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: int = 120,
    request_retries: int = 3,
    request_retry_backoff: float = 1.0,
    temperature: float | None = None,
    max_tokens: int | None = None,
    effort: str | None = None,
    extra_body: Json | None = None,
) -> Model:
    """Create a model from a provider:model reference."""
    provider_name, model_name = parse_model_ref(model_ref)
    settings = ModelSettings(temperature=temperature, max_tokens=max_tokens, effort=effort, extra_body=extra_body or {})
    if provider_name == "openai":
        provider = OpenAIProvider(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            request_retries=request_retries,
            request_retry_backoff=request_retry_backoff,
        )
        return OpenAIResponsesModel(model_name, provider=provider, settings=settings)
    if provider_name == "anthropic":
        provider = AnthropicProvider(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            request_retries=request_retries,
            request_retry_backoff=request_retry_backoff,
        )
        return AnthropicMessagesModel(model_name, provider=provider, settings=settings)
    if provider_name == "openrouter":
        provider = OpenRouterProvider(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            request_retries=request_retries,
            request_retry_backoff=request_retry_backoff,
        )
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


__all__ = [
    "AnthropicMessagesModel",
    "AnthropicMessagesSession",
    "AnthropicProvider",
    "AssistantEntry",
    "DEFAULT_ANTHROPIC_MAX_TOKENS",
    "Model",
    "ModelCapabilities",
    "ModelNotice",
    "ModelSession",
    "ModelSettings",
    "ModelToolCall",
    "ModelTurn",
    "OpenAIProvider",
    "OpenAIResponsesModel",
    "OpenAIResponsesSession",
    "OpenRouterModel",
    "OpenRouterProvider",
    "OpenRouterSession",
    "Provider",
    "ProviderError",
    "ReasoningPart",
    "RequestConstants",
    "ResumableModel",
    "StructuredOutputRequest",
    "TokenUsage",
    "ToolOutput",
    "ToolResultEntry",
    "TranscriptEntry",
    "UserEntry",
    "append_notices_to_content",
    "extract_finish_reason",
    "extract_response_model",
    "extract_token_usage",
    "infer_model",
    "model_capabilities",
    "parse_model_ref",
    "provider_prefix",
    "render_model_notices",
    "same_provider_model_ref",
    "session_image_blocks",
]
