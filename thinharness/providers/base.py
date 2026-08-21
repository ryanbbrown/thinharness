"""Provider-neutral model contracts and normalized request values."""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol

from pydantic import BaseModel, Field

from ..content import ImageBlock, NormalizedContent, Prompt, append_text_block
from ..tools.base import Json, ToolResult

if TYPE_CHECKING:
    from .transport import Provider


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
    cached_tokens: int | None = None


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
class ToolOutput:
    """A normalized local tool output."""

    call_id: str
    result: ToolResult
    wire_output: str | None = None

    def __post_init__(self) -> None:
        """Require the normalized tool-result contract."""
        if not isinstance(self.result, ToolResult):
            raise TypeError("ToolOutput.result must be a ToolResult")

    @property
    def output(self) -> str:
        """Return the exact provider-facing text when one is required."""
        return self.wire_output if self.wire_output is not None else self.result.to_json()


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

    Built once per run after generic plugin connection and run-start hooks,
    so the toolset and instructions are frozen for the run.
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
    max_tokens: int | None = Field(default=None, ge=1)
    effort: str | None = None
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
        prompt: Prompt,
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

    async def continue_with_user_content(
        self,
        content: Prompt,
        constants: RequestConstants,
        *,
        notices: list[ModelNotice] | None = None,
    ) -> ModelTurn:
        """Continue a model run with user content (a correction or resumed prompt)."""
        ...

    def dump_state(self) -> dict[str, Any] | None:
        """Serialize session state for resume, or None if unavailable."""
        ...


def render_model_notices(notices: list[ModelNotice] | None) -> str:
    """Render provider-neutral notices as deterministic text."""
    if not notices:
        return ""
    return "\n\n".join(
        f'<harness_notice kind="{notice.kind}">\n{notice.content}\n</harness_notice>'
        for notice in notices
    )


def append_notices_to_content(content: NormalizedContent, notices: list[ModelNotice] | None) -> NormalizedContent:
    """Append notices as one final text block."""
    return append_text_block(content, render_model_notices(notices))


def _data_url(block: ImageBlock) -> str:
    """Encode one image as a provider data URL."""
    return f"data:{block.media_type};base64,{base64.b64encode(block.data).decode('ascii')}"


def extract_token_usage(raw: Json) -> TokenUsage | None:
    """Best-effort normalized token usage from a raw provider response.

    Handles both key styles (input_tokens/output_tokens and
    prompt_tokens/completion_tokens), plus provider cache-read breakdowns;
    missing keys yield None fields.
    """
    usage = raw.get("usage")
    if not isinstance(usage, dict):
        return None
    input_tokens = usage.get("input_tokens", usage.get("prompt_tokens"))
    output_tokens = usage.get("output_tokens", usage.get("completion_tokens"))
    cached_tokens = usage.get("cache_read_input_tokens")
    if not isinstance(cached_tokens, int):
        input_details = usage.get("input_tokens_details")
        if isinstance(input_details, dict):
            cached_tokens = input_details.get("cached_tokens")
    if not isinstance(cached_tokens, int):
        prompt_details = usage.get("prompt_tokens_details")
        if isinstance(prompt_details, dict):
            cached_tokens = prompt_details.get("cached_tokens")
    return TokenUsage(
        input_tokens=input_tokens if isinstance(input_tokens, int) else None,
        output_tokens=output_tokens if isinstance(output_tokens, int) else None,
        cached_tokens=cached_tokens if isinstance(cached_tokens, int) else None,
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
