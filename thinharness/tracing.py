"""OpenTelemetry-compatible tracing helpers.

Model input messages are constructed from provider-neutral ModelTraceSnapshot
objects, never from provider payloads, because providers.py may already have
appended harness notices to those payloads. For top-level runs,
langfuse.trace.input stores the raw caller prompt while the first model span
stores the effective prompt after hooks. OTel GenAI message shapes follow the
semantic convention as retrieved on 2026-05-19:
https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/.
"""

from __future__ import annotations

import base64
import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from .providers import ModelNotice
from .tools.base import Json

try:
    from opentelemetry.trace import SpanKind, Status, StatusCode
except ImportError:  # pragma: no cover - exercised when optional deps are absent
    SpanKind = Status = StatusCode = None  # type: ignore[assignment]


class TracingOptions(BaseModel):
    """Configuration for tracing one harness run."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    tracer: Any
    agent_name: str = "thinharness"
    agent_description: str | None = None
    conversation_id: str | None = None
    capture_messages: bool = False
    capture_tool_args: bool = False
    capture_tool_results: bool = False


@dataclass
class OtlpTracing:
    """Handle returned by create_otlp_tracing."""

    tracer: Any
    provider: Any

    def force_flush(self) -> None:
        """Flush buffered spans."""
        self.provider.force_flush()

    def shutdown(self) -> None:
        """Shut down the tracer provider."""
        self.provider.shutdown()


@dataclass(frozen=True)
class ModelTraceSnapshot:
    """Canonical input for one model span."""

    kind: Literal["start", "resume", "tool_outputs", "correction", "output_retry_tool"]
    prompt: str | None = None
    tool_outputs: list[Json] | None = None
    notices: list[Json] | None = None
    structured_output: str | None = None

    def with_notices(self, notices: list[ModelNotice]) -> ModelTraceSnapshot:
        """Return a copy with model-facing notices attached."""
        serialized = [asdict(notice) for notice in notices]
        return replace(self, notices=serialized or None)


def create_otlp_tracing(
    *,
    service_name: str,
    endpoint: str | None = None,
    headers: dict[str, str] | None = None,
    tracer_name: str = "thinharness",
) -> OtlpTracing:
    """Create an OTLP HTTP tracer provider."""
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError("install thinharness[tracing] to use create_otlp_tracing") from exc

    exporter = OTLPSpanExporter(endpoint=endpoint, headers=headers)
    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    return OtlpTracing(tracer=provider.get_tracer(tracer_name), provider=provider)


def create_langfuse_tracing(
    *,
    service_name: str,
    public_key: str | None = None,
    secret_key: str | None = None,
    host: str | None = None,
    legacy_ingestion: bool = False,
    tracer_name: str = "thinharness",
) -> OtlpTracing:
    """Create a Langfuse OTLP tracer provider for live validation."""
    public_key = public_key or os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = secret_key or os.getenv("LANGFUSE_SECRET_KEY")
    if not public_key or not secret_key:
        raise RuntimeError("LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are required for create_langfuse_tracing")
    auth = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
    langfuse_host = host or os.getenv("LANGFUSE_HOST") or "https://us.cloud.langfuse.com"
    headers = {"Authorization": f"Basic {auth}"}
    if legacy_ingestion:
        # Current Langfuse docs recommend this for direct OTLP ingestion that
        # needs real-time Cloud Fast Preview visibility.
        headers["x-langfuse-ingestion-version"] = "4"
    return create_otlp_tracing(
        service_name=service_name,
        endpoint=langfuse_host.rstrip("/") + "/api/public/otel/v1/traces",
        headers=headers,
        tracer_name=tracer_name,
    )


class RunTracer:
    """Small wrapper around an OpenTelemetry tracer."""

    def __init__(self, options: TracingOptions | None) -> None:
        self.options = options

    @contextmanager
    def agent(self, *, conversation_id: str | None = None) -> Iterator[_SpanAdapter]:
        """Trace a harness run."""
        options = self.options
        name = options.agent_name if options else "thinharness"
        attributes = {
            "gen_ai.operation.name": "invoke_agent",
            "gen_ai.agent.name": name,
            "gen_ai.agent.description": options.agent_description if options else None,
            "gen_ai.conversation.id": options.conversation_id if options and options.conversation_id else conversation_id,
        }
        with self._span(f"invoke_agent {name}", "internal", attributes) as span:
            yield span

    @contextmanager
    def model(self, model: Any) -> Iterator[_SpanAdapter]:
        """Trace one model request."""
        model_name = str(getattr(model, "model", "unknown") or "unknown")
        provider_name = getattr(getattr(model, "provider", None), "name", None)
        attributes = {
            "gen_ai.operation.name": "chat",
            "gen_ai.provider.name": provider_name,
            "gen_ai.request.model": model_name,
        }
        with self._span(f"chat {model_name}", "client", attributes) as span:
            yield span

    @contextmanager
    def tool(self, *, tool_name: str, call_id: str, arguments: str) -> Iterator[_SpanAdapter]:
        """Trace one local tool execution."""
        options = self.options
        attributes = {
            "gen_ai.operation.name": "execute_tool",
            "gen_ai.tool.name": tool_name,
            "gen_ai.tool.call.id": call_id,
            "gen_ai.tool.type": "function",
            "gen_ai.tool.call.arguments": arguments if options and options.capture_tool_args else None,
        }
        with self._span(f"execute_tool {tool_name}", "internal", attributes) as span:
            yield span

    @contextmanager
    def _span(self, name: str, kind: str, attributes: Json) -> Iterator[_SpanAdapter]:
        """Start an OpenTelemetry span or a no-op span."""
        if not self.options:
            yield _SpanAdapter(_NoopSpan())
            return

        tracer = self.options.tracer
        kwargs: dict[str, Any] = {"attributes": _compact(attributes)}
        span_kind = _span_kind(kind)
        if span_kind is not None:
            kwargs["kind"] = span_kind

        if hasattr(tracer, "start_as_current_span"):
            with tracer.start_as_current_span(name, **kwargs) as span:
                yield _SpanAdapter(span)
            return

        span = tracer.start_span(name, **kwargs)
        try:
            yield _SpanAdapter(span)
        finally:
            span.end()


class _SpanAdapter:
    """Compatibility wrapper for OpenTelemetry spans."""

    def __init__(self, span: Any) -> None:
        self.span = span

    def set_attributes(self, attributes: Json) -> None:
        """Set non-empty span attributes."""
        compact = _compact(attributes)
        if compact and hasattr(self.span, "set_attributes"):
            self.span.set_attributes(compact)
        elif compact:
            for key, value in compact.items():
                self.set_attribute(key, value)

    def set_attribute(self, key: str, value: Any) -> None:
        """Set one span attribute."""
        if value is not None and hasattr(self.span, "set_attribute"):
            self.span.set_attribute(key, value)

    def record_exception(self, exc: BaseException) -> None:
        """Record an exception on the span."""
        if hasattr(self.span, "record_exception"):
            self.span.record_exception(exc)

    def set_error(self, message: str, error_type: str | None = None) -> None:
        """Mark the span as failed."""
        if Status is not None and StatusCode is not None and hasattr(self.span, "set_status"):
            self.span.set_status(Status(StatusCode.ERROR, message))
        elif hasattr(self.span, "set_status"):
            self.span.set_status({"code": "ERROR", "message": message})
        if error_type:
            self.set_attribute("error.type", error_type)


class _NoopSpan:
    """No-op span used when tracing is disabled."""

    def set_attributes(self, attributes: Json) -> None:
        """Ignore span attributes."""

    def set_attribute(self, key: str, value: Any) -> None:
        """Ignore one span attribute."""

    def record_exception(self, exc: BaseException) -> None:
        """Ignore recorded exceptions."""

    def set_status(self, status: Any) -> None:
        """Ignore span status."""


def annotate_model_request(span: _SpanAdapter, snapshot: ModelTraceSnapshot, *, capture_messages: bool) -> None:
    """Write opt-in request content for portable OTel and Langfuse display."""
    if not capture_messages:
        return
    input_payload = _model_request_input(snapshot)
    span.set_attributes({
        "gen_ai.input.messages": serialize_attribute_value(_otel_input_messages(snapshot)),
        "gen_ai.prompt": serialize_attribute_value(input_payload),
        "langfuse.observation.input": serialize_attribute_value(input_payload),
        "thinharness.model.request.kind": snapshot.kind,
        "thinharness.output.mode_requested": snapshot.structured_output,
        "thinharness.model.notices": serialize_attribute_value(snapshot.notices),
    })


def annotate_model_span(span: _SpanAdapter, turn: Any, *, capture_messages: bool = False) -> None:
    """Add model response attributes to a span."""
    raw = getattr(turn, "raw", {}) or {}
    text = getattr(turn, "text", "") or ""
    attributes = {
        "gen_ai.response.id": raw.get("id"),
        "gen_ai.response.model": _response_model(raw),
        "gen_ai.response.finish_reasons": _finish_reasons(raw),
        **_usage_attributes(raw),
    }
    if capture_messages and text:
        attributes.update({
            "langfuse.observation.output": serialize_attribute_value({"text": text}),
            "gen_ai.completion": text,
            "gen_ai.output.messages": serialize_attribute_value(_otel_output_messages(turn)),
        })
    span.set_attributes(attributes)


def annotate_agent_start(
    span: _SpanAdapter,
    *,
    prompt: str,
    instructions: str,
    capture_messages: bool,
    top_level: bool,
) -> None:
    """Write opt-in agent input attributes before provider work runs."""
    if not capture_messages:
        return
    if top_level:
        span.set_attributes({
            "langfuse.trace.input": prompt,
            "gen_ai.system_instructions": serialize_attribute_value([{"type": "text", "content": instructions}]),
        })
    else:
        span.set_attribute("langfuse.observation.input", prompt)


def annotate_agent_result(
    span: _SpanAdapter,
    *,
    result: Any,
    output_schema: Any | None,
    capture_messages: bool,
    top_level: bool,
) -> None:
    """Write opt-in agent trace or observation output attributes."""
    if not capture_messages:
        return
    output_payload = output_schema.dump(result.output) if output_schema is not None and result.output is not None else None
    output = {"text": result.text, "output": output_payload, "stop_reason": result.stop_reason}
    if top_level:
        span.set_attributes({
            "langfuse.trace.output": serialize_attribute_value(output),
            "gen_ai.completion": result.text,
        })
    else:
        span.set_attribute("langfuse.observation.output", serialize_attribute_value(output))


def _trace_output_mode(output_schema: Any | None) -> str | None:
    """Return the requested output mode for tracing."""
    return str(output_schema.mode) if output_schema is not None else None


def serialize_attribute_value(value: Any) -> str | None:
    """Serialize arbitrary data for a span attribute."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except TypeError:
        return str(value)


def _model_request_input(snapshot: ModelTraceSnapshot) -> Json | None:
    """Return the backend-compatible logical request payload."""
    if snapshot.kind in {"start", "resume"}:
        return {"prompt": snapshot.prompt}
    if snapshot.kind in {"tool_outputs", "output_retry_tool"}:
        return {"tool_outputs": snapshot.tool_outputs or []}
    if snapshot.kind == "correction":
        return {"correction": snapshot.prompt}
    return None


def _otel_input_messages(snapshot: ModelTraceSnapshot) -> list[Json] | None:
    """Return OTel-shaped logical input messages."""
    if snapshot.kind in {"start", "resume", "correction"}:
        if snapshot.prompt is None:
            return None
        return [{"role": "user", "parts": [{"type": "text", "content": snapshot.prompt}]}]
    if snapshot.kind in {"tool_outputs", "output_retry_tool"}:
        return [
            {
                "role": "tool",
                "parts": [{
                    "type": "tool_result",
                    "id": output.get("call_id"),
                    "content": output.get("output"),
                }],
            }
            for output in snapshot.tool_outputs or []
        ]
    return None


def _otel_output_messages(turn: Any) -> list[Json]:
    """Return OTel-shaped assistant output messages."""
    parts: list[Json] = []
    text = getattr(turn, "text", "") or ""
    if text:
        parts.append({"type": "text", "content": text})
    for call in getattr(turn, "tool_calls", []) or []:
        parts.append({
            "type": "tool_call",
            "id": getattr(call, "id", None),
            "name": getattr(call, "name", None),
            "arguments": getattr(call, "arguments", None),
        })
    return [{"role": "assistant", "parts": parts}]


def _usage_attributes(raw: Json) -> Json:
    """Extract common token usage attributes from provider responses."""
    usage = raw.get("usage") or {}
    input_tokens = usage.get("input_tokens", usage.get("prompt_tokens"))
    output_tokens = usage.get("output_tokens", usage.get("completion_tokens"))
    return {
        "gen_ai.usage.input_tokens": input_tokens,
        "gen_ai.usage.output_tokens": output_tokens,
        "gen_ai.usage.total_tokens": usage.get("total_tokens"),
    }


def _response_model(raw: Json) -> str | None:
    """Extract a response model name."""
    if isinstance(raw.get("model"), str):
        return raw["model"]
    choice = ((raw.get("choices") or [{}])[0] or {}) if isinstance(raw.get("choices"), list) else {}
    return choice.get("model")


def _finish_reasons(raw: Json) -> list[str] | None:
    """Extract provider finish reasons."""
    if isinstance(raw.get("stop_reason"), str):
        return [raw["stop_reason"]]
    if isinstance(raw.get("finish_reason"), str):
        return [raw["finish_reason"]]
    choices = raw.get("choices")
    if isinstance(choices, list):
        reasons = [
            reason
            for choice in choices
            if isinstance(choice, dict) and isinstance(reason := choice.get("finish_reason"), str)
        ]
        return reasons or None
    return None


def _compact(attributes: Json) -> Json:
    """Drop unset attributes."""
    return {key: value for key, value in attributes.items() if value is not None}


def _span_kind(kind: str) -> Any | None:
    """Return an OpenTelemetry span kind if available."""
    if SpanKind is None:
        return None
    if kind == "client":
        return SpanKind.CLIENT
    return SpanKind.INTERNAL
