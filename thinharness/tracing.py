"""OpenTelemetry-compatible tracing helpers."""

from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Any, Iterator

from pydantic import BaseModel, ConfigDict
from pydantic.dataclasses import dataclass

from .tools import Json

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


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
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


class RunTracer:
    """Small wrapper around an OpenTelemetry tracer."""

    def __init__(self, options: TracingOptions | None) -> None:
        self.options = options

    @contextmanager
    def agent(self, *, conversation_id: str | None = None) -> Iterator["_SpanAdapter"]:
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
    def model(self, model: Any) -> Iterator["_SpanAdapter"]:
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
    def tool(self, *, tool_name: str, call_id: str, arguments: str) -> Iterator["_SpanAdapter"]:
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
    def _span(self, name: str, kind: str, attributes: Json) -> Iterator["_SpanAdapter"]:
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


def annotate_model_span(span: _SpanAdapter, turn: Any, *, capture_messages: bool = False) -> None:
    """Add model response attributes to a span."""
    raw = getattr(turn, "raw", {}) or {}
    text = getattr(turn, "text", "") or ""
    span.set_attributes({
        "gen_ai.response.id": raw.get("id"),
        "gen_ai.response.model": _response_model(raw),
        "gen_ai.response.finish_reasons": _finish_reasons(raw),
        **_usage_attributes(raw),
        "gen_ai.completion": text if capture_messages and text else None,
    })


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
