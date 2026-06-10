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

import contextvars
import hashlib
import json
import re
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
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
class LocalTracing:
    """Handle returned by create_local_tracing."""

    tracer: Any
    trace_dir: Path


@dataclass(frozen=True)
class ModelTraceSnapshot:
    """Canonical input for one model span."""

    kind: Literal["start", "resume", "tool_outputs", "correction", "output_retry_tool", "background_completion"]
    prompt: str | None = None
    tool_outputs: list[Json] | None = None
    notices: list[Json] | None = None
    structured_output: str | None = None

    def with_notices(self, notices: list[ModelNotice]) -> ModelTraceSnapshot:
        """Return a copy with model-facing notices attached."""
        serialized = [asdict(notice) for notice in notices]
        return replace(self, notices=serialized or None)


def create_local_tracing(trace_dir: str | Path | None = None, *, project_root: str | Path | None = None) -> LocalTracing:
    """Create a plaintext JSONL tracer rooted in the local filesystem."""
    resolved = Path(trace_dir or "~/.thinharness/traces").expanduser().resolve()
    if project_root is not None:
        resolved = resolved / _encode_trace_project_path(project_root)
    return LocalTracing(tracer=_LocalTraceTracer(resolved), trace_dir=resolved)


def create_local_tracing_options(
    trace_dir: str | Path | None = None,
    *,
    project_root: str | Path | None = None,
    agent_name: str = "thinharness",
    agent_description: str | None = None,
    conversation_id: str | None = None,
) -> TracingOptions:
    """Create full-capture tracing options for a local JSONL trace sink."""
    local = create_local_tracing(trace_dir, project_root=project_root)
    return _local_tracing_options(
        local,
        agent_name=agent_name,
        agent_description=agent_description,
        conversation_id=conversation_id,
    )


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

    def __init__(self, options: list[TracingOptions]) -> None:
        self.options = options

    @contextmanager
    def agent(self, *, conversation_id: str | None = None) -> Iterator[_TraceSpan]:
        """Trace a harness run."""
        spans = [
            (
                option,
                f"invoke_agent {option.agent_name}",
                {
                    "gen_ai.operation.name": "invoke_agent",
                    "gen_ai.agent.name": option.agent_name,
                    "gen_ai.agent.description": option.agent_description,
                    "gen_ai.conversation.id": option.conversation_id or conversation_id,
                },
            )
            for option in self.options
        ]
        with self._span("internal", spans) as span:
            yield span

    @contextmanager
    def model(self, model: Any) -> Iterator[_TraceSpan]:
        """Trace one model request."""
        model_name = str(getattr(model, "model", "unknown") or "unknown")
        provider_name = getattr(getattr(model, "provider", None), "name", None)
        attributes = {
            "gen_ai.operation.name": "chat",
            "gen_ai.provider.name": provider_name,
            "gen_ai.request.model": model_name,
        }
        with self._span("client", [(option, f"chat {model_name}", attributes) for option in self.options]) as span:
            yield span

    @contextmanager
    def tool(self, *, tool_name: str, call_id: str, arguments: str) -> Iterator[_TraceSpan]:
        """Trace one local tool execution."""
        spans = [
            (
                option,
                f"execute_tool {tool_name}",
                {
                    "gen_ai.operation.name": "execute_tool",
                    "gen_ai.tool.name": tool_name,
                    "gen_ai.tool.call.id": call_id,
                    "gen_ai.tool.type": "function",
                    "gen_ai.tool.call.arguments": arguments if option.capture_tool_args else None,
                },
            )
            for option in self.options
        ]
        with self._span("internal", spans) as span:
            yield span

    @contextmanager
    def _span(self, kind: str, spans: list[tuple[TracingOptions, str, Json]]) -> Iterator[_TraceSpan]:
        """Start OpenTelemetry spans for every configured sink."""
        if not self.options:
            yield _TraceSpan([])
            return

        with ExitStack() as stack:
            adapters = [
                (stack.enter_context(_start_span(option, name, kind, attributes)), option)
                for option, name, attributes in spans
            ]
            yield _TraceSpan(adapters)


@contextmanager
def _start_span(option: TracingOptions, name: str, kind: str, attributes: Json) -> Iterator[_SpanAdapter]:
    """Start one OpenTelemetry-shaped span."""
    tracer = option.tracer
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


class _TraceSpan:
    """Span handle spanning every configured trace sink."""

    def __init__(self, spans: list[tuple[_SpanAdapter, TracingOptions]]) -> None:
        self.spans = spans

    def for_each(self, callback: Callable[[_SpanAdapter, TracingOptions], None]) -> None:
        """Run a callback with each child span and its capture policy."""
        for span, option in self.spans:
            callback(span, option)

    def set_attributes(self, attributes: Json) -> None:
        """Set attributes on every child span."""
        for span, _option in self.spans:
            span.set_attributes(attributes)

    def set_attribute(self, key: str, value: Any) -> None:
        """Set one attribute on every child span."""
        for span, _option in self.spans:
            span.set_attribute(key, value)

    def set_attribute_where(self, predicate: Callable[[TracingOptions], bool], key: str, value: Any) -> None:
        """Set one attribute on child spans whose capture policy allows it."""
        for span, option in self.spans:
            if predicate(option):
                span.set_attribute(key, value)

    def record_exception(self, exc: BaseException) -> None:
        """Record an exception on every child span."""
        for span, _option in self.spans:
            span.record_exception(exc)

    def set_error(self, message: str, error_type: str | None = None) -> None:
        """Mark every child span as failed."""
        for span, _option in self.spans:
            span.set_error(message, error_type)


def _local_tracing_options(
    local: LocalTracing,
    *,
    agent_name: str,
    agent_description: str | None,
    conversation_id: str | None,
) -> TracingOptions:
    """Return full-capture options for a local JSONL trace sink."""
    return TracingOptions(
        tracer=local.tracer,
        agent_name=agent_name,
        agent_description=agent_description,
        conversation_id=conversation_id,
        capture_messages=True,
        capture_tool_args=True,
        capture_tool_results=True,
    )


def _encode_trace_project_path(project_root: str | Path) -> str:
    """Encode one project root as a portable trace-directory segment."""
    resolved = str(Path(project_root).expanduser().resolve())
    name = Path(resolved).name or "root"
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip(".-") or "root"
    digest = hashlib.sha1(resolved.encode()).hexdigest()[:10]
    return f"{slug}-{digest}"


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


class _LocalTraceTracer:
    """OpenTelemetry-shaped tracer that writes ended spans as JSONL."""

    def __init__(self, trace_dir: Path) -> None:
        self.trace_dir = trace_dir
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.current_span: contextvars.ContextVar[_LocalTraceSpan | None] = contextvars.ContextVar("thinharness_local_span", default=None)
        self.lock = threading.Lock()

    def start_as_current_span(self, name: str, **kwargs: Any) -> Any:
        """Start a local span context."""
        return _LocalTraceSpanContext(self, name, kwargs)

    def _write(self, span: _LocalTraceSpan) -> None:
        """Append one span record to its trace file."""
        path = self.trace_dir / f"{span.trace_id}.jsonl"
        with self.lock, path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(span.to_record(), ensure_ascii=False, default=str) + "\n")


class _LocalTraceSpanContext:
    """Context manager for one local span."""

    def __init__(self, tracer: _LocalTraceTracer, name: str, kwargs: dict[str, Any]) -> None:
        self.tracer = tracer
        self.name = name
        self.kwargs = kwargs
        self.span: _LocalTraceSpan | None = None
        self.token: contextvars.Token[_LocalTraceSpan | None] | None = None

    def __enter__(self) -> _LocalTraceSpan:
        """Start and bind the local span."""
        parent = self.tracer.current_span.get()
        self.span = _LocalTraceSpan(
            tracer=self.tracer,
            name=self.name,
            kind=str(self.kwargs.get("kind") or ""),
            attributes=dict(self.kwargs.get("attributes") or {}),
            trace_id=parent.trace_id if parent is not None else uuid.uuid4().hex,
            span_id=uuid.uuid4().hex,
            parent_id=parent.span_id if parent is not None else None,
            started_at=time.time(),
        )
        self.token = self.tracer.current_span.set(self.span)
        return self.span

    def __exit__(self, exc_type: Any, exc: BaseException | None, tb: Any) -> None:
        """End and unbind the local span."""
        assert self.span is not None
        assert self.token is not None
        if exc is not None:
            self.span.record_exception(exc)
        self.tracer.current_span.reset(self.token)
        self.span.end()


class _LocalTraceSpan:
    """Mutable local span record."""

    def __init__(
        self,
        *,
        tracer: _LocalTraceTracer,
        name: str,
        kind: str,
        attributes: Json,
        trace_id: str,
        span_id: str,
        parent_id: str | None,
        started_at: float,
    ) -> None:
        self.tracer = tracer
        self.name = name
        self.kind = kind
        self.attributes = attributes
        self.trace_id = trace_id
        self.span_id = span_id
        self.parent_id = parent_id
        self.started_at = started_at
        self.ended_at: float | None = None
        self.status: Any = None
        self.exceptions: list[Json] = []

    def set_attributes(self, attributes: Json) -> None:
        """Set span attributes."""
        self.attributes.update(attributes)

    def set_attribute(self, key: str, value: Any) -> None:
        """Set one span attribute."""
        self.attributes[key] = value

    def record_exception(self, exc: BaseException) -> None:
        """Record an exception."""
        self.exceptions.append({"type": type(exc).__name__, "message": str(exc)})

    def set_status(self, status: Any) -> None:
        """Set span status."""
        self.status = serialize_attribute_value(status)

    def end(self) -> None:
        """End the span and append it to disk."""
        if self.ended_at is not None:
            return
        self.ended_at = time.time()
        self.tracer._write(self)

    def to_record(self) -> Json:
        """Return the JSONL record for this span."""
        return {
            "type": "span",
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "name": self.name,
            "kind": self.kind,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_ms": (self.ended_at - self.started_at) * 1000 if self.ended_at is not None else None,
            "attributes": self.attributes,
            "status": self.status,
            "exceptions": self.exceptions,
        }


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
    output_messages = _otel_output_messages(turn)
    if capture_messages and output_messages and output_messages[0]["parts"]:
        attributes.update({
            "langfuse.observation.output": serialize_attribute_value(
                {"text": text} if text else {"messages": output_messages}
            ),
            "gen_ai.output.messages": serialize_attribute_value(output_messages),
        })
        if text:
            attributes["gen_ai.completion"] = text
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
    if snapshot.kind in {"tool_outputs", "output_retry_tool"} or (snapshot.kind == "background_completion" and snapshot.tool_outputs is not None):
        return {"tool_outputs": snapshot.tool_outputs or []}
    if snapshot.kind == "background_completion":
        return {"background_completion": snapshot.prompt}
    if snapshot.kind == "correction":
        return {"correction": snapshot.prompt}
    return None


def _otel_input_messages(snapshot: ModelTraceSnapshot) -> list[Json] | None:
    """Return OTel-shaped logical input messages."""
    if snapshot.kind == "background_completion":
        if snapshot.prompt is not None:
            return [{"role": "user", "parts": [{"type": "text", "content": snapshot.prompt}]}]
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
    if snapshot.kind in {"start", "resume", "correction"} and snapshot.prompt is not None:
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
