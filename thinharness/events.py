"""Public stream events and internal stream delivery helpers."""

from __future__ import annotations

import asyncio
import contextvars
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field, replace
from typing import Literal, cast

from .types import ApprovalDecision, HarnessResult, Json, StopReason

StreamEventKind = Literal[
    "run_started",
    "model_request_started",
    "model_message",
    "tool_call_started",
    "tool_call_completed",
    "background_task_started",
    "background_task_completed",
    "model_retry",
    "limit_warning",
    "approval_resumed",
    "run_completed",
    "run_failed",
]


@dataclass(frozen=True)
class StreamToolCall:
    """A model-requested tool call summary."""

    id: str
    name: str


@dataclass(frozen=True, kw_only=True)
class StreamOptions:
    """Visibility controls for Harness.stream()."""

    include_model_text: bool = True
    include_subagents: bool = True


@dataclass(frozen=True, kw_only=True)
class StreamEvent:
    """Base event metadata shared by every harness stream event."""

    kind: StreamEventKind
    run_id: str
    sequence: int
    parent_run_id: str | None = None
    parent_tool_call_id: str | None = None
    agent_name: str | None = None
    metadata: Json = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class RunStartedEvent(StreamEvent):
    """A harness run has started."""

    kind: Literal["run_started"] = "run_started"
    prompt: str | None = None
    root: str = ""
    max_model_requests: int = 0
    max_tool_calls: int | None = None


@dataclass(frozen=True, kw_only=True)
class ApprovalResumedEvent(StreamEvent):
    """An approval pause has resumed with host decisions."""

    kind: Literal["approval_resumed"] = "approval_resumed"
    decisions: tuple[ApprovalDecision, ...] = ()


@dataclass(frozen=True, kw_only=True)
class ModelRequestStartedEvent(StreamEvent):
    """A provider request is about to be made."""

    kind: Literal["model_request_started"] = "model_request_started"
    request_kind: Literal["start", "resume", "tool_outputs", "approval_resume", "correction", "output_retry_tool", "background_completion"] = "start"
    model: str = ""
    provider: str | None = None


@dataclass(frozen=True, kw_only=True)
class ModelMessageEvent(StreamEvent):
    """A complete provider turn has been received."""

    kind: Literal["model_message"] = "model_message"
    text: str = ""
    tool_calls: tuple[StreamToolCall, ...] = ()
    finalized_output_mode: str | None = None


@dataclass(frozen=True, kw_only=True)
class ToolCallStartedEvent(StreamEvent):
    """A local tool call is about to execute or be cancelled by hooks."""

    kind: Literal["tool_call_started"] = "tool_call_started"
    call_id: str = ""
    tool_name: str = ""
    tool_index: int = 0
    arguments: str | None = None


@dataclass(frozen=True, kw_only=True)
class ToolCallCompletedEvent(StreamEvent):
    """A local tool call has reached a model-visible terminal output."""

    kind: Literal["tool_call_completed"] = "tool_call_completed"
    call_id: str = ""
    tool_name: str = ""
    ok: bool | None = None
    cancelled: bool = False
    retry_kind: str | None = None
    error_type: str | None = None
    message: str | None = None
    duration_ms: float | None = None
    output: str | None = None
    background_task_id: str | None = None
    background_status: Literal["running"] | None = None


@dataclass(frozen=True, kw_only=True)
class BackgroundTaskStartedEvent(StreamEvent):
    """A background tool task has been scheduled."""

    kind: Literal["background_task_started"] = "background_task_started"
    background_task_id: str = ""
    tool_call_id: str = ""
    tool_name: str = ""


@dataclass(frozen=True, kw_only=True)
class BackgroundTaskCompletedEvent(StreamEvent):
    """A background tool task has finished."""

    kind: Literal["background_task_completed"] = "background_task_completed"
    background_task_id: str = ""
    tool_call_id: str = ""
    tool_name: str = ""
    status: Literal["completed", "failed", "cancelled"] = "completed"
    elapsed_ms: float = 0.0
    output: str | None = None


@dataclass(frozen=True, kw_only=True)
class ModelRetryEvent(StreamEvent):
    """The harness is giving the model an in-budget retry opportunity."""

    kind: Literal["model_retry"] = "model_retry"
    retry_kind: Literal["structured_output", "tool_retry"] = "structured_output"
    message: str = ""
    call_id: str | None = None


@dataclass(frozen=True, kw_only=True)
class LimitWarningEvent(StreamEvent):
    """A near-limit warning has been sent to the model."""

    kind: Literal["limit_warning"] = "limit_warning"
    limit_kind: Literal["model_requests", "tool_calls"] = "model_requests"
    remaining: int = 0
    content: str = ""


@dataclass(frozen=True, kw_only=True)
class RunCompletedEvent(StreamEvent):
    """A harness run completed successfully."""

    kind: Literal["run_completed"] = "run_completed"
    result: HarnessResult


@dataclass(frozen=True, kw_only=True)
class RunFailedEvent(StreamEvent):
    """A harness run failed and async iteration will raise the same failure."""

    kind: Literal["run_failed"] = "run_failed"
    stop_reason: StopReason = "error"
    error_type: str = ""
    message: str = ""


HarnessStreamEvent = (
    RunStartedEvent
    | ModelRequestStartedEvent
    | ModelMessageEvent
    | ToolCallStartedEvent
    | ToolCallCompletedEvent
    | BackgroundTaskStartedEvent
    | BackgroundTaskCompletedEvent
    | ModelRetryEvent
    | LimitWarningEvent
    | ApprovalResumedEvent
    | RunCompletedEvent
    | RunFailedEvent
)


@dataclass
class RunStreamContext:
    """Internal stream identity and sequence state for one run."""

    run_id: str
    parent_run_id: str | None
    parent_tool_call_id: str | None
    agent_name: str | None
    options: StreamOptions
    sequence: int = 0


_STREAM_SENTINEL = object()


class StreamEmitter:
    """Unbounded per-run event queue producer."""

    def __init__(self, ctx: RunStreamContext) -> None:
        self.ctx = ctx
        self._queue: asyncio.Queue[HarnessStreamEvent | object] = asyncio.Queue()
        self._closed = False
        self._finished = False

    @property
    def queue(self) -> asyncio.Queue[HarnessStreamEvent | object]:
        """Return the consumer queue."""
        return self._queue

    def emit(self, event: HarnessStreamEvent) -> None:
        """Assign delivery sequence and enqueue an event."""
        if self._closed or self._finished:
            return
        self.ctx.sequence += 1
        sequenced = replace(event, sequence=self.ctx.sequence)
        self._queue.put_nowait(sequenced)

    def emit_forwarded(self, event: HarnessStreamEvent) -> None:
        """Forward a child event through this stream with this stream's sequence."""
        self.emit(event)

    def finish(self) -> None:
        """Signal stream completion to the consumer."""
        if self._finished:
            return
        self._finished = True
        self._queue.put_nowait(_STREAM_SENTINEL)

    def close(self) -> None:
        """Stop accepting new events and unblock the consumer."""
        self._closed = True
        self.finish()


_CURRENT_STREAM_EMITTER: contextvars.ContextVar[StreamEmitter | None] = contextvars.ContextVar("thinharness_stream_emitter", default=None)


def current_stream_emitter() -> StreamEmitter | None:
    """Return the current stream emitter for tool/subagent internals."""
    return _CURRENT_STREAM_EMITTER.get()


def new_run_id() -> str:
    """Return a framework-generated public run id."""
    return f"run_{uuid.uuid4().hex}"


class HarnessStream(AsyncIterator[HarnessStreamEvent]):
    """Async iterator and context manager returned by Harness.stream()."""

    def __init__(self, task: asyncio.Task[HarnessResult], emitter: StreamEmitter) -> None:
        self._task = task
        self._emitter = emitter
        self._closed = False

    @property
    def run_id(self) -> str:
        """Return the root run id for this stream."""
        return self._emitter.ctx.run_id

    def __aiter__(self) -> HarnessStream:
        return self

    async def __anext__(self) -> HarnessStreamEvent:
        if self._closed:
            raise StopAsyncIteration
        item = await self._emitter.queue.get()
        if item is _STREAM_SENTINEL:
            self._closed = True
            if self._task.cancelled():
                raise asyncio.CancelledError()
            if self._task.done():
                exc = self._task.exception()
                if exc is not None:
                    raise exc
            raise StopAsyncIteration
        event = cast(HarnessStreamEvent, item)
        return event

    async def aclose(self) -> None:
        """Cancel the underlying run if it is still active."""
        if self._closed:
            return
        self._closed = True
        if not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        elif not self._task.cancelled():
            self._task.exception()
        self._emitter.close()

    async def __aenter__(self) -> HarnessStream:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()


def create_stream_context(
    *,
    parent_run_id: str | None = None,
    parent_tool_call_id: str | None = None,
    agent_name: str | None = None,
    options: StreamOptions | None = None,
) -> RunStreamContext:
    """Create a fresh stream context for one run."""
    return RunStreamContext(
        run_id=new_run_id(),
        parent_run_id=parent_run_id,
        parent_tool_call_id=parent_tool_call_id,
        agent_name=agent_name,
        options=options or StreamOptions(),
    )
