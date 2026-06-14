"""Internal tool execution policy and per-call lifecycle."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from .events import (
    _CURRENT_STREAM_EMITTER,
    BackgroundTaskCompletedEvent,
    BackgroundTaskStartedEvent,
    RunStreamContext,
    StreamEmitter,
    ToolCallCompletedEvent,
    ToolCallStartedEvent,
)
from .hooks import _CURRENT_TOOL_CALL, _CURRENT_TOOL_RUNTIME, AfterToolCallContext, BeforeToolCallContext
from .providers import ModelToolCall, ToolOutput
from .tools.base import Json, ToolEnvelope, ToolResult, ToolSpec, _invoke_tool
from .tracing import RunTracer, serialize_attribute_value

if TYPE_CHECKING:
    from .core import Harness
    from .runtime import RunContext
    from .tracing import _TraceSpan


MAX_PARALLEL_TOOL_WORKERS = 16


@dataclass(frozen=True)
class ToolCallExecution:
    """Internal per-call execution data with control-flow signals."""

    envelope: ToolEnvelope
    output: str
    cancelled: bool
    retry_kind: str | None = None
    background_start: BackgroundToolStart | None = None


@dataclass(frozen=True)
class BackgroundToolStart:
    """Prepared background execution to start after the start-notice span closes."""

    task_id: str
    tool_call_id: str
    tool_name: str
    arguments: str
    spec: ToolSpec
    run_metadata: Json


@dataclass
class BackgroundToolTask:
    """Strong reference and metadata for one pending background tool task."""

    task_id: str
    tool_call_id: str
    tool_name: str
    arguments: str
    task: asyncio.Task[BackgroundToolCompletion]
    started_at: float


@dataclass(frozen=True)
class BackgroundToolCompletion:
    """Result of one background tool task."""

    task_id: str
    tool_call_id: str
    tool_name: str
    envelope: ToolEnvelope
    output: str
    elapsed_ms: float
    failed: bool
    event: str = "completed"

    def record(self) -> Json:
        """Return the tool_call_records entry for this completion."""
        return {
            "background": {
                "task_id": self.task_id,
                "tool_call_id": self.tool_call_id,
                "tool_name": self.tool_name,
                "event": self.event,
                "elapsed_ms": self.elapsed_ms,
            },
            "output": self.output,
        }


class BackgroundToolManager:
    """Own background tool tasks for one Harness.run invocation."""

    def __init__(self, *, run_tracer: RunTracer, emitter: StreamEmitter | None = None, stream: RunStreamContext | None = None) -> None:
        self.run_tracer = run_tracer
        self.emitter = emitter
        self.stream = stream
        self._next_id = 1
        self._pending: dict[asyncio.Task[BackgroundToolCompletion], BackgroundToolTask] = {}
        self._ready: list[BackgroundToolCompletion] = []
        self._ready_event = asyncio.Event()

    def allocate_id(self) -> str:
        """Return the next stable per-run background task id."""
        task_id = f"bg_{self._next_id}"
        self._next_id += 1
        return task_id

    def has_pending_or_ready(self) -> bool:
        """Return whether background work is pending or ready for delivery."""
        self._harvest_done()
        return bool(self._pending or self._ready)

    def start_many(self, starts: list[BackgroundToolStart]) -> None:
        """Start prepared background tasks under the current agent span."""
        for start in starts:
            self.start(start)

    def start(self, start: BackgroundToolStart) -> None:
        """Start one prepared background task."""
        started_at = time.perf_counter()
        if self.emitter is not None and self.stream is not None:
            self._emit(BackgroundTaskStartedEvent(
                **self._base(),
                background_task_id=start.task_id,
                tool_call_id=start.tool_call_id,
                tool_name=start.tool_name,
            ))
        task = asyncio.create_task(self._run(start, started_at))
        self._pending[task] = BackgroundToolTask(
            task_id=start.task_id,
            tool_call_id=start.tool_call_id,
            tool_name=start.tool_name,
            arguments=start.arguments,
            task=task,
            started_at=started_at,
        )
        task.add_done_callback(self._collect_task_completion)

    def drain_ready(self) -> list[BackgroundToolCompletion]:
        """Return all completed background results that have not been delivered."""
        self._harvest_done()
        ready = self._ready
        self._ready = []
        self._ready_event.clear()
        return ready

    async def wait_next_ready(self) -> BackgroundToolCompletion:
        """Wait for and return the next ready background completion."""
        while True:
            self._ready_event.clear()
            self._harvest_done()
            if self._ready:
                completion = self._ready.pop(0)
                return completion
            if not self._pending:
                raise RuntimeError("no pending background tasks")
            await self._ready_event.wait()

    async def cancel_and_drain(self) -> list[BackgroundToolCompletion]:
        """Cancel all pending tasks and return any terminal completion records."""
        completions = self.drain_ready()
        tasks = list(self._pending)
        if not tasks:
            return completions
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._harvest_done()
        completions.extend(self.drain_ready())
        return completions

    def _harvest_done(self) -> None:
        """Move completed task results into the ready list."""
        for task in list(self._pending):
            if task.done():
                self._collect_task_completion(task)

    def _collect_task_completion(self, task: asyncio.Task[BackgroundToolCompletion]) -> None:
        """Record one completed task exactly once."""
        meta = self._pending.pop(task, None)
        if meta is None:
            return
        try:
            completion = task.result()
        except asyncio.CancelledError:
            # _run normally converts cancellation into a completion; this is a fallback
            # for cancellation that prevents _run from returning its own record.
            completion = BackgroundToolCompletion(
                task_id=meta.task_id,
                tool_call_id=meta.tool_call_id,
                tool_name=meta.tool_name,
                envelope=ToolResult(False, "Background task cancelled", {"error_type": "Cancelled"}),
                output=ToolResult(False, "Background task cancelled", {"error_type": "Cancelled"}).to_json(),
                elapsed_ms=(time.perf_counter() - meta.started_at) * 1000,
                failed=True,
                event="cancelled",
            )
            self._emit_completion(completion)
        except Exception as exc:
            # _run normally converts failures into a completion; keep collection robust
            # if an exception escapes that normalization path.
            completion = BackgroundToolCompletion(
                task_id=meta.task_id,
                tool_call_id=meta.tool_call_id,
                tool_name=meta.tool_name,
                envelope=ToolResult(False, f"{type(exc).__name__}: {exc}", {"error_type": type(exc).__name__}),
                output=ToolResult(False, f"{type(exc).__name__}: {exc}", {"error_type": type(exc).__name__}).to_json(),
                elapsed_ms=(time.perf_counter() - meta.started_at) * 1000,
                failed=True,
                event="failed",
            )
            self._emit_completion(completion)
        self._ready.append(completion)
        self._ready_event.set()

    async def _run(self, start: BackgroundToolStart, started_at: float) -> BackgroundToolCompletion:
        """Invoke one background tool and normalize its completion."""
        call_token = _CURRENT_TOOL_CALL.set({"call_id": start.tool_call_id, "name": start.tool_name})
        runtime_token = _CURRENT_TOOL_RUNTIME.set({"run_metadata": dict(start.run_metadata)})
        emitter_token = _CURRENT_STREAM_EMITTER.set(self.emitter)
        try:
            try:
                with self.run_tracer.tool(tool_name=start.tool_name, call_id=start.tool_call_id, arguments=start.arguments) as span:
                    span.set_attributes({
                        "thinharness.background.task_id": start.task_id,
                        "thinharness.background.phase": "execution",
                        "thinharness.background.original_tool_call_id": start.tool_call_id,
                    })
                    envelope = await _invoke_tool(start.spec, start.arguments)
                    output = envelope.to_json()
                    failed = not envelope.ok
                    span.set_attribute_where(
                        lambda option: option.capture_tool_results,
                        "gen_ai.tool.call.result",
                        serialize_attribute_value(output),
                    )
                    if failed:
                        span.set_error(f'Background tool "{start.tool_name}" failed', envelope.error_type() or "ToolExecutionError")
                    completion = BackgroundToolCompletion(
                        task_id=start.task_id,
                        tool_call_id=start.tool_call_id,
                        tool_name=start.tool_name,
                        envelope=envelope,
                        output=output,
                        elapsed_ms=(time.perf_counter() - started_at) * 1000,
                        failed=failed,
                    )
                    self._emit_completion(completion)
                    return completion
            except asyncio.CancelledError:
                completion = BackgroundToolCompletion(
                    task_id=start.task_id,
                    tool_call_id=start.tool_call_id,
                    tool_name=start.tool_name,
                    envelope=ToolResult(False, "Background task cancelled", {"error_type": "Cancelled"}),
                    output=ToolResult(False, "Background task cancelled", {"error_type": "Cancelled"}).to_json(),
                    elapsed_ms=(time.perf_counter() - started_at) * 1000,
                    failed=True,
                    event="cancelled",
                )
                self._emit_completion(completion)
                return completion
            except Exception as exc:
                completion = BackgroundToolCompletion(
                    task_id=start.task_id,
                    tool_call_id=start.tool_call_id,
                    tool_name=start.tool_name,
                    envelope=ToolResult(False, f"{type(exc).__name__}: {exc}", {"error_type": type(exc).__name__}),
                    output=ToolResult(False, f"{type(exc).__name__}: {exc}", {"error_type": type(exc).__name__}).to_json(),
                    elapsed_ms=(time.perf_counter() - started_at) * 1000,
                    failed=True,
                    event="failed",
                )
                self._emit_completion(completion)
                return completion
        finally:
            _CURRENT_STREAM_EMITTER.reset(emitter_token)
            _CURRENT_TOOL_RUNTIME.reset(runtime_token)
            _CURRENT_TOOL_CALL.reset(call_token)

    def _base(self) -> dict[str, Any]:
        """Return common stream event fields for background events."""
        assert self.stream is not None
        return {
            "run_id": self.stream.run_id,
            "sequence": 0,
            "parent_run_id": self.stream.parent_run_id,
            "parent_tool_call_id": self.stream.parent_tool_call_id,
            "agent_name": self.stream.agent_name,
        }

    def _emit(self, event: Any) -> None:
        """Emit one background event if streaming is active."""
        if self.emitter is not None and self.stream is not None:
            self.emitter.emit(event)

    def _emit_completion(self, completion: BackgroundToolCompletion) -> None:
        """Emit a public background completion event."""
        if self.emitter is None or self.stream is None:
            return
        status: Literal["completed", "failed", "cancelled"]
        if completion.event == "cancelled":
            status = "cancelled"
        elif completion.failed:
            status = "failed"
        else:
            status = "completed"
        self._emit(BackgroundTaskCompletedEvent(
            **self._base(),
            background_task_id=completion.task_id,
            tool_call_id=completion.tool_call_id,
            tool_name=completion.tool_name,
            status=status,
            elapsed_ms=completion.elapsed_ms,
            output=completion.output,
        ))


def background_completion_message(completion: BackgroundToolCompletion) -> str:
    """Return the synthetic model-facing completion message."""
    status = "failed" if completion.failed else "completed"
    return (
        f"Background task {completion.task_id} completed.\n"
        f"Tool: {completion.tool_name}\n"
        f"Status: {status}\n"
        f"Elapsed: {completion.elapsed_ms:.0f} ms\n"
        "Output:\n"
        f"{completion.output}"
    )


class ToolBatchExecutor:
    """Execute one model-requested tool batch."""

    def __init__(
        self,
        *,
        harness: Harness,
        run_context: RunContext,
        tool_map: dict[str, ToolSpec],
        run_tracer: RunTracer,
        tool_execution: str,
    ) -> None:
        self.harness = harness
        self.run_context = run_context
        self.tool_map = tool_map
        self.run_tracer = run_tracer
        self.tool_execution = tool_execution
        self.call_executor = ToolCallExecutor(
            harness=harness,
            run_context=run_context,
            tool_map=tool_map,
            run_tracer=run_tracer,
            tool_execution=tool_execution,
        )

    async def execute_batch(
        self,
        calls: list[ModelToolCall],
        *,
        tool_indices: list[int] | None = None,
    ) -> tuple[list[Json], list[ToolOutput], list[ToolCallExecution]]:
        """Run one batch of model tool calls; preserve model order in returned outputs."""
        indices = tool_indices or list(range(len(calls)))
        if len(indices) != len(calls):
            raise ValueError("tool_indices length must match calls length")
        if self._should_run_sequentially(calls):
            results = []
            for index, call in zip(indices, calls, strict=True):
                execution = await self.call_executor.execute_one(call, index)
                self._start_background(execution)
                results.append(execution)
        else:
            results = await self._run_calls_concurrently(calls, indices)
        records = []
        for call, execution in zip(calls, results, strict=True):
            record = {"call": {"id": call.id, "name": call.name, "arguments": call.arguments}, "output": execution.output}
            if execution.cancelled:
                record["cancelled"] = True
            if execution.background_start is not None:
                record["background"] = {
                    "task_id": execution.background_start.task_id,
                    "status": "running",
                }
            records.append(record)
        outputs = [ToolOutput(call.id, execution.output) for call, execution in zip(calls, results, strict=True)]
        return records, outputs, results

    def _start_background(self, execution: ToolCallExecution) -> None:
        """Start a prepared background task once its start-notice span has closed."""
        if execution.background_start is None:
            return
        assert self.run_context.background is not None
        self.run_context.background.start(execution.background_start)

    def _should_run_sequentially(self, calls: list[ModelToolCall]) -> bool:
        """Decide whether the batch must execute serially."""
        if self.tool_execution == "sequential" or len(calls) <= 1:
            return True
        return any((spec := self.tool_map.get(str(call.name))) is not None and spec.sequential for call in calls)

    async def _run_calls_concurrently(self, calls: list[ModelToolCall], indices: list[int]) -> list[ToolCallExecution]:
        """Execute calls concurrently while preserving model request order."""
        sem = asyncio.Semaphore(MAX_PARALLEL_TOOL_WORKERS)

        async def invoke(index: int, call: ModelToolCall) -> ToolCallExecution:
            """Invoke one traced tool call under the shared concurrency limit."""
            async with sem:
                execution = await self.call_executor.execute_one(call, index)
                self._start_background(execution)
                return execution

        tasks = [asyncio.create_task(invoke(index, call)) for index, call in zip(indices, calls, strict=True)]
        task_index = {task: index for index, task in enumerate(tasks)}
        results: list[ToolCallExecution | None] = [None] * len(tasks)
        pending = set(tasks)
        try:
            while pending:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_EXCEPTION)
                for task in done:
                    exc = task.exception()
                    if exc is not None:
                        for sibling in pending:
                            sibling.cancel()
                        await asyncio.gather(*pending, return_exceptions=True)
                        raise exc
                    results[task_index[task]] = task.result()
        except BaseException:
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            raise
        return [result for result in results if result is not None]


class ToolCallExecutor:
    """Execute one model-requested tool call."""

    def __init__(
        self,
        *,
        harness: Harness,
        run_context: RunContext,
        tool_map: dict[str, ToolSpec],
        run_tracer: RunTracer,
        tool_execution: str,
    ) -> None:
        self.harness = harness
        self.run_context = run_context
        self.tool_map = tool_map
        self.run_tracer = run_tracer
        self.tool_execution = tool_execution

    async def execute_one(self, call: ModelToolCall, index: int) -> ToolCallExecution:
        """Execute one model tool call with tracing."""
        with self.run_tracer.tool(tool_name=call.name, call_id=call.id, arguments=call.arguments) as span:
            call_token = _CURRENT_TOOL_CALL.set({"call_id": call.id, "name": call.name})
            runtime_token = _CURRENT_TOOL_RUNTIME.set({"run_metadata": dict(self.run_context.metadata)})
            emitter_token = _CURRENT_STREAM_EMITTER.set(self.run_context.emitter)
            cancelled = False
            background_start: BackgroundToolStart | None = None
            start = time.perf_counter()
            completed_emitted = False
            envelope: ToolEnvelope | None = None
            output: str | None = None
            retry_kind: str | None = None
            try:
                spec = self.tool_map.get(str(call.name))
                before = BeforeToolCallContext(
                    harness=self.harness,
                    metadata=dict(self.run_context.metadata),
                    call_id=call.id,
                    tool_name=call.name,
                    arguments=call.arguments,
                    tool_spec=spec,
                    tool_index=index,
                )
                self.run_context.emit(ToolCallStartedEvent(
                    **self.run_context.stream_base(),
                    call_id=call.id,
                    tool_name=call.name,
                    tool_index=index,
                    arguments=call.arguments,
                ))
                self.harness.hooks.fire(before)
                if before.cancelled:
                    cancelled = True
                    reason = before.cancel_reason or "unspecified"
                    envelope = ToolResult(False, f"Tool execution blocked by hook: {reason}", {"error_type": "ToolCallCancelled"})
                else:
                    decision = self._background_decision(call, spec)
                    if decision.error_output is not None:
                        envelope = decision.error_output
                    elif decision.start is not None:
                        background_start = decision.start
                        envelope = _background_start_output(decision.start)
                    else:
                        envelope = await self._call_output(call.name, decision.arguments)
                output = envelope.to_json()
                retry_kind = None if cancelled else envelope.retry_kind()
                after = AfterToolCallContext(
                    harness=self.harness,
                    metadata=dict(self.run_context.metadata),
                    call_id=call.id,
                    tool_name=call.name,
                    arguments=call.arguments,
                    original_output=output,
                    output=output,
                    envelope=envelope,
                    duration_ms=(time.perf_counter() - start) * 1000,
                )
                self.harness.hooks.fire_after_tool_call(after)
                output = after.output
                envelope = after.envelope
                self._annotate_special_tool(span, call.name, envelope)
                span.set_attribute_where(
                    lambda option: option.capture_tool_results,
                    "gen_ai.tool.call.result",
                    serialize_attribute_value(output),
                )
                if retry_kind is not None:
                    span.set_error(f'Tool "{call.name}" failed', retry_kind)
                elif not envelope.ok:
                    span.set_error(f'Tool "{call.name}" failed', "ToolExecutionError")
                self._emit_completed(
                    call=call,
                    output=output,
                    envelope=envelope,
                    cancelled=cancelled,
                    retry_kind=retry_kind,
                    duration_ms=(time.perf_counter() - start) * 1000,
                    background_start=background_start,
                )
                completed_emitted = True
                return ToolCallExecution(envelope=envelope, output=output, cancelled=cancelled, retry_kind=retry_kind, background_start=background_start)
            except Exception as exc:
                if not completed_emitted:
                    self.run_context.emit(ToolCallCompletedEvent(
                        **self.run_context.stream_base(),
                        call_id=call.id,
                        tool_name=call.name,
                        ok=False,
                        cancelled=cancelled,
                        retry_kind=retry_kind,
                        error_type=type(exc).__name__,
                        message=str(exc),
                        duration_ms=(time.perf_counter() - start) * 1000,
                        output=output,
                    ))
                raise
            finally:
                _CURRENT_STREAM_EMITTER.reset(emitter_token)
                _CURRENT_TOOL_RUNTIME.reset(runtime_token)
                _CURRENT_TOOL_CALL.reset(call_token)

    def _emit_completed(
        self,
        *,
        call: ModelToolCall,
        output: str,
        envelope: ToolEnvelope,
        cancelled: bool,
        retry_kind: str | None,
        duration_ms: float,
        background_start: BackgroundToolStart | None,
    ) -> None:
        """Emit a public tool completion event."""
        self.run_context.emit(ToolCallCompletedEvent(
            **self.run_context.stream_base(),
            call_id=call.id,
            tool_name=call.name,
            ok=envelope.ok,
            cancelled=cancelled,
            retry_kind=retry_kind,
            error_type=envelope.error_type(),
            message=envelope.content if not envelope.ok else None,
            duration_ms=duration_ms,
            output=output,
            background_task_id=background_start.task_id if background_start is not None else None,
            background_status="running" if background_start is not None else None,
        ))

    async def _call_output(self, name: str, arguments: str) -> ToolEnvelope:
        """Execute one model tool call and format its output."""
        spec = self.tool_map.get(str(name))
        if not spec:
            return ToolResult(False, f"unknown tool {name}", {"tool": name})
        return await _invoke_tool(spec, arguments)

    def _background_decision(self, call: ModelToolCall, spec: ToolSpec | None) -> _BackgroundDecision:
        """Return how to handle the private _background argument for one call."""
        if spec is None:
            return _BackgroundDecision(arguments=call.arguments)
        parsed = _parse_background_args(call.arguments)
        mode = spec.background
        known_target = True
        strip_private_arg = False
        unsupported_message: str | None = None
        if spec.background_policy is not None and parsed.args is not None:
            policy = spec.background_policy(parsed.args)
            mode = policy.mode
            known_target = policy.known_target
            strip_private_arg = policy.strip_private_arg
            unsupported_message = policy.unsupported_message
        if mode == "model":
            if parsed.error is not None:
                return _BackgroundDecision(arguments=call.arguments, error_output=_retry_output("InvalidArguments", parsed.error))
            arguments = parsed.stripped_arguments if parsed.present else call.arguments
            if parsed.requested and self.tool_execution != "sequential":
                return _BackgroundDecision(arguments=arguments, start=self._background_start(call, spec, arguments))
            return _BackgroundDecision(arguments=arguments)
        if mode == "always":
            if self.tool_execution == "sequential":
                return _BackgroundDecision(arguments=call.arguments)
            arguments = parsed.stripped_arguments if strip_private_arg and parsed.present else call.arguments
            return _BackgroundDecision(arguments=arguments, start=self._background_start(call, spec, arguments))
        if parsed.present and parsed.requested and known_target and unsupported_message is not None:
            return _BackgroundDecision(
                arguments=parsed.stripped_arguments,
                error_output=_retry_output("InvalidArguments", unsupported_message),
            )
        if strip_private_arg and parsed.present:
            return _BackgroundDecision(arguments=parsed.stripped_arguments)
        return _BackgroundDecision(arguments=call.arguments)

    def _background_start(self, call: ModelToolCall, spec: ToolSpec, arguments: str) -> BackgroundToolStart:
        """Build a prepared background execution."""
        assert self.run_context.background is not None
        return BackgroundToolStart(
            task_id=self.run_context.background.allocate_id(),
            tool_call_id=call.id,
            tool_name=call.name,
            arguments=arguments,
            spec=spec,
            run_metadata=dict(self.run_context.metadata),
        )

    def _annotate_special_tool(self, span: _TraceSpan, name: str, envelope: ToolEnvelope) -> None:
        """Add tool-family trace attributes for framework and MCP tools."""
        if name == "subagent":
            span.set_attributes({
                "subagent.name": envelope.metadata.get("agent"),
                "subagent.tool_mode": envelope.metadata.get("tool_mode"),
                "subagent.tools": envelope.metadata.get("tools"),
            })
        spec = self.tool_map.get(str(name))
        if spec is not None and spec.mcp is not None:
            span.set_attributes({
                "mcp.server.id": spec.mcp.server_id,
                "mcp.tool.name": spec.mcp.tool_name,
            })


@dataclass(frozen=True)
class _ParsedBackgroundArgs:
    """Parsed private _background argument state."""

    present: bool
    requested: bool
    stripped_arguments: str
    args: Json | None
    error: str | None = None


@dataclass(frozen=True)
class _BackgroundDecision:
    """Decision for one call's background/private-argument handling."""

    arguments: str
    start: BackgroundToolStart | None = None
    error_output: ToolEnvelope | None = None


def _parse_background_args(arguments: str) -> _ParsedBackgroundArgs:
    """Parse and strip the private _background argument if present."""
    try:
        args = json.loads(arguments or "{}")
    except json.JSONDecodeError:
        return _ParsedBackgroundArgs(False, False, arguments, None)
    if not isinstance(args, dict):
        return _ParsedBackgroundArgs(False, False, arguments, None)
    if "_background" not in args:
        return _ParsedBackgroundArgs(False, False, arguments, args)
    raw = args["_background"]
    if not isinstance(raw, bool):
        stripped = dict(args)
        stripped.pop("_background", None)
        return _ParsedBackgroundArgs(
            True,
            False,
            json.dumps(stripped, ensure_ascii=False, separators=(",", ":")),
            stripped,
            "_background must be a boolean",
        )
    stripped = dict(args)
    stripped.pop("_background", None)
    return _ParsedBackgroundArgs(
        True,
        raw,
        json.dumps(stripped, ensure_ascii=False, separators=(",", ":")),
        stripped,
    )


def _background_start_output(start: BackgroundToolStart) -> ToolEnvelope:
    """Return the immediate model-visible start notice."""
    return ToolResult(
        True,
        f"Started background task {start.task_id} for tool {start.tool_name}. Continue other work; the harness will notify you when it finishes.",
        {
            "background_task_id": start.task_id,
            "tool_name": start.tool_name,
            "status": "running",
        },
    )


def _retry_output(error_type: str, message: str) -> ToolEnvelope:
    """Return a retryable argument error output."""
    return ToolResult(False, message, {"error_type": error_type, "retry": True})
