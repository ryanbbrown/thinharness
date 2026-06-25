"""Internal tool execution policy and per-call lifecycle."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .events import (
    _CURRENT_STREAM_EMITTER,
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
                results.append(execution)
        else:
            results = await self._run_calls_concurrently(calls, indices)
        records = []
        for call, execution in zip(calls, results, strict=True):
            record = {"call": {"id": call.id, "name": call.name, "arguments": call.arguments}, "output": execution.output}
            if execution.cancelled:
                record["cancelled"] = True
            records.append(record)
        outputs = [ToolOutput(call.id, execution.output) for call, execution in zip(calls, results, strict=True)]
        return records, outputs, results

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
                return await self.call_executor.execute_one(call, index)

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
    ) -> None:
        self.harness = harness
        self.run_context = run_context
        self.tool_map = tool_map
        self.run_tracer = run_tracer

    async def execute_one(self, call: ModelToolCall, index: int) -> ToolCallExecution:
        """Execute one model tool call with tracing."""
        with self.run_tracer.tool(tool_name=call.name, call_id=call.id, arguments=call.arguments) as span:
            call_token = _CURRENT_TOOL_CALL.set({"call_id": call.id, "name": call.name})
            runtime_token = _CURRENT_TOOL_RUNTIME.set({"run_metadata": dict(self.run_context.metadata)})
            emitter_token = _CURRENT_STREAM_EMITTER.set(self.run_context.emitter)
            cancelled = False
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
                    envelope = await self._call_output(call.name, call.arguments)
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
                )
                completed_emitted = True
                return ToolCallExecution(envelope=envelope, output=output, cancelled=cancelled, retry_kind=retry_kind)
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
        ))

    async def _call_output(self, name: str, arguments: str) -> ToolEnvelope:
        """Execute one model tool call and format its output."""
        spec = self.tool_map.get(str(name))
        if not spec:
            return ToolResult(False, f"unknown tool {name}", {"tool": name})
        return await _invoke_tool(spec, arguments)

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
