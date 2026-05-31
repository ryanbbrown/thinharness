"""Internal tool execution policy and per-call lifecycle."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .hooks import _CURRENT_TOOL_CALL, _CURRENT_TOOL_RUNTIME, AfterToolCallContext, BeforeToolCallContext
from .providers import ModelToolCall, ToolOutput
from .tools.base import Json, ToolSpec, _invoke_tool
from .tracing import RunTracer, serialize_attribute_value

if TYPE_CHECKING:
    from .core import Harness
    from .runtime import RunContext
    from .tracing import _TraceSpan


MAX_PARALLEL_TOOL_WORKERS = 16


@dataclass(frozen=True)
class ToolCallExecution:
    """Internal per-call execution data with control-flow signals."""

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
    ) -> tuple[list[Json], list[ToolOutput], list[ToolCallExecution]]:
        """Run one batch of model tool calls; preserve model order in returned outputs."""
        if self._should_run_sequentially(calls):
            results = [
                await self.call_executor.execute_one(call, index)
                for index, call in enumerate(calls)
            ]
        else:
            results = await self._run_calls_concurrently(calls)
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

    async def _run_calls_concurrently(self, calls: list[ModelToolCall]) -> list[ToolCallExecution]:
        """Execute calls concurrently while preserving model request order."""
        sem = asyncio.Semaphore(MAX_PARALLEL_TOOL_WORKERS)

        async def invoke(index: int, call: ModelToolCall) -> ToolCallExecution:
            """Invoke one traced tool call under the shared concurrency limit."""
            async with sem:
                return await self.call_executor.execute_one(call, index)

        tasks = [asyncio.create_task(invoke(index, call)) for index, call in enumerate(calls)]
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
            cancelled = False
            start = time.perf_counter()
            try:
                before = BeforeToolCallContext(
                    harness=self.harness,
                    metadata=dict(self.run_context.metadata),
                    call_id=call.id,
                    tool_name=call.name,
                    arguments=call.arguments,
                    tool_spec=self.tool_map.get(str(call.name)),
                    tool_index=index,
                )
                self.harness.hooks.fire(before)
                if before.cancelled:
                    cancelled = True
                    reason = before.cancel_reason or "unspecified"
                    output = json.dumps({
                        "ok": False,
                        "content": f"Tool execution blocked by hook: {reason}",
                        "metadata": {"error_type": "ToolCallCancelled"},
                    }, ensure_ascii=False)
                else:
                    output = await self._call_output(call.name, call.arguments)
                parsed = _parse_tool_output(output)
                retry_kind = None if cancelled else _tool_retry_kind(parsed)
                after = AfterToolCallContext(
                    harness=self.harness,
                    metadata=dict(self.run_context.metadata),
                    call_id=call.id,
                    tool_name=call.name,
                    arguments=call.arguments,
                    original_output=output,
                    output=output,
                    parsed_output=parsed,
                    duration_ms=(time.perf_counter() - start) * 1000,
                )
                self.harness.hooks.fire_after_tool_call(after)
                output = after.output
                parsed = _parse_tool_output(output)
                self._annotate_special_tool(span, call.name, parsed)
                span.set_attribute_where(
                    lambda option: option.capture_tool_results,
                    "gen_ai.tool.call.result",
                    serialize_attribute_value(output),
                )
                if retry_kind is not None:
                    span.set_error(f'Tool "{call.name}" failed', retry_kind)
                elif parsed.get("ok") is False:
                    span.set_error(f'Tool "{call.name}" failed', "ToolExecutionError")
                return ToolCallExecution(output=output, cancelled=cancelled, retry_kind=retry_kind)
            finally:
                _CURRENT_TOOL_RUNTIME.reset(runtime_token)
                _CURRENT_TOOL_CALL.reset(call_token)

    async def _call_output(self, name: str, arguments: str) -> str:
        """Execute one model tool call and format its output."""
        spec = self.tool_map.get(str(name))
        if not spec:
            return json.dumps({"ok": False, "content": f"unknown tool {name}", "metadata": {"tool": name}}, ensure_ascii=False)
        return await _invoke_tool(spec, arguments)

    def _annotate_special_tool(self, span: _TraceSpan, name: str, parsed: Json) -> None:
        """Add tool-family trace attributes for framework and MCP tools."""
        metadata_value = parsed.get("metadata")
        parsed_metadata: dict[str, Any] = metadata_value if isinstance(metadata_value, dict) else {}
        if name == "subagent":
            span.set_attributes({
                "subagent.name": parsed_metadata.get("agent"),
                "subagent.tool_mode": parsed_metadata.get("tool_mode"),
                "subagent.tools": parsed_metadata.get("tools"),
            })
        spec = self.tool_map.get(str(name))
        spec_metadata = spec.metadata if spec is not None else {}
        if spec_metadata.get("source") == "mcp":
            span.set_attributes({
                "mcp.server.id": spec_metadata.get("mcp_server_id"),
                "mcp.tool.name": spec_metadata.get("mcp_tool_name"),
            })


def _parse_tool_output(output: str) -> Json:
    """Parse a normalized tool output envelope."""
    try:
        parsed = json.loads(output)
    except json.JSONDecodeError:
        return {"ok": False, "content": output, "metadata": {"error_type": "InvalidToolOutput"}}
    return parsed if isinstance(parsed, dict) else {"ok": False, "content": output, "metadata": {"error_type": "InvalidToolOutput"}}


def _tool_retry_kind(parsed: Json) -> str | None:
    """Return the retry error type from a parsed tool output envelope."""
    metadata_value = parsed.get("metadata")
    metadata: dict[str, Any] = metadata_value if isinstance(metadata_value, dict) else {}
    error_type = metadata.get("error_type")
    if metadata.get("retry") is True and isinstance(error_type, str):
        return error_type
    return None
