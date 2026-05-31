# Plan: Parallel Tool Execution

## Overview
Execute multiple tool calls from the same model response concurrently by default when the called tools are safe to run in parallel. Use a Pydantic AI-style conservative rule for now: if any tool in the batch requires sequential execution, run the whole batch sequentially in model order.

## References
- `thinharness/core.py` currently executes tool calls serially inside `Harness.run()` by iterating over `turn.tool_calls` and calling `_traced_call_output(...)` one at a time before sending all `ToolOutput`s back to the model.
- `vendor/pydantic-ai/pydantic_ai_slim/pydantic_ai/tool_manager.py` uses a per-tool `sequential` flag and forces a batch to sequential mode if any requested tool has `sequential=True`.
- `vendor/pydantic-ai/pydantic_ai_slim/pydantic_ai/_agent_graph.py` starts all tool-call tasks for parallel batches, but preserves deterministic result handling.
- `~/code/pi-agent-sdk/node_modules/@mariozechner/pi-agent-core/dist/agent-loop.js` exposes `toolExecution: "parallel" | "sequential"`, defaults to parallel, prepares calls first, then starts runnable calls concurrently while returning final results in assistant source order.

## Steps

### 1. Add Tool Execution Policy to Configuration
Add a small execution mode to `HarnessConfig` in `thinharness/core.py`.

```python
from typing import Literal

@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class HarnessConfig:
    """Configuration for Harness."""

    tool_execution: Literal["auto", "sequential"] = "auto"
```

`auto` means same-response tool calls may run in parallel depending on tool metadata. `sequential` is an escape hatch that forces the current behavior for debugging, deterministic reproduction, or custom tools that are not thread-safe.

**Verify:** Add a construction test proving the default is `"auto"` and explicit `"sequential"` is accepted.

### 2. Add Sequential Metadata to ToolSpec
Extend `ToolSpec` in `thinharness/tools.py` with a per-tool marker.

```python
@dataclass(frozen=True, config=ConfigDict(arbitrary_types_allowed=True))
class ToolSpec:
    """A JSON-schema-described callable exposed to the model."""

    name: str
    description: str
    parameters: Json | type[BaseModel]
    handler: ToolHandler
    sequential: bool = False
```

Keep the provider-facing schema unchanged; `sequential` is runtime-only harness metadata and should not be sent to model providers.

**Verify:** Existing schema tests should still show the same `response_tool()` output, without a `sequential` field.

### 3. Mark Unsafe Built-in Tools Sequential
Update built-in tool creation in `FileTools.specs()` so read-only tools stay parallel-safe and mutating or process-running tools force sequential execution.

Initial classification:
- Parallel-safe: `read`, `search`, `list_files`, `skill_read` if present.
- Sequential: `write`, `edit`, `bash` if present, `skill_run` if present.

Use the conservative rule for anything that can mutate files, depend on external process state, or execute arbitrary code. Custom tools default to `sequential=False`, so callers can opt into safety by setting `sequential=True` on their own `ToolSpec`.

**Verify:** Add tests that inspect the built-in `ToolSpec` objects and confirm read/search are not sequential while write/edit are sequential.

### 4. Centralize Batch Execution in Harness
Replace the inline serial loop in `Harness.run()` with a helper that executes one model batch and returns both recorded call entries and provider `ToolOutput`s.

```python
def _execute_tool_batch(self, run_tracer: RunTracer, calls: list[ModelToolCall]) -> tuple[list[Json], list[ToolOutput]]:
    """Execute model tool calls according to the configured execution policy."""
    should_run_sequentially = (
        self.config.tool_execution == "sequential"
        or any(self._tool_map.get(str(call.name)) and self._tool_map[str(call.name)].sequential for call in calls)
        or len(calls) <= 1
    )
    if should_run_sequentially:
        return self._execute_tool_batch_sequential(run_tracer, calls)
    return self._execute_tool_batch_parallel(run_tracer, calls)
```

Keep final result ordering in the original `turn.tool_calls` order even when execution finishes out of order. The next model request should still receive one complete list of outputs for the assistant message.

**Verify:** Existing harness tool-loop tests still pass with no payload ordering changes.

### 5. Run Synchronous Tools Concurrently with Threads
Because current handlers are synchronous, implement parallel execution with `concurrent.futures.ThreadPoolExecutor`, not `asyncio.create_task`.

```python
with ThreadPoolExecutor(max_workers=len(calls)) as executor:
    futures = [
        executor.submit(self._traced_call_output, run_tracer, call.id, call.name, call.arguments)
        for call in calls
    ]
    outputs = [future.result() for future in futures]
```

Use a bounded worker count equal to the batch size for the initial implementation. Let `_traced_call_output()` keep converting tool exceptions into error strings so one failed tool does not cancel sibling calls.

**Verify:** Add a test with two fake slow tools where elapsed time is less than serial duration, and assert the returned `ToolOutput`s preserve model call order.

### 6. Preserve Sequential Fallback Behavior
For `tool_execution="sequential"` or any batch containing a sequential tool, execute the entire batch in the existing model order.

This intentionally matches Pydantic AI's simple policy instead of trying to partition read-only calls around write barriers. A mixed batch like `read`, `write`, `read` will run fully sequentially for now.

**Verify:** Add a test where one slow parallel-safe tool and one `sequential=True` tool are emitted in the same response, then assert elapsed time is approximately serial and call order is preserved.

### 7. Add Regression Coverage for Provider Continuation
Add tests around a fake model response with multiple tool calls to confirm the harness sends all outputs in the next provider continuation call exactly once.

Important assertions:
- The model receives one continuation after the batch, not one per tool call.
- The continuation output order matches the assistant tool-call order.
- Errors from one tool are represented as normal tool output strings and do not prevent other same-batch tools from completing.

**Verify:** Run `uv run --extra dev pytest -q tests/test_harness.py`.

### 8. Document the Runtime Policy
Update `README.md` or a focused docs section with the runtime distinction between model parallel tool emission and harness parallel tool execution.

Document the default behavior:
- Same-response calls run in parallel in `auto` mode when no called tool is marked sequential.
- Any sequential tool in the batch makes the whole batch sequential.
- The harness blocks at the batch boundary because provider protocols require all tool results before the next model turn.
- `HarnessConfig(tool_execution="sequential")` restores fully serial execution.

**Verify:** Documentation examples import and construct real public types.

## Considerations
- This is batch-level concurrency only. The harness should not ask the model for the next response until all tool results for the current assistant message are available.
- Threaded execution is the smallest change because all current tool handlers are synchronous. If async tool handlers are introduced later, add an async execution path instead of wrapping everything in threads.
- The whole-batch sequential fallback is conservative. A later optimization can partition read-only calls around sequential barriers, but that adds ordering complexity and is not needed for the first implementation.
- OpenAI/OpenRouter-style provider settings such as `parallel_tool_calls` control whether a model may emit multiple tool calls. This plan is about executing the emitted calls concurrently inside the harness.
