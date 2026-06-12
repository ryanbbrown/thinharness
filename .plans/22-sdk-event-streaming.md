# Plan: SDK Event Streaming

## Goal

Add a lightweight public event stream for `Harness.run()` visibility without provider token deltas.

SDK callers should be able to write:

```python
async for event in harness.stream("Process these records."):
    ...
```

and observe coarse run progress: run start/end, model turns, tool calls, background completions, structured-output retries, limit warnings, and child/subagent activity. The final successful event carries the same `HarnessResult` shape that `run()` returns today. Streaming is for workflow visibility and orchestration UX, not chatbot token streaming.

## Current State

The run loop has useful internal structure but no public live surface:

- `Harness.run()` owns the full loop and returns only `HarnessResult` at the end.
- `HarnessResult.responses` and `HarnessResult.tool_call_records` give post-run visibility, not in-flight progress.
- Hooks can observe lifecycle moments, but they are synchronous mutation/interception points, not an async SDK stream.
- Tracing captures rich model/tool spans, but tracing is an observability sink rather than app-facing control flow.
- `ModelTurn` is already provider-neutral and turn-level: full assistant text, full tool calls, raw response. It does not expose provider text deltas.
- Background tools already have start/completion concepts internally, but completions are only visible through final `tool_call_records` and traces.

This feature should reuse that turn-level shape instead of introducing provider streaming.

## Vendor Scan

Local vendor references point to a coarse typed-event stream as the right default:

- `vendor/codex/sdk/typescript/src/thread.ts` exposes `runStreamed(...)` returning an `AsyncGenerator<ThreadEvent>`, and implements non-streaming `run(...)` by consuming that generator. Events include `thread.started`, `turn.started`, item lifecycle events, and `turn.completed`.
- `vendor/codex/sdk/typescript/src/items.ts` models completed/updated items such as agent messages, command execution, MCP tool calls, web search, reasoning, and errors. Tool/MCP arguments and results are present in those item payloads, matching the expectation that SDK stream events are app-facing workflow data.
- `vendor/adk-python/src/google/adk/events/event.py` uses typed `Event` objects with `invocation_id`, `branch`, author, content, actions, and `long_running_tool_ids`. ADK uses branch strings such as `agent_1.agent_2` to represent nested subagent context.
- `vendor/adk-python/src/google/adk/flows/llm_flows/base_llm_flow.py` and `functions.py` compose agent execution as async generators of `Event`; subagent transfers yield child-agent events through the same stream.
- `vendor/smolagents/src/smolagents/agents.py` supports `run(..., stream=True)` by returning a generator over steps/tool calls/tool outputs/final answer, and can also include text deltas when `stream_outputs=True`. ThinHarness should copy the step/tool visibility, not the text-delta path.
- `vendor/agent-framework/python/packages/core/agent_framework/_types.py` has `ResponseStream`, an async iterable that also finalizes into an `AgentResponse` via `get_final_response()`. That supports the idea that streaming and a final result are not competing modes.

## Decisions

- Public API is `Harness.stream(...)`, an async iterator of typed event dataclasses.
- No provider text/token deltas in this plan. Model messages are emitted at turn boundaries after the provider returns a complete `ModelTurn`.
- `Harness.run(...)` should be implemented by consuming `Harness.stream(...)`. This makes streaming the canonical loop and prevents two subtly different run implementations.
- The stream terminal success event includes the normal `HarnessResult`; `run(...)` returns that exact result object.
- Event classes are frozen dataclasses with `kind: Literal[...]` discriminators, exported from `thinharness`.
- Events include stable run metadata: `run_id`, optional `parent_run_id`, optional `parent_tool_call_id`, optional `agent_name`, and monotonic `sequence`.
- `sequence` is delivery-order within the stream being consumed. In a flattened parent stream, forwarded child events are re-sequenced by the parent stream emitter while preserving the child's `run_id`; direct child streams still start at sequence 1.
- `run_id` is a new per-`Harness.run`/`Harness.stream` id, distinct from tracing span ids and caller-provided `conversation_id`.
- Caller `metadata["run_id"]` is not special. It remains ordinary caller metadata and must not override the framework-generated stream `run_id`.
- Subagent events are flattened into the parent stream with `parent_run_id` and `parent_tool_call_id`, not hidden inside the parent `subagent` tool event. This matches ADK's "one event stream with nested branch metadata" more than opaque tool-only delegation.
- Stream events include high-level prompt text, tool arguments, and model-visible tool outputs by default. They do not include raw provider response JSON.
- The default stream should include model assistant text and tool-call names/ids because those are the main progress signal.
- Background tools emit explicit events: original tool call starts/completes with a `background_task_id` and `status="running"`, then separate background completion events when the actual background task finishes. This is clearer than hiding completion as a synthetic model continuation.
- Hooks remain separate. Do not reframe streaming as hooks, and do not make hook dispatch async in this feature.
- `stream()` has no synchronous iterator counterpart in this plan. Sync callers can keep using `run_sync()` for final results; sync streaming can be considered later if a concrete host needs it.
- All event dataclasses must stay `frozen=True`. Pyright relies on frozen dataclasses for safe `kind: Literal[...]` narrowing in subclasses.

## Public API Shape

Add typed event dataclasses, likely in a new `thinharness/events.py` leaf-ish module:

```python
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
    "run_completed",
    "run_failed",
]

@dataclass(frozen=True)
class StreamToolCall:
    id: str
    name: str

@dataclass(frozen=True, kw_only=True)
class StreamEvent:
    kind: StreamEventKind
    run_id: str
    sequence: int
    parent_run_id: str | None = None
    parent_tool_call_id: str | None = None
    agent_name: str | None = None
    metadata: Json = field(default_factory=dict)

@dataclass(frozen=True, kw_only=True)
class RunStartedEvent(StreamEvent):
    kind: Literal["run_started"] = "run_started"
    prompt: str | None = None
    root: str = ""
    max_model_requests: int = 0
    max_tool_calls: int | None = None

@dataclass(frozen=True, kw_only=True)
class ModelRequestStartedEvent(StreamEvent):
    kind: Literal["model_request_started"] = "model_request_started"
    request_kind: Literal["start", "resume", "tool_outputs", "correction", "output_retry_tool", "background_completion"] = "start"
    model: str = ""
    provider: str | None = None

@dataclass(frozen=True, kw_only=True)
class ModelMessageEvent(StreamEvent):
    kind: Literal["model_message"] = "model_message"
    text: str = ""
    tool_calls: tuple[StreamToolCall, ...] = ()
    finalized_output_mode: str | None = None
@dataclass(frozen=True, kw_only=True)
class ToolCallStartedEvent(StreamEvent):
    kind: Literal["tool_call_started"] = "tool_call_started"
    call_id: str = ""
    tool_name: str = ""
    tool_index: int = 0
    arguments: str | None = None

@dataclass(frozen=True, kw_only=True)
class ToolCallCompletedEvent(StreamEvent):
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
    kind: Literal["background_task_started"] = "background_task_started"
    background_task_id: str = ""
    tool_call_id: str = ""
    tool_name: str = ""

@dataclass(frozen=True, kw_only=True)
class BackgroundTaskCompletedEvent(StreamEvent):
    kind: Literal["background_task_completed"] = "background_task_completed"
    background_task_id: str = ""
    tool_call_id: str = ""
    tool_name: str = ""
    status: Literal["completed", "failed", "cancelled"] = "completed"
    elapsed_ms: float = 0.0
    output: str | None = None

@dataclass(frozen=True, kw_only=True)
class ModelRetryEvent(StreamEvent):
    kind: Literal["model_retry"] = "model_retry"
    retry_kind: Literal["structured_output", "tool_retry"] = "structured_output"
    message: str = ""
    call_id: str | None = None

@dataclass(frozen=True, kw_only=True)
class LimitWarningEvent(StreamEvent):
    kind: Literal["limit_warning"] = "limit_warning"
    limit_kind: Literal["model_requests", "tool_calls"] = "model_requests"
    remaining: int = 0
    content: str = ""

@dataclass(frozen=True, kw_only=True)
class RunCompletedEvent(StreamEvent):
    kind: Literal["run_completed"] = "run_completed"
    result: HarnessResult

@dataclass(frozen=True, kw_only=True)
class RunFailedEvent(StreamEvent):
    kind: Literal["run_failed"] = "run_failed"
    stop_reason: StopReason = "error"
    error_type: str = ""
    message: str = ""
```

Use a union alias:

```python
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
    | RunCompletedEvent
    | RunFailedEvent
)
```

Add stream options:

```python
@dataclass(frozen=True)
class StreamOptions:
    include_model_text: bool = True
    include_subagents: bool = True
```

`Harness.stream(...)` signature:

```python
def stream(
    self,
    prompt: str,
    *,
    resume_from: dict[str, Any] | None = None,
    metadata: Json | None = None,
    stream_options: StreamOptions | None = None,
) -> HarnessStream:
    ...
```

Do not add `stream=True` to `run(...)`; that creates a mode-switching return type. Keep `run(...) -> HarnessResult` and `stream(...) -> HarnessStream`, where `HarnessStream` is async-iterable over `HarnessStreamEvent`.

`HarnessStream` is a public async-iterable/async-context-manager wrapper around the internal queue task. Export it so callers can annotate early-close cases:

```python
class HarnessStream(AsyncIterator[HarnessStreamEvent]):
    async def aclose(self) -> None: ...
    async def __aenter__(self) -> HarnessStream: ...
    async def __aexit__(self, *exc: object) -> None: ...
```

Update `thinharness/__init__.py` and `__all__` to export:

- `HarnessStream`
- `HarnessStreamEvent`
- `StreamEvent`
- `StreamEventKind`
- `StreamOptions`
- `StreamToolCall`
- all concrete event classes

## Payload Policy

Default payloads:

- `RunStartedEvent`: submitted prompt.
- `ModelRequestStartedEvent`: request kind, provider, model, no provider payload.
- `ModelMessageEvent`: assistant `text` unless `include_model_text=False`, normalized tool call id/name pairs, no raw provider response JSON.
- `ToolCallStartedEvent`: tool name/id/index and model-requested arguments.
- `ToolCallCompletedEvent`: status fields and model-visible output.
- `BackgroundTaskCompletedEvent`: status, elapsed time, and model-visible output.
- `RunCompletedEvent`: full `HarnessResult` always, because this is the explicit terminal result path. The result already contains `responses` and `tool_call_records`; callers who reach this event have opted into SDK-level completion handling.

This keeps streaming as a high-level SDK consumption surface. Raw provider response JSON remains available after completion through `HarnessResult.responses` rather than through live stream events.

## Run Ids And Nesting

Current ThinHarness has no public run id. It has:

- caller-provided `metadata["conversation_id"]` for tracing grouping;
- `parent_call_id` passed to child subagents through `_child_metadata()`;
- trace span parentage inside tracing sinks;
- `_is_child_run` to distinguish trace output behavior.

Add a per-run id managed by the stream runtime:

- Generate `run_id` at the beginning of every `stream(...)` call.
- Ignore caller-provided `metadata["run_id"]` for stream identity. It remains visible only as ordinary caller metadata where metadata is otherwise copied.
- For child harness streams, pass `parent_run_id` and `parent_tool_call_id` through a stream context object, not ad hoc user metadata.
- Keep `conversation_id` unchanged as a caller grouping key.

Implementation should use a small `RunStreamContext` with:

```python
@dataclass
class RunStreamContext:
    run_id: str
    parent_run_id: str | None
    parent_tool_call_id: str | None
    agent_name: str | None
    options: StreamOptions
    sequence: int = 0
```

`RunContext` should own this context for every run once `run()` is rebuilt on `stream()`. It should expose a small `emit(...)` helper that delegates to the active emitter. Keep `RunStreamContext` separate from `RunContext` so stream identity/options stay isolated from ordinary run bookkeeping and can be passed through contextvars for subagents.

Subagents:

- Change `run_subagent_tool(...)` to call `child.stream(...)` when a parent stream is active and `include_subagents=True`.
- Forward child events through the parent stream queue, preserving child `run_id` and setting `parent_run_id` to the parent run id.
- Still return the subagent tool `ToolResult` exactly as today after the child terminal result.
- If `include_subagents=False`, keep current behavior: parent sees only the `subagent` tool call lifecycle.

This requires an internal event sink/queue, because a tool handler cannot directly `yield` into the parent `Harness.stream(...)` async generator. Keep it narrowly scoped to stream emission; do not turn hooks or tracing into queue producers.

Use a contextvar similar to `_CURRENT_TOOL_CALL` / `_CURRENT_TOOL_RUNTIME`:

```python
_CURRENT_STREAM_EMITTER: contextvars.ContextVar[StreamEmitter | None]
```

`ToolCallExecutor` sets this context while invoking tool handlers. `run_subagent_tool(...)` reads it and forwards child stream events to the parent emitter when present.

## Internal Architecture

### 1. Add Event Sink Plumbing

Add an internal sink to `RunContext`:

```python
class StreamEmitter:
    def __init__(self, ctx: RunStreamContext) -> None: ...
    def emit(self, event: HarnessStreamEvent) -> None: ...
    async def close(self) -> None: ...
```

Implementation options:

- Simpler: use a list buffer on `RunContext` and have the stream-driving loop yield after every awaited operation.
- More complete: use an `asyncio.Queue[HarnessStreamEvent]` so child/background tasks can emit immediately.

Choose the queue. Background completions and subagent streams can happen while parent tool batches are in progress; a queue keeps event timing honest and avoids awkward "flush points" everywhere.

Constraints:

- Do not leave producer tasks running when the stream consumer stops early.
- Cancellation of the async generator must follow today's `Harness.run()` cancellation semantics: background tasks are cancelled/drained, `run_end` fires once, `_running` resets.
- Queue emission must never block the run loop indefinitely. Use an unbounded queue for v1. This means there is no consumer backpressure after `_run_streaming(...)` starts; that is acceptable because events are small and runs are bounded, but document it honestly.
- Use an explicit sentinel in the queue after the terminal event so the consumer can distinguish "no event yet" from "stream finished".
- `StreamEmitter.emit(...)` assigns the delivery-order `sequence` at enqueue time. Forwarded child events are copied/replaced with the parent stream's next sequence.
- Public event base `metadata` is reserved for stable event-specific metadata and should remain empty in v1 unless an event has a documented use. Do not dump caller metadata into every event.

Threading:

- `RunContext` owns `stream: RunStreamContext` and `emitter: StreamEmitter`.
- `ToolBatchExecutor` and `ToolCallExecutor` can access the emitter through `run_context.emit(...)`; do not pass a second emitter argument through their constructors.
- `BackgroundToolManager` should receive the emitter in its constructor, alongside `run_tracer`, because its `_run(...)` coroutine executes after the original tool call has returned.
- `BackgroundToolManager.cancel_and_drain()` should emit `BackgroundTaskCompletedEvent(status="cancelled")` for drained cancellation completions when a consumer is still active. If the stream consumer itself has cancelled and the emitter is closing, cleanup may be silent; cleanup correctness matters more than best-effort events no one can read.

### 2. Make Streaming The Canonical Runner

Refactor the current `Harness.run(...)` body into an internal coroutine that can emit events:

```python
async def _run_streaming(..., emitter: StreamEmitter) -> HarnessResult:
    ...
```

`Harness.stream(...)` should return a small stream object, not a bare async generator. The object must implement `__aiter__`, `__anext__`, `aclose()`, and async context-manager methods. Direct full consumption still works as requested:

```python
async for event in harness.stream("..."):
    ...
```

For early exits, document and test deterministic cleanup through `aclose()` or `async with`:

```python
async with harness.stream("...") as events:
    async for event in events:
        break
```

The stream object performs eager `_closed`/`_running` checks before returning, matching `run()` more closely than a bare async generator whose body starts only on first iteration.

`Harness.stream(...)`:

1. Creates `RunStreamContext` and `StreamEmitter`.
2. Starts `_run_streaming(...)` as one task.
3. Yields events from the queue as they arrive.
4. On successful completion, `_run_streaming(...)` emits `RunCompletedEvent(result=result)` and returns.
5. On failure, `_run_streaming(...)` emits `RunFailedEvent(...)` before re-raising or storing the exception.
6. `_run_streaming(...)` always enqueues a sentinel after terminal success/failure cleanup.
7. After yielding `RunFailedEvent`, the stream object raises the same exception to preserve normal async iteration error behavior.

`Harness.run(...)`:

```python
async def run(...) -> HarnessResult:
    result: HarnessResult | None = None
    async for event in self.stream(...):
        if isinstance(event, RunCompletedEvent):
            result = event.result
    if result is None:
        raise HarnessError("stream ended without a result")
    return result
```

If `stream(...)` raises, `run(...)` naturally raises the same exception. If stream completion ends without `RunCompletedEvent`, raise `HarnessError("stream ended without a result")`.

Potential issue: `RunCompletedEvent.result` includes full post-run `responses` and `tool_call_records`. That means event consumers see the complete result once, at the end. This is intended.

`run_sync(...)` should require no special path: it keeps calling `asyncio.run(self.run(...))`. Add a regression test because `run()` now consumes a stream object and `_run_streaming(...)` creates a task on the active event loop.

### 3. Emit Model Events

In `RunContext.advance_model(...)`:

- Before opening or inside the model span, emit `ModelRequestStartedEvent` with `trace_snapshot.kind`, provider, model.
- Emit `LimitWarningEvent` for each computed `ModelNotice(kind="limit_warning")`. These are warnings sent to the model, not hard `limit_reached` hook events.
- After `turn = await request(notices)` and output resolution, emit `ModelMessageEvent`.

Use normalized data:

- `text=turn.text` unless `StreamOptions.include_model_text` is false, in which case use `text=""`.
- `tool_calls=(StreamToolCall(id=call.id, name=call.name), ...)`
- keep model-requested arguments visible through `ToolCallStartedEvent`, not duplicated on model events.
- do not include `turn.raw` in stream events.

Do not emit partial text deltas. Provider sessions still call non-streaming endpoints and return complete turns.

### 4. Emit Tool Events

In `ToolCallExecutor.execute_one(...)`:

- Emit `ToolCallStartedEvent` after `BeforeToolCallContext` is built and before hook dispatch. This means every model-requested tool call has a visible start, even if a `before_tool_call` hook cancels it.
- If the hook cancels the call, emit `ToolCallCompletedEvent(cancelled=True, ok=False, ...)` with the cancellation output.
- If a strict `before_tool_call` hook raises, emit `ToolCallCompletedEvent(ok=False, error_type=type(exc).__name__, message=str(exc))` before propagating the exception. The run will then emit `RunFailedEvent`.
- If a strict `after_tool_call` hook raises, emit `ToolCallCompletedEvent(ok=False, error_type=type(exc).__name__, message=str(exc))` before propagating.
- Include model-requested `arguments`.
- Emit `ToolCallCompletedEvent` after `after_tool_call` hooks have had a chance to rewrite output, because that is the model-visible output.
- Include model-visible `output`.
- Preserve current retry accounting: event emission must not change `retry_kind` capture timing. The `retry_kind` field is the pre-`after_tool_call` budget signal, while `output` is the post-hook model-visible output. This mirrors `docs/decisions.md`: hooks own the message, the harness owns retry budget.

For parallel tool execution:

- Started/completed events may interleave by actual execution timing.
- Provider-facing outputs and `tool_call_records` still preserve model order, per existing decision.
- Tests should assert ordering only where deterministic: sequential tools, or started events in a single-call batch. For parallel batches, assert event sets and per-call start-before-complete.

### 5. Emit Background Events

For a background-starting tool call:

- The original tool call emits `ToolCallCompletedEvent(..., background_task_id="bg_1", background_status="running")` with the start-notice output.
- `BackgroundToolManager.start(...)` emits `BackgroundTaskStartedEvent`.
- `BackgroundToolManager._run(...)` emits `BackgroundTaskCompletedEvent` on `completed`, `failed`, or `cancelled`.

When a background completion is injected back into the model via `_defer_final_for_background(...)` or `_reject_batch_for_background(...)`, `RunContext.advance_model(...)` already emits `ModelRequestStartedEvent(kind="background_completion")`. Do not invent a separate "synthetic user message" event in v1; the background completion event plus model request event is enough.

### 6. Emit Retry And Terminal Events

Structured output retries:

- Before sending a retry tool output/user message, emit `ModelRetryEvent(retry_kind="structured_output", message=decision.retry_message, call_id=decision.retry_call_id)`.
- Do not emit retry events for every failed tool result unless the harness is actually consuming retry budget and allowing the model to retry.
- Emit `ModelRetryEvent(retry_kind="tool_retry", ...)` inside `RunContext.check_tool_retry_limits(...)` after incrementing a tool's retry count and before returning normally. If the tool retry budget is exceeded, the existing `limit_reached` path and `RunFailedEvent` cover the terminal failure instead.

Terminal success:

- `RunContext.finalize(...)` builds the `HarnessResult`, attaches resume state, fires run-end hooks, then emits `RunCompletedEvent(result=result)`.
- Run-end hooks should see the result before the stream terminal event is emitted, matching today's hook semantics.

Terminal failure:

- Existing exception classification sets `stop_reason` and `terminal_error`.
- Emit `RunFailedEvent(stop_reason=..., error_type=..., message=...)` after `fire_run_end_once()` so hooks keep their current "terminal bookkeeping happened first" role. Document and test this ordering.
- `Harness.stream(...)` raises the original/wrapped exception after yielding `RunFailedEvent`.
- For `ProviderError`, the event should reflect what callers actually observe: `stop_reason="provider_error"`, `error_type="HarnessError"`, and `message=str(wrapped_error)`, because current `run()` wraps provider failures in `HarnessError`.
- If `new_session()` fails before the first provider request, the stream may go directly from `run_started` to `run_failed` with no `model_request_started`; document this as acceptable because no model request was actually started.
- Emit exactly one terminal event: `RunCompletedEvent` on success, `RunFailedEvent` on failure, never both.

Cancellation:

- External cancellation of the stream consumer should cancel the run task.
- The run task should emit no best-effort `RunFailedEvent` if the consumer has cancelled and is no longer reading. It must still preserve current cleanup: cancel/drain background tasks, fire `run_end`, reset `_running`.

## Tests

Add focused tests rather than snapshotting every event:

1. `test_stream_returns_final_harness_result`
   - Scripted final model turn.
   - Consume all events.
   - Assert exactly one `RunCompletedEvent`, result text/output/usage matches `run(...)` behavior.

2. `test_run_consumes_stream_and_returns_same_result`
   - Use a subclass or monkeypatch around `stream(...)` only if practical.
   - Otherwise assert `run(...)` and manual stream consumption produce equivalent final `HarnessResult`.

3. `test_stream_emits_model_and_tool_lifecycle`
   - Model requests one tool then finalizes.
   - Assert event order: `run_started`, `model_request_started(start)`, `model_message`, `tool_call_started`, `tool_call_completed`, `model_request_started(tool_outputs)`, `model_message`, `run_completed`.
   - Assert `ToolCallStartedEvent.arguments` and `ToolCallCompletedEvent.output` are included by default.

4. `test_stream_payloads_are_high_level_without_raw_provider_payloads`
   - Assert prompt, arguments, and output appear.
   - Assert raw provider response JSON does not appear in stream events.

5. `test_stream_limit_warning_events`
   - Existing near-limit scripted cases.
   - Assert `LimitWarningEvent` mirrors `ModelNotice` content and remaining count.

6. `test_stream_structured_output_retry_event`
   - Tool-mode or prompted structured output retry.
   - Assert `ModelRetryEvent` appears before the retry model request.

7. `test_stream_background_events`
   - Background tool start and completion.
   - Assert tool start notice event, `BackgroundTaskStartedEvent`, `BackgroundTaskCompletedEvent`, and later `model_request_started(background_completion)`.

8. `test_stream_subagent_events_include_parent_ids`
   - Parent calls subagent.
   - Assert child events appear with distinct `run_id`, parent `run_id`, and parent tool call id.
   - Also test `StreamOptions(include_subagents=False)` hides child events while preserving parent result.
   - Include a background subagent case (`_background: true`) because child stream forwarding from inside a `BackgroundToolManager` task is the highest-risk context propagation path.

9. `test_stream_failure_yields_failed_event_then_raises`
   - ProviderError and UnexpectedModelBehavior cases.
   - Assert `RunFailedEvent` stop reason matches current `run_end` semantics and async iteration raises the same exception type expected from `run(...)`.

10. `test_stream_consumer_cancellation_cleans_up`
    - Start a long background tool or long provider request.
    - Cancel stream consumption.
    - Assert `_running` resets and a later run can start, matching existing cancellation tests.

11. `test_stream_close_after_early_break_cleans_up`
    - Use `async with harness.stream(...) as events`, break after the first event, and assert cleanup runs and a later run can start.

12. `test_run_external_cancellation_still_cleans_up`
    - Preserve the shape from `tests/test_harness.py::test_external_cancellation_records_run_end_and_allows_rerun` after `run()` is rebuilt on `stream()`.

13. `test_stream_strict_tool_hook_failures_complete_started_tool`
    - Strict `before_tool_call` exception: `tool_call_started`, `tool_call_completed(error_type=...)`, then `run_failed`.
    - Strict `after_tool_call` exception: same terminal shape after tool execution.
    - Parallel batch strict-hook failure still cancels siblings per current tests.

14. `test_stream_sequence_and_run_ids`
    - Two stream calls produce distinct `run_id`s.
    - Sequence is monotonic in delivery order.
    - Forwarded subagent events keep child `run_id` but receive parent stream sequence.

15. `test_stream_run_sync_still_works`
    - `run_sync()` still returns a `HarnessResult` and closes owned resources.

16. `test_stream_resume_from_emits_resume_kind`
    - Resumed runs emit `ModelRequestStartedEvent(request_kind="resume")`.

17. `test_stream_rejects_concurrent_calls`
    - Two simultaneous `stream()` calls on one harness preserve the current non-reentrant guard.

18. `test_stream_tool_retry_event`
    - Retryable tool output within budget emits `ModelRetryEvent(retry_kind="tool_retry")`.

19. `test_stream_parallel_tool_events`
    - Parallel tool events can interleave, but each call has start-before-complete and provider-facing output order is unchanged.

Validation gate:

```bash
uv run pytest
uv run ruff check .
uv run pyright
```

## Documentation

Update `README.md` and `docs/docs.md` with:

- a short `async for event in harness.stream(...)` example;
- the high-level stream payload policy and absence of raw provider response JSON in stream events;
- statement that streaming is coarse turn/tool/run streaming, not token delta streaming;
- guidance for workflow UIs: use `kind`, `run_id`, `parent_run_id`, and `parent_tool_call_id` to group nested subagent work.

Do not advertise streaming as a chatbot UI primitive.

## Do Not Touch

- Do not add provider streaming endpoints or token delta event types.
- Do not change `ModelSession` protocol signatures for provider-level streaming.
- Do not make hooks async.
- Do not change model-facing prompt/tool-output text.
- Do not change tracing capture policies.
- Do not change `HarnessResult` fields.
- Do not remove `responses` or `tool_call_records`; streaming supplements them.

## Risks And Open Questions

- The event queue introduces a second execution task around the run loop. Early consumer cancellation is the highest-risk behavior; implement and test it before subagent forwarding.
- Subagent streaming requires passing stream context across a tool invocation boundary. Use an internal contextvar or explicit runtime context; avoid putting framework-only stream ids into user metadata.
- `run()` on top of `stream()` makes stream terminal semantics load-bearing. The stream must emit `RunCompletedEvent` exactly once on success.
- Default `ModelMessageEvent.text`, tool arguments, and tool outputs may expose model or tool content that some callers consider sensitive. This is still the requested workflow-visibility signal; callers who need to hide model text can set `StreamOptions(include_model_text=False)`.
- `RunFailedEvent` plus raising may surprise some consumers. It is still preferable to swallowing failures in an event stream; document that terminal failure is both observable and raised.
