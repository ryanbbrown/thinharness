# Background Completion Semantics

## Goal

Implement background tool completion semantics that let a `Harness.run(...)`
continue useful foreground work while ensuring completed background results are
delivered to the model at the next legal provider request.

Background execution is still scoped to the active run. It is not a durable job
queue, does not survive run termination, and does not expose public job control.

## Assumptions

- Provider transcripts remain append-only. The harness must never mutate an
  in-flight provider request after it has been sent.
- Provider APIs cannot be interrupted or cancelled just because background work
  completed.
- A model-visible background start notice is still represented as the tool
  output for the original model tool call.
- A background completion is model-visible information, but it is not another
  provider-requested tool call and must not consume `max_tool_calls`.
- If background completion delivery requires an extra provider request, it still
  consumes `max_model_requests`.
- Existing approval behavior remains conservative: if a later turn requires
  human approval, pending background work is cancelled and reported through the
  approval resume notice.
- A completion that has finished but has not yet been delivered is still active
  run state. Finalization, approval pause, run cleanup, and limit handling must
  treat it as undelivered background work.

## Implemented State

- `ToolSpec.background` exposes model-selected or always-background tool modes.
- `ToolCallExecutor` returns an immediate start notice for background calls.
- `BackgroundToolManager` owns run-scoped asyncio tasks and emits start/completed
  stream events.
- `BackgroundToolManager` tracks ready completions separately from pending tasks.
- `RunContext` records and formats ready completions in batches.
- Final answers are deferred while any background completion is pending or ready
  but undelivered.
- Ready background completions are delivered at the next legal provider request,
  coalesced with foreground tool outputs when possible.

## Target Semantics

- Starting a background tool returns an immediate model-visible start notice with
  the background task id.
- The model may continue with foreground tool calls that remain useful regardless
  of the eventual background result.
- If the model has no useful independent work left, it should stop and let the
  harness wait for background completion rather than inventing filler work.
- The harness does not interrupt already-running foreground tools when
  background work completes.
- The harness does not mutate an in-flight provider request when background work
  completes.
- At the next provider request that is legal under transcript rules, the harness
  delivers completed background results to the model.
- Multiple completed background tasks are delivered together when possible, in
  completion order.
- A ready-but-undelivered completion must never be lost just because no task is
  still running.

## Delivery Cases

### Completion While Foreground Tools Execute

When background work completes during a foreground tool batch:

1. Finish the current foreground batch.
2. Drain all background completions that are already completed.
3. Record the foreground tool outputs and background completion records.
4. Send the foreground tool outputs and background completion content in one
   continuation request.

### Completion While Provider Generates

When background work completes while a provider request is in flight:

1. Let the provider request finish.
2. If the model turn contains tool calls, execute those tool calls normally.
3. Drain completed background results after that tool batch finishes.
4. Send those tool outputs and background completion content in the same
   continuation request.

The model's requested foreground tools are not rejected or cancelled merely
because background completion became available after the request was sent.

### Final Answer While Background Pending

When the model returns a final answer while background work is pending:

1. Do not accept the final answer.
2. Wait for the next background completion.
3. Drain any additional completions that are ready immediately afterward.
4. Send the completion content to the model and require it to produce the final
   answer again.

For text finalization, this can remain a synthetic user message. For
`final_result` tool finalization, pair the deferred `final_result` call with a
tool output explaining that the final answer was deferred.

### Multiple Background Tasks

If multiple background tasks complete before the next delivery point:

1. Deliver all currently completed results together.
2. Preserve completion order.
3. Do not wait for still-pending background tasks unless finalization requires
   waiting for at least one completion.

## Design

### 1. Track Ready Completions Separately From Pending Tasks

Extend `BackgroundToolManager` so completions can be collected without waiting:

- Internal `_ready: list[BackgroundToolCompletion]`.
- `drain_ready() -> list[BackgroundToolCompletion]`.
- `wait_next_ready() -> BackgroundToolCompletion`, which waits for the next
  task only when `_ready` is empty.
- `has_pending_or_ready()` returns true when either `_pending` or `_ready` is
  non-empty. Use this for final-answer deferral, max-tool-call background
  rejection, approval pause handling, and run-end cleanup.
- Ensure completions enter `_ready` exactly once and in completion order.

Implementation detail:

- Prefer an `asyncio.Queue[BackgroundToolCompletion]` or task
  `add_done_callback(...)` path so the task records its completion at the moment
  it finishes. Do not depend on iterating an `asyncio.wait(...)` `done` set or a
  pending-task dict to infer completion order; both are unsuitable when several
  tasks are already done.
- `drain_ready()` first inspects `_pending` for `task.done()` and collects those
  results before deciding that no completions are ready.
- Avoid `task.result()` before a task is done. The no-wait path must inspect only
  completed tasks.
- `cancel_and_drain()` must first return any ready completions as completed
  records, then cancel still-pending work and return cancellation records for the
  remainder.

### 2. Add RunContext Background Drain Helpers

Replace single-completion-only call sites with batch-oriented helpers:

- `drain_ready_background() -> tuple[list[BackgroundToolCompletion], str | None]`
- `drain_next_background_batch() -> tuple[list[BackgroundToolCompletion], str]`

Expected behavior:

- `drain_ready_background()` records every ready completion and returns a joined
  model-facing message, or `None` if nothing is ready.
- `drain_next_background_batch()` waits for one completion, then drains any other
  ready completions and returns one joined message.
- The joined message should preserve the existing per-completion format from
  `background_completion_message(...)`, separated with `\n\n---\n\n`.
- These helpers belong on `RunContext`; they delegate to
  `BackgroundToolManager.drain_ready()` and
  `BackgroundToolManager.wait_next_ready()`.

### 3. Coalesce Completion With Tool-Output Continuations

Change normal tool continuation flow in `Harness._execute_tool_turn(...)`:

1. Execute the requested tool batch.
2. Record foreground tool records.
3. Drain ready background completions.
4. Check foreground tool retry limits after ready completions have been recorded,
   so a retry-budget failure cannot drop an already-completed background result.
5. Continue the provider request with the foreground `ToolOutput`s plus the
   completion content.

Use the existing `TurnDriver.send_tool_outputs(..., extra_notices=...)` plumbing:

- If `drain_ready_background()` returns a message, pass
  `ModelNotice(kind="background_completion", content=message)` as an extra
  notice on the same `continue_with_tools(...)` request.
- If no completions are ready, pass no extra notice.
- Keep the request/trace kind as `tool_outputs` for coalesced delivery. Reserve
  the existing `background_completion` request kind for completion-only
  continuations, such as final-answer deferral.
- Do not extend `ModelSession.continue_with_tools(...)` with a mixed payload
  unless provider notice rendering proves insufficient.

### 4. Preserve Existing Special Cases

Update these paths to use the same batch drain helpers:

- `_defer_final_for_background(...)`
- `_reject_batch_for_background(...)`

`_reject_batch_for_background(...)` is still needed when `max_tool_calls` is
exhausted and background completion is pending. It should wait for a background
completion and send rejection outputs plus the completion notice in one request.

Approval and cleanup need separate lifecycle handling:

- If the model returns an approval-required turn while completions are ready,
  preserve those ready completions in the approval envelope or deliver them with
  the approval-resume continuation before any resumed model work proceeds.
  Prefer storing completed-but-undelivered records in the approval envelope so
  the user approval boundary remains explicit.
- If the run exits while completions are ready, record them as completed
  background records before cancelling still-pending tasks.
- Pending tasks that are cancelled for an approval pause should continue to be
  reported through `background_cancelled` notices on resume.

### 5. Update Notice Types And Serialization

Add `background_completion` as a `ModelNotice.kind` literal in:

- `thinharness.providers.ModelNotice`
- fake/session helpers used by tests, especially the local `SequenceSession` in
  `tests/test_background_tools.py`

Provider note rendering is already centralized and kind-agnostic through harness
notices, so concrete providers should need an audit rather than bespoke
serialization changes. Verify OpenAI Responses, Anthropic Messages, and
OpenRouter all pass the notice through unchanged.

Rules:

- Limit warnings continue to emit `LimitWarningEvent`.
- Background completion notices do not need a new public stream event because
  `BackgroundTaskCompletedEvent` already emits completion details.
- For coalesced delivery, tracing should show the foreground tool outputs in the
  normal tool-output input and the completion content in the model notices
  attribute. Do not assert a separate `background_completion` trace kind for
  coalesced delivery.

### 6. Strengthen Model Instructions

Update default background guidance to make the contract explicit:

> Use background execution only for long-running work that can proceed
> independently. After starting background work, continue only with tool calls
> that remain useful regardless of that background result. If the next action
> depends on the background result, stop and let the harness notify you when it
> completes.

Keep this instruction hidden when `tool_execution="sequential"` or when no
model-selectable background tools are exposed. Treat background-capable
subagents as model-selectable background tools when their schema exposes
`_background`.

The instruction is longer than the current sentence, but the added contract is
part of the behavior change and reduces the chance that the model starts
background work and then performs dependent filler tool calls.

## Test Plan

Add or update tests in `tests/test_background_tools.py`:

- `test_drain_ready_harvests_completed_tasks_in_completion_order`
  - Start at least three background tasks that finish in a known non-start order.
  - Drain after all are complete.
  - Assert the returned completions preserve finish order, not start order.

- `test_wait_next_ready_returns_ready_completion_without_waiting`
  - Put one completion in the ready queue.
  - Assert `wait_next_ready()` returns it immediately without waiting for another
    pending task.

- `test_ready_background_completion_is_coalesced_with_foreground_tool_outputs`
  - Start one background tool and one foreground tool.
  - Make the background task complete before the foreground task returns.
  - Assert the next model continuation is `continue_with_tools(...)`, not
    `continue_with_user_message(...)`.
  - Assert the continuation includes foreground tool output plus a
    `background_completion` notice.

- `test_background_completion_during_provider_turn_waits_until_next_tool_outputs`
  - After a background start notice, have the next provider turn take long enough
    for the background task to finish.
  - The model then asks for a foreground tool.
  - Assert completion is delivered with that foreground tool output.
  - Implement with a custom async `SequenceSession` method that awaits an
    `asyncio.Event` or sleeps inside `continue_with_tools(...)` before returning
    the tool-call turn.

- `test_final_text_drains_multiple_ready_background_completions`
  - Start two background tasks.
  - Let both complete before the model finalizes.
  - Assert the final deferral message contains both completions in completion
    order, separated by `\n\n---\n\n`, and only one continuation is needed before
    the new final.

- `test_background_completion_notice_does_not_consume_tool_calls`
  - Set a tight `max_tool_calls`.
  - Verify coalesced completion delivery does not increment tool-call usage.

- `test_background_completion_notice_tracing_is_coalesced`
  - Confirm the model request trace for the coalesced tool-output continuation
    has request kind `tool_outputs`, foreground tool outputs in the input, and
    background completion content in the notices attribute.

- `test_final_answer_defers_when_completion_is_ready_but_not_pending`
  - Move a background completion into the ready state before the model returns a
    final answer.
  - Assert the final answer is deferred and the ready completion is delivered.

- `test_ready_completion_is_preserved_across_approval_pause`
  - Let background work complete while the provider returns an approval-required
    turn.
  - Assert the ready completion is stored in resume state or delivered on resume,
    according to the implementation choice.

- `test_ready_completion_recorded_on_run_cleanup`
  - End the run through an error or limit path while a completion is ready.
  - Assert `tool_call_records` includes the completed background record.

Update existing tests that currently expect `session.user_messages` for ordinary
background completion delivery. Keep user-message expectations only for
completion-only final deferral.

Specific existing tests to revisit:

- `test_background_model_choice_returns_start_notice_before_tool_finishes`
- `test_multiple_background_completions_are_delivered_in_completion_order`
- `test_background_tracing_has_start_and_execution_spans`
- `test_stream_background_events`
- approval tests that assert background cancellation or approval-resume notices

Run:

```bash
uv run pytest tests/test_background_tools.py tests/test_streaming.py tests/test_approvals.py tests/test_tracing.py
uv run pyright
uv run ruff check thinharness tests
```

## Acceptance Criteria

- A background completion that is ready before a legal tool-output continuation
  is delivered on that continuation, not as a later standalone user message.
- Foreground tool batches that were already requested by the model still run to
  completion even if a background task finishes first.
- Final answers are deferred until pending background completion has been
  delivered back to the model.
- Multiple ready completions are delivered together in completion order.
- Background completions remain run-scoped and are cancelled/drained on run
  termination.
- Existing approval, retry, structured-output, streaming, and tracing behavior
  remains compatible.

## Non-Goals

- Do not add durable background jobs.
- Do not add polling APIs or public background task control.
- Do not require provider-specific streaming interruption support.
- Do not reject model-requested foreground tool calls just because a background
  task completed after the model requested them.
- Do not change the public `ToolSpec.background` API unless the implementation
  uncovers an unavoidable provider contract issue.
