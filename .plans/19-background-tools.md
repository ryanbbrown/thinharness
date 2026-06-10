# Background Tools Plan

## Goal

Add per-run background tool execution so a model can start selected long-running tools, continue other work, and receive the eventual result later in the same harness run.

This is not durable background execution. Background jobs are owned by the active `Harness.run()` invocation. A run must not finish while background jobs are pending; if the model tries to finalize early, the harness waits for the next background completion, injects it back into the conversation, and continues the model.

## Decisions

- Background work never outlives the owning harness run.
- Background jobs are allowed inside child harnesses. Each harness run owns its own background jobs. A child background completion is injected into the child conversation, not the parent conversation.
- Do not add public polling, persistence, or stop APIs in this version.
- Add developer opt-in on `ToolSpec` with three modes:
  - `"never"`: default; no background option.
  - `"always"`: tool always starts in background.
  - `"model"`: model may choose by passing `_background: true`.
- Disallow `sequential=True` with any background mode other than `"never"`.
- Preserve run-level `tool_execution="sequential"` semantics:
  - model-choice background is not exposed in schemas while the harness is in global sequential mode;
  - `background="always"` tools are rejected by `Harness` construction/addition when `tool_execution="sequential"`.
- For `background="always"` and `background="never"`, the model does not see `_background` in the tool schema.
- For `background="model"`, the harness adds `_background: bool` to the model-facing schema, defaulting to `false`, then strips it before normal tool argument validation.
- Framework default `subagent` gets model-choice background behavior.
- Named configured subagents get their own `SubAgentConfig.background` policy, defaulting to `"never"`. The single framework `subagent` tool must special-case background policy by inspecting the `agent` argument before honoring `_background`.
- Built-in `parallel_llm` gets `background="model"`.
- Filesystem and other default workspace tools stay `background="never"`.
- Background completion order follows actual completion order, not original model call order.
- Requested tool-call accounting counts only the original model-requested tool call. Background completion injection does not increment `RunUsage.tool_calls`.
- Background tool failures are delivered to the model as completion messages. They do not trigger the existing automatic retry-budget loop, but the model may choose to call the tool again normally.
- Background completion continuations are normal model requests and count against `max_model_requests`. If the budget is exhausted before a pending completion can be delivered, the run fails through the existing limit path and drains pending tasks.
- Hold off on new background lifecycle hooks until concrete use cases appear.

## Public API Shape

Add a background mode to `ToolSpec`:

```python
ToolBackgroundMode = Literal["never", "always", "model"]

@dataclass(frozen=True)
class ToolSpec:
    ...
    background: ToolBackgroundMode = "never"
```

Validation:

```python
if self.sequential and self.background != "never":
    raise ValueError("sequential tools cannot run in background")
```

Schema behavior:

- `ToolSpec.response_tool()` returns the normal schema for `"never"` and `"always"`.
- For `"model"`, the returned schema includes an optional `_background` boolean unless global sequential mode suppresses background exposure.
- Schema injection must deep-copy the model-facing schema before mutation so `ToolSpec.parameters` and caller-owned manual schemas are not mutated.
- For object schemas with `additionalProperties: false`, preserve the closed schema and add `_background` to `properties`. If `required` exists, do not add `_background` to it.
- If the tool's schema already defines `_background`, raise a configuration error rather than silently merging.
- The executor strips `_background` from arguments before `_prepare_args()` validates the tool's real argument model.

Because schema exposure depends on harness-level `tool_execution`, `Harness.tool_schemas()` may need to request model-facing schemas through a helper that receives the active harness config rather than calling `ToolSpec.response_tool()` blindly.

For subagents, add:

```python
class SubAgentConfig(BaseModel):
    ...
    background: ToolBackgroundMode = "never"
```

The default subagent, selected when `agent` is omitted, behaves as `"model"`. A named subagent follows its own `SubAgentConfig.background` policy. If a named subagent is `"never"` and the model passes `_background: true`, return a normal retryable argument error rather than starting it.

## Model-Facing Behavior

When the model starts a background tool, it receives an immediate normal tool result:

```json
{
  "ok": true,
  "content": "Started background task bg_1 for tool subagent. Continue other work; the harness will notify you when it finishes.",
  "metadata": {
    "background_task_id": "bg_1",
    "tool_name": "subagent",
    "status": "running"
  }
}
```

When the background job finishes, the harness usually continues the model with a synthetic user message rather than a provider tool-result item, because the original background tool call was already satisfied by the start notice.

Completion message content should be short and structured enough for the model:

```text
Background task bg_1 completed.
Tool: subagent
Status: completed
Elapsed: 12345 ms
Output:
{"ok":true,"content":"...","metadata":{...}}
```

Failure uses the same message shape with `Status: failed` and the normalized failed tool envelope in `Output`.

If the model tries to finalize via text/native/prompted output while background jobs are pending, the harness waits for the next completion and continues with `continue_with_user_message(...)`.

If the model tries to finalize via `final_result` tool output while background jobs are pending, the harness must preserve provider tool/result pairing:

1. `resolve_turn_output(...)` should expose the final-result tool call id, for example as `OutputTurnDecision.final_tool_call_id`.
2. The harness should wait for the next background completion.
3. It should continue with `continue_with_tools([ToolOutput(final_tool_call_id, message)])`, where `message` says the final answer was deferred because background work completed and includes the background completion payload.
4. The model must then produce a new final answer, normally by calling `final_result` again.

This avoids leaving an unanswered `final_result` tool call in Anthropic/OpenRouter-style transcripts.

## Run Loop Semantics

The existing loop currently handles:

1. model turn
2. tool batch
3. provider continuation with tool outputs
4. finalization

The background version should preserve that structure for normal tools and add a per-run background manager:

1. Execute a model-requested tool batch.
2. For normal calls, await actual tool output as today.
3. For background calls, start an owned task and immediately return a start-notice `ToolOutput`.
4. Continue the model with the full batch of provider-facing outputs.
5. If a later model turn finalizes while background tasks are pending, do not finalize. Wait for the next background completion and continue the model with either a synthetic user message or a paired `final_result` tool output as described above.
6. If background tasks remain when the run is cancelled or errors, cancel and drain them before leaving the run.

Synthetic background completions should use a new model trace snapshot kind such as `"background_completion"`. They usually call `continue_with_user_message(...)`; the `final_result` pending case instead uses `continue_with_tools(...)` to satisfy provider tool/result pairing.

Sketch:

```python
while True:
    run_ctx.responses.append(turn.raw)

    if decision.kind == "final":
        if run_ctx.background.has_pending():
            completion = await run_ctx.background.wait_next()
            run_ctx.record_background_completion(completion)
            message = background_completion_message(completion)
            if decision.finalized_via_output_tool:
                final_id = decision.final_tool_call_id
                assert final_id is not None
                turn, decision = await run_ctx.advance_model(
                    lambda notices: active_session.continue_with_tools(
                        [ToolOutput(final_id, message)],
                        instructions=instructions,
                        tools=self.tool_schemas(),
                        metadata=metadata,
                        structured_output=structured_output,
                        notices=notices,
                    ),
                    trace_snapshot=ModelTraceSnapshot(kind="background_completion", prompt=message),
                )
            else:
                turn, decision = await run_ctx.advance_model(
                    lambda notices: active_session.continue_with_user_message(
                        message,
                        instructions=instructions,
                        tools=self.tool_schemas(),
                        metadata=metadata,
                        structured_output=structured_output,
                        notices=notices,
                    ),
                    trace_snapshot=ModelTraceSnapshot(kind="background_completion", prompt=message),
                )
            continue
        return run_ctx.finalize(...)

    if decision.kind in {"retry_tool_output", "retry_user_message"}:
        ...  # existing retry behavior
        continue

    if decision.kind == "unexpected":
        raise UnexpectedModelBehavior(...)

    run_ctx.check_tool_limit(len(turn.tool_calls))
    run_ctx.usage.tool_calls += len(turn.tool_calls)
    recorded, outputs, executions = await tool_executor.execute_batch(turn.tool_calls)
    run_ctx.record_tool_batch(recorded)
    run_ctx.check_tool_retry_limits(turn.tool_calls, executions)
    turn, decision = await run_ctx.advance_model(... continue_with_tools(outputs) ...)
```

If a background completion continuation would exceed `max_model_requests`, the existing `RunContext.check_model_limit()` path should raise. The run error path must cancel and drain any other pending background tasks before returning control.

## Background Manager

Add an internal per-run helper, likely in `thinharness/tool_execution.py` or a small companion module. `RunContext` should own one manager instance and pass it to `ToolBatchExecutor`, so the run loop can check `has_pending()`, wait for completions, record them, and drain/cancel on exits.

```python
@dataclass
class BackgroundToolTask:
    task_id: str
    tool_call_id: str
    tool_name: str
    arguments: str
    task: asyncio.Task[BackgroundToolCompletion]
    started_at: float

@dataclass
class BackgroundToolCompletion:
    task_id: str
    tool_call_id: str
    tool_name: str
    output: str
    elapsed_ms: float
    failed: bool
```

The manager should:

- allocate stable per-run ids like `bg_1`, `bg_2`;
- retain strong references to tasks;
- expose `has_pending()`;
- expose `wait_next()` returning the next completion in completion order;
- cancel and drain all pending tasks during run cancellation/error cleanup;
- record completion data into `RunContext.tool_call_records`.

Use completion-order behavior naturally by waiting on the task set with `asyncio.wait(..., return_when=asyncio.FIRST_COMPLETED)`.

The manager should start actual background execution tasks after the start-notice tool span closes and while the agent span is still current. The background coroutine should explicitly re-establish captured `_CURRENT_TOOL_CALL` and `_CURRENT_TOOL_RUNTIME` values around the real tool invocation so backgrounded subagents still receive parent call id and run metadata. This should parent the background execution span to the agent span rather than to the already-ended start-notice tool span.

Sync handlers running through `asyncio.to_thread` cannot be force-killed. Cancellation/drain waits for those workers to return, matching current `_invoke_tool(...)` behavior. Tests for cancellation should use async handlers.

## Tool Execution Changes

`ToolBatchExecutor.execute_batch()` currently returns records, provider-facing outputs, and executions in model call order. Preserve that for provider-facing outputs.

For background starts:

- `ToolCallExecutor` should still fire `before_tool_call` hooks before starting the background task.
- If `before_tool_call` cancels, return the cancellation output synchronously as today and do not start a background task.
- The immediate start notice should still pass through `after_tool_call` hooks because it is the model-visible output for the original tool call.
- The actual background execution should invoke the tool without firing `before_tool_call` and `after_tool_call` a second time. The user-visible lifecycle hooks for this version remain the original tool call hooks only.
- Existing `after_tool_call` hooks on a backgrounded call see the start notice, not the eventual tool result. This is a behavior difference for hook users who opt into background mode. For subagent-specific lifecycle data, use `before_subagent_run` and `after_subagent_run`.
- The actual background execution should still normalize tool output through the same `_invoke_tool(...)` path.
- Retry metadata in the eventual normalized output should not trigger automatic retry accounting. If a background handler raises `ModelRetry`, its retry envelope is delivered in the completion message and the model may choose to call the tool again as a new requested tool call.

Argument handling:

- Parse raw JSON once enough to detect `_background`.
- Use `_background` only if `ToolSpec.background == "model"`.
- Strip `_background` before passing args to `_invoke_tool(...)` or validation helpers.
- The stripping should happen in `ToolCallExecutor` or a helper before calling `_invoke_tool(...)`, because `_prepare_args(...)` currently lives inside `_invoke_tool(...)`.
- If `_background` is present for a `"never"` or `"always"` tool, let normal argument validation fail unless it was framework-injected. This reliably fails for Pydantic-backed strict args. Manual-schema tools may accept extra keys today; the model should not see `_background` for those tools, so this is not a hard guarantee for manual schemas.
- Multiple calls to the same background-capable tool are allowed. The harness does not deduplicate or serialize them beyond existing tool execution policy.

## Tool Call Records

Keep the existing record shape for original model-requested calls:

```json
{
  "call": {
    "id": "call_1",
    "name": "subagent",
    "arguments": "{\"task\":\"research\",\"_background\":true}"
  },
  "output": "{\"ok\":true,\"content\":\"Started background task bg_1...\"}",
  "background": {
    "task_id": "bg_1",
    "status": "running"
  }
}
```

Record completions honestly without inventing a fake model tool call:

```json
{
  "background": {
    "task_id": "bg_1",
    "tool_call_id": "call_1",
    "tool_name": "subagent",
    "event": "completed",
    "elapsed_ms": 12345
  },
  "output": "{\"ok\":true,\"content\":\"...actual output...\"}"
}
```

If a background task fails by returning a normalized failed tool envelope, `event` can still be `"completed"` with failed output. If the task itself raises outside normalization or is cancelled during cleanup, record `"event": "failed"` or `"event": "cancelled"` as appropriate.

This intentionally broadens the `tool_call_records` de-facto shape. Update in-repo consumers that assume every record has `record["call"]` so they either filter for records with `call` or handle background completion records explicitly.

## Tracing

`tool_call_records` are not used by tracing today. Background tracing should be explicit.

Trace structure:

- The original model-requested background call creates a short normal tool span for the start notice.
- The actual background execution creates its own `execute_tool <name>` span with background attributes:
  - `thinharness.background.task_id`
  - `thinharness.background.phase = "execution"`
  - `thinharness.background.original_tool_call_id`
  - `gen_ai.tool.name`
  - `gen_ai.tool.call.id`
  - optional captured arguments and result under the existing tracing options.
- Failed background execution marks the background execution span as error.
- The model turn that receives a background completion uses a new `ModelTraceSnapshot(kind="background_completion", prompt=...)` or equivalent and annotates input as a user message.
- Extending `ModelTraceSnapshot.kind` requires updating the `Literal`, `_model_request_input(...)`, and `_otel_input_messages(...)` so `background_completion` captures the completion message as a user-message-like input.

Local JSONL traces should naturally show:

1. model tool call request;
2. start-notice tool span;
3. background execution tool span;
4. model continuation from background completion.

## Built-In Tool Prompting

Add concise guidance to tool descriptions or tool instructions:

- `subagent`: background mode is available for long independent delegated work; default to synchronous unless explicitly useful or requested.
- `parallel_llm`: background mode is available for large independent batches; default to synchronous unless explicitly useful or requested.
- default system/tool instructions should say background is optional and should not be used for normal short work.

Do not add background guidance to filesystem tool descriptions.

## Tests

Focused tests should cover:

- `ToolSpec(background=...)` defaults and validation.
- `sequential=True` with `background != "never"` raises.
- `"model"` schemas include `_background`; `"always"` and `"never"` schemas do not.
- `background="always"` starts in background automatically and does not expose `_background`.
- global `tool_execution="sequential"` suppresses/rejects background as specified.
- schema collision with existing `_background` raises.
- manual JSON-schema background tools do not mutate caller-owned schemas across repeated `tool_schemas()` calls.
- `_background` is stripped before Pydantic validation.
- `_background` stripping happens before invocation for manual-schema tools too.
- background start returns immediate directive output and continues the model before the long tool finishes.
- if model finalizes while background is pending, harness waits for completion and continues instead of returning.
- if model calls `final_result` while background is pending, the harness answers the `final_result` tool call with a paired tool output and then asks for a new final result.
- multiple background tasks are delivered in completion order.
- `RunUsage.tool_calls` counts only the original model-requested calls.
- `max_tool_calls` reached after a background start does not block already-pending completion delivery.
- `max_model_requests` exhausted while background work is pending fails through `limit_reached` and drains pending tasks.
- background completion does not trigger automatic retry-budget accounting.
- background tasks inside a child harness are delivered to the child conversation, and the parent receives only the subagent final result.
- run cancellation drains pending background tasks.
- backgrounded subagents preserve parent call id, run metadata, and child metadata.
- `before_tool_call` and `after_tool_call` fire exactly once for a backgrounded call, and `after_tool_call` sees the start notice.
- tracing includes separate start and background execution spans with expected attributes.
- trace input capture works for `ModelTraceSnapshot(kind="background_completion")`.
- `tool_call_records` include both the start record and honest completion record.
- in-repo examples/e2e consumers tolerate completion records without `call`.

Likely focused test files:

- `tests/test_parallel_tools.py`
- `tests/test_subagents.py`
- `tests/test_tool_retry.py`
- `tests/test_tracing.py`
- possibly a new `tests/test_background_tools.py` if the scenarios become large.

## Validation

After implementation:

- `uv run pytest tests/test_background_tools.py tests/test_parallel_tools.py tests/test_subagents.py tests/test_tool_retry.py tests/test_tracing.py`
- `uv run pyright`
- relevant `ruff` command for touched files, matching the repo's existing check style.

## Out Of Scope

- Durable background tasks.
- Public polling APIs.
- Public cancellation APIs.
- Background lifecycle hooks.
- Background filesystem tools.
- Automatic retry-budget handling for background completions.
- Changing provider adapters to send late tool results for already-satisfied tool calls.
