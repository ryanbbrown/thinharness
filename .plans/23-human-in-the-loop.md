# Plan: Human-in-the-Loop Tool Approval (v2)

## Goal

Add a small, loop-level human-in-the-loop primitive for tool calls.

ThinHarness should own the agent-loop mechanics: when the model asks for a tool that requires approval, pause before side effects happen, return the pending approval details to the host application, and resume the same loop with an approve or reject decision.

The host application still owns the platform: users, auth, roles, storage, queues, notifications, review UI, audit retention, and escalation policy.

## Common Across Frameworks

Most compared frameworks with HITL converge on the same core primitive:

- A tool call can be marked as requiring approval.
- The agent loop pauses before executing that tool.
- The caller receives the tool name, call id, arguments, and enough run state to resume.
- The caller can approve or reject.
- On approval, the framework executes the real tool call and sends its result back to the model.
- On rejection, the framework sends a model-visible rejection result so the model can explain, recover, or choose a different path.

ThinHarness should implement this shared loop primitive directly. Without it, callers must fake approval through structured output, execute the tool outside the loop, and manually re-enter the harness, or fork/customize the run loop.

## Decisions

- Approval is tool execution control flow, not structured output.
- Approval happens before any tool in the model's batch is executed. A mixed batch (one approval-required call plus normal calls) pauses the whole batch; nothing executes until the host resumes.
- A paused run returns a normal `HarnessResult` with `stop_reason="approval_required"`. It is not an exception and not a `RunFailedEvent`; the stream finishes with `RunCompletedEvent` so `Harness.run()` works unchanged.
- The pause state is one JSON blob on `HarnessResult.resume_state`, a harness-level envelope (`kind="approval_pause"`) that wraps the provider session state, the pending tool batch, and the paused run's history and accounting. One blob to persist, one blob to resume from.
- The resume path is a new explicit API (`resume_approvals(...)` / `stream_approvals(...)`), not a callback. The host decides when and where to resume, including from a different process or a freshly constructed `Harness` with the same configuration.
- Rejection is represented as a normal failed tool result sent back to the model (`error_type: "ApprovalRejected"`, optional host-supplied reason), not as an exception to the caller. It does not set `retry: true`, so it does not consume the tool retry budget.
- Approval pauses require a resumable model. Registering an approval-required tool on a harness whose model lacks `resume_kind`/`resume_session` raises at configuration time, not at pause time.
- Approval-required tools are not supported inside child (subagent) harnesses in this version: a paused child run cannot be surfaced through the `subagent` tool result coherently. Explicitly configured approval tools (`SubAgentConfig.tools`) raise; inherited parent approval tools are filtered out of inheritance the same way `subagent` and MCP tools already are in `_effective_custom_tools`, so registering one approval tool on a parent does not break every delegation.
- `requires_approval=True` requires `background="never"` on the same `ToolSpec`. Combining approval with background execution is deferred.
- Every model request of an approval-required tool pauses, including a re-request after a rejection or after a retryable tool failure. There is no approve-once memory; that is host policy.
- `before_tool_call` / `after_tool_call` hooks fire at execution time during the resume run, not at pause time. Human rejection bypasses tool hooks entirely; the rejection output is authoritative.
- Approval is an interrupted logical run, not two runs. The envelope carries `usage`, `responses`, `tool_call_records`, `emitted_limit_warnings`, and `metadata` from the paused run; the resume run restores them, so the post-resume result contains the whole run history and `max_tool_calls` / `max_model_requests` budgets span the pause instead of resetting.
- The paused batch counts toward `usage.tool_calls` exactly once, at pause time — consistent with the existing rule that model-requested calls count even when hook-cancelled. The resume path must not re-count it. The increment lives in `_execute_tool_turn` (`thinharness/core.py:545`), not in `ToolBatchExecutor.execute_batch`, so the resume path can reuse `execute_batch` for the executed subset without double counting.
- Logical-run budgets have a user-visible edge: a resume can hit `limit_reached` on its first model request if the paused run was near budget. That is correct under these semantics; document it and test it.
- If background tasks are pending when a pause occurs, the pause branch cancels and drains them itself (`await run_ctx.background.cancel_and_drain()`, recording completions) before the paused result is built. This keeps `BackgroundTaskCompletedEvent`s ahead of `RunCompletedEvent`, lets `run_end` hooks observe the cancellation records in `tool_call_records`, and makes the envelope's cancelled-task list authoritative rather than racing the run-exit `finally` cleanup. The resume run tells the model about the cancellations via a harness notice.

## Public API Shape

### ToolSpec flag

```python
@dataclass(frozen=True)
class ToolSpec:
    ...
    requires_approval: bool = False
```

Validation in `ToolSpec.__post_init__` (`thinharness/tools/base.py`):

```python
if self.requires_approval and self.background != "never":
    raise ValueError("approval-required tools cannot use background execution")
```

Harness-side validation (`thinharness/core.py`), applied in `__init__` and `add_tool`:

- if any tool has `requires_approval=True` and the model is not resumable (the existing `hasattr(model, "resume_kind") and hasattr(model, "resume_session")` check), raise `ValueError("approval-required tools require a resumable model")`;
- if `_is_child_run` and any tool has `requires_approval=True`, raise `ValueError("approval-required tools are not supported inside subagents")`.

Ordering caveat: in `Harness.__init__` the constructor's `add_tool` loop currently runs before `self._is_child_run` is assigned, so either assign `_is_child_run` before the loop or run both checks after it; otherwise the `add_tool`-side check reads an unset attribute for constructor-passed tools.

In `thinharness/subagents.py`, `_effective_custom_tools` additionally filters out `tool.requires_approval` tools when building inherited child tool lists, and `SubAgentConfig.validate_subagent` rejects explicit `tools` entries with `requires_approval=True`.

Built-in filesystem/skill/subagent/parallel_llm/MCP tools stay `requires_approval=False`; there is no config knob for them in this version. Hosts that want approval on built-in behavior can wrap a builtin in their own `ToolSpec`.

### Leaf types

In `thinharness/types.py`:

```python
StopReason = Literal[
    ...,
    "approval_required",
]

@dataclass(frozen=True)
class PendingApproval:
    """One model-requested tool call awaiting a host decision."""

    call_id: str
    tool_name: str
    arguments: str  # raw JSON string exactly as the model sent it

@dataclass(frozen=True)
class ApprovalDecision:
    """One host decision for a pending approval."""

    call_id: str
    approved: bool
    reason: str | None = None  # included in the model-visible rejection output

@dataclass
class HarnessResult:
    ...
    pending_approvals: list[PendingApproval] = field(default_factory=list)
```

`pending_approvals` is non-empty only when `stop_reason == "approval_required"`. The same records are embedded in the resume envelope so the host can persist a single blob.

Export `PendingApproval` and `ApprovalDecision` from `thinharness/__init__.py`.

### Resume envelope

The pause writes a harness-level envelope into `HarnessResult.resume_state`:

```json
{
  "kind": "approval_pause",
  "version": 1,
  "provider_state": { "kind": "anthropic", "version": 1, "model": "...", ... },
  "batch": [
    {"id": "call_1", "name": "deploy", "arguments": "{\"env\":\"prod\"}"},
    {"id": "call_2", "name": "read", "arguments": "{\"path\":\"a.txt\"}"}
  ],
  "approval_required_ids": ["call_1"],
  "cancelled_background_task_ids": [],
  "usage": { ... },
  "responses": [ ... ],
  "tool_call_records": [ ... ],
  "emitted_limit_warnings": [ ... ],
  "metadata": { ... }
}
```

- `provider_state` is exactly `session.dump_state()` at pause time and is validated later by the provider's own `resume_session` (`_validate_resume_state` in `thinharness/providers.py`).
- `batch` preserves the full model-requested batch in model order, including calls that do not require approval, because none of them executed.
- `usage` is the paused run's `RunUsage` as plain JSON, captured after the pending batch was counted. `responses` and `tool_call_records` carry the run history so far — including the background-cancellation records from the pause-time drain — so the post-resume result is the whole logical run. `emitted_limit_warnings` carries the serialized `LimitNoticeKey` entries; JSON round-trips tuples into lists, so restore must convert them back to tuples. `metadata` is the paused run's metadata.
- `responses` is the only unbounded-size field (raw provider turns for the run so far), so envelope size grows with run length. Add a doc note so hosts sizing storage are not surprised.
- Envelope validation mirrors the provider style: required keys, `version == 1`, batch call ids unique, `approval_required_ids` a non-empty subset of batch ids, unknown keys rejected, JSON-serializability enforced via the existing `json.loads(json.dumps(...))` isolation in the pause path. Stored `usage`, `responses`, `tool_call_records`, `emitted_limit_warnings`, and `metadata` are shape-checked enough to restore without attribute errors.
- Passing an `approval_pause` envelope to `run(prompt, resume_from=...)` raises a clear `HarnessError` directing the caller to `resume_approvals`; passing a plain provider state to `resume_approvals` raises the inverse error. Detection is by the `kind` field.
- Provider-state errors raised through `_validate_resume_state` currently say "resume_from ..."; if the implementation touches that function, prefer adding a `label` parameter so approval-path errors say "approval state ..." instead.

### Resume API

On `Harness`:

```python
async def resume_approvals(
    self,
    state: dict[str, Any],
    decisions: list[ApprovalDecision],
    *,
    metadata: Json | None = None,
) -> HarnessResult: ...

def stream_approvals(
    self,
    state: dict[str, Any],
    decisions: list[ApprovalDecision],
    *,
    metadata: Json | None = None,
    stream_options: StreamOptions | None = None,
) -> HarnessStream: ...

def resume_approvals_sync(self, state, decisions, *, metadata=None) -> HarnessResult: ...
```

`resume_approvals` is implemented over `stream_approvals` the same way `run` wraps `stream`; `resume_approvals_sync` mirrors `run_sync` (including `aclose()` on exit). The usual `_running` / `_closed` guards apply.

Decision validation, before any session or tool work:

- envelope shape validation (kind, version, required keys, unknown keys, unique batch call ids);
- exactly one decision per id in `approval_required_ids`; missing, duplicate, or unknown `call_id` values raise `HarnessError`;
- decisions for ids not in `approval_required_ids` raise (the host does not get to veto normal calls through this path).

`metadata` is not merged: caller-supplied resume metadata wins when provided; otherwise the restored envelope metadata is used. No silent deep-merge of the two.

Tool-existence validation runs later, after `_ensure_mcp_connected()` (MCP tools only enter `_tool_map` at connect time, so checking earlier would falsely fail any paused batch containing an MCP call on a freshly constructed harness):

- every *approval-required* call's tool must exist in `_tool_map`; a missing one raises `HarnessError` naming it — the host approved an action the harness can no longer perform, which is a configuration error;
- non-approval batch calls keep today's semantics: an unknown tool name flows through `ToolCallExecutor._call_output` and produces the model-visible `"unknown tool"` output instead of failing the run.

The resume harness must be configured equivalently to the paused one (same model ref — enforced by provider state validation — and a superset of the approval-required tool names).

## Run Loop Semantics

### Pausing

In `Harness._run_streaming`, immediately before `_execute_tool_turn(...)` (after the `decision.kind` branches and after the existing background over-budget rejection branch):

```python
approval_calls = [
    call for call in turn.tool_calls
    if (spec := self._tool_map.get(str(call.name))) is not None and spec.requires_approval
]
if approval_calls:
    run_ctx.check_tool_limit(len(turn.tool_calls))  # fail fast before bothering a human
    run_ctx.usage.tool_calls += len(turn.tool_calls)  # the batch counts exactly once, here
    cancelled_ids = await run_ctx.cancel_pending_background()  # drain + record before the result exists
    return run_ctx.pause_for_approval(turn, approval_calls, active_session, cancelled_ids)
```

Notes:

- `run_ctx.responses` already contains `turn.raw` (appended at the top of the loop), so the paused result faithfully includes the model turn that requested the tools.
- `check_tool_limit` runs at pause time so an over-budget batch fails through the existing `limit_reached` path instead of asking for an approval that could never execute. `usage.tool_calls` IS incremented at pause — the model-requested batch counts exactly once, here, and the envelope captures usage after the increment.
- No `before_tool_call` hooks fire, no `ToolCallStartedEvent` is emitted, and no tool spans open for the paused batch; all of that happens during the resume run.
- The `final_result` synthetic tool can never appear in a paused batch: `resolve_turn_output` resolves `final_result` turns to `final` / `retry` / `unexpected` before the `continue` path is reached, and `final_result` cannot be a registered approval tool (the name is reserved).

`RunContext.cancel_pending_background()` (new, in `thinharness/runtime.py`) wraps `background.cancel_and_drain()`, records each completion via `record_background_completion`, and returns the cancelled task ids. Running it in the pause branch — not leaving it to the run-exit `finally` — keeps `BackgroundTaskCompletedEvent`s ahead of `RunCompletedEvent` (matching every other terminal path), lets `run_end` hooks observe the cancellation records, and makes the envelope authoritative. The `finally` cleanup then finds nothing pending.

`RunContext.pause_for_approval(...)` (new, in `thinharness/runtime.py`, alongside `finalize`):

1. Sets `stop_reason = "approval_required"`.
2. Calls `session.dump_state()`; `None` raises `HarnessError("approval pause requires session resume state")`.
3. Builds the envelope (JSON-isolated copy) with `cancelled_background_task_ids` from the drain above, plus `usage` (captured after the batch increment), `responses`, `tool_call_records` (including the cancellation records), `emitted_limit_warnings`, and `metadata` from the run context.
4. Builds the result via `build_terminal_result(turn.text)` with `pending_approvals` populated and `resume_state` set to the envelope. `text` carries any commentary the model emitted alongside the tool calls.
5. Annotates the agent span (see Tracing), fires `fire_run_end_once()` (the `RunEndContext` sees `stop_reason="approval_required"` and the paused result), and emits `RunCompletedEvent`.

Because the pause path returns normally, `_classify_run_failure` and `RunFailedEvent` are not involved.

### Resuming

`stream_approvals` reuses `_run_streaming` with a third first-turn kind. `_start_or_resume_turn` currently handles `"start"` and `"resume"`; add `"approval_resume"`, which instead of calling `driver.start/resume`:

1. Validates the envelope shape and decisions (above).
2. Calls `self.model.resume_session(envelope["provider_state"])` (cast through the existing `ResumableModel` path).
3. Runs the start ceremony. Today `_prepare_run_start` (called unconditionally by `_run_streaming`) fires `RunStartContext`, connects MCP, and fires `UserPromptSubmitContext`; adapt it (parameter or split) so the approval-resume path fires `RunStartContext` with `prompt=""`, runs `_ensure_mcp_connected()`, and skips `UserPromptSubmitContext` — there is no user prompt. `RunStartedEvent` is emitted with `prompt=None` (the field is already `str | None`), followed immediately by `ApprovalResumedEvent` carrying the normalized decisions.
4. Validates tool existence for approval-required calls (now that MCP tools are in `_tool_map`), reconstructs the batch as `ModelToolCall` values, and processes it:
   - Before processing the batch, the `RunContext` is seeded from the envelope: `usage` restored as `RunUsage`, `responses` and `tool_call_records` as lists, `emitted_limit_warnings` as a `set` of `LimitNoticeKey` tuples (lists from JSON converted back), and metadata per the no-merge rule above.
   - No `check_tool_limit` and no `usage.tool_calls` increment for the batch — both happened at pause time and the restored usage already includes it. The resume batch path must still replicate `_execute_tool_turn`'s `cancelled_tool_calls` accounting for hook-cancelled approved calls (`thinharness/core.py:547`), since it bypasses that helper.
   - Approved calls and calls that never required approval execute through the existing `ToolBatchExecutor.execute_batch(...)` machinery: `before_tool_call` / `after_tool_call` hooks, tracing spans, stream events, retry-kind detection, and unknown-tool outputs all behave as for any normal batch. `check_tool_retry_limits` is called with only the executed subset of calls and executions — it zips them `strict=True`, so rejected calls must not be in either sequence.
   - Rejected calls do not execute and do not fire tool hooks. Each produces `ToolOutput(call.id, ToolResult(False, message, {"error_type": "ApprovalRejected"}).as_json())` where `message` is `"Tool call was rejected by a human reviewer."` plus `"\nReason: {reason}"` when provided. Each rejected call emits a paired `ToolCallStartedEvent` / `ToolCallCompletedEvent` (`ok=False`, `error_type="ApprovalRejected"`) and a short error-status tool span, mirroring the hook-cancellation shape.
   - Outputs are merged back into model order before being sent to the provider.
   - `tool_call_records` entries for the batch gain an `"approval"` key: `{"approved": true}` or `{"approved": false, "reason": ...}` on the records for calls that required approval.
5. Continues the model with `driver.send_tool_outputs(outputs, kind="approval_resume")`. If `cancelled_background_task_ids` is non-empty, the request also carries a harness notice (see below) telling the model those background tasks were cancelled by the pause.
6. Returns `(session, driver, turn, decision)` into the existing while-loop unchanged. A later turn can pause again (another envelope), finalize, retry structured output, etc.

Batch execution happens before the first model request of the resume run, so a resume that only executes tools and then finalizes costs one model request.

### Provider compatibility

All three built-in session shapes already support this resume point:

- OpenAI Responses: `previous_response_id` is updated in `_complete` to the response that contained the function calls; `continue_with_tools` sends `function_call_output` items against it. Cross-process resume relies on the response being stored server-side (no `store: false` in `extra_body`) — same constraint as existing resume, but worth a doc note since approval pauses make cross-process resume the headline use case.
- Anthropic Messages: `dump_state()` copies `messages` after `_complete` appended the assistant `tool_use` turn; `continue_with_tools` appends the `tool_result` user message.
- OpenRouter: same transcript shape with `role: "tool"` messages.

No provider adapter changes are needed beyond the notice extension below.

### Background interaction

A pause while background tasks are pending cannot wait (unbounded latency) and cannot deliver completions (the tool batch is outstanding, and Anthropic-style transcripts require tool results next). So the pause lets the existing run-exit cleanup cancel and drain them, records the cancellations in `tool_call_records`, and stores the task ids in the envelope.

To keep the resumed model coherent (it saw "Started background task bg_1" and will never see a completion), extend `ModelNotice.kind` to `Literal["limit_warning", "background_cancelled"]` and have the approval-resume continuation append one notice listing the cancelled task ids (no notice when the list is empty). `render_model_notices` already renders generically, so this is the only provider-layer touch, but the harness-side plumbing has two knock-on changes:

- `TurnDriver.send_tool_outputs` gains an `extra_notices: list[ModelNotice]` parameter that the request lambda concatenates with the limit notices `advance_model` passes in — the lambda is built inside `TurnDriver`, so there is no other injection point.
- `advance_model`'s notice-event loop currently asserts `limit_kind`/`remaining` are set on every notice before emitting `LimitWarningEvent`; it must emit limit events only for `kind == "limit_warning"` and skip other kinds. Relatedly, `LimitNoticeKey` (`types.py`) and `_limit_notice_dedup_key` (`runtime.py`) are typed against the `Literal["limit_warning"]` kind, so widening `ModelNotice.kind` needs a narrowing there for pyright to pass.

## Events and Tracing

- `ModelRequestStartedEvent.request_kind` and `TurnDriver.send_tool_outputs(kind=...)` / `ModelTraceSnapshot.kind` Literals gain `"approval_resume"`. Update `_model_request_input(...)` / `_otel_input_messages(...)` in `thinharness/tracing.py` so the kind captures tool outputs the same way `"tool_outputs"` does.
- No `ApprovalRequiredEvent`: terminal outcomes are conveyed once, through `RunCompletedEvent.result.stop_reason`, exactly as `limit_reached` is today (`runtime.py` finalizes both through `RunCompletedEvent`; `RunFailedEvent` is reserved for exception paths). A herald event immediately before the terminal event would duplicate `pending_approvals` with no information or timing gain — the pause is the last event either way. If kind-dispatch for terminal outcomes ever matters, the coherent design is a distinct terminal event replacing `RunCompletedEvent` for pauses, not a duplicate preceding one.
- Add `ApprovalResumedEvent` (`kind="approval_resumed"`, frozen `StreamEvent` dataclass carrying the normalized decisions as a tuple of `ApprovalDecision`), emitted immediately after `RunStartedEvent` on the resume stream. It makes resume streams self-describing for passive consumers (trace viewers, log sinks) — without it the only markers are `RunStartedEvent(prompt=None)` and tool events appearing before any model request. Extend `StreamEventKind`, the `HarnessStreamEvent` union, and the `thinharness/__init__.py` exports.
- At pause, annotate the agent span: `thinharness.approval.paused = true`, `thinharness.approval.pending_call_ids = [...]`, plus the normal terminal annotations with the paused result.
- On resume, rejected calls get a short tool span with error type `ApprovalRejected`; approved calls trace exactly like normal tool calls.

## Model-Facing Behavior

The model never sees the pause. From its perspective it requested a tool batch and, one provider turn later, received outputs:

Approved call — the real tool output, unchanged.

Rejected call:

```json
{
  "ok": false,
  "content": "Tool call was rejected by a human reviewer.\nReason: production deploys are frozen until Monday.",
  "metadata": {"error_type": "ApprovalRejected"}
}
```

No `retry: true`, so rejection never consumes the per-tool retry budget (`_tool_retry_kind` requires it). The model can explain, pick another path, or re-request the tool — which pauses again.

Tool descriptions and system instructions are not changed; approval is invisible to the model by design.

## Host-Facing Usage

```python
result = await harness.run("deploy the release")
if result.stop_reason == "approval_required":
    blob = result.resume_state          # persist anywhere; JSON
    pending = result.pending_approvals  # show to a human

    # later, possibly another process / fresh Harness with the same config:
    result = await harness.resume_approvals(
        blob,
        [ApprovalDecision(call_id=p.call_id, approved=True) for p in pending],
    )
```

## Code Placement

- `thinharness/types.py`: `PendingApproval`, `ApprovalDecision`, `StopReason` extension, `HarnessResult.pending_approvals`.
- `thinharness/approvals.py` (new, small): envelope build/validate helpers and the decided-batch split/merge logic, keeping `core.py` and `runtime.py` at their current altitude (consistent with the plan-20 decomposition).
- `thinharness/runtime.py`: `RunContext.pause_for_approval`, `RunContext.cancel_pending_background`, `send_tool_outputs` kind + `extra_notices` extensions, limit-notice narrowing in `advance_model` / `_limit_notice_dedup_key`.
- `thinharness/core.py`: pause check in the loop, `"approval_resume"` branch in `_start_or_resume_turn`, `_prepare_run_start` adaptation, the three public resume methods, construction-time validation.
- `thinharness/tools/base.py`: `ToolSpec.requires_approval` + validation.
- `thinharness/subagents.py`: inheritance filtering in `_effective_custom_tools`, explicit-tool rejection in `SubAgentConfig`.
- `thinharness/providers.py`: `ModelNotice.kind` extension only (plus the matching `LimitNoticeKey` narrowing in `types.py`).
- `thinharness/events.py`: `ApprovalResumedEvent`, `StreamEventKind` / union / export extensions, `request_kind` Literal. `thinharness/tracing.py`: input-capture mapping for `"approval_resume"`.

## Tests

New `tests/test_approvals.py`, using the existing fakes (`ScriptedModel`/`ScriptedSession`, `MultiCallClient`, per-provider fake providers from `tests/fakes.py` and the `test_resume.py` patterns):

- `ToolSpec(requires_approval=True)` defaults and validation; `requires_approval` + `background != "never"` raises.
- Harness construction/`add_tool` with an approval tool raises for a non-resumable model; tools passed through the constructor hit the same checks as `add_tool` (covers the `_is_child_run` assignment-ordering caveat).
- Subagents: `SubAgentConfig.tools` containing an approval tool raises at config validation; the default subagent and `inherit_parent_tools=True` subagents silently exclude the parent's approval tool from the child tool list and still run.
- An approval-required tool does not execute before approval: handler not called, no `before_tool_call` hook, no `ToolCallStartedEvent`, no tool span.
- The paused result has `stop_reason="approval_required"`, populated `pending_approvals`, an `approval_pause` envelope in `resume_state`, the model turn in `responses`, and the model's commentary in `text`; `run_end` hooks observed the paused result.
- A mixed batch pauses everything; the non-approval call also did not execute.
- Approving resumes the run, executes the original call with the original arguments, sends the real output back via `continue_with_tools`, and finishes; `tool_call_records` carry the `approval` annotation.
- Rejecting sends the `ApprovalRejected` output (with and without `reason`), the model sees it and can finish; the rejection does not increment `tool_retries`.
- Mixed approve/reject in one batch merges outputs in model order.
- Decision validation: missing, duplicate, unknown, and extra decisions each raise before any execution; envelopes with duplicate batch call ids are rejected.
- `run(prompt, resume_from=envelope)` and `resume_approvals(provider_state, ...)` both raise the directed errors; `resume_approvals` on a closed or already-running harness hits the usual guards; `resume_approvals_sync` works and closes the harness like `run_sync`.
- Resuming on a freshly constructed harness with the same config works, including after a string round-trip of the envelope (`json.loads(json.dumps(blob))`); a missing approval-required tool raises.
- A paused mixed batch containing an MCP tool call resumes on a fresh harness with the same MCP config (tool validation must wait for MCP connect); an unknown non-approval sibling tool still yields the model-visible `"unknown tool"` output rather than failing the run.
- Envelope round-trips through real provider session shapes for OpenAI, Anthropic, and OpenRouter fakes (pause after the function-call turn, resume, assert the continuation payload pairs tool outputs with the original call ids).
- Re-pause: the resumed model requests the approval tool again and the run pauses again with a fresh envelope.
- Limits and accounting: an over-budget approval batch fails `limit_reached` at pause time without producing an envelope; `usage.tool_calls` counts the paused batch exactly once across pause + resume; budgets span the pause — a run paused near `max_model_requests` hits `limit_reached` on the resume's first model request; limit-warning dedup keys survive the JSON round trip and near-limit notices are not re-sent after resume; `usage.tool_retries` survives the round trip.
- History: the post-resume result's `responses` and `tool_call_records` include the pre-pause turns and records, including background-cancellation records from the pause drain.
- Metadata: caller-supplied resume metadata wins; otherwise the restored envelope metadata is used; the two are never merged.
- Hooks on resume: `before_tool_call` can still cancel an approved call (hook-cancel output, counts as today); `after_tool_call` fires for approved calls only.
- Structured output: a run with `output_type` pauses and, after resume, finalizes through `final_result` normally.
- Tool retry: an approved call returning a retry envelope goes through normal retry accounting; the re-request pauses again. A mixed batch with one rejected call and one approved retry-envelope call works (`check_tool_retry_limits` only sees the executed subset).
- Background: pausing with a pending background task cancels it, records the cancellation in `tool_call_records` before `run_end` hooks fire, emits `BackgroundTaskCompletedEvent` before `RunCompletedEvent`, stores the id in the envelope, and the resume request carries the `background_cancelled` notice; an envelope with no cancelled ids produces no notice.
- Tracing: pause annotations on the agent span; `"approval_resume"` snapshot captures tool outputs; rejected-call span has `ApprovalRejected`.
- Streaming: pause stream ends with `RunCompletedEvent` (no herald event before it); resume stream emits `ApprovalResumedEvent` with the decisions immediately after `RunStartedEvent(prompt=None)`, then tool events for the batch, then `request_kind="approval_resume"` on the first model request.

## Validation

After implementation:

- `uv run pytest tests/test_approvals.py tests/test_resume.py tests/test_harness.py tests/test_hooks.py tests/test_background_tools.py tests/test_streaming.py tests/test_tracing.py`
- `uv run pyright`
- relevant `ruff` checks for touched files.
- Update `docs/docs.md` (tools + resume sections), `docs/decisions.md` (approval is loop control flow; `resume_from` semantics unchanged; logical-run accounting and budgets span pause/resume; envelope size grows with run length), and `CHANGELOG.md`.

## Deliberate Non-Goals

These vary across frameworks and are platform-shaped, so ThinHarness should not own them:

- Review UI.
- User identity, auth, roles, or permission checks.
- Persistent approval storage.
- Queues, notifications, reminders, escalation, or SLA handling.
- Audit-retention policy.
- Workflow-step approval separate from model tool calls.
- Multi-party approval.
- Hosted approval dashboards or control planes.
- Product-specific policy DSLs.

## Defer

These may be useful, but should wait until the base primitive is proven:

- Dynamic approval predicates based on arguments or run metadata.
- Argument editing before approval.
- Inline auto-approval callbacks (including approve-once / session-allowlist memory).
- Partial execution of mixed approved/non-approved batches (execute safe calls before pausing).
- Approval-required tools combined with background execution.
- Approval support inside subagent child harnesses.
- A dedicated `ApprovalRequiredEvent` stream event (see Events and Tracing — duplicative with `RunCompletedEvent`; would only return as part of a kind-per-terminal-outcome redesign).
- A config knob to mark built-in tools approval-required.
- Approval timeouts and policy composition.

## Verification

- A tool marked approval-required does not execute before approval.
- The run returns `approval_required`, pending approval records, and resume state.
- Approving resumes the same run, executes the original tool call, and sends the tool result back to the model.
- Rejecting resumes the same run and sends a rejection tool result back to the model.
- Mixed batches pause before any side effects.
- Approval pauses work with provider resume state for OpenAI, Anthropic, and OpenRouter session shapes.
- Approval pauses interact cleanly with tool-call limits, tool retries, background tools, hooks, tracing, and structured output.
- The paused batch counts toward `usage.tool_calls` exactly once across pause + resume; budgets and limit-warning dedup span the pause; the post-resume result carries the full logical-run history.
