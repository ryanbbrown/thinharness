# Tracing and event projection from transcript deltas

## Goal

Reduce duplicated conversation-shape construction in tracing and runtime now that built-in providers maintain a neutral transcript. The durable transcript is the canonical model-visible conversation state; tracing and streaming should be projections from the same per-request model-visible delta where that makes sense.

This is intentionally a behavior cleanup, not a backwards-compatibility exercise. Trace input should reflect the exact model-visible content, including rendered `<harness_notice>` text. Keeping a separate "logical but not quite what the model saw" trace view is unnecessary complexity.

## Current state

The new transcript introduced by `.plans/29-unified-transcript-log.md` is the durable model-visible log:

- `UserEntry(content, notice=False)`
- `AssistantEntry(text, tool_calls)`
- `ToolResultEntry(call_id, output)`
- `UserEntry(content, notice=True)` for notice text accompanying tool-result batches

Tracing currently has a parallel request-log model:

- `ModelTraceSnapshot` in `tracing.py` carries `kind`, `prompt`, `tool_outputs`, `notices`, and `structured_output`.
- `TurnDriver` builds that snapshot independently in `runtime.py` for `start`, `resume`, `send_tool_outputs`, and `send_user_message`.
- `tracing.py` then rebuilds model input payloads again via `_model_request_input()` and `_otel_input_messages()`.
- `tracing.py` rebuilds assistant output again via `_otel_output_messages(turn)`.
- `runtime.py` separately emits `ModelMessageEvent` by mapping `ModelTurn.tool_calls` to `StreamToolCall`.

The event stream is not just the transcript. It includes operational lifecycle notifications that are not durable model-visible conversation entries:

- run start/completion/failure
- model request start
- tool call start/completion
- background task start/completion
- retry and limit notifications
- approval resume

So the right target is not "make events.py equal transcript." The right target is "use one model-visible request/response delta as the source for durable transcript, model tracing payloads, and model-message stream projection." Operational events remain separate.

## Design sketch

Introduce an internal `ModelRequestDelta` (name flexible) near the provider/runtime boundary:

```python
@dataclass(frozen=True)
class ModelRequestDelta:
    kind: Literal["start", "resume", "tool_outputs", "approval_resume", "correction", "output_retry_tool", "background_completion"]
    entries: list[TranscriptEntry]
    notices: list[ModelNotice] = field(default_factory=list)
    structured_output: str | None = None
```

The delta represents the exact new model-visible input for one provider request, after hooks/background/limit notices have been rendered into user text where applicable. `notices` remains as structured metadata for observability, but the primary trace input should include the rendered notice text because that is what the model saw.

Examples:

- start/resume/correction: one `UserEntry(content=append_notices_to_text(prompt, notices))`
- tool output request: N `ToolResultEntry`s plus optional `UserEntry(notice=True)` if notices are sent as user text
- background completion as user message: one `UserEntry(content=...)`
- approval resume: same as tool output request, with `kind="approval_resume"`

Then project it to:

- tracing input attributes from `entries` plus structured `notices` metadata (`gen_ai.input.messages`, `gen_ai.prompt`, `langfuse.observation.input`, `thinharness.model.notices`)
- request-kind stream metadata (`ModelRequestStartedEvent.request_kind`)
- provider transcript append logic, if provider sessions can accept already-rendered entries without leaking into their native in-run request builders

For assistant output, add a helper from `AssistantEntry` / `ModelTurn` to:

- tracing output attributes (`gen_ai.output.messages`, `langfuse.observation.output`)
- `ModelMessageEvent`
- durable transcript append, if provider sessions can share that helper

## Boundaries

Keep these separate:

- **Conversation log:** model-visible entries only. This is what durable resume uses.
- **Operational stream:** lifecycle notifications. It includes some conversation projections, but also events that happen before or outside model-visible transcript entries.
- **Trace spans:** operational timing plus optional model-visible payload capture.

Do not add operational metadata to durable transcript entries. Avoid storing timing, stream sequence, tracing capture policy, or output-mode details in `resume_state`.

Public stream event dataclasses should stay mostly stable, with one intended simplification: remove `StreamOptions.include_model_text`. The stream is an observability/log surface; suppressing assistant text makes it less useful and adds branching that other agent frameworks do not appear to expose as a core option. If callers need redaction, that should be a separate filtering layer outside the core event schema.

## Implementation steps

1. Add projection helpers in a small internal module, e.g. `thinharness/projections.py`, so `providers.py` does not learn about tracing or stream-event concerns:
   - `trace_input_messages_from_entries(entries)`
   - `trace_output_messages_from_assistant(entry_or_turn)`
   - `model_request_input_from_delta(delta)`
   - `stream_tool_calls_from_assistant(entry_or_turn)`

2. Replace `ModelTraceSnapshot.prompt/tool_outputs/notices` with `ModelRequestDelta.entries/notices`, or adapt `ModelTraceSnapshot` into this shape as an intermediate step. Do not keep separate logical/model-visible entry lists.

3. Update `TurnDriver` to build one delta per model request after notices are known:
   - It currently builds `ModelTraceSnapshot` before `advance_model`, but final limit/background notices are only known inside `RunContext.advance_model`.
   - The likely place to finalize the delta is inside `advance_model`, after `all_notices` is computed and before `annotate_model_request`.

4. Update `annotate_model_request()` to consume the delta and remove `_otel_input_messages()` branches that duplicate request-kind logic. Trace input should include rendered notices because the delta is model-visible.

5. Update `annotate_model_span()` to use the same assistant-entry projection helper used by stream `ModelMessageEvent`.

6. Keep stream lifecycle event emission in `runtime.py` and tool execution code. Use the shared assistant projection for `ModelMessageEvent` and always include assistant text.

7. Remove `StreamOptions.include_model_text` and the associated runtime branch. Keep `StreamOptions.include_subagents` unless a separate review shows it is also unnecessary.

8. Leave provider-native in-run request builders untouched. This must preserve the dual-store boundary from plan 29: normal in-run provider payloads should stay byte-identical.

## Tests

This is mostly a refactor, with two intentional behavior/API changes:

- trace input messages now represent exact model-visible input, including rendered notices
- stream events no longer support suppressing assistant text via `StreamOptions.include_model_text`

Required checks:

- Existing tracing tests in `tests/test_tracing.py` should be updated where they currently expect notices to be absent from trace input messages.
- Existing streaming tests in `tests/test_streaming.py` and approval streaming tests should be updated if they cover `StreamOptions.include_model_text`.
- Existing provider payload tests should remain byte-identical; any changed in-run payload means transcript projection leaked into provider request construction.
- Full `uv run pyright`, relevant `ruff`, and full `pytest`.

New or adjusted tests worth adding:

- One tracing test that asserts a tool-output request with a notice includes the rendered notice in trace input and also preserves structured notice metadata.
- One tracing test that asserts a prompt-path notice includes the rendered notice in trace input.
- One tracing test that asserts assistant text + tool calls uses the same projection as `ModelMessageEvent`.
- One regression test for resume first-turn tracing: replayed transcript itself should not be double-counted as the new request delta unless that is intentionally exposed.
- One streaming test update proving model text is always included in `ModelMessageEvent`.

## Risks

- Tracing has sink-specific conventions (`gen_ai.*`, Langfuse) that are not identical to durable transcript shape. The shared helper should project transcript entries into those conventions rather than forcing tracing attributes to become transcript dictionaries.
- Including rendered notices in trace input is a trace-shape behavior change. This is accepted for simplicity and fidelity to what the model saw, but docs/tests should make it explicit.
- Limit notices are computed inside `advance_model`, while request kind/prompt/tool outputs are prepared in `TurnDriver`. Moving delta finalization too early will miss notices; moving too late can obscure the original request kind.
- Stream events include operational timing and partial lifecycle states. Treating the stream as "just the transcript" would lose useful host notifications like `tool_call_started`.

## Out of scope

- Broad changes to public stream event schemas beyond removing `StreamOptions.include_model_text`.
- Adding token streaming.
- Making tracing required or changing capture defaults.
- Moving tool execution records into durable transcript.
- Reducing provider-native in-run state or changing the transcript resume envelope.
