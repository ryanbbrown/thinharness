# Plan: OTel GenAI Tracing

Spec: none found.

## Overview
ThinHarness already has OpenTelemetry span plumbing for agent, model, tool, and subagent execution, but it does not write the standard GenAI request and response content attributes that tracing platforms need to show useful inputs and outputs. The fix is to add a platform-neutral OTel GenAI content writer in `thinharness/tracing.py`, wire explicit request snapshots through `core.py`, and keep Langfuse as one validation target via a small OTLP helper.

## Findings
ThinHarness currently creates the right span tree: `invoke_agent <name>` root spans, `chat <model>` generation spans, `execute_tool <name>` tool spans, and nested child agent spans through copied `TracingOptions` in `thinharness/subagents.py`.

`create_otlp_tracing()` is only transport setup. It creates an OTLP HTTP exporter, but there is no writer that sets OTel GenAI content attributes like `gen_ai.system_instructions`, `gen_ai.input.messages`, and `gen_ai.output.messages`, nor compatibility attributes that common backends map into observation input/output.

`TracingOptions.capture_messages=True` currently writes only `gen_ai.completion` from normalized model response text. It does not capture the user prompt, system instructions, tool continuation input, corrective retry message, provider request payload, final root output, or structured parsed result.

The prompt overwrite risk is real if tracing infers input from the final provider payload shape. Notices are appended into provider input in `thinharness/providers.py`; for tool continuations, OpenAI appends notices as a trailing user message after `function_call_output` items, Anthropic appends notices as a text block after `tool_result` blocks, and OpenRouter appends notices as a trailing user message after tool messages. A writer that uses "last user text" as the observation input can show the notice instead of the real prompt or tool outputs.

`~/code/pi-agent-sdk/src/tracing.ts` is the useful reference for span lifecycle and OTLP transport setup: it starts agent spans on `agent_start`, model spans on assistant `message_start`, tool spans by `toolCallId`, nests subagents under the delegate tool context, and uses `createOtlpTracing()` with Langfuse Basic auth plus `x-langfuse-ingestion-version: 4`. It is not a content-writer reference: it writes completion fallback data but does not write request input attributes, so ThinHarness should deliberately go beyond that implementation.

Current OpenTelemetry GenAI semconv says content capture should be opt-in and, when enabled, use `gen_ai.system_instructions`, `gen_ai.input.messages`, and `gen_ai.output.messages`: https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/. Current Langfuse docs recommend direct `langfuse.*` attributes for manual instrumentation because those take precedence over generic OTel mappings, and they map observation input/output from `langfuse.observation.input` / `langfuse.observation.output` or fallback `gen_ai.prompt` / `gen_ai.completion`: https://langfuse.com/integrations/native/opentelemetry.

## Steps

### 1. Add an explicit model trace snapshot shape
Add provider-neutral dataclasses in `thinharness/tracing.py` so model spans receive canonical request data from core rather than scraping provider payloads after notices have been rendered.

```python
# thinharness/tracing.py
@dataclass(frozen=True)
class ModelTraceSnapshot:
    """Canonical input for one model span."""

    kind: Literal["start", "resume", "tool_outputs", "correction", "output_retry_tool"]
    prompt: str | None = None
    tool_outputs: list[Json] | None = None
    notices: list[Json] | None = None
    structured_output: str | None = None

    def with_notices(self, notices: list[ModelNotice]) -> ModelTraceSnapshot:
        """Return a copy with model-facing notices attached."""
        serialized = [asdict(notice) for notice in notices]
        return replace(self, notices=serialized or None)
```

The important rule is that `prompt`, corrective `message`, and `tool_outputs` stay separate from `notices`. The writer can include notices in metadata, but notices must never replace the observation input.

Also define `_trace_output_mode(output_schema)` so core can record `"text"`, `"tool"`, `"native"`, `"prompted"`, or `None` without importing output internals into every call site. The writer must use this field for `thinharness.output.mode_requested`; if implementation pressure makes that field unnecessary, drop both `_trace_output_mode()` and `structured_output` rather than leaving dead snapshot data.

Keep `output_retry=True` as a separate `advance_model()` parameter. It is usage accounting, not tracing. The snapshot kind describes the provider call shape: `correction` means a user-message retry via `continue_with_user_message(...)`, while `output_retry_tool` means the structured-output tool-mode retry via `continue_with_tools([ToolOutput(final_id, retry_message)], ...)`.

**Verify:** Add a unit test that builds a `ModelTraceSnapshot(kind="tool_outputs", tool_outputs=[...], notices=[...])`, calls the writer, and asserts the serialized input contains the tool output object and not just the notice text. Add a one-line test that `serialize_attribute_value(None) is None` so missing content keeps being dropped rather than written as `"null"`.

### 2. Implement the OTel GenAI content writer
Add helper functions in `thinharness/tracing.py` that annotate spans with OTel GenAI attributes first, plus compatibility attributes for platforms that still map older or backend-specific names.

```python
# thinharness/tracing.py
def annotate_model_request(span: _SpanAdapter, snapshot: ModelTraceSnapshot, *, capture_messages: bool) -> None:
    """Write opt-in model request content attributes."""
    if not capture_messages:
        return
    input_payload = _model_request_input(snapshot)
    span.set_attributes({
        "gen_ai.input.messages": serialize_attribute_value(_otel_input_messages(snapshot)),
        "gen_ai.prompt": serialize_attribute_value(input_payload),
        "langfuse.observation.input": serialize_attribute_value(input_payload),
        "thinharness.model.request.kind": snapshot.kind,
        "thinharness.output.mode_requested": snapshot.structured_output,
        "thinharness.model.notices": serialize_attribute_value(snapshot.notices),
    })

def annotate_model_span(span: _SpanAdapter, turn: Any, *, capture_messages: bool = False) -> None:
    """Add model response attributes to a span."""
    ...
    if capture_messages and text:
        span.set_attributes({
            "langfuse.observation.output": serialize_attribute_value({"text": text}),
            "gen_ai.completion": text,
            "gen_ai.output.messages": serialize_attribute_value(_otel_output_messages(turn)),
        })
```

Define the private mapping helpers in the same file:

```python
def _model_request_input(snapshot: ModelTraceSnapshot) -> Json | None:
    """Return the backend-compatible logical request payload."""
    if snapshot.kind in {"start", "resume"}:
        return {"prompt": snapshot.prompt}
    if snapshot.kind in {"tool_outputs", "output_retry_tool"}:
        return {"tool_outputs": snapshot.tool_outputs or []}
    if snapshot.kind == "correction":
        return {"correction": snapshot.prompt}
    return None
```

`_otel_input_messages(snapshot)` should return OTel-shaped message objects: start/resume/correction become a single `{"role": "user", "parts": [{"type": "text", "content": prompt}]}` message, tool continuations and `output_retry_tool` become `{"role": "tool", "parts": [{"type": "tool_result", "id": call_id, "content": output}]}` entries, and `instructions` stays only on the root span's `gen_ai.system_instructions` instead of being duplicated into every model span. Preserve `ToolOutput.output` as the raw JSON envelope string because that is exactly what the model saw; do not parse and re-render it for prettier backend display. Keep notices out of `gen_ai.input.messages` even when a provider puts notice text in the same user-role payload as tool results, because notices are harness metadata rather than the logical user/tool input being traced. `_otel_output_messages(turn)` should return one assistant message with text parts and tool-call parts, including tool call id/name/arguments when present.

Keep content capture behind `capture_messages`, matching OTel guidance that prompt and completion content are sensitive and should be opt-in. Keep `gen_ai.completion` for current tests and older backend fallback mapping, and document in the writer docstring why both `gen_ai.input.messages` and `gen_ai.prompt` are written. The module docstring should explicitly say input messages are constructed from `ModelTraceSnapshot`, never from provider payloads, because `providers.py` may already have appended notices to those payloads. It should also pin the OTel GenAI semantic convention retrieval date used for message shapes, for example `OTel GenAI semconv as of 2026-05-19: https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/`.

**Verify:** Extend `tests/test_tracing.py::test_harness_tracing_records_agent_model_and_tool_spans` to assert the first chat span has `gen_ai.input.messages` and backend-compatible input, the root agent span has `gen_ai.system_instructions` exactly once, and the final chat span has `gen_ai.output.messages`, `gen_ai.completion`, and backend-compatible output. Add a `capture_messages=False` regression asserting none of `langfuse.observation.input`, `langfuse.observation.output`, `langfuse.trace.input`, `langfuse.trace.output`, `gen_ai.input.messages`, `gen_ai.output.messages`, `gen_ai.system_instructions`, `gen_ai.prompt`, or `gen_ai.completion` appears on any span.

### 3. Wire request snapshots through the run loop
Change `advance_model()` in `thinharness/core.py` to accept a `ModelTraceSnapshot` and call `annotate_model_request()` immediately after the model span opens and before the provider request runs. Fold the existing `thinharness.model.notices.count` and `thinharness.model.notices.kinds` block into the new writer, because the serialized `thinharness.model.notices` list is more useful and avoids two competing notice encodings.

```python
# thinharness/core.py
async def advance_model(request, *, trace_snapshot: ModelTraceSnapshot, output_retry: bool = False) -> ModelTurn:
    """Run one provider request with limit, usage, and tracing ceremony."""
    ...
    with run_tracer.model(self.model) as model_span:
        annotate_model_request(
            model_span,
            trace_snapshot.with_notices(notices),
            capture_messages=bool(self.tracing and self.tracing.capture_messages),
        )
        advanced_turn = await request(notices)
```

Build trace snapshots at each call site:

```python
trace_snapshot=ModelTraceSnapshot(kind="start", prompt=effective_prompt, structured_output=_trace_output_mode(self.output_schema))
trace_snapshot=ModelTraceSnapshot(kind="resume", prompt=effective_prompt, structured_output=_trace_output_mode(self.output_schema))
trace_snapshot=ModelTraceSnapshot(kind="tool_outputs", tool_outputs=[{"call_id": item.call_id, "output": item.output} for item in tool_outputs], structured_output=_trace_output_mode(self.output_schema))
trace_snapshot=ModelTraceSnapshot(kind="correction", prompt=retry_message, structured_output=_trace_output_mode(self.output_schema))
trace_snapshot=ModelTraceSnapshot(kind="output_retry_tool", tool_outputs=[{"call_id": final_id, "output": retry_message}], structured_output=_trace_output_mode(self.output_schema))
```

Split agent-level content writing into start and successful-result phases. At agent span open, after `instructions = structured_instructions(...)` is available and before the first provider request, call `annotate_agent_start(...)` so failed runs still carry the raw input prompt and system instructions. At successful terminal paths, call `annotate_agent_result(...)` from a single finalize helper to write output-only fields.

```python
# core.py, inside Harness.run() after the agent span opens:
instructions = structured_instructions(self.system_instructions(), self.output_schema)
annotate_agent_start(
    agent_span,
    prompt=prompt,
    instructions=instructions,
    capture_messages=bool(self.tracing and self.tracing.capture_messages),
    top_level=not self._is_child_run,
)
```

For top-level runs, `annotate_agent_start()` writes `langfuse.trace.input` and `gen_ai.system_instructions`; for child runs it writes `langfuse.observation.input` only. This deliberate split means `langfuse.trace.input` is the raw caller prompt, before hook context injection, while the first model span's `gen_ai.input.messages` is the `effective_prompt` actually sent to the model after hooks. Add this distinction to the writer module docstring and the docs attribute table.

Consolidate result writing into a single finalize helper inside `Harness.run()` instead of leaving the current five `agent_span.set_attribute("gen_ai.completion", ...)` terminal branches in `core.py`. The helper should wrap `result = build_terminal_result(...)`, `annotate_agent_result(...)`, `attach_resume_state(...)`, `fire_run_end_once()`, and `return result` so the writer fires exactly once per successful terminal path.

```python
# core.py, nested inside Harness.run()
def _finalize(
    text: str,
    *,
    output: Any | None = None,
    finalized_via_output_tool_value: bool = False,
) -> HarnessResult:
    """Run terminal bookkeeping for one successful run."""
    # Intentionally captures agent_span and active_session from the enclosing run scope.
    nonlocal result, finalized_via_output_tool
    if finalized_via_output_tool_value:
        finalized_via_output_tool = True
    result = build_terminal_result(text, output)
    annotate_agent_result(
        agent_span,
        result=result,
        output_schema=self.output_schema,
        capture_messages=bool(self.tracing and self.tracing.capture_messages),
        top_level=not self._is_child_run,
    )
    attach_resume_state(active_session)
    fire_run_end_once()
    return result
```

The current inline `turn.finalized_output_mode = ...` assignments can stay at the caller sites before `_finalize(...)`; only `finalized_via_output_tool` needs to flow into `_finalize(...)` so `attach_resume_state(...)` sees the correct nonlocal state.

```python
# thinharness/tracing.py
def annotate_agent_start(
    span: _SpanAdapter,
    *,
    prompt: str,
    instructions: str,
    capture_messages: bool,
    top_level: bool,
) -> None:
    """Write opt-in agent input attributes before any provider work runs."""
    if not capture_messages:
        return
    if top_level:
        span.set_attributes({
            "langfuse.trace.input": prompt,
            "gen_ai.system_instructions": serialize_attribute_value([{"type": "text", "content": instructions}]),
        })
    else:
        span.set_attribute("langfuse.observation.input", prompt)

def annotate_agent_result(
    span: _SpanAdapter,
    *,
    result: HarnessResult,
    output_schema: OutputSchema | None,
    capture_messages: bool,
    top_level: bool,
) -> None:
    """Write opt-in agent trace or observation output attributes."""
    if not capture_messages:
        return
    if output_schema is not None and result.output is not None:
        output_payload = output_schema.dump(result.output)
    else:
        output_payload = None
    output = {"text": result.text, "output": output_payload, "stop_reason": result.stop_reason}
    if top_level:
        span.set_attributes({
            "langfuse.trace.output": serialize_attribute_value(output),
            "gen_ai.completion": result.text,
        })
    else:
        span.set_attribute("langfuse.observation.output", serialize_attribute_value(output))
```

Only the topmost agent span may write `langfuse.trace.*`; child agent spans write observation input/output instead so a subagent cannot overwrite the root trace's input/output in Langfuse. Do not put this runtime flag on user-facing `TracingOptions`. Add an internal `_is_child_run: bool = False` constructor argument on `Harness`, set it from `build_child_harness()`, and pass `top_level=not self._is_child_run` into `annotate_agent_result()`.

Error paths should continue to mark spans as errors and record exceptions without writing `langfuse.trace.output`; a failed run has no `HarnessResult` payload to serialize. Because `annotate_agent_start()` runs before provider work, failed top-level runs still have `langfuse.trace.input` and `gen_ai.system_instructions` for debugging. Removing `thinharness.model.notices.count` and `thinharness.model.notices.kinds` is safe because no current tests assert those names, and the serialized `thinharness.model.notices` list replaces them.

**Verify:** Add regression tests for all request kinds: initial prompt with notices, normal tool continuation with notices, structured-output corrective user message, structured-output tool retry via `output_retry_tool`, and resumed prompt. The test should assert that notice text is present only under a notice/metadata field and does not overwrite the main prompt/tool input field. Add a subagent tracing regression asserting the child `invoke_agent` span has `langfuse.observation.input/output` and does not have `langfuse.trace.input/output`. Extend the concurrent subagent fanout tracing test with `capture_messages=True` so each child observation input is its own task and parent input does not leak into child spans. Add a provider-error regression asserting a failed top-level run has `langfuse.trace.input` and `gen_ai.system_instructions`, does not have `langfuse.trace.output`, and is marked error.

### 4. Add a Langfuse OTLP validation helper
Add `create_langfuse_tracing()` in `thinharness/tracing.py` and export it from `thinharness/__init__.py`. It should be a small wrapper around `create_otlp_tracing()` that uses the same Basic auth pattern as the working `pi-agent-sdk` manual trace script while following current Langfuse environment conventions.

```python
# thinharness/tracing.py
def create_langfuse_tracing(
    *,
    service_name: str,
    public_key: str | None = None,
    secret_key: str | None = None,
    host: str | None = None,
    legacy_ingestion: bool = False,
    tracer_name: str = "thinharness",
) -> OtlpTracing:
    """Create a Langfuse OTLP tracer provider."""
    public_key = public_key or os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = secret_key or os.getenv("LANGFUSE_SECRET_KEY")
    if not public_key or not secret_key:
        raise RuntimeError("LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are required for create_langfuse_tracing")
    auth = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
    langfuse_host = host or os.getenv("LANGFUSE_HOST") or "https://us.cloud.langfuse.com"
    headers = {"Authorization": f"Basic {auth}"}
    if legacy_ingestion:
        headers["x-langfuse-ingestion-version"] = "4"
    return create_otlp_tracing(
        service_name=service_name,
        endpoint=langfuse_host.rstrip("/") + "/api/public/otel/v1/traces",
        headers=headers,
        tracer_name=tracer_name,
    )
```

Verify the current Langfuse OTLP docs before implementation to confirm whether `x-langfuse-ingestion-version: 4` is still recommended or merely tolerated. Keep it behind `legacy_ingestion` unless the docs still require it. Document in the helper docstring that this helper is a backend-specific convenience for live validation and that the primary tracing contract remains standard OTel GenAI attributes. Also document that the helper reads `LANGFUSE_HOST`, `LANGFUSE_PUBLIC_KEY`, and `LANGFUSE_SECRET_KEY`, does not read the local `.m` file itself, and defaults to the US cloud host unless callers set `LANGFUSE_HOST` for another region. Document `legacy_ingestion=True` as an older/self-hosted deployment escape hatch to try only if Langfuse returns 400/422 without the `x-langfuse-ingestion-version: 4` header.

**Verify:** Add a unit test that monkeypatches Langfuse env vars, calls `create_langfuse_tracing()` with a fake/mocked `create_otlp_tracing()` boundary if needed, and asserts `LANGFUSE_HOST` derives the OTLP endpoint and `legacy_ingestion=True` adds the v4 header.

### 5. Update docs and e2e validation
Update `README.md` and `docs/architecture.md` so tracing no longer sounds like span structure only. Include a short OTel/Langfuse example that uses `create_langfuse_tracing()`, `TracingOptions(capture_messages=True, capture_tool_args=True, capture_tool_results=True)`, `force_flush()`, and `shutdown()`.

```python
tracing = create_langfuse_tracing(service_name="thinharness-dev")
harness = Harness(
    HarnessConfig(root=".", model="openrouter:anthropic/claude-haiku-4.5"),
    tracing=TracingOptions(tracer=tracing.tracer, agent_name="thin-agent", capture_messages=True),
)
try:
    result = harness.run_sync("Inspect the repo and summarize the tracing setup.")
finally:
    tracing.force_flush()
    tracing.shutdown()
```

Add a concrete tracing e2e script named `e2e/langfuse_tracing_journey.py` that imports `create_langfuse_tracing` from the public `thinharness` package and exits early with a clear message if provider or Langfuse env vars are missing, matching the existing journey-script pattern rather than pytest skip semantics. The scenario should force at least one subagent call and make the subagent do multiple observable steps, for example: create `outputs/subagent-draft.md`, create `outputs/subagent-notes.md`, read the draft back, edit or replace the draft with a revised version, then return a concise summary to the parent. Configure the parent harness with `builtin_tools=[]` and only the `subagent` tool so the parent cannot do the file work directly; configure the child subagent with explicit file tools such as `builtin_tools=["read", "write", "edit", "list"]` or the current equivalent tool names. Enable tracing with `capture_messages=True`, `capture_tool_args=True`, and `capture_tool_results=True` so the live trace exercises root input/output, generation input/output, subagent nesting, and multiple tool observations.

Add a small docs table in `docs/architecture.md` listing every content attribute the writer sets, when it is set, and what it serves: `gen_ai.system_instructions`, `gen_ai.input.messages`, `gen_ai.output.messages`, `gen_ai.prompt`, `gen_ai.completion`, `gen_ai.tool.call.arguments`, `gen_ai.tool.call.result`, `langfuse.observation.input`, `langfuse.observation.output`, `langfuse.trace.input`, `langfuse.trace.output`, `thinharness.model.request.kind`, `thinharness.output.mode_requested`, and `thinharness.model.notices`. Pin the OTel GenAI semantic convention retrieval date used for the message shapes, and distinguish `thinharness.output.mode_requested` on request spans from the existing `thinharness.output.mode` finalized-mode attribute.

**Verify:** Run `uv run pytest tests/test_tracing.py tests/test_providers.py tests/test_subagents.py tests/test_tool_retry.py`, `uv run ruff check thinharness tests`, and `uv run pyright`. For live validation, run the e2e tracing script with provider and Langfuse env vars loaded and confirm Langfuse shows a root trace with nested generation/tool/subagent observations, the subagent span is nested under the parent `subagent` tool span, the child run includes multiple file-tool observations, each generation observation has non-empty input/output, and tool observations visibly show input and output. If Langfuse does not show tool input/output from `gen_ai.tool.call.arguments/result`, mirroring tool args/results to `langfuse.observation.input/output` becomes required before closing this work.

## Test Checklist
Use this as the implementation checklist for the tests implied above:

- `serialize_attribute_value(None) is None` so absent content stays absent.
- `ModelTraceSnapshot(kind="tool_outputs", ...)` writer test asserts `gen_ai.input.messages` contains the raw tool output and `thinharness.model.notices` contains notice text.
- First chat span has `gen_ai.input.messages`; `gen_ai.system_instructions` appears only on the root agent span.
- Final chat span has `gen_ai.output.messages`, `gen_ai.completion`, and `langfuse.observation.output`.
- `capture_messages=False` regression asserts no `langfuse.observation.*`, `langfuse.trace.*`, `gen_ai.input.messages`, `gen_ai.output.messages`, `gen_ai.system_instructions`, `gen_ai.prompt`, or `gen_ai.completion` appears on any span.
- All five snapshot kinds are exercised: `start`, `resume`, `tool_outputs`, `correction`, and `output_retry_tool`; notice text must stay under `thinharness.model.notices` and must not overwrite `gen_ai.input.messages`.
- Subagent regression asserts child `invoke_agent` spans have `langfuse.observation.input/output` and do not have `langfuse.trace.input/output`.
- Concurrent subagent fanout with `capture_messages=True` asserts each child observation input is that child's own task and parent input does not leak into child spans.
- Provider-error regression asserts failed top-level runs still have `langfuse.trace.input` and `gen_ai.system_instructions`, omit `langfuse.trace.output`, and mark the span as failed.
- `create_langfuse_tracing()` env test asserts `LANGFUSE_HOST=https://eu.cloud.langfuse.com` derives the EU OTLP endpoint, missing keys raise the documented `RuntimeError`, and `legacy_ingestion=True` adds the v4 header while `False` omits it.
- No `FakeTracer` changes should be required because `tests/fakes.py::FakeSpan.set_attributes()` and `set_attribute()` already accept arbitrary string-valued attributes.

Implementation will need `dataclasses.replace` and `dataclasses.asdict` in `thinharness/tracing.py`, plus `Literal`. If `annotate_agent_result()` imports `OutputSchema` for typing, guard it with `TYPE_CHECKING` or use a string annotation to avoid circular-import risk.

## Considerations
Use `capture_messages` as the single opt-in for prompt/input/output content rather than adding another config flag. That keeps the public API small and matches the current meaning of "messages."

Do not serialize full provider payloads as the primary observation input. Provider payloads are useful metadata later, but using them as input is how notice-only continuations and provider-specific wrappers can hide the actual logical prompt or tool outputs.

Keep OTel `gen_ai.*` attributes as the primary portability contract and write backend-specific compatibility attributes only where they make the same trace easier to inspect in a known backend. Langfuse is the first live validation target, not the tracing model.

Root trace input/output, child agent observation input/output, and model observation input/output are different. The top-level root trace should show the caller prompt and final harness result; child agent spans should show that subagent's task and result without writing trace-level attributes; each model span should show the exact logical request for that model call and that call's normalized response.

Tool spans already have opt-in argument/result capture. Leave those fields on `gen_ai.tool.call.arguments` and `gen_ai.tool.call.result`, and optionally mirror to `langfuse.observation.input/output` only if Langfuse does not surface tool observations clearly after the generation writer lands.

Subagent nesting should remain context-driven. The plan should not change `_CURRENT_TOOL_CALL`, `_child_tracing()`, or the concurrent subagent tests except to assert the new attributes appear on child model spans too.
