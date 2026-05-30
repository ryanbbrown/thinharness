# Plan: Structured Output for `ParallelLlmTool`

## Goal

Add validated structured output to custom `ParallelLlmTool` without duplicating the structured-output orchestration currently embedded in `Harness.run()`.

The built-in `parallel_llm` remains generic/text-only. Custom tools should support:

```python
ParallelLlmTool(
    name="parallel_extract",
    model="openai:gpt-5.2-mini",
    output_type=InvoiceFields,
    output_mode="auto",
    output_retries=1,
).spec()
```

`output_mode` should mean the same thing as `HarnessConfig.output_mode`: `auto`, `native`, `tool`, `prompted`, or `text`, resolved against the selected model's capabilities.

## Current State

- `ParallelLlmTool(...).spec()` exists and is a normal custom `ToolSpec`.
- Built-in `parallel_llm` is still opt-in through `builtin_tools=["parallel_llm"]`.
- The built-in model/temperature are host-owned:
  - `HarnessConfig.builtin_parallel_llm_model`
  - `HarnessConfig.builtin_parallel_llm_temperature`
- Model-facing `ParallelLlmArgs` intentionally does not include model or temperature.
- `ParallelLlmTool` currently returns raw text per result entry.
- A direct duplicated structured-output implementation was considered and should be avoided.

## Design

Extract reusable structured-output orchestration into `thinharness/output.py`, then use it from both `Harness.run()` and `ParallelLlmTool`.

Do not maintain a model registry for temperature support or structured-output quirks. If a caller passes temperature or requests a mode a provider/model rejects, let the provider/capability error surface.

## Suggested Extraction

Add helpers to `output.py` around the existing `OutputSchema` primitives:

```python
def resolve_output_schema_for_model(
    model: Model,
    output_type: OutputSpec | None,
    output_mode: OutputMode,
) -> OutputSchema | None:
    ...

def structured_instructions(
    instructions: str,
    output_schema: OutputSchema | None,
) -> str:
    ...

def validate_turn_output(
    turn: ModelTurn,
    output_schema: OutputSchema | None,
) -> Any:
    ...

def structured_retry_prompt(prompt: str, error: OutputValidationError) -> str:
    ...
```

`resolve_output_schema_for_model` should contain the logic currently in `Harness._build_output_schema()`:

- `None` output type returns `None`.
- Resolve wrapper specs with `resolve_output_spec`.
- `auto` uses `model.capabilities.default_structured_output_mode`.
- Reject unsupported native/tool modes using `ModelCapabilities`.
- Build and return `OutputSchema`.

`validate_turn_output` should contain the common "turn -> validated value" logic:

- `None` schema: return `turn.text`.
- `text`: require no tool calls, validate text.
- `tool`: require exactly one `final_result` tool call and no siblings, validate tool arguments.
- `native` / `prompted`: require no tool calls, validate text.

`structured_instructions` should append prompted schema instructions only when mode is `prompted`.

Keep provider request construction outside `output.py`; callers use:

- `tools=output_schema.synthetic_tools() if output_schema else []`
- `structured_output=output_schema.structured_output_request() if output_schema else None`

## Harness Refactor

Refactor `Harness._build_output_schema()` to call `resolve_output_schema_for_model(...)`.

Refactor the repeated turn-finalization checks in `Harness.run()` only if it stays clear and low-risk. A minimal first pass can leave the run loop mostly intact and just share resolution helpers. Prefer exact behavior preservation over a large run-loop rewrite.

## Parallel Tool Implementation

Add to `ParallelLlmTool.__init__`:

```python
output_type: OutputSpec | None = None
output_mode: OutputMode = "auto"
output_retries: int = 1
```

Validation:

- `output_retries >= 0`.
- The built-in `create_parallel_llm_tool(parent)` should not pass `output_type`, so built-in stays text-only.

Per prompt:

1. Resolve `output_schema` once after resolving the batch model.
2. Build per-call instructions with `structured_instructions(args.system or "", output_schema)`.
3. Each provider attempt creates a fresh session and calls `start(...)` with:
   - `tools=output_schema.synthetic_tools()` when structured tool mode is active, else `[]`.
   - `structured_output=output_schema.structured_output_request()` for native mode, else `None`.
4. Validate with `validate_turn_output(...)`.
5. Success entry:
   - Text mode / no schema: `{"index": i, "ok": true, "result": turn.text}`
   - Structured schema: dump validated value to JSON-compatible Python using `output_schema.adapter.dump_python(value, mode="json")`.
6. Validation failure:
   - Retry up to `output_retries` with a fresh one-shot prompt containing validation feedback.
   - Do not hold the concurrency semaphore while preparing retries or sleeping provider backoff.
   - If exhausted, return sparse failure entry: `{"index": i, "ok": false, "error": "output validation failed: ..."}`

Provider retry and structured-output retry are separate:

- `max_attempts` handles retryable provider failures for one prompt attempt.
- `output_retries` handles invalid model output against the schema.
- Both increment `model_requests` because every provider request is real traffic.

## Tests

Extend `tests/test_parallel_llm.py`:

- Custom `ParallelLlmTool(output_type=Model, output_mode="prompted")` parses valid JSON text into JSON-compatible result values.
- Invalid structured text returns a failure entry when `output_retries=0`.
- Invalid then valid output succeeds when `output_retries=1`; assert fresh session/provider request count.
- Tool mode accepts a `final_result` tool call and rejects text/no tool call.
- Native mode sends `structured_output` to `session.start(...)`.
- `auto` resolves against model capabilities.
- Built-in `parallel_llm` remains text-only and does not expose output schema config.

Provider/session fakes may need to record `structured_output`, `tools`, and `instructions`.

## E2E

Update `e2e/parallel_llm_agent_journey.py` after structured support lands:

- Keep built-in `parallel_llm` for simple string outputs.
- Use custom `ParallelLlmTool(name="parallel_json", output_type=...)` for validated structured outputs.
- Assert result entries already contain parsed JSON-compatible dicts, not strings that the e2e script parses manually.

The direct tool e2e can optionally add one provider-specific structured smoke test, but avoid making all providers run structured output modes if it becomes slow/flaky.

## Docs

Update:

- `docs/docs.md`: show `output_type` / `output_mode` on custom `ParallelLlmTool`.
- `docs/architecture.md`: note shared structured-output helpers in `output.py`.
- `docs/decisions.md`: replace "validated structured parallel output is deferred" with the final decision once implemented.

