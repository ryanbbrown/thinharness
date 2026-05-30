# Plan — Structured output

## Goal

`Harness(HarnessConfig(output_type=MyModel)).run("...")` returns a `HarnessResult` with `output: MyModel` populated. Validation, retries, and markdown-fence stripping handled internally. All three strategies supported (tool, native, prompted), chosen automatically per provider with user override.

## Strategy summary

| Strategy | How it works | Provider support |
|---|---|---|
| **tool** | Register synthetic `final_result(...)` function tool whose schema is the output schema; intercept the call as the result | All three current adapters |
| **native** | Provider's structured-output knob: OpenAI Responses `text.format`, OpenRouter `response_format` | OpenAI ✅, OpenRouter ⚠️ depends on model, Anthropic ❌ |
| **prompted** | Render schema into system instructions, parse + fence-strip + validate text on end-turn | All — even models without function calling |

## Implementation sequencing

Keep the behavior changes reviewable by landing them in this order:

1. Extract the repeated model-call ceremony in `Harness.run()` into an `_advance_model(...)` helper that performs `check_model_limit()`, opens the model span, awaits the provider call, increments `usage.model_requests`, annotates provider exceptions, and runs `annotate_model_span(...)`.
2. Add shared types and protocol slots with no behavior wired yet: `StopReason` values, `UnexpectedModelBehavior`, `StructuredOutputRequest`, `ModelSession.continue_with_user_message`, fake-session updates, and provider capability fields.
3. Vendor and trim `_output.py` and dependencies.
4. Replace hook-filter mismatch warnings with strict validation against currently registered harness tools.
5. Wire capabilities, schema construction, synthetic tool schemas, native payload translation, retries, subagent serialization, tests, and docs.

## Step 1 — Vendor `_output.py` and dependencies

Copy from `vendor/pydantic-ai/pydantic_ai_slim/pydantic_ai/` into `thinharness/`:

| Source | Destination | Notes |
|---|---|---|
| `_output.py` | `thinharness/_output.py` | The big one. Copy wholesale, then trim. |
| `_function_schema.py` | `thinharness/_function_schema.py` | Used by `_output.py` for function-style outputs |
| `_json_schema.py` | `thinharness/_json_schema.py` | `GenerateToolJsonSchema` and friends |
| Selected `_utils.py` helpers | `thinharness/_pydantic_utils.py` | `strip_markdown_fences`, `check_object_json_schema`, `merge_json_schema_defs` only |
| `output.py` | `thinharness/output.py` | Public surface: `OutputSpec`, `NativeOutput`, `PromptedOutput`, `ToolStructuredOutput`, `TextOutput` markers |
| `exceptions.py` selected | merge into `thinharness/core.py` | `ModelRetry`, `ToolRetryError`, `UnexpectedModelBehavior` |
| `_messages.py` selected | new `thinharness/_messages.py` | `RetryPromptPart`, `ToolReturnPart` only |
| `_run_context.py` minimal | `thinharness/_run_context.py` | Strip dependency-injection — keep `tool_name`, `retry`, `model`, `usage` |

**Trim in the same commit so the diff is reviewable:**

- Drop `BinaryImage`, `DeferredToolRequests`, `BuiltinToolCallEvent`, `ToolsetTool`, `OutputToolset` glue
- Drop the four output hook variants (`before_output_validate`, `after_output_validate`, `before_output_process`, `after_output_process`) — keep only the validate-and-retry path
- Drop streaming output processors (`StreamedOutput*`) — comes back when we add streaming
- Drop pydantic-graph state (`ctx.state.consume_output_retry()`) — replace with a simple counter on `RunUsage`

Keep: `OutputObjectDefinition`, `BaseOutputProcessor`, `ObjectOutputProcessor`, `UnionOutputProcessor`, `PlainTextOutputProcessor`, `_flatten_output_spec`, `OutputSchema` + the three subclasses, `_make_retry_prompt`, `strip_markdown_fences`. Target ~600 lines after trim (down from 1633).

The trim must preserve these output shapes by name: `BaseModel`, dataclass, `TypedDict`, `list[T]`, and `Union[A, B]`. The `list[T]` / non-object wrapping behavior depends on `ObjectOutputProcessor.__init__`, `_flatten_output_spec`, and `_function_schema.function_schema`; do not trim those paths away just because the module still compiles.

After trimming, run `uv run python -m compileall thinharness tests` before wiring behavior into the run loop. Keep this first structured-output commit import-clean with no run-loop changes so accidental dependencies on pydantic-ai graph, streaming, message, or toolset internals are caught early.

Vendor policy: keep `vendor/pydantic-ai` as the upstream source reference, but runtime imports must use the trimmed local `thinharness/*` files. Add a short vendor note next to the trimmed files recording the upstream pydantic-ai commit hash, copied source files, and trim list so future re-syncs are mechanical. This is a maintained fork of selected files, not a runtime dependency on pydantic-ai.

## Step 2 — Capability flags on providers

In `providers.py` add to each `Model` class:

```python
class ModelCapabilities(BaseModel):
    supports_json_schema_output: bool = False
    supports_tools: bool = True
    permissive_native_override: bool = False
    default_structured_output_mode: Literal["native", "tool", "prompted"] = "tool"
```

| Adapter | native | tools | default mode |
|---|---|---|---|
| `OpenAIResponsesModel` | ✅ | ✅ | `native` |
| `AnthropicMessagesModel` | ❌ | ✅ | `tool` |
| `OpenRouterModel` | ⚠️ depends on underlying model | ✅ | `tool` (safest; `permissive_native_override=True`) |

Add `capabilities: ModelCapabilities` as a class attribute. Extend the `Model` Protocol, but make harness code tolerant of custom/fake models without the attribute:

```python
capabilities = getattr(self.model, "capabilities", ModelCapabilities())
```

OpenRouter native mode is permissive in v1: `auto` chooses `tool`, but explicit `output_mode="native"` is allowed through `permissive_native_override=True`, and provider/model rejection surfaces as `ProviderError` at request time. Anthropic native mode remains a construction-time validation error because the adapter has no native structured-output API.

Add the provider-neutral native request object in `providers.py`:

```python
@dataclass
class StructuredOutputRequest:
    """Provider-neutral structured-output request metadata."""

    name: str
    schema: Json
    strict: bool = True
    description: str | None = None
```

Adapter translation:

- OpenAI Responses maps this to `text.format`.
- OpenRouter maps this to `response_format = {"type": "json_schema", "json_schema": ...}`.
- Anthropic raises if it receives a non-`None` request.

`strict` is not blindly `True` for every schema. Detect strict compatibility from the generated JSON schema using the pydantic-ai output/schema helpers. If a schema is incompatible with OpenAI strict structured output, either build the request with `strict=False` when the provider supports that fallback or raise a clear build-time error before the provider returns a confusing 400.

## Step 3 — `HarnessConfig` surface

```python
output_type: Any | None = None                  # OutputSpec-style; OutputSchema validates supported forms
output_mode: Literal["auto", "native", "tool", "prompted"] = "auto"
output_retries: int = 1                         # corrective retry requests after the first invalid output
```

Do not validate model capabilities in a `HarnessConfig` validator. `HarnessConfig` only has a model string and cannot see an injected `model=` object or OpenRouter's underlying model support. `output_type=None` keeps current free-form-text behavior.

`output_retries` means corrective retry requests after the first invalid output, not total validation attempts:

- `output_retries=0` fails on the first invalid output.
- `output_retries=1` permits one invalid output and one corrective model request.

The default is `1` because one corrective request catches common schema-shape mistakes without hiding a consistently incompatible schema or prompt.

Add the same explicit fields to named `SubAgentConfig`:

```python
output_type: Any | None = None
output_mode: Literal["auto", "native", "tool", "prompted"] = "auto"
output_retries: int = 1
```

Parent structured-output config is never inherited by subagents. Only a named/custom `SubAgentConfig` can opt into structured output by setting `output_type` directly. The framework default subagent always runs with `output_type=None`.

Update `build_child_harness(...)` so the child config always sets:

- `output_type=config.output_type` for named subagents, otherwise `None`
- `output_mode=config.output_mode` for named subagents, otherwise `"auto"`
- `output_retries=config.output_retries` for named subagents, otherwise the default

Do not copy `parent_config.output_type`, `parent_config.output_mode`, or `parent_config.output_retries` into child harnesses unless they came from the child subagent's own explicit config.

## Step 4 — Wire `OutputSchema` into `Harness.__init__`

After model construction:

```python
if config.output_type is not None:
    mode = config.output_mode
    if mode == "auto":
        mode = capabilities.default_structured_output_mode
    if mode == "native" and not capabilities.supports_json_schema_output:
        if not capabilities.permissive_native_override:
            raise ValueError(f"{self.model.provider.name} does not support native structured output")
    if mode == "tool" and not capabilities.supports_tools:
        raise ValueError(f"{self.model.provider.name} does not support tool structured output")
    self.output_schema = OutputSchema.build(
        output_type=config.output_type,
        mode=mode,
    )
else:
    self.output_schema = None
```

Also reserve the synthetic `final_result` name only when `output_type` is set. If any user-provided or selected built-in tool has the same final tool name, raise during harness construction before the run starts.

Apply the same reservation when a named `SubAgentConfig.output_type` is set: the subagent's own explicit tools or selected built-ins cannot include `final_result`.

Expose synthetic tools through `OutputSchema.synthetic_tools() -> list[Json]`, returning `[final_result_schema]` only for tool mode and `[]` otherwise. Do not reference `OutputToolset` or a `toolset` attribute; Step 1 removes those abstractions. The synthetic schema must use the same neutral JSON tool schema shape returned by `ToolSpec.response_tool()`; provider adapters remain responsible for translating that neutral shape into OpenAI/OpenRouter/Anthropic request payloads.

Expose `OutputSchema.dump(value) -> str`, delegating to the same Pydantic `TypeAdapter` / output adapter that validated the value. This is the canonical serializer for `result.output` at text boundaries such as subagent tool results, docs examples, JSONL/result export, and logs.

## Step 5 — Per-strategy plumbing in the loop

Touch `core.py:run` at three points:

**(a) Before `session.start`** — apply the strategy's effect on the request:

```python
extra_instructions = ""
structured_output: StructuredOutputRequest | None = None

if self.output_schema is not None:
    if self.output_schema.mode == "native":
        structured_output = self.output_schema.structured_output_request()
    elif self.output_schema.mode == "prompted":
        extra_instructions = self.output_schema.build_instructions(...)
```

Compose prompted-mode instructions at the run call site without mutating `system_instructions()`:

```python
instructions = self.system_instructions()
if extra_instructions:
    instructions = f"{instructions}\n\n{extra_instructions}"
```

Add an optional `structured_output: StructuredOutputRequest | None = None` kwarg through session start/continue request construction. Provider adapters own the native payload translation: OpenAI maps to `text.format`, OpenRouter maps to `response_format`, and Anthropic raises if non-`None` reaches it. The harness should not pass raw JSON schema as a provider-specific `response_format`.

Append synthetic tool schemas through `Harness.tool_schemas()` when `self.output_schema` is in tool mode, rather than threading `extra_tools` separately through every run-loop call site. This keeps `session.start(...)` and all `session.continue_with_tools(...)` requests consistent. Keep the `final_result` reservation check next to this mechanism.

For native mode, define `extra_body` precedence explicitly inside adapters: apply `extra_body` first, then apply harness-injected structured-output fields. Harness structured output wins over `extra_body["text"]` or `extra_body["response_format"]` collisions.

`output_type=str` / `TextOutput` is a text-output path, not a structured JSON path. It skips synthetic tools, native schema payloads, and prompted schema instructions; on final text, set `result.output = result.text`.

**(b) After each `turn`** — check for the synthetic `final_result` tool call (tool mode) before normal tool dispatch:

```python
if self.output_schema and self.output_schema.mode == "tool":
    finals = [c for c in turn.tool_calls if c.name == "final_result"]
    if finals:
        if len(finals) > 1 or len(turn.tool_calls) > 1:
            raise UnexpectedModelBehavior("final_result must be the only tool call in its turn")
        return self._validate_and_finalize(finals[0].arguments)
```

`_validate_and_finalize` runs the schema's text-or-args processor, catches `ValidationError`, returns the populated `HarnessResult` or raises `ToolRetryError`.

Intercept `final_result` before tool accounting and normal tool dispatch. A successful or invalid synthetic final answer must not increment `usage.tool_calls` and must not append a `tool_call_record`.

If `final_result` appears with any sibling tool call, run zero tools and fail immediately with `UnexpectedModelBehavior`. This check must happen before `_execute_tool_batch(...)`, so parallel tool execution cannot leak side effects from sibling calls.

Structured finalization is not a real tool execution from the hook system's perspective. `before_tool_call`, `after_tool_call`, and `_CURRENT_TOOL_CALL` do not fire for `final_result`. `final_result` is not part of `self.tools`, is not hookable, and should fail hook-filter validation like any other unknown tool name.

When annotating the model span for a turn that finalizes through `final_result`, add an attribute such as `thinharness.output.mode = "tool"` or `gen_ai.output.finalized = true` so traces do not look like the harness dropped a tool call.

Preserve any assistant text emitted alongside a valid `final_result` in `result.text`; do not blank it. Most providers/models will return `""`, but if a provider gives text such as `"Done"` in the same turn, keep it verbatim.

**(c) On end-turn (no tool calls) for native/prompted** — validate the text:

```python
if not turn.tool_calls and self.output_schema and self.output_schema.mode in ("native", "prompted"):
    return self._validate_and_finalize(turn.text)
```

For tool mode, a no-tool text-only end turn is an output validation failure, not a successful unstructured result. Send a corrective retry prompt telling the model it must call `final_result(...)` to deliver the answer; this counts against `output_retries`.

Add symmetric trace attributes for native and prompted finalization turns, such as `thinharness.output.mode = "native"` / `"prompted"` and `gen_ai.output.finalized = true`.

## Step 6 — Retry path

When `_validate_and_finalize` raises `ToolRetryError`:

```python
usage.output_retries += 1
if usage.output_retries > self.config.output_retries:
    stop_reason = "output_validation_failed"
    raise HarnessError("output validation exceeded output_retries")

if self.output_schema.mode == "tool":
    outputs = [ToolOutput(call_id=final.id, output=retry_error.tool_retry.model_response_str())]
    turn = session.continue_with_tools(outputs, ...)
else:
    # native/prompted: inject as a user message
    turn = session.continue_with_user_message(retry_error.message, ...)
```

Output retries must use the same provider-request path as normal model calls:

1. `check_model_limit()`
2. `with run_tracer.model(self.model) as model_span`
3. provider call
4. `usage.model_requests += 1`
5. provider exception annotation
6. `annotate_model_span(...)`

This applies to tool-mode retry continuations and native/prompted `continue_with_user_message(...)` calls. Do not add a side path that can bypass `max_model_requests`, undercount usage, or disappear from traces.

Add a new `ModelSession.continue_with_user_message` method on the adapters and update `tests/fakes.py` / ad hoc fake sessions. It mirrors `continue_with_tools` but adds a corrective user message instead of tool outputs.

`continue_with_user_message(...)` is only used for native/prompted retry. Tool-mode retry must continue with a `ToolOutput` so Anthropic-style tool-use/tool-result pairing stays valid; add an assertion or adapter-level guard so tool-mode code cannot accidentally call the user-message path.

Add `output_retries: int = 0` to `RunUsage` and `"output_validation_failed"` to `StopReason`.

Extend `StopReason` with `"unexpected_model_behavior"` as well. `UnexpectedModelBehavior` should subclass `HarnessError`; add an explicit `except UnexpectedModelBehavior` branch before the generic `except HarnessError` branch so `run_end` and traces receive `stop_reason="unexpected_model_behavior"`.

If retry exhaustion occurs, set `stop_reason = "output_validation_failed"` before raising `HarnessError("output validation exceeded output_retries")`. The existing `run_end` helper should then fire exactly once with `stop_reason="output_validation_failed"` and `usage.output_retries` populated.

Output-retry continuations are cancellable like any other model call. If cancellation lands between an invalid output and the corrective request, the `_advance_model(...)` path should surface the cancellation cleanly; `usage.output_retries` should reflect only retry requests actually attempted.

## Step 6.5 — Hook filter validation

Replace `_warn_unmatched_hook_filters()` with strict validation against the harness's currently registered tools and known subagents. Hooks must be added after the tools they reference. If a hook filter names a tool that is not currently registered, raise a clear `ValueError`; do this consistently rather than conditionally warning.

Construction order should be: register built-ins, skills, subagent tool, and custom `tools=...`; validate unique tool names; then build/validate the hook registry. If a future `add_hook()` API exists, it should validate immediately against the current tool set. If `add_tool()` remains after hooks are registered, it can re-run validation to catch any now-resolved or newly-invalid state, but structured output should not depend on deferred hook matching.

`final_result` is synthetic structured-output plumbing, not a registered harness tool. Do not add it to the valid hook-filter set. A hook filtered to `tools=["final_result"]` should fail validation.

## Step 7 — Result surface

```python
@dataclass
class HarnessResult:
    text: str
    output: Any | None = None      # populated when output_type was set
    # ... existing fields
```

For tool-mode structured output, `result.output` is the validated typed value and `result.text` is the assistant text from the same turn, usually `""`. Do not serialize the typed output back into `text`; callers should use `result.output` when structured output is enabled.

`result.output` is the validated object as typed. Do not pre-serialize it inside `HarnessResult`. Callers that need JSON should use the output schema / `TypeAdapter(...).dump_python(result.output)` path, or `model_dump()` for `BaseModel` values.

Named/custom subagents may opt into structured output explicitly through their own `SubAgentConfig.output_type`. Parent structured-output config is not inherited. When a child subagent returns:

```python
if result.output is not None:
    content = child.output_schema.dump(result.output)
else:
    content = result.text
```

The parent-facing subagent tool should return `content` and include metadata such as `{"structured_output": result.output is not None}`. Use the child harness's output schema / `TypeAdapter` serializer where available rather than raw `json.dumps(result.output)`.

Generic typing: keep `Any` for v1. Full `Generic[OutputT]` typing of `Harness`/`HarnessResult` is mechanically straightforward but adds noise — defer until users complain.

## Step 8 — Tests

New file `tests/test_structured_output.py`. Use the existing `FakeModel` pattern from `tests/fakes.py`:

- BaseModel output via tool mode: model emits `final_result` call → result populated
- BaseModel output via native mode: model emits text JSON, no fences → parsed
- BaseModel output via prompted mode: model emits ```` ```json ... ``` ```` → fence stripped, parsed
- TypedDict + dataclass + `list[Model]` outputs (the TypedDict-wrapping trick)
- Validation failure → retry with error message → success on second try
- `output_retries=0` → first invalid output raises without a corrective request
- `output_retries=1` → one invalid output and one corrective model request are allowed
- Validation failure → exhaust `output_retries` → `HarnessError`, `stop_reason="output_validation_failed"`
- Exhausted validation fires `run_end` exactly once with `stop_reason="output_validation_failed"` and `usage.output_retries` populated
- Anthropic + `output_mode="native"` → raises at `Harness.__init__`
- OpenRouter + `output_mode="native"` → construction succeeds; provider rejection is a `ProviderError`
- Tool name collision with `final_result` while `output_type` is set → raises at `Harness.__init__`
- Tool-mode `final_result` mixed with ordinary tool calls or repeated in one turn → `UnexpectedModelBehavior`
- Tool-mode `final_result` mixed with sibling tool calls → no sibling tool dispatch and no `before_tool_call` hook fires
- Tool-mode model returns text-only end turn without `final_result` → corrective retry, then success or exhausted `output_retries`
- Tool-mode `final_result` populates `output` without incrementing `usage.tool_calls` or appending a `tool_call_record`
- Tool-mode `final_result` preserves same-turn assistant text in `result.text`
- `output_type=str` / `TextOutput` → `result.output == result.text`, no synthetic tool registered, no native payload sent, no prompted schema block
- Native/prompted validation retries are counted in `usage.model_requests` and traced as normal model spans
- Fake sessions implement `continue_with_user_message`
- `continue_with_user_message` is never used for tool-mode retry
- Cancellation between invalid output and corrective retry cleanly cancels; `usage.output_retries` reflects only attempted retries
- `extra_body` cannot silently override native structured-output payload fields
- Hook filters referencing unknown tools raise consistently after tools are registered; `final_result` is unknown/not hookable
- Tool-mode, native-mode, and prompted-mode finalization turns have trace attributes marking structured-output finalization
- Strict-incompatible schema in OpenAI native mode gets either `strict=False` when supported or a clear build-time error
- Parent `output_type` is not inherited by the framework default subagent or named subagents
- Named subagent with explicit `output_type` serializes `result.output` into the parent-facing subagent tool result and marks metadata with `structured_output=True`
- Named subagent without explicit `output_type` keeps returning `result.text`
- Named subagent finalizes via tool mode while parent runs native mode → both work independently, no config bleed

Port a handful of pydantic-ai's own structured-output tests (their `tests/test_agent.py` cases for fence stripping and union output) to verify the borrowed processors behave as upstream.

## Step 9 — Docs

Update `README.md` with one section per strategy. Minimal example:

```python
class Person(BaseModel):
    name: str
    age: int

result = Harness(HarnessConfig(output_type=Person)).run("Pick a name and age.")
print(result.output.name)  # typed
```

Also update `thinharness/__init__.py` to export the public structured-output markers, including the renamed `ToolStructuredOutput`. Document that structured output and ordinary tools coexist by default, but the synthetic `final_result` must be the only tool call in the final turn. Document that `result.text` is not a JSON serialization of `result.output`.

Document prompted-mode instruction precedence: `system_prompt` content is rendered first through `system_instructions()`, then the structured-output schema instructions are appended verbatim for that run. Do not add an extra prose wrapper around the schema block unless `OutputSchema.build_instructions(...)` owns it.

Document subagent behavior: structured output is only available on explicitly configured named subagents, never inherited from the parent or enabled on the framework default subagent. If a named subagent returns structured output, the parent receives a serialized representation of `result.output` as the subagent tool content.

Document hook filter validation: hooks must be registered after the tools they reference; unknown tool filters raise; synthetic `final_result` is not hookable.

## Open questions

1. Do we want generic typing (`Harness[Person]`, `HarnessResult[Person]`) on day one, or wait? Current plan: wait. Adding `Generic[OutputT]` later is non-breaking as long as the runtime shape keeps `HarnessResult.output`.

## Out of scope (deliberately)

- Streaming structured output (waits for the async + streaming work)
- The four output hook variants — wire in if a real use case appears
- `OutputToolset`/`AbstractToolset` abstraction — replaced by direct `final_result` injection in the loop
- Function-as-output-type (pydantic-ai's `output_type=my_function` style)
- `DeferredToolRequests`, `BinaryImage`, `BuiltinToolCallEvent` outputs
