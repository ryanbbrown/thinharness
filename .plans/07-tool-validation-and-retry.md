# Plan — ModelRetry + per-tool retry budget

## Goal

Give tools a clean way to ask the model to retry, and bound the loop with a per-tool retry budget.

- Tool handlers can raise `ModelRetry("hint")` to send a retry-prompt tool output back to the model.
- Pydantic `ValidationError` raised during arg validation (already runs today for `ToolSpec(parameters=BaseModel)`) becomes retryable through the same envelope.
- Malformed JSON / non-object args are also retryable.
- A configurable `tool_retries` budget per tool name per run prevents infinite loops; exceeding it terminates the run with a new `StopReason`.

Non-goals: function-as-tool (`add_tool(fn)` with signature-derived schema), `RunContext`/`deps` threading, streaming partial-arg validation, capability/hook introspection of `ToolDef`. See "Out of scope" for explicit rejections.

## What this plan does *not* need

Nothing from `vendor/pydantic-ai` gets copied. `ModelRetry` is ~10 lines we write ourselves; the retry-envelope shape is thinharness's existing `ToolResult` envelope with a new `metadata.retry` flag. No new dependencies.

## Step 1 — Add `ModelRetry`

Add to `thinharness/tools.py` (not `core.py` — `core.py` imports from `tools.py`, so reversing the direction would create a circular import; tool handlers are also the natural home for an exception they raise):

```python
class ModelRetry(Exception):
    """Raised by a tool handler to ask the model to try again with a hint message."""
    def __init__(self, message: str):
        self.message = message
        super().__init__(message)
```

Implementation re-export in `thinharness/__init__.py`:

```python
from .tools import ModelRetry
```

User import form:

```python
from thinharness import ModelRetry
```

That's the entire "new types" surface for this plan.

## Step 2 — Drop dict-form tool registration

**Land this as a preparatory commit (or separate PR) before the retry work.** It's mechanical type-signature cleanup, touches a lot of surface, and is unrelated to retry semantics. Doing it first lets the existing test suite catch any regressions in isolation before retry behavior is layered on top.

Independent cleanup, but simplifies `add_tool`'s type signature so the retry work has a smaller diff to land against. Today `add_tool` accepts `ToolSpec | dict`; greenfield, we drop the dict form. Sites to update in one commit:

| Location | Current | Becomes |
|---|---|---|
| `Harness.__init__(tools: list[ToolSpec \| Json] \| None)` | accepts both | `list[ToolSpec] \| None` |
| `Harness.add_tool(tool: ToolSpec \| Json)` (`core.py:331`) | accepts both | `ToolSpec` |
| `SubAgentConfig.tools: list[ToolSpec \| Json]` | accepts both | `list[ToolSpec]` |
| `_effective_custom_tools(...)` in `subagents.py` | normalizes dicts | passes `ToolSpec`s through |
| `_tool_name(...)` (looks up `dict["name"]`) | branch for dict | only `ToolSpec` now |
| Existing dict-style examples in README | dict form | explicit `ToolSpec(...)` |
| Existing tests passing dict-style configs | dict form | rewrite to `ToolSpec(...)` |

No `_coerce_tool` helper needed — without function-as-tool there's no normalization to do.

## Step 3 — Retry path in the run loop

This is the meat of the plan. Today `_traced_call_output` (`core.py:375`) returns `(output_str, cancelled)` and `_execute_tool_batch` builds `ToolOutput`s for `session.continue_with_tools`. There's no concept of a tool call needing to be re-issued by the model.

### 3a. Per-run retry state

Add to `RunUsage`:

```python
tool_retries: dict[str, int] = field(default_factory=dict)
"""Number of retryable tool failures observed per tool name during this run."""
```

(Structured output already adds `output_retries: int`. Different field — output retries are per-run, tool retries are per-tool-name.)

**Counter semantics — important to name explicitly.** `tool_retries[name]` counts *retryable failures observed*, including the over-budget failure that terminates the run. It is **not** a count of retry outputs delivered to the model. With `tool_retries=1`:

- Retryable failure #1 → counter becomes 1, retry envelope sent to model, model retries.
- Retryable failure #2 → counter becomes 2, budget check sees `2 > 1`, run terminates with `tool_retries_exceeded`. **The second failure is counted but its envelope is never sent.**

Pin this in the test list.

### 3b. Universal validation rule in `tools.py`

In `tools.py`, `call_tool` and `_invoke_tool` follow one rule for all tool kinds:

> **"If your tool has a schema, bad args trigger a retry. If you used the raw JSON-schema escape hatch, you trust the model."**

Matrix:

| Tool kind | Shape failure (malformed JSON / non-object) | Validation failure (Pydantic `ValidationError` during arg validation) |
|---|---|---|
| `ToolSpec(parameters=BaseModel)` (incl. builtins) | retry envelope | retry envelope |
| `ToolSpec(parameters: Json)` (raw JSON schema; escape hatch) | retry envelope | n/a — no validation runs |

(Note: the existing `parameters` field type is `Json \| type[BaseModel]` where `Json = dict[str, Any]` is a JSON schema object, not Python's `dict` type.)

Greenfield: builtins (which use `StrictArgs` BaseModels via `ToolSpec`) automatically get retry semantics on arg validation failure — that's a behavior change vs today's `{ok: false}` but it's the correct behavior (a `read({"path": 123})` failure is a model mistake worth retrying with budget enforcement).

**Two-phase try blocks are critical**: arg-validation `ValidationError` is retryable, but a *handler-internal* `ValidationError` (e.g. a handler doing its own Pydantic validation downstream of the validated args) is an ordinary tool error, not a retry. Split the try blocks accordingly:

```python
def call_tool(spec: ToolSpec, raw_args: str | Json) -> str:
    # Shape check — universal for all tool kinds
    try:
        args = json.loads(raw_args or "{}") if isinstance(raw_args, str) else raw_args
        if not isinstance(args, dict):
            raise ArgumentShapeError("tool arguments must be a JSON object")
    except (json.JSONDecodeError, ArgumentShapeError) as exc:
        return _retry_envelope("InvalidArguments", str(exc))

    # Arg validation — ValidationError here is retryable
    if _is_args_model(spec.parameters):
        try:
            validated = spec.parameters.model_validate(args)
        except ValidationError as exc:
            return _retry_envelope(
                "ValidationError",
                _format_validation_errors(exc),
                errors=exc.errors(include_url=False, include_context=False),
            )
    else:
        validated = args   # escape hatch — handler trusts the dict

    # Handler dispatch — ValidationError here is an ordinary error, NOT a retry
    try:
        result = spec.handler(validated)
    except ModelRetry as exc:
        return _retry_envelope("ModelRetry", exc.message)
    except Exception as exc:
        if getattr(exc, "_thinharness_strict_hook", False):
            raise
        return ToolResult(False, f"{type(exc).__name__}: {exc}", {"error_type": type(exc).__name__}).as_json()
    # ... normalize result
```

`_invoke_tool` (async) is parallel: same two-phase pattern, with the existing sync/async dispatch fork preserved in the handler phase.

### 3c. Retry envelope

Built by `_retry_envelope` inside `tools.py`. `content` is a string for envelope uniformity; structured pydantic errors live in `metadata.errors`. **No counter fields in the envelope** — the run loop is the sole counter owner (see 3e).

```python
{
    "ok": False,
    "content": "<retry message — error.message for ModelRetry, formatted multi-line for ValidationError>",
    "metadata": {
        "error_type": "ModelRetry" | "ValidationError" | "InvalidArguments",
        "retry": True,
        "errors": [...],  # ValidationError only; from error.errors(include_url=False, include_context=False)
    },
}
```

Format for `ValidationError` `content` — `_format_validation_errors`:

```
Invalid arguments:
- city: Input should be a valid string (got int)
- units: Input should be 'c' or 'f' (got 'kelvin')
- filters.0.name: Field required
- options.user.email: Value error, must contain '@'
- <root>: Value error, age must be greater than name length
```

Locations from `error.errors()` come as tuples (`('filters', 0, 'name')`); the helper joins them as dotted paths so prompts are actionable for nested args. Bare-integer indices stay numeric (`filters.0.name`). **Empty `loc == ()`** (model-level validators, root-level errors) renders as `<root>` so the bullet always has a key.

Same wire format as today's failed-tool envelope, so no provider adapter changes. The *behavioral* change is the retry-budget check in the run loop.

### 3d. Plumbing the retry signal past hooks

Hook output mutation can't be allowed to subvert budget enforcement (a hook that rewrites `metadata.retry` would effectively grant infinite retries). The fix is straightforward: **capture `retry_kind` by parsing the envelope *before* `after_tool_call` hooks fire**, and carry it alongside the (possibly mutated) string output in a small internal dataclass.

```python
# core.py — internal
@dataclass(frozen=True)
class ToolCallExecution:
    """Internal per-call execution data — control-flow signals separate from user-facing record."""
    output: str          # post-hook output, sent to the provider
    cancelled: bool      # short-circuited by before_tool_call hook
    retry_kind: str | None
    # retry_kind ∈ {None, "ModelRetry", "ValidationError", "InvalidArguments"}
```

`_traced_call_output` captures `retry_kind` from the pre-hook envelope immediately after `_invoke_tool`/`_call_output` returns, before firing `after_tool_call`:

```python
output = self._call_output(name, arguments)            # str — unchanged from today
parsed = _parse_tool_output(output)
metadata = parsed.get("metadata") if isinstance(parsed.get("metadata"), dict) else {}
retry_kind = metadata.get("error_type") if metadata.get("retry") is True else None
# ... fire after_tool_call (can rewrite output, but retry_kind is already locked in) ...
return ToolCallExecution(output=output, cancelled=cancelled, retry_kind=retry_kind)
```

Updated signatures (only at the `_traced_call_output` / `_execute_tool_batch` boundary; `call_tool` and `_invoke_tool` stay as `-> str`):

```python
# was: _traced_call_output(...) -> (output: str, cancelled: bool)
# now: _traced_call_output(...) -> ToolCallExecution

# was: _execute_tool_batch(...) -> (recorded: list[Json], outputs: list[ToolOutput])
# now: _execute_tool_batch(...) -> (recorded: list[Json], outputs: list[ToolOutput], executions: list[ToolCallExecution])

# also: _run_calls_concurrently(...) -> list[ToolCallExecution]   (was list[tuple[str, bool]])
```

**Trust model.** Anything that builds a tool output envelope with `metadata.retry: True` will be treated as a retry. In practice that's only our internal `_retry_envelope(...)` helper, since constructing the JSON shape by hand requires deliberate effort. If a user handler intentionally returns a fake retry envelope, that handler chose its own consequences. **Hooks own the message; the harness owns the budget** — and the harness reads the budget from the pre-hook capture, not from anything hooks can rewrite.

### 3e. Budget check (run loop owns the counter)

In `Harness.run`, after `_execute_tool_batch` and **before** `session.continue_with_tools(...)`:

```python
recorded, outputs, executions = await self._execute_tool_batch(run_tracer, turn.tool_calls)
usage.cancelled_tool_calls += sum(1 for execution in executions if execution.cancelled)
tool_call_records.extend(recorded)

for call, execution in zip(turn.tool_calls, executions, strict=True):
    if execution.retry_kind is None or execution.cancelled:
        continue
    usage.tool_retries[call.name] = usage.tool_retries.get(call.name, 0) + 1
    max_retries = self._tool_max_retries(call.name)
    if usage.tool_retries[call.name] > max_retries:
        self.hooks.fire(LimitReachedContext(
            harness=self, metadata=dict(run_metadata),
            limit_kind="tool_retries", limit_value=max_retries,
            current_count=usage.tool_retries[call.name],
        ))
        stop_reason = "tool_retries_exceeded"
        terminal_error = HarnessError(
            f"tool {call.name!r} exceeded max_retries={max_retries}"
        )
        raise terminal_error
```

Type updates needed:
- Add `"tool_retries_exceeded"` to `StopReason`.
- Add `"tool_retries"` to the `LimitReachedContext.limit_kind` `Literal` (today only `"model_requests" | "tool_calls"` — type-check will fail otherwise).

`_tool_max_retries(name)` returns the tool's `max_retries` if set, else `config.tool_retries`.

**Over-budget envelopes are not sent to the provider, and other calls in the same batch are not sent either.** When the budget check raises, the entire batch's `outputs` list never reaches `session.continue_with_tools(...)`. Concrete consequences:

- `max_retries=0`: the first failure fails the run immediately, no extra provider roundtrip.
- Mixed batch where one call succeeds and one call is the budget-exceeding retry: the successful output is recorded locally but the model never sees it; the conversation ends.
- Batch with two failures of the same tool, where the second crosses the budget: both records exist locally; neither output is sent.

**Diagnostic visibility on `tool_retries_exceeded` is limited by design.** Tool spans were created during `_traced_call_output` and `LimitReachedContext` fires before the raise, so tracing and the limit hook see the budget breach. But the run ends with a raised `HarnessError`, so `RunEndContext.result is None` and the local `tool_call_records` list isn't surfaced to callers. We don't add a partial-diagnostics API for this — no concrete consumer yet. Tests assert `result is None` plus `stop_reason == "tool_retries_exceeded"` plus the expected `LimitReachedContext` fire.

**Retry budget is per tool name per run, deliberately coarse.** Two calls to the same tool in one batch (different `call_id`s) share the budget — a bad first call consumes budget that an unrelated second call needs. Simple and predictable; if it becomes a problem in practice, the dict can be re-keyed on `(name, call_id)` without an API change.

### 3f. Tracing

`_traced_call_output` annotates the tool span with the pre-hook `retry_kind` (alongside the existing `cancelled` handling), not from re-parsing the post-hook output. A hook that rewrites `metadata.error_type` shouldn't change what the span reports:

```python
if execution.retry_kind is not None:
    span.set_error(f'Tool "{name}" failed', execution.retry_kind)
elif parsed.get("ok") is False:
    span.set_error(f'Tool "{name}" failed', "ToolExecutionError")
```

### 3g. Cancellation interaction

`BeforeToolCallContext.cancelled` already short-circuits to a `ToolCallCancelled` envelope. Cancellation is *not* a retry — `_traced_call_output` returns `ToolCallExecution(cancelled=True, retry_kind=None)` regardless of envelope contents. The `cancelled` check in 3e is belt-and-braces; the `retry_kind is None` check would catch it too. Tested explicitly.

## Step 4 — `ToolSpec.max_retries` field

Add a per-tool override to `ToolSpec`. Place the new field **after `metadata`** so existing positional `ToolSpec(name, description, parameters, handler, sequential, metadata)` constructions don't get their argument meanings shuffled:

```python
@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: Json | type[BaseModel]
    handler: ToolHandler
    sequential: bool = False
    metadata: Json = field(default_factory=dict)
    max_retries: int | None = None    # NEW; None = use HarnessConfig.tool_retries

    def __post_init__(self) -> None:
        if self.max_retries is not None and self.max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {self.max_retries}")
```

Non-negativity is enforced in `__post_init__` since `ToolSpec` is a frozen dataclass, not a Pydantic model. `max_retries` is keyword-only in practice.

## Step 5 — `HarnessConfig` + `SubAgentConfig` surface

```python
class HarnessConfig(BaseModel):
    # existing fields ...
    tool_retries: int = Field(default=1, ge=0)
    """Default number of times to retry a tool that raises ModelRetry or fails arg validation.
    Per-tool override via ToolSpec.max_retries."""
```

`1` matches pydantic-ai's default. `Field(ge=0)` matches the style of structured-output's `output_retries`.

Named subagents get an **independent** retry budget — they don't inherit from the parent. Only the unnamed/general subagent (which has no `SubAgentConfig` to carry a value) inherits, because there's nowhere else to read the value from. This matches how `output_retries` works today (`SubAgentConfig.output_retries: int = 1`, no `None` sentinel, no inheritance).

```python
class SubAgentConfig(BaseModel):
    # existing fields ...
    tool_retries: int = Field(default=1, ge=0)
    """Retry budget for tools invoked by this subagent.
    Independent of the parent; defaults to 1 (matching HarnessConfig default)."""
```

Child-harness builder logic (concrete — don't infer by analogy from `output_retries`'s current pattern, even though the final shape ends up similar):

```python
"tool_retries": (
    config.tool_retries          # named subagent: its own value (default 1)
    if config is not None
    else parent_config.tool_retries   # default subagent: inherit
),
```

Inheritance summary:

| Scenario | Child's `tool_retries` |
|---|---|
| Default subagent (no `SubAgentConfig`) | inherit `parent_config.tool_retries` |
| Named subagent, no explicit `tool_retries` set | `1` (the field default) |
| Named subagent, explicit `tool_retries=N` | `N` |

Plain English: **named subagents have their own budget; default subagent inherits the parent's.**

## Step 6 — Builtins automatically retry on arg validation

The existing builtins (`read`, `write`, `edit`, `search`, ...) already use `StrictArgs` Pydantic models via `ToolSpec(parameters=ReadArgs, ...)`. With Step 3b's universal rule, they **automatically** get retry envelopes on arg validation failure — no per-builtin change needed.

What still has to be decided per-builtin (deferred): whether to raise `ModelRetry(...)` from inside the handler for *logic* failures that today return `{ok: false}`. Examples:

- `edit` returning `"old_string not found"` (`tools.py:375`) — could become `ModelRetry("re-read the file and retry with exact text")`.
- `edit` returning `"old_string appears N times"` (`tools.py:379`) — same pattern.

Don't migrate the in-handler `ModelRetry` cases in this plan. Revisit once we have signal on whether `ModelRetry` actually drives better model behavior on the builtins than plain `{ok: false}`.

## Step 7 — Tests

New file `tests/test_tool_retry.py` (uses existing `FakeModel`):

- **Public import** — `from thinharness import ModelRetry` works
- Tool raises `ModelRetry("try again with X")` → next turn shows the retry envelope as a tool output → tool succeeds on retry → run completes
- Validation failure (model sends `{age: "five"}` for `age: int`) on a `ToolSpec(parameters=BaseModel)` tool → retry envelope → tool retried → success
- **Handler-internal `ValidationError`** (handler does its own Pydantic validation downstream of args) → ordinary failed tool envelope, NOT a retry, does NOT consume budget
- **Builtin validation failure is retryable** — call a builtin with wrong arg types, see the retry envelope (pins the builtins-auto-migrate behavior)
- **Malformed JSON args** (model sends `"{not json"`) → retry envelope with `error_type: "InvalidArguments"` → tool retried → success; also counts against budget
- **Nested validation error location formatting** — handler with `filters: list[FilterModel]`, invalid inner field → retry message contains `filters.0.name: ...`
- Exceed `tool_retries=2` → `HarnessError`, `stop_reason == "tool_retries_exceeded"`
- **`max_retries=0`** fails the run on the first bad call with no extra provider roundtrip (assert `session.continue_with_tools` not called for the over-budget batch)
- **`max_retries` per-tool override lower than the config default** (config `tool_retries=3`, tool `max_retries=1`) — the tool-level override wins; pinned separately from the `max_retries=0` edge case
- **Counter semantics** — with `tool_retries=1`, after the second retryable failure assert `usage.tool_retries[name] == 2` (the over-budget failure is counted even though its envelope is not sent)
- **Two calls to the same tool in one batch** share the retry budget AND, when the second crosses the limit, the batch's outputs are not sent at all (pin both the shared count and the no-continuation behavior)
- **Unknown tool errors do not consume retry budget** (calling a nonexistent tool returns `{ok: false}` without `retry: True`; `retry_kind` is None)
- `before_tool_call` cancellation does *not* consume retry budget (and `_traced_call_output` returns `retry_kind=None` for cancellations)
- `ModelRetry` raised by *async* handler is captured
- **Negative limits rejected** — `HarnessConfig(tool_retries=-1)` and `ToolSpec(max_retries=-1)` raise
- `_run_calls_concurrently` preserves `ToolCallExecution.retry_kind` in result order (parallel batch with one retry and one success)
- Subagent inheritance — **default subagent** (no `SubAgentConfig`) inherits parent `tool_retries`; **named subagent with default config** (no explicit `tool_retries`) runs with `1` regardless of parent; **named subagent with `tool_retries=N`** runs with `N`
- **`RunEndContext` on `tool_retries_exceeded`** — assert `stop_reason == "tool_retries_exceeded"`, `usage.tool_retries[name]` reflects the over-budget count, and `result is None` (the absence of a `HarnessResult` is intentional, not an oversight)
- **Root-level validation error** formats with `<root>` as the location, not an empty path (model-level validator failing on the args dict)

Hook/tracing integration tests (extend existing hook tests or co-locate):

- `after_tool_call.parsed_output` sees the retry envelope (with `metadata.retry=True`) for `ModelRetry`
- `after_tool_call.parsed_output` sees the retry envelope for `ValidationError`
- An `after_tool_call` hook that **mutates `after.output` to invalid JSON** does *not* break budget enforcement — `retry_kind` was captured pre-hook from the original envelope
- An `after_tool_call` hook that **rewrites `metadata.error_type`** does not affect the span's reported error type — tracing reads pre-hook `retry_kind` (Step 3f)
- Tool spans are marked failed for retry outputs, with `error_type` set to `ModelRetry` / `ValidationError` / `InvalidArguments`

Public `call_tool` tests stay separate from `_invoke_tool` tests — the sync helper still returns the existing `AsyncHandlerInSyncContext` envelope for async-only paths.

## Step 8 — Docs

README addition — `ModelRetry` example using the existing `ToolSpec` + `BaseModel` API:

```python
from pydantic import BaseModel
from thinharness import Harness, HarnessConfig, ModelRetry, ToolSpec

class UserLookupArgs(BaseModel):
    user_id: str

def lookup_user(args: UserLookupArgs) -> dict:
    user = db.get(args.user_id)
    if user is None:
        raise ModelRetry(f"user {args.user_id!r} not found; did you mean to search by email instead?")
    return user

harness = Harness(HarnessConfig(model="openai:gpt-5.2", tool_retries=2))
harness.add_tool(ToolSpec(
    name="lookup_user",
    description="Look up a user by id.",
    parameters=UserLookupArgs,
    handler=lookup_user,
))
```

Also note in the README that built-in tools now automatically retry on arg-validation failures (counted against `tool_retries`).

Update any existing README examples that pass `dict`-style tool configs to use `ToolSpec(...)` (see Step 2).

## Resolved

- **No function-as-tool feature** — `add_tool(my_func)` deferred. Users define `ToolSpec(parameters=BaseModel, handler=...)`. The schema-derivation infrastructure pydantic-ai provides isn't worth the ~500 lines + `griffe` dep for the ergonomic gain.
- **No vendoring** — `ModelRetry` is 10 lines we write ourselves. Nothing else from pydantic-ai is copied.
- **Retry envelope `content` is a string**, structured pydantic errors go in `metadata.errors`. Keeps envelope shape uniform across all tool outputs.
- **Per-tool retry counts as `dict[str, int]` on `RunUsage`**, matching the `tool_calls` style. No separate state object.
- **Universal validation rule** — all tools with a schema (BaseModel) retry on `ValidationError`; only raw-dict escape-hatch tools skip validation. Builtins automatically pick up retry semantics for free (greenfield migration).
- **Override kwargs on `add_tool(spec)` not introduced** — without function-as-tool there's no temptation to add them. `add_tool` keeps its current signature.
- **Tracing reads pre-hook `retry_kind`** — span `error_type` cannot be subverted by `after_tool_call` hooks rewriting envelope metadata.
- **`ToolCallExecution` carries control-flow data**, `tool_call_records` carries diagnostic data — kept separate so hook mutation of records can't subvert budget enforcement.
- **Two-phase try blocks during call dispatch** — arg-validation `ValidationError` is retryable; handler-internal `ValidationError` is an ordinary tool error.
- **`ModelRetry` lives in `tools.py`**, not `core.py`, to avoid a circular import (`core.py` already imports from `tools.py`).
- **Trust model is deliberately lenient** — we don't defend against handlers that intentionally forge `metadata.retry: True` in their output. Constructing the envelope by hand requires deliberate effort; users who do it own the result. The protection that *does* matter (hooks rewriting captured retry signal) is handled by capturing `retry_kind` from the pre-hook envelope.
- **`include_input=False` not used on Pydantic errors** — error metadata keeps the invalid input (default Pydantic behavior). Stripping it would defend against one secret-leak channel while bash, file reads, env access, and any custom tool remain wide open; the trade isn't worth the loss of debuggability. Secret hygiene is a top-level concern (env scrubbing, redacting hooks), not point-fix defense inside Pydantic error formatting.
- **Named subagents have independent retry budgets**, default subagent inherits the parent's. Matches `SubAgentConfig.output_retries`'s shape and avoids the surprise of a generous parent budget silently bleeding into a different-purpose subagent.
- **No partial-diagnostics API** on `tool_retries_exceeded` — tracing spans + `LimitReachedContext` cover the inspectability need; the local `tool_call_records` isn't surfaced (no `HarnessResult` returned on raised errors). Revisit if a concrete consumer appears.
- **Retry limits are `Field(ge=0)`** on `HarnessConfig.tool_retries` and `SubAgentConfig.tool_retries`; `ToolSpec.max_retries` validated in `__post_init__`.

## Out of scope

- **Function-as-tool (`add_tool(fn)` with signature-derived schema)** — would require vendoring `_function_schema.py`, `_griffe.py`, `_json_schema.py` plus a `griffe` dependency. Users define `BaseModel` args explicitly. Revisit if there's real demand.
- `RunContext` / `deps` threading into handlers — separate plan
- Streaming partial-arg validation (`allow_partial='trailing-strings'`)
- Function-as-output-type (pydantic-ai's `output_type=my_function`)
- Migrating builtins to in-handler `ModelRetry` envelopes for logic failures (arg-validation retry already happens automatically)
- Per-tool `wrap_tool_validate` / `before_tool_validate` capability hooks
- A `@harness.tool` decorator
