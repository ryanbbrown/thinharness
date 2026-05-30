# Plan: `parallel_llm` Tool

## Overview

Add a built-in `parallel_llm` tool that lets the agent fan out a list of one-shot LLM completions in parallel through the harness's existing provider layer. Each call is stateless (no tools, no loop, no harness overhead) — strictly cheaper than spawning N subagents when the agent just needs N independent completions (e.g. classify 30 snippets, summarize 20 chunks, translate a batch).

Related plans: `.plans/hooks-and-limits.md` (run limits, `RunUsage`), `.plans/subagents.md` (the heavier alternative this tool exists to undercut).

> **Note:** The original v1 of this plan also bundled a `thinharness/tools/` package reorganization. That work is already complete (`tools/` now contains `base.py`, `filesystem.py`, `jsonl.py`, `mcp.py`, `skills.py`, `__init__.py`), so this revision drops it.

## File layout

New: `thinharness/tools/parallel_llm.py`. Re-export the factory from `thinharness/tools/__init__.py` *and* from `thinharness/__init__.py` (mirrors `create_subagent_tool`, which is exported from both) so it's importable as:

```python
from thinharness import create_parallel_llm_tool   # public top-level
from thinharness.tools import create_parallel_llm_tool  # also valid
```

Add to `thinharness/__init__.py`'s `__all__`. Tests and external users can invoke the handler directly without going through `Harness`.

## Input schema

```python
class ParallelLlmArgs(StrictArgs):
    prompts: list[str] | None = None         # inline prompts
    prompts_file: str | None = None          # path to a JSON array of strings; jailed by read_paths
    system: str | None = None                # shared instructions for every call; see "system semantics"
    model: str | None = None                 # provider-prefixed ref; defaults to harness model
    output_file: str | None = None           # if set, results written here; summary returned to agent
    max_concurrency: int = Field(default=8, ge=1, le=32)
    temperature: float | None = None
```

Imports come from `.base` (`StrictArgs`, `PathPolicy`, `PathValidationError`, `coerce_args`, `Json`, `ToolResult`, `ToolSpec`), same pattern as the sibling modules. Use the existing `Json = dict[str, Any]` alias from `tools.base` rather than introducing a more precise recursive JSON type — consistency with the rest of the codebase that `pyright` already checks.

### Validation

- Exactly one of `prompts` / `prompts_file` must be set.
- `prompts_file` resolves through a `PathPolicy(root, config.read_paths, "read")`; `output_file` resolves through `PathPolicy(root, config.write_paths, "write")`. Both policies are constructed once in `create_parallel_llm_tool(parent)` and captured by the handler. Using the policies (not bare `contained_path`) ensures the model can't bypass user-configured read/write restrictions via this tool.
- `prompts_file` must parse as a JSON array of strings. Reject: empty file, whitespace-only file, JSON object (non-array root), array containing a non-string, empty array — all with the same `"prompts_file must be a non-empty JSON array of strings"` error family.
- Parent directories of `output_file` are created on demand, matching `FileTools.write`.
- Empty `prompts` list → error (no work to do).
- `len(prompts) > config.parallel_llm_max_prompts` → error before any provider call. See **Prompt count cap** below.
- `max_concurrency` bounded by Pydantic `Field(ge=1, le=32)` — consistent with how other tool args declare bounds (e.g. `ReadArgs.limit`).

### Prompt count cap and retry budget — host-controlled

`HarnessConfig` gains two new fields:

```python
parallel_llm_max_prompts: int = Field(default=100, ge=1)
parallel_llm_max_attempts: int = Field(default=4, ge=1, le=10)
```

Both host-controlled (not model-controlled — neither field appears in `ParallelLlmArgs`) so a confused agent can't override them. Workflow authors who need larger batches or different retry budgets set these deliberately at config time. The handler raises before any provider call when `len(prompts) > parallel_llm_max_prompts`; the error message names the configured limit so the workflow author sees what to bump.

- `parallel_llm_max_prompts` caps total work per call (complements `max_concurrency`, which caps in-flight).
- `parallel_llm_max_attempts` caps per-prompt retry budget. Default 4 attempts → sleeps after attempts 0..2 (~1s, 2s, 4s); per-prompt worst case ~7s of wait time on transient failures. Bumping to 5+ extends to 8s/16s sleeps; bumping to 1 disables retry entirely.

## Tool return contract

The handler always returns a `ToolResult` — `_invoke_tool` wraps every plain-dict return in a `ToolResult` envelope, so to keep the on-the-wire shape predictable the handler builds it explicitly. JSON formatting is **compact** for inline content and **pretty-printed** for `output_file`:

```python
payload = {...}  # the batch summary or summary-with-results
inline_content = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
return ToolResult(True, inline_content, metadata={...})

# output_file (when set):
file_text = json.dumps(file_payload, ensure_ascii=False, indent=2) + "\n"
```

The model receives the standard envelope:

```json
{"ok": true, "content": "{...batch JSON...}", "metadata": {...}}
```

Tests parse `content` to assert the batch payload.

### Inline payload (when `output_file` is unset)

```json
{
  "total": 50,
  "succeeded": 48,
  "failed": 2,
  "results": [
    {"index": 0, "ok": true,  "result": "..."},
    {"index": 1, "ok": false, "error": "provider error 429: ..."},
    ...
  ]
}
```

### File payload + inline summary (when `output_file` is set)

File contents — same shape as inline, top-level object including `results`:

```json
{
  "total": 50,
  "succeeded": 48,
  "failed": 2,
  "results": [
    {"index": 0, "ok": true, "result": "..."},
    {"index": 7, "ok": false, "error": "..."}
  ]
}
```

Inline summary (no `results` key):

```json
{
  "total": 50,
  "succeeded": 48,
  "failed": 2,
  "output_file": "results.json",
  "failed_indices": [7, 23]
}
```

`failed_indices` lets the agent re-run failures or `Read` just the failing entries without scanning the full file. The list is omitted if zero failures.

Per-entry shape is **sparse**: success entries are `{"index": i, "ok": true, "result": "..."}` and failure entries are `{"index": i, "ok": false, "error": "..."}`. Consumers branch on `ok`; the missing key on the other side is the contract, not an oversight. Including `index` in every result entry keeps re-runs and filtered partial output files unambiguous even if the surface evolves later.

Results are always in input order, regardless of which calls finished first (`asyncio.gather` preserves submission order). Each `run_one` returns the full entry object — symmetric success/error paths, no separate text-then-build step.

`output_file` resolves relative to the workspace `root` (same as every other path-jailed tool arg), not relative to `HarnessConfig.output_dir`.

### Atomic file writes

`output_file` is written atomically to avoid leaving a partial JSON file on crash or cancellation (this is exactly the scenario `output_file` exists for — large batches):

```python
def _atomic_write_json(output_path: Path, file_text: str) -> None:
    """Write file_text to output_path atomically via a same-dir temp file."""
    tmp = tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        delete=False,
    )
    temp_path = Path(tmp.name)
    try:
        with tmp:
            tmp.write(file_text)
        os.replace(temp_path, output_path)
    except BaseException:  # includes CancelledError
        temp_path.unlink(missing_ok=True)
        raise
```

Notes:
- **Same directory** for the temp file matters: `os.replace` is only atomic when source and destination are on the same filesystem.
- **Deterministic prefix/suffix** (`.{output_path.name}.…tmp`) makes the temp file findable by tests asserting cleanup.
- **`except BaseException`** is intentional — it catches `asyncio.CancelledError` so cleanup runs, then immediately re-raises. Cancellation propagation stays intact.

### Inline-vs-file size

There is no automatic cap or auto-redirect. Workflow authors enable `parallel_llm` deliberately — they know whether a batch's responses will fit in the agent's tool-output window. We can't predict response size from prompt count anyway, so any heuristic ("require `output_file` above N prompts") would be wrong in both directions. The tool docstring and `HarnessConfig`-level docs should be explicit: **for batches whose combined output may be large, the workflow author should require their agent to pass `output_file`**.

(If a batch overruns the downstream `max_tool_chars`, the existing tool-output truncation kicks in — same as any other tool. Loud failure at design time beats silent truncation in production for this tool.)

## Concurrency and retry

The harness is fully async (`ModelSession.start/continue_with_*` are `async def`, transport is `httpx.AsyncClient`), so the handler is `async def`. `_invoke_tool` already detects async handlers via `_is_async_callable`.

Implementation: a `Semaphore(max_concurrency)`, then `await asyncio.gather(*(run_one(i, p) for i, p in enumerate(prompts)))` (no `return_exceptions=True` — see cancellation below). The semaphore wraps each **provider attempt**, not the entire retry loop, so sleeping retries don't hold concurrency slots. **Each retry attempt creates a fresh `session = model.new_session()` and issues exactly one `start` call** — this is a stated requirement, both because it's the right semantics and so tests can count attempts deterministically.

```python
async def run_one(index: int, prompt: str) -> Json:
    """Run one prompt with retry; return the entry envelope (success or failure)."""
    try:
        for attempt in range(max_attempts):
            try:
                async with sem:
                    session = model.new_session()  # fresh per attempt
                    usage.parallel_model_requests += 1  # count attempt before await
                    with RunTracer(parent.tracing).model(model) as span:
                        turn = await session.start(prompt=prompt, instructions=instructions, tools=[])
                        annotate_model_span(span, turn, capture_messages=...)
                    return {"index": index, "ok": True, "result": turn.text}  # uses turn.text only
            except ProviderError as exc:
                if attempt == max_attempts - 1 or not _is_retryable(exc):
                    return {"index": index, "ok": False, "error": str(exc)}
                await _sleep_retry(_retry_delay(attempt))
    except Exception as exc:  # non-ProviderError
        return {"index": index, "ok": False, "error": f"{type(exc).__name__}: {exc}"}
```

The attempt is counted **before** `await session.start(...)`, not after — that way attempted traffic is recorded even if cancellation arrives while the HTTP request is in flight. Failed-and-retried attempts increment too (they're real traffic).

Catching `Exception` (not `BaseException`) so `asyncio.CancelledError` propagates cleanly and `gather` cancels in-flight calls when the surrounding tool call is cancelled. Skipping `return_exceptions=True` on the `gather` call ensures cancellation isn't silently folded into a per-entry "error".

Only `turn.text` is surfaced — the contract is the model's canonical assistant text projection. Raw provider payloads (`turn.raw`) are never exposed in the result entries.

### Model resolution

Mirror the logic in `subagents.build_child_harness`. `parse_model_ref(ref)` returns a tuple `(provider_name, model_id)`, not an object, so the same-provider check uses tuple destructuring. Reuse the existing `_same_provider` / `_provider_prefix` helpers from `subagents.py` (move them to a shared location like `providers.py` if needed) rather than introducing parallel implementations:

```python
def _resolve_batch_model(parent: Harness, model_ref: str | None, temperature: float | None) -> Model:
    """Return parent.model unchanged, or build a fresh one for an override."""
    if model_ref is None and temperature is None:
        return parent.model
    new_ref = model_ref or parent.model_ref
    new_provider, _ = parse_model_ref(new_ref)
    parent_provider = _provider_prefix(getattr(getattr(parent.model, "provider", None), "name", ""))
    same_provider = new_provider == parent_provider
    return infer_model(
        new_ref,
        api_key=parent.config.api_key if same_provider else None,
        base_url=parent.config.base_url if same_provider else None,
        timeout=parent.config.request_timeout,  # always carried — fan-out multiplies wrong-timeout impact
        temperature=temperature if temperature is not None else parent.config.temperature,
        extra_body=parent.config.extra_body,  # always carried, matches subagents (caller-owned global)
    )
```

The handler closes over `parent: Harness` (set at factory time) for defaults; the helper takes the parsed args.

**`extra_body` semantics**: always carried (matching `subagents.build_child_harness`), even across provider overrides. `extra_body` is global and caller-owned — workflow authors that need per-provider differences should split the call. Documented in the config and tool docstrings.

**`temperature=None` with `model` override**: falls back to `parent.config.temperature`, *not* the new provider's default. Documented in the config and tool docstrings so users don't expect a fresh-provider implicit default.

### `ProviderError.status_code` refactor (part of this feature)

To make retry classification structured rather than string-parsed, this feature also lands a small refactor of `ProviderError`:

```python
class ProviderError(RuntimeError):
    """Raised when a provider request fails."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
```

Set `status_code` at the single HTTP raise site in `Provider.post_json`:

```python
except httpx.HTTPStatusError as exc:
    raise ProviderError(
        f"provider error {exc.response.status_code}: {exc.response.text}",
        status_code=exc.response.status_code,
    ) from exc
```

All other raise sites (`"provider request failed:"` transport errors, missing-API-key, `"Anthropic does not support..."` capability errors, `"provider returned invalid JSON:"`) leave `status_code=None` — no HTTP response actually happened in those cases.

**Risk**: very low. `ProviderError` is currently bare (`class ProviderError(RuntimeError): pass`), so:
- Callers that raise `ProviderError("msg")` keep working (the new kwarg is keyword-only with default).
- Callers that catch `ProviderError` keep working (no change to the exception hierarchy).
- Existing `pytest.raises(ProviderError, match="...")` tests keep working (message strings preserved unchanged).
- Nothing reads `status_code` today; the only new consumer is `parallel_llm._is_retryable`.

**Provider-side tests** (additions to `tests/test_providers.py`): assert `exc.status_code == 429` on the existing 429 raise test; assert `exc.status_code is None` on the existing transport-error and invalid-JSON tests.

### Retry policy

With `status_code` available, retry classification is structured for HTTP errors and falls back to a single message-prefix check for the one ambiguous case (`status_code is None` could be transport-retryable or config-non-retryable):

```python
RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})

def _is_retryable(exc: ProviderError) -> bool:
    """True for transport errors and retryable HTTP codes; False for auth/config/capability."""
    if exc.status_code is None:
        # No HTTP response. Distinguish transport (retry) from config/feature (don't).
        # Transport errors are the only None-status case that's retryable today.
        return str(exc).startswith("provider request failed:")
    return exc.status_code in RETRYABLE_STATUS
```

Retryable: exponential backoff with jitter, capped at `HarnessConfig.parallel_llm_max_attempts` (default 4 → sleeps after attempts 0..2 ~1s, 2s, 4s; per-prompt worst case ~7s). Non-retryable (missing API key, capability mismatch, invalid response, 4xx other than 429/408/425): fail immediately so a misconfigured batch doesn't burn time per prompt.

If a future cleanup promotes the `None`-status distinction into structure (e.g. `kind` enum or `ProviderTransportError` subclass), `_is_retryable` simplifies further. Out of scope here — the current shape is already a strict improvement over pure message matching.

Delay calculation and sleeping are split into separate module-level helpers so tests can deterministically patch the sleeper while still asserting backoff bounds:

```python
def _retry_delay(attempt: int) -> float:
    """Return seconds to wait before the next attempt (with jitter)."""
    base = 2 ** attempt          # 1, 2, 4, 8, ...
    return base + random.uniform(0, base * 0.25)

async def _sleep_retry(delay: float) -> None:
    """Sleep for the computed retry delay."""
    await asyncio.sleep(delay)
```

With `parallel_llm_max_attempts=4`, attempts are indexed 0..3 and sleeps occur after attempts 0..2 (the last failed attempt doesn't sleep — it returns the failure entry). So `_retry_delay` is called with `attempt ∈ {0, 1, 2}` by default.

Sleep stays out of the semaphore (semaphore wraps the attempt, not the loop body).

### Failure-entry exception messages

- `ProviderError` (any attempt): `"error": str(exc)` — those messages are already user-facing.
- Path/validation errors: keep the policy's own message.
- Anything else: `"error": f"{type(exc).__name__}: {exc}"` so we don't accidentally leak raw repr or stack-y detail through the exception's `__str__`.

A failed entry never aborts the batch.

### `system` semantics

If `system` is provided, it's passed as `instructions` verbatim. If `system is None`, `instructions=""` — parallel calls get **no** harness system prompt, no workspace summary, no skill summary. This intentionally differs from `Harness.run`, which composes a full system prompt. The point of `parallel_llm` is raw stateless completions; carrying the parent's system prompt would leak context the workflow author may not want.

The tool docstring spells this out: *"If you need the parent harness system prompt, include the relevant instructions in `system`; it is not inherited automatically."*

## Limit accounting

`parallel_llm` counts as **one** call against `max_tool_calls` and contributes **nothing** to `max_model_requests`. Rationale:

- `max_model_requests` exists to bound the agent loop — how many times the model gets to take a turn before the harness gives up. Fan-out calls are not loop turns; they're sub-work of a single tool invocation the agent already chose to make.
- The user enables `parallel_llm` deliberately; if they want to bound fan-out they have two natural knobs already: `max_tool_calls` (caps how many *invocations* the agent can make), `max_concurrency` (caps in-flight at any moment), and now `parallel_llm_max_prompts` (caps total work per call).
- Double-counting against the agent-loop budget would conflate two different concerns and discourage use of the tool exactly when it's most valuable (large batches).

No pre-check against the agent-loop budget. The tool just runs. (`max_tool_calls` defaults to `None` in `HarnessConfig`, so most users don't hit this anyway.)

### Usage tracking

`RunUsage` gains a new counter so parallel traffic is observable without conflating it with the agent-loop budget:

```python
@dataclass
class RunUsage:
    model_requests: int = 0
    tool_calls: int = 0
    cancelled_tool_calls: int = 0
    output_retries: int = 0
    tool_retries: dict[str, int] = field(default_factory=dict)
    parallel_model_requests: int = 0  # NEW: provider attempts made by parallel_llm
```

Incremented **per provider attempt** (just before `await session.start(...)`) so attempted-but-cancelled traffic is recorded. Failed-and-retried attempts count too — they're real traffic.

#### How the handler reaches `RunUsage`

`RunUsage` is currently a local variable inside `Harness.run`, so there's no run-scoped access point a tool handler can read. Add one, mirroring the existing `self._current_run_metadata` lifecycle:

```python
# Harness.__init__
self._current_usage: RunUsage | None = None

# Harness.run
usage = RunUsage()
self._current_usage = usage
try:
    ...  # existing run body
finally:
    self._current_usage = None
    self._current_run_metadata = None
```

The `parallel_llm` handler asserts `parent._current_usage is not None` (it should only be invoked during an active run) and increments through that reference. Same pattern as `_current_run_metadata`; no new mechanism for finding run-local state.

Token-level cost tracking is still out of scope — that's a cross-cutting concern (main loop, subagents, parallel calls, all the same problem) and belongs in its own feature.

## Tracing

`RunTracer` is created per-run inside `Harness.run`; there is no `RunTracer` stored on `Harness` to capture at factory time. The handler captures `parent.tracing` (the `TracingOptions | None`) and instantiates `RunTracer(parent.tracing)` inside the call. With OTel's active context, model spans opened in the handler nest under the surrounding tool span automatically.

`RunTracer.model(model)` yields a `_SpanAdapter` — pass that to `annotate_model_span` along with the `ModelTurn`. With tracing disabled, both are no-ops.

## Tool registration

Add construction to `Harness.__init__` alongside `create_subagent_tool` — the handler needs the parent harness for model, tracing, config, and path policies:

```python
builtin_candidates = [
    *filesystem_tools,
    *self.skills.specs(),
    create_subagent_tool(self, self.config.subagents),
    create_parallel_llm_tool(self),
]
```

`parallel_llm` is **opt-in** via `HarnessConfig.builtin_tools=["parallel_llm", …]`, same pattern as `jsonl_search` and `subagent`. It is *not* added to `DEFAULT_BUILTIN_TOOLS` — users who want fan-out have to ask for it.

## Tool description (model-facing)

"Run N independent prompts as one-shot LLM completions in parallel. Each call is stateless: no tools, no memory, no continuation — only the model's text response is returned. Use this when you have a batch of independent prompts (classify, summarize, translate). For multi-step work, use the subagent tool instead. For large batches, pass `output_file` and read it back rather than receiving full results inline. If you need the parent harness system prompt, include the relevant instructions in `system`; it is not inherited automatically."

## Tests

Add `tests/test_parallel_llm.py` (separate from the existing `tests/test_parallel_tools.py`, which covers parallel execution of normal tools — different feature, similar name). Reuse `tests/fakes.py` helpers (`_fake_openai`, `MultiCallClient`) for the mock model layer. Monkeypatch `_sleep_retry` to a no-op so retry tests don't actually wait.

**Testing style**: prefer direct handler / factory tests for validation, path policy, retry classification, atomic write, `_resolve_batch_model`, and prompts-file parsing — same pattern as existing tests using `call_tool`, `_invoke_tool`, `create_subagent_tool`. Reserve harness-level (`Harness.run` end-to-end) tests for tool selection, limit accounting, tracing nesting, and real model-call integration. This keeps `tests/test_parallel_llm.py` smaller and avoids scripting full model loops for every validation case.

Coverage:

- **Envelope shape**: handler returns a `ToolResult` envelope; batch payload is JSON inside `content`. Inline `content` is compact (no indent); `output_file` content is `indent=2` and ends with a trailing newline.
- **Happy path with inline prompts** (mock model echoing input).
- **`prompts_file` happy path** + all reject cases as one parametrized test: both inputs given, neither given, empty file, whitespace-only file, JSON object (non-array), array with non-string entry, empty array.
- **`output_file`** writes top-level object including `results`, returns summary without `results`, populates `failed_indices` when any failure occurs, omits `failed_indices` when zero failures. Verifies `index` present on every result entry. Verifies parent directories are created on demand.
- **Sparse result shape**: success entry has `result` but no `error` key; failure entry has `error` but no `result` key.
- **Atomic `output_file`**: simulate write failure mid-stream (mock the write to raise) and assert the destination file is not created and no `.{output_file.name}.*.tmp` file remains in the parent directory.
- **Atomic write under cancellation**: cancel the task during the temp write; assert `CancelledError` propagates and no temp file remains. Verifies the `except BaseException` cleanup path.
- **Path policies**: `prompts_file` outside `read_paths` rejected; `output_file` outside `write_paths` rejected; bare `contained_path` (root-only) is *not* sufficient.
- **Prompt count cap**: `len(prompts) > config.parallel_llm_max_prompts` rejected before any provider call (mock model assertion: zero `start` invocations). Error message names the configured limit.
- **Retry budget configurability**: `parallel_llm_max_attempts=1` → no retries even on retryable status; `parallel_llm_max_attempts=2` → at most one retry.
- **Tool selection**: `HarnessConfig(builtin_tools=None)` does not expose `parallel_llm`; `HarnessConfig(builtin_tools=["parallel_llm"])` does expose it; unknown-builtin error lists `parallel_llm` as available.
- **Batch calls receive `tools=[]`**: assert the mock provider records empty tools on every per-prompt `start` call.
- **Limit accounting**: one `parallel_llm` invocation increments `usage.tool_calls` by 1 and leaves `usage.model_requests` unchanged; `usage.parallel_model_requests` equals total provider attempts (succeeded + failed-and-retried).
- **Retry on retryable HTTP status**: mock provider raises `ProviderError(..., status_code=429)` twice then succeeds; result is `ok`; `parallel_model_requests` += 3. Parametrize over `[429, 500, 502, 503, 504, 408, 425]`.
- **Retry on transport error**: mock raises `ProviderError("provider request failed: ...")` (no `status_code`) twice then succeeds.
- **Fast-fail on non-retryable HTTP status**: mock raises `ProviderError(..., status_code=401)` → exactly one attempt, entry is `{ok: false, error: ...}`, `_sleep_retry` never called. Parametrize over `[400, 401, 403, 404]`.
- **Fast-fail on non-retryable `None`-status**: mock raises `ProviderError("MISSING_KEY is required for X")` (no `status_code`, no transport prefix) → exactly one attempt, no retry.
- **`_is_retryable` unit tests** (no harness run): exhaustive table over `(status_code, message_prefix)` cases.
- **`ProviderError.status_code` field** (additions to `tests/test_providers.py`): existing 429 test asserts `exc.status_code == 429`; existing transport-error and invalid-JSON tests assert `exc.status_code is None`.
- **Fresh session per attempt**: assert `model.new_session` is invoked once per provider attempt (not once per prompt).
- **`_retry_delay` bounds**: returns a value in `[2**attempt, 2**attempt * 1.25]` for attempts 0..2 (the default cap; deterministic without patching jitter).
- **`_sleep_retry` patched in all retry tests** so the suite never waits in real time.
- **Concurrency cap honored**: counter inside the mock model + small await; assert max in-flight never exceeds `max_concurrency`.
- **Concurrency under retry**: with `max_concurrency=1` and one prompt sleeping in backoff, a second prompt can still execute — proves the semaphore wraps the attempt, not the loop body.
- **Partial failure**: non-`ProviderError` exception on one prompt → batch completes; that entry is `{ok: false, error: "TypeName: message"}`; other entries succeed.
- **Cancellation propagates**: cancelling the surrounding task cancels the gather (not converted into one failed entry per prompt). Asserts no `return_exceptions=True` swallowing.
- **Mixed-latency ordering**: prompt 1 returns before prompt 0 (mock with controlled awaits); assert result list is still `[index=0, index=1, ...]`. Cheap protection against a future switch to `asyncio.as_completed`.
- **`_resolve_batch_model`** unit tests (no harness run): omitted-both → returns `parent.model`; same-provider override → fresh model with parent `api_key`/`base_url`/`request_timeout`/`extra_body`; cross-provider override → fresh model **without** parent `api_key`/`base_url` but **with** `request_timeout`/`extra_body`; temperature rule (omitted → parent `config.temperature`, provided → that value).
- **`system` semantics**: omitted → `instructions=""`; provided → passed through verbatim.
- **Parent system prompt non-leakage**: configure parent harness with a distinctive system prompt (e.g. `"DISTINCTIVE_PARENT_MARKER"`); run a batch with `system=None`; assert no per-prompt `start` call's `instructions` contains the marker. Executable privacy check, not just an implementation-detail assertion.

Existing test suite must continue to pass.

## Implementation order

Each step is an independent unit; later steps depend on earlier ones:

1. **`ProviderError.status_code`** — add the `status_code` kwarg and set it at the single HTTP raise site in `Provider.post_json`. Update `tests/test_providers.py` to assert the field on the existing 429, transport-error, and invalid-JSON tests.
2. **`RunUsage.parallel_model_requests` + `Harness._current_usage`** — add the counter field and the run-scoped current-usage plumbing on `Harness` (mirror `_current_run_metadata` lifecycle).
3. **`HarnessConfig` fields** — add `parallel_llm_max_prompts` and `parallel_llm_max_attempts`. Public exports: re-export `create_parallel_llm_tool` from `thinharness/__init__.py` and `thinharness/tools/__init__.py`, add to `__all__`. Wire `create_parallel_llm_tool(self)` into `builtin_candidates` in `Harness.__init__`.
4. **`thinharness/tools/parallel_llm.py`** — implement the tool: path policies, `_resolve_batch_model` (reusing `subagents.py`'s provider-prefix helpers, moving them to a shared location if needed), `_is_retryable`, `_retry_delay` / `_sleep_retry`, `_atomic_write_json`, and the handler.
5. **`tests/test_parallel_llm.py`** — full coverage per the test section above, preferring direct handler/factory tests for edge cases.
6. **Docs**: update `README.md`, `docs/docs.md`, and `docs/architecture.md` for the new config fields (`parallel_llm_max_prompts`, `parallel_llm_max_attempts`), the opt-in builtin selection, the new `parallel_model_requests` usage counter, the `system`-not-inherited semantics, the `extra_body` global/caller-owned note, and the `temperature=None`-with-override-uses-config-default note. The architecture doc sections on `RunUsage` and limits should not become misleading.
7. **Verification**: `uv run pyright`, targeted `pytest tests/test_parallel_llm.py tests/test_providers.py`, and `ruff` on touched files.

This order reduces risk: the provider and usage changes (1, 2) are independent foundations; the config/exports (3) is wiring; then the tool (4) is built against stable primitives.

## Open Questions

- **Cross-provider in one batch?** No. Single `model` for the whole batch. Per-prompt model overrides are a future feature, not v1.
- **Streaming results?** No. Ordered list is the agent-facing contract. As-completed streaming buys nothing here because the tool result is one blob returned to the agent.
- **Further `ProviderError` granularity?** Not in v1. After the `status_code` addition lands, the only remaining ambiguity is `status_code is None` (transport vs. config), which the single message-prefix check handles. If we ever need finer classification (e.g. for distinct UX around capability errors), promote `kind` to a structured field or split into subclasses.
