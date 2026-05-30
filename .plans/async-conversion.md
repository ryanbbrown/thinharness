# Plan — Full async conversion

## Goal

Make thinharness async-native end to end, with a thin sync wrapper for callers who want it. Real concurrency for tool calls and provider HTTP. Foundation for streaming.

## Stance

Greenfield, no backward-compat constraints. We commit to:

- All `Model` / `ModelSession` methods are `async def`
- All HTTP is `httpx.AsyncClient`
- `Harness.run()` becomes `async def`; `Harness.run_sync()` is a wrapper using `asyncio.run`
- Tool fanout uses `asyncio.wait(..., return_when=FIRST_EXCEPTION)` so a strict-hook failure cancels pending siblings immediately — drop `ThreadPoolExecutor`
- Sync handlers run on a worker thread via `asyncio.to_thread`; if the threaded call returns an awaitable (e.g. decorated async handler) it is awaited on the loop. No reliance on `iscoroutinefunction(handler)`.
- Async-runtime: pure `asyncio` end to end. No `anyio` dependency — Python 3.11+ `asyncio.to_thread` propagates contextvars and is sufficient.

## Step 1 — Add httpx, drop urllib

`pyproject.toml`:

```toml
dependencies = [
    "pydantic>=2.13.4",
    "httpx>=0.27",
]
```

We deliberately do not pull in `anyio`. Python 3.11+ ships `asyncio.to_thread`, which is enough for sync tool dispatch and already propagates contextvars. The rest of the implementation (`create_task`, `Semaphore`, `wait`) is asyncio-native, so adding `anyio` would only buy us a runtime-neutrality story we don't actually keep.

Delete `_post_json` from `providers.py` and the `urllib` imports.

All three concrete provider constructors (`OpenAIProvider`, `AnthropicProvider`, `OpenRouterProvider` at `providers.py:133/148/169`) accept and forward `http_client: httpx.AsyncClient | None = None` to `super().__init__`. Without that, callers can't actually inject a client through the concrete classes.

`Provider` base class becomes:

```python
class Provider:
    name = "provider"
    api_key_env = ""
    default_base_url = ""

    def __init__(self, *, api_key=None, base_url=None, timeout=120, http_client: httpx.AsyncClient | None = None):
        self.api_key = api_key or (os.getenv(self.api_key_env) if self.api_key_env else None)
        self.base_url = (base_url or self.default_base_url).rstrip("/")
        self.timeout = timeout
        self._http_client = http_client    # injected for tests/sharing
        self._owns_client = http_client is None

    def _client(self) -> httpx.AsyncClient:
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=self.timeout)
        return self._http_client

    async def aclose(self) -> None:
        if self._owns_client and self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None

    async def post_json(self, path: str, payload: Json) -> Json:
        try:
            response = await self._client().post(f"{self.base_url}{path}", json=payload, headers=self.headers())
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise ProviderError(f"provider error {exc.response.status_code}: {exc.response.text}") from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"provider request failed: {exc}") from exc
        try:
            return response.json()
        except ValueError as exc:
            raise ProviderError(f"provider returned invalid JSON: {exc}") from exc
```

Preserves the three-class failure parity from the current `_post_json` (HTTP status, transport/timeout, invalid JSON).

Per-provider `create_response` / `create_message` / `create_chat_completion` become `async def` and `await self.post_json(...)`.

Pytest config also lands here so the new async tests can run from the start:

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
```

This avoids decorating every async test with `@pytest.mark.asyncio`.

## Step 2 — `Model` and `ModelSession` go async

```python
class Model(Protocol):
    model: str

    @property
    def provider(self) -> Provider: ...
    @property
    def api_key(self) -> str | None: ...

    def new_session(self) -> ModelSession: ...   # session creation stays sync


class ModelSession(Protocol):
    async def start(self, *, prompt, instructions, tools, metadata=None, previous_response_id=None) -> ModelTurn: ...
    async def continue_with_tools(self, outputs, *, tools, metadata=None) -> ModelTurn: ...
```

All three concrete sessions (`OpenAIResponsesSession`, `AnthropicMessagesSession`, `OpenRouterSession`): convert `def _complete` to `async def _complete` and `await self.model.provider.create_*(...)`. Methods are async all the way down — no sync inner helpers calling async outer.

## Step 3 — `Harness.run` becomes async

`core.py:run` rewritten as `async def run`. Mechanical changes:

- All `session.start(...)` and `session.continue_with_tools(...)` calls awaited
- `_execute_tool_batch` becomes `async`
- Replace `ThreadPoolExecutor` with `asyncio.create_task` fanout (Step 4)
- `RunTracer` context managers stay sync — they don't do I/O. OTel context propagation is automatic across `await`.

### External-cancellation contract

`asyncio.CancelledError` inherits from `BaseException`, not `Exception`. The current run loop catches `ProviderError`, `HarnessError`, and `Exception` (`core.py:306/319/325`) — all of those *miss* cancellation, so a converted run that is cancelled from outside would slip past every recording path and fire `run_end` with `terminal_error=None` and `stop_reason="end_turn"`. That's wrong on both axes.

Introduce a new `StopReason` literal `"cancelled"`, distinct from `"cancelled_by_hook"` (which is a normal harness outcome from `before.cancelled`). The cancel arm goes inside the same `with run_tracer.agent(...)` block as the existing `except ProviderError/HarnessError/Exception` chain so `agent_span` is in scope, and it sits *alongside* those arms, not above or below them. Concretely the rewritten skeleton is:

```python
self._running = True
try:
    try:
        with run_tracer.agent(conversation_id=...) as agent_span:
            try:
                # ... existing flow: hooks, session start, the while True provider/tool loop ...
            except asyncio.CancelledError as exc:
                stop_reason = "cancelled"
                terminal_error = exc
                agent_span.record_exception(exc)
                agent_span.set_error("run cancelled", "CancelledError")
                raise
            except ProviderError as exc:
                # unchanged from core.py:306
                ...
            except HarnessError as exc:
                # unchanged from core.py:312
                ...
            except Exception as exc:
                # unchanged from core.py:319
                ...
    finally:
        fire_run_end_once()
finally:
    self._current_run_metadata = None
    self._running = False
```

Key invariants this preserves:

- `fire_run_end_once()` is called from the outer `finally`, so it fires exactly once whether the run ends normally, via `Exception`, or via `CancelledError`. The existing `nonlocal run_end_fired` guard already handles "fire only once" — don't add a second call site.
- `self._running` is reset on the outermost `finally`, so a `Harness` instance can be re-used after a cancelled run. (Tests will assert this.)
- The inner `try` blocks around `session.start` / `session.continue_with_tools` (`core.py:264-276`, `core.py:297-304`) keep their `except Exception:` shape — they're for annotating the *model* span on provider failures. `CancelledError` deliberately slips past those inner arms and is annotated only at the agent-span level by the cancel arm above. Don't widen the inner arms to `BaseException`.
- `agent_span.record_exception(exc)` plus the surrounding `with run_tracer.agent(...)` context manager exit may both record the exception depending on the OTel tracer configuration. That's a minor double-record at worst and is consistent with how the existing `Exception` arm behaves today.

The Step 4 fanout already drains pending tool tasks on `BaseException`, so leaked tool work is bounded to "in-flight sync threads complete; everything else is cancelled before the `CancelledError` re-raises." `run_end` fires once with `error=exc` and `stop_reason="cancelled"`.

Add `run_sync`. The body has to close any harness-owned client *inside* the same `asyncio.run` loop, because `httpx.AsyncClient` binds to the loop it was created on — closing after `asyncio.run` returns means closing on a dead loop:

```python
def run_sync(self, prompt: str, **kwargs) -> HarnessResult:
    """Synchronous wrapper around `run`. Convenience for non-async callers."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        raise HarnessError("run_sync cannot be called from inside a running event loop; await run() instead")

    async def _run_and_close():
        try:
            return await self.run(prompt, **kwargs)
        finally:
            await self.aclose()

    return asyncio.run(_run_and_close())
```

`aclose()` respects the ownership flags from Step 7, so injected clients and shared subagent providers are not touched.

(Pydantic-ai uses a `_utils.get_event_loop()` helper that handles the nested-loop case via `nest_asyncio` shimming. We don't need that complexity — explicit error is cleaner.)

## Step 4 — Tool fanout

### Contract (lock this down before coding)

- **Normal tool handler errors** stay per-tool results, unchanged from today. They are caught inside `_invoke_tool` (see below) and never propagate to the fanout loop.
- **Normal `before.cancelled`** produces a `ToolCallCancelled` output for that one call; siblings continue. Unchanged from today.
- **Strict hook exception** aborts the batch: not-yet-started sibling tasks are cancelled before they run; pending awaits unblock with `CancelledError`; the original strict exception is re-raised after pending tasks drain.
- **In-flight sync handlers running on worker threads cannot be cancelled.** Python's `threading` API has no preemptive kill. The fanout abort waits for already-running threaded handlers to finish before re-raising. Mutating built-ins (`write`, `edit`, `skill_run`) carry `sequential=True`, so they are never in a parallel batch — only async or read-only sync handlers will be in flight when an abort fires.
- **The `subagent` tool is *not* sequential, by design.** Concurrent subagent calls are a feature, not a bug. The corollary is: if two parallel subagent calls each invoke `write`/`edit`/`skill_run` through their child harnesses, those writes happen concurrently across child harnesses. On parent strict-hook abort, pending children are cancelled; already-running child runs drain (and within each child, the sequential rule still serializes its own mutations). We don't flip `subagent.sequential` based on child tool surface — too clever, easy to break. If a user wants strict cross-subagent mutation serialization, set `tool_execution="sequential"` on the parent harness.
- **Result ordering**: outputs returned in original `calls` index order (not completion order), so the `ToolOutput` list handed back to the provider matches model-request order.
- **Sequential rule**: if any tool in the batch has `sequential=True`, or `tool_execution="sequential"`, the whole batch awaits in a loop. Unchanged from today.
- **Provider continuation** runs once, after the whole batch is either complete or has been converted into per-tool outputs.
- **`after_tool_call` firing rules** (preserving current behavior at `core.py:391-414`):
  - Normal handler success or per-tool failure → `after_tool_call` fires with the normalized envelope.
  - Normal `before.cancelled` → `after_tool_call` fires with the `ToolCallCancelled` envelope as `output`.
  - **Strict hook exception during `before_tool_call`** → `after_tool_call` does *not* fire for that call. The strict exception propagates out of `_traced_call_output`'s `try` block; the `finally` resets `_CURRENT_TOOL_CALL` but doesn't fire after-hooks.
  - **External cancellation during `_invoke_tool`** (semaphore await or `to_thread`) → `after_tool_call` does *not* fire. `CancelledError` propagates through `_traced_call_output` without producing a result envelope. The `RunTracer.tool` context manager records the cancellation as a tool-span exception on exit.
- **Semaphore + cancellation**: `async with sem:` releases on `CancelledError` via `__aexit__`, including when the task is cancelled mid-`to_thread`. A task cancelled while awaiting `sem.acquire()` never acquired the slot, so there's nothing to release. A strict-hook abort with 16 in-flight sync handlers therefore frees all 16 slots immediately, even though the worker threads keep running until their handlers return.

### Implementation

Replace `_run_calls_in_threads` with an `asyncio.wait(FIRST_EXCEPTION)` loop so a strict-hook failure cancels pending siblings immediately rather than waiting for them to complete:

```python
async def _run_calls_concurrently(self, run_tracer, calls):
    sem = asyncio.Semaphore(MAX_PARALLEL_TOOL_WORKERS)

    async def invoke(index, call):
        async with sem:
            return await self._traced_call_output(run_tracer, call.id, call.name, call.arguments, index)

    tasks = [asyncio.create_task(invoke(i, c)) for i, c in enumerate(calls)]
    task_index = {task: i for i, task in enumerate(tasks)}
    results: list[tuple[str, bool] | None] = [None] * len(tasks)
    pending = set(tasks)
    try:
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_EXCEPTION)
            for task in done:
                exc = task.exception()
                if exc is not None:
                    for sibling in pending:
                        sibling.cancel()
                    await asyncio.gather(*pending, return_exceptions=True)
                    raise exc
                results[task_index[task]] = task.result()
    except BaseException:
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        raise
    return results  # type: ignore[return-value]
```

Indexing by original task position preserves model-request order regardless of completion order. Normal handler errors are caught inside `_invoke_tool` and returned as `(output, cancelled)` tuples, so the only exceptions that reach this loop are strict hook failures, parent cancellation, and harness bugs.

Empty-batch invariant: `calls == []` skips the `while pending:` loop entirely and returns `[]`. In practice this can't happen — `_execute_tool_batch` is only called from `core.py:293` after the `if not turn.tool_calls:` early-return — but the shape is well-defined. Don't add a defensive guard.

### Handler dispatch

`_traced_call_output` becomes `async def`. `_call_output` invokes the handler once and inspects the return value:

```python
async def _call_output(self, name: str, arguments: str) -> str:
    spec = self._tool_map.get(str(name))
    if not spec:
        return json.dumps({"ok": False, ...})
    return await _invoke_tool(spec, arguments or {})
```

`_invoke_tool` lives in `tools.py` alongside the sync `call_tool`, and the two share argument parsing and result normalization through `_prepare_args` / `_normalize_result` so they cannot drift. Handlers are always called single-positional (`spec.handler(args)`), matching the current `tools.py:690` convention. Exception handling mirrors sync `call_tool`: normal handler exceptions become structured failed outputs, but `_thinharness_strict_hook` propagates.

```python
def _prepare_args(spec, raw_args) -> Any: ...        # JSON parse + pydantic validation
def _normalize_result(spec, raw) -> str: ...         # → ToolResult envelope

def call_tool(spec, raw_args) -> str:
    args = _prepare_args(spec, raw_args)
    if isinstance(args, str):  # _prepare_args returned an error envelope
        return args
    try:
        result = spec.handler(args)
    except Exception as exc:
        if getattr(exc, "_thinharness_strict_hook", False):
            raise
        return ToolResult(False, f"{type(exc).__name__}: {exc}", {"error_type": type(exc).__name__}).as_json()
    if inspect.isawaitable(result):
        # public sync path cannot drive an event loop — surface as a tool error and avoid the un-awaited coroutine warning
        close = getattr(result, "close", None)
        if close is not None:
            close()
        return ToolResult(False, "async handler requires harness execution", {"error_type": "AsyncHandlerInSyncContext"}).as_json()
    return _normalize_result(spec, result)

async def _invoke_tool(spec, raw_args) -> str:
    args = _prepare_args(spec, raw_args)
    if isinstance(args, str):
        return args
    try:
        # Thread-first: run the (possibly sync) handler on a worker thread so the loop stays unblocked.
        # asyncio.to_thread copies the calling task's Context into the worker thread (loop → thread, one-way),
        # so a _CURRENT_TOOL_CALL.set(...) performed on the loop side before this await is visible inside the
        # handler. Changes made *inside* the thread are NOT propagated back. The harness only sets
        # _CURRENT_TOOL_CALL from the loop side in _traced_call_output (core.py:378), which is correct.
        result = await asyncio.to_thread(spec.handler, args)
        if inspect.isawaitable(result):
            # Decorated async handler whose factory ran in the thread; await the coroutine on the loop.
            result = await result
    except Exception as exc:
        if getattr(exc, "_thinharness_strict_hook", False):
            raise
        return ToolResult(False, f"{type(exc).__name__}: {exc}", {"error_type": type(exc).__name__}).as_json()
    return _normalize_result(spec, result)
```

Key correctness points (each was a bug in earlier drafts):

- **No double-invocation**: the sync handler is called exactly once, on the worker thread. The earlier sketch ran `spec.handler(args)` on the loop and *then again* in a thread — that blocked the loop and duplicated side effects.
- **Single positional argument**, not `**args` — matches current handler signatures across built-ins, skills, the subagent tool, and test fakes.
- **Symmetric exception wrapping**: an async tool that raises `ValueError` becomes a per-tool failed result, just like its sync equivalent. Without this, an async tool error would tear down the whole run.
- **Public `call_tool` has a defined async-handler failure mode** rather than leaking an un-awaited coroutine warning. Async-capable execution requires going through `Harness.run`.
- Detecting awaitables on the *return value* (not on the handler with `iscoroutinefunction`) handles `functools.partial`, callable objects, and decorated handlers.

A note on `_prepare_args`'s `str | Any` return shape: it's a pragmatic consolidation of the three early returns in current `call_tool` (`tools.py:681-688`). It's mildly awkward — `_prepare_args` returning a "result envelope or parsed args" union means both callers branch on `isinstance(args, str)`. An alternative (`_prepare_args` raises a `_ArgParseError` carrying the envelope, callers `try/except`) is cleaner but adds a one-off exception type. Either works; pick during implementation. Either way the goal is "only one place parses and validates args."

`_invoke_tool` stays private — not exported from `thinharness.__init__`. `ToolSpec.handler` type widens to `Callable[[Any], Any | Awaitable[Any]]`.

All built-in tools stay sync and dispatch through the same threaded path — filesystem reads/writes, `search` (ripgrep subprocess), `skill_run`, `jsonl_search`. Python has no native async file I/O for regular files, so rewriting them async-native would only wrap the same thread pool.

`MAX_PARALLEL_TOOL_WORKERS` is enforced by the `Semaphore` in `_run_calls_concurrently` (preserves the existing 16-call ceiling so we don't fork-bomb the disk or the default thread pool).

`tool_execution: Literal["auto", "sequential"]` config still works — sequential just awaits each `_invoke_tool` in a loop.

## Step 5 — Subagents

`subagents.py:create_subagent_tool` returns a `ToolSpec` whose handler (`run_subagent_tool`, `subagents.py:76`) creates and runs a child harness. The handler becomes `async def` and `await`s the child's `run()`. No `to_thread` indirection needed — child harness is now async too.

### Handler shape with `aclose()` cleanup

The current `run_subagent_tool` fires the after-subagent hook from two different paths: success at line 140 and failure at line 120. The async rewrite must thread `await child.aclose()` through both without firing the after-hook twice and without masking a `child.run` exception with a later `aclose` failure. Concrete shape:

```python
async def run_subagent_tool(args: SubAgentArgs) -> ToolResult:
    config = _select_config(subagent_configs, args.agent)
    child = build_child_harness(parent, config)
    effective_tools = [tool.name for tool in child.tools]
    try:
        try:
            result = await child.run(args.task, metadata=_child_metadata(parent))
        finally:
            await child.aclose()
    except Exception as exc:
        # existing failure-path after-hook fire (subagents.py:120) goes here
        parent.hooks.fire(AfterSubAgentContext(..., error=exc))
        return ToolResult(False, str(exc), {"agent": agent_name, ...})
    # existing success-path after-hook fire (subagents.py:140) goes here
    parent.hooks.fire(AfterSubAgentContext(..., result=result))
    return ToolResult(True, result.text, {"agent": agent_name, ...})
```

Nesting invariants:

- `await child.aclose()` runs inside the inner `finally` so it always fires, but *before* the outer `except` catches — that way an `aclose` failure can't mask the original `child.run` exception (the outer `except` sees `child.run`'s exception, not `aclose`'s).
- The after-subagent hook still fires from exactly one of the two paths (success vs failure), unchanged from today.
- `aclose()` is unconditional. When `_owns_model=False` (shared parent), `Harness.aclose()` is a no-op for the provider (per Step 7), so calling it always is safe and keeps the control flow simple. When `_owns_model=True` (override child), it closes the freshly-inferred provider's `httpx.AsyncClient` inside the parent's loop — which is correct because the child created that client inside the same loop on its first request.

### Provider ownership across parent/child boundaries

Child harnesses commonly share the parent's `Model` instance (`child_model = parent.model` at `subagents.py:189`). If the child were to call `aclose()` on that shared provider, it would close the parent's HTTP client mid-run.

Current `build_child_harness` always passes `model=child_model` to the child constructor (`subagents.py:200-202`), regardless of whether `child_model` is the shared parent reference or a fresh `infer_model(...)` from the override path (line 192). So the naive rule "`_owns_model = (model is None)`" gets the override case exactly backwards: an override child looks non-owning because it received `model=`, even though it created that model itself.

Resolution: `Harness.__init__` takes an explicit `_owns_model: bool | None = None` kwarg. When `None`, it defaults to `model is None`. `build_child_harness` sets it explicitly:

```python
return Harness(
    child_config,
    model=child_model,
    _owns_model=(config is not None and config.model is not None),  # True only for override
    ...
)
```

Shared-parent path → `False`. Override path → `True`. The flag and rule are detailed in Step 7.

## Step 6 — Hooks

`hooks.py` stays mostly synchronous — hooks fire on already-collected context objects, no I/O. But:

- `HookRegistry.fire()` stays sync — call it from async code freely (hooks shouldn't do network I/O; if they do, that's the user's problem to wrap).
- For hooks that *do* want to await, add `async def fire_async(...)` later if asked. Not in v1.

## Step 7 — Provider lifecycle

### Contract

Two layers of ownership, because the harness and the provider can each own (or not own) their resource independently:

**Provider client ownership** (forced by httpx: `AsyncClient` binds to the loop it was first used on):

- Provider lazy-creates the client on first request and stores `_owns_client = True`.
- An injected `Provider(http_client=...)` sets `_owns_client = False`. `Provider.aclose()` only closes clients it owns.
- `Provider.aclose()` is idempotent.

**Harness model ownership** (new — addresses the subagent-sharing case):

- A `Harness` constructed *without* a `model=` argument infers the model itself and owns the provider. `_owns_model` defaults to `True`.
- A `Harness` constructed *with* `model=` defaults to non-owning (`_owns_model=False`). This covers child harnesses sharing `parent.model` and external callers passing in their own `Model`.
- Callers that build a `Model` and want the harness to own its lifecycle (the subagent-override case) pass `_owns_model=True` explicitly. `subagents.py:build_child_harness` does this when `config.model is not None`.
- The single `_owns_model` flag gates `Harness.aclose()`: when `False`, `aclose` is a no-op for the provider.

Operational consequences:

- **Async usage** (`async with Harness() as h: await h.run(...)`): top-level harness creates and persists the client across `await h.run(...)` calls in that loop; closed on `__aexit__`.
- **`run_sync`**: harness creates the client inside the wrapper's `asyncio.run` loop and closes it before that loop tears down (see Step 3 sample). Reuse across `run_sync` calls is impossible because the previous loop is dead.
- **Subagents sharing `parent.model`**: child `aclose()` is a no-op for the shared provider; the parent's client stays alive for the rest of the parent run.
- **Injected client**: never closed by anything except the injecting caller.

### Implementation

```python
class Harness:
    def __init__(self, config=None, *, model=None, _owns_model: bool | None = None, ...):
        self.model = model or infer_model(...)
        self._owns_model = _owns_model if _owns_model is not None else (model is None)
        ...

    async def aclose(self) -> None:
        """Close provider HTTP clients if this harness owns them. Idempotent."""
        if not self._owns_model:
            return
        aclose = getattr(self.model.provider, "aclose", None)
        if aclose is not None:
            await aclose()

    async def __aenter__(self): return self
    async def __aexit__(self, *exc): await self.aclose()
```

The `getattr(self.model.provider, "aclose", None)` check covers custom `Model` implementations whose provider isn't a `Provider` subclass (test fakes, userland adapters). Tightening the `Model` Protocol to require `aclose()` would be cleaner but forces every fake to grow a no-op method. The defensive lookup keeps the door open without complicating the contract.

`build_child_harness` (`subagents.py:200`) updates to pass `_owns_model=(config is not None and config.model is not None)`. Shared-parent children remain non-owning; override children own and close their freshly-inferred provider when the child run finishes (the subagent tool handler `await`s the child's `aclose()` in a `finally` — see Step 5).

## Step 8 — Tests

Test infrastructure:

- `tests/fakes.py` conversion is broader than just `FakeModel`. All of these are called by real sessions or by tests that exercise the provider HTTP path, and all need to become async (or get replaced):
  - `FakeModel` session methods (`start`/`continue_with_tools`) → `async def`.
  - `FakeClient.create_response` (`tests/fakes.py:25`) → `async def`.
  - `FakeAnthropicProvider.create_message` (`tests/fakes.py:124`) → `async def`.
  - `FakeOpenRouterProvider.create_chat_completion` (`tests/fakes.py:140`) → `async def`.
  - `MultiCallClient.create_response` (`tests/fakes.py:233`) → `async def`.
  - `test_providers.py:74/107` monkeypatches `_post_json`, which no longer exists. Replace with `httpx.MockTransport` injected via `Provider(http_client=httpx.AsyncClient(transport=...))` — cleaner than `respx` and uses the new injection path under test.
- Add `pytest-asyncio` to `dev` extras with `asyncio_mode = "auto"` (set in Step 1). The implementation is asyncio-native (`asyncio.create_task`, `asyncio.Semaphore`, `asyncio.wait`, `asyncio.to_thread`), so claiming runtime neutrality via `pytest-anyio` would be a fiction — it would also parametrize tests against trio, which the harness doesn't support.
- In `auto` mode, `pytest-asyncio` only wraps `async def` tests and fixtures. Sync tests stay sync, and the current suite has no project-level `@pytest.fixture` definitions (it relies on builtin `tmp_path`), so the auto-mode flip is safe — no existing tests or fixtures need rewriting beyond the explicit conversions in this Step.

Test split (don't blanket-convert; `run_sync` is part of the public API and deserves real coverage):

- **Convert to async** (`async def test_foo` + `await harness.run(...)`): provider/session tests, core loop tests, anything that needs async assertions (e.g. asserting tasks are cancelled, asserting client is closed).
- **Keep sync via `run_sync`**: representative harness, file-tool, hook, and subagent tests. Exercises the sync wrapper end-to-end.

New tests:

- `async with Harness() as h:` lifecycle, including `aclose()` idempotency and that a second `aclose()` is a no-op.
- Injected `httpx.AsyncClient` is never closed by `Harness.aclose()` or `run_sync`. Exercise through at least one concrete provider (`OpenAIProvider(http_client=...)`) so the concrete constructor wiring is covered, not just the base class.
- Child harness sharing `parent.model` does *not* close the parent's provider on subagent completion. Child harness with a model override *does* close its own provider after the child run (regression for the `_owns_model` flag — earlier `model is None` rule got this backwards because `build_child_harness` always passes `model=`).
- External cancellation: `task = asyncio.create_task(h.run(...))`, then `task.cancel()` mid-run. The `CancelledError` re-raises, pending tool tasks drain, `run_end` fires once with `error` set and `stop_reason="cancelled"` (the new external-cancel reason, distinct from `cancelled_by_hook`).
- Cancellation during `session.start` (provider HTTP in flight): same as above. `CancelledError` skips the inner `except Exception` arm and surfaces to the outer cancel arm at the agent-span level. `run_end` still fires once.
- Re-running the same `Harness` instance after an external cancellation succeeds. The outer `finally` resets `_running=False` even on `CancelledError`, so a subsequent `harness.run(...)` or `harness.run_sync(...)` works.
- `after_tool_call` does *not* fire when a tool is cancelled by an external `CancelledError` mid-`to_thread`, nor when a strict `before_tool_call` hook raises. It *does* fire for normal handler errors, normal `before.cancelled`, and successful tool calls.
- Concurrent subagent fanout: two parallel `subagent` calls; parent strict-hook abort cancels pending children and drains running ones without hanging.
- **Strict hook exception** during a tool batch cancels pending sibling tasks before they start, and the original exception surfaces promptly without waiting for slow async siblings to complete. Use an async handler that awaits on `asyncio.Event` so the cancellation point is testable (a sync handler running in a thread cannot be preempted — calling that out in the contract is intentional).
- **Normal `before.cancelled`** produces `ToolCallCancelled` for that one call; sibling calls complete; `usage.cancelled_tool_calls` increments. (Regression test — earlier plan draft inverted this behavior.)
- Tool batch outputs are returned in model-request order even when faster calls finish first.
- `run_sync` from inside a running loop raises `HarnessError`.
- Async tool handlers (`async def my_tool`) work end-to-end.
- Async tool handlers wrapped in `functools.partial` and decorated callables work end-to-end (regression for the `isawaitable`-on-result detection).
- Sync handlers run exactly once when invoked through `_invoke_tool` (regression for the thread-first dispatch — earlier draft executed sync handlers twice).
- Sync handler exceptions become structured failed tool outputs; `_thinharness_strict_hook` exceptions propagate. Same for async handlers.
- Public `call_tool(spec, args)` returns a structured `AsyncHandlerInSyncContext` error when handed an async-handler spec, without emitting an un-awaited-coroutine warning.
- `_CURRENT_TOOL_CALL` / tracing parent context is visible inside both a native-async handler and a sync handler dispatched through `asyncio.to_thread` (regression for subagent `parent_call_id`).
- HTTP status, transport timeout, and invalid-JSON failures all surface as `ProviderError` (mock with `respx` or `httpx.MockTransport`).

`vendor/pydantic-ai` tests use `vcrpy` for real-API recordings. We're smaller — stick with `respx` mocks of `httpx.AsyncClient` for v1. Real-API integration tests can come later.

## Step 9 — Docs

`README.md` rewrite:

```python
import asyncio
from thinharness import Harness, HarnessConfig

async def main():
    async with Harness(HarnessConfig(model="openai:gpt-5.2")) as h:
        result = await h.run("...")
        print(result.text)

asyncio.run(main())
```

Plus a "Synchronous usage" section showing `Harness(...).run_sync(...)`.

## Migration order

Repo is greenfield, so we skip the intermediate "sync harness calls `asyncio.run(...)` per provider request" states — they'd be slow, would break from active event loops (notebooks, async test runners), and exist only to keep tests green between commits we control. One focused conversion lands the spine, then follow-ups handle plumbing:

1. **Core async conversion** (one commit): add `httpx` dep, drop `urllib`, convert `Provider.post_json` + all `create_*` methods (including the concrete-provider `__init__` plumbing for `http_client`), convert `ModelSession.start`/`continue_with_tools`, convert `Harness.run` (including the `CancelledError` arm and new `"cancelled"` stop reason), add `Provider.aclose` + `Harness.aclose` + `_owns_model` (minimum lifecycle the `run_sync` body needs to compile), add `run_sync` with the `_run_and_close` body, convert *all* the async fakes from Step 8 plus the `_post_json` monkeypatch tests, replace `_run_calls_in_threads` with `_run_calls_concurrently`. Land `pytest-asyncio` + `asyncio_mode = "auto"` in `pyproject.toml` so the suite runs. This is a big commit, but every piece is load-bearing on every other — splitting it leaves the tree non-compiling.
2. **Lifecycle context manager + tests** (one commit): add `__aenter__`/`__aexit__`; add the broader new tests from Step 8 (idempotency, injected-client non-closure, override-child ownership, external cancellation, concurrent subagent fanout).
3. **Subagent handler conversion** (one commit): drop any `to_thread` indirection for child harnesses now that they're async; wire `_owns_model=True` through `build_child_harness` for the override path; add the `await child.aclose()` in the subagent handler's `finally`.
4. **Docs** (one commit): README rewrite.

Note: commit 1 is intentionally fat — the `run_sync` body calls `aclose()`, `aclose()` reads `_owns_model`, the fakes have to be async for any session test to pass, and the cancel arm has to exist before the cancellation test runs. Trying to land these incrementally would either leave the tree broken or require throwaway transitional code.

## Resolved decisions

1. **Async runtime**: `asyncio` only. No `anyio` dependency. `asyncio.to_thread` (Python 3.11+) propagates contextvars and is sufficient.
2. **Test runner**: `pytest-asyncio`. Implementation is asyncio-native; runtime neutrality via `pytest-anyio` would be a fiction.
3. **Shared client injection**: yes, via `Provider(http_client=...)`. Harness never closes injected clients.
4. **`run_sync` inside a running loop**: hard error. `nest_asyncio` is a footgun that hides re-entrancy bugs.
5. **Async hooks**: out of scope for v1. `HookRegistry.fire()` stays sync.
6. **Public `call_tool` with async handler**: returns a structured `AsyncHandlerInSyncContext` error envelope. Coroutine is `.close()`'d to suppress the un-awaited warning.
7. **In-flight sync handler cancellation**: not supported. Python threads can't be preempted. The fanout cancels pending tasks; running threaded handlers run to completion. Mutating built-ins are `sequential=True` and therefore never in a parallel batch.
8. **Pytest-asyncio mode**: `asyncio_mode = "auto"` in `pyproject.toml`. Avoids decorating every async test.
9. **`subagent` tool stays parallel**, not `sequential=True`. Concurrent subagents are a feature. Cross-subagent mutation serialization is the user's choice via `tool_execution="sequential"` on the parent.
10. **External cancellation**: new `StopReason` `"cancelled"`, distinct from `"cancelled_by_hook"`. `asyncio.CancelledError` propagates from `Harness.run` after `run_end` fires.
11. **`after_tool_call` firing**: success, per-tool failure, and normal `before.cancelled` all fire after-hooks. Strict-hook `before_tool_call` exceptions and external cancellation during `_invoke_tool` do *not* fire after-hooks.
12. **`asyncio.to_thread` contextvar direction**: one-way, loop → thread. Harness only sets `_CURRENT_TOOL_CALL` from the loop side in `_traced_call_output`.

## Out of scope

- Streaming responses (separate plan; the async foundation enables it but we're not building it here)
- Async hook variants
- `nest_asyncio` integration
- Shared connection pooling beyond "user can pass their own httpx client"
