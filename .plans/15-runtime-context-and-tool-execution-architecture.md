# Plan: Runtime Context and Tool Execution Architecture

## Overview
Split `Harness.run()` into explicit runtime modules so the main loop reads as orchestration: start run, advance model, execute tool batch, continue, finalize. The end state is a run context module for per-run state and model advancement ceremony, plus a tool execution module for batch and single-call lifecycle policy.

This is a behavior-preserving internal refactor. Public API, provider request shapes, hook semantics, tracing semantics, metadata semantics, resume behavior, and retry accounting should remain unchanged unless a separate explicit decision approves a behavior change.

## Decisions
- Create `thinharness/runtime.py` for internal run context and model advancement ceremony.
- Create `thinharness/tool_execution.py` for batch execution and one-call lifecycle.
- Move both batch and single-call execution out of `core.py`.
- Remove `_current_run_metadata` from `Harness` by passing run metadata explicitly through `RunContext` and tool/subagent execution.
- Keep provider-specific session methods in `providers.py`; runtime code should not know provider payload shapes.
- Write this as one implementation plan with two ordered phases: Phase A extracts run context, Phase B extracts tool execution.
- Keep `resolve_turn_output(...)` inside `RunContext.advance_model(...)` so output finalization attributes are written before the model span closes.
- Keep the public `current_tool_call_context()` return shape behavior-preserving: callers should still observe only `{"call_id": ..., "name": ...}`.
- Add an internal per-call runtime context channel in `hooks.py` for copied run metadata. This may be a separate private contextvar or a private helper that filters a richer context value before `current_tool_call_context()` returns it.
- Preserve existing direct imports unless tests are updated in the same pass. In particular, keep `_compute_limit_notices` importable from `thinharness.core` for the direct structured-output tests.

## Steps

### 1. Create `runtime.py` with `RunContext`
Add `thinharness/runtime.py` as an internal module. It should own per-run mutable state and the model advancement ceremony currently captured by closures inside `Harness.run()`.

A reasonable starting shape:

```python
@dataclass
class RunContext:
    harness: Harness
    prompt: str
    metadata: Json
    usage: RunUsage = field(default_factory=RunUsage)
    responses: list[Json] = field(default_factory=list)
    tool_call_records: list[Json] = field(default_factory=list)
    emitted_limit_warnings: set[LimitNoticeKey] = field(default_factory=set)
    tracer: RunTracer | None = None
    result: HarnessResult | None = None
    terminal_error: BaseException | None = None
    stop_reason: StopReason = "end_turn"
    run_end_fired: bool = False
    finalized_via_output_tool: bool = False
```

Avoid making `RunContext` public in `__init__.py` unless the implementation reveals a strong reason. It is an internal module.

Move or wrap these responsibilities from `Harness.run()`:
- `fire_run_end_once`.
- `check_model_limit`.
- `check_tool_limit`.
- `retry_or_fail()` for structured-output retry budget accounting.
- `check_tool_retry_limits(...)` for retryable tool failure accounting.
- terminal result construction.
- resume state attachment.
- model-facing limit notice computation.
- `advance_model(...)` ceremony: limit check, notices, output retry accounting, model tracing, request annotation, provider call, usage increment, response annotation, output resolution, output finalization span annotation, provider exception annotation.

`Harness.run()` should still choose which provider session method to call. It passes a callable into `RunContext.advance_model(...)`:

```python
turn, decision = await run_ctx.advance_model(
    lambda notices: active_session.start(..., notices=notices),
    trace_snapshot=ModelTraceSnapshot(...),
    output_retry=False,
)
```

That keeps provider-specific behavior in the session adapter while centralizing the repeated model-request ceremony.

`advance_model(...)` must keep the current `output_retry: bool = False` parameter behavior and increment `usage.output_retries` for structured-output correction turns. It should return both the `ModelTurn` and `OutputTurnDecision`, matching the existing span lifetime: `resolve_turn_output(...)` happens while the model span is still open.

`RunContext` needs access to the per-run `RunTracer` and the active agent span. Either store them on `RunContext` after `run_tracer.agent(...)` is entered, or keep finalization as a small closure in `Harness.run()` that captures the agent span. Do not lose current agent-span result annotation or error annotation behavior.

Avoid circular imports deliberately:
- Keep public dataclasses and exceptions that are already exported from `thinharness.core` importable from there.
- Use `from __future__ import annotations` in `runtime.py`.
- Import `Harness` only under `TYPE_CHECKING` in `runtime.py`; `RunContext.harness` can be annotated without a runtime import.
- Import `RunContext` locally inside `Harness.run()` if a top-level import would create a cycle.
- Keep `_compute_limit_notices` importable from `core.py`. If its implementation moves, leave a compatibility import or wrapper in `core.py`.

**Verify:** After this step, run `uv run pytest tests/test_harness.py tests/test_hooks.py tests/test_resume.py tests/test_tracing.py tests/test_structured_output.py`.

### 2. Refactor `Harness.run()` Around `RunContext`
Replace the local closure variables in `Harness.run()` with `run_ctx`.

The main loop should become structurally simple:

```python
run_ctx = RunContext(...)
with run_ctx.agent_span(...) as agent_span:
    run_ctx.fire_run_start(...)
    ...
    turn, decision = await run_ctx.advance_model(...)
    while True:
        run_ctx.responses.append(turn.raw)
        ...
        run_ctx.check_tool_limit(len(turn.tool_calls))
        run_ctx.usage.tool_calls += len(turn.tool_calls)
        recorded, outputs, executions = await self.tool_executor.execute_batch(run_ctx, turn.tool_calls)
        run_ctx.record_tool_batch(recorded, executions)
        run_ctx.usage.cancelled_tool_calls += sum(1 for execution in executions if execution.cancelled)
        run_ctx.check_tool_retry_limits(turn.tool_calls, executions)
        turn, decision = await run_ctx.advance_model(...)
```

Do not preserve `_current_run_metadata` as an instance variable. Any code that needs metadata receives `run_ctx.metadata`.

Keep `self._running` and `self._closed` on `Harness`; those are harness lifecycle flags, not per-run state.

Keep these existing control-flow details visible in the refactor:
- `await self._ensure_mcp_connected()` still runs before the first model request.
- The first-turn branch still chooses `active_session.start(...)` vs `active_session.continue_with_user_prompt(...)`; `RunContext.advance_model(...)` should not know provider session payload shapes.
- `fire_run_end_once()` still runs in an inner `finally`, while `self._running = False` remains in an outer `finally`, so a strict `run_end` hook cannot leave the harness stuck as running.
- Provider-facing `metadata` remains the original run metadata argument, while hook-facing metadata receives a fresh `dict(run_ctx.metadata)` copy for each hook fire.
- `check_tool_limit(len(turn.tool_calls))` and `usage.tool_calls += len(turn.tool_calls)` stay before batch execution, so attempted model-requested calls still count if a strict hook or unexpected tool exception aborts the batch.
- `usage.cancelled_tool_calls` is updated after batch execution returns. If the batch raises before returning, keep current behavior and do not invent partial accounting.

**Verify:** Add or update a test that proves a hook/tool/subagent still sees run metadata. Add or keep coverage proving strict hook failures still leave attempted tool calls counted in `run_end` usage. Then run `uv run pytest tests/test_hooks.py tests/test_subagents.py tests/test_tracing.py tests/test_structured_output.py`.

### 3. Create `tool_execution.py` with Batch and Single-Call Executors
Add `thinharness/tool_execution.py`. It should own both batch-level policy and one-call lifecycle policy.

Recommended modules:

```python
@dataclass(frozen=True)
class ToolCallExecution:
    output: str
    cancelled: bool
    retry_kind: str | None = None

class ToolBatchExecutor:
    async def execute_batch(self, calls: list[ModelToolCall]) -> tuple[list[Json], list[ToolOutput], list[ToolCallExecution]]:
        ...

class ToolCallExecutor:
    async def execute_one(self, call: ModelToolCall, index: int) -> ToolCallExecution:
        ...
```

`ToolBatchExecutor` owns:
- Sequential vs concurrent choice.
- Whole-batch sequential fallback if any called tool has `sequential=True`.
- Concurrency limit.
- Sibling cancellation when a strict hook or unexpected exception escapes.
- Preserving model call order.
- Building `tool_call_records`.
- Building provider `ToolOutput` values.

`ToolCallExecutor` owns:
- Setting `_CURRENT_TOOL_CALL`.
- Firing `BeforeToolCallContext`.
- Handling hook cancellation output.
- Invoking the tool through existing `_invoke_tool(...)`.
- Parsing normalized output.
- Capturing retry kind before `after_tool_call`.
- Firing `AfterToolCallContext`.
- Refreshing parsed output after hook mutation.
- Preparing trace attributes for MCP and subagent calls.
- Returning one `ToolCallExecution`.

`core.py` should not need to know MCP or subagent trace attribution details after this move.

Move these tool-execution helpers with the behavior they support:
- `ToolCallExecution`.
- `_parse_tool_output`.
- `_tool_retry_kind`.
- `MAX_PARALLEL_TOOL_WORKERS` if it remains only tool-execution policy.

`ToolCallExecutor.execute_one()` is the sole writer for per-call context: set it before hooks/tool invocation and reset it in `finally`. Preserve `current_tool_call_context()` as the public reader for `{"call_id": call_id, "name": name}`. Add a separate internal reader for runtime metadata, such as `current_tool_runtime_context()`, returning `{"run_metadata": dict(run_ctx.metadata)}`. Readers such as `subagents.py` should use the public helper for parent call id and the internal runtime helper for metadata.

**Verify:** Run `uv run pytest tests/test_parallel_tools.py tests/test_tool_retry.py tests/test_hooks.py tests/test_tracing.py tests/test_mcp.py tests/test_harness.py`.

### 4. Wire Tool Execution into `Harness`
Instantiate the tool execution module per run, after `run_ctx` and the per-run tracer are available. Do not construct a persistent executor in `Harness.__init__` if it needs run-specific state.

Keep the dependency direction simple:
- `core.py` owns `Harness`.
- `tool_execution.py` accepts focused runtime dependencies: harness instance for hook context values, tool map, hooks, config values, run tracer, run metadata, and an optional call invoker.
- Avoid importing `Harness` at runtime in `tool_execution.py`; use `TYPE_CHECKING` for annotations and pass the instance as an object dependency.

Use this shape unless implementation proves it awkward:

```python
executor = ToolBatchExecutor(
    harness=self,
    run_context=run_ctx,
    tool_map=self._tool_map,
    hooks=self.hooks,
    run_tracer=run_ctx.tracer,
    tool_execution=self.config.tool_execution,
)
```

Avoid mixing a persistent `self.tool_executor` with a captured per-run `RunContext`. If a lightweight factory on `Harness` is useful, it should still create a new executor for each run.

Remove these methods from `Harness` once replaced:
- `_traced_call_output`.
- `_execute_tool_batch`.
- `_should_run_sequentially`.
- `_run_calls_concurrently`.

Move `_call_output` into `ToolCallExecutor` as a private helper unless doing so creates an import cycle. Preserve unknown-tool behavior and invocation through the existing `_invoke_tool(...)` library function.

**Verify:** Run the same tool execution tests plus `uv run pytest tests/test_harness.py`.

### 5. Remove `_current_run_metadata` and Pass Metadata Explicitly
Update `subagents.py` and tool execution so child metadata no longer depends on `getattr(parent, "_current_run_metadata", None)`.

Current friction:
- `Harness.run()` writes `self._current_run_metadata`.
- Tool execution reads it for hook contexts.
- `subagents.py` reads it to build child metadata and hook metadata.

End state:
- `RunContext.metadata` is passed to tool execution.
- `ToolCallExecutor` builds hook contexts with that metadata.
- `ToolCallExecutor` stores `run_metadata=dict(run_ctx.metadata)` in the internal runtime context for the duration of the tool call.
- Subagent execution reads parent hook metadata from the internal runtime context, not from parent instance state.
- Subagent execution reads parent call id from `current_tool_call_context()`.

Do not rebuild subagent tool handlers per run. They are created during `Harness.__init__` and only receive tool args at invocation time. The runtime metadata injection point is the tool-call context set by `ToolCallExecutor`.

When firing hook contexts, keep the current defensive-copy behavior. Each `BeforeToolCallContext`, `AfterToolCallContext`, `BeforeSubagentRunContext`, and `AfterSubagentRunContext` should receive a new `dict(...)` copy so hook mutation cannot leak into later hooks or child metadata.

Preserve current child run metadata projection. Child provider metadata should still include only:
- `conversation_id` from parent run metadata, when present.
- `parent_call_id` from the active tool call, when present.

Do not leak arbitrary parent metadata keys into child provider metadata. Subagent hook contexts can receive full copied run metadata, matching current hook behavior.

Metadata reads must remain null-safe. Subagent helpers should tolerate missing runtime context with a fallback equivalent to `{}`.

**Verify:** Add or update a regression test proving mutation of one hook context's metadata does not leak into later tool/subagent hook metadata. Keep or add coverage proving extra parent metadata does not leak into child provider metadata. Then run `uv run pytest tests/test_subagents.py tests/test_hooks.py tests/test_tracing.py tests/test_mcp.py`.

### 6. Update Tests to Target the New Modules
Add focused tests for the new modules where useful, but avoid duplicating all integration tests.

Recommended test updates:
- A direct `RunContext` test for `run_end` firing once under success and error if this can be done without awkward fakes.
- A direct `ToolBatchExecutor` test for order preservation and sequential fallback if existing integration tests become too indirect.
- A structured-output tracing regression if existing coverage no longer directly proves finalization attributes are written on the model span before it closes.
- A `current_tool_call_context()` regression preserving the public `{"call_id", "name"}` shape while proving internal runtime metadata is available to subagent helpers.
- Existing integration tests remain the main proof for hooks, cancellation, tracing, MCP attribution, subagents, and retry budgets.

Prefer high-signal tests that would fail if policy leaks back into `core.py`.

**Verify:** Run pass-level validation commands.

### 7. Update Architecture Docs If the New Shape Lands Cleanly
Update `docs/architecture.md` after implementation:
- Add `runtime.py` as the owner of per-run state and model advancement ceremony.
- Add `tool_execution.py` as the owner of batch and one-call tool execution.
- Describe `core.py` as the high-level run orchestrator.
- Update the module tree and any `core.py` mechanics section that still lists moved helpers such as `fire_run_end_once`, `advance_model`, retry helpers, or `_execute_tool_batch`.

Update `docs/decisions.md` only if a decision changes, such as public behavior around tool metadata or subagent metadata. Do not rewrite old decisions just because files moved.

**Verify:** Documentation references real file names and current flow.

## Phase Order

### Phase A: Run Context
1. Create `runtime.py`.
2. Move run bookkeeping and model advancement ceremony.
3. Refactor `Harness.run()` to use `RunContext`.
4. Remove `_current_run_metadata` only if tool/subagent paths are ready; otherwise leave a temporary compatibility path documented in the code as transitional and remove it in Phase B.

### Phase B: Tool Execution
1. Create `tool_execution.py`.
2. Move batch execution and one-call lifecycle.
3. Pass `RunContext.metadata` into tool and subagent execution.
4. Add an internal per-call runtime metadata channel for subagent hooks and child run metadata while preserving public `current_tool_call_context()` shape.
5. Remove old `Harness` tool-execution methods and any transitional metadata path.

Phase A and Phase B are in one plan because they touch the same internal state. Implement in order; do not parallelize across agents unless write scopes are split very carefully.

## Test Strategy

### 1. Run Context Lifecycle Tests
- **Purpose:** Prove per-run state, limits, model advancement, tracing, and terminal bookkeeping stayed correct.
- **Tests:** Run start/end, provider errors, cancellation, limit reached, resume state, model request usage, notice emission.
- **How:** Existing scripted-model tests in `tests/test_harness.py`, `tests/test_hooks.py`, `tests/test_resume.py`, and `tests/test_tracing.py`.
- **Likely misses:** Tool-call ordering and hook mutation details.

### 2. Tool Execution Policy Tests
- **Purpose:** Prove batch and one-call execution policy moved without losing semantics.
- **Tests:** Parallel execution, sequential fallback, order preservation, strict hook cancellation, retry-kind capture before after-hooks, after-hook output mutation, unknown tools, MCP attribution, subagent attribution.
- **How:** Existing tests in `tests/test_parallel_tools.py`, `tests/test_tool_retry.py`, `tests/test_hooks.py`, `tests/test_tracing.py`, and `tests/test_mcp.py`.

### 3. Subagent Metadata Tests
- **Purpose:** Prove removing `_current_run_metadata` did not break child run context or hook metadata.
- **Tests:** Parent call id propagation, conversation id propagation, before/after subagent hooks, child tracing nesting.
- **How:** `tests/test_subagents.py`, `tests/test_tracing.py`, and MCP subagent tests.

## Spec Coverage Map
- Run bookkeeping locality -> Run Context Lifecycle Tests.
- Model advancement ceremony -> Run Context Lifecycle Tests.
- Tool batch policy -> Tool Execution Policy Tests.
- One-call hook/retry/trace lifecycle -> Tool Execution Policy Tests.
- Explicit run metadata flow -> Subagent Metadata Tests.

## Validation Commands
Run these before handing off as complete:

```bash
uv run pytest tests/test_harness.py tests/test_hooks.py tests/test_tool_retry.py tests/test_parallel_tools.py tests/test_subagents.py tests/test_tracing.py tests/test_mcp.py tests/test_resume.py tests/test_structured_output.py tests/test_parallel_llm.py
uv run ruff check .
uv run pyright
```

## Do Not Touch
- Do not redesign structured-output decisions in this pass; use the Pass 1 result as the output interface.
- Do not change provider payload translation.
- Do not refactor `FileTools` / `JsonlSearch`.
- Do not refactor `ModelSession` request shape beyond what `RunContext.advance_model(...)` needs.
- Do not export `RunContext`, `ToolBatchExecutor`, or `ToolCallExecutor` publicly unless the implementation proves they should be public.
- Do not change public behavior as part of this refactor without first updating this plan with the explicit behavior decision.

## Considerations
- Avoid a new god module. `runtime.py` should own per-run state and model advancement ceremony, while `tool_execution.py` owns tool execution. If either module starts importing everything from `core.py`, narrow the constructor inputs.
- The hardest part is strict hook exceptions and cancellation across concurrent tool calls. Preserve existing tests before simplifying control flow.
- Passing metadata explicitly means passing it through `RunContext` into tool execution, then through an internal per-call runtime metadata context into subagent helpers. Avoid a harness instance field for current-run metadata.
- If circular imports appear, use `TYPE_CHECKING`, local imports, move tiny shared dataclasses to the module that owns their behavior, or pass callbacks instead of importing `Harness`.
