# Plan: Runtime Context and Tool Execution Architecture

## Overview
Split `Harness.run()` into explicit runtime modules so the main loop reads as orchestration: start run, advance model, resolve output, execute tool batch, continue, finalize. The end state is a run context module for per-run state and model advancement ceremony, plus a tool execution module for batch and single-call lifecycle policy.

This plan intentionally assumes greenfield architecture. Backwards compatibility is not the priority, but existing behavior should remain unless a simpler ideal interface requires a deliberate test update.

## Decisions
- Create `thinharness/runtime.py` for internal run context and model advancement ceremony.
- Create `thinharness/tool_execution.py` for batch execution and one-call lifecycle.
- Move both batch and single-call execution out of `core.py`.
- Remove `_current_run_metadata` from `Harness` by passing run metadata explicitly through `RunContext` and tool/subagent execution.
- Keep provider-specific session methods in `providers.py`; runtime code should not know provider payload shapes.
- Write this as one implementation plan with two ordered phases: Phase A extracts run context, Phase B extracts tool execution.

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
    result: HarnessResult | None = None
    terminal_error: BaseException | None = None
    stop_reason: StopReason = "end_turn"
    run_end_fired: bool = False
```

Avoid making `RunContext` public in `__init__.py` unless the implementation reveals a strong reason. It is an internal module.

Move or wrap these responsibilities from `Harness.run()`:
- `fire_run_end_once`.
- `check_model_limit`.
- `check_tool_limit`.
- structured-output retry budget helper.
- tool retry budget helper.
- terminal result construction.
- resume state attachment.
- model-facing limit notice computation.
- `advance_model(...)` ceremony: limit check, notices, model tracing, request annotation, provider call, usage increment, response annotation, provider exception annotation.

`Harness.run()` should still choose which provider session method to call. It passes a callable into `RunContext.advance_model(...)`:

```python
turn = await run_ctx.advance_model(
    lambda notices: active_session.start(..., notices=notices),
    trace_snapshot=ModelTraceSnapshot(...),
)
```

That keeps provider-specific behavior in the session adapter while centralizing the repeated model-request ceremony.

**Verify:** After this step, run `uv run pytest tests/test_harness.py tests/test_hooks.py tests/test_resume.py tests/test_tracing.py`.

### 2. Refactor `Harness.run()` Around `RunContext`
Replace the local closure variables in `Harness.run()` with `run_ctx`.

The main loop should become structurally simple:

```python
run_ctx = RunContext(...)
with run_ctx.agent_span(...) as agent_span:
    run_ctx.fire_run_start(...)
    ...
    turn = await run_ctx.advance_model(...)
    while True:
        run_ctx.responses.append(turn.raw)
        decision = resolve_turn_output(...)
        ...
        recorded, outputs, executions = await self.tool_executor.execute_batch(run_ctx, turn.tool_calls)
        run_ctx.record_tool_batch(recorded, executions)
        run_ctx.check_tool_retry_limits(turn.tool_calls, executions)
        turn = await run_ctx.advance_model(...)
```

Do not preserve `_current_run_metadata` as an instance variable. Any code that needs metadata receives `run_ctx.metadata`.

Keep `self._running` and `self._closed` on `Harness`; those are harness lifecycle flags, not per-run state.

**Verify:** Add or update a test that proves a hook/tool/subagent still sees run metadata, then run `uv run pytest tests/test_hooks.py tests/test_subagents.py tests/test_tracing.py`.

### 3. Create `tool_execution.py` with Batch and Single-Call Executors
Add `thinharness/tool_execution.py`. It should own both batch-level policy and one-call lifecycle policy.

Recommended modules:

```python
@dataclass(frozen=True)
class ToolCallExecution:
    output: str
    cancelled: bool
    retry_kind: str | None = None
    parsed_output: Json | None = None
    trace_metadata: Json = field(default_factory=dict)

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

**Verify:** Run `uv run pytest tests/test_parallel_tools.py tests/test_tool_retry.py tests/test_hooks.py tests/test_tracing.py tests/test_mcp.py`.

### 4. Wire Tool Execution into `Harness`
In `Harness.__init__`, construct the tool execution module after tools, hooks, and MCP state are initialized enough for lookup.

Keep the dependency direction simple:
- `core.py` owns `Harness`.
- `tool_execution.py` can accept only the specific dependencies it needs: tool map, hooks, config, tracer, metadata, and call invoker.
- Avoid importing `Harness` at runtime in `tool_execution.py` unless needed for hook contexts. Use `TYPE_CHECKING` if possible.

One acceptable shape is to instantiate executors per run because they need `RunContext`:

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

Another acceptable shape is a lightweight factory on `Harness`. Pick whichever keeps imports and state easiest to understand.

Remove these methods from `Harness` once replaced:
- `_traced_call_output`.
- `_execute_tool_batch`.
- `_should_run_sequentially`.
- `_run_calls_concurrently`.

Keep `_call_output` on `Harness` only if it remains the cleanest way to preserve unknown-tool behavior and custom tool invocation. Otherwise move it into `ToolCallExecutor` as a private helper.

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
- Subagent execution receives parent metadata from the tool call lifecycle, not from parent instance state.
- `_CURRENT_TOOL_CALL` can remain for parent call id unless the new tool execution result makes explicit parent call id easier.

If the cleanest implementation requires changing `run_subagent_tool(parent, configs, args)` to accept optional runtime metadata, do that and update the subagent tool handler factory accordingly.

**Verify:** Run `uv run pytest tests/test_subagents.py tests/test_hooks.py tests/test_tracing.py tests/test_mcp.py`.

### 6. Update Tests to Target the New Modules
Add focused tests for the new modules where useful, but avoid duplicating all integration tests.

Recommended test updates:
- A direct `RunContext` test for `run_end` firing once under success and error if this can be done without awkward fakes.
- A direct `ToolBatchExecutor` test for order preservation and sequential fallback if existing integration tests become too indirect.
- Existing integration tests remain the main proof for hooks, cancellation, tracing, MCP attribution, subagents, and retry budgets.

Prefer high-signal tests that would fail if policy leaks back into `core.py`.

**Verify:** Run pass-level validation commands.

### 7. Update Architecture Docs If the New Shape Lands Cleanly
Update `docs/architecture.md` after implementation:
- Add `runtime.py` as the owner of per-run state and model advancement ceremony.
- Add `tool_execution.py` as the owner of batch and one-call tool execution.
- Describe `core.py` as the high-level run orchestrator.

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
4. Remove old `Harness` tool-execution methods and any transitional metadata path.

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
uv run pytest tests/test_harness.py tests/test_hooks.py tests/test_tool_retry.py tests/test_parallel_tools.py tests/test_subagents.py tests/test_tracing.py tests/test_mcp.py tests/test_resume.py
uv run ruff check .
uv run pyright
```

If Pass 1 has not landed yet, also run:

```bash
uv run pytest tests/test_structured_output.py tests/test_parallel_llm.py
```

## Do Not Touch
- Do not redesign structured-output decisions in this pass; use the Pass 1 result as the output interface.
- Do not change provider payload translation.
- Do not refactor `FileTools` / `JsonlSearch`.
- Do not refactor `ModelSession` request shape beyond what `RunContext.advance_model(...)` needs.
- Do not export `RunContext`, `ToolBatchExecutor`, or `ToolCallExecutor` publicly unless the implementation proves they should be public.

## Considerations
- Avoid a new god module. `runtime.py` should own per-run state and model advancement ceremony, while `tool_execution.py` owns tool execution. If either module starts importing everything from `core.py`, narrow the constructor inputs.
- The hardest part is strict hook exceptions and cancellation across concurrent tool calls. Preserve existing tests before simplifying control flow.
- Passing metadata explicitly may require small signature changes in subagent helpers. That is acceptable in this greenfield project, but keep the resulting interface simpler than the instance-state workaround it replaces.
- If circular imports appear, use `TYPE_CHECKING`, move tiny shared dataclasses to the module that owns their behavior, or pass callbacks instead of importing `Harness`.
