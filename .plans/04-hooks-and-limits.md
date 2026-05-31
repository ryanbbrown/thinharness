# Plan: Hooks and Explicit Run Limits

## Overview
Add a small lifecycle hook system plus explicit model-request and tool-call limits to the harness. The implementation should keep `core.py` focused on orchestration, put hook types and dispatch in `thinharness/hooks.py`, and treat subagent runs as normal nested harness runs with extra semantic hook events at the delegation boundary. Near-limit model guidance is intentionally split into follow-up work because it changes provider message construction.

Existing related plans: `.plans/03-subagents.md` and `.plans/subagents-fork-mode.md`.

Implementation order should follow the natural fault lines, with tests run after each phase before continuing:
1. Hooks core: `hooks.py`, runtime registration, `run_start`, `user_prompt_submit`, `before_tool_call`, and `after_tool_call`.
2. Hard limits and result shape: replace `max_turns`, add `max_model_requests`, `max_tool_calls`, `limit_reached`, `RunUsage`, and `tool_call_records`.
3. Subagent hooks and docs: add `before_subagent_run`, `after_subagent_run`, `subagent_hooks`, child hook behavior, and README updates.

These phases are for implementation and verification order, not required user checkpoints. Continue through the phases in one work session when the tests are green.

## Steps

### 1. Add a Hook Module and Public API
Create `thinharness/hooks.py` to own hook registration, filtering, context objects, and dispatch behavior.

Suggested shape:

```python
from __future__ import annotations

from typing import ClassVar

HookEvent = Literal[
    "run_start",
    "user_prompt_submit",
    "before_tool_call",
    "after_tool_call",
    "before_subagent_run",
    "after_subagent_run",
    "run_end",
    "limit_reached",
]

HookHandler = Callable[["HookContext"], None]

@dataclass(frozen=True)
class Hook:
    """One lifecycle hook registration."""

    event: HookEvent
    handler: HookHandler
    tools: list[str] | None = None
    agents: list[str] | None = None
```

Use dataclasses for hooks and runtime hook contexts. These objects contain callables and mutable runtime payloads, not serialized config, and contexts will be created on every hook dispatch. Keep Pydantic for config objects such as `HarnessConfig` and `SubAgentConfig`.

`HookContext` should be a mutable base class with stable common fields:

```python
@dataclass
class HookContext:
    """Mutable lifecycle context passed to hook handlers."""

    event: ClassVar[HookEvent]
    harness: "Harness"
    metadata: Json = field(default_factory=dict)
```

Use `TYPE_CHECKING` imports in `hooks.py` to reference `Harness` during static analysis without creating a runtime import cycle.

Use one event-specific subclass per hook event instead of a single untyped `data: Json` bag or one giant optional-field context. This keeps hook handlers typed and makes each event's payload obvious:

```python
@dataclass(kw_only=True)
class RunStartContext(HookContext):
    """Context for a run before the first model request."""

    event: ClassVar[HookEvent] = "run_start"
    prompt: str
    root: Path
    max_model_requests: int
    max_tool_calls: int | None = None


@dataclass(kw_only=True)
class UserPromptSubmitContext(HookContext):
    """Context for the submitted user prompt before querying the model."""

    event: ClassVar[HookEvent] = "user_prompt_submit"
    prompt: str
    additional_context: list[str] = field(default_factory=list)
    cancelled: bool = False
    cancel_reason: str = ""


@dataclass(kw_only=True)
class BeforeToolCallContext(HookContext):
    """Context for a tool call before it is executed."""

    event: ClassVar[HookEvent] = "before_tool_call"
    call_id: str
    tool_name: str
    arguments: str
    tool_spec: ToolSpec | None
    tool_index: int
    cancelled: bool = False
    cancel_reason: str = ""


@dataclass(kw_only=True)
class AfterToolCallContext(HookContext):
    """Context for a completed tool call before the output is finalized."""

    event: ClassVar[HookEvent] = "after_tool_call"
    call_id: str
    tool_name: str
    arguments: str
    original_output: str
    output: str
    parsed_output: Json | None = None
    duration_ms: float


@dataclass(kw_only=True)
class BeforeSubagentRunContext(HookContext):
    """Context for a subagent run before the child harness is built."""

    event: ClassVar[HookEvent] = "before_subagent_run"
    agent: str
    task: str
    inherited: bool
    tool_mode: str
    parent_harness: "Harness"
    parent_call_id: str | None = None
    cancelled: bool = False
    cancel_reason: str = ""


@dataclass(kw_only=True)
class AfterSubagentRunContext(HookContext):
    """Context for a completed or failed subagent run."""

    event: ClassVar[HookEvent] = "after_subagent_run"
    agent: str
    task: str
    result: HarnessResult | None = None
    error: BaseException | None = None
    tools: list[str] = field(default_factory=list)
    usage: RunUsage | None = None
    parent_call_id: str | None = None


@dataclass(kw_only=True)
class RunEndContext(HookContext):
    """Context for the terminal run outcome."""

    event: ClassVar[HookEvent] = "run_end"
    result: HarnessResult | None = None
    error: BaseException | None = None
    stop_reason: StopReason = "end_turn"
    usage: RunUsage | None = None
```

Define `LimitReachedContext` with the limit implementation in Step 5. Context subclasses use class-level `event` constants so hook authors cannot construct mismatched event/context pairs.

Do not add public hooks around raw model/provider requests in v1. Claude Code exposes stable product events such as prompt submit, tool use, subagent start/stop, and stop; it does not expose a configurable "before LLM request" hook. Keep provider request construction as internal orchestration so hooks do not need to understand provider-specific continuation state.

`user_prompt_submit` should not rewrite the original prompt directly. Hooks that want to affect the first model request should append strings to `ctx.additional_context`; the harness appends those as clearly labeled extra context after the submitted prompt. If a prompt guardrail needs to block the run before any model request, set `ctx.cancelled=True`.

Use a deterministic prompt injection format when `additional_context` is present:

```text
{original_prompt}

<hook_context>
{context_1}

{context_2}
</hook_context>
```

Keep the exact wrapper in one helper so tests can assert the final prompt string.

Add a tiny `HookRegistry`:

```python
class HookRegistry:
    """Dispatch lifecycle hooks in registration order."""

    def __init__(self, hooks: list[Hook] | None = None) -> None: ...
    def fire(self, ctx: HookContext) -> None: ...
```

Filtering rules:
- `tools=None` means all tools; `tools=["read", "search"]` only fires for those tool names.
- `agents=None` means all subagents; `agents=["default", "research"]` only fires for those subagent names.
- Empty filter lists are invalid. Use `None` for "all" and omit hooks that should never fire.
- Filters compare against final registered tool names and final subagent names exactly. They are case-sensitive; `tools=["Read"]` does not match the built-in `read` tool.
- Tool filters apply only to tool events.
- Agent filters apply only to subagent events.
- Reject invalid filters during hook registration: tool filters are valid only for tool events, and agent filters are valid only for subagent events. Do not silently ignore nonsensical filters such as `Hook(event="run_start", tools=["read"])` or `Hook(event="limit_reached", agents=["research"])`.
- Unmatched filter names should warn once after current tool and subagent registration is known. Do not raise, because users may still add tools later with `add_tool(...)`; exact matching remains the runtime rule.
- Hook handler exceptions should be logged with `logger.warning(...)`, including the event name and handler qualified name, so broken observability hooks do not disappear silently. Also log the traceback at debug level for diagnosis without requiring `strict_hooks=True`. When `strict_hooks=True`, re-raise handler exceptions after logging.
- Only `UserPromptSubmitContext`, `BeforeToolCallContext`, and `BeforeSubagentRunContext` expose `cancelled` and `cancel_reason`; only `user_prompt_submit`, `before_tool_call`, and `before_subagent_run` are cancellable in v1. Other hook events may observe or mutate their documented context fields, but do not expose cancellation fields.
- If a cancellable hook sets `cancelled=True`, stop dispatching the remaining handlers for that same event and operation. Use `ctx.cancel_reason` as the user-facing reason, or `"unspecified"` if empty.
- Hook dispatch is synchronous in v1. Async hook handlers are out of scope.
- Hook handlers receive the same mutable context object in registration order. Mutations made by one handler are visible to later handlers for the same event unless dispatch stops due to cancellation.
- Hook handlers are not automatically represented as trace spans in v1. Hook errors are logged; existing model/tool spans remain focused on model/provider calls and tool execution.

Export `Hook`, `HookContext`, all event-specific context classes, `HookEvent`, and `HookRegistry` from `thinharness/__init__.py`.

**Verify:** construction tests for unfiltered hooks, tool-filtered hooks, agent-filtered hooks, empty filter rejection, invalid event/filter combinations including `limit_reached` filters, unmatched-name warnings, exact case-sensitive matching, registration order, class-level event constants, required context fields populated with real values, cancellation fields existing only on cancellable contexts, cancellation stopping the handler chain only on supported events, and handler exceptions logged without crashing the run.

### 2. Wire Hooks into Harness Construction
Keep hooks out of Pydantic config objects and pass them as runtime constructor arguments. `HarnessConfig` and `SubAgentConfig` should remain serializable settings; hooks are live Python callables and cannot JSON round-trip cleanly.

```python
class HarnessConfig(BaseModel):
    """Configuration for Harness."""

    strict_hooks: bool = False
```

In `Harness.__init__`, create:

```python
def __init__(
    ...,
    hooks: list[Hook] | HookRegistry | None = None,
    subagent_hooks: dict[str, list[Hook] | HookRegistry] | None = None,
) -> None:
    self.hooks = hooks if isinstance(hooks, HookRegistry) else HookRegistry(hooks)
    self.subagent_hooks = subagent_hooks or {}
```

`hooks` applies to the current harness only. `subagent_hooks` is a runtime-only mapping keyed by resolved subagent name, including `DEFAULT_SUBAGENT_NAME` for the framework default subagent. This keeps child hook configuration explicit without embedding callables in `SubAgentConfig`.

After built-in tools, constructor `tools=[...]`, and configured subagents are registered, warn once for hook filters that currently match no known tool or subagent name. Invalid filter shape still raises during `HookRegistry` construction, but unmatched names should warn rather than raise because `add_tool(...)` can register tools after harness construction.

Subagents should not inherit parent hooks by default. A parent hook observes the parent run and the parent-side `subagent` tool boundary; the child run should only fire hooks explicitly configured for that child.

For the framework default subagent, use `subagent_hooks.get(DEFAULT_SUBAGENT_NAME)` when present and no child hooks otherwise. For named subagents, `build_child_harness(...)` should pass `hooks=subagent_hooks.get(config.name)` into `Harness(...)`. Parent config hooks are never copied implicitly because hooks no longer live in config.

Add and export:

```python
DEFAULT_SUBAGENT_NAME: Final[str] = "default"
```

Use that constant for default-subagent hook metadata and examples instead of repeating the string literal. Export it from `thinharness.subagents`; top-level package export is optional. Reserve this name for the framework default by rejecting `SubAgentConfig(name=DEFAULT_SUBAGENT_NAME)`. Also reject empty string names and names containing whitespace at `SubAgentConfig` construction.

**Verify:** a child harness built by `build_child_harness(parent, None)` does not inherit parent hooks; a named subagent with `subagent_hooks={"research": [...]}` fires those hooks inside the child run; parent hooks still fire for the parent `subagent` tool call; `SubAgentConfig(name=DEFAULT_SUBAGENT_NAME)`, empty names, and whitespace-containing names are rejected; unmatched hook filter names warn after current tool/subagent registration is known but do not prevent later `add_tool(...)` registration.

### 3. Fire Run and Prompt Lifecycle Hooks
Refactor `Harness.run()` so run state, prompt submission, model request counting, and terminal cleanup flow through one clear path.

Fire `run_start` at the top of `Harness.run()` after basic run state is initialized and before the first model request. Include the prompt, metadata, configured limits, and root path. `run_start` is observe-only in v1.

Add a same-harness re-entrancy guard around `Harness.run()`:

```python
if self._running:
    raise HarnessError("Harness.run is not re-entrant")
self._running = True
try:
    ...
finally:
    self._running = False
```

This prevents hook handlers from recursively calling `run()` on the same harness while one run is already mutating per-run state. The guard is per `Harness` instance: a hook may call a different harness or allow normal subagent execution, because those runs use separate child harness instances.

Fire `user_prompt_submit` once per `Harness.run(...)` after `run_start` and before the first model request. Include:
- `prompt`
- `additional_context`
- `metadata`

If `user_prompt_submit` appends `additional_context`, include that context in the first model request after the original prompt. Do not expose a raw provider-message mutation surface. If `user_prompt_submit` sets `cancelled=True`, do not call the model; raise `HarnessError("run blocked by hook: ...")` with `stop_reason="cancelled_by_hook"` and still fire `run_end`.

Do not fire hooks around every raw provider request in v1. Model request counting still belongs in a helper around `session.start(...)` and `session.continue_with_tools(...)`, but that helper should handle limits and tracing only. This keeps hooks aligned with stable harness concepts instead of provider-specific message plumbing.

Fire `run_end` for every terminal outcome from a single `finally`-style path: successful result, provider failure converted to `HarnessError`, hard-limit failure, and unexpected exception. Include:
- `result` on success
- `error` and `stop_reason` on failure; use values such as `"end_turn"`, `"provider_error"`, `"limit_reached"`, and `"error"`
- final `usage`

Type `stop_reason` as a `Literal` shared by `HarnessResult` and run-end contexts. Include `"cancelled_by_hook"` for prompt-level cancellation.

Do not fire `run_end` twice if an exception bubbles through nested try/finally blocks. Implement this with an explicit guard on run state:

```python
def _fire_run_end_once(...):
    if run_state.run_end_fired:
        return
    run_state.run_end_fired = True
    self.hooks.fire(...)
```

If a `run_end` hook raises under `strict_hooks=True`, suppress any second `run_end` emission via this guard and propagate the hook exception.

**Verify:** a no-tool run fires `run_start`, `user_prompt_submit`, and `run_end` exactly once each; `user_prompt_submit` appends additional context to the first model request using the documented `<hook_context>` wrapper; `user_prompt_submit` cancellation prevents the first model request and still fires `run_end`; no hooks fire per raw provider request; `run_end` includes the final `HarnessResult`; `run_end` fires once for provider errors and once for unexpected exceptions; recursive `harness.run(...)` on the same harness raises a clear `HarnessError`.

### 4. Fire Tool Hooks with Output Mutation Support
Integrate tool hooks inside `_traced_call_output(...)`, not outside it. The hook order matters because `after_tool_call` can mutate the output that the model will see.

`before_tool_call` context should include:
- `call_id`
- `tool_name`
- `arguments`
- `tool_spec`
- `tool_index` within the current model response

`tool_index` reflects the model-requested order in the provider response, not execution-completion order. Preserve this value even when tool calls run in parallel.

If `before_tool_call` sets `ctx.cancelled=True`, skip handler execution and return a normalized failed `ToolResult`:

```json
{"ok": false, "content": "Tool execution blocked by hook: ...", "metadata": {"error_type": "ToolCallCancelled"}}
```

`after_tool_call` context should include typed fields:
- `call_id`
- `tool_name`
- `arguments`
- `original_output`
- `output`
- parsed normalized output, if available
- `duration_ms`

Allow `after_tool_call` to replace the provider-facing output by assigning `ctx.output`. `ctx.original_output` remains the raw tool-produced output for debugging and auditability. If multiple `after_tool_call` hooks match, each later hook sees the current effective `ctx.output` and may replace it again. The final `ctx.output` is sent back to the model and used for tracing status.

`after_tool_call` should fire for successful results and normalized structured failures, including argument-validation errors, unknown tools, hook-cancelled tools, and tool handler exceptions converted by `_call_output(...)` to `ok: false`. It should not need a separate `error` field for normal tool failures because the existing tool contract is structured JSON output.

The final output flow inside `_traced_call_output(...)` should be:
1. Open the tool trace span and set the current tool-call context.
2. Fire `before_tool_call`; if cancelled, synthesize a normalized failed output.
3. Otherwise run the handler through `_call_output(...)`, which converts handler exceptions into normalized `ok: false` output.
4. Fire `after_tool_call`; handlers may mutate `ctx.output`.
5. Parse the final output.
6. Set trace attributes, including subagent metadata and captured tool result, from the final output.
7. Mark the span failed if the final parsed output has `ok: false`.

Parallel behavior:
- Per-tool hooks fire inside the worker context with the tool span current and may run concurrently for parallel tool batches.
- Hook handlers used with parallel-safe tools must not assume serial execution and must not share mutable state without coordination. The harness does not serialize hook execution with a lock in v1.
- Preserve provider continuation order regardless of hook timing.

**Verify:** tool-filtered hooks fire only for matching tools; `before_tool_call` can cancel one tool in a parallel batch without cancelling siblings; `tool_index` stays tied to model-requested order under parallel execution; `after_tool_call` can rewrite one output; `after_tool_call` fires for normalized handler exceptions; tracing captures the rewritten output and final error status; parallel hook handlers run without corrupting output order; hook exceptions raised under `strict_hooks=True` inside parallel worker contexts surface through the batch instead of being swallowed.

### 5. Replace max_turns with Explicit Limits
Remove `max_turns` from `HarnessConfig` and replace it with:

```python
max_model_requests: int = 64
max_tool_calls: int | None = None
```

Also update `SubAgentConfig`:

```python
max_model_requests: int | None = None
max_tool_calls: int | None = None
```

Child config inheritance in `build_child_harness(...)` should use explicit `is not None` override semantics:

```python
"max_model_requests": config.max_model_requests if config and config.max_model_requests is not None else parent_config.max_model_requests,
"max_tool_calls": config.max_tool_calls if config and config.max_tool_calls is not None else parent_config.max_tool_calls,
```

Counting semantics:
- `max_model_requests` counts provider calls only: one `session.start(...)` or `session.continue_with_tools(...)` is one model request.
- Tool calls are counted separately by `max_tool_calls`.
- A model response containing three tool calls still counts as one model request.
- A tool batch with three calls counts as three tool calls.

Limit enforcement:
- Check `max_model_requests` before every model request.
- Check `max_tool_calls` before executing a batch using projected total count, like Pydantic AI's `UsageLimits.check_before_tool_call(...)`.
- If a batch would exceed `max_tool_calls`, reject the batch before running any tools so partial execution does not surprise callers.
- `max_tool_calls` counts model-requested tool calls, including calls later blocked by `before_tool_call` hooks. A hook-cancelled tool still consumes one requested tool-call slot because the model attempted to use a tool and receives a tool result for that request.
- When a hard limit is reached, fire `limit_reached` and raise `HarnessError` at the parent harness level.
- Inside a subagent tool, child `HarnessError` is caught by `run_subagent_tool(...)` and returned as a structured `ok: false` tool result, preserving current subagent failure behavior.

Add a typed limit context:

```python
@dataclass(kw_only=True)
class LimitReachedContext(HookContext):
    """Context for a hard run limit being reached."""

    event: ClassVar[HookEvent] = "limit_reached"
    limit_kind: Literal["model_requests", "tool_calls"]
    limit_value: int
    current_count: int
```

`current_count` should equal `limit_value` when `limit_reached` fires: the limit has been consumed, and the next model request or tool-call batch would push the run over the limit.

`limit_reached` is observe-only. After it fires, the harness raises `HarnessError`; `run_end` still fires afterward with `stop_reason="limit_reached"`. The raised `HarnessError` is the canonical caller-facing outcome; `limit_reached` and `run_end` are observer notifications.

Default and migration note:
- `max_model_requests=64` is the current placeholder default. Tune before implementation if needed; exact migration from old `max_turns=32` would allow up to 33 provider calls because the existing loop does one initial request plus up to 32 continuations.
- Parent and child limits are local run limits. If a subagent inherits `max_model_requests=64`, it receives its own fresh child budget. Document this because nested subagents can multiply total provider calls; a future global budget object can handle shared caps explicitly.

Update error wording away from "turns":

```text
model did not finish within max_model_requests=...
tool calls would exceed max_tool_calls=...
```

**Verify:** `max_model_requests=1` succeeds for immediate final text; `max_model_requests=1` fails before continuing after a tool call; `max_model_requests=2` allows one tool batch and final text; `max_tool_calls=2` allows a two-tool batch; `max_tool_calls=2` rejects a three-tool batch before any tool executes; `limit_reached` fires with `current_count == limit_value` and is followed by exactly one `run_end`; a hook-cancelled tool still increments `usage.tool_calls` and `usage.cancelled_tool_calls`; subagent-level limit failure returns a structured failed parent tool result.

### 6. Add Subagent-Specific Hooks
Keep generic tool hooks for the `subagent` tool, but add semantic subagent events in `thinharness/subagents.py`.

Fire `before_subagent_run` in `run_subagent_tool(...)` after resolving the config and before `build_child_harness(...)`. This event is cancellable. Include:
- `agent`: resolved agent name, with `DEFAULT_SUBAGENT_NAME` for omitted agent
- `task`
- `inherited`
- `tool_mode`
- `parent_harness`
- `parent_call_id`, when available from `current_tool_call_context()`

Allow cancellation here. If cancelled, return a structured failed `ToolResult` with `error_type="SubAgentCancelled"` and content `"Subagent execution blocked by hook: {reason}"`, using `"unspecified"` when `cancel_reason` is empty.

Fire `after_subagent_run` after child completion or child failure. Include:
- `agent`
- `task`
- `result` on success
- `error` on failure
- `tools`
- `usage` from the child `HarnessResult` when available
- `parent_call_id`

Subagent event dispatch should use `parent.hooks.fire(...)` because the semantic subagent hooks live at the parent-side delegation boundary. Child lifecycle hooks come from the child harness only when explicitly configured through the parent's runtime `subagent_hooks` mapping; they are not inherited from the parent by default. The child hook events fire transparently because the child is a normal `Harness`, not because parent and child registries are linked.

Ordering for a successful subagent call should be:
1. Parent `before_tool_call` for tool `subagent`
2. Parent `before_subagent_run`
3. Child `run_start` / child `user_prompt_submit` hooks when configured through `subagent_hooks`
4. Child `run_end`
5. Parent `after_subagent_run`
6. Parent `after_tool_call` for tool `subagent`

If `before_subagent_run` cancels:
1. Parent `before_tool_call` for tool `subagent`
2. Parent `before_subagent_run`, which cancels
3. No child hooks fire because no child harness was constructed
4. Skip `after_subagent_run` because no child run started
5. Parent `after_tool_call` for tool `subagent` with the synthetic failed `ToolResult`

**Verify:** subagent hooks can filter by agent name; default subagent uses `DEFAULT_SUBAGENT_NAME`; cancellation returns `ok: false` without constructing a child harness; no child hooks fire on `before_subagent_run` cancellation; child `run_start`, `user_prompt_submit`, and `run_end` hooks fire only when configured through `subagent_hooks`; parent generic `after_tool_call` still sees the final structured subagent tool result; filters for unknown subagent names warn after registration is known and never match unless that name is later registered.

### 7. Update Result Metadata and Documentation
Add a small usage object and attach it to `HarnessResult`:

```python
@dataclass
class RunUsage:
    """Provider and tool usage for one harness run."""

    model_requests: int = 0
    tool_calls: int = 0
    cancelled_tool_calls: int = 0


@dataclass
class HarnessResult:
    """Final result returned by a harness run."""

    text: str
    responses: list[Json] = field(default_factory=list)
    tool_call_records: list[Json] = field(default_factory=list)
    usage: RunUsage = field(default_factory=RunUsage)
    stop_reason: StopReason = "end_turn"
```

Rename the existing `tool_calls: list[Json]` record list to `tool_call_records` so it is not confused with `usage.tool_calls`. `usage.tool_calls` should equal the number of model-requested tool calls recorded in `tool_call_records`; `usage.cancelled_tool_calls` separately reports how many of those calls were blocked by hooks. Keep `stop_reason` on `HarnessResult` and `RunEndContext`, not in `RunUsage`, so there is one canonical run outcome. Type `StopReason` as a shared `Literal`, not plain `str`.

Update README:
- Replace `max_turns` examples or prose with `max_model_requests` and `max_tool_calls`.
- Add a short hooks example with a tool filter.
- Add a subagent hook example showing `agents=["research"]`.
- Add an example that passes the same logging hook through both `hooks=[...]` and `subagent_hooks={"research": [...]}` for users who want observability inside children.
- Explain that model requests and tool calls are separate counters.
- Explain `result.usage` and the difference between `usage.tool_calls` and `tool_call_records`.
- Explain that parent hooks do not automatically apply inside subagent child runs; configure child hooks through `subagent_hooks`.
- Document that local parent/child budgets can multiply total provider calls in subagent-heavy workflows.
- Document that hook filters match final registered names exactly and case-sensitively.
- Document the breaking changes together: `max_turns` removal, `tool_calls` record list renamed to `tool_call_records`, and `result.usage` added for counters.
- Document that hooks are runtime constructor arguments, not `HarnessConfig` / `SubAgentConfig` fields.
- Document that hook handlers are synchronous in v1.

Update `.plans/03-subagents.md` references only if needed to avoid stale `max_turns` snippets. Do not rewrite the old plan wholesale; it is historical context.

**Verify:** README examples import exported hook classes successfully; no docs mention `max_turns` as the active config field except historical plans; `HarnessResult.tool_call_records` contains the same records previously exposed as `tool_calls`; `result.usage` reports model requests, tool calls, and cancelled tool calls; `result.stop_reason` reports the run outcome.

### 8. Follow-Up Plan: Near-Limit Guidance
Do not implement near-limit guidance in this hooks/limits pass. It changes model input and requires provider continuation design, so it should get a separate plan after Steps 1-7 are implemented, tested, and reviewed.

The follow-up plan should cover:
- A first-class `LimitWarning` / notice mechanism, not a raw model-request hook.
- Provider-neutral notice support for both `ModelSession.start(...)` and `ModelSession.continue_with_tools(...)`.
- The ordering invariant that tool outputs precede notice text in continuation payloads.
- Unit tests for OpenAI, Anthropic, and OpenRouter adapter payload shape.
- Manual integration tests gated by `.env` provider API keys.

**Verify later:** the separate plan exists before implementing near-limit guidance, and this core hooks/limits implementation does not include `limit_warnings` or provider `notices` parameters.

## Considerations
- `max_turns` removal is a breaking change, but the project is greenfield and the current name is ambiguous. Do not keep an alias unless tests or README reveal public usage we deliberately want to preserve.
- Parent hooks observe parent lifecycle and the parent-side subagent boundary only. Child lifecycle hooks must be configured through `subagent_hooks`, which keeps delegated agents self-contained and avoids surprising duplicate hook execution.
- A shared global budget across parent and subagents is out of scope. Parent limits and child limits are local run limits. If global budgeting is needed later, add an explicit shared budget object rather than coupling nested counters implicitly.
- Near-limit guidance is more than an observability hook because it changes model input. Keep it as a separate follow-up plan and implement it only after the core hooks/limits pass is stable.
- Per-tool hook filters should match final tool names exactly after builtin selection. Do not normalize custom tool names beyond the existing tool registration behavior.
- Hook handlers should not call `harness.run(...)` recursively on the same harness. Re-entrant runs on the same instance are unsupported, but separate `Harness` instances may run concurrently in the same process.
- Subagent fork-conversation mode should reuse these hooks without new event names. Fork mode can add `fork_mode` to subagent hook context when `.plans/subagents-fork-mode.md` is implemented.
