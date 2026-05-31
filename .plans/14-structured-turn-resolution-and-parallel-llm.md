# Plan: Structured Turn Resolution and Parallel LLM Reuse

## Overview
Deepen structured-output handling so `core.py` and `parallel_llm.py` ask `output.py` one question: given this `ModelTurn` and schema, what does this turn mean? The end state is a single structured-turn decision module that owns finalization, validation failure classification, retry intent, and trace facts for `Harness.run()` and `ParallelLlmTool`. Callers still own retry-budget exhaustion and transport.

This plan supersedes the older extraction idea in `.plans/12-parallel-llm-structured-output.md`: instead of sharing only helper functions such as `validate_turn_output(...)`, share the actual turn-resolution decision shape.

## Decisions
- Put the new structured-turn decision types and functions in `thinharness/output.py`.
- Remove `Harness._finalized_output_mode_for_turn`; tracing should use the structured decision rather than performing a second validation pass.
- Update `ParallelLlmTool` in the same pass so the architecture does not leave a duplicated structured-output path behind.
- Preserve current structured-output behavior: normal user tools may run before final structured output, and retry-budget exhaustion remains caller-specific.
- Treat `ParallelLlmTool` as one-shot in this pass. If a resolved turn asks to `continue` with ordinary tool calls, return a per-item failure instead of adding nested tool execution.
- Keep finalized structured-output trace facts on the model span by resolving the returned turn before that span closes.
- Do not tackle the broader `ModelSession` request-shape refactor in this pass.
- Do not rename provider capability fields in this pass unless a required code change directly forces it.
- Delete `validate_turn_output(...)` if the new resolver makes it unused; no compatibility wrapper is needed.

## Steps

### 1. Add Structured Turn Decision Types in `output.py`
Add small internal dataclasses or frozen dataclasses in `thinharness/output.py` that describe the resolved outcome of one model turn against an optional `OutputSchema`.

The exact names are flexible, but the interface should be clear and typed. A reasonable shape:

```python
@dataclass(frozen=True)
class OutputTurnDecision:
    kind: Literal["continue", "final", "retry_user_message", "retry_tool_output", "unexpected"]
    finalized_mode: ResolvedOutputMode | None = None
    finalized_via_output_tool: bool = False
    text: str = ""
    output: Any | None = None
    retry_message: str = ""
    retry_call_id: str | None = None
    error: OutputValidationError | None = None
    unexpected_message: str = ""
```

Add one main resolver function, for example:

```python
def resolve_turn_output(
    turn: ModelTurn,
    output_schema: OutputSchema | None,
) -> OutputTurnDecision:
    ...
```

The resolver owns these rules:
- No `output_schema`: final text when there are no tool calls; continue when there are tool calls.
- In every structured mode, ordinary non-`final_result` tool calls produce `kind="continue"` so normal tools can run before final structured output.
- Only tool-mode synthetic `final_result` receives special structured-output handling. In text, native, prompted, or no-schema mode, a user tool named `final_result` remains an ordinary tool unless existing tool-name collision checks reject it elsewhere.
- Do not reuse `validate_turn_output(...)` as the resolver kernel without changing its tool-call behavior. That helper currently raises on tool calls for modes where `Harness.run()` must continue and execute ordinary tools.
- `text` mode: final text when there are no tool calls; validation failure creates a user-message retry decision.
- `tool` mode: exactly one `final_result` call finalizes or creates a tool-output retry tied to that call id; text without `final_result` creates a user-message retry; `final_result` with siblings or repeated `final_result` returns an unexpected decision.
- `native` and `prompted` modes: no tool calls, validate text, final or user-message retry.
- The decision carries `finalized_mode` when validation succeeds so tracing does not revalidate.
- The decision carries `finalized_via_output_tool` when the run finalized through the synthetic `final_result` tool so `Harness.run()` can preserve `resume_state` omission without re-inferring the path.
- The resolver does not know retry budgets. It returns retry decisions whenever validation fails; `Harness.run()` and `ParallelLlmTool` decide whether the relevant retry budget is exhausted and shape the terminal failure.
- The decision must carry enough error text for both callers: harness retry prompts still need the existing "failed structured output validation" / "Call `final_result`" wording, while `ParallelLlmTool` failure entries should preserve the existing `"output validation failed: ..."` style and include `"final_result"` for tool-mode text-without-call failures.

Keep existing schema-building helpers such as `OutputSchema.build`, `resolve_output_schema_for_model`, `structured_instructions`, and `structured_output_request`. The new module should compose those helpers, not duplicate schema construction.

**Verify:** Add focused unit tests in `tests/test_structured_output.py` for `resolve_turn_output(...)` covering final, retry, continue, and unexpected decisions without running the full harness loop.

### 2. Refactor `Harness.run()` to Act on Decisions
In `thinharness/core.py`, replace the inline structured-output branch beginning around the `while True` loop with a smaller decision/action flow.

The loop should read roughly like:

```python
turn, decision = await advance_model(..., output_schema=self.output_schema)
while True:
    responses.append(turn.raw)

if decision.kind == "final":
    turn.finalized_output_mode = decision.finalized_mode
    return finalize(
        decision.text,
        active_session,
        output=decision.output,
        finalized_via_output_tool=decision.finalized_via_output_tool,
    )
if decision.kind == "retry_tool_output":
    retry_or_fail()
    turn, decision = await advance_model(
        lambda notices: active_session.continue_with_tools(...),
        trace_snapshot=...,
        output_schema=self.output_schema,
        output_retry=True,
    )
    continue
if decision.kind == "retry_user_message":
    retry_or_fail()
    turn, decision = await advance_model(
        lambda notices: active_session.continue_with_user_message(...),
        trace_snapshot=...,
        output_schema=self.output_schema,
        output_retry=True,
    )
    continue
if decision.kind == "continue":
    check_tool_limit(len(turn.tool_calls))
    usage.tool_calls += len(turn.tool_calls)
    recorded, outputs, executions = await execute_normal_tools(...)
    usage.cancelled_tool_calls += sum(1 for execution in executions if execution.cancelled)
    tool_call_records.extend(recorded)
    check_tool_retry_limits(turn.tool_calls, executions)
    turn, decision = await advance_model(
        lambda notices: active_session.continue_with_tools(...),
        trace_snapshot=...,
        output_schema=self.output_schema,
    )
    continue
if decision.kind == "unexpected":
    raise UnexpectedModelBehavior(decision.unexpected_message)
```

Remove `Harness._finalized_output_mode_for_turn`. In `advance_model(...)`, stop validating the turn through a separate helper for trace annotation. Instead, after the provider returns a `ModelTurn` and before the model span closes, call `resolve_turn_output(...)`, set finalized/retry trace attributes from that same decision, and return both the turn and decision to the main loop. Keep retry-budget exhaustion outside `advance_model(...)`; it belongs to the caller action path.

The loop shape should match the current flow: advance once before entering the loop, append/process that returned turn inside the loop, and then each action branch that needs more model output advances exactly once before continuing. Do not re-run the initial request at the top of every iteration.

Preserve retry accounting exactly:
- `retry_or_fail()` checks the output retry budget before any corrective model request starts.
- Corrective structured-output requests pass `output_retry=True`.
- `usage.output_retries` increments only after model-limit checks allow the corrective request, matching the current `advance_model(...)` ordering.

Preserve raw response recording exactly: each returned `turn.raw` is appended to `responses` once, including initial turns, retry turns, and turns after ordinary tool execution. Do not append in both `advance_model(...)` and the loop unless the ownership is explicit and tested.

Trace attribute contract:
- `kind="final"` with `decision.finalized_mode is not None` sets finalized structured-output attributes on the model span before it closes.
- `kind="final"` with `decision.finalized_mode is None` is a normal non-structured final result and must not set structured-output finalization attributes.
- `continue`, `retry_user_message`, `retry_tool_output`, and `unexpected` decisions do not set `gen_ai.output.finalized=True`.
- If retry/validation attributes are added for non-final decisions, define them separately and test them; do not reuse the finalization marker.

Important behavior to preserve or intentionally encode:
- A successful synthetic `final_result` does not count as a normal tool call.
- Invalid `final_result` arguments consume `output_retries`, not `tool_retries`.
- `final_result` mixed with sibling calls runs zero sibling tools and raises `UnexpectedModelBehavior`.
- Ordinary non-`final_result` tool calls still run before final structured output in text, tool, native, and prompted modes.
- Ordinary tool execution still enforces tool limits, increments `usage.tool_calls`, increments cancelled-tool accounting, records tool call records, and checks tool retry limits before continuing the model.
- Tool hooks do not fire for synthetic `final_result`.
- `resume_state` remains omitted for tool-mode `final_result` exits.
- `finalized_via_output_tool` is true only for final decisions produced by the synthetic `final_result` tool.
- If `turn.finalized_output_mode` remains useful for internal compatibility, set it from the decision; otherwise remove write-only assignments in the same pass after confirming no consumers.

**Verify:** Run `uv run pytest tests/test_structured_output.py tests/test_resume.py tests/test_tracing.py`.

### 3. Update `ParallelLlmTool` to Use the Same Decision Module
In `thinharness/tools/parallel_llm.py`, replace direct calls to `validate_turn_output(...)` with the new turn decision resolver.

For each prompt:
- Build `output_schema` once per batch using `resolve_output_schema_for_model(...)`.
- Build `instructions` with `structured_instructions(...)`.
- Send synthetic tools and native request metadata from `output_schema`.
- After each model turn, call `resolve_turn_output(turn, output_schema)`.
- On `final`, emit `_success_entry(...)`.
- On `retry_user_message` or `retry_tool_output`, check `self.output_retries`; if attempts remain, retry with a fresh prompt built from the original prompt and validation feedback.
- On `continue`, return a sparse failure entry because `ParallelLlmTool` is a normal one-shot model call in this pass and does not execute arbitrary nested tools.
- On `unexpected` or exhausted retry, return a sparse failure entry.
- Build failure entries from the decision's validation error/message so existing `ParallelLlmTool` error strings remain stable, including the `"output validation failed:"` prefix where tests expect it.
- Use the existing structured retry prompt shape for fresh-session retries so the retry prompt still includes the original prompt and validation feedback.
- Treat `continue` and `unexpected` as immediate failures, not retryable validation failures. This is an intentional behavior clarification for one-shot parallel calls.

The key architectural requirement is that `ParallelLlmTool` no longer contains its own independent rules for validating text, tool-mode `final_result`, native, or prompted output.

**Verify:** Run `uv run pytest tests/test_parallel_llm.py tests/test_structured_output.py`.

### 4. Update Tests to Target the New Interface
Revise tests so the structured-output rules are tested through both layers:
- Direct unit tests for the output decision module.
- Harness integration tests for hook behavior, retry budgets, finalization, tracing, and resume state.
- Parallel LLM tests for reused decision behavior.

Prefer tests that prove why the module exists:
- A trace finalized mode should come from the same decision that finalizes the run.
- A `final_result` sibling tool call should never execute.
- Ordinary user tools should still execute before later structured final output.
- `ParallelLlmTool` and `Harness.run()` should agree on valid and invalid structured turns.
- `ParallelLlmTool` should treat a shared `continue` decision as a per-item failure, not as a request to execute tools.
- `HarnessResult.responses` should contain exactly one raw response entry per model turn, including retry turns.
- Intermediate model spans should not be marked finalized.

**Verify:** Run the pass-level validation commands in the Test Strategy.

## Test Strategy

### 1. Output Decision Unit Tests
- **Purpose:** Prove that `output.py` owns structured turn classification.
- **Tests:** Final text, final `final_result`, invalid text retry, invalid tool retry, continue with normal tools in each structured mode, unexpected mixed final tool calls.
- **How:** Direct tests against `resolve_turn_output(...)` using `ModelTurn` and `ModelToolCall`.
- **Likely misses:** Provider payload translation and hook ordering.

### 2. Harness Integration Tests
- **Purpose:** Prove `Harness.run()` still performs the right side effects for each decision.
- **Tests:** Retry counts including model-limit-blocked corrective requests, `stop_reason`, `resume_state`, no hooks for `final_result`, no sibling tool execution, ordinary tools before final structured output, user tool named `final_result` outside tool mode if supported by existing collision checks, one raw response per model turn, ordinary tool accounting, tracing finalized attributes only on finalizing structured model spans.
- **How:** Existing scripted-model tests in `tests/test_structured_output.py`, `tests/test_resume.py`, and `tests/test_tracing.py`.
- **DOES NOT:** Re-test every Pydantic schema-generation edge case; those stay under `OutputSchema` tests.

### 3. Parallel LLM Reuse Tests
- **Purpose:** Prove `ParallelLlmTool` no longer has a separate structured-output implementation.
- **Tests:** Prompted, native, tool, text, invalid retry, exhausted retry, `continue` as unsupported failure, built-in text-only behavior.
- **How:** `tests/test_parallel_llm.py` with fake model sessions recording tools, instructions, and structured-output metadata.

## Spec Coverage Map
- Structured turn finalization -> Output Decision Unit Tests, Harness Integration Tests.
- Structured retry behavior -> Output Decision Unit Tests, Harness Integration Tests, Parallel LLM Reuse Tests.
- Normal tool calls before structured final output -> Output Decision Unit Tests, Harness Integration Tests.
- Raw response recording -> Harness Integration Tests.
- Ordinary tool accounting and retry limits -> Harness Integration Tests, Tool Retry Tests.
- Tracing finalized output mode on model spans -> Harness Integration Tests.
- `ParallelLlmTool` structured output -> Parallel LLM Reuse Tests.

## Validation Commands
Run these before handing off as complete:

```bash
uv run pytest tests/test_structured_output.py tests/test_parallel_llm.py tests/test_providers.py tests/test_resume.py tests/test_tracing.py
uv run pytest tests/test_harness.py tests/test_tool_retry.py
uv run ruff check .
uv run pyright
```

## Do Not Touch
- Do not move tool batch execution in this pass.
- Do not introduce `RunContext` in this pass.
- Do not rewrite provider request/session interfaces beyond what structured-output decision reuse requires.
- Do not rename `ModelCapabilities.permissive_native_override` in this pass unless it becomes unavoidable while preserving existing provider behavior.
- Do not change docs except for short architecture notes if the implementation changes public examples.

## Considerations
- The biggest risk is span lifetime: finalization/retry trace facts should come from `resolve_turn_output(...)`, but they still belong on the model span. Resolve the turn after the provider returns and before the model span closes.
- `ParallelLlmTool` remains one-shot. Use a fresh corrective prompt for both user-message and tool-output structured retries; do not add nested ordinary tool execution in this pass.
- If `validate_turn_output(...)` becomes orphaned after `ParallelLlmTool` switches to `resolve_turn_output(...)`, delete it in the same pass rather than leaving a stale duplicate validation path. Do not keep a compatibility wrapper.
- Keep `_structured_retry_message` wording stable whether it remains in `core.py` or moves closer to the resolver.
- Keep decision types internal unless the final shape is clearly valuable as public API.
