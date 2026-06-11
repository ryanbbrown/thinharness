# Plan: Leaf Types Module and Run-Loop Decomposition

## Overview
Two batched refactors from the 2026-06-10 architecture review (`.context/architecture-review.html`, findings 1 and 2). They batch well because finding 1 is a mechanical prerequisite that makes finding 2 clean: once shared types live in a leaf module, the run-loop helpers extracted in Phase B can use ordinary top-level imports instead of inheriting the deferred-import workaround.

**Phase A (finding 1):** extract `HarnessError`, `UnexpectedModelBehavior`, `HarnessResult`, `RunUsage`, `StopReason`, and `LimitNoticeKey` into a new leaf module so `core.py` stops being both the top and the bottom of the dependency graph. Deletes all eight deferred `from .core import ...` sites (seven in `runtime.py`, one in `providers.py`) and untangles the `TYPE_CHECKING` core imports in `runtime.py`, `providers.py`, and `hooks.py`.

**Phase B (finding 2):** decompose `Harness.run()` (currently ~300 lines, `core.py:346-643`) by extracting the session-call ceremony repeated at eight call sites into a turn driver, deduplicating the two background-drain blocks, and collapsing the five-rung exception ladder.

This is a behavior-preserving internal refactor. Public API, provider request shapes, hook semantics, tracing semantics, resume behavior, stop reasons, and retry accounting must remain unchanged. The existing test suite is the safety net.

## Decisions
- New module is `thinharness/types.py` and it must be a true leaf: stdlib imports only, nothing from the package. It must NOT import `Json` from `.tools.base` — importing `thinharness.tools.base` executes `thinharness/tools/__init__.py`, which eagerly imports `parallel_llm` → `output` → `providers`, and Phase A makes `providers.py` import `.types`; if `types` is imported before `core` (e.g. by the updated `__init__.py` import order), that chain re-enters the partially initialized `types` module and raises ImportError. Instead, move the alias `Json = dict[str, Any]` from `tools/base.py:17` into `types.py`, and have `tools/base.py` import it via `from ..types import Json` (the existing re-export through `thinharness/tools/__init__.py` keeps working; no other import sites change).
- This is a greenfield project with no external users: there is no backwards-compatibility obligation anywhere in this plan. `core.py` imports the moved names from `.types` because it still uses them — not as a re-export shim — and `thinharness/__init__.py` imports the moved leaf types directly from `.types` (keeping `Harness`/`HarnessConfig` from `.core`).
- `_compute_limit_notices`, `_limit_notice_dedup_key`, `_append_notice_once`, and `_build_resume_state` move from `core.py` to `runtime.py`, their only consumers' layer. Update the one external import (`tests/test_structured_output.py:31`) to `from thinharness.runtime import _compute_limit_notices` in the same pass rather than leaving a compatibility alias.
- `providers.py` deletes the `_resume_error` lazy-import helper and raises `HarnessError` imported top-level from `.types`. Do NOT introduce a new exception subclass — plain `HarnessError` is sufficient and `tests/test_resume.py` asserts on it; a subclass would be speculative.
- The turn driver (`TurnDriver`) lives in `runtime.py` next to `RunContext`, not in `core.py` and not in a new module. It is internal; do not export it.
- `TurnDriver` calls `harness.tool_schemas()` fresh on every request, exactly as today. Schema caching is review finding 11 and is out of scope.
- The exception ladder keeps `asyncio.CancelledError` as its own `except` clause. Do not collapse to a single `except BaseException` — that would newly capture `KeyboardInterrupt`/`SystemExit` into span annotation and stop-reason bookkeeping, which is a behavior change.
- `match` statements are fine (the codebase already uses one in `tools/skills.py`).
- Phase A lands and is fully validated before Phase B starts. Each phase should be a separate commit (conventional commits: `refactor(core): ...`).

## Steps

### Phase A — leaf types module

### 1. Create `thinharness/types.py`
Move from `core.py`, verbatim except for imports:
- `HarnessError(RuntimeError)` (core.py:247)
- `UnexpectedModelBehavior(HarnessError)` (core.py:251)
- `StopReason` Literal alias (core.py:59)
- `LimitNoticeKey` tuple alias (core.py:70)
- `RunUsage` dataclass (core.py:236)
- `HarnessResult` dataclass (core.py:223)

Also move the alias `Json = dict[str, Any]` (tools/base.py:17) into `types.py`, and change `tools/base.py` to `from ..types import Json` — `types.py` must stay stdlib-only (see Decisions for the cycle this prevents). `HarnessResult` then references `Json` and `StopReason` locally. Keep docstrings and field defaults byte-identical. Give the module a one-line docstring in the existing style (e.g. `"""Shared leaf types for harness runs."""`).

In `core.py`, replace the definitions with `from .types import HarnessError, HarnessResult, LimitNoticeKey, RunUsage, StopReason, UnexpectedModelBehavior` (all still used inside `core.py`). Update `thinharness/__init__.py` to import `HarnessError`, `HarnessResult`, `RunUsage`, and `UnexpectedModelBehavior` from `.types` directly, leaving `Harness` and `HarnessConfig` imported from `.core`. The exported names (`__all__`/public surface) are unchanged; only their source module moves.

**Verify:** `uv run pyright` clean; `uv run pytest tests/test_harness.py tests/test_structured_output.py` passes; `python -c "from thinharness import HarnessError, HarnessResult, RunUsage, UnexpectedModelBehavior"` works; `python -c "import thinharness.types, thinharness.providers, thinharness.hooks"` works (exercises the import chain that would break if `types.py` were not a true leaf).

### 2. Move limit-notice and resume-state helpers into `runtime.py`
Move `_limit_notice_dedup_key`, `_append_notice_once`, `_compute_limit_notices` (core.py:73-138) and `_build_resume_state` (core.py:141-159) into `runtime.py` unchanged. Their dependencies are all importable there without cycles: `ModelNotice`/`ModelSession` (already imported), `HarnessError` (now from `.types`), stdlib `json` (used by `_build_resume_state`'s round-trip; `runtime.py` does not currently import it), and `HarnessConfig` (annotation-only — keep under `TYPE_CHECKING`; `runtime.py` already has `from __future__ import annotations`).

Then delete all seven deferred imports in `runtime.py` (lines 89, 106, 130, 176, 189, 228, 237):
- `HarnessError` → top-level `from .types import HarnessError`.
- `HarnessResult` in `build_terminal_result` → top-level import from `.types`.
- `_compute_limit_notices` / `_build_resume_state` → now local to the module.

The `TYPE_CHECKING` block in `runtime.py` shrinks to `Harness` from `.core` plus the existing tool-execution/tracing types. The other four names currently in that block (`HarnessResult`, `LimitNoticeKey`, `RunUsage`, `StopReason`, runtime.py:21) do NOT disappear — they become one real top-level `from .types import ...` alongside `HarnessError`; the leaf is import-cheap, so don't guard-split annotation-only names. Update `tests/test_structured_output.py:31` to import `_compute_limit_notices` from `thinharness.runtime`.

`core.py` no longer references these four helpers after Phase B finishes; during Phase A, `run()`'s call sites are unchanged because both helpers are only called from `runtime.py` already — confirm with grep and remove the now-unused definitions from `core.py` in this step.

**Verify:** `grep -n "from .core import" thinharness/runtime.py` returns nothing; `uv run pytest tests/test_structured_output.py tests/test_resume.py tests/test_harness.py`.

### 3. Fix the providers → core inversion
In `providers.py`:
- Add top-level `from .types import HarnessError`.
- Delete `_resume_error` (providers.py:189-193) and the `TYPE_CHECKING` import of `core.HarnessError` (line 17).
- Replace every `raise _resume_error(...)` (11 sites: `_validate_resume_state` plus the three `resume_session` methods) with `raise HarnessError(...)` — same messages, same exception type, so `tests/test_resume.py` assertions on message text must pass unchanged.

In `hooks.py`, move `HarnessResult`, `RunUsage`, `StopReason` out of the `TYPE_CHECKING` block (hooks.py:29) into a real `from .types import ...` import, leaving only `Harness` type-checking-guarded. This is optional correctness polish — annotations work either way — but it makes the import graph honest now that it can be.

**Verify:** `grep -rn "from .core import\|from ..core import" thinharness/` shows only `TYPE_CHECKING` imports of `Harness` (`runtime.py`, `tool_execution.py`, `subagents.py:18`, `tools/parallel_llm.py:30`, `hooks.py`) plus the deferred `from .core import Harness` inside `build_child_harness` (`subagents.py:208`, kept — `core` imports `subagents` at top level, so that one stays deferred). Run `uv run pytest tests/test_resume.py tests/test_providers.py`.

### Phase B — run-loop decomposition

### 4. Add `TurnDriver` to `runtime.py`
A small class owning the active session and the per-run constants so call sites name only what varies:

```python
class TurnDriver:
    """Owns one active ModelSession plus the per-run request constants."""

    def __init__(self, *, session, run_ctx, harness, instructions, metadata, structured_output): ...

    async def start(self, prompt: str) -> tuple[ModelTurn, OutputTurnDecision]: ...
    async def resume(self, prompt: str) -> tuple[ModelTurn, OutputTurnDecision]: ...
    async def send_tool_outputs(
        self, outputs: list[ToolOutput], *,
        kind: Literal["tool_outputs", "output_retry_tool", "background_completion"] = "tool_outputs",
        output_retry: bool = False,
    ) -> tuple[ModelTurn, OutputTurnDecision]: ...
    async def send_user_message(
        self, message: str, *,
        kind: Literal["correction", "background_completion"],
        output_retry: bool = False,
    ) -> tuple[ModelTurn, OutputTurnDecision]: ...
```

Each method builds its `ModelTraceSnapshot` (the snapshot for tool outputs renders `[{"call_id": ..., "output": ...}]` exactly as today) and delegates to `run_ctx.advance_model(...)` so the existing limit/notice/tracing/output-resolution ceremony is untouched. `start` calls `session.start(...)`, `resume` calls `session.continue_with_user_prompt(...)`. Tools come from `harness.tool_schemas()` per call; instructions and structured-output request are the stored constants. The driver computes the snapshot `output_mode` itself in `__init__` via `_trace_output_mode(harness.output_schema)`.

**Metadata pitfall:** the driver stores the raw `metadata` argument passed to `run()` (possibly `None`) — NOT `run_metadata = dict(metadata or {})`, which exists only for hooks and `RunContext` (core.py:368-375). Every session call today passes the raw value, so sessions must keep receiving `None` when the caller passed no metadata; passing `run_metadata` would silently turn `None` into `{}` in provider payloads and no scripted test currently catches it. Before wiring the driver, add one scripted-session assertion that `metadata is None` for a plain `run(prompt)` call, locking the behavior.

Map all eight existing call sites onto driver methods:
| core.py site | replacement |
|---|---|
| 430 (`start`) | `driver.start(effective_prompt)` |
| 441 (`continue_with_user_prompt`) | `driver.resume(effective_prompt)` |
| 464 (deferred final, output tool) | `driver.send_tool_outputs([...], kind="background_completion")` |
| 480 (deferred final, text) | `driver.send_user_message(msg, kind="background_completion")` |
| 508 (`retry_tool_output`) | `driver.send_tool_outputs([...], kind="output_retry_tool", output_retry=True)` |
| 528 (`retry_user_message`) | `driver.send_user_message(msg, kind="correction", output_retry=True)` |
| 571 (budget-exhausted rejection) | `driver.send_tool_outputs(outputs, kind="background_completion")` |
| 594 (normal tool batch) | `driver.send_tool_outputs(outputs)` |

This deletes every `lambda notices, x=x: ...` default-arg capture in `run()`. Session selection (new vs resumed) stays in `Harness.run()`; the driver receives whichever session was chosen.

**Verify:** `uv run pytest tests/test_harness.py tests/test_structured_output.py tests/test_background_tools.py tests/test_resume.py tests/test_tracing.py` — tracing tests are the key check that snapshot kinds and capture behavior are unchanged.

### 5. Deduplicate the background-drain blocks
The shared portion of core.py:452-457 and 548-557 (wait for next completion, record it, format the message) becomes one helper on `RunContext`:

```python
async def drain_next_background(self) -> tuple[BackgroundToolCompletion, str]:
    """Wait for, record, and format the next background completion."""
```

The two call sites keep their genuinely different continuations:
- **Final-with-pending** (deferred final answer): branch on `decision.finalized_via_output_tool` to send either the deferred-final tool output or a user message — extract as `Harness._defer_final_for_background(decision, driver, run_ctx)` returning `(turn, decision)`.
- **Tool-budget-exhausted-with-pending**: build the `ToolCallsExceeded` rejection outputs for every requested call and send them — extract as `Harness._reject_batch_for_background(turn, driver, run_ctx)` returning `(turn, decision)`.

Keep the exact model-facing message text (`background_completion_message`, the "Final answer deferred..." wrapper, and the "Tool call was not executed because max_tool_calls=... is exhausted..." wrapper) byte-identical — tests and downstream prompts depend on it.

`runtime.py` imports `background_completion_message` (and `BackgroundToolCompletion` for the helper's return annotation) from `.tool_execution` at top level — safe because `tool_execution.py` imports `runtime` only under `TYPE_CHECKING` (tool_execution.py:18). Do not re-introduce a deferred import here; removing that pattern is the point of Phase A.

**Verify:** `uv run pytest tests/test_background_tools.py` (full file) plus `tests/test_harness.py`.

### 6. Collapse the exception ladder
Replace the five handlers (core.py:606-636) with two:

```python
except asyncio.CancelledError as exc:
    run_ctx.record_terminal_failure(exc, stop_reason="cancelled", span_message="run cancelled")
    raise
except Exception as exc:
    raise _classify_run_failure(run_ctx, agent_span, exc)  # or bare `raise` per classification
```

where the classifier preserves each branch's exact semantics:
- `ProviderError` → stop_reason `"provider_error"`, terminal_error = `HarnessError(str(exc))`, **raise the wrapper** `from exc`.
- `UnexpectedModelBehavior` → stop_reason `"unexpected_model_behavior"`, terminal_error = existing-or-exc, re-raise original.
- `HarnessError` → terminal_error = existing-or-exc; stop_reason becomes `"error"` only if still `"end_turn"`; re-raise original.
- other `Exception` → stop_reason `"error"`, terminal_error = exc, re-raise original.
- All branches: `agent_span.record_exception(exc)` + `agent_span.set_error(...)` with the same message/type strings as today (`"run cancelled"`/`"CancelledError"` for cancellation, `str(exc)`/`type(exc).__name__` otherwise).

**Subclass trap:** `UnexpectedModelBehavior` subclasses `HarnessError`. The current ladder gets the dispatch right for free via separate `except` clauses; an `isinstance`-based classifier must test in the same order — `ProviderError`, then `UnexpectedModelBehavior`, then `HarnessError`, then `Exception`. Testing `HarnessError` first makes the `UnexpectedModelBehavior` branch dead code and assigns the wrong stop reason. This is the single most likely bug in the refactor.

Implementation shape is flexible (one helper taking a precomputed `(stop_reason, wrap)` pair is fine); the non-negotiables are: CancelledError stays a separate clause, `ProviderError` is the only wrapped re-raise, and the stop-reason precedence rules above are preserved. Keep the inline session-creation `except` at core.py:424-429 as is — it is pre-loop and intentionally different (including that a `ProviderError` from `new_session()` maps to stop_reason `"error"`, not `"provider_error"`; do not unify it with the classifier in passing).

**Verify:** `uv run pytest tests/test_harness.py tests/test_hooks.py tests/test_providers.py` — these cover provider_error wrapping, cancelled stop reasons, and strict-hook HarnessError propagation with preserved `stop_reason` from `cancelled_by_hook`.

### 7. Final loop shape and cleanup
After steps 4-6, restructure the `while True` body in `run()` as a flat dispatch over `decision.kind` (if/elif chain or `match` — implementer's choice) where every branch is a few lines calling a named helper or driver method. Target: `run()` under ~120 lines with no nesting deeper than three levels inside the loop. Delete now-unused locals: `output_mode` and `structured_output` move into the driver, and since the driver computes `output_mode` itself, `core.py` also drops its `_trace_output_mode` import (core.py:53). `instructions` is NOT deletable — it is still needed pre-driver by `annotate_agent_start` (core.py:409-418) in addition to being passed into the driver. Do not change the surrounding structure: the `finally` blocks (background cancel-and-drain, `fire_run_end_once`, `self._running = False`), the MCP connect timing, the hook firing order, and the pre-loop resume branching all stay where they are.

**Verify:** Full validation commands below.

## Test Strategy

### 1. Existing integration suite as the primary safety net
- **Purpose:** Prove the refactor is behavior-preserving across every loop path.
- **How:** The scripted-model tests already cover start/resume, structured-output retries (both tool and user-message modes), background interleaving (deferred final + budget exhaustion), provider errors, cancellation, limit notices, and tracing snapshot kinds. No path through `run()` should exist that the suite does not exercise; if one is found during implementation, add a scripted-model test for it *before* refactoring that path.

### 2. Targeted additions
- A direct `TurnDriver` test is justified only if a bug slips through integration coverage — prefer not to duplicate.
- Snapshot-kind assertions already exist: `output_retry_tool` at `tests/test_tracing.py:386` and `background_completion` at `tests/test_background_tools.py:469`. Verify both `background_completion` forms (tool-output and user-message prompt) remain asserted after step 4; add an assertion only if one form is missing.
- A small table-driven unit test for the step-6 exception classifier IS justified (unlike `TurnDriver`): for each of `ProviderError` / `UnexpectedModelBehavior` / `HarnessError` / generic `Exception`, assert the resulting `stop_reason`, `terminal_error` identity, wrap-vs-reraise behavior, and the `cancelled_by_hook` precedence case. The subclass-ordering trap is exactly the kind of bug integration failures diagnose slowly.
- A scripted assertion that sessions receive `metadata=None` for a plain `run(prompt)` call (see step 4's metadata pitfall) — add before the driver lands.

### 3. Import-graph regression
- After Phase A, assert no module outside `core.py` imports from `core` at runtime (except the deferred `Harness` import in `build_child_harness`): a simple grep check in the validation gate is sufficient; do not add a permanent lint rule in this pass.
- `tests/fakes.py:179` does a deferred `from thinharness import HarnessError`; it keeps working since the package export set is unchanged — covered by the full-suite run, no action needed.

## Spec Coverage Map
- Leaf types extraction → import-graph grep + public import smoke check + full suite.
- Limit-notice/resume-state helper moves → `tests/test_structured_output.py` (updated import), `tests/test_resume.py`.
- Providers raising `HarnessError` directly → `tests/test_resume.py`, `tests/test_providers.py`.
- TurnDriver ceremony equivalence → `tests/test_tracing.py`, `tests/test_structured_output.py`.
- Background drain dedup → `tests/test_background_tools.py`.
- Exception ladder collapse → `tests/test_harness.py`, `tests/test_hooks.py`.

## Validation Commands
Run before handing off each phase as complete:

```bash
uv run pytest
uv run ruff check .
uv run pyright
uv run python -c "import thinharness.types, thinharness.providers, thinharness.hooks"   # leaf-import smoke check
grep -rn "from .core import\|from ..core import" thinharness/ | grep -v __pycache__   # only TYPE_CHECKING Harness imports + the deferred Harness import in subagents.py build_child_harness should remain
```

## Do Not Touch
- No public API changes: `thinharness/__init__.py` exports the same set of names (their internal source module changes to `.types`, which is fine — greenfield, no compat obligation).
- Findings 3-11 from the review are out of scope: no constructor cleanup, no tool-envelope changes, no `ToolSpec.metadata` rework, no provider session base class, no `HarnessConfig` regrouping, no schema caching.
- Do not change any model-facing message text (limit notices, background completion messages, retry prompts).
- Do not change `ModelSession` method signatures or provider payload shapes.
- Do not export `types` module contents, `TurnDriver`, or any new helper from `thinharness/__init__.py`.
- Do not rename `runtime.py` members that tests or tracing reference (`RunContext`, `advance_model`, `ModelTraceSnapshot` kinds).

## Considerations
- The riskiest step is 6 (exception ladder): the `HarnessError` branch's stop-reason precedence (`cancelled_by_hook` set before the raise must survive) and the ProviderError wrap-vs-reraise distinction are easy to flatten incorrectly. Write the classifier from the table in step 6, not from memory of the old code.
- `_build_resume_state`'s `require_dump_state` flag and the JSON round-trip isolation are load-bearing for resume tests — move, don't "improve".
- Watch for `__init__.py` star-of-names drift: `LimitNoticeKey` is not currently exported publicly and should stay that way after moving.
- If pyright complains about `HarnessConfig` annotations in moved `runtime.py` helpers, the `TYPE_CHECKING` import plus `from __future__ import annotations` (already present) resolves it; do not import `core` at runtime to fix a type error.
- Plan 15 (`.plans/15-runtime-context-and-tool-execution-architecture.md`) chose to keep `_compute_limit_notices` importable from `core` for tests; this plan supersedes that decision by updating the test import instead — flagging the conflict explicitly per repo convention.
