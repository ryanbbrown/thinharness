# Remove Background Tools Plan

## Goal

Fully remove the per-run background tool execution feature (added in `.plans/19-background-tools.md` and refined in `.plans/25-background-completion-semantics.md`). The feature lets a model start a long-running tool, continue other work, and receive the completion later in the same run. It is the single largest opt-in subsystem in the harness (~300 source references across 15 modules) and is mostly relevant to interactive coding agents, not the predefined business agents ThinHarness targets. Removing it shrinks the run loop, the tool-execution layer, the event/projection surface, and the approval-resume envelope.

No backwards compatibility is required. Saved resume/approval envelopes produced by the current version are allowed to break.

## Scope Decisions

- **Remove from**: `thinharness/` source, tests, and current-facing documentation (README, `docs/docs.md`, `docs/behavior.md`, the doc site, the run-loop diagram, CHANGELOG).
- **Keep as historical record** (do not edit): `.plans/19-background-tools.md`, `.plans/25-background-completion-semantics.md`, and other `.plans/*` that mention background in passing; the existing `## [0.2.0]` CHANGELOG "Added" entries. These are an engineering journal, like git history.
- **Add** a CHANGELOG "Removed" entry under a new top section for the next release.
- **Doc site**: edit sources, regenerate the about page via `scripts/build_site.py`, hand-edit `docs/site/explainer/index.html`, re-export the run-loop SVG, then push and verify the deploy (per `CLAUDE.md`'s post-push deploy check).
- **Explicitly NOT background-feature references — must stay untouched**:
  - `docs/site/assets/site.css` (~101 hits): CSS `background:` / `background-color:` properties.
  - `examples/web_research_report/outputs/**` and `.thinharness/outputs/**` (`*.jsonl`, `*.txt`): research content that happens to contain the word "background".
  - `docs/docs.md` Bash Prototype Tool line: "Background descendants left by a command are cleaned up when the shell exits" — this is about OS child processes, not the feature.
  - `docs/behavior.md` `### Requirements` is generic plan-template guidance; only the illustrative `BACKGROUND-1` prefix example is touched (see Docs section).

## Public API Removed

From `thinharness/__init__.py` and `thinharness/tools/__init__.py` `__all__` + imports:

- `BackgroundTaskStartedEvent`
- `BackgroundTaskCompletedEvent`
- `BackgroundPolicyDecision`
- `ToolBackgroundMode`

`ToolSpec` loses its `background` and `background_policy` fields. `SubAgentConfig` loses its `background` field. These are constructor-visible API changes; acceptable under "no backwards compatibility".

## Source Changes

Order chosen so that leaf modules (types, events) change before their consumers (runtime, core), making each pyright pass meaningful.

### `thinharness/tools/base.py`

- Delete `ToolBackgroundMode` (line 19) and `BackgroundPolicyDecision` (lines 33–40).
- `ToolSpec`: remove `background` (55) and `background_policy` (58) fields.
- `__post_init__`: remove the background-mode validation (63–64), the `sequential` + background check (67–68), and the `requires_approval` + background check (69–70). Keep the `kind` and `max_retries` validation.
- `response_tool(...)`: drop the `include_background` parameter and the `_add_background_parameter(parameters)` call; return the plain schema (74–84).
- Delete `_add_background_parameter(...)` (371–382).

### `thinharness/tool_execution.py`

This module is ~113 background references; most of it is background-only.

- Delete dataclasses: `BackgroundToolStart` (45–54), `BackgroundToolTask` (57–67), `BackgroundToolCompletion` (69–93).
- Delete `BackgroundToolManager` class entirely (96–322).
- Delete `background_completion_message(...)` (325–335).
- `ToolCallExecution`: remove `background_start` field (42).
- `ToolBatchExecutor.execute_batch(...)`: drop `_start_background(execution)` calls in both sequential and concurrent paths; in record construction, drop the `if execution.background_start is not None: record["background"] = ...` block (386–390).
- Delete `ToolBatchExecutor._start_background(...)` (395–400).
- `ToolCallExecutor.execute_one(...)`: remove `background_start` local; replace the `_background_decision` branch (497–504) with a direct `envelope = await self._call_output(call.name, call.arguments)`; drop `background_start=...` from the `_emit_completed(...)` call and the returned `ToolCallExecution`.
- `ToolCallExecutor._emit_completed(...)`: remove the `background_start` parameter and the `background_task_id` / `background_status` kwargs passed to `ToolCallCompletedEvent` (562–587).
- Delete `_background_decision(...)` (596–630), `_background_start(...)` (632–642).
- Delete `_ParsedBackgroundArgs` (660–668), `_BackgroundDecision` (671–677), `_parse_background_args(...)` (680–708), `_background_start_output(...)` (711–721).
- `_retry_output(...)` (724–726): used only by `_background_decision`. Confirm via grep, then delete.
- Update imports: drop `BackgroundTaskCompletedEvent`, `BackgroundTaskStartedEvent`, `StreamEmitter` (only the deleted manager used it), `RunStreamContext` (verify), `Any` (only the deleted `_base`/`_emit` used it), and now-unused `json`, `Literal`. **Keep** `RunTracer`, `serialize_attribute_value`, and `_CURRENT_STREAM_EMITTER` — the surviving executors still use them. (pyright + ruff F401 backstop this.)

### `thinharness/runtime.py`

- Drop import of `BackgroundToolCompletion, background_completion_message` (30) and the `TYPE_CHECKING` `BackgroundToolManager` import (42).
- Delete `BACKGROUND_COMPLETION_SEPARATOR` (47).
- `TurnDriver.send_tool_outputs(...)` and `send_user_message(...)`: remove `"background_completion"` from the `kind` Literals (105, 129).
- `RunContext`: remove `background: BackgroundToolManager | None` (305) and `ready_background_completion_messages` (308) fields.
- Request-kind handling (422–432): remove `"background_completion"` from the asserts/casts for both request and tool-output Literals.
- `cancel_pending_background(...)` (484–493): delete.
- `pause_for_approval(...)` (around 501–515): remove the `cancelled_background_task_ids` parameter and the `ready_background_completion_messages=` argument passed to `build_approval_envelope`.
- `record_background_completion(...)` (616–617), `drain_ready_background(...)` (620–628), `drain_next_background_batch(...)` (630–638): delete.
- `_join_background_messages(...)` (641–643): delete.
- **Dead `extra_notices` plumbing:** the background sites were the only non-`None` callers of `extra_notices` (threaded through `send_tool_outputs` (107), `_run_model_request` (155), `advance_model` (389)). After this removal it is always `None` and no linter flags it. Remove the parameter from the chain as an orphan (CLAUDE.md "remove orphans your changes made" / simplicity-first), unless deliberately kept as a notice-extension seam.

### `thinharness/core.py`

- Constructor validation (159–161): delete the `background="always"` subagents + sequential check.
- `_run_streaming(...)`: drop `BackgroundToolManager` from the import (395); delete `run_ctx.background = BackgroundToolManager(...)` (429).
- Run loop:
  - Finalize branch (527–538): remove the `run_ctx.background.has_pending_or_ready()` deferral; finalize directly.
  - max-tool-calls branch (563–571): remove the background-aware rejection; keep the normal limit check that already lives in `_execute_tool_turn` / `check_tool_limit`.
  - Approval path (572–577): drop `cancelled_ids = await run_ctx.cancel_pending_background()`; call `run_ctx.pause_for_approval(turn, approval_calls, active_session)` without the cancelled ids.
  - `finally` block (590–594): remove the background `cancel_and_drain` drain loop; keep `run_ctx.fire_run_end_once()`.
- `_resume_approval_batch(...)` / approval resume: `_approval_resume_notices(...)` (774–778, 826–839) — delete the method and its call; resume with no background notices. Verify `send_tool_outputs(..., extra_notices=...)` callers are updated.
- `_execute_tool_turn(...)` (864–880): remove `drain_ready_background()` and the background-completion `notices`; return `await driver.send_tool_outputs(outputs)`.
- Delete `_defer_final_for_background(...)` (882–902) and `_reject_batch_for_background(...)` (904–928).
- `tool_schemas(...)` (990–999): remove `expose_background` logic; call `tool.response_tool()` plainly.
- `system_instructions(...)` (1001–1010+): delete the background-guidance paragraph and its `any(tool.background == "model" ...)` guard.
- `_validate_tool_background_policy(...)` (1089–1096) and `_validate_tool_background_policy_for(...)` (1098–1112): this static method also enforces the **approval** invariants (`requires_approval` needs a resumable model; not allowed in child runs). Keep those approval checks. Remove only the `background == "always"` check (1107–1108). That makes `tool_execution` the only-now-unused parameter through the chain (`_validate_tool_spec` → forwarded at 1084 → here, plus the instance variant and call sites at 223/980/1169); **drop the `tool_execution` parameter** from the chain (no linter flags it). Rename the methods to approval-only (e.g. `_validate_tool_approval_policy[_for]`) and update the call sites.

### `thinharness/subagents.py`

- Drop `BackgroundPolicyDecision, ToolBackgroundMode` from the `tools.base` import (15).
- `SubAgentConfig`: remove `background` field (46).
- Subagent tool construction (92–97): remove `background="model"` and `background_policy=...`; `_subagent_tool_description(...)` loses its `background_available` argument.
- `_subagent_background_policy(...)` (359–375): delete.
- `_subagent_tool_description(...)` (378–391): remove the `background_available` parameter and the trailing background guidance line.

### `thinharness/tools/parallel_llm.py`

- Tool spec (148): remove `background="model"`.
- Re-exports (36–37) and the `description: str = DEFAULT_PARALLEL_LLM_DESCRIPTION` default param (88): **leave pointing at the kept constant names** (do not rename).
- `_build_parallel_llm_spec`-style code (277–279): drop the `background_available` plumbing; reference `_defaults.DEFAULT_PARALLEL_LLM_DESCRIPTION` / `_INSTRUCTIONS` directly.
- `_parallel_llm_description(...)` / `_parallel_llm_instructions(...)` (300–311): with `background_available` gone these just return the constants — delete them and inline the constants at the call sites.

### `thinharness/defaults.py`

- **Keep the public constant names** `DEFAULT_PARALLEL_LLM_DESCRIPTION` and `DEFAULT_PARALLEL_LLM_INSTRUCTIONS` — they have live consumers a literal-"background" grep misses: `parallel_llm.py:36-37` (re-export) and `:88` (default param), `test_parallel_llm.py:15/699`, `test_harness.py:42/308/331`. Set each to its current `_BASE` content (drop the background sentence) and delete the now-unused `_BASE` and `_BACKGROUND` variants (77–78, 87–90). Net: same two public names, value = base text, no background variants. (Build-breaker if missed — flagged by all three reviewers.)

### `thinharness/events.py`

- Remove `"background_task_started"`, `"background_task_completed"` from the event-kind list (20–21).
- `ModelRequestStartedEvent.request_kind` Literal (82): remove `"background_completion"`.
- `ToolCallCompletedEvent`: remove `background_task_id` (122) and `background_status` (123) fields.
- Delete `BackgroundTaskStartedEvent` (127–133) and `BackgroundTaskCompletedEvent` (137–147).
- Remove both from the `StreamEvent` union (193–194).

### `thinharness/projections.py`

- `ModelRequestKind` (22) and the two Literals (37, 53): remove `"background_completion"`.
- Delete the `if delta.kind == "background_completion": return {"background_completion": content}` branch (120–121).

### `thinharness/providers.py`

- `ModelNotice.kind` Literal (98): remove `"background_cancelled"` and `"background_completion"`. Verify no `ModelNotice` rendering switch elsewhere needs those arms; the generic `<harness_notice kind="...">` wrapper is kind-agnostic, so likely nothing else changes.

### `thinharness/approvals.py`

- `_ENVELOPE_KEYS` (15–28): remove `"cancelled_background_task_ids"` and `"ready_background_completion_messages"`.
- `ApprovalPause` (52–53): remove both fields.
- `build_approval_envelope(...)` (61–94): remove both parameters and both envelope keys.
- `validate_approval_pause_state(...)` (97–135): remove the `missing.discard(...)` line (106), and the `cancelled_ids` / `ready_messages` parsing + `ApprovalPause(...)` args.
- **Envelope version**: keep `APPROVAL_ENVELOPE_VERSION = 1` (all three reviewers concur). Old envelopes carrying the two removed keys now fail the "unknown keys" check — acceptable under "no backwards compatibility". (Bumping to v2 would only give a friendlier stale-envelope error; not worth it here.)

### `thinharness/__init__.py` and `thinharness/tools/__init__.py`

- Remove the four symbols above from imports and `__all__`.

## Tests

Background-only tests are deleted; mixed files have their background cases removed. (Confirmed: `test_tracing.py`, `test_parallel_tools.py`, `test_tool_retry.py`, `test_harness.py`, `test_resume.py`, `test_hooks.py`, and `tests/fakes.py` contain **no** background references, so they need no edits.)

- **Delete** `tests/test_background_tools.py` entirely.
- `tests/test_approvals.py`: delete `test_background_task_is_cancelled_and_reported_when_later_turn_pauses` (455–505) and `test_ready_background_completion_is_preserved_across_approval_pause` (507–562). In `test_approval_tool_requires_resumable_model_and_no_background` (786–798), remove the background `pytest.raises(... "approval-required tools cannot use background execution")` block (795–796), **keep** the resumable-model block (797–798), and rename to drop `_and_no_background`. **Delete the whole** `test_approval_pause_state_accepts_old_envelopes_without_ready_messages` (911–930) — it only verifies the now-removed backward-compat tolerance, so stripping its key/assertion would leave a hollow, misleadingly-named test. Remove the `cancelled_background_task_ids` key from the remaining envelope fixture (~900) and drop the `BackgroundTaskCompletedEvent` import (15).
- `tests/test_streaming.py`: delete `test_stream_background_events` (277–**299**) and the `BackgroundTaskCompletedEvent` / `BackgroundTaskStartedEvent` imports (10–11). `SequenceSession` (33) is shared with other tests — do not remove it.
- `tests/test_providers.py`: delete **only** the `background` variable (46) and its assertion (57); **keep** `first`/`second` (44–45) and the `limit_warning` assertions (48–56), which still use them. (Deleting 46–57 wholesale orphans `first`/`second` → ruff F841.)
- `tests/test_subagents.py:81`: **update** the expected property set `{"task", "agent", "_background"}` → `{"task", "agent"}` (do not delete the line; line 82's `"tools" not in …` is a separate, kept assertion).
- `tests/test_web_research_report_example.py`: **delete the whole** `test_build_harness_keeps_critical_path_tools_synchronous` (239–247). Its only assertions are the `_background`-absent checks (245–246), which become tautological once the feature is gone; stripping them leaves a no-assertion test.
- `tests/test_parallel_llm.py` and `tests/test_harness.py`: no background refs, but both consume `DEFAULT_PARALLEL_LLM_*` (see `defaults.py`). Keeping the constant names means `test_parallel_llm.py:699` and `test_harness.py:308/331` should still pass; the constants lose their background sentence, so run both and update any assertion that pinned the background suffix (e.g. a `.endswith(...)` on the old text).

## Docs

### `README.md`

- Line 222 (No token streaming): drop "background" from the emitted-events list.
- Line 266: drop "background" from the streaming events list.
- Line 274 (Custom typed tools): change "sequential/background/approval flags" → "sequential/approval flags".
- **Delete** the "Background tools" feature bullet (283).
- Line 285 (Event streaming): drop "background" from the events list.
- Verify the comparison table has no Background row/column (confirmed none today; re-check after edits).
- **LOC row:** this removal drops a large chunk of `thinharness/`. Re-run `tokei thinharness/ -t Python`, update the ThinHarness LOC value in the README comparison table (currently `8,658`, ~line 61), the "measured from the current working tree on …" date in `docs/table.md` (~line 15), and the LOC snapshot in `docs/site/explainer/index.html` (~line 58). The about page picks up the README value on regeneration.

### Doc site

- `docs/site/about/index.html`: **regenerated** from README — do not hand-edit. Run `uv run python scripts/build_site.py` after the README edits, then `--check` to confirm clean.
- `docs/site/explainer/index.html` (hand-written, not generated): remove the nav link (41) and the entire `Background Mode` section (`<section id="background-mode">` 323–367); remove background mentions in the run-loop prose (88), the streaming-event list (129), the tool-attribute prose (226, 292), and the approval/architecture rows (658, 666, 746, 907). The 706–708 item is a whole prose paragraph ("Background-capable tools use the same execution path…") containing the `#background-mode` cross-link (707) — **delete the entire paragraph**, not just the link. Then confirm no remaining `#background-mode` reference and that `grep -in background docs/site/explainer/index.html` is 0. (LOC snapshot at line 58 handled in the LOC item above.)

### `docs/docs.md`

- Streaming event lists (58, 64): drop background events / `BackgroundTaskCompletedEvent.output` mention.
- Approval section (252): remove the "They cannot use background execution, and" clause; keep the subagent-restriction sentence.
- **Delete** the `## Background Tools` section (≈292–330) including the `background="model"` example and the "Background modes" list.
- Subagent capabilities line (433): remove "and background policy".
- Leave the Bash Prototype Tool "Background descendants" sentence (268) untouched.

### `docs/behavior.md`

- No Background requirements section exists (the feature was never given one). The only reference is the illustrative prefix example on line 14: change "such as `BACKGROUND-1` or `TOOL-APPROVAL-1`" → "such as `TOOL-APPROVAL-1`" (or another live section prefix) so the template stops citing a removed feature. No new section is added.

### `CHANGELOG.md`

- Keep the historical `## 0.2.0` "Added background tool execution…" and "Added SDK event streaming with … background task …" lines as record.
- Add a new top `## X.Y.Z` section with a **flat** `- Removed …` bullet (the CHANGELOG uses flat bullets under version headings, not `###` subsections): "Removed background tool execution and background completion semantics, including `ToolSpec.background`/`background_policy`, `SubAgentConfig.background`, `BackgroundTask*Event`, and the approval-envelope background fields."

### Run-loop diagram

`assets/thinharness-run-loop.drawio` contains a background branch at the finalize node: cells `e-final-background-advance`, `label-background-pending` ("background pending: wait + deliver completion"), and `label-no-background`. Remove these feature cells and reconnect the finalize node straight to "done". Preserve the canvas background rect (`mxCell id="background"`), which is the diagram backdrop, not a feature node — confirm by inspecting its style before deleting anything. Re-export to `docs/site/assets/thinharness-run-loop.svg` (and `assets/thinharness-run-loop.svg` if both are tracked) with `--svg-theme light` per `CLAUDE.md`.

## Validation

1. `uv run pyright` — clean.
2. `uv run pytest` — full suite green. Touched files include `tests/test_approvals.py tests/test_streaming.py tests/test_providers.py tests/test_subagents.py tests/test_web_research_report_example.py`, **plus `tests/test_parallel_llm.py` and `tests/test_harness.py`** (they consume `DEFAULT_PARALLEL_LLM_*`), plus the run-loop/tool-execution coverage in `tests/test_parallel_tools.py tests/test_tool_retry.py tests/test_tracing.py tests/test_resume.py tests/test_hooks.py` for regressions.
3. `ruff check` on every touched file (match the repo's existing invocation).
4. `uv run python scripts/build_site.py --check` — about page regenerated and clean.
5. Final grep gate: `grep -rin "background" thinharness/ tests/ docs/docs.md docs/behavior.md README.md`. The **expected `thinharness/` survivors are exactly two non-feature docstrings** — `tools/mcp.py:37` ("background session task") and `tools/bash.py:133` ("background descendants"); no survivors in tests or those docs. A separate grep over `docs/site/` should show only `site.css` CSS properties; `grep -in background docs/site/explainer/index.html` should be 0.
6. Confirm no orphaned imports/symbols remain (pyright + ruff F401 cover this). Optionally add one positive intent assertion (the dedicated tests are deleted): public tool schemas expose no `_background`, and `thinharness`/`thinharness.tools` `__all__` no longer export the removed background symbols.
7. Push to `main`, wait ~30s, confirm the deploy succeeded (`CLAUDE.md`).

## Behavior Contract

**Decision (user-confirmed): do NOT add any background section to `docs/behavior.md`, not even a "removed capability" note.** Docs and app represent current state only, which has no background. There is no existing background section to delete; the sole mechanical edit is dropping the illustrative `BACKGROUND-1` from the line-14 prefix example so the template no longer cites the removed feature. (This overrides Codex's suggestion to add a positive "no background surface" contract section.)

## Out of Scope

- Editing historical `.plans/*` documents.
- Removing the historical `## [0.2.0]` CHANGELOG "Added" entries.
- The bash tool's OS "background descendants" cleanup behavior and its docs.
- CSS `background:` properties in `docs/site/assets/site.css`.
- Example research-output files containing the word "background".
- Any approval, parallel-tool, or subagent behavior beyond stripping the background hooks.
