# Batch Edit Tool Plan

## Decision

Change the built-in `edit` tool to accept a list of replacements instead of a single one. The schema becomes:

```python
class EditOperation(StrictArgs):
    path: str
    old_string: str
    new_string: str
    all: bool = False
    expected_replacements: int | None = Field(default=None, ge=1)

class EditArgs(StrictArgs):
    edits: list[EditOperation] = Field(min_length=1)
```

List-only, no backwards-compatible flat form. A single edit is a one-element list; one canonical shape is more reliable than two. This is a breaking change to the tool schema, accepted deliberately.

Per-edit matching semantics are unchanged: exact substring match, `old_string` must be unique in the file unless `all=true`, empty `old_string` rejected. `expected_replacements` stays a pure count assertion (it does not bypass the uniqueness check — multi-occurrence replacement still requires `all=true`, matching current `filesystem.py:247-250` behavior). The instructions should state this interaction so models pair `expected_replacements > 1` with `all=true`.

## Why

The harness already serializes edit execution (`sequential=True`, consumed in `tool_execution.py`), and the system prompt already asks the model to emit independent edits in one assistant turn (`defaults.py:11-12`). GPT models ignore that prompt because they are trained around single-call batching (`apply_patch` carries many hunks in one call), not parallel tool-call emission. A list parameter works with that prior instead of fighting it, collapses N model round-trips into one, and makes sample transcripts readable (one call with a list instead of 14 stacked calls).

We considered porting codex `apply_patch` (vendored at `vendor/codex/codex-rs/apply-patch/`) and rejected it: the format is off-distribution for non-GPT providers, its fuzzy matching (trim-both-sides, Unicode normalization) can silently apply at the wrong indentation, and it takes the first match with no uniqueness requirement — all in tension with fail-loud. The list-of-exact-edits keeps the safety properties and gets the same batching benefit.

## Application semantics

- Edits apply sequentially in list order. Each edit sees the file as modified by earlier edits in the same call.
- Apply-all, not atomic and not stop-at-first-failure: every edit is attempted, and the result reports a per-edit outcome. This differs from codex (which stops at the first failure) by design — codex hunks are positionally coupled through a file cursor; these edits are independent exact matches, so continuing past a failure is safe and minimizes retry cost (the model resends only the failed items).
- The dependent-edit hazard (an earlier edit making a later edit's `old_string` ambiguous, or consuming text a later edit needed) is already caught loudly by the existing per-edit checks: ambiguity fails with "appears N times", a consumed match fails with "not found". No new mechanism needed.
- Textual checks do not catch *semantic* coupling (e.g. edit 1 changing a signature fails while edit 2 updating its call sites succeeds, leaving transiently broken code). Considered and rejected: halting remaining same-file edits after a failure (codex-style). The per-edit report makes the failure loud, the model fixes the failed item on the next call, and the end state converges; a halt policy adds complexity without changing the converged outcome. Instead, the tool instructions must say batches should contain independent edits, and that after any per-edit failure the model must resolve it before relying on the file's state.
- Per-item errors that today abort the whole call (path-policy violations, file not found) become per-edit failures like any other; sibling edits still apply.

## Result shape

- Message: one line per edit, numbered, with path and outcome, e.g.
  - `1. ok src/a.py: replaced 1 occurrence`
  - `3. FAILED src/b.py: old_string appears 2 times; add more context or set all=true`
- `ok` is `True` only when every edit applied (fail loud). On partial application, `ok=False` and the message must make unambiguous which edits were applied, so the model does not re-send already-applied edits.
- Metadata: `{"applied": N, "failed": M, "results": [...]}` with per-edit path/replacements/error. Per-item error details (including `error_type` for path-policy failures) live inside `results[i]`, not at the top level. This is a deliberate metadata shape change — the old top-level `path`/`replacements` keys go away; no in-repo consumer reads them outside tests, but the change should be noted in the CHANGELOG.

## Changes

1. `thinharness/tools/filesystem.py` — new `EditOperation`/`EditArgs` models; rework `FileTools.edit` to loop the existing single-edit logic, collect per-edit outcomes, and build the combined result. Keep `sequential=True` on the spec.
2. `thinharness/defaults.py` —
   - `DEFAULT_EDIT_DESCRIPTION`: describe the list form ("apply one or more exact text replacements...").
   - `DEFAULT_EDIT_INSTRUCTIONS`: document list order, that each edit sees the file as left by earlier edits, keep edits independent or order them deliberately, per-edit uniqueness rules, the `all`/`expected_replacements` interaction (count assertion only; multi-occurrence needs `all=true`), and that failed items should be retried individually after reading the per-edit results.
   - `DEFAULT_SYSTEM_PROMPT` lines 11-12: replace "emit multiple edit calls in the same assistant turn" with guidance covering three points: batch independent edits into one `edit` call, order dependent edits deliberately within the list, and retry only failed items after reading per-edit results.
3. `tests/test_file_tools.py` — update the description map and the existing `edit` call site (becomes `{"edits": [{...}]}`); add tests for:
   - multi-edit across files, including two edits in different files sharing the same `old_string` (no cross-file state bleed);
   - same-file ordering (later edit matches text produced by an earlier edit);
   - partial failure reporting (`ok=False`, applied edits unambiguously listed, failed edit has reason) and the all-edits-fail case (`applied=0`, message lists every failure);
   - ambiguity introduced by an earlier edit;
   - per-item `all`; per-item `expected_replacements` count mismatch mid-batch (fails that item, siblings continue); `expected_replacements` matching count > 1 with `all=false` still fails (pins the count-assertion-only semantics);
   - per-item path-policy violation and per-item file-not-found, with sibling edits still applying;
   - empty `old_string` item;
   - empty `edits` list and the old flat `{path, old_string, new_string}` shape both rejected — direct calls go through `coerce_args`, so assert with `pytest.raises(ValidationError)` (the retry envelope only applies via the tool-layer path);
   - provider-facing schema (modeled on `test_file_tools_validate_glob_selectors`): `response_tool()` for edit contains no `$ref`/`$defs` after inlining, the nested `EditOperation` object has `additionalProperties: false`, and `edits` has `minItems: 1`.
4. `docs/docs.md:74` — update the `edit` tool bullet to match the new description (e.g. "apply one or more exact text replacements to UTF-8 files; edits apply in order. This tool is sequential.").
5. `CHANGELOG.md` — entry for the breaking tool-schema and metadata-shape change.

## Verification

- `uv run pytest tests/test_file_tools.py tests/test_parallel_tools.py` (the latter asserts the edit spec's sequential flag), plus `tests/test_harness.py` if it exercises edit indirectly → pass.
- `uv run pyright` → clean.
- `uv run ruff check` on touched files → clean.
- One manual run of `e2e/workspace_tools_journey.py` against a live model — it is the only end-to-end check that a model actually copes with the new list-only shape.
- Optional follow-up, not part of this change: regenerate the sample transcript for the docs site so the published trace shows one batched edit call (README/about.html are unaffected unless README changes; `docs/site/about.html` is generated from README via `scripts/build_site.py`).
