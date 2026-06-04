# Search Tool Improvements Plan

## Context

ThinHarness currently has two filesystem search tools:

- `search`: text search backed by `rg --json`, with output formatting adapted from PGR's code-search use case.
- `jsonl_search`: structured JSONL row search, optionally prefiltered by ripgrep and then filtered/projected in Python.

The product direction is business/document agents, not code-navigation agents. The current code-oriented search output spends tokens on ranking concepts such as definitions, source/test buckets, and next-step file reads that are less useful for business documents.

## Goals

1. Make both search tools return compact, grouped, document-friendly output.
2. Keep path policy safety intact for restricted read allowlists.
3. Preserve spill-to-file behavior, but make spilled files obvious and readable.
4. Treat partial ripgrep failures as recoverable when usable matches are available.
5. Raise search display limits to fit the normal JSONL/document search use case.

## Non-Goals

- Do not add a second code-specific search mode yet.
- Do not add new dependencies.
- Do not change write/edit/list/glob behavior.
- Do not broaden filesystem access beyond configured read paths, except for harness-owned spill artifacts.

## Proposed Behavior

### Shared Spill Output

Change the default spill directory from `.fsharness/outputs` to project-local `.thinharness/outputs`.

Keep `output_dir` configurable through the existing `FileTools(..., output_dir=...)` and `HarnessConfig(output_dir=...)` settings, but require it to stay under the workspace root. This preserves the sandbox/cloud pattern where generated artifacts live in the shared workspace filesystem and can be read back by root-scoped tools.

When output exceeds `max_chars`, keep saving the full output to a file, but update the returned preview to explicitly tell the model how to read it safely:

```text
[truncated 92341 chars to 40000; full output saved to .thinharness/outputs/search-...txt]
Read the saved output with read(path=".thinharness/outputs/search-...txt", offset=1, limit=400), then continue with later offsets as needed.

<head preview>
...
<tail preview>
```

Keep the current head/tail split because it preserves both the beginning of the result set and the end-of-output notes. Do not switch to first-N-only unless later evidence shows tail previews are confusing.

Metadata should include:

- `truncated: true`
- `saved_to`: absolute path for callers
- `saved_to_display`: workspace-relative path for model-facing follow-up reads, produced with `self._display(artifact)`
- `chars`: full output character count

Make generated spill artifacts readable through the `read` tool even when `read_paths` is restricted. The read allowlist should remain strict for user/workspace files.

The read exception must be artifact-specific:

- Resolve the requested path under the workspace root before applying the exception.
- Allow only paths under the resolved `self.output_dir`.
- Track generated spill artifact paths and allow only those exact artifacts, not arbitrary pre-existing files under `output_dir`.
- Keep rejecting path traversal, root escapes, and symlink escapes after resolution.
- Apply the same rule to custom workspace-local `output_dir` values, not only the default `.thinharness/outputs`.

Add `.thinharness/` or `.thinharness/outputs/` to `.gitignore` so runtime spill files do not appear as source changes.

### `search` Output

Replace code-navigation formatting with pure grouped text search formatting.

Current style:

```text
  summary:
    query: refund
    scope: all files
    files: 2 total, 2 shown
    buckets: 2 source, 0 test, 0 low-priority
    definition_candidates: 0
    best_next_step: read policies/refunds.md around line 12

policies/refunds.md
  why: reference, source
  12-12:
    12| Refunds are available within 30 days.
```

New style:

```text
summary:
  query: refund
  scope: all readable files
  files: 2 total, 2 shown
  matches: 4 shown, 0 omitted

policies/refunds.md
  12: Refunds are available within 30 days.
  18: Refund exceptions require manager approval.
  ... 2 more match(es)

claims/customer_1024.txt
  7: Customer requests a refund for a damaged item.
  14: Previous refund was denied due to missing receipt.
```

Remove these code-specific concepts from default output:

- definition detection
- source/test/low-priority buckets
- `why: definition/reference`
- `best_next_step`
- code-specific sorting

Keep:

- `rg --json`
- grouping by path
- line numbers
- line preview truncation
- `path_glob`, `file_type`, and configured exclude globs
- read allowlist filtering on every parsed match
- no-match guidance
- per-file and total omitted match counts
- spill-to-file for long output

Sort grouped output deterministically by path and then line number after policy filtering. Do not preserve ripgrep filesystem traversal order, because it is not semantically meaningful for document search and is weaker for repeatable agent behavior.

Remove the public code-ranking configuration because it no longer has product meaning for document search:

- `search_low_priority_dirs`
- `search_test_dirs`
- `DEFAULT_SEARCH_LOW_PRIORITY_DIRS`
- `DEFAULT_SEARCH_TEST_DIRS`

Also remove ranking-only helpers and fields that become unused, including definition detection and file-priority helpers.

### `jsonl_search` Output

Change row output from repeating `path:line:` on every row to grouped file blocks.

Current style:

```text
events.jsonl:1: {"user.name": "alice", "msg": "logi..."}
events.jsonl:3: {"user.name": "carol", "msg": "logi..."}
```

New style:

```text
summary:
  query: login
  scope: glob=*.jsonl
  fields: user.name, msg
  files: 1 total, 1 shown
  rows_matched: 2

events.jsonl
  1: {"user.name": "alice", "msg": "logi..."}
  3: {"user.name": "carol", "msg": "logi..."}
```

If rows are omitted for a shown file:

```text
  ... 12 more row(s)
```

If files are omitted:

```text
note: 4 more file(s) omitted
```

Keep:

- default `path_glob="**/*.jsonl"`
- optional ripgrep prefilter
- Python JSON parsing
- `where` filters
- field projection
- JSON parse error count
- read allowlist filtering
- spill-to-file for long output

Sort grouped JSONL output deterministically by path and then line number before applying display limits.

### Ripgrep Return Code Handling

Ripgrep return code meanings relevant here:

- `0`: matches found
- `1`: no matches
- `2`: error

Current behavior treats return code `2` as fatal. New behavior:

- If return code is `2` and parsed `rg --json` output contains usable match rows, return `ok=True` with partial results.
- Include warning metadata such as:
  - `returncode: 2`
  - `warning: "ripgrep returned 2; showing parsed partial matches"`
  - `warning_excerpt`: a compact excerpt from combined ripgrep output if available
- If return code is `2` and no usable matches were parsed, keep returning `ok=False`.

This lets searches survive nonfatal traversal/read errors while still failing loudly on invalid regexes or fully broken commands.

`jsonl_search` needs an explicit scan metadata path for this. Change `_candidates()` so it can return parsed candidate rows plus warning metadata, then merge that metadata into the final successful `ToolResult` after `_truncate()`. The current `(iterator, ToolResult | None)` shape cannot represent partial success with warning metadata.

### Missing Search Roots

Keep existing behavior: configured read roots that do not exist are ignored for search commands.

If no configured search roots exist, return a successful no-match result with metadata showing `returncode: 1`. Do not call ripgrep with nonexistent roots.

### Default Limits

Raise display defaults for document/JSONL search:

- `SearchArgs.max_files`: from `10` to `50`
- `SearchArgs.max_matches_per_file`: from `3` to `10`
- `JsonlSearchArgs.max_files`: from `10` to `100`
- `JsonlSearchArgs.max_matches_per_file`: from `3` to `25`

Keep global `max_tool_chars` as the real context protection mechanism, with spill-to-file for full output.

## Implementation Steps

1. Update spill directory and read access.
   - Change default `output_dir` to `.thinharness/outputs`.
   - Preserve configurable workspace-local `output_dir`.
   - Add `saved_to_display`.
   - Add model-facing read guidance with `offset`/`limit`.
   - Permit `read` access to generated spill artifacts even when `read_paths` is restricted.
   - Add `.thinharness/` or `.thinharness/outputs/` to `.gitignore`.

2. Simplify `search` formatting.
   - Remove code-ranking sort from default output path.
   - Replace `_format_search_output` with grouped path/line formatting.
   - Keep `_parse_contained_rg_json` and path policy checks.
   - Include per-file and total omitted match counts.
   - Remove public ranking config fields and unused ranking helpers.

3. Group `jsonl_search` formatting.
   - Emit one path header per file.
   - Emit indented `line: projected_json` rows.
   - Keep per-file and total omission notes.
   - Sort grouped output by path and line number.

4. Handle `rg` return code `2` as partial success when possible.
   - Apply to both `search` and query-prefiltered `jsonl_search`.
   - Add scan warning metadata threading for `jsonl_search`.
   - Preserve hard failure when no matches can be parsed.

5. Raise default result limits.
   - Update Pydantic arg defaults.
   - Use higher defaults for `jsonl_search` because JSONL rows are usually compact and the tool is often used for structured corpus scans.
   - Update tests and any docs that mention defaults.

6. Update model-facing text and docs.
   - Change the `search` tool description away from code/ranking language.
   - Update default prompt wording if it still implies symbols/code search.
   - Update architecture docs that mention source/test ranking, definition candidates, search ranking bucket config, or `.fsharness/outputs`.

## Tests

Add or update tests for:

- `search` grouped document-style output.
- `search` no longer emits `why`, `buckets`, `definition_candidates`, or `best_next_step`.
- existing search tests no longer assert code-ranking order or code-ranking reasons.
- `search` reports per-file and total omitted match counts when display limits hide matches.
- `jsonl_search` emits one path header with multiple row lines.
- `jsonl_search` omission notes do not repeat the path unnecessarily.
- spill files are written under `.thinharness/outputs`.
- `.thinharness/` or `.thinharness/outputs/` is ignored by git.
- spill result includes `saved_to_display` and read guidance with `offset`/`limit`.
- spilled output can be read when `read_paths` is restricted.
- non-spill files under `output_dir` remain blocked when `read_paths` is restricted.
- spill read access rejects traversal, root escapes, and symlink escapes.
- `rg` return code `2` with parseable matches returns `ok=True` and warning metadata.
- `rg` return code `2` with no parseable matches returns `ok=False`.
- `jsonl_search` preserves rc `2` warning metadata on its final successful result.
- nonexistent read roots are ignored for both `search` and `jsonl_search`.
- raised JSONL defaults surface more files and rows than the old `10`/`3` defaults.

Run:

```bash
uv run pytest tests/test_file_tools.py tests/test_parallel_tools.py
uv run ruff check thinharness/tools/filesystem.py thinharness/tools/jsonl.py tests/test_file_tools.py tests/test_parallel_tools.py
uv run pyright
```

## Decisions

- No backwards compatibility is needed for `.fsharness/outputs`; use a clean rename to `.thinharness/outputs`.
- Overflow artifacts should be project-local by default, not under `~/.thinharness`, because workspace tools need root-scoped relative paths that work in local, sandboxed, and cloud runs.
- `output_dir` remains configurable, but must resolve under the workspace root.
- `search` output should sort deterministically by path and line number.
- `search` should report omitted match counts.
- `jsonl_search` should use higher default display limits than `search` because rows are compact and the use case is structured corpus search.
- Remove the public code-ranking configuration fields instead of keeping dormant knobs.
