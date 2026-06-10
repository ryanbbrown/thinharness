# Search Support Extraction

## Goal

Move the search-only helpers that currently live in `base.py` and inside `FileTools` into one small `tools/search_support.py` module, so that `search` (in `filesystem.py`) and `jsonl_search` (in `jsonl.py`) both depend on the same stateless helpers instead of `JsonlSearch` reaching back into `FileTools` through injected callbacks.

No new class. No new abstraction layer. Just relocate pure functions and shrink the callback surface from four to one.

## Why

`JsonlSearch` does not import `filesystem.py` — at the import level it is already independent. But `FileTools` injects four of its own bound methods/lambdas into `JsonlSearch` at construction:

- `truncate` — spill/truncate behavior
- `parse_rg_json` — contained ripgrep JSON parsing
- `path_allowed` — read-policy containment check
- `search_roots` — existing readable search roots as display paths

Three of those four only need `root` + `read_policy`, which are plain values, not behavior. They can become stateless helpers that both tools import. The fourth — `truncate` — is genuinely `FileTools` behavior: it registers spilled artifacts in `FileTools._spill_artifacts`, which `read` consumes so a spilled result stays readable under a restricted `read_paths`. That one stays a callback.

End state: **4 callbacks -> 1 callback (`truncate`) + 1 plain value (`read_policy`).**

This is not about making `JsonlSearch` deletable — deleting it is already a four-line edit (drop the import, the construction block, and `self.jsonl.spec()` from `specs()`). It is about (a) getting search-only helpers out of the generic `base.py`, and (b) removing the awkward "parse rg output via an injected callback" wiring.

## Non-Goals

- No `SearchSupport` dataclass or any new class. (An earlier draft proposed one; dropped — a frozen dataclass could not absorb `truncate` without dragging the spill lifecycle, so it bought nothing.)
- Do not move `PathPolicy` or path-policy helpers out of `base.py`.
- Do not move `_timeout_error_message` out of `base.py` (resolved below).
- Do not rename `filesystem.py` or merge `jsonl.py` into it.
- Do not change tool names, argument schemas, output format, ranking, truncation semantics, or error messages.
- Do not touch `parallel_llm.py`, `skills.py`, or MCP.

## What Moves Where

### New: `thinharness/tools/search_support.py` (depends only on `base.py`)

Moved from `base.py`:
- `validate_glob_selector`
- `_rg_error_message`
- `_rg_partial_warning_metadata`
- `_compact_rg_excerpt`
- `_is_rg_json_match_or_context`

Moved from `filesystem.py`:
- `SearchMatch`, `SearchFile`
- `_parse_rg_json` (was a `FileTools` staticmethod; becomes a module-private function)

New, extracted from existing `FileTools` code so both tools share one copy:
- `parse_contained_rg_json(stdout, root, read_policy) -> list[SearchFile]` — was `FileTools._parse_contained_rg_json` + `_search_file_allowed`
- `search_root_display_paths(root, read_policy) -> list[str]` — was the `search_roots=` lambda, also duplicated inline in `FileTools.search`

`search_support.py` imports `PathPolicy`, `contained_path`, `PathValidationError` from `base.py`. No circular import (`base.py` never imports `search_support.py`).

Keep the existing helper names (including the leading underscores) to avoid rename churn; cross-module import of underscore-prefixed helpers is already the established pattern here (`jsonl.py` imports `_rg_error_message` from `base.py` today).

### From bound methods to plain functions

The mechanical heart of this change: each helper that was a `FileTools` method becomes a standalone module function, and the `self.*` values it used to reach for implicitly are passed in as explicit arguments instead. These helpers only ever read `self.root` and `self.read_policy` (plain values), so they convert cleanly:

| Before (method on `FileTools`) | After (function in `search_support.py`) |
|---|---|
| `@staticmethod _parse_rg_json(stdout)` | `_parse_rg_json(stdout)` — already `self`-free; just relocated |
| `self._parse_contained_rg_json(stdout)` | `parse_contained_rg_json(stdout, root, read_policy)` |
| `self._search_file_allowed(path)` | folded into `parse_contained_rg_json` (it needs `root` + `read_policy`) |
| `search_roots=lambda: [self._display(p) for p in self.read_policy.existing_search_roots()]` | `search_root_display_paths(root, read_policy)` |

`self._truncate` is the one helper that does **not** convert this way: it reads and mutates `self._spill_artifacts`, shared state owned by `FileTools` and consumed by `read`. An explicit-args function can't reproduce that, so it stays a bound-method callback.

### `base.py` (after)

Keeps generic tool contracts, invocation, schema helpers, and all path-policy code. Keeps `_timeout_error_message` — `skills.py` uses it for skill-script timeouts, so it is a generic timeout string, not search-specific. Loses only the five search helpers listed above.

### `filesystem.py` (after)

- Imports `SearchMatch`, `SearchFile`, `parse_contained_rg_json`, `search_root_display_paths`, `validate_glob_selector`, `_rg_error_message`, `_rg_partial_warning_metadata` from `search_support.py`.
- `FileTools.search` calls `parse_contained_rg_json(...)` and `search_root_display_paths(...)` instead of its own methods.
- Removes `FileTools._parse_rg_json`, `_parse_contained_rg_json`, `_search_file_allowed`.
- Keeps `_truncate`, `_display`, `_format_search_output`, `_no_matches_message`, `_describe_search_scope`, `_truncate_line`, `_exclude_glob`, and all read/write/edit/list/glob code.
- Constructs `JsonlSearch` with `root`, `read_policy`, and the single `truncate` callback.

### `jsonl.py` (after)

- `JsonlSearch.__init__(root, read_policy, *, max_tool_chars, rg_timeout, truncate)`.
- `_candidates` uses `parse_contained_rg_json(...)` and `search_root_display_paths(...)`.
- `_iter_jsonl_files` uses `read_policy.allows(...)`.
- Imports the pure helpers from `search_support.py`; keeps `ToolResult`, `ToolSpec`, `Json`, `StrictArgs`, `PathPolicy`, `PathValidationError`, `_path_error`, `_timeout_error_message`, `coerce_args` from `base.py`.
- Keeps all JSONL where/projection/jq logic unchanged.

## Callback Reduction

| Callback (current) | Future |
|---|---|
| `truncate=self._truncate` | KEPT — bound to `FileTools._spill_artifacts`, consumed by `read` |
| `parse_rg_json=self._parse_contained_rg_json` | REMOVED -> `parse_contained_rg_json(stdout, root, read_policy)` |
| `path_allowed=self.read_policy.allows` | REMOVED -> `read_policy` passed as a value; `JsonlSearch` calls `read_policy.allows` |
| `search_roots=lambda: ...` | REMOVED -> `search_root_display_paths(root, read_policy)` |

## Implementation Order

1. Create `search_support.py`; move the five search helpers from `base.py`; update imports in `filesystem.py` and `jsonl.py`. Run checks — pure relocation, no behavior change.
2. Move `SearchMatch` / `SearchFile` / `_parse_rg_json` from `filesystem.py` to `search_support.py`; add `parse_contained_rg_json` and `search_root_display_paths`. Point `FileTools.search` at the shared helpers; delete the now-dead `FileTools` methods.
3. Change `JsonlSearch.__init__` to take `root` + `read_policy` + `truncate`; update the `FileTools` construction block; update `_candidates` / `_iter_jsonl_files` to use the shared helpers and `read_policy`.

Each step is independently testable; run the focused checks between steps.

## Validation

```bash
uv run pytest tests/test_file_tools.py
uv run ruff check thinharness/tools/base.py thinharness/tools/filesystem.py thinharness/tools/jsonl.py thinharness/tools/search_support.py
uv run pyright
```

Tests import only `FileTools`, `PathValidationError`, `SearchArgs`, and `JsonlSearchArgs`, so moving the internal helpers requires no test changes. Run any JSONL-specific test file too if one exists separately.

## Success Criteria

- `JsonlSearch` receives one callback (`truncate`) plus plain values; the three policy/parse callbacks are gone.
- `search` and `jsonl_search` share rg parsing, containment filtering, and search-root resolution through `search_support.py`.
- `base.py` no longer carries search-only helpers but still owns generic tool and path-policy code.
- Public tool behavior (names, schemas, output, truncation, errors) is unchanged.
