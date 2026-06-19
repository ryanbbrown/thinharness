# JSONL Field Search Snippets Plan

## Goal

Add a compact field-level search mode to `jsonl_search` so callers can select JSONL rows with the existing query/where machinery, then search inside one or more large string fields and return only matching internal lines/snippets.

This is meant for structured JSONL rows that contain large multiline strings such as accessibility trees, browser state dumps, transcript blobs, logs, or rendered document text. The caller often does not need the whole field; they need the few lines inside the field that match a term or regex.

## Problem

Current `jsonl_search` has grep capabilities at the row-selection stage:

1. Optional ripgrep query finds candidate JSONL lines/rows.
2. Python parses candidate rows.
3. `where` filters keep or reject rows.
4. `fields` projects whole values from each kept row.

That works when projected fields are small. It is weak when one projected field is a huge multiline string. A row may match, but returning `accessibility_tree` or `text` still produces one large blob. If the caller sets a per-field cap, the important lines can be clipped. If the caller sets `0` for no per-field truncation, the whole tool output can still hit the global tool cap and spill to a file.

The missing behavior is not more row filtering. It is an output transformation: after a row is selected, search inside a selected string field, split it into lines, and return only the matching lines plus optional nearby context.

## Proposed API Shape

Add `field_searches` to `JsonlSearchArgs`:

```python
class JsonlFieldSearch(StrictArgs):
    field: str
    query: str
    regex: bool = False
    case_sensitive: bool = False
    context_lines: int = 0
    max_matches: int = 20
    max_line_chars: int = 300

class JsonlSearchArgs(StrictArgs):
    query: str = ""
    path: str = "."
    fields: dict[str, int] = Field(default_factory=dict)
    where: list[JsonlWhereFilter] = Field(default_factory=list)
    field_searches: list[JsonlFieldSearch] = Field(default_factory=list)
    max_files: int = 100
    max_matches_per_file: int = 25
    timeout: int | None = None
    max_chars: int | None = None
```

Example call:

```json
{
  "path": "trajectories/1d56a4d6/states.jsonl",
  "where": [{"field": "state_index", "op": "eq", "value": "11"}],
  "fields": {"state_index": 0, "url": 0},
  "field_searches": [
    {
      "field": "accessibility_tree",
      "query": "Incident|-- None --|Edit personal filters",
      "regex": true,
      "context_lines": 0,
      "max_matches": 20,
      "max_line_chars": 300
    }
  ]
}
```

Example output:

```text
summary:
  query: (none)
  scope: path=trajectories/1d56a4d6/states.jsonl
  where: state_index eq '11'
  fields: state_index, url
  field_searches: accessibility_tree
  files: 1 total, 1 shown
  rows_matched: 1

trajectories/1d56a4d6/states.jsonl
  12: {"state_index": 11, "url": "https://..."}
    accessibility_tree matches:
      426: [a800] menuitem 'Edit personal filters', visible
      428: [a802] menuitem '-- None --', visible
      433: [a807] menuitem 'Incident Mobile', visible
      434: [a808] menuitem 'Incident Portal', visible
      435: [a809] menuitem 'My Open Incidents', visible
```

## Semantics

`query` keeps its current meaning: optional ripgrep prefilter over JSONL rows. `where` keeps its current meaning: structured row filtering after JSON parsing. `fields` keeps its current meaning: normal projected row values.

`field_searches` runs only after a row has already passed `query` and `where`. It does not decide candidate rows by itself unless `query` is omitted and `where` selects rows. Each field search:

- Resolves `field` with the same jq-style field path parser used by `fields` and `where`.
- Requires the resolved field value to be a string. Missing, null, object, array, number, and boolean values produce a compact note and no matches for that field.
- Splits the string with `splitlines()`.
- Matches each internal field line with either substring search or regex search.
- Includes `context_lines` lines before and after each match, merging overlapping ranges.
- Returns at most `max_matches` primary matches per field search per row. Context lines do not count as primary matches.
- Truncates each returned internal line to `max_line_chars`.

`context_lines` is the same idea as `grep -C`: `0` returns only matching lines, `1` returns one neighboring line before and after each match, and so on.

## Multiple Field Searches

Support multiple searches per row. This lets callers inspect more than one large field without issuing multiple tool calls:

```json
{
  "fields": {"state_index": 0},
  "field_searches": [
    {"field": "accessibility_tree", "query": "Incident", "regex": false},
    {"field": "thought", "query": "Filters", "regex": false, "context_lines": 1}
  ]
}
```

Output should group snippets under the field name:

```text
  12: {"state_index": 11}
    accessibility_tree matches:
      ...
    thought matches:
      ...
```

If the same field appears multiple times in `field_searches`, preserve call order and include a short query label so the outputs are distinguishable.

## Validation Rules

- `field` must be a non-empty string.
- `query` must be a non-empty string.
- `context_lines` must be `>= 0`.
- `max_matches` must be `>= 1`.
- `max_line_chars` must be `>= 1`.
- Invalid regex should fail the search with `invalid field_search regex`.
- Invalid field paths should fail the search with `invalid field path`, matching existing projection behavior.

## Output Details

If no internal field lines match for a selected row, show a compact zero-match line only when no normal fields were projected and all field searches missed. Avoid noisy no-match blocks when normal projected fields already make the row useful.

Suggested zero-match output:

```text
  12: {}
    accessibility_tree matches: none
```

When matches are omitted because `max_matches` was reached, include:

```text
      ... 14 more match(es)
```

This omitted count should count additional primary matching lines, not context lines.

The global spill/truncation behavior should remain unchanged. The new mode is intended to reduce how often useful output spills by returning snippets instead of full large fields.

## Implementation Sketch

1. Add `JsonlFieldSearch` and `field_searches` to `thinharness/tools/jsonl.py`.
2. Compile field searches before scanning rows:
   - parse field path once
   - compile regex once when `regex=true`
   - normalize case behavior
3. Extend the row formatting path in `JsonlSearch.search()`:
   - keep existing `_project_fields(row, fields)` behavior for `fields`
   - compute `field_search` snippet blocks for each shown row
   - render projected fields and snippet blocks together under the row line
4. Add helpers:
   - `_compile_field_searches(...)`
   - `_field_search_matches(row, compiled_searches) -> list[RenderedFieldSearch]`
   - `_line_ranges_for_matches(...)` to merge context windows
   - `_truncate_internal_line(...)`
5. Update the summary header to list searched fields when `field_searches` is non-empty.
6. Update `DEFAULT_JSONL_SEARCH_INSTRUCTIONS` with a concise note that `field_searches` is for extracting matching lines from large string fields.
7. Add focused tests in `tests/test_file_tools.py`.

## Test Plan

Add tests for:

- Field search returns matching internal lines from a large multiline string.
- Field search works when `query` is omitted and `where` selects the row.
- Top-level `query` still acts only as row prefilter.
- Normal `fields` projection and `field_searches` render together.
- Multiple field searches render in order.
- Duplicate field searches are distinguishable.
- `context_lines` includes neighboring lines.
- Overlapping context ranges merge without duplicate output.
- `max_matches` limits primary matches and reports omitted match count.
- `max_line_chars` truncates returned internal lines.
- Substring search is case-insensitive by default.
- `case_sensitive=true` changes substring matching.
- Regex search works.
- Invalid regex fails the search with a clear error.
- Missing/non-string field values do not crash and produce compact no-match behavior.
- Invalid field path fails consistently with existing field projection errors.
- Existing `jsonl_search` behavior without `field_searches` is unchanged.
- Tool schema exposes `field_searches` and its validation constraints.

Validation commands:

```bash
uv run pytest tests/test_file_tools.py -k jsonl_search
uv run pytest tests/test_file_tools.py
uv run ruff check thinharness/tools/jsonl.py thinharness/defaults.py tests/test_file_tools.py
uv run pyright
```

## Non-Goals

- Do not add a benchmark-specific trajectory inspection tool.
- Do not remove global tool output truncation or spill-to-file behavior.
- Do not change the meaning of top-level `query`.
- Do not make `fields` do two jobs; normal projection stays in `fields`, snippet extraction goes in `field_searches`.
- Do not add OR filters, sorting, scoring, or ranking.
- Do not add a dependency for regex or text extraction.

## Open Checks During Implementation

- Confirm whether the model schema remains understandable with nested `field_searches`; if needed, add examples to the default tool instructions rather than broad docs rewrites.
- Decide whether no-match snippet blocks should always be shown or only when the row would otherwise be empty. Prefer compact output unless tests show ambiguity.
- Confirm line numbering should be 1-based within the field string, not source-file line numbers. The row already has the JSONL source line number.
