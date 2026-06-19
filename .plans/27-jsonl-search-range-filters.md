# JSONL Search Range Filters Plan

## Goal

Add `gt`, `gte`, `lt`, and `lte` operators to `jsonl_search` so callers can filter JSONL rows by numeric and date-like scalar fields.

The feature should remain small, predictable, and compatible with the current `where` shape:

```json
{"field": "score", "op": "gte", "value": "0.8", "type": "number"}
{"field": "published_at", "op": "lt", "value": "2026-06-12", "type": "date"}
```

## Current Behavior

`jsonl_search` is implemented in `thinharness/tools/jsonl.py`.

Current `where` operators:

- `eq`
- `ne`
- `in`
- `contains`
- `regex`
- `exists`

Current comparison behavior renders non-string JSON values to strings for most operators. That is useful for text filters but too ambiguous for range filters.

## Decisions

1. Add the new operators to the existing `where` API.
2. Require an explicit `type` for range operators.
3. Support `type: "number"` and `type: "date"` initially.
4. For `type: "number"`, only real JSON numbers in rows are comparable. Numeric strings in row data do not match.
5. For `type: "date"`, support ISO-ish strings only. Do not support Unix timestamps as date filters in this feature.
6. Date-only filters and date-only row values compare by calendar date. Datetime row values compared to date-only filter values are compared by their calendar date, so `2026-06-12T15:30:00Z <= 2026-06-12` is true.
7. Rows with missing, null, object, array, invalid, or wrong-type comparison values do not match.
8. Search output should warn when at least one row could not be compared for a range filter.
9. Invalid filter definitions should still fail the whole search.
10. Range filters are pre-validated before scanning rows, so invalid filters fail even when the scope has zero candidate rows.

## Proposed API Shape

Update `JsonlWhereFilter`:

```python
class JsonlWhereFilter(StrictArgs):
    field: str
    op: Literal["eq", "ne", "in", "contains", "regex", "exists", "gt", "gte", "lt", "lte"]
    value: str | None = None
    values: list[str] | None = None
    type: Literal["number", "date"] | None = None
```

Rules:

- `gt`, `gte`, `lt`, and `lte` require `value`.
- `gt`, `gte`, `lt`, and `lte` require `type`.
- Existing non-range operators reject `type` if supplied.
- Range operators reject `values` if supplied.
- Keep `value` as `str | None` for now. The filter target is parsed according to `type`.
- Empty string range values are invalid filter definitions and fail the search.

## Number Semantics

For `type: "number"`:

- Filter target must parse as a finite number. Parsing with `float()` is not enough; explicitly reject `NaN`, `Infinity`, and `-Infinity` with `math.isfinite()`.
- Row value must be an `int` or `float`, excluding booleans.
- `NaN`, `Infinity`, and `-Infinity` should not be treated as comparable if encountered.
- Numeric strings in rows do not match.

Examples:

```json
{"score": 9.5}
```

passes:

```json
{"field": "score", "op": "gt", "value": "8", "type": "number"}
```

but:

```json
{"score": "9.5"}
```

does not pass the same filter.

## Date Semantics

For `type: "date"`:

- Filter target must be an ISO-ish date or datetime string.
- Row value must be a string parseable as an ISO-ish date or datetime.
- Supported examples:
  - `2026-06-12`
  - `2026-06-12T14:03:00`
  - `2026-06-12T14:03:00Z`
  - `2026-06-12T14:03:00-04:00`
- Use the Python standard library if possible, likely `datetime.date.fromisoformat()` and `datetime.datetime.fromisoformat()` with a small `Z` to `+00:00` normalization.
- Do not add a dependency just for date parsing unless the standard library approach proves too brittle.

Comparison normalization:

- If either side is date-only, reduce both sides to calendar dates before comparing. Strict operators remain strict calendar-date comparisons, so `2026-06-12T15:30:00Z > 2026-06-12` is false and `2026-06-12T15:30:00Z >= 2026-06-12` is true.
- For timezone-aware datetimes reduced to dates, use the date as written in the row string, not a UTC-normalized date. For example, `2026-06-12T23:30:00-04:00` compares as calendar date `2026-06-12` against a date-only filter.
- If both sides are datetimes, compare datetimes.
- If both datetimes are timezone-aware, compare by instant.
- If one datetime is timezone-aware and the other is naive, treat the row as not comparable rather than guessing a timezone.

This avoids surprising date-only behavior. A filter like `lte 2026-06-12` naturally includes rows from any time on `2026-06-12`.

## Warning Behavior

Rows that cannot be compared should not fail the whole search. They should be counted and treated as non-matching.

Warning count semantics:

- `compare_warnings` is a count of distinct candidate rows where a range comparison was actually attempted and failed because the row value was non-comparable.
- A row is counted at most once even if multiple range filters on that row are non-comparable.
- Rows filtered out by the ripgrep prefilter are not candidate rows and are not counted.
- Rows that fail an earlier `where` filter before a later range filter is attempted are not counted for the later range filter. This keeps the current AND short-circuit behavior.
- Missing, null, object, array, wrong-type, non-finite number, invalid date string, and aware/naive datetime mismatch values count as non-comparable when the relevant range comparison is attempted.

The result header should include a warning only when the count is nonzero, for example:

```text
summary:
  query: (none)
  scope: path=events.jsonl
  where: score gte '0.8' (number)
  files: 1 total, 1 shown
  rows_matched: 3
  compare_warnings: 2 row(s) had non-comparable values
```

Metadata should also expose the warning count for programmatic callers:

```json
{"compare_warnings": 2}
```

Do not use the generic metadata key `warning` for comparison warnings. Ripgrep partial results already use `warning` and `warning_excerpt`; comparison warnings must use `compare_warnings` so both can coexist.

## Implementation Sketch

1. Extend `JsonlWhereFilter.op` and add `type`.
2. Add a compiled/pre-validated representation for `where` filters before candidate iteration starts:
   - validate required and forbidden fields
   - parse range filter targets exactly once
   - fail the search with `invalid where filter` before reading rows when filter definitions are invalid
3. Add helper types/functions in `thinharness/tools/jsonl.py`:
   - parse row scalar values by requested comparison type
   - evaluate `gt/gte/lt/lte`
   - return pass/fail plus whether the attempted range comparison was non-comparable, likely via a small result tuple or enum rather than a bare bool
4. Keep existing string operator behavior unchanged.
5. Update `_describe_where()` to show range filter types only for range filters. Existing operator display should not change.
6. Update `JsonlSearch.search()` to include warning counts in output and metadata.
7. Update `DEFAULT_JSONL_SEARCH_INSTRUCTIONS` with one concise bullet for range filters.
8. Add focused tests in `tests/test_file_tools.py`.

## Test Plan

Add tests for:

- `number` range filters match JSON numbers.
- `gt` and `lt` are exclusive for equal numeric values.
- Numeric strings in row values do not match and increment warning count.
- Booleans do not count as numbers.
- Negative filter targets parse correctly.
- Non-finite row numbers such as `NaN`, `Infinity`, and `-Infinity` do not match and increment warning count if parsed by `json.loads`.
- Non-finite filter targets such as `NaN`, `Infinity`, and `-Infinity` fail as invalid filters.
- `date` filters match ISO date strings.
- Datetime-to-datetime filters work when both sides are naive or both sides are aware.
- Date-only filters include datetimes on the same calendar day for `lte` and `gte`.
- Date-only filters exclude datetimes on the same calendar day for strict `lt` and `gt`.
- Near-midnight timezone-offset datetimes compared to date-only filters use the date as written, not UTC-normalized date.
- Aware-vs-naive datetime comparison does not match and increments warning count.
- Invalid row date strings do not match and increment warning count.
- Missing/null/object/array fields do not match and increment warning count for range filters.
- Invalid range filter targets fail with `invalid where filter`.
- Invalid range filter targets fail even when there are zero candidate rows.
- Missing `type` for range ops fails.
- `type` supplied with a non-range op fails.
- `values` supplied with a range op fails.
- `compare_warnings` is absent from header and metadata when zero.
- `compare_warnings` coexists with ripgrep partial-warning metadata without overwriting `warning`.
- Multiple range filters count each candidate row at most once.
- A row that fails an earlier filter before a later range comparison is not counted for that later comparison.
- The tool schema exposes `gt`, `gte`, `lt`, `lte`, and `type: "number" | "date"` in `where` filters.
- `_describe_where()` includes `(number)` or `(date)` only for range filters.
- Existing `eq`, `regex`, `contains`, `in`, and `exists` behavior remains unchanged.

Validation commands after implementation:

```bash
uv run pytest tests/test_file_tools.py -k jsonl_search
uv run pytest tests/test_file_tools.py
uv run ruff check thinharness/tools/jsonl.py tests/test_file_tools.py
uv run pyright
```

## Non-Goals

- No OR filters.
- No sorting.
- No automatic type inference.
- No Unix timestamp date mode.
- No dependency addition unless standard-library ISO parsing is insufficient.
- No broad docs rewrite beyond the tool instructions and any narrow README/docs mention already covering `jsonl_search`.

## Open Checks During Implementation

- Confirm pydantic accepts the field name `type` cleanly in `StrictArgs`; expected outcome is that a bare `type` field works. If it does not, use an internal alias while preserving the external JSON field name.
- Confirm schema generation includes `type` clearly enough for model tool calls. Add a schema assertion if needed.
