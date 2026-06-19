# Behavior

This file records durable product behavior so plan reviews can check the intended behavior contract before implementation.

<!--
## Feature Name

### Purpose

One short paragraph describing the behavior from the user or system perspective.

### Requirements

Use an uppercase, readable requirement prefix from the section name, such as `BACKGROUND-1` or `TOOL-APPROVAL-1`.

- FEATURE-1: A concrete externally meaningful behavior.
- FEATURE-2: A behavior constraint, including any important exclusion or boundary.

### Scenarios

Use this section only when ordering, lifecycle, concurrency, retries, streaming, cancellation, or multi-actor behavior matters.
-->

## JSONL Field Search Snippets

### Purpose

`jsonl_search` can extract matching internal lines from large multiline string fields after a JSONL row has been selected by the existing row query and `where` filters.

### Requirements

- JSONL-FIELD-SEARCH-1: The top-level `query` remains a row prefilter over JSONL lines; `field_searches` runs only after JSON parsing and `where` filtering.
- JSONL-FIELD-SEARCH-2: Each field search resolves the same jq-style field paths used by `fields` and `where`, requires a non-empty query, and searches only string field values.
- JSONL-FIELD-SEARCH-3: Field searches support substring or regex matching, case-insensitive matching by default, optional case-sensitive matching, context lines, per-row match limits, and per-line truncation.
- JSONL-FIELD-SEARCH-4: Output preserves normal `fields` projection and renders matching field snippets beneath each selected row, without changing the existing global tool truncation and spill behavior.

## JSONL Search Range Filters

### Purpose

`jsonl_search` can filter rows by numeric and date-like scalar fields using explicit range operators in the existing `where` filter shape.

### Requirements

- JSONL-RANGE-1: Range filters use `gt`, `gte`, `lt`, and `lte` operators and require an explicit `type` of `number` or `date`.
- JSONL-RANGE-2: Number range filters compare only JSON number values, excluding booleans and non-finite numbers; numeric strings do not match.
- JSONL-RANGE-3: Date range filters compare ISO-like date and datetime strings, compare date-only values by calendar date, and treat aware/naive datetime mismatches as non-comparable.
- JSONL-RANGE-4: Invalid range filter definitions fail before scanning rows with `invalid where filter`.
- JSONL-RANGE-5: Non-comparable row values do not match and increment `compare_warnings` once per candidate row where a range comparison was attempted.
- JSONL-RANGE-6: Comparison warnings appear in result metadata under `compare_warnings` without replacing ripgrep partial-result warning metadata.
