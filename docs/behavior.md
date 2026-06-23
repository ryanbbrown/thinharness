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

## JSONL Typed Equality Filters

### Purpose

`jsonl_search` can apply typed numeric and date-like equality filters so scalar comparisons do not fall back to JSON string rendering.

### Requirements

- JSONL-TYPED-EQUALITY-1: Typed equality filters use `eq` and `ne` operators with an explicit `type` of `number` or `date`.
- JSONL-TYPED-EQUALITY-2: Number equality filters compare only JSON number values, excluding booleans and non-finite numbers; numeric strings do not match.
- JSONL-TYPED-EQUALITY-3: Date equality filters compare ISO-like date and datetime strings, compare date-only values by calendar date, and treat aware/naive datetime mismatches as non-comparable.
- JSONL-TYPED-EQUALITY-4: Non-comparable row values do not match either `eq` or `ne` and increment `compare_warnings` once per candidate row where a typed equality comparison was attempted.
- JSONL-TYPED-EQUALITY-5: Invalid typed equality filter definitions fail before scanning rows with `invalid where filter`.

## Resume State

### Purpose

Built-in provider resume state is a self-contained, provider-agnostic transcript that can be replayed by any built-in provider or model while preserving the run lifecycle rules for when resume state is available.

### Requirements

- RESUME-1: `resume_state` is a provider-agnostic transcript; resume across built-in providers and across built-in models is supported.
- RESUME-2: `resume_state` is self-contained and does not depend on provider continuation tokens such as OpenAI `previous_response_id`; an OpenAI run that never received a response id is still resumable.
- RESUME-3: Resume state version 2 preserves reasoning as visible transcript text only and does not preserve provider-specific reasoning chains.
- RESUME-4: Built-in provider resume state uses `version` 2; version 1 state and old provider-native `kind` values are rejected with a regenerate error.
- RESUME-5: On resume, the live system prompt from the resuming harness config is re-injected; captured system prompts are not stored or restored.
- RESUME-6: A session seeded via `OpenAIResponsesSession.start(previous_response_id=...)` captures only new transcript entries, so externally seeded prior turns are not present when later resumed from `resume_state`.

## Model Observability Projections

### Purpose

Tracing and streaming expose projections of the same neutral per-request model-visible input and assistant output without changing provider-native in-run request construction.

### Requirements

- MODEL-OBSERVABILITY-1: Model trace input is built from the new model-visible entries for that provider request, including rendered `<harness_notice>` text whenever notices are sent to the model.
- MODEL-OBSERVABILITY-2: Structured model notice metadata remains available on model spans separately from rendered input messages.
- MODEL-OBSERVABILITY-3: Replayed resume transcript entries are not counted as new model request input for the first resumed provider request; only the new resume prompt or continuation delta is traced as request input.
- MODEL-OBSERVABILITY-4: `ModelMessageEvent.text` always includes assistant text from the completed provider turn; stream text suppression is not part of `StreamOptions`.
- MODEL-OBSERVABILITY-5: Stream lifecycle events remain operational events and are not stored in durable provider transcript entries.
- MODEL-OBSERVABILITY-6: Core tracing emits OTel/GenAI-oriented attributes and does not include sink-specific display namespaces.
