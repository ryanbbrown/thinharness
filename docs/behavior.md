# Behavior

This file records durable product behavior so plan reviews can check the intended behavior contract before implementation.

<!--
## Feature Name

### Purpose

One short paragraph describing the behavior from the user or system perspective.

### Requirements

Use an uppercase, readable requirement prefix from the section name, such as `TOOL-APPROVAL-1`.

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
- RESUME-3: Resuming on the originating provider preserves native reasoning (Anthropic thinking signatures, OpenAI `encrypted_content`, OpenRouter `reasoning_details`); resuming on a different provider degrades each reasoning part to a leading `<thinking>`-tagged text block and drops the opaque blob. Native re-emit additionally requires the resuming run to be able to accept the block: OpenAI re-emits the native reasoning item only when the resuming model is reasoning-capable, and Anthropic only when extended thinking is enabled; otherwise both use the text fallback. So a reasoning-model capture resumed on a non-reasoning model of the same provider degrades to text.
- RESUME-4: Built-in provider resume state uses `version` 3; version 1 and version 2 state and old provider-native `kind` values are rejected with a regenerate error.
- RESUME-5: On resume, the live system prompt from the resuming harness config is re-injected; captured system prompts are not stored or restored.
- RESUME-6: A session seeded via `OpenAIResponsesSession.start(prompt, constants, previous_response_id=...)` captures only new transcript entries, so externally seeded prior turns are not present when later resumed from `resume_state`. This is unrelated to reasoning fidelity and is not changed by RESUME-3/RESUME-7.
- RESUME-7: For reasoning-capable OpenAI Responses models the harness requests `include=["reasoning.encrypted_content"]` so reasoning survives resume; non-reasoning models are unaffected. Captured `resume_state` therefore contains encrypted reasoning blobs (OpenAI/OpenRouter) and signed thinking (Anthropic) and should be treated as sensitive, consistent with the local-trace sensitivity note.

## Run Toolset Freeze

### Purpose

The set of tools a model can call is fixed when a run starts, so every provider request in one run sees the same tool schemas.

### Requirements

- TOOLSET-FREEZE-1: The run's tool schemas, system instructions, request metadata, and structured-output request are captured once per run after run-start hooks and MCP connection, and every provider request in that run uses that captured set.
- TOOLSET-FREEZE-2: A tool added with `add_tool` during an in-flight run does not appear in that run's later provider requests; it takes effect on the next run.

## Run Token Accounting

### Purpose

Harness runs report provider token usage as run-level totals so hosts can meter cost without parsing raw provider responses.

### Requirements

- TOKEN-USAGE-1: `RunUsage` exposes `input_tokens` and `output_tokens` run totals accumulated per provider request and surfaced on `HarnessResult.usage`, including retry turns and approval-resume turns in the same logical run.
- TOKEN-USAGE-2: Provider responses with missing or partial usage contribute only the token counts they report; absent counts add nothing and do not error.
- TOKEN-USAGE-3: Approval envelopes written before token accounting existed still resume: missing token keys default to 0, while token keys that are present with a wrong type are rejected.

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
- MODEL-OBSERVABILITY-7: Model spans pin `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`, `gen_ai.usage.total_tokens`, `gen_ai.response.model`, and `gen_ai.response.finish_reasons`. `finish_reasons` is always a list wrapping the normalized reason. `total_tokens` passes through a raw provider `total_tokens` when present and is otherwise computed as input+output only when both are present; partial usage yields no total.
- MODEL-OBSERVABILITY-8: Custom `Model` implementations that do not populate normalized `ModelTurn` usage fields keep their `gen_ai.usage.*` span attributes via best-effort extraction from the raw response.
