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
- RESUME-3: Resuming on the originating provider preserves native reasoning (Anthropic thinking signatures, OpenAI `encrypted_content`, OpenRouter `reasoning_details`); resuming on a different provider degrades each reasoning part to a leading `<thinking>`-tagged text block and drops the opaque blob. Native re-emit additionally requires the resuming run to be able to accept the block: OpenAI re-emits the native reasoning item only when the resuming model is reasoning-capable, and Anthropic uses the thinking gate in RESUME-3A; otherwise both use the text fallback. So a reasoning-model capture resumed on a non-reasoning model of the same provider degrades to text.
- RESUME-3A: Anthropic resume treats explicit `extra_body["thinking"]` as authoritative: `enabled` and `adaptive` accept signed thinking replay, while `disabled`, unknown, or malformed values suppress native replay. Without an explicit thinking key, `HarnessConfig.effort` implies adaptive thinking; otherwise Anthropic models outside the legacy off-by-default families (`claude-opus-4`, `claude-sonnet-4`, `claude-haiku-4`, and `claude-3`) are assumed to run thinking by default and keep signed thinking blocks on resume.
- RESUME-4: Built-in provider resume state uses `version` 3; version 1 and version 2 state and old provider-native `kind` values are rejected with a regenerate error.
- RESUME-5: On resume, the live system prompt from the resuming harness config is re-injected; captured system prompts are not stored or restored.
- RESUME-6: A session seeded via `OpenAIResponsesSession.start(prompt, constants, previous_response_id=...)` captures only new transcript entries, so externally seeded prior turns are not present when later resumed from `resume_state`. This is unrelated to reasoning fidelity and is not changed by RESUME-3/RESUME-7.
- RESUME-7: For reasoning-capable OpenAI Responses models the harness requests `include=["reasoning.encrypted_content"]` so reasoning survives resume; non-reasoning models are unaffected. Captured `resume_state` therefore contains encrypted reasoning blobs (OpenAI/OpenRouter) and signed thinking (Anthropic) and should be treated as sensitive, consistent with the local-trace sensitivity note.

## Plugin Composition

### Purpose

Callers compose optional harness behavior explicitly while independent custom tools and hooks stay direct constructor inputs.

### Requirements

- PLUGIN-1: `Harness` accepts plugins in caller order through `plugins=`; no plugin is loaded through entry points, directories, manifests, or implicit defaults.
- PLUGIN-2: Plugin names are non-empty and unique within one harness. A duplicate name fails before either plugin binds.
- PLUGIN-3: Plugin binding is synchronous and performs no file or network I/O. Static tools, instructions, and hooks are validated and visible immediately after harness construction.
- PLUGIN-4: `Harness.connect()` or the first run opens connected plugin bindings once in caller order. Concurrent connection calls share that attempt, and connection completes before `run_start` hooks fire.
- PLUGIN-5: Dynamic tools, instructions, and hooks are staged and receive the same complete validation as static contributions. ThinHarness commits the full dynamic set only after every binding opens successfully.
- PLUGIN-6: A connection failure, including cancellation, closes entered bindings in reverse order, installs no dynamic contribution, and leaves connection retryable. `run_start` and `run_end` do not fire for an attempt that fails during connection.
- PLUGIN-7: Closing a harness closes plugin bindings in reverse order before closing a model owned by the harness. Repeated close calls have no effect.
- PLUGIN-8: Contribution order is plugin static contributions, direct `tools=` and `hooks=`, then plugin dynamic contributions. System instructions are the configured system prompt, plugin instructions, the transitional skill summary, and per-tool instructions; structured-output instructions are added through the existing output path.
- PLUGIN-9: ThinHarness copies caller-supplied hook registries before adding plugin hooks. Plugin composition never mutates a caller-owned registry.
- PLUGIN-10: Plugins are trusted in-process code. ThinHarness does not isolate them or resolve dependencies between them.

## Filesystem Plugin

### Purpose

Callers opt into root-scoped workspace tools without making filesystem behavior part of the core harness.

### Requirements

- FILESYSTEM-PLUGIN-1: `Harness` has no implicit filesystem tools. `FilesystemPlugin` provides `read`, `write`, `edit`, `search`, `list`, and `glob` by default; callers select an ordered subset explicitly.
- FILESYSTEM-PLUGIN-2: `jsonl_search` is an opt-in tool of `FilesystemPlugin` and shares its root, read policy, search process, truncation, and spill-output handling.
- FILESYSTEM-PLUGIN-3: `HarnessConfig.root` is the one run root. `FilesystemPlugin` uses that root and cannot configure a different root.
- FILESYSTEM-PLUGIN-4: Harness construction and plugin binding do not create the root. A harness without `FilesystemPlugin` has a generic default prompt, adds no workspace-root instruction, and has no filesystem side effect.
- FILESYSTEM-PLUGIN-5: Filesystem limits, output location, search settings, and path policies belong to `FilesystemPlugin`. `HarnessConfig.read_paths` and `write_paths` remain temporarily as parallel-LLM policy and do not configure filesystem plugin tools.
- FILESYSTEM-PLUGIN-6: Independent custom tools continue to use `tools=[ToolSpec(...)]`; callers do not need to wrap one tool in a plugin.

## Run Toolset Freeze

### Purpose

The set of tools a model can call is fixed when a run starts, so every provider request in one run sees the same tool schemas.

### Requirements

- TOOLSET-FREEZE-1: The run's tool schemas, system instructions, request metadata, and structured-output request are captured once per run after harness connection and run-start hooks, and every provider request in that run uses that captured set.
- TOOLSET-FREEZE-2: A tool added with `add_tool` during an in-flight run does not appear in that run's later provider requests; it takes effect on the next run.
- TOOLSET-FREEZE-3: The executable tool map is frozen with the schemas: a model call naming a tool added mid-run resolves as an unknown tool for the current run, and approval-required detection uses the same frozen map.

## Run Token Accounting

### Purpose

Harness runs report provider token usage as run-level totals so hosts can meter cost without parsing raw provider responses.

### Requirements

- TOKEN-USAGE-1: `RunUsage` exposes `input_tokens` and `output_tokens` run totals accumulated per provider request and surfaced on `HarnessResult.usage`, including retry turns and approval-resume turns in the same logical run.
- TOKEN-USAGE-2: Provider responses with missing or partial usage contribute only the token counts they report; absent counts add nothing and do not error.
- TOKEN-USAGE-3: Approval envelopes written before token accounting existed still resume: missing token keys default to 0, while token keys that are present with a wrong type are rejected.

## Anthropic Prompt Caching

### Purpose

Anthropic Messages requests opt into provider prompt caching by default so multi-request runs reuse the growing prompt prefix instead of paying full input price on every request.

### Requirements

- ANTHROPIC-CACHE-1: Every Anthropic Messages request sends top-level `cache_control: {"type": "ephemeral"}`, letting the API place the cache breakpoint on the last cacheable block automatically; a `cache_control` key in model `extra_body` replaces the default.
- ANTHROPIC-CACHE-2: Cache reads surface through the existing normalized usage fields (`TokenUsage.cached_tokens`, `RunUsage.cached_tokens`); `input_tokens` remains the provider-reported value, which for Anthropic is the uncached remainder, and cache-write tokens are not separately accounted.

## Provider Request Settings

### Purpose

Provider-neutral request settings let callers tune output length and reasoning depth without hand-writing each provider's payload dialect.

### Requirements

- PROVIDER-SETTINGS-1: `HarnessConfig.max_tokens` and `ModelSettings.max_tokens` accept positive integers only. OpenAI Responses sends `max_output_tokens` only when configured, OpenRouter sends `max_tokens` only when configured, and Anthropic always sends `max_tokens`, defaulting to 16384 when neither the model constructor nor settings provide a value.
- PROVIDER-SETTINGS-2: A directly constructed `AnthropicMessagesModel(max_tokens=...)` overrides `ModelSettings.max_tokens`; a top-level `extra_body["max_tokens"]` overrides both because tuning keys are applied before `extra_body`.
- PROVIDER-SETTINGS-3: `HarnessConfig.effort` and `ModelSettings.effort` pass through as provider-neutral strings. OpenAI Responses and OpenRouter send `reasoning: {"effort": ...}`; Anthropic sends `output_config.effort` and injects adaptive thinking unless `extra_body` supplies its own top-level `thinking` key.
- PROVIDER-SETTINGS-4: The harness does not client-validate provider/model-specific `effort`, `temperature`, or thinking combinations. Invalid combinations surface as provider API errors.
- PROVIDER-SETTINGS-5: `HarnessConfig.request_timeout` applies independently to every built-in provider transport attempt, including retry attempts.

## Provider Request Retries

### Purpose

Built-in provider requests recover from transient HTTP failures without repeating model-session mutations or consuming another logical model request.

### Requirements

- PROVIDER-RETRY-1: Every built-in OpenAI, Anthropic, and OpenRouter HTTP request permits three retries by default after the first attempt. `request_retries` accepts 0 through 10, and `request_retry_backoff` accepts non-negative seconds; direct provider constructors enforce the same bounds as `HarnessConfig`.
- PROVIDER-RETRY-2: HTTP 408, 409, 425, 429, all 5xx responses, `httpx.TimeoutException`, `httpx.NetworkError`, and `httpx.RemoteProtocolError` are retryable. Other 4xx responses, other HTTP errors, authentication failures, invalid JSON, response validation failures, and arbitrary custom model failures are not retryable.
- PROVIDER-RETRY-3: Retry delay uses `request_retry_backoff * 2**retry_index` plus up to 25 percent positive jitter. A valid numeric or HTTP-date `Retry-After` can increase that delay, and every delay is capped at 60 seconds.
- PROVIDER-RETRY-4: Cancellation during a request or delay propagates immediately. Exhaustion raises the final attempt's `ProviderError`, preserving provider-error run classification.
- PROVIDER-RETRY-5: Transport attempts stay inside one logical model request. They do not increase model request limits, usage counts, stream event counts, trace span counts, parallel completion request counts, or provider session history.
- PROVIDER-RETRY-6: Named subagent override models and inferred parallel completion models inherit the parent request retry settings. The parallel LLM tool has no separate provider retry loop or attempt budget.
- PROVIDER-RETRY-7: Retries use at-least-once HTTP delivery. A transport failure after provider acceptance can cause duplicate provider work or charges because built-in providers do not share a portable idempotency-key contract.
- PROVIDER-RETRY-8: A custom `http_client` can apply its own retry policy below the provider retry loop. Callers set `request_retries=0` when the custom client owns retries to avoid multiplying attempt budgets.

## Structured Output Provider Modes

### Purpose

Structured output uses the strongest provider-native contract available while preserving explicit fallback modes for older models and provider-specific limits.

### Requirements

- STRUCTURED-PROVIDER-1: Anthropic supports native JSON Schema structured output and defaults `output_mode="auto"` to native, matching OpenAI. Tool and prompted modes remain available explicitly.
- STRUCTURED-PROVIDER-2: For Anthropic native structured output, provider requests send `output_config.format: {"type": "json_schema", "schema": ...}`. The schema is written after `extra_body` so a caller's `output_config.effort` or other fields survive, but the structured-output schema cannot be silently removed.
- STRUCTURED-PROVIDER-3: Anthropic model or schema floors are enforced by the API, not by client-side capability sniffing. Callers targeting older Anthropic models can request explicit tool mode.

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
- MODEL-OBSERVABILITY-7: Model spans pin `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`, `gen_ai.usage.cache_read.input_tokens`, `gen_ai.usage.total_tokens`, `gen_ai.response.model`, and `gen_ai.response.finish_reasons`. `cache_read.input_tokens` carries provider-reported cached input tokens when present and is omitted when unreported. `finish_reasons` is always a list wrapping the normalized reason. `total_tokens` passes through a raw provider `total_tokens` when present and is otherwise computed as input+output only when both are present; partial usage yields no total.
- MODEL-OBSERVABILITY-8: Custom `Model` implementations that do not populate normalized `ModelTurn` usage fields keep their `gen_ai.usage.*` span attributes via best-effort extraction from the raw response.

## MCP Client Layer

### Purpose

ThinHarness exposes tools from MCP servers through wrapper objects whose transport, session, and connection sharing come from the FastMCP client, while ThinHarness keeps tool selection, conversion, error envelopes, and trace attribution.

### Requirements

- MCP-1: `MCPServer` accepts exactly one FastMCP `ClientTransport`, including `FastMCPTransport` for an MCP server object in the same Python process; URL strings, script paths, server objects, and configuration dictionaries are rejected with `TypeError`. The stdio, SSE, and Streamable HTTP compatibility wrappers take command- or URL-based constructors and derive their ids from them; the generic class derives its default id from the transport class name, and duplicate ids get `-2`, `-3` suffixes.
- MCP-2: Connections open lazily on `Harness.connect()` or the first run; one discovered tool snapshot is reused for all runs of one harness; `Harness.aclose()` closes harness-entered servers, and a partial multi-server connection failure closes servers that were already opened.
- MCP-3: A wrapper owns the FastMCP client built on its transport: nested and concurrent entries share one connection, the final exit closes the transport (terminating a stdio child process), and the same wrapper can reconnect afterwards. One stateful transport object must not be reused across wrappers; reusing the same wrapper shares one session.
- MCP-4: Final close is bounded — the bound comes from FastMCP's `client_disconnect_timeout` setting (default 5 seconds) — and a caller cancellation consumed by transport cleanup is re-raised after cleanup completes. Cancelling a first connection or a final close propagates the cancellation and leaves the wrapper reusable.
- MCP-5: `include_tools` and `exclude_tools` match original MCP tool names before prefixing and normalization; `tool_prefix`, schema cleanup, sanitized-name collision errors, and cross-harness tool collision errors are ThinHarness behavior. Discovered MCP tools are ordinary `ToolSpec` objects with `kind="mcp"` and `McpToolInfo` attribution, and tracing reads that attribution from the `ToolSpec`, so an after-tool hook cannot erase it.
- MCP-6: Successful `structuredContent` is returned as a JSON string; text, image, audio, embedded-resource, and resource-link blocks convert in order to model-visible text. A protocol-level tool failure (`isError`) returns a failed `ToolResult` with `error_type="MCPToolError"` and `retry=True`; known transport and protocol failures during a tool call return `error_type="MCPError"`, including when wrapped in an exception group or explicit cause chain — a group whose members are all `Exception`s is normalized when any member's cause chain holds a known failure, even alongside sibling exception noise from teardown. An exception group carrying cancellation or any other non-`Exception` failure propagates, and exceptions with no known failure in their group or cause chain propagate as programming errors.
- MCP-7: MCP tools and connection details never enter resume state, and subagents see MCP servers only through explicit `inherit_mcp_servers` and `mcp_servers` settings.
- MCP-8: The base install works without MCP packages: importing ThinHarness and constructing any wrapper needs no extra, and opening a connection without `mcp` or `fastmcp` raises `MCPDependencyError` with the `thinharness[mcp]` install hint.
- MCP-9: `timeout` bounds MCP initialization and HTTP connection establishment; `read_timeout` bounds MCP requests, HTTP reads, and SSE reads.
