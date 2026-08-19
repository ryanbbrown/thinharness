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
- PLUGIN-3: `PluginContext` contains the canonical harness root, configured model, and a narrow `ChildHarnessHost`; it does not expose the parent harness. Plugin binding is synchronous and performs no file, provider, or network I/O. Static tools, instructions, hooks, and agent names are validated and visible immediately after harness construction.
- PLUGIN-4: `Harness.connect()` or the first run opens connected plugin bindings once in caller order. Concurrent connection calls share that attempt, and connection completes before `run_start` hooks fire.
- PLUGIN-5: Dynamic tools, instructions, and hooks are staged and receive the same complete validation as static contributions. ThinHarness commits the full dynamic set only after every binding opens successfully.
- PLUGIN-6: A connection failure, including cancellation, closes entered bindings in reverse order, installs no dynamic contribution, and leaves connection retryable. `run_start` and `run_end` do not fire for an attempt that fails during connection.
- PLUGIN-7: Closing a harness closes plugin bindings in reverse order before closing a model owned by the harness. Repeated close calls have no effect.
- PLUGIN-8: Contribution order is plugin static contributions, direct `tools=` and `hooks=`, then plugin dynamic contributions. System instructions are the configured system prompt, plugin instructions in caller plugin order, and all per-tool instructions; structured-output instructions are added through the existing output path. A skill summary is an ordinary plugin instruction at the `SkillsPlugin` position. `SubagentsPlugin` contributes its ordinary `subagent` tool at its plugin position. In a child, automatically inherited plugins keep parent plugin order, explicit child plugins follow them, inherited direct tools follow inherited plugins, and explicit child tools are last.
- PLUGIN-9: ThinHarness copies caller-supplied hook registries before adding plugin hooks. Plugin composition never mutates a caller-owned registry.
- PLUGIN-10: Plugins are trusted in-process code. ThinHarness does not isolate them or resolve dependencies between them.
- PLUGIN-11: Each static plugin binding supplies an immutable agent-name tuple. Core combines these names before validating agent-filtered hooks and rejects blank or duplicate names. Connected contributions cannot change the agent catalog.
- PLUGIN-12: Core records authoritative direct, plugin, and delegation composition roles independently of caller-visible `ToolOrigin`; these records control inheritance and delegation tracing and cannot be forged through tool metadata.
- PLUGIN-13: Automatic child inheritance is explicit and structural. Only a plugin with synchronous `for_child()` is rebound against the child context; its returned plugin must be valid and keep the expected fixed name.

## Subagents Plugin

### Purpose

Callers add delegation explicitly through `SubagentsPlugin`, which owns the model-facing tool, child recipes, inheritance policy, child hooks, and delegation result shaping.

### Requirements

- SUBAGENTS-PLUGIN-1: A plain harness has no delegation tool. `SubagentsPlugin` has fixed name `"subagents"`, contributes one ordinary tool named `subagent`, and provides an unnamed `"default"` child plus ordered optional named children. Normal tool and plugin collision rules apply; no tool name is reserved.
- SUBAGENTS-PLUGIN-2: The unnamed child uses parent-derived model, prompt, limits, output, and additive inheritance defaults. Named configurations can override the model, prompt, limits, output, hooks, plugins, and tools; named children with no tools are valid.
- SUBAGENTS-PLUGIN-3: `inherit_parent=True` rebinds only plugins that explicitly implement `for_child()` and inherits eligible direct tools from the active run snapshot. Inherited sources keep parent order, explicit child plugins and tools follow them, and duplicates fail rather than replace inherited values. Approval-required tools never enter a child.
- SUBAGENTS-PLUGIN-4: Child hooks belong to the selected child configuration. Default-child hooks come from `default_hooks`; named hooks come from `SubAgentConfig.hooks`. Parent `before_subagent_run` and `after_subagent_run` hooks remain on the parent and can filter against the plugin's static agent catalog.
- SUBAGENTS-PLUGIN-5: Every top-level plugin context receives a narrow child host that accepts delegation only during an active parent tool call. Every child context receives a disabled host, and `SubagentsPlugin` is invalid in explicit child plugins, so children cannot create grandchildren through built-in or custom plugin paths.
- SUBAGENTS-PLUGIN-6: Each child is a fresh, independently budgeted run with no parent transcript. A parent-model child borrows the model; an override creates and closes its own model while projecting parent request settings and only same-provider credentials.
- SUBAGENTS-PLUGIN-7: Child events remain nested unless subagent streaming is enabled, then preserve parent run and tool-call correlation. Real delegation is traced from authoritative composition with `subagent.delegation=true` from tool-span start; a same-named direct tool or forged `ToolOrigin` remains an ordinary tool event.
- SUBAGENTS-PLUGIN-8: Successful results preserve agent, inheritance mode, effective tools, child request usage, and structured-output metadata. Unknown agents, provider failures, hook failures, strict sibling cancellation, and parent cancellation preserve existing error and cleanup behavior; every created child closes without hiding the original error.
- SUBAGENTS-PLUGIN-9: Child configuration, child hosts, plugin state, and model objects never enter resume or approval state. Parent usage counts one delegation tool call while child requests and tokens remain child usage.
- SUBAGENTS-PLUGIN-10: Child inheritance has three result modes: `"inherited"` for default or inherited-only children, `"inherited+explicit"` for additive inherited and explicit sources, and `"explicit"` when parent inheritance is disabled.

## Filesystem Plugin

### Purpose

Callers opt into root-scoped workspace tools without making filesystem behavior part of the core harness.

### Requirements

- FILESYSTEM-PLUGIN-1: `Harness` has no implicit filesystem tools. `FilesystemPlugin` provides `read`, `write`, `edit`, `search`, `list`, and `glob` by default; callers select an ordered subset explicitly.
- FILESYSTEM-PLUGIN-2: `jsonl_search` is an opt-in tool of `FilesystemPlugin` and shares its root, read policy, search process, truncation, and spill-output handling.
- FILESYSTEM-PLUGIN-3: `HarnessConfig.root` is the one run root. `FilesystemPlugin` uses that root and cannot configure a different root.
- FILESYSTEM-PLUGIN-4: Harness construction and plugin binding do not create the workspace root. A harness without `FilesystemPlugin` has a generic default prompt, adds no workspace-root instruction, and has no workspace filesystem side effect. Observability sinks keep their independent configured storage behavior.
- FILESYSTEM-PLUGIN-5: Filesystem limits, output location, search settings, and path policies belong to `FilesystemPlugin`.
- FILESYSTEM-PLUGIN-6: Independent custom tools continue to use `tools=[ToolSpec(...)]`; callers do not need to wrap one tool in a plugin.
- FILESYSTEM-PLUGIN-7: `FilesystemPlugin` has the runtime-fixed name `"filesystem"`. Its constructor configuration is frozen: mutation of constructor inputs, returned property values, or plugin attributes cannot change later parent or child bindings.

## Skills Plugin

### Purpose

Callers explicitly compose a fixed skill catalog and select which skill operations a harness can use.

### Requirements

- SKILLS-PLUGIN-1: A harness accepts at most one runtime-fixed `SkillsPlugin` named `"skills"`. The plugin requires one or more ordered skill directories and an explicit non-empty ordered selection of `skill_read`, `skill_run`, or both.
- SKILLS-PLUGIN-2: The plugin discovers and validates its catalog during construction. Its selected tools and summary are static and visible immediately after harness construction, and binding performs no I/O.
- SKILLS-PLUGIN-3: Relative skill directories resolve from the process working directory, not from `HarnessConfig.root`. Reusing one plugin object across harnesses reuses the same registry and catalog.
- SKILLS-PLUGIN-4: Catalog names, paths, metadata, selection, and summary are frozen at plugin construction. Existing skill content, file trees, and scripts remain live and are read or executed when a tool is invoked. A new plugin is required to discover added or removed skills.
- SKILLS-PLUGIN-5: A non-empty catalog contributes selected tools in caller order and one compact summary. The summary mentions `skill_read` only when that tool is selected. A catalog with no skills contributes no tools or summary.
- SKILLS-PLUGIN-6: `skill_read` preserves live content, tree, containment, and truncation behavior and is parallel-safe. `skill_run` preserves runner, working-directory, merged-output, timeout, metadata, and containment behavior and runs sequentially.
- SKILLS-PLUGIN-7: `SkillsPlugin` explicitly opts into safe child inheritance and rebinds through the generic plugin contract. Inherited bindings share the exact constructor-time registry, frozen catalog, tool order, and one summary without another discovery pass. Its constructor configuration is frozen so later input, property-value, or attribute mutation cannot change a binding.

## Parallel LLM Plugin

### Purpose

Callers explicitly compose a stateless parallel batch tool and choose whether it borrows a model or owns per-call provider construction.

### Requirements

- PARALLEL-LLM-PLUGIN-1: A harness accepts at most one runtime-fixed `ParallelLlmPlugin` named `"parallel_llm"`. It contributes one text-only `parallel_llm` tool with no model-visible model override.
- PARALLEL-LLM-PLUGIN-2: With no model argument, the plugin borrows the model from the context where it is bound, so an inherited binding uses the child model. With a model object, it borrows that caller-owned object. The plugin does not close either borrowed model. Constructor configuration is frozen for parent and inherited bindings.
- PARALLEL-LLM-PLUGIN-3: A string model uses plugin-owned provider and request settings, creates a provider for each batch invocation, and closes it after success, schema-resolution failure, request failure, or cancellation. Provider and request settings are rejected for borrowed models.
- PARALLEL-LLM-PLUGIN-4: The plugin uses the canonical harness root. Read and write policies are root-scoped, outputs are atomic JSON files, prompt count and concurrency are bounded, and ordered sparse results report batch-local request, total, success, and failure counts.
- PARALLEL-LLM-PLUGIN-5: Batch prompts use independent fresh sessions and receive no parent system prompt, tools, memory, or continuation. Batch requests and tokens are outside parent `RunUsage` and `max_model_requests`; the parent counts one batch invocation toward `max_tool_calls`.
- PARALLEL-LLM-PLUGIN-6: Provider transport retries remain inside one logical batch request. Cancellation propagates and closes plugin-owned string-model providers.
- PARALLEL-LLM-PLUGIN-7: Callers that need a renamed tool or structured batch output use `ParallelLlmTool(...).spec()` directly. Direct specifications use the same ordinary `ToolSpec` contract as all other tools.

## Run Toolset Freeze

### Purpose

The set of tools a model can call is fixed when a run starts, so every provider request in one run sees the same tool schemas.

### Requirements

- TOOLSET-FREEZE-1: The run's tool schemas, executable map, authoritative direct/plugin/delegation composition roles, system instructions, request metadata, and structured-output request are captured once per run after harness connection and run-start hooks. Every provider request and delegated child in that run uses that snapshot.
- TOOLSET-FREEZE-2: A direct tool added with `add_tool` during an in-flight run does not appear in that run's later provider requests or delegated children; it takes effect on the next run.
- TOOLSET-FREEZE-3: A model call naming a tool added mid-run resolves as an unknown tool for the current run. Approval detection, direct-tool inheritance, and delegation tracing use the same frozen authoritative map rather than live tools or caller-supplied origins.

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
- PROVIDER-SETTINGS-6: Anthropic request metadata includes only a string `user_id`, the sole metadata field accepted by the Messages API. Other run metadata remains available to harness hooks, tracing, and child correlation but is not sent in the Anthropic payload.

## Provider Request Retries

### Purpose

Built-in provider requests recover from transient HTTP failures without repeating model-session mutations or consuming another logical model request.

### Requirements

- PROVIDER-RETRY-1: Every built-in OpenAI, Anthropic, and OpenRouter HTTP request permits three retries by default after the first attempt. `request_retries` accepts 0 through 10, and `request_retry_backoff` accepts non-negative seconds; direct provider constructors enforce the same bounds as `HarnessConfig`.
- PROVIDER-RETRY-2: HTTP 408, 409, 425, 429, all 5xx responses, `httpx.TimeoutException`, `httpx.NetworkError`, and `httpx.RemoteProtocolError` are retryable. Other 4xx responses, other HTTP errors, authentication failures, invalid JSON, response validation failures, and arbitrary custom model failures are not retryable.
- PROVIDER-RETRY-3: Retry delay uses `request_retry_backoff * 2**retry_index` plus up to 25 percent positive jitter. A valid numeric or HTTP-date `Retry-After` can increase that delay, and every delay is capped at 60 seconds.
- PROVIDER-RETRY-4: Cancellation during a request or delay propagates immediately. Exhaustion raises the final attempt's `ProviderError`, preserving provider-error run classification.
- PROVIDER-RETRY-5: Transport attempts stay inside one logical model request. They do not increase model request limits, usage counts, stream event counts, trace span counts, parallel completion request counts, or provider session history.
- PROVIDER-RETRY-6: Named child override models inherit the parent request retry settings. A `ParallelLlmPlugin` with a string model uses its own request retry settings; one rebound with no model uses the child model settings; one configured with a model object uses that model's settings. The parallel LLM tool has no separate provider retry loop or attempt budget.
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

ThinHarness exposes tools from MCP servers through explicit `MCPPlugin` composition. Server wrappers retain FastMCP transport, session, and connection sharing, while the plugin owns harness binding, discovery, lifecycle, and tool attribution.

### Requirements

- MCP-1: `MCPServer` accepts exactly one FastMCP `ClientTransport`, including `FastMCPTransport` for an MCP server object in the same Python process; URL strings, script paths, server objects, and configuration dictionaries are rejected with `TypeError`. The stdio, SSE, and Streamable HTTP compatibility wrappers take command- or URL-based constructors and derive their public base ids from them; the generic class derives its base id from the transport class name. Each `MCPPlugin` binding resolves duplicate ids locally with deterministic `-2`, `-3` suffixes without mutating shared wrappers.
- MCP-2: MCP is enabled only by adding one fixed-name `MCPPlugin` to `Harness(plugins=...)`. Generic plugin connection opens servers lazily on `Harness.connect()` or the first run, discovers one tool snapshot per binding, and reuses that snapshot for all runs. The complete discovered contribution is validated and installed atomically. A connection, discovery, validation, or cancellation failure closes entered servers in reverse order, installs no tools, and permits retry; `Harness.aclose()` closes the plugin once.
- MCP-3: A wrapper owns the FastMCP client built on its transport: nested and concurrent entries share one connection, the final exit closes the transport (terminating a stdio child process), and the same wrapper can reconnect afterwards. One stateful transport object must not be reused across wrappers; reusing the same wrapper across parent, child, or independent harness bindings shares one reference-counted session.
- MCP-4: Final close is bounded — the bound comes from FastMCP's `client_disconnect_timeout` setting (default 5 seconds) — and a caller cancellation consumed by transport cleanup is re-raised after cleanup completes. Cancelling a first connection or a final close propagates the cancellation and leaves the wrapper reusable.
- MCP-5: `include_tools` and `exclude_tools` match original MCP tool names before prefixing and normalization; `tool_prefix`, schema cleanup, sanitized-name collision errors, and cross-contribution tool collision errors are ThinHarness behavior. Discovered MCP tools are ordinary `ToolSpec` objects with `ToolOrigin(plugin="mcp", source=resolved_server_id, attributes={"tool_name": original_tool_name})`. Tracing reads this origin from the `ToolSpec`, so an after-tool hook cannot erase attribution. Model-visible result metadata uses the same binding-local server id.
- MCP-6: Successful `structuredContent` is returned as a JSON string; text, image, audio, embedded-resource, and resource-link blocks convert in order to model-visible text. A protocol-level tool failure (`isError`) returns a failed `ToolResult` with `error_type="MCPToolError"` and `retry=True`; known transport and protocol failures during a tool call return `error_type="MCPError"`, including when wrapped in an exception group or explicit cause chain — a group whose members are all `Exception`s is normalized when any member's cause chain holds a known failure, even alongside sibling exception noise from teardown. An exception group carrying cancellation or any other non-`Exception` failure propagates, and exceptions with no known failure in their group or cause chain propagate as programming errors.
- MCP-7: MCP tools and connection details never enter resume state. `MCPPlugin` does not inherit automatically into children. A child that needs MCP lists an explicit `MCPPlugin` in its plugin configuration; that child binding owns its connection lifecycle, while reuse of the same server wrapper keeps the wrapper's reference-counted session behavior.
- MCP-8: The base install works without MCP packages: importing ThinHarness and constructing any wrapper or `MCPPlugin` needs no extra, and opening a connection without `mcp` or `fastmcp` raises `MCPDependencyError` with the `thinharness[mcp]` install hint.
- MCP-9: `timeout` bounds MCP initialization and HTTP connection establishment; `read_timeout` bounds MCP requests, HTTP reads, and SSE reads.
