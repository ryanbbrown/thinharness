# Architecture

ThinHarness is a small, provider-agnostic agent loop for SDK use. The core idea is intentionally narrow: normalize provider turns, expose a small set of explicit tools, run model-requested tools, feed tool outputs back to the provider, and stop when the model produces a final answer or a configured limit is reached.

The repository contains large vendored reference trees under `vendor/`. Those are useful for comparison, LOC measurement, and borrowed ideas such as the pgr-style search ranking, but the ThinHarness runtime is the `thinharness/` package.

## Repository Tree

This tree expands project-owned files and collapses generated, cache, and vendored content.

```text
.
|-- .context/
|   `-- *.md                         # Planning, feedback, and historical implementation artifacts.
|-- .github/
|   `-- workflows/
|       `-- ci.yml                   # GitHub Actions lint and test workflow.
|-- assets/
|   |-- ThinHarness.svg              # README logo.
|   `-- agno-a.svg                   # Agno icon used in README comparison table.
|-- docs/
|   |-- architecture.md              # This architecture guide.
|   |-- decisions.md                 # Design decisions and deferred choices.
|   |-- docs.md                      # User-facing resume documentation.
|   |-- table.md                     # README comparison-table methodology and cell rationale.
|   `-- THIRD_PARTY_NOTICES.md       # Third-party attribution notices.
|-- e2e/
|   |-- control_plane_journey.py     # Live-provider control-plane, hooks, and retry journey.
|   |-- mcp_journey.py               # Live-provider MCP journey.
|   |-- parallel_llm_agent_journey.py # Live-provider built-in and custom parallel LLM journey.
|   |-- parallel_llm_tool_journey.py # Live-provider direct parallel LLM tool journey.
|   |-- skills_journey.py            # Live-provider skills journey.
|   |-- structured_output_journey.py # Live-provider structured-output journey.
|   `-- workspace_tools_journey.py   # Live-provider filesystem tool journey.
|-- AGENTS.md                        # Repository-specific agent validation instructions.
|-- tests/
|   |-- fakes.py                     # Scripted models, fake providers, and fake tracers.
|   |-- test_file_tools.py           # Filesystem, search, and JSONL tool tests.
|   |-- test_harness.py              # Core run loop, custom tools, lifecycle, and resource tests.
|   |-- test_hooks.py                # Hook dispatch, cancellation, strict hooks, and limit hooks.
|   |-- test_mcp.py                  # MCP lifecycle, discovery, collisions, subagents, and tracing tests.
|   |-- test_mcp_optional_dependency.py # Optional MCP dependency behavior.
|   |-- test_parallel_llm.py         # Built-in parallel one-shot LLM tool tests.
|   |-- test_parallel_tools.py       # Same-turn parallel and sequential tool execution tests.
|   |-- test_providers.py            # Provider payload conversion and error wrapping tests.
|   |-- test_resume.py               # Resume state contracts across providers and failure modes.
|   |-- test_skills.py               # Skill discovery, reading, running, selection, and duplicates.
|   |-- test_structured_output.py    # Native, prompted, tool, text, and subagent structured output tests.
|   |-- test_subagents.py            # Child harness construction, inheritance, hooks, and model overrides.
|   |-- test_tool_retry.py           # Retry envelopes, budgets, validation retries, and tracing semantics.
|   `-- test_tracing.py              # OpenTelemetry span shape and parent-child tracing tests.
|-- thinharness/
|   |-- __init__.py                  # Public API exports.
|   |-- core.py                      # HarnessConfig, HarnessResult, Harness, and the agent run loop.
|   |-- defaults.py                  # Default system prompt.
|   |-- hooks.py                     # Hook registration, event context dataclasses, and dispatch.
|   |-- output.py                    # Structured-output schema building, validation, and serialization.
|   |-- providers.py                 # Provider transports, model sessions, and provider-format adapters.
|   |-- py.typed                     # PEP 561 marker for typed package consumers.
|   |-- subagents.py                 # SubAgentConfig and the built-in subagent tool.
|   |-- tracing.py                   # OpenTelemetry-compatible tracing helpers.
|   `-- tools/
|       |-- __init__.py              # Tool package exports.
|       |-- base.py                  # ToolSpec, ToolResult, argument validation, and path policy.
|       |-- filesystem.py            # read, write, edit, search, list, glob, and jsonl_search wiring.
|       |-- jsonl.py                 # JSONL-specific search and projection engine.
|       |-- mcp.py                   # Optional MCP server connection and tool conversion support.
|       |-- parallel_llm.py          # Opt-in parallel one-shot model completion tool.
|       `-- skills.py                # Explicit skill discovery plus skill_read and skill_run.
|-- vendor/
|   |-- pgr/                         # Search-ranking reference implementation.
|   |-- claude-agent-sdk/            # Comparison/reference submodule.
|   |-- deepagents/                  # Comparison/reference submodule.
|   |-- openai-agents/               # Comparison/reference submodule.
|   |-- pydantic-ai/                 # Comparison/reference submodule.
|   |-- smolagents/                  # Comparison/reference submodule.
|   |-- adk-python/                  # Comparison/reference submodule.
|   |-- agno/                        # Comparison/reference submodule.
|   |-- strands/                     # Comparison/reference submodule.
|   |-- agent-framework/             # Comparison/reference submodule.
|   `-- other small harness references
|-- .gitmodules                      # Vendor submodule definitions.
|-- .gitignore                       # Ignore rules for local generated artifacts.
|-- LICENSE                          # MIT license.
|-- pyproject.toml                   # Package metadata, dependencies, pytest, coverage, ruff, and pyright config.
|-- README.md                        # Project motivation, comparison table, and basic usage.
`-- uv.lock                          # Locked dependency graph.
```

Generated or local-only directories such as `.venv/`, `.pytest_cache/`, `.ruff_cache/`, `__pycache__/`, and `thinharness.egg-info/` are not architectural. They should not be read as part of the runtime design.

## Runtime Shape

The runtime has five main layers:

1. `Harness` in `core.py` owns one run loop.
2. `ModelSession` implementations in `providers.py` turn provider APIs into normalized `ModelTurn` objects.
3. `ToolSpec` handlers in `tools/` define everything the model can call.
4. `HookRegistry` in `hooks.py` lets callers observe or mutate selected lifecycle events.
5. `RunTracer` in `tracing.py` wraps agent, model, and tool work in optional OpenTelemetry spans.

The key boundary is `ModelTurn`: providers can have very different APIs, but the harness only needs `text`, `tool_calls`, and `raw` provider JSON. Once a provider returns a `ModelTurn`, `core.py` can apply the same tool execution, structured output, hook, retry, tracing, and limit logic for OpenAI, Anthropic, OpenRouter, or a custom model object that implements the protocol.

## Core Flow

A normal `Harness.run(prompt)` follows this path:

1. Reject closed or re-entrant use.
2. Create or resume a provider `ModelSession`.
3. Initialize per-run state: raw responses, tool call records, `RunUsage`, limit-warning dedupe state, tracing, and run metadata.
4. Fire `run_start`.
5. Connect MCP servers if configured.
6. Fire `user_prompt_submit`, allowing hooks to cancel or append prompt context.
7. Build system instructions from the configured prompt, workspace root, and skill summary.
8. Resolve structured-output request metadata if configured.
9. Send the initial prompt to the model session.
10. Repeat until terminal:
    - Store the raw provider response.
    - If structured output is configured, check whether the turn finalizes output.
    - If the model returned no tool calls, return final text or validated structured output.
    - Enforce `max_tool_calls`.
    - Execute the tool-call batch, either serially or concurrently.
    - Track retryable tool failures and enforce retry budgets.
    - Continue the provider session with normalized tool outputs.
11. Attach `resume_state` only for clean resumable exits.
12. Fire `run_end` exactly once.
13. Clear per-run state and allow another run.

The run loop is async-native. `run_sync()` is a wrapper for callers outside an event loop; it runs `run()` with `asyncio.run()` and then closes owned async resources.

## Core Decisions

The design choices are recorded in `docs/decisions.md`. The most important decisions for reading the code are:

- The harness is async-native, and provider calls use `httpx.AsyncClient`.
- Conversation state is per-run `ModelSession` state. Reusing a `Harness` does not mean reusing provider transcript state.
- A single `Harness` rejects concurrent `run()` calls. Parallel branches should use separate harness instances.
- Tools always return a normalized JSON envelope: `{"ok": bool, "content": str, "metadata": dict}`.
- Argument JSON errors and Pydantic validation failures are model-retryable mistakes.
- Same-turn tool calls run concurrently when every tool in the batch is parallel-safe.
- One sequential tool forces the whole batch to execute serially in model order.
- Hooks can change tool output, but the harness captures retry control flow before `after_tool_call` hooks run.
- Structured output is provider-neutral at the harness boundary and provider-specific only inside adapters.
- Structured-output schema resolution and turn validation are shared through `thinharness/output.py`.
- The synthetic `final_result` structured-output tool is not a normal registered tool.
- Resume is a clean-new-turn API, not interrupted continuation or failed-request retry.
- Subagents are opt-in through the `subagent` tool, and child runs start fresh.
- MCP support is optional and discovered tools are appended once per harness lifecycle.
- `parallel_llm` is available as an opt-in text-only built-in and as a configurable `ParallelLlmTool(...).spec()` that can opt into structured output.

## File Mechanics

### `thinharness/__init__.py`

This file defines the public import surface. It re-exports the main runtime types (`Harness`, `HarnessConfig`, `HarnessResult`), provider models, hook contexts, structured-output markers, subagent config, built-in tool classes, MCP types, and tracing helpers.

The file does not implement behavior. Its main architectural job is making common SDK use pleasant:

```python
from thinharness import Harness, HarnessConfig, ToolSpec
```

Because ThinHarness is a small SDK, this explicit `__all__` list doubles as an API contract. If something is not exported here, consumers can still import it by module path, but it is less clearly public.

### `thinharness/core.py`

`core.py` is the center of the package.

`HarnessConfig` is a Pydantic model for serializable setup knobs: provider reference, root path, tool selection, filesystem limits, search ranking options, model request settings, tracing config, subagent definitions, structured-output settings, retry budgets, parallel LLM batch limits, and MCP servers.

`HarnessResult` is the final return object. It carries:

- `text`: final model text.
- `output`: validated structured output, if configured.
- `responses`: raw provider responses seen during the run.
- `tool_call_records`: normalized call and output history.
- `usage`: model request counts, tool call counts, cancellations, and retry counters.
- `stop_reason`: terminal reason.
- `resume_state`: opaque provider state for clean continuation, when available.

`Harness.__init__()` wires runtime dependencies:

- Resolves `root` and creates it if needed.
- Uses `HARNESS_MODEL` or `config.model`.
- Infers a provider model unless an explicit `model` object is injected.
- Builds filesystem tools with path policies and configured limits.
- Builds a `SkillRegistry`.
- Adds selected built-ins. The default built-ins are only `read`, `write`, `edit`, `search`, `list`, and `glob`; `parallel_llm` and `subagent` are opt-in by name.
- Adds custom tools.
- Builds structured-output schema if configured.
- Stores MCP servers without connecting them yet.
- Validates tool uniqueness, skill selection, `final_result` collisions, and hook filters.

`Harness.run()` owns the agent loop. A few inner helpers matter:

- `fire_run_end_once()` guarantees a single `run_end` event across success, errors, cancellation, and limit exits.
- `check_model_limit()` prevents provider requests beyond `max_model_requests`.
- `check_tool_limit()` rejects a whole model-emitted batch if it would exceed `max_tool_calls`.
- `advance_model()` wraps one provider request with limit checks, model-facing limit notices, usage accounting, tracing, and structured-output finalization annotations.
- `retry_or_fail()` enforces structured-output retry budgets.
- `check_tool_retry_limits()` enforces retryable tool-error budgets per tool name.

Tool execution flows through `_execute_tool_batch()`. In `tool_execution="auto"`, a batch is concurrent unless any called `ToolSpec` is sequential. Concurrent execution still preserves model call order in `tool_call_records` and provider continuation outputs.

`parallel_llm` internal provider attempts are tool-specific work. The tool invocation consumes one `tool_calls` slot, but its internal attempts are reported as `model_requests` in the tool payload and metadata rather than in `RunUsage.model_requests`; `max_model_requests` remains the budget for agent-loop turns.

MCP connection is lazy. `_ensure_mcp_connected()` enters each server context, lists tools, checks collisions, and appends discovered `ToolSpec`s exactly once. It also reserves `final_result` when tool-mode structured output is active, so an MCP tool cannot collide with the synthetic output tool.

### `thinharness/defaults.py`

This file contains `DEFAULT_SYSTEM_PROMPT`. The prompt is deliberately operational: search first, read bounded ranges, use edit/write for changes, and answer concisely with changes and verification. It reflects the package's default filesystem-agent behavior without baking in provider-specific instructions.

### `thinharness/hooks.py`

`hooks.py` defines the lifecycle extension surface.

The hook events are:

- `run_start`
- `user_prompt_submit`
- `before_tool_call`
- `after_tool_call`
- `before_subagent_run`
- `after_subagent_run`
- `run_end`
- `limit_reached`

Each event has a typed dataclass context. Some contexts are mutable by design:

- `UserPromptSubmitContext` can cancel the run or add prompt context.
- `BeforeToolCallContext` can cancel a tool call.
- `BeforeSubagentRunContext` can cancel a child run.
- `AfterToolCallContext` can rewrite the model-visible output.

`HookRegistry.fire()` dispatches matching hooks in registration order. Tool filters only apply to tool events, and agent filters only apply to subagent events. Unknown tool filter names are allowed because tools can be added later and MCP tools are discovered lazily. Subagent filters are validated against known subagent names.

The module also owns `_CURRENT_TOOL_CALL`, a context variable used so nested handlers, especially subagent handlers, can discover the parent tool call id.

### `thinharness/output.py`

`output.py` turns a configured output type into a provider-neutral `OutputSchema`.

There are four modes:

- `native`: ask the provider for JSON schema output.
- `tool`: expose a synthetic `final_result` tool.
- `prompted`: add JSON schema instructions to the system prompt and validate text.
- `text`: populate `HarnessResult.output` with plain text.

Marker dataclasses (`NativeOutput`, `PromptedOutput`, `ToolStructuredOutput`, `TextOutput`) let callers force a mode at the output-type level.

`OutputSchema.build()` uses Pydantic `TypeAdapter` to create a JSON schema and validator. Object outputs can be used directly as tool arguments. Non-object outputs, such as lists or unions, are wrapped under a `value` argument because function tools require object arguments.

The module also normalizes schemas for provider use:

- Inlines simple Pydantic `$defs` references.
- Removes Pydantic-only decoration.
- Sets `additionalProperties: false` on object nodes.
- Determines whether a native schema is strict-compatible.
- Strips one surrounding Markdown JSON fence for prompted-mode validation.

### `thinharness/providers.py`

`providers.py` separates provider transport from harness control flow.

The normalized runtime dataclasses are:

- `ModelToolCall`: one provider tool call with `id`, `name`, and raw JSON `arguments`.
- `ModelTurn`: one provider response, normalized to `text`, `tool_calls`, and raw JSON.
- `ToolOutput`: one local tool output bound to a provider call id.
- `ModelNotice`: provider-neutral harness notice, currently used for near-limit guidance.
- `StructuredOutputRequest`: provider-neutral schema request for native structured output.

`Model`, `ResumableModel`, and `ModelSession` are protocols. This is the main custom-provider extension point: implement the protocol and inject a model into `Harness(..., model=...)`. Every provider request method accepts an optional `notices` list so the harness can pass provider-neutral model-facing guidance without storing hidden pending state on the session.

Provider transports are thin `httpx` wrappers:

- `OpenAIProvider` posts to `/responses`.
- `AnthropicProvider` posts to `/messages` and uses Anthropic auth headers.
- `OpenRouterProvider` posts to `/chat/completions` and can add attribution headers.

`ProviderError` preserves the existing user-facing message and also carries `status_code` when an HTTP response exists. Transport, auth, capability, and invalid-JSON failures leave `status_code=None`; this lets retrying callers classify HTTP failures without parsing message text.

Model adapters translate the normalized session contract into provider-specific payloads:

- `OpenAIResponsesModel` uses server-side continuation through `previous_response_id`. Native structured output maps to Responses `text.format`.
- `AnthropicMessagesModel` stores a local Messages transcript. Tool specs are converted from Responses-style function tools to Anthropic `input_schema` tools. Native structured output is rejected.
- `OpenRouterModel` stores a local Chat Completions transcript. Tools are converted to Chat Completions function tools. Native structured output can be passed through because OpenRouter may forward it to capable upstream models.

Resume state is provider-bound and model-bound:

- OpenAI stores `previous_response_id`.
- Anthropic stores `system` plus `messages`.
- OpenRouter stores `messages`.

`_validate_resume_state()` enforces shape, kind, version, model, required fields, unknown-key rejection, and JSON serializability before mutating session state.

### `thinharness/subagents.py`

Subagents are implemented as a normal-looking built-in tool named `subagent`, but their construction is framework-owned.

`SubAgentConfig` describes named child agents. A named subagent must explicitly define a tool surface through inherited parent tools, explicit built-ins, explicit custom tools, inherited MCP servers, or explicit MCP servers. It cannot expose `subagent` again, which structurally prevents recursive delegation.

Calling `subagent` without an `agent` argument uses the framework default subagent. That route inherits parent tools, except it drops the parent `subagent` tool and MCP-sourced tools.

`run_subagent_tool()` performs the parent-facing tool behavior:

1. Select the named config or default route.
2. Fire `before_subagent_run`.
3. Build a child harness.
4. Connect child MCP servers.
5. Run the child with minimal metadata.
6. Close the child.
7. Fire `after_subagent_run`.
8. Return child text or serialized structured output as a `ToolResult`.

`build_child_harness()` is where most decisions live. It copies the parent config, overrides child-specific fields, clears `subagents`, decides whether to reuse the parent model or create a new model, shares tracing with a child agent name, and passes only the appropriate custom tools.

### `thinharness/tracing.py`

`tracing.py` is a lightweight OpenTelemetry adapter.

`TracingOptions` accepts a tracer and controls whether message text, tool args, and tool results are captured. Capture is off by default because those fields may contain user data.

`RunTracer` creates three span types:

- Agent span: `invoke_agent {name}`
- Model span: `chat {model}`
- Tool span: `execute_tool {tool_name}`

The code accepts tracers that expose either `start_as_current_span()` or `start_span()`, which keeps tests simple and avoids coupling too tightly to one tracing implementation.

`create_otlp_tracing()` is an optional-extra helper. It imports OpenTelemetry SDK packages lazily and raises a clear install error if `thinharness[tracing]` is not installed.

`create_langfuse_tracing()` wraps the OTLP helper for Langfuse. It reads `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and `LANGFUSE_HOST`, defaults to the US cloud host, and sends traces to `/api/public/otel/v1/traces`. Current Langfuse OTLP docs recommend `x-langfuse-ingestion-version: 4` for direct ingestion that should appear in Cloud Fast Preview in real time; ThinHarness exposes that as `legacy_ingestion=True` so callers can opt in deliberately for validation or older deployments.

When `capture_messages=True`, ThinHarness writes request content from an explicit `ModelTraceSnapshot`, not from provider payloads after harness notices have been appended. The top-level agent span records the raw caller prompt as `langfuse.trace.input`; model spans record the effective prompt or tool outputs actually sent to the provider. Child agent spans write observation input/output instead of trace input/output so subagents cannot overwrite the root trace in Langfuse. Message shapes follow the OTel GenAI semantic convention as retrieved on 2026-05-19.

| Attribute | Set on | Purpose |
| --- | --- | --- |
| `gen_ai.system_instructions` | Top-level agent start | System instructions sent to the run, recorded once. |
| `gen_ai.input.messages` | Model request | OTel GenAI logical request messages. |
| `gen_ai.output.messages` | Model response | OTel GenAI assistant response messages. |
| `gen_ai.prompt` | Model request | Backend-compatible logical input fallback. |
| `gen_ai.completion` | Model response and top-level result | Backend-compatible text output fallback. |
| `gen_ai.tool.call.arguments` | Tool span | Opt-in tool arguments when `capture_tool_args=True`. |
| `gen_ai.tool.call.result` | Tool span | Opt-in tool result when `capture_tool_results=True`. |
| `langfuse.observation.input` | Model request and child agent start | Langfuse observation input. |
| `langfuse.observation.output` | Model response and child agent result | Langfuse observation output. |
| `langfuse.trace.input` | Top-level agent start | Raw caller prompt before hook context injection. |
| `langfuse.trace.output` | Top-level successful result | Final harness result payload. |
| `thinharness.model.request.kind` | Model request | Snapshot kind: `start`, `resume`, `tool_outputs`, `correction`, or `output_retry_tool`. |
| `thinharness.output.mode_requested` | Model request | Requested structured-output mode. Distinct from finalized `thinharness.output.mode`. |
| `thinharness.model.notices` | Model request | Serialized harness-owned notices kept separate from logical input messages. |

Example:

```python
from thinharness import Harness, HarnessConfig, TracingOptions, create_langfuse_tracing

tracing = create_langfuse_tracing(service_name="thinharness-dev")
harness = Harness(
    HarnessConfig(root=".", model="openrouter:anthropic/claude-haiku-4.5"),
    tracing=TracingOptions(
        tracer=tracing.tracer,
        agent_name="thin-agent",
        capture_messages=True,
        capture_tool_args=True,
        capture_tool_results=True,
    ),
)
try:
    result = harness.run_sync("Inspect the repo and summarize the tracing setup.")
finally:
    tracing.force_flush()
    tracing.shutdown()
```

### `thinharness/tools/__init__.py`

This file re-exports the public tool-layer API: `ToolSpec`, `ToolResult`, `ModelRetry`, path helpers, filesystem tools, JSONL search types, `ParallelLlmTool`, the `parallel_llm` factory and args type, MCP server classes, and skill registry types.

### `thinharness/tools/base.py`

`base.py` defines the shared tool contract.

`ToolSpec` is the model-facing schema plus the Python handler. Important fields:

- `name`
- `description`
- `parameters`, either a JSON schema dict or Pydantic model class
- `handler`
- `sequential`, used by batch execution
- `metadata`, used for framework and MCP attribution
- `max_retries`, an optional per-tool retry budget override

`ToolResult` is the normalized output envelope sent back to the provider.

`ModelRetry` is the intended way for a handler to say "the model made a fixable mistake; ask it to call this tool again."

`call_tool()` supports synchronous direct invocation. `_invoke_tool()` is the async harness path. `_invoke_tool()` directly awaits async handlers and runs sync handlers in a worker thread so the event loop is not blocked. Both paths normalize results:

- `ToolResult` is serialized as-is.
- `str` becomes successful content.
- JSON-serializable data becomes pretty JSON content.
- malformed JSON arguments and Pydantic validation errors become retry envelopes.
- internal handler exceptions become non-retry tool failures unless the exception came from strict hook machinery.

`PathPolicy` and `contained_path()` enforce workspace-root containment and optional read/write allowlists. This is the foundation for filesystem tool safety.

Schema helpers turn Pydantic models into provider-ready JSON schemas by inlining local `$defs`, removing titles, simplifying nullable `anyOf`, and defaulting object schemas to `additionalProperties: false`.

### `thinharness/tools/filesystem.py`

`filesystem.py` implements the default workspace tools.

`FileTools` owns:

- `root`
- `output_dir` for large truncated outputs
- read and write `PathPolicy` instances
- read, search, and tool-output size limits
- search ranking bucket configuration
- an embedded `JsonlSearch` instance

The tool specs are:

- `read`: reads a UTF-8 file with line numbers, offset, limit, and max char caps.
- `write`: creates, overwrites, or appends a UTF-8 file. Sequential.
- `edit`: exact string replacement with uniqueness and expected-replacement checks. Sequential.
- `search`: runs `rg --json`, groups matches by file, ranks likely definitions and source files first, and formats next-step-oriented output.
- `list`: lists files or directories under a readable path.
- `glob`: returns newest matching files or directories under a readable path.
- `jsonl_search`: delegates to `tools/jsonl.py`.

The search behavior is a core product choice. It does not dump raw ripgrep output. It ranks and explains matches so a model can decide what to read next. Definition-looking lines rank before references; source paths rank before tests; configured low-priority directories such as `vendor` and `node_modules` rank lower.

Large output truncation writes the full text to `.fsharness/outputs/` and returns a head/tail preview with metadata pointing to the saved artifact.

### `thinharness/tools/jsonl.py`

`jsonl.py` implements `jsonl_search`, a structured-data companion to text search.

The tool can:

- Search only `**/*.jsonl` by default.
- Use ripgrep as a prefilter when `query` is provided.
- Scan all scoped JSONL rows when `query` is empty.
- Apply AND-ed `where` filters.
- Project selected fields using jq-style paths.
- Truncate individual projected field displays.
- Preserve file and row counts even when display is limited.

Supported `where` operations are `eq`, `ne`, `in`, `contains`, `regex`, and `exists`. Field paths support dotted keys, numeric indexes, and quoted bracket keys.

The reason this is separate from generic `search` is that JSONL is naturally chunked by line. It gives agents a way to work over structured records without maintaining an embedding index.

### `thinharness/tools/parallel_llm.py`

`parallel_llm.py` implements `ParallelLlmTool`, a normal configurable `ToolSpec` wrapper for batches of independent one-shot model calls. The built-in `parallel_llm` is created by constructing this class from the parent harness's model, root, path policies, and configured prompt/retry caps.

`ParallelLlmArgs` accepts either inline `prompts` or a `prompts_file` containing a JSON array of strings. `prompts_file` is resolved through the tool's read `PathPolicy`; `output_file` is resolved through the write `PathPolicy` and written atomically as pretty JSON. Inline results are compact JSON inside `ToolResult.content`; file mode returns only a summary and failed indices.

The tool deliberately does not inherit the parent system prompt. If `system` is omitted, per-prompt calls use `instructions=""`; if shared instructions are needed, the caller must pass them explicitly. Each attempt uses a fresh `model.new_session()`, so there is no memory or continuation state.

Batch size and retry budget are host-controlled by `ParallelLlmTool.max_prompts` and `ParallelLlmTool.max_attempts`; the built-in fills these from `HarnessConfig.parallel_llm_max_prompts` and `HarnessConfig.parallel_llm_max_attempts`. The built-in model and temperature can be pinned with `HarnessConfig.builtin_parallel_llm_model` and `HarnessConfig.builtin_parallel_llm_temperature`; otherwise it uses the parent harness model and temperature. The model controls only `max_concurrency`, bounded from 1 to 32. Retries are structured around `ProviderError.status_code`: retryable HTTP statuses are 408, 425, 429, 500, 502, 503, and 504; transport errors with the existing `provider request failed:` prefix are also retried.

The model-facing arguments do not include model, temperature, or output-schema overrides. Custom `ParallelLlmTool` instances own provider parsing, API key, base URL, request timeout, temperature, `extra_body`, and optional `output_type` / `output_mode` / `output_retries`. Structured custom tools use the shared `OutputSchema` helpers from `output.py`; successful structured entries are serialized to JSON-compatible Python values, while the built-in `parallel_llm` remains text-only.

### `thinharness/tools/skills.py`

`skills.py` implements explicit skill use.

`SkillRegistry` discovers skills only from configured directories. It looks for `SKILL.md` files recursively and top-level `*.md` files, parses simple frontmatter, and rejects duplicate names.

If skills exist and the harness exposes skill tools, the system prompt gets a compact skill summary. The model still has to call `skill_read` to inspect details. This matches the decision that skills are tools, not auto-discovered prompt stuffing.

The exposed tools are:

- `skill_read`: reads `SKILL.md` or another contained skill file and includes a compact file tree.
- `skill_run`: runs a contained script inside the skill directory. Python scripts run through `uv run`, shell scripts run through `bash`, JavaScript files run through `node`, Go files run through `go run`, and other files run directly. This tool is sequential.

No sandboxing is applied by `skill_run`; the assumption is that SDK callers choose trusted skill directories.

### `thinharness/tools/mcp.py`

`mcp.py` adds optional Model Context Protocol support.

MCP imports are lazy. Importing ThinHarness does not require `mcp`; using MCP without the optional dependency raises `MCPDependencyError` with an install hint.

`MCPServer` is the abstract base for concrete transports. It owns:

- tool prefixing
- include/exclude filters
- connection and read timeouts
- a readable server id
- a reference-counted background session task
- name sanitization and collision detection

Concrete transports are:

- `MCPServerStdio`
- `MCPServerSSE`
- `MCPServerStreamableHTTP`

`list_tools()` connects, asks the server for tools, filters them, sanitizes public names, cleans input schemas, and returns `ToolSpec`s whose handlers call back into the server.

`call_tool()` normalizes MCP results into `ToolResult`. MCP tool errors are retryable. Structured content is serialized as JSON when available; otherwise content blocks are rendered as text placeholders for text, images, audio, and resources.

## Provider Continuation Mechanics

The harness continues providers in batch units. When a model emits multiple tool calls in one turn:

- The harness waits for all results from that batch.
- Outputs are sent back in the same order as the model's calls.
- There is one provider continuation request for the whole batch.

Provider-specific continuation differs:

- OpenAI Responses sends `function_call_output` items plus `previous_response_id`.
- Anthropic Messages appends a user message containing `tool_result` blocks.
- OpenRouter appends Chat Completions `tool` role messages.

Near-limit notices are provider-neutral `ModelNotice` values. Providers render them as either appended text or additional user content, depending on their API shape. The ordering invariant is the same everywhere: provider-required tool outputs come first, hook-supplied prompt context comes before harness notices, and harness notices are last in the user input for that provider request.

Provider notice rendering follows each API's native shape:

- OpenAI initial, corrective, and resumed user prompts append notice text to the string input. Tool continuations append all `function_call_output` items first, then add a Responses `message` item with `input_text` notice content.
- Anthropic initial, corrective, and resumed user prompts append notice text to the user text. Tool continuations put every `tool_result` block first, then append a text block containing the notice.
- OpenRouter initial, corrective, and resumed user prompts append notice text to the user message content. Tool continuations append all `role="tool"` messages first, then a user notice message.

Notices are real model input. For stateless providers they become part of the local transcript; for OpenAI Responses they become part of server-side conversation state behind `previous_response_id`. Deduplication is per `Harness.run(...)`, not per resumed conversation.

## Retry Mechanics

There are two retry systems:

Tool retries happen when a tool output envelope has `metadata.retry == true` and an `error_type`. This includes malformed JSON arguments, argument shape errors, Pydantic validation errors, explicit `ModelRetry`, and MCP tool errors. Retry budget is per tool name per run.

Structured-output retries happen when the model fails to produce valid configured output. For tool mode, invalid `final_result` arguments are answered as tool outputs tied to the failed call id. For text/prompted/native modes, invalid text is corrected with a user message. These retries count against `output_retries`.

In both systems, over-budget failures stop the run before sending a misleading continuation to the provider.

## Tests and Verification Layout

The tests are mostly scripted-model tests. `tests/fakes.py` provides deterministic provider and model sessions so the suite can assert exact payloads, continuation state, hook ordering, retry accounting, and span attributes without live provider calls.

The e2e scripts are intentionally separate from pytest's default path. They use live providers when credentials are present and skip when running in CI or without the required key. They validate that real models can follow the control-plane contracts the unit tests simulate.

CI runs:

```text
uv sync --locked --all-extras --group dev
uv run ruff check .
uv run pytest
```

It also installs `ripgrep`, which is required by `search` and query-prefiltered `jsonl_search`.

## Non-Runtime Files

`README.md` explains the motivation, comparison table, opinions, and quick usage.

`docs/table.md` supports the README comparison table with methodology and per-cell rationale.

`docs/docs.md` currently documents the resume API.

`docs/decisions.md` is the highest-signal design record. When behavior in code looks stricter or simpler than a larger framework would choose, this file usually explains why.

`.context/*.md` files are implementation planning and feedback notes. They are useful historical context, but not runtime source.

`vendor/*` submodules are reference material and comparison inputs. They should not be read as code that ThinHarness imports at runtime.

## Reading Path

For a granular codebase read, use this order:

1. `docs/decisions.md` for the design rules.
2. `thinharness/core.py` for the run loop.
3. `thinharness/providers.py` for provider normalization.
4. `thinharness/tools/base.py` for the tool contract.
5. `thinharness/tools/filesystem.py`, `thinharness/tools/jsonl.py`, and `thinharness/tools/parallel_llm.py` for built-in tools.
6. `thinharness/output.py` for structured output.
7. `thinharness/hooks.py` for lifecycle customization.
8. `thinharness/subagents.py` for delegation.
9. `thinharness/tools/mcp.py` for optional external tool discovery.
10. The matching `tests/test_*.py` files after each runtime file.
