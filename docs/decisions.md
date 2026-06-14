# Decisions

## Runtime Model

- **The harness is async-native.** `Harness.run()` is the primary API and provider calls use `httpx.AsyncClient`. `run_sync()` is a convenience wrapper that owns the event loop and closes owned async resources before returning.
- **Models are reusable configuration objects.** Mutable conversation state lives in per-run `ModelSession` instances created by `model.new_session()` or `model.resume_session(...)`, so repeated harness runs do not share provider transcript state by accident.
- **Harness runs are not re-entrant.** A single `Harness` instance supports one active `run()` at a time. Callers that want parallel branches should construct separate harnesses.
- **Control-plane mutation is setup-time.** `add_tool(...)` exists for registration, but the implementation assumes callers do not mutate a harness's tools while a run is active.
- **Runtime objects stay runtime-shaped.** `HarnessConfig`, `SubAgentConfig`, tracing options, and model settings are Pydantic models. Tool specs, results, turns, usage, and hook contexts stay dataclass/protocol-style runtime objects because they carry callables or live state.

## Tools

- **Provider-facing tool output is always normalized.** Tool handlers may return `ToolResult`, a string, or JSON-serializable data, but the model receives an `{"ok", "content", "metadata"}` JSON envelope.
- **Argument mistakes are retryable model mistakes.** Malformed JSON, non-object arguments, and Pydantic argument validation failures return retry envelopes. Handler-internal exceptions are ordinary tool failures unless the handler raises `ModelRetry`.
- **Tool retry budgets are per tool name per run.** Two calls to the same tool share the same retry counter. This is intentionally coarse and avoids tracking per-call retry identities.
- **Over-budget retry batches are not continued.** When any tool exceeds its retry budget, the harness records local execution, fires `limit_reached`, and ends the run without sending that batch's outputs back to the provider.
- **Hooks can rewrite tool output but not retry control flow.** The retry signal is captured before `after_tool_call` hooks run, so hooks own the message while the harness owns the budget.
- **Requested tool calls count against `max_tool_calls`.** A tool blocked by `before_tool_call` still consumed a model-requested call slot. Cancelled calls are tracked separately in `RunUsage.cancelled_tool_calls`.
- **Human approval is tool execution control flow.** `requires_approval=True` pauses before executing any call in the model-emitted batch and returns `stop_reason="approval_required"` with pending approval details. Rejection is represented as a normal failed tool result sent back to the model, not as a caller exception or retry-budget event.
- **Skill scripts use extension-based runners.** `skill_run` keeps the simple `script` plus `args` interface, but treats Python and shell as first-class skill helper languages: `.py` runs through `uv run`, and `.sh`/`.bash` runs through `bash`, so CLI subcommands and flags work without executable bits. JavaScript and Go get basic file-runner support through `node` and `go run`, but richer package-manager flows such as npm scripts, Go module setup, or Python installed console commands are deferred until real skills need them.
- **Parallel LLM model settings are host-owned.** The model-facing `parallel_llm` arguments cannot override model, temperature, or output schema. The built-in exposes `builtin_parallel_llm_model` and `builtin_parallel_llm_temperature` on `HarnessConfig`; custom `ParallelLlmTool` instances take model settings and optional structured-output settings at construction. Provider/model-specific temperature support is not registry-validated by ThinHarness; unsupported settings surface as provider errors.
- **Parallel LLM prompt source is structurally discriminated.** `parallel_llm` uses one `source` object with `kind="inline"` plus `prompts` or `kind="file"` plus `path`, instead of optional sibling fields. This makes invalid mixed prompt sources structurally harder for models to produce.

## Parallel Tool Execution

- **Same-turn tool calls run concurrently by default.** `tool_execution="auto"` runs a model-emitted batch in parallel when every called tool is parallel-safe.
- **One sequential tool makes the whole batch sequential.** Mutating built-ins such as `write`, `edit`, and `skill_run` are marked `sequential=True`; mixed batches run serially in model order instead of partitioning around barriers.
- **Output order follows model order.** Parallel execution may finish out of order, but the provider continuation and `tool_call_records` preserve the original assistant tool-call order.
- **Parallel execution is concurrency-limited inside the harness.** Tool fanout schedules one asyncio task per model-emitted call, but a semaphore limits how many calls execute at once.
- **Provider continuation remains batch-oriented.** The harness waits for every result in the current assistant tool-call batch before asking the provider for the next turn.

## Hooks And Limits

- **Hooks are runtime-only.** Hook callables are passed to `Harness(...)` or `subagent_hooks`, not embedded in `HarnessConfig`, because they are not serializable configuration.
- **Hook dispatch is synchronous.** Async hook handlers and hook trace spans are out of scope for now.
- **Hook contexts are event-specific dataclasses.** The API prefers typed context objects over a generic data bag, and only prompt/tool/subagent-before contexts are cancellable.
- **Tool filters are passive exact matches.** Hook tool filters match final tool names case-sensitively at dispatch time. The harness does not reject unknown tool-filter names because tools can be added later and MCP tools are discovered asynchronously.
- **Agent filters are validated against known subagents.** Subagent hook filters must reference `default` or a configured named subagent.
- **Parent hooks do not automatically enter child runs.** A parent observes the parent run and the `subagent` tool boundary. Child run hooks are supplied explicitly through `subagent_hooks`.
- **`run_end` fires once.** The run loop uses a guard so success, errors, limit exits, hook cancellation, and external cancellation all produce at most one `run_end` event.
- **Near-limit guidance is deterministic model input.** Hard limits remain authoritative, but the harness now emits provider-facing `ModelNotice` input shortly before configured model-request and tool-call limits are exhausted. These notices are not hook events, are deduped per `Harness.run(...)`, and are computed from each parent or child run's own local budget.
- **Model notice categories may grow.** `ModelNotice.kind` includes budget warnings plus harness notices such as background-task cancellation. Future releases may add notice categories without exposing raw provider payload structures. `tool_retries` near-limit guidance remains deferred and retry exhaustion is still only a hard failure.

## Structured Output

- **Structured output is adapter-neutral at the harness boundary.** The harness builds an `OutputSchema`; providers translate native requests into their own payload fields.
- **`auto` follows provider capabilities.** OpenAI defaults to native structured output, Anthropic defaults to tool mode, and OpenRouter defaults to tool mode while allowing explicit native mode to pass through to the provider.
- **Plain text output is a separate path.** `TextOutput` / `output_type=str` populates `HarnessResult.output` from final text without synthetic tools, native schema payloads, or prompted JSON instructions.
- **`final_result` is synthetic, not a normal tool.** Tool-mode structured output reserves the `final_result` name, does not expose it in `self.tools`, does not fire tool hooks, and does not count it in `usage.tool_calls`.
- **`final_result` must be alone.** If the model emits `final_result` with sibling tool calls, the harness runs no tools and raises `UnexpectedModelBehavior`.
- **Structured output retries are corrective model requests.** `output_retries` counts retry requests after invalid structured output, not total validation attempts.
- **Subagents do not inherit parent structured output.** The default subagent runs unstructured. Named subagents opt into structured output only through their own `SubAgentConfig`.
- **Validated subagent output is serialized with `OutputSchema.dump`.** Structured child results cross the parent tool boundary as canonical JSON text.
- **Custom parallel LLM tools can validate structured output.** The built-in `parallel_llm` remains text-only, while `ParallelLlmTool(...).spec()` can opt into `output_type`, `output_mode`, and `output_retries`. It uses the same schema-resolution and turn-validation helpers as `Harness.run()`, and successful entries contain JSON-compatible parsed values.
- **Streaming structured output is deferred.** The current run loop is turn-based; partial validation and streaming processors are not part of the current design.

## Resume

- **`resume_from` is a new-turn API.** It appends a new user prompt to a completed prior run. It is not a failed-request retry, interrupted-tool continuation, or assistant-response continuation.
- **Resume state is opaque provider-session state.** Callers persist and replay the dict, but do not construct it. State is versioned, provider-bound, model-bound, strictly shaped, and JSON-serializable.
- **Raw OpenAI response IDs are not public harness API.** `previous_response_id` remains an OpenAI session implementation detail; `Harness.run()` accepts only `resume_from`.
- **Resume state is emitted only after clean resumable exits.** Errors, cancellation, limit exits, output validation failure, tool retry exhaustion, and unexpected model behavior all return no checkpoint.
- **Tool-mode `final_result` exits are not resumable.** The provider transcript would contain an unanswered synthetic tool call, so `resume_state` is intentionally `None`.
- **Limit notices are part of provider history.** Model-facing notices are real provider input, so stateless provider resume state may include prior notice text and a resumed run may emit fresh notices for its own budget.
- **No transcript repair in v1.** The harness does not synthesize missing tool results, drop unpaired calls, or reshape provider history to make a non-clean exit resumable.
- **Config compatibility beyond provider and model is caller-owned.** Resume validation checks kind, version, model, and state shape; it does not verify that tools, system prompt, or other harness settings match the original run.
- **Approval resume is not `resume_from`.** Approval pauses are interrupted logical runs, so their `approval_pause` envelope is resumed only through `resume_approvals(...)` / `stream_approvals(...)`. The envelope wraps provider state, the full pending tool batch, run history, metadata, usage, retry counters, and emitted limit-warning keys.
- **Approval budgets span the interruption.** The paused batch counts toward `usage.tool_calls` at pause time exactly once. Resuming restores usage and prior history before processing decisions, so model-request and tool-call budgets continue from the original run rather than resetting.
- **Approval envelope size is run-size dependent.** The envelope stores prior raw provider responses to produce a post-resume result with the whole logical history, so hosts should size approval-state storage for long runs.

## Subagents

- **Subagents are an opt-in built-in tool.** The `subagent` tool is available when included in `builtin_tools` or manually added; the default built-in tool set does not expose delegation.
- **The omitted-agent route is framework-owned.** Calling `subagent` without `agent` runs the framework default subagent named `default`.
- **`default` is reserved.** User-defined `SubAgentConfig` entries cannot use the framework default name.
- **Child runs start fresh.** Subagents do not resume or fork the parent provider session. Forked-conversation subagents remain a separate future feature.
- **No call-time tool narrowing.** A named subagent's tool surface comes from its config, and inherited child tool copying drops the parent `subagent` tool plus MCP-sourced tools.
- **Recursion is structurally disabled for built-in and explicit tools.** Child harness configs always set `subagents=[]`, and `SubAgentConfig` rejects built-in or explicit custom `subagent` tools.
- **Inherited tools are passed as live `ToolSpec` objects.** This preserves bound filesystem, skill, and custom handlers rather than reconstructing tools from names.
- **Approval tools do not enter child harnesses.** Explicit approval-required subagent tools are rejected, and inherited parent tool copying filters approval-required tools out because a paused child run cannot be surfaced coherently through one `subagent` tool result.
- **MCP tools are not inherited as ordinary custom tools.** Named subagents can opt into MCP with `inherit_mcp_servers=True` or explicit `mcp_servers`; inherited parent tool copying filters out MCP-sourced tools.
- **Subagent model overrides are credential-light.** Omitted models reuse the parent model object. Same-provider overrides may reuse parent API settings; different-provider overrides fall back to that provider's normal environment/config path.
- **Child output directories are shared with the parent.** The current implementation keeps artifact handling simple instead of creating per-child output subdirectories.

## MCP

- **MCP support is optional.** Importing the package does not require the `mcp` dependency; using MCP surfaces a dependency error with an install hint.
- **MCP construction is cheap.** Server objects can be placed in config without opening transports. Connections and tool discovery happen on `connect()` or at run start.
- **Discovered tools are appended once per harness lifecycle.** MCP tools become part of the live harness tool map after connection rather than being serialized into config.
- **Resolved server IDs remain server-local.** MCP servers store their resolved ID on the server object. Reusing the same server instance across unrelated harnesses can carry that resolved ID with it, but the v4 plan chose this shape and the edge case is narrow.
- **Content block parsing stays lightweight.** MCP tool result text extraction currently uses the block's `type` value instead of SDK-specific `isinstance` checks. Richer dispatch is deferred until the SDK surface makes it necessary.
- **Serialized MCP config is out of scope.** `HarnessConfig.model_dump()` may include live `MCPServer` objects that are not JSON-roundtrippable. Declarative, serialized MCP configuration can be added as a separate feature.
- **Collision errors remain concise.** Tool name collisions identify the duplicated tool and point users toward `tool_prefix` or `exclude_tools`. Source-specific collision diagnostics are deferred.
- **Lock-held shutdown stays simple.** A rare shared-server timing path can allow a replacement session while the old session finishes tearing down. The current shutdown behavior is accepted instead of adding more lifecycle locking.
- **MCP dynamic capability updates are deferred.** `tools/list_changed` notifications are not handled; the harness uses the tool snapshot discovered at connection time.
- **MCP prompts, resources, sampling, OAuth, and `.mcp.json` discovery are out of scope.** The current feature only turns MCP tools into harness tools over the supported transports.
