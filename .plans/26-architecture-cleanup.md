# Plan: Architecture Cleanup After Review

## Goal

Clean up the remaining high-leverage architecture issues from
`.context/architecture-review.html` while preserving the current user-facing
behavior.

The architecture review is partly stale. `thinharness/types.py`,
`runtime.TurnDriver`, model-request ceremony extraction, background deferral
helpers, and exception classification already exist. This plan starts from the
current tree and focuses only on the remaining work that still improves
locality and leverage:

- keep tool output envelopes structured internally until the provider seam;
- replace framework-tool metadata conventions with typed `ToolSpec` fields;
- remove constructor-time temporal coupling in `Harness.__init__`;
- apply small cleanup items that are independently useful.

`HarnessConfig` regrouping is deliberately out of scope. The flat config is
wide, but nested config objects may make the public interface heavier without
enough payoff right now.

## Assumptions

- This is still pre-production greenfield code. Public breakage is acceptable
  when it produces a cleaner interface, but the implementation should avoid
  gratuitous churn.
- Provider-facing tool outputs remain strings via `ToolOutput.output`.
- Tool handlers may continue returning `ToolResult`, strings, or JSON-compatible
  values.
- Hook semantics remain recognizable, but `AfterToolCallContext` may gain a
  structured primary field and make string mutation secondary.
- No provider payload shapes, model-facing retry text, background completion
  text, or stop reasons should change unless explicitly called out in this plan.

## Current Friction

### 1. Tool output envelope is a stringly typed internal contract

`ToolResult` is structured, but the executor serializes it immediately and then
multiple modules recover meaning by reparsing JSON:

- `ToolCallExecutor.execute_one(...)` parses the output to compute retry state.
- `HookRegistry.fire_after_tool_call(...)` reparses before each after-tool hook.
- `ToolCallExecutor.execute_one(...)` parses again after hooks have run.
- Background execution parses string output to determine failure state.

That spreads knowledge of the `{"ok", "content", "metadata"}` envelope across
`tools/base.py`, `tool_execution.py`, and `hooks.py`.

### 2. `ToolSpec.metadata` carries framework behavior

The generic tool executor currently knows subagent and MCP conventions through
magic metadata keys:

- `"framework_tool": "subagent"`
- `"subagent_background": {...}`
- `"source": "mcp"`
- `"mcp_server_id"` / `"mcp_tool_name"`

This makes `ToolSpec.metadata` an informal type system and forces subagent
background policy into `tool_execution.py`.

### 3. `Harness.__init__` depends on assignment order

The constructor assigns a placeholder `HookRegistry`, calls `add_tool(...)`,
then replaces the hook registry. `add_tool(...)` also probes
`getattr(self, "output_schema", None)` because it can run before
`output_schema` exists.

The code works, but the interface between construction and normal mutation is
too implicit.

### 4. Small cleanup items remain

Small issues from the review are still worthwhile:

- misleading `validate_skills` validator name;
- model capability lookup is duplicated and informal;
- `MCPServer._resolved_id` is set by `Harness`;
- duplicated provider-prefix credential inheritance helper;
- duplicate workspace-relative display helper;
- parallel-LLM description editing uses exact-sentence replacement.

## Target Shape

### Tool output envelope

Add a first-class internal envelope near `ToolResult` in `thinharness/tools/base.py`.
The exact name can be chosen during implementation, but the module should expose
one structured value with this behavior:

- `ok: bool`
- `content: str`
- `metadata: Json`
- `to_json() -> str` for the provider seam
- `from_json(output: str) -> ToolEnvelope` for legacy/custom string rewrites
- `retry_kind() -> str | None`
- `error_type() -> str | None`

`ToolResult` can either become that envelope or become a small compatibility
constructor for it. Prefer the smallest migration that keeps tool handler return
typing clear.

Internally, `ToolCallExecution` and background completion paths should carry the
structured envelope plus the rendered provider string where needed for events,
tracing, and records. `ToolOutput` remains provider-facing and still stores a
string.

### Hook context

`AfterToolCallContext` should make the structured envelope the primary field.
A reasonable shape:

```python
@dataclass(kw_only=True)
class AfterToolCallContext(HookContext):
    ...
    original_output: str
    output: str
    envelope: ToolEnvelope
    duration_ms: float
```

Rules:

- Existing hooks that read or assign `ctx.output` should still work during this
  refactor.
- If a hook changes `ctx.output`, the registry or executor reparses once after
  that hook and refreshes `ctx.envelope`.
- New code should use `ctx.envelope`.
- `parsed_output` should be removed in the envelope phase. Because this is
  greenfield, tests and examples should move to `ctx.envelope` in the same
  change.

The main win is that parsing lives in one module and the executor/hook system no
longer duplicate the envelope parser.

### Typed tool metadata

Extend `ToolSpec` with typed fields instead of framework string conventions.
Likely fields:

```python
ToolKind = Literal["user", "subagent", "parallel_llm", "mcp"]

@dataclass(frozen=True)
class McpToolInfo:
    server_id: str
    tool_name: str

@dataclass(frozen=True)
class BackgroundPolicyDecision:
    mode: ToolBackgroundMode
    known_target: bool = True
    strip_private_arg: bool = True
    unsupported_message: str | None = None

@dataclass(frozen=True)
class ToolSpec:
    ...
    kind: ToolKind = "user"
    background_policy: Callable[[Json], BackgroundPolicyDecision] | None = None
    mcp: McpToolInfo | None = None
```

Implementation details may vary, but the intended locality is fixed:

- subagent-specific background helpers move to `subagents.py`;
- MCP trace attributes read a typed `mcp` field, not `"source": "mcp"`;
- `metadata` remains available for user-defined out-of-band data, not framework
  control flow.
- the typed MCP field replaces only `ToolSpec.metadata`; MCP tool result
  envelope metadata returned by `MCPServer.call_tool(...)` remains byte-identical
  because it is model-facing output.

The background policy callable receives parsed tool arguments after the private
`_background` key has been stripped. It must be able to preserve all current
subagent cases:

- omitted `agent` means the framework default subagent and supports
  model-selected background;
- known named subagent with `background="never"` and `_background: true`
  produces the current retryable `"selected subagent does not support background
  execution"` result;
- unknown named subagent has `_background` stripped and then reaches the
  subagent handler, which returns the existing `UnknownSubAgent` envelope.

### Constructor

Make construction collect and validate complete state before assigning the final
runtime attributes:

1. Resolve config, root, model, model capabilities, skills, and output schema.
2. Build built-in tool candidates.
3. Select built-ins and append constructor-supplied tools into local lists.
4. Build the final hook registry as a local.
5. Validate tools, output-schema collisions, hook filters, skill selection,
   background/approval policies, MCP server ids.
6. Assign `self.tools`, `self._tool_map`, `self.hooks`, and related fields once.

`add_tool(...)` remains the setup-time mutation API after construction. Shared
validation should move into helpers that take explicit parameters instead of
probing partially initialized `self`.

## Phases

### Phase 1 - Small cleanup foundation

This phase should be small and behavior-preserving.

Tasks:

- Rename `HarnessConfig.validate_skills` to a broader validator name, such as
  `validate_config`.
- Add one shared capability helper, for example
  `model_capabilities(model) -> ModelCapabilities`, and use it from both
  `Harness.__init__` and `resolve_output_schema_for_model(...)`. Do not make
  `capabilities` a required runtime attribute for injected custom models in this
  cleanup; keeping the default preserves the lightweight custom-model seam while
  still removing duplicated defensive lookups.
- Add `MCPServer.resolve_id(...)` or equivalent ownership method and stop
  assigning `_resolved_id` directly from `Harness`.
- Extract one provider-prefix helper for parent/child same-provider decisions
  used by subagents and parallel LLM.
- Replace parallel-LLM description `str.replace(...)` with explicit description
  composition. Prefer splitting the default text into base text plus optional
  background sentence instead of duplicating literals at call sites.
- Defer tool schema caching. It is a valid cleanup, but it has cache invalidation
  edges around `add_tool(...)` and MCP connection and is not required for the
  main architecture goal.

Verify:

```bash
uv run pytest tests/test_harness.py tests/test_subagents.py tests/test_parallel_llm.py tests/test_mcp.py
uv run ruff check thinharness tests
uv run pyright
```

### Phase 2 - Constructor cleanup

Tasks:

- Rework `Harness.__init__` to avoid the placeholder hook registry.
- Stop using `getattr(self, "output_schema", None)` inside tool registration.
- Do not call public `add_tool(...)` from the constructor. Extract an internal
  registration/validation helper that can build constructor-local tool lists
  without firing hook-filter validation prematurely; public `add_tool(...)`
  should keep validating after post-construction mutations.
- Extract shared tool validation helpers that accept explicit state:
  `output_schema`, `tool_map`, `is_child_run`, model resume capability, and
  `tool_execution`.
- Keep `add_tool(...)` as a post-construction setup method and preserve duplicate
  tool and reserved-name errors.
- Preserve the current `final_result` rule exactly: a custom `final_result` tool
  remains legal when no structured output is configured, and is rejected only
  when `output_schema is not None and output_schema.mode != "text"`.
- Do not move filesystem side effects (`root.mkdir`) in this phase unless the
  change is trivial and tests already cover it.

Verify:

```bash
uv run pytest tests/test_harness.py tests/test_hooks.py tests/test_subagents.py tests/test_structured_output.py tests/test_approvals.py
uv run ruff check thinharness tests
uv run pyright
```

### Phase 3 - Typed framework tool fields

Tasks:

- Add typed `ToolSpec` fields for framework/tool kind and optional special
  policy data.
- Update subagent tool construction to supply a typed background policy instead
  of `"framework_tool"` and `"subagent_background"` metadata.
- Move `_subagent_background_mode(...)` and `_subagent_agent_known(...)` out of
  `tool_execution.py` or delete them in favor of a typed policy supplied by
  `subagents.py`. The replacement must preserve known-vs-unknown subagent
  behavior described in the target shape.
- Update MCP tool creation to set typed MCP info.
- Update tracing annotation in `tool_execution.py` to use typed fields.
- Update inherited subagent tool filtering to use typed fields instead of
  `metadata.get("source") == "mcp"`.
- Update the reserved-name guard in `Harness.add_tool(...)` from
  `spec.metadata.get("framework_tool")` to the typed field.
- Keep `metadata` only for user-visible or tool-specific payloads that are not
  framework dispatch.

Verify:

```bash
uv run pytest tests/test_background_tools.py tests/test_subagents.py tests/test_mcp.py tests/test_tracing.py
uv run ruff check thinharness tests
uv run pyright
```

### Phase 4 - Structured tool envelope internally

Tasks:

- Introduce the structured envelope in `tools/base.py`.
- Centralize parsing and serialization there.
- Update `_normalize_result`, `_retry_envelope`, `call_tool`, and `_invoke_tool`
  so they produce or render the envelope through the central type.
- Update `ToolCallExecutor`, `ToolBatchExecutor`, `BackgroundToolManager`, and
  approval rejection output paths to use the structured envelope internally.
- Update `AfterToolCallContext` and hook dispatch so hooks get a refreshed
  structured envelope with at most one parse after each string rewrite.
- Remove `AfterToolCallContext.parsed_output` rather than keeping a transition
  alias. This is greenfield code, so tests and examples should move to the new
  `ctx.envelope` field in the same phase. Export the envelope type from
  `thinharness/__init__.py` if it appears in public hook context fields.
- Remove duplicate parsers from `tool_execution.py` and `hooks.py`.
- Ensure retry accounting still captures retry intent before hooks rewrite the
  model-visible output, preserving the existing decision that hooks can rewrite
  messages but not retry control flow.
- Preserve stream event and trace output strings exactly where they are public or
  model-facing.

Verify:

```bash
uv run pytest tests/test_hooks.py tests/test_tool_retry.py tests/test_background_tools.py tests/test_streaming.py tests/test_mcp.py tests/test_harness.py tests/test_approvals.py
uv run ruff check thinharness tests
uv run pyright
```

### Phase 5 - Final consolidation

Tasks:

- Remove dead metadata branches, duplicate parser helpers, and obsolete tests.
- Audit public exports for stale names after hook context changes.
- Run the full validation suite.
- Record any durable non-obvious learning in `.agent/learnings.jsonl` if the
  implementation uncovers a reusable project rule.

Verify:

```bash
uv run pytest
uv run ruff check .
uv run pyright
```

## Test Strategy

- Prefer existing scripted-model integration tests over low-value unit tests.
- Add targeted tests where the refactor changes an interface:
  - hooks can read the structured envelope;
  - hooks can still rewrite `ctx.output`;
  - retry accounting still uses pre-hook retry metadata;
  - known named subagent with `background="never"` plus `_background: true`
    still gets the current retryable unsupported-background envelope;
  - unknown named subagent plus `_background: true` still gets `UnknownSubAgent`;
  - omitted-agent default subagent plus `_background: true` still backgrounds;
  - `_background: false` is stripped before subagent argument validation;
  - subagent background policy no longer depends on `ToolSpec.metadata`;
  - MCP trace attributes still include server and tool names;
  - MCP result-envelope metadata remains unchanged;
  - an injected custom model without `capabilities` still receives default
    capability behavior;
  - constructor-provided hooks validate after all constructor-provided tools are
    known;
  - `final_result` remains legal as a custom tool without structured output and
    remains rejected with structured output.
- Add regression tests before refactoring any path whose behavior is not already
  covered.

## Do Not Touch

- Do not regroup `HarnessConfig` into nested config objects in this plan.
- Do not change provider request payload shapes.
- Do not change model-facing background, retry, approval, or limit-warning text.
- Do not change stop reasons.
- Do not introduce new dependencies.
- Do not split `providers.py` or `tracing.py`; those are separate decisions.
- Do not implement tool schema caching in this plan.

## Open Design Choices For Implementation

- Whether `ToolResult` itself becomes the envelope or a compatibility wrapper.
- Whether `ToolResult` usage in existing tool implementations should return the
  new envelope directly or continue being normalized into it.
- Whether `ToolSpec.kind` plus `mcp` and `background_policy` is sufficient, or a
  richer framework-info object would make the interface clearer.

## Success Criteria

- The generic tool executor no longer branches on subagent-specific metadata.
- MCP trace metadata no longer depends on stringly typed `ToolSpec.metadata`.
- Tool envelope parsing exists in one module.
- `Harness.__init__` no longer needs a placeholder hook registry or partially
  initialized `self` checks.
- Full `uv run pytest`, `uv run ruff check .`, and `uv run pyright` pass.
