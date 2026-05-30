# Plan: Subagents

## Overview
Build subagents as an in-process delegation tool around the existing `Harness`, `ToolSpec`, `ModelSession`, and tracing abstractions. The parent harness can expose a `subagent` tool either as a configured built-in tool or as a manually passed custom tool, and each invocation runs a child harness with configurable tools, system prompt, and nested trace spans.

This plan covers fresh-context subagents only. Forked-conversation mode (where the child reuses the parent's running session state) is a separate feature with provider-specific plumbing; see `plan-subagents-fork-mode.md`.

Prerequisite before adding subagents: tighten the first-party tool-output contract so built-ins, custom tools, and the subagent tool all normalize to the same provider-facing result shape:

```json
{"ok": true, "content": "...", "metadata": {...}}
{"ok": false, "content": "...", "metadata": {...}}
```

Tool handlers may still return `ToolResult`, a plain string, or JSON-serializable data for authoring convenience, but `call_tool()` should wrap every successful return value as `ok: true`. Argument errors, unknown tools, and handler exceptions should normalize to `ok: false`. Tracing should parse normalized tool output and mark a tool span failed when `ok` is `false`; remove the existing raw `"error:"` string path instead of carrying a compatibility fallback. All existing tests affected by this contract change must be updated and passing before starting the subagent implementation.

## References
- `~/code/pi-agent-sdk/src/sub-agent.ts` is the closest shape: `SubAgentDefinition` contains `name`, `description`, `systemPrompt`, `tools`, optional `model`, and `createSubAgentTool()` exposes one parent-facing delegate tool that selects a child agent by name and returns the child output.
- `~/code/pi-agent-sdk/src/agent.ts` shows the integration point: assemble normal tools first, then append the delegate tool when subagents are configured and pass the parent model as the child default.
- `vendor/pydantic-ai/pydantic_ai_slim/pydantic_ai/agent/spec.py` models agents as specs with `model`, `name`, `description`, `instructions`, `model_settings`, and `capabilities`; use the same idea for a lightweight `SubAgentConfig` instead of adding many constructor-only parameters.
- `vendor/pydantic-ai/pydantic_ai_slim/pydantic_ai/capabilities/abstract.py` treats tools, instructions, model settings, and hooks as composable capability contributions; mirror that by letting each subagent define its own system prompt and tool selection while keeping runtime assembly centralized.
- `/Users/ryanbrown/code/claude-code/src/tools/AgentTool/AgentTool.tsx` makes `subagent_type` optional but does not expose a call-time `tools` argument. Regular agents resolve tools from their definitions; the built-in general-purpose agent has `tools: ["*"]`, and the built-in Explore agent bakes its read-only behavior into its own definition with `disallowedTools`.
- `/Users/ryanbrown/code/claude-code/src/tools/AgentTool/runAgent.ts` registers subagents with a parent id for trace hierarchy visualization and resolves the final child tool pool before running the child agent.

## Steps

### 0. Prerequisite: Normalize Tool Outputs and Tool Failure Tracing
Update the harness tool boundary before adding the subagent tool:

- Keep authoring convenience: handlers may return `ToolResult`, `str`, or JSON-serializable values.
- Normalize every provider-facing tool output to `ToolResult(...).as_json()`.
- Convert invalid JSON args, validation errors, unknown tools, and handler exceptions to `ToolResult(False, ...)`.
- Update tracing so `_traced_call_output()` parses normalized JSON and marks the tool span failed when `ok` is `false`; remove the current `"error:"` string check.
- Update existing tests that assert raw custom string/dict outputs or raw parallel tool outputs. After this prerequisite, every provider continuation should receive structured `ok/content/metadata` JSON.

**Verify**: built-ins still return `ok/content/metadata`; a string custom tool is wrapped as `ok: true`; a dict custom tool is wrapped as `ok: true` with JSON content; malformed JSON in `function_call.arguments`, validation errors, handler exceptions, and unknown tools return `ok: false`; tracing marks spans failed for any normalized `ok: false` tool output.

### 1. Make Skill Loading Explicit and Move Skill Validation After Final Tool Assembly
Before the BaseModel conversion, remove implicit skill discovery and move skill-tool validation to the final tool assembly point.

Skill loading should be explicit:
- `HarnessConfig.skills_dir=None` means no skills are configured.
- There is no automatic fallback to `root/.agents/skills`.
- Callers that want workspace skills must pass `skills_dir` explicitly.
- `selected_skills` filters skills discovered from explicit `skills_dir`; if `selected_skills` is set without `skills_dir`, raise `ValueError`.

Add `skills: SkillRegistry | None = None` as an optional keyword-only argument to `Harness.__init__`. When provided, `skills_dir` and `selected_skills` must be unset, and the harness reuses that registry instead of discovering skills from config. This gives inherited children a concrete way to share the parent's skill prompt source and inherited `skill_read`/`skill_run` handlers.

Move skill-tool validation and `_skills_enabled` calculation in `Harness.__init__` until after built-ins and custom tools have both been added and `self.tools` is final. Validation should check the final exposed tool names, not only `HarnessConfig.builtin_tools`.

Precise anchor point: run validation after the `for tool in tools or []: self.add_tool(tool)` loop in `Harness.__init__`. The check should inspect final names:

```python
tool_names = {tool.name for tool in self.tools}
```

**Verify**: no skills are discovered when `skills_dir=None`; explicit `skills_dir` loads skills; `selected_skills` without `skills_dir` fails; `skills=` with `skills_dir` or `selected_skills` fails; explicit parent skills still require `skill_read` or `skill_run` in the final exposed tool list; a parent configured with `skills_dir` and `builtin_tools=["skill_read"]` passes; a child inheriting the parent's skill registry and final skill tools passes with `builtin_tools=[]` and ends with `_skills_enabled=True`; a child with explicit skills but no final skill tools still fails.

### 2. Convert Public Config Objects and Add Subagent Configuration Types
Because the project is greenfield, convert public/config-style objects to `pydantic.BaseModel` before adding subagents. This gives cleaner `@model_validator(mode="after")` validation, `model_dump()`/debug serialization, schema-oriented field metadata, and `model_copy(update=...)` for child config inheritance.

Convert these public config objects:
- `HarnessConfig`
- `SubAgentConfig`
- `TracingOptions`
- `ModelSettings`

Keep runtime structs such as `HarnessResult`, `ModelTurn`, `ModelToolCall`, `ToolOutput`, `ToolResult`, `ToolSpec`, `SearchMatch`, and `SearchFile` as dataclasses unless a separate refactor needs otherwise. `ToolSpec` contains a callable handler and is a runtime object, not a config payload.

Add `thinharness/subagents.py` with the small public subagent config model, plus exports from `thinharness/__init__.py`.

Import `Json` from `.tools`, not from `pydantic`. `pydantic.Json` is a special encoded-string validator and will break dict-style tool definitions in confusing ways. Keep the union ordered as `ToolSpec | Json` so Pydantic first accepts real `ToolSpec` objects and only then falls back to dict-style tool definitions.

```python
class SubAgentConfig(BaseModel):
    """Configuration for one delegated child harness."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.-]+$")
    description: str = Field(min_length=1)
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    inherit_parent_tools: bool = False
    builtin_tools: list[str] = Field(default_factory=list)
    tools: list[ToolSpec | Json] = Field(default_factory=list)
    model: str | None = None
    max_turns: int | None = None
```

Add fields to `HarnessConfig` after converting it to `BaseModel`:

```python
subagents: list[SubAgentConfig] = Field(default_factory=list)
```

`SubAgentConfig` represents user-defined named subagents only. The framework-created default subagent is not represented in `HarnessConfig.subagents`.

- **Selection default**: if the parent calls the subagent tool without an `agent` arg, route to the framework-created default subagent. The default subagent runs in fresh context, uses tracing name `subagent.default`, and inherits the parent effective tool universe minus the `subagent` tool.
- **Parent-tool inheritance**: parent-tool inheritance is controlled only by `inherit_parent_tools=True`, not by the agent name.

**Construction validation**:
- Put BaseModel construction checks in `@model_validator(mode="after")`, not dataclass-style `__post_init__`.
- Raise `ValueError` if `inherit_parent_tools=True` is combined with a non-empty `builtin_tools` or `tools` on the same config. Inheriting and specifying your own pool are mutually exclusive; silent ignoring is a footgun.
- Raise `ValueError` if any named, non-inheriting subagent has neither `builtin_tools` nor `tools`. Named subagents must always have an explicit tool set; they should never accidentally receive all filesystem built-ins through `HarnessConfig.builtin_tools=None`.
- Named inherited agents are allowed. They inherit the parent tool universe and cannot use per-call narrowing.
- Validate `name` and `description` as non-empty single-line fields. `name` should be a stable identifier safe for tool arguments, tracing, and metadata; `description` is rendered into the parent-facing tool description.

**Model field**: `model: str | None` mirrors `HarnessConfig.model` (a model ref like `"openai:gpt-5.2"`), resolved at child build time via `infer_model` only when it is explicitly set. If `model` is omitted, the child should reuse `parent.model`, not reconstruct from `parent.config.model`, so custom parent models, providers, clients, and gateway settings are preserved. This is safe because the current model layer creates isolated per-run `ModelSession` objects with `new_session()`.

Document that a subagent model override uses normal provider credential resolution. Same-provider overrides may inherit the parent's `api_key` and `base_url`; different-provider overrides must not receive the parent's provider credentials and should fall back to that provider's env/config path, such as `ANTHROPIC_API_KEY` / `ANTHROPIC_BASE_URL`. Missing or invalid child credentials should surface as a child provider failure and be returned by the subagent tool as `ok: false`; do not add provider-specific preflight validation in the subagent layer. The plan does not add per-subagent credential fields.

**Import boundary**: avoid a `core.py` <-> `subagents.py` circular import while keeping the public API typed.
- Move `DEFAULT_SYSTEM_PROMPT` to a tiny shared module such as `thinharness/defaults.py`.
- Keep `SubAgentConfig` in `subagents.py`.
- In `subagents.py`, import `Harness`/`HarnessConfig` only under `TYPE_CHECKING` at module load. Runtime helpers that need them should use local imports.
- In `core.py`, import `SubAgentConfig` at runtime for `HarnessConfig.subagents: list[SubAgentConfig]`, because Pydantic resolves that annotation. Locally import `create_subagent_tool` inside `Harness.__init__` when assembling built-in candidates.
- Keep broader module-boundary and vendoring cleanup as a later refactor rather than weakening the initial public API.

**Verify**: run the full suite after the BaseModel conversion before adding subagent runtime code: `uv run --extra dev pytest -q`. Then add import/export and construction tests for `SubAgentConfig`, including a config whose `tools` contains a `ToolSpec` and a dict-style tool side by side. Add validation tests for inherited/custom tool conflicts, invalid names/descriptions, named agents without explicit tools, and named inherited agents.

### 3. Introduce a Parent-Facing Delegate Tool
Implement `create_subagent_tool(parent: Harness, configs: list[SubAgentConfig]) -> ToolSpec` in `thinharness/subagents.py`. The tool schema supports named delegation to user-defined subagents and omission for the framework-created default subagent. It does not expose call-time tool selection; each user-defined subagent has a fixed effective tool surface.

```python
class SubAgentArgs(BaseModel):
    """Arguments for subagent delegation."""

    task: str
    agent: str | None = Field(default=None, min_length=1, description="Optional subagent name; omit to use the framework default subagent.")


def create_subagent_tool(parent: Harness, configs: list[SubAgentConfig]) -> ToolSpec:
    """Create the parent-facing subagent delegation tool."""
    return ToolSpec(
        "subagent",
        _subagent_tool_description(configs),
        SubAgentArgs,
        lambda args: run_subagent_tool(parent, configs, args),
    )
```

Selection rules:
- If `args.agent` is provided, match the named config. Blank strings are invalid input and should return the normal structured argument-validation error instead of being treated as omission.
- If `args.agent` is omitted, run the framework-created default subagent.
- Unknown named agents return a structured tool error listing available user-defined agents.

Tool-surface rule:
- The parent-facing tool does not accept `tools` or any other call-time narrowing argument.
- The framework-created default subagent inherits the parent's effective tool universe minus the `subagent` tool.
- A user-defined named subagent's effective tools come only from its `SubAgentConfig`: either the full inherited parent tool universe when `inherit_parent_tools=True`, or its explicit `builtin_tools`/`tools` when `inherit_parent_tools=False`.
- If a future internal prebuilt default variant needs Explore-like behavior, model it as a separate framework-defined policy rather than exposing tool selection to the parent model at call time.

Tool description:
- `_subagent_tool_description(configs)` should list only explicitly callable named agents as `- name: description`.
- Mention the omitted-agent route separately as "Omit `agent` to use the framework default subagent."
- The description is rendered when `create_subagent_tool(...)` is called. Dynamic subagent registration is out of scope; callers that change subagent configs later should recreate the delegate tool or harness.

Example description:

```text
Delegate one self-contained task to a sub-helper. Each subagent runs in isolated context.

Available agents:
- research: Searches and reads code without editing.

Omit `agent` to route to the framework default subagent.
```

**Verify**: tests for unknown agent, blank agent string validation, omitted agent using the framework default subagent, omitted agent working even when `configs=[]`, tool description listing callable named agents while describing the omitted-agent default route separately, the provider-facing schema not containing a `tools` field, framework default subagent receiving all parent tools minus `subagent`, and named inherited agents receiving all parent tools minus `subagent`.

### 4. Build Child Harnesses from Parent Defaults plus Agent Overrides
Add a helper that creates a child harness for one invocation. Inherit the parent root, provider model, limits, tracing settings, and search settings unless the subagent overrides them; override system prompt and derive the effective tool pool from the agent config.

```python
def build_child_harness(parent: Harness, config: SubAgentConfig | None) -> Harness:
    """Create an isolated child harness for one subagent invocation."""
    parent_config = parent.config
    inherit_tools = config is None or config.inherit_parent_tools
    child_wants_skills = bool(config and any(name in {"skill_read", "skill_run"} for name in config.builtin_tools))
    child_skills = parent.skills if inherit_tools else None
    child_config = parent_config.model_copy(update={
        "model": config.model if config is not None and config.model is not None else parent_config.model,
        "root": parent.root,
        "system_prompt": DEFAULT_SYSTEM_PROMPT if config is None else config.system_prompt,
        "builtin_tools": [] if inherit_tools else config.builtin_tools,
        "skills_dir": parent_config.skills_dir if child_wants_skills and not inherit_tools else None,
        "selected_skills": parent_config.selected_skills if child_wants_skills and not inherit_tools else None,
        "max_turns": config.max_turns if config is not None and config.max_turns is not None else parent_config.max_turns,
        "subagents": [],
    })
    child_model = parent.model
    if config is not None and config.model is not None:
        same_provider = _same_provider(parent.model, config.model)
        child_model = infer_model(
            config.model,
            api_key=parent_config.api_key if same_provider else None,
            base_url=parent_config.base_url if same_provider else None,
            timeout=parent_config.request_timeout,
            temperature=parent_config.temperature,
            extra_body=parent_config.extra_body,
        )
    return Harness(child_config, model=child_model, tools=effective_custom_tools(parent, config), tracing=_child_tracing(parent, config), skills=child_skills)
```

Use explicit `is not None` override semantics for every optional subagent override. Do not use `config.max_turns or parent_config.max_turns` or `config.model or parent_config.model`; falsy values should not silently fall through.

Tool pool resolution:
- For the framework default subagent (`config is None`) and inherited named subagents (`inherit_parent_tools=True`): create the child with `builtin_tools=[]`, then start from the parent harness's currently exposed `ToolSpec` instances and drop the `subagent` tool (by name). Transfer the `ToolSpec` objects unchanged; do not reconstruct inherited tools from names. This preserves bound handlers such as filesystem tools tied to the parent's root/options and skill tools tied to the parent's `SkillRegistry`.
- If the inherited parent tool set includes `skill_read` or `skill_run`, the child must not create a second independent `SkillRegistry` for those same inherited handlers. Pass `skills=parent.skills` into the child harness, set child `skills_dir=None` and `selected_skills=None`, and reuse the parent's skill registry/prompt source so the child system prompt and the inherited skill handlers describe and execute against the same skill set.
- For named agents: `builtin_tools` is forwarded to the child `HarnessConfig` so the standard `_select_builtin_tools` path applies. Custom `tools` in the config are registered on the child harness via `Harness.add_tool`, which is the single entry point for both `ToolSpec` and dict-style tool definitions. This guarantees flags such as `sequential` are honored on dict-style tools without re-implementing the conversion in subagents.py.
- For named, non-inherited agents that do not include `skill_read` or `skill_run` in `builtin_tools`, set child `skills_dir=None` and `selected_skills=None` even if the parent has skills configured. For named agents that explicitly include skill tools, keep the parent's explicit `skills_dir`/`selected_skills` so the child builds a matching skill registry for its own skill tool specs.
- Resolve inherited parent tools when the subagent tool is invoked, not when the `subagent` tool is created. Parent construction selects built-ins first and then adds custom tools, and callers may also call `harness.add_tool(...)` later.
- `create_subagent_tool` always emits the canonical tool name `"subagent"`. Do not support overriding the delegate tool name in the base implementation; filtering inherited tools by `tool.name != "subagent"` is the recursion guard.
- Child harness configs always set `subagents=[]`, regardless of named or inherited mode. This makes recursion structurally impossible instead of relying on each named subagent to omit `"subagent"` from `builtin_tools`.
- Reading `parent.tools` inside the subagent handler assumes `Harness.add_tool(...)` is not called concurrently with `Harness.run(...)`. The harness remains single-run/single-thread for control-plane mutations.

Model override credential resolution:
- Parse the provider prefix from `config.model` and compare it to the parent model's provider.
- Implement the comparison explicitly: parse `config.model` with `parse_model_ref(...)`, normalize `parent.model.provider.name` to the same provider-prefix vocabulary, and compare those strings. If the parent was constructed from a handcrafted `model=` object rather than the normal `infer_model(...)` path, `parent_config.api_key` and `base_url` may not describe that object; callers are responsible for custom-model credential behavior.
- Forward `parent_config.api_key` and `parent_config.base_url` only for same-provider overrides.
- For different-provider overrides, pass `api_key=None` and `base_url=None` into `infer_model(...)` so the child provider uses its own environment/config path.
- Do not add per-subagent credential fields in this implementation.

**Verify**: unit tests for: framework default child inherits the parent's actual exposed `ToolSpec` objects without duplicate built-ins; inherited named child with explicit parent skills does not fail construction when the final inherited tool set includes skill tools; inherited child skill prompts and inherited skill handlers use the same registry; named child without explicit tools is rejected; named child with `builtin_tools=[]` plus custom tools does not expose filesystem tools or parent skills; named child with explicit skill tools uses skills from the parent's explicit skill config; named child configured with a custom echo tool can use it even if the parent lacks that tool; child config has `subagents=[]` in all modes; omitted model reuses the parent model object; model override wins over the parent model; same-provider model override forwards parent credentials; different-provider model override does not forward parent credentials; model override credential failures are returned as `ok: false`; child `HarnessConfig.model` records `config.model if config.model is not None else parent_config.model`; dict-style custom tool registered via `SubAgentConfig.tools` with `"sequential": True` is treated as sequential in the child harness.

### 5. Wire the Tool as Either Built-in or Custom
Add the subagent tool to the built-in candidate list before `_select_builtin_tools()` in `Harness.__init__`, so existing `builtin_tools` selection controls exposure. This candidate exists even when there are no user-defined subagents because omitted `agent` routes to the framework default subagent.

```python
builtin_candidates = [*filesystem_tools, *self.skills.specs()]
from .subagents import create_subagent_tool

builtin_candidates.append(create_subagent_tool(self, self.config.subagents))
builtin = self._select_builtin_tools(builtin_candidates, self.config.builtin_tools)
```

This gives three supported usage patterns:
- Automatic built-in exposure: leave `builtin_tools=None`, and the parent sees `subagent` even if there are no user-defined subagents, because omitted `agent` uses the framework default subagent.
- Explicit built-in exposure: set `builtin_tools=["read", "search", "subagent"]`.
- Custom tool exposure: construct the parent harness first, then call `harness.add_tool(create_subagent_tool(harness, configs))` when the caller does not want config-driven built-ins.

Document that `subagent` is a normal parent tool option. `builtin_tools=[]` disables the subagent tool entirely, and `builtin_tools` must include `"subagent"` for the model to delegate when explicit tool selection is used. The canonical tool name `"subagent"` is reserved whenever the delegate tool is exposed; adding a custom tool with the same name should raise a clear duplicate/reserved-name error.

**Verify**: tests cover `builtin_tools=None` includes `subagent`, `builtin_tools=[]` excludes it, `builtin_tools=["subagent"]` exposes only the delegate tool even when there are no user-defined subagents, duplicate custom `"subagent"` tools fail clearly, and post-construction `harness.add_tool(create_subagent_tool(harness, configs))` works.

### 6. Make Nested Tracing First-Class
Ensure child runs execute inside the parent `execute_tool subagent` span and reuse the same tracer/context so observability platforms render the child as nested under the parent tool call.

Target span shape:

```text
invoke_agent thinharness
  chat gpt-5.2
  execute_tool subagent
    invoke_agent subagent.default
      chat gpt-5.2
      execute_tool read
      chat gpt-5.2
```

Implementation details:
- `_traced_call_output()` already opens a tool span before calling the tool handler; keep child `Harness.run()` inside that handler so nested tracing follows the active OpenTelemetry context.
- Add a small harness-private `ContextVar` for the current tool call id/name only, set inside `_traced_call_output()` before invoking the handler and reset in `finally`. The subagent handler uses this to include `parent_call_id` in child metadata. Do not duplicate the active OpenTelemetry context or parent run metadata inside this variable.
- Context variables do not automatically propagate into `ThreadPoolExecutor` workers. Update `_run_calls_in_threads()` so each submitted tool call runs with a copied context, for example with `contextvars.copy_context().run(invoke, call)`. Prefer this single propagation primitive over bespoke OpenTelemetry-only attach/detach helpers, so the current tool call id/name and OTel context propagate together. Parallel subagent calls must each retain their own parent tool context.
- Store parent run metadata for the duration of `Harness.run()` in a private harness attr and clear it in `finally`, so the subagent tool can derive the child conversation id without changing public custom tool handler signatures.
- `_child_tracing(parent, config)` should return `None` immediately when `parent.tracing is None`. Otherwise it should clone the parent's `TracingOptions` with `agent_name=f"subagent.{config.name if config else 'default'}"`, `agent_description=config.description if config else "Framework default subagent"`, and the same tracer/capture settings. Do not add a `parent_context` field to `TracingOptions`; with copied context propagation, `RunTracer._span()` and `start_as_current_span()` should naturally parent child spans under the current parent tool span.
- Do not create a disconnected tracer provider for child runs; reuse the existing tracer instance.
- Record child errors on both the child agent span and the parent `subagent` tool span so traces show failure at both the delegation boundary and the child run boundary. This should use the generic normalized tool-output contract: any tool result with `ok: false` marks the tool span failed, regardless of whether the tool is `read`, `search`, `skill_run`, `subagent`, or a custom tool normalized by the harness.
- Add trace attributes for `subagent.name`, `subagent.tool_mode = "inherited" | "explicit"`, and `subagent.tools = [<effective tool names>]`.

**Verify**: extend the fake tracer tests to assert parent pointers exactly: parent model span and `execute_tool subagent` span are children of the root parent agent span; child `invoke_agent subagent.<name>` is a child of `execute_tool subagent`; child model/tool spans are children of the child agent span. Also assert the child span carries `gen_ai.agent.name == "subagent.<name>"`, the subagent attributes are set, child metadata includes only the intended conversation id and parent call id, and generic tool tracing tests prove `ok:false` results mark spans failed. Add tests for `tracing=None` parent runs that assert no spurious tracer/span objects are created, and for concurrent subagent fan-out where two same-response subagent calls each nest under their own parent tool span.

### 7. Add Result Formatting and Error Behavior
Return the normalized tool result shape: `{"ok": true, "content": "...", "metadata": {"agent": "..."}}`. Tool failures such as unknown named agent or child execution failures should return `ok: false` instead of raising through the parent loop, matching normal tool failure behavior.

On success, `content` is `child_result.text`, the child's final assistant text. Metadata should include `{"agent": <name or "default">, "inherited": <bool>, "tools": [<final effective tool names>], "turns": len(child_result.responses)}`. Keep full child transcripts out of the parent context by default; add them later behind an explicit debug option if needed.

Child run metadata should be minimal: include `conversation_id` from the parent run metadata when present and `parent_call_id` from the current subagent tool call when available. Do not forward the parent `metadata` dict wholesale.

Wrap the entire child build/run path in `try/except Exception` inside the subagent handler and return `ToolResult(False, str(exc), {"agent": <name or "default">, "error_type": type(exc).__name__})`. This gives subagent-specific error metadata for build failures, provider failures, duplicate tool failures, and `HarnessError` instead of relying on the generic tool-call exception wrapper.

**Verify**: unknown named agent, child build failures, child provider failures, and child `HarnessError` return `ok: false` with subagent metadata; successful child runs return `ok: true`; parent tool span records the JSON result when `capture_tool_results=True` and is marked failed when the result has `ok: false`.

### 8. Document Usage
Update `README.md` with focused examples for the framework default subagent and specialized named subagents.

```python
harness = Harness(HarnessConfig(
    root=".",
    subagents=[
        SubAgentConfig(
            name="research",
            description="Searches and reads code without editing.",
            system_prompt="Investigate and report findings. Do not edit files.",
            builtin_tools=["read", "search", "glob"],
        )
    ],
))
```

Also show explicit built-in exposure:

```python
HarnessConfig(
    builtin_tools=["read", "write", "edit", "subagent"],
    subagents=[SubAgentConfig(name="research", description="Research helper.", builtin_tools=["read", "search", "glob"])],
)
```

**Verify**: README snippets should be valid imports after `__init__.py` exports are added.

## Considerations
- The framework default subagent is selected by omitting `agent`; it is not configured through `SubAgentConfig`. It inherits the parent tool universe minus the `subagent` tool.
- Named inherited subagents also receive the parent tool universe minus the `subagent` tool. There is no call-time tool narrowing in the provider-facing schema; if a caller wants a different tool surface, they should define a different named agent.
- Avoid recursive subagents. Child harnesses don't get the parent subagent tool; nested delegation is out of scope for this implementation.
- Skill loading is explicit. `skills_dir=None` means no skills, and callers that want workspace skills should pass the directory they want loaded.
- The plan intentionally keeps outputs text-first and JSON-wrapped like existing tools. Full child transcripts can be added behind a metadata/debug flag later, but they should not be dumped into parent context by default.
- Nested tracing should be treated as part of correctness, not polish. A subagent implementation that returns the right text but produces disconnected child spans is incomplete.
- Parallel tool execution composes with subagents: the parent can emit multiple subagent calls in one response and each child can independently run parallel tools. With the 16-worker cap per harness, worst-case thread fan-out is 16 parent workers × 16 child workers = 256 for I/O-bound work. Recursion is structurally impossible because children have `subagents=[]`.
- Subagent-level timeout/cancellation is deferred. A child stuck in a provider call will occupy the parent tool call until the underlying provider timeout fires.
