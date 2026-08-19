# Subagents plugin — plan v2

Move delegation from the last built-in tool path to an explicit `SubagentsPlugin`. The plugin owns the `subagent` tool, child definitions, inheritance policy, child hooks, and delegation results. Core keeps only a narrow child-harness execution module that any trusted plugin can call without receiving the parent `Harness` object.

This is a clean pre-1.0 break. Do not add aliases, fallback reads, deprecated paths, dual configuration, or migration constructors.

## Resolved decisions

1. **Delegation is explicit.** A harness gets delegation only from one `SubagentsPlugin`. Plain `Harness(...)` has no model-callable delegation tool.
2. **The unnamed child remains.** `SubagentsPlugin()` contributes `subagent`; omitting the model-visible `agent` argument selects the framework default child. Named children are optional.
3. **Children cannot delegate.** `SubagentsPlugin` never inherits into a child and is rejected in explicit child plugin configuration. Every child receives a disabled child-harness host that rejects before creating resources, so an inherited custom plugin cannot create a grandchild through `PluginContext`. A custom tool named `subagent` is not delegation and remains an ordinary tool.
4. **Inheritance rebinds plugins.** Replace `inherit_parent_tools` with additive `inherit_parent=True`. Safe parent plugins bind again against the child context rather than copying their parent-bound handlers.
5. **Safe plugin inheritance is explicit.** A plugin opts in through a structural `for_child() -> Plugin` method. `FilesystemPlugin`, `SkillsPlugin`, and `ParallelLlmPlugin` opt in and return themselves from a frozen constructor configuration. `MCPPlugin` and `SubagentsPlugin` do not opt in. A custom plugin without `for_child()` does not inherit.
6. **Parallel batches follow the child model.** Rebinding `ParallelLlmPlugin(model=None)` makes it borrow the child model. An explicit model object or model string keeps its configured model behavior.
7. **MCP is explicit in children.** Remove `SubAgentConfig.mcp_servers` and `inherit_mcp_servers`. A child that needs MCP includes `MCPPlugin(...)` in its own `plugins` list. Parent MCP connections never inherit implicitly.
8. **Inheritance is additive.** A named child may combine `inherit_parent=True` with explicit child `plugins` and `tools`. Inherited values come first. Duplicate plugin or tool names fail through normal atomic composition; explicit values do not silently replace inherited values.
9. **Child hooks stay with child configuration.** Named-child lifecycle hooks live in `SubAgentConfig.hooks`. Hooks for the unnamed child live in `SubagentsPlugin(default_hooks=...)`. Parent `before_subagent_run` and `after_subagent_run` hooks remain ordinary `Harness(..., hooks=...)` hooks.
10. **No special tool kind or reserved name remains.** Remove `ToolKind`, `ToolSpec.kind`, the `"subagent"` kind, and the core name reservation. A direct custom tool may use the name `subagent` when no plugin tool has that name. Normal duplicate-tool validation rejects a collision with `SubagentsPlugin`.
11. **Core does not receive subagent configuration.** Remove `HarnessConfig.builtin_tools`, `HarnessConfig.subagents`, and `Harness(subagent_hooks=...)`. Core does not import `SubAgentConfig`, `SubagentsPlugin`, or subagent tool construction helpers.
12. **Named children may have no tools.** A system-prompt-only or model-only named child is valid. Hooks alone do not expose tools, and no artificial tool-source requirement remains.
13. **Direct-tool inheritance follows run freezing.** The child sees the direct tools frozen for the active parent run. A direct tool added during that run is first available to a child in the next run. This is an intentional change from the current live-list behavior.
14. **Other execution behavior stays stable.** Child runs remain fresh, one level deep, independently budgeted, streamed and traced under the parent, and always closed. Model ownership, provider-setting projection, structured results, cancellation, errors, metadata, and before/after hooks keep their current meaning.

## Target interface

```python
from thinharness import (
    FilesystemPlugin,
    Harness,
    HarnessConfig,
    Hook,
    MCPPlugin,
    MCPServerStdio,
    SubAgentConfig,
    SubagentsPlugin,
)

research_mcp = MCPPlugin(
    servers=[MCPServerStdio("python", ["research_server.py"])],
)

harness = Harness(
    HarnessConfig(model="openai:gpt-5.5"),
    plugins=[
        FilesystemPlugin(tools=["read", "search"]),
        SubagentsPlugin(
            default_hooks=[
                Hook("run_start", prepare_default_child),
            ],
            agents=[
                SubAgentConfig(
                    name="researcher",
                    description="Research one focused question.",
                    system_prompt="Return concise findings with sources.",
                    model="anthropic:claude-opus-4-6",
                    inherit_parent=True,
                    plugins=[research_mcp],
                    tools=[custom_research_tool],
                    hooks=[
                        Hook("run_start", prepare_researcher),
                        Hook("run_end", inspect_researcher_result),
                    ],
                ),
            ],
        ),
    ],
    hooks=[
        Hook("before_subagent_run", approve_delegation),
        Hook("after_subagent_run", record_delegation),
    ],
)
```

`SubagentsPlugin` has the fixed runtime name `"subagents"` and contributes one tool named `subagent`. Its constructor is keyword-only:

```python
SubagentsPlugin(
    *,
    agents: Sequence[SubAgentConfig] = (),
    default_hooks: Sequence[Hook] | HookRegistry | None = None,
)
```

Reject unordered agent collections, duplicate names, the reserved config name `"default"`, invalid plugin values, and any child configuration that contains `SubagentsPlugin`. Perform the recursion check in `thinharness/plugins/subagents.py` by object type, not by plugin name in core. Reusing one plugin object across harnesses binds it to each harness's own child host, root, and model.

The unnamed child intentionally keeps today's fixed parent-derived model, system prompt, limits, output behavior, and inheritance policy. `default_hooks` is its only plugin-level override. Do not add a second default-child configuration type in this slice.

`SubAgentConfig` keeps:

```python
SubAgentConfig(
    *,
    name: str,
    description: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    inherit_parent: bool = False,
    plugins: Sequence[Plugin] = (),
    tools: Sequence[ToolSpec] = (),
    hooks: Sequence[Hook] | HookRegistry | None = None,
    model: str | None = None,
    max_model_requests: int | None = None,
    max_tool_calls: int | None = None,
    output_type: Any | None = None,
    output_mode: Literal["auto", "native", "tool", "prompted"] = "auto",
    output_retries: int = 1,
    tool_retries: int = 1,
)
```

Keep the current name and one-line description validation. Reject approval-required explicit tools because approval pauses remain unsupported inside children. Reject `SubagentsPlugin` in explicit child plugins by object type. Child hook collections must not contain `before_subagent_run` or `after_subagent_run` hooks and must not use `Hook.agents`; children cannot delegate, so these settings are invalid. Validate named and default child hooks when `SubagentsPlugin` is constructed, not when a child first runs. Remove `inherit_parent_tools`, `inherit_mcp_servers`, `mcp_servers`, and `builtin_tools`; reject these removed names with direct migration errors rather than accepting them.

## Narrow child-harness host

Do not put `Harness`, `HarnessConfig`, plugin lookup, mutable tool maps, tracing internals, or provider credentials into `PluginContext`.

Add one narrow host capability to `PluginContext`:

```python
@dataclass(frozen=True)
class PluginContext:
    root: Path
    model: Model
    child_harnesses: ChildHarnessHost
```

Define the public `ChildHarnessHost` protocol and immutable request/outcome types in `thinharness/children.py`; export them from `thinharness/__init__.py` so third-party plugins can type and fake `PluginContext.child_harnesses`. Keep the parent-holding implementation private in the same neutral module. `thinharness/plugins/base.py` imports the host type only under `TYPE_CHECKING`, which avoids a core↔plugins runtime cycle.

The request carries the plugin-owned delegation vocabulary as opaque data: agent name, agent description, trace agent name, task, `inherited`, `tool_mode`, child system prompt, model override, explicit plugins/tools/hooks, limits, output settings, and retry settings. Core does not construct `"subagent."` names or interpret the tool-mode values. The outcome carries the child result, effective tool names, structured-output serialization data, and any failure needed for the plugin's `ToolResult`.

`tool_mode` has three stable values:

- `"inherited"` for the default child or a named child with `inherit_parent=True` and no explicit plugins/tools;
- `"inherited+explicit"` for `inherit_parent=True` plus explicit plugins or tools;
- `"explicit"` for `inherit_parent=False`, including a model-only child.

The host owns parent-dependent mechanics:

- register a tool returned by its delegation interface in core-owned composition provenance; this internal role is not a `ToolSpec` field and survives generic plugin normalization;
- build one child harness from parent defaults plus explicit child overrides;
- obtain the active parent run's frozen direct-tool snapshot from core-owned tool runtime state, never from the live `Harness.tools` list;
- rebind child-inheritable plugins in parent plugin order;
- append explicit child plugins and direct tools in caller order;
- infer and own an override model, or borrow the parent model;
- project same-provider credentials and the existing request settings exactly as today;
- derive child tracing options and child hooks;
- connect the child, forward nested stream events, and preserve parent run and tool-call correlation;
- fire existing parent `before_subagent_run` and `after_subagent_run` hooks with the actual parent harness in their contexts;
- close the child after success, failure, or cancellation without hiding the original run error.

A top-level host rejects a request made outside an active parent tool call before model inference, plugin connection, filesystem access, or child construction. Every child `PluginContext` receives a disabled host that always rejects before creating resources. This blocks grandchild creation even when an inherited custom plugin captures and calls its child context host.

`SubagentsPlugin.bind()` captures only `context.child_harnesses` in its static tool handler and registers that returned tool through the host's delegation interface. Binding stays synchronous and I/O-free. Child model inference, plugin connection, filesystem access, and child execution happen only when the tool runs.

Add contract coverage for all in-repo direct `PluginContext(...)` constructions in `tests/unit/test_plugins.py`, `tests/unit/test_parallel_llm.py`, and `tests/unit/test_skills.py`. Also prove that one plugin object bound to two parent harnesses receives independent child hosts and cannot cross parent roots, models, tools, hooks, metadata, or streams.

## Child plugin inheritance

Add and publicly export a runtime-checkable structural protocol without changing the base `Plugin.bind()` interface:

```python
@runtime_checkable
class ChildInheritablePlugin(Protocol):
    def for_child(self) -> Plugin:
        """Return the plugin object to bind to one inherited child."""
```

Rules:

- `for_child()` is synchronous and performs no file, provider, or network I/O.
- `FilesystemPlugin.for_child()`, `SkillsPlugin.for_child()`, and `ParallelLlmPlugin.for_child()` return `self`.
- Make these three plugins use one private immutable constructor snapshot for every bind. Copy mutable inputs on construction, expose no mutable configuration container used by binding, and reject configuration assignment/deletion after construction. Mutation of an original input or a value obtained from a public property cannot change later parent or child bindings. Make `FilesystemPlugin.name` runtime-fixed like the other built-in plugin names so child rebinding cannot change identity.
- Returning `self` therefore reuses the same frozen configuration while normal child binding creates independent `ToolSpec` values and instructions.
- `SkillsPlugin` shares its constructor-time registry and frozen catalog while live skill files remain live.
- `ParallelLlmPlugin(model=None)` sees the child `PluginContext.model`; explicit plugin models keep their existing ownership rules.
- `MCPPlugin` has no `for_child()`. A child may still list the same MCP plugin object explicitly; normal binding and reference-counted server lifecycle apply.
- `SubagentsPlugin` has no `for_child()` and is also rejected if listed explicitly in a child's plugins.
- A custom plugin opts in only by implementing `for_child()`. It may return itself or a fresh configured plugin. Validate the returned object and fixed name before child construction. The custom plugin owns its snapshot and thread-safety contract.
- A child-inheritable plugin must not contribute approval-required tools. Reject known static violations while the parent harness is constructed; reject connected violations atomically when the child connects. Do not silently filter plugin tools or leave their instructions/hooks behind.
- Preserve parent plugin order among inherited plugins. Append explicit child plugins after inherited plugins.
- Duplicate names across inherited and explicit child plugins fail. Do not add implicit replacement, exclusion lists, clone fallbacks, or plugin-name special cases. `inherit_parent=True` cannot mean “inherit some and override one”; callers use `inherit_parent=False` and list the wanted plugins/tools explicitly.
- `SubagentsPlugin.bind()` registers each static child recipe with the child host. After all parent plugins bind, the host validates named-child plugin/tool collisions and known approval violations before `Harness(...)` returns. Revalidate direct-tool collisions after `Harness.add_tool()` and connected composition where the relevant names become known. Only genuinely dynamic child connection collisions may surface when the child connects.

## Direct tool inheritance and run freezing

`ToolOrigin` is caller-visible attribution metadata, not authoritative ownership. A direct caller may forge any origin value. Track tool ownership in a separate core-owned composition record that never comes from `ToolSpec.origin`.

Track direct tools and plugin tools as explicit internal composition sources:

- tools passed through `Harness(tools=...)` are direct;
- `Harness.add_tool()` adds a direct tool;
- tools contributed by plugins are not direct;
- freeze the eligible direct-tool list and authoritative plugin ownership with the run toolset;
- place that frozen composition snapshot in the existing core-owned tool runtime context before any tool handler starts, so the child host reads the active snapshot without inspecting live harness state;
- a direct tool added during a parent run does not enter a child delegated during that run, but it is available to children in the next run;
- a valid child-host request outside an active tool runtime fails clearly instead of falling back to current direct tools;
- inherited direct tools preserve parent order and object identity;
- exclude approval-required direct tools;
- append explicit child tools after inherited direct tools;
- reject duplicate tool names through normal child composition;
- do not filter by the string name `subagent`; the actual delegation tool comes from a non-inheritable plugin and is never in the direct source.

The default child always uses `inherit_parent=True`. A named child uses its `inherit_parent` value. A named child with `inherit_parent=False`, no plugins, and no tools receives no model-callable tools and remains valid.

## Plugin contribution and hook-filter names

Core currently imports subagent configuration only to validate `Hook.agents` filters. Replace that feature-specific path with static plugin binding metadata.

Extend `PluginBinding` with an immutable tuple of valid agent names, empty by default. `SubagentsPlugin` contributes `("default", *named_agent_names)` when it binds.

Core requirements:

- combine static agent names from every plugin binding before validating caller and plugin hooks;
- reject blank names and duplicate names within or across bindings;
- validate `Hook.agents` at harness construction, after the existing `Harness.add_tool()` revalidation point, and after connected contributions;
- do not add a public `Harness.add_hook()` method in this slice;
- connected plugins cannot change agent names;
- a harness without `SubagentsPlugin` rejects any agent-filtered hook because no agent names exist;
- dynamic agent catalogs are out of scope.

Child lifecycle hooks from `SubAgentConfig.hooks` or `default_hooks` belong to the child harness. A supplied `HookRegistry` keeps its own `strict_hooks` value; a plain hook sequence uses the parent harness `strict_hooks` setting, matching current child behavior. Copy caller-owned hook registries before child composition. Parent subagent hooks remain in the parent registry and continue to support `agents=[...]` filters.

## SubagentsPlugin behavior

Preserve these model-visible and runtime behaviors:

- tool arguments remain `task: str` and optional non-empty `agent: str`;
- omitting `agent` selects `"default"`;
- an unknown name returns a failed result with the requested name, sorted available names, and `UnknownSubAgent`;
- the description lists named agents in caller order and tells the model how to select the default;
- each child starts a fresh provider session with its own system prompt and no parent transcript;
- parent and child limits remain independent;
- the parent counts one `subagent` tool call; child requests and tokens stay in child usage and result metadata;
- override models inherit current timeout, retry, backoff, temperature, token, effort, and extra-body settings; only same-provider overrides receive parent API key and base URL;
- parent-model children borrow the model and never close it; override models are child-owned and close once;
- child structured output becomes the tool content and reports `structured_output=True`;
- successful metadata keeps agent, inheritance mode, effective tools, model requests, and structured-output status;
- cancellation and strict-hook failures do not hang sibling tool calls;
- child close runs after success, provider failure, hook failure, stream cancellation, or parent cancellation;
- child events remain flattened only when `StreamOptions.include_subagents=True`;
- conversation id and parent call id propagation remain unchanged;
- before-hook metadata mutation does not alter child metadata;
- plugin state, child configuration, child host, and models never enter resume or approval state.

The generic plugin normalizer assigns `ToolOrigin(plugin="subagents", source="subagent")`.

## Remove core built-ins and special tool state

Remove from `HarnessConfig`:

- `builtin_tools`;
- `subagents`.

Add explicit removed-field guards in `thinharness/_migration.py` that point both names to plugin composition and name `SubagentsPlugin` for delegation. Delete `_select_builtin_tools()` and every built-in migration branch from core. A plain harness now has no implicit or selected built-in tool path. The changelog must state that old filesystem, skills, and parallel values inside `builtin_tools` now move to their respective plugins rather than treating the generic removed-field message as feature-specific guidance.

Remove from `Harness`:

- `subagent_hooks=`;
- `self.subagent_hooks`;
- direct imports of subagent configuration or tool builders;
- subagent-specific hook-filter lookup;
- the `subagent` reserved-name check.

Keep an internal child-harness marker only where core needs it to enforce no approval-required tools inside children, install the disabled child host, and suppress top-level-only local tracing. Rename the marker and the approval error to say `child harnesses` rather than `subagents`.

Remove from tool contracts:

- `ToolKind`;
- `ToolSpec.kind`;
- runtime kind validation.

All `ToolSpec` values then use one ordinary contract. Update direct construction, tests, exports, documentation, and changelog together.

Delete the public low-level composition helpers `create_subagent_tool()` and `build_child_harness()` and remove their top-level exports. Do not leave wrappers or aliases. Move retained public types (`SubAgentConfig`, `SubAgentArgs`, and `DEFAULT_SUBAGENT_NAME`) behind the plugin module. Export them and `SubagentsPlugin` from both `thinharness/plugins/__init__.py` and `thinharness/__init__.py`. Export `ChildInheritablePlugin`, `ChildHarnessHost`, and the child request/outcome types through the same public plugin-contract surface.

Delete `thinharness/subagents.py` when its retained implementation has moved behind `thinharness/plugins/subagents.py` and the narrow child host module. Do not keep a pass-through module.

## Tracing and transcript classification

Keep current public event names and result trace attributes for real delegation. Detect delegation from the core-owned frozen composition source, not from the tool name or caller-forgeable `ToolOrigin`:

- when a tool span starts, look up the frozen authoritative composition role for that exact tool and set `subagent.delegation=true` before any hook or handler can cancel or fail;
- only a tool registered through the child host's delegation interface receives that marker; ordinary plugin origin metadata cannot create the role;
- `ToolOrigin` remains attribution metadata and never decides delegation control flow or transcript kind;
- a custom direct tool named `subagent`, including one with `ToolOrigin(plugin="subagents")`, receives ordinary tool tracing;
- keep `subagent.name`, `subagent.tool_mode`, and `subagent.tools` when a real delegation outcome supplies them;
- a cancelled or early-failing real delegation still has the marker even when result metadata is absent;
- keep child `invoke_agent subagent.<name>` spans and agent descriptions;
- update `scripts/build_transcripts.py` to classify a span as a subagent only when `subagent.delegation` is true;
- do not add a legacy tool-name fallback for stored traces; update the checked-in web-research trace's known real delegation spans with the truthful marker, or regenerate that trace with the new implementation, before rebuilding transcript HTML;
- retain rendered transcript output for the checked-in real subagent trace;
- add regressions for a normal delegation, a before-tool-cancelled delegation, an early failure, and a forged-origin direct tool.

`subagent.delegation` is the only new trace attribute. Do not rename existing public streaming events, hook events, trace attributes, stop reasons, or result metadata.

## Behavior contract changes before implementation

After plan review and before code changes, update only affected sections of `docs/behavior.md`:

- extend PLUGIN-3 with the narrow child-harness host while keeping binding synchronous and I/O-free;
- extend the plugin binding contract with static agent names, authoritative core composition provenance, and explicit `for_child()` inheritance;
- update PLUGIN-8 for the `SubagentsPlugin` tool and child plugin ordering;
- update FILESYSTEM-PLUGIN requirements for a fixed plugin name and frozen constructor configuration;
- replace SKILLS-PLUGIN-7 with generic safe-plugin rebinding and shared registry behavior, and state that inheritable plugin configuration is frozen;
- update PARALLEL-LLM-PLUGIN-2 so an inherited borrowed-model plugin uses the child model and frozen plugin configuration;
- update PARALLEL-LLM-PLUGIN-7 to remove the obsolete ordinary `"user"` tool-kind statement;
- update PROVIDER-RETRY-6 for child model override and rebound parallel model settings;
- replace MCP-7's temporary bridge with explicit child `MCPPlugin` composition;
- update TOOLSET-FREEZE-1 through TOOLSET-FREEZE-3 so the same run snapshot carries direct-tool ownership used by delegated children;
- add a Subagents Plugin section covering explicit composition, the default child, named configs, additive inheritance, safe plugin rebinding, direct-tool freezing, child hooks, disabled child hosts, no recursion, model ownership, limits, streaming, tracing, results, cancellation, and state exclusion;
- remove every statement that calls subagent a built-in tool or documents `builtin_tools`.

Do not change unrelated behavior sections.

## Architecture guards

Extend `tests/unit/test_architecture.py` with AST import and named-identifier checks rather than a blanket `"subagents" not in source` substring assertion. Direct `thinharness/core.py` source rejects:

- imports of `subagents` or `plugins.subagents`;
- `SubAgentConfig`, `SubagentsPlugin`, `SubAgentArgs`, and `DEFAULT_SUBAGENT_NAME`;
- `create_subagent_tool`, `build_child_harness`, and `_select_builtin_tools`;
- `builtin_tools`, `subagent_hooks`, and direct subagent name reservation;
- filesystem, skills, parallel-LLM, or MCP inheritance logic.

Add structural checks that:

- `thinharness/subagents.py` no longer exists;
- core constructs only the narrow child host and does not expose the parent harness through `PluginContext`;
- `SubagentsPlugin` owns the delegation tool and configuration;
- only plugins with `for_child()` enter automatic child plugin composition;
- no name-based delegation detection remains in core or tool execution.

## Tests

Retain and migrate the existing subagent behavior suite. Add focused coverage for:

### Plugin construction and composition

- `SubagentsPlugin()` contributes one static `subagent` tool and the default agent name;
- no plugin means no delegation tool;
- fixed plugin name, origin, description, schema, order, and normal collisions;
- ordered agent validation, duplicate names, reserved `default`, invalid descriptions, and plugin reuse across harnesses with different roots and models;
- duplicate agent names within or across binding metadata fail;
- a named child with no tools is valid;
- a child cannot list `SubagentsPlugin` explicitly, and the error comes from plugin-module validation rather than core;
- plugin bind is I/O-free and does not infer a child model;
- all direct third-party-style `PluginContext(...)` constructions receive a fake child host cleanly;
- a top-level child host rejects outside an active tool runtime, and a child disabled host rejects an inherited custom plugin's grandchild attempt before resources open;
- plugin state and host references do not serialize into resume or approval state.

### Inheritance

- default and named `inherit_parent=True` children rebind filesystem, skills, and parallel plugins in parent order;
- skills reuse the exact registry and summary once;
- `ParallelLlmPlugin(model=None)` uses a named child's cross-provider override model without inheriting the parent API key or base URL;
- parallel plugins with explicit model objects or strings keep those models and ownership rules;
- MCP does not inherit, while an explicit child MCP plugin connects and closes through its own binding;
- a custom plugin with `for_child()` inherits; one without it does not;
- an invalid `for_child()` return fails clearly;
- mutation attempts and mutable constructor inputs cannot change later filesystem, skills, or parallel child bindings;
- inherited and explicit plugins/tools are additive and preserve their stated order;
- a known inherited/explicit filesystem or direct-tool collision fails while the parent is constructed, and later `add_tool()` collisions revalidate before a run;
- dynamic child connection collisions fail atomically;
- `inherit_parent=True` plus explicit sources reports `tool_mode="inherited+explicit"` in hooks, results, and traces;
- plugin tools are not copied again as direct tools;
- direct tools with caller-supplied or forged origins still inherit and trace as direct tools;
- approval-required direct and explicit tools never enter children;
- an inheritable plugin with a known approval-required tool fails parent construction rather than being silently filtered;
- a direct tool added during a run appears only in children of the next run.

### Hooks and execution

- named hooks come from `SubAgentConfig.hooks`; default hooks come from `default_hooks`;
- child hooks with subagent events or any `agents=` filter fail at plugin construction;
- caller-owned `HookRegistry` values are copied; registry strictness is preserved, while plain sequences use parent strictness;
- parent before/after hooks, cancellation, metadata isolation, and agent filters remain stable;
- a default child works with `default_hooks` and parent hooks filtered to `agents=["default"]`;
- agent-filtered parent hooks fail without the plugin and for unknown names;
- default, unknown, blank, model-only, structured-output, provider-failure, and child-close paths;
- shared parent model versus owned override model lifecycle;
- two concurrent delegations to one override-model config create two owned child providers and close each once;
- same-provider credential forwarding and cross-provider credential isolation;
- fresh budgets, retries, notices, metadata, and usage accounting;
- concurrent delegation and strict sibling abort do not hang;
- nested streaming correlation and `include_subagents` behavior;
- tracing attributes and child spans remain stable;
- `subagent.delegation=true` exists from span start for normal, cancelled, and early-failing real delegation;
- a direct tool named `subagent` with a forged `ToolOrigin(plugin="subagents")` remains an ordinary trace and transcript event.

### Removal

- each removed `HarnessConfig` field raises its own `SubagentsPlugin` migration error;
- removed `SubAgentConfig` fields fail by name;
- `Harness(subagent_hooks=...)`, `create_subagent_tool`, and `build_child_harness` are gone;
- `ToolSpec` has no `kind` field and no `ToolKind` remains;
- a direct custom tool named `subagent`, even with forged subagents origin metadata, works without the plugin and has ordinary tracing;
- the same custom tool collides normally when the plugin is present;
- core has no feature-specific subagent composition.

## Caller, documentation, and site migration

Update every checked-in caller. This includes:

- `README.md` feature text and examples;
- `docs/docs.md` configuration, hooks, subagents, MCP, tracing, and plugin examples, including the tool-surface sentence, rejected Bash example, and SkillsPlugin child wording;
- `docs/site/explainer/index.html` architecture and composition text;
- `docs/site/about/index.html` through `scripts/build_site.py`;
- the checked-in web-research trace marker and `docs/site/examples/index.html` through transcript regeneration;
- `CHANGELOG.md` with all breaking removals and inheritance changes;
- `examples/web_research_report/agent.py`, including required-tool and trace-name assertions;
- `examples/mcp_plugin.py`;
- `tests/e2e/langfuse_tracing_journey.py`;
- `tests/unit/test_harness.py` built-in migration tests and the removed add-after-construction delegation capability;
- every test and journey that passes `builtin_tools=[]` or `builtin_tools=["subagent"]`;
- every import of `create_subagent_tool()` or `build_child_harness()`;
- every direct `PluginContext(...)` constructor affected by the required host field.

Delete `builtin_tools=[]` rather than replacing it: no built-in tools remain. Replace enabled delegation with `plugins=[..., SubagentsPlugin(...)]` in caller order.

Add `tests/e2e/subagents_journey.py` with real provider calls. It must prove that the plugin delegates to the default child and to a named override-model child, returns child output to the parent, prevents child delegation, and emits the expected metadata without relying only on mocks.

## Implementation steps

1. Update the affected behavior contracts.
2. Add the narrow child-harness host and its request/outcome types without moving feature configuration into core.
3. Extend `PluginContext` and `PluginBinding` with the child host and static agent names.
4. Add `ChildInheritablePlugin` and implement `for_child()` on filesystem, skills, and parallel plugins.
5. Add `SubagentsPlugin`, move the retained public configuration types behind it, and implement static tool binding.
6. Move child construction, execution, hooks, streaming, tracing, and cleanup behind the narrow host.
7. Implement explicit direct-tool source tracking and run-frozen child inheritance.
8. Implement additive inherited and explicit child composition with normal collision errors.
9. Remove MCP and skills/filesystem temporary bridges and the parent-bound parallel handler path.
10. Migrate child hooks to `SubAgentConfig.hooks` and `default_hooks`; replace core agent-filter lookup with binding metadata.
11. Remove built-in selection, old config fields, constructor arguments, public helpers, `ToolKind`, `ToolSpec.kind`, and reserved-name logic.
12. Change tracing and transcript classification to use core-owned composition provenance and the start-of-span delegation marker.
13. Migrate unit tests, examples, documentation, site pages, and all end-to-end journeys.
14. Delete the obsolete `thinharness/subagents.py` module and add architecture guards.
15. Regenerate checked-in generated pages and run all validation.

## Validation

Run:

```bash
uv run pytest tests/unit/test_subagents.py tests/unit/test_plugins.py tests/unit/test_harness.py tests/unit/test_hooks.py tests/unit/test_architecture.py
uv run pytest tests/unit/test_streaming.py tests/unit/test_tracing.py tests/unit/test_approvals.py tests/unit/test_structured_output.py tests/unit/test_tool_retry.py tests/unit/test_mcp.py tests/unit/test_parallel_llm.py
uv run pytest tests/unit/test_web_research_report_example.py
uv run pytest
uv run ruff check .
uv run pyright
uv run scripts/build_site.py
uv run scripts/build_site.py --check
uv run scripts/build_transcripts.py
for journey in tests/e2e/*_journey.py; do uv run --env-file .env python "$journey"; done
git diff --check
```

Keep the migrated suite at `tests/unit/test_subagents.py`; do not rename it. Report every journey separately. Credential- or service-based skips are not passes.

## Success criteria

- ThinHarness has no built-in tool selector or implicit feature tool path.
- Delegation exists only through explicit `SubagentsPlugin` composition.
- The plugin owns child definitions, the default child, tool description, inheritance choices, child hooks, and result shaping.
- Core does not import subagent configuration or plugin implementation and exposes only a narrow child-harness host.
- Children receive a disabled child host and cannot create grandchildren through built-in or custom inherited plugins, but a custom ordinary tool may use the same model-facing name.
- Safe parent plugins rebind against the child; unknown custom plugins do not inherit without `for_child()`.
- Inherited parallel LLM batches borrow the child model when configured with `model=None`.
- Parent MCP never inherits implicitly; explicit child MCP keeps independent binding and cleanup.
- Additive inherited and explicit composition is ordered and collision-safe.
- Child inheritance uses the active parent run's frozen direct tools.
- Child hooks are local to each named config or the plugin default.
- Streaming, tracing, hooks, usage, limits, model ownership, structured output, errors, and cancellation retain their current behavior except for the approved child-model parallel rebinding and run-frozen direct-tool inheritance changes.
- Core, docs, examples, tests, and site pages contain no obsolete subagent built-in configuration.
- Focused suites, the full suite, Ruff, Pyright, generated-page checks, and every live end-to-end journey pass.

## Out of scope

- Nested or recursive delegation.
- Forking the parent conversation into a child.
- Background children, detached jobs, scheduling, or persistent child sessions.
- Child approval pauses.
- Dynamic agent catalogs or plugin discovery.
- Automatic inheritance of MCP or unknown custom plugins.
- Selective inherited-plugin exclusion, implicit override by name, or dependency resolution.
- Changing child result shape, stream event names, trace attribute names, hook event names, provider retry policy, or structured-output semantics.
- Compatibility aliases, deprecation periods, config migrations, or fallback reads.

## Review record

One Codex, Claude, and GLM panel round reviewed plan v1. Plan v2 applies the verified findings on disabled child hosts, authoritative tool provenance, start-of-span delegation marking, immutable inheritable plugin configuration, approval-tool policy, eager collision validation, child-hook validation, child request fields, tool-mode values, public host types, direct `PluginContext` callers, run-frozen tool plumbing, behavior-contract coverage, AST-scoped architecture guards, transcript regressions, and caller migration. No second plan-review round is scheduled.
