# Plugin architecture deepening — plan v2

Tighten the 0.7 plugin architecture in three places: share the repeated built-in plugin rules, make the child host own its agent catalog, and replace the private dictionary-shaped tool runtime context with a typed runtime scope.

This is one architecture change. The three parts simplify related plugin and child-run interfaces without changing model-visible tool behavior.

## Decisions

1. **Keep the public plugin contract structural.** Custom plugins continue to satisfy the `Plugin` protocol without inheriting from a ThinHarness base class.
2. **Share only built-in policy.** Add private support for fixed built-in names and frozen built-in configuration. Do not expose this support as a public extension interface.
3. **Keep MCP configuration mutable.** `MCPPlugin` shares the fixed-name rule but does not gain the freeze rule. `FilesystemPlugin`, `SkillsPlugin`, `ParallelLlmPlugin`, `SubagentsPlugin`, and `BashPlugin` keep their current frozen configuration behavior.
4. **The parent child host owns the agent catalog.** Delegation recipes already contain their agent names. `PluginBinding` no longer repeats those names in a separate field. The catalog and sealing controls stay private to the core host; the public `ChildHarnessHost` protocol remains unchanged.
5. **Delegation registration is static and atomic.** A plugin registers delegation tools and recipes during synchronous `bind()`. Core seals registration after all bindings and before reading the catalog. A failed or late registration changes no provenance, recipes, or names.
6. **Agent names identify hook-filter groups, not one tool.** The host records an ordered unique catalog. Repeated names across registrations are valid, which allows two delegation tools to use the same agent identity. Blank names fail with `ValueError`.
7. **Runtime requests must use the sealed catalog.** `run()` rejects an unregistered agent name before hooks or child creation. Other request fields may differ from the registered recipe because the delegation plugin owns call-time request construction.
8. **Bind-time registration covers connected tools.** A tool registered during `bind()` enters the static agent catalog even when the tool is contributed later by `connect()`. Registration from `connect()` or a live tool handler fails because the host is sealed.
9. **The runtime scope stays private and narrow.** It contains only the active lease, copied run metadata, and the active run's tool and composition snapshot maps. It does not become a general capability or policy object.
10. **Preserve runtime identity.** The typed scope keeps references to the same tool and composition maps, and copied async contexts share the same mutable lease. Lease revocation must still block detached tasks after a tool call ends.
11. **Do not add compatibility paths.** This is a pre-1.0 interface cleanup. Do not preserve `PluginBinding.agent_names` as an alias or deprecated field.

## 1. Share built-in plugin policy

Add one private module under `thinharness/plugins/` for built-in plugin identity and freezing rules.

- Use a shared private metaclass that receives and installs each built-in's fixed name when that built-in class is created. It must allow the first built-in declaration while rejecting later class assignment, class deletion, and subclass replacement of the name.
- Use the fixed-name implementation for `MCPPlugin`, `FilesystemPlugin`, `SkillsPlugin`, `ParallelLlmPlugin`, `SubagentsPlugin`, and `BashPlugin`.
- Add one private frozen-plugin base for the five plugins that are already frozen. Its freeze checks must use direct object state rather than `getattr()`, so `ParallelLlmPlugin.__getattr__` cannot intercept the check.
- Preserve each plugin's constructor validation, copied configuration, properties, `for_child()` behavior, binding behavior, fixed-name error detail, and model-visible tools.
- Make instance assignment and deletion of `.name` raise the fixed-name error for all six built-ins. This standardizes MCP instance deletion with its existing assignment rule.
- Preserve `ParallelLlmPlugin.__getattr__` behavior.
- Keep `MCPPlugin` mutable for every attribute except `name`.
- Keep the public `Plugin` protocol unchanged.

Use one parameterized fixed-name contract test for all six built-ins: instance assignment, instance deletion, class assignment, class deletion, and subclass name replacement fail. Use a second parameterized test for post-construction assignment and deletion on the five frozen plugins. Prove separately that non-name MCP assignment and deletion remain possible and that parallel-LLM public values still return detached data.

## 2. Move agent names into the parent child host

Make `_ParentChildHarnessHost` the one source of truth for statically registered child recipes and their agent names without widening `ChildHarnessHost`.

- Keep the ordered catalog accessor and sealing operation private to `_ParentChildHarnessHost`.
- `register_delegation_tool()` first validates the tool, ordered recipe sequence, recipe values, and every non-empty agent name. It commits delegation provenance, recipes, and new catalog names only after all validation succeeds.
- Preserve registration order and append each agent name only once. Repeated names within or across registrations remain one catalog entry.
- Seal the parent host immediately after every configured plugin has returned a valid binding. Seal before contribution normalization, hook-filter validation, connection, or any run.
- A registration attempt after sealing raises `HarnessError` and changes no host state.
- `_DisabledChildHarnessHost` stays stateless and keeps rejecting registration and execution with `HarnessError`.
- Remove `agent_names` from `PluginBinding`.
- Remove core's binding-level agent-name collection and validation. Core validates `Hook.agents` against the sealed parent-host catalog after static plugin composition and after connected hook composition.
- `SubagentsPlugin.bind()` registers its existing default and named recipes but no longer returns a duplicate agent-name tuple.
- `_ParentChildHarnessHost.run()` keeps the active lease and registered-delegation-tool guards and also rejects any request whose `agent_name` is absent from the sealed catalog.
- Preserve current default-name handling, named-agent order, hook dispatch, child construction, and connected-contribution atomicity.

Before implementation, update only the affected current-state requirements in `docs/behavior.md`: `PLUGIN-3`, `PLUGIN-11`, and `SUBAGENTS-PLUGIN-4`. The contract must say that synchronous child-host registration supplies the static catalog, bind-time registration covers a tool contributed through `connect()`, registration is sealed before connection, and connected contributions cannot change the catalog. Do not add migration wording.

Tests must cover:

- catalog order within one plugin and across two delegation plugins;
- repeated names across alias tools without duplicate catalog entries;
- blank-name and invalid-recipe failures that leave provenance, recipes, and the catalog unchanged;
- valid default, named, custom, and connect-contributed agent filters;
- unknown hook filters;
- registration rejection from a connector and from a live tool handler, with no host mutation;
- runtime rejection of an unregistered request name before hooks or child creation;
- disabled-host registration and execution rejection;
- a non-delegation tool calling `host.run()` and receiving the registered-delegation-tool error;
- reuse of one `SubagentsPlugin` across two harnesses, proving sealing is per host.

## 3. Add a typed tool runtime scope

In `thinharness/hooks.py`, replace `_CURRENT_TOOL_RUNTIME`'s `dict[str, Any]` value with one private frozen dataclass. Import `_ToolComposition` only under `TYPE_CHECKING` to avoid the existing `children.py` to `hooks.py` runtime dependency becoming a cycle.

```python
@dataclass(frozen=True)
class _ToolRuntimeScope:
    lease: _ToolRuntimeLease
    run_metadata: Json
    tool_map: dict[str, ToolSpec]
    tool_composition: dict[str, _ToolComposition]
```

- `ToolCallExecutor` creates the scope for each active tool call.
- Child execution reads named fields instead of string keys.
- Remove only defensive dictionary-shape parsing made unnecessary by the private typed producer. Keep lease activity validation and authoritative delegation-composition validation.
- Metadata remains copied when the scope is created.
- Tool and composition fields keep the active run's existing snapshot maps by reference; the frozen dataclass does not claim to make those dictionaries deeply immutable.
- The lease remains one shared mutable object and is revoked on every tool-call exit path.
- Keep `current_tool_runtime_context()` internal and do not export the scope type from the package.

Tests must prove:

- the runtime context is absent outside a tool call;
- metadata is copied while tool-map and composition-map identity is preserved inside a tool call;
- copied async contexts receive the same lease;
- lease revocation blocks later child execution after normal completion, handler exception, and cancellation;
- the registered-delegation guard remains effective.

## Files expected to change

- `docs/behavior.md`
- `thinharness/plugins/_builtin.py` or one equivalent private module
- `thinharness/plugins/{bash,filesystem,mcp,parallel_llm,skills,subagents}.py`
- `thinharness/plugins/base.py`
- `thinharness/children.py`
- `thinharness/core.py`
- `thinharness/hooks.py`
- `thinharness/tool_execution.py`
- focused tests, especially `tests/unit/test_subagents.py`, `test_plugins.py`, `test_mcp.py`, `test_skills.py`, `test_parallel_llm.py`, `test_bash_plugin.py`, `test_hooks.py`, `test_tool_retry.py`, and `test_architecture.py`

Do not change README copy, package exports, model-visible schemas, provider behavior, MCP provenance, tool-composition snapshots, release automation, or version numbers in this change. Do not add a release note for an intermediate plugin interface that has not been released.

## Acceptance checks

- The six built-in plugin names and the existing five frozen plugin configurations keep their intended behavior with one shared implementation.
- `MCPPlugin` remains mutable except for its fixed name.
- `PluginBinding` contains only `static` and `connect`.
- The public `ChildHarnessHost` protocol remains unchanged.
- Agent-filtered hooks use the sealed private host catalog, including custom and connect-contributed delegation tools registered during `bind()`.
- Registration after synchronous binding cannot change agent names, recipes, or delegation provenance.
- Runtime child requests cannot use names outside the sealed catalog or run from an ordinary tool.
- Child delegation still uses the active run's existing tool and composition snapshot maps.
- Copied async contexts share lease revocation and cannot delegate after the parent tool call ends.
- No public compatibility layer or migration documentation is added.
- Focused plugin, subagent, hook, retry, and architecture tests pass.
- `uv run ruff check .` passes.
- `uv run pyright` passes.
- `uv run pytest` passes.
- `git diff --check` passes.
