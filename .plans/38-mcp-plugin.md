# MCP plugin — plan v2

Migrate MCP after `.plans/37-plugin-system-and-filesystem.md` lands and its implementation review is complete. This is the second implementation slice and the first connected production plugin. No release is cut between the two slices.

## Approved decisions

1. **One MCP plugin per harness.** Plugin names are unique, `MCPPlugin.name` is fixed to `"mcp"`, and one plugin groups every server for that harness.
2. **Keep server adapters.** `MCPServer`, `MCPServerStdio`, `MCPServerSSE`, and `MCPServerStreamableHTTP` remain transport and tool-conversion adapters. `MCPPlugin` owns harness composition and lifecycle.
3. **One discovered snapshot per binding.** MCP tool changes after connection are ignored until a new harness binding is created. Do not implement `notifications/tools/list_changed`.
4. **Subagent MCP settings stay temporarily.** Existing `SubAgentConfig.inherit_mcp_servers` and `mcp_servers` remain until the subagent plugin plan. The bridge moves out of core and builds child `MCPPlugin` values explicitly.
5. **Connection precedes `run_start`.** MCP setup failure occurs before a run starts and does not fire run lifecycle hooks.

This plan removes `HarnessConfig.mcp_servers`. Do not support both configuration paths.

## Goal

After this plan:

```python
harness = Harness(
    HarnessConfig(root="."),
    model=model,
    plugins=[
        MCPPlugin(servers=[
            MCPServerStdio("python", ["server.py"], tool_prefix="docs"),
        ]),
    ],
)
```

`Harness.connect()` is fully generic. Core does not import MCP classes, resolve MCP server ids, discover MCP tools, or own an MCP-specific exit stack.

## Plugin behavior

Add `thinharness/plugins/mcp.py` with `MCPPlugin`.

- `bind()` resolves and validates configuration without I/O.
- The binding has no static tools.
- Its connector owns a private `AsyncExitStack`, enters servers in caller order, discovers one tool snapshot from each server, and returns one dynamic `PluginContribution`.
- The connector keeps entered servers open until the harness closes.
- Generic harness composition validates the complete discovered set before committing any tool.
- Because an async context manager whose `__aenter__` fails is not owned by the harness stack, the MCP connector itself catches `BaseException`, closes its private stack in reverse order, and re-raises. This covers connection, discovery, schema, collision, and cancellation failures.
- Repeated `connect()` and repeated runs reuse the same binding and discovered tools.
- `Harness.aclose()` closes the plugin once. Reusing the same `MCPServer` wrapper across parent and child bindings keeps the current FastMCP reference-counted session sharing.

Keep server filtering, prefixing, schema cleanup, error conversion, timeout behavior, optional dependency behavior, and transport ownership inside the existing MCP implementation module.

## Generic tool origin

Plan 37 adds `ToolOrigin`. Use it for MCP tools:

```python
ToolOrigin(
    plugin="mcp",
    source=resolved_server_id,
    attributes={"tool_name": original_tool_name},
)
```

Remove the MCP-specific fields from the core tool contract:

- remove `McpToolInfo`;
- remove `ToolSpec.mcp`;
- remove `"mcp"` from `ToolKind` after all checks use `ToolOrigin`.

Keep `ToolKind` only for remaining framework control behavior such as the reserved subagent tool. Do not turn every origin into a new `ToolKind` value.

`MCPPlugin.name` is fixed to `"mcp"`, so tracing derives `mcp.server.id` and `mcp.tool.name` from `ToolOrigin`, and subagent parent-tool inheritance excludes `origin.plugin == "mcp"` unless MCP inheritance is explicit. Tool result metadata remains unchanged because it is model-visible behavior.

## Server identity

Move duplicate server-id resolution from `Harness` into the MCP plugin binding. Preserve the current deterministic base ids and `-2`, `-3` suffix behavior.

Server identity must be binding-local. Remove `MCPServer._resolved_id` and `resolve_id()` rather than leaving mutable identity on a shared wrapper. A private bound-server adapter carries `(server, resolved_id)` and builds handlers that pass the resolved id into result normalization. Both `ToolOrigin.source` and model-visible `ToolResult.metadata["mcp_server_id"]` use that bound id. Keep the public server wrapper responsible for connection and raw calls.

Add tests for:

- duplicate ids inside the one plugin;
- one server object reused by two harnesses with different collision neighbors;
- stable trace and result metadata after both harnesses connect;
- the public base id before binding.

## Subagent bridge

The subagent module is not core, so it can keep a temporary MCP-specific bridge until `SubagentsPlugin` exists.

- Explicit `SubAgentConfig.mcp_servers` values become a child `MCPPlugin`.
- `inherit_mcp_servers=True` finds the fixed-name MCP plugin in the read-only `parent.plugins` tuple, copies its server adapters by identity, then adds explicit child servers without duplicate objects.
- Default parent-tool inheritance still excludes resolved MCP `ToolSpec` values. The child must connect its own MCP plugin so handlers and lifecycle are valid.
- Remove reads of `parent._mcp_servers` and `child.config.mcp_servers`; do not add a replacement MCP field to core.
- Keep current sharing, override, union, failure rollback, tracing, and cleanup behavior.
- Record the bridge for deletion in the later subagent plugin plan; do not add a generic plugin inheritance flag now.

## Implementation steps

1. Add `MCPPlugin` and public exports.
2. Remove mutable resolved ids from server wrappers and move resolution plus result attribution into binding-local MCP state.
3. Move server entry and discovery from `Harness._ensure_mcp_connected()` into the plugin connector, with its own failed-entry cleanup.
4. Delete `_mcp_servers`, `_mcp_stack`, `_mcp_connected`, `_resolve_mcp_server_ids()`, and `_ensure_mcp_connected()` from core.
5. Make `Harness.connect()` delegate only to generic plugin connection management from plan 37.
6. Remove `HarnessConfig.mcp_servers` and migrate every caller to `plugins=[MCPPlugin(...)]`.
7. Replace `McpToolInfo` with `ToolOrigin` in MCP conversion, tracing, tests, and subagent inheritance.
8. Implement the temporary subagent bridge without adding MCP knowledge back to core.
9. Add an architecture test that fails if `thinharness/core.py` directly imports MCP modules or contains MCP lifecycle logic. Transitive MCP reachability through the temporary subagent bridge remains until `SubagentsPlugin` migrates.
10. Update README, `docs/docs.md`, end-to-end journeys, examples, exports, changelog, and MCP-1/MCP-5 in `docs/behavior.md`.

## Behavior contract changes before implementation

After this plan review and before code changes, update the MCP section in `docs/behavior.md`:

- replace `HarnessConfig.mcp_servers` with explicit `MCPPlugin` composition;
- state lazy generic plugin connection and one discovered snapshot per binding;
- state binding-local server ids and origin metadata;
- preserve filter, prefix, schema, result, error, timeout, optional dependency, session sharing, cancellation, and bounded cleanup rules;
- preserve explicit subagent inheritance while naming the temporary bridge;
- update MCP-1 server-id ownership and MCP-5 from `kind="mcp"`/`McpToolInfo` to `ToolOrigin`;
- leave the run toolset freeze wording owned by plan 37.

Edit only affected sections.

## Tests

Retain or migrate all existing MCP behavior tests. Add focused coverage for:

- MCP tools absent before connect and installed atomically after connect;
- explicit `connect()` and first-run lazy connection;
- repeated runs reuse one discovered snapshot;
- a second fixed-name `MCPPlugin` is rejected during harness construction;
- collisions with static plugin tools, direct tools, structured-output tools, and tools from another MCP server;
- failed second-server connection and failed discovery close the first server and allow retry;
- cancellation during connect and close;
- no dynamic contribution after failed connection;
- reverse close order across MCP and other connected plugins;
- binding-local server ids and one server wrapper shared across harnesses;
- tracing attribution after an after-tool hook rewrites output;
- resume and approval flows do not serialize MCP connection details;
- subagent no-inherit, explicit, inherit, union, shared-session, and failure cases, including a child whose second server fails while the parent keeps a shared server live;
- base import and wrapper construction without MCP dependencies.

Run:

```bash
uv run pytest tests/unit/test_mcp.py tests/unit/test_mcp_optional_dependency.py
uv run pytest tests/unit/test_subagents.py tests/unit/test_tracing.py tests/unit/test_resume.py tests/unit/test_approvals.py
uv run pytest
uv run ruff check .
uv run pyright
```

Run the deterministic MCP end-to-end journey. Report credential-based skips separately and do not count them as passes.

## Success criteria

- MCP is enabled only through `MCPPlugin`.
- Core contains no direct MCP imports, state, lifecycle, identity, or discovery logic; the temporary transitive dependency is isolated in the subagent module.
- Generic plugin rollback and cleanup preserve every current MCP lifecycle guarantee.
- `ToolSpec` contains generic origin data instead of MCP-specific fields.
- Parent and child harnesses retain explicit MCP inheritance and safe shared-session behavior.
- Existing MCP transport and result semantics do not change.
- The focused suite, full suite, Ruff, Pyright, and deterministic MCP journey pass.

## Out of scope

- Replacing FastMCP or changing its pinned version.
- MCP prompts, resources, sampling, elicitation, OAuth, tasks, server instructions, or provider-native MCP.
- Dynamic MCP tool-list updates.
- Generic plugin inheritance or child-harness factories.
- Subagents as a plugin.
- Package distribution splitting.

## Review record

One Codex, Claude, and GLM panel round reviewed plan v1 together with the plugin-system plan. Plan v2 applies the verified findings on unique plugin identity, failed connector entry, binding-local server ids, result metadata, subagent access, architecture-test scope, behavior-document ownership, and combined failure tests. The approved product decisions are recorded above. No second plan-review round will run.
