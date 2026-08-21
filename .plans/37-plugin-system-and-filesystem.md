# Plugin system and filesystem plugin — plan v2

Build the plugin seam and migrate the filesystem tools through it. This is the first implementation slice. It must land and receive implementation review before the MCP plugin starts. No release is cut between this plan and the MCP plugin plan.

## Approved decisions

1. **No implicit filesystem tools.** `Harness(...)` has no tools unless the caller passes `tools=` or `plugins=`. The replacement is `plugins=[FilesystemPlugin()]`.
2. **One canonical root.** `HarnessConfig.root` stays in core as run context. `FilesystemPlugin` uses `PluginContext.root`; it does not accept a second root.
3. **Static contributions are visible before connection.** `FilesystemPlugin` tools appear in `harness.tools` and `tool_schemas()` immediately after construction. Connected plugins can add a second, dynamic contribution during `connect()`.
4. **Direct extension stays simple.** Independent custom tools still use `tools=[ToolSpec(...)]`, and independent hooks still use `hooks=[Hook(...)]`.
5. **Plugin names are unique within one harness.** Names identify contribution origin. `MCPPlugin` therefore groups all servers for a harness.
6. **Plugins connect before `run_start`.** A connected hook applies to the first lazy run. A connection failure happens before a run starts, so `run_start` and `run_end` do not fire for that attempt.
7. **Parallel-LLM policy remains in core temporarily.** Keep `HarnessConfig.read_paths` and `write_paths` until `ParallelLlmPlugin` migrates.

These are breaking pre-1.0 changes. Do not add compatibility aliases or an implicit default plugin.

## Goal

After this plan:

```python
harness = Harness(
    HarnessConfig(root="."),
    model=model,
    plugins=[
        FilesystemPlugin(tools=["read", "write", "edit", "search", "list", "glob", "jsonl_search"]),
    ],
    tools=[custom_tool],
    hooks=[custom_hook],
)
```

The core run loop knows how to compose plugins, but it does not import or construct filesystem tools.

## Public interface

Add `thinharness/plugins/base.py` with these concepts. Exact private helper names can change, but the public shape and lifecycle must stay small.

```python
@dataclass(frozen=True)
class PluginContext:
    root: Path

@dataclass(frozen=True)
class ToolOrigin:
    plugin: str
    source: str | None = None
    attributes: Json = field(default_factory=dict)

@dataclass(frozen=True)
class PluginContribution:
    tools: tuple[ToolSpec, ...] = ()
    instructions: tuple[str, ...] = ()
    hooks: tuple[Hook, ...] = ()

@dataclass(frozen=True)
class PluginBinding:
    static: PluginContribution = PluginContribution()
    connect: Callable[[], AsyncContextManager[PluginContribution]] | None = None

class Plugin(Protocol):
    name: str
    def bind(self, context: PluginContext) -> PluginBinding: ...
```

Rules:

- `bind()` is synchronous and does no file or network I/O.
- One plugin object can bind to more than one harness. Each call returns independent binding state.
- Plugin names are non-empty and unique within one harness. Reject duplicates before binding.
- `PluginBinding.static` is validated and installed during harness construction.
- `connect` is optional. ThinHarness enters it lazily during `Harness.connect()` or before `run_start` on the first run. It can return dynamic tools, instructions, and hooks.
- Concurrent `connect()` calls share one connection attempt and cannot enter a binding twice.
- Dynamic contributions go through the full tool, hook-filter, approval, reserved-name, structured-output, callable-handler, and collision validation path. Commit the complete staged set only after every plugin connects successfully.
- A connection failure catches `BaseException`, closes entered bindings in reverse order, leaves no dynamic contribution installed, and allows a later retry. Cancellation propagates after cleanup.
- `Harness.aclose()` closes plugin bindings in reverse order, then closes an owned model.
- `Harness.plugins` is a read-only tuple of configured plugin objects for temporary feature bridges such as subagents. Core composition uses bindings, not plugin-specific inspection.
- Plugins run in-process and are trusted. There is no isolation, entry-point discovery, hot reload, dependency ordering, or plugin-to-plugin lookup.

Observers are not part of this first interface. Current tracing does not implement a neutral read-only observer seam, and adding an unused interface now would be speculative. A later observability plan can add it with a real adapter.

## Composition order

Use one deterministic order:

1. plugin static contributions in caller plugin order;
2. direct `tools=` and `hooks=` contributions;
3. plugin dynamic contributions in caller plugin order.

System instructions are assembled as:

1. `HarnessConfig.system_prompt`;
2. plugin-level instruction strings in contribution order;
3. the current skill summary while skills remain on the transitional built-in path;
4. `ToolSpec.instructions` in final tool order;
5. structured-output instructions through the existing output path.

Plugin hooks use the existing strict/non-strict behavior and execute in contribution order. Always build a fresh `HookRegistry` from copied caller hooks plus plugin hooks; never mutate a caller-supplied registry. Validate dynamic hook filters before commit. Do not add topological ordering or hook priority.

## Filesystem plugin

Add `thinharness/plugins/filesystem.py` with `FilesystemPlugin`.

- The plugin wraps the existing `FileTools` implementation instead of copying tool logic.
- Its default tool list is `read`, `write`, `edit`, `search`, `list`, and `glob` in that order.
- `jsonl_search` remains opt-in and stays behind this plugin because it shares the root, read policy, ripgrep process, truncation, and spill-output handling.
- `tools=` accepts an ordered sequence, rejects duplicates and unknown names, and preserves caller order. Do not accept a set because set order is not part of the interface.
- The plugin contributes `Workspace root: <resolved root>` as plugin-level instructions. A harness without this plugin does not claim that it has a model-visible workspace.
- The plugin owns the current filesystem settings: `output_dir`, read and write path policies, read and output limits, search line limit, ripgrep timeout, and search exclusion globs.
- Keep `FileTools` as a public low-level deep module for callers that want direct `ToolSpec` values. Remove only the public `builtin_tools()` helper after all in-repo callers move to `FilesystemPlugin`.
- `bind()` must not create the root. Remove eager root creation from plugin composition. Missing-root read, list, glob, search, and JSONL calls return their normal empty or not-found result; write and spill paths create required parents when used.

Keep `HarnessConfig.root`. Remove these filesystem-only fields from `HarnessConfig`:

- `output_dir`
- `max_read_chars`
- `max_read_bytes`
- `max_tool_chars`
- `max_search_line_chars`
- `rg_timeout`
- `search_exclude_globs`

Keep `read_paths` and `write_paths` temporarily as parallel-LLM policy. `FilesystemPlugin` receives its own path-policy settings explicitly. Document this temporary duplication and remove the core fields when `ParallelLlmPlugin` migrates.

`Harness` and `FilesystemPlugin` must not create the root during construction or binding. A harness without the plugin has no filesystem side effect and no workspace instruction.

## Transitional built-ins

`HarnessConfig.builtin_tools` stays temporarily for skills, subagents, and parallel LLM until their own plugin plans land. It no longer accepts filesystem tool names. `builtin_tools=None` selects no transitional built-ins; every remaining built-in is explicit. An old filesystem name fails with an error that points to `FilesystemPlugin`.

Update `SubAgentConfig` and `build_child_harness` only as much as needed to preserve current child filesystem choices:

- add an explicit `plugins` field and count it as a valid tool source in `SubAgentConfig.validate_subagent`;
- pass child plugins through `build_child_harness` into `Harness`;
- convert in-repo child filesystem selections to `FilesystemPlugin`;
- default child inheritance can continue to pass already-resolved non-MCP `ToolSpec` values;
- do not redesign subagent factories or plugin inheritance in this slice.

Mark the temporary `builtin_tools` path for removal in the later skills, parallel-LLM, and subagent plugin plans. Do not create a second hidden filesystem construction path for children.

## Implementation steps

1. Add plugin contracts, exports, and focused contract tests.
2. Refactor harness construction into collection, validation, and final assignment so static plugin contributions are atomic.
3. Add generic plugin connection management. During this unreleased intermediate slice, connect generic plugins first and the existing MCP stack second, both before `run_start`. If either path fails, close and reset both paths and remove all staged dynamic contributions so retry starts clean. Close MCP first, then generic plugins, then an owned model.
4. Add full dynamic contribution staging, validation, rollback, retry, concurrent-connect serialization, and reverse-order close.
5. Add `ToolOrigin` to `ToolSpec`. Stamp plugin tools with their unique plugin name while direct tools can keep `origin=None`.
6. Implement `FilesystemPlugin` on top of `FileTools` and move filesystem configuration except temporary parallel-LLM path policy out of `HarnessConfig`.
7. Migrate unit tests, end-to-end journeys, examples, README, `docs/docs.md`, and the hand-maintained `docs/site/explainer/index.html` to explicit filesystem plugins.
8. Remove core imports of `tools.filesystem`, `DEFAULT_BUILTIN_TOOLS`, and the exported `builtin_tools()` helper. Migrate its direct caller in `tests/unit/test_tool_retry.py`.
9. Add an architecture test that fails if `thinharness/core.py` imports `thinharness.plugins.filesystem` or `thinharness.tools.filesystem`.

## Behavior contract changes before implementation

After this plan review and before code changes, update `docs/behavior.md` with:

- explicit plugin composition and no implicit filesystem tools;
- bind, static contribution, lazy connect, atomic commit, retry, and reverse close rules;
- deterministic tool, instruction, and hook ordering;
- duplicate-name errors and plugin origin;
- one canonical root and the absence of filesystem side effects without `FilesystemPlugin`;
- plugin connection occurring before `run_start`, and the run toolset freeze occurring after both;

Edit only affected sections.

## Tests

Add focused tests for:

- static plugin tools and instructions visible before `connect()`;
- direct custom tools combined with plugin tools;
- duplicate plugin names, duplicate tools across plugins, and collisions between a plugin and `tools=`;
- copied hook registries, plugin hook ordering, dynamic hook-filter validation, and existing strict-hook behavior;
- lazy connection before `run_start`, one connection across several runs, concurrent connection, reverse close, failed-open rollback, cancellation, and successful retry;
- full dynamic tool validation, including reserved names, structured output, approval policy, and callable handlers;
- no partial dynamic tools, instructions, or hooks after failure;
- toolset freeze after plugin connection and run-start hooks;
- reuse of one plugin object across two harnesses with independent binding state;
- no root creation and no workspace instruction without `FilesystemPlugin`, plus missing-root tool behavior with the plugin;
- filesystem default selection, explicit ordering, empty selection, duplicates, unknown names, path rules, spill files, and opt-in JSONL search;
- approval, resume, streaming, tracing, and subagent behavior with plugin-provided tools.

Run:

```bash
uv run pytest tests/unit/test_harness.py tests/unit/test_hooks.py tests/unit/test_streaming.py tests/unit/test_resume.py tests/unit/test_approvals.py
uv run pytest tests/unit/test_file_tools.py tests/unit/test_subagents.py tests/unit/test_tool_retry.py tests/unit/test_parallel_llm.py tests/unit/test_parallel_tools.py tests/unit/test_skills.py
uv run pytest
uv run ruff check .
uv run pyright
```

Use the actual filesystem test file names present at implementation time if they differ.

## Success criteria

- The plugin seam supports static and connected adapters without exposing resources to callers.
- Filesystem tools are available only through explicit `FilesystemPlugin` use or direct `ToolSpec` registration.
- `jsonl_search` is an option on `FilesystemPlugin`, not a separate plugin.
- Independent custom tools and hooks remain direct constructor inputs.
- Core does not import or construct filesystem implementations.
- Failed plugin connection leaves the harness clean and retryable.
- The full test suite, Ruff, and Pyright pass.

## Out of scope

- MCP migration, except for making generic connection management possible.
- Skills, subagents, parallel LLM, providers, or tracing as plugins.
- Package distribution splitting.
- Plugin discovery, package manifests, hot reload, UI contributions, sandboxing, or dependency graphs.
- Dynamic tool-list change notifications after connection.

## Review record

One Codex, Claude, and GLM panel round reviewed plan v1 together with the MCP plan. Plan v2 applies the verified findings on parallel-LLM path policy, root side effects, hook registry copying, instruction order, dynamic validation, concurrent connection, temporary MCP coexistence, subagent validation, and missing tests. The approved product decisions are recorded above. No second plan-review round will run.
