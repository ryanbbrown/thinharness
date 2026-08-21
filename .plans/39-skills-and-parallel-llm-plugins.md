# Skills and parallel LLM plugins — plan v2

Migrate the remaining non-subagent built-in tools onto the plugin seam created by plans 37 and 38. This slice adds `SkillsPlugin` and `ParallelLlmPlugin`, removes their configuration and composition logic from core, and leaves subagent composition for a later plan.

## Resolved decisions

1. **Keep both low-level modules.** `SkillRegistry` remains the deep module for skill discovery and execution. `ParallelLlmTool` remains the deep, renameable module for one-shot batches. The plugins own harness composition, defaults, and origin attribution rather than copying either implementation.
2. **Skills stay static.** `SkillsPlugin` discovers one catalog when the plugin object is constructed. Its `bind()` method performs no I/O, and its selected tools and summary are visible immediately after harness construction. Reusing one plugin object across harnesses reuses that catalog; callers construct a new plugin to rediscover added or removed skills.
3. **The skill catalog is frozen, not skill contents.** Discovery metadata, selected names, and the summary are fixed at plugin construction. Existing `SKILL.md` content, trees, and scripts remain live and are read when `skill_read` or `skill_run` executes.
4. **Skill paths keep their current meaning.** Relative `skills_dir` values resolve from the process working directory through `SkillRegistry`, not from `HarnessConfig.root`. Documentation must state this difference from root-scoped filesystem and parallel tools.
5. **Skill tool selection stays explicit.** `SkillsPlugin` requires a non-empty ordered selection of `skill_read`, `skill_run`, or both. Enabling skill discovery does not silently enable script execution.
6. **The parallel plugin uses the harness model by default.** Add the configured harness `Model` to `PluginContext`. `ParallelLlmPlugin(model=None)` borrows that model; an explicit model object is caller-owned; an explicit model string uses plugin-owned provider settings and the existing per-call create/close behavior.
7. **Alternate parallel models do not inherit hidden parent settings.** Provider and request settings are valid only when `model` is a string. Reject them for `model=None` or a model object instead of storing ignored values.
8. **Plugin names are runtime-fixed.** `SkillsPlugin.name == "skills"` and `ParallelLlmPlugin.name == "parallel_llm"` cannot be changed on an instance or class, including a subclass. One plugin of each type is allowed per harness; one SkillsPlugin accepts several skill directories.
9. **Subagents are not redesigned here.** Named child agents use their existing `plugins` field for explicit skills and parallel LLM. Remove the now-empty child `builtin_tools` path. A small SkillsPlugin-specific inheritance bridge preserves default and `inherit_parent_tools=True` behavior until `SubagentsPlugin` replaces child composition.
10. **The main built-in selector stays temporarily.** `HarnessConfig.builtin_tools` remains only to select the deferred `subagent` tool. Skill and parallel names fail with errors that point to their plugins.

These are breaking pre-1.0 changes. Do not add aliases, fallback configuration reads, or compatibility constructors.

## Goal

After this plan, callers compose both features explicitly:

```python
from thinharness import Harness, HarnessConfig, ParallelLlmPlugin, SkillsPlugin

harness = Harness(
    HarnessConfig(root="."),
    plugins=[
        # Relative skill paths use the process working directory, not root.
        SkillsPlugin(".agents/skills", tools=["skill_read"]),
        ParallelLlmPlugin(),
    ],
)
```

`thinharness/core.py` does not import skill or parallel-LLM implementation modules, construct either tool family, discover skills, render skill summaries, own parallel path policy, or copy parallel provider settings.

## Plugin context

Extend the existing context by one stable core dependency:

```python
@dataclass(frozen=True)
class PluginContext:
    root: Path
    model: Model
```

Core constructs the model before binding plugins, as it does now. Do not add `Harness`, `HarnessConfig`, credentials, provider settings, plugin lookup, child factories, tracing, or mutable services to the context.

A plugin must not close `context.model`; model ownership remains with the harness or caller. Add contract coverage that one plugin object bound to two harnesses sees each harness's own root and model.

## SkillsPlugin

Add `thinharness/plugins/skills.py`:

```python
SkillsPlugin(
    skills_dir: str | Path | Sequence[str | Path],
    *,
    selected_skills: Sequence[str] | None = None,
    tools: Sequence[Literal["skill_read", "skill_run"]],
)
```

Behavior:

- The fixed name is `"skills"`.
- Reject a set for `skills_dir` or `tools`. Reject an empty skill-directory sequence, an empty tool selection, duplicate tools, and unknown tools when the plugin is constructed.
- Construct one `SkillRegistry` in the plugin constructor. Discovery and selected-skill validation therefore occur before harness binding.
- A valid tool selection with no discovered skills contributes no tools and no summary. `selected_skills` naming a missing skill still fails during construction.
- `bind()` selects the requested `ToolSpec` values in caller order and returns one static `PluginContribution` without calling filesystem metadata functions.
- When at least one selected tool exists, contribute the compact skill summary once as plugin instructions.
- Make summary wording conditional. When `skill_read` is absent, do not tell the model to call it.
- Generic normalization assigns `ToolOrigin(plugin="skills", source=<tool name>)`.
- `skill_run` remains sequential. `skill_read` remains parallel-safe.
- Keep current frontmatter, containment, tree rendering, truncation, runner selection, working directory, merged output, timeout, and result metadata behavior inside `SkillRegistry`.
- Keep `Skill`, `SkillRegistry`, argument types, and parser helpers public at their current module level.

The frozen catalog contains skill names, paths, metadata, selection, and summary text. `skill_read` still reads current file content and builds the current tree at invocation time. `skill_run` still executes the current script file. Adding or removing skill entries after plugin construction does not change the catalog; editing a discovered skill's files remains visible.

The skill summary becomes a normal plugin instruction. Its position follows caller plugin order, before all per-tool instructions under PLUGIN-8. This intentionally replaces the old fixed transitional-summary position.

## ParallelLlmPlugin

Add `thinharness/plugins/parallel_llm.py` with fixed name `"parallel_llm"`. It contributes one static tool named `parallel_llm` by constructing the existing `ParallelLlmTool` at bind time with the canonical context root and resolved model.

The plugin accepts:

- `model: Model | str | None = None`;
- `description` and `instructions`, defaulting to the current parallel defaults;
- `read_paths`, `write_paths`, and `max_prompts`;
- for a string model only: `api_key`, `base_url`, request timeout, retry count, retry backoff, temperature, max tokens, effort, and extra body.

Use `None` sentinels for optional provider/request arguments on the plugin. For a string model, map omitted values to the current `ParallelLlmTool` defaults. For `model=None` or a model object, reject any supplied provider or request argument because the borrowed model already owns those settings.

Reject `max_prompts < 1` during plugin construction. Do not add `root`, output schema settings, or a caller-settable tool name to the plugin. Callers that need a renamed or structured-output batch tool continue to use `ParallelLlmTool(...).spec()` through direct `tools=` composition.

### I/O-free binding

`ParallelLlmTool.__init__` currently resolves its root. Add a private `_root_is_resolved: bool = False` argument, matching `FileTools`. The plugin passes the already-resolved `PluginContext.root` with `_root_is_resolved=True`. Do not call `Path.resolve()`, `Path.exists()`, `Path.stat()`, provider inference, or network code during `ParallelLlmPlugin.bind()`.

### Model ownership

- `model=None`: bind to `PluginContext.model`; the plugin does not close it.
- `model=<Model>`: bind to that object; the caller owns it and the plugin does not close it.
- `model="provider:model"`: preserve `ParallelLlmTool` behavior — infer a model for each batch invocation and close the created provider after the batch, including schema-resolution, request, and cancellation failures.

Preserve all current parallel behavior:

- independent stateless prompts and fresh sessions;
- ordered sparse results with bounded concurrency;
- no parent system prompt, tools, memory, or continuation;
- file input and output policies under `PluginContext.root`;
- prompt caps, atomic JSON files, cancellation, and batch-local request counts;
- provider transport retries inside one logical request;
- no model override in model-visible arguments;
- default description and per-tool instructions;
- text-only plugin batches.

Batch model requests and tokens remain outside parent `RunUsage` and `max_model_requests`. The parent counts one `parallel_llm` tool call toward `max_tool_calls`; batch metadata reports its own `model_requests`, `total`, `succeeded`, and `failed` values.

Generic normalization assigns `ToolOrigin(plugin="parallel_llm", source="parallel_llm")`.

Remove `"parallel_llm"` from `ToolKind` because no core control flow uses it. Remove `kind="parallel_llm"` from `ParallelLlmTool.spec()`, so direct and plugin-created specs use the valid default kind `"user"`. Update the `ToolKind` literal and its runtime validation set, tests, and changelog together. The low-level tool's execution behavior does not change.

## Core removals and fail-loud guards

Remove from `HarnessConfig`:

- `skills_dir`;
- `selected_skills`;
- `read_paths`;
- `write_paths`;
- `builtin_parallel_llm_model`;
- `builtin_parallel_llm_temperature`;
- `parallel_llm_max_prompts`.

Add a `model_validator(mode="before")` that rejects each removed field by name with a plugin migration message. Do not rely on Pydantic's default extra-field behavior. Replace the old `selected_skills requires skills_dir` test with explicit removed-field tests. Do not change all unknown extras to `extra="forbid"` in this slice.

Remove from `Harness`:

- the `skills=` constructor argument;
- `self.skills` and `_skills_enabled`;
- skill discovery, selected-skill validation, summary rendering, and skill tool validation;
- `create_parallel_llm_tool(self)` and parallel provider-setting projection;
- direct imports of `SkillRegistry` and the parallel-LLM module.

Delete `create_parallel_llm_tool(parent)` and its exports. Keep `ParallelLlmTool` as the direct low-level interface.

Reduce the main built-in candidate list to `subagent`. Keep `HarnessConfig.builtin_tools` until the subagent plugin plan. Requests for `skill_read` or `skill_run` must point to `SkillsPlugin`; requests for `parallel_llm` must point to `ParallelLlmPlugin`. Other unknown values keep the normal unknown-built-in error.

Remove `SubAgentConfig.builtin_tools`. Extend its existing before-validator to reject the removed field by name, as it already does for `background`, so a second valid tool source cannot hide the mistake. Update validation to:

- detect recursive `subagent` exposure only through direct `tools`;
- reject `inherit_parent_tools` combined with `plugins` or direct `tools`;
- require named subagents to define `plugins`, `tools`, `inherit_parent_tools=True`, `inherit_mcp_servers=True`, or `mcp_servers`;
- use that exact source list in the validation error.

Migrate every caller, including the rejected Bash built-in test, instead of leaving obsolete child configuration examples.

## Temporary subagent behavior

Preserve these behaviors while deferring `SubagentsPlugin`:

- `SubAgentConfig.plugins` accepts `SkillsPlugin` and `ParallelLlmPlugin` for explicit child composition.
- A non-inheriting named child that wants skills must configure its own `SkillsPlugin`; it does not implicitly reuse the parent's catalog.
- A default child and a child with `inherit_parent_tools=True` continue to receive the parent's resolved, non-approval tools.
- Resolved MCP tools remain excluded unless MCP inheritance is explicit.
- Find the exact parent `SkillsPlugin` with an `isinstance` scan over the read-only `parent.plugins` tuple, matching the MCP bridge pattern.
- Only when that plugin exists, exclude inherited resolved tools whose origin plugin is `"skills"` **and** whose names are in that exact plugin's configured tool selection. Generic direct tools or another custom plugin with a caller-supplied `ToolOrigin(plugin="skills")` remain inherited.
- Rebind the same parent SkillsPlugin object in the child instead of copying those selected specs. The child receives the same registry object, catalog, tool order, and one summary without a second discovery pass.
- Insert the rebound SkillsPlugin after the current filesystem instruction-only plugin in the temporary inherited-plugin list. This fixes child instruction order: filesystem root instruction, skill summary, then per-tool instructions.
- Other inherited static tools, including `parallel_llm`, keep the current resolved-handler inheritance behavior. A child with a model override therefore keeps the parent batch handler and parent batch model, matching current `inherit_parent_tools` behavior.
- Keep the existing filesystem instruction-only bridge and MCP server bridge.
- Child model, tracing, hooks, limits, output, lifecycle, and cleanup behavior do not change.

Mark the SkillsPlugin-specific bridge, filesystem instruction bridge, and MCP bridge for deletion in the later subagent plugin plan. Do not add generic plugin inheritance, clone methods, or child factories now.

## Behavior contract changes before implementation

After plan review and before code changes, update only affected sections of `docs/behavior.md`:

- extend PLUGIN-3 to state that `PluginContext` contains the canonical root and configured model while binding remains synchronous and I/O-free;
- update PLUGIN-8 so the skill summary is a plugin instruction ordered by plugin position;
- remove the temporary parallel-LLM path-policy wording from FILESYSTEM-PLUGIN-5;
- add a Skills Plugin section covering one fixed-name plugin per harness, explicit tool selection, constructor-time catalog discovery, static visibility, cwd-relative paths, frozen catalog/live content, summary conditions, tool behavior, and temporary child inheritance;
- add a Parallel LLM Plugin section covering one fixed-name plugin per harness, explicit composition, model ownership, provider-setting validation, root/path policy, text-only behavior, parent-run accounting exclusion, results, retries, cancellation, and the direct `ParallelLlmTool` escape hatch;
- update PROVIDER-RETRY-6 so plugin-owned string models use their own settings while a plugin borrowing the harness model uses that model's configured transport retries.

Do not update unrelated behavior sections.

## Architecture guard

Extend `tests/unit/test_architecture.py` with a core-only check. Reject these tokens in `thinharness/core.py`:

- `tools.skills`, `SkillRegistry`, `skills_dir`, `selected_skills`, `_skills_enabled`, and `prompt_summary`;
- `tools.parallel_llm`, `create_parallel_llm_tool`, `builtin_parallel_llm`, and `parallel_llm_max_prompts`.

Scope the test to direct core source. Temporary imports and bridges in `thinharness/subagents.py` remain allowed until its plugin migration.

Also test that `ParallelLlmPlugin.bind()` does not call root-resolution or filesystem metadata methods and does not infer or contact a provider.

## Implementation steps

1. Update the affected behavior contracts.
2. Add `model` to `PluginContext` and migrate every binding call and contract test.
3. Add `SkillsPlugin`, focused tests, and public exports.
4. Add the private resolved-root path to `ParallelLlmTool`; add `ParallelLlmPlugin`, focused tests, and public exports.
5. Migrate main harness and explicit child callers from skill and parallel built-ins to plugins.
6. Implement the exact temporary inherited-skills bridge.
7. Add fail-loud removed-field guards, then remove the listed core fields, constructor arguments, state, helpers, imports, `ToolKind` value, and child built-in path.
8. Add the concrete architecture and I/O-free binding tests.
9. Update README, `docs/docs.md`, all checked-in site pages, examples, exports, changelog, and end-to-end journeys.
10. Regenerate the README-derived site with `uv run scripts/build_site.py` and verify `uv run scripts/build_site.py --check`.

## Tests

Retain the existing `SkillRegistry` and `ParallelLlmTool` behavior suites. Add focused coverage for:

### Plugin context

- root and model identity passed to static and connected test plugins;
- one plugin object bound to two harnesses with independent context values;
- plugins do not close caller-owned models.

### Skills

- constructor rejects unordered or empty skill directories, and unordered, empty, duplicate, or unknown tool selections;
- constructor discovers recursively, filters selected skills, and rejects duplicate or missing selected names;
- relative skill paths preserve process-working-directory resolution across harnesses with different roots;
- selected tools and summary are static and visible immediately after harness construction;
- caller tool order and plugin-order-dependent summary order are preserved;
- no discovered skills contributes no tools or summary, while a missing selected skill fails;
- adding a skill after construction is ignored; editing a discovered `SKILL.md` or script remains visible to execution; a new plugin sees the added skill;
- plugin reuse across harnesses shares one catalog and registry object;
- runtime name mutation and subclass overrides are rejected;
- origin stamping and collisions with direct or other plugin tools;
- custom direct tools and custom plugins with `origin.plugin == "skills"` remain inherited unless they match the exact parent SkillsPlugin selection;
- summary appears once, and its wording does not mention unavailable `skill_read`;
- existing read, run, timeout, containment, truncation, runner, and sequential behavior;
- explicit child plugin composition and inherited parent skill tools plus exactly one summary.

### Parallel LLM

- static visibility, runtime-fixed plugin/tool name, origin, description, and instructions;
- I/O-free bind with the resolved context root;
- `model=None` binds each harness model independently when one plugin object is reused;
- explicit model objects are borrowed and never closed;
- provider/request options are rejected unless model is a string;
- explicit model strings use plugin request settings and close created providers on success, failure, schema failure, and cancellation;
- `max_prompts` fails during plugin construction;
- canonical root, read/write policies, prompt cap, concurrency, ordering, batch-local accounting, output files, retries, cancellation, and text-only output;
- one parent tool call is counted while nested batch requests and tokens stay outside parent run usage and limits;
- no parent system prompt or nested tool execution;
- direct `ParallelLlmTool` remains renameable, supports structured output, and now emits kind `"user"`;
- explicit child plugin composition and inherited resolved-tool behavior, including a child model override retaining the parent's batch model.

### Removal and integration

- each removed `HarnessConfig` field raises its own migration error;
- `Harness(skills=...)` raises rather than being ignored;
- `builtin_tools=["skill_read"]`, `builtin_tools=["skill_run"]`, and `builtin_tools=["parallel_llm"]` give plugin migration errors;
- `builtin_tools=["subagent"]` still works and other unknown names keep the normal error;
- `SubAgentConfig.builtin_tools` raises even when another valid source is present;
- updated named-subagent validation uses the exact remaining source list;
- plugin/direct/structured-output tool collisions remain atomic;
- resume and approval state do not serialize plugin configuration or `PluginContext.model`;
- system instructions preserve plugin order and place all per-tool instructions last.

Run:

```bash
uv run pytest tests/unit/test_skills.py tests/unit/test_parallel_llm.py tests/unit/test_plugins.py tests/unit/test_architecture.py tests/unit/test_file_tools.py
uv run pytest tests/unit/test_harness.py tests/unit/test_subagents.py tests/unit/test_tracing.py tests/unit/test_resume.py tests/unit/test_approvals.py
uv run pytest
uv run ruff check .
uv run pyright
uv run scripts/build_site.py --check
uv run --env-file .env python tests/e2e/skills_journey.py
uv run --env-file .env python tests/e2e/parallel_llm_agent_journey.py
uv run --env-file .env python tests/e2e/parallel_llm_tool_journey.py
```

Report credential-based skips separately and do not count them as passes.

## Documentation and caller migration

Update:

- README feature text and examples, including the current `builtin_parallel_llm_model` text;
- `docs/docs.md` configuration, plugin, skills, parallel batch, and child-agent examples;
- `docs/site/explainer/index.html` architecture tree, composition text, and feature tables;
- hand-written `docs/site/index.html` parallel-LLM card;
- the README-derived `docs/site/about/index.html` through the site builder;
- `tests/e2e/skills_journey.py` and `parallel_llm_agent_journey.py`;
- examples and tests using core skill or parallel settings;
- `examples/web_research_report/agent.py` to remove obsolete core `read_paths` and `write_paths` while keeping its direct structured `ParallelLlmTool`;
- the rejected Bash child built-in test;
- public exports and changelog breaking entries, including direct `ParallelLlmTool` specs changing from kind `"parallel_llm"` to `"user"`.

Do not replace direct `ParallelLlmTool` values that use a custom name or structured output. Do not wrap independent custom tools in plugins.

## Success criteria

- Skills and the default parallel batch tool are enabled only through explicit plugins.
- `PluginContext` contains only the canonical root and configured model.
- Core contains no direct skill or parallel-LLM imports, configuration, discovery, summary, path policy, model projection, or tool construction.
- Skill tools and summary are static, ordered, and based on one constructor-time catalog while discovered files remain live.
- Parallel plugin binding is I/O-free, and model ownership and provider-setting rules are explicit and tested.
- Low-level `SkillRegistry` and `ParallelLlmTool` behavior remains available without plugin composition.
- Main `builtin_tools` selects only `subagent`; the child built-in selector is gone and rejected loudly.
- Existing child tool inheritance remains safe through the exact temporary bridge.
- Focused suites, full suite, Ruff, Pyright, site drift check, and the three relevant end-to-end journeys pass.

## Out of scope

- `SubagentsPlugin`, child-harness factories, or generic plugin inheritance.
- Removing `HarnessConfig.builtin_tools`, `HarnessConfig.subagents`, or the main `subagent` built-in.
- Replacing `SkillRegistry` discovery or frontmatter parsing.
- Sandboxing skill scripts or changing supported runners.
- Dynamic skill catalog refresh or filesystem watchers.
- Nested tools, memory, or multi-turn sessions inside parallel LLM batches.
- Changing the low-level `ParallelLlmTool` structured-output interface.
- Provider, tracing, structured output, approval, resume, limit, hook, Bash, or direct custom-tool migration.
- Plugin discovery, package manifests, hot reload, dependency ordering, or plugin-to-plugin lookup.

## Review record

One Codex, Claude, and GLM panel round reviewed plan v1. Plan v2 applies the verified findings on I/O-free parallel binding, exact skill inheritance, frozen catalog versus live contents, removed-field rejection, model-setting validation, prompt order, fixed names, batch accounting, subagent validation, `ToolKind`, documentation coverage, summary wording, and focused tests. No second plan-review round is scheduled.
