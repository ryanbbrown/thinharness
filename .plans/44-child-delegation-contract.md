# Child delegation contract — plan

Fix three small pre-0.7 defects in the public child-delegation interface.

## Scope

1. **Use the real child model for plugin validation.** Parent construction must not bind an override-model child's plugins against the parent model. For a child without a model override, keep eager plugin-tool validation against the borrowed parent model. For an override-model child, validate plugin names, direct tools, and other model-independent recipe rules during parent construction, then validate plugin contributions when the real child model is constructed. Do not infer or open the override model during parent construction.
2. **Keep one inheritance state.** Remove `ChildHarnessRequest.inherited`. `tool_mode` is authoritative: `"explicit"` does not inherit; `"inherited"` and `"inherited+explicit"` inherit. Continue exposing the derived boolean in hook contexts and delegation result metadata.
3. **Keep one parent harness reference.** Remove `BeforeSubagentRunContext.parent_harness`. Both before- and after-subagent hooks use the inherited `HookContext.harness`, which is the parent harness.

This is a clean pre-1.0 interface correction. Do not add aliases, deprecated properties, fallback reads, migration documentation, or compatibility constructors.

## Tests

- Reproduce the override-model false duplicate rejection and prove parent construction now accepts the valid recipe.
- Prove an override-model plugin collision or approval violation is rejected when the real child is constructed.
- Prove parent-model child plugin collisions still fail during parent construction.
- Prove each `tool_mode` produces the correct inherited composition, hook boolean, and result metadata.
- Prove before- and after-subagent hooks receive the parent through `ctx.harness` and that the before context has no `parent_harness` field.
- Update all direct `ChildHarnessRequest` constructions and type assertions.

## Non-goals

Do not change provider modules, model lifecycle, plugin registration, the sealed agent catalog, tool composition provenance, README copy, versions, or release automation.

## Validation

- Focused subagent, hook, tracing, and architecture tests.
- `uv run ruff check .`
- `uv run pyright`
- `uv run pytest`
- `git diff --check`
