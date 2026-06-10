# Tool Prompt Defaults Plan

## Goal

Centralize framework-owned prompt text for built-in tools and add a minimal mechanism for tool-specific usage instructions to be included in model instructions.

This should improve tool reliability, especially for `parallel_llm`, without adding a runtime customization API.

## Decisions

- Move built-in tool descriptions out of constructors and into `thinharness/defaults.py`.
- Add framework-owned tool usage instruction constants in `thinharness/defaults.py`.
- Keep customization fork/source-edit oriented for now; do not add `HarnessConfig` overrides.
- Treat tool descriptions and tool usage instructions as different surfaces:
  - tool descriptions stay in provider tool schemas
  - tool usage instructions are appended to the harness system instructions when the tool is enabled
- Keep instruction blocks short and operational, not full agent workflows.

## Proposed API Shape

Add an optional field to `ToolSpec`:

```python
instructions: str | None = None
```

`ToolSpec.response_tool()` remains unchanged. The new field is only used by the harness when building system instructions.

`Harness.system_instructions()` should append enabled tool instructions after the user/system prompt, workspace root, and skill summary.

## Default Constants

Add constants such as:

```python
DEFAULT_READ_DESCRIPTION = "Read a UTF-8 text file with line numbers, offset, and limit."
DEFAULT_PARALLEL_LLM_DESCRIPTION = "..."
DEFAULT_PARALLEL_LLM_INSTRUCTIONS = "..."
```

Use these constants from built-in tool constructors.

## Parallel LLM Instruction Content

The `parallel_llm` instruction block should clarify:

- Use it only for independent one-shot prompts.
- It does not inherit the parent system prompt.
- If using inline prompts, include `prompts` and omit `prompts_file`.
- If using a prompt file, include `prompts_file` and omit `prompts`.
- Do not fill unused optional fields with placeholder values.
- For large or structured batches, write results to an output file and read that file afterward.

This is a prompt reliability improvement, not a wrapper that silently fixes malformed calls.

## Scope

In scope:

- `ToolSpec` data model update.
- Built-in filesystem descriptions moved to constants.
- `parallel_llm` description moved to constants.
- Optional tool instruction inclusion in `system_instructions()`.
- Focused tests for instruction inclusion and tool schema preservation.

Out of scope:

- Runtime prompt override config.
- Public customization API.
- Rewriting tool schemas.
- Changing `parallel_llm` argument shape.
- Updating example agents.

## Validation

- Existing tool schemas should remain unchanged except for text sourced from constants.
- `system_instructions()` should include instructions only for enabled tools that define them.
- User-provided `system_prompt` should still be preserved and should not suppress framework-owned tool instructions.
- Run focused tests for harness instruction assembly and tool schema generation.
- Run `uv run pyright` after implementation.

