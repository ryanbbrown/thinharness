# Structured output: local implementation vs pydantic-ai copy

## Summary

The current implementation uses a small local `thinharness/output.py` module built on Pydantic's public `TypeAdapter` API. It does not copy pydantic-ai's structured-output internals into runtime code.

This covers the current thinharness surface:

- `BaseModel`
- dataclasses
- `TypedDict`
- `list[T]`
- `Union[A, B]`
- `str` / `TextOutput`
- tool, native, and prompted modes
- validation retries
- markdown fence stripping
- `final_result` wrapping for non-object outputs

It is intentionally narrower than pydantic-ai's output layer. If structured output becomes a larger compatibility surface, copying and trimming pydantic-ai's implementation may be safer than growing local edge-case handling one case at a time.

## What we did locally

`thinharness/output.py` is about 230 lines. It handles:

- marker wrappers: `NativeOutput`, `PromptedOutput`, `ToolStructuredOutput`, `TextOutput`
- schema generation via `TypeAdapter.json_schema()`
- schema cleanup using existing local helpers from `tools.py`
- tool-mode argument wrapping for non-object schemas
- validation via `TypeAdapter.validate_python()` / `validate_json()`
- serialization via `TypeAdapter.dump_python(mode="json")`
- markdown fence stripping
- strict native schema compatibility checks for object schemas

Runtime imports do not depend on `vendor/pydantic-ai`.

## What a fuller pydantic-ai copy would involve

The plan originally called for copying and trimming files from:

- `vendor/pydantic-ai/pydantic_ai_slim/pydantic_ai/_output.py` - 1,633 lines
- `vendor/pydantic-ai/pydantic_ai_slim/pydantic_ai/_function_schema.py` - 396 lines
- `vendor/pydantic-ai/pydantic_ai_slim/pydantic_ai/_json_schema.py` - 208 lines
- `vendor/pydantic-ai/pydantic_ai_slim/pydantic_ai/output.py` - 434 lines
- selected helpers from `vendor/pydantic-ai/pydantic_ai_slim/pydantic_ai/_utils.py` - 921 lines total upstream
- selected exceptions from `vendor/pydantic-ai/pydantic_ai_slim/pydantic_ai/exceptions.py` - 297 lines total upstream
- selected message parts from `vendor/pydantic-ai/pydantic_ai_slim/pydantic_ai/messages.py` - 2,687 lines total upstream
- selected run context pieces from `vendor/pydantic-ai/pydantic_ai_slim/pydantic_ai/_run_context.py` - 152 lines total upstream

At upstream commit `ac684b2638ee1095077ece25b7fed5abe6d14a25`, those candidate source files total roughly 6,700 lines before trimming. The original estimate was about 600 lines after an aggressive trim, but that would still require maintaining a forked subset and resolving dependencies between copied internals.

## What pydantic-ai likely handles better

The local implementation is more minimal. Areas where pydantic-ai's implementation is likely more mature:

- complex JSON Schema normalization for different providers
- edge cases in unions, discriminators, recursive schemas, and constrained types
- strict native schema compatibility across nested schema structures
- richer output processors and retry prompt construction
- function-as-output-type support
- output validator hooks
- streaming or partial structured output paths
- a broader corpus of provider-specific tests

## Known local limitations

The local implementation deliberately does not support:

- function-as-output-type
- output validation hooks
- streaming structured output
- pydantic-ai's `OutputToolset` abstraction
- deferred tool requests or built-in output event types
- broad provider schema compatibility beyond the covered test cases

The current strict-schema handling recursively marks object nodes with `additionalProperties: false` and only sends `strict=True` when object nodes are fully required. This is enough for the covered object schemas, but pydantic-ai may still handle more schema forms correctly.

## When to reconsider copying pydantic-ai

Consider switching to a trimmed pydantic-ai copy if we need:

- production-grade support for arbitrary Pydantic schemas
- more provider-specific schema compatibility
- structured output streaming
- output validators/hooks
- function outputs
- fewer local decisions around schema normalization and retry semantics

Until then, the local `TypeAdapter` implementation keeps thinharness small and matches the current API goals.
