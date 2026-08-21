# Provider package reorganization — plan

Replace the 1,820-line `thinharness/providers.py` module with a focused `thinharness/providers/` package. Preserve behavior and the existing public import path.

This is a pure refactor. Do not change provider payloads, request settings, retries, resume state, model/session protocols, inference behavior, exports, or error messages.

## Package structure

- `providers/base.py`: provider-neutral model turns, tool outputs, notices, request constants, capabilities, settings, model/session protocols, and neutral normalization helpers.
- `providers/transport.py`: shared HTTP transport policy, retry parsing and delays, `ProviderError`, and the base provider lifecycle.
- `providers/transcript.py`: provider-neutral transcript entries, resume-state encoding and validation, transcript mutation, image recovery, and shared reasoning fallback behavior.
- `providers/openai.py`: `OpenAIProvider`, `OpenAIResponsesModel`, `OpenAIResponsesSession`, OpenAI payload rendering, transcript replay, structured-output conversion, and response extraction.
- `providers/anthropic.py`: `AnthropicProvider`, `AnthropicMessagesModel`, `AnthropicMessagesSession`, Anthropic payload rendering, transcript replay, structured-output conversion, and response extraction.
- `providers/openrouter.py`: `OpenRouterProvider`, `OpenRouterModel`, `OpenRouterSession`, OpenRouter payload rendering, transcript replay, structured-output conversion, and response extraction.
- `providers/__init__.py`: explicit public re-exports plus model-reference parsing, provider-prefix resolution, capability lookup, same-provider checks, and `infer_model` dispatch.

Each provider file must own its complete wire dialect. Shared modules must not import concrete provider adapters. Avoid wildcard imports and pass-through wrapper functions.

## Interface rules

- `from thinharness.providers import ...` continues to expose every current public name used by ThinHarness, tests, examples, and applications.
- Top-level `from thinharness import ...` exports remain unchanged.
- Internal callers that currently import private shared helpers from `thinharness.providers` continue to resolve through explicit package exports or move to the correct focused module.
- Do not add aliases for removed private locations. The package path itself is the current public interface.
- A custom application model or transport remains definable outside ThinHarness and injectable through `Harness(model=...)` or an existing built-in model's `provider=` argument.
- Delete `thinharness/providers.py` after the package is complete. Do not keep both forms.

## Tests

Split `tests/unit/test_providers.py` into focused files for neutral/factory behavior, transport/retries, transcript behavior, OpenAI, Anthropic, and OpenRouter. Keep cross-provider image, reasoning, resume, tracing, and harness tests in their existing files.

Preserve every existing assertion. Add or adjust architecture tests so they prove:

- `providers.py` no longer exists and the focused package modules do;
- each provider adapter lives in its own module;
- current public imports resolve from `thinharness.providers` and `thinharness`;
- a small external-style custom model and a custom built-in transport still inject without source edits.

## Validation

- Focused provider, resume, image, reasoning, tracing, parallel-LLM, subagent, and architecture tests.
- `uv run ruff check .`
- `uv run pyright`
- `uv run pytest`
- `git diff --check`
- Build a wheel in a temporary directory, install it into a clean temporary virtual environment, and import all current top-level exports plus all public `thinharness.providers` exports.
