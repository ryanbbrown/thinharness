# Changelog

## Unreleased

- **Breaking:** `ModelSession` narrowed to three request methods — `start(prompt, constants, ...)`, `continue_with_tools(outputs, constants, ...)`, and `continue_with_user_text(text, constants, ...)` — each taking a per-run `RequestConstants` positional parameter; the `continue_with_user_message`/`continue_with_user_prompt` pair is removed.
- **Breaking:** Removed `ModelTurn.finalized_output_mode`; the finalized mode is carried on `OutputTurnDecision` and `ModelMessageEvent.finalized_output_mode` is unchanged.
- **Breaking:** `OutputTurnDecision` and `resolve_turn_output` moved from `thinharness.output` to the new `thinharness.turns` module (no compatibility re-export).
- Changed the run toolset to freeze at run start: tools added with `add_tool` during an in-flight run take effect on the next run.
- Changed Anthropic model spans to gain `gen_ai.usage.total_tokens`, computed as input+output when the provider reports both and no raw total exists.
- Added `RunUsage.input_tokens`/`output_tokens` run totals accumulated per provider request and surfaced on `HarnessResult.usage`; approval envelopes written before these fields existed still resume.
- Added run-state codecs `RunUsage.to_json`/`RunUsage.from_json` and limit-notice key encode/decode in `thinharness.types`.
- Added normalized `ModelTurn.usage` (`TokenUsage`), `ModelTurn.finish_reason`, and `ModelTurn.response_model`; tracing prefers them and falls back to raw extraction for custom models.
- Added `RequestConstants` and `TokenUsage` to the public exports.

## 0.4.0 - 2026-06-25

- Removed background tool execution and background completion semantics, including `ToolSpec.background`/`background_policy`, `SubAgentConfig.background`, `BackgroundTask*Event`, and the approval-envelope background fields.

## 0.3.0 - 2026-06-23

- Added JSONL search range filters, typed equality filters, and field snippets.
- Added a unified, provider-agnostic transcript resume state shared across Anthropic, OpenAI Responses, and OpenRouter, replacing the previous provider-specific resume payloads.
- Added same-provider reasoning fidelity: native model reasoning (Anthropic thinking signatures, OpenAI `encrypted_content`, OpenRouter `reasoning_details`) is preserved when resuming on the same provider and degraded to a leading `<thinking>`-tagged text block on cross-provider resume.
- Changed `resume_state` to a `kind="transcript"`, version 3 format; resume state captured by 0.2.0 (version 1) is rejected and must be regenerated.

## 0.2.0 - 2026-06-15

- Added SDK event streaming with typed run, model, tool, background task, retry, limit warning, completion, and failure events.
- Added approval-required custom tools with `Harness.resume_approvals(...)` / `stream_approvals(...)`, pending approval results, and approval resume events.
- Added background tool execution and background completion semantics.
- Added an opt-in `BashTool` for exploratory runs; it is custom-registration only, not a default or named built-in tool.
- Added local tracing helpers and expanded tracing coverage.
- Added a generated documentation site and web research report example.
- Changed search output to be document-oriented.
- Changed the built-in `edit` tool to use a list-only `edits` schema and report per-edit metadata, replacing the previous flat single-edit arguments and top-level edit metadata.

## 0.1.0 - 2026-05-18

- Initial PyPI release.
