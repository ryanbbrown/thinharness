# Changelog

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
