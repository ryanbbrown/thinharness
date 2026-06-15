# Changelog

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
