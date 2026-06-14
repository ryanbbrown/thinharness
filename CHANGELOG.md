# Changelog

## 0.1.0 - Unreleased

- Added approval-required custom tools with `Harness.resume_approvals(...)` / `stream_approvals(...)`, pending approval results, and approval resume events.
- Added an opt-in `BashTool` for exploratory runs; it is custom-registration only, not a default or named built-in tool.
- Changed the built-in `edit` tool to use a list-only `edits` schema and report per-edit metadata, replacing the previous flat single-edit arguments and top-level edit metadata.
- Initial PyPI release.
