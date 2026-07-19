# Changelog

## 0.5.4 - 2026-07-19

- Added in-process MCP server support: `MCPServer` is now a concrete class that accepts any FastMCP `ClientTransport`, including `FastMCPTransport` for an MCP server object in the same Python process.
- Changed the MCP connection layer to the FastMCP client (`fastmcp-slim[client]==3.4.4`, added to the `mcp` extra); ThinHarness no longer maintains its own MCP session lifecycle. `MCPServerStdio`, `MCPServerSSE`, and `MCPServerStreamableHTTP` keep their existing constructors, ids, timeout semantics, tool filtering, and error envelopes. The final-close bound now comes from FastMCP's `client_disconnect_timeout` setting (default 5 seconds, previously a fixed 3 seconds).

## 0.5.3 - 2026-07-07

- Added provider-neutral `max_tokens` and `effort` settings, raised Anthropic's default `max_tokens` to 16384, and translated those settings to each built-in provider's payload dialect.
- Changed Anthropic structured output to support native `output_config.format` by default while preserving explicit tool and prompted modes.
- Fixed Anthropic resume replay for adaptive and default-on thinking models so signed thinking blocks are preserved when the resumed request can accept them.

## 0.5.2 - 2026-07-05

- Changed Anthropic Messages requests to opt into provider prompt caching via top-level `cache_control: {"type": "ephemeral"}` (automatic breakpoint placement); a `cache_control` key in model `extra_body` overrides the default.

## 0.5.1 - 2026-07-04

- Added provider-reported cached input token counts to normalized usage and OTel GenAI model span attributes, including OpenAI Responses, Chat Completions, and Anthropic cache-read usage.
- Added `RunUsage.cached_tokens` run totals accumulated per provider request; approval envelopes written before this field existed still resume.

## 0.5.0 - 2026-07-03

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
