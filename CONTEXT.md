# Domain Context

Named terms for the ThinHarness runtime. These are the ubiquitous names used in code, docs, and reviews; use them exactly.

## Request Constants

`RequestConstants` (`thinharness/providers.py`) is the frozen per-run bundle of everything constant across one run's provider requests: system instructions, tool schemas, request metadata, and the structured-output request. It is built once per run in `core.py`, after run-start hooks fire and MCP servers connect, and passed positionally to every `ModelSession` request method. Consequence: the run's toolset is frozen at run start — `add_tool` during an in-flight run affects the next run, not the current one. Per-request values (notices) stay keyword arguments on the session methods.

## Turn State Machine

`thinharness/turns.py` owns the loop that drives one run from its first model turn to a terminal result. `advance_until_terminal(start, session, constants, harness, run_ctx, tool_executor)` produces the first turn for all three entry paths (start, resume, approval-resume including the approved-batch replay), dispatches explicitly on all five `OutputTurnDecision` kinds (`final`, `retry_tool_output`, `retry_user_message`, `unexpected`, `continue`), spends the structured-output retry budget, and calls `run_ctx.finalize` or `run_ctx.pause_for_approval` itself. `core.py` does run-scoped setup (session creation, envelope validation, stream/tracing wiring), makes one call into the machine, and owns exception classification around it. `runtime.py`'s `RunContext` keeps per-request ceremony (limits, notices, tracing, usage) and terminal bookkeeping.

## Token Usage

`TokenUsage` (`thinharness/providers.py`) is the normalized per-turn token report on `ModelTurn.usage`: `input_tokens` and `output_tokens`, each `None` when the provider did not report it. `RunUsage.input_tokens`/`output_tokens` are the run totals, accumulated per provider request in `RunContext.advance_model` and surfaced on `HarnessResult.usage`; they survive approval pauses via the run-state codec (`RunUsage.to_json`/`from_json`), which defaults missing token keys to 0 so pre-token envelopes still resume. Tracing prefers the normalized fields and falls back to best-effort raw extraction for custom models; `gen_ai.usage.total_tokens` passes a raw total through when present, otherwise computes input+output only when both are known.
