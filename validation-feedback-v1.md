# Validation feedback — ModelRetry + per-tool retry budget

Implementation under review vs. `.context/plan-tool-validation-and-retry.md`. Overall: the diff faithfully implements every step of the plan, respects every entry in `## Resolved`, and the public surface matches. Below are the deviations and gaps worth flagging.

## Structural deviations from the plan

### 1. `_prepare_args` uses sentinel typing instead of two-phase try blocks in the callers

Plan Step 3b showed the shape-check / validation / dispatch logic inlined in `call_tool` (and a mirror in `_invoke_tool`) with three separate `try` blocks. The implementation instead factors shape + validation into `_prepare_args(spec, raw_args) -> str | Any` and uses `isinstance(args, str)` in callers as an in-band signal that an envelope was already produced (`thinharness/tools.py:698-717`, `tools.py:765-769`, `tools.py:790-794`).

This works because validated args are never `str` instances (they're either a dict or a `BaseModel`), and it deduplicates the shape/validation logic between the sync and async paths. But the plan's stated reason for splitting the try blocks was *clarity about which `ValidationError` is retryable* — "arg-validation `ValidationError` is retryable, but a handler-internal `ValidationError` is an ordinary tool error" (plan line 101). The current shape is still correct (the handler dispatch is in a separate `try` from `_prepare_args`, and `test_handler_internal_validation_error_is_not_retry` pins the behavior), but a reader has to chase the sentinel return to confirm that property.

Low-priority cleanup: either keep the inlined version per the plan, or rename `_prepare_args` to make the discriminator obvious (e.g. return a small `PreparedArgs | RetryEnvelope` tagged class) so the contract is in the type signature rather than `isinstance` checks.

### 2. `_format_validation_errors` adds `(got …)` to "missing" errors

`tools.py:737-746` always appends `(got <input>)` when the Pydantic error item has an `input` key. Pydantic v2 *does* populate `input` for `type=missing` errors — but it's the *parent* dict, not the missing value. So a missing nested field renders as:

```
- filters.0.name: Field required (got dict)
```

The plan example (line 161) shows just `- filters.0.name: Field required`. The "(got dict)" suffix is noise for missing-field cases and could mislead the model.

Suggested fix: skip the `(got …)` suffix when `item.get("type") == "missing"`, or whitelist a small set of error types where the input is genuinely informative (`int_type`, `string_type`, `enum`, value errors).

The `test_nested_and_root_validation_error_formatting` test asserts only substring presence (`"filters.0.name" in nested["content"]`), so it doesn't catch this.

## Test-coverage gaps relative to the plan

Plan Step 7 enumerates ~20 test cases. All but two are covered. Missing:

1. **`_run_calls_concurrently` preserves `retry_kind` in result order for a mixed parallel batch.** The plan asked for "parallel batch with one retry and one success" pinned explicitly (line 364). The closest existing test is `test_two_calls_same_tool_share_budget_and_skip_batch_continuation`, but both calls in that test are the same `ModelRetry`-raising tool, and the harness routes same-tool batches through the sequential path anyway. No test asserts retry/success ordering through `_run_calls_concurrently`.

2. **Async handler-internal `ValidationError`.** `test_handler_internal_validation_error_is_not_retry` covers the sync handler path. The async path (`_invoke_tool` at `tools.py:790`) has its own try block and isn't directly exercised. Low risk since the implementations are parallel, but the plan called for parity.

## Minor nits

- `test_tool_retries_exceeded_counts_over_budget_failure` uses `tool_retries=1`; the plan's checklist (line 355) wrote `tool_retries=2`. Functionally equivalent — both pin the same counter semantics — just a difference from the plan's literal example.

- `ToolCallExecution` is declared with `retry_kind: str | None = None` (default) at `core.py:118-124`. The plan's snippet (line 178-184) had no default. Not a bug; just slightly looser than the plan. Worth keeping if it simplifies any synthetic construction in tests, otherwise drop the default to match.

- README "Tool retry" section (README.md:256-) covers the new behavior and the builtin auto-retry note that the plan asked for. Good.

## What the implementation gets right (worth noting because they're easy to break)

- **Pre-hook `retry_kind` capture is correct** (`core.py:605-606`). `parsed` is computed *before* `AfterToolCallContext` fires, `retry_kind` is locked in, and the post-hook re-parse is only used for `subagent` span metadata and the `ok is False` fallback. `test_after_tool_hook_sees_retry_envelope_and_cannot_break_budget` and `test_tracing_uses_pre_hook_retry_kind` pin this against the obvious regression.

- **Budget check raises before `session.continue_with_tools`** (`core.py:453-455`), so the over-budget batch's outputs never reach the provider. Both the `max_retries=0` case and the same-tool-twice case assert `session.tool_outputs == []` directly.

- **Counter semantics match the plan exactly** — `test_tool_retries_exceeded_counts_over_budget_failure` asserts `usage.tool_retries == {"flaky": 2}` with `tool_retries=1`, pinning that the over-budget failure is counted even though its envelope is never sent.

- **Subagent inheritance** matches the plan's three-row table (default subagent inherits, named subagent defaults to 1, named subagent honors its explicit value). `test_subagent_tool_retry_budget_inheritance` covers all three.

- **`LimitReachedContext` fires before the raise**, and the `tool_retries` `limit_kind` literal is properly added to `hooks.py:184`. `test_tool_retries_exceeded_counts_over_budget_failure` asserts hook order via the events list.
