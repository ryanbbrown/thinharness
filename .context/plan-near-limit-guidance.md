# Plan: Near-Limit Guidance

## Overview
Add provider-neutral model notices so the harness can warn the model when it is about to exhaust local run limits. This builds directly on `.context/plan-hooks-and-limits.md` Step 8: warnings are first-class harness-generated model input, not raw model-request hooks, and provider adapters are responsible for preserving each provider's continuation payload shape.

## Steps

### 1. Add Provider-Neutral Notice Types
Define a small provider-facing notice type in `thinharness/providers.py` and extend the model session protocol to accept notices on both initial and continuation requests.

Suggested shape:

```python
@dataclass
class ModelNotice:
    """Provider-neutral notice appended to model input."""

    kind: Literal["limit_warning"]
    content: str
    metadata: Json = Field(default_factory=dict)
```

Update `ModelSession.start(...)` and `ModelSession.continue_with_tools(...)`:

```python
def start(..., notices: list[ModelNotice] | None = None) -> ModelTurn: ...
def continue_with_tools(..., notices: list[ModelNotice] | None = None) -> ModelTurn: ...
```

Keep this type provider-neutral and non-hook-specific. Do not add a public model-request hook or expose provider-specific message structures to hook handlers.

`ModelNotice` is public because it appears in the `ModelSession` protocol. Export it from `thinharness/__init__.py` alongside `ToolOutput`, `ModelTurn`, and `ModelToolCall`.

Files: `thinharness/providers.py`, fake sessions in `tests/test_harness.py`, `thinharness/__init__.py`.

**Verify:** Protocol implementers and fake sessions accept `notices=None`; all existing tests still pass before notices are generated.

### 2. Define Harness Limit Warning Policy
Add a harness-side helper in `thinharness/core.py` that computes notices immediately before every logical provider request, after hard-limit checks and before `session.start(...)` or `session.continue_with_tools(...)`.

Use simple v1 warning rules:

- If `max_model_requests - usage.model_requests == 1`, send one `limit_warning` notice telling the model this is the final allowed model request and it must produce a final answer now.
- If `max_tool_calls is not None` and `max_tool_calls - usage.tool_calls == 1`, send one `limit_warning` notice telling the model only one tool call remains and it should avoid tool fan-out.
- If `max_tool_calls is not None` and `max_tool_calls - usage.tool_calls == 0`, send one `limit_warning` notice telling the model no tool calls remain and it must answer without requesting tools.

Deduplicate each warning kind once per run:

```python
emitted_limit_warnings: set[str] = set()

def limit_notices() -> list[ModelNotice]:
    notices: list[ModelNotice] = []
    if remaining_model_requests == 1 and "final_model_request" not in emitted_limit_warnings:
        notices.append(ModelNotice(
            kind="limit_warning",
            content="This is the final allowed model request for this run. Provide the final answer now and do not request more tools.",
            metadata={"limit_kind": "model_requests", "remaining": 1},
        ))
        emitted_limit_warnings.add("final_model_request")
    if remaining_tool_calls == 0 and "no_tool_calls_remaining" not in emitted_limit_warnings:
        notices.append(ModelNotice(
            kind="limit_warning",
            content="No tool calls remain for this run. Provide the final answer without requesting tools.",
            metadata={"limit_kind": "tool_calls", "remaining": 0},
        ))
        emitted_limit_warnings.add("no_tool_calls_remaining")
    elif remaining_tool_calls == 1 and "one_tool_call_remaining" not in emitted_limit_warnings:
        notices.append(ModelNotice(
            kind="limit_warning",
            content="Only one tool call remains for this run. Avoid tool fan-out and request at most one tool only if it is necessary.",
            metadata={"limit_kind": "tool_calls", "remaining": 1},
        ))
        emitted_limit_warnings.add("one_tool_call_remaining")
    return notices
```

Check `remaining_tool_calls == 0` before `remaining_tool_calls == 1` so an exhausted budget produces the stronger no-tools warning. The one-tool warning cannot prevent every same-batch overage because a model can still request multiple tools in a single response, but it gives the model guidance before the next opportunity to fan out.

Do not add `limit_warnings` config knobs in v1. The warnings are deterministic consequences of configured hard limits. Do not emit warning hooks; existing `limit_reached` remains the observer event for actual hard failures.

Files: `thinharness/core.py`.

**Verify:** `max_model_requests=1` immediate-final runs send a final-request notice on `session.start(...)`; `max_model_requests=2` tool-then-final runs send the final-request notice only on `continue_with_tools(...)`; `max_tool_calls=1` sends the one-tool warning on the initial request; `max_tool_calls=1` after one requested tool sends the no-tools warning on continuation; warnings are emitted at most once per kind per run; warning-only runs do not fire `limit_reached`, do not emit a new hook event, and do not add a `HarnessConfig.limit_warnings` field.

### 3. Render Notices Through a Single Helper
Add one helper in `thinharness/providers.py` to render notices into deterministic text. Keep the exact wrapper centralized so provider tests can assert payload shape.

Suggested format:

```text
<harness_notice kind="limit_warning">
This is the final allowed model request for this run. Provide the final answer now and do not request more tools.
</harness_notice>
```

If multiple notices are present, render them in order separated by a blank line. Do not reuse `<hook_context>` because notices are harness-generated guidance, not hook-supplied prompt context.

Files: `thinharness/providers.py`.

**Verify:** Unit tests assert exact rendered notice text for one notice and two notices.

### 4. Define Prompt Composition Order
Prompt text may already include hook-provided context from `apply_prompt_context(...)`. Notices must be appended after that combined prompt so the full initial request order is:

```text
{original_prompt}

<hook_context>
...
</hook_context>

<harness_notice kind="limit_warning">
...
</harness_notice>
```

When no hook context is present, append notices directly after the original prompt separated by one blank line. Keep hook context construction in `thinharness/hooks.py` and notice construction in `thinharness/providers.py`; the harness should pass the already hook-augmented prompt plus notice objects into `session.start(...)`.

Files: `thinharness/core.py`, `thinharness/hooks.py`, `thinharness/providers.py`, `tests/test_harness.py`.

**Verify:** A test with both `user_prompt_submit` additional context and a final-request notice asserts the exact initial prompt string order and separators.

### 5. Inject Notices in OpenAI Responses Payloads
Update `OpenAIResponsesSession.start(...)` and `continue_with_tools(...)`.

Initial request: append the rendered notice text after the prompt string, using the same deterministic pattern as other provider-neutral prompt additions:

```python
input_payload = _append_notices_to_text(prompt, notices)
```

Continuation request: preserve the invariant that tool outputs come first. Append a text input item after all `function_call_output` items:

```python
input_payload = [
    {"type": "function_call_output", "call_id": output.call_id, "output": output.output}
    for output in outputs
]
if notice_text:
    input_payload.append({
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": notice_text}],
    })
```

This exact continuation item shape was verified with a live OpenAI Responses API probe using `.env` credentials: a `message` item containing an `input_text` content block is accepted after `function_call_output` items. The shorter `{"role": "user", "content": notice_text}` shape was also accepted by the live API, but use the explicit `type="message"` shape because it is less ambiguous. A bare `{"type": "input_text", "text": ...}` item was rejected and must not be used.

Files: `thinharness/providers.py`, `tests/test_harness.py`.

**Verify:** OpenAI fake-client tests assert the first payload input includes notice text for initial warnings; continuation payloads assert all `function_call_output` items precede the notice item and `previous_response_id` behavior is unchanged.

### 6. Inject Notices in Anthropic Messages Payloads
Update `AnthropicMessagesSession.start(...)` and `continue_with_tools(...)`.

Initial request: append rendered notice text to the first user message content string.

Continuation request: keep tool results first inside the user message content list, then append a text block:

```python
content = [{"type": "tool_result", "tool_use_id": output.call_id, "content": output.output} for output in outputs]
if notice_text:
    content.append({"type": "text", "text": notice_text})
self.messages.append({"role": "user", "content": content})
```

Files: `thinharness/providers.py`, `tests/test_harness.py`.

**Verify:** Anthropic payload tests assert notice text appears after all `tool_result` blocks on continuation and does not disturb assistant/tool history.

### 7. Inject Notices in OpenRouter Chat Payloads
Update `OpenRouterSession.start(...)` and `continue_with_tools(...)`.

Initial request: append rendered notice text to the initial user message content string.

Continuation request: append every tool message first, then append a user message containing rendered notice text:

```python
for output in outputs:
    self.messages.append({"role": "tool", "tool_call_id": output.call_id, "content": output.output})
if notice_text:
    self.messages.append({"role": "user", "content": notice_text})
```

Files: `thinharness/providers.py`, `tests/test_harness.py`.

**Verify:** OpenRouter payload tests assert all `role="tool"` messages precede the notice user message and existing assistant/tool message ordering remains intact.

### 8. Wire Notices Into Harness Model Requests
Call the helper from `Harness.run(...)` immediately before `session.start(...)` and `session.continue_with_tools(...)`.

Ordering inside each model request should be:

1. Check hard model-request limit.
2. Compute near-limit notices from current `RunUsage`.
3. Open the model trace span.
4. Call `session.start(..., notices=notices)` or `session.continue_with_tools(..., notices=notices)`.
5. Increment `usage.model_requests` after the request succeeds.

For continuation after a tool batch, `usage.tool_calls` has already been incremented, so the no-tool-calls-remaining warning reflects the budget state the model is about to face.

Do not count notices as model requests or tool calls. Do not add new trace spans or tracing attributes for notices in v1; current tracing records model responses/completions, not request payloads.

Files: `thinharness/core.py`.

Update fake `ScriptedSession.start(...)` and `continue_with_tools(...)` signatures to accept `notices=None`. Also update their `on_start` and `on_continue` callback contracts so tests can inspect notices consistently:

```python
def on_start(prompt, instructions, tools, metadata, previous_response_id, notices): ...
def on_continue(outputs, tools, metadata, notices): ...
```

Files: `thinharness/core.py`, `tests/test_harness.py`.

**Verify:** A fake `ScriptedSession` captures notice objects and confirms the harness passes them before the final allowed provider request and after tool budget reaches zero; a combined-warning case such as `max_model_requests=2` and `max_tool_calls=1` confirms tool outputs precede notice text and notices render in deterministic order with the final-model-request warning before the tool-call warning.

### 9. Update Docs and Manual Integration Notes
Update README with a short paragraph under limits explaining that the harness may add model-facing limit notices before hard limits are reached. Document that this is deterministic harness behavior, separate from lifecycle hooks, and that parent and child runs receive local notices from their own budgets.

Add a manual integration section or developer note for live provider checks gated by provider API keys. The repo does not currently depend on `python-dotenv`, so `.env` support means shell-loading the file before running the tests:

```bash
set -a; source .env; set +a
uv run --extra dev pytest -q tests/test_harness.py -k openai_notice_payload_live
uv run --extra dev pytest -q tests/test_harness.py -k anthropic_notice_payload_live
uv run --extra dev pytest -q tests/test_harness.py -k openrouter_notice_payload_live
```

Keep live tests skipped unless the relevant API key is present in the process environment. Do not add a dotenv dependency just for this plan unless the project adopts dotenv elsewhere.

Files: `README.md`, optionally `tests/test_harness.py`.

**Verify:** README mentions near-limit notices; no docs describe notices as hooks; skipped live tests remain skipped without provider API keys.

## Considerations
- The warning text should be short and directive. Long warnings consume the very budget they are trying to conserve.
- The v1 threshold warns at the final model request, one remaining tool call, and no remaining tool calls. This still cannot prevent all same-batch tool overages because the model may request multiple tools in one response.
- Provider adapters must preserve continuation ordering. Tool outputs always precede notice text so providers can correlate tool results before reading additional guidance.
- Notices are local to each harness run. Subagents receive their own notices based on their own inherited or overridden limits; there is still no shared global budget.
- Do not add raw model-request hooks as part of this plan. Hooks can observe lifecycle and hard `limit_reached` events, while near-limit notices are internal model input shaping.
- If a provider rejects the proposed OpenAI Responses notice input item shape, adjust only the OpenAI adapter and tests while preserving the provider-neutral `ModelNotice` API and the tool-output-before-notice invariant.
