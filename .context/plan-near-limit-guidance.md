# Plan: Near-Limit Guidance

## Overview
Add provider-neutral model notices so the harness can warn the model before local run limits are exhausted. The current code has hard `max_model_requests`, `max_tool_calls`, and `tool_retries` enforcement, and `decisions.md` still records near-limit guidance as deferred; this plan updates the older design for the current four-method `ModelSession` protocol, resume flow, structured-output retries, shared test fakes, and provider-specific payload tests.

## Current State
- `thinharness/core.py` enforces hard limits in `check_model_limit()` and `check_tool_limit()`, then sends provider requests through `advance_model(...)`.
- `ModelSession` now has four request methods: `start(...)`, `continue_with_tools(...)`, `continue_with_user_message(...)`, and `continue_with_user_prompt(...)`.
- Structured-output retry paths use `continue_with_tools(...)` for invalid `final_result` tool calls and `continue_with_user_message(...)` for text-only or invalid prompted/native output.
- Resume uses `continue_with_user_prompt(...)` for the first request of a resumed run.
- Shared fakes live in `tests/fakes.py`, but several tests also define local fake sessions that will need signature updates once the harness passes notices.
- Provider payload coverage is concentrated in `tests/test_providers.py`; harness loop behavior is mostly covered in `tests/test_harness.py`, `tests/test_hooks.py`, `tests/test_resume.py`, and `tests/test_structured_output.py`.
- `UserPromptSubmitContext.additional_context` already injects hook-owned text into the initial prompt only; notices must stay separate because they are harness-owned and can fire before any provider request, including continuations and resumed prompts.
- Notices are sent as user input, not instructions. That keeps model-facing guidance close to the turn it affects and preserves provider prompt-caching behavior for system/instruction text.

## Steps

### 1. Add Provider-Neutral Notice Types
Define a small provider-facing notice type in `thinharness/providers.py` and extend every `ModelSession` request method to accept notices.

```python
from dataclasses import dataclass
from typing import Literal

@dataclass(frozen=True)
class ModelNotice:
    """Provider-neutral notice appended to model input.

    limit_kind and remaining are populated for kind="limit_warning".
    """

    kind: Literal["limit_warning"]
    content: str
    limit_kind: Literal["model_requests", "tool_calls"] | None = None
    remaining: int | None = None
```

Keep `kind` as `Literal["limit_warning"]` in v1 for type clarity, but document in `decisions.md` that the public union may grow when new notice categories are introduced. Keep `limit_kind` and `remaining` as first-class fields instead of burying them in `metadata`, because dedup depends on them and should not silently break on dict key typos.

Update `ModelSession` so every method that can trigger a provider request has `notices: list[ModelNotice] | None = None`:

```python
async def start(..., notices: list[ModelNotice] | None = None) -> ModelTurn: ...
async def continue_with_tools(..., notices: list[ModelNotice] | None = None) -> ModelTurn: ...
async def continue_with_user_message(..., notices: list[ModelNotice] | None = None) -> ModelTurn: ...
async def continue_with_user_prompt(..., notices: list[ModelNotice] | None = None) -> ModelTurn: ...
```

Keep `ModelNotice` provider-neutral and non-hook-specific. Do not add raw model-request hooks or expose provider message structures to hooks.

Use explicit per-call kwargs rather than session-level pending state. This widens the protocol, but it keeps request construction explicit and testable, and avoids hidden adapter state that can leak between provider calls.

Export `ModelNotice` from `thinharness/__init__.py` alongside `ToolOutput`, `ModelTurn`, and `ModelToolCall`.

Files: `thinharness/providers.py`, `thinharness/core.py`, `thinharness/__init__.py`, `tests/fakes.py`, local fake sessions in `tests/test_harness.py`, `tests/test_hooks.py`, `tests/test_mcp.py`, `tests/test_resume.py`, and `tests/test_tool_retry.py`.

**Verify:** `uv run --extra dev pytest -q` still passes after adding optional `notices=None` parameters but before generating any notices.

### 2. Update Test Fakes
Update `tests/fakes.py::ScriptedSession` and local fake sessions to accept notices before changing harness behavior.

Use one test-only capture list:

```python
self.notice_calls: list[tuple[str, list[ModelNotice]]] = []
```

Append one entry for every fake session request, even when `notices is None`:

```python
self.notice_calls.append(("start", list(notices or [])))
self.notice_calls.append(("continue_with_tools", list(notices or [])))
self.notice_calls.append(("continue_with_user_message", list(notices or [])))
self.notice_calls.append(("continue_with_user_prompt", list(notices or [])))
```

Keep existing `on_start(prompt, instructions, tools, metadata, previous_response_id)` and `on_continue(outputs_or_message, tools, metadata)` callback arity for compatibility. Tests that need notices should inspect `notice_calls` instead of changing every existing callback.

Adding notices will create expected assertion churn in existing tests that use `max_model_requests=1`, especially provider payload assertions and fake-session call assertions. Prefer asserting `notice_calls` for harness policy tests and use provider payload tests only for rendered adapter shape.

Files: `tests/fakes.py`, local fake classes in `tests/test_harness.py`, `tests/test_hooks.py`, `tests/test_mcp.py`, `tests/test_resume.py`, `tests/test_tool_retry.py`.

**Verify:** Existing tests do not fail due to callback arity changes, and fake calls record method names plus an empty list when no notices were passed.

### 3. Render Notices Centrally
Add provider helpers in `thinharness/providers.py` so all adapters render notices consistently.

```python
def render_model_notices(notices: list[ModelNotice] | None) -> str:
    """Render provider-neutral notices as deterministic text."""
    if not notices:
        return ""
    return "\n\n".join(
        f'<harness_notice kind="{notice.kind}">\n{notice.content}\n</harness_notice>'
        for notice in notices
    )


def append_notices_to_text(text: str, notices: list[ModelNotice] | None) -> str:
    """Append rendered notices to provider text input."""
    notice_text = render_model_notices(notices)
    return text if not notice_text else f"{text}\n\n{notice_text}"
```

Use XML-like tags because the project already uses tagged context blocks, models handle this shape predictably, and the wrapper gives provider tests a deterministic boundary to assert. Do not reuse `<hook_context>`; notices are harness-generated guidance, while hook context remains owned by `thinharness/hooks.py`. Notice content is harness-controlled in v1, so this helper does not escape XML-like characters; if future notice content includes user data, add escaping or a different serialization format first.

Files: `thinharness/providers.py`, `tests/test_providers.py`.

**Verify:** Unit tests assert exact rendered output for one notice, multiple notices separated by one blank line, and no notices returning an empty string / unchanged text.

### 4. Define Harness Warning Policy
Add a small helper near the run-loop helpers in `thinharness/core.py` that computes notices immediately before every logical provider request, after `check_model_limit()` and before entering the model trace span.

Use simple v1 rules:
- If `max_model_requests - usage.model_requests == 1`, emit a final-model-request warning.
- If `max_tool_calls is not None` and `max_tool_calls - usage.tool_calls == 1`, emit a one-tool-call warning.
- If `max_tool_calls is not None` and `max_tool_calls - usage.tool_calls == 0`, emit a no-tool-calls warning.

`check_model_limit()` still runs before notice computation. A config with `max_model_requests=0` should hard-reject before any notice is computed or provider request is made.

The first-request `max_model_requests=1` warning is intentional. It tells the model this is a single-shot run and should be tested as normal behavior, not treated as an edge case to suppress.

Make warning text conditional on `output_schema.mode == "tool"`. Only tool-mode structured output exposes the synthetic `final_result` tool; native and prompted structured output must not mention `final_result`.

The `final_result` exception depends on the current structured-output invariant: synthetic `final_result` calls are handled before the ordinary tool budget path and do not increment `usage.tool_calls`. Pin this with a regression test. If a future refactor makes `final_result` consume `max_tool_calls`, the warning text must change because telling the model to call `final_result` with no tool budget would be contradictory.

```python
LimitNoticeKey = tuple[Literal["limit_warning"], Literal["model_requests", "tool_calls"], int]

def _limit_notice_dedup_key(notice: ModelNotice) -> LimitNoticeKey:
    """Return the once-per-run key for a model notice."""
    assert notice.limit_kind is not None and notice.remaining is not None
    return (notice.kind, notice.limit_kind, notice.remaining)

def _append_notice_once(notices: list[ModelNotice], emitted: set[LimitNoticeKey], notice: ModelNotice) -> None:
    """Append a notice once per run."""
    key = _limit_notice_dedup_key(notice)
    if key in emitted:
        return
    notices.append(notice)
    emitted.add(key)

def _compute_limit_notices(config: HarnessConfig, usage: RunUsage, emitted: set[LimitNoticeKey], *, final_result_tool_available: bool) -> list[ModelNotice]:
    """Return model-facing warnings for the current run budget state."""
    notices: list[ModelNotice] = []
    final_model_text = (
        "Final request: produce the answer now; only call final_result if required."
        if final_result_tool_available
        else "Final request: produce the answer now; do not request tools."
    )
    remaining_model_requests = config.max_model_requests - usage.model_requests
    if remaining_model_requests == 1:
        _append_notice_once(notices, emitted, ModelNotice(
            kind="limit_warning",
            content=final_model_text,
            limit_kind="model_requests",
            remaining=1,
        ))

    if config.max_tool_calls is None:
        return notices
    remaining_tool_calls = config.max_tool_calls - usage.tool_calls
    if remaining_tool_calls == 0:
        no_tools_text = (
            "Tool calls are not available on this run; only call final_result if required."
            if final_result_tool_available and config.max_tool_calls == 0
            else "No tool calls remain: answer now; only call final_result if required."
            if final_result_tool_available
            else "Tool calls are not available on this run; answer without tools."
            if config.max_tool_calls == 0
            else "No tool calls remain: answer now without tools."
        )
        _append_notice_once(notices, emitted, ModelNotice(
            kind="limit_warning",
            content=no_tools_text,
            limit_kind="tool_calls",
            remaining=0,
        ))
    elif remaining_tool_calls == 1:
        tool_phrase = "non-final_result tool" if final_result_tool_available else "tool"
        _append_notice_once(notices, emitted, ModelNotice(
            kind="limit_warning",
            content=f"One {tool_phrase} call remains: avoid fan-out.",
            limit_kind="tool_calls",
            remaining=1,
        ))
    return notices
```

Initialize `emitted_limit_warnings: set[LimitNoticeKey] = set()` near the top of each `Harness.run(...)` invocation, next to `usage = RunUsage()`. This is run-scoped state only: it is not part of `RunUsage`, is not persisted in `resume_state`, and each subagent's child `Harness.run(...)` gets its own fresh set.

Check `remaining_tool_calls == 0` before `remaining_tool_calls == 1` so an exhausted tool budget produces the stronger warning. Same-turn overages still produce a hard `limit_reached` before any continuation request, so the harness does not send a corrective no-tools notice for that over-budget batch. Dedup exists mainly for structured-output retry requests, which can call `advance_model(..., output_retry=True)` repeatedly without changing model/tool usage state; without dedup, the same near-limit notice would reappear on each corrective retry. Do not add `HarnessConfig.limit_warnings` knobs in v1; warnings are deterministic consequences of configured hard limits. Do not emit warning hooks; existing `limit_reached` remains the observer event for actual hard failures.

Subagent runs use only their own fresh `HarnessConfig` limits, whether inherited or overridden. Inheritance copies the configured limit value into the child run; it is not a slice of the parent's remaining budget. Do not propagate parent remaining budget into a child run, and do not warn a subagent about parent budget state.

Files: `thinharness/core.py`, `tests/test_harness.py`, `tests/test_subagents.py`.

**Verify:** Harness tests cover `max_model_requests=0` hard-rejecting before provider request, `max_model_requests=1` on initial start, `max_model_requests=2` on tool continuation, `max_tool_calls=1` on initial start, `max_tool_calls=0` on initial start, and `max_tool_calls=1` after one requested tool producing the no-tool warning on continuation. Tests also assert same-turn tool overage raises `limit_reached` without sending a continuation notice; `final_result` resolution does not increment `usage.tool_calls`; only tool-mode structured output mentions `final_result`; native, prompted, text, and unstructured runs do not; combined notices render in stable model-warning-then-tool-warning order for `max_model_requests=1` with `max_tool_calls=1`, and for `max_model_requests=1` with `max_tool_calls=0`; each warning key is emitted at most once per run, including across structured-output retries at the same usage state; warning-only runs do not fire `limit_reached`; subagents only see fresh child-budget notices; and `HarnessConfig` does not grow a notice toggle.

### 5. Pass Notices Through `advance_model`
Refactor `advance_model(...)` so it computes notices once for each provider request and passes them into the request call.

```python
async def advance_model(request, *, is_output_retry: bool = False) -> ModelTurn:
    """Run one provider request with limit, usage, and tracing ceremony."""
    check_model_limit()
    notices = _compute_limit_notices(
        self.config,
        usage,
        emitted_limit_warnings,
        final_result_tool_available=self.output_schema is not None and self.output_schema.mode == "tool",
    )
    if is_output_retry:
        usage.output_retries += 1
    with run_tracer.model(self.model) as model_span:
        advanced_turn = await request(notices)
        usage.model_requests += 1
        ...
```

Update every call site:

```python
turn = await advance_model(lambda notices: active_session.start(..., notices=notices))
turn = await advance_model(lambda notices: active_session.continue_with_user_prompt(..., notices=notices))
turn = await advance_model(lambda notices: active_session.continue_with_tools(..., notices=notices), is_output_retry=True)
turn = await advance_model(lambda notices: active_session.continue_with_user_message(..., notices=notices), is_output_retry=True)
turn = await advance_model(lambda notices: active_session.continue_with_tools(tool_outputs, ..., notices=notices))
```

The lambda call sites are acceptable in v1 because they preserve the current request structure. If implementation reads poorly, use an equivalent `advance_model(active_session.start, ..., notices=...)` helper shape, but keep the behavior: `advance_model` computes notices exactly once and injects them into the provider request.

For continuations after a normal tool batch, `usage.tool_calls` is already incremented before `advance_model(...)`, so the no-tool-calls-remaining warning reflects the state the model is about to face. For structured-output retries, `usage.output_retries` should keep its current semantics and notices should not count as output retries.

For tracing, add lightweight request attributes such as notice count and kinds to the existing model span; do not record notice content in traces by default because count/kinds are enough signal and full notice text adds noise.

Files: `thinharness/core.py`, `tests/fakes.py`, `tests/test_structured_output.py`, `tests/test_resume.py`.

**Verify:** Tests capture notice objects from `ScriptedSession` for start, tool continuation, structured-output correction via user message, invalid `final_result` correction via tool output, and resumed user prompt. Existing output retry and resume tests should still pass. Add one assertion that notice generation reads only model/tool usage in v1, so incrementing `usage.output_retries` before the corrective request does not change notice content.

### 6. Inject Notices in OpenAI Responses Payloads
Update `OpenAIResponsesSession.start(...)`, `continue_with_tools(...)`, `continue_with_user_message(...)`, and `continue_with_user_prompt(...)`.

Initial/user-message/user-prompt requests should append rendered notice text to the string input payload:

```python
input_payload = append_notices_to_text(prompt, notices)
payload = self.model.build_payload(input_payload=input_payload, ...)
```

For initial `start(...)`, notices must be appended after hook-provided prompt context because the harness already calls `apply_prompt_context(...)` before `session.start(...)`. The provider receives:

```text
{original prompt}

<hook_context>
...
</hook_context>

<harness_notice kind="limit_warning">
...
</harness_notice>
```

Universal ordering rule: notices always come last in the user input for that provider request. On initial `start(...)`, they come after hook-injected context. On resumed `continue_with_user_prompt(...)` and structured-output `continue_with_user_message(...)`, no prompt-submit hooks run, so the order is just request text followed by notices.

Tool continuation should preserve the invariant that function outputs come first, then append a user message item containing the notice text:

```python
input_payload = [
    {"type": "function_call_output", "call_id": output.call_id, "output": output.output}
    for output in outputs
]
notice_text = render_model_notices(notices)
if notice_text:
    input_payload.append({
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": notice_text}],
    })
```

Keep `previous_response_id` behavior unchanged: `start(...)` can accept an explicit `previous_response_id`, continuations include the current `previous_response_id` when one exists, and `continue_with_user_prompt(...)` should not gain new validation beyond the current resume-state validation path. `OpenAIResponsesModel.build_payload(...)` intentionally accepts both string input payloads and list-based tool continuation payloads.

Files: `thinharness/providers.py`, `tests/test_providers.py`, `tests/fakes.py`.

**Verify:** OpenAI fake-client tests assert initial input includes notice text, tool continuation payloads keep all `function_call_output` items before the notice message, corrective user messages append notices to the message string, resumed prompts preserve `previous_response_id`, and no-notice payloads are unchanged. Include a shared provider-ordering assertion that all continuation tool result items precede notice text.

### 7. Inject Notices in Anthropic Messages Payloads
Update `AnthropicMessagesSession.start(...)`, `continue_with_tools(...)`, `continue_with_user_message(...)`, and `continue_with_user_prompt(...)`.

Initial/user-message/user-prompt requests should append notices to the user text content. Tool continuation should keep all `tool_result` blocks first, then append a text block:

```python
content = [
    {"type": "tool_result", "tool_use_id": output.call_id, "content": output.output}
    for output in outputs
]
notice_text = render_model_notices(notices)
if notice_text:
    content.append({"type": "text", "text": notice_text})
self.messages.append({"role": "user", "content": content})
```

Files: `thinharness/providers.py`, `tests/test_providers.py`, `tests/test_resume.py`.

**Verify:** Anthropic payload tests assert notice text appears after all `tool_result` blocks on continuation, start/user-message/user-prompt text receives appended notices, the initial prompt order is original prompt, hook context, then notice, and assistant/tool history remains in the same order. Include the same provider-ordering assertion used for OpenAI. Resume tests should continue to serialize a JSON-safe transcript. Before considering the provider gate complete, run a small live Anthropic probe with a current Haiku model to confirm a user message containing tool_result blocks followed by a text notice block is accepted; if rejected, keep tool outputs first but send the notice as a second user message after the tool-result message.

### 8. Inject Notices in OpenRouter Chat Payloads
Update `OpenRouterSession.start(...)`, `continue_with_tools(...)`, `continue_with_user_message(...)`, and `continue_with_user_prompt(...)`.

Initial/user-message/user-prompt requests should append notices to the user message content. Tool continuation should append all tool messages first, then append a user notice message:

```python
for output in outputs:
    self.messages.append({"role": "tool", "tool_call_id": output.call_id, "content": output.output})
notice_text = render_model_notices(notices)
if notice_text:
    self.messages.append({"role": "user", "content": notice_text})
```

Files: `thinharness/providers.py`, `tests/test_providers.py`, `tests/test_resume.py`.

This tool-output-then-user-notice shape was validated live against OpenRouter on 2026-05-17 using `openai/gpt-4o-mini`, `anthropic/claude-3.5-haiku`, `anthropic/claude-3-haiku`, and `google/gemini-2.0-flash-001`. All accepted an assistant tool call followed by a tool result and synthetic user notice, then produced a normal final answer. The probe did not use `tool_choice`, which matches the current ThinHarness adapter.

**Verify:** OpenRouter payload tests assert all `role="tool"` messages precede the notice user message, start/user-message/user-prompt text receives appended notices, the initial prompt order is original prompt, hook context, then notice, existing assistant/tool ordering remains intact, and resume state stays JSON-safe. Include the same provider-ordering assertion used for OpenAI and Anthropic.

### 9. Update Docs and Decisions
Update public docs after behavior is implemented.

- Keep the existing deferred note in `decisions.md` for history and add a new implemented-rule bullet: hard limits remain authoritative, near-limit guidance is deterministic provider-facing input emitted shortly before configured limits are exhausted, and `ModelNotice.kind` is a public literal union that may grow in future releases.
- Add a short README paragraph near the harness limits / hooks discussion explaining that the harness may add model-facing limit notices, that these are not hooks, and that parent and child harness runs compute notices from their own local budgets only.
- If `docs.md` remains resume-focused, add one sentence noting that provider-visible notices are part of the conversation state returned by `resume_state`, because they were sent to the model.
- Document the v1 limitation plainly: near-limit notices may remain visible in resumed conversations because they were real model input, so their text is intentionally scoped to "this run".
- Document that notice dedup is per `Harness.run(...)`, not per conversation. A resumed run may re-emit a notice already visible in prior conversation history.
- Document that `tool_retries` near-limit guidance is deferred; retry exhaustion remains a hard failure only in this feature.
- Document that notice text is English-only and not user-configurable in v1.

Files: `README.md`, `docs.md`, `decisions.md`.

**Verify:** Docs do not describe notices as hook events, do not imply notices are configurable in v1, and clearly scope warning text to "this run" because historical notices may remain visible in resumed provider context.

### 10. Manual Provider Checks
Keep live provider checks optional and skipped unless the relevant API key is present. The repo does not depend on `python-dotenv`, so shell-load `.env` if needed:

```bash
set -a; source .env; set +a
uv run --extra dev pytest -q tests/test_providers.py -k openai_notice_payload_live
uv run --extra dev pytest -q tests/test_providers.py -k anthropic_notice_payload_live
uv run --extra dev pytest -q tests/test_providers.py -k openrouter_notice_payload_live
```

Do not add a dotenv dependency just for these checks.

Files: `tests/test_providers.py`, optionally `README.md` or a developer note if live tests are added.

**Verify:** Live tests skip cleanly without API keys, and mocked provider tests cover payload shape in normal CI. If live behavior tests are added, they should construct a harness with a near-limit configuration such as `max_model_requests=1`, send one prompt, and verify the model produces a final answer without requesting tools. OpenRouter live tests should keep the adapter's current no-`tool_choice` behavior.

## Considerations
- Notices are real model input. For stateless providers, they will be part of the local message transcript; for OpenAI Responses, they will be part of the server-side conversation behind `previous_response_id`. Do not try to strip them in v1; instead keep the text scoped to "this run".
- Dedup is per run, not per conversation. A resumed run may re-emit notices that are already present in the prior provider-visible conversation history.
- The v1 thresholds warn at final model request, one tool call remaining, and no tool calls remaining. Structured-output warning text distinguishes non-`final_result` tools from the synthetic `final_result` output tool. Same-turn overages are still possible because a model can request multiple tools in one response.
- Provider adapters must preserve continuation ordering. Tool outputs always precede notice text so providers can correlate tool results before reading additional guidance.
- Subagents receive notices only from their own inherited or overridden budgets. There is no shared global budget across parent and child runs, and child runs are not warned about the parent's remaining budget.
- `tool_retries` exhaustion remains a hard failure only. Do not add retry-budget notices in this plan unless a later design defines model-facing retry guidance.
- Notice text is English-only and not configurable in v1.
- Keep notice generation outside hooks. Hooks can observe lifecycle and hard `limit_reached` events; near-limit notices are internal model input shaping.
- If a provider rejects a proposed notice payload shape, adjust only that adapter and its tests while preserving the provider-neutral `ModelNotice` API and the tool-output-before-notice invariant.
