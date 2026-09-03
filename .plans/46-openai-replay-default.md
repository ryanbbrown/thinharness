# OpenAI replay default

### Goal

After this change, the OpenAI Responses adapter sends the complete ordered conversation as client-managed input on every request by default, with `store:false` and no `previous_response_id`. The replayed items are the exact raw Responses items the API returned (reasoning items with `encrypted_content` and `id`, assistant messages with `id`, `status`, and `phase`, `function_call` items with `id` and `call_id`) plus the exact input items the adapter sent (user messages and `function_call_output` items). Server-side `previous_response_id` continuation remains available only as an explicit opt-in mode with its current behavior. Resume state carries the raw OpenAI item history so a crash resume, an approval resume, or a fork produces the same next request as an uninterrupted run. Malformed item histories are rejected before any provider call. The plan does not authorize any automatic retry of a `cyber_policy` refusal through the other state mode, any compaction or truncation of replayed history, or any change to the Anthropic or OpenRouter adapters beyond the shared resume envelope.

Evidence and scope come from the research report at `/Users/ryanbrown/.bb/thread-storage/thr_eyt3rrumv4/comparison-harness-research/report.md`: one preserved context was blocked twice under continuation and accepted under exact replay, and Pi, DeepSeek Harness, and Pydantic AI all default to client-managed history.

### Current state

- `thinharness/providers/openai.py`: `OpenAIResponsesSession` stores the returned response id and sends `previous_response_id` on every later request. It sends only the new items per request. It never sets `store`. `resume_session` replays a rendering of the neutral transcript once (`_render_openai_transcript`), then returns to id chaining.
- `thinharness/providers/openai.py`: `_render_openai_transcript` rebuilds Responses items from the neutral transcript. It drops assistant message `id`, `status`, and `phase`, drops `function_call` item `id`, and rebuilds reasoning items from `ReasoningPart`. It is the only OpenAI wire history today, so it must stay for cross-provider and non-reasoning-model resume, but it is not exact.
- `thinharness/providers/transcript.py`: the neutral resume envelope is `{"kind": "transcript", "version": 4, "origin_provider", "origin_model", "entries"}`. `_validate_resume_state` rejects unknown keys and any other version with a regenerate error (`docs/behavior.md` RESUME-4).
- `thinharness/runtime.py:403` calls `dump_state()` at an approval pause, after an assistant turn whose `function_call` items have no outputs yet. `thinharness/core.py:665` resumes that state and then continues with tool outputs. A trailing unanswered call batch is therefore a valid history shape.
- `thinharness/core.py:209` and `thinharness/providers/__init__.py:44`: `HarnessConfig` builds the model through `infer_model`. Callers who need a non-default model construct it and pass `Harness(config, model=...)`.
- `thinharness/providers/base.py:166`: `ModelSession.start` accepts `previous_response_id`; Anthropic and OpenRouter sessions raise `ProviderError` when it is given.
- `tests/unit/fakes.py` `FakeClient` and `tests/unit/test_reasoning_fidelity.py` `ReasoningOpenAIProvider` return minimal output items without `id`, `status`, or `phase`.

### Decisions

- `OpenAIResponsesModel` gains a constructor keyword `state_mode: Literal["replay", "continuation"] = "replay"`. Any other value raises `ValueError` at construction. Every `OpenAIResponsesSession` created by that model uses the model's mode. `HarnessConfig` and `infer_model` do not expose the mode; an operator who wants continuation constructs the model and passes it to `Harness(config, model=...)`.
- In replay mode the constructor rejects `settings.extra_body` that contains `input`, `previous_response_id`, or `store` with `ValueError`, so no request can bypass replay. Continuation mode does not reserve these keys.
- In replay mode, the session keeps an ordered list of raw Responses items called the item history. Each request's `input` is a new list built from the item history followed by the new items for that request; the session never passes its own history list object as the payload. The new items are appended to the history before the provider call, at the same point the neutral transcript is appended, so both lists stay aligned when the call fails. After a successful response, a deep copy of every `response["output"]` item is appended unchanged. Only `response["output"]` contributes assistant items; `output_text` is never turned into an item. Each request sets `store: False` and never sets `previous_response_id`. Instructions, tools, metadata, `include`, and settings are built exactly as today.
- In replay mode, every user turn, including the first prompt and later `continue_with_user_content` calls, is sent as a `message` item inside the list; replay mode never sends scalar text input. Continuation mode keeps sending scalar text input for all-text prompts.
- In replay mode, a response that contains a `reasoning` item without `encrypted_content` raises `ProviderError` after the response is received and before any later request. The message names the two remedies: add the model family to `_openai_supports_encrypted_reasoning`, or use `state_mode="continuation"`. Nothing is dropped silently.
- In replay mode, `start(..., previous_response_id=...)` with a non-`None` value raises `ProviderError`. Continuation mode keeps honoring it as today.
- `store: False` is applied inside `build_payload` when the session passes it. A direct `build_payload` call without it adds no `store` key. Because replay mode reserves `store` in `extra_body`, every replay request sends `store: false`.
- Continuation mode keeps its current behavior unchanged: id chaining, scalar text input, no `store` key, one-time neutral transcript replay on resume, and a neutral-only resume dump.
- The neutral resume envelope moves to `version` 5 and gains one optional key, `openai_items`, the raw item history list. Only OpenAI replay-mode sessions emit it; Anthropic, OpenRouter, and OpenAI continuation-mode envelopes omit the key entirely and their dump code does not change. Version 4 state is rejected with the existing regenerate error; no migration is written, because RESUME-4 already declares that older state is rejected.
- On resume in replay mode, the session uses `openai_items` verbatim as its item history when it is present and the resuming model can accept it: the list contains no `reasoning` item, or `_openai_supports_encrypted_reasoning(model)` is true. Otherwise the session builds its initial item history from `_render_openai_transcript(entries)` exactly as the existing RESUME-3 text-degradation path does. In both cases the session also restores the neutral transcript from `entries`, so a later dump still supports cross-provider resume.
- `_validate_resume_state` keeps its current signature and return value (the entries). It accepts the optional `openai_items` key, requires a list when present, and validates the history before any session is created or any provider call is made, raising `HarnessError` with a `resume_from` prefix like the other checks. Anthropic and OpenRouter call it unchanged and ignore the key. OpenAI `resume_session` reads the validated history from the state after the call. The history is rejected when any of these holds: an element is not a dict or lacks a string `type`; a `function_call_output` has no earlier unanswered `function_call` with the same `call_id` (this covers dropped calls, duplicated outputs, and an output placed before its call); a `function_call` is still unanswered when a later user-role `message` item or a later `reasoning` item appears; a `reasoning` item is the last element or is immediately followed by a user-role `message` or a `function_call_output`. Unanswered `function_call` items at the tail are accepted because an approval pause dumps state in that shape.
- `_extract_responses_tool_calls` recognizes only `function_call` items and uses `call_id` as the call id, so every extracted call pairs with its `function_call_output` under the rules above.
- Forks work by resuming the same state dict into separate sessions. `resume_session` deep-copies both the entries and the item history list, so neither session mutates the state or the other session.
- No automatic fallback or retry between modes exists anywhere, including after a `cyber_policy` refusal.

### Changes

### 1. Replay-mode session and opt-in continuation

- **Files**: modify `thinharness/providers/openai.py`; modify `tests/unit/fakes.py`; modify `tests/unit/test_provider_openai.py`; create `tests/unit/test_provider_openai_replay.py`; update every test that asserts `previous_response_id` chaining or delta-only second requests, including `tests/unit/test_harness.py`, `tests/unit/test_provider_transport.py`, `tests/unit/test_parallel_tools.py`, `tests/unit/test_bash_plugin.py`, `tests/unit/test_tool_retry.py`, `tests/unit/test_streaming.py`, `tests/unit/test_subagents.py`, `tests/unit/test_hooks.py`, `tests/unit/test_mcp.py`, `tests/unit/test_parallel_llm.py`, `tests/unit/test_approvals.py`, `tests/unit/test_architecture.py`, and `tests/unit/test_image_inputs.py`.
- **Change**: `OpenAIResponsesModel` stores `state_mode` and passes it to each session. `OpenAIResponsesSession` holds the item history and, in replay mode, builds every request from it as described in Decisions, records new input items and raw output items after each `_complete`, sets `store: False` through `build_payload`, and rejects a seeded `previous_response_id`. Continuation mode is the existing code path selected by mode. `dump_state` passes the item history to the envelope builder from unit 2 in replay mode and omits it in continuation mode. `tests/unit/fakes.py` gains one realistic Responses fixture provider whose scripted responses carry reasoning items with `id`, `summary`, and `encrypted_content`, assistant messages with `id`, `status`, `phase`, and `content`, and `function_call` items with `id`, `call_id`, `status`, `name`, and `arguments`, across at least two tool rounds and a final text response. All OpenAI fakes deep-copy recorded payloads. `FakeClient`, `MultiCallClient`, and the reasoning-fidelity and resume fakes return their final text turn as an assistant `message` item inside `output` instead of `output_text` only; other `output_text`-only stubs change only where their test asserts replayed history content. Existing tests that assert `previous_response_id` chaining either assert the replay shape or move to `state_mode="continuation"` when the test is about continuation; tests that assert scalar `input` are updated for replay's list form.
- **Tests**:
  - Uninterrupted replay run through start, two `continue_with_tools`, and a final `continue_with_user_content`: every payload has `store` false, no `previous_response_id`, identical `instructions`, `tools`, `metadata`, and `include`, and `input` equals the full prior item history plus the new items in order.
  - The replayed reasoning, message, and function_call items in a later payload are deep-equal to the raw output items the fixture returned, including `id`, `encrypted_content`, `status`, and `phase`.
  - A replay-mode request with a tool output that carries `wire_output` replays that exact text, and a notice appends a trailing user message item after the outputs.
  - Replay mode with a non-reasoning model name sends no `include` key and still sets `store` false.
  - `start(..., previous_response_id="x")` in replay mode raises `ProviderError` and makes no provider call.
  - `state_mode="continuation"` reproduces today's chain: `previous_response_id` on later requests, scalar text input, no `store` key, and one-time transcript replay on resume.
  - `build_payload` without a store value adds no `store` key.
  - Replay-mode construction with `extra_body` containing `store`, `input`, or `previous_response_id` raises `ValueError`; continuation mode accepts the same `extra_body`.
  - An invalid `state_mode` raises `ValueError` at construction.
  - A replay-mode response with a `reasoning` item lacking `encrypted_content` (model name outside the detection list) raises `ProviderError` and no later request is sent.
  - A provider failure on the second request leaves `dump_state()` entries and `openai_items` aligned (both include the tool outputs).

### 2. Resume envelope with raw item history and validation

- **Files**: modify `thinharness/providers/transcript.py`; modify `thinharness/providers/openai.py`; `thinharness/providers/anthropic.py` and `thinharness/providers/openrouter.py` stay unchanged; modify `tests/unit/test_provider_openai_replay.py`; modify `tests/unit/test_resume.py`; modify `tests/unit/test_reasoning_fidelity.py`; modify `tests/unit/test_provider_transcript.py`; modify `tests/unit/test_approvals.py`; modify `tests/unit/test_image_inputs.py` (version assertions at lines 231, 374, 416).
- **Change**: the envelope builder emits `version` 5 and includes `openai_items` only when a raw item history is supplied. The validator accepts the key as optional, requires a list when present, runs the item-history rules from Decisions, and keeps returning the entries. Anthropic and OpenRouter ignore the key when resuming. OpenAI `resume_session` applies the acceptance rule from Decisions to choose between the raw history and the rendered neutral transcript, deep-copies both, and restores the neutral transcript. Tests that assert `version == 4` assert 5, and the rejected-version parametrization includes 4.
- **Tests**:
  - Crash resume: dump state after the first tool round, JSON round-trip it, resume in a new model, and continue with the same outputs; the next payload equals the uninterrupted run's corresponding payload exactly.
  - Approval-shaped resume: state dumped after an assistant response with unanswered `function_call` items validates and, after `continue_with_tools`, produces the same payload as the uninterrupted run.
  - Follow-up resume: state dumped after the final text response, resumed with `continue_with_user_content`, sends the full raw history followed by the new user message item.
  - Fork: resume the same state dict into two sessions and continue each with different outputs; both payloads share the identical raw prefix, differ only in the new items, and the state dict is unchanged after both runs.
  - Reasoning-model state resumed on a non-reasoning OpenAI model uses the rendered transcript with `<thinking>` text and no `reasoning` item (existing `test_openai_cross_model_falls_back_to_text` continues to pass).
  - Continuation-mode state omits `openai_items` and resumes in replay mode through the rendered transcript.
  - Anthropic and OpenRouter envelopes omit `openai_items`; an envelope with `openai_items: null` is rejected as a wrong type.
  - Cross-provider: OpenAI replay-mode state resumed on Anthropic and OpenRouter still renders from `entries` (existing tests).
  - A version 5 OpenAI envelope with valid `openai_items` resumes on Anthropic and OpenRouter from `entries`.
  - Malformed `openai_items` inside an approval envelope fails with the `approval state provider_state` prefix.
  - Malformed histories, each rejected with `HarnessError` before any provider call: output placed before its call; output for a call that was dropped; duplicated output for one call; a call left unanswered before a later reasoning item; a call left unanswered before a later user message; reasoning item last; reasoning item followed by a `function_call_output`; a non-dict element; an element without a string `type`.
  - Version 4 state is rejected with the regenerate error.
  - Harness-level: `test_openai_resume_full_replays_transcript_for_followup` asserts replay behavior (no `previous_response_id`, `store` false, full history in the second request).

### 3. Documentation

- **Files**: modify `docs/docs.md`; modify `CHANGELOG.md`; `docs/behavior.md` is updated by the orchestrator before implementation and is not edited by the writer.
- **Change**: `docs/docs.md` resume section states version 5, describes `openai_items`, describes the replay default and the `state_mode="continuation"` opt-in, and replaces the `previous_response_id` escape-hatch bullet with the continuation-mode statement. `CHANGELOG.md` gains an entry for the replay default, the continuation opt-in, and the breaking version 5 envelope.
- **Tests**: none.

### Acceptance checks

- A default `OpenAIResponsesModel` session sends `store: false`, never sends `previous_response_id`, and sends the full ordered raw item history plus new items on every request.
- Replayed items are deep-equal to the raw output items received, including `encrypted_content`, item `id`, `call_id`, `status`, and `phase`.
- OpenAI replay resume state is version 5 with `openai_items`; other envelopes omit the key; crash resume, approval-shaped resume, and forks produce the same next request as an uninterrupted run.
- Malformed, reordered, or incomplete item histories raise `HarnessError` before any provider call.
- `OpenAIResponsesModel(..., state_mode="continuation")` reproduces the previous chaining behavior.
- No code path retries a request through the other state mode.

### Validation

```bash
uv run pytest tests/unit -q
uv run ruff check thinharness tests
uv run pyright
```

### Deferred

- Recording the raw item history in continuation mode so continuation captures resume exactly in replay mode.
- Exposing the state mode through `HarnessConfig` or `infer_model`.
- Compaction of long replayed histories.
- Rejecting a raw history whose `origin_model` differs from the resuming model; the provider reports incompatible items as an API error today.
- Dropping a bare trailing `reasoning` item from a truncated response instead of rejecting the state on resume; no such response has been observed in this repository.
- Sending `prompt_cache_key` with replayed requests.
