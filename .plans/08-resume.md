# Resume functionality plan (v5)

Adding multi-turn conversation continuation to thinharness — sending an additional user message that builds on prior `Harness.run()` calls.

Revised after feedback v1–v4 (see `.reviews/plans/resume/resume-plan-feedback-v{1..4}.md`), then trimmed in v5 to remove over-defensive layers found during self-review.

## Goal

A caller should be able to:

1. Call `result1 = await harness.run("first prompt")`.
2. Persist `result1.resume_state` (a JSON-serializable opaque dict) anywhere — memory, file, DB.
3. Later — possibly in a new process — call `result2 = await harness.run("follow-up", resume_from=state)` and have the model see the full prior exchange.

The wire format differs by provider, but the API is uniform:

- **OpenAI Responses** — server-side state. Send `previous_response_id` + the new user turn.
- **Anthropic Messages** — stateless. Re-send the full `messages: [...]` array each call.
- **OpenRouter (chat completions)** — stateless. Same as Anthropic, with `role: "tool"` instead of `tool_result` blocks.

## Contract (read this first)

`resume_from` is a **new-turn API**. It means "the prior conversation completed; append a new user message and continue." It explicitly does **not** mean:

- "retry the failed request" — fix the input and call `run()` fresh.
- "resume an interrupted tool call" — provider transcript invariants don't allow it.
- "continue the assistant's previous response" — separate concept, not supported.

`resume_state` is:

- **Opaque** — callers persist and replay it, they don't construct or read fields.
- **JSON-serializable** — `json.dumps(resume_state)` always works. This is part of the adapter contract: if `dump_state()` produces something non-serializable, it's a thinharness bug and the exception propagates (we do not silently omit state).
- **Provider-bound** — tagged with `kind`; rejected if the resuming harness uses a different provider.
- **Model-bound** — tagged with `model`; rejected if the resuming harness uses a different model name.
- **Versioned** — tagged with `version: 1`; unknown versions raise.
- **Strictly shaped** — unknown top-level keys are rejected. Forward compatibility is handled by bumping `version`, not by reserving extension fields.
- **Only emitted on final result objects** after a clean, resumable terminal state — never on intermediate tool turns, validation retries, or partial responses. See "Lifecycle" below.
- **Persistence-safe, not session-identity-bound** — the resuming harness/session can be a freshly constructed instance. Round-tripping through `json.dumps` / `json.loads` between processes is the expected path.
- **Reusable for sequential branching** — the same `resume_state` can be passed to `run()` multiple times with different prompts to fork the conversation. The harness deep-copies on dump and on load, so branches don't cross-contaminate. Concurrent branching from one `Harness` instance is blocked by the existing `_running` re-entrancy guard; callers wanting parallel branches construct multiple harnesses.
- **Expected to be used with the same `HarnessConfig`** that produced it (system prompt, tools). Mismatches outside `kind` + `model` are not validated; document this.

## Current state

`Harness.run(prompt, *, previous_response_id=None, metadata=None)` plumbs an OpenAI-only `previous_response_id` through:

- `core.py:204` — public `run()` accepts `previous_response_id`.
- `core.py:364` — passes it to `session.start(...)`.
- `providers.py:330` — `OpenAIResponsesSession.start` stores it and uses it on every `_complete`.
- `providers.py:441` / `providers.py:540` — Anthropic and OpenRouter sessions explicitly **raise** if `previous_response_id` is supplied.

The harness never surfaces the last response id back on `HarnessResult`. Anthropic/OpenRouter have no resume path — `AnthropicMessagesSession.messages` is reset on each `new_session()` and discarded when `run()` returns.

## Design — opaque, versioned, model-bound provider-session state

Each `ModelSession` knows how to serialize and rehydrate its own state.

```python
# OpenAI
{"kind": "openai", "version": 1, "model": "gpt-5.2", "previous_response_id": "resp_abc"}

# Anthropic
{"kind": "anthropic", "version": 1, "model": "claude-sonnet-4-6",
 "system": "...", "messages": [...]}

# OpenRouter
{"kind": "openrouter", "version": 1, "model": "anthropic/claude-sonnet-4-6",
 "messages": [{"role": "system", ...}, ...]}
```

Notes on shape:
- `kind` is the provider discriminator (matched against `Model.resume_kind`).
- `model` is the provider's model name (matched against `Model.model` byte-for-byte).
- `version` is `1` for v1; bumped on any breaking shape change. Unknown versions raise `HarnessError` before any provider call.
- For stateless providers, state size grows with the conversation — that's expected. OpenAI state stays small because the conversation lives server-side.

### Public API

```python
from thinharness import Harness, HarnessConfig

harness = Harness(HarnessConfig(model="anthropic:claude-sonnet-4-6"))

r1 = await harness.run("Tell me a joke.")
# r1.resume_state is a JSON-serializable dict (opaque)

r2 = await harness.run("Explain it.", resume_from=r1.resume_state)
```

Same shape for any provider; only the bytes inside `resume_state` differ.

`previous_response_id` is **removed** from `Harness.run` and `Harness.run_sync`. The low-level `OpenAIResponsesSession.start(..., previous_response_id=...)` is kept — it's provider-adapter internals, not a public harness mechanism. README documents only `resume_from`.

`resume_state` is **strictly opaque** — never documented as constructable. Callers with raw OpenAI response ids from outside thinharness drop down to the provider session API.

`resume_from` **only accepts the structured dict** — no raw strings, no shorthand.

## Implementation steps

### 1. `providers.py` — add resume to the session/model contract

Imports to add: `copy` (used by `dump_state` / `resume_session` for in-adapter deep-copies; `providers.py` currently imports `json` but not `copy`).

**Resume contract is a separate protocol.** `Model` stays unchanged; resume support lives on a sibling protocol that built-ins implement and custom models opt into:

```python
class Model(Protocol):
    model: str
    # ... existing members unchanged ...
    def new_session(self) -> ModelSession: ...


class ResumableModel(Model, Protocol):
    """A Model that also supports resume_from on Harness.run()."""
    resume_kind: str   # "openai" | "anthropic" | "openrouter" | (custom)

    def resume_session(self, state: dict[str, Any]) -> ModelSession: ...
```

Built-in `OpenAIResponsesModel` / `AnthropicMessagesModel` / `OpenRouterModel` all implement `ResumableModel`. Custom models that don't care about resume implement only `Model` — type-checkers don't force them to grow new methods. The harness's `hasattr` check in `core.py` is the runtime gate: passing `resume_from=...` to a non-`ResumableModel` raises `HarnessError`. Explicit kind string (not `isinstance`) keeps subclass/wrapper models compatible.

**`ModelSession` protocol additions:**

```python
class ModelSession(Protocol):
    async def start(...): ...
    async def continue_with_tools(...): ...
    async def continue_with_user_message(...): ...

    # NEW: first turn after resuming a prior conversation
    async def continue_with_user_prompt(
        self,
        prompt: str,
        *,
        instructions: str,
        tools: list[Json],
        metadata: Json | None = None,
        structured_output: StructuredOutputRequest | None = None,
    ) -> ModelTurn: ...

    def dump_state(self) -> dict[str, Any] | None:
        """Serialize session state for resume, or None if no usable continuation exists."""
        ...
```

Why a new method: `start()` overwrites `self.previous_response_id` from its kwarg, which would clobber loaded state. `continue_with_user_message` is the structured-output-retry path and shouldn't grow `instructions`. A dedicated entry point keeps each path explicit.

`continue_with_user_prompt`'s `prompt` argument is the **post-hook `effective_prompt`** — the same value `start()` receives on non-resumed runs. The harness applies `UserPromptSubmitContext` exactly once to compute it before calling either method, so resumed runs go through the same hook pipeline as fresh runs.

#### State validation helper

A private module-level helper (`providers.py`) used by every adapter:

```python
_BASE_RESUME_KEYS = frozenset({"kind", "version", "model"})

def _validate_resume_state(
    state: dict[str, Any],
    *,
    expected_kind: str,
    expected_model: str,
    required_fields: dict[str, type | tuple[type, ...]],
) -> None:
    """Validate resume state shape before any session mutation. Raises HarnessError."""
    if not isinstance(state, dict):
        raise HarnessError("resume_from must be a dict")
    if state.get("kind") != expected_kind:
        raise HarnessError(f"resume_from kind {state.get('kind')!r} does not match {expected_kind!r}")
    if state.get("version") != 1:
        raise HarnessError(f"resume_from version {state.get('version')!r} is not supported")
    if state.get("model") != expected_model:
        raise HarnessError(
            f"resume_from model {state.get('model')!r} does not match current model {expected_model!r}"
        )
    for field, expected_type in required_fields.items():
        if field not in state:
            raise HarnessError(f"resume_from missing required field: {field!r}")
        if not isinstance(state[field], expected_type):
            raise HarnessError(f"resume_from field {field!r} has wrong type")
    allowed = _BASE_RESUME_KEYS | required_fields.keys()
    unknown = set(state) - allowed
    if unknown:
        raise HarnessError(f"resume_from has unknown keys: {sorted(unknown)!r}")
    try:
        json.dumps(state)
    except (TypeError, ValueError) as exc:
        raise HarnessError("resume_from must be JSON-serializable") from exc
```

The trailing `json.dumps` is the inbound serializability check. Wrapping it in `HarnessError` keeps with the contract that bad caller input always surfaces as `HarnessError`, never raw `TypeError`. Outbound enforcement in `_build_resume_state` stays unwrapped — a non-serializable dump is an adapter bug, not a caller issue.

Forward-compat note: when a future change needs new fields, bump `version` and update validation. There is no reserved "extensions" field — version bumps are the explicit mechanism.

`HarnessError`, not `ProviderError` — bad caller input is a harness API error, not a remote failure.

Every `resume_session` calls this helper **first**, before any session field is touched. Tests assert that failed validation does not mutate any reachable state.

#### `OpenAIResponsesSession`

- `dump_state(self)`:
  ```python
  if not self.previous_response_id:
      return None
  return {
      "kind": "openai", "version": 1, "model": self.model.model,
      "previous_response_id": self.previous_response_id,
  }
  ```
- `continue_with_user_prompt(prompt, *, instructions, tools, ...)`: build a Responses payload with `input=prompt`, `instructions=instructions`, `previous_response_id=self.previous_response_id`, `tools=tools`.

`OpenAIResponsesModel.resume_session(state)`:
- `_validate_resume_state(state, expected_kind="openai", expected_model=self.model, required_fields={"previous_response_id": str})`.
- Additionally assert `state["previous_response_id"]` is non-empty.
- After validation: construct fresh `OpenAIResponsesSession(self)`; set `session.previous_response_id = state["previous_response_id"]`.

#### `AnthropicMessagesSession`

- `dump_state(self)`:
  ```python
  return {
      "kind": "anthropic", "version": 1, "model": self.model.model,
      "system": self.system,
      "messages": copy.deepcopy(self.messages),
  }
  ```
  Deep-copy isolates the dumped messages from the live session. Core-level JSON round-trip (in `_build_resume_state`) enforces serializability and produces the final caller-facing copy.
- `continue_with_user_prompt(prompt, *, instructions, tools, ...)`: `instructions` is ignored (the system prompt was rehydrated from state). Append `{"role": "user", "content": prompt}` to `self.messages`, `_complete(...)`.

`AnthropicMessagesModel.resume_session(state)`:
- `_validate_resume_state(state, expected_kind="anthropic", expected_model=self.model, required_fields={"system": (str, list), "messages": list})`. The `system` field accepts both `str` and `list` because Anthropic supports list-of-content-blocks for prompt-caching markers.
- Per-message validation: every item in `messages` is a dict. No further per-role/per-content checks — bad hand-edited messages surface as a `ProviderError` from Anthropic with their error text, which is sufficient.
- After validation: construct fresh session; set `session.system = state["system"]`; set `session.messages = copy.deepcopy(state["messages"])` (deep-copy in, so caller mutations after the call don't bleed into the session).

#### `OpenRouterSession`

- `dump_state(self)`: `{"kind": "openrouter", "version": 1, "model": self.model.model, "messages": copy.deepcopy(self.messages)}`. No `system` — it's `messages[0]`. Core-level JSON round-trip enforces serializability.
- `continue_with_user_prompt(prompt, *, instructions, tools, ...)`: ignore `instructions`. Append user message, `_complete(...)`.

`OpenRouterModel.resume_session(state)`:
- `_validate_resume_state(state, expected_kind="openrouter", expected_model=self.model, required_fields={"messages": list})`.
- Per-message validation: every item in `messages` is a dict. No per-role/per-content rules — bad hand-edited messages surface as a `ProviderError` from OpenRouter, which is sufficient.
- After validation: construct fresh session; `session.messages = copy.deepcopy(state["messages"])`.

### 2. `core.py` — `resume_from` replaces `previous_response_id`

- Drop `previous_response_id` from `Harness.run` and `Harness.run_sync`.
- Add `resume_from: dict[str, Any] | None = None` to both.

**Single-call validation, with explicit ordering at the top of `run()`:**

1. `if self._running: raise HarnessError("Harness.run is not re-entrant")` — concurrent calls always see the re-entrancy error first, regardless of what's in `resume_from`.
2. If `resume_from is not None`: validate + construct the session via `model.resume_session(resume_from)`. Full validation (kind, version, model, payload shape, unknown keys, JSON-serializability) and session rehydration happen here as one call. If anything is wrong, `HarnessError` raises before any side effects.
3. Initialize run bookkeeping (`_running = True`, `_current_run_metadata = ...`).
4. Fire `RunStartContext` and `UserPromptSubmitContext` hooks.

Custom models that don't support resume are detected at the same site with a small `hasattr` check, since `Model` is a duck-typed protocol:

```python
session: ModelSession
if resume_from is None:
    session = self.model.new_session()
    first_turn_kind = "start"
else:
    if not hasattr(self.model, "resume_kind") or not hasattr(self.model, "resume_session"):
        raise HarnessError(f"model {type(self.model).__name__} does not support resume")
    session = self.model.resume_session(resume_from)   # raises HarnessError on bad state
    first_turn_kind = "resume"

# ...later, after hooks and effective_prompt are computed:
if first_turn_kind == "start":
    first_turn_call = lambda: session.start(
        prompt=effective_prompt, instructions=instructions, tools=self.tool_schemas(),
        metadata=metadata, structured_output=structured_output,
    )
else:
    first_turn_call = lambda: session.continue_with_user_prompt(
        prompt=effective_prompt, instructions=instructions, tools=self.tool_schemas(),
        metadata=metadata, structured_output=structured_output,
    )
```

This collapses what was a two-layer validation (early check in core + full check in adapter) into one. The session-building call is the validation. If it succeeds, we hold the session in a local until we need it; if it raises, no hooks fired and `_running` was never set — the harness is left exactly as it was before the call.

After the run finishes, attach `resume_state` per the lifecycle rule (next section).

### 3. `core.py` — `HarnessResult.resume_state`

```python
@dataclass
class HarnessResult:
    text: str
    output: Any | None = None
    responses: list[Json] = field(default_factory=list)
    tool_call_records: list[Json] = field(default_factory=list)
    usage: RunUsage = field(default_factory=lambda: RunUsage())
    stop_reason: StopReason = "end_turn"
    resume_state: dict[str, Any] | None = None      # NEW
```

### 4. Lifecycle — when `resume_state` is emitted

`resume_state` is **only** ever set on the terminal `HarnessResult` returned by `run()`. It is never exposed on intermediate tool turns, validation retries, partial provider responses, or any future streaming events. Treat it as a property of the completed run, not a running cursor.

It is attached **only** when **all** of these hold:

1. `stop_reason == "end_turn"` — no in-flight tool calls, no errors. This excludes `provider_error`, `limit_reached`, `error`, `cancelled_by_hook`, `cancelled`, `output_validation_failed`, `tool_retries_exceeded`, `unexpected_model_behavior`.
2. `session.dump_state()` returns a non-`None` dict (handles OpenAI "no response id yet" case).
3. The run did **not** finalize via the synthetic `final_result` tool (structured-output tool mode). Reason: that turn leaves an unanswered `tool_use` / `tool_calls` in the stateless transcript and an unanswered function call in OpenAI's server state — neither can safely accept a fresh user turn.

In every other case, `resume_state` is `None`. This includes max-turn exhaustion, provider/tool errors, cancellation, and structured-output `final_result` finalization.

#### Detecting `final_result` termination

Use a **core-owned local boolean** rather than inspecting the last turn's `finalized_output_mode`. Inspecting the last turn is fragile — `_finalized_output_mode_for_turn` marks turns that *look like* finalization candidates, even ones the loop later rejects/retries, so its presence on the final turn isn't a reliable terminator signal across future loop refactors.

Concretely in `run()`:

```python
finalized_via_output_tool = False
# ...inside the loop, at the exact site we accept a final_result tool call:
if self.output_schema is not None and self.output_schema.mode == "tool":
    finals = [call for call in turn.tool_calls if call.name == FINAL_RESULT_TOOL_NAME]
    if finals and len(finals) == 1 and len(turn.tool_calls) == 1:
        try:
            value = self.output_schema.validate_tool_arguments(finals[0].arguments)
        except OutputValidationError:
            ...
        else:
            finalized_via_output_tool = True   # <- set here, only on accept
            result = build_terminal_result(turn.text, value)
            result.resume_state = _build_resume_state(
                session, stop_reason, finalized_via_output_tool
            )
            ...
```

And the helper that decides:

```python
def _build_resume_state(session, stop_reason, finalized_via_output_tool):
    """Apply lifecycle rule and produce the final resume_state value."""
    if stop_reason != "end_turn" or finalized_via_output_tool:
        return None
    state = session.dump_state()
    if state is None:
        return None
    # Core-level JSON enforcement: a non-serializable dump is a thinharness bug,
    # not silently dropped. Round-trip also gives an isolation copy.
    return json.loads(json.dumps(state))
```

The JSON round-trip here (a) enforces the adapter contract that `dump_state()` returns JSON-serializable data — applies uniformly to every adapter including OpenAI and custom models, not just the stateless ones that already round-trip internally — and (b) gives a fresh copy so the caller can mutate `result.resume_state` without affecting anything else. If `json.dumps` raises, it propagates as a `TypeError`; that's the right signal that an adapter is misbehaving.

**Assignment ordering — `RunEndContext` must see `resume_state`.** Every successful terminal path uses the same ordering:

1. Build the `HarnessResult` (`build_terminal_result(...)`).
2. Set `result.resume_state = _build_resume_state(session, stop_reason, finalized_via_output_tool)`.
3. Fire `fire_run_end_once()` — `RunEndContext.result` now points at the result with `resume_state` populated.
4. Return.

`RunEndContext` currently receives `result` as part of its context (`core.py:222-235`); without step 2 happening before step 3, hooks would see `result.resume_state is None` even on clean resumable runs. The `final_result` branch and the regular `end_turn` branch both follow this ordering.

**No transcript repair in v1.** The harness does not synthesize missing tool results, drop unpaired calls, or otherwise reshape the dumped transcript to make a non-clean exit resumable. If the run didn't end cleanly, the caller doesn't get a checkpoint. This keeps the contract simple and removes a class of subtle invariant bugs.

#### Resumed runs and structured output

A resumed run can request structured output in any mode supported by the selected provider/config (existing provider limits still apply — e.g. Anthropic still rejects native structured output, `providers.py:428-440`). The lifecycle rule applies the same way: if the resumed run terminates through the synthetic `final_result` tool, the returned `HarnessResult` has `resume_state = None`. Text and native modes (where supported) terminate cleanly and produce a fresh `resume_state` normally. This is consistent — `final_result` non-resumability is a property of the termination shape, not of whether the run was a resume.

`AnthropicMessagesSession.continue_with_user_prompt(...)` mirrors the existing `start()` / `continue_with_user_message()` guard: `if structured_output is not None: raise ProviderError("Anthropic does not support native structured output")`. Same for any future provider that doesn't support native — keep the guard symmetric across all three session entry points.

If we later need "resume after `final_result`", the cleanest path is sending a synthetic tool_result back to the provider during finalization. Larger behavioral change; defer.

### 5. `__init__.py`

No new exported types. `resume_state` is `dict[str, Any]`; documenting it as opaque is more useful than exporting a `ResumeState` alias.

### 6. Subagents (`subagents.py`)

Verify `build_child_harness` doesn't accidentally accept `resume_from` from the parent — each subagent invocation is a fresh one-shot. The child run site passes nothing for `resume_from` (default `None`).

### 7. Tracing

`RunTracer.agent` takes `conversation_id` from `metadata` (`core.py:333`). v1 leaves correlation to the caller — no auto-generation, no propagation through `resume_state`. Callers wanting cross-resume trace correlation stamp their own `conversation_id` into `metadata` each call.

### 8. Docs (README)

Add a section near the existing usage examples:

````markdown
### Continuing a conversation

`HarnessResult.resume_state` is an opaque, JSON-serializable token that lets you
continue a conversation with a new user message:

```python
first = await harness.run("Summarize this repository.")
if first.resume_state is None:
    raise RuntimeError("run cannot be continued")  # see notes on when this happens
save_json(first.resume_state)

state = load_json()
second = await harness.run("Now turn that into a checklist.", resume_from=state)
```

The contract, in five lines:

- Save `result.resume_state` exactly as JSON.
- Pass it back as `resume_from` with the next user message.
- Use the same provider, model, system prompt, and tools as the run that produced it.
- Expect no state after failed, cancelled, partial, or exhausted runs.
- Treat the contents as unstable provider-owned details — don't read or construct them.

Other notes:
- `resume_state` is `None` when the run did not end cleanly (provider/tool errors,
  limits, cancellation) or when it ended via structured-output `final_result`.
- The same `resume_state` can be reused to start multiple follow-up conversations
  (sequential branching). For parallel branches, use separate `Harness` instances.
- Stateless providers (Anthropic, OpenRouter) embed the full transcript in
  `resume_state`, so it grows with the conversation. OpenAI state stays small
  because the conversation lives server-side.
- OpenAI requires the prior response to still be retained server-side (deleted
  after ~30 days by default). An expired response id surfaces as a `ProviderError`
  with the OpenAI error text. v1 does not translate this into a dedicated
  expired-state error type.
- thinharness has no separate "message history" input parameter — `resume_from`
  is the only mechanism for carrying prior context across `run()` calls.
````

## Tests

New file `tests/test_resume.py`:

1. **OpenAI happy path** — round-trip via `json.dumps`/`json.loads`; assert second request includes `previous_response_id` and omits the original prompt.
2. **Anthropic happy path** — assert second request's `messages` contains original user turn, assistant turn, AND new user turn.
3. **OpenRouter happy path** — same shape, chat-completion messages.
4. **Cross-provider mismatch** — OpenAI state → Anthropic harness → `HarnessError("resume_from kind ...")`.
5. **Model mismatch (same provider)** — capture state from a harness using `claude-sonnet-4-6`, resume with `claude-opus-4-7`, expect `HarnessError`.
6. **Version mismatch** — hand-edit `version` to `2`, expect `HarnessError("resume_from version 2 is not supported")`.
7. **Missing version** — drop `version` field, expect `HarnessError`.
8. **Structured-output `final_result` ⇒ no `resume_state`** — for each provider, run with `output_type=` in tool mode, assert `resume_state is None` even though `stop_reason == "end_turn"`.
9. **No usable continuation token** — OpenAI session that never received a response id ⇒ `dump_state()` returns `None` ⇒ `resume_state is None`.
10. **Non-clean exits omit `resume_state`** — separate cases for `limit_reached`, `provider_error`, `tool_retries_exceeded`, `cancelled_by_hook`, `output_validation_failed`. Each asserts `resume_state is None`.
11. **Malformed `resume_from` shapes** — missing `kind`, wrong types for `messages`/`previous_response_id`, missing required fields, malformed message dicts. All raise `HarnessError` (not `ProviderError`).
12. **Failed validation does not mutate caller's session/model** — call `model.resume_session(bad_state)` directly (adapter-level, not through `Harness.run`); expect `HarnessError`; then call `model.new_session()` and exercise it normally; assert it works. The harness-level path is mostly exercised by early validation, so this test must stay at the adapter to catch ordering bugs in `resume_session` itself. (v4 L3.)
13. **Detached state — outbound** — mutate `result.resume_state["messages"][0]` after the run; resume from a separately-stashed copy; verify the original session capture is unaffected.
14. **Detached state — inbound** — pass a `resume_from` dict to `run()`; after the run, verify the caller's original dict is untouched (no in-place mutation).
15. *(removed — covered by test 18, which is a strict superset: fresh harness instance + JSON round-trip)*
16. **No raw string accepted** — pass `resume_from="resp_abc"`, expect `HarnessError`.
17. **Multi-turn run with tool calls** — confirm tool_use/tool_result pairing is preserved in dumped Anthropic/OpenRouter messages (falls out of the lifecycle rule + `final_result` exclusion, but explicit test guards regression).
18. **Fresh-harness persistence** — run with harness A; JSON-roundtrip `resume_state`; construct a fresh harness B with identical config; resume from B. Verifies persistence is not tied to in-memory session/harness identity. (v3 H5.)
19. **Sequential branching** — run, then resume from the same `resume_state` twice with two different follow-up prompts. Assert both branches succeed and produce independent continuations. Verifies state reusability + deep-copy isolation under reuse. (v3 M2.)
20. **Unknown top-level key rejection** — add a `"foo": "bar"` field to an otherwise-valid `resume_state`; expect `HarnessError`. Assert the message contains `"unknown keys"` and `"foo"`; don't pin the exact `sorted(...)` repr (Python version drift). (v3 H3, v4 L2.)
21. **Early validation — hooks don't fire** — pass a malformed `resume_from`; assert that `RunStartContext` and `UserPromptSubmitContext` hooks are never invoked. (v3 M3.)
22. **Adapter JSON-serialization contract** — a fake/test session that returns a non-serializable object from `dump_state()` should cause `Harness.run()` to raise rather than silently omit state. Confirms JSON serialization is part of the adapter contract, not a caller responsibility. (v3 H2.)
23. **Custom model without resume support** — a `Model` implementation that lacks `resume_kind` / `resume_session` attributes is allowed to run *without* `resume_from` (existing surface stays compatible). Calling `run(..., resume_from={...})` against such a model raises `HarnessError("model ... does not support resume")` before hooks fire. (v4 M4.)
24. **Resumed run with structured output** — resume from a clean prior state and request `output_type=` in tool mode. Assert the resumed run completes through `final_result` normally, and that the new result has `resume_state is None`. Confirms resumed runs can use structured output but follow the same lifecycle rule. (v4 H3.)
25. **Re-entrancy beats resume validation** — start a long-running `run()` on a harness in one task; concurrently call `run(..., resume_from=<malformed>)` on the same harness. Assert the re-entrancy `HarnessError` raises, not a resume-shape error. (v5 H2.)
26. **Inbound non-JSON-serializable state** — pass a `resume_from` whose nested payload contains a non-serializable object (e.g. a `datetime` inside `messages`); expect `HarnessError("resume_from must be JSON-serializable")`, not a raw `TypeError`. (v5 H3.)
27. **`RunEndContext` sees `resume_state`** — register a hook that captures `context.result.resume_state`. On a clean resumable run, assert the captured value is non-`None` and equals what the returned `HarnessResult` exposes. Confirms the assignment ordering in section 4. (v5 M1.)

Existing tests to migrate:

- `tests/test_harness.py:52` **stays as-is.** It asserts that within a single `run()`, the OpenAI session's second payload carries `previous_response_id` from the first response — that's internal tool-loop continuation, not the public resume API. Likewise `tests/test_providers.py:47` (low-level `OpenAIResponsesSession.start(previous_response_id=...)`) stays.
- `tests/fakes.py` — every fake session adds `dump_state()` and `continue_with_user_prompt(...)`; every fake model adds `resume_session(state)` and `resume_kind`. Keep fakes generic so individual tests can set the dumped shape.
- Public-API call sites that actually pass `previous_response_id=` to `Harness.run(...)` / `Harness.run_sync(...)` — if any exist — migrate to `resume_from=`. Audit before editing rather than searching blindly.

## Summary of feedback resolution

### v1 feedback

| Item | Resolution |
|---|---|
| H1 — structured-output `final_result` | `resume_state = None` when run terminated via `final_result`. Defer transcript repair. |
| H2 — OpenAI instructions on resumed turns | New `continue_with_user_prompt(..., instructions=...)` session method; OpenAI uses it, stateless providers ignore. |
| H3 — useless `previous_response_id: None` | `dump_state()` returns `None` when no continuation token; harness propagates. |
| H4 — state shape validation | `_validate_resume_state` helper called by every adapter before any mutation. |
| H5 — shallow-copy mutation | `json.loads(json.dumps(...))` on dump and on load. |
| H6 — system prompt drift | Trust caller for system prompt + tools; document. Model name IS now validated (see v2 H5). |
| M1 — protocol blast radius | All built-ins + fakes updated. Custom models without `resume_session` ⇒ `HarnessError`. |
| M2 — discriminator | `Model.resume_kind: str`. |
| M3 — `run_sync` | Both signatures migrate. |
| M4 — low-level provider test | `OpenAIResponsesSession.start(previous_response_id=)` stays; harness test migrates. |
| M5 — JSON serializability | JSON round-trip at dump time + persistence round-trip test. |
| M6 — same-kind model changes | Rejected (see v2 H5). |

### v2 feedback

| Item | Resolution |
|---|---|
| H1 — explicit state version | `"version": 1` in every state dict; unknown ⇒ `HarnessError`. |
| H2 — `resume_state` lifecycle | Emitted only on clean `end_turn` + non-`None` `dump_state()` + non-`final_result` terminal turn. Section 4 enumerates exclusions. |
| H3 — new-turn API only | Contract block at top of plan. Tests assert it; README documents it. |
| H4 — validate before mutating | `_validate_resume_state` runs first; adapters construct fresh sessions after. Test 12 confirms. |
| H5 — config identity in state | `model` name included and validated. System prompt / tool fingerprint deferred — documented as caller responsibility. |
| M1 — type alias | Skipped — keeping it as `dict[str, Any]` with strong docs is clearer than a half-typed alias. |
| M2 — detached state both directions | Deep-copy on both dump and load; tests 13 + 14 cover both. |
| M3 — keep OpenAI escape hatch private | `OpenAIResponsesSession.start(previous_response_id=)` documented as provider-adapter internals only; README only mentions `resume_from`. |
| M4 — size expectations | README documents stateless-provider state growth vs OpenAI's small state. |
| M5 — one error type | `HarnessError` for all resume-validation failures. `ProviderError` reserved for remote failures. |
| M6 — no raw strings | `resume_from` accepts only the structured dict; test 16 confirms. |

### v3 feedback

| Item | Resolution |
|---|---|
| H1 — only on final result objects | Lifecycle section made explicit: `resume_state` lives on the terminal `HarnessResult` only, never on intermediate turns or future streaming events. |
| H2 — JSON serialization in adapter contract | Stated in contract block; test 22 confirms non-serializable dumps propagate the `TypeError` rather than silently omitting state. |
| H3 — reject unknown top-level keys | `_validate_resume_state` rejects keys outside the allowed set. **Pushed back on the reserved-extension-field suggestion** — version bumps are the explicit forward-compat mechanism. Test 20 confirms. |
| H4 — ownership of transcript repair | v1: no repair, ever. Stated in section 4. |
| H5 — independent of session identity | Test 18 constructs a fresh `Harness` instance for the resume. |
| M1 — caller-perspective naming | Contract block and README use "continue this conversation with a new user message." Internal jargon (checkpoint, cursor, thread, restore) avoided in public docs. |
| M2 — single-use vs reusable | Reusable for sequential branching; documented. Concurrent branching needs separate harness instances (`_running` guard). Test 19 confirms. |
| M3 — validate before side effects | Early-validation step added to `core.py` step 2: kind/version/model checked before hooks fire. Test 21 confirms hooks don't fire on bad input. |
| M4 — stable cheap fingerprints | N/A — fingerprints not adopted in v1. |
| M5 — retention limits documented | README documents OpenAI server-side expiry. **Deferred translating expired-response errors into a dedicated error type** — keeping provider-specific error parsing out of the harness for v1. Surfaces as `ProviderError`. |
| M6 — no merging with input history | N/A — thinharness has no separate history-input parameter. Documented in README. |

### v4 feedback

| Item | Resolution |
|---|---|
| H1 — early validation internal inconsistency | Resolved. v5 collapsed the two-layer validation into one: `resume_session(state)` is called at the top of `run()` (before hooks), and that single call does the full validation + session construction. The `hasattr` check for custom-model opt-out is the only extra logic in core. Same "no side effects on bad input" guarantee with half the code. |
| H2 — final-result detection mechanism | Resolved via a core-owned `finalized_via_output_tool` local boolean set at the exact site `final_result` is accepted, then consumed by `_build_resume_state(...)`. No longer relies on inspecting `turn.finalized_output_mode` after the loop. |
| H3 — resumed structured-output tool mode | Documented explicitly in the lifecycle section: resumed runs can use structured output in any mode, but `final_result` termination ⇒ `resume_state = None`, same as non-resumed runs. Test 24 covers this. |
| M1 — `continue_with_user_prompt` prompt ownership | Section 1 explicitly states it receives the post-hook `effective_prompt`, so `UserPromptSubmitContext` runs once on resumed runs just like on fresh ones. |
| M2 — tighter stateless message validation | **Pushed back in v5** — per-role/per-content rules removed. Anthropic/OpenRouter validate only that `messages` is a list of dicts. The opaque-state contract tells callers not to hand-construct or hand-edit; the tighter validation only fires when they've done what we said not to, and the cost is a `ProviderError` instead of `HarnessError` (worse message, same fix). Anthropic's `system` field accepts both `str` and `list` (fixes a real bug — list form is needed for prompt-caching markers). |
| M3 — core-level JSON enforcement | `_build_resume_state(...)` in `core.py` does `json.loads(json.dumps(state))` after every successful `dump_state()`, regardless of provider. Adapter-level dumps now use `copy.deepcopy` (since serializability is enforced once at the core boundary). Test 22 confirms. |
| M4 — custom model failure mode | `_validate_resume_from_for_harness` `hasattr`-checks `resume_kind` / `resume_session` and raises `HarnessError` for models that don't support resume. Models without `resume_from` continue to run normally — no mandatory protocol upgrade. Test 23 confirms both halves. |
| L1 — README None-handling | README example now shows `if first.resume_state is None: raise ...` before the `save_json` call. |
| L2 — unknown-key error determinism | Test 20 asserts on `"unknown keys"` substring + offending key name, not the exact `sorted(...)` repr. |
| L3 — test 12 target | Test 12 now explicitly calls `model.resume_session(...)` directly, not through `Harness.run`. |

### v5 feedback

| Item | Resolution |
|---|---|
| H1 — full validation runs after hooks | Already resolved by the v5 simplification (reviewer had a stale version). Section 2 now states the ordering explicitly. |
| H2 — don't move resume validation ahead of re-entrancy guard | Explicit ordering in section 2: (1) `_running` check, (2) resume validation/session-construction, (3) bookkeeping, (4) hooks. Test 25 confirms re-entrancy wins. |
| H3 — inbound JSON-copy failures as `HarnessError` | `_validate_resume_state` now ends with a `try: json.dumps(state); except: raise HarnessError(...)`. Outbound `_build_resume_state` stays unwrapped (an adapter bug, not caller input). Test 26 confirms. |
| M1 — `RunEndContext.result.resume_state` | Section 4 now spells out the assignment ordering: build result → set resume_state → fire run_end → return. Test 27 confirms hooks see the populated value. |
| M2 — tighten OpenRouter `tool_calls` | **Pushed back.** Reverts the v5 simplification you and I just settled on (callers shouldn't hand-construct/hand-edit; malformed messages surface as `ProviderError` with provider error text). |
| M3 — tighten Anthropic nested content | **Pushed back** for the same reason as M2. |
| M4 — `test_harness.py:52` migration note was wrong | Corrected. That assertion is internal tool-loop continuation, not the public resume API. The migration note now says it stays as-is, and asks for an audit of actual public-API call sites instead of a blind search. |
| L1 — "any structured output mode" overstated | Reworded to "any mode supported by the selected provider/config." `AnthropicMessagesSession.continue_with_user_prompt` now gets the same `structured_output is not None ⇒ ProviderError` guard as `start` / `continue_with_user_message`. |
| L2 — `Model` protocol vs optional resume support | Split off a sibling `ResumableModel(Model, Protocol)`. `Model` stays unchanged; resume support is opt-in for custom models, type-checkers stay happy, runtime `hasattr` check is the gate. |
| L3 — `copy` import | Implementation checklist for step 1 explicitly calls it out. |

## Open questions

None blocking. Deferred items:

1. **Resume after structured-output `final_result`.** Would require sending a synthetic tool_result back to the provider before dumping. Larger behavioral change; defer.
2. **OpenAI server-side response expiry surfacing as a dedicated error.** Currently `ProviderError` with the OpenAI error text. Adding a `ResumeStateExpiredError` subclass would require parsing OpenAI-specific error codes inside the adapter — provider-specific knowledge we're explicitly avoiding for v1. Defer until user demand.
3. **System prompt / tool schema fingerprinting.** Deferred per user. Resuming with mismatched prompt/tools may produce confusing behavior; documented as caller responsibility.
