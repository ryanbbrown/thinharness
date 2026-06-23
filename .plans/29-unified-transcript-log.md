# Unified neutral transcript (one canonical log) — plan v3

Introduce a single provider-agnostic, JSON-serializable conversation transcript as the harness's canonical record of a run, and make `resume_state` (and the approval-pause `provider_state`) a projection of it instead of a per-provider native blob.

Revised after two multi-review rounds (`.reviews/plans/unified-transcript-log/*-v{1,2}.md`). All reviewers endorsed the core design across both rounds; v2 fixed the v1 correctness bugs (OpenAI deferred replay, system re-injection, Anthropic coalescing, notice preservation, version bump), and v3 resolves the v2 implementation-spec precision items (chiefly: the backing store is explicitly **dual-store / additive**, not a replacement). Findings-resolution tables for both rounds are at the end.

## Goal

A run's durable state should have **one representation**, not three. Today the conversation exists as:

1. Provider-native session state (`AnthropicMessagesSession.messages` in Anthropic block format, `OpenRouterSession.messages` in chat format, `OpenAIResponsesSession.previous_response_id` as a server-side pointer).
2. `HarnessResult.resume_state` — whatever (1) serializes to, tagged by provider `kind`.
3. The live `StreamEvent` queue and OTel spans — separate ephemeral projections.

This plan collapses (1) and (2) into a neutral `Transcript` that every provider renders to/from. After this change:

- `resume_state` is provider-agnostic: a run captured on `anthropic:claude-...` can be resumed on `openai:gpt-...` (with documented graceful degradation), and vice versa.
- `resume_state` is self-contained and no longer depends on OpenAI server-side response retention (the ~30-day expiry footgun in `.plans/08-resume.md`).
- The approval-pause envelope's `provider_state` is the same neutral transcript, so both resume paths unify.
- The neutral transcript is positioned to become the single source the event stream and tracing derive from (deferred — see "Out of scope").

This is the architecture pydantic-ai, strands, and agno converged on. The cost is bounded because thinharness has only three built-in providers and is already half-neutral on the harness↔provider boundary.

## What this is and is not

**Is:** a neutral *transcript / log* — one durable representation of what happened — added *alongside* the existing provider-native in-run request builders, which are unchanged.

**Is not:** a neutral *capability surface* (LiteLLM-style lowest-common-denominator), and **not** a replacement of the in-run request path. Provider-specific request settings (`ModelSettings.extra_body`, Anthropic `max_tokens`, structured-output modes) keep flowing through unchanged. The transcript records the conversation, not the request knobs, and is consulted only to produce/restore durable state.

## Current state (file:line)

The harness↔provider boundary is **already neutral in both directions**:

- **Out:** every provider returns `ModelTurn` (`providers.py:31` — `text`, `tool_calls: list[ModelToolCall]`, `raw`). The run loop never reads provider-native responses; it appends `turn.raw` to `run_ctx.responses` (`core.py:526`) and reads `turn.text` / `turn.tool_calls`.
- **In:** the loop drives sessions with neutral inputs — `start(prompt=...)`, `continue_with_tools(outputs: list[ToolOutput])`, `continue_with_user_message`, `continue_with_user_prompt`, plus `ModelNotice` / `StructuredOutputRequest` (`providers.py:115-173`).

What is **not** neutral is only the accumulated transcript each session keeps, appended after each completion:

- `AnthropicMessagesSession` — Anthropic block format; appends at `providers.py:614` (start user), `:634` (tool_result batch), `:653` (corrective user), `:669` (resumed user prompt), `:697` (assistant turn). Notices are baked into user content via `append_notices_to_text` at `:614/:653/:669` and as a trailing text block in the tool batch at `:632-633`.
- `OpenRouterSession` — chat format; appends at `:762/:780/:797/:811/:845`. Notices baked in at `:764/:797/:811` and a trailing user message at `:781-783`.
- `OpenAIResponsesSession` — no client transcript; only `previous_response_id`, set from each response id (`:541`). Notices appended to input text at `:440/:495/:517` and as a user message item in the tool path at `:467-471`.

`TurnDriver` can attach `ModelNotice`s to `start`, resumed prompts, corrective messages, and tool outputs (`runtime.py:67/:81/:122`), so notices are model-visible on every entry point, not just tool continuations.

`dump_state()` (`providers.py:171`) serializes whichever native blob the session kept, tagged with a provider `kind`. That tag is the entire reason resume is provider-bound. Both durable-state consumers funnel through `dump_state()`:

- `HarnessResult.resume_state` via `run_ctx.finalize(..., require_dump_state=model_supports_resume)` (`core.py:532-538`).
- The approval-pause envelope's `provider_state` (built in `run_ctx.pause_for_approval`, `core.py:577`; restored via `self._resume_approval_session(approval_pause.provider_state)` → `resume_session(...)`, `core.py:465`, `:720`).

So neutralizing `dump_state`/`resume_session` fixes **both** durable paths.

The validation/lifecycle rules from `.plans/08-resume.md` (clean-`end_turn`-only, no resume after `final_result`, no transcript repair) stay in force.

## Design

### 1. The neutral `Transcript` type (`providers.py`, new leaf types)

An ordered list of neutral entries. Pure data, JSON-serializable via an explicit to-dict/from-dict mapping with a `role` discriminator (dataclasses are not directly `json.dumps`-able, and the durable path does `json.loads(json.dumps(state))` — `runtime.py`/`approvals.py:87`).

```python
@dataclass
class AssistantEntry:
    text: str
    tool_calls: list[ModelToolCall]            # reuse existing leaf type

@dataclass
class UserEntry:
    content: str                                # fully rendered, provider-neutral user text
    notice: bool = False                        # True only for batch-accompanying notice text (grouping hint)

@dataclass
class ToolResultEntry:
    call_id: str
    output: str

TranscriptEntry = AssistantEntry | UserEntry | ToolResultEntry
```

**Serialization (load-bearing — specify exactly).** `dump_state` emits plain dicts with a `role` discriminator; the validator/renderers reconstruct from those dicts:

```python
{"role": "user", "content": "...", "notice": false}
{"role": "assistant", "text": "...", "tool_calls": [{"id": "...", "name": "...", "arguments": "..."}]}
{"role": "tool", "call_id": "...", "output": "..."}
```

`ModelToolCall.arguments` stays a JSON string in the transcript. Renderers needing a parsed object (Anthropic `tool_use.input`) call `json.loads(arguments)` (mirroring `_extract_anthropic_tool_calls`'s `json.dumps(input)` at `providers.py:994`).

**Notice text is captured in `UserEntry.content`, faithfully (no notice is dropped).** Because notices are model-visible on *all* entry points (current state above), v3 records them wherever they appear:

- `start` / `continue_with_user_prompt` / `continue_with_user_message`: the `UserEntry.content` stores the **notice-appended** text — exactly the value `append_notices_to_text(text, notices)` produces and that current in-run tests pin (`test_providers.py:148/:154-155/:172`) — not the raw prompt. `notice=False` (it is a normal user turn that happens to carry appended guidance).
- `continue_with_tools`: tool results become `ToolResultEntry`s, and any rendered notice text becomes a trailing `UserEntry(notice=True)` so the Anthropic renderer can fold it back into the tool-result user message (role alternation; §3).

Dropping stale notices from the durable transcript is a separate, intentional behavior change deferred to a follow-up; v3 preserves current behavior.

Notes:

- **No `system` in the stream.** The system prompt and tool schemas are the run's "versioned constant" (supplied by `HarnessConfig` each turn via `instructions`/`tools`), not part of the transcript. Resume re-derives them from live config — see §3.
- **v1 stores reasoning as text only.** No opaque provider blobs (Anthropic thinking `signature`, OpenAI `encrypted_content`/reasoning item ids). Same-provider reasoning fidelity is deferred (Out of scope).
- **No per-entry `provider_name` and no `_parse_extras` in v1.** Envelope-level `origin_provider` covers diagnostics; per-entry provenance is only needed by deferred fidelity work.
- **Assistant content ordering is intentionally flattened.** `AssistantEntry` stores concatenated `text` + a separate `tool_calls` list, matching what the harness already sees (`ModelTurn.text` is the concatenation of all text blocks via `_extract_anthropic_text`, `providers.py:998`). A `text → tool_use → text` native turn round-trips as `text + tool_use`. This is consistent with every existing downstream consumer; an accepted, tested degradation, not a regression. (See §2 — it affects only `dump_state`, never the in-run request.)
- `ModelToolCall.id` is the portable correlation key between an `AssistantEntry.tool_calls[i]` and the matching `ToolResultEntry.call_id`.

### 2. Provider contract: dual store (native in-run builder + parallel neutral transcript)

**This is Option A and it is explicit: the native in-run request builders are retained; the neutral transcript is purely additive.** Concretely:

- `AnthropicMessagesSession.messages`/`.system`, `OpenRouterSession.messages`, and `OpenAIResponsesSession.previous_response_id` **stay** and remain the source of every in-run request, byte-for-byte as today (OpenRouter still stores the raw assistant message at `providers.py:845`, preserving any `reasoning`/`reasoning_details` during the live run; OpenAI still chains `previous_response_id` and sends only new items).
- Each session **also** maintains a neutral `Transcript`, appending entries as it already receives/produces them: `start`/`continue_with_user_*` append a `UserEntry` (notice-appended content per §1); `continue_with_tools` appends `ToolResultEntry`s + optional `UserEntry(notice=True)`; `_complete` appends an `AssistantEntry(text, tool_calls)` from the existing `_extract_*` helpers (`providers.py:968-1009`).
- `_render_input(entries, *, instructions)` is invoked **only on the first post-resume turn**, never during a normal in-run turn. It is gated on a per-session resume flag (`_resume_entries` for stateless providers, `_pending_replay` for OpenAI) that is set by `resume_session` and consumed exactly once.

Consequence (and the implementation guard): **all in-run payload tests must remain byte-identical** — `test_harness.py:226-242`, `test_providers.py:60-74`, and `test_providers.py:92-175` (minus the resume sub-case at `:106-120`). If any in-run payload test changes, the dual-store boundary leaked (Option B crept in) and must be corrected. The assistant-text flattening and any per-turn re-rendering apply to resumed turns only.

The four `continue_with_*` bodies stay distinct — each renders to its native wire format and gains one neutral-append line. They do not "collapse."

### 3. Provider-agnostic `dump_state` / `resume_session` lifecycle

```python
# neutral envelope — same shape for every provider; allowed keys are exactly this set
{"kind": "transcript", "version": 2,
 "origin_provider": "anthropic",          # normalized model-ref prefix (see below); required
 "origin_model": "claude-...",            # required
 "entries": [ {"role": "user", ...}, {"role": "assistant", ...}, {"role": "tool", ...}, ... ]}
```

- `dump_state()` serializes the neutral transcript. `origin_provider` is the **normalized lowercase prefix** via `provider_prefix(self.model.provider.name)` (`providers.py:888-895`) — `self.model.provider.name` is capitalized (`"Anthropic"`/`"OpenAI"`/`"OpenRouter"`, `:279/:303/:329`), so it must be normalized to match model-ref prefixes. `origin_provider`/`origin_model` are **required** envelope fields; the allowed-key set is exactly `{kind, version, origin_provider, origin_model, entries}`.
- `resume_session(state)` (any provider) validates the neutral envelope and prepares the session for full replay. It **must not make a model request** (it runs before hooks/`_running`), so the rendered replay is deferred to the first post-resume turn, not produced inside `resume_session`.

**Replay injection is entry-point-agnostic.** A resumed run's first provider call is `continue_with_user_prompt` (normal resume) or `continue_with_tools` (approval resume — `core.py:492` sets `skip_user_prompt`; `_resume_approval_batch` enters through `send_tool_outputs` → `continue_with_tools`, `core.py:778`). The first such turn renders the transcript prefix and injects it, then proceeds normally; the resume flag is consumed so subsequent turns use the native in-run path.

- **Anthropic / OpenRouter (stateless).** `resume_session` stores `self._resume_entries = entries` (it cannot render yet — no `instructions`). The first post-resume turn renders entries into `self.messages`, sets/prepends the system prompt from the live `instructions`, appends the new turn's content, completes, and clears `_resume_entries`.
- **OpenAI (server-side).** `resume_session` stores `self._pending_replay = entries`, leaves `previous_response_id = None`. The first post-resume turn renders entries into Responses `input` items and produces the final `input` as `[*replay_items, *entry_point_items]`, sends with **no** `previous_response_id`, then clears `_pending_replay`. **`continue_with_user_prompt` builds a *string* `input`** (`providers.py:516`); when `_pending_replay` is set, that string must first be wrapped as a `{type:message, role:user, content:[{type:input_text, text}]}` item before prepending replay items (you cannot prepend list items to a string). `continue_with_tools` already builds a list, so it prepends directly. Once the response yields an `id`, chaining resumes.

**System prompt re-injection** (reverses plan-08's "instructions ignored on resume" — `.plans/08-resume.md:226/:236`):

- **Anthropic** — the first resumed turn sets `self.system = instructions` (today it ignores `instructions`, relying on rehydrated `self.system`, `providers.py:656-670`). Gated on the resume flag so non-resumed in-run turns are untouched.
- **OpenRouter** — the resume render prepends `{"role": "system", "content": instructions}` as `messages[0]` (today system is the rehydrated `messages[0]`, `providers.py:800-812`).
- **OpenAI** — already passes `instructions` live every turn (`providers.py:407`); no change.

**Per-provider render (`_render_input`):**

- **Anthropic** — entries → `messages`. `AssistantEntry` → one assistant message with `{type:tool_use, id, name, input: json.loads(arguments)}` per call, **plus a leading `{type:text}` block only when `text` is non-empty** (an empty text block would shift `content[0]` off the `tool_use` block that `test_resume.py:61` pins). **A maximal run of consecutive `ToolResultEntry`s + an immediately-following `UserEntry(notice=True)` coalesces into one `user` message** with N `tool_result` blocks (+ trailing `text` block) — separate user messages would violate role alternation. A `UserEntry(notice=True)` with *zero* preceding tool results (defensive: shouldn't occur) renders as a plain user message with a text block. Plain `UserEntry` → its own user message.
- **OpenRouter** — entries → chat messages. `AssistantEntry` → `{role:assistant, content, tool_calls:[{id, type:function, function:{name, arguments}}]}`; each `ToolResultEntry` → `{role:tool, tool_call_id, content}` (consecutive tool messages allowed); `UserEntry` → `{role:user, content}`.
- **OpenAI Responses** — entries → `input` items: `UserEntry` → `{type:message, role:user, content:[{type:input_text, text}]}`; `AssistantEntry` → optional `{type:message, role:assistant, content:[{type:output_text, text}]}` (omit when text empty) + one `{type:function_call, call_id, name, arguments}` per call; `ToolResultEntry` → `{type:function_call_output, call_id, output}`.

**Tool-call id portability (claim softened).** Replaying both sides of a call/result pair from the stored `ModelToolCall.id` preserves *internal* correlation, but does **not** guarantee the receiving provider accepts a foreign-format id (e.g. OpenAI may reject a `toolu_*` `call_id`). Acceptance must be verified against the real API (residual risks). Likewise, on a cross-provider replay into Anthropic, `json.loads(arguments)` could raise if a non-Anthropic origin stored non-JSON `arguments` (the OpenAI/OpenRouter extractors pass the raw provider string, `providers.py:973/:1008`); treat a parse failure as a documented cross-provider failure mode adjacent to the foreign-`call_id` risk.

### 4. Cross-provider / cross-model resume + version + validator coupling

- `_validate_resume_state`'s model/provider mismatch rejection (`providers.py:197-202`) is **removed**. `origin_provider`/`origin_model` are retained for diagnostics only.
- **`version` bumps to `2`.** The envelope shape changed, so per plan-08's contract the version must bump. The validator rejects old state with a clear, **`"resume_from"`-prefixed** `HarnessError` instructing the caller to regenerate — both old `version: 1` *and* old `kind` values (`"openai"`/`"anthropic"`/`"openrouter"`). thinharness is greenfield (no deployed persisted state); no migration shim.
- **The validator must keep the `"resume_from"` message prefix.** `_resume_approval_session` relabels only errors whose message `startswith("resume_from")` into `"approval state provider_state…"` (`core.py:724-728`); `test_approvals.py:733` depends on this. If the new validator changes the prefix, either preserve it or update `core.py:726` to match — state the coupling.
- Validation still rejects: non-dict, wrong/missing `version`, missing/non-list/malformed `entries`, missing/wrong-typed `origin_*`, unknown top-level keys, non-JSON-serializable payloads, malformed entry dicts (bad `role`/missing fields). `HarnessError`, not `ProviderError`.
- v1 degradation is simple because reasoning is text: switching providers loses nothing structural — message/tool history replays cleanly.

### 5. `resume_kind`, capability gates, and the custom-model boundary

`resume_kind` is no longer used for envelope validation (`kind` is fixed to `"transcript"`), but it is **kept as the capability marker** for the two runtime `hasattr` gates that opt a model into resume: `core.py:460` (`model_supports_resume`, also drives finalize's `require_dump_state`) and `core.py:1135` (`_model_supports_approval_resume`, also referenced by approval-tool config validation in `.plans/24-human-in-the-loop.md`). Both gates and the fakes/tests that set `resume_kind` stay. The field is intentionally vestigial for the envelope and load-bearing only for capability detection — documented as such.

**Custom-model boundary.** The neutral-transcript contract binds the **three built-in providers** (and the real-provider-backed test fakes). Custom `ResumableModel` implementations keep their own opaque `dump_state`/`resume_session` protocol — the harness delegates to `model.resume_session(state)` and does not impose the neutral schema on them. The scripted/sequence test fakes are exactly such custom models (`kind == "scripted"`) and are **left unchanged** (see §Tests). Document this boundary so adapter authors don't assume they must emit `{"kind": "transcript", ...}`.

### 6. Approval-envelope unification (`approvals.py`, `core.py`)

`build_approval_envelope`'s `provider_state` already carries `dump_state()` output, restored via `resume_session` (`core.py:465`, `:720-727`). With §3 it becomes the neutral transcript automatically. Required work:

- Update the field's documented shape; note that the envelope now embeds the full transcript inside `provider_state` (net-new size growth for OpenAI, previously a single id).
- `APPROVAL_ENVELOPE_VERSION` **stays `1`** — intentionally. The outer envelope shape is unchanged; only the *inner* `provider_state` reshapes. The two versions are independent. An approval envelope captured under the old code carries a native `provider_state` that now fails at `resume_session` time (not at `validate_approval_pause_state`, which only checks `provider_state` is a dict, `approvals.py:110`) with the relabeled `"approval state provider_state…"` error — consistent with the greenfield/no-migration stance. State this rather than leaving it silent.
- Fix `test_approval_resume_labels_inner_provider_state_errors` (`test_approvals.py:731-734`): it mutates `provider_state["kind"]="wrong"`; since `kind` is fixed to `"transcript"`, switch the malformed trigger to a bad `version`/`entries` so it still exercises the relabeling path (and the error must keep the `"resume_from"` prefix per §4). The scripted-backed approval envelopes (`test_approvals.py:860/:884`) use the custom protocol and are unaffected.

### 7. `final_result` / lifecycle parity

Unchanged from `.plans/08-resume.md`: `resume_state` only on clean `end_turn`, never after the synthetic `final_result` tool, never on non-clean exits. `finalized_via_output_tool` gating in `run_ctx.finalize` (`core.py:536`) stays.

### 8. Low-level escape hatch — a resume *semantic regression* to document

`OpenAIResponsesSession.start(previous_response_id=...)` (the documented escape hatch, `providers.py:438`, exercised by `test_providers.py:76-90`) still works in-run. But if a caller seeds a session from an *external* OpenAI response id and the run later dumps state, the neutral transcript captures only the new prompt onward — the externally-seeded prior turns are **absent from a full replay**. Today those turns survive via server-side chaining; after this change they do not. Because this changes resume *semantics* for an existing supported escape hatch (not merely an absent feature), it is surfaced in `behavior.md` (RESUME-6), not just noted here.

## Behavior changes (update `docs/behavior.md` after review, before implementation)

Add a `RESUME` section per the template (`docs/behavior.md:6-22`):

- **RESUME-1** — `resume_state` is a provider-agnostic transcript; resume across providers and across models is supported.
- **RESUME-2** — `resume_state` is self-contained and does not depend on any provider continuation token (e.g. OpenAI server-side response retention); an OpenAI run that never received a response id is still resumable.
- **RESUME-3** — v1 does not preserve provider-specific reasoning chains across resume (reasoning replays as text).
- **RESUME-4** — `resume_state` `version` is `2`; v1-shaped state (old `version` or old provider `kind`) is rejected with a regenerate error.
- **RESUME-5** — on resume, the *live* system prompt (from the resuming harness's config) is re-injected; the captured system prompt is not stored or restored.
- **RESUME-6** — a session seeded via the `start(previous_response_id=…)` escape hatch loses its externally-seeded prior turns on resume (the transcript captures only from the new prompt onward).

## Implementation steps

1. **`providers.py`** — add transcript leaf types + dict (de)serialization; add the neutral-envelope validator (drop kind/model rejection, require `version == 2`, keep `"resume_from"` prefix, reject old `kind` values, exact allowed-key set); per session add the parallel transcript (dual-store) + `_resume_entries`/`_pending_replay` flags, neutral-append in `start`/`continue_*`/`_complete`, `_render_input` (Anthropic coalescing + omit-empty-text + system set; OpenAI pending-replay with string→item wrap; OpenRouter system prepend), rewrite `dump_state`/`resume_session` to the neutral envelope with normalized `origin_provider`. Keep native in-run stores and `resume_kind`.
2. **`core.py`** — no structural loop change; confirm `_resume_approval_session` (`:720`) passes the neutral envelope through and the relabel prefix still matches; both capability gates (`:460`, `:1135`) unchanged.
3. **`approvals.py`** — update `provider_state` docs/shape; record version-independence + size-growth notes.
4. **Tests — targeted audit (§"Tests").** Update only real-provider-backed fakes; **leave the scripted/sequence fakes unchanged**.
5. **`docs/behavior.md`** — add `RESUME-1..6` (after review).
6. **Docs surfaces** — update `docs/docs.md` stale contract at `:248` (approval wraps provider resume state), `:252`, `:558`, `:562`, `:564` (same provider/model + provider-owned details), `:570` (size growth), `:572`; **add** a resume section to `README.md` (none exists — `:295` is a one-line bullet); regenerate site artifacts (`scripts/build_site.py`) if part of the docs workflow.
7. **Run** `uv run pyright`, ruff, and the full pytest suite (per project `CLAUDE.md`).

## Tests

**Do NOT migrate the scripted/sequence fakes.** `ScriptedModel.resume_session`/`ScriptedSession.dump_state` (`fakes.py:177-183/:210/:240-242`) and `SequenceSession` defaults (`test_mcp.py:102`, `test_streaming.py:62`) model custom resumable models with their own `kind == "scripted"` protocol; they never touch `_render_input` or the neutral validator. `test_stream_resume_from_emits_resume_kind` (`test_streaming.py:352`) **passes unchanged** — drop it from the change list. Migrating these would cascade-break `test_resume.py:179/:201/:223/:264/:305/:346/:405`, `test_tracing.py:358`, `test_mcp.py:102`, `test_streaming.py:62`, `test_approvals.py:860/:884`. Only `FakeClient`/`FakeAnthropicProvider`/`FakeOpenRouterProvider`-backed tests change.

Existing assertions to **invert or restructure** (not "migrate"):

- `test_resume.py:30` — resume-turn `previous_response_id` assertions invert; **also** `:43` (`input == "follow-up"` string→list) and `:46` (`"first" not in json.dumps` — now `"first"` *is* present). In-run `payloads[1]["previous_response_id"]` stays.
- `test_resume.py:88` — remove `kind`/`model` rejection branches (cross-provider/cross-model now succeed); keep version + unknown-key branches, retargeted to `version: 2`.
- `test_resume.py:116` (`test_resume_rejects_malformed_shapes_before_hooks_fire`) — **restructure** against the neutral schema. Every sub-case uses dead v1 field names (`:132/:134/:139/:141`), and the JSON-serializable case (`:143-147`) now trips the unknown-key check (which runs before `json.dumps`, `providers.py:208-214`), so it would raise "unknown keys" not "JSON-serializable". Rebuild the cases (and assert the JSON-serializable case still raises the JSON error — adjust validator order if needed).
- `test_resume.py:248` (`test_no_openai_response_id_omits_resume_state`) — **inverts**: a no-id OpenAI run is now resumable (RESUME-2).
- `test_resume.py:361-374` (detachment) — index changes from `["messages"]` to `["entries"]`; the mutated element must be a `UserEntry` (`entries[0]["content"]`), since `entries[1]` is the `AssistantEntry` (no `content` key).
- `test_resume.py:399` (`assert session.messages == []`) — **stays valid** under dual-store (native attribute retained).
- `test_approvals.py:600/:609/:629` — provider-native `provider_state` + OpenAI `previous_response_id`-replay assertions become `kind:"transcript"` + full-replay.
- `test_approvals.py:731-734` — change malformed trigger per §6; keep the `"resume_from"` prefix so relabeling fires.
- `test_providers.py:106-120` — neutral envelope + full-replay shape.

New cases:

1. **Cross-provider resume after a real tool round-trip** — capture on an Anthropic-shaped fake *through an actual tool turn*, resume on OpenAI- and OpenRouter-shaped fakes; assert the assistant tool call and its result replay with matching ids.
2. **Cross-model same-provider resume** — previously rejected; now succeeds.
3. **OpenAI approval-resume full replay** — pause on an approval-required tool (OpenAI-shaped fake), resume; assert the first `continue_with_tools` request replays the full prior input (user + assistant `function_call`) + the `function_call_output`, with **no** `previous_response_id`.
4. **OpenAI normal-resume shape** — assert `payloads[2]["input"]` is the list `[{user "first"}, {assistant function_call}, {function_call_output}, {user "follow-up"}]` with no `previous_response_id`, and `payloads[1]["previous_response_id"]` (in-run chaining) preserved.
5. **Multi-tool batch resume** — assistant turn with ≥2 tool calls: Anthropic render = one `user` message with N `tool_result` blocks; OpenRouter = N `role:tool` messages. (Requires fakes emitting multiple tool calls — current `echo_tool` fakes emit one, `fakes.py:133/:154`.)
6. **System-prompt re-derivation on resume** — resume with a harness whose system prompt differs from the capturing one; assert the replayed input carries the *new* system text (Anthropic `system` / OpenRouter `messages[0]` / OpenAI `instructions`), never `""`/absent (RESUME-5).
7. **Notice preservation, tool and non-tool** — (a) resume after a `continue_with_tools` carrying a background-completion/cancellation notice and a limit warning; assert the notice text survives and (Anthropic) is folded into the tool-result user message; (b) attach a limit notice to `start`/`continue_with_user_prompt` (e.g. low `max_model_requests`), resume, assert the rendered `<harness_notice …>` text survives in the replayed `UserEntry.content`.
8. **Round-trip serialization** — `json.loads(json.dumps(dump_state()))` equals `dump_state()` for a transcript with all three entry kinds and a multi-tool assistant turn; the non-JSON-serializable-dump test (`test_resume.py:402`) still raises.
9. **OpenAI no-id resume** — `resume_state` non-`None` for a no-id run; resume full-replays (RESUME-2).
10. **Version + kind rejection** — v1-shaped state (`version:1` *and* old `kind` values) raises `HarnessError` (RESUME-4).
11. **In-run byte-identical guard** — confirm `test_harness.py:226-242`, `test_providers.py:60-74`, `test_providers.py:92-175` (minus `:106-120`) are unchanged; any change means Option B leaked.
12. **Wire-shape pins still pass** — `test_resume.py:49/:61/:68` (string-content plain user messages; exact `tool_use`/`tool_result` and `type:"function"` shapes).
13. Carry over (retargeted): `final_result` ⇒ no state, non-clean exits ⇒ no state, malformed envelopes ⇒ `HarnessError`, fresh-harness persistence.

A test fake that **rejects an unpaired `function_call_output`** is recommended so the OpenAI renderer's pairing is proven against something stricter than permissive `FakeClient`. Live notice tests (`test_providers.py:300/:324`, skipped without keys) seed `session.system`/`session.messages` directly on a fresh session — confirm dual-store keeps that working.

## Out of scope (deferred)

- **Same-provider reasoning fidelity** — a `provider_extras` field stamped with `origin_provider`, re-emitted only on same-provider resume (Anthropic thinking `signature`, OpenAI `encrypted_content`), with per-entry `provider_name` added then. v1 stores reasoning as text.
- **Dropping stale notices** from the durable transcript — intentional behavior change; v1 preserves current behavior.
- **Tracing/event-stream as projections of the transcript** — pydantic-ai has each canonical part own its OTel serialization; thinharness's `tracing.py`/`events.py` could pull payload-building from the transcript. Separate change.
- **Extended-thinking in-run block-ordering constraints** — current code does not enable extended thinking; revisit if it does.
- **OpenAI same-provider `previous_response_id` fast-path on resume** — v1 favors uniform full replay (intentional; full replay is larger payloads for the common same-provider case, accepted for uniformity + retention-independence).

## Residual risks

- **Real-provider divergence from fakes (highest).** The OpenAI render path (full `input`-item replay, foreign `call_id` acceptance, `output_text`/`input_text` content types, `function_call`/`function_call_output` pairing without `previous_response_id`, reasoning-item-free replay against gpt-5-class models) is only exercised against fakes. Gate "RESUME-1 cross-provider supported" (vs experimental) on at least one real-API smoke test per provider pair: capture on Anthropic, resume on OpenAI Responses and OpenRouter; confirm acceptance or capture the exact rejection.
- **Cross-provider malformed `arguments`** — `json.loads(arguments)` in the Anthropic render can raise if a non-Anthropic origin stored non-JSON `arguments`; documented failure mode adjacent to foreign-`call_id`.
- **OpenAI `input` wire shape** changes from string to list on the resumed first turn (handled per §3); tests asserting string `input` update.
- **Size growth** — OpenAI `resume_state`/approval-envelope grows O(1)→O(conversation); also in-run *stored* state for OpenAI grows O(1)→O(n) (the parallel transcript), bounded by `max_model_requests`. Document for OpenAI callers.

## LOC estimate

~400–600 net lines added against the 1009-line `providers.py`: per-provider `_render_input`, Anthropic coalescing, OpenAI pending-replay (with string→item wrap), system re-injection, dual-store bookkeeping, and explicit entry (de)serialization. A few hundred net lines, not a doubling.

## Findings resolution — v1 review round

| Finding (reviewer) | Resolution |
|---|---|
| OpenAI replay can't run in `resume_session`; approval-resume re-enters via `continue_with_tools` (claude H1, glm H1) | §3 deferred entry-point-agnostic pending-replay; tests 3-4. |
| System prompt empty on Anthropic/OpenRouter resume (claude #3, glm H2) | §3 system re-injection from live `instructions`; RESUME-5; test 6. |
| Anthropic must coalesce consecutive tool results into one user message (claude #2, glm M2) | §3 coalescing rule; test 5. |
| Notices dropped from durable transcript (codex #1) | §1 notice capture; test 7. |
| Entry dataclasses not JSON-serializable / no discriminator (claude #4, glm L2) | §1 explicit dict shape + `json.loads(arguments)`. |
| Test blast radius understated; assertions invert (codex, claude #5, glm H3) | §Tests enumerates inversions. |
| No-id OpenAI run becomes resumable (claude #6) | RESUME-2; test 9. |
| `provider_name`/`_parse_extras` speculative (claude #7, glm L3) | Dropped from v1 (§1). |
| `version` not bumped (glm M1) | Bumped to `2`; v1 rejected (§4); RESUME-4. |
| `resume_kind` / two gates (glm M3) | §5 vestigial capability marker; both gates unchanged. |
| Tool-call id cross-provider acceptance over-claimed (glm M4) | §3 softened; real-API residual risk. |
| Assistant content ordering loss (codex #2) | §1 accepted+documented flattening; test 5. |
| Stale `docs/docs.md`; README has no section (codex #3, glm L1) | Step 6 updates docs.md + adds README section. |
| "Collapse `continue_with_*`" inaccurate (glm L4) | §2 corrected. |
| Wrong append-site citations (glm L5) | "Current state" anchored on real sites. |
| Escape-hatch interaction (claude #9, glm residual) | §8 + RESUME-6. |
| Envelope size growth (claude open-q1, glm residual) | §6 + residual risks. |
| behavior.md should be `RESUME-*` (glm/claude open-q) | RESUME-1..6. |
| OpenAI replay only tested vs fake (claude #8) | Residual risk + strict fake. |

## Findings resolution — v2 review round

| Finding (reviewer) | Resolution |
|---|---|
| Backing-store wording self-contradictory; pick Option A (claude #1 High, glm #2) | §2 rewritten: explicit dual-store; in-run byte-identical guard (test 11). |
| OpenAI `continue_with_user_prompt` builds a string `input` (glm #1) | §3 string→message-item wrap before prepending replay items. |
| Notice asymmetry for non-tool paths (codex #1, claude #4) | §1 `UserEntry.content` stores notice-appended text on those paths; test 7b. |
| Don't migrate scripted/sequence fakes (claude #3) | §5 custom-model boundary; §Tests leaves them unchanged; `test_streaming.py:352` dropped from change list. |
| `test_resume.py:116` malformed-shapes needs restructure (validator order) (glm #3) | §Tests restructure + assert JSON-serializable case still raises that error. |
| `test_resume.py:361-374` detachment retarget (`messages`→`entries`, mutate `UserEntry`) (claude #2, glm #3) | §Tests explicit retarget. |
| `test_resume.py:43/:46` invert (glm #3) | §Tests enumerated. |
| `_resume_approval_session` relabel depends on `"resume_from"` prefix (claude #6, glm #4) | §4 validator keeps the prefix (or update `core.py:726`); stated coupling. |
| behavior.md missing system-on-resume change (glm #5) | RESUME-5. |
| Escape-hatch is a semantic regression, belongs in behavior.md (claude #9, glm #5) | RESUME-6 + §8. |
| `origin_provider`/`origin_model` required + normalized (glm #6) | §3 required, exact allowed-key set, `provider_prefix` normalization. |
| Anthropic omit empty assistant text block (glm #7) | §3 leading text block only when non-empty. |
| docs.md enumeration incomplete (claude #7) | Step 6 adds `:248/:562/:564/:570`. |
| Approval envelope version independence (claude #8, glm #10) | §6 stays `1`, independent, old envelopes rejected at resume time. |
| Coalescing zero-tool-results edge (glm #8) | §3 defensive: plain user message. |
| Cross-provider malformed `arguments` (glm #9) | §3 + residual risks documented failure mode. |
| OpenAI in-run memory O(1)→O(n) (glm #11) | Residual risks note. |
| Custom `ResumableModel` schema boundary (codex open-q) | §5 custom-model boundary. |

## Open questions

None blocking. Deferred items are listed under "Out of scope."
