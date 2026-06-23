# Same-provider reasoning fidelity in the neutral transcript — plan v1

Capture native model reasoning into the neutral `Transcript` and re-emit it natively when a run is resumed on the **same provider**, so reasoning models do not lose their chain-of-thought across a resume boundary. Cross-provider resume degrades reasoning to text (unchanged intent of RESUME-3).

This executes the item plan-29 (`.plans/29-unified-transcript-log.md`) explicitly deferred under "Out of scope": *"Same-provider reasoning fidelity — a `provider_extras` field stamped with `origin_provider`, re-emitted only on same-provider resume (Anthropic thinking `signature`, OpenAI `encrypted_content`)… v1 stores reasoning as text."*

## Goal

Plan-29 made `resume_state` a neutral transcript of `UserEntry`/`AssistantEntry`/`ToolResultEntry`. `AssistantEntry` keeps only `text` + `tool_calls`; `_append_assistant_turn` discards `turn.raw` (`providers.py:306-307`), which is the only place native reasoning lives. Consequences today:

- **OpenAI Responses (sharpest regression).** Before plan-29, OpenAI `dump_state` serialized just `previous_response_id`; on resume OpenAI rebuilt the full prior conversation server-side **including reasoning items**. Plan-29 switched to self-contained client-side full replay with **no** `previous_response_id` and a transcript that carries no reasoning items → reasoning is dropped on every resume of a reasoning model. The retention-independence win was real; the casualty was the server-held reasoning.
- **OpenRouter.** `reasoning_details` arrive on the raw assistant message and are kept in-run via `self.messages`, but never reach the transcript → lost on resume.
- **Anthropic.** Only relevant once extended thinking is enabled (the harness does not enable it by default; see §6). When enabled, thinking blocks + signatures are kept in-run via `self.messages` but lost on resume.

After this change a run captured and resumed on the same provider preserves native reasoning; a run resumed on a different provider keeps the reasoning **text** (a leading `<thinking>`-tagged block) and drops the opaque blob.

## What this is and is not

**Is:** an additive `reasoning` field on `AssistantEntry`, populated from each turn's raw response and rendered back natively only when the resuming provider matches the originating provider.

**Is not:** a change to the in-run request/response normalization (`ModelTurn.text`/`tool_calls` are unchanged), and **not** a cross-provider reasoning translator. The neutral part is an opaque carrier gated by provenance, exactly as pydantic-ai's `ThinkingPart` does.

## Live API verification (done — gates the residual risk)

The plan-29 residual risk was "real-provider divergence from fakes." The happy path (one tool round-trip, a multiply tool, low reasoning effort) was verified against all three live APIs on 2026-06-22:

- **OpenAI** (`gpt-5-mini`): with `store` defaulted true (harness in-run mode) **plus** `include=["reasoning.encrypted_content"]`, the response's `reasoning` item carries `encrypted_content`. In-run `previous_response_id` chaining still works alongside `include`. A reasoning blob captured under `store=true` **replays statelessly** — full `input` `[reasoning, function_call, function_call_output]`, `store=false`, **no** `previous_response_id` → **200**, terse cached-reasoning answer. Replaying with `store` defaulted true (no `previous_response_id`) also → 200. Dropping the reasoning item also → 200 (model re-reasons).
- **Anthropic** (`claude-sonnet-4-6`, `thinking={type:enabled,budget_tokens:1024}`): T1 returns `[thinking{thinking,signature}, text, tool_use]`. Reconstructing the assistant message as `[thinking(signature), text, tool_use]` + a user `tool_result` → **200**. (Missing thinking block also accepted here, but native preservation is the goal.)
- **OpenRouter** (`reasoning={effort:"low"}`): `openai/gpt-5-mini` returns `reasoning_details=[{type:"reasoning.encrypted", data:"…"}]`; `anthropic/claude-sonnet-4.5` returns `[{type:"reasoning.text", text, format:"anthropic-claude-v1", index, signature}]`. Echoing `reasoning_details` verbatim on the assistant message + a `role:tool` result → **200** for both.

Smoke scripts are reproducible from `.env` keys; see §Tests for the cases to keep as a guarded live suite. **Not yet verified** (residual risks below): redacted_thinking, multiple reasoning items per turn, interleaved-thinking constraints, long conversations, and signature acceptance across a model-version change.

## Design

### 1. Neutral `ReasoningPart` + `AssistantEntry.reasoning` (`providers.py`, leaf types)

```python
@dataclass
class ReasoningPart:
    text: str = ""                         # plain reasoning text — always kept; cross-provider fallback
    signature: str | None = None           # opaque blob: Anthropic signature / redacted data,
                                            #             OpenAI encrypted_content, OpenRouter signature|data
    id: str | None = None                  # provider reasoning-item id (OpenAI rs_…; "redacted_thinking" marker)
    provider_name: str | None = None       # origin provider prefix; native re-emit only when this matches
    provider_details: Json | None = None   # spillover: OpenAI summary raw_content; OpenRouter raw reasoning_details entry

@dataclass
class AssistantEntry:
    text: str
    tool_calls: list[ModelToolCall]
    reasoning: list[ReasoningPart] = field(default_factory=list)   # additive, defaulted
```

A list because a turn can carry several reasoning parts (Anthropic thinking + redacted_thinking; OpenAI multiple summaries sharing one id; OpenRouter multiple `reasoning_details` entries).

`ModelTurn` gains `reasoning: list[ReasoningPart] = field(default_factory=list)`, populated in each `_complete` from `turn.raw`; `_append_assistant_turn` copies `turn.reasoning` (deep-copied) into the `AssistantEntry`.

**Provenance is per-part, not per-envelope.** A transcript can accumulate entries from multiple providers after cross-provider resumes, so the gate keys on `ReasoningPart.provider_name`, not the envelope's `origin_provider`. This mirrors pydantic-ai's `provider_name == self.system` gate (`models/anthropic.py:1348`, `models/openai.py:3044`).

### 2. Serialization (load-bearing)

- Add `ReasoningPart` to/from dict; `signature`/`id`/`provider_name`/`provider_details` are `None`-omitted-or-present per the existing explicit style.
- Extend the assistant branch of `_transcript_entry_to_dict`/`_transcript_entry_from_dict` (`providers.py:258-289`) to include `"reasoning": [ … ]`.
- **`_TRANSCRIPT_ENTRY_KEYS["assistant"]` (`providers.py:213`) must gain `"reasoning"`.** The validator does strict `set(value) != _TRANSCRIPT_ENTRY_KEYS[role]` (`:274`), so an assistant entry without the key would fail — hence the version bump (§4) and: `dump_state` always emits `"reasoning"` (empty list when none), keeping the key-set exact.
- Round-trips through `json.loads(json.dumps(...))` like every other entry, so it is also persisted incrementally by the transcript-delta tracing path (`tracing.py:467`).

### 3. Per-provider capture (IN) and native re-emit (OUT)

The gate, applied in every `_render_*_transcript` when rendering an `AssistantEntry.reasoning[i]`:

```
native_ok = part.provider_name == <this provider prefix> and <blob present> and <provider can accept it now>
if native_ok:  emit native reasoning block
elif part.text:  emit a leading "<thinking>\n{part.text}\n</thinking>" block (text fallback)
else:  drop
```

The text fallback is emitted as a leading content block (Anthropic `text` / OpenRouter `content` / OpenAI `output_text` message), **before** the assistant text and tool calls, matching pydantic-ai's `thinking_tags` degradation.

**OpenAI Responses.**
- *Capture:* add `include=["reasoning.encrypted_content"]` to `build_payload` (`providers.py:493-513`) so every response's `reasoning` items carry `encrypted_content` even under in-run `store=true` chaining (verified). In `_complete`, extract items where `type == "reasoning"`: `ReasoningPart(text=joined summary text or "", signature=encrypted_content, id=rs_id, provider_name="openai", provider_details={"raw_content": [...]} if present)`.
- *Re-emit:* in `_render_openai_transcript` (`providers.py:1133`), for a matching part emit `{"type":"reasoning","id":part.id,"encrypted_content":part.signature,"summary":[]}` **before** the assistant `output_text` message and the `function_call` items. The resume path already sends no `previous_response_id`; default `store` is accepted for the replay (verified), so **no `store` toggle is required** (optionally set `store=false` for ZDR — call-out, not a requirement).

**Anthropic Messages.**
- *Capture:* in `_complete` (`providers.py:793`), extract from `response["content"]`: `thinking` → `ReasoningPart(text=thinking, signature=signature, provider_name="anthropic")`; `redacted_thinking` → `ReasoningPart(text="", signature=data, id="redacted_thinking", provider_name="anthropic")`.
- *Re-emit:* in `_render_anthropic_transcript` (`providers.py:1060`), **prepend** reasoning blocks to the assistant `content` (before the optional text block and the `tool_use` blocks — Anthropic requires thinking first): `{"type":"thinking","thinking":part.text,"signature":part.signature}`, or `{"type":"redacted_thinking","data":part.signature}` when `id == "redacted_thinking"`.
- *Constraint:* Anthropic only accepts thinking blocks when the **resuming** request enables thinking. Gate the native emit additionally on "thinking enabled for this run" (derived from `self.model.settings.extra_body`); otherwise use the text fallback. Because the harness does not enable thinking by default (no `thinking` key is sent anywhere today), Anthropic native preservation is effectively inert until a caller turns thinking on — consistent with plan-29's note that extended thinking is out of scope. The in-run block-ordering constraints for live extended thinking remain out of scope (deferred in plan-29).

**OpenRouter.**
- *Capture:* in `_complete` (`providers.py:944`), read `message.get("reasoning_details")`. Store each entry faithfully: `ReasoningPart(text=entry.get("text",""), signature=entry.get("signature") or entry.get("data"), id=entry.get("id"), provider_name="openrouter", provider_details=entry)` — keeping the full raw entry in `provider_details` so the self-describing OpenRouter shape (`type`/`format`/`index`) round-trips exactly.
- *Re-emit:* in `_render_openrouter_transcript` (`providers.py:1101`), reattach `message["reasoning_details"] = [part.provider_details for matching parts]` on the assistant message (verbatim from `provider_details`). OpenRouter accepted both encrypted and text forms echoed verbatim (verified).

### 4. Version + validator coupling

- **`version` bumps `2` → `3`.** The assistant entry shape changed (new required `reasoning` key in the exact-key-set check). Per plan-08/plan-29's contract a shape change bumps the version. Greenfield (no deployed persisted state); reject old `version: 2` and `version: 1` plus old `kind` values with the existing **`"resume_from"`-prefixed** `HarnessError` instructing regeneration (`providers.py:227-230`).
- The `_resume_approval_session` relabel still keys on the `"resume_from"` prefix (`core.py:724-728`, `test_approvals.py:733`); keep it.
- `_validate_anthropic_tool_arguments` (`providers.py:310`) is unaffected (reasoning carries no tool-arg JSON).

### 5. Capability/boundary parity (unchanged)

`resume_kind`, both capability gates (`core.py:460`, `:1135`), and the custom-`ResumableModel` boundary (custom models keep their opaque protocol) are untouched. The neutral-reasoning contract binds only the three built-in providers and their real-provider-backed fakes.

## Behavior changes (update `docs/behavior.md` after review, before implementation)

- **Update RESUME-3** — was "v1 does not preserve provider-specific reasoning chains across resume (reasoning replays as text)." Now: *same-provider* resume preserves native reasoning (Anthropic thinking signatures, OpenAI `encrypted_content`, OpenRouter `reasoning_details`); *cross-provider* resume degrades reasoning to text.
- **Add RESUME-7** — for OpenAI Responses, the harness requests `include=["reasoning.encrypted_content"]` so reasoning survives resume; this is captured into `resume_state` (which therefore contains encrypted reasoning blobs — treat as sensitive, consistent with the existing local-trace sensitivity note).
- **RESUME-6 is unchanged and still applies** — the `start(previous_response_id=…)` escape hatch still loses externally-seeded prior turns on resume. This plan does **not** fix RESUME-6; state that explicitly so "reasoning regression fixed" is not misread as "all resume regressions fixed."
- `version` is now `3` (extends RESUME-4's regenerate-on-old-state rule).

## Implementation steps

1. **`providers.py`** — add `ReasoningPart`; add `reasoning` to `AssistantEntry` and `ModelTurn`; `ReasoningPart` (de)serialization; extend assistant entry (de)serialization + `_TRANSCRIPT_ENTRY_KEYS["assistant"]`; bump validator to `version == 3` (reject 1/2 + old `kind`, keep `"resume_from"` prefix); per-provider capture in the three `_complete`s; `_append_assistant_turn` copies `turn.reasoning`; OpenAI `include` in `build_payload`; native re-emit + text fallback in the three `_render_*_transcript`s with the per-part provenance gate (Anthropic also gated on thinking-enabled).
2. **`docs/behavior.md`** — update RESUME-3, add RESUME-7, restate RESUME-6 scope, note `version: 3` (after review).
3. **`README.md` / `docs/docs.md`** — update the Resume section: line 303 ("Provider-specific reasoning chains are not preserved") becomes "preserved on same-provider resume, degraded to text cross-provider"; note the OpenAI `include`/sensitivity point.
4. **Tests** — unit (fakes) + the guarded live suite (§Tests).
5. **Run** `uv run pyright`, ruff, full pytest (project `CLAUDE.md`).

## Tests

Real-provider-backed fakes (`FakeClient`/`FakeAnthropicProvider`/`FakeOpenRouterProvider`) only; leave scripted/sequence fakes unchanged (per plan-29 §Tests). Fakes must be extended to emit reasoning in their raw responses (OpenAI `reasoning` item w/ `encrypted_content`; Anthropic `thinking` block w/ `signature`; OpenRouter `reasoning_details`).

Unit cases:
1. **Capture** — each provider's `_complete` populates `AssistantEntry.reasoning` with the right `provider_name`, `signature`/`id`, and (OpenAI) `include` is present in the payload.
2. **Same-provider native re-emit** — resume on the same provider renders the native block (OpenAI `reasoning` item with `encrypted_content` ahead of the `function_call`, no `previous_response_id`; Anthropic `thinking{signature}` first in assistant content; OpenRouter `reasoning_details` reattached).
3. **Cross-provider text fallback** — capture on Anthropic-shaped fake, resume on OpenAI/OpenRouter-shaped fakes: assert a leading `<thinking>`-tagged text block, no opaque blob, no foreign reasoning item.
4. **Anthropic thinking-disabled fallback** — same-provider resume but thinking not enabled in the resuming config → text fallback, not a `thinking` block.
5. **redacted_thinking** — Anthropic `redacted_thinking` round-trips to `{"type":"redacted_thinking","data":…}`.
6. **Multi-part turn** — ≥2 reasoning parts and ≥1 tool call render in the right order.
7. **Round-trip serialization** — `json.loads(json.dumps(dump_state()))` equals `dump_state()` for an assistant turn carrying reasoning; `version == 3`.
8. **Version rejection** — `version: 2` and `version: 1` state raise the `"resume_from"`-prefixed `HarnessError`.
9. **In-run guard** — Anthropic/OpenRouter in-run payloads stay byte-identical; OpenAI in-run payload changes **only** by the added `include` key (assert exactly that delta — the deliberate exception to plan-29's byte-identical guard).

Live suite (guarded behind keys, mirrors the verified smoke tests; one tool round-trip each): OpenAI stateless reasoning replay; Anthropic reconstructed thinking(signature) acceptance; OpenRouter `reasoning_details` echo for one encrypted and one text model. Gate "same-provider reasoning preserved" on these passing.

## Residual risks

- **Verified only on the happy path.** redacted_thinking, multiple reasoning items per turn, interleaved-thinking ordering, long multi-turn conversations, and signature acceptance across a model-version change between capture and resume are **not** yet verified — add live cases or document as experimental.
- **`resume_state` now contains encrypted reasoning blobs** for OpenAI/OpenRouter (and signed thinking for Anthropic). Bigger payloads and sensitive content; documented in RESUME-7.
- **Anthropic native re-emit depends on thinking being enabled** in the resuming run; mismatched config silently uses the text fallback (intended, but worth a doc line).
- **OpenAI `include` changes every in-run payload** (the one accepted exception to plan-29's byte-identical guard).

## Out of scope (deferred)

- Cross-provider reasoning translation (kept as text by design).
- In-run extended-thinking block-ordering constraints (deferred in plan-29; revisit if the harness enables live thinking).
- RESUME-6 (`previous_response_id` escape-hatch seeding) — unrelated; not addressed here.
- OpenAI same-provider `previous_response_id` fast-path on resume (plan-29 chose uniform full replay).
