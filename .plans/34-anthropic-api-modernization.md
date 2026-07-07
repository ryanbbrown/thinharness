# Anthropic API modernization: max_tokens, provider-neutral effort, adaptive-thinking resume, native structured output — plan v3

Four related fixes to the Anthropic Messages path, in one change because they share the same payload-builder and config seams: (1) make `max_tokens` configurable and raise the silent-truncation default of 1024; (2) add a provider-neutral `effort` setting that each provider translates to its own dialect; (3) fix the resume-time thinking gate so adaptive thinking (the only mode on Opus 4.7+/Sonnet 5/Claude 5 family) replays signed thinking blocks; (4) support Anthropic native structured output via `output_config.format`, which is GA on current models (no beta header; the older `output_format` parameter is deprecated).

v2 incorporated round-1 review feedback (`.reviews/plans/anthropic-api-modernization/*-v1.md`): split payload precedence so structured output survives `extra_body` (matching OpenAI/OpenRouter), a model-aware thinking gate, validated `max_tokens`, enumerated the §4 signature/call-site/test edits, and pinned the effort × explicit-thinking and temperature × effort interactions. v3 applies two product decisions: the thinking-default table is inverted to enumerate the closed set of legacy off-by-default model families (unknown and future models are assumed on-by-default), and native becomes Anthropic's default structured-output mode (nothing depends on the tool-mode default yet).

## Goal

Today the Anthropic path has four gaps:

- **`max_tokens` is stuck at 1024.** `AnthropicMessagesModel` defaults `max_tokens: int = 1024` (`providers.py:725`), `infer_model()` never passes it (`providers.py:1023-1025`), and `HarnessConfig` has no field for it. The only override is smuggling `"max_tokens"` through `extra_body`, which works only because `payload.update(extra_body)` (`providers.py:838`) happens to run after the default is written (`providers.py:826`). 1024 silently truncates any real agentic turn — the caller sees only `finish_reason`.
- **Effort is provider dialect at the config layer.** Callers must know OpenAI wants `{"reasoning": {"effort": ...}}` while Anthropic wants `output_config: {"effort": ...}` (paired with `thinking: {"type": "adaptive"}`), and encode that in `extra_body`. Swapping providers means rewriting app-level translators.
- **Resume drops thinking signatures under adaptive thinking.** `_anthropic_thinking_enabled` (`providers.py:1090-1093`) recognizes only `thinking.type == "enabled"`. On models where `"adaptive"` is the only supported mode (manual `budget_tokens` returns 400 on Opus 4.7/4.8, Sonnet 5, Claude 5 family), the gate returns False, so `_render_anthropic_transcript` (`providers.py:1111-1118`) degrades signed thinking blocks to `<thinking>` text — and drops them entirely when the thinking text is empty, which is what `display: "omitted"` (the default on Opus 4.7+/Sonnet 5/Fable 5) returns. A resume that lands mid-tool-loop then replays an assistant `tool_use` turn without its thinking block and the API rejects the request.
- **Native structured output is hard-rejected.** `AnthropicMessagesModel` declares `supports_json_schema_output=False` (`providers.py:716`) and raises `ProviderError("Anthropic does not support native structured output")` in all three session methods (`providers.py:770-771, 788-789, 810-811`). That was true when written; Anthropic structured outputs are now GA on current models via `output_config: {"format": {"type": "json_schema", "schema": ...}}` with no beta header.

## What this is and is not

**Is:** config/settings plumbing (`HarnessConfig` → `infer_model` → `ModelSettings` → payload builders), a corrected resume gate, and a new Anthropic payload branch for `output_config`. In-run request/response normalization (`ModelTurn`), the neutral transcript format, and resume-state serialization are unchanged — no resume-state version bump.

**Not:** streaming support (the real fix for very large `max_tokens`; deferred), strict tool-parameter schemas (`strict: true` on Anthropic tool definitions), or refusal/fallback handling for Fable 5.

## Payload precedence (one rule, all providers)

Review round 1 caught that "extra_body wins wholesale per top-level key" would let a pre-existing `extra_body["output_config"]` silently delete the structured-output schema while the harness still validates as native. The existing codebase already splits precedence — OpenAI applies native `text` after the `extra_body` update (`providers.py:592-594`), OpenRouter applies `response_format` after it (`providers.py:978-980`), and tests pin that structured output beats `extra_body` (`tests/unit/test_providers.py:287, 382`). Anthropic follows the same split:

- **Tuning keys lose to `extra_body`:** `temperature`, `max_tokens`, `thinking`, `output_config.effort`, `reasoning.effort` are written before `payload.update(settings.extra_body)`. An `extra_body` top-level key replaces the harness-built value wholesale (existing `cache_control` precedent, pinned at `tests/unit/test_providers.py:346`).
- **The structured-output contract key wins over `extra_body`:** after the update, when a `StructuredOutputRequest` is present, `payload.setdefault("output_config", {})["format"] = {...}` deep-sets the format into whatever `output_config` survived — harness-built effort, or a caller's `extra_body` override. A caller's `output_config.effort` is preserved; the schema cannot be silently dropped. Same outcome as OpenAI/OpenRouter: the harness never validates against a schema it didn't send.

## Design

### 1. `max_tokens` on `ModelSettings` + `HarnessConfig`

- `ModelSettings` (`providers.py:148-152`) gains `max_tokens: int | None = Field(default=None, ge=1)`; `HarnessConfig` (`core.py:104-145`) gains the same field next to `temperature`. All presence checks use `is not None` (never truthiness — `0` must fail validation, not silently mean "unset").
- `Harness.__init__` threads it through `infer_model` (`core.py:178-185`), which forwards it into `ModelSettings` (`providers.py:1019`).
- Per-provider translation, matching how `temperature` already works:
  - **Anthropic:** `payload["max_tokens"]` = ctor arg if not None, else `settings.max_tokens` if not None, else module constant `DEFAULT_ANTHROPIC_MAX_TOKENS = 16384` (`providers.py:826`). The API requires the field; 16384 is the practical ceiling for non-streaming requests per current API guidance ("stream for anything above ~16K output") — `create_message` is a plain non-streaming POST (`providers.py:488-490`) behind `request_timeout` (default 120s), so a larger default would trade silent truncation for client timeouts.
  - **OpenAI Responses:** emit `payload["max_output_tokens"] = settings.max_tokens` in `build_payload` (`providers.py:573-595`) only when not None; the Responses API default (model maximum) is fine.
  - **OpenRouter:** emit `payload["max_tokens"] = settings.max_tokens` in `_complete` (`providers.py:960-980`) only when not None.
- The `AnthropicMessagesModel` ctor arg becomes `max_tokens: int | None = None`; when not None it takes precedence over `settings.max_tokens` (more specific, explicitly constructed; existing direct constructions like the live test's `max_tokens=2048` keep working).
- `extra_body["max_tokens"]` still overrides (tuning key, see precedence section) — kept deliberate, documented in the `_complete`/`build_payload` docstrings, and pinned by a test.
- Threading parity: the two other `infer_model` call sites that forward `temperature`/`extra_body` — subagent child models (`subagents.py:276-283`) and the parallel-LLM tool (`tools/parallel_llm.py`, ctor + `create_parallel_llm_tool`) — forward `max_tokens` (and `effort`, §2) the same way. Deliberate asymmetry with `builtin_parallel_llm_temperature` (`core.py:142`): no dedicated `builtin_parallel_llm_max_tokens`/`_effort` knobs — the parallel-LLM tool inherits the parent values; a dedicated override can be added later if a real need appears.

### 2. Provider-neutral `effort`

- `ModelSettings` gains `effort: str | None = None`; `HarnessConfig` gains the same field; threaded through all `infer_model` call sites as in §1.
- Type is `str`, not a Literal: the vocabularies differ per provider and model (Anthropic: `low`/`medium`/`high`/`max` everywhere effort is GA, plus `xhigh` on Opus 4.7/4.8, Sonnet 5, and Fable 5; `max` errors on Sonnet 4.5/Haiku 4.5; OpenAI: `minimal`/`low`/`medium`/`high`). The API is the validator; client-side enums would chase provider releases.
- Translation, written before the `extra_body` update (tuning keys — see precedence section):
  - **OpenAI Responses:** `payload["reasoning"] = {"effort": settings.effort}`.
  - **OpenRouter:** `payload["reasoning"] = {"effort": settings.effort}` (OpenRouter's unified reasoning parameter, already live-verified in plan 31).
  - **Anthropic:** `payload["output_config"] = {"effort": settings.effort}` and `payload["thinking"] = {"type": "adaptive"}`. Rationale for auto-adaptive: effort is the thinking-depth control on 4.6+; adaptive is the pairing the API recommends; Fable 5 accepts explicit adaptive; models that reject adaptive (pre-4.6) only support `effort` behind a beta header this harness never sends, so no supported combination is lost.
- **`effort` composes with an explicit `extra_body["thinking"]` — no client-side rejection.** Both "contradictory-looking" combos are API-sanctioned: `thinking: {"type": "disabled"}` + `output_config.effort` is the documented Sonnet 4.6 fast-path shape, and `thinking: {"type": "enabled", "budget_tokens": N}` + `effort` is the documented transitional escape hatch. So: the injected adaptive `thinking` is written before the update and an explicit `extra_body["thinking"]` replaces it wholesale; `output_config.effort` is still emitted; the resume gate (§3) honors the explicit dict. Invalid combos (e.g. effort on Haiku 4.5) surface as descriptive API 400s.
- Note for `docs/behavior.md`: unset `effort` means the API-side default, which is `high` on Sonnet 4.6+ — not minimal. `effort` is a hint layered on that default.
- `temperature` × `effort`: both are passed through independently. On every model where `effort` is GA without a beta header (Opus 4.7/4.8, Sonnet 5, Fable 5 reject sampling params outright; 4.6-family accepts them), some combinations 400. Consistent with the "API is the validator" stance — the harness does not referee; pinned by a payload test and stated in behavior.md.

### 3. Adaptive-aware resume thinking gate

Replace `_anthropic_thinking_enabled(settings)` (`providers.py:1090-1093`) with a model-aware helper pair:

```python
_ANTHROPIC_THINKING_DEFAULT_OFF_PREFIXES = ("claude-opus-4", "claude-sonnet-4", "claude-haiku-4", "claude-3")


def _anthropic_thinking_on_by_default(model_name: str) -> bool:
    """Return whether this model runs thinking when the thinking key is omitted.

    Enumerates the closed, historical set of off-by-default families rather
    than the open set of on-by-default ones: thinking-off-when-omitted is
    legacy behavior (Opus 4.0-4.8, Sonnet 4.x, Haiku 4.x, Claude 3.x), while
    the 5-era models run thinking with the key omitted (Fable/Mythos always-on
    — disabled is a 400; Sonnet 5 on-by-default but disable-able). Unknown and
    future models are assumed on. Maintenance contract: only a future
    off-by-default model needs a prefix added here; callers can always pin
    behavior with an explicit thinking config in extra_body.
    """
    return not model_name.startswith(_ANTHROPIC_THINKING_DEFAULT_OFF_PREFIXES)


def _anthropic_thinking_enabled(model: AnthropicMessagesModel) -> bool:
    """Return whether the resuming Anthropic request runs extended thinking."""
    if "thinking" in model.settings.extra_body:
        thinking = model.settings.extra_body["thinking"]
        return isinstance(thinking, dict) and thinking.get("type") in ("enabled", "adaptive")
    return model.settings.effort is not None or _anthropic_thinking_on_by_default(model.model)
```

- An explicit `thinking` key in `extra_body` is authoritative in both directions: `"enabled"`/`"adaptive"` → replay; `"disabled"`, any other type, or a present-but-non-dict value → no replay (a malformed `thinking` is explicit-and-unrecognized, not a license to infer — the request itself will 400 or run thinking-off, and emitting thinking blocks into it would be wrong either way).
- Without the key: `effort` set implies the harness injects adaptive (§2) → replay; otherwise the off-by-default table decides. The guess for unlisted models is deliberately "on": guessing "off" re-introduces the intermittent signature-drop bug for every new on-by-default model until a table edit ships (round-1 finding against Sonnet 5, whose omitted-key default runs adaptive with `display: "omitted"` — empty-text signed blocks), while guessing "on" can only misfire if Anthropic ships a future off-by-default model against its current direction.
- On-by-default models get **no** injected `thinking` in the outgoing payload when `effort` is unset — the API default is relied on; the gate and the wire shape are decoupled by design (injection is an `effort` side effect only).
- The call site (`providers.py:857-860`) changes with the helper in the same commit: `_apply_resume` passes `self.model` instead of `self.model.settings` (gate + helper signature + call site are one atomic change).
- The render path (`providers.py:1111-1118`) already emits signed parts with empty text as `{"type": "thinking", "thinking": "", "signature": ...}` when the gate is on — Fable 5/Sonnet 5 replay rules require exactly that. No render change expected; pin it with a test rather than assuming (the `elif part.text:` fallback must remain unreachable for signed parts when the gate is on).
- Test knock-on: non-legacy fake model names (`"claude-test"` in the unit fakes) now land on the on side of the gate. Resume tests that need thinking-off-by-omission must use a legacy-prefixed name (e.g. `"claude-opus-4-8"`) or explicit `{"type": "disabled"}`.

### 4. Native structured output via `output_config.format`

- Flip the capability: `ModelCapabilities(supports_json_schema_output=True, default_structured_output_mode="native")` (`providers.py:716`), matching OpenAI (`providers.py:548`). Product decision (v3): native is the better mode and nothing depends on the current tool-mode default yet, so `output_type` under the default `output_mode="auto"` now resolves to native for Anthropic. Tool and prompted modes remain available explicitly (`ToolStructuredOutput(...)`/`output_mode="tool"`, `PromptedOutput(...)`). `output_mode="native"` today raises in `resolve_output_schema_for_model` (`output.py:191-193`) and will now pass.
- Remove the three `ProviderError("Anthropic does not support native structured output")` raises (`providers.py:770-771, 788-789, 810-811`).
- Concrete edits (round-1 ask — enumerate the surface): `_complete` (`providers.py:822`) gains a `structured_output: StructuredOutputRequest | None = None` keyword; the three session-method call sites (`providers.py:778, 800, 816`) forward `structured_output=constants.structured_output`, mirroring OpenRouter (`providers.py:924, 941, 955`).
- Inside `_complete`, after `payload.update(settings.extra_body)`: when `structured_output` is present, `payload.setdefault("output_config", {})["format"] = {"type": "json_schema", "schema": request.schema}` (precedence section). `request.name`/`description`/`strict` have no Anthropic wire equivalent and are ignored (they exist for the OpenAI translation, `providers.py:1236`).
- No response-side changes: the API returns the JSON document as an ordinary `text` content block, so `_extract_anthropic_text` and the provider-neutral native-output validation of `turn.text` (`output.py:138-146`) apply unchanged. The guarded live smoke asserts the raw response content shape to prove this against the real API, not just the fake.
- Two existing tests assert the capability gate being removed and must be replaced (round-1 finding): `tests/unit/test_structured_output.py:542-546` (`test_anthropic_native_mode_is_rejected`) and `:549-553` (`test_native_output_marker_respects_provider_capabilities`) both `pytest.raises(ValueError, match="does not support native structured output")`. Replace with: native mode constructs without raising, and the payload carries `output_config.format` (mirror the OpenRouter payload-assertion pattern at `tests/unit/test_providers.py:382-401`). Additionally, audit `tests/unit/test_structured_output.py` for tests that assert Anthropic `auto` resolves to tool mode (e.g. `final_result` tool advertised by default) — those flip to asserting native resolution, with explicit `output_mode="tool"` coverage retained.
- Model-floor behavior: structured outputs are GA without a beta header on current models (Claude 5 family, Opus 4.6+, Sonnet 4.6+, Haiku 4.5). On older models the API returns 400 — and because native is now the `auto` default, an `output_type` run against a pre-4.6 Anthropic model that previously worked via tool mode starts failing. Accepted (v3 decision — no active users); such callers set `output_mode="tool"` explicitly. No capability sniffing.
- Anthropic enforces a JSON Schema subset; unsupported keywords fail at the API with a descriptive 400. Pass-through, no client-side schema filtering.

## Behavior changes (update `docs/behavior.md` after review, before implementation — four distinct sections, edit only affected ones)

- Anthropic default `max_tokens` becomes 16384; `HarnessConfig.max_tokens` (≥1) sets it explicitly for any provider (dialect-translated); `extra_body` still overrides.
- New `HarnessConfig.effort`, dialect-translated per provider; on Anthropic it also enables adaptive thinking unless `extra_body` sets `thinking`; unset means the API-side default (`high` on Sonnet 4.6+); `temperature` + `effort` conflicts surface as API errors.
- Resume replays Anthropic thinking signatures under adaptive thinking, effort-implied adaptive, and whenever the model runs thinking by default with the key omitted (every model outside the legacy off-by-default families: Opus 4.x, Sonnet 4.x, Haiku 4.x, Claude 3.x) — amends the existing thinking-replay requirement rather than adding a new one.
- Anthropic's default structured-output mode becomes native (matching OpenAI); tool and prompted modes remain available explicitly; the structured-output schema always survives `extra_body` (all providers); pre-4.6 Anthropic models need explicit `output_mode="tool"`.

## Implementation steps

1. §3 gate fix: helper pair + call-site edit (`providers.py:857-860`) + empty-text render pin, one atomic change → unit tests on `_render_anthropic_transcript` via resume fakes.
2. §1 `max_tokens`: `ModelSettings`/`HarnessConfig` fields with `ge=1`, three payload builders, `infer_model` + subagents + parallel-LLM threading, ctor precedence.
3. §2 `effort`: fields, three translations, adaptive injection, interaction with §3's gate.
4. §4 structured output: capability flip to native default, remove raises, `_complete` signature + three call-site forwards, post-update `format` deep-set, replace the two capability-gate tests and flip any Anthropic auto-resolves-to-tool assertions.
5. Tests + `uv run pyright` + ruff; update `docs/behavior.md` (four sections) and `docs/docs.md` (provider-settings list at `docs/docs.md:99` gains `max_tokens`, `effort`); CHANGELOG entry.

## Tests

- **Payload assertions (fake provider):** Anthropic payload carries `max_tokens=16384` by default, `HarnessConfig.max_tokens` value when set, ctor-arg precedence over settings, and `extra_body["max_tokens"]` overriding both (pins the tuning-key ordering; same precedent as the `cache_control` pin at `tests/unit/test_providers.py:346`). OpenAI payload gains `max_output_tokens` only when set; OpenRouter `max_tokens` only when set. The exact-key-set assertion at `tests/unit/test_reasoning_fidelity.py:381` stays valid for default settings; add variants asserting the new keys appear only when configured. Validation tests: `max_tokens=0` and negative rejected on both `HarnessConfig` and `ModelSettings`.
- **Effort translation:** per provider, effort lands in the right dialect; Anthropic gets `thinking: {"type": "adaptive"}` + `output_config.effort`. `extra_body` overrides of `thinking` (each of `adaptive`, `enabled`+budget, `disabled`), `reasoning`, and `output_config` win per top-level key. `temperature` + `effort` both emitted (pins pass-through as intended).
- **Precedence collision (round-1 finding):** `extra_body={"output_config": {"effort": "high"}}` together with `NativeOutput` → payload `output_config` contains **both** the caller's `effort` and the harness `format` (the deep-set semantics), on the fake provider.
- **Resume gate:** parametrize the reasoning-fidelity resume tests over `{"type": "enabled", "budget_tokens": N}` (legacy) and `{"type": "adaptive"}`; add cases for effort-implied adaptive, omitted key on an on-by-default model (`claude-sonnet-5`) and on an unknown model name (assumed on), omitted key on a legacy model (`claude-opus-4-8` → no replay), explicit `disabled` suppressing replay, a present-but-non-dict `thinking` value suppressing replay, and an empty-text signed thinking part rendering as a `{"type": "thinking", "thinking": "", "signature": ...}` block.
- **Structured output:** replace the two capability-gate tests (§4); `auto` now resolves to native for Anthropic (flip any tool-default assertions) while explicit `output_mode="tool"` still works; native mode payload contains `output_config.format` with the schema; effort + format merge. End-to-end success path needs a JSON-returning fake (`FakeAnthropicProvider` at `tests/unit/fakes.py:122-136` returns `"done"`, not valid JSON) asserting `turn.text` parses/validates like OpenAI native output. Truncation path: fake returning `stop_reason: "max_tokens"` with cut-off JSON → the existing `OutputValidationError`/`output_retries` path engages.
- **Threading parity:** payload-level assertions that `effort`/`max_tokens` set on the parent reach subagent child models (`subagents.py:276-283`) and the parallel-LLM tool's model resolution.
- **Live (guarded, keys from `.env` via `uv run --env-file .env`):** the legacy-dialect resume case stays on its current default `claude-sonnet-4-5` (pre-4.6: `enabled`+`budget_tokens` still functional there — it is current models that 400). Add: adaptive-dialect resume on `claude-sonnet-5`; omitted-config resume on `claude-sonnet-5` (the intermittent-drop case from round 1); native structured-output smoke on a current model asserting the raw response content shape (JSON in a `text` block) alongside a tool loop. Re-run the existing live surface at `tests/unit/test_providers.py:426-427` (defaults `claude-haiku-4-5`) to confirm the 1024→16384 default causes no timeout.

## Residual risks

- **Adaptive replay of signed blocks after truncation/interleaving is asserted from API docs, not yet live-verified in this harness.** Plan 31 live-verified signature replay under legacy `enabled` on `claude-sonnet-4-6`; the adaptive and omitted-config paths on current models are covered by the new guarded live tests — run them before merging.
- **`effort` auto-adaptive on a pre-4.6 Anthropic model** produces a 400 (`adaptive` unsupported there; `effort` itself was beta-gated pre-4.6 behind a header this harness never sends). Acceptable: the API error names the parameter.
- **The off-by-default prefix table bets that future models run thinking by default.** If Anthropic ships a future off-by-default model outside the legacy prefixes, resuming a thinking transcript on it with no explicit config would emit thinking blocks into a thinking-off request → 400. Mitigations: the bet matches the Sonnet 5/Fable 5 direction; an explicit `extra_body` thinking config pins behavior either way; the fix is a one-line prefix addition.
- **Non-streaming ceiling:** 16384 keeps long turns inside typical timeouts but a turn that genuinely produces ~16K tokens can still exceed 120s; callers can raise `request_timeout`. Streaming is the real fix and is out of scope.

## Out of scope (deferred)

- Streaming responses (unlocks 64K–128K `max_tokens`).
- `strict: true` on Anthropic tool-parameter schemas.
- Fable 5 refusal `stop_reason` handling and server-side fallbacks.
- Structured-output beta headers for pre-4.6 models.
- Dedicated `builtin_parallel_llm_max_tokens`/`builtin_parallel_llm_effort` overrides (parallel-LLM tool inherits parent values).
