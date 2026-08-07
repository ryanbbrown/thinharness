# Provider request retries with backoff — plan v2

Add bounded retries with backoff to every HTTP request made by ThinHarness's built-in OpenAI, Anthropic, and OpenRouter providers. Keep retries below model-session state so a failed HTTP attempt cannot duplicate transcript entries or tool outputs.

This revision incorporates the single review round in `.reviews/plans/provider-request-retries/`. It also replaces the separate `parallel_llm` provider retry loop with the shared transport policy. One request must have one retry budget.

## Goal

A transient provider failure currently aborts a normal agent run. `Provider.post_json()` performs one HTTP request and converts each HTTP failure into `ProviderError`. All built-in model adapters use this transport for initial turns, tool continuations, structured-output corrections, resume requests, approval resumes, subagents, and one-shot parallel completions.

After this change:

- transient failures retry automatically before they fail a run or one-shot completion;
- retries use bounded exponential backoff with jitter and respect a usable `Retry-After` value;
- permanent client errors still fail immediately;
- the final exhausted error keeps the existing `ProviderError` and run-failure behavior;
- retries do not consume logical model request limits or duplicate session state;
- callers can set the retry count and base backoff in `HarnessConfig`;
- all built-in request paths use one retry policy and one attempt budget.

## Retry seam and custom-model boundary

Implement retries in `Provider.post_json()`, not around `ModelSession.start()` or continuation methods. Session methods can append user and tool state before they call the provider. Retrying those methods could append the same state more than once.

The shared HTTP transport covers every built-in provider request without repeating session mutation. It does not retry arbitrary exceptions from custom `Model` or `ModelSession` implementations. ThinHarness cannot safely repeat an arbitrary stateful session method after it raises. Custom models can retry behind their own session boundary.

Remove the outer provider retry loop from `parallel_llm`. Delete `parallel_llm_max_attempts`, `ParallelLlmTool.max_attempts`, and the private status, delay, and sleep helpers. Thread the shared request retry settings into inferred alternate parallel models. A direct `ParallelLlmTool` model reference receives the same request retry constructor settings. A supplied model object owns its provider settings.

This is an intentional breaking cleanup. Retaining both retry loops would permit sixteen HTTP attempts under the current defaults and would keep two divergent status policies.

## Retry classification

Retry these failures:

- HTTP 408, 409, 425, and 429;
- HTTP 500 through 599;
- `httpx.TransportError`, except `httpx.UnsupportedProtocol`.

HTTP 409 and all 5xx responses match the broad retry behavior used by common provider SDKs. HTTP 425 preserves ThinHarness's existing parallel completion policy.

Do not retry:

- other HTTP 400 through 499 responses;
- `httpx.UnsupportedProtocol`;
- other `httpx.HTTPError` subclasses outside `TransportError`;
- local authentication failures raised while building headers;
- successful responses with invalid JSON;
- response normalization or structured-output validation failures;
- arbitrary custom model or session exceptions.

Retries are at-least-once HTTP delivery. A read, write, protocol, or timeout failure can occur after the provider accepts the POST. A replay can therefore duplicate provider work or charges. The built-in APIs do not share a portable idempotency-key contract, so this feature documents that risk instead of adding provider-specific keys.

## Public configuration

Add these fields to `HarnessConfig`:

```python
request_retries: int = Field(default=3, ge=0, le=10)
request_retry_backoff: float = Field(default=1.0, ge=0)
```

`request_retries` counts retries after the first attempt. The default permits four total attempts. Zero preserves single-attempt behavior.

`request_retry_backoff` is the initial delay in seconds. The exponential base for retry index `n` is `base * 2**n`. Add random positive jitter from zero through 25% of that base. Cap the final delay at 60 seconds.

A zero base removes exponential delay. It does not suppress a valid `Retry-After` value.

Thread both values through `Harness.__init__` and `infer_model()` into each built-in `Provider`. Add matching keyword arguments to `Provider`, `OpenAIProvider`, `AnthropicProvider`, and `OpenRouterProvider`. Direct provider construction receives the same defaults and enforces the same bounds with `ValueError`.

Thread the parent values into named subagent model overrides and inferred parallel models. Shared parent and child model objects already share their configured provider.

Do not add a dependency. Use `asyncio.sleep()`, `random`, and standard-library date parsing.

## Retry-After and delay behavior

For retryable HTTP responses, inspect `Retry-After` through `httpx.Headers`.

- Accept a non-negative numeric delay in seconds.
- Accept an HTTP date and convert it to a non-negative delay from current UTC time.
- Treat a parsed date without timezone information as UTC.
- Ignore malformed, negative, non-finite, or otherwise unusable values.
- Use the larger of the jittered exponential delay and the parsed `Retry-After` delay.
- Cap the final sleep at 60 seconds.
- Never let `Retry-After` reduce the exponential delay.

Do not read provider-specific rate-limit reset headers in this feature.

## Attempt lifecycle

`Provider.post_json()` performs at most `request_retries + 1` attempts.

- Build and validate headers once before the first attempt.
- Reuse the same URL, JSON payload, headers, and `AsyncClient` for every attempt.
- Drain each intermediate retryable response before sleeping so the connection returns to the pool.
- On the final attempt, raise the error from that attempt. A final transport error after earlier status errors has `status_code=None` and the transport-error message.
- Parse JSON only after a successful status. Invalid JSON remains non-retryable.
- Let `asyncio.CancelledError` propagate during requests and sleeps. Never catch `BaseException`.
- Log each scheduled retry at info level with provider name, failure class or status, retry number, retry limit, and delay. Do not log payloads or response bodies.

`request_timeout` remains a per-attempt timeout. One logical request is bounded by the configured attempt count, each attempt timeout, and three capped default retry delays. This feature does not add a separate total deadline.

## Accounting, streaming, and tracing

A retry is another transport attempt for the same logical model request.

- Increment `RunUsage.model_requests` only after a complete `ModelTurn`, as today.
- Check `max_model_requests` once before the logical model request, as today.
- Do not add transport attempts to `RunUsage`. Failed responses have no normalized token usage, and provider transport does not own run state.
- Emit one `ModelRequestStartedEvent` and at most one `ModelMessageEvent` for one logical model request.
- Keep one model span for one logical request. A recovered transport failure does not mark the successful model span as failed.
- If all attempts fail, preserve current classification: the run ends with `stop_reason="provider_error"` and raises `HarnessError` from the final `ProviderError` message.
- In `parallel_llm`, count one logical session request per prompt or structured-output correction. Internal transport attempts do not increase its `model_requests` result field.

Do not add a public retry stream event. `ModelRetryEvent` describes a model-visible corrective turn after structured-output or tool failure. An HTTP replay is not model-visible.

## Implementation steps

### 1. Record behavior before code changes

After this plan review and before code implementation, add a `Provider Request Retries` section to `docs/behavior.md`.

Record the retryable failures, permanent exclusions, defaults, backoff and jitter, delay cap, `Retry-After`, at-least-once risk, cancellation, logical accounting, custom-model boundary, subagent behavior, and single-policy parallel behavior.

Update the existing `Provider Request Settings` section to state that `request_timeout` applies to each attempt.

### 2. Add and propagate retry settings

- Add `HarnessConfig.request_retries` and `HarnessConfig.request_retry_backoff` in `thinharness/core.py`.
- Pass them from `Harness.__init__` to `infer_model()`.
- Extend `infer_model()` and all built-in provider constructors.
- Validate direct constructor values before assignment.
- Pass both values when a named subagent infers an override model.
- Pass both values when `ParallelLlmTool` infers a model reference.
- Pass parent values through `create_parallel_llm_tool()`.

### 3. Consolidate `parallel_llm` retries

- Remove `parallel_llm_max_attempts` from `HarnessConfig`.
- Remove `max_attempts` from `ParallelLlmTool`.
- Delete `RETRYABLE_STATUS`, `_is_retryable`, `_retry_delay`, `_sleep_retry`, and the `random` import from `parallel_llm.py`.
- Replace the outer attempt loop with one session start. Let built-in provider transport handle transient HTTP retries.
- Update tests, documentation, examples, and end-to-end scripts that set the removed fields.

### 4. Implement the transport loop

- Add private helpers in `thinharness/providers.py` for constructor validation, retry classification, `Retry-After` parsing, and delay calculation.
- Inject or patch the random and sleep seams in tests. Do not duplicate the algorithm in fakes.
- Make `Provider.post_json()` implement the attempt lifecycle above.
- Keep final `ProviderError` text and `status_code` compatible.

### 5. Add behavior-level tests

Use `httpx.MockTransport` or an equivalent async client seam. Existing provider tests that only assert one failure must pass `request_retries=0` so they never sleep.

Cover:

- status matrix: 408, 409, 425, 429, 500, and 599 retry; 400, 401, 404, and `UnsupportedProtocol` do not retry;
- retryable status followed by success repeats the exact URL, payload, and headers;
- transport failure followed by success retries;
- exponential delays include controlled jitter and remain capped;
- numeric, aware-date, and naive-date `Retry-After` values influence delay;
- past, smaller, malformed, negative, and non-finite `Retry-After` values cannot reduce or break fallback backoff;
- valid `Retry-After` still sleeps when configured backoff is zero;
- exhausted status and transport attempts raise the final compatible `ProviderError`;
- mixed exhaustion reports the last attempt's failure;
- intermediate response bodies do not prevent client reuse;
- invalid JSON and permanent client failures make one attempt;
- cancellation during an in-flight request and during backoff propagates without another attempt;
- `request_retries=0` makes one attempt for status and transport failures;
- direct provider and `HarnessConfig` boundaries reject retries below zero, retries above ten, and negative backoff; accept zero and ten;
- settings reach inferred OpenAI, Anthropic, OpenRouter, named subagent, and alternate parallel models;
- a recovered agent request reports one logical `model_requests`, one request-start event, and one model message;
- retrying a tool continuation does not duplicate provider-visible tool output, `previous_response_id`, replay input, or transcript state;
- parallel completion performs only the shared transport attempts and reports one logical request;
- retry logs omit request and response content.

### 6. Update documentation

- Add the behavior contract before implementation.
- Update the provider settings group and limits text in `docs/docs.md`.
- Update the parallel LLM section and examples to remove `max_attempts` and explain the shared request policy.
- Update README and generated site feature text that claims parallel LLM has separate retry settings.
- Add an `Unreleased` section to `CHANGELOG.md`. Do not bump `pyproject.toml`; package versions change only in the release workflow.

### 7. Validate

Run:

```bash
uv run pytest tests/unit/test_providers.py tests/unit/test_harness.py tests/unit/test_streaming.py tests/unit/test_tracing.py tests/unit/test_subagents.py tests/unit/test_parallel_llm.py
uv run pytest
uv run ruff check thinharness/core.py thinharness/providers.py thinharness/subagents.py thinharness/tools/parallel_llm.py tests/unit/test_providers.py tests/unit/test_harness.py tests/unit/test_streaming.py tests/unit/test_tracing.py tests/unit/test_subagents.py tests/unit/test_parallel_llm.py
uv run pyright
```

Run the retry-focused tests repeatedly to confirm they are deterministic and perform no real backoff sleeps.

## Success criteria

- One transient built-in provider failure no longer aborts a normal agent run when retry budget remains.
- Every built-in provider request path uses one shared retry policy.
- Permanent failures remain fast failures.
- Backoff has jitter, stays bounded, and honors valid `Retry-After` guidance.
- A recovered retry leaves logical limits, usage, events, tracing, parallel counts, and session history unchanged.
- Exhaustion preserves current provider-error terminal behavior.
- Focused tests, full tests, Ruff, and Pyright pass.

## Out of scope

- Retrying arbitrary custom model session methods.
- Provider-specific idempotency keys.
- Retrying invalid JSON or semantic response-normalization failures.
- Retrying local tools, MCP calls, hooks, or structured-output validation.
- Circuit breakers, cross-run retry budgets, overall run deadlines, or distributed rate limiting.

## Review record

Exactly one review panel round ran against plan v1. Codex, Claude, and GLM reports are in `.reviews/plans/provider-request-retries/`. Plan v2 accepts the required findings on retry-policy consolidation, configuration propagation, direct constructor validation, exception classification, jitter, HTTP-date parsing, cancellation, state safety, and documentation. No second plan review round is part of this task. No unresolved decision requires user input.
