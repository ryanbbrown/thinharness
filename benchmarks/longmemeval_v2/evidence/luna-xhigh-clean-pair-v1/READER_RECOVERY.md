# Reader recovery

## Why the original readers returned 429

The reader model's small size did not prevent a capacity limit. Every final error body said that qwen/qwen3.5-9b was temporarily rate-limited upstream. OpenRouter identified provider_name as Parasail, is_byok as false, and limit_source as upstream_provider_shared_pool. The request went to the OpenRouter API, pinned to Parasail with fallback disabled. OpenRouter returned the HTTP 429, but its structured body identifies the constrained resource as Parasail's shared upstream pool rather than an OpenRouter account-credit or billing limit.

Each cell contained one question. Reader concurrency was one, prompt-build concurrency was one, and the outer benchmark ran cells sequentially. The failures were therefore not caused by concurrent requests from this benchmark.

## Original retry policy and evidence

The official harness created an OpenAI Python 3.7.0 AsyncOpenAI client with max_retries set to 10. The SDK makes the initial request plus up to 10 retries, so each failed reader exhausted 11 HTTP attempts before raising the preserved final RateLimitError. It retries HTTP 429 responses. If retry-after-ms or Retry-After is present, positive, and at most 120 seconds, the SDK uses that delay. A Retry-After above 120 seconds stops retrying. Without a usable header, the ten delays use bases 0.5, 1, 2, 4, and then six 8-second delays, each multiplied by random jitter from 0.75 through 1.0. The total fallback wait is therefore between 41.625 and 55.5 seconds.

The original runner did not log individual transport attempts, calculated delays, or response headers. It preserved only each final error body. Therefore, the exact original jittered delays and whether any original response contained Retry-After cannot be recovered. No Retry-After, rate-limit, request-ID, or provider response headers appear anywhere in the preserved raw run or terminal log. Provider identity comes from the error body, not an original response header. This telemetry limit is stated rather than filled with an estimate.

The four original final failures were:

| Cell | Final response | Route evidence |
| --- | --- | --- |
| `03-eaba5c44-native` | HTTP 429 after 11 SDK attempts | Parasail; upstream_provider_shared_pool; not BYOK |
| `05-b54161f8-thinharness` | HTTP 429 after 11 SDK attempts | Parasail; upstream_provider_shared_pool; not BYOK |
| `06-11cc7ac2-native` | HTTP 429 after 11 SDK attempts | Parasail; upstream_provider_shared_pool; not BYOK |
| `10-af2ebaed-thinharness` | HTTP 429 after 11 SDK attempts | Parasail; upstream_provider_shared_pool; not BYOK |

The original logs are `.benchmark-runs/longmemeval-v2-luna-xhigh-clean-pair-v1/logs/<cell>.log`. Their hashes and final error lines are frozen in `reader_recovery_preflight.json`. The original terminal log is `/Users/ryanbrown/.bb/thread-storage/thr_dgzrszp3tc/terminal-jobs/job_ee6c3d6e-12bc-4e39-81e5-cb5721ffabd3/output.log`.

## Recovery policy and outcome

Recovery reused each preserved prompt_rows.jsonl message payload. It did not invoke either query harness. It kept qwen/qwen3.5-9b, temperature 0.6, top_p 0.95, top_k 20, thinking enabled, 20,000 maximum output tokens, the OpenRouter endpoint, and the Parasail-only no-fallback route. Calls were sequential. The recovery client disabled hidden SDK retries so every HTTP attempt and safe response header could be recorded. It honored any positive Retry-After value. When no such header existed, it waited 5, 10, 20, 40, 80, then at most 120 seconds between temporary failures, with no fixed temporary-error attempt limit.

| Cell | Recovery HTTP attempts | Temporary 429s | Waits | Retry-After observed | Success provider | Score |
| --- | ---: | ---: | --- | --- | --- | ---: |
| `03-eaba5c44-native` | 5 | 4 | 5, 10, 20, 40 seconds | None | Parasail | 1 |
| `05-b54161f8-thinharness` | 1 | 0 | None | None | Parasail | 1 |
| `06-11cc7ac2-native` | 1 | 0 | None | None | Parasail | 0 |
| `10-af2ebaed-thinharness` | 2 | 1 | 5 seconds | None | Parasail | 1 |

All five recovery 429 responses again named Parasail and upstream_provider_shared_pool in their bodies. Their only captured safe headers were server: cloudflare and distinct cf-ray values. They contained no Retry-After, x-ratelimit, x-openrouter, or x-request-id header. The successful response metadata named Parasail for all four readers.

The recovered readers added 0.00595175 USD provider-reported cost. Three required GPT-5.2 evaluator calls added 0.00572250 USD API-equivalent. The recovery added 0.01167425 USD, bringing the complete run total to 0.52499143 USD API-equivalent. The query total and ratio did not change.

## Complete paired accuracy

Native scored 9 of 10; ThinHarness scored 8 of 10. Both scored 5 of 5 on deterministic questions. Native scored 4 of 5 and ThinHarness 3 of 5 on LLM-judged questions. The mean paired difference for ThinHarness minus native is -0.10. The only discordant pair is eaba5c44, where native scored 1 and ThinHarness 0. Both scored 0 on 11cc7ac2. Nine pairs tied. One discordant observation is not enough to establish a stable accuracy difference.

Original failure receipts remain at each recovered cell's `reader_recovery/original_failure_receipt.json`. Every transport request is in `reader_recovery/transport_attempts.jsonl`; reader outputs, evaluator receipts, and scored checkpoints are beside it. Validation still finds exactly 20 query summaries, all under attempt_001, and no attempt_002. The recovery terminal log is `/Users/ryanbrown/.bb/thread-storage/thr_dgzrszp3tc/terminal-jobs/job_dc9c110b-8d24-4199-901c-d68cf29d7cff/output.log`.
