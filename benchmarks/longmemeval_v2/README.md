# LongMemEval-V2 Luna xhigh paired wave

This benchmark compares the official AgentRunbook-C V2 query harness with native ThinHarness on the same frozen 14-question LongMemEval-V2-Small wave.

## Frozen design

- Selection: [`selection.json`](selection.json)
- Seed: `thinharness-lmev2-luna-xhigh-wave1-v1`
- Selection rule: lowest SHA-256 rank in each domain × seven-question-type cell
- Order: question type order in `selection.json`, web before enterprise
- Execution order: official/native cell, then ThinHarness cell, for each question
- Query model: `gpt-5.6-luna`, xhigh reasoning, direct OpenAI API
- Official query method: `agentrunbook_c_v2`, online learning disabled
- ThinHarness tools: `read`, `search`, `jsonl_search`, `list`, and `glob`
- Reader: `qwen/qwen3.5-9b`, temperature 0.6, top-p 0.95, top-k 20, thinking enabled, 200K memory-context limit, 20K output limit
- Reader route: direct OpenRouter request pinned to Parasail, no fallback, required parameters, data collection denied
- Evaluator: `gpt-5.2`, medium reasoning, direct OpenAI API, official evaluator specifications
- Official harness revision: `2cc8c540bdb87fe6761629b585e727e1c4704520`
- Dataset revision: `f152293e235517d504809563c833d7190b8c713b`

The official harness receives the committed instrumentation patch in [`official-direct-api-instrumentation.patch`](official-direct-api-instrumentation.patch). The patch only adds explicit reader routing and response/evaluator usage receipts. It does not change query prompts, query tools, reader prompts, or scoring rules.

## Retry and checkpoint policy

Each query harness can make at most three query attempts when it fails to produce a valid memory result. The official OpenAI Agents SDK and ThinHarness provider transports make no hidden HTTP retries for query calls. The unchanged official reader and evaluator OpenAI clients allow at most 10 transport retries. A complete cell is never rerun. The runner checkpoints one receipt after every cell and preserves every query attempt.

The runner stops before launching another cell when an API response reports a genuine credit, billing, or quota exhaustion condition. Other scored model failures remain scored outcomes. An unscored infrastructure failure exits for repair instead of silently rerunning the question.

## Durable artifacts

Raw results live under `.benchmark-runs/longmemeval-v2-luna-xhigh-wave1/` and are excluded from Git because they include large memory workspaces and traces. Compact receipts, hashes, paired scores, and costs are written under `benchmarks/longmemeval_v2/evidence/luna-xhigh-wave1/` for local commit after the run.

The terminal command injects `.env` through `uv`; scripts record only API-key variable names and presence checks. They do not print or persist secret values.
