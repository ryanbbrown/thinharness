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
- ThinHarness runtime requirement: `rg` must resolve on `PATH`; preflight executes a real `jsonl_search` probe and fails before model calls if it does not
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

The ThinHarness-only ripgrep diagnostic uses [`rg_diagnostic_selection.json`](rg_diagnostic_selection.json) and [`run_rg_diagnostics.py`](run_rg_diagnostics.py). Its four cells are new stochastic replicates. They never replace the original paired-wave evidence and the runner cannot launch native cells.

The prompt-alignment diagnostic uses [`prompt_alignment_selection.json`](prompt_alignment_selection.json) and [`run_prompt_alignment.py`](run_prompt_alignment.py). It runs five new ThinHarness-only replicates selected by the largest positive original ThinHarness-minus-native query-cost difference. [`prompt_alignment_diff.md`](prompt_alignment_diff.md) records every localized change from the current native prompt and instruction. The runner validates those native sources, the rendered prompts, the selection, ripgrep, routes, receipts, hashes, scoring evidence, accessed data forms, and secret boundaries.

The clean paired comparison uses [`clean_pair_selection.json`](clean_pair_selection.json), [`clean_pair_config.json`](clean_pair_config.json), [`clean_pair_prompt_diff.json`](clean_pair_prompt_diff.json), and [`clean_pair_cost_estimate.json`](clean_pair_cost_estimate.json). Its seeded ten-question selection covers all seven question types, five questions per domain, all three web environments, five deterministic and five LLM-judged questions, and both image paths. Prior costs and outcomes are not selection inputs. Native and ThinHarness each run one query attempt, with native-aligned retrieval policy, working ripgrep in both runtimes, and a 2 USD dynamic API-equivalent hard cap. [`resume_clean_pair_readers.py`](resume_clean_pair_readers.py) resumes only failed reader and scoring stages from preserved prompt rows. Results are in [`evidence/luna-xhigh-clean-pair-v1/REPORT.md`](evidence/luna-xhigh-clean-pair-v1/REPORT.md).

The terminal command injects `.env` through `uv`; scripts record only API-key variable names and presence checks. They do not print or persist secret values.
