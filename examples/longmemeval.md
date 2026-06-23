# LongMemEval-V2 - ThinHarness Benchmark

## Context

I used a [fork](https://github.com/ryanbbrown/LongMemEval-V2) of the LongMemEval-V2 benchmark to test whether ThinHarness is a strong general-purpose agent harness on a nontrivial non-coding task: information retrieval over long trajectory haystacks.

I scoped the comparison to the 127 dynamic questions in the small tier: `dynamic-environment` and `dynamic-environment-abs` across the web and enterprise domains. After reviewing the benchmark categories, this subset looked like the best test for my purposes: given a long history of interaction traces, can the memory layer efficiently and accurately find the state change, UI behavior, or environment fact needed to answer the question?

## Run Details

To get a local baseline for more detailed metrics than the leaderboard provides, I ran AgentRunbook-C with the resumable wrapper in `evaluation/scripts/run_agentrunbook_c_dynamic.sh`. That script plans the 127-question dynamic set, runs one question per output directory, and can be re-run to fill only missing questions. Running one question at a time is slower but was helpful when encountering intermittent issues running on my local machine. The matching ThinHarness wrapper is `evaluation/scripts/run_thinharness_dynamic.sh`.

This is a best-faith reproduction rather than an exact reproduction of the paper. The paper setup expects a local `Qwen/Qwen3.5-9B` reader deployment; I used `qwen/qwen3.5-9b` through OpenRouter. The paper uses Codex v0.117.0 for Codex and AgentRunbook-C; my rerun used the local Codex CLI available, `codex-cli 0.141.0`. Both the AgentRunbook-C rerun and ThinHarness run used `gpt-5.4-mini` with `xhigh` reasoning for the query-time memory agent and `gpt-5.2` for the evaluator.

For the ThinHarness run, I used only its generic built-in filesystem tools (`read`, `search`, `jsonl_search`, `list`, and `glob`). I did restructure the memory files into a JSONL-friendly corpus so its generic `jsonl_search` tool could work well; that seems acceptable here because AgentRunbook-C also creates a custom trajectory structure rather than using the vanilla Codex raw layout. I tried to keep the query-time system prompt as close as practical to AgentRunbook-C, changing the tool instructions only where ThinHarness needed to know how to use its built-in tools.

## Results

Across all 127 dynamic questions in the small tier:

| Metric | AgentRunbook-C rerun | ThinHarness |
| --- | --- | --- |
| Dynamic score | 72.4% (92/127) | 74.0% (94/127) |
| Non-abstention | 86.0% (74/86) | 84.9% (73/86) |
| Abstention | 43.9% (18/41) | 51.2% (21/41) |
| Memory query time | 151.9s avg, 129.1s median | 99.7s avg, 87.9s median |
| Memory-agent tokens / usage | 114.77M input, 1.32M output, 116.09M total | 60.14M input, 2.10M output, 62.24M total |
| Dynamic-subset LAFS | 3.76 | 9.73 (+5.96) |

The 72.4% accuracy for AgentRunbook-C matches the paper, but I would not treat this single consolidated run as a statistically signficant claim that ThinHarness has higher accuracy than AgentRunbook-C--I saw meaningful variance on a portion of the questions when doing targeted reruns. The result does make me reasonably confident that ThinHarness at least matches AgentRunbook-C's performance on this slice, and the published leaderboard reference for vanilla Codex is materially lower than both.

The memory query time is the harness-measured time around `memory.query(...)`: it includes the query-time memory retrieval agent, but not the downstream reader, scorer, or prior runtime input generation. The timing comparison isn't perfect (local Codex CLI vs. OpenAI API), but the 46.4% lower token usage indicates that ~34% time savings is probably in the right ballpark. Note that the paper only provides a single aggregate query time figure across all questions, 108.3s, which is far lower than the 151.9s above but includes all questions in the small tier (some of which may have been faster).

