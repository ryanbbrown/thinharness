# LongMemEval native-prompt alignment diagnostic

Five new ThinHarness-only stochastic replicates. The query prompt is the only intentionally changed runtime variable.

| Question | Evidence | Input | Tools | Failures | Latency | Query cost | Score |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 109b334c | original native | 362,313 | 9 | 1 | 43.21s | $0.02261250 | 1 |
| 109b334c | original ThinHarness | 4,217,890 | 47 | 4 | 157.82s | $0.18527498 | 1 |
| 109b334c | aligned-prompt ThinHarness | 664,239 | 21 | 2 | 66.04s | $0.03310812 | 1 |
| f61a096f | original native | 531,836 | 13 | 1 | 45.66s | $0.02635228 | 1 |
| f61a096f | original ThinHarness | 3,929,999 | 43 | 4 | 147.90s | $0.15035950 | 1 |
| f61a096f | aligned-prompt ThinHarness | 180,877 | 14 | 0 | 50.40s | $0.02125538 | 0 |
| ee75b921 | original native | 275,701 | 14 | 1 | 47.28s | $0.01456892 | 0 |
| ee75b921 | original ThinHarness | 898,559 | 22 | 4 | 95.63s | $0.05556418 | 1 |
| ee75b921 | aligned-prompt ThinHarness | 111,384 | 10 | 0 | 32.58s | $0.00977430 | 0 |
| 18b91103 | original native | 1,005,361 | 18 | 1 | 106.45s | $0.04945982 | 1 |
| 18b91103 | original ThinHarness | 1,603,048 | 24 | 4 | 101.01s | $0.07806182 | 0 |
| 18b91103 | aligned-prompt ThinHarness | 190,584 | 11 | 0 | 31.16s | $0.01836762 | 1 |
| b82d0dd6 | original native | 720,375 | 13 | 1 | 72.63s | $0.03832578 | 1 |
| b82d0dd6 | original ThinHarness | 1,002,698 | 21 | 2 | 101.64s | $0.06043504 | 1 |
| b82d0dd6 | aligned-prompt ThinHarness | 412,488 | 14 | 0 | 70.09s | $0.03169098 | 0 |

## Data forms actually accessed

Access means a successful direct tool call or a completed/timed-out recursive search scope. Visible-only file names from listings do not count as file access.

| Question | Global JSONL | Per-trajectory JSONL | Raw trajectory JSON | Summaries | Full spills read |
| --- | ---: | ---: | ---: | ---: | ---: |
| 109b334c | no | yes | yes | yes | 0 |
| f61a096f | no | yes | no | yes | 0 |
| ee75b921 | no | yes | no | yes | 0 |
| 18b91103 | no | yes | no | yes | 0 |
| b82d0dd6 | no | yes | no | yes | 0 |

- Full spill artifacts: 14 created, 0 read.
- Failed tool calls: 3; failed search/jsonl_search calls: 2.
- No aligned cell used global JSONL. All five used summaries and targeted per-trajectory JSONL.

## Totals

- Aligned/original ThinHarness query-cost ratio: 0.216x.
- Aligned/original native query-cost ratio: 0.755x.
- Correct: native 4/5, original ThinHarness 4/5, aligned prompt 2/5.
- New-cell query API-equivalent cost: $0.11419640.
- New-cell total API-equivalent cost: $0.12998585.
- Scores are new stochastic outcomes and do not replace the paired wave.
