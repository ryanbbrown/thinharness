# LongMemEval ripgrep diagnostic replicates

These are four new ThinHarness-only stochastic replicates. They preserve and do not replace the original paired-wave cells.

| Question | Evidence | Input tokens | Tool calls | Search failures | Latency | Query cost | Score |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| f61a096f | original native | 531,836 | 13 | 1 | 45.66s | $0.02635228 | 1 |
| f61a096f | original ThinHarness | 3,929,999 | 43 | 4 | 147.90s | $0.15035950 | 1 |
| f61a096f | rg ThinHarness replicate | 331,310 | 13 | 0 | 36.51s | $0.02147764 | 1 |
| 109b334c | original native | 362,313 | 9 | 1 | 43.21s | $0.02261250 | 1 |
| 109b334c | original ThinHarness | 4,217,890 | 47 | 4 | 157.82s | $0.18527498 | 1 |
| 109b334c | rg ThinHarness replicate | 1,300,709 | 33 | 0 | 82.68s | $0.07021264 | 1 |
| b82d0dd6 | original native | 720,375 | 13 | 1 | 72.63s | $0.03832578 | 1 |
| b82d0dd6 | original ThinHarness | 1,002,698 | 21 | 2 | 101.64s | $0.06043504 | 1 |
| b82d0dd6 | rg ThinHarness replicate | 2,306,426 | 42 | 2 | 133.46s | $0.11890048 | 0 |
| 18b91103 | original native | 1,005,361 | 18 | 1 | 106.45s | $0.04945982 | 1 |
| 18b91103 | original ThinHarness | 1,603,048 | 24 | 4 | 101.01s | $0.07806182 | 0 |
| 18b91103 | rg ThinHarness replicate | 2,078,603 | 33 | 1 | 92.80s | $0.08067142 | 1 |

## Totals

- Replicate/original ThinHarness input ratio: 0.560x.
- Replicate/original ThinHarness query-cost ratio: 0.614x.
- Replicate/original native input ratio: 2.297x.
- Replicate/original native query-cost ratio: 2.130x.
- Search failures: 14 original ThinHarness, 3 replicate; replicate rg-unavailable failures: 0.
- Correct scores: 4/4 native, 3/4 original ThinHarness, 3/4 replicate.
- Replicate final-cell API-equivalent cost: $0.37462468.
- Scores are stochastic outcomes. This diagnostic cannot attribute score changes only to ripgrep.
