# LongMemEval-V2 Luna xhigh paired wave results

The 14-question wave finished all 28 cells: 14 official AgentRunbook-C V2 cells and the same 14 native ThinHarness cells.

## Quality

| Result | Official/native | ThinHarness |
| --- | ---: | ---: |
| Overall | 11/14 (78.6%) | 11/14 (78.6%) |
| Deterministic scoring | 6/6 | 5/6 |
| LLM-judged scoring | 5/8 | 6/8 |
| Web | 4/7 | 6/7 |
| Enterprise | 7/7 | 5/7 |

The harnesses agreed on 10 questions. ThinHarness alone was correct on 2; the official harness alone was correct on 2. The exact McNemar test on the four disagreements has p=1.0. The approximate 95% interval for the paired accuracy difference is -29 to +29 percentage points. Each harness's 11/14 Wilson interval is 52.4% to 92.4%.

This sample does not establish parity or superiority. It shows no quality difference in this wave.

## Query-harness efficiency

| Query metric | Official/native | ThinHarness | ThinHarness / native |
| --- | ---: | ---: | ---: |
| Mean latency | 49.68s | 78.61s | 1.58× |
| Median latency | 46.47s | 77.11s | 1.66× |
| Input tokens | 5,518,990 | 14,741,564 | 2.67× |
| Cached input tokens | 4,757,910 | 12,679,002 | 2.66× |
| Ordinary input tokens | 761,080 | 2,062,562 | 2.71× |
| Output tokens | 48,940 | 75,593 | 1.54× |
| Reasoning output tokens | 30,487 | 55,970 | 1.84× |
| Model requests | 176 | 169 | 0.96× |
| Tool calls | 162 | 281 | 1.73× |
| Query API-equivalent cost | $0.3061 | $0.7568 | 2.47× |

The official harness's 49.68-second mean is close to the published 48.94-second full-set mean. In this wave, ThinHarness matched quality but used more tokens, tools, time, and API-equivalent spend. ThinHarness made slightly fewer model requests, so the difference came from larger turns and more tool work rather than more model turns.

## Cost

The 28 final cells recorded:

- Query API-equivalent cost: $1.06290624.
- Parasail reader provider-reported cost: $0.06938665.
- GPT-5.2 evaluator API-equivalent cost: $0.03637025.
- Total final-cell API-equivalent cost: $1.16866314.

OpenAI responses report tokens, not billed dollars. Query and evaluator costs use the frozen price assumptions in the runner. Parasail reported its reader cost in each response.

Setup recovery added $0.00050585 of confirmed reader cost. One interrupted native query and the first reader/evaluator responses for one recovered cell have unknown cost. The exact account charge is therefore not available. [`setup_incidents.json`](setup_incidents.json) preserves these exclusions.

## Reproducibility and validation

- Selection: [`../../selection.json`](../../selection.json)
- Final paired summary: [`final_summary.json`](final_summary.json)
- Compact cell receipts: [`cell_receipts.json`](cell_receipts.json)
- Validation receipt: [`validation.json`](validation.json)
- Raw artifact hashes: [`raw_artifact_manifest.json`](raw_artifact_manifest.json)
- Raw run root: `.benchmark-runs/longmemeval-v2-luna-xhigh-wave1/`

Validation recomputed all 12 deterministic scores, matched all 16 LLM-judged scores to evaluator receipts, verified 43,453 cell artifacts and 44,086 raw-run files, confirmed Parasail handled all 28 reader calls, and found no injected API-key value in the scanned artifacts.
