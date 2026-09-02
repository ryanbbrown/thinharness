# LongMemEval clean ten-pair report

## Conclusion

The fresh query-cost ratio is 1.152 times: 0.25249354 USD API-equivalent for ThinHarness versus 0.21919964 USD for native. The paired bootstrap 95% interval is 0.758 to 1.786 times. This run rejects the old 2.47-times figure as a description of the aligned, working-ripgrep setup, but ten questions do not establish a stable cost difference.

The run does not provide a complete ten-pair accuracy comparison. Temporary Parasail reader rate limits left two native and two ThinHarness query cells unscored. Six pairs received scores on both sides, and both harnesses were correct on all six. Across independently scored cells, native was correct on 8 of 8 and ThinHarness on 6 of 8, but the unmatched missing scores mean those rates are not a paired quality estimate.

## Frozen selection

The seed was `thinharness-lmev2-luna-xhigh-clean-pair-v1`. The selector chose the lowest SHA-256 rank in ten predeclared question-type, domain, and environment slots. Prior cost, score, trace, and outcome data were not inputs. The selection covers all seven question types, five questions per domain, all three web environments, five deterministic and five LLM-judged questions, and both image paths.

1. `10466872` — web static environment
2. `634973c3` — enterprise static environment
3. `eaba5c44` — web static-environment abstention
4. `dea446d0` — web dynamic environment
5. `b54161f8` — enterprise dynamic environment
6. `11cc7ac2` — enterprise dynamic-environment abstention
7. `07b49858` — web procedure
8. `7df9e2ff` — enterprise procedure abstention
9. `77258cda` — web errors and gotchas, image
10. `af2ebaed` — enterprise errors and gotchas, image

## Controlled configuration

Both sides used direct OpenAI GPT-5.6 Luna with xhigh reasoning, at most 30 model requests or turns, one query attempt, zero query HTTP retries, no output retry, online learning disabled, the same question order, and the same official reader, evaluator, and scoring rules. The reader was qwen/qwen3.5-9b through OpenRouter pinned to Parasail without fallback. The evaluator was direct OpenAI GPT-5.2 with medium reasoning.

Both preflights selected `/Users/ryanbrown/.pi/agent/bin/rg`. Native completed a shell search probe, and ThinHarness completed a real jsonl_search probe. The frozen prompt diff preserves native wording and retrieval policy except for documented tool, corpus-path, question-delivery, image-delivery, and structured-output differences. The intended remaining variables were the harness and tool implementation and the normalized ThinHarness JSONL corpus.

## Results

| Metric | Native | ThinHarness | Thin / native |
| --- | ---: | ---: | ---: |
| Query API-equivalent cost | 0.21919964 USD | 0.25249354 USD | 1.152 |
| Input tokens | 4,018,120 | 3,760,928 | 0.936 |
| Cached input | 3,449,962 | 2,993,627 | 0.868 |
| Ordinary input | 568,158 | 767,301 | 1.351 |
| Output tokens | 30,474 | 32,634 | 1.071 |
| Reasoning tokens | 18,731 | 19,730 | 1.053 |
| Requests | 112 | 86 | 0.768 |
| Tool calls | 102 | 139 | 1.363 |
| Search calls | 48 | 77 | 1.604 |
| Tool-result characters | 3,363,327 | 2,035,805 | 0.605 |
| Query wall time | 418.61 seconds | 402.01 seconds | 0.960 |

ThinHarness produced 11 valid spans covering 18 states; native produced 11 valid spans covering 19 states. Neither produced an invalid span. ThinHarness sent 107,572 evidence-context tokens to readers; native sent 120,295. The detailed per-question cost, score, and telemetry table is in `RESULTS.md`.

## Cost and incidents

Combined query cost was 0.47169318 USD API-equivalent. Successful reader calls reported 0.02450550 USD. Evaluator cost was 0.01711850 USD API-equivalent. The preserved final-cell total was 0.51331718 USD, well below the 2 USD cap and the conservative 1.86048304 USD estimate. OpenAI did not report billed query or evaluator dollars, and failed rate-limited reader calls have no provider cost receipt, so exact account spend is unavailable.

All 20 paid query cells completed exactly one attempt. No query was rerun. Four reader calls ended with a Parasail upstream 429 after the configured transport retries: native `eaba5c44`, ThinHarness `b54161f8`, native `11cc7ac2`, and ThinHarness `af2ebaed`. They remain unscored. The earlier missing-dependency setup launch made zero model requests and is preserved separately.

## Cost cause and next step

Extra uncached context caused the small remaining cost gap. ThinHarness's extra 199,143 ordinary input tokens added 0.03982860 USD. Its 456,335 fewer cached tokens saved 0.00912670 USD, and extra output added 0.00259200 USD. ThinHarness stopped in fewer requests and returned less raw result text, but used more typed tool and search calls and ended with 35.1% larger final request contexts. The detailed artifact-only analysis is in `ROOT_CAUSE.md`.

Do not selectively rerun the four missing reader outcomes. The smallest no-paid improvement is request-component telemetry that separates fixed prompt, serialized tool schemas, assistant output, each tool result, image input, and accumulated history. The smallest supported tool-guidance correction is to make the jsonl_search span example match the accepted in-filter schema and state clearly that procedure_notes can be absent. Any new paid accuracy run needs new approval.
