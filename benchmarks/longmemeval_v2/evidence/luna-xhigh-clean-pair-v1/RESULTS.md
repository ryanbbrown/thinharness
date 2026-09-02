# LongMemEval clean paired comparison

Scored outcomes: native 9/10; ThinHarness 8/10. Unscored reader failures: native 0; ThinHarness 0.
All 10 pairs received scores: native 9/10; ThinHarness 8/10; mean paired difference -0.100.
Fresh query API-equivalent cost: native 0.21919964 USD; ThinHarness 0.25249354 USD; ratio 1.152x (paired bootstrap 95% interval 0.758x to 1.786x).

| Question | Type | Native score | Thin score | Native query cost | Thin query cost | Native / Thin input | Native / Thin tools |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 10466872 | static-environment | 1.0 | 1.0 | 0.00729780 USD | 0.02853454 USD | 104,361 / 391,325 | 7 / 14 |
| 634973c3 | static-environment | 1.0 | 1.0 | 0.01918440 USD | 0.00965874 USD | 343,263 / 114,021 | 10 / 9 |
| eaba5c44 | static-environment-abs | 1.0 | 0.0 | 0.02095474 USD | 0.05891048 USD | 364,508 / 1,090,405 | 13 / 26 |
| dea446d0 | dynamic-environment | 1.0 | 1.0 | 0.00858866 USD | 0.00695162 USD | 133,057 / 51,517 | 8 / 9 |
| b54161f8 | dynamic-environment | 1.0 | 1.0 | 0.01950130 USD | 0.03462456 USD | 343,133 / 391,077 | 11 / 18 |
| 11cc7ac2 | dynamic-environment-abs | 0.0 | 0.0 | 0.03399876 USD | 0.03939102 USD | 735,801 / 799,926 | 14 / 14 |
| 07b49858 | procedure | 1.0 | 1.0 | 0.05924146 USD | 0.03603366 USD | 1,234,949 / 459,900 | 15 / 15 |
| 7df9e2ff | procedure-abs | 1.0 | 1.0 | 0.01884622 USD | 0.01357512 USD | 251,174 / 160,593 | 7 / 12 |
| 77258cda | errors-gotchas | 1.0 | 1.0 | 0.01087284 USD | 0.00833970 USD | 165,246 / 65,418 | 8 / 8 |
| af2ebaed | errors-gotchas | 1.0 | 1.0 | 0.02071346 USD | 0.01647410 USD | 342,628 / 236,746 | 9 / 14 |

## Aggregate telemetry

| Metric | Native | ThinHarness |
| --- | ---: | ---: |
| Input tokens | 4,018,120 | 3,760,928 |
| Cached input tokens | 3,449,962 | 2,993,627 |
| Ordinary input tokens | 568,158 | 767,301 |
| Output tokens | 30,474 | 32,634 |
| Reasoning tokens | 18,731 | 19,730 |
| Requests | 112 | 86 |
| Tool calls | 102 | 139 |
| Search calls | 48.00 | 77.00 |
| Failed tool calls | 0.00 | 5.00 |
| Tool-result characters | 3,363,327.00 | 2,035,805.00 |
| Query wall time | 418.61 | 402.01 |
| Evidence context tokens | 120,295.00 | 107,572.00 |
| Valid evidence spans | 11.00 | 11.00 |
| Invalid evidence spans | 0.00 | 0.00 |
| Evidence states | 19.00 | 18.00 |

## Cost

- Native query: 0.21919964 USD; ThinHarness query: 0.25249354 USD API-equivalent.
- Combined query: 0.47169318 USD API-equivalent.
- Reader: 0.03045725 USD API-equivalent; 0.03045725 USD provider-reported.
- Evaluator: 0.02284100 USD API-equivalent.
- Total: 0.52499143 USD API-equivalent, below the 2 USD cap.
- OpenAI responses report tokens rather than billed dollars; query and evaluator costs use the frozen rates.
