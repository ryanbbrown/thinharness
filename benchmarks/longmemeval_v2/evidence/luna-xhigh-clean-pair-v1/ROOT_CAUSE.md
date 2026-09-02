# Clean paired cost analysis

ThinHarness was 1.152 times native query cost in this fresh run, not 2.47 times. Native query cost was 0.21919964 USD API-equivalent; ThinHarness was 0.25249354 USD. The paired bootstrap 95% interval is 0.758 to 1.786 times, so ten questions do not establish a stable cost disadvantage.

The 0.03329390 USD difference came from uncached input. ThinHarness had 199,143 more ordinary input tokens, adding 0.03982860 USD. It had 456,335 fewer cached input tokens, saving 0.00912670 USD, and 2,160 more output tokens, adding 0.00259200 USD. Thus extra ordinary input accounts for 119.6% of the net gap; lower cached input offsets 27.4%, and output adds 7.8%.

| Metric | Native | ThinHarness | Thin / native |
| --- | ---: | ---: | ---: |
| Total input | 4,018,120 | 3,760,928 | 0.936 |
| Ordinary input | 568,158 | 767,301 | 1.351 |
| Cached input | 3,449,962 | 2,993,627 | 0.868 |
| Final request context, summed | 567,852 | 767,073 | 1.351 |
| First request input, summed | 17,430 | 33,660 | 1.931 |
| Requests | 112 | 86 | 0.768 |
| Tool calls | 102 | 139 | 1.363 |
| Search calls | 48 | 77 | 1.604 |
| Tool-result characters | 3,363,327 | 2,035,805 | 0.605 |
| Query wall time | 418.61 seconds | 402.01 seconds | 0.960 |

The fixed prompt wording is close. The native query prompt is 582 bytes and its instruction is 7,575 bytes. The localized ThinHarness versions are 611 and 7,751 bytes. ThinHarness receives them inline, while native reads its instruction as a tool result. ThinHarness also exposes five typed tools instead of native's shell and apply_patch tools. The traces do not preserve the final serialized tool-schema token count, so the exact split between prompt, schema, assistant text, and tool arguments cannot be calculated. The first-request difference is 16,230 tokens across ten cells, about 8.1% of the final-context difference.

ThinHarness did not return more raw result text or run for more turns. It returned 39.5% fewer tool-result characters, made 23.2% fewer model requests, and finished query work 4.0% faster. However, it made 36.3% more tool calls and 60.4% more search calls. Its final conversations were 35.1% larger and had a lower cache share: 79.6% versus 85.9%. The extra tool invocations, typed arguments, JSONL formatting, inline fixed instructions, and assistant reasoning therefore produced more new context even though the aggregate result text was smaller.

The cost difference is concentrated, not general. ThinHarness was cheaper on six questions. Three high-exploration ThinHarness cells produced most positive cost: eaba5c44 added 0.03795574 USD, 10466872 added 0.02123674 USD, and b54161f8 added 0.01512326 USD relative to native. Native's procedure cell 07b49858 alone offset 0.02320780 USD. This concentration explains the wide cost-ratio interval.

The corpus transformation was used narrowly. Every ThinHarness cell read summaries and targeted per-trajectory JSONL. Only one cell reached a broad scope that exposed global JSONL and raw trajectory JSON. Twenty-nine truncated spill files were created and none was read. Data duplication was therefore available but was not broadly consumed. ThinHarness had five failed tool calls: three invalid jsonl_search in-filter shapes and two missing procedure_notes directories. Ripgrep was available in both runtimes and caused no failures.

Evidence size was comparable: each side returned 11 valid spans and no invalid spans. Native selected 19 states and 120,295 reader-context tokens; ThinHarness selected 18 states and 107,572 tokens. Six pairs received reader scores on both sides, and all six were correct for both. Four other pairs cannot support a paired accuracy comparison because temporary Parasail rate limits left two native and two ThinHarness cells unscored. ThinHarness's two scored failures occurred where the matching native cell was unscored, so this run cannot establish an accuracy difference.

The smallest next no-paid step is to persist serialized request-component sizes before any future run: fixed prompt, tool schemas, assistant output, each tool result, image payload, and accumulated history. The smallest code correction supported by these traces is to make jsonl_search span examples match the accepted in-filter schema and state that procedure_notes can be absent. Any new paid accuracy run needs separate approval; these twenty query cells must not be rerun selectively.
