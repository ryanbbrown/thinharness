# LongMemEval prompt-alignment variance check

Result: 1 of 2 aligned ThinHarness regressions recovered on a second stochastic replicate. This is evidence of variance for f61a096f. b82d0dd6 failed in both aligned runs with the same evidence error. Two replicates cannot estimate either error rate.

## Results

| Question | Run | Reward | Requests | Tools | Input | Cached | Ordinary | Cache write | Output | Reasoning | Search failures | Query latency | Query cost |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| f61a096f | native | 1 | 14 | 13 | 531,836 | 463,594 | 68,242 | 0 | 2,860 | 1,159 | 1 | 45.66s | 0.02635228 USD |
| f61a096f | aligned attempt 1 | 0 | 7 | 14 | 180,877 | 111,509 | 69,368 | 69,347 | 4,293 | 3,232 | 0 | 50.40s | 0.02125538 USD |
| f61a096f | aligned replicate 2 | 1 | 19 | 21 | 1,346,806 | 1,185,693 | 161,113 | 161,056 | 4,875 | 3,254 | 0 | 72.79s | 0.06178646 USD |
| b82d0dd6 | native | 1 | 14 | 13 | 720,375 | 627,249 | 93,126 | 0 | 5,963 | 4,315 | 1 | 72.63s | 0.03832578 USD |
| b82d0dd6 | aligned attempt 1 | 0 | 9 | 14 | 412,488 | 317,899 | 94,589 | 94,562 | 5,346 | 4,099 | 0 | 70.09s | 0.03169098 USD |
| b82d0dd6 | aligned replicate 2 | 0 | 17 | 27 | 950,366 | 858,272 | 92,094 | 92,043 | 6,765 | 4,256 | 0 | 88.89s | 0.04370224 USD |

### f61a096f: recovered

- Gold and replicate-2 answer: nine.
- Attempt 1 found no direct customer-detail evidence and guessed five from a generic toolbar description.
- Replicate 2 found trajectory 06c70de4, state 38. The state lists all nine toolbar actions. The official reader returned nine, and deterministic scoring gave reward 1.
- Replicate 2 used 7.45 times attempt-1 input and cost 2.91 times as much for the query. It used 2.53 times native input and cost 2.34 times as much as native.

The query completed once. The first scoring runtime then failed because the official token counter did not have its optional torch and torchvision dependencies. Recovery rebuilt the prompt row from the preserved structured output and linked state, then ran the unchanged official reader once. The query was not rerun. This question uses deterministic scoring, so no evaluator call was needed.

### b82d0dd6: did not recover

- Gold: the incident-assignment workflow does not open a separate category-to-priority lookup page before editing incidents.
- Replicate-2 answer: Knowledge Search / Knowledge Portal — the Company Protocols article.
- Both aligned attempts anchored on a Search Knowledge link in the task form. Replicate 2 stated that the run did not display the lookup table, but still inferred that the portal was the required first page. The correct response rejects that premise.
- Replicate 2 used 2.30 times attempt-1 input and cost 1.38 times as much for the query. It used 1.32 times native input and cost 1.14 times as much as native.

This repeated failure is consistent with a stable retrieval or interpretation weakness for this question, but two aligned attempts are not enough to prove that the prompt caused it.

## New spend

| Question | Query | Reader API-equivalent | Reader provider-reported | Evaluator | Total API-equivalent |
| --- | ---: | ---: | ---: | ---: | ---: |
| f61a096f | 0.06178646 USD | 0.00056630 USD | 0.00056630 USD | 0 USD | 0.06235276 USD |
| b82d0dd6 | 0.04370224 USD | 0.00142120 USD | 0.00142120 USD | 0.00225575 USD | 0.04737919 USD |
| Total | 0.10548870 USD | 0.00198750 USD | 0.00198750 USD | 0.00225575 USD | 0.10973195 USD |

The two replicate queries cost 1.99 times the two aligned attempt-1 queries and 1.63 times the existing native queries. This large spread also shows that cost and token use vary materially across runs.

## Validation

- Exactly two new ThinHarness query attempts completed: one per selected question. There were 36 query response receipts, no native cells, no whole-cell reruns, no HTTP retries, no output retries, and no tool retries.
- The model, effort, prompt bytes, output contract, tools, corpus, task references, official reader, evaluator policy, ripgrep probe, source revision, source hashes, and API routes match the frozen diagnostic.
- Validation rechecked 857 cell artifacts and 461,728,266 bytes. A supplemental pass rehashed all 868 raw-manifest files and 461,886,185 bytes.
- Secret scans found no persisted injected key. The durable terminal log also had zero exact secret matches.
