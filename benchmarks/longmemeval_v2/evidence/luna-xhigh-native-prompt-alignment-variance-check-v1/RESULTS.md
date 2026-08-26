# LongMemEval native-prompt alignment variance check

Two additional ThinHarness stochastic replicates. No native cells were run.

| Question | Evidence | Score | Requests | Tools | Input | Output | Reasoning | Search failures | Latency | Query cost | Total cost |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| f61a096f | native | 1 | 14 | 13 | 531,836 | 2,860 | 1,159 | 1 | 45.66s | 0.02635228 USD | 0.02687643 USD |
| f61a096f | aligned attempt 1 | 0 | 7 | 14 | 180,877 | 4,293 | 3,232 | 0 | 50.40s | 0.02125538 USD | 0.02231018 USD |
| f61a096f | aligned replicate 2 | 1 | 19 | 21 | 1,346,806 | 4,875 | 3,254 | 0 | 72.79s | 0.06178646 USD | 0.06235276 USD |
| b82d0dd6 | native | 1 | 14 | 13 | 720,375 | 5,963 | 4,315 | 1 | 72.63s | 0.03832578 USD | 0.04396498 USD |
| b82d0dd6 | aligned attempt 1 | 0 | 9 | 14 | 412,488 | 5,346 | 4,099 | 0 | 70.09s | 0.03169098 USD | 0.03659483 USD |
| b82d0dd6 | aligned replicate 2 | 0 | 17 | 27 | 950,366 | 6,765 | 4,256 | 0 | 88.89s | 0.04370224 USD | 0.04737919 USD |

Recovered: 1/2.
New query API-equivalent cost: 0.10548870 USD.
New reader API-equivalent cost: 0.00198750 USD.
New evaluator API-equivalent cost: 0.00225575 USD.
New total API-equivalent cost: 0.10973195 USD.

These two outcomes estimate stochastic variance only. They do not replace prior cells or establish a stable regression rate.
