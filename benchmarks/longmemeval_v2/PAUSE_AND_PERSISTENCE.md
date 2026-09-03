# LongMemEval V2 pause and persistence

LongMemEval work is paused. Do not run more benchmark cells or continue the cost-telemetry analysis without new approval.

## Tracked and reproducible

The canonical repository is `/Users/ryanbrown/code/thinharness`, branch `main`, remote `https://github.com/ryanbbrown/thinharness.git`.

Reusable benchmark code and frozen configuration are under `benchmarks/longmemeval_v2/`. This includes selection manifests, prompt diffs, API and scoring settings, the official-harness instrumentation patch, preparation and run scripts, recovery scripts, tests, and validation code.

Compact reports, receipts, comparisons, validations, and raw-artifact hash manifests are under these tracked directories:

- `benchmarks/longmemeval_v2/evidence/luna-xhigh-wave1/`
- `benchmarks/longmemeval_v2/evidence/luna-xhigh-rg-diagnostic-replicates-v1/`
- `benchmarks/longmemeval_v2/evidence/luna-xhigh-native-prompt-alignment-diagnostic-v1/`
- `benchmarks/longmemeval_v2/evidence/luna-xhigh-native-prompt-alignment-variance-check-v1/`
- `benchmarks/longmemeval_v2/evidence/luna-xhigh-clean-pair-v1/`

The generated official runtime lock is archived as `benchmarks/longmemeval_v2/official-runtime.uv.lock`. Its SHA-256 is `513c0c4953f84cba9a12bff8f29a41c78de82ced946ccd79b2cf922b9b1c64dc`, which matches `clean_pair_config.json`. The official source checkout itself is reproducible from revision `2cc8c540bdb87fe6761629b585e727e1c4704520` plus `official-direct-api-instrumentation.patch`.

## Durable local raw artifacts

Raw traces and large generated corpora remain local because they are about 11 GB and are excluded by `.gitignore`. They are under `/Users/ryanbrown/code/thinharness/.benchmark-runs/`:

- `longmemeval-v2-luna-xhigh-wave1/` — 4.6 GB
- `longmemeval-v2-luna-xhigh-clean-pair-v1/` — 3.1 GB
- `longmemeval-v2-luna-xhigh-native-prompt-alignment-diagnostic-v1/` — 1.1 GB
- `longmemeval-v2-luna-xhigh-rg-diagnostic-replicates-v1/` — 990 MB
- `longmemeval-v2-luna-xhigh-native-prompt-alignment-variance-check-v1/` — 443 MB
- `longmemeval-v2-frozen-official-2cc8c540/` and two small setup-failure roots

Tracked raw-artifact manifests provide hashes for the runs used in the reports.

All reusable LongMemEval files that were previously only in BB thread storage were copied to `/Users/ryanbrown/code/thinharness/.benchmark-runs/longmemeval-v2-bb-archive/`. This 536 MB archive contains 823 source files from threads `thr_eq5ch6tgjd`, `thr_mjxejnc8k9`, and `thr_dgzrszp3tc`, including the readiness receipt, smoke and paired-pilot data, route tests, analysis notes, and terminal logs. `MANIFEST.json` records every copied file's path, size, and SHA-256. A secret scan found zero injected API-key values. The original BB copies can disappear without removing the local archive.

The raw runs and BB archive have no remote backup in this repository. They remain local-only. The compact evidence needed to interpret and reproduce the work is tracked.

## Temporary checkout

`/Users/ryanbrown/.cache/thinharness-benchmarks/LongMemEval-V2-clean-pair/` is a disposable official source checkout and Python environment. No unique configuration depends on it: revision, patch, runtime lock, source hashes, and run metadata are persisted above.
