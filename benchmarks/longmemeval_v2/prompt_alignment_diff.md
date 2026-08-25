# LongMemEval native prompt alignment

This artifact defines the prompt-only difference between current native LongMemEval V2 and the ThinHarness diagnostic.

Native sources:

- Official revision: 2cc8c540bdb87fe6761629b585e727e1c4704520
- Native query prompt SHA-256: a0a70c0fb9df19ebe6861812bfb3bed647e403299aaeee870e2e198773ee7adb
- Native INSTRUCTION.md SHA-256: 6add63451d46ab786296963aef4cc36fcc63a83d9eba9b330d1ce6fd7b139a6c

The runtime starts from those exact native sources. It fails if any source fragment below is absent or repeated. It applies only these ordered replacements.

## Query prompt replacements

1. Replace the instruction to read INSTRUCTION.md and question.json with an instruction to follow the inline instruction and question. ThinHarness does not create those sandbox files.
2. Replace the trajectories/ corpus root with corpus/. Existing ThinHarness corpus files remain unchanged.
3. Replace the question.json image reference with the attached-image reference. Existing ThinHarness image delivery remains unchanged.
4. Replace writing memory_module_output.json with configured structured output. Existing ThinHarness output mode remains unchanged.
5. Replace the scripts/inspect_trajectory.py hint with bounded read and search for summaries and per-trajectory jsonl_search for exact verification. Existing ThinHarness tools remain unchanged.

## INSTRUCTION.md replacements

1. Replace the question.json and trajectories/ locations with the inline question and corpus/ root.
2. Replace writing memory_module_output.json with configured structured output.
3. Replace the fixed trajectories/ root description with corpus/.
4. Replace per-trajectory paths with corpus/trajectories/ and state that selected screenshots go to the reader. ThinHarness query tools do not expose trajectory screenshots.
5. Replace native state field names text and thoughts with existing ThinHarness fields accessibility_tree and thought, and document existing action_annotated.
6. Replace both summary paths with their existing corpus/ paths.
7. Replace the question.json image wording with attached-image wording.
8. Replace the full-summary path with corpus/TRAJECTORY_SUMMARY_FULL.md.
9. Replace shell helper examples with equivalent bounded read, summary-scoped search, and per-trajectory jsonl_search examples. Preserve shortlist-first retrieval, exact verification, the warning against broad raw searches, and the stop rules.
10. Replace the scratch-file permission with bounded tool-result guidance because the fixed tool set has no write tool.

All other native wording and ordering remain unchanged, including the role, task overview, output field names, evidence limits, triage stages, shortlist-first policy, exact-match rules, contradiction and uncertainty rules, small evidence package, and final reminder.

## Remaining non-equivalences held fixed

- ThinHarness system and filesystem-plugin instructions remain active. Changing them would exceed the query-prompt-only scope.
- The instruction and question are inline instead of being returned by a shell tool call. The fixed ThinHarness surface has no shell or native sandbox files.
- ThinHarness uses read and jsonl_search instead of shell and inspect_trajectory.py. Tool schemas and implementations are fixed.
- ThinHarness uses native structured output instead of a writable output file. Output behavior is fixed.
- Corpus paths and normalized field names match the existing ThinHarness corpus. Corpus and duplicated data exposure are fixed.
- The question image remains attached directly. Image delivery is fixed.
- Responses history, request and tool limits, parallel behavior, reader expansion, model settings, and all other variables remain unchanged.
