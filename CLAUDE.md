# Agent Instructions

## Branching
- Make all changes directly on `main` unless the user instructs otherwise.

## Validation
- Run `uv run pyright` after Python changes so type checking passes.
- Keep running the relevant pytest and ruff checks for the files you touched.
- Export draw.io SVGs with `--svg-theme light` so rendered diagrams are always light mode, regardless of viewer color scheme.
- After pushing to `main`, check deployment status after about 30 seconds and confirm the deploy succeeded.

## Behavior Contracts
Update `docs/behavior.md` after the plan review cycle and before implementation for any nontrivial change that affects durable product behavior. Follow the structure in that file. Usually this means adding a new section, but review existing behavior sections and update affected requirements when the planned implementation changes or clarifies them. Edit only affected sections; avoid wording churn.

Do not update `docs/behavior.md` for pure refactors, internal cleanup, renames, file moves, dependency updates, or implementation-only API changes unless they change the product behavior described there. If the intended behavior cannot be stated clearly, stop and clarify before implementation.

## Project Learnings
Agents should capture durable project learnings when they discover a non-obvious pattern, pitfall, user preference, architecture constraint, tool behavior, or workflow fix that would save future agents time.

Do not add every lesson directly to this file. Prefer appending a structured learning record to `.agent/learnings.jsonl`. The user will periodically review those records and promote important ones into this file.

Use this JSONL shape:

```json
{"skill":"review","type":"pitfall","key":"short-stable-key","insight":"Actionable rule future agents should follow.","confidence":8,"source":"observed","files":["path/to/relevant-file"]}
```

Types: `pattern`, `pitfall`, `preference`, `architecture`, `tool`, `operational`, `investigation`.

Sources: `observed`, `user-stated`, `inferred`, `cross-model`.

Confidence: 1-10. Use 8-9 for verified observations, 4-5 for uncertain inference, and 10 for explicit user-stated preferences.

Only log learnings that are reusable, specific, and likely to prevent a future mistake. Do not log obvious facts, one-off transient errors, or broad preferences inferred without evidence.
