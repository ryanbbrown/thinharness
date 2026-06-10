# Agent Instructions

## Branching
- Make all changes directly on `main` unless the user instructs otherwise.

## Validation
- Run `uv run pyright` after Python changes so type checking passes.
- Keep running the relevant pytest and ruff checks for the files you touched.

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
