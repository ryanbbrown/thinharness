# Web Research Report Agent

Run a live Exa-backed research report workflow with GPT-5.5 through OpenAI:

```bash
uv run --env-file .env python examples/web_research_report/agent.py
```

The run writes the final report to `outputs/market_landscape_report.md`, source artifacts under `outputs/sources/`, and source notes in `outputs/source_notes_batch.json`. Local traces use ThinHarness's default project-scoped trace directory under `~/.thinharness/traces/`.
