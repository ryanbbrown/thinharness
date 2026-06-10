# Web Research Report Agent

Run a live Exa-backed research report workflow with DeepSeek through OpenRouter:

```bash
uv run --env-file .env python examples/web_research_report/agent.py
```

The run writes the final report to `outputs/market_landscape_report.md`, source artifacts under `outputs/sources/`, source notes in `outputs/source_notes_batch.json`, and local traces under `outputs/traces/`.
