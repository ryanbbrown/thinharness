# E2E Journeys

These scripts run real provider calls against temporary workspaces. They are intentionally not wired into pytest or CI.

Run one script with environment from `.env`:

```bash
uv run --env-file .env python e2e/workspace_tools_journey.py
```

Each script skips when `CI` is set or when the required provider key is missing. Model defaults can be overridden with the per-script `E2E_*_MODEL` environment variable.

Current journeys:

- `workspace_tools_journey.py`: filesystem tools plus `jsonl_search`.
- `skills_journey.py`: skill discovery, `skill_read`, and `skill_run`.
- `control_plane_journey.py`: hooks, sequential execution, and retry-limit behavior.
- `structured_output_journey.py`: Pydantic structured output after tool use.
- `mcp_journey.py`: local stdio MCP tool discovery and execution.
