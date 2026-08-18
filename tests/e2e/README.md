# E2E Journeys

Most scripts run real provider calls against temporary workspaces. The deterministic MCP journey uses a local scripted model, needs the `mcp` extra, and needs no provider credentials. Journeys are intentionally not wired into pytest or CI.

Run one script with environment from `.env`:

```bash
uv run --env-file .env python tests/e2e/workspace_tools_journey.py
```

Credential-based scripts skip when `CI` is set or when the required provider key is missing. Their model defaults can be overridden with the per-script `E2E_*_MODEL` environment variable.

Current journeys:

- `workspace_tools_journey.py`: filesystem tools plus `jsonl_search`.
- `skills_journey.py`: skill discovery, `skill_read`, and `skill_run`.
- `control_plane_journey.py`: hooks, sequential execution, and retry-limit behavior.
- `structured_output_journey.py`: Pydantic structured output after tool use.
- `mcp_journey.py`: deterministic local stdio MCP tool discovery, execution, and cleanup. It reports a skip when the `mcp` extra is not installed and does not use provider credentials.
- `parallel_llm_tool_journey.py`: direct `ParallelLlmTool` calls across all configured providers.
- `parallel_llm_agent_journey.py`: an agent run using both built-in `parallel_llm` and a renamed custom `ParallelLlmTool`.
- `prompt_caching_journey.py`: Anthropic prompt caching — asserts a multi-request run reports cached input tokens.
- `anthropic_modernization_journey.py`: Anthropic native structured output, max-token/effort payloads, and adaptive/default-on thinking resume.
