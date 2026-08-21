<p align="center">
  <img src="assets/ThinHarness.svg" alt="ThinHarness" width="360">
</p>

<p align="center">
  <br/>
  A compact, SDK-only agent harness:
  <br/>
  maximum performance, minimum framework code
  <br/><br/>
</p>

<div align="center">

[![CI](https://img.shields.io/github/actions/workflow/status/ryanbbrown/thinharness/ci.yml?branch=main&label=CI)](https://github.com/ryanbbrown/thinharness/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/ryanbbrown/thinharness/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/thinharness.svg)](https://pypi.org/project/thinharness/)

</div>

## Why keep it small

Minimal agent harnesses are becoming more common. [Pi](https://github.com/earendil-works/pi) has shown how far a focused, token-efficient harness can go, and [Deep Agents](https://github.com/langchain-ai/deepagents) is close behind for cost/performance on [Composio's benchmark](https://composio.dev/content/best-agent-harness-deepseek-v4-flash). Vercel recently released [fx](https://github.com/vercel-labs/fx), which is deliberately tiny and embeddable. ThinHarness takes the same direction to its limit: how little framework code can you keep without giving up capability or performance?

Larger harnesses are larger for good reasons. They support more providers, storage systems, sandboxes, durable jobs, deployment targets, integrations, and interactive interfaces. ThinHarness makes fewer promises. The core owns the model and tool loop and the behavior that must stay consistent across every run. Optional capabilities live in explicit plugins.

For a side project, that means less framework code to learn, configure, debug, update, and carry in a fork. It also means the project can become complete. ThinHarness does not need to keep growing into an agent platform after it has the features its agents need.

## Benchmarking

Benchmarking is in progress.

ThinHarness powers [Retrodict](https://github.com/ryanbbrown/Retrodict), a specialized ARC-AGI-3 agent that leads the reported cost-performance frontier for public ARC-AGI-3 harnesses. Its official competition-mode scorecard reports 99.86% mean RHAE across all 25 public games, with all 183 levels solved. See the [ARC-AGI Community Leaderboard submission](https://github.com/arcprize/ARC-AGI-Community-Leaderboard/tree/main/submissions/retrodict). This is application evidence, not an isolated comparison of generic harness loops.

## Features

The core contains the run behavior every configuration shares. Plugins add complete capabilities without making them implicit dependencies.

### Core

- **Plugin composition:** explicit Python plugins can contribute tools, instructions, hooks, and connected resources.
- **Provider adapters:** built-in OpenAI, Anthropic, and OpenRouter adapters, plus public model and session protocols for implementing another provider.
- **Custom typed tools:** define sync or async `ToolSpec` handlers with Pydantic argument models, normalized `ToolResult` envelopes, parallel and approval flags, and per-tool retry settings.
- **Structured output:** Pydantic-validated results with native, tool, prompted, and text modes.
- **Text and images:** ordered text and image input in prompts and tool results, with JPEG, PNG, GIF, and WebP support across the built-in providers.
- **Resume:** self-contained transcript state can replay text and images across built-in providers and models, preserve native reasoning on same-provider resume, and degrade it to text across providers.
- **Parallel tool calls:** same-turn tool batches run concurrently when every called tool is parallel-safe.
- **Human approvals:** approval-required tools pause before side effects and return the pending call plus the state needed to continue after an approve or reject decision.
- **Tool retries:** tools raise `ModelRetry` to send structured feedback to the model and retry within a per-tool budget.
- **Limits and notices:** request, tool-call, output-retry, and tool-retry budgets bound each run; near-limit guidance can warn the model before a budget is exhausted.
- **Hooks and events:** lifecycle hooks can inspect or intercept prompts, tool calls, subagents, limits, and run boundaries; async streaming emits coarse run, model, tool, retry, limit, and subagent events.
- **Tracing:** local plaintext JSONL traces plus OpenTelemetry-compatible spans for runs, provider calls, tools, and subagents.

### Plugins

- **Filesystem:** root-scoped `read`, `write`, batched exact-replacement `edit`, `search`, `list`, and `glob`, plus opt-in bounded `read_image`.
- **JSONL search:** an opt-in `FilesystemPlugin` tool for structured search over line-delimited data, with ripgrep prefiltering, field projection, equality, contains, regex, and range filters, plus field-level snippets from large multiline values.
- **Bash:** one-shot non-interactive commands with a contained working directory, filtered environment, bounded output, timeouts, cancellation cleanup, and optional approval. It is not a sandbox.
- **MCP:** `MCPPlugin` support built on the FastMCP client, including in-process servers, lazy tool discovery, and collision checks.
- **Subagents:** a default child, named child configurations, explicit safe-plugin inheritance, local child hooks, and no recursive delegation.
- **Parallel LLM:** batches of independent one-shot prompts, with an optional separate model, structured results, and explicit read and write paths.
- **Skills:** ordered `skill_read` and `skill_run` tools, with Python, shell, JavaScript, and Go script runners.

## Install

```bash
uv add thinharness     # or pip install thinharness
```

Requires Python 3.11+.

## Quick start

```python
import asyncio
from thinharness import FilesystemPlugin, Harness, HarnessConfig

async def main():
    async with Harness(
        HarnessConfig(root=".", model="openai:gpt-5.5"),
        plugins=[FilesystemPlugin(tools=["read"])],
    ) as harness:
        result = await harness.run("Read README.md and summarize it.")
        print(result.text)

asyncio.run(main())
```

There's a synchronous wrapper too: `Harness(...).run_sync(...)`.

## Size

These are source lines of code in the smallest first-party opinionated configuration for each harness. The count includes required first-party runtime packages and excludes tests, documentation, examples, hosted services, and unrelated optional interfaces where the project structure makes that separation possible.

| Harness | Source LOC | What is counted |
| --- | --- | --- |
| **ThinHarness** | **10,230** | 6,073 core + 4,157 bundled plugins and tools |
| [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness) | 72,737 | The base and headless profiles + their first-party package dependency closure |
| [Pydantic AI Coder](https://github.com/pydantic/pydantic-ai-harness) | 85,633 | Pydantic AI runtime + the Coder capability and everything it composes |
| [Pi](https://github.com/earendil-works/pi) | 95,276 | `pi-coding-agent` + its first-party workspace dependency closure |
| [Deep Agents](https://github.com/langchain-ai/deepagents) | 118,977 | Deep Agents + its required LangChain and LangGraph runtime |

LOC is not a quality or performance score. It measures how much framework code comes with the comparable agent configuration. Moving code from a core package into required plugins does not make that configuration smaller, so the table counts both.

## License

MIT. See [LICENSE](LICENSE).