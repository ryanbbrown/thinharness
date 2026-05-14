# thinharness

A minimal SDK-only filesystem harness for agents using provider-native tool APIs.

The harness provides:

- A small provider-agnostic tool loop.
- Responses-like model classes for OpenAI Responses, Anthropic Messages, and OpenRouter.
- Provider classes for auth, base URLs, client injection, and gateway/proxy customization.
- OpenTelemetry-compatible tracing for agent runs, model calls, and tool calls.
- Built-in filesystem tools: `read`, `write`, `edit`, `search`, `list`, `glob`, and `jsonl_search`.
- Agent-oriented code search adapted from `pgr`, backed by `rg --json`.
- Contained path handling for structured filesystem tools.
- Frontmatter-based skill discovery with `skill_read` and `skill_run`.
- Custom Pydantic or JSON-schema tools with Python handlers.

## Usage

```python
from thinharness import Harness, HarnessConfig

harness = Harness(HarnessConfig(root=".", model="openai:gpt-5.2"))
result = harness.run("Read README.md and summarize it.")
print(result.text)
```

Model refs select the provider:

```python
HarnessConfig(model="openai:gpt-4.1-mini")
HarnessConfig(model="anthropic:claude-haiku-4-5-20251001")
HarnessConfig(model="openrouter:openai/gpt-4.1-mini")
```

Model refs must include a provider prefix. API keys are read from `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, and `OPENROUTER_API_KEY` unless you pass `api_key` or provide a custom model/provider/client.

Custom providers let you keep the model protocol while changing transport details:

```python
from thinharness import Harness, OpenAIProvider, OpenAIResponsesModel

provider = OpenAIProvider(base_url="https://company.internal/llm/v1", api_key="...")
model = OpenAIResponsesModel("gpt-5.2", provider=provider)
harness = Harness(model=model)
```

Useful harness limits live on `HarnessConfig`:

```python
from pathlib import Path

HarnessConfig(
    builtin_tools=["read", "search", "skill_read"],
    skills_dir=[Path(".agents/skills"), Path("vendor/skills")],
    selected_skills=["python"],
    max_read_chars=40_000,
    max_read_bytes=1_000_000,
    max_tool_chars=40_000,
    max_search_line_chars=180,
)
```

Files up to `max_read_bytes` use the fast whole-file read path, then apply `offset` and `limit` in memory. Larger files must be read with an explicit bounded range, which is streamed so skipped content is not accumulated in memory.

`builtin_tools` selects harness-provided tools by lowercase name. `None` exposes all built-ins, `[]` exposes none, and a list such as `["read", "search"]` exposes only those tools. Skill access is selected the same way with `skill_read` and `skill_run`.

Built-in tools return JSON strings with `ok`, `content`, and `metadata` fields. Failed tools return `ok: false` instead of raising through the model loop when the failure is part of normal tool execution, such as invalid arguments, missing files, ripgrep errors, or timeouts.

Custom tools can use a Pydantic args model as the source of truth for validation and provider JSON Schema:

```python
from pydantic import BaseModel, Field
from thinharness import Harness, HarnessConfig, ToolSpec

class EchoArgs(BaseModel):
    value: str
    count: int = Field(default=1, ge=1)

def echo(args: EchoArgs) -> dict[str, str]:
    return {"echo": args.value * args.count}

harness = Harness(
    HarnessConfig(builtin_tools=["read", "search"]),
    tools=[ToolSpec("echo", "Echo typed input.", EchoArgs, echo)],
)
```

## Search

`search` wraps `rg --json`, groups matches by file, ranks likely definitions and source files first, and formats the result so an agent can decide what to read next.

```python
{
    "query": "CheckpointStore",
    "path_glob": "**/*.py",
    "file_type": "py",
    "max_files": 10,
    "max_matches_per_file": 3,
}
```

The implementation is a Python port of the core `pgr` search behavior. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for attribution.

`max_search_line_chars` only affects `search` match previews. `jsonl_search` uses `fields` for field-level projection and truncation, plus `max_tool_chars` for the total output size.

Search ranking and scope are configurable. By default, pgr-style ranking shows likely definitions first, then source files before tests before low-priority directories such as `vendor`, `examples`, `fixtures`, and `node_modules`. Exclude globs are passed to ripgrep, so excluded paths are not searched at all.

```python
HarnessConfig(
    search_exclude_globs=["vendor/**", "node_modules/**"],
    search_low_priority_dirs=["examples", "fixtures", "third_party"],
    search_test_dirs=["tests", "specs"],
)
```

## Skills

Skills are Markdown files with intentionally small frontmatter. Use flat `key: value` metadata such as `name` and `description`; values may be plain strings, quoted strings, booleans, or JSON literals. Nested YAML is not supported.

`skills_dir` may be one path or a list of paths. The directories are treated as one skill namespace, so duplicate skill names raise an error. Use `selected_skills` to expose only specific skill names. If `skills_dir` or `selected_skills` is configured and matching skills are available, `builtin_tools` must explicitly include `skill_read` or `skill_run`.

`skill_run` executes local scripts from the selected skill directory without sandboxing. Only expose it when those scripts are trusted for the workspace.

## Tracing

Pass an OpenTelemetry tracer to emit spans for the harness run, model calls, and tool executions:

```python
from thinharness import Harness, HarnessConfig, TracingOptions, create_otlp_tracing

otlp = create_otlp_tracing(service_name="my-agent")
harness = Harness(HarnessConfig(root=".", tracing=TracingOptions(tracer=otlp.tracer)))
result = harness.run("Read README.md and summarize it.")
otlp.force_flush()
```

Install the optional tracing dependencies with `pip install "thinharness[tracing]"`.

## Runtime model

Provider model instances keep conversation state while a run is in progress. Treat a `Harness` instance as single-run/single-thread unless you provide a stateless custom model.

## Development

```bash
uv run --extra dev pytest -q
```
