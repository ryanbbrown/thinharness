# thinharness

A minimal SDK-only filesystem harness for agents using provider-native tool APIs.

The harness provides:

- A small provider-agnostic tool loop.
- Responses-like model classes for OpenAI Responses, Anthropic Messages, and OpenRouter.
- Provider classes for auth, base URLs, client injection, and gateway/proxy customization.
- OpenTelemetry-compatible tracing for agent runs, model calls, and tool calls.
- Built-in filesystem tools: `read`, `write`, `edit`, `search`, `list`, and `glob`.
- Agent-oriented code search adapted from `pgr`, backed by `rg --json`.
- Contained path handling for structured filesystem tools.
- Frontmatter-based skill discovery with `skill_read` and `skill_run`.
- Custom JSON-schema tools with Python handlers.

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

## Development

```bash
uv run --extra dev pytest -q
```
