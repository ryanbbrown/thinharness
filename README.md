# thinharness

A minimal SDK-only filesystem harness for agents using provider-native tool APIs.

The harness provides:

- A small provider-agnostic tool loop.
- Responses-like model classes for OpenAI Responses, Anthropic Messages, and OpenRouter.
- Provider classes for auth, base URLs, and gateway/proxy customization.
- OpenTelemetry-compatible tracing for agent runs, model calls, and tool calls.
- Built-in filesystem tools: `read`, `write`, `edit`, `search`, `list`, and `glob`.
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
    max_model_requests=64,
    max_tool_calls=200,
    max_read_chars=40_000,
    max_read_bytes=1_000_000,
    max_tool_chars=40_000,
    max_search_line_chars=180,
    read_paths=["src", "tests"],
    write_paths=["src"],
)
```

`max_model_requests` counts provider calls. `max_tool_calls` counts model-requested tool calls separately, including calls blocked by hooks. A response with three tool calls uses one model request and three tool calls; a batch that would exceed `max_tool_calls` is rejected before any tool in that batch runs.

Files up to `max_read_bytes` use the fast whole-file read path, then apply `offset` and `limit` in memory. Larger files must be read with an explicit bounded range, which is streamed so skipped content is not accumulated in memory.

`builtin_tools` selects harness-provided tools by lowercase name. `None` exposes the default filesystem tools (`read`, `write`, `edit`, `search`, `list`, and `glob`), `[]` exposes none, and a list such as `["read", "search"]` exposes only those tools. Specialized tools are opt-in: add `jsonl_search` for JSONL datasets, `subagent` for delegation, and `skill_read` or `skill_run` for configured skills.

Tool outputs sent back to providers are JSON strings with `ok`, `content`, and `metadata` fields. Failed tools return `ok: false` instead of raising through the model loop when the failure is part of normal tool execution, such as invalid arguments, handler exceptions, missing files, ripgrep errors, or timeouts.

`read_paths` and `write_paths` narrow filesystem access under `root`. Relative entries are resolved from the workspace root; absolute entries must still be inside the workspace root. When omitted, both default to the full workspace root. `read_paths` applies to `read`, `list`, `glob`, `search`, and `jsonl_search`; `write_paths` applies to `write` and `edit`. Glob-style selectors are validated separately, so absolute patterns and `..` path components are rejected.

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

Run results include final text, raw model responses, tool records, and usage counters:

```python
result = harness.run("Inspect the workspace.")
print(result.usage.model_requests)
print(result.usage.tool_calls)
print(result.usage.cancelled_tool_calls)
print(result.tool_call_records)
```

`usage.tool_calls` is the cheap count of requested tool calls, and it should equal `len(result.tool_call_records)` for completed runs. `tool_call_records` is the ordered audit list of each tool call and provider-facing output; hook-blocked records include `cancelled: True`, while the provider-facing output also carries `metadata.error_type == "ToolCallCancelled"`.

## Hooks

Hooks are synchronous runtime constructor arguments. They are not part of `HarnessConfig` or `SubAgentConfig` because handlers are live Python callables. If you pass a prebuilt `HookRegistry`, its `strict_hooks` setting is preserved; if you pass a hook list, `HarnessConfig.strict_hooks` controls whether handler exceptions are logged and swallowed or re-raised.

```python
from thinharness import BeforeToolCallContext, Harness, Hook

def log_reads(ctx: BeforeToolCallContext) -> None:
    print(ctx.call_id, ctx.tool_name, ctx.arguments)

harness = Harness(
    hooks=[Hook("before_tool_call", log_reads, tools=["read"])],
)
```

Filters match final registered tool and subagent names exactly and case-sensitively. `tools=None` and `agents=None` mean all matching event targets. Hook handlers run in registration order; tool hooks run inside the tool execution span and may run concurrently for parallel tool batches.

`user_prompt_submit` hooks can append provider-neutral context to the first model request through `ctx.additional_context` or block the run with `ctx.cancelled = True`. `before_tool_call` and `before_subagent_run` are also cancellable. `after_tool_call` may replace the output sent back to the model by assigning `ctx.output`.

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

`max_search_line_chars` only affects `search` match previews. `jsonl_search` is an opt-in built-in for JSONL datasets; it uses `fields` for field-level projection and truncation, plus `max_tool_chars` for the total output size.

```python
HarnessConfig(builtin_tools=["read", "search", "jsonl_search"])
```

Search ranking and scope are configurable. By default, pgr-style ranking shows likely definitions first, then source files before tests before low-priority directories such as `vendor`, `examples`, `fixtures`, and `node_modules`. Exclude globs are passed to ripgrep, so excluded paths are not searched at all.

```python
HarnessConfig(
    search_exclude_globs=["vendor/**", "node_modules/**"],
    search_low_priority_dirs=["examples", "fixtures", "third_party"],
    search_test_dirs=["tests", "specs"],
)
```

## Subagents

`subagent` is an opt-in built-in tool. Include it in `builtin_tools` to allow delegation. It can route to the framework default subagent by omitting `agent`; the default subagent runs in fresh context and inherits the parent's current tool set except `subagent`.

```python
from thinharness import Harness, HarnessConfig

harness = Harness(HarnessConfig(root=".", builtin_tools=["read", "search", "glob", "subagent"]))
result = harness.run("Use a subagent to inspect README.md, then summarize the result.")
```

Specialized named subagents use fixed tool surfaces:

```python
from thinharness import Harness, HarnessConfig, SubAgentConfig

harness = Harness(HarnessConfig(
    root=".",
    subagents=[
        SubAgentConfig(
            name="research",
            description="Searches and reads code without editing.",
            system_prompt="Investigate and report findings. Do not edit files.",
            builtin_tools=["read", "search", "glob"],
        )
    ],
))
```

When explicit tool selection is used, include `subagent` to allow delegation:

```python
HarnessConfig(
    builtin_tools=["read", "write", "edit", "subagent"],
    subagents=[
        SubAgentConfig(name="research", description="Research helper.", builtin_tools=["read", "search", "glob"]),
    ],
)
```

`builtin_tools=None` does not expose `subagent`; it must be selected explicitly. `builtin_tools=[]` disables all built-ins. A named subagent can also set `inherit_parent_tools=True` to inherit the parent's effective tool universe minus `subagent`; otherwise it must define `builtin_tools` or `tools`.

Parent hooks observe the parent run and the parent-side `subagent` tool boundary. They do not automatically run inside child harnesses. Configure child hooks explicitly with `subagent_hooks`:

```python
from thinharness import Harness, HarnessConfig, Hook, SubAgentConfig

def log_subagent(ctx) -> None:
    print(ctx.agent, ctx.task)

def log_child_run(ctx) -> None:
    print("child", ctx.prompt)

harness = Harness(
    HarnessConfig(
        subagents=[
            SubAgentConfig(
                name="research",
                description="Searches and reads code.",
                builtin_tools=["read", "search", "glob"],
            ),
        ],
    ),
    hooks=[Hook("before_subagent_run", log_subagent, agents=["research"])],
    subagent_hooks={"research": [Hook("run_start", log_child_run)]},
)
```

Pass the same hook in both places if you want observability at the parent boundary and inside the child run:

```python
logging_hook = Hook("run_end", lambda ctx: print(ctx.stop_reason))
harness = Harness(
    hooks=[logging_hook],
    subagent_hooks={"research": [logging_hook]},
)
```

Parent and child budgets are local to each harness run. If a child inherits `max_model_requests=64`, each child invocation receives its own fresh budget, so subagent-heavy workflows can multiply total provider calls.

## Skills

Skills are Markdown files with intentionally small frontmatter. Use flat `key: value` metadata such as `name` and `description`; values may be plain strings, quoted strings, booleans, or JSON literals. Nested YAML is not supported.

`skills_dir` may be one path or a list of paths. The directories are treated as one skill namespace, so duplicate skill names raise an error. Use `selected_skills` to expose only specific skill names. Skills are loaded only when `skills_dir` is set; there is no automatic workspace skill discovery. If `builtin_tools` is an explicit list and matching skills are available, include `skill_read` or `skill_run`.

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

## Parallel tool execution

When a model emits multiple tool calls in one response, the harness runs them in parallel by default. Mutating built-ins (`write`, `edit`, `skill_run`) are marked `sequential=True` and force the entire batch to run serially in model order. Read-only built-ins (`read`, `search`, `list`, `glob`, `jsonl_search`, `skill_read`) execute concurrently in a thread pool.

```python
HarnessConfig(tool_execution="auto")        # default: same-response calls may run in parallel
HarnessConfig(tool_execution="sequential")  # escape hatch: always serial, in model order
```

Custom tools default to `sequential=False`. Set `sequential=True` on a `ToolSpec` for tools that mutate shared state or are not thread-safe; including one such tool in a batch makes the whole batch sequential. The harness always waits for every result in the current batch before sending the next provider continuation, and returned outputs preserve the original model call order regardless of completion order.

## Runtime model

Provider model instances keep conversation state while a run is in progress. Treat a `Harness` instance as single-run/single-thread unless you provide a stateless custom model.

## Breaking changes

- `max_turns` was replaced by `max_model_requests` and `max_tool_calls`.
- `HarnessResult.tool_calls` was renamed to `tool_call_records`.
- `HarnessResult.usage` now contains model request, tool call, and cancelled tool call counters.

If you previously passed `max_turns=N`, the closest provider-call budget is `max_model_requests=N + 1`, because the old loop allowed one initial request plus up to `N` continuations.

## Development

```bash
uv run --extra dev pytest -q
```
