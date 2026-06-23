# ThinHarness User Guide

ThinHarness is a small SDK for purpose-built agent loops. The host application chooses the model, tools, limits, context, output contract, and lifecycle hooks. The model gets enough room to plan and use tools, but the run stays bounded by configuration.

For code ownership and the run-loop mental model, see `docs/site/explainer.html`. This file is the user-facing API guide.

## Install

```bash
uv add thinharness
```

Optional extras:

```bash
uv add "thinharness[mcp]"
uv add "thinharness[tracing]"
```

ThinHarness requires Python 3.11+.

## First Run

```python
import asyncio

from thinharness import Harness, HarnessConfig


async def main() -> None:
    async with Harness(HarnessConfig(root=".", model="openai:gpt-5.5")) as harness:
        result = await harness.run("Read README.md and summarize it.")
        print(result.text)


asyncio.run(main())
```

`Harness.run()` is the primary API. `Harness.run_sync(...)` is a convenience wrapper for synchronous callers outside an existing event loop.

A `Harness` instance is reusable, but one instance cannot run two prompts concurrently. Use separate harness instances for parallel branches.

## Streaming Progress

Use `Harness.stream(...)` when a host app needs live workflow progress and the final `HarnessResult`:

```python
from thinharness import RunCompletedEvent


async for event in harness.stream("Process these records."):
    if event.kind == "model_message":
        print(event.text)
    if isinstance(event, RunCompletedEvent):
        result = event.result
```

Streaming is coarse turn/tool/run streaming, not token-delta streaming. Provider calls still return complete model turns. Events cover run start/end, provider request starts, complete model messages, tool call start/completion, background task start/completion, structured-output and tool retries, limit warnings, and child subagent runs.

Stream events are high-level workflow events intended for app consumption:

- `RunStartedEvent.prompt` includes the submitted prompt.
- `ToolCallStartedEvent.arguments` includes the model-requested tool arguments.
- `ToolCallCompletedEvent.output` and `BackgroundTaskCompletedEvent.output` include model-visible tool output.
- Raw provider response JSON is not part of stream events; use `HarnessResult.responses` for raw provider responses after completion.
- `ModelMessageEvent.text` includes assistant text from the completed provider turn.
- Child subagent events are flattened by default; set `include_subagents=False` to keep only the parent `subagent` tool lifecycle.

Workflow UIs should group nested work with `kind`, `run_id`, `parent_run_id`, and `parent_tool_call_id`. The final successful `RunCompletedEvent` carries the same full `HarnessResult` returned by `run()`, including `responses`, `tool_call_records`, and `resume_state`; treat that terminal result as completion data rather than a progress update.

`stream()` starts the run eagerly when called and uses an unbounded in-process event queue. If you might stop before the terminal event, close the stream so the run task is cancelled and cleaned up:

```python
async with harness.stream("Process records.") as events:
    async for event in events:
        if should_stop(event):
            break
```

## Configuration Shape

Most behavior is configured with `HarnessConfig`:

```python
config = HarnessConfig(
    root=".",
    model="openai:gpt-5.5",
    system_prompt="You are a focused research agent.",
    max_model_requests=32,
    max_tool_calls=80,
    read_paths=["inputs", "docs"],
    write_paths=["outputs"],
)
```

Important groups:

- `root`, `read_paths`, `write_paths`, and `output_dir` define filesystem scope.
- `model`, `api_key`, `base_url`, `temperature`, `extra_body`, and `request_timeout` define provider settings.
- `builtin_tools`, `tools`, `subagents`, `mcp_servers`, and `skills_dir` define the model-callable surface.
- `max_model_requests`, `max_tool_calls`, `output_retries`, and `tool_retries` bound the run.
- `output_type` and `output_mode` define structured output.
- `tracing`, `local_tracing`, and `local_trace_dir` define observability.

## Built-In Filesystem Tools

When `builtin_tools` is omitted, the model gets these filesystem tools:

- `read`: read bounded UTF-8 file ranges with line numbers.
- `write`: create, overwrite, or append UTF-8 files. This tool is sequential.
- `edit`: apply one or more exact text replacements to UTF-8 files; edits apply in order. This tool is sequential.
- `search`: ripgrep-backed grouped path/line search.
- `list`: list files or directories.
- `glob`: find files by glob pattern.

When `builtin_tools` is provided, it is an explicit replacement list. Include every built-in tool the model should see.

`jsonl_search` is available as an opt-in built-in:

```python
harness = Harness(HarnessConfig(
    root=".",
    builtin_tools=["read", "search", "jsonl_search"],
))
```

Use `query` as a ripgrep row prefilter, `fields` to project only the values the model needs, and `where` for structured filters over jq-style field paths:

```python
result = await harness.run(
    "Use jsonl_search on support/events.jsonl. Find open tickets with priority p1 and return id, customer.name, and updated_at."
)
```

The model can call `jsonl_search` with arguments like:

```json
{
  "path": "support/events.jsonl",
  "query": "ticket",
  "where": [
    {"field": "status", "op": "eq", "value": "open"},
    {"field": "priority", "op": "eq", "value": "p1"}
  ],
  "fields": {"id": 0, "customer.name": 0, "updated_at": 0}
}
```

Range filters use `op` values `gt`, `gte`, `lt`, or `lte` with an explicit `type` of `number` or `date`. Number ranges match JSON numbers only; date ranges compare ISO-like date or datetime strings:

```json
{
  "path": "support/events.jsonl",
  "where": [
    {"field": "score", "op": "gte", "value": "0.8", "type": "number"},
    {"field": "created_at", "op": "gte", "value": "2026-06-01", "type": "date"}
  ],
  "fields": {"id": 0, "score": 0, "created_at": 0}
}
```

For large multiline string fields, `field_searches` returns matching internal lines without rendering the whole field. It runs after JSON parsing and `where` filtering, so it is best used with `fields` for the row summary and snippets for the bulky field:

```json
{
  "path": "states.jsonl",
  "where": [{"field": "state_index", "op": "eq", "value": "11"}],
  "fields": {"state_index": 0, "url": 0},
  "field_searches": [
    {
      "field": "accessibility_tree",
      "query": "Incident|-- None --|Edit personal filters",
      "regex": true,
      "context_lines": 1,
      "max_matches": 5,
      "max_line_chars": 160
    }
  ]
}
```

Filesystem tools enforce the configured read and write policies. Paths must resolve under `root`; escape attempts through absolute paths outside `root`, `..`, or symlinks are rejected.

```python
harness = Harness(HarnessConfig(
    root="/repo",
    read_paths=["src", "tests"],
    write_paths=["outputs"],
))
```

With this configuration, `read` can access `src/app.py` and `tests/test_app.py`, but not `docs/notes.md`. `write` can create or update `outputs/report.md`, but not `src/generated.py`. Omit `read_paths` or `write_paths` to allow that operation anywhere under `root`.

## Custom Tools

Custom tools are registered as `ToolSpec` objects. A handler may return a `ToolResult`, a string, or JSON-serializable data. The model always receives a JSON envelope with `ok`, `content`, and `metadata`.

```python
from pydantic import BaseModel

from thinharness import Harness, HarnessConfig, ToolSpec


class LookupArgs(BaseModel):
    account_id: str


def lookup_account(args: LookupArgs) -> dict[str, str]:
    return {"account_id": args.account_id, "tier": "enterprise"}


harness = Harness(
    HarnessConfig(root="."),
    tools=[
        ToolSpec(
            name="lookup_account",
            description="Look up account metadata by account id.",
            parameters=LookupArgs,
            handler=lookup_account,
        )
    ],
)
```

Use a Pydantic model for `parameters` when you want argument validation. Invalid JSON, non-object arguments, and Pydantic validation errors are treated as model-retryable mistakes.

If a handler detects a domain mistake the model can fix, raise `ModelRetry`:

```python
from thinharness import ModelRetry


def lookup_account(args: LookupArgs) -> dict[str, str]:
    if not args.account_id.startswith("acct_"):
        raise ModelRetry("account_id must start with acct_")
    return {"account_id": args.account_id, "tier": "enterprise"}
```

Tool retry budgets are per tool name per run. Ordinary handler exceptions become failed tool results, but they are not retryable unless the tool result says so.

### Approval-Required Tools

Set `requires_approval=True` on a custom `ToolSpec` when the host application must review a model-requested action before the handler runs. Approval is loop control flow, not structured output: if any call in a model-emitted batch requires approval, ThinHarness pauses before executing the whole batch and returns a normal `HarnessResult` with `stop_reason="approval_required"`.

The paused result includes:

- `pending_approvals`: call id, tool name, and raw JSON arguments for each approval-required call.
- `resume_state`: one JSON-serializable approval envelope that wraps the provider resume payload, the full paused tool batch, run history, usage, metadata, and accounting needed to continue the same logical run. For built-in providers, the nested `provider_state` is the full neutral transcript.

Resume with `resume_approvals(...)`, `stream_approvals(...)`, or `resume_approvals_sync(...)` and one `ApprovalDecision` per pending approval. Approved calls execute through the normal tool machinery, including hooks, tracing, retry accounting, and stream events. Rejected calls do not execute or fire tool hooks; the model receives a failed tool result with `error_type="ApprovalRejected"` and can explain, recover, or request another tool.

Approval-required tools need a resumable model because the harness must continue after the paused assistant tool-call turn. They cannot use background execution, and they are not supported inside child subagent harnesses. Built-in tools remain non-approval tools in this version; wrap built-in behavior in a custom `ToolSpec` when host review is required.

### Bash Prototype Tool

`BashTool` is an opt-in custom tool for exploratory agent runs. It is not part of the default built-ins, and `builtin_tools=["bash"]` is intentionally rejected.

```python
from thinharness import BashTool, Harness, HarnessConfig


harness = Harness(
    HarnessConfig(root="."),
    tools=[BashTool(root=".").spec()],
)
```

The tool runs one `bash -c` command from a workspace-contained cwd and marks itself sequential because commands may mutate state. The cwd check is not a sandbox: commands can still access absolute paths, network tools, environment variables, and anything the host process can access. The tool has a configured `max_tool_chars` output cap; the model can pass `max_chars` on an individual call only to request a lower cap. The final limit is `min(max_chars, max_tool_chars)`, applied independently to stdout and stderr. Background descendants left by a command are cleaned up when the shell exits; this is not a persistent job runner. Use it to prototype workflow shape, then promote repeated shell logic into typed tools.

## Tool Execution Policy

By default, same-turn tool calls run concurrently when all called tools are parallel-safe. The provider-facing outputs and `tool_call_records` still preserve the model's original tool-call order.

Set `sequential=True` on a `ToolSpec` when calls to that tool must not overlap with sibling tool calls:

```python
ToolSpec(
    name="send_invoice",
    description="Send an invoice email.",
    parameters=InvoiceArgs,
    handler=send_invoice,
    sequential=True,
)
```

If any tool in a model-emitted batch is sequential, the whole batch runs serially in model order. You can force every batch to run serially with:

```python
HarnessConfig(tool_execution="sequential")
```

## Background Tools

Background tools let long-running independent work stop blocking the agent loop. The work is still owned by the current `Harness.run(...)`: the model gets a start notice, continues other work, and later receives the completion as provider input.

There is no detached job queue, polling API, or job-control surface.

```python
import asyncio

from pydantic import BaseModel

from thinharness import ToolSpec


class ReportArgs(BaseModel):
    topic: str


async def build_report(args: ReportArgs) -> str:
    await asyncio.sleep(10)
    return f"report for {args.topic}"


report_tool = ToolSpec(
    name="build_report",
    description="Build a long report for a topic.",
    parameters=ReportArgs,
    handler=build_report,
    background="model",
)
```

Background modes:

- `background="never"` is the default.
- `background="model"` exposes a private `_background: true` argument when `tool_execution` is not sequential. The model chooses whether to start that call in the background.
- `background="always"` always starts the tool in the background and does not expose a model-facing switch.

Sequential tools cannot run in the background. `tool_execution="sequential"` disables model-facing background arguments and rejects `background="always"` tools.

## Structured Output

Set `output_type` to validate the final result with Pydantic. `result.text` remains the final text; `result.output` contains the parsed value.

```python
from pydantic import BaseModel

from thinharness import Harness, HarnessConfig


class Summary(BaseModel):
    title: str
    bullets: list[str]


harness = Harness(HarnessConfig(
    root=".",
    output_type=Summary,
    output_mode="auto",
))

result = await harness.run("Summarize README.md.")
summary: Summary = result.output
```

`output_mode` options:

- `auto`: choose the best mode for the provider.
- `native`: request provider-native JSON-schema output.
- `tool`: expose a synthetic `final_result` tool.
- `prompted`: ask for JSON in prompt instructions and validate the text.
- `text`: copy final text into `result.output`.

ThinHarness resolves structured output when the harness is constructed and eagerly checks the requested mode against the model provider's declared capabilities. For example, `output_mode="native"` raises immediately for a provider that does not support native JSON-schema output. `output_mode="auto"` chooses a supported default for the provider.

Tool-mode structured output uses a harness-created `final_result` tool. It is not a normal registered tool, does not fire tool hooks, and clean exits through it are not resumable because the provider transcript would contain an unanswered synthetic tool call.

## Hooks

Hooks are runtime callables registered on a `Harness`. They can observe lifecycle events, append prompt context, cancel selected before-events, or rewrite tool output.

```python
from thinharness import Hook, Harness, HarnessConfig


def add_policy(ctx) -> None:
    ctx.additional_context.append("Use the internal refund policy dated 2026-06.")


harness = Harness(
    HarnessConfig(root="."),
    hooks=[Hook("user_prompt_submit", add_policy)],
)
```

Hook events:

- `run_start`
- `user_prompt_submit`
- `before_tool_call`
- `after_tool_call`
- `before_subagent_run`
- `after_subagent_run`
- `limit_reached`
- `run_end`

`user_prompt_submit`, `before_tool_call`, and `before_subagent_run` are cancellable. `after_tool_call` can rewrite `ctx.output`, but retry control flow is captured before that rewrite. Tool filters apply only to tool events; agent filters apply only to subagent events.

By default, hook exceptions are logged and the run continues. Set `strict_hooks=True` to make hook exceptions fail the run.

## Subagents

The `subagent` tool is opt-in. It lets the parent delegate a bounded task to a child harness. Child runs start fresh; they do not inherit the parent provider transcript.

```python
from thinharness import Harness, HarnessConfig, SubAgentConfig


harness = Harness(HarnessConfig(
    root=".",
    builtin_tools=["read", "search", "subagent"],
    subagents=[
        SubAgentConfig(
            name="reviewer",
            description="Review a draft for factual and citation issues.",
            system_prompt="You are a careful review agent.",
            inherit_parent_tools=True,
            max_model_requests=12,
        )
    ],
))
```

Calling `subagent` without an `agent` argument uses the framework default subagent, which inherits parent tools except for recursive `subagent` access and MCP-discovered tools. Named subagents use their own `SubAgentConfig`.

Named subagents can:

- inherit parent tools with `inherit_parent_tools=True`
- choose explicit `builtin_tools`
- receive explicit custom `tools`
- opt into MCP with `inherit_mcp_servers=True` or `mcp_servers=[...]`
- use their own model, limits, structured output, and background policy

`default` is reserved for the framework default subagent name.

## Parallel LLM Batches

`parallel_llm` is an opt-in built-in tool for batches of independent one-shot prompts:

```python
harness = Harness(HarnessConfig(
    root=".",
    builtin_tools=["parallel_llm"],
    builtin_parallel_llm_model="openai:gpt-5.5-mini",
    builtin_parallel_llm_temperature=0,
    parallel_llm_max_prompts=100,
    parallel_llm_max_attempts=4,
))
```

Each batch call is stateless. Per-prompt calls receive no tools, no memory, no continuation, and no inherited parent harness system prompt. Pass `system` when the batch needs shared instructions.

The model-facing prompt source is structurally discriminated:

- inline prompts: `{"kind": "inline", "prompts": [...]}`
- prompt file: `{"kind": "file", "path": "prompts.json"}`

Use `output_file` when combined results may be large. Inline output returns compact JSON in `ToolResult.content`; file output writes pretty JSON under the write path policy and returns a summary.

`max_concurrency` is model-controlled per tool call and only limits in-flight attempts. `parallel_llm_max_prompts` and `parallel_llm_max_attempts` are host-controlled `HarnessConfig` fields. Internal parallel attempts are reported in the tool payload and metadata; they do not consume `max_model_requests`, while the `parallel_llm` invocation itself still counts as one tool call.

For a custom, renameable version, construct `ParallelLlmTool` directly:

```python
from pydantic import BaseModel

from thinharness import Harness, HarnessConfig, ParallelLlmTool


class InvoiceFields(BaseModel):
    vendor: str
    total: float


extract_tool = ParallelLlmTool(
    name="parallel_extract",
    description="Extract fields from independent chunks.",
    model="openai:gpt-5.5-mini",
    root=".",
    read_paths=["inputs"],
    write_paths=["outputs"],
    max_prompts=50,
    max_attempts=2,
    output_type=InvoiceFields,
    output_mode="auto",
    output_retries=1,
).spec()

harness = Harness(HarnessConfig(root="."), tools=[extract_tool])
```

When `output_type` is set on a custom `ParallelLlmTool`, successful entries contain parsed JSON-compatible values rather than raw text.

## Skills

Skills are explicit tools, not auto-discovery. Configure `skills_dir`, then expose `skill_read` and/or `skill_run` through `builtin_tools`.

```python
harness = Harness(HarnessConfig(
    root=".",
    skills_dir="skills",
    selected_skills=["invoice-review"],
    builtin_tools=["read", "search", "skill_read", "skill_run"],
))
```

If skills are configured and skill tools are exposed, the system prompt includes a compact skill summary. The model still has to call `skill_read` to inspect details.

`skill_run` runs scripts from trusted skill directories. Python scripts run through `uv run`; shell scripts run through `bash`; JavaScript and Go files use `node` and `go run`.

## MCP

MCP support is optional. Importing ThinHarness does not require the `mcp` package; using MCP requires the `mcp` extra.

```python
from thinharness import Harness, HarnessConfig, MCPServerStdio


harness = Harness(HarnessConfig(
    root=".",
    mcp_servers=[
        MCPServerStdio(
            "uvx",
            ["my-mcp-server"],
            tool_prefix="external",
            include_tools=["lookup"],
        )
    ],
))
```

MCP servers connect lazily during harness startup. Discovered MCP tools become normal `ToolSpec` objects in the live harness tool map. Name collisions are rejected; use `tool_prefix`, `include_tools`, or `exclude_tools` to keep the model-facing tool surface explicit.

Available transports:

- `MCPServerStdio`
- `MCPServerSSE`
- `MCPServerStreamableHTTP`

ThinHarness only turns MCP tools into harness tools. MCP prompts, resources, sampling, OAuth flows, and `.mcp.json` discovery are outside the current scope.

## Resume

`HarnessResult.resume_state` is an opaque, JSON-serializable token that lets callers continue a completed conversation with a new user message.

```python
first = await harness.run("Summarize this repository.")
if first.resume_state is None:
    raise RuntimeError("run cannot be continued")

second = await harness.run(
    "Now turn that into a checklist.",
    resume_from=first.resume_state,
)
```

The contract:

- Save `result.resume_state` exactly as JSON.
- Pass it back as `resume_from` with the next user message.
- Built-in provider state is a self-contained transcript and can be resumed by any built-in provider or model.
- The resuming harness supplies the live system prompt and tool schemas; captured system prompts are not stored or restored.
- Expect no state after failed, cancelled, partial, or exhausted runs.
- Treat the contents as harness-owned details; persist them exactly, but do not construct them by hand.

`resume_from` is a new-turn API. The prior run completed, and the next call appends a new user message. It is not a retry mechanism, interrupted-tool-call recovery, or a way to continue the assistant's previous response.

Approval pauses use a separate resume path. When `stop_reason == "approval_required"`, persist the returned `resume_state` envelope and call `resume_approvals(...)` with approval decisions instead of passing that envelope to `run(..., resume_from=...)`. The post-resume result carries the full logical run history: pre-pause responses, tool records, usage counters, and metadata are restored before the approved or rejected batch is processed.

Budgets span the pause. The paused batch counts against `usage.tool_calls` exactly once at pause time, and a resumed run can immediately hit `limit_reached` if the logical run was already at its configured model-request or tool-call limit. The approval envelope also includes the full nested provider transcript and raw provider responses, so its stored size grows with run length. `APPROVAL_ENVELOPE_VERSION` remains independent from the nested provider-state version; old nested provider states fail when the provider resume step validates them.

Built-in provider resume details:

- `resume_state["kind"] == "transcript"` and `version == 2`.
- The transcript is provider-agnostic and no longer depends on OpenAI server-side response retention.
- Provider-specific reasoning chains are not preserved; visible reasoning text, ordinary assistant text, tool calls, user messages, tool results, and harness notices are replayed.
- Cross-provider resume is supported by the built-in renderers, but real providers may reject foreign-format tool-call ids or malformed tool-call argument JSON.
- `OpenAIResponsesSession.start(previous_response_id=...)` remains available as a low-level escape hatch, but later resume state captures only the new prompt onward, not the externally seeded prior turns.

The same `resume_state` can be reused for sequential branching:

```python
base = await harness.run("Draft three product names.")
one = await harness.run("Make them more formal.", resume_from=base.resume_state)
two = await harness.run("Make them more playful.", resume_from=base.resume_state)
```

For parallel branches, use separate `Harness` instances.

## Limits And Notices

Hard limits stop runs:

- `max_model_requests`: maximum provider turns in one run.
- `max_tool_calls`: maximum model-requested ordinary tool calls in one run.
- `tool_retries`: default retry budget per tool name.
- `output_retries`: structured-output retry budget.

Near-limit notices are deterministic model input emitted before some limits are exhausted. They are not hooks, and they do not replace hard limit enforcement. Parent and child runs compute notices from their own local budgets.

Because notices are real provider input, they may become part of provider history and resume state.

## Tracing

Local tracing is on by default. It writes plaintext JSONL traces under:

```text
~/.thinharness/traces/<encoded-project-root>/
```

Those traces can include prompts, model outputs, tool arguments, and tool results. Treat them as sensitive local data.

Disable local trace files with:

```python
HarnessConfig(local_tracing=False)
```

or:

```bash
THINHARNESS_DISABLE_LOCAL_TRACING=1
```

External tracing uses OpenTelemetry-compatible tracers:

```python
from thinharness import Harness, HarnessConfig, TracingOptions, create_otlp_tracing


otlp = create_otlp_tracing(
    service_name="thinharness-agent",
    endpoint="https://otel.example.com/v1/traces",
)

harness = Harness(
    HarnessConfig(root="."),
    tracing=[TracingOptions(
        tracer=otlp.tracer,
        agent_name="support-review-agent",
        capture_messages=False,
        capture_tool_args=False,
        capture_tool_results=False,
    )],
)
```

Each tracing sink owns its capture policy. External spans can exist without recording raw prompts or tool payloads unless capture flags are enabled.

## Result Object

`Harness.run(...)` returns `HarnessResult`:

- `text`: final model text.
- `output`: parsed structured output, if configured.
- `responses`: raw provider responses.
- `tool_call_records`: normalized tool call and output records.
- `usage`: model request counts, tool call counts, cancellations, and retry counters.
- `stop_reason`: terminal reason.
- `resume_state`: opaque continuation state when the run is cleanly resumable.
- `pending_approvals`: pending human approval records when `stop_reason == "approval_required"`.

Most applications should read `text`, `output`, `usage`, and `resume_state`; the raw provider responses and tool records are mainly for debugging, auditing, and tests.
