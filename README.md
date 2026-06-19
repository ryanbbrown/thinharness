<p align="center">
  <img src="assets/ThinHarness.svg" alt="ThinHarness" width="360">
</p>

<p align="center">
  <br/>
  A minimal, opinionated agent harness &mdash;
  <br/>
  focused scope, straightforward code, easy to fork.
  <br/><br/>
</p>

<div align="center">

[![CI](https://img.shields.io/github/actions/workflow/status/ryanbbrown/thinharness/ci.yml?branch=main&label=CI)](https://github.com/ryanbbrown/thinharness/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/ryanbbrown/thinharness/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/thinharness.svg)](https://pypi.org/project/thinharness/)

</div>

## Why this exists

Production agents rarely stop at framework configuration. Things like orchestration, permissions, user/session storage, and deployment become specific to the application and its users.

ThinHarness exists for the gap between building the agent loop yourself and adopting a large agent runtime where the loop comes bundled with assumptions you don’t need and can’t easily change.

It owns a focused set of agent-loop primitives that generalize well and are tedious to rebuild, leaving the rest of the application stack for you to own.

I started building ThinHarness after running into this gap in practice. Filesystem-enabled agents are simple but powerful, yet most frameworks don't include them out of the box, and the ones that do are among the largest and heaviest options (Claude Code -> Claude Agent SDK, LangChain -> deepagents, Agno). I usually needed only a small slice of the functionality, but that slice came with coupled assumptions that didn't match my application. Making it fit meant writing enough wrappers, adapters, and fixes that I ended up owning framework-shaped code anyway.

<!--
  LOC measurement scope: strict framework-only. Each row strips clearly
  non-framework code from the upstream package — platform/deployment layers,
  domain-specific modalities (voice/realtime), eval/optimizer suites, UI/CLI
  tools, A2A/declarative wire protocols, code-executor backends. Provider
  implementations stay IN (they're part of what you import to use the library).
  The exact tokei command + upstream commit hash for each upstream row is in an
  HTML comment above the row, so the number is reproducible. Upstream rows were
  measured 2026-06-15 against the pinned commits; ThinHarness was remeasured
  from this working tree on 2026-06-19.
-->

<div align="center">

<table>
  <thead>
    <tr>
      <td align="left" width="256" bgcolor="#eaeef2"><b>Library</b></td>
      <td align="center" width="70" bgcolor="#eaeef2"><b>LOC<sup>1</sup></b></td>
      <td align="center" width="62" bgcolor="#eaeef2"><b>Tool<br>retries<sup>2</sup></b></td>
      <td align="center" width="70" bgcolor="#eaeef2"><b>Subagents</b></td>
      <td align="center" width="68" bgcolor="#eaeef2"><b>Structured<br>output</b></td>
      <td align="center" width="52" bgcolor="#eaeef2"><b>Skills</b></td>
      <td align="center" width="82" bgcolor="#eaeef2"><b>FS<br>tools</b></td>
      <td align="center" width="62" bgcolor="#eaeef2"><b>OTel<br>tracing</b></td>
    </tr>
  </thead>
  <tbody>
    <!-- LOC: tokei thinharness/ -t Python  ·  ryanbbrown/thinharness working tree, measured 2026-06-19 -->
    <tr>
      <td align="left" bgcolor="#f6f8fa"><b>ThinHarness</b></td>
      <td align="right" bgcolor="#f6f8fa"><b>8,197</b></td>
      <td align="center" bgcolor="#f6f8fa"><b>✅</b></td>
      <td align="center" bgcolor="#f6f8fa"><b>✅</b></td>
      <td align="center" bgcolor="#f6f8fa"><b>✅</b></td>
      <td align="center" bgcolor="#f6f8fa"><b>✅</b></td>
      <td align="center" bgcolor="#f6f8fa"><b>✅</b></td>
      <td align="center" bgcolor="#f6f8fa"><b>✅</b></td>
    </tr>
    <!-- LOC: tokei src/claude_agent_sdk/ -t Python --exclude testing  ·  anthropics/claude-agent-sdk-python @ 634c2f6 -->
    <tr>
      <td align="left" bgcolor="#ffffff">
        <img src="https://cdn.jsdelivr.net/npm/simple-icons@latest/icons/anthropic.svg" width="20" height="20" align="absmiddle" alt="">
        &nbsp;Claude&nbsp;Agent&nbsp;SDK<sup>3</sup>
      </td>
      <td align="right" bgcolor="#ffffff">8,263</td>
      <td align="center" bgcolor="#ffffff">❌</td>
      <td align="center" bgcolor="#ffffff">✅</td>
      <td align="center" bgcolor="#ffffff">❌</td>
      <td align="center" bgcolor="#ffffff">✅</td>
      <td align="center" bgcolor="#ffffff">✅</td>
      <td align="center" bgcolor="#ffffff">⚠️</td>
    </tr>
    <!-- LOC: tokei src/smolagents/ -t Python --exclude cli.py --exclude gradio_ui.py --exclude vision_web_browser.py  ·  huggingface/smolagents @ e8b988d -->
    <tr>
      <td align="left" bgcolor="#ffffff">
        <img src="https://cdn.jsdelivr.net/npm/simple-icons@latest/icons/huggingface.svg" width="20" height="20" align="absmiddle" alt="">
        &nbsp;smolagents
      </td>
      <td align="right" bgcolor="#ffffff">9,840</td>
      <td align="center" bgcolor="#ffffff">❌</td>
      <td align="center" bgcolor="#ffffff">✅</td>
      <td align="center" bgcolor="#ffffff">✅</td>
      <td align="center" bgcolor="#ffffff">❌</td>
      <td align="center" bgcolor="#ffffff">❌</td>
      <td align="center" bgcolor="#ffffff">✅</td>
    </tr>
    <!-- LOC: tokei libs/deepagents/deepagents/ -t Python  ·  langchain-ai/deepagents @ 5975503 -->
    <!-- Substrate (see footnote 4): deepagents is a thin wrapper over LangChain/LangGraph.
         Effective import surface ≈111k LOC, measured with the same strict filter as the rest of the table:
           tokei libs/langgraph/langgraph/ libs/prebuilt/langgraph/ -t Python  ·  langchain-ai/langgraph @ 9732084  =>  26,987
           tokei libs/core/langchain_core/ -t Python --exclude document_loaders --exclude documents --exclude embeddings --exclude indexing --exclude retrievers.py --exclude vectorstores --exclude cross_encoders.py  ·  langchain-ai/langchain @ 62f2559  =>  55,442
           tokei libs/langchain_v1/langchain/ -t Python --exclude embeddings  ·  langchain-ai/langchain @ 62f2559  =>  11,663
           deepagents itself: 17,039
    -->
    <tr>
      <td align="left" bgcolor="#ffffff">
        <img src="https://cdn.jsdelivr.net/npm/simple-icons@latest/icons/langchain.svg" width="20" height="20" align="absmiddle" alt="">
        &nbsp;deepagents<sup>4</sup>
      </td>
      <td align="right" bgcolor="#ffffff">17,039</td>
      <td align="center" bgcolor="#ffffff">❌</td>
      <td align="center" bgcolor="#ffffff">✅</td>
      <td align="center" bgcolor="#ffffff">❌</td>
      <td align="center" bgcolor="#ffffff">✅</td>
      <td align="center" bgcolor="#ffffff">✅</td>
      <td align="center" bgcolor="#ffffff">❌</td>
    </tr>
    <!-- LOC: tokei strands-py/src/strands/ -t Python --exclude experimental --exclude vended_plugins --exclude multiagent/a2a  ·  strands-agents/sdk-python @ a92502f -->
    <tr>
      <td align="left" bgcolor="#ffffff">
        <img src="https://cdn.jsdelivr.net/npm/simple-icons@latest/icons/amazonwebservices.svg" width="20" height="20" align="absmiddle" alt="">
        &nbsp;AWS Strands
      </td>
      <td align="right" bgcolor="#ffffff">28,157</td>
      <td align="center" bgcolor="#ffffff">⚠️</td>
      <td align="center" bgcolor="#ffffff">✅</td>
      <td align="center" bgcolor="#ffffff">✅</td>
      <td align="center" bgcolor="#ffffff">❌</td>
      <td align="center" bgcolor="#ffffff">❌</td>
      <td align="center" bgcolor="#ffffff">✅</td>
    </tr>
    <!-- LOC: tokei python/packages/core/agent_framework/ -t Python --exclude _evaluation.py --exclude a2a --exclude ag_ui --exclude chatkit --exclude declarative --exclude devui --exclude hyperlight --exclude lab --exclude orchestrations --exclude mem0 --exclude redis --exclude microsoft  ·  microsoft/agent-framework @ ed4ff18 -->
    <tr>
      <td align="left" bgcolor="#ffffff">
        <img src="https://cdn.jsdelivr.net/npm/simple-icons@latest/icons/microsoft.svg" width="20" height="20" align="absmiddle" alt="">
        &nbsp;Microsoft<br>
        Agent Framework
      </td>
      <td align="right" bgcolor="#ffffff">40,514</td>
      <td align="center" bgcolor="#ffffff">❌</td>
      <td align="center" bgcolor="#ffffff">✅</td>
      <td align="center" bgcolor="#ffffff">✅</td>
      <td align="center" bgcolor="#ffffff">✅</td>
      <td align="center" bgcolor="#ffffff">❌</td>
      <td align="center" bgcolor="#ffffff">✅</td>
    </tr>
    <!-- LOC: tokei pydantic_ai_slim/pydantic_ai/ -t Python --exclude _a2a.py --exclude ag_ui.py --exclude ui --exclude durable_exec --exclude embeddings --exclude ext  ·  pydantic/pydantic-ai @ fabeacc -->
    <tr>
      <td align="left" bgcolor="#ffffff">
        <img src="https://cdn.jsdelivr.net/npm/simple-icons@latest/icons/pydantic.svg" width="20" height="20" align="absmiddle" alt="">
        &nbsp;Pydantic AI
      </td>
      <td align="right" bgcolor="#ffffff">59,034</td>
      <td align="center" bgcolor="#ffffff">✅</td>
      <td align="center" bgcolor="#ffffff">❌</td>
      <td align="center" bgcolor="#ffffff">✅</td>
      <td align="center" bgcolor="#ffffff">❌</td>
      <td align="center" bgcolor="#ffffff">❌</td>
      <td align="center" bgcolor="#ffffff">✅</td>
    </tr>
    <!-- LOC: tokei src/google/adk/ -t Python --exclude a2a --exclude apps --exclude cli --exclude cloud --exclude code_executors --exclude environment --exclude evaluation --exclude examples --exclude integrations --exclude optimization --exclude platform  ·  google/adk-python @ 22adbe1 -->
    <tr>
      <td align="left" bgcolor="#ffffff">
        <img src="https://cdn.jsdelivr.net/npm/simple-icons@latest/icons/google.svg" width="20" height="20" align="absmiddle" alt="">
        &nbsp;Google ADK
      </td>
      <td align="right" bgcolor="#ffffff">64,890</td>
      <td align="center" bgcolor="#ffffff">⚠️</td>
      <td align="center" bgcolor="#ffffff">✅</td>
      <td align="center" bgcolor="#ffffff">✅</td>
      <td align="center" bgcolor="#ffffff">✅</td>
      <td align="center" bgcolor="#ffffff">❌</td>
      <td align="center" bgcolor="#ffffff">✅</td>
    </tr>
    <!-- LOC: tokei src/agents/ -t Python --exclude realtime --exclude voice --exclude extensions/experimental --exclude extensions/visualization.py  ·  openai/openai-agents-python @ c359c20 -->
    <tr>
      <td align="left" bgcolor="#ffffff">
        <img src="https://cdn.jsdelivr.net/npm/simple-icons@latest/icons/openai.svg" width="20" height="20" align="absmiddle" alt="">
        &nbsp;OpenAI&nbsp;Agents&nbsp;SDK
      </td>
      <td align="right" bgcolor="#ffffff">73,139</td>
      <td align="center" bgcolor="#ffffff">✅</td>
      <td align="center" bgcolor="#ffffff">✅</td>
      <td align="center" bgcolor="#ffffff">✅</td>
      <td align="center" bgcolor="#ffffff">❌</td>
      <td align="center" bgcolor="#ffffff">❌</td>
      <td align="center" bgcolor="#ffffff">✅</td>
    </tr>
    <!-- LOC: tokei libs/agno/agno/{agent,agents,approval,compression,factory,guardrails,hooks,memory,models,reasoning,registry,run,session,skills,team,tools,tracing,utils} -t Python  ·  agno-agi/agno @ 5cf1ed7 -->
    <tr>
      <td align="left" bgcolor="#ffffff">
        <img src="assets/agno-a.svg" width="20" height="20" align="absmiddle" alt="">
        &nbsp;Agno
      </td>
      <td align="right" bgcolor="#ffffff">111,539</td>
      <td align="center" bgcolor="#ffffff">⚠️</td>
      <td align="center" bgcolor="#ffffff">✅</td>
      <td align="center" bgcolor="#ffffff">✅</td>
      <td align="center" bgcolor="#ffffff">✅</td>
      <td align="center" bgcolor="#ffffff">✅</td>
      <td align="center" bgcolor="#ffffff">✅</td>
    </tr>
  </tbody>
</table>
<p align="center"><sub><em>* Table focuses on harness-level features that differentiate the libraries. All listed also support MCP, lifecycle hooks, multi-turn conversations, and human-in-the-loop. It intentionally does not compare framework/platform features like vector DB integrations, hosted deployment, memory/session stores, or broad SaaS connectors.</em></sub></p>

<p align="left">
  <sub>1. LOC excludes anything that is not the core agent harness framework. See raw README source comments for exact commands.<br>
  2. Tool retries: a documented primitive (e.g. Pydantic AI's <code>ModelRetry</code>) that lets tools signal "model passed bad args — retry with this feedback," distinct from generic exception propagation.<br>
  3. Claude Agent SDK shells out to the Claude Code CLI binary, which is 200k+ LOC.<br>
  4. deepagents is a thin wrapper over LangChain/LangGraph; effective import surface is ≈111k LOC.</sub>
</p>

</div>

<br>

See [docs/table.md](docs/table.md) for per-cell rationale and how the LOC numbers are measured.

## Opinions

ThinHarness has opinions. They are the reason it stays small.

**Purpose-built agents, not universal agents.** ThinHarness is for bounded agent loops inside software you control, not open-ended interactive assistants. For business use cases, focused agent loops orchestrated by deterministic code are usually a better fit than sprawling multi-agent systems with broad authority.

**No bash by default.** Purpose-built business agents usually don't need a shell. Bash is a broad security and reliability surface: it gives the model open-ended authority instead of typed, bounded actions. ThinHarness keeps bash out of the default and built-in tool sets, but exposes an opt-in `BashTool` for exploratory runs before the workflow is hardened with typed tools.

**Skills are tools, not auto-discovery.** Skills live in directories you point at explicitly. The agent calls `skill_read` and `skill_run` like any other tool. No interactive scan of the workspace, no global skill marketplace, no magic. SDK use is deliberate; the auto-discovery design is for interactive coding agents and doesn't belong here.

**Search is a top priority.** The `search` tool exposes ripgrep as compact grouped path/line results, tuned for document and business-workflow agents rather than code navigation. There's also a `jsonl_search` variant, because JSONL is the right shape when you're replacing RAG with agent-driven search over structured data: ripgrep row prefiltering, jq-style field projection, `where` filters, range filters, and snippets from large multiline fields.

**Parallel LLM calls, built in.** Fan out from inside the harness when a workflow needs reliability beyond a single agent loop — majority vote, ensembled extraction. Set `builtin_parallel_llm_model` to enable the default `parallel_llm` tool for plain-text batches; for validated structured output per call, instantiate `ParallelLlmTool` yourself with `output_type` (a Pydantic model). Each call is stateless, and large batches can write JSON to `output_file`.

**Background tools are simple.** Some long-running tools can start in the background so the agent can keep working. There is no detached job queue, polling API, or job-control surface; the current run still owns the task, and the completion is sent back to the model when it finishes.

**No token streaming.** Streaming is for workflow progress, not live chatbot text. ThinHarness emits run, model-turn, tool, retry, limit, background, and subagent events, but it does not stream provider token deltas. Token streaming would add provider-specific plumbing, event merging, cancellation edge cases, and more surface area to keep stable. For workflow-style agents, step-level updates are usually the useful signal.

**Three providers, no matrix.** ThinHarness ships small provider classes for OpenAI, Anthropic, and OpenRouter. If your gateway speaks one of those protocols, you swap a base URL and move on. If not, the provider classes are small enough to fork or replace, and ignoring the bundled ones costs you nothing

**No compaction.** Compaction is a workaround for context windows filling up across long, accumulating runs — useful for interactive coding sessions that sprawl over hours. For SDK-based business agents, the right answer to "context is getting big" is almost always better task decomposition: shorter runs, separate harness instances, narrower subagents.

**No deployment layer.** Agents still need serving, auth, durable jobs, user/session storage, and deployment in production. ThinHarness does not try to own that stack. A bundled deployment layer might work for some teams, but it will miss plenty of real production shapes; instead of adding more code and more options, ThinHarness leaves that application stack for you to own.

## Install

```bash
uv add thinharness     # or pip install thinharness
```

Requires Python 3.11+.

## Use

```python
import asyncio
from thinharness import Harness, HarnessConfig

async def main():
    async with Harness(HarnessConfig(root=".", model="openai:gpt-5.5")) as harness:
        result = await harness.run("Read README.md and summarize it.")
        print(result.text)

asyncio.run(main())
```

There's a synchronous wrapper too: `Harness(...).run_sync(...)`.

For workflow visibility, use `Harness.stream(...)`:

```python
from thinharness import RunCompletedEvent

async for event in harness.stream("Process these records."):
    if event.kind == "tool_call_started":
        print(event.tool_name)
    if isinstance(event, RunCompletedEvent):
        result = event.result
```

Streaming emits coarse run, model, tool, background, retry, limit, and subagent events, then finishes with the same `HarnessResult` returned by `run()`.

## Features

- **Filesystem tools:** `read`, `write`, batched exact-replacement `edit`, `search`, `list`, and `glob` with root-scoped path policies.
- **JSONL search:** opt-in `jsonl_search` for structured line-delimited data, with ripgrep prefiltering, field projection, equality/contains/regex/range `where` filters, and field-level snippets from large multiline string values.
- **Bash prototype tool:** opt-in `BashTool` for exploratory shell commands. It is lightweight, custom-registration only, and is not included in the default or built-in tool set.
- **Provider adapters:** built-in OpenAI, Anthropic, and OpenRouter adapters, plus public model/session protocols for implementing another provider.
- **Custom typed tools:** define sync or async `ToolSpec` handlers with Pydantic argument models, normalized `ToolResult` envelopes, sequential/background/approval flags, and per-tool retry settings.
- **Structured output:** Pydantic-validated results with native, tool, prompted, and text modes.
- **Hooks:** lifecycle and tool-call interception for prompt submission, tool calls, subagents, limits, and run boundaries.
- **Subagents:** opt-in delegation through a built-in `subagent` tool and explicit `SubAgentConfig`.
- **Parallel LLM:** opt-in `parallel_llm` fan-out for batches of independent one-shot prompts, plus `ParallelLlmTool(...).spec()` for renameable tools with explicit model, path, prompt, and retry settings.
- **Skills:** explicit `skill_read` and `skill_run` tools for selected skill directories, with Python, shell, JavaScript, and Go script runners.
- **Resume:** clean new-turn continuation through opaque provider session state.
- **MCP:** optional MCP client support with lazy tool discovery and collision checks.
- **Parallel tool calls:** same-turn tool batches run concurrently when every called tool is parallel-safe.
- **Background tools:** opt-in long-running tool calls return a start notice immediately, keep the agent loop moving, and deliver completion back to the model when ready.
- **Human approvals:** mark custom tools as approval-required so a run pauses before side effects, returns pending call details plus resume state, then continues after an approve/reject decision.
- **Event streaming:** async coarse-grained run, model, tool, background, retry, limit, and subagent events for workflow visibility.
- **Tool retries:** tools raise `ModelRetry` to send structured feedback back to the model and trigger a retry within a per-tool budget.
- **Limits and notices:** configured request, tool-call, output-retry, and tool-retry budgets bound each run; near-limit guidance can warn the model before request or tool-call budgets are exhausted.
- **Tracing:** local plaintext JSONL traces plus OpenTelemetry-compatible spans for runs, provider calls, tools, and subagents.

## Tracing

Local tracing is on by default. It writes full plaintext JSONL traces under `~/.thinharness/traces/<encoded-project-root>/`, including prompts, model outputs, tool arguments, and tool results, so treat that directory as sensitive local data.

Set `local_tracing=False` or `THINHARNESS_DISABLE_LOCAL_TRACING=1` to disable local trace files. External tracing is generic OpenTelemetry: pass any tracer with `start_as_current_span(...)` or `start_span(...)` in `TracingOptions`, and each sink keeps its own capture policy.

## Status

Pre-1.0. APIs may shift, but I don't expect dramatic changes. Forking is a real option, not just a theoretical one: the codebase is small enough that pulling upstream changes into your fork by hand stays cheap. Each major feature (MCP, subagents, jsonl_search, parallel_llm, background tools, skills) lives in its own file with no hidden dependencies. If you don't use one, that's even less code to worry about. If you want to delete it entirely, that's a one-shot 10-word prompt to a coding agent.

ThinHarness was built with coding agents, but isn't vibe-coded. I have used it, iterated on it, and reviewed its design + behavior. The [docs site](https://ryanbbrown.com/thinharness/) includes a [codebase explainer](https://ryanbbrown.com/thinharness/explainer.html) that I iterated on to understand the library, and the [web research example](https://ryanbbrown.com/thinharness/examples.html) has the transcript from a non-trivial agent run to show that it works effectively.

## License

MIT. See [LICENSE](LICENSE).
