<p align="center">
  <img src="assets/ThinHarness.svg" alt="ThinHarness" width="360">
</p>

<p align="center">
  <br/>
  A minimal, opinionated agent harness &mdash;
  <br/>
  focused scope, readable core, easy to fork.
  <br/><br/>
</p>

<div align="center">

[![CI](https://img.shields.io/github/actions/workflow/status/ryanbbrown/thinharness/ci.yml?branch=main&label=CI)](https://github.com/ryanbbrown/thinharness/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/ryanbbrown/thinharness/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/thinharness.svg)](https://pypi.org/project/thinharness/)

</div>

## Why this exists

Filesystem-based agent harnesses are simple but powerful: easily auditable, flexible, and they work just as well for non-coding business tasks like research over a corpus, workflow automation, or multi-step analysis. But the harnesses that provide filesystem primitives are either coding agents (Claude Agent SDK) or are massive and highly abstracted (deepagents, Agno). Even if you don't want filesystem tools, the general-purpose agent harness libraries are missing features (see table below) — or large enough that it's a pain when you (inevitably) need to customize.

So I built one. The core agent loop isn't that complicated. Provider call, parse tool calls, run them, feed results back, repeat. ThinHarness is **5,528 LOC** across 18 Python files. The whole thing. Small enough to actually read. You can audit it. You can fork it without inheriting a fork-maintenance problem, because there isn't much there to drift.

<!--
  LOC measurement scope: strict framework-only. Each row strips clearly
  non-framework code from the upstream package — platform/deployment layers,
  domain-specific modalities (voice/realtime), eval/optimizer suites, UI/CLI
  tools, A2A/declarative wire protocols, code-executor backends. Provider
  implementations stay IN (they're part of what you import to use the library).
  The exact tokei command + upstream commit hash for each row is in an HTML
  comment above the row, so the number is reproducible. Measured 2026-05-16
  against the commit pinned in each row's comment.
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
    <!-- LOC: tokei thinharness/ -t Python  ·  ryanbbrown/thinharness working tree, measured 2026-06-07 -->
    <tr>
      <td align="left" bgcolor="#f6f8fa"><b>ThinHarness</b></td>
      <td align="right" bgcolor="#f6f8fa"><b>5,528</b></td>
      <td align="center" bgcolor="#f6f8fa"><b>✅</b></td>
      <td align="center" bgcolor="#f6f8fa"><b>✅</b></td>
      <td align="center" bgcolor="#f6f8fa"><b>✅</b></td>
      <td align="center" bgcolor="#f6f8fa"><b>✅</b></td>
      <td align="center" bgcolor="#f6f8fa"><b>✅</b></td>
      <td align="center" bgcolor="#f6f8fa"><b>✅</b></td>
    </tr>
    <!-- LOC: tokei src/claude_agent_sdk/ -t Python --exclude testing  ·  anthropics/claude-agent-sdk-python @ c352a50 -->
    <tr>
      <td align="left" bgcolor="#ffffff">
        <img src="https://cdn.jsdelivr.net/npm/simple-icons@latest/icons/anthropic.svg" width="20" height="20" align="absmiddle" alt="">
        &nbsp;Claude&nbsp;Agent&nbsp;SDK<sup>3</sup>
      </td>
      <td align="right" bgcolor="#ffffff">8,202</td>
      <td align="center" bgcolor="#ffffff">❌</td>
      <td align="center" bgcolor="#ffffff">✅</td>
      <td align="center" bgcolor="#ffffff">❌</td>
      <td align="center" bgcolor="#ffffff">✅</td>
      <td align="center" bgcolor="#ffffff">✅</td>
      <td align="center" bgcolor="#ffffff">⚠️</td>
    </tr>
    <!-- LOC: tokei src/smolagents/ -t Python --exclude cli.py --exclude gradio_ui.py --exclude vision_web_browser.py  ·  huggingface/smolagents @ 025b6ad -->
    <tr>
      <td align="left" bgcolor="#ffffff">
        <img src="https://cdn.jsdelivr.net/npm/simple-icons@latest/icons/huggingface.svg" width="20" height="20" align="absmiddle" alt="">
        &nbsp;smolagents
      </td>
      <td align="right" bgcolor="#ffffff">10,091</td>
      <td align="center" bgcolor="#ffffff">❌</td>
      <td align="center" bgcolor="#ffffff">✅</td>
      <td align="center" bgcolor="#ffffff">✅</td>
      <td align="center" bgcolor="#ffffff">❌</td>
      <td align="center" bgcolor="#ffffff">❌</td>
      <td align="center" bgcolor="#ffffff">✅</td>
    </tr>
    <!-- LOC: tokei libs/deepagents/deepagents/ -t Python  ·  langchain-ai/deepagents @ 7465d77 -->
    <!-- Substrate (see footnote 4): deepagents is a thin wrapper over LangChain/LangGraph.
         Effective import surface ≈105k LOC, measured with the same strict filter as the rest of the table:
           tokei libs/langgraph/langgraph/ libs/prebuilt/langgraph/ -t Python  ·  langchain-ai/langgraph @ 076e2a3  =>  26,144
           tokei libs/core/langchain_core/ -t Python --exclude document_loaders --exclude documents --exclude embeddings --exclude indexing --exclude retrievers.py --exclude vectorstores --exclude cross_encoders.py  ·  langchain-ai/langchain @ 73d4fd9  =>  54,992
           tokei libs/langchain_v1/langchain/ -t Python --exclude embeddings  ·  langchain-ai/langchain @ 73d4fd9  =>  ~9,000
           deepagents itself: 15,369
    -->
    <tr>
      <td align="left" bgcolor="#ffffff">
        <img src="https://cdn.jsdelivr.net/npm/simple-icons@latest/icons/langchain.svg" width="20" height="20" align="absmiddle" alt="">
        &nbsp;deepagents<sup>4</sup>
      </td>
      <td align="right" bgcolor="#ffffff">15,369</td>
      <td align="center" bgcolor="#ffffff">❌</td>
      <td align="center" bgcolor="#ffffff">✅</td>
      <td align="center" bgcolor="#ffffff">❌</td>
      <td align="center" bgcolor="#ffffff">✅</td>
      <td align="center" bgcolor="#ffffff">✅</td>
      <td align="center" bgcolor="#ffffff">❌</td>
    </tr>
    <!-- LOC: tokei src/strands/ -t Python --exclude experimental --exclude vended_plugins --exclude multiagent/a2a  ·  strands-agents/sdk-python @ 1232230 -->
    <tr>
      <td align="left" bgcolor="#ffffff">
        <img src="https://cdn.jsdelivr.net/npm/simple-icons@latest/icons/amazonwebservices.svg" width="20" height="20" align="absmiddle" alt="">
        &nbsp;AWS Strands
      </td>
      <td align="right" bgcolor="#ffffff">25,494</td>
      <td align="center" bgcolor="#ffffff">⚠️</td>
      <td align="center" bgcolor="#ffffff">✅</td>
      <td align="center" bgcolor="#ffffff">✅</td>
      <td align="center" bgcolor="#ffffff">❌</td>
      <td align="center" bgcolor="#ffffff">❌</td>
      <td align="center" bgcolor="#ffffff">✅</td>
    </tr>
    <!-- LOC: tokei python/packages/core/agent_framework/ -t Python --exclude _evaluation.py --exclude a2a --exclude ag_ui --exclude chatkit --exclude declarative --exclude devui --exclude hyperlight --exclude lab --exclude orchestrations --exclude mem0 --exclude redis --exclude microsoft  ·  microsoft/agent-framework @ a60e541 -->
    <tr>
      <td align="left" bgcolor="#ffffff">
        <img src="https://cdn.jsdelivr.net/npm/simple-icons@latest/icons/microsoft.svg" width="20" height="20" align="absmiddle" alt="">
        &nbsp;Microsoft<br>
        Agent Framework
      </td>
      <td align="right" bgcolor="#ffffff">34,751</td>
      <td align="center" bgcolor="#ffffff">❌</td>
      <td align="center" bgcolor="#ffffff">✅</td>
      <td align="center" bgcolor="#ffffff">✅</td>
      <td align="center" bgcolor="#ffffff">✅</td>
      <td align="center" bgcolor="#ffffff">❌</td>
      <td align="center" bgcolor="#ffffff">✅</td>
    </tr>
    <!-- LOC: tokei pydantic_ai_slim/pydantic_ai/ -t Python --exclude _a2a.py --exclude ag_ui.py --exclude ui --exclude durable_exec --exclude embeddings --exclude ext  ·  pydantic/pydantic-ai @ ac684b2 -->
    <tr>
      <td align="left" bgcolor="#ffffff">
        <img src="https://cdn.jsdelivr.net/npm/simple-icons@latest/icons/pydantic.svg" width="20" height="20" align="absmiddle" alt="">
        &nbsp;Pydantic AI
      </td>
      <td align="right" bgcolor="#ffffff">51,231</td>
      <td align="center" bgcolor="#ffffff">✅</td>
      <td align="center" bgcolor="#ffffff">❌</td>
      <td align="center" bgcolor="#ffffff">✅</td>
      <td align="center" bgcolor="#ffffff">❌</td>
      <td align="center" bgcolor="#ffffff">❌</td>
      <td align="center" bgcolor="#ffffff">✅</td>
    </tr>
    <!-- LOC: tokei src/google/adk/ -t Python --exclude a2a --exclude apps --exclude cli --exclude cloud --exclude code_executors --exclude environment --exclude evaluation --exclude examples --exclude integrations --exclude optimization --exclude platform  ·  google/adk-python @ bd062ec -->
    <tr>
      <td align="left" bgcolor="#ffffff">
        <img src="https://cdn.jsdelivr.net/npm/simple-icons@latest/icons/google.svg" width="20" height="20" align="absmiddle" alt="">
        &nbsp;Google ADK
      </td>
      <td align="right" bgcolor="#ffffff">57,392</td>
      <td align="center" bgcolor="#ffffff">⚠️</td>
      <td align="center" bgcolor="#ffffff">✅</td>
      <td align="center" bgcolor="#ffffff">✅</td>
      <td align="center" bgcolor="#ffffff">✅</td>
      <td align="center" bgcolor="#ffffff">❌</td>
      <td align="center" bgcolor="#ffffff">✅</td>
    </tr>
    <!-- LOC: tokei src/agents/ -t Python --exclude realtime --exclude voice --exclude extensions/experimental --exclude extensions/visualization.py  ·  openai/openai-agents-python @ 4bd459e -->
    <tr>
      <td align="left" bgcolor="#ffffff">
        <img src="https://cdn.jsdelivr.net/npm/simple-icons@latest/icons/openai.svg" width="20" height="20" align="absmiddle" alt="">
        &nbsp;OpenAI&nbsp;Agents&nbsp;SDK
      </td>
      <td align="right" bgcolor="#ffffff">72,410</td>
      <td align="center" bgcolor="#ffffff">✅</td>
      <td align="center" bgcolor="#ffffff">✅</td>
      <td align="center" bgcolor="#ffffff">✅</td>
      <td align="center" bgcolor="#ffffff">❌</td>
      <td align="center" bgcolor="#ffffff">❌</td>
      <td align="center" bgcolor="#ffffff">✅</td>
    </tr>
    <!-- LOC: tokei libs/agno/agno/{agent,agents,approval,compression,factory,guardrails,hooks,memory,models,reasoning,registry,run,session,skills,team,tools,tracing,utils} -t Python  ·  agno-agi/agno @ bb7ddb0 -->
    <tr>
      <td align="left" bgcolor="#ffffff">
        <img src="assets/agno-a.svg" width="20" height="20" align="absmiddle" alt="">
        &nbsp;Agno
      </td>
      <td align="right" bgcolor="#ffffff">106,852</td>
      <td align="center" bgcolor="#ffffff">⚠️</td>
      <td align="center" bgcolor="#ffffff">✅</td>
      <td align="center" bgcolor="#ffffff">✅</td>
      <td align="center" bgcolor="#ffffff">✅</td>
      <td align="center" bgcolor="#ffffff">✅</td>
      <td align="center" bgcolor="#ffffff">✅</td>
    </tr>
  </tbody>
</table>
<p align="center"><sub><em>* Table focuses on features that differentiate the harnesses. All listed also support MCP, lifecycle hooks, and multi-turn conversations.</em></sub></p>

<p align="left">
  <sub>1. LOC excludes anything that is not the core agent harness framework. See raw README source comments for exact commands.<br>
  2. Tool retries: a documented primitive (e.g. Pydantic AI's <code>ModelRetry</code>) that lets tools signal "model passed bad args — retry with this feedback," distinct from generic exception propagation.<br>
  3. Claude Agent SDK shells out to the Claude Code CLI binary, which is 200k+ LOC.<br>
  4. deepagents is a thin wrapper over LangChain/LangGraph; effective import surface is ≈105k LOC.</sub>
</p>

</div>

<br>

See [docs/table.md](docs/table.md) for per-cell rationale and how the LOC numbers are measured.

## Opinions

ThinHarness has opinions. They are the reason it stays small.

**No bash.** Business agents don't need a shell. Bash is a giant security surface, and agents mess up when writing shell commands more often than you'd initially expect. Cut it and most of those failures stop being possible.

**Skills are tools, not auto-discovery.** Skills live in directories you point at explicitly. The agent calls `skill_read` and `skill_run` like any other tool. No interactive scan of the workspace, no global skill marketplace, no magic. SDK use is deliberate; the auto-discovery design is for interactive coding agents and doesn't belong here.

**Search is a top priority.** The `search` tool exposes ripgrep as compact grouped path/line results, tuned for document and business-workflow agents rather than code navigation. There's also a `jsonl_search` variant, because JSONL is the right shape when you're replacing RAG with agent-driven search over structured data (line-delimited, naturally chunked, `jq` + `rg`).

**Parallel LLM calls, built in.** Fan out from inside the harness when a workflow needs reliability beyond a single agent loop — majority vote, ensembled extraction. Set `builtin_parallel_llm_model` to enable the default `parallel_llm` tool for plain-text batches; for validated structured output per call, instantiate `ParallelLlmTool` yourself with `output_type` (a Pydantic model). Each call is stateless, and large batches can write JSON to `output_file`.

**Three providers, no matrix.** ThinHarness ships small provider classes for OpenAI, Anthropic, and OpenRouter. If your gateway speaks one of those protocols, you swap a base URL and move on. If not, the provider classes are small enough to fork or replace, and ignoring the bundled ones costs you nothing

**No compaction.** Compaction is a workaround for context windows filling up across long, accumulating runs — useful for interactive coding sessions that sprawl over hours. For SDK-based business agents, the right answer to "context is getting big" is almost always better task decomposition: shorter runs, separate harness instances, narrower subagents.

**No deployment layer.** Agents still need serving, auth, storage, retries, and observability in production. ThinHarness does not try to own that stack. A bundled deployment layer might work for some teams, but it will miss plenty of real production shapes; instead of adding more code and more options, ThinHarness stays an SDK and lets the host application own deployment.

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
    async with Harness(HarnessConfig(root=".", model="openai:gpt-5.2")) as harness:
        result = await harness.run("Read README.md and summarize it.")
        print(result.text)

asyncio.run(main())
```

There's a synchronous wrapper (`Harness(...).run_sync(...)`), Pydantic-typed structured output, lifecycle hooks, subagents, and path-scoped FS tools. The whole library is 18 Python files; the loop you care about is in [`thinharness/core.py`](thinharness/core.py) and the tools live in [`thinharness/tools/`](thinharness/tools/). Reading those files is faster than reading the docs would be.

## Features

- **Filesystem tools:** `read`, `write`, `edit`, `search`, `list`, `glob`, and `jsonl_search` with root-scoped path policies.
- **Structured output:** Pydantic-validated results with native, tool, prompted, and text modes.
- **Hooks:** lifecycle and tool-call interception for prompt submission, tool calls, subagents, limits, and run boundaries.
- **Subagents:** opt-in delegation through a built-in `subagent` tool and explicit `SubAgentConfig`.
- **Parallel LLM:** opt-in `parallel_llm` fan-out for batches of independent one-shot prompts, plus `ParallelLlmTool(...).spec()` for renameable tools with explicit model, path, prompt, and retry settings.
- **Resume:** clean new-turn continuation through opaque provider session state.
- **MCP:** optional MCP client support with lazy tool discovery and collision checks.
- **Parallel tool calls:** same-turn tool batches run concurrently when every called tool is parallel-safe.
- **Tool retries:** tools raise `ModelRetry` to send structured feedback back to the model and trigger a retry within a per-tool budget.
- **Limit notices:** near-limit guidance can warn the model before configured request or tool-call budgets are exhausted. Notices are harness-owned model input, not hooks or callbacks; parent and child runs compute them from their own local budgets.
- **Tracing:** local plaintext JSONL traces plus OpenTelemetry-compatible spans for runs, provider calls, tools, and subagents.

## Tracing

Local tracing is on by default. It writes full plaintext JSONL traces under `~/.thinharness/traces/<encoded-project-root>/`, including prompts, model outputs, tool arguments, and tool results, so treat that directory as sensitive local data.

Set `local_tracing=False` or `THINHARNESS_DISABLE_LOCAL_TRACING=1` to disable local trace files. External tracing is generic OpenTelemetry: pass any tracer with `start_as_current_span(...)` or `start_span(...)` in `TracingOptions`, and each sink keeps its own capture policy.

## Status

Pre-1.0. APIs may shift, but I don't expect dramatic changes. Forking is a real option, not just a theoretical one: the codebase is small enough that pulling upstream changes into your fork by hand stays cheap. Each major feature (MCP, subagents, jsonl_search, parallel_llm, skills) lives in its own file with no hidden dependencies. If you don't use one, that's even less code to worry about. If you want to delete it entirely, that's a one-shot 10-word prompt to a coding agent.

## License

MIT. See [docs/THIRD_PARTY_NOTICES.md](docs/THIRD_PARTY_NOTICES.md).
