# Comparison Notes

LOC is `tokei` "Code" lines (excludes comments and blanks). Each row is measured
**strict framework-only**: the measurement strips clearly non-framework code from
the upstream package: platform/deployment layers, domain-specific modalities
(voice, realtime), eval/optimizer suites, UI/CLI tools, A2A/declarative wire
protocols, code-executor backends, and similar non-framework layers. Provider
implementations stay in because they are part of what you import to use the
library. Tests, examples, and docs are always excluded.

The README is the canonical rendered table. The raw README source comments keep
the command beside each row. To reproduce locally, clone each upstream repo at
the pinned commit and run the command shown below.

Measured 2026-05-16 for upstream libraries. ThinHarness was remeasured from the
current working tree on 2026-05-17.

## LOC Commands

- **ThinHarness** — `tokei thinharness/ -t Python`
- **Claude Agent SDK** — `tokei src/claude_agent_sdk/ -t Python --exclude testing` at `anthropics/claude-agent-sdk-python @ c352a50`
- **smolagents** — `tokei src/smolagents/ -t Python --exclude cli.py --exclude gradio_ui.py --exclude vision_web_browser.py` at `huggingface/smolagents @ 025b6ad`
- **deepagents** — `tokei libs/deepagents/deepagents/ -t Python` at `langchain-ai/deepagents @ 7465d77`
- **AWS Strands** — `tokei src/strands/ -t Python --exclude experimental --exclude vended_plugins --exclude multiagent/a2a` at `strands-agents/sdk-python @ 1232230`
- **Microsoft Agent Framework** — `tokei python/packages/core/agent_framework/ -t Python --exclude _evaluation.py --exclude a2a --exclude ag_ui --exclude chatkit --exclude declarative --exclude devui --exclude hyperlight --exclude lab --exclude orchestrations --exclude mem0 --exclude redis --exclude microsoft` at `microsoft/agent-framework @ a60e541`
- **Pydantic AI** — `tokei pydantic_ai_slim/pydantic_ai/ -t Python --exclude _a2a.py --exclude ag_ui.py --exclude ui --exclude durable_exec --exclude embeddings --exclude ext` at `pydantic/pydantic-ai @ ac684b2`
- **Google ADK** — `tokei src/google/adk/ -t Python --exclude a2a --exclude apps --exclude cli --exclude cloud --exclude code_executors --exclude environment --exclude evaluation --exclude examples --exclude integrations --exclude optimization --exclude platform` at `google/adk-python @ bd062ec`
- **OpenAI Agents SDK** — `tokei src/agents/ -t Python --exclude realtime --exclude voice --exclude extensions/experimental --exclude extensions/visualization.py` at `openai/openai-agents-python @ 4bd459e`
- **Agno** — `tokei libs/agno/agno/{agent,agents,approval,compression,factory,guardrails,hooks,memory,models,reasoning,registry,run,session,skills,team,tools,tracing,utils} -t Python` at `agno-agi/agno @ bb7ddb0`

Claude Agent SDK also shells out to the Claude Code CLI binary, which is 200k+
LOC. The table counts the Python SDK package and footnotes that relationship.

## Notes on the Marks

- **Tool retries** — only Pydantic AI (`ModelRetry`) and OpenAI Agents (`ModelRetryAdvice` / `ModelRetrySettings`) ship a documented, named primitive that lets a tool function signal "model passed bad args — please retry with this feedback," distinct from generic exception propagation. AWS Strands has hook-based retry via `AfterToolCallEvent.retry=True`, Google ADK has a `ReflectAndRetryToolPlugin`, and Agno has a `RetryAgentRun` exception that retries the whole agent run rather than a single tool — these are marked `⚠️`. Claude Agent SDK, smolagents, deepagents, and Microsoft Agent Framework have no named primitive and are marked `❌`.
- **Subagents** — Pydantic AI documents an "agent delegation" pattern, where one agent is called inside another's tool function, but ships no class, decorator, or middleware for it. Its own multi-agent docs point users to deepagents for that case, so it is marked `❌`.
- **Structured output** — Claude Agent SDK and deepagents return free-form messages with no built-in validation step, so they are marked `❌`.
- **Skills** — Pydantic AI, OpenAI Agents SDK, smolagents, and AWS Strands have no Markdown/frontmatter skills primitive, so they are marked `❌`.
- **Built-in FS tools** — Pydantic AI, OpenAI Agents SDK, Google ADK, smolagents, AWS Strands, and Microsoft Agent Framework do not ship typed read/write/edit/search-style filesystem tools in the core package, so they are marked `❌`. Generic shell or code-exec tools do not count here.
- **OTel tracing** — deepagents leans on LangSmith rather than emitting OTel from its own code, so it is marked `❌`. Claude Agent SDK is marked `⚠️` because the Python SDK itself ships no instrumentation beyond W3C traceparent propagation into the CLI subprocess, while the Claude Code CLI it shells out to has beta OTel support.

## What "Strict Framework-Only" Excludes

Carving principle: keep everything you'd import to *use the library as an agent
framework*: agent loop, hooks, tools, structured output, skills, subagents,
memory, session, tracing, provider/model implementations, and MCP. Strip what's
clearly outside that scope: deployment layers, evals, UI/CLI tools,
voice/realtime modalities, declarative wire protocols, and code-executor
backends.

Per-library exclusions:

- **ThinHarness, deepagents** — nothing stripped; already framework-only.
- **Claude Agent SDK** — `testing/` (user-facing test helpers).
- **smolagents** — `cli.py`, `gradio_ui.py`, `vision_web_browser.py` (CLI + UI).
- **AWS Strands** — `experimental/`, `vended_plugins/`, `multiagent/a2a/` (experimental APIs, opt-in plugins, A2A wire protocol).
- **Microsoft Agent Framework** — `_evaluation.py`, `a2a/`, `ag_ui/`, `chatkit/`, `declarative/`, `devui/`, `hyperlight/`, `lab/`, `orchestrations/`, `mem0/`, `redis/`, `microsoft/` (evals, wire protocols, UI, runtime backends, storage backends, lab/experimental).
- **Pydantic AI** — `_a2a.py`, `ag_ui.py`, `ui/`, `durable_exec/`, `embeddings/`, `ext/` (A2A, UI, durable execution runtime, embedding models, ext).
- **Google ADK** — `a2a/`, `apps/`, `cli/`, `cloud/`, `code_executors/`, `environment/`, `evaluation/`, `examples/`, `integrations/`, `optimization/`, `platform/`.
- **OpenAI Agents SDK** — `realtime/`, `voice/`, `extensions/experimental`, `extensions/visualization.py`.
- **Agno** — `api/`, `client/`, `cloud/`, `db/`, `integrations/`, `knowledge/`, `learn/`, `os/`, `remote/`, `scheduler/`, `vectordb/`, `context/`, `culture/`, plus boundary cases `workflow/` and `eval/`. As shipped, Agno is 254,377 LOC.
