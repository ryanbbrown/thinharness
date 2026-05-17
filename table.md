# Comparison

LOC is `tokei` "Code" lines (excludes comments and blanks). Each row is measured
**strict framework-only**: we strip clearly non-framework code from the upstream
package — platform/deployment layers, domain-specific modalities (voice,
realtime), eval/optimizer suites, UI/CLI tools, A2A/declarative wire protocols,
code-executor backends. Provider implementations stay in (they're part of what
you import to use the library). Tests, examples, and docs are always excluded.

The exact `tokei` command + upstream commit hash for each row appears in an HTML
comment directly above the row in the table source — view source to verify any
number. To reproduce locally: clone each upstream repo at the pinned commit, run
the command shown.

Measured 2026-05-16.

<table>
  <thead>
    <tr>
      <th align="left">Library</th>
      <th align="right">LOC</th>
      <th align="center">Hooks</th>
      <th align="center">Subagents</th>
      <th align="center">Structured&nbsp;output</th>
      <th align="center">Skills</th>
      <th align="center">Multi&#8209;provider</th>
      <th align="center">MCP</th>
      <th align="center">Built&#8209;in&nbsp;FS&nbsp;tools</th>
      <th align="center">OTel&nbsp;tracing</th>
    </tr>
  </thead>
  <tbody>
    <!-- LOC: tokei thinharness/ -t Python  ·  ryanbbrown/thinharness @ 14ac220 -->
    <tr>
      <td align="left"><b>ThinHarness</b></td>
      <td align="right"><b>3,348</b></td>
      <td align="center">✅</td>
      <td align="center">✅</td>
      <td align="center">✅</td>
      <td align="center">✅</td>
      <td align="center">✅</td>
      <td align="center">❌</td>
      <td align="center">✅</td>
      <td align="center">✅</td>
    </tr>
    <!-- LOC: tokei src/claude_agent_sdk/ -t Python --exclude testing  ·  anthropics/claude-agent-sdk-python @ c352a50 -->
    <tr>
      <td align="left">
        <img src="https://cdn.jsdelivr.net/npm/simple-icons@latest/icons/anthropic.svg" width="20" height="20" alt="">
        &nbsp;Claude Agent SDK<sup>*</sup>
      </td>
      <td align="right">8,202</td>
      <td align="center">✅</td>
      <td align="center">✅</td>
      <td align="center">❌</td>
      <td align="center">✅</td>
      <td align="center">❌</td>
      <td align="center">✅</td>
      <td align="center">✅</td>
      <td align="center">⚠️</td>
    </tr>
    <!-- LOC: tokei src/smolagents/ -t Python --exclude cli.py --exclude gradio_ui.py --exclude vision_web_browser.py  ·  huggingface/smolagents @ 025b6ad -->
    <tr>
      <td align="left">
        <img src="https://cdn.jsdelivr.net/npm/simple-icons@latest/icons/huggingface.svg" width="20" height="20" alt="">
        &nbsp;smolagents
      </td>
      <td align="right">10,091</td>
      <td align="center">⚠️</td>
      <td align="center">✅</td>
      <td align="center">✅</td>
      <td align="center">❌</td>
      <td align="center">✅</td>
      <td align="center">✅</td>
      <td align="center">❌</td>
      <td align="center">✅</td>
    </tr>
    <!-- LOC: tokei libs/deepagents/deepagents/ -t Python  ·  langchain-ai/deepagents @ 7465d77 -->
    <tr>
      <td align="left">
        <img src="https://cdn.jsdelivr.net/npm/simple-icons@latest/icons/langchain.svg" width="20" height="20" alt="">
        &nbsp;deepagents
      </td>
      <td align="right">15,369</td>
      <td align="center">✅</td>
      <td align="center">✅</td>
      <td align="center">❌</td>
      <td align="center">✅</td>
      <td align="center">✅</td>
      <td align="center">✅</td>
      <td align="center">✅</td>
      <td align="center">❌</td>
    </tr>
    <!-- LOC: tokei src/strands/ -t Python --exclude experimental --exclude vended_plugins --exclude multiagent/a2a  ·  strands-agents/sdk-python @ 1232230 -->
    <tr>
      <td align="left">
        <img src="https://cdn.jsdelivr.net/npm/simple-icons@latest/icons/amazonwebservices.svg" width="20" height="20" alt="">
        &nbsp;AWS Strands
      </td>
      <td align="right">25,494</td>
      <td align="center">✅</td>
      <td align="center">✅</td>
      <td align="center">✅</td>
      <td align="center">❌</td>
      <td align="center">✅</td>
      <td align="center">✅</td>
      <td align="center">❌</td>
      <td align="center">✅</td>
    </tr>
    <!-- LOC: tokei python/packages/core/agent_framework/ -t Python --exclude _evaluation.py --exclude a2a --exclude ag_ui --exclude chatkit --exclude declarative --exclude devui --exclude hyperlight --exclude lab --exclude orchestrations --exclude mem0 --exclude redis --exclude microsoft  ·  microsoft/agent-framework @ a60e541 -->
    <tr>
      <td align="left">
        <img src="https://cdn.jsdelivr.net/npm/simple-icons@latest/icons/microsoft.svg" width="20" height="20" alt="">
        &nbsp;Microsoft Agent Framework
      </td>
      <td align="right">34,751</td>
      <td align="center">✅</td>
      <td align="center">✅</td>
      <td align="center">✅</td>
      <td align="center">✅</td>
      <td align="center">✅</td>
      <td align="center">✅</td>
      <td align="center">❌</td>
      <td align="center">✅</td>
    </tr>
    <!-- LOC: tokei pydantic_ai_slim/pydantic_ai/ -t Python --exclude _a2a.py --exclude ag_ui.py --exclude ui --exclude durable_exec --exclude embeddings --exclude ext  ·  pydantic/pydantic-ai @ ac684b2 -->
    <tr>
      <td align="left">
        <img src="https://cdn.jsdelivr.net/npm/simple-icons@latest/icons/pydantic.svg" width="20" height="20" alt="">
        &nbsp;Pydantic AI
      </td>
      <td align="right">51,231</td>
      <td align="center">✅</td>
      <td align="center">❌</td>
      <td align="center">✅</td>
      <td align="center">❌</td>
      <td align="center">✅</td>
      <td align="center">✅</td>
      <td align="center">❌</td>
      <td align="center">✅</td>
    </tr>
    <!-- LOC: tokei src/google/adk/ -t Python --exclude a2a --exclude apps --exclude cli --exclude cloud --exclude code_executors --exclude environment --exclude evaluation --exclude examples --exclude integrations --exclude optimization --exclude platform  ·  google/adk-python @ bd062ec -->
    <tr>
      <td align="left">
        <img src="https://cdn.jsdelivr.net/npm/simple-icons@latest/icons/google.svg" width="20" height="20" alt="">
        &nbsp;Google ADK
      </td>
      <td align="right">57,392</td>
      <td align="center">✅</td>
      <td align="center">✅</td>
      <td align="center">✅</td>
      <td align="center">✅</td>
      <td align="center">✅</td>
      <td align="center">✅</td>
      <td align="center">❌</td>
      <td align="center">✅</td>
    </tr>
    <!-- LOC: tokei src/agents/ -t Python --exclude realtime --exclude voice --exclude extensions/experimental --exclude extensions/visualization.py  ·  openai/openai-agents-python @ 4bd459e -->
    <tr>
      <td align="left">
        <img src="https://cdn.jsdelivr.net/npm/simple-icons@latest/icons/openai.svg" width="20" height="20" alt="">
        &nbsp;OpenAI Agents SDK
      </td>
      <td align="right">72,410</td>
      <td align="center">✅</td>
      <td align="center">✅</td>
      <td align="center">✅</td>
      <td align="center">❌</td>
      <td align="center">✅</td>
      <td align="center">✅</td>
      <td align="center">❌</td>
      <td align="center">✅</td>
    </tr>
    <!-- LOC: tokei libs/agno/agno/{agent,agents,approval,compression,factory,guardrails,hooks,memory,models,reasoning,registry,run,session,skills,team,tools,tracing,utils} -t Python  ·  agno-agi/agno @ bb7ddb0 -->
    <tr>
      <td align="left">Agno</td>
      <td align="right">106,852</td>
      <td align="center">✅</td>
      <td align="center">✅</td>
      <td align="center">✅</td>
      <td align="center">✅</td>
      <td align="center">✅</td>
      <td align="center">✅</td>
      <td align="center">✅</td>
      <td align="center">✅</td>
    </tr>
  </tbody>
</table>

<sub>* shells out to the Claude Code CLI binary, which is 200k+ LOC</sub>

<sub>Out of scope: multi-agent orchestration frameworks like CrewAI, AutoGen, and AG2 follow a different paradigm (agent-to-agent conversation / role workflows) and don't map onto these columns. LlamaIndex centers on RAG and Semantic Kernel on broader AI orchestration — both have agent abstractions, but their scope is wider than what's compared here.</sub>

## Notes on the marks

- **Hooks** — lifecycle / tool-call interception primitive exposed by the framework. ThinHarness `Hook(...)`; Claude Agent SDK `hooks=` on `ClaudeSDKClient`; deepagents middleware (`permissions.py`, etc.); Pydantic AI `capabilities/hooks.py`; OpenAI Agents SDK `lifecycle.py` (`RunHooks`/`AgentHooks`); Google ADK `before_model_callbacks` etc.; Agno `@hook` decorator with explicit pre/post events; AWS Strands `hooks/` module with event registry; Microsoft Agent Framework `_middleware.py` (`AgentMiddleware`/`FunctionMiddleware`). smolagents only exposes step-level `step_callbacks` (no per-tool-call interception), marked ⚠️.
- **Subagents** — first-class delegation primitive. ThinHarness ships a `subagent` built-in tool and `SubAgentConfig`; Claude Agent SDK exposes Claude Code's Task tool; deepagents has `middleware/subagents.py`; OpenAI Agents SDK has `handoffs/`; smolagents has managed agents; ADK has multi-agent delegation; Agno has `team/` (Teams); AWS Strands has `multiagent/` (graph, swarm); Microsoft Agent Framework has `_workflows/` with agent executors. Pydantic AI documents an "agent delegation" *pattern* (call one agent inside another's tool function) but ships no class, decorator, or middleware for it — their own multi-agent docs point users to deepagents for that case, so it's marked ❌.
- **Structured output** — typed final outputs with a validation mechanism. ThinHarness `OutputSchema` with native/tool/prompted modes; Pydantic AI is the reference implementation; OpenAI Agents SDK `agent_output.py`; ADK Pydantic `output_schema`; Agno `response_model`; AWS Strands `tools/structured_output/`; Microsoft Agent Framework `response_format` plumbed through `_types.py`/`_tools.py`; smolagents uses `FinalAnswerTool` + callable `final_answer_checks`. Claude Agent SDK and deepagents return free-form messages with no built-in validation step.
- **Skills** — Markdown/frontmatter skill discovery and invocation. ThinHarness `skill_read` / `skill_run`; Claude Agent SDK via Claude Code skills; deepagents `middleware/skills.py`; Google ADK ships a `skills/` module with a `SkillRegistry` (marked experimental upstream, but the feature exists); Agno has a full `skills/` module with a Claude-Code-shaped `Skill` dataclass (SKILL.md frontmatter, scripts, references); Microsoft Agent Framework `_skills.py` implements the [agentskills.io](https://agentskills.io/) progressive-disclosure spec (`FileSkill`/`InlineSkill`/`ClassSkill`). Pydantic AI, OpenAI Agents SDK, smolagents, and AWS Strands have no skills primitive.
- **Multi-provider** — supports more than one model provider directly. Claude Agent SDK is Claude-only (it wraps the Claude Code CLI), so ❌.
- **MCP** — first-class Model Context Protocol client/server support. ThinHarness does not currently include MCP.
- **Built-in FS tools** — ships read/write/edit/search-style filesystem tools out of the box. Pydantic AI's slim core ships web/search/image-gen common tools but no filesystem set. OpenAI Agents SDK defines an `apply_patch` editor *protocol* but ships no FS implementation. Google ADK ships `bash_tool` (generic shell) and several hosted-search tools but no FS tool primitives. smolagents executes Python code in a sandbox but has no dedicated FS tool surface. Agno ships `tools/file.py` and `tools/local_file_system.py` for direct FS access. AWS Strands keeps the core SDK tool-free; FS tools live in a separate `strands-agents-tools` package and don't count here. Microsoft Agent Framework ships no FS tool primitives in its core package. Generic shell or code-exec tools don't count here — the column is specifically about typed read/write/edit/search-style primitives.
- **OTel tracing** — native OpenTelemetry spans for runs/model-calls/tool-calls. ThinHarness emits its own spans; Pydantic AI integrates with Logfire/OTel; OpenAI Agents SDK has a built-in tracing system exportable to OTel; smolagents `monitoring.py` + OpenTelemetry instrumentation; ADK ships a `telemetry/` module; Agno ships a `tracing/` module with an OTel exporter; AWS Strands ships a `telemetry/` module; Microsoft Agent Framework ships `_telemetry.py` with OTel instrumentation. Claude Agent SDK gets ⚠️: the Python SDK itself ships no instrumentation (only W3C traceparent propagation into the CLI subprocess), and there is an open issue ([#452](https://github.com/anthropics/claude-agent-sdk-python/issues/452)) asking for native SDK tracing — but the Claude Code CLI it shells out to does emit OTel spans (`claude_code.interaction`/`llm_request`/`tool`), with subagents documented to nest under parent `claude_code.tool` spans. The CLI's tracing is officially in beta, and real-world users report rough edges (broken nested traces in some cases, custom glue needed to integrate with MLflow). deepagents leans on LangSmith rather than emitting OTel from its own code.

## What "strict framework-only" excludes

Carving principle: keep everything you'd import to *use the library as an agent framework* — agent loop, hooks, tools, structured output, skills, subagents, memory, session, tracing, provider/model implementations, MCP. Strip what's clearly outside that scope: deployment layers, evals, UI/CLI tools, voice/realtime modalities, declarative wire protocols.

Per-library exclusions for the curious (full commands in the table source HTML comments):

- **ThinHarness, deepagents** — nothing stripped; already framework-only.
- **Claude Agent SDK** — `testing/` (user-facing test helpers).
- **smolagents** — `cli.py`, `gradio_ui.py`, `vision_web_browser.py` (CLI + UI).
- **AWS Strands** — `experimental/`, `vended_plugins/`, `multiagent/a2a/` (experimental APIs, opt-in plugins, A2A wire protocol).
- **Microsoft Agent Framework** — `_evaluation.py`, `a2a/`, `ag_ui/`, `chatkit/`, `declarative/`, `devui/`, `hyperlight/`, `lab/`, `orchestrations/`, `mem0/`, `redis/`, `microsoft/` (evals, wire protocols, UI, runtime backends, storage backends, lab/experimental).
- **Pydantic AI** — `_a2a.py`, `ag_ui.py`, `ui/`, `durable_exec/`, `embeddings/`, `ext/` (A2A, UI, durable execution runtime, embedding models, ext).
- **Google ADK** — `a2a/`, `apps/`, `cli/`, `cloud/`, `code_executors/`, `environment/`, `evaluation/`, `examples/`, `integrations/`, `optimization/`, `platform/`.
- **OpenAI Agents SDK** — `realtime/`, `voice/`, `extensions/experimental`, `extensions/visualization.py`.
- **Agno** — `api/`, `client/`, `cloud/`, `db/`, `integrations/`, `knowledge/`, `learn/`, `os/`, `remote/`, `scheduler/`, `vectordb/`, `context/`, `culture/`, plus boundary cases `workflow/` and `eval/`. (Agno bundles a full platform layer alongside its agent framework; as-shipped is 254,377 LOC.)
