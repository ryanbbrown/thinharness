"""SDK-only provider-agnostic agent loop."""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .defaults import DEFAULT_SYSTEM_PROMPT
from .hooks import (
    Hook,
    HookRegistry,
    RunStartContext,
    UserPromptSubmitContext,
    apply_prompt_context,
)
from .output import (
    FINAL_RESULT_TOOL_NAME,
    OutputMode,
    OutputSchema,
    OutputSpec,
    resolve_output_schema_for_model,
    structured_instructions,
)
from .providers import (
    Model,
    ModelCapabilities,
    ModelNotice,
    ModelSession,
    ProviderError,
    ResumableModel,
    StructuredOutputRequest,
    ToolOutput,
    infer_model,
)
from .subagents import DEFAULT_SUBAGENT_NAME, SubAgentConfig, create_subagent_tool
from .tools.base import Json, ToolSpec
from .tools.filesystem import builtin_tools as make_builtin_tools
from .tools.mcp import MCPServer
from .tools.parallel_llm import create_parallel_llm_tool
from .tools.skills import SkillRegistry
from .tracing import (
    LocalTracing,
    ModelTraceSnapshot,
    RunTracer,
    TracingOptions,
    _trace_output_mode,
    annotate_agent_start,
    create_local_tracing,
)

DEFAULT_BUILTIN_TOOLS = {"read", "write", "edit", "search", "list", "glob"}
StopReason = Literal[
    "end_turn",
    "provider_error",
    "limit_reached",
    "error",
    "cancelled_by_hook",
    "cancelled",
    "output_validation_failed",
    "tool_retries_exceeded",
    "unexpected_model_behavior",
]
LimitNoticeKey = tuple[Literal["limit_warning"], Literal["model_requests", "tool_calls"], int]


def _limit_notice_dedup_key(notice: ModelNotice) -> LimitNoticeKey:
    """Return the once-per-run key for a model notice."""
    assert notice.limit_kind is not None and notice.remaining is not None
    return (notice.kind, notice.limit_kind, notice.remaining)


def _append_notice_once(notices: list[ModelNotice], emitted: set[LimitNoticeKey], notice: ModelNotice) -> None:
    """Append a notice once per run."""
    key = _limit_notice_dedup_key(notice)
    if key in emitted:
        return
    notices.append(notice)
    emitted.add(key)


def _compute_limit_notices(
    config: HarnessConfig,
    usage: RunUsage,
    emitted: set[LimitNoticeKey],
    *,
    final_result_tool_available: bool,
) -> list[ModelNotice]:
    """Return model-facing warnings for the current run budget state."""
    notices: list[ModelNotice] = []
    final_model_text = (
        "Final request: produce the answer now with final_result."
        if final_result_tool_available
        else "Final request: produce the answer now; do not request tools."
    )
    remaining_model_requests = config.max_model_requests - usage.model_requests
    if remaining_model_requests == 1:
        _append_notice_once(notices, emitted, ModelNotice(
            kind="limit_warning",
            content=final_model_text,
            limit_kind="model_requests",
            remaining=1,
        ))

    if config.max_tool_calls is None:
        return notices
    remaining_tool_calls = config.max_tool_calls - usage.tool_calls
    if remaining_tool_calls == 0:
        no_tools_text = (
            "Tool calls are not available on this run; produce the answer with final_result."
            if final_result_tool_available and config.max_tool_calls == 0
            else "No tool calls remain: produce the answer with final_result."
            if final_result_tool_available
            else "Tool calls are not available on this run; answer without tools."
            if config.max_tool_calls == 0
            else "No tool calls remain: answer now without tools."
        )
        _append_notice_once(notices, emitted, ModelNotice(
            kind="limit_warning",
            content=no_tools_text,
            limit_kind="tool_calls",
            remaining=0,
        ))
    elif remaining_tool_calls == 1:
        tool_phrase = "tool call remains besides final_result" if final_result_tool_available else "tool call remains"
        _append_notice_once(notices, emitted, ModelNotice(
            kind="limit_warning",
            content=f"One {tool_phrase}: avoid fan-out.",
            limit_kind="tool_calls",
            remaining=1,
        ))
    return notices


def _build_resume_state(
    session: ModelSession,
    stop_reason: StopReason,
    finalized_via_output_tool: bool,
    require_dump_state: bool,
) -> dict[str, Any] | None:
    """Apply resume lifecycle rules and return an isolated JSON copy."""
    if stop_reason != "end_turn" or finalized_via_output_tool:
        return None
    dump_state = getattr(session, "dump_state", None)
    if dump_state is None:
        # Non-resumable custom models may omit dump_state; resumable models must provide it.
        if require_dump_state:
            raise HarnessError("resumable model session is missing dump_state()")
        return None
    state = dump_state()
    if state is None:
        return None
    return json.loads(json.dumps(state))


def _local_tracing_enabled(configured: bool) -> bool:
    """Return whether local plaintext tracing should be active."""
    disabled = os.getenv("THINHARNESS_DISABLE_LOCAL_TRACING", "").lower() in {"1", "true", "yes"}
    return configured and not disabled


class HarnessConfig(BaseModel):
    """Configuration for Harness."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: str = "openai:gpt-5.2"
    root: str | Path = "."
    api_key: str | None = None
    base_url: str | None = None
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    skills_dir: str | Path | list[str | Path] | None = None
    selected_skills: list[str] | None = None
    builtin_tools: list[str] | None = None
    output_dir: str | Path | None = None
    max_model_requests: int = 64
    max_tool_calls: int | None = None
    strict_hooks: bool = False
    request_timeout: int = 120
    max_read_chars: int = 40_000
    max_read_bytes: int = 1_000_000
    max_tool_chars: int = 40_000
    max_search_line_chars: int = 180
    rg_timeout: int = 30
    search_exclude_globs: list[str] = Field(default_factory=list)
    read_paths: list[str | Path] | None = None
    write_paths: list[str | Path] | None = None
    temperature: float | None = None
    extra_body: dict[str, Any] = Field(default_factory=dict)
    tracing: list[TracingOptions] = Field(default_factory=list)
    local_tracing: bool = True
    local_trace_dir: str | Path = "~/.thinharness/traces"
    tool_execution: Literal["auto", "sequential"] = "auto"
    subagents: list[SubAgentConfig] = Field(default_factory=list)
    output_type: OutputSpec | None = None
    output_mode: OutputMode = "auto"
    output_retries: int = Field(default=1, ge=0)
    tool_retries: int = Field(default=1, ge=0)
    builtin_parallel_llm_model: str | None = None
    builtin_parallel_llm_temperature: float | None = None
    parallel_llm_max_prompts: int = Field(default=100, ge=1)
    parallel_llm_max_attempts: int = Field(default=4, ge=1, le=10)
    mcp_servers: list[MCPServer] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_skills(self) -> HarnessConfig:
        """Validate explicit skill discovery settings."""
        if self.selected_skills is not None and self.skills_dir is None:
            raise ValueError("selected_skills requires skills_dir")
        return self


@dataclass
class HarnessResult:
    """Final result returned by a harness run."""

    text: str
    output: Any | None = None
    responses: list[Json] = field(default_factory=list)
    tool_call_records: list[Json] = field(default_factory=list)
    usage: RunUsage = field(default_factory=lambda: RunUsage())
    stop_reason: StopReason = "end_turn"
    resume_state: dict[str, Any] | None = None


@dataclass
class RunUsage:
    """Provider and tool usage for one harness run."""

    model_requests: int = 0
    tool_calls: int = 0
    cancelled_tool_calls: int = 0
    output_retries: int = 0
    tool_retries: dict[str, int] = field(default_factory=dict)


class HarnessError(RuntimeError):
    """Raised when the harness cannot complete a run."""


class UnexpectedModelBehavior(HarnessError):
    """Raised when the model returns an invalid tool/finalization pattern."""


class Harness:
    """A non-interactive filesystem agent harness for SDK use."""

    def __init__(
        self,
        config: HarnessConfig | None = None,
        *,
        model: Model | None = None,
        tools: list[ToolSpec] | None = None,
        tracing: list[TracingOptions] | None = None,
        skills: SkillRegistry | None = None,
        hooks: list[Hook] | HookRegistry | None = None,
        subagent_hooks: dict[str, list[Hook] | HookRegistry] | None = None,
        _owns_model: bool | None = None,
        _is_child_run: bool = False,
    ) -> None:
        self.config = config or HarnessConfig()
        if skills is not None and (self.config.skills_dir is not None or self.config.selected_skills is not None):
            raise ValueError("skills cannot be combined with skills_dir or selected_skills")
        self.root = Path(self.config.root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.model_ref = os.getenv("HARNESS_MODEL", self.config.model)
        self.model = model or infer_model(
            self.model_ref,
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            timeout=self.config.request_timeout,
            temperature=self.config.temperature,
            extra_body=self.config.extra_body,
        )
        self._owns_model = _owns_model if _owns_model is not None else model is None
        self.model_capabilities = getattr(self.model, "capabilities", ModelCapabilities())
        self.skills = skills or SkillRegistry(self.config.skills_dir, selected_skills=self.config.selected_skills)
        filesystem_tools = make_builtin_tools(
            self.root,
            output_dir=self.config.output_dir,
            max_read_chars=self.config.max_read_chars,
            max_read_bytes=self.config.max_read_bytes,
            max_tool_chars=self.config.max_tool_chars,
            max_search_line_chars=self.config.max_search_line_chars,
            rg_timeout=self.config.rg_timeout,
            search_exclude_globs=self.config.search_exclude_globs,
            read_paths=self.config.read_paths,
            write_paths=self.config.write_paths,
        )
        builtin_candidates = [
            *filesystem_tools,
            *self.skills.specs(),
            create_subagent_tool(self, self.config.subagents),
            create_parallel_llm_tool(self),
        ]
        builtin = self._select_builtin_tools(builtin_candidates, self.config.builtin_tools)
        self.tools: list[ToolSpec] = builtin
        self._validate_unique_tools(self.tools)
        self._tool_map = {tool.name: tool for tool in self.tools}
        raw_hooks = hooks
        self.hooks = HookRegistry([], strict_hooks=self.config.strict_hooks)
        self.subagent_hooks = subagent_hooks or {}
        for tool in tools or []:
            self.add_tool(tool)
        self.output_schema = self._build_output_schema()
        self._validate_final_result_collision()
        self._mcp_servers = list(self.config.mcp_servers)
        self._resolve_mcp_server_ids()
        self._mcp_stack: AsyncExitStack | None = None
        self._mcp_connected = False
        self.hooks = raw_hooks if isinstance(raw_hooks, HookRegistry) else HookRegistry(raw_hooks, strict_hooks=self.config.strict_hooks)
        self._validate_skill_tool_selection()
        self._skills_enabled = bool(self.skills.skills) and any(tool.name in {"skill_read", "skill_run"} for tool in self.tools)
        self.local_tracing: LocalTracing | None = None
        external_tracing = list(self.config.tracing if tracing is None else tracing)
        if _local_tracing_enabled(self.config.local_tracing) and not _is_child_run:
            self.local_tracing = create_local_tracing(self.config.local_trace_dir, project_root=self.root)
            self.tracing = [
                TracingOptions(
                    tracer=self.local_tracing.tracer,
                    capture_messages=True,
                    capture_tool_args=True,
                    capture_tool_results=True,
                ),
                *external_tracing,
            ]
        else:
            self.tracing = external_tracing
        self._is_child_run = _is_child_run
        self._running = False
        self._closed = False
        self._validate_hook_filters()

    async def run(self, prompt: str, *, resume_from: dict[str, Any] | None = None, metadata: Json | None = None) -> HarnessResult:
        """Run one prompt to completion."""
        from .runtime import RunContext
        from .tool_execution import ToolBatchExecutor

        if self._closed:
            raise HarnessError("harness is closed")
        if self._running:
            raise HarnessError("Harness.run is not re-entrant")

        session: ModelSession | None
        model_supports_resume = hasattr(self.model, "resume_kind") and hasattr(self.model, "resume_session")
        if resume_from is None:
            session = None
            first_turn_kind = "start"
        else:
            if not model_supports_resume:
                raise HarnessError(f"model {type(self.model).__name__} does not support resume")
            session = cast(ResumableModel, self.model).resume_session(resume_from)
            first_turn_kind = "resume"

        run_tracer = RunTracer(self.tracing)
        run_metadata = dict(metadata or {})
        run_ctx = RunContext(
            harness=self,
            prompt=prompt,
            metadata=run_metadata,
            usage=RunUsage(),
            tracer=run_tracer,
        )
        self._running = True

        try:
            try:
                conversation_id = str(metadata.get("conversation_id")) if metadata and metadata.get("conversation_id") else None
                with run_tracer.agent(conversation_id=conversation_id) as agent_span:
                    run_ctx.agent_span = agent_span
                    tool_executor = ToolBatchExecutor(
                        harness=self,
                        run_context=run_ctx,
                        tool_map=self._tool_map,
                        run_tracer=run_tracer,
                        tool_execution=self.config.tool_execution,
                    )
                    try:
                        self.hooks.fire(RunStartContext(
                            harness=self,
                            metadata=dict(run_metadata),
                            prompt=prompt,
                            root=self.root,
                            max_model_requests=self.config.max_model_requests,
                            max_tool_calls=self.config.max_tool_calls,
                        ))
                        await self._ensure_mcp_connected()
                        prompt_ctx = UserPromptSubmitContext(harness=self, metadata=dict(run_metadata), prompt=prompt)
                        self.hooks.fire(prompt_ctx)
                        if prompt_ctx.cancelled:
                            reason = prompt_ctx.cancel_reason or "unspecified"
                            run_ctx.stop_reason = "cancelled_by_hook"
                            run_ctx.terminal_error = HarnessError(f"run blocked by hook: {reason}")
                            raise run_ctx.terminal_error
                        effective_prompt = apply_prompt_context(prompt, prompt_ctx.additional_context)
                        instructions = structured_instructions(self.system_instructions(), self.output_schema)
                        agent_span.for_each(
                            lambda span, option: annotate_agent_start(
                                span,
                                prompt=prompt,
                                instructions=instructions,
                                capture_messages=option.capture_messages,
                                top_level=not self._is_child_run,
                            )
                        )
                        structured_output = self._structured_output_request()
                        output_mode = _trace_output_mode(self.output_schema)
                        if first_turn_kind == "start":
                            try:
                                active_session = self.model.new_session()
                            except Exception as exc:
                                run_ctx.stop_reason = "error"
                                run_ctx.terminal_error = exc
                                agent_span.record_exception(exc)
                                agent_span.set_error(str(exc), type(exc).__name__)
                                raise
                            turn, decision = await run_ctx.advance_model(lambda notices: active_session.start(
                                prompt=effective_prompt,
                                instructions=instructions,
                                tools=self.tool_schemas(),
                                metadata=metadata,
                                structured_output=structured_output,
                                notices=notices,
                            ), trace_snapshot=ModelTraceSnapshot(kind="start", prompt=effective_prompt, structured_output=output_mode))
                        else:
                            assert session is not None
                            active_session = session
                            turn, decision = await run_ctx.advance_model(lambda notices: active_session.continue_with_user_prompt(
                                prompt=effective_prompt,
                                instructions=instructions,
                                tools=self.tool_schemas(),
                                metadata=metadata,
                                structured_output=structured_output,
                                notices=notices,
                            ), trace_snapshot=ModelTraceSnapshot(kind="resume", prompt=effective_prompt, structured_output=output_mode))
                        while True:
                            run_ctx.responses.append(turn.raw)
                            if decision.kind == "final":
                                return run_ctx.finalize(
                                    decision.text,
                                    active_session,
                                    output=decision.output,
                                    finalized_via_output_tool_value=decision.finalized_via_output_tool,
                                    require_dump_state=model_supports_resume,
                                )
                            if decision.kind == "retry_tool_output":
                                run_ctx.retry_or_fail()
                                final_id = decision.retry_call_id
                                assert final_id, "tool-mode final_result retry requires a tool call id"
                                retry_message = decision.retry_message
                                turn, decision = await run_ctx.advance_model(
                                    lambda notices, final_id=final_id, retry_message=retry_message: active_session.continue_with_tools(
                                        [ToolOutput(final_id, retry_message)],
                                        instructions=instructions,
                                        tools=self.tool_schemas(),
                                        metadata=metadata,
                                        structured_output=structured_output,
                                        notices=notices,
                                    ),
                                    trace_snapshot=ModelTraceSnapshot(
                                        kind="output_retry_tool",
                                        tool_outputs=[{"call_id": final_id, "output": retry_message}],
                                        structured_output=output_mode,
                                    ),
                                    output_retry=True,
                                )
                                continue
                            if decision.kind == "retry_user_message":
                                run_ctx.retry_or_fail()
                                retry_message = decision.retry_message
                                turn, decision = await run_ctx.advance_model(
                                    lambda notices, retry_message=retry_message: active_session.continue_with_user_message(
                                        retry_message,
                                        instructions=instructions,
                                        tools=self.tool_schemas(),
                                        metadata=metadata,
                                        structured_output=structured_output,
                                        notices=notices,
                                    ),
                                    trace_snapshot=ModelTraceSnapshot(
                                        kind="correction",
                                        prompt=retry_message,
                                        structured_output=output_mode,
                                    ),
                                    output_retry=True,
                                )
                                continue
                            if decision.kind == "unexpected":
                                raise UnexpectedModelBehavior(decision.unexpected_message)
                            run_ctx.check_tool_limit(len(turn.tool_calls))
                            run_ctx.usage.tool_calls += len(turn.tool_calls)
                            recorded, outputs, executions = await tool_executor.execute_batch(turn.tool_calls)
                            run_ctx.usage.cancelled_tool_calls += sum(1 for execution in executions if execution.cancelled)
                            run_ctx.record_tool_batch(recorded)
                            run_ctx.check_tool_retry_limits(turn.tool_calls, executions)
                            tool_outputs = outputs
                            turn, decision = await run_ctx.advance_model(lambda notices, tool_outputs=tool_outputs: active_session.continue_with_tools(
                                tool_outputs,
                                instructions=instructions,
                                tools=self.tool_schemas(),
                                metadata=metadata,
                                structured_output=structured_output,
                                notices=notices,
                            ), trace_snapshot=ModelTraceSnapshot(
                                kind="tool_outputs",
                                tool_outputs=[{"call_id": item.call_id, "output": item.output} for item in tool_outputs],
                                structured_output=output_mode,
                            ))
                    except asyncio.CancelledError as exc:
                        run_ctx.stop_reason = "cancelled"
                        run_ctx.terminal_error = exc
                        agent_span.record_exception(exc)
                        agent_span.set_error("run cancelled", "CancelledError")
                        raise
                    except ProviderError as exc:
                        run_ctx.stop_reason = "provider_error"
                        run_ctx.terminal_error = HarnessError(str(exc))
                        agent_span.record_exception(exc)
                        agent_span.set_error(str(exc), type(exc).__name__)
                        raise run_ctx.terminal_error from exc
                    except UnexpectedModelBehavior as exc:
                        run_ctx.stop_reason = "unexpected_model_behavior"
                        run_ctx.terminal_error = run_ctx.terminal_error or exc
                        agent_span.record_exception(exc)
                        agent_span.set_error(str(exc), type(exc).__name__)
                        raise
                    except HarnessError as exc:
                        run_ctx.terminal_error = run_ctx.terminal_error or exc
                        if run_ctx.stop_reason == "end_turn":
                            run_ctx.stop_reason = "error"
                        agent_span.record_exception(exc)
                        agent_span.set_error(str(exc), type(exc).__name__)
                        raise
                    except Exception as exc:
                        run_ctx.stop_reason = "error"
                        run_ctx.terminal_error = exc
                        agent_span.record_exception(exc)
                        agent_span.set_error(str(exc), type(exc).__name__)
                        raise
            finally:
                run_ctx.fire_run_end_once()
        finally:
            self._running = False

    def run_sync(self, prompt: str, *, resume_from: dict[str, Any] | None = None, metadata: Json | None = None) -> HarnessResult:
        """Synchronous wrapper around run."""
        if self._running:
            raise HarnessError("Harness.run is not re-entrant")
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise HarnessError("run_sync cannot be called from inside a running event loop; await run() instead")

        async def _run_and_close() -> HarnessResult:
            """Run and close owned async resources in the same loop."""
            try:
                return await self.run(prompt, resume_from=resume_from, metadata=metadata)
            finally:
                await self.aclose()

        return asyncio.run(_run_and_close())

    async def aclose(self) -> None:
        """Close MCP servers and owned provider HTTP clients."""
        if self._closed:
            return
        try:
            if self._mcp_stack is not None:
                await self._mcp_stack.aclose()
                self._mcp_stack = None
                self._mcp_connected = False
            if self._owns_model:
                aclose = getattr(self.model.provider, "aclose", None)
                if aclose is not None:
                    await aclose()
        finally:
            self._closed = True

    async def __aenter__(self) -> Harness:
        """Enter an async harness lifecycle."""
        return self

    async def __aexit__(self, *exc: object) -> None:
        """Close owned async resources when leaving a lifecycle."""
        await self.aclose()

    def add_tool(self, tool: ToolSpec) -> None:
        """Register a custom tool using a ToolSpec."""
        spec = tool
        if not callable(spec.handler):
            raise TypeError(f"handler for tool {spec.name!r} is not callable")
        if spec.name == "subagent" and spec.metadata.get("framework_tool") != "subagent":
            raise ValueError("subagent is a reserved tool name")
        output_schema = getattr(self, "output_schema", None)
        if (
            spec.name == FINAL_RESULT_TOOL_NAME
            and output_schema is not None
            and output_schema.mode != "text"
        ):
            raise ValueError(f"{FINAL_RESULT_TOOL_NAME} is reserved for structured output")
        if spec.name in self._tool_map:
            raise ValueError(f"duplicate tool name: {spec.name}")
        self.tools.append(spec)
        self._tool_map[spec.name] = spec
        self._validate_hook_filters()

    def tool_schemas(self) -> list[Json]:
        """Return normalized Responses-style tool definitions."""
        tools = [tool.response_tool() for tool in self.tools]
        if self.output_schema is not None:
            tools.extend(self.output_schema.synthetic_tools())
        return tools

    def system_instructions(self) -> str:
        """Return the full instruction text sent to the model."""
        parts = [self.config.system_prompt, f"Workspace root: {self.root}"]
        if self._skills_enabled:
            skill_summary = self.skills.prompt_summary()
            if skill_summary:
                parts.append(skill_summary)
        tool_instructions = []
        for tool in self.tools:
            if tool.instructions is None:
                continue
            instructions = tool.instructions.strip()
            if instructions:
                tool_instructions.append(instructions)
        parts.extend(tool_instructions)
        return "\n\n".join(parts)

    def _tool_max_retries(self, name: str) -> int:
        """Return the retry budget for one tool name."""
        spec = self._tool_map.get(str(name))
        if spec is not None and spec.max_retries is not None:
            return spec.max_retries
        return self.config.tool_retries

    @staticmethod
    def _validate_unique_tools(tools: list[ToolSpec]) -> None:
        """Reject duplicate tool names before sending schemas to a provider."""
        seen: set[str] = set()
        for tool in tools:
            if tool.name in seen:
                raise ValueError(f"duplicate tool name: {tool.name}")
            seen.add(tool.name)

    def _validate_skill_tool_selection(self) -> None:
        """Require explicit skill tool selection when skills are explicitly configured."""
        if not self.skills.skills:
            return
        tool_names = {tool.name for tool in self.tools}
        if not tool_names.intersection({"skill_read", "skill_run"}):
            raise ValueError("configured skills require exposing skill_read or skill_run")

    def _validate_hook_filters(self) -> None:
        """Validate hook filters against registered subagents."""
        agent_names = {DEFAULT_SUBAGENT_NAME, *(config.name for config in self.config.subagents)}
        self.hooks.validate_filters(agent_names=agent_names)

    def _resolve_mcp_server_ids(self) -> None:
        """Assign stable suffixes to duplicate MCP server ids."""
        counts: dict[str, int] = {}
        for server in self._mcp_servers:
            base_id = server.id
            counts[base_id] = counts.get(base_id, 0) + 1
            resolved = base_id if counts[base_id] == 1 else f"{base_id}-{counts[base_id]}"
            server._resolved_id = resolved

    async def connect(self) -> None:
        """Open MCP server connections and discover their tools."""
        if self._closed:
            raise HarnessError("harness is closed")
        await self._ensure_mcp_connected()

    async def _ensure_mcp_connected(self) -> None:
        """Connect MCP servers and append their discovered tools once."""
        if self._mcp_connected:
            return
        if not self._mcp_servers:
            self._mcp_connected = True
            return
        stack = AsyncExitStack()
        try:
            mcp_tools: list[ToolSpec] = []
            seen = set(self._tool_map)
            if self.output_schema is not None and self.output_schema.mode == "tool":
                seen.add(FINAL_RESULT_TOOL_NAME)
            for server in self._mcp_servers:
                await stack.enter_async_context(server)
                for tool in await server.list_tools():
                    if tool.name in seen:
                        raise HarnessError(
                            f"MCP tool name collision for {tool.name!r}; use tool_prefix or exclude_tools to disambiguate"
                        )
                    seen.add(tool.name)
                    mcp_tools.append(tool)
            self.tools.extend(mcp_tools)
            self._tool_map.update({tool.name: tool for tool in mcp_tools})
            self._mcp_stack = stack
            self._mcp_connected = True
        except BaseException:
            await stack.aclose()
            raise

    def _build_output_schema(self) -> OutputSchema | None:
        """Build structured-output validation if configured."""
        return resolve_output_schema_for_model(self.model, self.config.output_type, self.config.output_mode)

    def _validate_final_result_collision(self) -> None:
        """Reserve final_result for synthetic structured output."""
        if self.output_schema is None or self.output_schema.mode == "text":
            return
        if FINAL_RESULT_TOOL_NAME in self._tool_map:
            raise ValueError(f"{FINAL_RESULT_TOOL_NAME} is reserved for structured output")

    def _structured_output_request(self) -> StructuredOutputRequest | None:
        """Return native structured-output request metadata."""
        if self.output_schema is None:
            return None
        return self.output_schema.structured_output_request()

    @staticmethod
    def _select_builtin_tools(tools: list[ToolSpec], selected_names: list[str] | None) -> list[ToolSpec]:
        """Return all or the explicitly selected built-in tools."""
        by_name = {tool.name: tool for tool in tools}
        if selected_names is None:
            return [tool for tool in tools if tool.name in DEFAULT_BUILTIN_TOOLS]
        selected: list[ToolSpec] = []
        seen: set[str] = set()
        for name in selected_names:
            if name in seen:
                raise ValueError(f"duplicate selected builtin tool: {name}")
            if name not in by_name:
                available = ", ".join(sorted(by_name)) or "none"
                raise ValueError(f"unknown builtin tool: {name}; available: {available}")
            selected.append(by_name[name])
            seen.add(name)
        return selected
