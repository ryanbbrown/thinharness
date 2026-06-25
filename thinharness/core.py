"""SDK-only provider-agnostic agent loop."""

from __future__ import annotations

import asyncio
import os
import time
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .approvals import (
    ApprovalPause,
    copy_restored_run_state,
    is_approval_pause_state,
    validate_approval_decisions,
    validate_approval_pause_state,
)
from .defaults import DEFAULT_SYSTEM_PROMPT
from .events import (
    ApprovalResumedEvent,
    HarnessStream,
    RunCompletedEvent,
    RunFailedEvent,
    RunStartedEvent,
    StreamEmitter,
    StreamOptions,
    ToolCallCompletedEvent,
    ToolCallStartedEvent,
    create_stream_context,
)
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
    OutputTurnDecision,
    resolve_output_schema_for_model,
    structured_instructions,
)
from .providers import (
    Model,
    ModelSession,
    ModelToolCall,
    ModelTurn,
    ProviderError,
    ResumableModel,
    StructuredOutputRequest,
    ToolOutput,
    infer_model,
    model_capabilities,
)
from .subagents import DEFAULT_SUBAGENT_NAME, SubAgentConfig, create_subagent_tool
from .tools.base import ToolResult, ToolSpec
from .tools.filesystem import builtin_tools as make_builtin_tools
from .tools.mcp import MCPServer
from .tools.parallel_llm import create_parallel_llm_tool
from .tools.skills import SkillRegistry
from .tracing import (
    LocalTracing,
    RunTracer,
    TracingOptions,
    annotate_agent_start,
    create_local_tracing,
    serialize_attribute_value,
)
from .types import ApprovalDecision, HarnessError, HarnessResult, Json, PendingApproval, RunUsage, UnexpectedModelBehavior

DEFAULT_BUILTIN_TOOLS = {"read", "write", "edit", "search", "list", "glob"}


def _local_tracing_enabled(configured: bool) -> bool:
    """Return whether local plaintext tracing should be active."""
    disabled = os.getenv("THINHARNESS_DISABLE_LOCAL_TRACING", "").lower() in {"1", "true", "yes"}
    return configured and not disabled


def _classify_run_failure(run_ctx: Any, agent_span: Any, exc: Exception) -> Exception:
    """Record a run failure and return the exception to raise."""
    agent_span.record_exception(exc)
    agent_span.set_error(str(exc), type(exc).__name__)
    if isinstance(exc, ProviderError):
        run_ctx.stop_reason = "provider_error"
        run_ctx.terminal_error = HarnessError(str(exc))
        return run_ctx.terminal_error
    if isinstance(exc, UnexpectedModelBehavior):
        run_ctx.stop_reason = "unexpected_model_behavior"
        run_ctx.terminal_error = run_ctx.terminal_error or exc
        return exc
    if isinstance(exc, HarnessError):
        run_ctx.terminal_error = run_ctx.terminal_error or exc
        if run_ctx.stop_reason == "end_turn":
            run_ctx.stop_reason = "error"
        return exc
    run_ctx.stop_reason = "error"
    run_ctx.terminal_error = exc
    return exc


class HarnessConfig(BaseModel):
    """Configuration for Harness."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: str = "openai:gpt-5.5"
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
    def validate_config(self) -> HarnessConfig:
        """Validate cross-field configuration settings."""
        if self.selected_skills is not None and self.skills_dir is None:
            raise ValueError("selected_skills requires skills_dir")
        return self


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
        self._is_child_run = _is_child_run
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
        self.model_capabilities = model_capabilities(self.model)
        self.skills = skills or SkillRegistry(self.config.skills_dir, selected_skills=self.config.selected_skills)
        output_schema = resolve_output_schema_for_model(self.model, self.config.output_type, self.config.output_mode)
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
        configured_tools = [*builtin, *(tools or [])]
        self._validate_tool_list(
            configured_tools,
            output_schema=output_schema,
            model_supports_approval_resume=self._model_supports_approval_resume(),
            is_child_run=self._is_child_run,
        )
        tool_map = {tool.name: tool for tool in configured_tools}
        hook_registry = hooks if isinstance(hooks, HookRegistry) else HookRegistry(hooks, strict_hooks=self.config.strict_hooks)
        self._validate_hook_registry(hook_registry, self.config.subagents)
        self._validate_skill_tool_selection_for(self.skills, configured_tools)

        self.tools = configured_tools
        self._tool_map = tool_map
        self.output_schema = output_schema
        self.hooks = hook_registry
        self.subagent_hooks = subagent_hooks or {}
        self._mcp_servers = list(self.config.mcp_servers)
        self._resolve_mcp_server_ids()
        self._mcp_stack: AsyncExitStack | None = None
        self._mcp_connected = False
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
        self._running = False
        self._closed = False

    async def run(self, prompt: str, *, resume_from: dict[str, Any] | None = None, metadata: Json | None = None) -> HarnessResult:
        """Run one prompt to completion."""
        result: HarnessResult | None = None
        stream = self.stream(prompt, resume_from=resume_from, metadata=metadata)
        async with stream as events:
            async for event in events:
                if isinstance(event, RunCompletedEvent) and event.run_id == stream.run_id:
                    result = event.result
        if result is None:
            raise HarnessError("stream ended without a result")
        return result

    async def resume_approvals(
        self,
        state: dict[str, Any],
        decisions: list[ApprovalDecision],
        *,
        metadata: Json | None = None,
    ) -> HarnessResult:
        """Resume a paused approval run with host decisions."""
        result: HarnessResult | None = None
        stream = self.stream_approvals(state, decisions, metadata=metadata)
        async with stream as events:
            async for event in events:
                if isinstance(event, RunCompletedEvent) and event.run_id == stream.run_id:
                    result = event.result
        if result is None:
            raise HarnessError("stream ended without a result")
        return result

    def resume_approvals_sync(
        self,
        state: dict[str, Any],
        decisions: list[ApprovalDecision],
        *,
        metadata: Json | None = None,
    ) -> HarnessResult:
        """Synchronous wrapper around resume_approvals."""
        if self._running:
            raise HarnessError("Harness.run is not re-entrant")
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise HarnessError("resume_approvals_sync cannot be called from inside a running event loop; await resume_approvals() instead")

        async def _run_and_close() -> HarnessResult:
            """Resume and close owned async resources in the same loop."""
            try:
                return await self.resume_approvals(state, decisions, metadata=metadata)
            finally:
                await self.aclose()

        return asyncio.run(_run_and_close())

    def stream(
        self,
        prompt: str,
        *,
        resume_from: dict[str, Any] | None = None,
        metadata: Json | None = None,
        stream_options: StreamOptions | None = None,
        _parent_run_id: str | None = None,
        _parent_tool_call_id: str | None = None,
        _agent_name: str | None = None,
    ) -> HarnessStream:
        """Stream coarse run lifecycle events for one prompt."""
        if self._closed:
            raise HarnessError("harness is closed")
        if self._running:
            raise HarnessError("Harness is not re-entrant")
        if is_approval_pause_state(resume_from):
            raise HarnessError("approval pause state must be resumed with resume_approvals()")
        stream_context = create_stream_context(
            parent_run_id=_parent_run_id,
            parent_tool_call_id=_parent_tool_call_id,
            agent_name=_agent_name,
            options=stream_options,
        )
        emitter = StreamEmitter(stream_context)
        loop = asyncio.get_running_loop()
        task = loop.create_task(self._run_streaming(
            prompt,
            resume_from=resume_from,
            approval_state=None,
            approval_decisions=None,
            metadata=metadata,
            emitter=emitter,
            stream_context=stream_context,
        ))
        self._running = True
        return HarnessStream(task, emitter)

    def stream_approvals(
        self,
        state: dict[str, Any],
        decisions: list[ApprovalDecision],
        *,
        metadata: Json | None = None,
        stream_options: StreamOptions | None = None,
    ) -> HarnessStream:
        """Stream an approval-pause resume."""
        if self._closed:
            raise HarnessError("harness is closed")
        if self._running:
            raise HarnessError("Harness is not re-entrant")
        stream_context = create_stream_context(options=stream_options)
        emitter = StreamEmitter(stream_context)
        loop = asyncio.get_running_loop()
        task = loop.create_task(self._run_streaming(
            "",
            resume_from=None,
            approval_state=state,
            approval_decisions=decisions,
            metadata=metadata,
            emitter=emitter,
            stream_context=stream_context,
        ))
        self._running = True
        return HarnessStream(task, emitter)

    async def _run_streaming(
        self,
        prompt: str,
        *,
        resume_from: dict[str, Any] | None,
        approval_state: dict[str, Any] | None,
        approval_decisions: list[ApprovalDecision] | None,
        metadata: Json | None,
        emitter: StreamEmitter,
        stream_context: Any,
    ) -> HarnessResult:
        """Run one prompt while emitting stream events."""
        from .runtime import RunContext
        from .tool_execution import ToolBatchExecutor

        try:
            run_tracer = RunTracer(self.tracing)
            approval_pause: ApprovalPause | None = None
            approval_decision_map: dict[str, ApprovalDecision] | None = None
            restored_responses: list[Json] = []
            restored_records: list[Json] = []
            restored_warnings: set[Any] = set()
            if approval_state is not None:
                approval_pause = validate_approval_pause_state(approval_state)
                approval_decision_map = validate_approval_decisions(approval_decisions or [], approval_pause.approval_required_ids)
                restored_metadata, restored_usage, restored_responses, restored_records, restored_warnings = copy_restored_run_state(
                    approval_pause,
                    metadata,
                )
                run_metadata = restored_metadata
                usage = restored_usage
            else:
                run_metadata = dict(metadata or {})
                usage = RunUsage()
            run_ctx = RunContext(
                harness=self,
                prompt=prompt,
                metadata=run_metadata,
                usage=usage,
                tracer=run_tracer,
                stream=stream_context,
                emitter=emitter,
            )
            if approval_pause is not None:
                run_ctx.responses = restored_responses
                run_ctx.tool_call_records = restored_records
                run_ctx.emitted_limit_warnings = restored_warnings
            run_ctx.emit(RunStartedEvent(
                **run_ctx.stream_base(),
                prompt=None if approval_pause is not None else prompt,
                root=str(self.root),
                max_model_requests=self.config.max_model_requests,
                max_tool_calls=self.config.max_tool_calls,
            ))
            if approval_pause is not None:
                run_ctx.emit(ApprovalResumedEvent(
                    **run_ctx.stream_base(),
                    decisions=tuple(approval_decisions or []),
                ))
        except Exception as exc:
            self._running = False
            emitter.emit(RunFailedEvent(
                run_id=stream_context.run_id,
                sequence=0,
                parent_run_id=stream_context.parent_run_id,
                parent_tool_call_id=stream_context.parent_tool_call_id,
                agent_name=stream_context.agent_name,
                stop_reason="error",
                error_type=type(exc).__name__,
                message=str(exc),
            ))
            emitter.finish()
            raise

        try:
            try:
                session: ModelSession | None
                model_supports_resume = hasattr(self.model, "resume_kind") and hasattr(self.model, "resume_session")
                if approval_pause is not None:
                    if not model_supports_resume:
                        run_ctx.terminal_error = HarnessError(f"model {type(self.model).__name__} does not support approval resume")
                        raise run_ctx.terminal_error
                    session = self._resume_approval_session(approval_pause.provider_state)
                    first_turn_kind = "approval_resume"
                elif resume_from is None:
                    session = None
                    first_turn_kind = "start"
                else:
                    if not model_supports_resume:
                        run_ctx.terminal_error = HarnessError(f"model {type(self.model).__name__} does not support resume")
                        raise run_ctx.terminal_error
                    session = cast(ResumableModel, self.model).resume_session(resume_from)
                    first_turn_kind = "resume"
                conversation_id = str(run_metadata.get("conversation_id")) if run_metadata.get("conversation_id") else None
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
                        effective_prompt, instructions = await self._prepare_run_start(
                            prompt,
                            run_metadata,
                            run_ctx,
                            agent_span,
                            skip_user_prompt=approval_pause is not None,
                        )
                        structured_output = self._structured_output_request()
                        if first_turn_kind == "approval_resume":
                            assert approval_pause is not None
                            assert approval_decision_map is not None
                            assert session is not None
                            driver = self._turn_driver(
                                session=session,
                                instructions=instructions,
                                metadata=run_metadata,
                                structured_output=structured_output,
                                run_ctx=run_ctx,
                            )
                            active_session = session
                            turn, decision = await self._resume_approval_batch(
                                approval_pause,
                                approval_decision_map,
                                driver,
                                run_ctx,
                                tool_executor,
                            )
                        else:
                            active_session, driver, turn, decision = await self._start_or_resume_turn(
                                first_turn_kind=cast(Literal["start", "resume"], first_turn_kind),
                                session=session,
                                effective_prompt=effective_prompt,
                                instructions=instructions,
                                metadata=metadata,
                                structured_output=structured_output,
                                run_ctx=run_ctx,
                                agent_span=agent_span,
                            )
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
                                run_ctx.emit_retry_event("structured_output", retry_message, final_id)
                                turn, decision = await driver.send_tool_outputs(
                                    [ToolOutput(final_id, retry_message)],
                                    kind="output_retry_tool",
                                    output_retry=True,
                                )
                                continue
                            if decision.kind == "retry_user_message":
                                run_ctx.retry_or_fail()
                                retry_message = decision.retry_message
                                run_ctx.emit_retry_event("structured_output", retry_message, decision.retry_call_id)
                                turn, decision = await driver.send_user_message(
                                    retry_message,
                                    kind="correction",
                                    output_retry=True,
                                )
                                continue
                            if decision.kind == "unexpected":
                                raise UnexpectedModelBehavior(decision.unexpected_message)
                            approval_calls = self._approval_required_calls(turn.tool_calls)
                            if approval_calls:
                                run_ctx.check_tool_limit(len(turn.tool_calls))
                                run_ctx.usage.tool_calls += len(turn.tool_calls)
                                return run_ctx.pause_for_approval(turn, approval_calls, active_session)
                            turn, decision = await self._execute_tool_turn(turn, driver, run_ctx, tool_executor)
                    except asyncio.CancelledError as exc:
                        run_ctx.stop_reason = "cancelled"
                        run_ctx.terminal_error = exc
                        agent_span.record_exception(exc)
                        agent_span.set_error("run cancelled", "CancelledError")
                        raise
                    except Exception as exc:
                        failure = _classify_run_failure(run_ctx, agent_span, exc)
                        if failure is exc:
                            raise
                        raise failure from exc
            finally:
                run_ctx.fire_run_end_once()
        except Exception as exc:
            if run_ctx.terminal_error is None:
                run_ctx.terminal_error = exc
                if run_ctx.stop_reason == "end_turn":
                    run_ctx.stop_reason = "error"
            run_ctx.emit(RunFailedEvent(
                **run_ctx.stream_base(),
                stop_reason=run_ctx.stop_reason,
                error_type=type(exc).__name__,
                message=str(exc),
            ))
            raise
        finally:
            self._running = False
            emitter.finish()

    async def _prepare_run_start(
        self,
        prompt: str,
        run_metadata: Json,
        run_ctx: Any,
        agent_span: Any,
        *,
        skip_user_prompt: bool = False,
    ) -> tuple[str, str]:
        """Fire start hooks and return the effective prompt plus instructions."""
        self.hooks.fire(RunStartContext(
            harness=self,
            metadata=dict(run_metadata),
            prompt=prompt,
            root=self.root,
            max_model_requests=self.config.max_model_requests,
            max_tool_calls=self.config.max_tool_calls,
        ))
        await self._ensure_mcp_connected()
        effective_prompt = prompt
        if not skip_user_prompt:
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
        return effective_prompt, instructions

    async def _start_or_resume_turn(
        self,
        *,
        first_turn_kind: Literal["start", "resume"],
        session: ModelSession | None,
        effective_prompt: str,
        instructions: str,
        metadata: Json | None,
        structured_output: StructuredOutputRequest | None,
        run_ctx: Any,
        agent_span: Any,
    ) -> tuple[ModelSession, Any, ModelTurn, OutputTurnDecision]:
        """Create the turn driver and make the initial model request."""
        from .runtime import TurnDriver

        if first_turn_kind == "start":
            try:
                active_session = self.model.new_session()
            except Exception as exc:
                run_ctx.stop_reason = "error"
                run_ctx.terminal_error = exc
                agent_span.record_exception(exc)
                agent_span.set_error(str(exc), type(exc).__name__)
                raise
            driver = TurnDriver(
                session=active_session,
                run_ctx=run_ctx,
                harness=self,
                instructions=instructions,
                metadata=metadata,
                structured_output=structured_output,
            )
            turn, decision = await driver.start(effective_prompt)
            return active_session, driver, turn, decision

        assert session is not None
        driver = TurnDriver(
            session=session,
            run_ctx=run_ctx,
            harness=self,
            instructions=instructions,
            metadata=metadata,
            structured_output=structured_output,
        )
        turn, decision = await driver.resume(effective_prompt)
        return session, driver, turn, decision

    def _turn_driver(
        self,
        *,
        session: ModelSession,
        instructions: str,
        metadata: Json | None,
        structured_output: StructuredOutputRequest | None,
        run_ctx: Any,
    ) -> Any:
        """Create a TurnDriver for an already prepared session."""
        from .runtime import TurnDriver

        return TurnDriver(
            session=session,
            run_ctx=run_ctx,
            harness=self,
            instructions=instructions,
            metadata=metadata,
            structured_output=structured_output,
        )

    def _resume_approval_session(self, provider_state: Json) -> ModelSession:
        """Resume a provider session for an approval envelope with approval-specific errors."""
        try:
            return cast(ResumableModel, self.model).resume_session(provider_state)
        except HarnessError as exc:
            message = str(exc)
            if message.startswith("resume_from"):
                message = f"approval state provider_state{message[len('resume_from'):]}"
            raise HarnessError(message) from exc

    async def _resume_approval_batch(
        self,
        approval_pause: ApprovalPause,
        decisions: dict[str, ApprovalDecision],
        driver: Any,
        run_ctx: Any,
        tool_executor: Any,
    ) -> tuple[ModelTurn, OutputTurnDecision]:
        """Execute or reject the paused batch, then continue the model."""
        self._validate_approval_resume_tools(approval_pause)
        batch = [
            ModelToolCall(id=call.id, name=call.name, arguments=call.arguments)
            for call in approval_pause.batch
        ]
        executed: list[tuple[int, ModelToolCall]] = [
            (index, call)
            for index, call in enumerate(batch)
            if call.id not in approval_pause.approval_required_ids or decisions[call.id].approved
        ]
        executed_indices = [index for index, _call in executed]
        executed_calls = [call for _index, call in executed]
        executed_by_id: dict[str, tuple[Json, ToolOutput, Any]] = {}
        if executed_calls:
            recorded, outputs, executions = await tool_executor.execute_batch(executed_calls, tool_indices=executed_indices)
            run_ctx.usage.cancelled_tool_calls += sum(1 for execution in executions if execution.cancelled)
            for call, record, output, execution in zip(executed_calls, recorded, outputs, executions, strict=True):
                if call.id in approval_pause.approval_required_ids:
                    record["approval"] = {"approved": True}
                executed_by_id[call.id] = (record, output, execution)
            run_ctx.check_tool_retry_limits(executed_calls, executions)

        ordered_records: list[Json] = []
        ordered_outputs: list[ToolOutput] = []
        for index, call in enumerate(batch):
            decision = decisions.get(call.id)
            if decision is not None and not decision.approved:
                record, output = self._reject_approval_call(call, index, decision, run_ctx)
                ordered_records.append(record)
                ordered_outputs.append(output)
                continue
            record, output, _execution = executed_by_id[call.id]
            ordered_records.append(record)
            ordered_outputs.append(output)
        run_ctx.record_tool_batch(ordered_records)
        return await driver.send_tool_outputs(ordered_outputs, kind="approval_resume")

    def _reject_approval_call(
        self,
        call: ModelToolCall,
        index: int,
        decision: ApprovalDecision,
        run_ctx: Any,
    ) -> tuple[Json, ToolOutput]:
        """Create model-visible rejection output without running hooks."""
        message = "Tool call was rejected by a human reviewer."
        if decision.reason:
            message = f"{message}\nReason: {decision.reason}"
        output = ToolResult(False, message, {"error_type": "ApprovalRejected"}).to_json()
        start = time.perf_counter()
        with run_ctx.tracer.tool(tool_name=call.name, call_id=call.id, arguments=call.arguments) as span:
            run_ctx.emit(ToolCallStartedEvent(
                **run_ctx.stream_base(),
                call_id=call.id,
                tool_name=call.name,
                tool_index=index,
                arguments=call.arguments,
            ))
            span.set_attribute_where(
                lambda option: option.capture_tool_results,
                "gen_ai.tool.call.result",
                serialize_attribute_value(output),
            )
            span.set_error("Tool call rejected by human reviewer", "ApprovalRejected")
            run_ctx.emit(ToolCallCompletedEvent(
                **run_ctx.stream_base(),
                call_id=call.id,
                tool_name=call.name,
                ok=False,
                error_type="ApprovalRejected",
                message=message,
                duration_ms=(time.perf_counter() - start) * 1000,
                output=output,
            ))
        return (
            {
                "call": {"id": call.id, "name": call.name, "arguments": call.arguments},
                "output": output,
                "approval": {"approved": False, "reason": decision.reason},
            },
            ToolOutput(call.id, output),
        )

    def _validate_approval_resume_tools(self, approval_pause: ApprovalPause) -> None:
        """Require approval-required tools to still be configured before execution."""
        current_required_ids: set[str] = set()
        for call in approval_pause.batch:
            spec = self._tool_map.get(str(call.name))
            if call.id in approval_pause.approval_required_ids and spec is None:
                raise HarnessError(f"approval-required tool {call.name!r} is not configured")
            if spec is not None and spec.requires_approval:
                current_required_ids.add(call.id)
        if current_required_ids != set(approval_pause.approval_required_ids):
            raise HarnessError("approval state approval_required_ids do not match configured approval-required tools")

    def _approval_required_calls(self, calls: list[ModelToolCall]) -> list[ModelToolCall]:
        """Return approval-required calls from a model batch."""
        return [
            call for call in calls
            if (spec := self._tool_map.get(str(call.name))) is not None and spec.requires_approval
        ]

    def _pending_approval_record(self, call: ModelToolCall) -> PendingApproval:
        """Return the host-facing pending approval shape for one call."""
        return PendingApproval(call_id=call.id, tool_name=call.name, arguments=call.arguments)

    async def _execute_tool_turn(
        self,
        turn: ModelTurn,
        driver: Any,
        run_ctx: Any,
        tool_executor: Any,
    ) -> tuple[ModelTurn, OutputTurnDecision]:
        """Execute a normal tool-call turn and continue with its outputs."""
        run_ctx.check_tool_limit(len(turn.tool_calls))
        run_ctx.usage.tool_calls += len(turn.tool_calls)
        recorded, outputs, executions = await tool_executor.execute_batch(turn.tool_calls)
        run_ctx.usage.cancelled_tool_calls += sum(1 for execution in executions if execution.cancelled)
        run_ctx.record_tool_batch(recorded)
        run_ctx.check_tool_retry_limits(turn.tool_calls, executions)
        return await driver.send_tool_outputs(outputs)

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
        self._validate_tool_spec(
            spec,
            output_schema=self.output_schema,
            model_supports_approval_resume=self._model_supports_approval_resume(),
            is_child_run=self._is_child_run,
        )
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

    @classmethod
    def _validate_tool_list(
        cls,
        tools: list[ToolSpec],
        *,
        output_schema: OutputSchema | None,
        model_supports_approval_resume: bool,
        is_child_run: bool,
    ) -> None:
        """Validate a complete tool list before assigning it to a harness."""
        cls._validate_unique_tools(tools)
        for tool in tools:
            cls._validate_tool_spec(
                tool,
                output_schema=output_schema,
                model_supports_approval_resume=model_supports_approval_resume,
                is_child_run=is_child_run,
            )

    @staticmethod
    def _validate_tool_spec(
        spec: ToolSpec,
        *,
        output_schema: OutputSchema | None,
        model_supports_approval_resume: bool,
        is_child_run: bool,
    ) -> None:
        """Validate one tool against explicit harness state."""
        if not callable(spec.handler):
            raise TypeError(f"handler for tool {spec.name!r} is not callable")
        if spec.name == "subagent" and spec.kind != "subagent":
            raise ValueError("subagent is a reserved tool name")
        if (
            spec.name == FINAL_RESULT_TOOL_NAME
            and output_schema is not None
            and output_schema.mode != "text"
        ):
            raise ValueError(f"{FINAL_RESULT_TOOL_NAME} is reserved for structured output")
        Harness._validate_tool_approval_policy_for(
            spec,
            model_supports_approval_resume=model_supports_approval_resume,
            is_child_run=is_child_run,
        )

    def _validate_tool_approval_policy(self, tool: ToolSpec) -> None:
        """Reject approval policies incompatible with this harness configuration."""
        self._validate_tool_approval_policy_for(
            tool,
            model_supports_approval_resume=self._model_supports_approval_resume(),
            is_child_run=self._is_child_run,
        )

    @staticmethod
    def _validate_tool_approval_policy_for(
        tool: ToolSpec,
        *,
        model_supports_approval_resume: bool,
        is_child_run: bool,
    ) -> None:
        """Reject approval policies incompatible with explicit harness state."""
        if tool.requires_approval and not model_supports_approval_resume:
            raise ValueError("approval-required tools require a resumable model")
        if tool.requires_approval and is_child_run:
            raise ValueError("approval-required tools are not supported inside subagents")

    @staticmethod
    def _validate_skill_tool_selection_for(skills: SkillRegistry, tools: list[ToolSpec]) -> None:
        """Require explicit skill tool selection for explicit skills and tool state."""
        if not skills.skills:
            return
        tool_names = {tool.name for tool in tools}
        if not tool_names.intersection({"skill_read", "skill_run"}):
            raise ValueError("configured skills require exposing skill_read or skill_run")

    def _validate_hook_filters(self) -> None:
        """Validate hook filters against registered subagents."""
        self._validate_hook_registry(self.hooks, self.config.subagents)

    @staticmethod
    def _validate_hook_registry(hooks: HookRegistry, subagents: list[SubAgentConfig]) -> None:
        """Validate hook filters against explicit subagent configuration."""
        agent_names = {DEFAULT_SUBAGENT_NAME, *(config.name for config in subagents)}
        hooks.validate_filters(agent_names=agent_names)

    def _model_supports_approval_resume(self) -> bool:
        """Return whether this harness model can resume provider sessions."""
        return hasattr(self.model, "resume_kind") and hasattr(self.model, "resume_session")

    def _resolve_mcp_server_ids(self) -> None:
        """Assign stable suffixes to duplicate MCP server ids."""
        counts: dict[str, int] = {}
        for server in self._mcp_servers:
            server.resolve_id(counts)

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
                    self._validate_tool_approval_policy(tool)
                    seen.add(tool.name)
                    mcp_tools.append(tool)
            self.tools.extend(mcp_tools)
            self._tool_map.update({tool.name: tool for tool in mcp_tools})
            self._mcp_stack = stack
            self._mcp_connected = True
        except BaseException:
            await stack.aclose()
            raise

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
