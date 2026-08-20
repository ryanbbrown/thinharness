"""SDK-only provider-agnostic agent loop."""

from __future__ import annotations

import asyncio
import os
from contextlib import AsyncExitStack
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ._migration import REMOVED_HARNESS_CONFIG_FIELDS, reject_removed_fields
from .approvals import (
    ApprovalPause,
    copy_restored_run_state,
    is_approval_pause_state,
    validate_approval_decisions,
    validate_approval_pause_state,
)
from .children import ChildHarnessHost, _ParentChildHarnessHost, _ToolComposition
from .content import ImageBlock, NormalizedContent, Prompt, normalize_content, redact_image_data, redacted_content_string, text_only_value
from .defaults import DEFAULT_SYSTEM_PROMPT
from .events import (
    ApprovalResumedEvent,
    HarnessStream,
    RunCompletedEvent,
    RunFailedEvent,
    RunStartedEvent,
    StreamEmitter,
    StreamOptions,
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
    resolve_output_schema_for_model,
    structured_instructions,
)
from .plugins.base import Plugin, PluginBinding, PluginContext, PluginContribution
from .providers import (
    Model,
    ModelSession,
    ModelToolCall,
    ProviderError,
    RequestConstants,
    ResumableModel,
    StructuredOutputRequest,
    infer_model,
    model_capabilities,
    session_image_blocks,
)
from .tools.base import ToolOrigin, ToolSpec
from .tracing import (
    LocalTracing,
    RunTracer,
    TracingOptions,
    annotate_agent_start,
    create_local_tracing,
)
from .turns import TurnStart, advance_until_terminal
from .types import ApprovalDecision, HarnessError, HarnessResult, Json, PendingApproval, RunUsage, UnexpectedModelBehavior


def _local_tracing_enabled(configured: bool) -> bool:
    """Return whether local plaintext tracing should be active."""
    disabled = os.getenv("THINHARNESS_DISABLE_LOCAL_TRACING", "").lower() in {"1", "true", "yes"}
    return configured and not disabled


def _error_type(exc: BaseException) -> str:
    """Return the stable public error classification for an exception."""
    value = getattr(exc, "_thinharness_error_type", None)
    return value if isinstance(value, str) else type(exc).__name__


def _sanitized_failure(exc: Exception, message: str) -> Exception:
    """Return an exception with a safe message and stable classification."""
    if message == str(exc):
        return exc
    if isinstance(exc, ProviderError):
        sanitized_provider = ProviderError(message, status_code=exc.status_code)
        sanitized_provider.__dict__["_thinharness_sanitized"] = True
        return sanitized_provider
    sanitized = HarnessError(message)
    sanitized.__dict__["_thinharness_error_type"] = _error_type(exc)
    sanitized.__dict__["_thinharness_sanitized"] = True
    if getattr(exc, "_thinharness_strict_hook", False):
        sanitized.__dict__["_thinharness_strict_hook"] = True
    return sanitized


def _classify_run_failure(run_ctx: Any, agent_span: Any, exc: Exception) -> Exception:
    """Record a run failure and return the exception to raise."""
    message = redact_image_data(str(exc), run_ctx.image_blocks)
    safe_exc = _sanitized_failure(exc, message)
    error_type = _error_type(safe_exc)
    agent_span.record_exception(safe_exc)
    agent_span.set_error(message, error_type)
    if isinstance(safe_exc, ProviderError) or getattr(safe_exc, "_thinharness_provider_error", False):
        run_ctx.stop_reason = "provider_error"
        terminal = HarnessError(message)
        terminal.__dict__["_thinharness_error_type"] = error_type
        status_code = getattr(safe_exc, "status_code", None)
        if status_code is not None:
            terminal.__dict__["status_code"] = status_code
        if getattr(safe_exc, "_thinharness_sanitized", False):
            terminal.__dict__["_thinharness_sanitized"] = True
        run_ctx.terminal_error = terminal
        return terminal
    if isinstance(safe_exc, UnexpectedModelBehavior):
        run_ctx.stop_reason = "unexpected_model_behavior"
        run_ctx.terminal_error = run_ctx.terminal_error or safe_exc
        return safe_exc
    if isinstance(safe_exc, HarnessError):
        run_ctx.terminal_error = run_ctx.terminal_error or safe_exc
        if run_ctx.stop_reason == "end_turn":
            run_ctx.stop_reason = "error"
        return safe_exc
    run_ctx.stop_reason = "error"
    run_ctx.terminal_error = safe_exc
    return safe_exc


def _normalize_public_prompt(prompt: Prompt, *, label: str) -> NormalizedContent:
    """Normalize a public or hook prompt as a harness validation error."""
    try:
        return normalize_content(prompt, label=label)
    except (TypeError, ValueError) as exc:
        raise HarnessError(str(exc)) from exc


def _event_prompt(content: NormalizedContent) -> str:
    """Keep text-only stream values and redact multimodal values."""
    text = text_only_value(content)
    return text if text is not None else redacted_content_string(content)


class HarnessConfig(BaseModel):
    """Configuration for Harness."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: str = "openai:gpt-5.5"
    root: str | Path = "."
    api_key: str | None = None
    base_url: str | None = None
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    max_model_requests: int = 64
    max_tool_calls: int | None = None
    strict_hooks: bool = False
    request_timeout: int = 120
    request_retries: int = Field(default=3, ge=0, le=10)
    request_retry_backoff: float = Field(default=1.0, ge=0, allow_inf_nan=False)
    temperature: float | None = None
    max_tokens: int | None = Field(default=None, ge=1)
    effort: str | None = None
    extra_body: dict[str, Any] = Field(default_factory=dict)
    tracing: list[TracingOptions] = Field(default_factory=list)
    local_tracing: bool = True
    local_trace_dir: str | Path = "~/.thinharness/traces"
    tool_execution: Literal["auto", "sequential"] = "auto"
    output_type: OutputSpec | None = None
    output_mode: OutputMode = "auto"
    output_retries: int = Field(default=1, ge=0)
    tool_retries: int = Field(default=1, ge=0)

    @model_validator(mode="before")
    @classmethod
    def reject_removed_fields(cls, data: object) -> object:
        """Fail loudly when callers use configuration moved to plugins."""
        return reject_removed_fields(
            data,
            owner="HarnessConfig",
            migrations=REMOVED_HARNESS_CONFIG_FIELDS,
        )


class Harness:
    """A non-interactive agent harness for SDK use."""

    def __init__(
        self,
        config: HarnessConfig | None = None,
        *,
        model: Model | None = None,
        plugins: list[Plugin] | None = None,
        tools: list[ToolSpec] | None = None,
        tracing: list[TracingOptions] | None = None,
        hooks: list[Hook] | HookRegistry | None = None,
        _owns_model: bool | None = None,
        _is_child_harness: bool = False,
        _child_harnesses: ChildHarnessHost | None = None,
    ) -> None:
        self.config = config or HarnessConfig()
        self._is_child_harness = _is_child_harness
        self.root = Path(self.config.root).expanduser().resolve()
        self.model_ref = os.getenv("HARNESS_MODEL", self.config.model)
        self.model = model or infer_model(
            self.model_ref,
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            timeout=self.config.request_timeout,
            request_retries=self.config.request_retries,
            request_retry_backoff=self.config.request_retry_backoff,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            effort=self.config.effort,
            extra_body=self.config.extra_body,
        )
        self._owns_model = _owns_model if _owns_model is not None else model is None
        self.model_capabilities = model_capabilities(self.model)
        output_schema = resolve_output_schema_for_model(self.model, self.config.output_type, self.config.output_mode)
        self.output_schema = output_schema

        child_harnesses = _child_harnesses or _ParentChildHarnessHost(self)
        configured_plugins = tuple(plugins or [])
        plugin_names = [plugin.name for plugin in configured_plugins]
        if any(not isinstance(name, str) or not name.strip() for name in plugin_names):
            raise ValueError("plugin name must be a non-empty string")
        duplicate_plugin = next((name for index, name in enumerate(plugin_names) if name in plugin_names[:index]), None)
        if duplicate_plugin is not None:
            raise ValueError(f"duplicate plugin name: {duplicate_plugin}")
        plugin_context = PluginContext(root=self.root, model=self.model, child_harnesses=child_harnesses)
        bindings = tuple(plugin.bind(plugin_context) for plugin in configured_plugins)
        for plugin, binding in zip(configured_plugins, bindings, strict=True):
            if not isinstance(binding, PluginBinding):
                raise TypeError(f"plugin {plugin.name!r} returned an invalid binding")
        static_tools: list[ToolSpec] = []
        static_compositions: list[_ToolComposition] = []
        static_instructions: list[str] = []
        static_hooks: list[Hook] = []
        agent_names: list[str] = []
        for plugin_index, (plugin, binding) in enumerate(zip(configured_plugins, bindings, strict=True)):
            contribution = self._normalize_contribution(plugin.name, binding.static)
            static_tools.extend(contribution.tools)
            static_compositions.extend(
                _ToolComposition(
                    source="plugin",
                    plugin_index=plugin_index,
                    delegation=isinstance(child_harnesses, _ParentChildHarnessHost)
                    and child_harnesses.is_delegation_tool(raw_tool),
                )
                for raw_tool in binding.static.tools
            )
            static_instructions.extend(contribution.instructions)
            static_hooks.extend(contribution.hooks)
            agent_names.extend(self._validate_binding_agent_names(binding.agent_names, agent_names))

        direct_tools = list(tools or [])
        configured_tools = [*static_tools, *direct_tools]
        configured_compositions = [
            *static_compositions,
            *(_ToolComposition(source="direct") for _ in direct_tools),
        ]
        self._validate_tool_list(
            configured_tools,
            output_schema=output_schema,
            model_supports_approval_resume=self._model_supports_approval_resume(),
            is_child_harness=self._is_child_harness,
        )
        tool_map = {tool.name: tool for tool in configured_tools}
        caller_hooks = list(hooks.hooks) if isinstance(hooks, HookRegistry) else list(hooks or [])
        strict_hooks = hooks.strict_hooks if isinstance(hooks, HookRegistry) else self.config.strict_hooks
        hook_registry = HookRegistry([*static_hooks, *caller_hooks], strict_hooks=strict_hooks)
        self._validate_hook_registry(hook_registry, set(agent_names))

        self.plugins = configured_plugins
        self._plugin_bindings = bindings
        self._base_tools = list(configured_tools)
        self._base_compositions = list(configured_compositions)
        self._base_instructions = list(static_instructions)
        self._strict_hooks = strict_hooks
        self.tools = configured_tools
        self._tool_map = tool_map
        self._tool_composition = {
            tool.name: composition
            for tool, composition in zip(configured_tools, configured_compositions, strict=True)
        }
        self._plugin_instructions = list(static_instructions)
        self.hooks = hook_registry
        self._agent_names = set(agent_names)
        self._child_harnesses = child_harnesses
        if isinstance(child_harnesses, _ParentChildHarnessHost):
            child_harnesses.validate_recipes(configured_tools, configured_compositions)
        self._plugin_stack: AsyncExitStack | None = None
        self._connected = False
        self._connect_lock = asyncio.Lock()
        self._connect_task: asyncio.Task[None] | None = None
        self._connect_waiters = 0
        self.local_tracing: LocalTracing | None = None
        external_tracing = list(self.config.tracing if tracing is None else tracing)
        if _local_tracing_enabled(self.config.local_tracing) and not _is_child_harness:
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

    async def run(self, prompt: Prompt, *, resume_from: dict[str, Any] | None = None, metadata: Json | None = None) -> HarnessResult:
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
        prompt: Prompt,
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
        task = loop.create_task(
            self._run_streaming(
                prompt,
                resume_from=resume_from,
                approval_state=None,
                approval_decisions=None,
                metadata=metadata,
                emitter=emitter,
                stream_context=stream_context,
            )
        )
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
        task = loop.create_task(
            self._run_streaming(
                "",
                resume_from=None,
                approval_state=state,
                approval_decisions=decisions,
                metadata=metadata,
                emitter=emitter,
                stream_context=stream_context,
            )
        )
        self._running = True
        return HarnessStream(task, emitter)

    async def _run_streaming(
        self,
        prompt: Prompt,
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
            await self._ensure_connected()
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
            raw_prompt: NormalizedContent = () if approval_pause is not None else _normalize_public_prompt(prompt, label="prompt")
            run_ctx = RunContext(
                harness=self,
                prompt=raw_prompt,
                metadata=run_metadata,
                usage=usage,
                tracer=run_tracer,
                stream=stream_context,
                emitter=emitter,
            )
            run_ctx.register_image_blocks(tuple(block for block in raw_prompt if isinstance(block, ImageBlock)))
            if approval_pause is not None:
                run_ctx.responses = restored_responses
                run_ctx.tool_call_records = restored_records
                run_ctx.emitted_limit_warnings = restored_warnings
            run_ctx.emit(
                RunStartedEvent(
                    **run_ctx.stream_base(),
                    prompt=None if approval_pause is not None else _event_prompt(raw_prompt),
                    root=str(self.root),
                    max_model_requests=self.config.max_model_requests,
                    max_tool_calls=self.config.max_tool_calls,
                )
            )
            if approval_pause is not None:
                run_ctx.emit(
                    ApprovalResumedEvent(
                        **run_ctx.stream_base(),
                        decisions=tuple(approval_decisions or []),
                    )
                )
        except BaseException as exc:
            self._running = False
            emitter.emit(
                RunFailedEvent(
                    run_id=stream_context.run_id,
                    sequence=0,
                    parent_run_id=stream_context.parent_run_id,
                    parent_tool_call_id=stream_context.parent_tool_call_id,
                    agent_name=stream_context.agent_name,
                    stop_reason="cancelled" if isinstance(exc, asyncio.CancelledError) else "error",
                    error_type=type(exc).__name__,
                    message=str(exc),
                )
            )
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
                if session is not None:
                    run_ctx.register_image_blocks(session_image_blocks(session))
                conversation_id = str(run_metadata.get("conversation_id")) if run_metadata.get("conversation_id") else None
                with run_tracer.agent(conversation_id=conversation_id) as agent_span:
                    run_ctx.agent_span = agent_span
                    try:
                        effective_prompt, instructions = await self._prepare_run_start(
                            raw_prompt,
                            run_metadata,
                            run_ctx,
                            agent_span,
                            skip_user_prompt=approval_pause is not None,
                        )
                        constants = RequestConstants(
                            instructions=instructions,
                            tools=self.tool_schemas(),
                            metadata=run_metadata,
                            structured_output=self._structured_output_request(),
                        )
                        # Snapshot the executable tool map alongside the frozen
                        # schemas so a tool added mid-run is neither advertised
                        # nor executable within this run.
                        tool_executor = ToolBatchExecutor(
                            harness=self,
                            run_context=run_ctx,
                            tool_map=dict(self._tool_map),
                            tool_composition=dict(self._tool_composition),
                            run_tracer=run_tracer,
                            tool_execution=self.config.tool_execution,
                        )
                        active_session = session if session is not None else self.model.new_session()
                        start = TurnStart(
                            kind=cast(Literal["start", "resume", "approval_resume"], first_turn_kind),
                            prompt=effective_prompt,
                            approval_pause=approval_pause,
                            approval_decisions=approval_decision_map,
                        )
                        return await advance_until_terminal(start, active_session, constants, self, run_ctx, tool_executor)
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
                        if getattr(failure, "_thinharness_sanitized", False):
                            raise failure from None
                        raise failure from exc
            finally:
                run_ctx.fire_run_end_once()
        except Exception as exc:
            if run_ctx.terminal_error is None:
                run_ctx.terminal_error = exc
                if run_ctx.stop_reason == "end_turn":
                    run_ctx.stop_reason = "error"
            run_ctx.emit(
                RunFailedEvent(
                    **run_ctx.stream_base(),
                    stop_reason=run_ctx.stop_reason,
                    error_type=_error_type(exc),
                    message=redact_image_data(str(exc), run_ctx.image_blocks),
                )
            )
            raise
        finally:
            self._running = False
            emitter.finish()

    async def _prepare_run_start(
        self,
        prompt: NormalizedContent,
        run_metadata: Json,
        run_ctx: Any,
        agent_span: Any,
        *,
        skip_user_prompt: bool = False,
    ) -> tuple[NormalizedContent, str]:
        """Fire start hooks and return effective normalized content plus instructions."""
        initial: NormalizedContent = () if skip_user_prompt else prompt
        start_ctx = RunStartContext(
            harness=self,
            metadata=dict(run_metadata),
            prompt=initial,
            root=self.root,
            max_model_requests=self.config.max_model_requests,
            max_tool_calls=self.config.max_tool_calls,
        )
        self.hooks.fire(start_ctx)
        effective_prompt = initial if skip_user_prompt else _normalize_public_prompt(start_ctx.prompt, label="run_start prompt")
        if not skip_user_prompt:
            prompt_ctx = UserPromptSubmitContext(harness=self, metadata=dict(run_metadata), prompt=effective_prompt)
            self.hooks.fire(prompt_ctx)
            if prompt_ctx.cancelled:
                reason = prompt_ctx.cancel_reason or "unspecified"
                run_ctx.stop_reason = "cancelled_by_hook"
                run_ctx.terminal_error = HarnessError(f"run blocked by hook: {reason}")
                raise run_ctx.terminal_error
            submitted = _normalize_public_prompt(prompt_ctx.prompt, label="user_prompt_submit prompt")
            effective_prompt = apply_prompt_context(submitted, prompt_ctx.additional_context)
            run_ctx.set_prompt_content(effective_prompt)
        instructions = structured_instructions(self.system_instructions(), self.output_schema)
        agent_span.for_each(
            lambda span, option: annotate_agent_start(
                span,
                prompt=None if skip_user_prompt else initial,
                instructions=instructions,
                capture_messages=option.capture_messages,
                top_level=not self._is_child_harness,
            )
        )
        return effective_prompt, instructions

    def _resume_approval_session(self, provider_state: Json) -> ModelSession:
        """Resume a provider session for an approval envelope with approval-specific errors."""
        try:
            return cast(ResumableModel, self.model).resume_session(provider_state)
        except HarnessError as exc:
            message = str(exc)
            if message.startswith("resume_from"):
                message = f"approval state provider_state{message[len('resume_from') :]}"
            raise HarnessError(message) from exc

    def _pending_approval_record(self, call: ModelToolCall) -> PendingApproval:
        """Return the host-facing pending approval shape for one call."""
        return PendingApproval(call_id=call.id, tool_name=call.name, arguments=call.arguments)

    def run_sync(self, prompt: Prompt, *, resume_from: dict[str, Any] | None = None, metadata: Json | None = None) -> HarnessResult:
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
        """Close connected plugins and an owned model."""
        async with self._connect_lock:
            if self._closed:
                return
            self._closed = True
            connect_task = self._connect_task
            caller_cancelled = False
            if connect_task is not None and not connect_task.done():
                current_task = asyncio.current_task()
                pending_cancels = current_task.cancelling() if current_task is not None else 0
                connect_task.cancel()
                try:
                    await connect_task
                except asyncio.CancelledError:
                    caller_cancelled = current_task is not None and current_task.cancelling() > pending_cancels
                except BaseException:
                    pass
            plugin_stack = self._plugin_stack
            self._plugin_stack = None
            self._connected = False
            close_error = await self._close_resources(
                plugin_stack=plugin_stack,
                close_model=self._owns_model,
            )
            if caller_cancelled:
                cancellation = asyncio.CancelledError()
                if close_error is not None:
                    cancellation.add_note(f"cleanup also failed: {type(close_error).__name__}: {close_error}")
                raise cancellation
            if close_error is not None:
                raise close_error

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
            is_child_harness=self._is_child_harness,
        )
        if spec.name in self._tool_map:
            raise ValueError(f"duplicate tool name: {spec.name}")
        composition = _ToolComposition(source="direct")
        candidate_tools = [*self.tools, spec]
        candidate_compositions = [*self._tool_composition.values(), composition]
        if isinstance(self._child_harnesses, _ParentChildHarnessHost):
            self._child_harnesses.validate_recipes(candidate_tools, candidate_compositions)
        self.tools.append(spec)
        self._tool_map[spec.name] = spec
        self._tool_composition[spec.name] = composition
        if not self._connected:
            self._base_tools.append(spec)
            self._base_compositions.append(composition)
        self._validate_hook_filters()

    def tool_schemas(self) -> list[Json]:
        """Return normalized Responses-style tool definitions."""
        tools = [tool.response_tool() for tool in self.tools]
        if self.output_schema is not None:
            tools.extend(self.output_schema.synthetic_tools())
        return tools

    def system_instructions(self) -> str:
        """Return the full instruction text sent to the model."""
        parts = [self.config.system_prompt, *self._plugin_instructions]
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
        seen: dict[str, ToolSpec] = {}
        for tool in tools:
            previous = seen.get(tool.name)
            if previous is not None:
                first = previous.origin.plugin if previous.origin is not None else "direct"
                second = tool.origin.plugin if tool.origin is not None else "direct"
                raise ValueError(f"duplicate tool name: {tool.name} ({first} and {second})")
            seen[tool.name] = tool

    @classmethod
    def _validate_tool_list(
        cls,
        tools: list[ToolSpec],
        *,
        output_schema: OutputSchema | None,
        model_supports_approval_resume: bool,
        is_child_harness: bool,
    ) -> None:
        """Validate a complete tool list before assigning it to a harness."""
        cls._validate_unique_tools(tools)
        for tool in tools:
            cls._validate_tool_spec(
                tool,
                output_schema=output_schema,
                model_supports_approval_resume=model_supports_approval_resume,
                is_child_harness=is_child_harness,
            )

    @staticmethod
    def _validate_tool_spec(
        spec: ToolSpec,
        *,
        output_schema: OutputSchema | None,
        model_supports_approval_resume: bool,
        is_child_harness: bool,
    ) -> None:
        """Validate one tool against explicit harness state."""
        if not callable(spec.handler):
            raise TypeError(f"handler for tool {spec.name!r} is not callable")
        if spec.name == FINAL_RESULT_TOOL_NAME and output_schema is not None and output_schema.mode != "text":
            raise ValueError(f"{FINAL_RESULT_TOOL_NAME} is reserved for structured output")
        Harness._validate_tool_approval_policy_for(
            spec,
            model_supports_approval_resume=model_supports_approval_resume,
            is_child_harness=is_child_harness,
        )

    @staticmethod
    def _validate_tool_approval_policy_for(
        tool: ToolSpec,
        *,
        model_supports_approval_resume: bool,
        is_child_harness: bool,
    ) -> None:
        """Reject approval policies incompatible with explicit harness state."""
        if tool.requires_approval and not model_supports_approval_resume:
            raise ValueError("approval-required tools require a resumable model")
        if tool.requires_approval and is_child_harness:
            raise ValueError("approval-required tools are not supported inside child harnesses")

    def _validate_hook_filters(self) -> None:
        """Validate hook filters against statically contributed agent names."""
        self._validate_hook_registry(self.hooks, self._agent_names)

    @staticmethod
    def _validate_hook_registry(hooks: HookRegistry, agent_names: set[str]) -> None:
        """Validate hook filters against statically contributed agent names."""
        hooks.validate_filters(agent_names=agent_names)

    @staticmethod
    def _validate_binding_agent_names(names: tuple[str, ...], previous: list[str]) -> list[str]:
        """Validate immutable agent names within and across plugin bindings."""
        if not isinstance(names, tuple):
            raise TypeError("PluginBinding.agent_names must be a tuple")
        accepted: list[str] = []
        for name in names:
            if not isinstance(name, str) or not name.strip():
                raise ValueError("plugin agent name must be a non-empty string")
            if name in previous or name in accepted:
                raise ValueError(f"duplicate plugin agent name: {name}")
            accepted.append(name)
        return accepted

    def _model_supports_approval_resume(self) -> bool:
        """Return whether this harness model can resume provider sessions."""
        return hasattr(self.model, "resume_kind") and hasattr(self.model, "resume_session")

    async def connect(self) -> None:
        """Open connected plugins."""
        if self._closed:
            raise HarnessError("harness is closed")
        await self._ensure_connected()

    async def _ensure_connected(self) -> None:
        """Share one connection attempt and commit its contributions atomically."""
        if self._closed:
            raise HarnessError("harness is closed")
        if self._connected:
            return
        async with self._connect_lock:
            if self._closed:
                raise HarnessError("harness is closed")
            if self._connected:
                return
            task = self._connect_task
            if task is None or (task.done() and self._connect_waiters == 0):
                task = asyncio.create_task(self._connect_once())
                self._connect_task = task
            self._connect_waiters += 1
        try:
            await task
        finally:
            async with self._connect_lock:
                self._connect_waiters -= 1
                if self._connect_waiters == 0 and task.done() and not self._connected and self._connect_task is task:
                    self._connect_task = None

    async def _connect_once(self) -> None:
        """Open every dynamic contribution for one shared connection attempt."""
        plugin_stack = AsyncExitStack()
        base_hooks = list(self.hooks.hooks)
        try:
            dynamic_tools: list[ToolSpec] = []
            dynamic_compositions: list[_ToolComposition] = []
            dynamic_instructions: list[str] = []
            dynamic_hooks: list[Hook] = []
            for plugin_index, (plugin, binding) in enumerate(zip(self.plugins, self._plugin_bindings, strict=True)):
                if binding.connect is None:
                    continue
                contribution = await plugin_stack.enter_async_context(binding.connect())
                normalized = self._normalize_contribution(plugin.name, contribution)
                dynamic_tools.extend(normalized.tools)
                dynamic_compositions.extend(
                    _ToolComposition(
                        source="plugin",
                        plugin_index=plugin_index,
                        delegation=isinstance(self._child_harnesses, _ParentChildHarnessHost)
                        and self._child_harnesses.is_delegation_tool(raw_tool),
                    )
                    for raw_tool in contribution.tools
                )
                dynamic_instructions.extend(normalized.instructions)
                dynamic_hooks.extend(normalized.hooks)

            candidate_tools = [*self._base_tools, *dynamic_tools]
            candidate_compositions = [*self._base_compositions, *dynamic_compositions]
            self._validate_tool_list(
                candidate_tools,
                output_schema=self.output_schema,
                model_supports_approval_resume=self._model_supports_approval_resume(),
                is_child_harness=self._is_child_harness,
            )
            if isinstance(self._child_harnesses, _ParentChildHarnessHost):
                self._child_harnesses.validate_recipes(candidate_tools, candidate_compositions)
            candidate_hooks = HookRegistry([*base_hooks, *dynamic_hooks], strict_hooks=self._strict_hooks)
            self._validate_hook_registry(candidate_hooks, self._agent_names)

            if self._closed:
                raise HarnessError("harness is closed")

            self.tools = candidate_tools
            self._tool_map = {tool.name: tool for tool in candidate_tools}
            self._tool_composition = {
                tool.name: composition
                for tool, composition in zip(candidate_tools, candidate_compositions, strict=True)
            }
            self._plugin_instructions = [*self._base_instructions, *dynamic_instructions]
            self.hooks = candidate_hooks
            self._plugin_stack = plugin_stack
            self._connected = True
        except BaseException as exc:
            cleanup_error = await self._close_resources(
                plugin_stack=plugin_stack,
                close_model=False,
            )
            self.tools = list(self._base_tools)
            self._tool_map = {tool.name: tool for tool in self.tools}
            self._tool_composition = {
                tool.name: composition
                for tool, composition in zip(self.tools, self._base_compositions, strict=True)
            }
            self._plugin_instructions = list(self._base_instructions)
            self.hooks = HookRegistry(base_hooks, strict_hooks=self._strict_hooks)
            if cleanup_error is not None:
                exc.add_note(f"cleanup also failed: {type(cleanup_error).__name__}: {cleanup_error}")
            raise

    async def _close_resources(
        self,
        *,
        plugin_stack: AsyncExitStack | None,
        close_model: bool,
    ) -> BaseException | None:
        """Attempt every close in order and return the first failure."""
        first_error: BaseException | None = None
        if plugin_stack is not None:
            try:
                await plugin_stack.aclose()
            except BaseException as exc:
                first_error = exc
        if close_model:
            aclose = getattr(self.model.provider, "aclose", None)
            if aclose is not None:
                try:
                    await aclose()
                except BaseException as exc:
                    if first_error is None:
                        first_error = exc
        return first_error

    @staticmethod
    def _normalize_contribution(plugin_name: str, contribution: PluginContribution) -> PluginContribution:
        """Validate contribution values and stamp missing tool provenance."""
        if not isinstance(contribution, PluginContribution):
            raise TypeError(f"plugin {plugin_name!r} returned an invalid contribution")
        instructions = tuple(instruction for instruction in contribution.instructions if instruction.strip())
        tools = tuple(
            replace(
                tool,
                origin=ToolOrigin(
                    plugin=plugin_name,
                    source=tool.origin.source if tool.origin is not None else tool.name,
                    attributes=dict(tool.origin.attributes) if tool.origin is not None else {},
                ),
            )
            for tool in contribution.tools
        )
        return PluginContribution(tools=tools, instructions=instructions, hooks=tuple(contribution.hooks))

    def _structured_output_request(self) -> StructuredOutputRequest | None:
        """Return native structured-output request metadata."""
        if self.output_schema is None:
            return None
        return self.output_schema.structured_output_request()
