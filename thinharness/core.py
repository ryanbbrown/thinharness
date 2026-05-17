"""SDK-only provider-agnostic agent loop."""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .defaults import DEFAULT_SYSTEM_PROMPT
from .hooks import (
    _CURRENT_TOOL_CALL,
    AfterToolCallContext,
    BeforeToolCallContext,
    Hook,
    HookRegistry,
    LimitReachedContext,
    RunEndContext,
    RunStartContext,
    UserPromptSubmitContext,
    apply_prompt_context,
)
from .output import FINAL_RESULT_TOOL_NAME, OutputMode, OutputSchema, OutputSpec, OutputValidationError, resolve_output_spec
from .providers import (
    Model,
    ModelCapabilities,
    ModelSession,
    ModelToolCall,
    ModelTurn,
    ProviderError,
    ResumableModel,
    StructuredOutputRequest,
    ToolOutput,
    infer_model,
)
from .skills import SkillRegistry
from .subagents import DEFAULT_SUBAGENT_NAME, SubAgentConfig, create_subagent_tool
from .tools import DEFAULT_SEARCH_LOW_PRIORITY_DIRS, DEFAULT_SEARCH_TEST_DIRS, Json, ToolSpec, _invoke_tool
from .tools import builtin_tools as make_builtin_tools
from .tracing import RunTracer, TracingOptions, annotate_model_span, serialize_attribute_value

MAX_PARALLEL_TOOL_WORKERS = 16
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
    search_low_priority_dirs: list[str] = Field(default_factory=lambda: list(DEFAULT_SEARCH_LOW_PRIORITY_DIRS))
    search_test_dirs: list[str] = Field(default_factory=lambda: list(DEFAULT_SEARCH_TEST_DIRS))
    read_paths: list[str | Path] | None = None
    write_paths: list[str | Path] | None = None
    temperature: float | None = None
    extra_body: dict[str, Any] = Field(default_factory=dict)
    tracing: TracingOptions | None = None
    tool_execution: Literal["auto", "sequential"] = "auto"
    subagents: list[SubAgentConfig] = Field(default_factory=list)
    output_type: OutputSpec | None = None
    output_mode: OutputMode = "auto"
    output_retries: int = Field(default=1, ge=0)
    tool_retries: int = Field(default=1, ge=0)

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


@dataclass(frozen=True)
class ToolCallExecution:
    """Internal per-call execution data with control-flow signals."""

    output: str
    cancelled: bool
    retry_kind: str | None = None


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
        tracing: TracingOptions | None = None,
        skills: SkillRegistry | None = None,
        hooks: list[Hook] | HookRegistry | None = None,
        subagent_hooks: dict[str, list[Hook] | HookRegistry] | None = None,
        _owns_model: bool | None = None,
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
            search_low_priority_dirs=self.config.search_low_priority_dirs,
            search_test_dirs=self.config.search_test_dirs,
            read_paths=self.config.read_paths,
            write_paths=self.config.write_paths,
        )
        builtin_candidates = [*filesystem_tools, *self.skills.specs(), create_subagent_tool(self, self.config.subagents)]
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
        self.hooks = raw_hooks if isinstance(raw_hooks, HookRegistry) else HookRegistry(raw_hooks, strict_hooks=self.config.strict_hooks)
        self._validate_skill_tool_selection()
        self._skills_enabled = bool(self.skills.skills) and any(tool.name in {"skill_read", "skill_run"} for tool in self.tools)
        self.tracing = tracing or self.config.tracing
        self._current_run_metadata: Json | None = None
        self._running = False
        self._closed = False
        self._validate_hook_filters()

    async def run(self, prompt: str, *, resume_from: dict[str, Any] | None = None, metadata: Json | None = None) -> HarnessResult:
        """Run one prompt to completion."""
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

        responses: list[Json] = []
        tool_call_records: list[Json] = []
        usage = RunUsage()
        result: HarnessResult | None = None
        terminal_error: BaseException | None = None
        stop_reason: StopReason = "end_turn"
        finalized_via_output_tool = False
        run_end_fired = False
        run_tracer = RunTracer(self.tracing)
        run_metadata = dict(metadata or {})
        self._current_run_metadata = run_metadata
        self._running = True
        self._closed = False

        def fire_run_end_once() -> None:
            """Emit run_end exactly once for this run."""
            nonlocal run_end_fired
            if run_end_fired:
                return
            run_end_fired = True
            self.hooks.fire(RunEndContext(
                harness=self,
                metadata=dict(run_metadata),
                result=result,
                error=terminal_error,
                stop_reason=stop_reason,
                usage=usage,
            ))

        def check_model_limit() -> None:
            """Raise if another provider request would exceed the configured limit."""
            nonlocal terminal_error, stop_reason
            if usage.model_requests < self.config.max_model_requests:
                return
            self.hooks.fire(LimitReachedContext(
                harness=self,
                metadata=dict(run_metadata),
                limit_kind="model_requests",
                limit_value=self.config.max_model_requests,
                current_count=usage.model_requests,
            ))
            stop_reason = "limit_reached"
            terminal_error = HarnessError(f"model did not finish within max_model_requests={self.config.max_model_requests}")
            raise terminal_error

        def check_tool_limit(batch_size: int) -> None:
            """Raise if a requested tool batch would exceed the configured limit."""
            nonlocal terminal_error, stop_reason
            if self.config.max_tool_calls is None or usage.tool_calls + batch_size <= self.config.max_tool_calls:
                return
            self.hooks.fire(LimitReachedContext(
                harness=self,
                metadata=dict(run_metadata),
                limit_kind="tool_calls",
                limit_value=self.config.max_tool_calls,
                current_count=usage.tool_calls + batch_size,
            ))
            stop_reason = "limit_reached"
            terminal_error = HarnessError(f"tool calls would exceed max_tool_calls={self.config.max_tool_calls}")
            raise terminal_error

        async def advance_model(request, *, output_retry: bool = False) -> ModelTurn:
            """Run one provider request with limit, usage, and tracing ceremony."""
            check_model_limit()
            if output_retry:
                usage.output_retries += 1
            with run_tracer.model(self.model) as model_span:
                try:
                    advanced_turn = await request()
                    usage.model_requests += 1
                except Exception as exc:
                    model_span.record_exception(exc)
                    model_span.set_error(str(exc), type(exc).__name__)
                    raise
                annotate_model_span(model_span, advanced_turn, capture_messages=bool(self.tracing and self.tracing.capture_messages))
                if finalized_mode := self._finalized_output_mode_for_turn(advanced_turn):
                    advanced_turn.finalized_output_mode = finalized_mode
                    model_span.set_attributes({
                        "thinharness.output.mode": finalized_mode,
                        "gen_ai.output.finalized": True,
                    })
                return advanced_turn

        def build_terminal_result(text: str, output: Any | None = None) -> HarnessResult:
            """Create the terminal HarnessResult for this run."""
            return HarnessResult(
                text=text,
                output=output,
                responses=responses,
                tool_call_records=tool_call_records,
                usage=usage,
                stop_reason=stop_reason,
            )

        def attach_resume_state(session_to_dump: ModelSession) -> None:
            """Attach final resume state before run_end hooks observe the result."""
            if result is not None:
                result.resume_state = _build_resume_state(
                    session_to_dump,
                    stop_reason,
                    finalized_via_output_tool,
                    model_supports_resume,
                )

        def retry_or_fail() -> None:
            """Track one structured-output validation retry or fail the run."""
            nonlocal terminal_error, stop_reason
            if usage.output_retries >= self.config.output_retries:
                stop_reason = "output_validation_failed"
                terminal_error = HarnessError("output validation exceeded output_retries")
                raise terminal_error

        def check_tool_retry_limits(calls: list[ModelToolCall], executions: list[ToolCallExecution]) -> None:
            """Track retryable tool failures and raise if any tool exceeds its budget."""
            nonlocal terminal_error, stop_reason
            for call, execution in zip(calls, executions, strict=True):
                if execution.retry_kind is None or execution.cancelled:
                    continue
                usage.tool_retries[call.name] = usage.tool_retries.get(call.name, 0) + 1
                max_retries = self._tool_max_retries(call.name)
                if usage.tool_retries[call.name] > max_retries:
                    self.hooks.fire(LimitReachedContext(
                        harness=self,
                        metadata=dict(run_metadata),
                        limit_kind="tool_retries",
                        limit_value=max_retries,
                        current_count=usage.tool_retries[call.name],
                    ))
                    stop_reason = "tool_retries_exceeded"
                    terminal_error = HarnessError(f"tool {call.name!r} exceeded max_retries={max_retries}")
                    raise terminal_error

        try:
            try:
                conversation_id = str(metadata.get("conversation_id")) if metadata and metadata.get("conversation_id") else None
                with run_tracer.agent(conversation_id=conversation_id) as agent_span:
                    try:
                        self.hooks.fire(RunStartContext(
                            harness=self,
                            metadata=dict(run_metadata),
                            prompt=prompt,
                            root=self.root,
                            max_model_requests=self.config.max_model_requests,
                            max_tool_calls=self.config.max_tool_calls,
                        ))
                        prompt_ctx = UserPromptSubmitContext(harness=self, metadata=dict(run_metadata), prompt=prompt)
                        self.hooks.fire(prompt_ctx)
                        if prompt_ctx.cancelled:
                            reason = prompt_ctx.cancel_reason or "unspecified"
                            stop_reason = "cancelled_by_hook"
                            terminal_error = HarnessError(f"run blocked by hook: {reason}")
                            raise terminal_error
                        effective_prompt = apply_prompt_context(prompt, prompt_ctx.additional_context)
                        instructions = self.system_instructions()
                        structured_output = self._structured_output_request()
                        if self.output_schema is not None and self.output_schema.mode == "prompted":
                            instructions = f"{instructions}\n\n{self.output_schema.build_instructions()}"
                        if first_turn_kind == "start":
                            try:
                                active_session = self.model.new_session()
                            except Exception as exc:
                                stop_reason = "error"
                                terminal_error = exc
                                agent_span.record_exception(exc)
                                agent_span.set_error(str(exc), type(exc).__name__)
                                raise
                            turn = await advance_model(lambda: active_session.start(
                                prompt=effective_prompt,
                                instructions=instructions,
                                tools=self.tool_schemas(),
                                metadata=metadata,
                                structured_output=structured_output,
                            ))
                        else:
                            assert session is not None
                            active_session = session
                            turn = await advance_model(lambda: active_session.continue_with_user_prompt(
                                prompt=effective_prompt,
                                instructions=instructions,
                                tools=self.tool_schemas(),
                                metadata=metadata,
                                structured_output=structured_output,
                            ))
                        while True:
                            responses.append(turn.raw)
                            if self.output_schema is not None:
                                if self.output_schema.mode == "text":
                                    if not turn.tool_calls:
                                        agent_span.set_attribute("gen_ai.completion", turn.text if self.tracing and self.tracing.capture_messages else None)
                                        turn.finalized_output_mode = self.output_schema.mode
                                        result = build_terminal_result(turn.text, self.output_schema.validate_text(turn.text))
                                        attach_resume_state(active_session)
                                        fire_run_end_once()
                                        return result
                                elif self.output_schema.mode == "tool":
                                    finals = [call for call in turn.tool_calls if call.name == FINAL_RESULT_TOOL_NAME]
                                    if finals:
                                        if len(finals) > 1 or len(turn.tool_calls) > 1:
                                            raise UnexpectedModelBehavior("final_result must be the only tool call in its turn")
                                        final = finals[0]
                                        try:
                                            value = self.output_schema.validate_tool_arguments(final.arguments)
                                        except OutputValidationError as exc:
                                            retry_or_fail()
                                            retry_message = _structured_retry_message(str(exc), "Call final_result again with valid arguments.")
                                            final_id = final.id
                                            assert final_id, "tool-mode final_result retry requires a tool call id"
                                            turn = await advance_model(
                                                lambda final_id=final_id, retry_message=retry_message: active_session.continue_with_tools(
                                                    [ToolOutput(final_id, retry_message)],
                                                    tools=self.tool_schemas(),
                                                    metadata=metadata,
                                                    structured_output=structured_output,
                                                ),
                                                output_retry=True,
                                            )
                                            continue
                                        agent_span.set_attribute("gen_ai.completion", turn.text if self.tracing and self.tracing.capture_messages else None)
                                        turn.finalized_output_mode = self.output_schema.mode
                                        finalized_via_output_tool = True
                                        result = build_terminal_result(turn.text, value)
                                        attach_resume_state(active_session)
                                        fire_run_end_once()
                                        return result
                                    if not turn.tool_calls:
                                        retry_or_fail()
                                        retry_message = _structured_retry_message(
                                            "model returned text instead of final_result",
                                            "Call final_result with the final answer.",
                                        )
                                        # There is no final_result call id to answer here, so this correction must be a user message.
                                        turn = await advance_model(lambda retry_message=retry_message: active_session.continue_with_user_message(
                                            retry_message,
                                            tools=self.tool_schemas(),
                                            metadata=metadata,
                                            structured_output=structured_output,
                                        ), output_retry=True)
                                        continue
                                elif not turn.tool_calls:
                                    try:
                                        value = self.output_schema.validate_text(turn.text)
                                    except OutputValidationError as exc:
                                        error_message = str(exc)
                                        retry_or_fail()
                                        retry_message = _structured_retry_message(
                                            error_message,
                                            "Return only valid JSON for the requested schema.",
                                        )
                                        turn = await advance_model(lambda retry_message=retry_message: active_session.continue_with_user_message(
                                            retry_message,
                                            tools=self.tool_schemas(),
                                            metadata=metadata,
                                            structured_output=structured_output,
                                        ), output_retry=True)
                                        continue
                                    agent_span.set_attribute("gen_ai.completion", turn.text if self.tracing and self.tracing.capture_messages else None)
                                    turn.finalized_output_mode = self.output_schema.mode
                                    result = build_terminal_result(turn.text, value)
                                    attach_resume_state(active_session)
                                    fire_run_end_once()
                                    return result
                            if not turn.tool_calls:
                                agent_span.set_attribute("gen_ai.completion", turn.text if self.tracing and self.tracing.capture_messages else None)
                                result = build_terminal_result(turn.text)
                                attach_resume_state(active_session)
                                fire_run_end_once()
                                return result
                            check_tool_limit(len(turn.tool_calls))
                            usage.tool_calls += len(turn.tool_calls)
                            recorded, outputs, executions = await self._execute_tool_batch(run_tracer, turn.tool_calls)
                            usage.cancelled_tool_calls += sum(1 for execution in executions if execution.cancelled)
                            tool_call_records.extend(recorded)
                            check_tool_retry_limits(turn.tool_calls, executions)
                            tool_outputs = outputs
                            turn = await advance_model(lambda tool_outputs=tool_outputs: active_session.continue_with_tools(
                                tool_outputs,
                                tools=self.tool_schemas(),
                                metadata=metadata,
                                structured_output=structured_output,
                            ))
                    except asyncio.CancelledError as exc:
                        stop_reason = "cancelled"
                        terminal_error = exc
                        agent_span.record_exception(exc)
                        agent_span.set_error("run cancelled", "CancelledError")
                        raise
                    except ProviderError as exc:
                        stop_reason = "provider_error"
                        terminal_error = HarnessError(str(exc))
                        agent_span.record_exception(exc)
                        agent_span.set_error(str(exc), type(exc).__name__)
                        raise terminal_error from exc
                    except UnexpectedModelBehavior as exc:
                        stop_reason = "unexpected_model_behavior"
                        terminal_error = terminal_error or exc
                        agent_span.record_exception(exc)
                        agent_span.set_error(str(exc), type(exc).__name__)
                        raise
                    except HarnessError as exc:
                        terminal_error = terminal_error or exc
                        if stop_reason == "end_turn":
                            stop_reason = "error"
                        agent_span.record_exception(exc)
                        agent_span.set_error(str(exc), type(exc).__name__)
                        raise
                    except Exception as exc:
                        stop_reason = "error"
                        terminal_error = exc
                        agent_span.record_exception(exc)
                        agent_span.set_error(str(exc), type(exc).__name__)
                        raise
            finally:
                fire_run_end_once()
        finally:
            self._current_run_metadata = None
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
        """Close provider HTTP clients if this harness owns them."""
        if not self._owns_model:
            return
        if self._closed:
            return
        aclose = getattr(self.model.provider, "aclose", None)
        if aclose is not None:
            await aclose()
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
        skill_summary = self.skills.prompt_summary() if self._skills_enabled else "No skills are configured."
        return f"{self.config.system_prompt}\n\nWorkspace root: {self.root}\n\n{skill_summary}"

    async def _call_output(self, name: str, arguments: str) -> str:
        """Execute one model tool call and format its output."""
        spec = self._tool_map.get(str(name))
        if not spec:
            return json.dumps({"ok": False, "content": f"unknown tool {name}", "metadata": {"tool": name}}, ensure_ascii=False)
        return await _invoke_tool(spec, arguments or {})

    async def _traced_call_output(self, run_tracer: RunTracer, call_id: str, name: str, arguments: str, tool_index: int) -> ToolCallExecution:
        """Execute one model tool call with tracing."""
        with run_tracer.tool(tool_name=name, call_id=call_id, arguments=arguments) as span:
            token = _CURRENT_TOOL_CALL.set({"call_id": call_id, "name": name})
            cancelled = False
            start = time.perf_counter()
            try:
                before = BeforeToolCallContext(
                    harness=self,
                    metadata=dict(self._current_run_metadata or {}),
                    call_id=call_id,
                    tool_name=name,
                    arguments=arguments,
                    tool_spec=self._tool_map.get(str(name)),
                    tool_index=tool_index,
                )
                self.hooks.fire(before)
                if before.cancelled:
                    cancelled = True
                    reason = before.cancel_reason or "unspecified"
                    output = json.dumps({
                        "ok": False,
                        "content": f"Tool execution blocked by hook: {reason}",
                        "metadata": {"error_type": "ToolCallCancelled"},
                    }, ensure_ascii=False)
                else:
                    output = await self._call_output(name, arguments)
                parsed = _parse_tool_output(output)
                retry_kind = None if cancelled else _tool_retry_kind(parsed)
                after = AfterToolCallContext(
                    harness=self,
                    metadata=dict(self._current_run_metadata or {}),
                    call_id=call_id,
                    tool_name=name,
                    arguments=arguments,
                    original_output=output,
                    output=output,
                    parsed_output=parsed,
                    duration_ms=(time.perf_counter() - start) * 1000,
                )
                self.hooks.fire_after_tool_call(after)
                output = after.output
                parsed = _parse_tool_output(output)
                metadata_value = parsed.get("metadata")
                parsed_metadata: dict[str, Any] = metadata_value if isinstance(metadata_value, dict) else {}
                if name == "subagent":
                    span.set_attributes({
                        "subagent.name": parsed_metadata.get("agent"),
                        "subagent.tool_mode": parsed_metadata.get("tool_mode"),
                        "subagent.tools": parsed_metadata.get("tools"),
                    })
                if self.tracing and self.tracing.capture_tool_results:
                    span.set_attribute("gen_ai.tool.call.result", serialize_attribute_value(output))
                if retry_kind is not None:
                    span.set_error(f'Tool "{name}" failed', retry_kind)
                elif parsed.get("ok") is False:
                    span.set_error(f'Tool "{name}" failed', "ToolExecutionError")
                return ToolCallExecution(output=output, cancelled=cancelled, retry_kind=retry_kind)
            finally:
                _CURRENT_TOOL_CALL.reset(token)

    async def _execute_tool_batch(self, run_tracer: RunTracer, calls: list[ModelToolCall]) -> tuple[list[Json], list[ToolOutput], list[ToolCallExecution]]:
        """Run one batch of model tool calls; preserve model order in returned outputs."""
        if self._should_run_sequentially(calls):
            results = [
                await self._traced_call_output(run_tracer, call.id, call.name, call.arguments, index)
                for index, call in enumerate(calls)
            ]
        else:
            results = await self._run_calls_concurrently(run_tracer, calls)
        recorded = []
        for call, execution in zip(calls, results, strict=True):
            record = {"call": {"id": call.id, "name": call.name, "arguments": call.arguments}, "output": execution.output}
            if execution.cancelled:
                record["cancelled"] = True
            recorded.append(record)
        outputs = [ToolOutput(call.id, execution.output) for call, execution in zip(calls, results, strict=True)]
        return recorded, outputs, results

    def _should_run_sequentially(self, calls: list[ModelToolCall]) -> bool:
        """Decide whether the batch must execute serially."""
        if self.config.tool_execution == "sequential" or len(calls) <= 1:
            return True
        return any((spec := self._tool_map.get(str(call.name))) is not None and spec.sequential for call in calls)

    async def _run_calls_concurrently(self, run_tracer: RunTracer, calls: list[ModelToolCall]) -> list[ToolCallExecution]:
        """Execute calls concurrently while preserving model request order."""
        sem = asyncio.Semaphore(MAX_PARALLEL_TOOL_WORKERS)

        async def invoke(index: int, call: ModelToolCall) -> ToolCallExecution:
            """Invoke one traced tool call under the shared concurrency limit."""
            async with sem:
                return await self._traced_call_output(run_tracer, call.id, call.name, call.arguments, index)

        tasks = [asyncio.create_task(invoke(index, call)) for index, call in enumerate(calls)]
        task_index = {task: index for index, task in enumerate(tasks)}
        results: list[ToolCallExecution | None] = [None] * len(tasks)
        pending = set(tasks)
        try:
            while pending:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_EXCEPTION)
                for task in done:
                    exc = task.exception()
                    if exc is not None:
                        for sibling in pending:
                            sibling.cancel()
                        await asyncio.gather(*pending, return_exceptions=True)
                        raise exc
                    results[task_index[task]] = task.result()
        except BaseException:
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            raise
        return [result for result in results if result is not None]

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
        """Validate hook filters against registered tools and subagents."""
        agent_names = {DEFAULT_SUBAGENT_NAME, *(config.name for config in self.config.subagents)}
        self.hooks.validate_filters(tool_names={tool.name for tool in self.tools}, agent_names=agent_names)

    def _build_output_schema(self) -> OutputSchema | None:
        """Build structured-output validation if configured."""
        if self.config.output_type is None:
            return None
        _, mode = resolve_output_spec(self.config.output_type, self.config.output_mode)
        if mode == "auto":
            mode = self.model_capabilities.default_structured_output_mode
        if mode == "native" and not self.model_capabilities.supports_json_schema_output:
            if not self.model_capabilities.permissive_native_override:
                raise ValueError(f"{self.model.provider.name} does not support native structured output")
        if mode == "tool" and not self.model_capabilities.supports_tools:
            raise ValueError(f"{self.model.provider.name} does not support tool structured output")
        return OutputSchema.build(self.config.output_type, mode)

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

    def _finalized_output_mode_for_turn(self, turn: ModelTurn) -> str | None:
        """Return the structured-output mode if a turn successfully finalizes."""
        if self.output_schema is None:
            return None
        if self.output_schema.mode == "text":
            return "text" if not turn.tool_calls else None
        if self.output_schema.mode == "tool":
            finals = [call for call in turn.tool_calls if call.name == FINAL_RESULT_TOOL_NAME]
            if len(finals) != 1 or len(turn.tool_calls) != 1:
                return None
            try:
                self.output_schema.validate_tool_arguments(finals[0].arguments)
            except OutputValidationError:
                return None
            return "tool"
        if turn.tool_calls:
            return None
        try:
            self.output_schema.validate_text(turn.text)
        except OutputValidationError:
            return None
        return self.output_schema.mode

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


def _parse_tool_output(output: str) -> Json:
    """Parse a normalized tool output envelope."""
    try:
        parsed = json.loads(output)
    except json.JSONDecodeError:
        return {"ok": False, "content": output, "metadata": {"error_type": "InvalidToolOutput"}}
    return parsed if isinstance(parsed, dict) else {"ok": False, "content": output, "metadata": {"error_type": "InvalidToolOutput"}}


def _tool_retry_kind(parsed: Json) -> str | None:
    """Return the retry error type from a parsed tool output envelope."""
    metadata_value = parsed.get("metadata")
    metadata: dict[str, Any] = metadata_value if isinstance(metadata_value, dict) else {}
    error_type = metadata.get("error_type")
    if metadata.get("retry") is True and isinstance(error_type, str):
        return error_type
    return None


def _structured_retry_message(error: str, instruction: str) -> str:
    """Build a corrective structured-output retry prompt."""
    return f"The previous response failed structured output validation.\n\n{error}\n\n{instruction}"
