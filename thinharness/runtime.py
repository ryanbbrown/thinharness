"""Internal per-run context and model advancement ceremony."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from .hooks import LimitReachedContext, RunEndContext
from .output import OutputTurnDecision, resolve_turn_output
from .providers import ModelNotice, ModelSession, ModelTurn
from .tracing import (
    ModelTraceSnapshot,
    RunTracer,
    annotate_agent_result,
    annotate_model_request,
    annotate_model_span,
)

if TYPE_CHECKING:
    from .core import Harness, HarnessResult, LimitNoticeKey, RunUsage, StopReason
    from .tool_execution import BackgroundToolCompletion, BackgroundToolManager
    from .tools.base import Json
    from .tracing import _TraceSpan


ModelRequest = Callable[[list[ModelNotice]], Awaitable[ModelTurn]]


class _ToolRetryCall(Protocol):
    """Minimal model tool call shape needed for retry accounting."""

    @property
    def name(self) -> str:
        """Return the model-requested tool name."""
        ...


class _ToolRetryExecution(Protocol):
    """Minimal tool execution shape needed for retry accounting."""

    @property
    def retry_kind(self) -> str | None:
        """Return the retryable error kind, if any."""
        ...

    @property
    def cancelled(self) -> bool:
        """Return whether the tool call was cancelled by a hook."""
        ...


@dataclass
class RunContext:
    """Mutable state for one harness run."""

    harness: Harness
    prompt: str
    metadata: Json
    usage: RunUsage
    responses: list[Json] = field(default_factory=list)
    tool_call_records: list[Json] = field(default_factory=list)
    emitted_limit_warnings: set[LimitNoticeKey] = field(default_factory=set)
    tracer: RunTracer | None = None
    result: HarnessResult | None = None
    terminal_error: BaseException | None = None
    stop_reason: StopReason = "end_turn"
    run_end_fired: bool = False
    finalized_via_output_tool: bool = False
    agent_span: _TraceSpan | None = None
    background: BackgroundToolManager | None = None

    def fire_run_end_once(self) -> None:
        """Emit run_end exactly once for this run."""
        if self.run_end_fired:
            return
        self.run_end_fired = True
        self.harness.hooks.fire(RunEndContext(
            harness=self.harness,
            metadata=dict(self.metadata),
            result=self.result,
            error=self.terminal_error,
            stop_reason=self.stop_reason,
            usage=self.usage,
        ))

    def check_model_limit(self) -> None:
        """Raise if another provider request would exceed the configured limit."""
        from .core import HarnessError

        if self.usage.model_requests < self.harness.config.max_model_requests:
            return
        self.harness.hooks.fire(LimitReachedContext(
            harness=self.harness,
            metadata=dict(self.metadata),
            limit_kind="model_requests",
            limit_value=self.harness.config.max_model_requests,
            current_count=self.usage.model_requests,
        ))
        self.stop_reason = "limit_reached"
        self.terminal_error = HarnessError(f"model did not finish within max_model_requests={self.harness.config.max_model_requests}")
        raise self.terminal_error

    def check_tool_limit(self, batch_size: int) -> None:
        """Raise if a requested tool batch would exceed the configured limit."""
        from .core import HarnessError

        max_tool_calls = self.harness.config.max_tool_calls
        if max_tool_calls is None or self.usage.tool_calls + batch_size <= max_tool_calls:
            return
        self.harness.hooks.fire(LimitReachedContext(
            harness=self.harness,
            metadata=dict(self.metadata),
            limit_kind="tool_calls",
            limit_value=max_tool_calls,
            current_count=self.usage.tool_calls + batch_size,
        ))
        self.stop_reason = "limit_reached"
        self.terminal_error = HarnessError(f"tool calls would exceed max_tool_calls={max_tool_calls}")
        raise self.terminal_error

    async def advance_model(
        self,
        request: ModelRequest,
        *,
        trace_snapshot: ModelTraceSnapshot,
        output_retry: bool = False,
    ) -> tuple[ModelTurn, OutputTurnDecision]:
        """Run one provider request with limit, usage, and tracing ceremony."""
        from .core import _compute_limit_notices

        assert self.tracer is not None
        self.check_model_limit()
        notices = _compute_limit_notices(
            self.harness.config,
            self.usage,
            self.emitted_limit_warnings,
            final_result_tool_available=self.harness.output_schema is not None and self.harness.output_schema.mode == "tool",
        )
        if output_retry:
            self.usage.output_retries += 1
        with self.tracer.model(self.harness.model) as model_span:
            snapshot = trace_snapshot.with_notices(notices)
            model_span.for_each(
                lambda span, option: annotate_model_request(
                    span,
                    snapshot,
                    capture_messages=option.capture_messages,
                )
            )
            try:
                turn = await request(notices)
                self.usage.model_requests += 1
            except Exception as exc:
                model_span.record_exception(exc)
                model_span.set_error(str(exc), type(exc).__name__)
                raise
            model_span.for_each(
                lambda span, option: annotate_model_span(
                    span,
                    turn,
                    capture_messages=option.capture_messages,
                )
            )
            decision = resolve_turn_output(turn, self.harness.output_schema)
            if finalized_mode := decision.finalized_mode:
                turn.finalized_output_mode = finalized_mode
                model_span.set_attributes({
                    "thinharness.output.mode": finalized_mode,
                    "gen_ai.output.finalized": True,
                })
            return turn, decision

    def build_terminal_result(self, text: str, output: Any | None = None) -> HarnessResult:
        """Create the terminal HarnessResult for this run."""
        from .core import HarnessResult

        return HarnessResult(
            text=text,
            output=output,
            responses=self.responses,
            tool_call_records=self.tool_call_records,
            usage=self.usage,
            stop_reason=self.stop_reason,
        )

    def attach_resume_state(self, session: ModelSession, *, require_dump_state: bool) -> None:
        """Attach final resume state before run_end hooks observe the result."""
        from .core import _build_resume_state

        if self.result is not None:
            self.result.resume_state = _build_resume_state(
                session,
                self.stop_reason,
                self.finalized_via_output_tool,
                require_dump_state,
            )

    def finalize(
        self,
        text: str,
        active_session: ModelSession,
        *,
        output: Any | None = None,
        finalized_via_output_tool_value: bool = False,
        require_dump_state: bool,
    ) -> HarnessResult:
        """Run terminal bookkeeping for one successful run."""
        assert self.agent_span is not None
        if finalized_via_output_tool_value:
            self.finalized_via_output_tool = True
        self.result = self.build_terminal_result(text, output)
        self.agent_span.for_each(
            lambda span, option: annotate_agent_result(
                span,
                result=self.result,
                output_schema=self.harness.output_schema,
                capture_messages=option.capture_messages,
                top_level=not self.harness._is_child_run,
            )
        )
        self.attach_resume_state(active_session, require_dump_state=require_dump_state)
        self.fire_run_end_once()
        return self.result

    def retry_or_fail(self) -> None:
        """Track one structured-output validation retry or fail the run."""
        from .core import HarnessError

        if self.usage.output_retries >= self.harness.config.output_retries:
            self.stop_reason = "output_validation_failed"
            self.terminal_error = HarnessError("output validation exceeded output_retries")
            raise self.terminal_error

    def check_tool_retry_limits(self, calls: Sequence[_ToolRetryCall], executions: Sequence[_ToolRetryExecution]) -> None:
        """Track retryable tool failures and raise if any tool exceeds its budget."""
        from .core import HarnessError

        for call, execution in zip(calls, executions, strict=True):
            if execution.retry_kind is None or execution.cancelled:
                continue
            self.usage.tool_retries[call.name] = self.usage.tool_retries.get(call.name, 0) + 1
            max_retries = self.harness._tool_max_retries(call.name)
            if self.usage.tool_retries[call.name] > max_retries:
                self.harness.hooks.fire(LimitReachedContext(
                    harness=self.harness,
                    metadata=dict(self.metadata),
                    limit_kind="tool_retries",
                    limit_value=max_retries,
                    current_count=self.usage.tool_retries[call.name],
                ))
                self.stop_reason = "tool_retries_exceeded"
                self.terminal_error = HarnessError(f"tool {call.name!r} exceeded max_retries={max_retries}")
                raise self.terminal_error

    def record_tool_batch(self, records: list[Json]) -> None:
        """Append provider-facing tool call records to this run."""
        self.tool_call_records.extend(records)

    def record_background_completion(self, completion: BackgroundToolCompletion) -> None:
        """Append a background completion record to this run."""
        self.tool_call_records.append(completion.record())
