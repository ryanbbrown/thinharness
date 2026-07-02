"""Turn state machine driving harness runs to a terminal result."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from .events import ToolCallCompletedEvent, ToolCallStartedEvent
from .output import FINAL_RESULT_TOOL_NAME, OutputSchema, OutputValidationError, ResolvedOutputMode
from .providers import ModelSession, ModelToolCall, ModelTurn, RequestConstants, ToolOutput
from .tools.base import ToolResult
from .tracing import _trace_output_mode, serialize_attribute_value
from .types import ApprovalDecision, HarnessError, HarnessResult, Json, UnexpectedModelBehavior

if TYPE_CHECKING:
    from .approvals import ApprovalPause
    from .core import Harness
    from .runtime import RunContext
    from .tool_execution import ToolBatchExecutor


@dataclass(frozen=True)
class OutputTurnDecision:
    """Resolved meaning of one model turn against an optional output schema."""

    kind: Literal["continue", "final", "retry_user_message", "retry_tool_output", "unexpected"]
    finalized_mode: ResolvedOutputMode | None = None
    finalized_via_output_tool: bool = False
    text: str = ""
    output: Any | None = None
    retry_message: str = ""
    retry_call_id: str | None = None
    final_tool_call_id: str | None = None
    error: OutputValidationError | None = None
    unexpected_message: str = ""


@dataclass(frozen=True)
class TurnStart:
    """First-turn production input for one run."""

    kind: Literal["start", "resume", "approval_resume"]
    prompt: str = ""
    approval_pause: ApprovalPause | None = None
    approval_decisions: dict[str, ApprovalDecision] | None = None


def resolve_turn_output(turn: ModelTurn, output_schema: OutputSchema | None) -> OutputTurnDecision:
    """Resolve the control-flow meaning of a model turn."""
    if output_schema is None:
        if turn.tool_calls:
            return OutputTurnDecision(kind="continue")
        return OutputTurnDecision(kind="final", text=turn.text)
    if output_schema.mode == "text":
        if turn.tool_calls:
            return OutputTurnDecision(kind="continue")
        value = output_schema.validate_text(turn.text)
        return OutputTurnDecision(kind="final", finalized_mode="text", text=turn.text, output=value)
    if output_schema.mode == "tool":
        finals = [call for call in turn.tool_calls if call.name == FINAL_RESULT_TOOL_NAME]
        if finals:
            if len(finals) > 1 or len(turn.tool_calls) > 1:
                return OutputTurnDecision(kind="unexpected", unexpected_message="final_result must be the only tool call in its turn")
            final = finals[0]
            try:
                value = output_schema.validate_tool_arguments(final.arguments)
            except OutputValidationError as exc:
                return OutputTurnDecision(
                    kind="retry_tool_output",
                    retry_message=_structured_retry_message(str(exc), "Call final_result again with valid arguments."),
                    retry_call_id=final.id,
                    error=exc,
                )
            return OutputTurnDecision(
                kind="final",
                finalized_mode="tool",
                finalized_via_output_tool=True,
                final_tool_call_id=final.id,
                text=turn.text,
                output=value,
            )
        if turn.tool_calls:
            return OutputTurnDecision(kind="continue")
        error = OutputValidationError("model returned text instead of final_result")
        return OutputTurnDecision(
            kind="retry_user_message",
            retry_message=_structured_retry_message(str(error), "Call final_result with the final answer."),
            error=error,
        )
    if turn.tool_calls:
        return OutputTurnDecision(kind="continue")
    try:
        value = output_schema.validate_text(turn.text)
    except OutputValidationError as exc:
        return OutputTurnDecision(
            kind="retry_user_message",
            retry_message=_structured_retry_message(str(exc), "Return only valid JSON for the requested schema."),
            error=exc,
        )
    return OutputTurnDecision(kind="final", finalized_mode=output_schema.mode, text=turn.text, output=value)


def _structured_retry_message(error: str, instruction: str) -> str:
    """Build a corrective structured-output retry prompt."""
    return f"The previous response failed structured output validation.\n\n{error}\n\n{instruction}"


async def advance_until_terminal(
    start: TurnStart,
    session: ModelSession,
    constants: RequestConstants,
    harness: Harness,
    run_ctx: RunContext,
    tool_executor: ToolBatchExecutor,
) -> HarnessResult:
    """Drive one run through model turns to a terminal result or approval pause.

    Raises on failure; the caller owns exception classification and spans.
    """
    output_mode = _trace_output_mode(harness.output_schema)
    require_dump_state = harness._model_supports_approval_resume()

    async def send_user_text(
        text: str,
        *,
        kind: Literal["resume", "correction"],
        output_retry: bool = False,
    ) -> tuple[ModelTurn, OutputTurnDecision]:
        """Continue the run with user text."""
        return await run_ctx.advance_model(
            lambda notices: session.continue_with_user_text(text, constants, notices=notices),
            request_kind=kind,
            prompt=text,
            structured_output=output_mode,
            output_retry=output_retry,
        )

    async def send_tool_outputs(
        outputs: list[ToolOutput],
        *,
        kind: Literal["tool_outputs", "approval_resume", "output_retry_tool"] = "tool_outputs",
        output_retry: bool = False,
    ) -> tuple[ModelTurn, OutputTurnDecision]:
        """Continue the run with tool outputs."""
        return await run_ctx.advance_model(
            lambda notices: session.continue_with_tools(outputs, constants, notices=notices),
            request_kind=kind,
            tool_outputs=outputs,
            structured_output=output_mode,
            output_retry=output_retry,
        )

    if start.kind == "start":
        turn, decision = await run_ctx.advance_model(
            lambda notices: session.start(start.prompt, constants, notices=notices),
            request_kind="start",
            prompt=start.prompt,
            structured_output=output_mode,
        )
    elif start.kind == "resume":
        turn, decision = await send_user_text(start.prompt, kind="resume")
    else:
        assert start.approval_pause is not None
        assert start.approval_decisions is not None
        outputs = await _resolve_approval_batch(start.approval_pause, start.approval_decisions, harness, run_ctx, tool_executor)
        turn, decision = await send_tool_outputs(outputs, kind="approval_resume")

    while True:
        run_ctx.record_response(turn)
        if decision.kind == "final":
            return run_ctx.finalize(
                decision.text,
                session,
                output=decision.output,
                finalized_via_output_tool_value=decision.finalized_via_output_tool,
                require_dump_state=require_dump_state,
            )
        if decision.kind == "retry_tool_output":
            run_ctx.retry_or_fail()
            final_id = decision.retry_call_id
            assert final_id, "tool-mode final_result retry requires a tool call id"
            retry_message = decision.retry_message
            run_ctx.emit_retry_event("structured_output", retry_message, final_id)
            turn, decision = await send_tool_outputs(
                [ToolOutput(final_id, retry_message)],
                kind="output_retry_tool",
                output_retry=True,
            )
            continue
        if decision.kind == "retry_user_message":
            run_ctx.retry_or_fail()
            retry_message = decision.retry_message
            run_ctx.emit_retry_event("structured_output", retry_message, decision.retry_call_id)
            turn, decision = await send_user_text(retry_message, kind="correction", output_retry=True)
            continue
        if decision.kind == "unexpected":
            raise UnexpectedModelBehavior(decision.unexpected_message)
        assert decision.kind == "continue"
        approval_calls = _approval_required_calls(harness, turn.tool_calls)
        if approval_calls:
            run_ctx.check_tool_limit(len(turn.tool_calls))
            run_ctx.usage.tool_calls += len(turn.tool_calls)
            return run_ctx.pause_for_approval(turn, approval_calls, session)
        run_ctx.check_tool_limit(len(turn.tool_calls))
        run_ctx.usage.tool_calls += len(turn.tool_calls)
        recorded, outputs, executions = await tool_executor.execute_batch(turn.tool_calls)
        run_ctx.usage.cancelled_tool_calls += sum(1 for execution in executions if execution.cancelled)
        run_ctx.record_tool_batch(recorded)
        run_ctx.check_tool_retry_limits(turn.tool_calls, executions)
        turn, decision = await send_tool_outputs(outputs)


async def _resolve_approval_batch(
    approval_pause: ApprovalPause,
    decisions: dict[str, ApprovalDecision],
    harness: Harness,
    run_ctx: RunContext,
    tool_executor: ToolBatchExecutor,
) -> list[ToolOutput]:
    """Execute or reject the paused batch and return its ordered model outputs.

    The batch's tool calls were counted against limits at pause time, so this
    replay does not re-run check_tool_limit or re-count usage.tool_calls.
    """
    _validate_approval_resume_tools(harness, approval_pause)
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
            record, output = _reject_approval_call(call, index, decision, run_ctx)
            ordered_records.append(record)
            ordered_outputs.append(output)
            continue
        record, output, _execution = executed_by_id[call.id]
        ordered_records.append(record)
        ordered_outputs.append(output)
    run_ctx.record_tool_batch(ordered_records)
    return ordered_outputs


def _reject_approval_call(
    call: ModelToolCall,
    index: int,
    decision: ApprovalDecision,
    run_ctx: RunContext,
) -> tuple[Json, ToolOutput]:
    """Create model-visible rejection output without running hooks."""
    message = "Tool call was rejected by a human reviewer."
    if decision.reason:
        message = f"{message}\nReason: {decision.reason}"
    output = ToolResult(False, message, {"error_type": "ApprovalRejected"}).to_json()
    start = time.perf_counter()
    assert run_ctx.tracer is not None
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


def _validate_approval_resume_tools(harness: Harness, approval_pause: ApprovalPause) -> None:
    """Require approval-required tools to still be configured before execution."""
    current_required_ids: set[str] = set()
    for call in approval_pause.batch:
        spec = harness._tool_map.get(str(call.name))
        if call.id in approval_pause.approval_required_ids and spec is None:
            raise HarnessError(f"approval-required tool {call.name!r} is not configured")
        if spec is not None and spec.requires_approval:
            current_required_ids.add(call.id)
    if current_required_ids != set(approval_pause.approval_required_ids):
        raise HarnessError("approval state approval_required_ids do not match configured approval-required tools")


def _approval_required_calls(harness: Harness, calls: list[ModelToolCall]) -> list[ModelToolCall]:
    """Return approval-required calls from a model batch."""
    return [
        call for call in calls
        if (spec := harness._tool_map.get(str(call.name))) is not None and spec.requires_approval
    ]
