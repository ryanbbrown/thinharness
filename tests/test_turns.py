from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from fakes import ScriptedProvider, ScriptedSession
from pydantic import BaseModel

from thinharness import Harness, HarnessConfig, ModelMessageEvent, RequestConstants, ToolSpec, UnexpectedModelBehavior
from thinharness.approvals import ApprovalPause, ApprovalToolCall
from thinharness.providers import ModelToolCall, ModelTurn, ToolOutput
from thinharness.tracing import RunTracer
from thinharness.turns import OutputTurnDecision, TurnStart, advance_until_terminal
from thinharness.types import ApprovalDecision, HarnessResult, RunUsage


class Person(BaseModel):
    name: str
    age: int


class FakeRunContext:
    """Minimal RunContext double recording the machine's ordered actions."""

    def __init__(self, script: list[tuple[ModelTurn, OutputTurnDecision]]) -> None:
        self.script = list(script)
        self.actions: list[tuple] = []
        self.responses: list[dict] = []
        self.usage = RunUsage()
        self.tracer = RunTracer([])
        self.events: list = []

    async def advance_model(self, request, *, request_kind, structured_output, prompt=None, tool_outputs=None, output_retry=False):
        self.actions.append(("advance_model", request_kind, output_retry))
        return self.script.pop(0)

    def record_response(self, turn: ModelTurn) -> None:
        self.responses.append(turn.raw)

    def finalize(self, text, session, *, output=None, finalized_via_output_tool_value=False, require_dump_state):
        self.actions.append(("finalize", text))
        return HarnessResult(text=text, output=output, responses=self.responses, usage=self.usage)

    def pause_for_approval(self, turn, approval_calls, session):
        self.actions.append(("pause_for_approval", [call.id for call in approval_calls]))
        return HarnessResult(text=turn.text, responses=self.responses, usage=self.usage, stop_reason="approval_required")

    def retry_or_fail(self) -> None:
        self.actions.append(("retry_or_fail",))

    def emit_retry_event(self, retry_kind, message, call_id) -> None:
        self.actions.append(("emit_retry_event", retry_kind, call_id))

    def check_tool_limit(self, batch_size: int) -> None:
        self.actions.append(("check_tool_limit", batch_size))

    def record_tool_batch(self, records) -> None:
        self.actions.append(("record_tool_batch", len(records)))

    def check_tool_retry_limits(self, calls, executions) -> None:
        self.actions.append(("check_tool_retry_limits", len(list(calls))))

    def emit(self, event) -> None:
        self.events.append(event)

    def stream_base(self) -> dict:
        return {"run_id": "run_test", "sequence": 0, "parent_run_id": None, "parent_tool_call_id": None, "agent_name": None}


class FakeHarness:
    """Minimal harness double exposing what the turn machine reads."""

    def __init__(self) -> None:
        self.output_schema = None

    def _model_supports_approval_resume(self) -> bool:
        return True


class FakeToolExecutor:
    """Tool executor double with a frozen tool map and canned outputs."""

    def __init__(self, tool_map: dict | None = None, cancelled_ids: set[str] | None = None) -> None:
        self.tool_map = tool_map or {}
        self.cancelled_ids = cancelled_ids or set()
        self.batches: list[list[str]] = []

    async def execute_batch(self, calls, tool_indices=None):
        self.batches.append([call.id for call in calls])
        records = [{"call": {"id": call.id, "name": call.name, "arguments": call.arguments}, "output": "ok"} for call in calls]
        outputs = [ToolOutput(call.id, "ok") for call in calls]
        executions = [SimpleNamespace(cancelled=call.id in self.cancelled_ids, retry_kind=None) for call in calls]
        return records, outputs, executions


def _approval_spec(name: str) -> ToolSpec:
    """Build an approval-required tool spec."""
    return ToolSpec(name, name, {"type": "object", "properties": {}}, lambda args: "ok", requires_approval=True)


def _plain_spec(name: str) -> ToolSpec:
    """Build a normal tool spec."""
    return ToolSpec(name, name, {"type": "object", "properties": {}}, lambda args: "ok")


CONSTANTS = RequestConstants(instructions="system", tools=[])


async def _run(script, *, kind="start", executor=None, approval_pause=None, approval_decisions=None):
    """Drive the machine over a scripted (turn, decision) sequence."""
    run_ctx = FakeRunContext(script)
    result = await advance_until_terminal(
        TurnStart(kind=kind, prompt="go", approval_pause=approval_pause, approval_decisions=approval_decisions),
        object(),
        CONSTANTS,
        FakeHarness(),
        run_ctx,
        executor or FakeToolExecutor(),
    )
    return run_ctx, result


async def test_final_decision_finalizes_and_records_terminal_turn_once() -> None:
    turn = ModelTurn(text="done", raw={"id": "terminal"})
    run_ctx, result = await _run([(turn, OutputTurnDecision(kind="final", text="done"))])

    assert result.text == "done"
    assert run_ctx.responses == [{"id": "terminal"}]
    assert run_ctx.actions == [("advance_model", "start", False), ("finalize", "done")]


async def test_continue_decision_executes_tools_and_spends_tool_budget() -> None:
    tool_turn = ModelTurn(tool_calls=[ModelToolCall(id="call_1", name="echo", arguments="{}")], raw={"id": "tools"})
    final_turn = ModelTurn(text="done", raw={"id": "done"})
    executor = FakeToolExecutor({"echo": _plain_spec("echo")})
    run_ctx, result = await _run([
        (tool_turn, OutputTurnDecision(kind="continue")),
        (final_turn, OutputTurnDecision(kind="final", text="done")),
    ], executor=executor)

    assert run_ctx.usage.tool_calls == 1
    assert run_ctx.responses == [{"id": "tools"}, {"id": "done"}]
    assert run_ctx.actions == [
        ("advance_model", "start", False),
        ("check_tool_limit", 1),
        ("record_tool_batch", 1),
        ("check_tool_retry_limits", 1),
        ("advance_model", "tool_outputs", False),
        ("finalize", "done"),
    ]


async def test_retry_tool_output_decision_spends_retry_budget_before_request() -> None:
    bad_turn = ModelTurn(tool_calls=[ModelToolCall(id="call_final", name="final_result", arguments="{}")], raw={"id": "bad"})
    final_turn = ModelTurn(text="done", raw={"id": "good"})
    run_ctx, _result = await _run([
        (bad_turn, OutputTurnDecision(kind="retry_tool_output", retry_message="fix it", retry_call_id="call_final")),
        (final_turn, OutputTurnDecision(kind="final", text="done")),
    ])

    assert run_ctx.actions == [
        ("advance_model", "start", False),
        ("retry_or_fail",),
        ("emit_retry_event", "structured_output", "call_final"),
        ("advance_model", "output_retry_tool", True),
        ("finalize", "done"),
    ]
    assert run_ctx.responses == [{"id": "bad"}, {"id": "good"}]


async def test_retry_user_message_decision_sends_correction() -> None:
    bad_turn = ModelTurn(text="not json", raw={"id": "bad"})
    final_turn = ModelTurn(text="done", raw={"id": "good"})
    run_ctx, _result = await _run([
        (bad_turn, OutputTurnDecision(kind="retry_user_message", retry_message="fix it")),
        (final_turn, OutputTurnDecision(kind="final", text="done")),
    ])

    assert run_ctx.actions == [
        ("advance_model", "start", False),
        ("retry_or_fail",),
        ("emit_retry_event", "structured_output", None),
        ("advance_model", "correction", True),
        ("finalize", "done"),
    ]


async def test_unexpected_decision_raises_after_recording_response() -> None:
    turn = ModelTurn(raw={"id": "weird"})
    run_ctx = FakeRunContext([(turn, OutputTurnDecision(kind="unexpected", unexpected_message="bad pattern"))])

    with pytest.raises(UnexpectedModelBehavior, match="bad pattern"):
        await advance_until_terminal(TurnStart(kind="start", prompt="go"), object(), CONSTANTS, FakeHarness(), run_ctx, FakeToolExecutor())

    assert run_ctx.responses == [{"id": "weird"}]


async def test_approval_pause_counts_full_batch_and_records_paused_turn() -> None:
    turn = ModelTurn(
        tool_calls=[
            ModelToolCall(id="call_1", name="deploy", arguments="{}"),
            ModelToolCall(id="call_2", name="echo", arguments="{}"),
        ],
        raw={"id": "paused"},
    )
    executor = FakeToolExecutor({"deploy": _approval_spec("deploy"), "echo": _plain_spec("echo")})
    run_ctx, result = await _run([(turn, OutputTurnDecision(kind="continue"))], executor=executor)

    assert result.stop_reason == "approval_required"
    assert run_ctx.usage.tool_calls == 2
    assert run_ctx.responses == [{"id": "paused"}]
    assert run_ctx.actions == [
        ("advance_model", "start", False),
        ("check_tool_limit", 2),
        ("pause_for_approval", ["call_1"]),
    ]


async def test_continue_decision_counts_hook_cancelled_tools() -> None:
    tool_turn = ModelTurn(tool_calls=[ModelToolCall(id="call_1", name="echo", arguments="{}")], raw={"id": "tools"})
    final_turn = ModelTurn(text="done", raw={"id": "done"})
    executor = FakeToolExecutor({"echo": _plain_spec("echo")}, cancelled_ids={"call_1"})
    run_ctx, _result = await _run([
        (tool_turn, OutputTurnDecision(kind="continue")),
        (final_turn, OutputTurnDecision(kind="final", text="done")),
    ], executor=executor)

    assert run_ctx.usage.cancelled_tool_calls == 1


async def test_resume_start_kind_uses_user_text_request() -> None:
    turn = ModelTurn(text="done", raw={"id": "resumed"})
    run_ctx, _result = await _run([(turn, OutputTurnDecision(kind="final", text="done"))], kind="resume")

    assert run_ctx.actions[0] == ("advance_model", "resume", False)


def _pause(batch: list[ApprovalToolCall], required: set[str]) -> ApprovalPause:
    """Build a minimal approval pause envelope value."""
    return ApprovalPause(
        provider_state={},
        batch=batch,
        approval_required_ids=frozenset(required),
        usage=RunUsage(),
        responses=[],
        tool_call_records=[],
        emitted_limit_warnings=set(),
        metadata={},
    )


async def test_approval_resume_replay_does_not_recount_tool_calls() -> None:
    pause = _pause(
        [ApprovalToolCall(id="call_1", name="deploy", arguments="{}"), ApprovalToolCall(id="call_2", name="echo", arguments="{}")],
        {"call_1"},
    )
    decisions = {"call_1": ApprovalDecision(call_id="call_1", approved=True)}
    executor = FakeToolExecutor({"deploy": _approval_spec("deploy"), "echo": _plain_spec("echo")})
    run_ctx, result = await _run(
        [(ModelTurn(text="done", raw={"id": "done"}), OutputTurnDecision(kind="final", text="done"))],
        kind="approval_resume",
        executor=executor,
        approval_pause=pause,
        approval_decisions=decisions,
    )

    assert result.text == "done"
    # The batch was counted at pause time: the replay neither re-checks the tool
    # limit nor re-counts usage.tool_calls.
    assert run_ctx.usage.tool_calls == 0
    assert ("check_tool_limit", 2) not in run_ctx.actions
    assert executor.batches == [["call_1", "call_2"]]
    assert run_ctx.actions == [
        ("check_tool_retry_limits", 2),
        ("record_tool_batch", 2),
        ("advance_model", "approval_resume", False),
        ("finalize", "done"),
    ]


async def test_approval_resume_counts_cancelled_tools_and_rejections() -> None:
    pause = _pause(
        [ApprovalToolCall(id="call_1", name="deploy", arguments="{}"), ApprovalToolCall(id="call_2", name="echo", arguments="{}")],
        {"call_1"},
    )
    decisions = {"call_1": ApprovalDecision(call_id="call_1", approved=False, reason="too risky")}
    executor = FakeToolExecutor({"deploy": _approval_spec("deploy"), "echo": _plain_spec("echo")}, cancelled_ids={"call_2"})
    run_ctx, _result = await _run(
        [(ModelTurn(text="done", raw={"id": "done"}), OutputTurnDecision(kind="final", text="done"))],
        kind="approval_resume",
        executor=executor,
        approval_pause=pause,
        approval_decisions=decisions,
    )

    # Only the approved call executes; the hook-cancelled execution is counted.
    assert executor.batches == [["call_2"]]
    assert run_ctx.usage.cancelled_tool_calls == 1
    rejection_events = [event for event in run_ctx.events if getattr(event, "error_type", None) == "ApprovalRejected"]
    assert [event.call_id for event in rejection_events] == ["call_1"]


class _ScriptedResumeModel:
    """Small resumable model for machine-level resume tests."""

    resume_kind = "scripted"

    def __init__(self, sessions: list) -> None:
        self.model = "scripted"
        self.provider = ScriptedProvider()
        self.api_key = "key"
        self.sessions = list(sessions)

    def new_session(self):
        """Return the next scripted session."""
        return self.sessions.pop(0)

    def resume_session(self, state):
        """Return the next scripted session for a resumed run."""
        return self.sessions.pop(0)


def test_correction_following_resume_uses_same_session(tmp_path: Path) -> None:
    first_session = ScriptedSession(
        start_turn=ModelTurn(text='{"name":"Ada","age":37}', raw={"id": "first"}),
        dump_state={"kind": "scripted", "version": 1, "model": "scripted"},
    )
    resumed_session = ScriptedSession(
        start_turn=ModelTurn(text="not json", raw={"id": "resumed-bad"}),
        continue_turn=ModelTurn(text='{"name":"Ada","age":37}', raw={"id": "corrected"}),
    )
    model = _ScriptedResumeModel([first_session, resumed_session])
    first = Harness(HarnessConfig(root=tmp_path, builtin_tools=[]), model=model).run_sync("first")

    resumed = Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[], output_type=Person, output_mode="prompted"),
        model=model,
    ).run_sync("follow-up", resume_from=first.resume_state)

    assert resumed.output == Person(name="Ada", age=37)
    assert resumed.usage.output_retries == 1
    # The resumed session answers the resume prompt first, then the correction
    # lands on the same session as a continuation.
    assert [method for method, _notices in resumed_session.notice_calls] == [
        "continue_with_user_text",
        "continue_with_user_text",
    ]
    assert resumed.responses == [{"id": "resumed-bad"}, {"id": "corrected"}]


async def test_model_message_event_finalized_output_mode_populated(tmp_path: Path) -> None:
    from fakes import ScriptedModel

    session = ScriptedSession(
        start_turn=ModelTurn(
            tool_calls=[ModelToolCall(id="call_final", name="final_result", arguments='{"name":"Ada","age":37}')],
            raw={"id": "final"},
        ),
    )
    harness = Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[], output_type=Person, output_mode="tool"),
        model=ScriptedModel([session]),
    )

    events = []
    async for event in harness.stream("make a person"):
        events.append(event)

    model_messages = [event for event in events if isinstance(event, ModelMessageEvent)]
    assert [event.finalized_output_mode for event in model_messages] == ["tool"]
