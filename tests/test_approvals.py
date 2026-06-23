from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from fakes import ContextFakeTracer, FakeAnthropicProvider, FakeClient, FakeOpenRouterProvider, ScriptedModel, ScriptedSession, _fake_openai
from pydantic import BaseModel

from thinharness import (
    AnthropicMessagesModel,
    ApprovalDecision,
    ApprovalResumedEvent,
    BackgroundTaskCompletedEvent,
    Harness,
    HarnessConfig,
    HarnessError,
    Hook,
    ModelRequestStartedEvent,
    ModelToolCall,
    ModelTurn,
    OpenRouterModel,
    PendingApproval,
    RunCompletedEvent,
    RunStartedEvent,
    SubAgentConfig,
    ToolCallCompletedEvent,
    ToolCallStartedEvent,
    ToolSpec,
    TracingOptions,
)
from thinharness.approvals import validate_approval_pause_state
from thinharness.tools.base import ToolResult


def approval_tool(called: list[dict] | None = None, *, name: str = "deploy") -> ToolSpec:
    """Create an approval-required test tool."""
    sink = called if called is not None else []
    return ToolSpec(
        name,
        "Deploy something.",
        {"type": "object", "properties": {"env": {"type": "string"}}, "required": ["env"]},
        lambda args: sink.append(args) or f"deployed {args['env']}",
        requires_approval=True,
    )


def echo_tool(called: list[dict] | None = None, *, name: str = "echo") -> ToolSpec:
    """Create a normal test tool."""
    sink = called if called is not None else []
    return ToolSpec(
        name,
        "Echo input.",
        {"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]},
        lambda args: sink.append(args) or args["value"],
    )


def approval_echo_tool(called: list[dict] | None = None) -> ToolSpec:
    """Create an approval-required echo test tool."""
    sink = called if called is not None else []
    return ToolSpec(
        "echo",
        "Echo input.",
        {"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]},
        lambda args: sink.append(args) or args["value"],
        requires_approval=True,
    )


async def test_approval_required_tool_pauses_without_executing_or_hooks(tmp_path: Path) -> None:
    called: list[dict] = []
    hook_calls: list[str] = []
    run_end_stop_reasons: list[str] = []
    session = ScriptedSession(
        start_turn=ModelTurn(
            text="I need to deploy.",
            tool_calls=[ModelToolCall(id="call_1", name="deploy", arguments='{"env":"prod"}')],
            raw={"id": "start"},
        ),
    )
    harness = Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[]),
        model=ScriptedModel([session]),
        tools=[approval_tool(called)],
        hooks=[
            Hook("before_tool_call", lambda ctx: hook_calls.append(ctx.tool_name)),
            Hook("run_end", lambda ctx: run_end_stop_reasons.append(ctx.stop_reason)),
        ],
    )

    result = await harness.run("deploy")

    assert result.stop_reason == "approval_required"
    assert result.text == "I need to deploy."
    assert called == []
    assert hook_calls == []
    assert run_end_stop_reasons == ["approval_required"]
    assert result.pending_approvals == [PendingApproval(call_id="call_1", tool_name="deploy", arguments='{"env":"prod"}')]
    assert result.pending_approvals[0].call_id == "call_1"
    assert result.pending_approvals[0].tool_name == "deploy"
    assert result.pending_approvals[0].arguments == '{"env":"prod"}'
    assert result.resume_state is not None
    assert result.resume_state["kind"] == "approval_pause"
    assert result.resume_state["batch"] == [{"id": "call_1", "name": "deploy", "arguments": '{"env":"prod"}'}]
    assert result.resume_state["approval_required_ids"] == ["call_1"]
    assert result.resume_state["responses"] == [{"id": "start"}]
    assert result.usage.tool_calls == 1


async def test_approval_resume_approve_executes_original_call_and_finishes(tmp_path: Path) -> None:
    called: list[dict] = []
    first = ScriptedSession(
        start_turn=ModelTurn(
            tool_calls=[ModelToolCall(id="call_1", name="deploy", arguments='{"env":"prod"}')],
            raw={"id": "start"},
        ),
    )
    resumed = ScriptedSession(start_turn=ModelTurn(raw={"unused": True}), continue_turn=ModelTurn(text="done", raw={"id": "done"}))
    model = ScriptedModel([first, resumed])
    harness = Harness(HarnessConfig(root=tmp_path, builtin_tools=[]), model=model, tools=[approval_tool(called)])

    paused = await harness.run("deploy")
    result = await harness.resume_approvals(
        json.loads(json.dumps(paused.resume_state)),
        [ApprovalDecision(call_id="call_1", approved=True)],
    )

    assert result.stop_reason == "end_turn"
    assert result.text == "done"
    assert called == [{"env": "prod"}]
    assert result.responses == [{"id": "start"}, {"id": "done"}]
    assert result.usage.tool_calls == 1
    assert resumed.notice_calls[0][0] == "continue_with_tools"
    outputs = resumed.on_continue or None
    assert outputs is None
    record = result.tool_call_records[-1]
    assert record["call"]["id"] == "call_1"
    assert record["approval"] == {"approved": True}
    assert json.loads(record["output"])["content"] == "deployed prod"


def test_resume_approvals_sync_success_on_fresh_harness(tmp_path: Path) -> None:
    called: list[dict] = []
    first = ScriptedSession(
        start_turn=ModelTurn(
            tool_calls=[ModelToolCall(id="call_1", name="deploy", arguments='{"env":"prod"}')],
            raw={"id": "start"},
        ),
    )
    resumed = ScriptedSession(start_turn=ModelTurn(raw={"unused": True}), continue_turn=ModelTurn(text="done", raw={"id": "done"}))
    model = ScriptedModel([first, resumed])
    paused = Harness(HarnessConfig(root=tmp_path, builtin_tools=[]), model=model, tools=[approval_tool(called)]).run_sync("deploy")

    result = Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[]),
        model=model,
        tools=[approval_tool(called)],
    ).resume_approvals_sync(
        json.loads(json.dumps(paused.resume_state)),
        [ApprovalDecision(call_id="call_1", approved=True)],
    )

    assert result.text == "done"
    assert called == [{"env": "prod"}]


async def test_approval_resume_reject_sends_model_visible_rejection(tmp_path: Path) -> None:
    seen_outputs = []
    first = ScriptedSession(
        start_turn=ModelTurn(
            tool_calls=[ModelToolCall(id="call_1", name="deploy", arguments='{"env":"prod"}')],
            raw={"id": "start"},
        ),
    )
    resumed = ScriptedSession(
        start_turn=ModelTurn(raw={"unused": True}),
        continue_turn=ModelTurn(text="not deployed", raw={"id": "done"}),
        on_continue=lambda outputs, _tools, _metadata: seen_outputs.extend(outputs),
    )
    harness = Harness(HarnessConfig(root=tmp_path, builtin_tools=[]), model=ScriptedModel([first, resumed]), tools=[approval_tool([])])

    paused = await harness.run("deploy")
    result = await harness.resume_approvals(
        paused.resume_state,
        [ApprovalDecision(call_id="call_1", approved=False, reason="freeze")],
    )

    assert result.text == "not deployed"
    assert result.usage.tool_retries == {}
    parsed = json.loads(seen_outputs[0].output)
    assert parsed == {
        "ok": False,
        "content": "Tool call was rejected by a human reviewer.\nReason: freeze",
        "metadata": {"error_type": "ApprovalRejected"},
    }
    record = result.tool_call_records[-1]
    assert record["approval"] == {"approved": False, "reason": "freeze"}


async def test_approval_resume_reject_without_reason_omits_reason_line(tmp_path: Path) -> None:
    seen_outputs = []
    first = ScriptedSession(
        start_turn=ModelTurn(
            tool_calls=[ModelToolCall(id="call_1", name="deploy", arguments='{"env":"prod"}')],
            raw={"id": "start"},
        ),
    )
    resumed = ScriptedSession(
        start_turn=ModelTurn(raw={"unused": True}),
        continue_turn=ModelTurn(text="not deployed", raw={"id": "done"}),
        on_continue=lambda outputs, _tools, _metadata: seen_outputs.extend(outputs),
    )
    harness = Harness(HarnessConfig(root=tmp_path, builtin_tools=[]), model=ScriptedModel([first, resumed]), tools=[approval_tool([])])

    paused = await harness.run("deploy")
    result = await harness.resume_approvals(paused.resume_state, [ApprovalDecision(call_id="call_1", approved=False)])

    assert result.text == "not deployed"
    assert json.loads(seen_outputs[0].output)["content"] == "Tool call was rejected by a human reviewer."


async def test_mixed_batch_pauses_everything_and_resumes_in_model_order(tmp_path: Path) -> None:
    normal_called: list[dict] = []
    seen_call_ids: list[str] = []
    hook_indices: list[int] = []
    first = ScriptedSession(
        start_turn=ModelTurn(
            tool_calls=[
                ModelToolCall(id="call_1", name="deploy", arguments='{"env":"prod"}'),
                ModelToolCall(id="call_2", name="echo", arguments='{"value":"ok"}'),
            ],
            raw={"id": "start"},
        ),
    )
    resumed = ScriptedSession(
        start_turn=ModelTurn(raw={"unused": True}),
        continue_turn=ModelTurn(text="done", raw={"id": "done"}),
        on_continue=lambda outputs, _tools, _metadata: seen_call_ids.extend(output.call_id for output in outputs),
    )
    harness = Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[]),
        model=ScriptedModel([first, resumed]),
        tools=[approval_tool([]), echo_tool(normal_called)],
        hooks=[Hook("before_tool_call", lambda ctx: hook_indices.append(ctx.tool_index))],
    )

    paused = await harness.run("go")

    assert normal_called == []
    assert paused.resume_state["batch"] == [
        {"id": "call_1", "name": "deploy", "arguments": '{"env":"prod"}'},
        {"id": "call_2", "name": "echo", "arguments": '{"value":"ok"}'},
    ]

    result = await harness.resume_approvals(paused.resume_state, [ApprovalDecision(call_id="call_1", approved=False)])

    assert seen_call_ids == ["call_1", "call_2"]
    assert normal_called == [{"value": "ok"}]
    assert hook_indices == [1]
    assert [record["call"]["id"] for record in result.tool_call_records] == ["call_1", "call_2"]


async def test_approval_resume_tracing_uses_restored_conversation_id(tmp_path: Path) -> None:
    tracer = ContextFakeTracer()
    first = ScriptedSession(
        start_turn=ModelTurn(
            tool_calls=[ModelToolCall(id="call_1", name="deploy", arguments='{"env":"prod"}')],
            raw={"id": "start"},
        ),
    )
    resumed = ScriptedSession(start_turn=ModelTurn(raw={"unused": True}), continue_turn=ModelTurn(text="done", raw={"id": "done"}))
    harness = Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[]),
        model=ScriptedModel([first, resumed]),
        tools=[approval_tool([])],
        tracing=[TracingOptions(tracer=tracer)],
    )

    paused = await harness.run("deploy", metadata={"conversation_id": "conv_1"})
    await harness.resume_approvals(paused.resume_state, [ApprovalDecision(call_id="call_1", approved=True)])

    agent_spans = [span for span in tracer.spans if span.name == "invoke_agent thinharness"]
    assert agent_spans[-1].attributes["gen_ai.conversation.id"] == "conv_1"


async def test_approval_decision_validation_happens_before_execution(tmp_path: Path) -> None:
    called: list[dict] = []
    first = ScriptedSession(
        start_turn=ModelTurn(
            tool_calls=[ModelToolCall(id="call_1", name="deploy", arguments='{"env":"prod"}')],
            raw={"id": "start"},
        ),
    )
    harness = Harness(HarnessConfig(root=tmp_path, builtin_tools=[]), model=ScriptedModel([first]), tools=[approval_tool(called)])
    paused = await harness.run("deploy")

    with pytest.raises(HarnessError, match="missing approval decision"):
        await harness.resume_approvals(paused.resume_state, [])
    with pytest.raises(HarnessError, match="unknown approval decision"):
        await harness.resume_approvals(paused.resume_state, [ApprovalDecision(call_id="other", approved=True)])
    with pytest.raises(HarnessError, match="duplicate approval decision"):
        await harness.resume_approvals(
            paused.resume_state,
            [ApprovalDecision(call_id="call_1", approved=True), ApprovalDecision(call_id="call_1", approved=False)],
        )
    with pytest.raises(HarnessError, match="non-bool approved"):
        await harness.resume_approvals(
            paused.resume_state,
            [ApprovalDecision(call_id="call_1", approved="false")],  # type: ignore[arg-type]
        )
    with pytest.raises(HarnessError, match="non-string reason"):
        await harness.resume_approvals(
            paused.resume_state,
            [ApprovalDecision(call_id="call_1", approved=False, reason=123)],  # type: ignore[arg-type]
        )

    assert called == []


async def test_tampered_approval_required_ids_fail_closed(tmp_path: Path) -> None:
    approval_called: list[dict] = []
    normal_called: list[dict] = []
    first = ScriptedSession(
        start_turn=ModelTurn(
            tool_calls=[
                ModelToolCall(id="call_1", name="deploy", arguments='{"env":"prod"}'),
                ModelToolCall(id="call_2", name="echo", arguments='{"value":"ok"}'),
            ],
            raw={"id": "start"},
        ),
    )
    resumed = ScriptedSession(start_turn=ModelTurn(raw={"unused": True}), continue_turn=ModelTurn(text="done", raw={"id": "done"}))
    harness = Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[]),
        model=ScriptedModel([first, resumed]),
        tools=[approval_tool(approval_called), echo_tool(normal_called)],
    )
    paused = await harness.run("go")
    tampered = json.loads(json.dumps(paused.resume_state))
    tampered["approval_required_ids"] = ["call_2"]

    with pytest.raises(HarnessError, match="approval_required_ids do not match"):
        await harness.resume_approvals(tampered, [ApprovalDecision(call_id="call_2", approved=True)])

    assert approval_called == []
    assert normal_called == []


async def test_approved_call_can_still_be_cancelled_by_before_tool_hook(tmp_path: Path) -> None:
    called: list[dict] = []
    first = ScriptedSession(
        start_turn=ModelTurn(
            tool_calls=[ModelToolCall(id="call_1", name="deploy", arguments='{"env":"prod"}')],
            raw={"id": "start"},
        ),
    )
    resumed = ScriptedSession(start_turn=ModelTurn(raw={"unused": True}), continue_turn=ModelTurn(text="done", raw={"id": "done"}))

    def cancel(ctx) -> None:
        ctx.cancelled = True
        ctx.cancel_reason = "blocked by policy"

    harness = Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[]),
        model=ScriptedModel([first, resumed]),
        tools=[approval_tool(called)],
        hooks=[Hook("before_tool_call", cancel)],
    )

    paused = await harness.run("deploy")
    result = await harness.resume_approvals(paused.resume_state, [ApprovalDecision(call_id="call_1", approved=True)])

    assert called == []
    assert result.usage.cancelled_tool_calls == 1
    assert result.tool_call_records[-1]["approval"] == {"approved": True}
    assert result.tool_call_records[-1]["cancelled"] is True


async def test_approved_retry_output_uses_retry_accounting_and_can_repause(tmp_path: Path) -> None:
    first = ScriptedSession(
        start_turn=ModelTurn(
            tool_calls=[ModelToolCall(id="call_1", name="deploy", arguments='{"env":"prod"}')],
            raw={"id": "start"},
        ),
    )
    resumed = ScriptedSession(
        start_turn=ModelTurn(raw={"unused": True}),
        continue_turn=ModelTurn(
            tool_calls=[ModelToolCall(id="call_2", name="deploy", arguments='{"env":"prod"}')],
            raw={"id": "retry"},
        ),
    )
    tool = ToolSpec(
        "deploy",
        "Deploy something.",
        {"type": "object", "properties": {"env": {"type": "string"}}, "required": ["env"]},
        lambda args: ToolResult(False, "try again", {"error_type": "RetryMe", "retry": True}),
        requires_approval=True,
    )
    harness = Harness(HarnessConfig(root=tmp_path, builtin_tools=[], tool_retries=2), model=ScriptedModel([first, resumed]), tools=[tool])

    paused = await harness.run("deploy")
    result = await harness.resume_approvals(paused.resume_state, [ApprovalDecision(call_id="call_1", approved=True)])

    assert result.stop_reason == "approval_required"
    assert result.pending_approvals[0].call_id == "call_2"
    assert result.usage.tool_retries == {"deploy": 1}
    assert result.usage.tool_calls == 2


async def test_resume_budget_spans_approval_pause(tmp_path: Path) -> None:
    called: list[dict] = []
    first = ScriptedSession(
        start_turn=ModelTurn(
            tool_calls=[ModelToolCall(id="call_1", name="deploy", arguments='{"env":"prod"}')],
            raw={"id": "start"},
        ),
    )
    resumed = ScriptedSession(start_turn=ModelTurn(raw={"unused": True}), continue_turn=ModelTurn(text="done", raw={"id": "done"}))
    harness = Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[], max_model_requests=1),
        model=ScriptedModel([first, resumed]),
        tools=[approval_tool(called)],
    )
    paused = await harness.run("deploy")

    with pytest.raises(HarnessError, match="max_model_requests=1"):
        await harness.resume_approvals(paused.resume_state, [ApprovalDecision(call_id="call_1", approved=True)])

    assert called == [{"env": "prod"}]


async def test_over_budget_approval_batch_fails_before_pause(tmp_path: Path) -> None:
    called: list[dict] = []
    session = ScriptedSession(
        start_turn=ModelTurn(
            tool_calls=[ModelToolCall(id="call_1", name="deploy", arguments='{"env":"prod"}')],
            raw={"id": "start"},
        ),
    )
    harness = Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[], max_tool_calls=0),
        model=ScriptedModel([session]),
        tools=[approval_tool(called)],
    )

    with pytest.raises(HarnessError, match="max_tool_calls=0"):
        await harness.run("deploy")

    assert called == []


async def test_background_task_is_cancelled_and_reported_when_later_turn_pauses(tmp_path: Path) -> None:
    release = asyncio.Event()
    resumed_notices = []
    first = ScriptedSession(
        start_turn=ModelTurn(
            tool_calls=[ModelToolCall(id="call_bg", name="slow", arguments='{"_background":true}')],
            raw={"id": "start"},
        ),
        continue_turn=ModelTurn(
            tool_calls=[ModelToolCall(id="call_approval", name="deploy", arguments='{"env":"prod"}')],
            raw={"id": "approval"},
        ),
    )
    resumed = ScriptedSession(
        start_turn=ModelTurn(raw={"unused": True}),
        continue_turn=ModelTurn(text="done", raw={"id": "done"}),
        on_continue=lambda _outputs, _tools, _metadata: resumed_notices.extend(resumed.notice_calls[-1][1]),
    )

    async def slow(_args) -> str:
        await release.wait()
        return "late"

    harness = Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[]),
        model=ScriptedModel([first, resumed]),
        tools=[
            ToolSpec("slow", "Slow", {"type": "object", "properties": {}, "additionalProperties": False}, slow, background="model"),
            approval_tool([]),
        ],
    )

    events = []
    async for event in harness.stream("go"):
        events.append(event)
    paused = next(event.result for event in events if isinstance(event, RunCompletedEvent))

    assert paused.stop_reason == "approval_required"
    assert paused.resume_state["cancelled_background_task_ids"] == ["bg_1"]
    assert paused.tool_call_records[-1]["background"]["event"] == "cancelled"
    background_completed_at = next(index for index, event in enumerate(events) if isinstance(event, BackgroundTaskCompletedEvent))
    run_completed_at = next(index for index, event in enumerate(events) if isinstance(event, RunCompletedEvent))
    assert background_completed_at < run_completed_at

    result = await harness.resume_approvals(paused.resume_state, [ApprovalDecision(call_id="call_approval", approved=True)])

    assert result.text == "done"
    assert [(notice.kind, notice.content) for notice in resumed_notices] == [
        ("background_cancelled", "Background task was cancelled when the run paused for human approval: bg_1.")
    ]


async def test_ready_background_completion_is_preserved_across_approval_pause(tmp_path: Path) -> None:
    background_finished = asyncio.Event()
    resumed_notices = []

    class WaitingSession(ScriptedSession):
        async def continue_with_tools(self, outputs, *, instructions=None, tools, metadata=None, structured_output=None, notices=None):
            await asyncio.wait_for(background_finished.wait(), timeout=0.5)
            return await super().continue_with_tools(
                outputs,
                instructions=instructions,
                tools=tools,
                metadata=metadata,
                structured_output=structured_output,
                notices=notices,
            )

    first = WaitingSession(
        start_turn=ModelTurn(
            tool_calls=[ModelToolCall(id="call_bg", name="fast", arguments='{"_background":true}')],
            raw={"id": "start"},
        ),
        continue_turn=ModelTurn(
            tool_calls=[ModelToolCall(id="call_approval", name="deploy", arguments='{"env":"prod"}')],
            raw={"id": "approval"},
        ),
    )
    resumed = ScriptedSession(
        start_turn=ModelTurn(raw={"unused": True}),
        continue_turn=ModelTurn(text="done", raw={"id": "done"}),
        on_continue=lambda _outputs, _tools, _metadata: resumed_notices.extend(resumed.notice_calls[-1][1]),
    )

    async def fast(_args) -> str:
        background_finished.set()
        return "ready"

    harness = Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[]),
        model=ScriptedModel([first, resumed]),
        tools=[
            ToolSpec("fast", "Fast", {"type": "object", "properties": {}, "additionalProperties": False}, fast, background="model"),
            approval_tool([]),
        ],
    )

    paused = await harness.run("go")

    assert paused.stop_reason == "approval_required"
    assert paused.resume_state["cancelled_background_task_ids"] == []
    assert len(paused.resume_state["ready_background_completion_messages"]) == 1
    assert paused.tool_call_records[-1]["background"]["event"] == "completed"

    result = await harness.resume_approvals(paused.resume_state, [ApprovalDecision(call_id="call_approval", approved=True)])

    assert result.text == "done"
    assert [(notice.kind, "Tool: fast" in notice.content) for notice in resumed_notices] == [("background_completion", True)]


async def test_openai_approval_pause_round_trips_provider_state(tmp_path: Path) -> None:
    called: list[dict] = []
    client = FakeClient()
    model = _fake_openai(client)
    harness = Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[]),
        model=model,
        tools=[
            ToolSpec(
                "read",
                "Read something.",
                {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
                lambda args: called.append(args) or "read-ok",
                requires_approval=True,
            )
        ],
    )

    paused = await harness.run("read")
    result = await Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[]),
        model=model,
        tools=[
            ToolSpec(
                "read",
                "Read something.",
                {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
                lambda args: called.append(args) or "read-ok",
                requires_approval=True,
            )
        ],
    ).resume_approvals(json.loads(json.dumps(paused.resume_state)), [ApprovalDecision(call_id="call_1", approved=True)])

    assert result.text == "done"
    assert called == [{"path": "hello.txt"}]
    assert paused.resume_state["provider_state"]["kind"] == "transcript"
    assert paused.resume_state["provider_state"]["version"] == 3
    assert [entry["role"] for entry in paused.resume_state["provider_state"]["entries"]] == ["user", "assistant"]
    assert "previous_response_id" not in client.payloads[1]
    assert [item["type"] for item in client.payloads[1]["input"]] == ["message", "function_call", "function_call_output"]
    assert client.payloads[1]["input"][-1] == {
        "type": "function_call_output",
        "call_id": "call_1",
        "output": result.tool_call_records[-1]["output"],
    }


async def test_anthropic_approval_pause_round_trips_provider_state(tmp_path: Path) -> None:
    called: list[dict] = []
    provider = FakeAnthropicProvider()
    model = AnthropicMessagesModel("claude-test", provider=provider)
    paused = await Harness(HarnessConfig(root=tmp_path, builtin_tools=[]), model=model, tools=[approval_echo_tool(called)]).run("first")

    result = await Harness(HarnessConfig(root=tmp_path, builtin_tools=[]), model=model, tools=[approval_echo_tool(called)]).resume_approvals(
        json.loads(json.dumps(paused.resume_state)),
        [ApprovalDecision(call_id="toolu_1", approved=True)],
    )

    assert result.text == "done"
    assert called == [{"value": "first"}]
    assistant_tool_use = provider.payloads[1]["messages"][1]["content"][0]
    user_tool_result = provider.payloads[1]["messages"][-1]["content"][0]
    assert assistant_tool_use["id"] == "toolu_1"
    assert user_tool_result["type"] == "tool_result"
    assert user_tool_result["tool_use_id"] == assistant_tool_use["id"]


async def test_openrouter_approval_pause_round_trips_provider_state(tmp_path: Path) -> None:
    called: list[dict] = []
    provider = FakeOpenRouterProvider()
    model = OpenRouterModel("openai/test", provider=provider)
    paused = await Harness(HarnessConfig(root=tmp_path, builtin_tools=[]), model=model, tools=[approval_echo_tool(called)]).run("first")

    result = await Harness(HarnessConfig(root=tmp_path, builtin_tools=[]), model=model, tools=[approval_echo_tool(called)]).resume_approvals(
        json.loads(json.dumps(paused.resume_state)),
        [ApprovalDecision(call_id="call_1", approved=True)],
    )

    assert result.text == "done"
    assert called == [{"value": "first"}]
    assistant_call = provider.payloads[1]["messages"][2]["tool_calls"][0]
    tool_message = provider.payloads[1]["messages"][-1]
    assert assistant_call["id"] == "call_1"
    assert tool_message["role"] == "tool"
    assert tool_message["tool_call_id"] == assistant_call["id"]


async def test_structured_output_finalizes_after_approval_resume(tmp_path: Path) -> None:
    class Answer(BaseModel):
        value: str

    first = ScriptedSession(
        start_turn=ModelTurn(
            tool_calls=[ModelToolCall(id="call_1", name="deploy", arguments='{"env":"prod"}')],
            raw={"id": "start"},
        ),
    )
    resumed = ScriptedSession(
        start_turn=ModelTurn(raw={"unused": True}),
        continue_turn=ModelTurn(
            tool_calls=[ModelToolCall(id="call_final", name="final_result", arguments='{"value":"ok"}')],
            raw={"id": "final"},
        ),
    )
    harness = Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[], output_type=Answer, output_mode="tool"),
        model=ScriptedModel([first, resumed]),
        tools=[approval_tool([])],
    )

    paused = await harness.run("deploy")
    result = await harness.resume_approvals(paused.resume_state, [ApprovalDecision(call_id="call_1", approved=True)])

    assert result.output == Answer(value="ok")
    assert result.resume_state is None


async def test_approval_resume_metadata_override_replaces_envelope_metadata(tmp_path: Path) -> None:
    seen_metadata = {}
    first = ScriptedSession(
        start_turn=ModelTurn(
            tool_calls=[ModelToolCall(id="call_1", name="deploy", arguments='{"env":"prod"}')],
            raw={"id": "start"},
        ),
    )
    resumed = ScriptedSession(
        start_turn=ModelTurn(raw={"unused": True}),
        continue_turn=ModelTurn(text="done", raw={"id": "done"}),
        on_continue=lambda _outputs, _tools, metadata: seen_metadata.update(metadata),
    )
    harness = Harness(HarnessConfig(root=tmp_path, builtin_tools=[]), model=ScriptedModel([first, resumed]), tools=[approval_tool([])])

    paused = await harness.run("deploy", metadata={"conversation_id": "original", "keep": "old"})
    await harness.resume_approvals(
        paused.resume_state,
        [ApprovalDecision(call_id="call_1", approved=True)],
        metadata={"conversation_id": "new", "other": "value"},
    )

    assert seen_metadata == {"conversation_id": "new", "other": "value"}


async def test_approval_resume_rejects_duplicate_batch_call_ids(tmp_path: Path) -> None:
    first = ScriptedSession(
        start_turn=ModelTurn(
            tool_calls=[ModelToolCall(id="call_1", name="deploy", arguments='{"env":"prod"}')],
            raw={"id": "start"},
        ),
    )
    harness = Harness(HarnessConfig(root=tmp_path, builtin_tools=[]), model=ScriptedModel([first]), tools=[approval_tool([])])
    paused = await harness.run("deploy")
    state = json.loads(json.dumps(paused.resume_state))
    state["batch"].append({"id": "call_1", "name": "echo", "arguments": '{"value":"ok"}'})

    with pytest.raises(HarnessError, match="duplicate call id"):
        await harness.resume_approvals(state, [ApprovalDecision(call_id="call_1", approved=True)])


async def test_approval_resume_labels_inner_provider_state_errors(tmp_path: Path) -> None:
    first = ScriptedSession(
        start_turn=ModelTurn(
            tool_calls=[ModelToolCall(id="call_1", name="deploy", arguments='{"env":"prod"}')],
            raw={"id": "start"},
        ),
    )
    resumed = ScriptedSession(start_turn=ModelTurn(raw={"unused": True}), continue_turn=ModelTurn(text="done", raw={"id": "done"}))
    harness = Harness(HarnessConfig(root=tmp_path, builtin_tools=[]), model=ScriptedModel([first, resumed]), tools=[approval_tool([])])
    paused = await harness.run("deploy")
    state = json.loads(json.dumps(paused.resume_state))
    state["provider_state"]["kind"] = "wrong"

    with pytest.raises(HarnessError, match="approval state provider_state kind 'wrong' does not match 'scripted'"):
        await harness.resume_approvals(state, [ApprovalDecision(call_id="call_1", approved=True)])


async def test_approval_resume_labels_builtin_provider_state_errors(tmp_path: Path) -> None:
    client = FakeClient()
    model = _fake_openai(client)
    harness = Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[]),
        model=model,
        tools=[
            ToolSpec(
                "read",
                "Read something.",
                {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
                lambda _args: "read-ok",
                requires_approval=True,
            )
        ],
    )
    paused = await harness.run("read")
    state = json.loads(json.dumps(paused.resume_state))
    state["provider_state"]["version"] = 1

    with pytest.raises(HarnessError, match="approval state provider_state version 1 is not supported"):
        await harness.resume_approvals(state, [ApprovalDecision(call_id="call_1", approved=True)])


async def test_limit_warning_dedup_keys_survive_approval_round_trip(tmp_path: Path) -> None:
    first = ScriptedSession(
        start_turn=ModelTurn(
            tool_calls=[ModelToolCall(id="call_1", name="deploy", arguments='{"env":"prod"}')],
            raw={"id": "start"},
        ),
    )
    resumed = ScriptedSession(start_turn=ModelTurn(raw={"unused": True}), continue_turn=ModelTurn(text="done", raw={"id": "done"}))
    harness = Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[], max_tool_calls=1),
        model=ScriptedModel([first, resumed]),
        tools=[approval_tool([])],
    )

    paused = await harness.run("deploy")
    result = await harness.resume_approvals(json.loads(json.dumps(paused.resume_state)), [ApprovalDecision(call_id="call_1", approved=True)])

    assert result.text == "done"
    assert paused.resume_state["emitted_limit_warnings"] == [["limit_warning", "tool_calls", 1]]
    assert [(notice.limit_kind, notice.remaining) for _method, notices in resumed.notice_calls for notice in notices] == [("tool_calls", 0)]


def test_approval_tool_requires_resumable_model_and_no_background(tmp_path: Path) -> None:
    class NonResumableModel:
        model = "non-resumable"
        provider = type("Provider", (), {"name": "test"})()
        api_key = "key"

        def new_session(self):
            raise AssertionError("unused")

    with pytest.raises(ValueError, match="approval-required tools cannot use background execution"):
        ToolSpec("x", "X", {"type": "object"}, lambda args: "x", requires_approval=True, background="model")
    with pytest.raises(ValueError, match="approval-required tools require a resumable model"):
        Harness(HarnessConfig(root=tmp_path, builtin_tools=[]), model=NonResumableModel(), tools=[approval_tool([])])


def test_subagents_reject_explicit_approval_tools_and_filter_inherited(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="approval-required tools are not supported inside subagents"):
        SubAgentConfig(name="helper", description="Helper.", tools=[approval_tool([])])

    parent = Harness(HarnessConfig(root=tmp_path, builtin_tools=[]), model=ScriptedModel([]), tools=[approval_tool([]), echo_tool([])])
    from thinharness.subagents import build_child_harness

    child = build_child_harness(parent, None)

    assert "deploy" not in {tool.name for tool in child.tools}
    assert "echo" in {tool.name for tool in child.tools}


async def test_inherit_parent_tools_subagent_runs_without_parent_approval_tool(tmp_path: Path) -> None:
    subagent_outputs = []
    echo_called: list[dict] = []
    child_session = ScriptedSession(
        start_turn=ModelTurn(
            tool_calls=[ModelToolCall(id="call_child", name="echo", arguments='{"value":"child"}')],
            raw={"id": "child_start"},
        ),
        continue_turn=ModelTurn(text="child done", raw={"id": "child_done"}),
    )
    parent_session = ScriptedSession(
        start_turn=ModelTurn(
            tool_calls=[ModelToolCall(id="call_parent", name="subagent", arguments='{"task":"use echo","agent":"helper"}')],
            raw={"id": "parent_start"},
        ),
        continue_turn=ModelTurn(text="parent done", raw={"id": "parent_done"}),
        on_continue=lambda outputs, _tools, _metadata: subagent_outputs.extend(outputs),
    )
    model = ScriptedModel([parent_session, child_session])
    harness = Harness(
        HarnessConfig(
            root=tmp_path,
            builtin_tools=["subagent"],
            subagents=[SubAgentConfig(name="helper", description="Helper.", inherit_parent_tools=True)],
        ),
        model=model,
        tools=[approval_tool([]), echo_tool(echo_called)],
    )

    result = await harness.run("delegate")

    assert result.text == "parent done"
    assert echo_called == [{"value": "child"}]
    parsed = json.loads(subagent_outputs[0].output)
    assert parsed["metadata"]["tools"] == ["echo"]


async def test_resume_approvals_closed_harness_guard(tmp_path: Path) -> None:
    session = ScriptedSession(start_turn=ModelTurn(text="done", raw={"id": "done"}))
    harness = Harness(HarnessConfig(root=tmp_path, builtin_tools=[]), model=ScriptedModel([session]))
    await harness.aclose()

    with pytest.raises(HarnessError, match="harness is closed"):
        await harness.resume_approvals({}, [])
    with pytest.raises(HarnessError, match="harness is closed"):
        harness.stream_approvals({}, [])


async def test_streaming_approval_resume_marker_and_request_kind(tmp_path: Path) -> None:
    first = ScriptedSession(
        start_turn=ModelTurn(
            tool_calls=[ModelToolCall(id="call_1", name="deploy", arguments='{"env":"prod"}')],
            raw={"id": "start"},
        ),
    )
    resumed = ScriptedSession(start_turn=ModelTurn(raw={"unused": True}), continue_turn=ModelTurn(text="done", raw={"id": "done"}))
    harness = Harness(HarnessConfig(root=tmp_path, builtin_tools=[]), model=ScriptedModel([first, resumed]), tools=[approval_tool([])])
    paused = await harness.run("deploy")

    events = []
    async for event in harness.stream_approvals(paused.resume_state, [ApprovalDecision(call_id="call_1", approved=True)]):
        events.append(event)

    assert isinstance(events[0], RunStartedEvent)
    assert events[0].prompt is None
    assert isinstance(events[1], ApprovalResumedEvent)
    assert events[1].decisions == (ApprovalDecision(call_id="call_1", approved=True),)
    assert any(isinstance(event, ToolCallStartedEvent) for event in events)
    assert any(isinstance(event, ToolCallCompletedEvent) for event in events)
    assert any(isinstance(event, ModelRequestStartedEvent) and event.request_kind == "approval_resume" for event in events)
    assert any(isinstance(event, RunCompletedEvent) and event.result.text == "done" for event in events)


def test_directed_resume_api_errors(tmp_path: Path) -> None:
    session = ScriptedSession(start_turn=ModelTurn(text="ready", raw={"id": "first"}))
    provider_state = Harness(HarnessConfig(root=tmp_path, builtin_tools=[]), model=ScriptedModel([session])).run_sync("first").resume_state

    with pytest.raises(HarnessError, match="approval state kind"):
        Harness(HarnessConfig(root=tmp_path / "resume", builtin_tools=[]), model=ScriptedModel([])).resume_approvals_sync(provider_state, [])

    approval_state = {
        "kind": "approval_pause",
        "version": 1,
        "provider_state": provider_state,
        "batch": [{"id": "call_1", "name": "deploy", "arguments": "{}"}],
        "approval_required_ids": ["call_1"],
        "cancelled_background_task_ids": [],
        "usage": {"model_requests": 1, "tool_calls": 1, "cancelled_tool_calls": 0, "output_retries": 0, "tool_retries": {}},
        "responses": [],
        "tool_call_records": [],
        "emitted_limit_warnings": [],
        "metadata": {},
    }
    with pytest.raises(HarnessError, match="approval pause state must be resumed with resume_approvals"):
        Harness(HarnessConfig(root=tmp_path / "other", builtin_tools=[]), model=ScriptedModel([])).run_sync("next", resume_from=approval_state)


def test_approval_pause_state_accepts_old_envelopes_without_ready_messages(tmp_path: Path) -> None:
    session = ScriptedSession(start_turn=ModelTurn(text="ready", raw={"id": "first"}))
    provider_state = Harness(HarnessConfig(root=tmp_path, builtin_tools=[]), model=ScriptedModel([session])).run_sync("first").resume_state
    approval_state = {
        "kind": "approval_pause",
        "version": 1,
        "provider_state": provider_state,
        "batch": [{"id": "call_1", "name": "deploy", "arguments": "{}"}],
        "approval_required_ids": ["call_1"],
        "cancelled_background_task_ids": [],
        "usage": {"model_requests": 1, "tool_calls": 1, "cancelled_tool_calls": 0, "output_retries": 0, "tool_retries": {}},
        "responses": [],
        "tool_call_records": [],
        "emitted_limit_warnings": [],
        "metadata": {},
    }

    pause = validate_approval_pause_state(approval_state)

    assert pause.ready_background_completion_messages == []
