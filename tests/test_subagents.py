from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from fakes import (
    FailingSession,
    FakeTracer,
    RecordingModel,
    ScriptedModel,
    ScriptedSession,
    echo_tool,
    tool_output,
)

from thinharness import (
    DEFAULT_SUBAGENT_NAME,
    AfterSubagentRunContext,
    BeforeSubagentRunContext,
    Harness,
    HarnessConfig,
    Hook,
    HookRegistry,
    SubAgentConfig,
    ToolSpec,
    TracingOptions,
    build_child_harness,
    call_tool,
    create_subagent_tool,
)
from thinharness.hooks import current_tool_call_context, current_tool_runtime_context
from thinharness.providers import ModelToolCall, ModelTurn


class ClosingProvider:
    def __init__(self) -> None:
        self.name = "OpenAI"
        self.closed = 0

    async def aclose(self) -> None:
        self.closed += 1


def test_subagent_config_validation_accepts_tool_specs() -> None:
    spec = echo_tool()
    sequential_tool = ToolSpec("sequential_echo", "Sequential echo", {"type": "object", "properties": {}}, lambda args: "ok", sequential=True)
    config = SubAgentConfig(name="research.1", description="Research helper.", tools=[spec, sequential_tool])
    inherited = SubAgentConfig(name="general", description="General helper.", inherit_parent_tools=True)

    assert config.tools == [spec, sequential_tool]
    assert inherited.inherit_parent_tools is True
    with pytest.raises(ValueError, match="inherit_parent_tools"):
        SubAgentConfig(name="bad", description="Bad helper.", inherit_parent_tools=True, builtin_tools=["read"])
    with pytest.raises(ValueError, match="cannot be exposed"):
        SubAgentConfig(name="recursive", description="Recursive helper.", builtin_tools=["subagent"])
    with pytest.raises(ValueError, match="cannot be exposed"):
        SubAgentConfig(
            name="recursive-custom",
            description="Recursive helper.",
            tools=[ToolSpec("subagent", "Recursive", {"type": "object", "properties": {}}, lambda args: "bad")],
        )
    with pytest.raises(ValueError, match="must define"):
        SubAgentConfig(name="empty", description="No tools.")
    with pytest.raises(ValueError):
        SubAgentConfig(name="bad name", description="Bad helper.", builtin_tools=["read"])
    with pytest.raises(ValueError, match="non-empty single line"):
        SubAgentConfig(name="ok", description="   ", builtin_tools=["read"])
    with pytest.raises(ValueError, match="non-empty single line"):
        SubAgentConfig(name="ok", description="Bad\nhelper.", builtin_tools=["read"])

def test_subagent_builtin_exposure_is_selectable(tmp_path: Path) -> None:
    default = Harness(HarnessConfig(root=tmp_path), model=ScriptedModel([]))
    disabled = Harness(HarnessConfig(root=tmp_path, builtin_tools=[]), model=ScriptedModel([]))
    only_subagent = Harness(HarnessConfig(root=tmp_path, builtin_tools=["subagent"]), model=ScriptedModel([]))

    assert "subagent" not in [tool["name"] for tool in default.tool_schemas()]
    assert "subagent" not in [tool["name"] for tool in disabled.tool_schemas()]
    assert [tool["name"] for tool in only_subagent.tool_schemas()] == ["subagent"]
    schema = only_subagent.tool_schemas()[0]["parameters"]
    assert set(schema["properties"]) == {"task", "agent", "_background"}
    assert "tools" not in schema["properties"]

def test_default_subagent_runs_child_with_inherited_tools_and_structured_result(tmp_path: Path) -> None:
    parent_call = ModelTurn(
        tool_calls=[ModelToolCall(id="call_1", name="subagent", arguments='{"task":"help"}')],
        raw={"id": "parent-start"},
    )
    child_start_metadata = {}

    def on_child_start(prompt, _instructions, tools, metadata, _previous_response_id):
        child_start_metadata.update({"prompt": prompt, "tools": [tool["name"] for tool in tools], "metadata": metadata})

    def on_parent_continue(outputs, _tools, _metadata):
        envelope = tool_output(outputs[0].output)
        assert envelope["ok"] is True
        assert envelope["content"] == "child done"
        assert envelope["metadata"]["agent"] == "default"
        assert envelope["metadata"]["inherited"] is True
        assert envelope["metadata"]["tools"] == ["echo"]

    child = ScriptedSession(start_turn=ModelTurn(text="child done", raw={"id": "child"}), on_start=on_child_start)
    parent = ScriptedSession(start_turn=parent_call, continue_turn=ModelTurn(text="parent done", raw={"id": "parent-done"}), on_continue=on_parent_continue)
    model = ScriptedModel([parent, child])
    harness = Harness(HarnessConfig(root=tmp_path, builtin_tools=[]), model=model, tools=[echo_tool()])
    harness.add_tool(create_subagent_tool(harness, []))

    result = harness.run_sync("delegate", metadata={"conversation_id": "conv-1", "extra": "ignored"})

    assert result.text == "parent done"
    assert child_start_metadata == {
        "prompt": "help",
        "tools": ["echo"],
        "metadata": {"conversation_id": "conv-1", "parent_call_id": "call_1"},
    }

def test_subagent_metadata_uses_runtime_context_without_leaking_hook_mutation(tmp_path: Path) -> None:
    parent_call = ModelTurn(
        tool_calls=[ModelToolCall(id="call_1", name="subagent", arguments='{"task":"help"}')],
        raw={"id": "parent-start"},
    )
    child_start_metadata = {}
    hook_metadata = []
    tool_contexts = []

    def on_child_start(prompt, _instructions, _tools, metadata, _previous_response_id):
        child_start_metadata.update({"prompt": prompt, "metadata": metadata})

    def before_subagent(ctx):
        assert isinstance(ctx, BeforeSubagentRunContext)
        hook_metadata.append(("before", dict(ctx.metadata)))
        tool_contexts.append((current_tool_call_context(), current_tool_runtime_context()))
        ctx.metadata["conversation_id"] = "mutated"
        ctx.metadata["new"] = "ignored"

    def after_subagent(ctx):
        assert isinstance(ctx, AfterSubagentRunContext)
        hook_metadata.append(("after", dict(ctx.metadata)))

    child = ScriptedSession(start_turn=ModelTurn(text="child done", raw={"id": "child"}), on_start=on_child_start)
    parent = ScriptedSession(start_turn=parent_call, continue_turn=ModelTurn(text="parent done", raw={"id": "parent-done"}))
    harness = Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[]),
        model=ScriptedModel([parent, child]),
        tools=[echo_tool()],
        hooks=[Hook("before_subagent_run", before_subagent), Hook("after_subagent_run", after_subagent)],
    )
    harness.add_tool(create_subagent_tool(harness, []))

    assert harness.run_sync("delegate", metadata={"conversation_id": "conv-1", "extra": "hook-only"}).text == "parent done"

    assert hook_metadata == [
        ("before", {"conversation_id": "conv-1", "extra": "hook-only"}),
        ("after", {"conversation_id": "conv-1", "extra": "hook-only"}),
    ]
    assert child_start_metadata == {"prompt": "help", "metadata": {"conversation_id": "conv-1", "parent_call_id": "call_1"}}
    assert tool_contexts == [
        (
            {"call_id": "call_1", "name": "subagent"},
            {"run_metadata": {"conversation_id": "conv-1", "extra": "hook-only"}},
        )
    ]

def test_subagent_receives_fresh_child_budget_notices(tmp_path: Path) -> None:
    parent_call = ModelTurn(
        tool_calls=[
            ModelToolCall(id="call_1", name="echo", arguments='{"value":"parent"}'),
            ModelToolCall(id="call_2", name="subagent", arguments='{"task":"help"}'),
        ],
        raw={"id": "parent-start"},
    )
    parent = ScriptedSession(start_turn=parent_call, continue_turn=ModelTurn(text="parent done", raw={"id": "parent-done"}))
    child = ScriptedSession(start_turn=ModelTurn(text="child done", raw={"id": "child"}))
    model = ScriptedModel([parent, child])
    harness = Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[], max_model_requests=2, max_tool_calls=2),
        model=model,
        tools=[echo_tool()],
    )
    harness.add_tool(create_subagent_tool(harness, []))

    assert harness.run_sync("delegate").text == "parent done"

    assert parent.notice_calls[0][1] == []
    assert [(notice.limit_kind, notice.remaining) for notice in parent.notice_calls[1][1]] == [("model_requests", 1), ("tool_calls", 0)]
    assert child.notice_calls[0][1] == []

def test_default_subagent_does_not_close_shared_parent_provider(tmp_path: Path) -> None:
    parent_call = ModelTurn(
        tool_calls=[ModelToolCall(id="call_1", name="subagent", arguments='{"task":"help"}')],
        raw={"id": "parent-start"},
    )
    parent = ScriptedSession(start_turn=parent_call, continue_turn=ModelTurn(text="parent done", raw={"id": "parent-done"}))
    child = ScriptedSession(start_turn=ModelTurn(text="child done", raw={"id": "child"}))
    model = ScriptedModel([parent, child])
    provider = ClosingProvider()
    model.provider = provider
    harness = Harness(HarnessConfig(root=tmp_path, builtin_tools=[]), model=model, tools=[echo_tool()])
    harness.add_tool(create_subagent_tool(harness, []))

    assert harness.run_sync("delegate").text == "parent done"

    assert provider.closed == 0

def test_named_inherited_subagent_gets_parent_tools_without_subagent(tmp_path: Path) -> None:
    parent_echo = echo_tool()
    parent = Harness(HarnessConfig(root=tmp_path, builtin_tools=[]), model=ScriptedModel([]), tools=[parent_echo])
    parent.add_tool(create_subagent_tool(parent, []))

    child = build_child_harness(parent, SubAgentConfig(name="general", description="General helper.", inherit_parent_tools=True))

    assert child.tools == [parent_echo]
    assert child.skills is parent.skills
    assert child.config.subagents == []

def test_inherited_subagent_reuses_parent_skill_registry(tmp_path: Path) -> None:
    skill = tmp_path / "skills" / "demo"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: demo\ndescription: Demo skill\n---\nDemo body", encoding="utf-8")
    parent = Harness(
        HarnessConfig(root=tmp_path, skills_dir=tmp_path / "skills", builtin_tools=["skill_read"]),
        model=ScriptedModel([]),
    )

    child = build_child_harness(parent, SubAgentConfig(name="general", description="General helper.", inherit_parent_tools=True))

    assert child.skills is parent.skills
    assert "demo - Demo skill" in child.system_instructions()
    skill_read = next(tool for tool in child.tools if tool.name == "skill_read")
    assert skill_read.handler.__self__ is parent.skills

def test_explicit_subagent_skill_tools_use_parent_skill_config(tmp_path: Path) -> None:
    skill = tmp_path / "skills" / "demo"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: demo\ndescription: Demo skill\n---\nDemo body", encoding="utf-8")
    parent = Harness(
        HarnessConfig(root=tmp_path, skills_dir=tmp_path / "skills", builtin_tools=["skill_read"]),
        model=ScriptedModel([]),
    )

    child = build_child_harness(parent, SubAgentConfig(name="skilled", description="Skill helper.", builtin_tools=["skill_read"]))

    assert child.skills is not parent.skills
    assert [tool.name for tool in child.tools] == ["skill_read"]
    assert "demo - Demo skill" in child.system_instructions()

def test_subagent_model_override_credential_forwarding(tmp_path: Path, monkeypatch) -> None:
    calls = []

    def fake_infer_model(model_ref, **kwargs):
        calls.append((model_ref, kwargs))
        return ScriptedModel([])

    monkeypatch.setattr("thinharness.subagents.infer_model", fake_infer_model)
    parent = Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[], model="openai:parent", api_key="parent-key", base_url="https://parent.example"),
        model=ScriptedModel([]),
    )
    same_provider = SubAgentConfig(name="same", description="Same provider.", model="openai:child", tools=[echo_tool()])
    other_provider = SubAgentConfig(name="other", description="Other provider.", model="anthropic:child", tools=[echo_tool()])

    same_child = build_child_harness(parent, same_provider)
    other_child = build_child_harness(parent, other_provider)

    assert same_child.config.model == "openai:child"
    assert other_child.config.model == "anthropic:child"
    assert calls[0][1]["api_key"] == "parent-key"
    assert calls[0][1]["base_url"] == "https://parent.example"
    assert calls[1][1]["api_key"] is None
    assert calls[1][1]["base_url"] is None

def test_subagent_model_override_is_used_for_child_run(tmp_path: Path, monkeypatch) -> None:
    child_model = RecordingModel([ScriptedSession(start_turn=ModelTurn(text="child done", raw={"id": "child"}))], model="child-model")
    parent_model = RecordingModel([], model="parent-model")

    def fake_infer_model(_model_ref, **_kwargs):
        return child_model

    monkeypatch.setattr("thinharness.subagents.infer_model", fake_infer_model)
    parent = Harness(HarnessConfig(root=tmp_path, builtin_tools=[]), model=parent_model)
    child = build_child_harness(parent, SubAgentConfig(name="special", description="Special helper.", model="openai:child", tools=[echo_tool()]))

    assert child.model is child_model
    assert child.run_sync("delegate").text == "child done"
    assert child_model.session_requests == 1
    assert parent_model.session_requests == 0

def test_subagent_model_override_closes_child_provider(tmp_path: Path, monkeypatch) -> None:
    parent_call = ModelTurn(
        tool_calls=[ModelToolCall(id="call_1", name="subagent", arguments='{"task":"help","agent":"special"}')],
        raw={"id": "parent-start"},
    )
    parent = ScriptedSession(start_turn=parent_call, continue_turn=ModelTurn(text="parent done", raw={"id": "parent-done"}))
    child = ScriptedSession(start_turn=ModelTurn(text="child done", raw={"id": "child"}))
    child_model = ScriptedModel([child], model="child-model")
    child_provider = ClosingProvider()
    child_model.provider = child_provider

    def fake_infer_model(_model_ref, **_kwargs):
        return child_model

    monkeypatch.setattr("thinharness.subagents.infer_model", fake_infer_model)
    harness = Harness(
        HarnessConfig(
            root=tmp_path,
            builtin_tools=[],
            subagents=[SubAgentConfig(name="special", description="Special helper.", model="openai:child", tools=[echo_tool()])],
        ),
        model=ScriptedModel([parent]),
    )
    harness.add_tool(create_subagent_tool(harness, harness.config.subagents))

    assert harness.run_sync("delegate").text == "parent done"

    assert child_provider.closed == 1

async def test_concurrent_subagent_strict_abort_does_not_hang(tmp_path: Path) -> None:
    parent_call = ModelTurn(
        tool_calls=[
            ModelToolCall(id="call_1", name="subagent", arguments='{"task":"one"}'),
            ModelToolCall(id="call_2", name="subagent", arguments='{"task":"two"}'),
        ],
        raw={"id": "parent-start"},
    )

    async def wait_forever(_args):
        await asyncio.Event().wait()

    child_tool = ToolSpec("wait", "Wait", {"type": "object", "properties": {}}, wait_forever)

    def fail_second_subagent(ctx):
        if ctx.call_id == "call_2":
            raise RuntimeError("strict abort")

    harness = Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[], strict_hooks=True),
        model=ScriptedModel([
            ScriptedSession(start_turn=parent_call),
            ScriptedSession(start_turn=ModelTurn(tool_calls=[ModelToolCall(id="child_call", name="wait", arguments="{}")], raw={"id": "child"})),
        ]),
        tools=[child_tool],
        hooks=[Hook("before_tool_call", fail_second_subagent)],
    )
    harness.add_tool(create_subagent_tool(harness, []))

    task = asyncio.create_task(harness.run("delegate"))

    with pytest.raises(RuntimeError, match="strict abort"):
        await asyncio.wait_for(task, timeout=1)

def test_subagent_child_provider_failure_returns_tool_error(tmp_path: Path) -> None:
    parent_call = ModelTurn(
        tool_calls=[ModelToolCall(id="call_1", name="subagent", arguments='{"task":"help"}')],
        raw={"id": "parent-start"},
    )

    def on_parent_continue(outputs, _tools, _metadata):
        envelope = tool_output(outputs[0].output)
        assert envelope["ok"] is False
        assert envelope["metadata"]["agent"] == "default"
        assert envelope["metadata"]["inherited"] is True
        assert envelope["metadata"]["tool_mode"] == "inherited"
        assert envelope["metadata"]["tools"] == ["echo"]
        assert envelope["metadata"]["error_type"] == "HarnessError"

    parent = ScriptedSession(start_turn=parent_call, continue_turn=ModelTurn(text="parent done", raw={"id": "parent-done"}), on_continue=on_parent_continue)
    tracer = FakeTracer()
    harness = Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[]),
        model=ScriptedModel([parent, FailingSession()]),
        tools=[echo_tool()],
        tracing=[TracingOptions(tracer=tracer)],
    )
    harness.add_tool(create_subagent_tool(harness, []))

    assert harness.run_sync("delegate").text == "parent done"
    subagent_tool = next(span for span in tracer.spans if span.name == "execute_tool subagent")
    assert subagent_tool.attributes["subagent.name"] == "default"
    assert subagent_tool.attributes["subagent.tool_mode"] == "inherited"
    assert subagent_tool.attributes["subagent.tools"] == ["echo"]
    assert subagent_tool.status is not None

def test_unknown_named_subagent_returns_structured_error(tmp_path: Path) -> None:
    harness = Harness(HarnessConfig(root=tmp_path, builtin_tools=[]), model=ScriptedModel([]))
    tool = create_subagent_tool(harness, [SubAgentConfig(name="research", description="Research helper.", builtin_tools=["read"])])

    output = tool_output(asyncio.run(tool.handler(tool.parse_args({"task": "x", "agent": "missing"}))).as_json())

    assert output["ok"] is False
    assert output["metadata"]["available"] == ["research"]
    assert output["metadata"]["error_type"] == "UnknownSubAgent"

def test_blank_subagent_name_is_normal_argument_validation_error(tmp_path: Path) -> None:
    harness = Harness(HarnessConfig(root=tmp_path, builtin_tools=[]), model=ScriptedModel([]))
    tool = create_subagent_tool(harness, [])
    harness.add_tool(tool)

    output = tool_output(call_tool(tool, '{"task":"x","agent":""}'))

    assert output["ok"] is False
    assert output["metadata"]["error_type"] == "ValidationError"
    assert "Invalid arguments" in output["content"]

def test_subagent_tool_name_is_reserved_for_custom_tools(tmp_path: Path) -> None:
    custom = ToolSpec("subagent", "Not the framework tool.", {"type": "object", "properties": {}}, lambda args: "bad")
    harness = Harness(HarnessConfig(root=tmp_path, builtin_tools=[]), model=ScriptedModel([]))

    with pytest.raises(ValueError, match="reserved tool name"):
        Harness(HarnessConfig(root=tmp_path, builtin_tools=[]), model=ScriptedModel([]), tools=[custom])
    with pytest.raises(ValueError, match="reserved tool name"):
        harness.add_tool(custom)

def test_framework_subagent_tool_can_be_added_after_construction(tmp_path: Path) -> None:
    harness = Harness(HarnessConfig(root=tmp_path, builtin_tools=[]), model=ScriptedModel([]))

    harness.add_tool(create_subagent_tool(harness, []))

    assert [tool["name"] for tool in harness.tool_schemas()] == ["subagent"]

def test_explicit_subagent_hook_registry_strict_mode_is_preserved(tmp_path: Path) -> None:
    parent_call = ModelTurn(
        tool_calls=[ModelToolCall(id="call_1", name="subagent", arguments='{"task":"help"}')],
        raw={"id": "parent-start"},
    )
    outputs_seen = []

    def on_continue(outputs, _tools, _metadata):
        outputs_seen.append(tool_output(outputs[0].output))

    parent = ScriptedSession(
        start_turn=parent_call,
        continue_turn=ModelTurn(text="parent done", raw={"id": "parent-done"}),
        on_continue=on_continue,
    )
    child_registry = HookRegistry([Hook("run_start", lambda ctx: (_ for _ in ()).throw(RuntimeError("strict child")))], strict_hooks=True)
    harness = Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[], strict_hooks=False),
        model=ScriptedModel([parent, ScriptedSession(start_turn=ModelTurn(text="child", raw={"id": "child"}))]),
        subagent_hooks={DEFAULT_SUBAGENT_NAME: child_registry},
    )
    harness.add_tool(create_subagent_tool(harness, []))

    assert harness.run_sync("go").text == "parent done"
    assert child_registry.strict_hooks is True
    assert outputs_seen[0]["ok"] is False
    assert outputs_seen[0]["metadata"]["error_type"] == "RuntimeError"

def test_subagent_hooks_and_child_hooks_are_explicit(tmp_path: Path) -> None:
    parent_call = ModelTurn(
        tool_calls=[ModelToolCall(id="call_1", name="subagent", arguments='{"task":"help","agent":"research"}')],
        raw={"id": "parent-start"},
    )
    child = ScriptedSession(start_turn=ModelTurn(text="child done", raw={"id": "child"}))
    parent = ScriptedSession(start_turn=parent_call, continue_turn=ModelTurn(text="parent done", raw={"id": "parent-done"}))
    events = []

    def before_subagent(ctx):
        assert isinstance(ctx, BeforeSubagentRunContext)
        events.append((ctx.event, ctx.agent, ctx.parent_call_id))

    def child_start(ctx):
        events.append((ctx.event, DEFAULT_SUBAGENT_NAME if ctx.harness.config.subagents else "child"))

    def after_subagent(ctx):
        assert isinstance(ctx, AfterSubagentRunContext)
        events.append((ctx.event, ctx.agent, ctx.usage.model_requests))

    harness = Harness(
        HarnessConfig(
            root=tmp_path,
            builtin_tools=[],
            subagents=[SubAgentConfig(name="research", description="Research helper.", tools=[echo_tool()])],
        ),
        model=ScriptedModel([parent, child]),
        hooks=[
            Hook("before_subagent_run", before_subagent, agents=["research"]),
            Hook("after_subagent_run", after_subagent, agents=["research"]),
        ],
        subagent_hooks={"research": [Hook("run_start", child_start)]},
    )
    harness.add_tool(create_subagent_tool(harness, harness.config.subagents))

    assert build_child_harness(harness, harness.config.subagents[0]).hooks.hooks
    assert build_child_harness(harness, None).hooks.hooks == []
    assert harness.run_sync("delegate").text == "parent done"
    assert events == [
        ("before_subagent_run", "research", "call_1"),
        ("run_start", "child"),
        ("after_subagent_run", "research", 1),
    ]

def test_subagent_hook_can_cancel_default_agent_without_child_run(tmp_path: Path) -> None:
    parent_call = ModelTurn(
        tool_calls=[ModelToolCall(id="call_1", name="subagent", arguments='{"task":"help"}')],
        raw={"id": "parent-start"},
    )
    events = []
    outputs_seen = []

    def on_continue(outputs, _tools, _metadata):
        outputs_seen.append(tool_output(outputs[0].output))

    parent = ScriptedSession(
        start_turn=parent_call,
        continue_turn=ModelTurn(text="parent done", raw={"id": "parent-done"}),
        on_continue=on_continue,
    )

    def cancel(ctx):
        assert isinstance(ctx, BeforeSubagentRunContext)
        events.append((ctx.agent, ctx.parent_call_id))
        ctx.cancelled = True
        ctx.cancel_reason = "blocked"

    harness = Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[]),
        model=ScriptedModel([parent]),
        hooks=[Hook("before_subagent_run", cancel, agents=[DEFAULT_SUBAGENT_NAME])],
        subagent_hooks={DEFAULT_SUBAGENT_NAME: [Hook("run_start", lambda ctx: pytest.fail("child should not run"))]},
    )
    harness.add_tool(create_subagent_tool(harness, []))

    assert harness.run_sync("delegate").text == "parent done"
    assert events == [(DEFAULT_SUBAGENT_NAME, "call_1")]
    assert outputs_seen[0]["ok"] is False
    assert outputs_seen[0]["content"] == "Subagent execution blocked by hook: blocked"
    assert outputs_seen[0]["metadata"]["error_type"] == "SubAgentCancelled"

def test_default_subagent_name_is_reserved() -> None:
    with pytest.raises(ValueError, match="reserved"):
        SubAgentConfig(name=DEFAULT_SUBAGENT_NAME, description="Reserved.", builtin_tools=["read"])
