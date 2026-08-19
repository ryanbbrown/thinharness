from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from fakes import FailingSession, FakeTracer, ScriptedModel, ScriptedSession, echo_tool, tool_output

from thinharness import (
    DEFAULT_SUBAGENT_NAME,
    AfterSubagentRunContext,
    BeforeSubagentRunContext,
    ChildHarnessOutcome,
    ChildHarnessRequest,
    FilesystemPlugin,
    Harness,
    HarnessConfig,
    Hook,
    HookRegistry,
    PluginBinding,
    PluginContext,
    PluginContribution,
    SkillsPlugin,
    SubAgentConfig,
    SubagentsPlugin,
    ToolOrigin,
    ToolResult,
    ToolSpec,
    TracingOptions,
    call_tool,
)
from thinharness.providers import ModelToolCall, ModelTurn


class FakeChildHost:
    def __init__(self, outcome: ChildHarnessOutcome | None = None) -> None:
        self.outcome = outcome
        self.tools: list[ToolSpec] = []
        self.recipes: list[ChildHarnessRequest] = []
        self.requests: list[ChildHarnessRequest] = []

    def register_delegation_tool(self, tool: ToolSpec, recipes) -> ToolSpec:
        self.tools.append(tool)
        self.recipes.extend(recipes)
        return tool

    async def run(self, request: ChildHarnessRequest) -> ChildHarnessOutcome:
        self.requests.append(request)
        if self.outcome is None:
            raise AssertionError("unexpected child run")
        return self.outcome


class ClosingProvider:
    name = "OpenAI"

    def __init__(self) -> None:
        self.closed = 0

    async def aclose(self) -> None:
        self.closed += 1


def _parent_call(*, agent: str | None = None, task: str = "help", call_id: str = "call_1") -> ModelTurn:
    agent_arg = f',"agent":"{agent}"' if agent is not None else ""
    return ModelTurn(
        tool_calls=[ModelToolCall(id=call_id, name="subagent", arguments=f'{{"task":"{task}"{agent_arg}}}')],
        raw={"id": "parent-start"},
    )


def _plugin_tool(plugin: SubagentsPlugin, tmp_path: Path) -> tuple[FakeChildHost, ToolSpec, PluginBinding]:
    host = FakeChildHost()
    binding = plugin.bind(PluginContext(root=tmp_path, model=ScriptedModel([]), child_harnesses=host))
    return host, binding.static.tools[0], binding


def test_plugin_contributes_static_tool_and_default_agent(tmp_path: Path) -> None:
    host, tool, binding = _plugin_tool(SubagentsPlugin(), tmp_path)

    assert binding.agent_names == (DEFAULT_SUBAGENT_NAME,)
    assert host.tools == [tool]
    assert tool.name == "subagent"
    assert tool.origin is None
    assert set(tool.response_tool()["parameters"]["properties"]) == {"task", "agent"}
    assert "Omit `agent`" in tool.description
    assert host.recipes[0].tool_mode == "inherited"


def test_plain_harness_has_no_delegation_and_same_named_direct_tool_is_valid(tmp_path: Path) -> None:
    direct = ToolSpec(
        "subagent",
        "An ordinary custom tool.",
        {"type": "object", "properties": {}},
        lambda _args: "ordinary",
        origin=ToolOrigin(plugin="subagents"),
    )
    harness = Harness(HarnessConfig(root=tmp_path), model=ScriptedModel([]), tools=[direct])

    assert harness.tools == [direct]
    with pytest.raises(ValueError, match="duplicate tool name"):
        Harness(
            HarnessConfig(root=tmp_path),
            model=ScriptedModel([]),
            plugins=[SubagentsPlugin()],
            tools=[direct],
        )


def test_subagent_config_accepts_additive_and_model_only_children() -> None:
    spec = echo_tool()
    config = SubAgentConfig(
        name="research.1",
        description="Research helper.",
        inherit_parent=True,
        plugins=[FilesystemPlugin(tools=["read"])],
        tools=[spec],
    )
    model_only = SubAgentConfig(name="model-only", description="No tools.", model="openai:child")

    assert config.inherit_parent is True
    assert config.tools == (spec,)
    assert model_only.plugins == ()
    assert model_only.tools == ()


@pytest.mark.parametrize("field", ["inherit_parent_tools", "inherit_mcp_servers", "mcp_servers", "builtin_tools"])
def test_removed_subagent_config_fields_fail_by_name(field: str) -> None:
    with pytest.raises(ValueError, match=rf"SubAgentConfig\.{field} has been removed"):
        SubAgentConfig(name="old", description="Old helper.", **{field: True})


def test_subagent_config_and_plugin_validate_order_names_plugins_and_tools() -> None:
    with pytest.raises(ValueError, match="reserved"):
        SubAgentConfig(name="default", description="Reserved.")
    with pytest.raises(ValueError, match="single line"):
        SubAgentConfig(name="bad", description="Bad\nline")
    with pytest.raises(ValueError):
        SubAgentConfig(name="bad name", description="Bad name.")
    with pytest.raises(TypeError, match="ordered sequence"):
        SubagentsPlugin(agents={SubAgentConfig(name="a", description="A.")})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="duplicate subagent name"):
        SubagentsPlugin(agents=[SubAgentConfig(name="a", description="A."), SubAgentConfig(name="a", description="Again.")])
    with pytest.raises(TypeError, match="Plugin values"):
        SubAgentConfig(name="bad-plugin", description="Bad.", plugins=[object()])
    with pytest.raises(ValueError, match="SubagentsPlugin cannot"):
        SubAgentConfig(name="nested", description="Nested.", plugins=[SubagentsPlugin()])
    approval = ToolSpec("approve", "Approve", {"type": "object"}, lambda _args: "ok", requires_approval=True)
    with pytest.raises(ValueError, match="child harnesses"):
        SubAgentConfig(name="approval", description="Approval.", tools=[approval])


def test_child_hook_validation_and_registry_copy(tmp_path: Path) -> None:
    registry = HookRegistry([Hook("run_start", lambda _ctx: None)], strict_hooks=True)
    config = SubAgentConfig(name="safe", description="Safe.", hooks=registry)
    plugin = SubagentsPlugin(default_hooks=registry, agents=[config])
    registry.hooks.clear()
    host, _tool, _binding = _plugin_tool(plugin, tmp_path)

    assert isinstance(plugin.default_hooks, HookRegistry)
    assert len(plugin.default_hooks.hooks) == 1
    assert all(isinstance(recipe.hooks, HookRegistry) for recipe in host.recipes)
    assert all(recipe.hooks.strict_hooks for recipe in host.recipes if isinstance(recipe.hooks, HookRegistry))
    assert all(len(recipe.hooks.hooks) == 1 for recipe in host.recipes if isinstance(recipe.hooks, HookRegistry))
    with pytest.raises(ValueError, match="lifecycle"):
        SubagentsPlugin(default_hooks=[Hook("before_subagent_run", lambda _ctx: None)])
    invalid = SubAgentConfig(
        name="bad",
        description="Bad.",
        hooks=[Hook("after_subagent_run", lambda _ctx: None, agents=["bad"])],
    )
    with pytest.raises(ValueError, match="lifecycle"):
        SubagentsPlugin(agents=[invalid])


def test_plugin_name_and_configuration_are_frozen() -> None:
    plugin = SubagentsPlugin(agents=[SubAgentConfig(name="a", description="A.")])

    with pytest.raises(AttributeError, match="fixed"):
        plugin.name = "other"  # type: ignore[misc]
    with pytest.raises(AttributeError, match="frozen"):
        plugin._agents = ()


def test_unknown_and_blank_agent_are_normal_tool_results(tmp_path: Path) -> None:
    host, tool, _ = _plugin_tool(
        SubagentsPlugin(agents=[SubAgentConfig(name="research", description="Research.")]),
        tmp_path,
    )

    unknown = tool_output(asyncio.run(tool.handler(tool.parse_args({"task": "x", "agent": "missing"}))).to_json())
    blank = tool_output(call_tool(tool, '{"task":"x","agent":""}'))

    assert unknown["metadata"] == {
        "agent": "missing",
        "available": ["research"],
        "error_type": "UnknownSubAgent",
    }
    assert blank["metadata"]["error_type"] == "ValidationError"
    assert host.requests == []


def test_default_child_runs_with_frozen_direct_tools_and_returns_metadata(tmp_path: Path) -> None:
    child_seen = {}

    def on_child_start(prompt, _instructions, tools, metadata, _previous_response_id):
        child_seen.update(prompt=prompt, tools=[tool["name"] for tool in tools], metadata=metadata)

    parent = ScriptedSession(
        start_turn=_parent_call(),
        continue_turn=ModelTurn(text="parent done", raw={"id": "parent-done"}),
    )
    child = ScriptedSession(start_turn=ModelTurn(text="child done", raw={"id": "child"}), on_start=on_child_start)
    harness = Harness(
        HarnessConfig(root=tmp_path),
        model=ScriptedModel([parent, child]),
        plugins=[SubagentsPlugin()],
        tools=[echo_tool()],
    )

    result = harness.run_sync("delegate", metadata={"conversation_id": "conv", "private": "parent"})
    envelope = tool_output(parent.continue_calls[0][0][0].output)

    assert result.text == "parent done"
    assert child_seen == {
        "prompt": "help",
        "tools": ["echo"],
        "metadata": {"conversation_id": "conv", "parent_call_id": "call_1"},
    }
    assert envelope["metadata"] == {
        "agent": "default",
        "inherited": True,
        "tool_mode": "inherited",
        "tools": ["echo"],
        "model_requests": 1,
        "structured_output": False,
    }


def test_named_model_only_child_has_no_tools(tmp_path: Path) -> None:
    child_seen = {}

    def on_child_start(_prompt, _instructions, tools, _metadata, _previous_response_id):
        child_seen["tools"] = tools

    parent = ScriptedSession(
        start_turn=_parent_call(agent="solo"),
        continue_turn=ModelTurn(text="done", raw={}),
    )
    child_model = ScriptedModel([
        ScriptedSession(start_turn=ModelTurn(text="solo answer", raw={}), on_start=on_child_start)
    ])
    child_model.provider = ClosingProvider()
    harness = Harness(
        HarnessConfig(root=tmp_path),
        model=ScriptedModel([parent]),
        plugins=[SubagentsPlugin(agents=[SubAgentConfig(name="solo", description="Solo.", model="openai:child")])],
    )

    import thinharness.children as children_module

    original = children_module.infer_model
    children_module.infer_model = lambda *_args, **_kwargs: child_model
    try:
        assert harness.run_sync("delegate").text == "done"
    finally:
        children_module.infer_model = original

    envelope = tool_output(parent.continue_calls[0][0][0].output)
    assert child_seen["tools"] == []
    assert envelope["metadata"]["tool_mode"] == "explicit"
    assert child_model.provider.closed == 1


def test_override_model_closes_when_child_construction_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    parent_model = ScriptedModel([])
    child_model = ScriptedModel([])
    child_provider = ClosingProvider()
    child_model.provider = child_provider

    class ModelSensitivePlugin:
        name = "model-sensitive"

        def bind(self, context):
            tools = () if context.model is parent_model else (
                ToolSpec("duplicate", "Duplicate", {"type": "object"}, lambda _args: "plugin"),
            )
            return PluginBinding(static=PluginContribution(tools=tools))

    parent = ScriptedSession(
        start_turn=_parent_call(agent="broken"),
        continue_turn=ModelTurn(text="done", raw={}),
    )
    monkeypatch.setattr("thinharness.children.infer_model", lambda *_args, **_kwargs: child_model)
    harness = Harness(
        HarnessConfig(root=tmp_path),
        model=parent_model,
        plugins=[SubagentsPlugin(agents=[
            SubAgentConfig(
                name="broken",
                description="Fails during child construction.",
                model="openai:child",
                plugins=[ModelSensitivePlugin()],
                tools=[ToolSpec("duplicate", "Duplicate", {"type": "object"}, lambda _args: "direct")],
            )
        ])],
    )
    parent_model.sessions.append(parent)

    assert harness.run_sync("delegate").text == "done"
    envelope = tool_output(parent.continue_calls[0][0][0].output)
    assert envelope["metadata"]["error_type"] == "ValueError"
    assert child_provider.closed == 1


def test_additive_inheritance_rebinds_plugins_and_appends_direct_tools(tmp_path: Path) -> None:
    child_seen = {}

    def on_child_start(_prompt, instructions, tools, _metadata, _previous_response_id):
        child_seen.update(instructions=instructions, tools=[tool["name"] for tool in tools])

    parent = ScriptedSession(start_turn=_parent_call(agent="mixed"), continue_turn=ModelTurn(text="done", raw={}))
    child = ScriptedSession(start_turn=ModelTurn(text="child", raw={}), on_start=on_child_start)
    explicit = ToolSpec("explicit", "Explicit", {"type": "object"}, lambda _args: "ok")
    harness = Harness(
        HarnessConfig(root=tmp_path),
        model=ScriptedModel([parent, child]),
        plugins=[
            FilesystemPlugin(tools=["read"]),
            SubagentsPlugin(agents=[
                SubAgentConfig(
                    name="mixed",
                    description="Mixed.",
                    inherit_parent=True,
                    tools=[explicit],
                )
            ]),
        ],
        tools=[echo_tool()],
    )

    harness.run_sync("delegate")
    envelope = tool_output(parent.continue_calls[0][0][0].output)

    assert child_seen["tools"] == ["read", "echo", "explicit"]
    assert child_seen["instructions"].count(f"Workspace root: {tmp_path.resolve()}") == 1
    assert envelope["metadata"]["tool_mode"] == "inherited+explicit"


def test_skills_plugin_reuses_registry_and_summary_in_child(tmp_path: Path) -> None:
    skill = tmp_path / "skills" / "demo"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: demo\ndescription: Demo skill\n---\nBody", encoding="utf-8")
    plugin = SkillsPlugin(tmp_path / "skills", tools=["skill_read"])
    child_seen = {}

    def on_child_start(_prompt, instructions, tools, _metadata, _previous_response_id):
        child_seen.update(instructions=instructions, tools=[tool["name"] for tool in tools])

    harness = Harness(
        HarnessConfig(root=tmp_path),
        model=ScriptedModel([
            ScriptedSession(start_turn=_parent_call(), continue_turn=ModelTurn(text="done", raw={})),
            ScriptedSession(start_turn=ModelTurn(text="child", raw={}), on_start=on_child_start),
        ]),
        plugins=[plugin, SubagentsPlugin()],
    )
    harness.run_sync("go")

    assert plugin.for_child() is plugin
    assert child_seen["tools"] == ["skill_read"]
    assert child_seen["instructions"].count("demo - Demo skill") == 1


def test_mcp_and_non_inheritable_custom_plugins_do_not_inherit(tmp_path: Path) -> None:
    class OrdinaryPlugin:
        name = "ordinary"

        def bind(self, _context):
            return PluginBinding(static=PluginContribution(tools=(ToolSpec("ordinary", "Ordinary", {"type": "object"}, lambda _args: "ok"),)))

    seen = {}

    def on_child_start(_prompt, _instructions, tools, _metadata, _previous_response_id):
        seen["tools"] = [tool["name"] for tool in tools]

    harness = Harness(
        HarnessConfig(root=tmp_path),
        model=ScriptedModel([
            ScriptedSession(start_turn=_parent_call(), continue_turn=ModelTurn(text="done", raw={})),
            ScriptedSession(start_turn=ModelTurn(text="child", raw={}), on_start=on_child_start),
        ]),
        plugins=[OrdinaryPlugin(), SubagentsPlugin()],
    )
    harness.run_sync("go")

    assert seen["tools"] == []


def test_custom_for_child_plugin_rebinds_and_invalid_return_fails(tmp_path: Path) -> None:
    roots = []

    class Inheritable:
        name = "custom"

        def for_child(self):
            return self

        def bind(self, context):
            roots.append(context.root)
            return PluginBinding(static=PluginContribution(tools=(ToolSpec("custom", "Custom", {"type": "object"}, lambda _args: "ok"),)))

    plugin = Inheritable()
    harness = Harness(
        HarnessConfig(root=tmp_path),
        model=ScriptedModel([
            ScriptedSession(start_turn=_parent_call(), continue_turn=ModelTurn(text="done", raw={})),
            ScriptedSession(start_turn=ModelTurn(text="child", raw={})),
        ]),
        plugins=[plugin, SubagentsPlugin()],
    )
    harness.run_sync("go")
    assert roots == [tmp_path.resolve(), tmp_path.resolve()]

    class Invalid(Inheritable):
        name = "invalid"

        def for_child(self):
            return object()

    with pytest.raises(TypeError, match="for_child"):
        Harness(HarnessConfig(root=tmp_path), model=ScriptedModel([]), plugins=[Invalid(), SubagentsPlugin()])


def test_known_child_plugin_and_tool_collisions_fail_parent_construction(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="duplicate plugin name"):
        Harness(
            HarnessConfig(root=tmp_path),
            model=ScriptedModel([]),
            plugins=[
                FilesystemPlugin(tools=["read"]),
                SubagentsPlugin(agents=[
                    SubAgentConfig(
                        name="collision",
                        description="Collision.",
                        inherit_parent=True,
                        plugins=[FilesystemPlugin(tools=["write"])],
                    )
                ]),
            ],
        )

    direct = echo_tool()
    harness = Harness(
        HarnessConfig(root=tmp_path),
        model=ScriptedModel([]),
        plugins=[SubagentsPlugin(agents=[
            SubAgentConfig(name="collision", description="Collision.", inherit_parent=True, tools=[direct])
        ])],
    )
    with pytest.raises(ValueError, match="duplicate tool name"):
        harness.add_tool(direct)


def test_inheritable_plugin_approval_tool_fails_parent_construction(tmp_path: Path) -> None:
    class Unsafe:
        name = "unsafe"

        def for_child(self):
            return self

        def bind(self, _context):
            tool = ToolSpec("unsafe", "Unsafe", {"type": "object"}, lambda _args: "ok", requires_approval=True)
            return PluginBinding(static=PluginContribution(tools=(tool,)))

    with pytest.raises(ValueError, match="child harnesses"):
        Harness(HarnessConfig(root=tmp_path), model=ScriptedModel([]), plugins=[Unsafe(), SubagentsPlugin()])


def test_direct_tool_added_during_run_is_visible_only_next_run(tmp_path: Path) -> None:
    late = ToolSpec("late", "Late", {"type": "object"}, lambda _args: "ok")
    seen: list[list[str]] = []

    def add_late(ctx):
        if ctx.tool_name == "subagent" and "late" not in [tool.name for tool in ctx.harness.tools]:
            ctx.harness.add_tool(late)

    def child_start(_prompt, _instructions, tools, _metadata, _previous_response_id):
        seen.append([tool["name"] for tool in tools])

    harness = Harness(
        HarnessConfig(root=tmp_path),
        model=ScriptedModel([
            ScriptedSession(start_turn=_parent_call(), continue_turn=ModelTurn(text="one", raw={})),
            ScriptedSession(start_turn=ModelTurn(text="child one", raw={}), on_start=child_start),
            ScriptedSession(start_turn=_parent_call(call_id="call_2"), continue_turn=ModelTurn(text="two", raw={})),
            ScriptedSession(start_turn=ModelTurn(text="child two", raw={}), on_start=child_start),
        ]),
        plugins=[SubagentsPlugin()],
        hooks=[Hook("before_tool_call", add_late, tools=["subagent"])],
    )

    assert asyncio.run(harness.run("first")).text == "one"
    assert asyncio.run(harness.run("second")).text == "two"
    assert seen == [[], ["late"]]


def test_parent_hooks_child_hooks_metadata_and_cancellation(tmp_path: Path) -> None:
    events = []

    def before(ctx):
        assert isinstance(ctx, BeforeSubagentRunContext)
        events.append((ctx.event, ctx.agent, dict(ctx.metadata)))
        ctx.metadata["changed"] = True

    def child_start(ctx):
        events.append((ctx.event, "child", dict(ctx.metadata)))

    def after(ctx):
        assert isinstance(ctx, AfterSubagentRunContext)
        events.append((ctx.event, ctx.agent, dict(ctx.metadata)))

    harness = Harness(
        HarnessConfig(root=tmp_path),
        model=ScriptedModel([
            ScriptedSession(start_turn=_parent_call(), continue_turn=ModelTurn(text="done", raw={})),
            ScriptedSession(start_turn=ModelTurn(text="child", raw={})),
        ]),
        plugins=[SubagentsPlugin(default_hooks=[Hook("run_start", child_start)])],
        hooks=[
            Hook("before_subagent_run", before, agents=["default"]),
            Hook("after_subagent_run", after, agents=["default"]),
        ],
    )
    harness.run_sync("go", metadata={"conversation_id": "c"})

    assert events == [
        ("before_subagent_run", "default", {"conversation_id": "c"}),
        ("run_start", "child", {"conversation_id": "c", "parent_call_id": "call_1"}),
        ("after_subagent_run", "default", {"conversation_id": "c"}),
    ]

    def cancel(ctx):
        ctx.cancelled = True
        ctx.cancel_reason = "blocked"

    parent = ScriptedSession(start_turn=_parent_call(), continue_turn=ModelTurn(text="blocked", raw={}))
    blocked = Harness(
        HarnessConfig(root=tmp_path),
        model=ScriptedModel([parent]),
        plugins=[SubagentsPlugin()],
        hooks=[Hook("before_subagent_run", cancel, agents=["default"])],
    )
    blocked.run_sync("go")
    envelope = tool_output(parent.continue_calls[0][0][0].output)
    assert envelope["metadata"]["error_type"] == "SubAgentCancelled"


def test_agent_filtered_hook_requires_plugin_and_known_name(tmp_path: Path) -> None:
    hook = Hook("before_subagent_run", lambda _ctx: None, agents=["default"])
    with pytest.raises(ValueError, match="unknown subagent name"):
        Harness(HarnessConfig(root=tmp_path), model=ScriptedModel([]), hooks=[hook])
    with pytest.raises(ValueError, match="missing"):
        Harness(
            HarnessConfig(root=tmp_path),
            model=ScriptedModel([]),
            plugins=[SubagentsPlugin()],
            hooks=[Hook("before_subagent_run", lambda _ctx: None, agents=["missing"])],
        )


def test_child_provider_failure_and_close_are_reported(tmp_path: Path) -> None:
    parent = ScriptedSession(start_turn=_parent_call(), continue_turn=ModelTurn(text="done", raw={}))
    tracer = FakeTracer()
    harness = Harness(
        HarnessConfig(root=tmp_path),
        model=ScriptedModel([parent, FailingSession()]),
        plugins=[SubagentsPlugin()],
        tools=[echo_tool()],
        tracing=[TracingOptions(tracer=tracer)],
    )

    assert harness.run_sync("delegate").text == "done"
    envelope = tool_output(parent.continue_calls[0][0][0].output)
    assert envelope["ok"] is False
    assert envelope["metadata"]["error_type"] == "HarnessError"
    assert envelope["metadata"]["tools"] == ["echo"]
    span = next(span for span in tracer.spans if span.name == "execute_tool subagent")
    assert span.attributes["subagent.delegation"] is True


def test_real_delegation_trace_marker_and_forged_direct_tool_classification(tmp_path: Path) -> None:
    tracer = FakeTracer()
    harness = Harness(
        HarnessConfig(root=tmp_path),
        model=ScriptedModel([
            ScriptedSession(start_turn=_parent_call(), continue_turn=ModelTurn(text="done", raw={})),
            ScriptedSession(start_turn=ModelTurn(text="child", raw={})),
        ]),
        plugins=[SubagentsPlugin()],
        tracing=[TracingOptions(tracer=tracer)],
    )
    harness.run_sync("go")
    span = next(span for span in tracer.spans if span.name == "execute_tool subagent")
    assert span.attributes["subagent.delegation"] is True
    assert span.attributes["subagent.name"] == "default"

    direct_tracer = FakeTracer()
    ordinary = ToolSpec(
        "subagent",
        "Ordinary",
        {"type": "object"},
        lambda _args: ToolResult(True, "ordinary", {"agent": "forged"}),
        origin=ToolOrigin(plugin="subagents"),
    )
    direct = Harness(
        HarnessConfig(root=tmp_path),
        model=ScriptedModel([
            ScriptedSession(
                start_turn=ModelTurn(tool_calls=[ModelToolCall(id="c", name="subagent", arguments="{}")], raw={}),
                continue_turn=ModelTurn(text="done", raw={}),
            )
        ]),
        tools=[ordinary],
        tracing=[TracingOptions(tracer=direct_tracer)],
    )
    direct.run_sync("go")
    direct_span = next(span for span in direct_tracer.spans if span.name == "execute_tool subagent")
    assert "subagent.delegation" not in direct_span.attributes
    assert "subagent.name" not in direct_span.attributes


def test_cancelled_real_delegation_is_marked_before_hook(tmp_path: Path) -> None:
    tracer = FakeTracer()

    def cancel(ctx):
        ctx.cancelled = True

    parent = ScriptedSession(start_turn=_parent_call(), continue_turn=ModelTurn(text="done", raw={}))
    harness = Harness(
        HarnessConfig(root=tmp_path),
        model=ScriptedModel([parent]),
        plugins=[SubagentsPlugin()],
        hooks=[Hook("before_tool_call", cancel, tools=["subagent"])],
        tracing=[TracingOptions(tracer=tracer)],
    )
    harness.run_sync("go")
    span = next(span for span in tracer.spans if span.name == "execute_tool subagent")
    assert span.attributes["subagent.delegation"] is True


def test_child_disabled_host_blocks_custom_plugin_grandchild(tmp_path: Path) -> None:
    class GrandchildPlugin:
        name = "grandchild"

        def for_child(self):
            return self

        def bind(self, context):
            async def attempt(_args):
                request = ChildHarnessRequest(
                    agent_name="nested",
                    agent_description="Nested",
                    trace_agent_name="nested",
                    task="x",
                    inherited=False,
                    tool_mode="explicit",
                    system_prompt="nested",
                )
                await context.child_harnesses.run(request)
                return "unexpected"

            return PluginBinding(static=PluginContribution(tools=(ToolSpec("attempt", "Attempt", {"type": "object"}, attempt),)))

    parent = ScriptedSession(start_turn=_parent_call(), continue_turn=ModelTurn(text="done", raw={}))
    child = ScriptedSession(
        start_turn=ModelTurn(tool_calls=[ModelToolCall(id="nested", name="attempt", arguments="{}")], raw={}),
        continue_turn=ModelTurn(text="child done", raw={}),
    )
    harness = Harness(
        HarnessConfig(root=tmp_path),
        model=ScriptedModel([parent, child]),
        plugins=[GrandchildPlugin(), SubagentsPlugin()],
    )
    harness.run_sync("go")
    nested = tool_output(child.continue_calls[0][0][0].output)
    assert nested["ok"] is False
    assert nested["metadata"]["error_type"] == "HarnessError"
    assert "cannot create nested" in nested["content"]


async def test_top_level_host_rejects_outside_active_tool_runtime(tmp_path: Path) -> None:
    harness = Harness(HarnessConfig(root=tmp_path), model=ScriptedModel([]), plugins=[SubagentsPlugin()])
    request = ChildHarnessRequest(
        agent_name="x",
        agent_description="X",
        trace_agent_name="x",
        task="x",
        inherited=False,
        tool_mode="explicit",
        system_prompt="x",
    )

    with pytest.raises(Exception, match="active parent tool call"):
        await harness._child_harnesses.run(request)


def test_plugin_object_reuse_binds_independent_hosts(tmp_path: Path) -> None:
    plugin = SubagentsPlugin()
    first = Harness(HarnessConfig(root=tmp_path / "one"), model=ScriptedModel([]), plugins=[plugin])
    second = Harness(HarnessConfig(root=tmp_path / "two"), model=ScriptedModel([]), plugins=[plugin])

    assert first.tools[0].handler is not second.tools[0].handler
    assert first._child_harnesses is not second._child_harnesses


def test_harness_removed_fields_and_constructor_helpers_are_gone() -> None:
    for field in ("builtin_tools", "subagents"):
        with pytest.raises(ValueError, match=rf"HarnessConfig\.{field} has been removed.*SubagentsPlugin"):
            HarnessConfig(**{field: []})
    with pytest.raises(TypeError, match="subagent_hooks"):
        Harness(subagent_hooks={})  # type: ignore[call-arg]

    import thinharness

    assert not hasattr(thinharness, "create_subagent_tool")
    assert not hasattr(thinharness, "build_child_harness")
    assert "kind" not in ToolSpec.__dataclass_fields__
