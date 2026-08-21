from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from fakes import FailingSession, FakeTracer, ScriptedModel, ScriptedSession, echo_tool, tool_output

from thinharness import (
    DEFAULT_SUBAGENT_NAME,
    AfterSubagentRunContext,
    BeforeSubagentRunContext,
    ChildHarnessOutcome,
    ChildHarnessRequest,
    ChildInheritablePlugin,
    FilesystemPlugin,
    Harness,
    HarnessConfig,
    Hook,
    HookRegistry,
    ParallelLlmPlugin,
    PluginBinding,
    PluginContext,
    PluginContribution,
    SkillsPlugin,
    StreamOptions,
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
    assert roots
    assert all(root == tmp_path.resolve() for root in roots)

    class Invalid(Inheritable):
        name = "invalid"

        def for_child(self):
            return object()

    with pytest.raises(TypeError, match="for_child"):
        Harness(HarnessConfig(root=tmp_path), model=ScriptedModel([]), plugins=[Invalid(), SubagentsPlugin()])


def test_rebound_child_static_tool_collision_fails_parent_construction(tmp_path: Path) -> None:
    class ReboundPlugin:
        name = "rebound"

        def bind(self, _context: PluginContext) -> PluginBinding:
            tool = ToolSpec("collision", "Child collision.", {"type": "object"}, lambda _args: "child")
            return PluginBinding(static=PluginContribution(tools=(tool,)))

    class ParentPlugin:
        name = "rebound"

        def bind(self, _context: PluginContext) -> PluginBinding:
            tool = ToolSpec("parent_only", "Parent only.", {"type": "object"}, lambda _args: "parent")
            return PluginBinding(static=PluginContribution(tools=(tool,)))

        def for_child(self) -> ReboundPlugin:
            return ReboundPlugin()

    plugin = ParentPlugin()
    assert isinstance(plugin, ChildInheritablePlugin)

    with pytest.raises(ValueError, match="duplicate tool name: collision"):
        Harness(
            HarnessConfig(root=tmp_path),
            model=ScriptedModel([]),
            plugins=[
                plugin,
                SubagentsPlugin(agents=[
                    SubAgentConfig(
                        name="collision",
                        description="Collision.",
                        inherit_parent=True,
                        tools=[
                            ToolSpec(
                                "collision",
                                "Explicit collision.",
                                {"type": "object"},
                                lambda _args: "explicit",
                            )
                        ],
                    )
                ]),
            ],
        )


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


async def test_detached_delegation_task_loses_active_call_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release = asyncio.Event()
    detached: list[asyncio.Task[ChildHarnessOutcome]] = []
    inferred: list[str] = []
    request = ChildHarnessRequest(
        agent_name="late",
        agent_description="Late child",
        trace_agent_name="subagent.late",
        task="late",
        inherited=False,
        tool_mode="explicit",
        system_prompt="late",
        model="openai:late",
    )

    class DetachedPlugin:
        name = "detached"

        def bind(self, context: PluginContext) -> PluginBinding:
            async def invoke(_args: Any) -> str:
                async def delayed() -> ChildHarnessOutcome:
                    await release.wait()
                    return await context.child_harnesses.run(request)

                detached.append(asyncio.create_task(delayed()))
                return "scheduled"

            tool = ToolSpec("delegate_later", "Delegate later", {"type": "object"}, invoke)
            registered = context.child_harnesses.register_delegation_tool(tool, [request])
            return PluginBinding(static=PluginContribution(tools=(registered,)))

    def infer(model_ref: str, **_kwargs: Any) -> ScriptedModel:
        inferred.append(model_ref)
        return ScriptedModel([])

    monkeypatch.setattr("thinharness.children.infer_model", infer)
    parent = ScriptedSession(
        start_turn=ModelTurn(
            tool_calls=[ModelToolCall(id="late_call", name="delegate_later", arguments="{}")],
            raw={},
        ),
        continue_turn=ModelTurn(text="done", raw={}),
    )
    harness = Harness(HarnessConfig(root=tmp_path), model=ScriptedModel([parent]), plugins=[DetachedPlugin()])

    assert (await harness.run("go")).text == "done"
    release.set()
    with pytest.raises(Exception, match="active parent tool call"):
        await detached[0]
    assert inferred == []


async def test_connected_registered_delegation_keeps_provenance_and_host_access(tmp_path: Path) -> None:
    tracer = FakeTracer()
    marker_at_hook: list[bool] = []

    class ConnectedDelegationPlugin:
        name = "connected-delegation"

        def bind(self, context: PluginContext) -> PluginBinding:
            request = ChildHarnessRequest(
                agent_name="connected",
                agent_description="Connected child",
                trace_agent_name="subagent.connected",
                task="",
                inherited=False,
                tool_mode="explicit",
                system_prompt="connected",
            )

            async def invoke(args: dict[str, Any]) -> str:
                outcome = await context.child_harnesses.run(replace(request, task=str(args["task"])))
                return outcome.content

            raw = ToolSpec(
                "connected_delegate",
                "Connected delegate",
                {"type": "object", "properties": {"task": {"type": "string"}}, "required": ["task"]},
                invoke,
            )
            registered = context.child_harnesses.register_delegation_tool(raw, [request])

            @asynccontextmanager
            async def connect():
                yield PluginContribution(tools=(registered,))

            return PluginBinding(connect=connect)

    parent = ScriptedSession(
        start_turn=ModelTurn(
            tool_calls=[ModelToolCall(id="connected_call", name="connected_delegate", arguments='{"task":"help"}')],
            raw={},
        ),
        continue_turn=ModelTurn(text="parent done", raw={}),
    )
    child = ScriptedSession(start_turn=ModelTurn(text="connected child done", raw={}))
    harness = Harness(
        HarnessConfig(root=tmp_path),
        model=ScriptedModel([parent, child]),
        plugins=[ConnectedDelegationPlugin()],
        hooks=[Hook(
            "before_tool_call",
            lambda _ctx: marker_at_hook.append(tracer.stack[-1].attributes.get("subagent.delegation") is True),
            tools=["connected_delegate"],
        )],
        tracing=[TracingOptions(tracer=tracer)],
    )

    assert (await harness.run("go")).text == "parent done"
    output = tool_output(parent.continue_calls[0][0][0].output)
    span = next(span for span in tracer.spans if span.name == "execute_tool connected_delegate")
    assert output["content"] == "connected child done"
    assert marker_at_hook == [True]
    assert span.attributes["subagent.delegation"] is True


@pytest.mark.parametrize(
    ("child_ref", "expected_key", "expected_base"),
    [
        ("openai:child", "parent-key", "https://parent.test"),
        ("anthropic:child", None, None),
    ],
)
async def test_child_override_projects_only_same_provider_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    child_ref: str,
    expected_key: str | None,
    expected_base: str | None,
) -> None:
    captured: dict[str, Any] = {}
    child_model = ScriptedModel([ScriptedSession(start_turn=ModelTurn(text="child", raw={}))])

    def infer(model_ref: str, **kwargs: Any) -> ScriptedModel:
        captured.update(model_ref=model_ref, **kwargs)
        return child_model

    monkeypatch.setattr("thinharness.children.infer_model", infer)
    parent = ScriptedSession(start_turn=_parent_call(agent="override"), continue_turn=ModelTurn(text="done", raw={}))
    harness = Harness(
        HarnessConfig(
            root=tmp_path,
            api_key="parent-key",
            base_url="https://parent.test",
            request_timeout=17,
            request_retries=2,
            temperature=0.4,
        ),
        model=ScriptedModel([parent]),
        plugins=[SubagentsPlugin(agents=[
            SubAgentConfig(name="override", description="Override.", model=child_ref)
        ])],
    )

    assert (await harness.run("go")).text == "done"
    assert captured["model_ref"] == child_ref
    assert captured["api_key"] == expected_key
    assert captured["base_url"] == expected_base
    assert captured["timeout"] == 17
    assert captured["request_retries"] == 2
    assert captured["temperature"] == 0.4


async def test_default_child_borrows_parent_model_without_closing_provider(tmp_path: Path) -> None:
    parent = ScriptedSession(start_turn=_parent_call(), continue_turn=ModelTurn(text="done", raw={}))
    model = ScriptedModel([parent, ScriptedSession(start_turn=ModelTurn(text="child", raw={}))])
    provider = ClosingProvider()
    model.provider = provider
    harness = Harness(HarnessConfig(root=tmp_path), model=model, plugins=[SubagentsPlugin()])

    assert (await harness.run("go")).text == "done"
    assert provider.closed == 0


async def test_concurrent_override_delegations_own_and_close_distinct_models(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    models = [
        ScriptedModel([ScriptedSession(start_turn=ModelTurn(text="one", raw={}))]),
        ScriptedModel([ScriptedSession(start_turn=ModelTurn(text="two", raw={}))]),
    ]
    providers = [ClosingProvider(), ClosingProvider()]
    for model, provider in zip(models, providers, strict=True):
        model.provider = provider
    created: list[ScriptedModel] = []

    def infer(*_args: Any, **_kwargs: Any) -> ScriptedModel:
        model = models[len(created)]
        created.append(model)
        return model

    monkeypatch.setattr("thinharness.children.infer_model", infer)
    parent = ScriptedSession(
        start_turn=ModelTurn(
            tool_calls=[
                ModelToolCall(id="one", name="subagent", arguments='{"task":"one","agent":"worker"}'),
                ModelToolCall(id="two", name="subagent", arguments='{"task":"two","agent":"worker"}'),
            ],
            raw={},
        ),
        continue_turn=ModelTurn(text="done", raw={}),
    )
    harness = Harness(
        HarnessConfig(root=tmp_path),
        model=ScriptedModel([parent]),
        plugins=[SubagentsPlugin(agents=[
            SubAgentConfig(name="worker", description="Worker.", model="openai:child")
        ])],
    )

    assert (await harness.run("go")).text == "done"
    assert len({id(model) for model in created}) == 2
    assert [provider.closed for provider in providers] == [1, 1]


async def test_strict_sibling_abort_does_not_hang_concurrent_delegation(tmp_path: Path) -> None:
    class BlockingSession:
        async def start(self, *_args: Any, **_kwargs: Any) -> ModelTurn:
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

        async def continue_with_tools(self, *_args: Any, **_kwargs: Any) -> ModelTurn:
            raise AssertionError("unreachable")

        async def continue_with_user_content(self, *_args: Any, **_kwargs: Any) -> ModelTurn:
            raise AssertionError("unreachable")

        def dump_state(self) -> None:
            return None

    def fail_sibling(ctx: Any) -> None:
        if ctx.tool_name == "fail":
            raise RuntimeError("strict sibling abort")

    parent = ScriptedSession(
        start_turn=ModelTurn(
            tool_calls=[
                ModelToolCall(id="delegate", name="subagent", arguments='{"task":"wait"}'),
                ModelToolCall(id="fail", name="fail", arguments="{}"),
            ],
            raw={},
        )
    )
    harness = Harness(
        HarnessConfig(root=tmp_path, strict_hooks=True),
        model=ScriptedModel([parent, BlockingSession()]),
        plugins=[SubagentsPlugin()],
        tools=[ToolSpec("fail", "Fail", {"type": "object"}, lambda _args: "unused")],
        hooks=[Hook("before_tool_call", fail_sibling)],
    )

    with pytest.raises(RuntimeError, match="strict sibling abort"):
        await asyncio.wait_for(harness.run("go"), timeout=1)


async def test_child_budgets_notices_and_tool_retry_fallback_are_fresh(tmp_path: Path) -> None:
    observed: list[tuple[str, int, int | None, int]] = []

    def record(label: str):
        def hook(ctx: Any) -> None:
            observed.append((label, ctx.max_model_requests, ctx.max_tool_calls, ctx.harness.config.tool_retries))

        return hook

    first_parent = ScriptedSession(start_turn=_parent_call(), continue_turn=ModelTurn(text="first", raw={}))
    first_child = ScriptedSession(
        start_turn=ModelTurn(
            tool_calls=[ModelToolCall(id="echo", name="echo", arguments='{"value":"ok"}')],
            raw={},
        ),
        continue_turn=ModelTurn(text="child first", raw={}),
    )
    second_parent = ScriptedSession(
        start_turn=_parent_call(agent="named", call_id="named_call"),
        continue_turn=ModelTurn(text="second", raw={}),
    )
    second_child = ScriptedSession(start_turn=ModelTurn(text="child second", raw={}))
    harness = Harness(
        HarnessConfig(root=tmp_path, max_model_requests=2, max_tool_calls=8, tool_retries=4),
        model=ScriptedModel([first_parent, first_child, second_parent, second_child]),
        plugins=[SubagentsPlugin(
            default_hooks=[Hook("run_start", record("default"))],
            agents=[SubAgentConfig(
                name="named",
                description="Named.",
                max_model_requests=3,
                max_tool_calls=2,
                hooks=[Hook("run_start", record("named"))],
            )],
        )],
        tools=[echo_tool()],
    )

    assert (await harness.run("first")).text == "first"
    assert (await harness.run("second")).text == "second"
    first_envelope = tool_output(first_parent.continue_calls[0][0][0].output)
    assert first_envelope["metadata"]["model_requests"] == 2
    assert [(notice.limit_kind, notice.remaining) for notice in first_child.notice_calls[1][1]] == [
        ("model_requests", 1)
    ]
    assert observed == [
        ("default", 2, 8, 4),
        ("named", 3, 2, 1),
    ]


async def test_inherited_parallel_model_resolution_follows_frozen_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import thinharness.plugins.parallel_llm as parallel_module

    real_tool = parallel_module.ParallelLlmTool
    captured: list[Any] = []

    def capture(**kwargs: Any):
        captured.append(kwargs["model"])
        return real_tool(**kwargs)

    monkeypatch.setattr(parallel_module, "ParallelLlmTool", capture)

    async def run_case(plugin: ParallelLlmPlugin) -> tuple[Any, Any, Any]:
        parent_model = ScriptedModel([
            ScriptedSession(start_turn=_parent_call(agent="worker"), continue_turn=ModelTurn(text="done", raw={}))
        ])
        child_model = ScriptedModel([ScriptedSession(start_turn=ModelTurn(text="child", raw={}))])
        monkeypatch.setattr("thinharness.children.infer_model", lambda *_args, **_kwargs: child_model)
        before = len(captured)
        harness = Harness(
            HarnessConfig(root=tmp_path),
            model=parent_model,
            plugins=[plugin, SubagentsPlugin(agents=[
                SubAgentConfig(name="worker", description="Worker.", inherit_parent=True, model="openai:child")
            ])],
        )
        await harness.run("go")
        return parent_model, child_model, tuple(captured[before:])

    borrowed_parent, borrowed_child, borrowed = await run_case(ParallelLlmPlugin())
    explicit_model = ScriptedModel([])
    _object_parent, _object_child, object_models = await run_case(ParallelLlmPlugin(explicit_model))
    _string_parent, _string_child, string_models = await run_case(ParallelLlmPlugin("openai:fixed"))

    assert borrowed[-1] is borrowed_child
    assert borrowed[:-1]
    assert all(model is borrowed_parent for model in borrowed[:-1])
    assert object_models
    assert all(model is explicit_model for model in object_models)
    assert string_models
    assert all(model == "openai:fixed" for model in string_models)


async def test_reused_subagents_plugin_keeps_parent_runs_fully_isolated(tmp_path: Path) -> None:
    plugin = SubagentsPlugin()
    hook_metadata: list[tuple[str, dict[str, Any]]] = []
    child_inputs: list[tuple[str, list[str], dict[str, Any]]] = []

    def parent(label: str, child_text: str, tool_name: str) -> Harness:
        parent_session = ScriptedSession(start_turn=_parent_call(), continue_turn=ModelTurn(text=f"{label} parent", raw={}))

        def child_start(_prompt: str, instructions: str, tools: list[dict[str, Any]], metadata: dict[str, Any], _previous: Any) -> None:
            child_inputs.append((instructions, [tool["name"] for tool in tools], dict(metadata)))

        child_session = ScriptedSession(
            start_turn=ModelTurn(text=child_text, raw={}),
            on_start=child_start,
        )
        return Harness(
            HarnessConfig(root=tmp_path / label),
            model=ScriptedModel([parent_session, child_session], model=f"{label}-model"),
            plugins=[FilesystemPlugin(tools=["read"]), plugin],
            tools=[ToolSpec(tool_name, tool_name, {"type": "object"}, lambda _args: label)],
            hooks=[Hook(
                "before_subagent_run",
                lambda ctx: hook_metadata.append((label, dict(ctx.metadata))),
                agents=["default"],
            )],
        )

    first = parent("first", "first child", "first_tool")
    second = parent("second", "second child", "second_tool")

    async def collect(harness: Harness, metadata: dict[str, Any]) -> tuple[str, list[Any]]:
        events: list[Any] = []
        stream = harness.stream("go", metadata=metadata, stream_options=StreamOptions(include_subagents=True))
        async with stream as values:
            async for event in values:
                events.append(event)
        result = next(event.result for event in events if event.kind == "run_completed" and event.parent_run_id is None)
        return result.text, events

    first_text, first_events = await collect(first, {"conversation_id": "first-conversation"})
    second_text, second_events = await collect(second, {"conversation_id": "second-conversation"})

    assert (first_text, second_text) == ("first parent", "second parent")
    assert [entry[1] for entry in child_inputs] == [["read", "first_tool"], ["read", "second_tool"]]
    assert str((tmp_path / "first").resolve()) in child_inputs[0][0]
    assert str((tmp_path / "second").resolve()) in child_inputs[1][0]
    assert child_inputs[0][2]["conversation_id"] == "first-conversation"
    assert child_inputs[1][2]["conversation_id"] == "second-conversation"
    assert hook_metadata == [
        ("first", {"conversation_id": "first-conversation"}),
        ("second", {"conversation_id": "second-conversation"}),
    ]
    assert [event.text for event in first_events if event.kind == "model_message" and event.agent_name == "default"] == ["first child"]
    assert [event.text for event in second_events if event.kind == "model_message" and event.agent_name == "default"] == ["second child"]


async def test_failed_child_run_with_cancelled_cleanup_propagates_cancellation(tmp_path: Path) -> None:
    class CancelOnClosePlugin:
        name = "cancel-on-close"

        def bind(self, _context: PluginContext) -> PluginBinding:
            @asynccontextmanager
            async def connect():
                try:
                    yield PluginContribution()
                finally:
                    raise asyncio.CancelledError

            return PluginBinding(connect=connect)

    parent = ScriptedSession(start_turn=_parent_call(agent="cancel"), continue_turn=ModelTurn(text="unused", raw={}))
    harness = Harness(
        HarnessConfig(root=tmp_path),
        model=ScriptedModel([parent, FailingSession()]),
        plugins=[SubagentsPlugin(agents=[
            SubAgentConfig(name="cancel", description="Cancel cleanup.", plugins=[CancelOnClosePlugin()])
        ])],
    )

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(harness.run("go"), timeout=1)


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
