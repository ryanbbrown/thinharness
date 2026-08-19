"""Explicit subagent delegation plugin and child configuration."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..children import ChildHarnessOutcome, ChildHarnessRequest
from ..defaults import DEFAULT_SYSTEM_PROMPT
from ..hooks import AGENT_EVENTS, Hook, HookRegistry
from ..tools.base import ToolResult, ToolSpec
from .base import Plugin, PluginBinding, PluginContext, PluginContribution

DEFAULT_SUBAGENT_NAME: Final[str] = "default"
_REMOVED_CONFIG_FIELDS = (
    "inherit_parent_tools",
    "inherit_mcp_servers",
    "mcp_servers",
    "builtin_tools",
)


class SubAgentConfig(BaseModel):
    """Configuration for one named delegated child harness."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)

    name: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.-]+$")
    description: str = Field(min_length=1)
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    inherit_parent: bool = False
    plugins: tuple[Any, ...] = ()
    tools: tuple[ToolSpec, ...] = ()
    hooks: Any = None
    model: str | None = None
    max_model_requests: int | None = None
    max_tool_calls: int | None = None
    output_type: Any | None = None
    output_mode: Literal["auto", "native", "tool", "prompted"] = "auto"
    output_retries: int = Field(default=1, ge=0)
    tool_retries: int = Field(default=1, ge=0)

    @model_validator(mode="before")
    @classmethod
    def reject_removed_fields(cls, data: object) -> object:
        """Fail loudly when callers pass fields removed from this configuration."""
        if isinstance(data, dict):
            for field_name in _REMOVED_CONFIG_FIELDS:
                if field_name in data:
                    raise ValueError(
                        f"SubAgentConfig.{field_name} has been removed; use inherit_parent, plugins, or tools"
                    )
            if "background" in data:
                raise ValueError("SubAgentConfig.background has been removed")
            for field_name in ("plugins", "tools"):
                if isinstance(data.get(field_name), (set, frozenset)):
                    raise TypeError(f"SubAgentConfig {field_name} must be an ordered sequence, not a set")
        return data

    @model_validator(mode="after")
    def validate_child(self) -> SubAgentConfig:
        """Validate display, plugin, tool, and child-hook policy."""
        if self.name == DEFAULT_SUBAGENT_NAME:
            raise ValueError(f"{DEFAULT_SUBAGENT_NAME!r} is reserved for the framework default subagent")
        if not self.description.strip() or "\n" in self.description or "\r" in self.description:
            raise ValueError("subagent description must be a non-empty single line")
        _validate_plugins(self.plugins)
        if any(isinstance(plugin, SubagentsPlugin) for plugin in self.plugins):
            raise ValueError("SubagentsPlugin cannot be configured inside a child harness")
        if any(tool.requires_approval for tool in self.tools):
            raise ValueError("approval-required tools are not supported inside child harnesses")
        return self


class SubAgentArgs(BaseModel):
    """Arguments for subagent delegation."""

    model_config = ConfigDict(extra="forbid")

    task: str
    agent: str | None = Field(
        default=None,
        min_length=1,
        description="Optional subagent name; omit to use the framework default subagent.",
    )


class _SubagentsPluginMeta(type):
    """Keep the subagents plugin name fixed on the class hierarchy."""

    def __setattr__(cls, attribute: str, value: object) -> None:
        if attribute == "name":
            raise AttributeError("SubagentsPlugin.name is fixed to 'subagents'")
        super().__setattr__(attribute, value)

    def __delattr__(cls, attribute: str) -> None:
        if attribute == "name":
            raise AttributeError("SubagentsPlugin.name is fixed to 'subagents'")
        super().__delattr__(attribute)


class SubagentsPlugin(metaclass=_SubagentsPluginMeta):
    """Contribute one delegation tool backed by isolated child harnesses."""

    name = "subagents"
    _agents: tuple[SubAgentConfig, ...]
    _default_hooks: tuple[Hook, ...] | HookRegistry | None
    _agent_hooks: tuple[tuple[Hook, ...] | HookRegistry | None, ...]
    _frozen: bool

    def __init_subclass__(cls) -> None:
        """Reject subclasses that replace the fixed plugin name."""
        super().__init_subclass__()
        if "name" in cls.__dict__:
            raise TypeError("SubagentsPlugin subclasses cannot override the fixed name 'subagents'")

    def __setattr__(self, attribute: str, value: object) -> None:
        """Reject configuration changes after construction."""
        if attribute == "name":
            raise AttributeError("SubagentsPlugin.name is fixed to 'subagents'")
        if getattr(self, "_frozen", False):
            raise AttributeError("SubagentsPlugin configuration is frozen")
        object.__setattr__(self, attribute, value)

    def __delattr__(self, attribute: str) -> None:
        """Reject configuration deletion after construction."""
        if attribute == "name":
            raise AttributeError("SubagentsPlugin.name is fixed to 'subagents'")
        if getattr(self, "_frozen", False):
            raise AttributeError("SubagentsPlugin configuration is frozen")
        object.__delattr__(self, attribute)

    def __init__(
        self,
        *,
        agents: Sequence[SubAgentConfig] = (),
        default_hooks: Sequence[Hook] | HookRegistry | None = None,
    ) -> None:
        if isinstance(agents, (set, frozenset)):
            raise TypeError("SubagentsPlugin agents must be an ordered sequence, not a set")
        configured = tuple(agents)
        if any(not isinstance(agent, SubAgentConfig) for agent in configured):
            raise TypeError("SubagentsPlugin agents must contain SubAgentConfig values")
        names = [agent.name for agent in configured]
        duplicate = next((name for index, name in enumerate(names) if name in names[:index]), None)
        if duplicate is not None:
            raise ValueError(f"duplicate subagent name: {duplicate}")
        normalized_default_hooks = _normalize_hooks(default_hooks, label="SubagentsPlugin.default_hooks")
        normalized_agent_hooks = tuple(
            _normalize_hooks(agent.hooks, label=f"SubAgentConfig({agent.name!r}).hooks")
            for agent in configured
        )
        object.__setattr__(self, "_agents", configured)
        object.__setattr__(self, "_default_hooks", normalized_default_hooks)
        object.__setattr__(self, "_agent_hooks", normalized_agent_hooks)
        object.__setattr__(self, "_frozen", True)

    @property
    def agents(self) -> tuple[SubAgentConfig, ...]:
        """Return the ordered frozen named-child catalog."""
        return self._agents

    @property
    def default_hooks(self) -> tuple[Hook, ...] | HookRegistry | None:
        """Return a copy of default-child hook configuration."""
        hooks = self._default_hooks
        if isinstance(hooks, HookRegistry):
            return HookRegistry(list(hooks.hooks), strict_hooks=hooks.strict_hooks)
        return hooks

    def bind(self, context: PluginContext) -> PluginBinding:
        """Bind one static delegation tool without creating child resources."""
        host = context.child_harnesses
        agents = self._agents
        recipes = (
            self._default_recipe(),
            *(self._recipe(config, hooks) for config, hooks in zip(agents, self._agent_hooks, strict=True)),
        )
        recipes_by_name = {
            agent.name: recipe
            for agent, recipe in zip(agents, recipes[1:], strict=True)
        }
        available_names = tuple(sorted(recipes_by_name))

        async def handler(args: SubAgentArgs) -> ToolResult:
            """Run one selected child and shape its model-visible result."""
            if args.agent is None:
                recipe = recipes[0]
            else:
                recipe = recipes_by_name.get(args.agent)
                if recipe is None:
                    return ToolResult(
                        False,
                        f"unknown subagent: {args.agent}",
                        {
                            "agent": args.agent,
                            "available": list(available_names),
                            "error_type": "UnknownSubAgent",
                        },
                    )
            outcome = await host.run(replace(recipe, task=args.task))
            return _tool_result(recipe, outcome)

        tool = ToolSpec(
            "subagent",
            _tool_description(agents),
            SubAgentArgs,
            handler,
        )
        registered = host.register_delegation_tool(tool, recipes)
        return PluginBinding(
            static=PluginContribution(tools=(registered,)),
            agent_names=(DEFAULT_SUBAGENT_NAME, *(agent.name for agent in self._agents)),
        )

    def _default_recipe(self) -> ChildHarnessRequest:
        """Return the fixed parent-derived unnamed-child recipe."""
        return ChildHarnessRequest(
            agent_name=DEFAULT_SUBAGENT_NAME,
            agent_description="Framework default subagent",
            trace_agent_name=f"subagent.{DEFAULT_SUBAGENT_NAME}",
            task="",
            inherited=True,
            tool_mode="inherited",
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            hooks=self._default_hooks,
            tool_retries=None,
        )

    @staticmethod
    def _recipe(
        config: SubAgentConfig,
        hooks: tuple[Hook, ...] | HookRegistry | None,
    ) -> ChildHarnessRequest:
        """Translate one plugin-owned named configuration into a host request."""
        explicit = bool(config.plugins or config.tools)
        tool_mode: Literal["inherited", "inherited+explicit", "explicit"]
        if config.inherit_parent:
            tool_mode = "inherited+explicit" if explicit else "inherited"
        else:
            tool_mode = "explicit"
        return ChildHarnessRequest(
            agent_name=config.name,
            agent_description=config.description,
            trace_agent_name=f"subagent.{config.name}",
            task="",
            inherited=config.inherit_parent,
            tool_mode=tool_mode,
            system_prompt=config.system_prompt,
            model=config.model,
            plugins=tuple(config.plugins),
            tools=tuple(config.tools),
            hooks=hooks,
            max_model_requests=config.max_model_requests,
            max_tool_calls=config.max_tool_calls,
            output_type=config.output_type,
            output_mode=config.output_mode,
            output_retries=config.output_retries,
            tool_retries=config.tool_retries,
        )


def _tool_result(recipe: ChildHarnessRequest, outcome: ChildHarnessOutcome) -> ToolResult:
    """Shape a child-host outcome as the delegation tool contract."""
    metadata = {
        "agent": recipe.agent_name,
        "inherited": recipe.inherited,
        "tool_mode": recipe.tool_mode,
        "tools": list(outcome.tools),
    }
    if outcome.error_type is not None:
        metadata["error_type"] = outcome.error_type
        return ToolResult(False, outcome.error_message or outcome.error_type, metadata)
    assert outcome.result is not None
    metadata.update({
        "model_requests": outcome.result.usage.model_requests,
        "structured_output": outcome.structured_output,
    })
    return ToolResult(True, outcome.content, metadata)


def _validate_plugins(plugins: Sequence[object]) -> None:
    """Reject invalid plugin values through the structural public contract."""
    for plugin in plugins:
        if not isinstance(plugin, Plugin):
            raise TypeError("SubAgentConfig plugins must contain Plugin values")


def _normalize_hooks(
    hooks: Sequence[Hook] | HookRegistry | None,
    *,
    label: str,
) -> tuple[Hook, ...] | HookRegistry | None:
    """Copy and validate hooks that will run inside a non-delegating child."""
    if hooks is None:
        return None
    if isinstance(hooks, HookRegistry):
        normalized: tuple[Hook, ...] | HookRegistry = HookRegistry(
            list(hooks.hooks),
            strict_hooks=hooks.strict_hooks,
        )
        values = normalized.hooks
    else:
        if isinstance(hooks, (set, frozenset)):
            raise TypeError(f"{label} must be an ordered sequence, not a set")
        values = list(hooks)
        if any(not isinstance(hook, Hook) for hook in values):
            raise TypeError(f"{label} must contain Hook values")
        normalized = tuple(values)
    for hook in values:
        if hook.event in AGENT_EVENTS:
            raise ValueError(f"{label} cannot contain subagent lifecycle hooks")
        if hook.agents is not None:
            raise ValueError(f"{label} cannot use Hook.agents")
    return normalized


def _tool_description(agents: Sequence[SubAgentConfig]) -> str:
    """Render the model-facing delegation tool description."""
    lines = [
        "Delegate one self-contained task to a sub-helper. Each subagent runs in isolated context.",
        "",
    ]
    if agents:
        lines.append("Available agents:")
        lines.extend(f"- {agent.name}: {agent.description}" for agent in agents)
        lines.append("")
    lines.append("Omit `agent` to use the framework default subagent.")
    return "\n".join(lines)


__all__ = [
    "DEFAULT_SUBAGENT_NAME",
    "SubAgentArgs",
    "SubAgentConfig",
    "SubagentsPlugin",
]
