"""Subagent configuration and delegation tool support."""

from __future__ import annotations

from types import TracebackType
from typing import TYPE_CHECKING, Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .defaults import DEFAULT_SYSTEM_PROMPT
from .hooks import AfterSubagentRunContext, BeforeSubagentRunContext, HookRegistry, current_tool_call_context
from .providers import infer_model, parse_model_ref, provider_prefix
from .tools.base import Json, ToolResult, ToolSpec
from .tools.mcp import MCPServer
from .tracing import TracingOptions

if TYPE_CHECKING:
    from .core import Harness


DEFAULT_SUBAGENT_NAME: Final[str] = "default"


class SubAgentConfig(BaseModel):
    """Configuration for one delegated child harness."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.-]+$")
    description: str = Field(min_length=1)
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    inherit_parent_tools: bool = False
    inherit_mcp_servers: bool = False
    builtin_tools: list[str] = Field(default_factory=list)
    tools: list[ToolSpec] = Field(default_factory=list)
    mcp_servers: list[MCPServer] = Field(default_factory=list)
    model: str | None = None
    max_model_requests: int | None = None
    max_tool_calls: int | None = None
    output_type: Any | None = None
    output_mode: Literal["auto", "native", "tool", "prompted"] = "auto"
    output_retries: int = Field(default=1, ge=0)
    tool_retries: int = Field(default=1, ge=0)

    @model_validator(mode="after")
    def validate_subagent(self) -> SubAgentConfig:
        """Validate subagent tool policy and display fields."""
        if self.name == DEFAULT_SUBAGENT_NAME:
            raise ValueError(f"{DEFAULT_SUBAGENT_NAME!r} is reserved for the framework default subagent")
        if not self.description.strip() or "\n" in self.description or "\r" in self.description:
            raise ValueError("subagent description must be a non-empty single line")
        exposes_subagent = any(name.lower() == "subagent" for name in self.builtin_tools) or any(
            _tool_name(tool).lower() == "subagent" for tool in self.tools
        )
        if exposes_subagent:
            raise ValueError("subagent cannot be exposed inside a child subagent")
        if self.inherit_parent_tools and (self.builtin_tools or self.tools):
            raise ValueError("inherit_parent_tools cannot be combined with builtin_tools or tools")
        if not (
            self.inherit_parent_tools
            or self.builtin_tools
            or self.tools
            or self.inherit_mcp_servers
            or self.mcp_servers
        ):
            raise ValueError("named subagents must define builtin_tools, tools, inherit_parent_tools=True, inherit_mcp_servers=True, or mcp_servers")
        return self


class SubAgentArgs(BaseModel):
    """Arguments for subagent delegation."""

    model_config = ConfigDict(extra="forbid")

    task: str
    agent: str | None = Field(default=None, min_length=1, description="Optional subagent name; omit to use the framework default subagent.")


def create_subagent_tool(parent: Harness, configs: list[SubAgentConfig]) -> ToolSpec:
    """Create the parent-facing subagent delegation tool."""
    async def handler(args: SubAgentArgs) -> ToolResult:
        """Run the selected subagent."""
        return await run_subagent_tool(parent, configs, args)

    return ToolSpec(
        "subagent",
        _subagent_tool_description(configs),
        SubAgentArgs,
        handler,
        metadata={"framework_tool": "subagent"},
    )


async def run_subagent_tool(parent: Harness, configs: list[SubAgentConfig], args: SubAgentArgs) -> ToolResult:
    """Run a child harness and return its final text as a tool result."""
    config = _select_config(configs, args.agent)
    if args.agent and config is None:
        available = sorted(cfg.name for cfg in configs)
        return ToolResult(
            False,
            f"unknown subagent: {args.agent}",
            {"agent": args.agent, "available": available, "error_type": "UnknownSubAgent"},
        )
    agent_name = config.name if config is not None else DEFAULT_SUBAGENT_NAME
    inherited = config is None or config.inherit_parent_tools
    tool_mode = "inherited" if inherited else "explicit"
    effective_tools: list[str] = []
    parent_call_id = _parent_call_id()
    before = BeforeSubagentRunContext(
        harness=parent,
        metadata=dict(getattr(parent, "_current_run_metadata", None) or {}),
        agent=agent_name,
        task=args.task,
        inherited=inherited,
        tool_mode=tool_mode,
        parent_harness=parent,
        parent_call_id=parent_call_id,
    )
    parent.hooks.fire(before)
    if before.cancelled:
        reason = before.cancel_reason or "unspecified"
        return ToolResult(
            False,
            f"Subagent execution blocked by hook: {reason}",
            {
                "agent": agent_name,
                "inherited": inherited,
                "tool_mode": tool_mode,
                "tools": effective_tools,
                "error_type": "SubAgentCancelled",
            },
        )
    try:
        child = build_child_harness(parent, config)
        await child.connect()
        effective_tools = [tool.name for tool in child.tools]
        result = None
        run_error: BaseException | None = None
        run_traceback: TracebackType | None = None
        try:
            result = await child.run(args.task, metadata=_child_metadata(parent))
        except BaseException as exc:
            # Preserve cancellation and other BaseException exits while still closing the child harness below.
            run_error = exc
            run_traceback = exc.__traceback__
        try:
            await child.aclose()
        except Exception:
            if run_error is None:
                raise
        if run_error is not None:
            raise run_error.with_traceback(run_traceback)
    except Exception as exc:
        parent.hooks.fire(AfterSubagentRunContext(
            harness=parent,
            metadata=dict(getattr(parent, "_current_run_metadata", None) or {}),
            agent=agent_name,
            task=args.task,
            error=exc,
            tools=effective_tools,
            parent_call_id=parent_call_id,
        ))
        return ToolResult(
            False,
            str(exc),
            {
                "agent": agent_name,
                "inherited": inherited,
                "tool_mode": tool_mode,
                "tools": effective_tools,
                "error_type": type(exc).__name__,
            },
        )
    assert result is not None
    parent.hooks.fire(AfterSubagentRunContext(
        harness=parent,
        metadata=dict(getattr(parent, "_current_run_metadata", None) or {}),
        agent=agent_name,
        task=args.task,
        result=result,
        tools=effective_tools,
        usage=result.usage,
        parent_call_id=parent_call_id,
    ))
    structured_output = result.output is not None
    content = child.output_schema.dump(result.output) if structured_output and child.output_schema is not None else result.text
    return ToolResult(
        True,
        content,
        {
            "agent": agent_name,
            "inherited": inherited,
            "tool_mode": tool_mode,
            "tools": effective_tools,
            "model_requests": result.usage.model_requests,
            "structured_output": structured_output,
        },
    )


def build_child_harness(parent: Harness, config: SubAgentConfig | None) -> Harness:
    """Create an isolated child harness for one subagent invocation."""
    from .core import Harness

    parent_config = parent.config
    inherit_tools = config is None or config.inherit_parent_tools
    child_wants_skills = bool(config and any(name.lower() in {"skill_read", "skill_run"} for name in config.builtin_tools))
    if inherit_tools:
        child_builtin_tools: list[str] = []
    else:
        assert config is not None
        child_builtin_tools = config.builtin_tools
    child_mcp_servers: list[MCPServer] = []
    if config is not None and config.inherit_mcp_servers:
        child_mcp_servers.extend(parent._mcp_servers)
    if config is not None:
        for server in config.mcp_servers:
            if not any(server is existing for existing in child_mcp_servers):
                child_mcp_servers.append(server)
    child_config = parent_config.model_copy(update={
        "model": config.model if config is not None and config.model is not None else parent_config.model,
        "root": parent.root,
        "system_prompt": DEFAULT_SYSTEM_PROMPT if config is None else config.system_prompt,
        "builtin_tools": child_builtin_tools,
        "skills_dir": parent_config.skills_dir if child_wants_skills and not inherit_tools else None,
        "selected_skills": parent_config.selected_skills if child_wants_skills and not inherit_tools else None,
        "max_model_requests": (
            config.max_model_requests
            if config is not None and config.max_model_requests is not None
            else parent_config.max_model_requests
        ),
        "max_tool_calls": (
            config.max_tool_calls
            if config is not None and config.max_tool_calls is not None
            else parent_config.max_tool_calls
        ),
        "output_type": config.output_type if config is not None else None,
        "output_mode": config.output_mode if config is not None else "auto",
        "output_retries": config.output_retries if config is not None else 1,
        "tool_retries": config.tool_retries if config is not None else parent_config.tool_retries,
        "subagents": [],
        "mcp_servers": child_mcp_servers,
    })
    child_model = parent.model
    if config is not None and config.model is not None:
        same_provider = _same_provider(parent, config.model)
        child_model = infer_model(
            config.model,
            api_key=parent_config.api_key if same_provider else None,
            base_url=parent_config.base_url if same_provider else None,
            timeout=parent_config.request_timeout,
            temperature=parent_config.temperature,
            extra_body=parent_config.extra_body,
        )
    return Harness(
        child_config,
        model=child_model,
        tools=_effective_custom_tools(parent, config),
        tracing=_child_tracing(parent, config),
        skills=parent.skills if inherit_tools else None,
        hooks=_child_hooks(parent, config),
        subagent_hooks={},
        _owns_model=config is not None and config.model is not None,
        _is_child_run=True,
    )


def _select_config(configs: list[SubAgentConfig], agent: str | None) -> SubAgentConfig | None:
    """Return the named subagent config, or None for the default route."""
    if agent is None:
        return None
    for config in configs:
        if config.name == agent:
            return config
    return None


def _effective_custom_tools(parent: Harness, config: SubAgentConfig | None) -> list[ToolSpec]:
    """Return custom tools to register on the child harness."""
    if config is None or config.inherit_parent_tools:
        return [tool for tool in parent.tools if tool.name != "subagent" and tool.metadata.get("source") != "mcp"]
    return list(config.tools)


def _child_tracing(parent: Harness, config: SubAgentConfig | None) -> TracingOptions | None:
    """Return child tracing options that share the parent's tracer."""
    if parent.tracing is None:
        return None
    name = config.name if config is not None else DEFAULT_SUBAGENT_NAME
    return parent.tracing.model_copy(update={
        "agent_name": f"subagent.{name}",
        "agent_description": config.description if config is not None else "Framework default subagent",
    })


def _child_metadata(parent: Harness) -> Json:
    """Build minimal metadata for a child run."""
    metadata: Json = {}
    parent_metadata = getattr(parent, "_current_run_metadata", None) or {}
    if conversation_id := parent_metadata.get("conversation_id"):
        metadata["conversation_id"] = conversation_id
    if tool_call := current_tool_call_context():
        metadata["parent_call_id"] = tool_call["call_id"]
    return metadata


def _parent_call_id() -> str | None:
    """Return the current parent tool call id when running as a tool."""
    tool_call = current_tool_call_context()
    return str(tool_call["call_id"]) if tool_call else None


def _child_hooks(parent: Harness, config: SubAgentConfig | None) -> HookRegistry | list | None:
    """Return the explicitly configured child hook registry."""
    return parent.subagent_hooks.get(config.name if config is not None else DEFAULT_SUBAGENT_NAME)


def _same_provider(parent: Harness, child_model_ref: str) -> bool:
    """Return whether a child model ref uses the same provider as the parent model."""
    child_provider, _ = parse_model_ref(child_model_ref)
    parent_provider = provider_prefix(getattr(getattr(parent.model, "provider", None), "name", ""))
    return child_provider == parent_provider


def _subagent_tool_description(configs: list[SubAgentConfig]) -> str:
    """Render the parent-facing subagent tool description."""
    lines = [
        "Delegate one self-contained task to a sub-helper. Each subagent runs in isolated context.",
        "",
    ]
    if configs:
        lines.append("Available agents:")
        lines.extend(f"- {config.name}: {config.description}" for config in configs)
        lines.append("")
    lines.append("Omit `agent` to use the framework default subagent.")
    return "\n".join(lines)


def _tool_name(tool: ToolSpec) -> str:
    """Return a tool name from a ToolSpec."""
    return tool.name
