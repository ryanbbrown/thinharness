"""Narrow child-harness execution contracts and core host implementation."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from types import TracebackType
from typing import TYPE_CHECKING, Any, Literal, Protocol

from .events import RunCompletedEvent, current_stream_emitter
from .hooks import (
    AfterSubagentRunContext,
    BeforeSubagentRunContext,
    Hook,
    HookRegistry,
    _ToolRuntimeScope,
    current_tool_call_context,
    current_tool_runtime_context,
)
from .providers import Model, infer_model, same_provider_model_ref
from .tools.base import Json, ToolSpec
from .types import HarnessError, HarnessResult

if TYPE_CHECKING:
    from .core import Harness
    from .plugins.base import Plugin


@dataclass(frozen=True)
class ChildHarnessRequest:
    """Opaque plugin-owned recipe for one fresh child harness run."""

    agent_name: str
    agent_description: str
    trace_agent_name: str
    task: str
    inherited: bool
    tool_mode: Literal["inherited", "inherited+explicit", "explicit"]
    system_prompt: str
    model: str | None = None
    plugins: tuple[Plugin, ...] = ()
    tools: tuple[ToolSpec, ...] = ()
    hooks: HookRegistry | tuple[Hook, ...] | None = None
    max_model_requests: int | None = None
    max_tool_calls: int | None = None
    output_type: Any | None = None
    output_mode: Literal["auto", "native", "tool", "prompted"] = "auto"
    output_retries: int = 1
    tool_retries: int | None = 1


@dataclass(frozen=True)
class ChildHarnessOutcome:
    """Result data returned by a child host to its delegation plugin."""

    result: HarnessResult | None
    tools: tuple[str, ...]
    content: str = ""
    structured_output: bool = False
    error: BaseException | None = None
    error_type: str | None = None
    error_message: str | None = None


class ChildHarnessHost(Protocol):
    """Narrow host capability available to trusted plugins."""

    def register_delegation_tool(
        self,
        tool: ToolSpec,
        recipes: Sequence[ChildHarnessRequest],
    ) -> ToolSpec:
        """Register authoritative delegation provenance and static child recipes."""
        ...

    async def run(self, request: ChildHarnessRequest) -> ChildHarnessOutcome:
        """Build, run, and close one child harness."""
        ...


@dataclass(frozen=True)
class _ToolComposition:
    """Core-owned provenance for one composed tool."""

    source: Literal["direct", "plugin"]
    plugin_index: int | None = None
    delegation: bool = False


class _DisabledChildHarnessHost:
    """Reject every child request before any resource can be created."""

    def register_delegation_tool(self, tool: ToolSpec, recipes: Sequence[ChildHarnessRequest]) -> ToolSpec:
        del tool, recipes
        raise HarnessError("child harnesses cannot create nested child harnesses")

    async def run(self, request: ChildHarnessRequest) -> ChildHarnessOutcome:
        del request
        raise HarnessError("child harnesses cannot create nested child harnesses")


_DISABLED_CHILD_HOST = _DisabledChildHarnessHost()


class _ParentChildHarnessHost:
    """Parent-holding core implementation of the narrow child host."""

    def __init__(self, parent: Harness) -> None:
        self._parent = parent
        self._delegation_tools: set[int] = set()
        self._recipes: list[ChildHarnessRequest] = []
        self._agent_names: list[str] = []
        self._sealed = False

    def register_delegation_tool(self, tool: ToolSpec, recipes: Sequence[ChildHarnessRequest]) -> ToolSpec:
        if self._sealed:
            raise HarnessError("child harness registration is sealed")
        if not isinstance(tool, ToolSpec):
            raise TypeError("delegation tool must be a ToolSpec")
        if not isinstance(recipes, Sequence) or isinstance(recipes, str | bytes):
            raise TypeError("child recipes must be an ordered sequence")
        registered = tuple(recipes)
        if any(not isinstance(recipe, ChildHarnessRequest) for recipe in registered):
            raise TypeError("child recipe must be a ChildHarnessRequest")
        names = tuple(recipe.agent_name for recipe in registered)
        if any(not isinstance(name, str) or not name.strip() for name in names):
            raise ValueError("child recipe agent name must be a non-empty string")

        self._delegation_tools.add(id(tool))
        self._recipes.extend(registered)
        for name in names:
            if name not in self._agent_names:
                self._agent_names.append(name)
        return tool

    def _seal_registration(self) -> None:
        """Prevent all later delegation registration."""
        self._sealed = True

    def _agent_catalog(self) -> tuple[str, ...]:
        """Return the ordered unique catalog from static registration."""
        return tuple(self._agent_names)

    def is_delegation_tool(self, tool: ToolSpec) -> bool:
        """Return whether a plugin registered this exact static tool."""
        return id(tool) in self._delegation_tools

    def validate_recipes(
        self,
        tools: Sequence[ToolSpec],
        compositions: Sequence[_ToolComposition],
    ) -> None:
        """Validate statically knowable child composition before a parent is usable."""
        for recipe in self._recipes:
            self._validate_recipe(recipe, tools, compositions)

    def _validate_recipe(
        self,
        recipe: ChildHarnessRequest,
        tools: Sequence[ToolSpec],
        compositions: Sequence[_ToolComposition],
    ) -> None:
        from .plugins.base import ChildInheritablePlugin, PluginBinding, PluginContext

        child_plugins: list[Plugin] = []
        if recipe.inherited:
            for plugin in self._parent.plugins:
                if not isinstance(plugin, ChildInheritablePlugin):
                    continue
                rebound = plugin.for_child()
                _validate_rebound_plugin(plugin, rebound)
                child_plugins.append(rebound)
        child_plugins.extend(recipe.plugins)
        _validate_plugin_names(child_plugins)

        child_tool_names: list[str] = []
        context = PluginContext(root=self._parent.root, model=self._parent.model, child_harnesses=_DISABLED_CHILD_HOST)
        for plugin in child_plugins:
            binding = plugin.bind(context)
            if not isinstance(binding, PluginBinding):
                raise TypeError(f"plugin {plugin.name!r} returned an invalid binding")
            for tool in binding.static.tools:
                if tool.requires_approval:
                    raise ValueError("approval-required tools are not supported inside child harnesses")
                child_tool_names.append(tool.name)

        if recipe.inherited:
            for tool, composition in zip(tools, compositions, strict=True):
                if composition.source == "direct" and not tool.requires_approval:
                    child_tool_names.append(tool.name)
        for tool in recipe.tools:
            if tool.requires_approval:
                raise ValueError("approval-required tools are not supported inside child harnesses")
            child_tool_names.append(tool.name)
        duplicate = next((name for index, name in enumerate(child_tool_names) if name in child_tool_names[:index]), None)
        if duplicate is not None:
            raise ValueError(f"duplicate tool name: {duplicate}")

    async def run(self, request: ChildHarnessRequest) -> ChildHarnessOutcome:
        """Run one child using the active parent's frozen tool composition."""
        runtime = current_tool_runtime_context()
        tool_call = current_tool_call_context()
        if runtime is None or tool_call is None:
            raise HarnessError("child harness request requires an active parent tool call")
        if not runtime.lease.active:
            raise HarnessError("child harness request requires an active parent tool call")
        active_name = str(tool_call.get("name", ""))
        active_composition = runtime.tool_composition.get(active_name)
        if not isinstance(active_composition, _ToolComposition) or not active_composition.delegation:
            raise HarnessError("child harness request requires a registered delegation tool")
        if request.agent_name not in self._agent_names:
            raise HarnessError(f"child harness request uses an unregistered agent name: {request.agent_name}")

        parent_metadata = _parent_run_metadata(runtime)
        parent_call_id = str(tool_call["call_id"]) if tool_call.get("call_id") else None
        before = BeforeSubagentRunContext(
            harness=self._parent,
            metadata=dict(parent_metadata),
            agent=request.agent_name,
            task=request.task,
            inherited=request.inherited,
            tool_mode=request.tool_mode,
            parent_harness=self._parent,
            parent_call_id=parent_call_id,
        )
        self._parent.hooks.fire(before)
        if before.cancelled:
            reason = before.cancel_reason or "unspecified"
            return ChildHarnessOutcome(
                result=None,
                tools=(),
                error_type="SubAgentCancelled",
                error_message=f"Subagent execution blocked by hook: {reason}",
            )

        effective_tools: tuple[str, ...] = ()
        try:
            child_model, owns_model = self._resolve_child_model(request)
            try:
                child = self._build_child(
                    request,
                    runtime.tool_map,
                    runtime.tool_composition,
                    child_model=child_model,
                    owns_model=owns_model,
                )
            except BaseException as build_error:
                if owns_model:
                    await _close_model_after_failure(child_model, build_error)
                raise
            result: HarnessResult | None = None
            run_error: BaseException | None = None
            run_traceback: TracebackType | None = None
            try:
                await child.connect()
                effective_tools = tuple(tool.name for tool in child.tools)
                emitter = current_stream_emitter()
                child_metadata = _child_metadata(parent_metadata, parent_call_id)
                if emitter is not None and emitter.ctx.options.include_subagents:
                    child_stream = child.stream(
                        request.task,
                        metadata=child_metadata,
                        stream_options=emitter.ctx.options,
                        _parent_run_id=emitter.ctx.run_id,
                        _parent_tool_call_id=parent_call_id,
                        _agent_name=request.agent_name,
                    )
                    try:
                        async for event in child_stream:
                            emitter.emit_forwarded(event)
                            if isinstance(event, RunCompletedEvent) and event.run_id == child_stream.run_id:
                                result = event.result
                    finally:
                        await child_stream.aclose()
                else:
                    result = await child.run(request.task, metadata=child_metadata)
            except BaseException as exc:
                run_error = exc
                run_traceback = exc.__traceback__
            try:
                await child.aclose()
            except asyncio.CancelledError:
                raise
            except BaseException as close_error:
                if run_error is None:
                    raise
                run_error.add_note(f"cleanup also failed: {type(close_error).__name__}: {close_error}")
            if run_error is not None:
                raise run_error.with_traceback(run_traceback)
        except Exception as exc:
            self._parent.hooks.fire(
                AfterSubagentRunContext(
                    harness=self._parent,
                    metadata=dict(parent_metadata),
                    agent=request.agent_name,
                    task=request.task,
                    error=exc,
                    tools=list(effective_tools),
                    parent_call_id=parent_call_id,
                )
            )
            return ChildHarnessOutcome(
                result=None,
                tools=effective_tools,
                error=exc,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )

        assert result is not None
        self._parent.hooks.fire(
            AfterSubagentRunContext(
                harness=self._parent,
                metadata=dict(parent_metadata),
                agent=request.agent_name,
                task=request.task,
                result=result,
                tools=list(effective_tools),
                usage=result.usage,
                parent_call_id=parent_call_id,
            )
        )
        structured_output = result.output is not None
        content = child.output_schema.dump(result.output) if structured_output and child.output_schema is not None else result.text
        return ChildHarnessOutcome(
            result=result,
            tools=effective_tools,
            content=content,
            structured_output=structured_output,
        )

    def _build_child(
        self,
        request: ChildHarnessRequest,
        frozen_tool_map: dict[str, ToolSpec],
        frozen_composition: dict[str, _ToolComposition],
        *,
        child_model: Model,
        owns_model: bool,
    ) -> Harness:
        from .core import Harness
        from .plugins.base import ChildInheritablePlugin

        parent_config = self._parent.config
        child_plugins: list[Plugin] = []
        if request.inherited:
            for plugin in self._parent.plugins:
                if isinstance(plugin, ChildInheritablePlugin):
                    rebound = plugin.for_child()
                    _validate_rebound_plugin(plugin, rebound)
                    child_plugins.append(rebound)
        child_plugins.extend(request.plugins)
        _validate_plugin_names(child_plugins)

        child_tools: list[ToolSpec] = []
        if request.inherited:
            for name, tool in frozen_tool_map.items():
                composition = frozen_composition.get(name)
                if composition is not None and composition.source == "direct" and not tool.requires_approval:
                    child_tools.append(tool)
        child_tools.extend(request.tools)

        child_config = parent_config.model_copy(
            update={
                "model": request.model if request.model is not None else parent_config.model,
                "root": self._parent.root,
                "system_prompt": request.system_prompt,
                "max_model_requests": request.max_model_requests if request.max_model_requests is not None else parent_config.max_model_requests,
                "max_tool_calls": request.max_tool_calls if request.max_tool_calls is not None else parent_config.max_tool_calls,
                "output_type": request.output_type,
                "output_mode": request.output_mode,
                "output_retries": request.output_retries,
                "tool_retries": request.tool_retries if request.tool_retries is not None else parent_config.tool_retries,
            }
        )
        hooks: HookRegistry | list[Hook] | None
        if isinstance(request.hooks, HookRegistry):
            hooks = HookRegistry(list(request.hooks.hooks), strict_hooks=request.hooks.strict_hooks)
        else:
            hooks = list(request.hooks) if request.hooks is not None else None
        tracing = [
            option.model_copy(
                update={
                    "agent_name": request.trace_agent_name,
                    "agent_description": request.agent_description,
                }
            )
            for option in self._parent.tracing
        ]
        return Harness(
            child_config,
            model=child_model,
            plugins=child_plugins,
            tools=child_tools,
            tracing=tracing,
            hooks=hooks,
            _owns_model=owns_model,
            _is_child_harness=True,
            _child_harnesses=_DISABLED_CHILD_HOST,
        )

    def _resolve_child_model(self, request: ChildHarnessRequest) -> tuple[Model, bool]:
        """Borrow the parent model or infer one owned override model."""
        if request.model is None:
            return self._parent.model, False
        parent_config = self._parent.config
        same_provider = same_provider_model_ref(self._parent.model, request.model)
        return infer_model(
            request.model,
            api_key=parent_config.api_key if same_provider else None,
            base_url=parent_config.base_url if same_provider else None,
            timeout=parent_config.request_timeout,
            request_retries=parent_config.request_retries,
            request_retry_backoff=parent_config.request_retry_backoff,
            temperature=parent_config.temperature,
            max_tokens=parent_config.max_tokens,
            effort=parent_config.effort,
            extra_body=parent_config.extra_body,
        ), True


async def _close_model_after_failure(model: Model, original_error: BaseException) -> None:
    """Close an inferred model after child construction fails without hiding that failure."""
    aclose = getattr(model.provider, "aclose", None)
    if aclose is None:
        return
    try:
        await aclose()
    except asyncio.CancelledError:
        raise
    except BaseException as close_error:
        original_error.add_note(f"cleanup also failed: {type(close_error).__name__}: {close_error}")


def _validate_rebound_plugin(parent_plugin: Plugin, rebound: object) -> None:
    """Validate one structural child-inheritance result."""
    from .plugins.base import Plugin

    if not isinstance(rebound, Plugin):
        raise TypeError(f"plugin {parent_plugin.name!r} for_child() returned an invalid plugin")
    if rebound.name != parent_plugin.name:
        raise ValueError(
            f"plugin {parent_plugin.name!r} for_child() changed its fixed name to {rebound.name!r}"
        )


def _validate_plugin_names(plugins: Sequence[Plugin]) -> None:
    """Validate one child plugin list with normal fixed-name rules."""
    names = [plugin.name for plugin in plugins]
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("plugin name must be a non-empty string")
    duplicate = next((name for index, name in enumerate(names) if name in names[:index]), None)
    if duplicate is not None:
        raise ValueError(f"duplicate plugin name: {duplicate}")


def _parent_run_metadata(runtime: _ToolRuntimeScope) -> Json:
    """Copy parent metadata from the active runtime."""
    return dict(runtime.run_metadata)


def _child_metadata(parent_metadata: Json, parent_call_id: str | None) -> Json:
    """Project correlation metadata into one child run."""
    metadata: Json = {}
    if conversation_id := parent_metadata.get("conversation_id"):
        metadata["conversation_id"] = conversation_id
    if parent_call_id is not None:
        metadata["parent_call_id"] = parent_call_id
    return metadata


__all__ = ["ChildHarnessHost", "ChildHarnessOutcome", "ChildHarnessRequest"]
