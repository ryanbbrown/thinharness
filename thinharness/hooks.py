"""Lifecycle hook types and dispatch helpers."""

from __future__ import annotations

import contextvars
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal

from .tools.base import Json, ToolSpec

_CURRENT_TOOL_CALL: contextvars.ContextVar[Json | None] = contextvars.ContextVar("thinharness_current_tool_call", default=None)


def current_tool_call_context() -> Json | None:
    """Return the current tool call context for nested tool handlers."""
    return _CURRENT_TOOL_CALL.get()

if TYPE_CHECKING:
    from .core import Harness, HarnessResult, RunUsage, StopReason


logger = logging.getLogger(__name__)

HookEvent = Literal[
    "run_start",
    "user_prompt_submit",
    "before_tool_call",
    "after_tool_call",
    "before_subagent_run",
    "after_subagent_run",
    "run_end",
    "limit_reached",
]
HookHandler = Callable[["HookContext"], None]

TOOL_EVENTS: set[HookEvent] = {"before_tool_call", "after_tool_call"}
AGENT_EVENTS: set[HookEvent] = {"before_subagent_run", "after_subagent_run"}
CANCELLABLE_EVENTS: set[HookEvent] = {"user_prompt_submit", "before_tool_call", "before_subagent_run"}
VALID_EVENTS: set[HookEvent] = {
    "run_start",
    "user_prompt_submit",
    "before_tool_call",
    "after_tool_call",
    "before_subagent_run",
    "after_subagent_run",
    "run_end",
    "limit_reached",
}


@dataclass(frozen=True)
class Hook:
    """One lifecycle hook registration."""

    event: HookEvent
    handler: HookHandler
    tools: list[str] | None = None
    agents: list[str] | None = None

    def __post_init__(self) -> None:
        """Reject nonsensical filter combinations at registration time."""
        if self.event not in VALID_EVENTS:
            raise ValueError(f"unknown hook event: {self.event!r}")
        if not callable(self.handler):
            raise TypeError("hook handler must be callable")
        if self.tools == []:
            raise ValueError("hook tools filter cannot be empty; use None for all tools")
        if self.agents == []:
            raise ValueError("hook agents filter cannot be empty; use None for all agents")
        if self.tools is not None and self.event not in TOOL_EVENTS:
            raise ValueError(f"tools filter is only valid for tool events, not {self.event!r}")
        if self.agents is not None and self.event not in AGENT_EVENTS:
            raise ValueError(f"agents filter is only valid for subagent events, not {self.event!r}")


@dataclass
class HookContext:
    """Mutable lifecycle context passed to hook handlers."""

    event: ClassVar[HookEvent]
    harness: Harness
    metadata: Json = field(default_factory=dict)


@dataclass(kw_only=True)
class RunStartContext(HookContext):
    """Context for a run before the first model request."""

    event: ClassVar[HookEvent] = "run_start"
    prompt: str
    root: Path
    max_model_requests: int
    max_tool_calls: int | None = None


@dataclass(kw_only=True)
class UserPromptSubmitContext(HookContext):
    """Context for the submitted user prompt before querying the model."""

    event: ClassVar[HookEvent] = "user_prompt_submit"
    prompt: str
    additional_context: list[str] = field(default_factory=list)
    cancelled: bool = False
    cancel_reason: str = ""


@dataclass(kw_only=True)
class BeforeToolCallContext(HookContext):
    """Context for a tool call before it is executed."""

    event: ClassVar[HookEvent] = "before_tool_call"
    call_id: str
    tool_name: str
    arguments: str
    tool_spec: ToolSpec | None
    tool_index: int
    cancelled: bool = False
    cancel_reason: str = ""


@dataclass(kw_only=True)
class AfterToolCallContext(HookContext):
    """Context for a completed tool call before the output is finalized."""

    event: ClassVar[HookEvent] = "after_tool_call"
    call_id: str
    tool_name: str
    arguments: str
    original_output: str
    output: str
    parsed_output: Json | None = None
    duration_ms: float


@dataclass(kw_only=True)
class BeforeSubagentRunContext(HookContext):
    """Context for a subagent run before the child harness is built."""

    event: ClassVar[HookEvent] = "before_subagent_run"
    agent: str
    task: str
    inherited: bool
    tool_mode: str
    parent_harness: Harness
    parent_call_id: str | None = None
    cancelled: bool = False
    cancel_reason: str = ""


@dataclass(kw_only=True)
class AfterSubagentRunContext(HookContext):
    """Context for a completed or failed subagent run."""

    event: ClassVar[HookEvent] = "after_subagent_run"
    agent: str
    task: str
    result: HarnessResult | None = None
    error: BaseException | None = None
    tools: list[str] = field(default_factory=list)
    usage: RunUsage | None = None
    parent_call_id: str | None = None


@dataclass(kw_only=True)
class RunEndContext(HookContext):
    """Context for the terminal run outcome."""

    event: ClassVar[HookEvent] = "run_end"
    result: HarnessResult | None = None
    error: BaseException | None = None
    stop_reason: StopReason = "end_turn"
    usage: RunUsage | None = None


@dataclass(kw_only=True)
class LimitReachedContext(HookContext):
    """Context for a hard run limit being reached."""

    event: ClassVar[HookEvent] = "limit_reached"
    limit_kind: Literal["model_requests", "tool_calls", "tool_retries"]
    limit_value: int
    current_count: int


class HookRegistry:
    """Dispatch lifecycle hooks in registration order."""

    def __init__(self, hooks: list[Hook] | None = None, *, strict_hooks: bool = False) -> None:
        self.hooks = list(hooks or [])
        self.strict_hooks = strict_hooks

    def fire(self, ctx: HookContext) -> None:
        """Dispatch matching hooks to one mutable context."""
        for hook in self.hooks:
            if not self._matches(hook, ctx):
                continue
            try:
                hook.handler(ctx)
            except Exception as exc:
                name = _handler_name(hook.handler)
                logger.warning("hook handler failed for event %s: %s", ctx.event, name)
                logger.debug("hook handler traceback for event %s: %s", ctx.event, name, exc_info=True)
                if self.strict_hooks:
                    _mark_strict_hook_exception(exc)
                    raise
            if ctx.event in CANCELLABLE_EVENTS and getattr(ctx, "cancelled", False):
                return

    def fire_after_tool_call(self, ctx: AfterToolCallContext) -> None:
        """Dispatch after-tool hooks while keeping parsed output current."""
        for hook in self.hooks:
            if not self._matches(hook, ctx):
                continue
            ctx.parsed_output = _parse_hook_output(ctx.output)
            try:
                hook.handler(ctx)
            except Exception as exc:
                name = _handler_name(hook.handler)
                logger.warning("hook handler failed for event %s: %s", ctx.event, name)
                logger.debug("hook handler traceback for event %s: %s", ctx.event, name, exc_info=True)
                if self.strict_hooks:
                    _mark_strict_hook_exception(exc)
                    raise

    def validate_filters(self, *, agent_names: set[str]) -> None:
        """Raise for agent filters that do not match registered names."""
        for hook in self.hooks:
            for name in hook.agents or []:
                if name not in agent_names:
                    available = ", ".join(sorted(agent_names)) or "none"
                    raise ValueError(f"hook filter references unknown subagent name: {name}; available: {available}")

    def _matches(self, hook: Hook, ctx: HookContext) -> bool:
        """Return whether a hook applies to a context."""
        if hook.event != ctx.event:
            return False
        if hook.tools is not None and getattr(ctx, "tool_name", None) not in hook.tools:
            return False
        if hook.agents is not None and getattr(ctx, "agent", None) not in hook.agents:
            return False
        return True


def apply_prompt_context(prompt: str, additional_context: list[str]) -> str:
    """Append hook-provided context to the submitted prompt."""
    if not additional_context:
        return prompt
    context = "\n\n".join(additional_context)
    return f"{prompt}\n\n<hook_context>\n{context}\n</hook_context>"


def _handler_name(handler: HookHandler) -> str:
    """Return a readable hook handler name for logs."""
    module = getattr(handler, "__module__", "")
    qualname = getattr(handler, "__qualname__", repr(handler))
    return f"{module}.{qualname}" if module else qualname


def _parse_hook_output(output: str) -> Json | None:
    """Parse current hook output for later after-tool handlers."""
    try:
        parsed = json.loads(output)
    except json.JSONDecodeError:
        return {"ok": False, "content": output, "metadata": {"error_type": "InvalidToolOutput"}}
    return parsed if isinstance(parsed, dict) else {"ok": False, "content": output, "metadata": {"error_type": "InvalidToolOutput"}}


def _mark_strict_hook_exception(exc: BaseException) -> None:
    """Mark strict hook failures so tool plumbing can preserve them."""
    try:
        exc.__dict__["_thinharness_strict_hook"] = True
    except Exception:
        return
