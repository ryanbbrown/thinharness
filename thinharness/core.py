"""SDK-only provider-agnostic agent loop."""

from __future__ import annotations

import contextvars
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .defaults import DEFAULT_SYSTEM_PROMPT
from .hooks import (
    _CURRENT_TOOL_CALL,
    AfterToolCallContext,
    BeforeToolCallContext,
    Hook,
    HookRegistry,
    LimitReachedContext,
    RunEndContext,
    RunStartContext,
    UserPromptSubmitContext,
    apply_prompt_context,
)
from .providers import Model, ModelToolCall, ProviderError, ToolOutput, infer_model
from .skills import SkillRegistry
from .subagents import DEFAULT_SUBAGENT_NAME, SubAgentConfig, create_subagent_tool
from .tools import DEFAULT_SEARCH_LOW_PRIORITY_DIRS, DEFAULT_SEARCH_TEST_DIRS, Json, ToolSpec, call_tool
from .tools import builtin_tools as make_builtin_tools
from .tracing import RunTracer, TracingOptions, annotate_model_span, serialize_attribute_value

MAX_PARALLEL_TOOL_WORKERS = 16
StopReason = Literal["end_turn", "provider_error", "limit_reached", "error", "cancelled_by_hook"]


class HarnessConfig(BaseModel):
    """Configuration for Harness."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: str = "openai:gpt-5.2"
    root: str | Path = "."
    api_key: str | None = None
    base_url: str | None = None
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    skills_dir: str | Path | list[str | Path] | None = None
    selected_skills: list[str] | None = None
    builtin_tools: list[str] | None = None
    output_dir: str | Path | None = None
    max_model_requests: int = 64
    max_tool_calls: int | None = None
    strict_hooks: bool = False
    request_timeout: int = 120
    max_read_chars: int = 40_000
    max_read_bytes: int = 1_000_000
    max_tool_chars: int = 40_000
    max_search_line_chars: int = 180
    rg_timeout: int = 30
    search_exclude_globs: list[str] = Field(default_factory=list)
    search_low_priority_dirs: list[str] = Field(default_factory=lambda: list(DEFAULT_SEARCH_LOW_PRIORITY_DIRS))
    search_test_dirs: list[str] = Field(default_factory=lambda: list(DEFAULT_SEARCH_TEST_DIRS))
    read_paths: list[str | Path] | None = None
    write_paths: list[str | Path] | None = None
    temperature: float | None = None
    extra_body: dict[str, Any] = Field(default_factory=dict)
    tracing: TracingOptions | None = None
    tool_execution: Literal["auto", "sequential"] = "auto"
    subagents: list[SubAgentConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_skills(self) -> HarnessConfig:
        """Validate explicit skill discovery settings."""
        if self.selected_skills is not None and self.skills_dir is None:
            raise ValueError("selected_skills requires skills_dir")
        return self


@dataclass
class HarnessResult:
    """Final result returned by a harness run."""

    text: str
    responses: list[Json] = field(default_factory=list)
    tool_call_records: list[Json] = field(default_factory=list)
    usage: RunUsage = field(default_factory=lambda: RunUsage())
    stop_reason: StopReason = "end_turn"


@dataclass
class RunUsage:
    """Provider and tool usage for one harness run."""

    model_requests: int = 0
    tool_calls: int = 0
    cancelled_tool_calls: int = 0


class HarnessError(RuntimeError):
    """Raised when the harness cannot complete a run."""


class Harness:
    """A non-interactive filesystem agent harness for SDK use."""

    def __init__(
        self,
        config: HarnessConfig | None = None,
        *,
        model: Model | None = None,
        tools: list[ToolSpec | Json] | None = None,
        tracing: TracingOptions | None = None,
        skills: SkillRegistry | None = None,
        hooks: list[Hook] | HookRegistry | None = None,
        subagent_hooks: dict[str, list[Hook] | HookRegistry] | None = None,
    ) -> None:
        self.config = config or HarnessConfig()
        if skills is not None and (self.config.skills_dir is not None or self.config.selected_skills is not None):
            raise ValueError("skills cannot be combined with skills_dir or selected_skills")
        self.root = Path(self.config.root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.model_ref = os.getenv("HARNESS_MODEL", self.config.model)
        self.model = model or infer_model(
            self.model_ref,
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            timeout=self.config.request_timeout,
            temperature=self.config.temperature,
            extra_body=self.config.extra_body,
        )
        self.skills = skills or SkillRegistry(self.config.skills_dir, selected_skills=self.config.selected_skills)
        filesystem_tools = make_builtin_tools(
            self.root,
            output_dir=self.config.output_dir,
            max_read_chars=self.config.max_read_chars,
            max_read_bytes=self.config.max_read_bytes,
            max_tool_chars=self.config.max_tool_chars,
            max_search_line_chars=self.config.max_search_line_chars,
            rg_timeout=self.config.rg_timeout,
            search_exclude_globs=self.config.search_exclude_globs,
            search_low_priority_dirs=self.config.search_low_priority_dirs,
            search_test_dirs=self.config.search_test_dirs,
            read_paths=self.config.read_paths,
            write_paths=self.config.write_paths,
        )
        builtin_candidates = [*filesystem_tools, *self.skills.specs(), create_subagent_tool(self, self.config.subagents)]
        builtin = self._select_builtin_tools(builtin_candidates, self.config.builtin_tools)
        self.tools: list[ToolSpec] = builtin
        self._validate_unique_tools(self.tools)
        self._tool_map = {tool.name: tool for tool in self.tools}
        self.hooks = hooks if isinstance(hooks, HookRegistry) else HookRegistry(hooks, strict_hooks=self.config.strict_hooks)
        self.subagent_hooks = subagent_hooks or {}
        self._suppress_hook_filter_warnings = True
        for tool in tools or []:
            self.add_tool(tool)
        self._suppress_hook_filter_warnings = False
        self._validate_skill_tool_selection()
        self._skills_enabled = bool(self.skills.skills) and any(tool.name in {"skill_read", "skill_run"} for tool in self.tools)
        self.tracing = tracing or self.config.tracing
        self._current_run_metadata: Json | None = None
        self._running = False
        self._warn_unmatched_hook_filters()

    def run(self, prompt: str, *, previous_response_id: str | None = None, metadata: Json | None = None) -> HarnessResult:
        """Run one prompt to completion."""
        if self._running:
            raise HarnessError("Harness.run is not re-entrant")

        responses: list[Json] = []
        tool_call_records: list[Json] = []
        usage = RunUsage()
        result: HarnessResult | None = None
        terminal_error: BaseException | None = None
        stop_reason: StopReason = "end_turn"
        run_end_fired = False
        run_tracer = RunTracer(self.tracing)
        run_metadata = dict(metadata or {})
        self._current_run_metadata = run_metadata
        self._running = True

        def fire_run_end_once() -> None:
            """Emit run_end exactly once for this run."""
            nonlocal run_end_fired
            if run_end_fired:
                return
            run_end_fired = True
            self.hooks.fire(RunEndContext(
                harness=self,
                metadata=dict(run_metadata),
                result=result,
                error=terminal_error,
                stop_reason=stop_reason,
                usage=usage,
            ))

        def check_model_limit() -> None:
            """Raise if another provider request would exceed the configured limit."""
            nonlocal terminal_error, stop_reason
            if usage.model_requests < self.config.max_model_requests:
                return
            self.hooks.fire(LimitReachedContext(
                harness=self,
                metadata=dict(run_metadata),
                limit_kind="model_requests",
                limit_value=self.config.max_model_requests,
                current_count=usage.model_requests,
            ))
            stop_reason = "limit_reached"
            terminal_error = HarnessError(f"model did not finish within max_model_requests={self.config.max_model_requests}")
            raise terminal_error

        def check_tool_limit(batch_size: int) -> None:
            """Raise if a requested tool batch would exceed the configured limit."""
            nonlocal terminal_error, stop_reason
            if self.config.max_tool_calls is None or usage.tool_calls + batch_size <= self.config.max_tool_calls:
                return
            self.hooks.fire(LimitReachedContext(
                harness=self,
                metadata=dict(run_metadata),
                limit_kind="tool_calls",
                limit_value=self.config.max_tool_calls,
                current_count=usage.tool_calls + batch_size,
            ))
            stop_reason = "limit_reached"
            terminal_error = HarnessError(f"tool calls would exceed max_tool_calls={self.config.max_tool_calls}")
            raise terminal_error

        try:
            try:
                conversation_id = str(metadata.get("conversation_id")) if metadata and metadata.get("conversation_id") else None
                with run_tracer.agent(conversation_id=conversation_id) as agent_span:
                    try:
                        self.hooks.fire(RunStartContext(
                            harness=self,
                            metadata=dict(run_metadata),
                            prompt=prompt,
                            root=self.root,
                            max_model_requests=self.config.max_model_requests,
                            max_tool_calls=self.config.max_tool_calls,
                        ))
                        prompt_ctx = UserPromptSubmitContext(harness=self, metadata=dict(run_metadata), prompt=prompt)
                        self.hooks.fire(prompt_ctx)
                        if prompt_ctx.cancelled:
                            reason = prompt_ctx.cancel_reason or "unspecified"
                            stop_reason = "cancelled_by_hook"
                            terminal_error = HarnessError(f"run blocked by hook: {reason}")
                            raise terminal_error
                        effective_prompt = apply_prompt_context(prompt, prompt_ctx.additional_context)
                        try:
                            session = self.model.new_session()
                        except Exception as exc:
                            stop_reason = "error"
                            terminal_error = exc
                            agent_span.record_exception(exc)
                            agent_span.set_error(str(exc), type(exc).__name__)
                            raise

                        check_model_limit()
                        with run_tracer.model(self.model) as model_span:
                            try:
                                turn = session.start(
                                    prompt=effective_prompt,
                                    instructions=self.system_instructions(),
                                    tools=self.tool_schemas(),
                                    metadata=metadata,
                                    previous_response_id=previous_response_id,
                                )
                                usage.model_requests += 1
                            except Exception as exc:
                                model_span.record_exception(exc)
                                model_span.set_error(str(exc), type(exc).__name__)
                                raise
                            annotate_model_span(model_span, turn, capture_messages=bool(self.tracing and self.tracing.capture_messages))
                        while True:
                            responses.append(turn.raw)
                            if not turn.tool_calls:
                                agent_span.set_attribute("gen_ai.completion", turn.text if self.tracing and self.tracing.capture_messages else None)
                                result = HarnessResult(
                                    text=turn.text,
                                    responses=responses,
                                    tool_call_records=tool_call_records,
                                    usage=usage,
                                    stop_reason=stop_reason,
                                )
                                fire_run_end_once()
                                return result
                            check_tool_limit(len(turn.tool_calls))
                            usage.tool_calls += len(turn.tool_calls)
                            recorded, outputs = self._execute_tool_batch(run_tracer, turn.tool_calls)
                            usage.cancelled_tool_calls += sum(1 for record in recorded if record.get("cancelled") is True)
                            tool_call_records.extend(recorded)
                            check_model_limit()
                            with run_tracer.model(self.model) as model_span:
                                try:
                                    turn = session.continue_with_tools(outputs, tools=self.tool_schemas(), metadata=metadata)
                                    usage.model_requests += 1
                                except Exception as exc:
                                    model_span.record_exception(exc)
                                    model_span.set_error(str(exc), type(exc).__name__)
                                    raise
                                annotate_model_span(model_span, turn, capture_messages=bool(self.tracing and self.tracing.capture_messages))
                    except ProviderError as exc:
                        stop_reason = "provider_error"
                        terminal_error = HarnessError(str(exc))
                        agent_span.record_exception(exc)
                        agent_span.set_error(str(exc), type(exc).__name__)
                        raise terminal_error from exc
                    except HarnessError as exc:
                        terminal_error = terminal_error or exc
                        if stop_reason == "end_turn":
                            stop_reason = "error"
                        agent_span.record_exception(exc)
                        agent_span.set_error(str(exc), type(exc).__name__)
                        raise
                    except Exception as exc:
                        stop_reason = "error"
                        terminal_error = exc
                        agent_span.record_exception(exc)
                        agent_span.set_error(str(exc), type(exc).__name__)
                        raise
            finally:
                fire_run_end_once()
        finally:
            self._current_run_metadata = None
            self._running = False

    def add_tool(self, tool: ToolSpec | Json) -> None:
        """Register a custom tool using a ToolSpec or API-style dict."""
        if isinstance(tool, dict):
            parameters = tool.get("args_model") or tool.get("parameters") or {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            }
            spec = ToolSpec(
                name=str(tool["name"]),
                description=str(tool.get("description", "")),
                parameters=parameters,
                handler=tool["handler"],
                sequential=bool(tool.get("sequential", False)),
            )
        else:
            spec = tool
        if not callable(spec.handler):
            raise TypeError(f"handler for tool {spec.name!r} is not callable")
        if spec.name == "subagent" and spec.metadata.get("framework_tool") != "subagent":
            raise ValueError("subagent is a reserved tool name")
        if spec.name in self._tool_map:
            raise ValueError(f"duplicate tool name: {spec.name}")
        self.tools.append(spec)
        self._tool_map[spec.name] = spec
        if not getattr(self, "_suppress_hook_filter_warnings", False):
            self._warn_unmatched_hook_filters()

    def tool_schemas(self) -> list[Json]:
        """Return normalized Responses-style tool definitions."""
        return [tool.response_tool() for tool in self.tools]

    def system_instructions(self) -> str:
        """Return the full instruction text sent to the model."""
        skill_summary = self.skills.prompt_summary() if self._skills_enabled else "No skills are configured."
        return f"{self.config.system_prompt}\n\nWorkspace root: {self.root}\n\n{skill_summary}"

    def _call_output(self, name: str, arguments: str) -> str:
        """Execute one model tool call and format its output."""
        spec = self._tool_map.get(str(name))
        if not spec:
            return json.dumps({"ok": False, "content": f"unknown tool {name}", "metadata": {"tool": name}}, ensure_ascii=False)
        return call_tool(spec, arguments or {})

    def _traced_call_output(self, run_tracer: RunTracer, call_id: str, name: str, arguments: str, tool_index: int) -> tuple[str, bool]:
        """Execute one model tool call with tracing."""
        with run_tracer.tool(tool_name=name, call_id=call_id, arguments=arguments) as span:
            token = _CURRENT_TOOL_CALL.set({"call_id": call_id, "name": name})
            cancelled = False
            start = time.perf_counter()
            try:
                before = BeforeToolCallContext(
                    harness=self,
                    metadata=dict(self._current_run_metadata or {}),
                    call_id=call_id,
                    tool_name=name,
                    arguments=arguments,
                    tool_spec=self._tool_map.get(str(name)),
                    tool_index=tool_index,
                )
                self.hooks.fire(before)
                if before.cancelled:
                    cancelled = True
                    reason = before.cancel_reason or "unspecified"
                    output = json.dumps({
                        "ok": False,
                        "content": f"Tool execution blocked by hook: {reason}",
                        "metadata": {"error_type": "ToolCallCancelled"},
                    }, ensure_ascii=False)
                else:
                    output = self._call_output(name, arguments)
                parsed = _parse_tool_output(output)
                after = AfterToolCallContext(
                    harness=self,
                    metadata=dict(self._current_run_metadata or {}),
                    call_id=call_id,
                    tool_name=name,
                    arguments=arguments,
                    original_output=output,
                    output=output,
                    parsed_output=parsed,
                    duration_ms=(time.perf_counter() - start) * 1000,
                )
                self.hooks.fire_after_tool_call(after)
                output = after.output
                parsed = _parse_tool_output(output)
                metadata = parsed.get("metadata") if isinstance(parsed.get("metadata"), dict) else {}
                if name == "subagent":
                    span.set_attributes({
                        "subagent.name": metadata.get("agent"),
                        "subagent.tool_mode": metadata.get("tool_mode"),
                        "subagent.tools": metadata.get("tools"),
                    })
                if self.tracing and self.tracing.capture_tool_results:
                    span.set_attribute("gen_ai.tool.call.result", serialize_attribute_value(output))
                if parsed.get("ok") is False:
                    span.set_error(f'Tool "{name}" failed', "ToolExecutionError")
                return output, cancelled
            finally:
                _CURRENT_TOOL_CALL.reset(token)

    def _execute_tool_batch(self, run_tracer: RunTracer, calls: list[ModelToolCall]) -> tuple[list[Json], list[ToolOutput]]:
        """Run one batch of model tool calls; preserve model order in returned outputs."""
        if self._should_run_sequentially(calls):
            results = [self._traced_call_output(run_tracer, call.id, call.name, call.arguments, index) for index, call in enumerate(calls)]
        else:
            results = self._run_calls_in_threads(run_tracer, calls)
        recorded = []
        for call, (output, cancelled) in zip(calls, results, strict=True):
            record = {"call": {"id": call.id, "name": call.name, "arguments": call.arguments}, "output": output}
            if cancelled:
                record["cancelled"] = True
            recorded.append(record)
        outputs = [ToolOutput(call.id, output) for call, (output, _cancelled) in zip(calls, results, strict=True)]
        return recorded, outputs

    def _should_run_sequentially(self, calls: list[ModelToolCall]) -> bool:
        """Decide whether the batch must execute serially."""
        if self.config.tool_execution == "sequential" or len(calls) <= 1:
            return True
        return any((spec := self._tool_map.get(str(call.name))) is not None and spec.sequential for call in calls)

    def _run_calls_in_threads(self, run_tracer: RunTracer, calls: list[ModelToolCall]) -> list[tuple[str, bool]]:
        """Execute calls concurrently while keeping the OpenTelemetry parent context."""
        def invoke(index: int, call: ModelToolCall) -> tuple[str, bool]:
            return self._traced_call_output(run_tracer, call.id, call.name, call.arguments, index)

        with ThreadPoolExecutor(max_workers=min(len(calls), MAX_PARALLEL_TOOL_WORKERS)) as executor:
            futures = [executor.submit(contextvars.copy_context().run, invoke, index, call) for index, call in enumerate(calls)]
            return [future.result() for future in futures]

    @staticmethod
    def _validate_unique_tools(tools: list[ToolSpec]) -> None:
        """Reject duplicate tool names before sending schemas to a provider."""
        seen: set[str] = set()
        for tool in tools:
            if tool.name in seen:
                raise ValueError(f"duplicate tool name: {tool.name}")
            seen.add(tool.name)

    def _validate_skill_tool_selection(self) -> None:
        """Require explicit skill tool selection when skills are explicitly configured."""
        if not self.skills.skills:
            return
        tool_names = {tool.name for tool in self.tools}
        if not tool_names.intersection({"skill_read", "skill_run"}):
            raise ValueError("configured skills require exposing skill_read or skill_run")

    def _warn_unmatched_hook_filters(self) -> None:
        """Warn for currently unmatched hook filter names."""
        agent_names = {DEFAULT_SUBAGENT_NAME, *(config.name for config in self.config.subagents)}
        self.hooks.warn_unmatched_filters(tool_names={tool.name for tool in self.tools}, agent_names=agent_names)

    @staticmethod
    def _select_builtin_tools(tools: list[ToolSpec], selected_names: list[str] | None) -> list[ToolSpec]:
        """Return all or the explicitly selected built-in tools."""
        if selected_names is None:
            return tools
        by_name = {tool.name: tool for tool in tools}
        selected: list[ToolSpec] = []
        seen: set[str] = set()
        for name in selected_names:
            if name in seen:
                raise ValueError(f"duplicate selected builtin tool: {name}")
            if name not in by_name:
                available = ", ".join(sorted(by_name)) or "none"
                raise ValueError(f"unknown builtin tool: {name}; available: {available}")
            selected.append(by_name[name])
            seen.add(name)
        return selected


def _parse_tool_output(output: str) -> Json:
    """Parse a normalized tool output envelope."""
    try:
        parsed = json.loads(output)
    except json.JSONDecodeError:
        return {"ok": False, "content": output, "metadata": {"error_type": "InvalidToolOutput"}}
    return parsed if isinstance(parsed, dict) else {"ok": False, "content": output, "metadata": {"error_type": "InvalidToolOutput"}}
