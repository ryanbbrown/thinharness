"""SDK-only provider-agnostic agent loop."""

from __future__ import annotations

import contextvars
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.dataclasses import dataclass

from .defaults import DEFAULT_SYSTEM_PROMPT
from .providers import Model, ModelToolCall, ProviderError, ResponsesClient, ToolOutput, infer_model
from .skills import SkillRegistry
from .subagents import SubAgentConfig
from .tools import DEFAULT_SEARCH_LOW_PRIORITY_DIRS, DEFAULT_SEARCH_TEST_DIRS, Json, ToolSpec, call_tool, object_schema
from .tools import builtin_tools as make_builtin_tools
from .tracing import RunTracer, TracingOptions, annotate_model_span, serialize_attribute_value


MAX_PARALLEL_TOOL_WORKERS = 16
_CURRENT_TOOL_CALL: contextvars.ContextVar[Json | None] = contextvars.ContextVar("thinharness_current_tool_call", default=None)


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
    max_turns: int = 32
    request_timeout: int = 120
    max_read_chars: int = 40_000
    max_read_bytes: int = 1_000_000
    max_tool_chars: int = 40_000
    max_search_line_chars: int = 180
    rg_timeout: int = 30
    search_exclude_globs: list[str] = Field(default_factory=list)
    search_low_priority_dirs: list[str] = Field(default_factory=lambda: list(DEFAULT_SEARCH_LOW_PRIORITY_DIRS))
    search_test_dirs: list[str] = Field(default_factory=lambda: list(DEFAULT_SEARCH_TEST_DIRS))
    temperature: float | None = None
    extra_body: dict[str, Any] = Field(default_factory=dict)
    tracing: TracingOptions | None = None
    tool_execution: Literal["auto", "sequential"] = "auto"
    subagents: list[SubAgentConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_skills(self) -> "HarnessConfig":
        """Validate explicit skill discovery settings."""
        if self.selected_skills is not None and self.skills_dir is None:
            raise ValueError("selected_skills requires skills_dir")
        return self


@dataclass
class HarnessResult:
    """Final result returned by a harness run."""

    text: str
    responses: list[Json] = Field(default_factory=list)
    tool_calls: list[Json] = Field(default_factory=list)


class HarnessError(RuntimeError):
    """Raised when the harness cannot complete a run."""


class Harness:
    """A non-interactive filesystem agent harness for SDK use."""

    def __init__(
        self,
        config: HarnessConfig | None = None,
        *,
        model: Model | None = None,
        adapter: Model | None = None,
        client: ResponsesClient | Any | None = None,
        tools: list[ToolSpec | Json] | None = None,
        tracing: TracingOptions | None = None,
        skills: SkillRegistry | None = None,
    ) -> None:
        self.config = config or HarnessConfig()
        if skills is not None and (self.config.skills_dir is not None or self.config.selected_skills is not None):
            raise ValueError("skills cannot be combined with skills_dir or selected_skills")
        self.root = Path(self.config.root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.model_ref = os.getenv("HARNESS_MODEL", self.config.model)
        self.model = model or adapter or infer_model(
            self.model_ref,
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            timeout=self.config.request_timeout,
            temperature=self.config.temperature,
            extra_body=self.config.extra_body,
            client=client,
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
        )
        from .subagents import create_subagent_tool

        builtin_candidates = [*filesystem_tools, *self.skills.specs(), create_subagent_tool(self, self.config.subagents)]
        builtin = self._select_builtin_tools(builtin_candidates, self.config.builtin_tools)
        self.tools: list[ToolSpec] = builtin
        self._validate_unique_tools(self.tools)
        self._tool_map = {tool.name: tool for tool in self.tools}
        for tool in tools or []:
            self.add_tool(tool)
        self._validate_skill_tool_selection()
        self._skills_enabled = bool(self.skills.skills) and any(tool.name in {"skill_read", "skill_run"} for tool in self.tools)
        self.tracing = tracing or self.config.tracing
        self._current_run_metadata: Json | None = None

    def run(self, prompt: str, *, previous_response_id: str | None = None, metadata: Json | None = None) -> HarnessResult:
        """Run one prompt to completion."""
        responses: list[Json] = []
        tool_calls: list[Json] = []
        run_tracer = RunTracer(self.tracing)
        session = self.model.new_session()
        self._current_run_metadata = dict(metadata or {})

        try:
            with run_tracer.agent(conversation_id=str(metadata.get("conversation_id")) if metadata and metadata.get("conversation_id") else None) as agent_span:
                try:
                    with run_tracer.model(self.model) as model_span:
                        try:
                            turn = session.start(
                                prompt=prompt,
                                instructions=self.system_instructions(),
                                tools=self.tool_schemas(),
                                metadata=metadata,
                                previous_response_id=previous_response_id,
                            )
                        except Exception as exc:
                            model_span.record_exception(exc)
                            model_span.set_error(str(exc), type(exc).__name__)
                            raise
                        annotate_model_span(model_span, turn, capture_messages=bool(self.tracing and self.tracing.capture_messages))
                    for _ in range(self.config.max_turns):
                        responses.append(turn.raw)
                        if not turn.tool_calls:
                            agent_span.set_attribute("gen_ai.completion", turn.text if self.tracing and self.tracing.capture_messages else None)
                            return HarnessResult(text=turn.text, responses=responses, tool_calls=tool_calls)
                        recorded, outputs = self._execute_tool_batch(run_tracer, turn.tool_calls)
                        tool_calls.extend(recorded)
                        with run_tracer.model(self.model) as model_span:
                            try:
                                turn = session.continue_with_tools(outputs, tools=self.tool_schemas(), metadata=metadata)
                            except Exception as exc:
                                model_span.record_exception(exc)
                                model_span.set_error(str(exc), type(exc).__name__)
                                raise
                            annotate_model_span(model_span, turn, capture_messages=bool(self.tracing and self.tracing.capture_messages))
                except ProviderError as exc:
                    agent_span.record_exception(exc)
                    agent_span.set_error(str(exc), type(exc).__name__)
                    raise HarnessError(str(exc)) from exc
                except Exception as exc:
                    agent_span.record_exception(exc)
                    agent_span.set_error(str(exc), type(exc).__name__)
                    raise
                error = HarnessError(f"model did not finish within max_turns={self.config.max_turns}")
                agent_span.set_error(str(error), type(error).__name__)
                raise error
        finally:
            self._current_run_metadata = None

    def add_tool(self, tool: ToolSpec | Json) -> None:
        """Register a custom tool using a ToolSpec or API-style dict."""
        if isinstance(tool, dict):
            parameters = tool.get("args_model") or tool.get("parameters") or object_schema({})
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
        if spec.name in self._tool_map:
            raise ValueError(f"duplicate tool name: {spec.name}")
        self.tools.append(spec)
        self._tool_map = {tool.name: tool for tool in self.tools}

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

    def _traced_call_output(self, run_tracer: RunTracer, call_id: str, name: str, arguments: str) -> str:
        """Execute one model tool call with tracing."""
        with run_tracer.tool(tool_name=name, call_id=call_id, arguments=arguments) as span:
            token = _CURRENT_TOOL_CALL.set({"call_id": call_id, "name": name})
            try:
                output = self._call_output(name, arguments)
            finally:
                _CURRENT_TOOL_CALL.reset(token)
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
            return output

    def _execute_tool_batch(self, run_tracer: RunTracer, calls: list[ModelToolCall]) -> tuple[list[Json], list[ToolOutput]]:
        """Run one batch of model tool calls; preserve model order in returned outputs."""
        if self._should_run_sequentially(calls):
            results = [self._traced_call_output(run_tracer, call.id, call.name, call.arguments) for call in calls]
        else:
            results = self._run_calls_in_threads(run_tracer, calls)
        recorded = [
            {"call": {"id": call.id, "name": call.name, "arguments": call.arguments}, "output": output}
            for call, output in zip(calls, results)
        ]
        outputs = [ToolOutput(call.id, output) for call, output in zip(calls, results)]
        return recorded, outputs

    def _should_run_sequentially(self, calls: list[ModelToolCall]) -> bool:
        """Decide whether the batch must execute serially."""
        if self.config.tool_execution == "sequential" or len(calls) <= 1:
            return True
        return any((spec := self._tool_map.get(str(call.name))) is not None and spec.sequential for call in calls)

    def _run_calls_in_threads(self, run_tracer: RunTracer, calls: list[ModelToolCall]) -> list[str]:
        """Execute calls concurrently while keeping the OpenTelemetry parent context."""
        def invoke(call: ModelToolCall) -> str:
            return self._traced_call_output(run_tracer, call.id, call.name, call.arguments)

        with ThreadPoolExecutor(max_workers=min(len(calls), MAX_PARALLEL_TOOL_WORKERS)) as executor:
            futures = [executor.submit(contextvars.copy_context().run, invoke, call) for call in calls]
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

    @staticmethod
    def _select_builtin_tools(tools: list[ToolSpec], selected_names: list[str] | None) -> list[ToolSpec]:
        """Return all or the explicitly selected built-in tools."""
        if selected_names is None:
            return tools
        by_name = {tool.name: tool for tool in tools}
        selected: list[ToolSpec] = []
        seen: set[str] = set()
        for raw_name in selected_names:
            name = raw_name.lower()
            if name in seen:
                raise ValueError(f"duplicate selected builtin tool: {name}")
            if name not in by_name:
                available = ", ".join(sorted(by_name)) or "none"
                raise ValueError(f"unknown builtin tool: {name}; available: {available}")
            selected.append(by_name[name])
            seen.add(name)
        return selected


def current_tool_call_context() -> Json | None:
    """Return the current tool call context for nested tool handlers."""
    return _CURRENT_TOOL_CALL.get()


def _parse_tool_output(output: str) -> Json:
    """Parse a normalized tool output envelope."""
    try:
        parsed = json.loads(output)
    except json.JSONDecodeError:
        return {"ok": False, "content": output, "metadata": {"error_type": "InvalidToolOutput"}}
    return parsed if isinstance(parsed, dict) else {"ok": False, "content": output, "metadata": {"error_type": "InvalidToolOutput"}}
