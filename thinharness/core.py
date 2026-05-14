"""SDK-only provider-agnostic agent loop."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass

from .providers import Model, ProviderError, ResponsesClient, ToolOutput, infer_model
from .skills import SkillRegistry
from .tools import DEFAULT_SEARCH_LOW_PRIORITY_DIRS, DEFAULT_SEARCH_TEST_DIRS, Json, ToolSpec, call_tool, object_schema
from .tools import builtin_tools as make_builtin_tools
from .tracing import RunTracer, TracingOptions, annotate_model_span, serialize_attribute_value


DEFAULT_SYSTEM_PROMPT = """You are a filesystem automation agent working inside the workspace root.

Use search to find symbols, definitions, references, filenames, and repeated patterns.
Use read to inspect bounded file sections before editing.
Use edit for targeted replacements and write for creating or replacing files.
Start narrow, broaden only if needed, and prefer bounded reads over full-file reads.

When finished, respond concisely with what changed and any verification run."""


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class HarnessConfig:
    """Configuration for Harness."""

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
    ) -> None:
        self.config = config or HarnessConfig()
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
        skills_dir = self.config.skills_dir if self.config.skills_dir is not None else self.root / ".agents" / "skills"
        self.skills = SkillRegistry(skills_dir, selected_skills=self.config.selected_skills)
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
        self._validate_skill_tool_selection()
        builtin = self._select_builtin_tools([*filesystem_tools, *self.skills.specs()], self.config.builtin_tools)
        self._skills_enabled = bool(self.skills.skills) and any(tool.name in {"skill_read", "skill_run"} for tool in builtin)
        self.tools: list[ToolSpec] = builtin
        self._validate_unique_tools(self.tools)
        self._tool_map = {tool.name: tool for tool in self.tools}
        for tool in tools or []:
            self.add_tool(tool)
        self.tracing = tracing or self.config.tracing

    def run(self, prompt: str, *, previous_response_id: str | None = None, metadata: Json | None = None) -> HarnessResult:
        """Run one prompt to completion."""
        responses: list[Json] = []
        tool_calls: list[Json] = []
        run_tracer = RunTracer(self.tracing)

        with run_tracer.agent(conversation_id=str(metadata.get("conversation_id")) if metadata and metadata.get("conversation_id") else None) as agent_span:
            try:
                with run_tracer.model(self.model) as model_span:
                    try:
                        turn = self.model.start(
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
                    outputs = []
                    for call in turn.tool_calls:
                        output = self._traced_call_output(run_tracer, call.id, call.name, call.arguments)
                        tool_calls.append({"call": {"id": call.id, "name": call.name, "arguments": call.arguments}, "output": output})
                        outputs.append(ToolOutput(call.id, output))
                    with run_tracer.model(self.model) as model_span:
                        try:
                            turn = self.model.continue_with_tools(outputs, tools=self.tool_schemas(), metadata=metadata)
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

    def add_tool(self, tool: ToolSpec | Json) -> None:
        """Register a custom tool using a ToolSpec or API-style dict."""
        if isinstance(tool, dict):
            parameters = tool.get("args_model") or tool.get("parameters") or object_schema({})
            spec = ToolSpec(
                name=str(tool["name"]),
                description=str(tool.get("description", "")),
                parameters=parameters,
                handler=tool["handler"],
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
            return f"error: unknown tool {name}"
        try:
            return call_tool(spec, arguments or {})
        except Exception as exc:
            return f"error: {type(exc).__name__}: {exc}"

    def _traced_call_output(self, run_tracer: RunTracer, call_id: str, name: str, arguments: str) -> str:
        """Execute one model tool call with tracing."""
        with run_tracer.tool(tool_name=name, call_id=call_id, arguments=arguments) as span:
            output = self._call_output(name, arguments)
            if self.tracing and self.tracing.capture_tool_results:
                span.set_attribute("gen_ai.tool.call.result", serialize_attribute_value(output))
            if output.startswith("error:"):
                span.set_error(f'Tool "{name}" failed', "ToolExecutionError")
            return output

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
        if not self.skills.skills or (self.config.skills_dir is None and self.config.selected_skills is None):
            return
        selected = {name.lower() for name in self.config.builtin_tools or []}
        if not selected.intersection({"skill_read", "skill_run"}):
            raise ValueError("skills_dir or selected_skills requires selecting skill_read or skill_run in builtin_tools")

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
