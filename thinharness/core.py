"""SDK-only provider-agnostic agent loop."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .providers import Model, ProviderError, ResponsesClient, ToolOutput, infer_model
from .skills import SkillRegistry
from .tools import Json, ToolSpec, call_tool, object_schema
from .tools import builtin_tools as make_builtin_tools


@dataclass
class HarnessConfig:
    """Configuration for Harness."""

    model: str = "openai:gpt-5.2"
    root: str | Path = "."
    api_key: str | None = None
    base_url: str | None = None
    system_prompt: str = "You are a concise filesystem automation agent. Use tools to inspect files before changing them."
    skills_dir: str | Path | None = None
    output_dir: str | Path | None = None
    max_turns: int = 32
    request_timeout: int = 120
    max_read_chars: int = 40_000
    max_tool_chars: int = 40_000
    rg_timeout: int = 30
    temperature: float | None = None
    extra_body: dict[str, Any] = field(default_factory=dict)


@dataclass
class HarnessResult:
    """Final result returned by a harness run."""

    text: str
    responses: list[Json] = field(default_factory=list)
    tool_calls: list[Json] = field(default_factory=list)


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
        skills_dir = self.config.skills_dir or self.root / ".agents" / "skills"
        self.skills = SkillRegistry(skills_dir)
        builtin = make_builtin_tools(
            self.root,
            output_dir=self.config.output_dir,
            max_read_chars=self.config.max_read_chars,
            max_tool_chars=self.config.max_tool_chars,
            rg_timeout=self.config.rg_timeout,
        )
        self.tools: list[ToolSpec] = [*builtin, *self.skills.specs()]
        for tool in tools or []:
            self.add_tool(tool)
        self._tool_map = {tool.name: tool for tool in self.tools}

    def run(self, prompt: str, *, previous_response_id: str | None = None, metadata: Json | None = None) -> HarnessResult:
        """Run one prompt to completion."""
        responses: list[Json] = []
        tool_calls: list[Json] = []

        try:
            turn = self.model.start(
                prompt=prompt,
                instructions=self.system_instructions(),
                tools=self.tool_schemas(),
                metadata=metadata,
                previous_response_id=previous_response_id,
            )
            for _ in range(self.config.max_turns):
                responses.append(turn.raw)
                if not turn.tool_calls:
                    return HarnessResult(text=turn.text, responses=responses, tool_calls=tool_calls)
                outputs = []
                for call in turn.tool_calls:
                    output = self._call_output(call.name, call.arguments)
                    tool_calls.append({"call": {"id": call.id, "name": call.name, "arguments": call.arguments}, "output": output})
                    outputs.append(ToolOutput(call.id, output))
                turn = self.model.continue_with_tools(outputs, tools=self.tool_schemas(), metadata=metadata)
        except ProviderError as exc:
            raise HarnessError(str(exc)) from exc
        raise HarnessError(f"model did not finish within max_turns={self.config.max_turns}")

    def add_tool(self, tool: ToolSpec | Json) -> None:
        """Register a custom tool using a ToolSpec or API-style dict."""
        if isinstance(tool, dict):
            spec = ToolSpec(
                name=str(tool["name"]),
                description=str(tool.get("description", "")),
                parameters=dict(tool.get("parameters") or object_schema({})),
                handler=tool["handler"],
            )
        else:
            spec = tool
        if not callable(spec.handler):
            raise TypeError(f"handler for tool {spec.name!r} is not callable")
        self.tools.append(spec)
        self._tool_map = {tool.name: tool for tool in self.tools}

    def tool_schemas(self) -> list[Json]:
        """Return normalized Responses-style tool definitions."""
        return [tool.response_tool() for tool in self.tools]

    def system_instructions(self) -> str:
        """Return the full instruction text sent to the model."""
        return f"{self.config.system_prompt}\n\nWorkspace root: {self.root}\n\n{self.skills.prompt_summary()}"

    def _call_output(self, name: str, arguments: str) -> str:
        """Execute one model tool call and format its output."""
        spec = self._tool_map.get(str(name))
        if not spec:
            return f"error: unknown tool {name}"
        try:
            return call_tool(spec, arguments or {})
        except Exception as exc:
            return f"error: {type(exc).__name__}: {exc}"
