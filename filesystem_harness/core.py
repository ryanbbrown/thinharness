"""SDK-only Responses API agent loop."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .skills import SkillRegistry
from .tools import Json, ToolSpec, call_tool, object_schema
from .tools import builtin_tools as make_builtin_tools


@dataclass
class HarnessConfig:
    """Configuration for Harness."""

    model: str = "gpt-5.2"
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


class ResponsesClient:
    """Tiny stdlib client for OpenAI-compatible /responses endpoints."""

    def __init__(self, *, api_key: str | None = None, base_url: str | None = None, timeout: int = 120) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.timeout = timeout

    def create(self, payload: Json) -> Json:
        """Create a response through the configured endpoint."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            f"{self.base_url}/responses",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise HarnessError(f"Responses API error {exc.code}: {body}") from exc


class Harness:
    """A non-interactive filesystem agent harness for SDK use."""

    def __init__(
        self,
        config: HarnessConfig | None = None,
        *,
        client: ResponsesClient | Any | None = None,
        tools: list[ToolSpec | Json] | None = None,
    ) -> None:
        self.config = config or HarnessConfig()
        self.root = Path(self.config.root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.model = os.getenv("OPENAI_MODEL", self.config.model)
        self.client = client or ResponsesClient(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            timeout=self.config.request_timeout,
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
        if not getattr(self.client, "api_key", True):
            raise HarnessError("OPENAI_API_KEY is required unless api_key or a custom client is passed")
        input_payload: Any = prompt
        previous = previous_response_id
        responses: list[Json] = []
        tool_calls: list[Json] = []

        for turn in range(self.config.max_turns):
            payload = self._payload(input_payload, previous_response_id=previous, metadata=metadata, include_instructions=turn == 0)
            response = self.client.create(payload)
            responses.append(response)
            previous = response.get("id") or previous
            calls = self._function_calls(response)
            if not calls:
                return HarnessResult(text=self._text(response), responses=responses, tool_calls=tool_calls)
            outputs = []
            for call in calls:
                output = self._call_output(call)
                tool_calls.append({"call": call, "output": output["output"]})
                outputs.append(output)
            input_payload = outputs
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
        """Return Responses API tool definitions."""
        return [tool.response_tool() for tool in self.tools]

    def system_instructions(self) -> str:
        """Return the full instruction text sent to the model."""
        return f"{self.config.system_prompt}\n\nWorkspace root: {self.root}\n\n{self.skills.prompt_summary()}"

    def _payload(self, input_payload: Any, *, previous_response_id: str | None, metadata: Json | None, include_instructions: bool) -> Json:
        """Build a Responses API payload."""
        payload: Json = {
            "model": self.model,
            "input": input_payload,
            "tools": self.tool_schemas(),
        }
        if include_instructions:
            payload["instructions"] = self.system_instructions()
        if previous_response_id:
            payload["previous_response_id"] = previous_response_id
        if metadata:
            payload["metadata"] = metadata
        if self.config.temperature is not None:
            payload["temperature"] = self.config.temperature
        payload.update(self.config.extra_body)
        return payload

    def _call_output(self, call: Json) -> Json:
        """Execute one model tool call and format its output."""
        name = call.get("name")
        spec = self._tool_map.get(str(name))
        if not spec:
            output = f"error: unknown tool {name}"
        else:
            try:
                output = call_tool(spec, call.get("arguments") or {})
            except Exception as exc:
                output = f"error: {type(exc).__name__}: {exc}"
        return {"type": "function_call_output", "call_id": call.get("call_id") or call.get("id"), "output": output}

    @staticmethod
    def _function_calls(response: Json) -> list[Json]:
        """Extract function calls from a Responses API object."""
        calls = []
        for item in response.get("output", []) or []:
            if item.get("type") in {"function_call", "tool_call"}:
                calls.append(item)
        return calls

    @staticmethod
    def _text(response: Json) -> str:
        """Extract final text from a Responses API object."""
        if isinstance(response.get("output_text"), str):
            return response["output_text"]
        chunks: list[str] = []
        for item in response.get("output", []) or []:
            for content in item.get("content", []) or []:
                if isinstance(content, dict) and content.get("type") in {"output_text", "text"}:
                    chunks.append(str(content.get("text", "")))
        return "".join(chunks)
