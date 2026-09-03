from __future__ import annotations

import contextvars
import copy
import json
import threading
import time
from typing import Any

from thinharness import (
    AnthropicProvider,
    ChildHarnessOutcome,
    ChildHarnessRequest,
    OpenAIProvider,
    OpenAIResponsesModel,
    OpenRouterProvider,
    ToolSpec,
)
from thinharness.providers import ModelNotice, ModelTurn, ProviderError

SCRIPTED_MODEL_NAME = "scripted-model"


class FakeChildHarnessHost:
    """No-op child host for direct third-party-style plugin bindings."""

    def register_delegation_tool(self, tool: ToolSpec, recipes) -> ToolSpec:
        del recipes
        return tool

    async def run(self, request: ChildHarnessRequest) -> ChildHarnessOutcome:
        del request
        raise AssertionError("unexpected child harness request")


class FakeClient(OpenAIProvider):
    def __init__(self) -> None:
        super().__init__(api_key="fake")
        self.calls = 0
        self.payloads = []

    async def create_response(self, payload):
        self.calls += 1
        self.payloads.append(copy.deepcopy(payload))
        if self.calls == 1:
            return {
                "id": "resp_1",
                "output": [{
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_1",
                    "status": "completed",
                    "name": "read",
                    "arguments": '{"path":"hello.txt"}',
                }],
            }
        return {
            "id": "resp_2",
            "output": [{
                "type": "message",
                "id": "msg_2",
                "status": "completed",
                "phase": "final_answer",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "done"}],
            }],
        }

def _fake_openai(client: FakeClient) -> OpenAIResponsesModel:
    """Build a Model that routes Responses calls through a fake OpenAIProvider."""
    return OpenAIResponsesModel("test-model", provider=client)


class ReplayOpenAIProvider(OpenAIProvider):
    """Responses fake with exact raw items across two tool rounds."""

    responses = [
        {
            "id": "resp_1",
            "output": [
                {
                    "type": "reasoning",
                    "id": "rs_1",
                    "summary": [{"type": "summary_text", "text": "first thought"}],
                    "encrypted_content": "encrypted-1",
                },
                {
                    "type": "message",
                    "id": "msg_1",
                    "status": "completed",
                    "phase": "commentary",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Reading now."}],
                },
                {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_1",
                    "status": "completed",
                    "name": "read",
                    "arguments": '{"path":"one.txt"}',
                },
            ],
        },
        {
            "id": "resp_2",
            "output": [
                {
                    "type": "reasoning",
                    "id": "rs_2",
                    "summary": [{"type": "summary_text", "text": "second thought"}],
                    "encrypted_content": "encrypted-2",
                },
                {
                    "type": "function_call",
                    "id": "fc_2",
                    "call_id": "call_2",
                    "status": "completed",
                    "name": "read",
                    "arguments": '{"path":"two.txt"}',
                },
            ],
        },
        {
            "id": "resp_3",
            "output": [{
                "type": "message",
                "id": "msg_3",
                "status": "completed",
                "phase": "final_answer",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "done"}],
            }],
        },
        {
            "id": "resp_4",
            "output": [{
                "type": "message",
                "id": "msg_4",
                "status": "completed",
                "phase": "final_answer",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "followed up"}],
            }],
        },
    ]

    def __init__(self, responses=None, *, fail_at: int | None = None) -> None:
        super().__init__(api_key="fake")
        self.script = copy.deepcopy(responses if responses is not None else self.responses)
        self.fail_at = fail_at
        self.payloads: list[dict[str, Any]] = []

    async def create_response(self, payload):
        self.payloads.append(copy.deepcopy(payload))
        if self.fail_at == len(self.payloads):
            raise ProviderError("scripted failure")
        return copy.deepcopy(self.script[len(self.payloads) - 1])


class FakeSpan:
    def __init__(self, name, attributes, parent=None) -> None:
        self.name = name
        self.attributes = dict(attributes or {})
        self.parent = parent
        self.status = None
        self.exceptions = []
        self.ended = False

    def set_attributes(self, attributes) -> None:
        self.attributes.update(attributes)

    def set_attribute(self, key, value) -> None:
        self.attributes[key] = value

    def set_status(self, status) -> None:
        self.status = status

    def record_exception(self, exc) -> None:
        self.exceptions.append(exc)

    def end(self) -> None:
        self.ended = True

class FakeSpanContext:
    def __init__(self, tracer, span) -> None:
        self.tracer = tracer
        self.span = span

    def __enter__(self):
        self.tracer.stack.append(self.span)
        return self.span

    def __exit__(self, exc_type, exc, tb) -> None:
        self.tracer.stack.pop()
        self.span.end()

class FakeTracer:
    def __init__(self) -> None:
        self.stack = []
        self.spans = []

    def start_as_current_span(self, name, **kwargs):
        span = FakeSpan(name, kwargs.get("attributes"), self.stack[-1] if self.stack else None)
        self.spans.append(span)
        return FakeSpanContext(self, span)

class ContextFakeSpanContext:
    def __init__(self, tracer, span) -> None:
        self.tracer = tracer
        self.span = span
        self.token = None

    def __enter__(self):
        stack = [*self.tracer.stack_var.get(), self.span]
        self.token = self.tracer.stack_var.set(stack)
        return self.span

    def __exit__(self, exc_type, exc, tb) -> None:
        self.tracer.stack_var.reset(self.token)
        self.span.end()

class ContextFakeTracer:
    def __init__(self) -> None:
        self.stack_var = contextvars.ContextVar("test_trace_stack", default=())
        self.spans = []
        self.lock = threading.Lock()

    def start_as_current_span(self, name, **kwargs):
        stack = self.stack_var.get()
        span = FakeSpan(name, kwargs.get("attributes"), stack[-1] if stack else None)
        with self.lock:
            self.spans.append(span)
        return ContextFakeSpanContext(self, span)

class FakeAnthropicProvider(AnthropicProvider):
    def __init__(self) -> None:
        super().__init__(api_key="key")
        self.payloads = []

    async def create_message(self, payload):
        """Capture Anthropic payloads and return a tool loop response."""
        self.payloads.append(copy.deepcopy(payload))
        last = payload["messages"][-1]
        if isinstance(last["content"], str):
            return {
                "content": [{"type": "tool_use", "id": f"toolu_{len(self.payloads)}", "name": "echo", "input": {"value": last["content"]}}],
                "stop_reason": "tool_use",
            }
        return {"content": [{"type": "text", "text": "done"}], "stop_reason": "end_turn"}

class FakeOpenRouterProvider(OpenRouterProvider):
    def __init__(self) -> None:
        super().__init__(api_key="key")
        self.payloads = []

    async def create_chat_completion(self, payload):
        """Capture OpenRouter payloads and return a tool loop response."""
        self.payloads.append(copy.deepcopy(payload))
        last = payload["messages"][-1]
        if last["role"] == "user":
            return {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "tool_calls": [{
                            "id": f"call_{len(self.payloads)}",
                            "type": "function",
                            "function": {"name": "echo", "arguments": json.dumps({"value": last["content"]})},
                        }],
                    }
                }]
            }
        return {"choices": [{"message": {"role": "assistant", "content": "done"}}]}

class ScriptedProvider:
    name = "OpenAI"

class ScriptedModel:
    def __init__(self, sessions, *, model: str = SCRIPTED_MODEL_NAME) -> None:
        self.model = model
        self.provider = ScriptedProvider()
        self.api_key = "scripted-key"
        self.sessions = list(sessions)
        self.resume_kind = "scripted"

    def new_session(self):
        """Return the next scripted session."""
        return self.sessions.pop(0)

    def resume_session(self, state):
        """Return the next scripted session for a resumed run."""
        if state.get("kind") != self.resume_kind:
            from thinharness import HarnessError

            raise HarnessError(f"resume_from kind {state.get('kind')!r} does not match {self.resume_kind!r}")
        return self.sessions.pop(0)

class RecordingModel(ScriptedModel):
    def __init__(self, sessions, *, model: str = "recording-model") -> None:
        super().__init__(sessions, model=model)
        self.session_requests = 0

    def new_session(self):
        """Record session requests and return the next scripted session."""
        self.session_requests += 1
        return super().new_session()

class ScriptedSession:
    def __init__(
        self,
        *,
        start_turn: ModelTurn,
        continue_turn: ModelTurn | None = None,
        on_start=None,
        on_continue=None,
        dump_state=None,
    ) -> None:
        self.start_turn = start_turn
        self.continue_turn = continue_turn or ModelTurn(text="done", raw={"id": "done"})
        self.on_start = on_start
        self.on_continue = on_continue
        self.notice_calls: list[tuple[str, list[ModelNotice]]] = []
        self.continue_calls: list[tuple[Any, Any, Any]] = []
        self._dump_state = dump_state if dump_state is not None else {"kind": "scripted", "version": 1, "model": SCRIPTED_MODEL_NAME}

    async def start(self, prompt, constants, *, previous_response_id=None, notices=None):
        """Return the scripted start turn."""
        from thinharness.content import normalize_content, text_only_value

        prompt_value = text_only_value(normalize_content(prompt)) or prompt
        self.notice_calls.append(("start", list(notices or [])))
        if self.on_start:
            self.on_start(prompt_value, constants.instructions, constants.tools, constants.metadata, previous_response_id)
        return self.start_turn

    async def continue_with_tools(self, outputs, constants, *, notices=None):
        """Return the scripted continuation turn."""
        self.notice_calls.append(("continue_with_tools", list(notices or [])))
        self.continue_calls.append((outputs, constants.tools, constants.metadata))
        if self.on_continue:
            self.on_continue(outputs, constants.tools, constants.metadata)
        return self.continue_turn

    async def continue_with_user_content(self, content, constants, *, notices=None):
        """Return the first resumed turn or a scripted correction turn."""
        from thinharness.content import normalize_content, text_only_value

        content_value = text_only_value(normalize_content(content)) or content
        is_resume = not self.notice_calls
        self.notice_calls.append(("continue_with_user_content", list(notices or [])))
        if is_resume:
            if self.on_start:
                self.on_start(content_value, constants.instructions, constants.tools, constants.metadata, None)
            return self.start_turn
        if self.on_continue:
            self.on_continue(content_value, constants.tools, constants.metadata)
        return self.continue_turn

    def dump_state(self):
        """Return scripted resume state."""
        return copy.deepcopy(self._dump_state)

class FailingSession:
    async def start(self, prompt, constants, *, previous_response_id=None, notices=None):
        """Raise a provider failure from the child run."""
        raise ProviderError("child failed")

    async def continue_with_tools(self, outputs, constants, *, notices=None):
        """Never continue after a failed start."""
        raise AssertionError("should not continue")

    async def continue_with_user_content(self, content, constants, *, notices=None):
        """Never continue after a failed start."""
        raise AssertionError("should not continue")

    def dump_state(self):
        """Never provide resume state after a failed start."""
        return None

def echo_tool() -> ToolSpec:
    """Create a custom echo tool."""
    return ToolSpec(
        "echo",
        "Echo input",
        {"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]},
        lambda args: args["value"],
    )

def tool_output(output: str | Any) -> dict:
    """Parse a normalized tool output envelope."""
    if hasattr(output, "to_json"):
        output = output.to_json()
    return json.loads(output)

class MultiCallClient(OpenAIProvider):
    """Fake Responses provider that emits a configured batch of tool calls once, then finishes."""

    def __init__(self, calls):
        super().__init__(api_key="fake")
        self.calls_to_emit = calls
        self.payloads = []
        self.invocations = 0

    async def create_response(self, payload):
        self.invocations += 1
        self.payloads.append(copy.deepcopy(payload))
        if self.invocations == 1:
            return {
                "id": "resp_1",
                "output": [
                    {
                        "type": "function_call",
                        "id": f"fc_{i}",
                        "call_id": f"call_{i}",
                        "status": "completed",
                        "name": name,
                        "arguments": args,
                    }
                    for i, (name, args) in enumerate(self.calls_to_emit, start=1)
                ],
            }
        return {
            "id": "resp_2",
            "output": [{
                "type": "message",
                "id": "msg_2",
                "status": "completed",
                "phase": "final_answer",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "done"}],
            }],
        }

def slow_tool(name: str, delay: float, *, sequential: bool = False) -> ToolSpec:
    """Create a tool that sleeps for delay seconds and echoes its name."""
    return ToolSpec(
        name,
        f"Sleeps {delay}s and returns its name.",
        {"type": "object", "properties": {}, "additionalProperties": False},
        lambda args: (time.sleep(delay), name)[1],
        sequential=sequential,
    )
