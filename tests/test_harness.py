from __future__ import annotations

import copy
import contextvars
import json
import subprocess
import threading
import time
import urllib.error
from pathlib import Path

import pytest
from pydantic import BaseModel, Field

from thinharness import (
    AnthropicMessagesModel,
    AnthropicProvider,
    Harness,
    HarnessConfig,
    OpenRouterModel,
    OpenRouterProvider,
    OpenAIProvider,
    OpenAIResponsesModel,
    SkillRegistry,
    SubAgentConfig,
    ToolSpec,
    TracingOptions,
    build_child_harness,
    create_subagent_tool,
    parse_model_ref,
)
from thinharness.providers import ModelToolCall, ModelTurn, ProviderError, ToolOutput
from thinharness.tools import FileTools


class FakeClient:
    def __init__(self) -> None:
        self.api_key = "fake"
        self.calls = 0
        self.payloads = []

    def create(self, payload):
        self.calls += 1
        self.payloads.append(payload)
        if self.calls == 1:
            return {
                "id": "resp_1",
                "output": [{
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "read",
                    "arguments": '{"path":"hello.txt"}',
                }],
            }
        return {"id": "resp_2", "output_text": "done"}


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
        self.stack_var = contextvars.ContextVar("test_trace_stack", default=[])
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

    def create_message(self, payload):
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

    def create_chat_completion(self, payload):
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
    def __init__(self, sessions, *, model: str = "scripted-model") -> None:
        self.model = model
        self.provider = ScriptedProvider()
        self.api_key = "scripted-key"
        self.sessions = list(sessions)

    def new_session(self):
        """Return the next scripted session."""
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
    def __init__(self, *, start_turn: ModelTurn, continue_turn: ModelTurn | None = None, on_start=None, on_continue=None) -> None:
        self.start_turn = start_turn
        self.continue_turn = continue_turn or ModelTurn(text="done", raw={"id": "done"})
        self.on_start = on_start
        self.on_continue = on_continue

    def start(self, *, prompt, instructions, tools, metadata=None, previous_response_id=None):
        """Return the scripted start turn."""
        if self.on_start:
            self.on_start(prompt, instructions, tools, metadata, previous_response_id)
        return self.start_turn

    def continue_with_tools(self, outputs, *, tools, metadata=None):
        """Return the scripted continuation turn."""
        if self.on_continue:
            self.on_continue(outputs, tools, metadata)
        return self.continue_turn


class FailingSession:
    def start(self, *, prompt, instructions, tools, metadata=None, previous_response_id=None):
        """Raise a provider failure from the child run."""
        raise ProviderError("child failed")

    def continue_with_tools(self, outputs, *, tools, metadata=None):
        """Never continue after a failed start."""
        raise AssertionError("should not continue")


def echo_tool() -> ToolSpec:
    """Create a custom echo tool."""
    return ToolSpec(
        "echo",
        "Echo input",
        {"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]},
        lambda args: args["value"],
    )


def tool_output(output: str) -> dict:
    """Parse a normalized tool output envelope."""
    return json.loads(output)


def test_file_tools_read_write_edit_and_list(tmp_path: Path) -> None:
    tools = FileTools(tmp_path)
    assert tools.write({"path": "notes/todo.txt", "content": "one\ntwo\n"}).ok
    read = tools.read({"path": "notes/todo.txt", "offset": 2, "limit": 1})
    assert read.ok
    assert "2\ttwo" in read.content
    edit = tools.edit({"path": "notes/todo.txt", "old_string": "two", "new_string": "TWO"})
    assert edit.ok
    listed = tools.list_files({"path": ".", "recursive": True})
    assert "notes/todo.txt" in listed.content


def test_file_tools_large_reads_require_and_stream_bounded_range(tmp_path: Path) -> None:
    path = tmp_path / "large.txt"
    path.write_text("one\ntwo\nthree\nfour\nfive\n", encoding="utf-8")
    tools = FileTools(tmp_path, max_read_bytes=10)

    unbounded = tools.read({"path": "large.txt"})
    assert not unbounded.ok
    assert "pass offset and limit" in unbounded.content

    bounded = tools.read({"path": "large.txt", "offset": 3, "limit": 2})
    assert bounded.ok
    assert "3\tthree" in bounded.content
    assert "4\tfour" in bounded.content
    assert "one" not in bounded.content
    assert bounded.metadata["total_lines"] is None


def test_file_tools_reject_path_escape(tmp_path: Path) -> None:
    tools = FileTools(tmp_path)
    read = tools.read({"path": "../outside.txt"})
    assert not read.ok
    assert read.metadata["error_type"] == "PathValidationError"
    write = tools.write({"path": "/tmp/outside.txt", "content": "no"})
    assert not write.ok
    assert write.metadata["error_type"] == "PathValidationError"


def test_file_tools_enforce_read_and_write_paths(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "docs").mkdir()
    (tmp_path / "src" / "app.py").write_text("def Target():\n    pass\n", encoding="utf-8")
    (tmp_path / "tests" / "test_app.py").write_text("Target()\n", encoding="utf-8")
    (tmp_path / "docs" / "note.md").write_text("Target\n", encoding="utf-8")
    tools = FileTools(tmp_path, read_paths=["src", "tests"], write_paths=["src"])

    assert tools.read({"path": "src/app.py"}).ok
    blocked_read = tools.read({"path": "docs/note.md"})
    assert not blocked_read.ok
    assert blocked_read.metadata["error_type"] == "PathValidationError"

    assert tools.write({"path": "src/generated.py", "content": "ok\n"}).ok
    blocked_write = tools.write({"path": "tests/generated.py", "content": "no\n"})
    assert not blocked_write.ok
    assert blocked_write.metadata["error_type"] == "PathValidationError"

    search = tools.search({"query": "Target"})
    assert search.ok
    assert "src/app.py" in search.content
    assert "tests/test_app.py" in search.content
    assert "docs/note.md" not in search.content


def test_file_tools_validate_glob_selectors(tmp_path: Path) -> None:
    tools = FileTools(tmp_path)
    for result in [
        tools.search({"query": "x", "path_glob": "../*.py"}),
        tools.list_files({"path": ".", "glob": "../*"}),
        tools.glob({"path": ".", "pattern": "/tmp/*"}),
        tools.jsonl_search({"path_glob": "src/../../*.jsonl"}),
    ]:
        assert not result.ok
        assert result.metadata["error_type"] == "PathValidationError"


def test_search_ranks_and_formats_agent_results(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "src" / "app.py").write_text("def HandleRequest():\n    return 'ok'\n", encoding="utf-8")
    (tmp_path / "tests" / "test_app.py").write_text("from src.app import HandleRequest\n", encoding="utf-8")
    tools = FileTools(tmp_path)
    result = tools.search({"query": "HandleRequest", "max_files": 5})
    assert result.ok
    assert "summary:" in result.content
    assert "best_next_step: read src/app.py around line 1" in result.content
    assert result.content.index("src/app.py") < result.content.index("tests/test_app.py")
    assert "why: definition, source" in result.content
    assert result.metadata["cmd"] == ["rg", "--json", "--", "HandleRequest", "."]


def test_search_no_matches_has_refinement_hint(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("hello\n", encoding="utf-8")
    result = FileTools(tmp_path).search({"query": "MissingThing", "path_glob": "**/*.py"})
    assert result.ok
    assert "No matches found." in result.content
    assert "scope: glob=**/*.py" in result.content


def test_search_reports_ripgrep_errors(tmp_path: Path) -> None:
    result = FileTools(tmp_path).search({"query": "["})
    assert not result.ok
    assert "ripgrep failed" in result.content
    assert result.metadata["returncode"] not in (0, 1)


def test_search_line_preview_limit_is_search_only(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("target = '" + ("x" * 40) + "'\n", encoding="utf-8")
    result = FileTools(tmp_path, max_search_line_chars=12).search({"query": "target"})
    assert result.ok
    assert "target = 'xx..." in result.content


def test_search_excludes_and_priority_are_configurable(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "vendor").mkdir()
    (tmp_path / "custom_low").mkdir()
    (tmp_path / "src" / "app.py").write_text("def Target():\n    pass\n", encoding="utf-8")
    (tmp_path / "vendor" / "lib.py").write_text("def Target():\n    pass\n", encoding="utf-8")
    (tmp_path / "custom_low" / "lib.py").write_text("def Target():\n    pass\n", encoding="utf-8")

    excluded = FileTools(tmp_path, search_exclude_globs=["vendor/**"]).search({"query": "Target"})
    assert excluded.ok
    assert "vendor/lib.py" not in excluded.content
    assert excluded.metadata["cmd"][:4] == ["rg", "--json", "--glob", "!vendor/**"]

    ranked = FileTools(tmp_path, search_low_priority_dirs=["custom_low"]).search({"query": "Target"})
    assert ranked.ok
    assert "custom_low/lib.py\n  why: definition, low-priority" in ranked.content
    assert "vendor/lib.py\n  why: definition, source" in ranked.content


def test_search_timeout_returns_structured_result(tmp_path: Path, monkeypatch) -> None:
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(kwargs.get("args", "rg"), timeout=1)

    monkeypatch.setattr("subprocess.run", timeout)
    result = FileTools(tmp_path).search({"query": "Target", "timeout": 1})

    assert not result.ok
    assert result.content == "ripgrep timed out after 1s"
    assert result.metadata["timeout"] == 1


def test_jsonl_search_filters_projects_and_formats(tmp_path: Path) -> None:
    rows = [
        {"id": 1, "user": {"name": "alice", "tags": ["admin", "ops"]}, "msg": "login ok"},
        {"id": 2, "user": {"name": "bob", "tags": ["user"]}, "msg": "login ok"},
        {"id": 3, "user": {"name": "carol", "tags": ["admin"]}, "msg": "login fail"},
    ]
    data = tmp_path / "events.jsonl"
    data.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    result = FileTools(tmp_path).jsonl_search({
        "path_glob": "*.jsonl",
        "fields": {"user.name": 0, "msg": 4},
        "where": [
            {"field": "user.tags[0]", "op": "eq", "value": "admin"},
            {"field": 'user["name"]', "op": "regex", "value": "^[ac]"},
        ],
    })
    assert result.ok, result.content
    assert "rows_matched: 2" in result.content
    assert 'events.jsonl:1: {"user.name": "alice", "msg": "logi…"}' in result.content
    assert 'events.jsonl:3: {"user.name": "carol", "msg": "logi…"}' in result.content
    assert "bob" not in result.content


def test_jsonl_search_uses_ripgrep_prefilter(tmp_path: Path) -> None:
    data = tmp_path / "events.jsonl"
    data.write_text(
        '{"id":1,"msg":"login ok"}\n{"id":2,"msg":"logout ok"}\n{"id":3,"msg":"login fail"}\n',
        encoding="utf-8",
    )
    result = FileTools(tmp_path).jsonl_search({
        "query": "login",
        "path_glob": "*.jsonl",
        "where": [{"field": "msg", "op": "contains", "value": "fail"}],
        "fields": {"id": 0},
    })
    assert result.ok
    assert "rows_matched: 1" in result.content
    assert 'events.jsonl:3: {"id": 3}' in result.content


def test_jsonl_search_reports_ripgrep_errors(tmp_path: Path) -> None:
    result = FileTools(tmp_path).jsonl_search({"query": "[", "path_glob": "*.jsonl"})
    assert not result.ok
    assert "ripgrep failed" in result.content


def test_jsonl_search_limits_display_without_losing_counts(tmp_path: Path) -> None:
    data = tmp_path / "events.jsonl"
    data.write_text(
        "\n".join(json.dumps({"id": i, "msg": "hit"}) for i in range(1, 5)) + "\n",
        encoding="utf-8",
    )

    result = FileTools(tmp_path).jsonl_search({"path_glob": "*.jsonl", "max_matches_per_file": 2})

    assert result.ok
    assert "rows_matched: 4" in result.content
    assert 'events.jsonl:1: {"id": 1, "msg": "hit"}' in result.content
    assert 'events.jsonl:2: {"id": 2, "msg": "hit"}' in result.content
    assert "events.jsonl:3" not in result.content
    assert "... 2 more row(s) in events.jsonl" in result.content


def test_jsonl_search_timeout_returns_structured_result(tmp_path: Path, monkeypatch) -> None:
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(kwargs.get("args", "rg"), timeout=1)

    monkeypatch.setattr("subprocess.run", timeout)
    result = FileTools(tmp_path).jsonl_search({"query": "hit", "path_glob": "*.jsonl", "timeout": 1})

    assert not result.ok
    assert result.content == "ripgrep timed out after 1s"
    assert result.metadata["timeout"] == 1


def test_skill_registry_reads_and_runs_skill(tmp_path: Path) -> None:
    skill = tmp_path / "skills" / "demo"
    (skill / "scripts").mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: demo\ndescription: Demo skill\n---\nBody", encoding="utf-8")
    script = skill / "scripts" / "echo.py"
    script.write_text("import sys\nprint('hi', *sys.argv[1:])\n", encoding="utf-8")
    registry = SkillRegistry(tmp_path / "skills")
    assert "demo - Demo skill" in registry.prompt_summary()
    read = registry.skill_read({"skill_name": "demo"})
    assert read.ok
    assert "SKILL.md" in read.content
    run = registry.skill_run({"skill_name": "demo", "script": "scripts/echo.py", "args": ["there"]})
    assert run.ok
    assert "hi there" in run.content


def test_skill_run_timeout_returns_structured_result(tmp_path: Path, monkeypatch) -> None:
    skill = tmp_path / "skills" / "demo"
    (skill / "scripts").mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: demo\n---\nBody", encoding="utf-8")
    (skill / "scripts" / "slow.py").write_text("print('slow')\n", encoding="utf-8")
    registry = SkillRegistry(tmp_path / "skills")

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(kwargs.get("args", "python"), timeout=1)

    monkeypatch.setattr("subprocess.run", timeout)
    result = registry.skill_run({"skill_name": "demo", "script": "scripts/slow.py", "timeout": 1})

    assert not result.ok
    assert result.content == "skill script timed out after 1s"
    assert result.metadata["timeout"] == 1


def test_skill_registry_aggregates_dirs_and_filters_selected_skills(tmp_path: Path) -> None:
    alpha = tmp_path / "a" / "alpha"
    beta = tmp_path / "b" / "beta"
    alpha.mkdir(parents=True)
    beta.mkdir(parents=True)
    (alpha / "SKILL.md").write_text("---\nname: alpha\ndescription: Alpha skill\n---\nAlpha", encoding="utf-8")
    (beta / "SKILL.md").write_text("---\nname: beta\ndescription: Beta skill\n---\nBeta", encoding="utf-8")

    registry = SkillRegistry([tmp_path / "a", tmp_path / "b"], selected_skills=["beta"])

    assert list(registry.skills) == ["beta"]
    assert "beta - Beta skill" in registry.prompt_summary()
    assert "alpha - Alpha skill" not in registry.prompt_summary()


def test_skill_registry_rejects_duplicate_skill_names(tmp_path: Path) -> None:
    first = tmp_path / "first" / "demo"
    second = tmp_path / "second" / "demo"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    (first / "SKILL.md").write_text("---\nname: demo\n---\nFirst", encoding="utf-8")
    (second / "SKILL.md").write_text("---\nname: demo\n---\nSecond", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate skill name: demo"):
        SkillRegistry([tmp_path / "first", tmp_path / "second"])


def test_harness_tool_loop_with_custom_client(tmp_path: Path) -> None:
    (tmp_path / "hello.txt").write_text("hello", encoding="utf-8")
    client = FakeClient()
    harness = Harness(HarnessConfig(root=tmp_path, model="openai:test-model"), client=client)
    result = harness.run("read hello", metadata={"case": "test"})
    assert result.text == "done"
    assert client.payloads[0]["tools"]
    assert client.payloads[0]["metadata"] == {"case": "test"}
    assert client.payloads[1]["previous_response_id"] == "resp_1"
    assert client.payloads[1]["input"][0]["type"] == "function_call_output"
    assert "hello" in client.payloads[1]["input"][0]["output"]


def test_anthropic_harness_reuses_model_without_message_leak(tmp_path: Path) -> None:
    provider = FakeAnthropicProvider()
    model = AnthropicMessagesModel("claude-test", provider=provider)
    harness = Harness(HarnessConfig(root=tmp_path, builtin_tools=[]), model=model, tools=[echo_tool()])

    assert harness.run("first").text == "done"
    assert harness.run("second").text == "done"

    assert provider.payloads[0]["messages"] == [{"role": "user", "content": "first"}]
    assert provider.payloads[2]["messages"] == [{"role": "user", "content": "second"}]


def test_openrouter_harness_reuses_model_without_message_leak(tmp_path: Path) -> None:
    provider = FakeOpenRouterProvider()
    model = OpenRouterModel("openai/test", provider=provider)
    harness = Harness(HarnessConfig(root=tmp_path, builtin_tools=[]), model=model, tools=[echo_tool()])

    assert harness.run("first").text == "done"
    assert harness.run("second").text == "done"

    assert provider.payloads[0]["messages"] == [
        {"role": "system", "content": harness.system_instructions()},
        {"role": "user", "content": "first"},
    ]
    assert provider.payloads[2]["messages"] == [
        {"role": "system", "content": harness.system_instructions()},
        {"role": "user", "content": "second"},
    ]


def test_harness_tracing_records_agent_model_and_tool_spans(tmp_path: Path) -> None:
    (tmp_path / "hello.txt").write_text("hello", encoding="utf-8")
    tracer = FakeTracer()
    harness = Harness(
        HarnessConfig(root=tmp_path, model="openai:test-model"),
        client=FakeClient(),
        tracing=TracingOptions(
            tracer=tracer,
            agent_name="test-agent",
            capture_messages=True,
            capture_tool_args=True,
            capture_tool_results=True,
        ),
    )

    result = harness.run("read hello", metadata={"conversation_id": "conv-1"})

    assert result.text == "done"
    assert [span.name for span in tracer.spans] == [
        "invoke_agent test-agent",
        "chat test-model",
        "execute_tool read",
        "chat test-model",
    ]
    root, first_chat, tool, second_chat = tracer.spans
    assert first_chat.parent is root
    assert tool.parent is root
    assert second_chat.parent is root
    assert root.attributes["gen_ai.operation.name"] == "invoke_agent"
    assert root.attributes["gen_ai.conversation.id"] == "conv-1"
    assert root.attributes["gen_ai.completion"] == "done"
    assert first_chat.attributes["gen_ai.provider.name"] == "OpenAI"
    assert first_chat.attributes["gen_ai.request.model"] == "test-model"
    assert tool.attributes["gen_ai.tool.name"] == "read"
    assert tool.attributes["gen_ai.tool.call.id"] == "call_1"
    assert tool.attributes["gen_ai.tool.call.arguments"] == '{"path":"hello.txt"}'
    assert "hello" in tool.attributes["gen_ai.tool.call.result"]
    assert second_chat.attributes["gen_ai.completion"] == "done"


def test_tool_tracing_marks_normalized_failures(tmp_path: Path) -> None:
    failing = ToolSpec("fail", "Returns failure.", {"type": "object", "properties": {}}, lambda args: ToolResult(False, "nope"))
    client = MultiCallClient([("fail", "{}")])
    tracer = FakeTracer()
    harness = Harness(
        HarnessConfig(root=tmp_path, model="openai:test-model", builtin_tools=[]),
        client=client,
        tools=[failing],
        tracing=TracingOptions(tracer=tracer),
    )

    harness.run("go")

    tool = next(span for span in tracer.spans if span.name == "execute_tool fail")
    assert tool.status is not None
    assert tool.attributes["error.type"] == "ToolExecutionError"


def test_subagent_tracing_nests_child_under_parent_tool_span(tmp_path: Path) -> None:
    parent_call = ModelTurn(
        tool_calls=[ModelToolCall(id="call_1", name="subagent", arguments='{"task":"help"}')],
        raw={"id": "parent-start"},
    )
    child = ScriptedSession(start_turn=ModelTurn(text="child done", raw={"id": "child"}))
    parent = ScriptedSession(start_turn=parent_call, continue_turn=ModelTurn(text="parent done", raw={"id": "parent-done"}))
    tracer = FakeTracer()
    harness = Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[]),
        model=ScriptedModel([parent, child]),
        tools=[echo_tool()],
        tracing=TracingOptions(tracer=tracer),
    )
    harness.add_tool(create_subagent_tool(harness, []))

    harness.run("delegate")

    assert [span.name for span in tracer.spans] == [
        "invoke_agent thinharness",
        "chat scripted-model",
        "execute_tool subagent",
        "invoke_agent subagent.default",
        "chat scripted-model",
        "chat scripted-model",
    ]
    root, first_chat, subagent_tool, child_agent, child_chat, final_chat = tracer.spans
    assert first_chat.parent is root
    assert subagent_tool.parent is root
    assert child_agent.parent is subagent_tool
    assert child_chat.parent is child_agent
    assert final_chat.parent is root
    assert child_agent.attributes["gen_ai.agent.name"] == "subagent.default"
    assert subagent_tool.attributes["subagent.name"] == "default"
    assert subagent_tool.attributes["subagent.tool_mode"] == "inherited"
    assert subagent_tool.attributes["subagent.tools"] == ["echo"]


def test_custom_tool_is_exposed_and_callable(tmp_path: Path) -> None:
    custom = ToolSpec(
        "echo_json",
        "Echo input",
        {"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]},
        lambda args: {"echo": args["value"]},
    )
    harness = Harness(HarnessConfig(root=tmp_path), client=FakeClient(), tools=[custom])
    assert any(tool["name"] == "echo_json" for tool in harness.tool_schemas())
    output = tool_output(harness._call_output("echo_json", '{"value":"ok"}'))
    assert output["ok"] is True
    assert json.loads(output["content"]) == {"echo": "ok"}


def test_custom_tool_can_use_pydantic_args_model(tmp_path: Path) -> None:
    class EchoArgs(BaseModel):
        value: str
        count: int = Field(default=1, ge=1)

    custom = ToolSpec("echo_typed", "Echo typed input", EchoArgs, lambda args: {"echo": args.value * args.count})
    harness = Harness(HarnessConfig(root=tmp_path), client=FakeClient(), tools=[custom])

    schema = next(tool for tool in harness.tool_schemas() if tool["name"] == "echo_typed")["parameters"]
    assert schema["properties"]["count"]["minimum"] == 1
    output = tool_output(harness._call_output("echo_typed", '{"value":"ok","count":2}'))
    assert json.loads(output["content"]) == {"echo": "okok"}
    invalid = tool_output(harness._call_output("echo_typed", '{"value":"ok","count":0}'))
    assert invalid["ok"] is False
    assert "invalid arguments" in invalid["content"]


def test_custom_tool_invalid_json_is_structured(tmp_path: Path) -> None:
    custom = ToolSpec("echo", "Echo input", {"type": "object", "properties": {}}, lambda args: "ok")
    harness = Harness(HarnessConfig(root=tmp_path), client=FakeClient(), tools=[custom])

    output = json.loads(harness._call_output("echo", "{bad json"))

    assert output["ok"] is False
    assert "invalid JSON arguments" in output["content"]


def test_builtin_tool_selection_is_explicit(tmp_path: Path) -> None:
    harness = Harness(HarnessConfig(root=tmp_path, builtin_tools=["read", "search"]), client=FakeClient())
    assert [tool["name"] for tool in harness.tool_schemas()] == ["read", "search"]


def test_skill_dirs_require_selected_skill_tools(tmp_path: Path) -> None:
    skill = tmp_path / "skills" / "demo"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: demo\n---\nDemo", encoding="utf-8")

    harness = Harness(HarnessConfig(root=tmp_path, skills_dir=tmp_path / "skills"), client=FakeClient())
    assert "skill_read" in [tool["name"] for tool in harness.tool_schemas()]
    with pytest.raises(ValueError, match="skill_read or skill_run"):
        Harness(HarnessConfig(root=tmp_path, skills_dir=tmp_path / "skills", builtin_tools=["read"]), client=FakeClient())


def test_skills_are_not_discovered_without_explicit_skills_dir(tmp_path: Path) -> None:
    skill = tmp_path / ".agents" / "skills" / "demo"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: demo\n---\nDemo", encoding="utf-8")

    harness = Harness(HarnessConfig(root=tmp_path), client=FakeClient())

    assert "skill_read" not in [tool["name"] for tool in harness.tool_schemas()]
    assert "No skills are configured." in harness.system_instructions()


def test_selected_skills_are_exposed_when_skill_tool_is_selected(tmp_path: Path) -> None:
    demo = tmp_path / "skills" / "demo"
    other = tmp_path / "skills" / "other"
    demo.mkdir(parents=True)
    other.mkdir(parents=True)
    (demo / "SKILL.md").write_text("---\nname: demo\ndescription: Demo skill\n---\nDemo", encoding="utf-8")
    (other / "SKILL.md").write_text("---\nname: other\ndescription: Other skill\n---\nOther", encoding="utf-8")

    harness = Harness(
        HarnessConfig(
            root=tmp_path,
            skills_dir=tmp_path / "skills",
            selected_skills=["demo"],
            builtin_tools=["read", "skill_read"],
        ),
        client=FakeClient(),
    )

    assert [tool["name"] for tool in harness.tool_schemas()] == ["read", "skill_read"]
    assert "demo - Demo skill" in harness.system_instructions()
    assert "other - Other skill" not in harness.system_instructions()


def test_selected_skills_without_skills_dir_fails() -> None:
    with pytest.raises(ValueError, match="selected_skills requires skills_dir"):
        HarnessConfig(selected_skills=["demo"])


def test_subagent_config_validation_accepts_tool_specs_and_dict_tools() -> None:
    spec = echo_tool()
    dict_tool = {
        "name": "dict_echo",
        "description": "Dict echo",
        "parameters": {"type": "object", "properties": {}},
        "handler": lambda args: "ok",
        "sequential": True,
    }
    config = SubAgentConfig(name="research.1", description="Research helper.", tools=[spec, dict_tool])
    inherited = SubAgentConfig(name="general", description="General helper.", inherit_parent_tools=True)

    assert config.tools == [spec, dict_tool]
    assert inherited.inherit_parent_tools is True
    with pytest.raises(ValueError, match="inherit_parent_tools"):
        SubAgentConfig(name="bad", description="Bad helper.", inherit_parent_tools=True, builtin_tools=["read"])
    with pytest.raises(ValueError, match="cannot be exposed"):
        SubAgentConfig(name="recursive", description="Recursive helper.", builtin_tools=["subagent"])
    with pytest.raises(ValueError, match="cannot be exposed"):
        SubAgentConfig(
            name="recursive-custom",
            description="Recursive helper.",
            tools=[ToolSpec("subagent", "Recursive", {"type": "object", "properties": {}}, lambda args: "bad")],
        )
    with pytest.raises(ValueError, match="must define"):
        SubAgentConfig(name="empty", description="No tools.")
    with pytest.raises(ValueError):
        SubAgentConfig(name="bad name", description="Bad helper.", builtin_tools=["read"])
    with pytest.raises(ValueError, match="description must not be empty"):
        SubAgentConfig(name="ok", description="   ", builtin_tools=["read"])
    with pytest.raises(ValueError, match="single line"):
        SubAgentConfig(name="ok", description="Bad\nhelper.", builtin_tools=["read"])


def test_subagent_builtin_exposure_is_selectable(tmp_path: Path) -> None:
    default = Harness(HarnessConfig(root=tmp_path), model=ScriptedModel([]))
    disabled = Harness(HarnessConfig(root=tmp_path, builtin_tools=[]), model=ScriptedModel([]))
    only_subagent = Harness(HarnessConfig(root=tmp_path, builtin_tools=["subagent"]), model=ScriptedModel([]))

    assert "subagent" in [tool["name"] for tool in default.tool_schemas()]
    assert "subagent" not in [tool["name"] for tool in disabled.tool_schemas()]
    assert [tool["name"] for tool in only_subagent.tool_schemas()] == ["subagent"]
    schema = only_subagent.tool_schemas()[0]["parameters"]
    assert set(schema["properties"]) == {"task", "agent"}
    assert "tools" not in schema["properties"]


def test_default_subagent_runs_child_with_inherited_tools_and_structured_result(tmp_path: Path) -> None:
    parent_call = ModelTurn(
        tool_calls=[ModelToolCall(id="call_1", name="subagent", arguments='{"task":"help"}')],
        raw={"id": "parent-start"},
    )
    child_start_metadata = {}

    def on_child_start(prompt, _instructions, tools, metadata, _previous_response_id):
        child_start_metadata.update({"prompt": prompt, "tools": [tool["name"] for tool in tools], "metadata": metadata})

    def on_parent_continue(outputs, _tools, _metadata):
        envelope = tool_output(outputs[0].output)
        assert envelope["ok"] is True
        assert envelope["content"] == "child done"
        assert envelope["metadata"]["agent"] == "default"
        assert envelope["metadata"]["inherited"] is True
        assert envelope["metadata"]["tools"] == ["echo"]

    child = ScriptedSession(start_turn=ModelTurn(text="child done", raw={"id": "child"}), on_start=on_child_start)
    parent = ScriptedSession(start_turn=parent_call, continue_turn=ModelTurn(text="parent done", raw={"id": "parent-done"}), on_continue=on_parent_continue)
    model = ScriptedModel([parent, child])
    harness = Harness(HarnessConfig(root=tmp_path, builtin_tools=[]), model=model, tools=[echo_tool()])
    harness.add_tool(create_subagent_tool(harness, []))

    result = harness.run("delegate", metadata={"conversation_id": "conv-1", "extra": "ignored"})

    assert result.text == "parent done"
    assert child_start_metadata == {
        "prompt": "help",
        "tools": ["echo"],
        "metadata": {"conversation_id": "conv-1", "parent_call_id": "call_1"},
    }


def test_child_harness_tool_surfaces_follow_subagent_policy(tmp_path: Path) -> None:
    parent_echo = echo_tool()
    explicit_tool = {
        "name": "explicit",
        "description": "Explicit sequential tool",
        "parameters": {"type": "object", "properties": {}},
        "handler": lambda args: "ok",
        "sequential": True,
    }
    parent = Harness(HarnessConfig(root=tmp_path, builtin_tools=[]), model=ScriptedModel([]), tools=[parent_echo])
    parent.add_tool(create_subagent_tool(parent, []))

    default_child = build_child_harness(parent, None)
    explicit_child = build_child_harness(parent, SubAgentConfig(name="special", description="Special helper.", tools=[explicit_tool]))

    assert default_child.tools == [parent_echo]
    assert default_child.config.subagents == []
    assert [tool.name for tool in explicit_child.tools] == ["explicit"]
    assert explicit_child.tools[0].sequential is True
    assert explicit_child.config.subagents == []
    assert build_child_harness(parent, None).model is parent.model


def test_named_inherited_subagent_gets_parent_tools_without_subagent(tmp_path: Path) -> None:
    parent_echo = echo_tool()
    parent = Harness(HarnessConfig(root=tmp_path, builtin_tools=[]), model=ScriptedModel([]), tools=[parent_echo])
    parent.add_tool(create_subagent_tool(parent, []))

    child = build_child_harness(parent, SubAgentConfig(name="general", description="General helper.", inherit_parent_tools=True))

    assert child.tools == [parent_echo]
    assert child.skills is parent.skills
    assert child.config.subagents == []


def test_inherited_subagent_reuses_parent_skill_registry(tmp_path: Path) -> None:
    skill = tmp_path / "skills" / "demo"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: demo\ndescription: Demo skill\n---\nDemo body", encoding="utf-8")
    parent = Harness(
        HarnessConfig(root=tmp_path, skills_dir=tmp_path / "skills", builtin_tools=["skill_read"]),
        model=ScriptedModel([]),
    )

    child = build_child_harness(parent, SubAgentConfig(name="general", description="General helper.", inherit_parent_tools=True))

    assert child.skills is parent.skills
    assert "demo - Demo skill" in child.system_instructions()
    skill_read = next(tool for tool in child.tools if tool.name == "skill_read")
    assert skill_read.handler.__self__ is parent.skills


def test_explicit_subagent_skill_tools_use_parent_skill_config(tmp_path: Path) -> None:
    skill = tmp_path / "skills" / "demo"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: demo\ndescription: Demo skill\n---\nDemo body", encoding="utf-8")
    parent = Harness(
        HarnessConfig(root=tmp_path, skills_dir=tmp_path / "skills", builtin_tools=["skill_read"]),
        model=ScriptedModel([]),
    )

    child = build_child_harness(parent, SubAgentConfig(name="skilled", description="Skill helper.", builtin_tools=["skill_read"]))

    assert child.skills is not parent.skills
    assert [tool.name for tool in child.tools] == ["skill_read"]
    assert "demo - Demo skill" in child.system_instructions()


def test_subagent_model_override_credential_forwarding(tmp_path: Path, monkeypatch) -> None:
    calls = []

    def fake_infer_model(model_ref, **kwargs):
        calls.append((model_ref, kwargs))
        return ScriptedModel([])

    monkeypatch.setattr("thinharness.subagents.infer_model", fake_infer_model)
    parent = Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[], model="openai:parent", api_key="parent-key", base_url="https://parent.example"),
        model=ScriptedModel([]),
    )
    same_provider = SubAgentConfig(name="same", description="Same provider.", model="openai:child", tools=[echo_tool()])
    other_provider = SubAgentConfig(name="other", description="Other provider.", model="anthropic:child", tools=[echo_tool()])

    same_child = build_child_harness(parent, same_provider)
    other_child = build_child_harness(parent, other_provider)

    assert same_child.config.model == "openai:child"
    assert other_child.config.model == "anthropic:child"
    assert calls[0][1]["api_key"] == "parent-key"
    assert calls[0][1]["base_url"] == "https://parent.example"
    assert calls[1][1]["api_key"] is None
    assert calls[1][1]["base_url"] is None


def test_subagent_model_override_is_used_for_child_run(tmp_path: Path, monkeypatch) -> None:
    child_model = RecordingModel([ScriptedSession(start_turn=ModelTurn(text="child done", raw={"id": "child"}))], model="child-model")
    parent_model = RecordingModel([], model="parent-model")

    def fake_infer_model(_model_ref, **_kwargs):
        return child_model

    monkeypatch.setattr("thinharness.subagents.infer_model", fake_infer_model)
    parent = Harness(HarnessConfig(root=tmp_path, builtin_tools=[]), model=parent_model)
    child = build_child_harness(parent, SubAgentConfig(name="special", description="Special helper.", model="openai:child", tools=[echo_tool()]))

    assert child.model is child_model
    assert child.run("delegate").text == "child done"
    assert child_model.session_requests == 1
    assert parent_model.session_requests == 0


def test_subagent_child_provider_failure_returns_tool_error(tmp_path: Path) -> None:
    parent_call = ModelTurn(
        tool_calls=[ModelToolCall(id="call_1", name="subagent", arguments='{"task":"help"}')],
        raw={"id": "parent-start"},
    )

    def on_parent_continue(outputs, _tools, _metadata):
        envelope = tool_output(outputs[0].output)
        assert envelope["ok"] is False
        assert envelope["metadata"]["agent"] == "default"
        assert envelope["metadata"]["inherited"] is True
        assert envelope["metadata"]["tool_mode"] == "inherited"
        assert envelope["metadata"]["tools"] == ["echo"]
        assert envelope["metadata"]["error_type"] == "HarnessError"

    parent = ScriptedSession(start_turn=parent_call, continue_turn=ModelTurn(text="parent done", raw={"id": "parent-done"}), on_continue=on_parent_continue)
    tracer = FakeTracer()
    harness = Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[]),
        model=ScriptedModel([parent, FailingSession()]),
        tools=[echo_tool()],
        tracing=TracingOptions(tracer=tracer),
    )
    harness.add_tool(create_subagent_tool(harness, []))

    assert harness.run("delegate").text == "parent done"
    subagent_tool = next(span for span in tracer.spans if span.name == "execute_tool subagent")
    assert subagent_tool.attributes["subagent.name"] == "default"
    assert subagent_tool.attributes["subagent.tool_mode"] == "inherited"
    assert subagent_tool.attributes["subagent.tools"] == ["echo"]
    assert subagent_tool.status is not None


def test_subagent_runs_with_tracing_disabled(tmp_path: Path) -> None:
    parent_call = ModelTurn(
        tool_calls=[ModelToolCall(id="call_1", name="subagent", arguments='{"task":"help"}')],
        raw={"id": "parent-start"},
    )
    child = ScriptedSession(start_turn=ModelTurn(text="child done", raw={"id": "child"}))
    parent = ScriptedSession(start_turn=parent_call, continue_turn=ModelTurn(text="parent done", raw={"id": "parent-done"}))
    harness = Harness(HarnessConfig(root=tmp_path, builtin_tools=[]), model=ScriptedModel([parent, child]), tools=[echo_tool()])
    harness.add_tool(create_subagent_tool(harness, []))

    assert build_child_harness(harness, None).tracing is None
    assert harness.run("delegate").text == "parent done"


def test_concurrent_subagent_fanout_keeps_each_child_under_own_tool_span(tmp_path: Path) -> None:
    parent_call = ModelTurn(
        tool_calls=[
            ModelToolCall(id="call_1", name="subagent", arguments='{"task":"first"}'),
            ModelToolCall(id="call_2", name="subagent", arguments='{"task":"second"}'),
        ],
        raw={"id": "parent-start"},
    )
    child_a = ScriptedSession(start_turn=ModelTurn(text="child a", raw={"id": "child-a"}))
    child_b = ScriptedSession(start_turn=ModelTurn(text="child b", raw={"id": "child-b"}))
    parent = ScriptedSession(start_turn=parent_call, continue_turn=ModelTurn(text="parent done", raw={"id": "parent-done"}))
    tracer = ContextFakeTracer()
    harness = Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[]),
        model=ScriptedModel([parent, child_a, child_b]),
        tools=[echo_tool()],
        tracing=TracingOptions(tracer=tracer),
    )
    harness.add_tool(create_subagent_tool(harness, []))

    assert harness.run("delegate").text == "parent done"

    root = next(span for span in tracer.spans if span.name == "invoke_agent thinharness")
    subagent_tools = [span for span in tracer.spans if span.name == "execute_tool subagent"]
    child_agents = [span for span in tracer.spans if span.name == "invoke_agent subagent.default"]
    child_model_spans = [span for span in tracer.spans if span.name == "chat scripted-model" and span.parent in child_agents]
    assert len(subagent_tools) == 2
    assert len(child_agents) == 2
    assert len(child_model_spans) == 2
    assert all(span.parent is root for span in subagent_tools)
    assert {id(span.parent) for span in child_agents} == {id(span) for span in subagent_tools}
    assert {id(span.parent) for span in child_model_spans} == {id(span) for span in child_agents}


def test_unknown_named_subagent_returns_structured_error(tmp_path: Path) -> None:
    harness = Harness(HarnessConfig(root=tmp_path, builtin_tools=[]), model=ScriptedModel([]))
    tool = create_subagent_tool(harness, [SubAgentConfig(name="research", description="Research helper.", builtin_tools=["read"])])

    output = tool_output(tool.handler(tool.parse_args({"task": "x", "agent": "missing"})).as_json())

    assert output["ok"] is False
    assert output["metadata"]["available"] == ["research"]
    assert output["metadata"]["error_type"] == "UnknownSubAgent"


def test_unknown_named_subagent_trace_marks_failed_without_child_tool_mode(tmp_path: Path) -> None:
    parent_call = ModelTurn(
        tool_calls=[ModelToolCall(id="call_1", name="subagent", arguments='{"task":"help","agent":"missing"}')],
        raw={"id": "parent-start"},
    )

    def on_parent_continue(outputs, _tools, _metadata):
        envelope = tool_output(outputs[0].output)
        assert envelope["ok"] is False
        assert envelope["metadata"]["agent"] == "missing"
        assert envelope["metadata"]["error_type"] == "UnknownSubAgent"
        assert "tool_mode" not in envelope["metadata"]
        assert "tools" not in envelope["metadata"]

    tracer = FakeTracer()
    parent = ScriptedSession(start_turn=parent_call, continue_turn=ModelTurn(text="parent done", raw={"id": "parent-done"}), on_continue=on_parent_continue)
    harness = Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[]),
        model=ScriptedModel([parent]),
        tracing=TracingOptions(tracer=tracer),
    )
    harness.add_tool(create_subagent_tool(harness, [SubAgentConfig(name="research", description="Research helper.", builtin_tools=["read"])]))

    assert harness.run("delegate").text == "parent done"
    subagent_tool = next(span for span in tracer.spans if span.name == "execute_tool subagent")
    assert subagent_tool.attributes["subagent.name"] == "missing"
    assert "subagent.tool_mode" not in subagent_tool.attributes
    assert "subagent.tools" not in subagent_tool.attributes
    assert subagent_tool.status is not None


def test_blank_subagent_name_is_normal_argument_validation_error(tmp_path: Path) -> None:
    harness = Harness(HarnessConfig(root=tmp_path, builtin_tools=[]), model=ScriptedModel([]))
    tool = create_subagent_tool(harness, [])
    harness.add_tool(tool)

    output = tool_output(harness._call_output(tool.name, '{"task":"x","agent":""}'))

    assert output["ok"] is False
    assert "invalid arguments" in output["content"]


def test_duplicate_tool_names_are_rejected(tmp_path: Path) -> None:
    duplicate = ToolSpec("read", "Duplicate read", {"type": "object", "properties": {}}, lambda args: "ok")
    with pytest.raises(ValueError, match="duplicate tool name: read"):
        Harness(HarnessConfig(root=tmp_path), client=FakeClient(), tools=[duplicate])


def test_subagent_tool_name_is_reserved_for_custom_tools(tmp_path: Path) -> None:
    custom = ToolSpec("subagent", "Not the framework tool.", {"type": "object", "properties": {}}, lambda args: "bad")
    harness = Harness(HarnessConfig(root=tmp_path, builtin_tools=[]), model=ScriptedModel([]))

    with pytest.raises(ValueError, match="reserved tool name"):
        Harness(HarnessConfig(root=tmp_path, builtin_tools=[]), model=ScriptedModel([]), tools=[custom])
    with pytest.raises(ValueError, match="reserved tool name"):
        harness.add_tool(custom)


def test_framework_subagent_tool_can_be_added_after_construction(tmp_path: Path) -> None:
    harness = Harness(HarnessConfig(root=tmp_path, builtin_tools=[]), model=ScriptedModel([]))

    harness.add_tool(create_subagent_tool(harness, []))

    assert [tool["name"] for tool in harness.tool_schemas()] == ["subagent"]


def test_model_refs_require_provider_prefix() -> None:
    assert parse_model_ref("openai:gpt-4.1-mini") == ("openai", "gpt-4.1-mini")
    assert parse_model_ref("anthropic:claude-3-5-haiku-latest") == ("anthropic", "claude-3-5-haiku-latest")
    with pytest.raises(ValueError):
        parse_model_ref("gpt-4.1-mini")


def test_model_sessions_advance_independently() -> None:
    provider = FakeAnthropicProvider()
    model = AnthropicMessagesModel("claude-test", provider=provider)
    tools = [{"type": "function", "name": "echo", "description": "Echo", "parameters": {"type": "object", "properties": {}}}]
    first = model.new_session()
    second = model.new_session()

    first_turn = first.start(prompt="first", instructions="system", tools=tools)
    second_turn = second.start(prompt="second", instructions="system", tools=tools)
    first.continue_with_tools([ToolOutput(first_turn.tool_calls[0].id, "first result")], tools=tools)
    second.continue_with_tools([ToolOutput(second_turn.tool_calls[0].id, "second result")], tools=tools)

    assert provider.payloads[2]["messages"][0] == {"role": "user", "content": "first"}
    assert provider.payloads[2]["messages"][-1]["content"][0]["content"] == "first result"
    assert provider.payloads[3]["messages"][0] == {"role": "user", "content": "second"}
    assert provider.payloads[3]["messages"][-1]["content"][0]["content"] == "second result"


def test_openai_previous_response_id_is_session_scoped() -> None:
    client = FakeClient()
    provider = OpenAIProvider(api_key="key", client=client)
    model = OpenAIResponsesModel("gpt-test", provider=provider)
    tools = [{"type": "function", "name": "echo", "description": "Echo", "parameters": {"type": "object", "properties": {}}}]
    first = model.new_session()
    second = model.new_session()

    first.start(prompt="first", instructions="system", tools=tools, previous_response_id="existing")
    first.continue_with_tools([ToolOutput("call_1", "ok")], tools=tools)
    second.start(prompt="second", instructions="system", tools=tools)

    assert client.payloads[0]["previous_response_id"] == "existing"
    assert client.payloads[1]["previous_response_id"] == "resp_1"
    assert "previous_response_id" not in client.payloads[2]


def test_anthropic_provider_model_tool_loop(monkeypatch) -> None:
    calls = []

    def fake_post(url, payload, headers, timeout):
        calls.append((url, payload, headers, timeout))
        if len(calls) == 1:
            return {
                "content": [{"type": "tool_use", "id": "toolu_1", "name": "echo", "input": {"value": "hi"}}],
                "stop_reason": "tool_use",
            }
        assert payload["messages"][-1]["content"][0]["type"] == "tool_result"
        return {"content": [{"type": "text", "text": "done"}], "stop_reason": "end_turn"}

    monkeypatch.setattr("thinharness.providers._post_json", fake_post)
    provider = AnthropicProvider(api_key="key")
    model = AnthropicMessagesModel("claude-test", provider=provider)
    session = model.new_session()
    tools = [{"type": "function", "name": "echo", "description": "Echo", "parameters": {"type": "object", "properties": {}}}]

    first = session.start(prompt="hi", instructions="system", tools=tools)
    assert first.tool_calls[0].name == "echo"
    second = session.continue_with_tools([ToolOutput(first.tool_calls[0].id, "ok")], tools=tools)
    assert second.text == "done"
    assert calls[0][1]["tools"][0]["input_schema"]["type"] == "object"


def test_openrouter_provider_model_tool_loop(monkeypatch) -> None:
    calls = []

    def fake_post(url, payload, headers, timeout):
        calls.append((url, payload, headers, timeout))
        if len(calls) == 1:
            return {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "echo", "arguments": '{"value":"hi"}'},
                        }],
                    }
                }]
            }
        assert payload["messages"][-1]["role"] == "tool"
        return {"choices": [{"message": {"role": "assistant", "content": "done"}}]}

    monkeypatch.setattr("thinharness.providers._post_json", fake_post)
    provider = OpenRouterProvider(api_key="key")
    model = OpenRouterModel("openai/test", provider=provider)
    session = model.new_session()
    tools = [{"type": "function", "name": "echo", "description": "Echo", "parameters": {"type": "object", "properties": {}}}]

    first = session.start(prompt="hi", instructions="system", tools=tools)
    assert first.tool_calls[0].id == "call_1"
    second = session.continue_with_tools([ToolOutput("call_1", "ok")], tools=tools)
    assert second.text == "done"
    assert calls[0][1]["tools"][0]["function"]["name"] == "echo"


def test_provider_wraps_transport_errors(monkeypatch) -> None:
    def fail_urlopen(request, timeout):
        raise urllib.error.URLError("offline")

    monkeypatch.setattr("urllib.request.urlopen", fail_urlopen)
    provider = OpenAIProvider(api_key="key", base_url="http://example.invalid")
    with pytest.raises(ProviderError, match="provider request failed"):
        provider.post_json("/responses", {})


def test_provider_wraps_invalid_json(monkeypatch) -> None:
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def read(self):
            return b"not json"

    def fake_urlopen(request, timeout):
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    provider = OpenAIProvider(api_key="key", base_url="http://example.invalid")
    with pytest.raises(ProviderError, match="invalid JSON"):
        provider.post_json("/responses", {})


class MultiCallClient:
    """Fake Responses client that emits a configured batch of tool calls once, then finishes."""

    def __init__(self, calls):
        self.api_key = "fake"
        self.calls_to_emit = calls
        self.payloads = []
        self.invocations = 0

    def create(self, payload):
        self.invocations += 1
        self.payloads.append(payload)
        if self.invocations == 1:
            return {
                "id": "resp_1",
                "output": [
                    {"type": "function_call", "call_id": f"call_{i}", "name": name, "arguments": args}
                    for i, (name, args) in enumerate(self.calls_to_emit, start=1)
                ],
            }
        return {"id": "resp_2", "output_text": "done"}


def slow_tool(name: str, delay: float, *, sequential: bool = False) -> ToolSpec:
    """Create a tool that sleeps for delay seconds and echoes its name."""
    return ToolSpec(
        name,
        f"Sleeps {delay}s and returns its name.",
        {"type": "object", "properties": {}, "additionalProperties": False},
        lambda args: (time.sleep(delay), name)[1],
        sequential=sequential,
    )


def test_harness_config_defaults_to_auto_tool_execution() -> None:
    assert HarnessConfig().tool_execution == "auto"
    assert HarnessConfig(tool_execution="sequential").tool_execution == "sequential"


def test_tool_spec_sequential_default_and_not_in_schema() -> None:
    spec = ToolSpec("echo", "Echo", {"type": "object", "properties": {}}, lambda args: "ok")
    assert spec.sequential is False
    assert "sequential" not in spec.response_tool()
    flagged = ToolSpec("write_thing", "writes", {"type": "object", "properties": {}}, lambda args: "ok", sequential=True)
    assert flagged.sequential is True
    assert "sequential" not in flagged.response_tool()


def test_builtin_tools_mark_mutating_specs_sequential(tmp_path: Path) -> None:
    by_name = {spec.name: spec for spec in FileTools(tmp_path).specs()}
    assert by_name["read"].sequential is False
    assert by_name["search"].sequential is False
    assert by_name["list"].sequential is False
    assert by_name["glob"].sequential is False
    assert by_name["jsonl_search"].sequential is False
    assert by_name["write"].sequential is True
    assert by_name["edit"].sequential is True


def test_parallel_safe_batch_runs_concurrently(tmp_path: Path) -> None:
    delay = 0.2
    client = MultiCallClient([("slow_a", "{}"), ("slow_b", "{}")])
    harness = Harness(
        HarnessConfig(root=tmp_path, model="openai:test-model", builtin_tools=[]),
        client=client,
        tools=[slow_tool("slow_a", delay), slow_tool("slow_b", delay)],
    )

    start = time.monotonic()
    result = harness.run("go")
    elapsed = time.monotonic() - start

    assert result.text == "done"
    assert elapsed < delay * 1.8, f"expected concurrent execution, elapsed={elapsed:.3f}s"
    assert len(client.payloads) == 2
    continuation_inputs = client.payloads[1]["input"]
    assert [item["call_id"] for item in continuation_inputs] == ["call_1", "call_2"]
    assert [tool_output(item["output"])["content"] for item in continuation_inputs] == ["slow_a", "slow_b"]


def test_sequential_tool_forces_serial_batch(tmp_path: Path) -> None:
    delay = 0.2
    client = MultiCallClient([("slow_a", "{}"), ("slow_b", "{}")])
    harness = Harness(
        HarnessConfig(root=tmp_path, model="openai:test-model", builtin_tools=[]),
        client=client,
        tools=[slow_tool("slow_a", delay), slow_tool("slow_b", delay, sequential=True)],
    )

    start = time.monotonic()
    result = harness.run("go")
    elapsed = time.monotonic() - start

    assert result.text == "done"
    assert elapsed >= delay * 1.9, f"expected serial execution, elapsed={elapsed:.3f}s"
    continuation_inputs = client.payloads[1]["input"]
    assert [item["call_id"] for item in continuation_inputs] == ["call_1", "call_2"]


def test_tool_execution_sequential_forces_serial_even_for_safe_tools(tmp_path: Path) -> None:
    delay = 0.15
    client = MultiCallClient([("slow_a", "{}"), ("slow_b", "{}")])
    harness = Harness(
        HarnessConfig(root=tmp_path, model="openai:test-model", builtin_tools=[], tool_execution="sequential"),
        client=client,
        tools=[slow_tool("slow_a", delay), slow_tool("slow_b", delay)],
    )

    start = time.monotonic()
    harness.run("go")
    elapsed = time.monotonic() - start

    assert elapsed >= delay * 1.9


def test_parallel_batch_preserves_model_call_order(tmp_path: Path) -> None:
    client = MultiCallClient([("slow_first", "{}"), ("fast_second", "{}")])
    harness = Harness(
        HarnessConfig(root=tmp_path, model="openai:test-model", builtin_tools=[]),
        client=client,
        tools=[slow_tool("slow_first", 0.2), slow_tool("fast_second", 0.01)],
    )

    harness.run("go")

    continuation_inputs = client.payloads[1]["input"]
    assert [item["call_id"] for item in continuation_inputs] == ["call_1", "call_2"]
    assert [tool_output(item["output"])["content"] for item in continuation_inputs] == ["slow_first", "fast_second"]


def test_parallel_batch_continues_when_one_tool_errors(tmp_path: Path) -> None:
    client = MultiCallClient([("boom", "{}"), ("ok", "{}")])

    def boom(_args):
        raise RuntimeError("nope")

    boom_spec = ToolSpec("boom", "Always raises.", {"type": "object", "properties": {}}, boom)
    ok_spec = ToolSpec("ok", "Returns ok.", {"type": "object", "properties": {}}, lambda args: "ok")
    harness = Harness(
        HarnessConfig(root=tmp_path, model="openai:test-model", builtin_tools=[]),
        client=client,
        tools=[boom_spec, ok_spec],
    )

    result = harness.run("go")

    assert result.text == "done"
    continuation_inputs = client.payloads[1]["input"]
    assert continuation_inputs[0]["call_id"] == "call_1"
    assert "RuntimeError" in continuation_inputs[0]["output"]
    assert continuation_inputs[1]["call_id"] == "call_2"
    assert tool_output(continuation_inputs[1]["output"])["content"] == "ok"


def test_parallel_batch_makes_one_provider_continuation(tmp_path: Path) -> None:
    client = MultiCallClient([("a", "{}"), ("b", "{}"), ("c", "{}")])
    harness = Harness(
        HarnessConfig(root=tmp_path, model="openai:test-model", builtin_tools=[]),
        client=client,
        tools=[slow_tool("a", 0.01), slow_tool("b", 0.01), slow_tool("c", 0.01)],
    )

    harness.run("go")

    assert client.invocations == 2
    assert len(client.payloads[1]["input"]) == 3


def test_truncate_spill_files_do_not_collide_under_parallel_reads(tmp_path: Path) -> None:
    big = "x" * 200 + "\n" + "y" * 200 + "\n"
    (tmp_path / "a.txt").write_text(big, encoding="utf-8")
    (tmp_path / "b.txt").write_text(big, encoding="utf-8")
    client = MultiCallClient([("read", '{"path":"a.txt","max_chars":50}'), ("read", '{"path":"b.txt","max_chars":50}')])
    harness = Harness(
        HarnessConfig(root=tmp_path, model="openai:test-model", max_tool_chars=50, max_read_chars=50),
        client=client,
    )

    harness.run("go")

    saved_paths = []
    for item in client.payloads[1]["input"]:
        body = json.loads(item["output"])
        assert body["metadata"]["truncated"] is True
        saved_paths.append(body["metadata"]["saved_to"])
    assert saved_paths[0] != saved_paths[1]
    assert all(Path(path).exists() for path in saved_paths)


def test_dict_tool_can_opt_into_sequential(tmp_path: Path) -> None:
    delay = 0.15
    sequential_dict_tool = {
        "name": "slow_b",
        "description": "Slow, sequential",
        "parameters": {"type": "object", "properties": {}},
        "handler": lambda args: (time.sleep(delay), "slow_b")[1],
        "sequential": True,
    }
    client = MultiCallClient([("slow_a", "{}"), ("slow_b", "{}")])
    harness = Harness(
        HarnessConfig(root=tmp_path, model="openai:test-model", builtin_tools=[]),
        client=client,
        tools=[slow_tool("slow_a", delay), sequential_dict_tool],
    )

    start = time.monotonic()
    harness.run("go")
    elapsed = time.monotonic() - start

    assert elapsed >= delay * 1.9, f"expected serial execution, elapsed={elapsed:.3f}s"


def test_parallel_batch_with_more_calls_than_worker_cap(tmp_path: Path) -> None:
    batch = [(f"t{i}", "{}") for i in range(20)]
    client = MultiCallClient(batch)
    tools = [slow_tool(name, 0.01) for name, _ in batch]
    harness = Harness(
        HarnessConfig(root=tmp_path, model="openai:test-model", builtin_tools=[]),
        client=client,
        tools=tools,
    )

    harness.run("go")

    continuation_inputs = client.payloads[1]["input"]
    assert [item["call_id"] for item in continuation_inputs] == [f"call_{i+1}" for i in range(20)]
    assert [tool_output(item["output"])["content"] for item in continuation_inputs] == [name for name, _ in batch]


def test_parallel_batch_tools_execute_in_separate_threads(tmp_path: Path) -> None:
    client = MultiCallClient([("track_a", "{}"), ("track_b", "{}")])
    seen_threads: list[int] = []
    barrier = threading.Barrier(2, timeout=2)

    def run_a(_args):
        seen_threads.append(threading.get_ident())
        barrier.wait()
        return "a"

    def run_b(_args):
        seen_threads.append(threading.get_ident())
        barrier.wait()
        return "b"

    harness = Harness(
        HarnessConfig(root=tmp_path, model="openai:test-model", builtin_tools=[]),
        client=client,
        tools=[
            ToolSpec("track_a", "a", {"type": "object", "properties": {}}, run_a),
            ToolSpec("track_b", "b", {"type": "object", "properties": {}}, run_b),
        ],
    )

    harness.run("go")

    assert len(seen_threads) == 2
    assert seen_threads[0] != seen_threads[1]
