from __future__ import annotations

import copy
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
    ToolSpec,
    TracingOptions,
    parse_model_ref,
)
from thinharness.providers import ProviderError, ToolOutput
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


def echo_tool() -> ToolSpec:
    """Create a custom echo tool."""
    return ToolSpec(
        "echo",
        "Echo input",
        {"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]},
        lambda args: args["value"],
    )


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
    with pytest.raises(ValueError):
        tools.read({"path": "../outside.txt"})
    with pytest.raises(ValueError):
        tools.write({"path": "/tmp/outside.txt", "content": "no"})


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


def test_custom_tool_is_exposed_and_callable(tmp_path: Path) -> None:
    custom = ToolSpec(
        "echo_json",
        "Echo input",
        {"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]},
        lambda args: {"echo": args["value"]},
    )
    harness = Harness(HarnessConfig(root=tmp_path), client=FakeClient(), tools=[custom])
    assert any(tool["name"] == "echo_json" for tool in harness.tool_schemas())
    output = harness._call_output("echo_json", '{"value":"ok"}')
    assert '"echo": "ok"' in output


def test_custom_tool_can_use_pydantic_args_model(tmp_path: Path) -> None:
    class EchoArgs(BaseModel):
        value: str
        count: int = Field(default=1, ge=1)

    custom = ToolSpec("echo_typed", "Echo typed input", EchoArgs, lambda args: {"echo": args.value * args.count})
    harness = Harness(HarnessConfig(root=tmp_path), client=FakeClient(), tools=[custom])

    schema = next(tool for tool in harness.tool_schemas() if tool["name"] == "echo_typed")["parameters"]
    assert schema["properties"]["count"]["minimum"] == 1
    assert '"echo": "okok"' in harness._call_output("echo_typed", '{"value":"ok","count":2}')
    assert "invalid arguments" in harness._call_output("echo_typed", '{"value":"ok","count":0}')


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

    with pytest.raises(ValueError, match="skill_read or skill_run"):
        Harness(HarnessConfig(root=tmp_path, skills_dir=tmp_path / "skills"), client=FakeClient())
    with pytest.raises(ValueError, match="skill_read or skill_run"):
        Harness(HarnessConfig(root=tmp_path, skills_dir=tmp_path / "skills", builtin_tools=["read"]), client=FakeClient())


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


def test_duplicate_tool_names_are_rejected(tmp_path: Path) -> None:
    duplicate = ToolSpec("read", "Duplicate read", {"type": "object", "properties": {}}, lambda args: "ok")
    with pytest.raises(ValueError, match="duplicate tool name: read"):
        Harness(HarnessConfig(root=tmp_path), client=FakeClient(), tools=[duplicate])


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
    assert [item["output"] for item in continuation_inputs] == ["slow_a", "slow_b"]


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
    assert [item["output"] for item in continuation_inputs] == ["slow_first", "fast_second"]


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
    assert continuation_inputs[1]["output"] == "ok"


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
    assert [item["output"] for item in continuation_inputs] == [name for name, _ in batch]


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
