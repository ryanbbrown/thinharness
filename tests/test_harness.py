from __future__ import annotations

import json
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
    tools = [{"type": "function", "name": "echo", "description": "Echo", "parameters": {"type": "object", "properties": {}}}]

    first = model.start(prompt="hi", instructions="system", tools=tools)
    assert first.tool_calls[0].name == "echo"
    second = model.continue_with_tools([ToolOutput(first.tool_calls[0].id, "ok")], tools=tools)
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
    tools = [{"type": "function", "name": "echo", "description": "Echo", "parameters": {"type": "object", "properties": {}}}]

    first = model.start(prompt="hi", instructions="system", tools=tools)
    assert first.tool_calls[0].id == "call_1"
    second = model.continue_with_tools([ToolOutput("call_1", "ok")], tools=tools)
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
