from __future__ import annotations

from pathlib import Path

import pytest

from filesystem_harness import (
    AnthropicMessagesModel,
    AnthropicProvider,
    Harness,
    HarnessConfig,
    OpenRouterModel,
    OpenRouterProvider,
    SkillRegistry,
    ToolSpec,
    parse_model_ref,
)
from filesystem_harness.providers import ToolOutput
from filesystem_harness.tools import FileTools


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

    monkeypatch.setattr("filesystem_harness.providers._post_json", fake_post)
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

    monkeypatch.setattr("filesystem_harness.providers._post_json", fake_post)
    provider = OpenRouterProvider(api_key="key")
    model = OpenRouterModel("openai/test", provider=provider)
    tools = [{"type": "function", "name": "echo", "description": "Echo", "parameters": {"type": "object", "properties": {}}}]

    first = model.start(prompt="hi", instructions="system", tools=tools)
    assert first.tool_calls[0].id == "call_1"
    second = model.continue_with_tools([ToolOutput("call_1", "ok")], tools=tools)
    assert second.text == "done"
    assert calls[0][1]["tools"][0]["function"]["name"] == "echo"
