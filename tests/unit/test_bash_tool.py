from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from fakes import MultiCallClient, ScriptedModel, ScriptedSession, _fake_openai, slow_tool, tool_output

from thinharness import BashArgs, BashTool, Harness, HarnessConfig, SubAgentConfig, call_tool
from thinharness.providers import ModelToolCall, ModelTurn
from thinharness.subagents import build_child_harness


def test_bash_spec_exposes_expected_schema() -> None:
    spec = BashTool().spec()

    assert spec.name == "bash"
    assert spec.description == (
        "Run one bash command from a workspace-contained cwd. Intended for exploratory workflows; prefer typed tools for production."
    )
    assert spec.parameters is BashArgs
    assert spec.sequential is True
    schema = spec.response_tool()["parameters"]
    assert schema["additionalProperties"] is False
    assert schema["properties"]["timeout"]["minimum"] == 1
    assert schema["properties"]["timeout"]["maximum"] == 120


def test_successful_command_captures_output_and_metadata(tmp_path: Path) -> None:
    tool = BashTool(tmp_path)

    result = tool.run({"command": "printf out; printf err >&2"})

    assert result.ok is True
    assert "stdout:\nout" in result.content
    assert "stderr:\nerr" in result.content
    assert result.metadata["exit_code"] == 0
    assert result.metadata["timed_out"] is False
    assert result.metadata["cwd"] == str(tmp_path)
    assert result.metadata["stdout_truncated"] is False
    assert result.metadata["stderr_truncated"] is False
    assert isinstance(result.metadata["duration_seconds"], float)


def test_non_zero_command_is_tool_failure_with_stderr(tmp_path: Path) -> None:
    result = BashTool(tmp_path).run({"command": "printf nope >&2; exit 7"})

    assert result.ok is False
    assert "stderr:\nnope" in result.content
    assert result.metadata["exit_code"] == 7
    assert result.metadata["error_type"] == "NonZeroExit"


def test_timeout_terminates_command_group_and_returns_metadata(tmp_path: Path) -> None:
    result = BashTool(tmp_path).run({"command": "printf before; sleep 2", "timeout": 1})

    assert result.ok is False
    assert result.metadata["timed_out"] is True
    assert result.metadata["error_type"] == "Timeout"
    assert result.metadata["exit_code"] is not None
    assert "before" in result.content


def test_timeout_returns_when_command_ignores_sigterm(tmp_path: Path) -> None:
    start = time.monotonic()
    result = BashTool(tmp_path).run({"command": "trap '' TERM; printf before; sleep 30", "timeout": 1})
    elapsed = time.monotonic() - start

    assert result.ok is False
    assert result.metadata["timed_out"] is True
    assert elapsed < 5
    assert "before" in result.content


def test_cwd_cannot_escape_workspace(tmp_path: Path) -> None:
    result = BashTool(tmp_path).run({"command": "pwd", "cwd": ".."})

    assert result.ok is False
    assert result.metadata["error_type"] == "PathValidationError"


def test_nonexistent_and_file_cwd_fail_cleanly(tmp_path: Path) -> None:
    file_path = tmp_path / "file.txt"
    file_path.write_text("x", encoding="utf-8")

    missing = BashTool(tmp_path).run({"command": "pwd", "cwd": "missing"})
    file_result = BashTool(tmp_path).run({"command": "pwd", "cwd": "file.txt"})

    assert missing.ok is False
    assert missing.metadata["error_type"] == "PathNotFound"
    assert file_result.ok is False
    assert file_result.metadata["error_type"] == "NotADirectory"


def test_output_truncation_sets_metadata(tmp_path: Path) -> None:
    result = BashTool(tmp_path, max_tool_chars=5).run({"command": "printf 1234567890; printf abcdefghij >&2"})

    assert result.ok is True
    assert "...[truncated]"[:5] in result.content
    assert result.metadata["stdout_truncated"] is True
    assert result.metadata["stderr_truncated"] is True


def test_per_call_max_chars_can_lower_but_not_raise_constructor_cap(tmp_path: Path) -> None:
    lower = BashTool(tmp_path, max_tool_chars=100).run({"command": "printf 1234567890", "max_chars": 5})
    clamped = BashTool(tmp_path, max_tool_chars=5).run({"command": "printf 1234567890", "max_chars": 100})

    assert lower.metadata["stdout_truncated"] is True
    assert "12345" not in lower.content
    assert clamped.metadata["stdout_truncated"] is True
    assert "12345" not in clamped.content


def test_invalid_bash_args_return_retry_envelope(tmp_path: Path) -> None:
    output = json.loads(call_tool(BashTool(tmp_path).spec(), "{}"))

    assert output["ok"] is False
    assert output["metadata"]["error_type"] == "ValidationError"
    assert output["metadata"]["retry"] is True


def test_bash_is_available_through_explicit_custom_registration(tmp_path: Path) -> None:
    call = ModelTurn(
        tool_calls=[ModelToolCall(id="call_1", name="bash", arguments='{"command":"printf custom"}')],
        raw={"id": "start"},
    )
    captured = {}

    def on_continue(outputs, _tools, _metadata) -> None:
        captured["output"] = tool_output(outputs[0].output)

    session = ScriptedSession(start_turn=call, continue_turn=ModelTurn(text="done", raw={"id": "done"}), on_continue=on_continue)
    harness = Harness(
        HarnessConfig(root=tmp_path, builtin_tools=[]),
        model=ScriptedModel([session]),
        tools=[BashTool(tmp_path).spec()],
    )

    assert [tool["name"] for tool in harness.tool_schemas()] == ["bash"]
    assert harness.run_sync("go").text == "done"
    assert captured["output"]["ok"] is True
    assert "custom" in captured["output"]["content"]


def test_bash_is_not_a_builtin_tool(tmp_path: Path) -> None:
    default = Harness(HarnessConfig(root=tmp_path), model=ScriptedModel([]))

    assert "bash" not in [tool["name"] for tool in default.tool_schemas()]
    with pytest.raises(ValueError, match="unknown builtin tool: bash"):
        Harness(HarnessConfig(root=tmp_path, builtin_tools=["bash"]), model=ScriptedModel([]))


def test_named_subagent_cannot_opt_into_bash_as_builtin(tmp_path: Path) -> None:
    parent = Harness(HarnessConfig(root=tmp_path, builtin_tools=[]), model=ScriptedModel([]))
    config = SubAgentConfig(name="shell", description="Shell helper.", builtin_tools=["bash"])

    with pytest.raises(ValueError, match="unknown builtin tool: bash"):
        build_child_harness(parent, config)


def test_mixed_batch_containing_bash_runs_sequentially(tmp_path: Path) -> None:
    client = MultiCallClient([("bash", '{"command":"sleep 0.2; printf bash"}'), ("slow", "{}")])
    harness = Harness(
        HarnessConfig(root=tmp_path, model="openai:test-model", builtin_tools=[]),
        model=_fake_openai(client),
        tools=[BashTool(tmp_path).spec(), slow_tool("slow", 0.2)],
    )

    start = time.monotonic()
    harness.run_sync("go")
    elapsed = time.monotonic() - start

    assert elapsed >= 0.38
    continuation_inputs = client.payloads[1]["input"]
    assert [item["call_id"] for item in continuation_inputs] == ["call_1", "call_2"]
