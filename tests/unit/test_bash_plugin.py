from __future__ import annotations

import asyncio
import json
import os
import shlex
import signal
import sys
import time
from pathlib import Path
from typing import Any

import pytest
from fakes import ContextFakeTracer, MultiCallClient, ScriptedModel, ScriptedSession, _fake_openai, slow_tool

import thinharness
import thinharness.plugins.bash as bash_module
import thinharness.tools as tools_module
from thinharness import (
    ApprovalDecision,
    BashPlugin,
    Harness,
    HarnessConfig,
    Hook,
    ModelToolCall,
    ModelTurn,
    SubAgentConfig,
    SubagentsPlugin,
    ToolOrigin,
    ToolResult,
    TracingOptions,
    call_tool,
)


def _call_turn(arguments: dict[str, Any], *, call_id: str = "call_1") -> ModelTurn:
    return ModelTurn(
        tool_calls=[ModelToolCall(id=call_id, name="bash", arguments=json.dumps(arguments))],
        raw={"id": "start"},
    )


async def _run_bash(
    root: Path,
    arguments: dict[str, Any],
    *,
    plugin: BashPlugin | None = None,
    hooks: list[Hook] | None = None,
    tracing: list[TracingOptions] | None = None,
) -> tuple[ToolResult, Any]:
    captured: list[ToolResult] = []

    def on_continue(outputs, _tools, _metadata) -> None:
        captured.append(ToolResult.from_json(outputs[0].output))

    session = ScriptedSession(
        start_turn=_call_turn(arguments),
        continue_turn=ModelTurn(text="done", raw={"id": "done"}),
        on_continue=on_continue,
    )
    harness = Harness(
        HarnessConfig(root=root),
        model=ScriptedModel([session]),
        plugins=[plugin or BashPlugin()],
        hooks=hooks,
        tracing=tracing,
    )
    run_result = await harness.run("go")
    assert len(captured) == 1
    return captured[0], run_result


def _stdout(result: ToolResult) -> str:
    if result.content == "(no output)":
        return ""
    return result.content.split("stdout:\n", 1)[1].split("\nstderr:\n", 1)[0]


def _stderr(result: ToolResult) -> str:
    if result.content == "(no output)":
        return ""
    return result.content.split("\nstderr:\n", 1)[1]


def _group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False


async def _wait_for_file(path: Path, timeout: float = 2) -> None:
    async with asyncio.timeout(timeout):
        while not path.exists():
            await asyncio.sleep(0.01)


def test_plugin_is_explicit_static_and_uses_fixed_identity(tmp_path: Path) -> None:
    plain = Harness(HarnessConfig(root=tmp_path), model=ScriptedModel([]))
    harness = Harness(HarnessConfig(root=tmp_path), model=ScriptedModel([]), plugins=[BashPlugin()])

    assert plain.tools == []
    assert [tool.name for tool in harness.tools] == ["bash"]
    tool = harness.tools[0]
    assert tool.sequential is True
    assert tool.origin == ToolOrigin(plugin="bash", source="bash")
    assert tool.requires_approval is False
    schema = tool.response_tool()["parameters"]
    assert set(schema["properties"]) == {"command", "cwd", "timeout"}
    assert schema["additionalProperties"] is False
    assert "non-interactive" in tool.description
    assert "does not share shell state" in tool.description
    assert "best-effort" in tool.description
    assert "capped by the host" in tool.description
    assert harness.system_instructions() == plain.system_instructions()


def test_binding_uses_canonical_root_without_creating_it(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    harness = Harness(HarnessConfig(root=missing), model=ScriptedModel([]), plugins=[BashPlugin()])

    assert not missing.exists()
    assert harness.tools[0].name == "bash"


def test_plugin_configuration_is_frozen_and_copies_environment() -> None:
    environment = {"HOST_VALUE": "before"}
    plugin = BashPlugin(env=environment)
    environment["HOST_VALUE"] = "after"

    assert plugin.env == {"HOST_VALUE": "before"}
    detached = plugin.env
    detached["HOST_VALUE"] = "mutated"
    assert plugin.env == {"HOST_VALUE": "before"}
    with pytest.raises(AttributeError, match="frozen"):
        plugin.max_output_bytes = 1
    with pytest.raises(AttributeError, match="fixed"):
        plugin.name = "other"
    with pytest.raises(AttributeError, match="fixed"):
        BashPlugin.name = "other"


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"default_timeout": True}, TypeError),
        ({"default_timeout": "1"}, TypeError),
        ({"default_timeout": float("inf")}, ValueError),
        ({"default_timeout": 10**400}, ValueError),
        ({"default_timeout": 0}, ValueError),
        ({"max_timeout": False}, TypeError),
        ({"max_timeout": float("nan")}, ValueError),
        ({"default_timeout": 2, "max_timeout": 1}, ValueError),
        ({"max_output_bytes": True}, TypeError),
        ({"max_output_bytes": 1.5}, TypeError),
        ({"max_output_bytes": 0}, ValueError),
        ({"inherit_env": 1}, TypeError),
        ({"requires_approval": "yes"}, TypeError),
        ({"env": []}, TypeError),
        ({"env": {1: "value"}}, TypeError),
        ({"env": {"NAME": 1}}, TypeError),
        ({"env": {"": "value"}}, ValueError),
        ({"env": {"A=B": "value"}}, ValueError),
        ({"env": {"A\0B": "value"}}, ValueError),
        ({"env": {"NAME": "a\0b"}}, ValueError),
    ],
)
def test_plugin_constructor_rejects_invalid_values(kwargs: dict[str, Any], error: type[Exception]) -> None:
    with pytest.raises(error):
        BashPlugin(**kwargs)


def test_duplicate_plugin_and_tool_collisions_use_normal_validation(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="duplicate plugin name: bash"):
        Harness(HarnessConfig(root=tmp_path), model=ScriptedModel([]), plugins=[BashPlugin(), BashPlugin()])
    with pytest.raises(ValueError, match="duplicate tool name: bash"):
        Harness(
            HarnessConfig(root=tmp_path),
            model=ScriptedModel([]),
            plugins=[BashPlugin()],
            tools=[slow_tool("bash", 0)],
        )


def test_old_public_bash_interfaces_are_removed() -> None:
    assert not hasattr(thinharness, "BashTool")
    assert not hasattr(thinharness, "BashArgs")
    assert not hasattr(tools_module, "BashTool")
    assert not hasattr(tools_module, "BashArgs")


async def test_success_keeps_streams_separate_and_reports_metadata(tmp_path: Path) -> None:
    result, _ = await _run_bash(tmp_path, {"command": "printf out; printf err >&2"})

    assert result.ok is True
    assert _stdout(result) == "out"
    assert _stderr(result) == "err"
    assert result.metadata["exit_code"] == 0
    assert result.metadata["timed_out"] is False
    assert result.metadata["cwd"] == str(tmp_path)
    assert result.metadata["timeout_seconds"] == 30
    assert result.metadata["stdout_bytes"] == 3
    assert result.metadata["stderr_bytes"] == 3
    assert result.metadata["stdout_truncated"] is False
    assert result.metadata["stderr_truncated"] is False
    assert isinstance(result.metadata["duration_seconds"], float)


async def test_one_empty_stream_keeps_both_exact_labels(tmp_path: Path) -> None:
    stdout_only, _ = await _run_bash(tmp_path, {"command": "printf out"})
    stderr_only, _ = await _run_bash(tmp_path, {"command": "printf err >&2"})

    assert stdout_only.content == "stdout:\nout\nstderr:\n(no output)"
    assert stderr_only.content == "stdout:\n(no output)\nstderr:\nerr"


async def test_nonzero_and_signal_exit_preserve_output(tmp_path: Path) -> None:
    failed, _ = await _run_bash(tmp_path, {"command": "printf before; printf nope >&2; exit 7"})
    signalled, _ = await _run_bash(tmp_path, {"command": "printf signal; kill -TERM $$"})

    assert failed.ok is False
    assert failed.metadata["error_type"] == "NonZeroExit"
    assert failed.metadata["exit_code"] == 7
    assert _stdout(failed) == "before"
    assert _stderr(failed) == "nope"
    assert signalled.ok is False
    assert signalled.metadata["error_type"] == "NonZeroExit"
    assert signalled.metadata["exit_code"] == -signal.SIGTERM
    assert signalled.metadata["signal"] == signal.SIGTERM
    assert _stdout(signalled) == "signal"


async def test_empty_output_uses_single_marker(tmp_path: Path) -> None:
    result, _ = await _run_bash(tmp_path, {"command": ":"})

    assert result.ok is True
    assert result.content == "(no output)"


async def test_timeout_keeps_partial_output_and_escalates_for_ignored_term(tmp_path: Path) -> None:
    started = time.monotonic()
    result, _ = await _run_bash(
        tmp_path,
        {"command": "trap '' TERM; printf before; while :; do sleep 1; done", "timeout": 0.1},
    )
    elapsed = time.monotonic() - started

    assert result.ok is False
    assert result.metadata["error_type"] == "Timeout"
    assert result.metadata["timed_out"] is True
    assert result.metadata["signal"] == signal.SIGKILL
    assert _stdout(result) == "before"
    assert 1 <= elapsed < 4


async def test_default_timeout_per_call_timeout_and_host_clamp(tmp_path: Path) -> None:
    plugin = BashPlugin(default_timeout=0.4, max_timeout=0.6)
    default, _ = await _run_bash(tmp_path, {"command": "printf default"}, plugin=plugin)
    requested, _ = await _run_bash(tmp_path, {"command": "printf requested", "timeout": 0.2}, plugin=plugin)
    clamped, _ = await _run_bash(tmp_path, {"command": "printf clamped", "timeout": 10}, plugin=plugin)

    assert default.metadata["timeout_seconds"] == 0.4
    assert requested.metadata["timeout_seconds"] == 0.2
    assert clamped.metadata["timeout_seconds"] == 0.6


async def test_pipe_setup_set_inheritable_failure_closes_both_descriptors(monkeypatch: pytest.MonkeyPatch) -> None:
    descriptors: list[int] = []
    real_pipe = os.pipe

    def tracked_pipe() -> tuple[int, int]:
        pair = real_pipe()
        descriptors.extend(pair)
        return pair

    def fail_inheritable(_fd: int, _inheritable: bool) -> None:
        raise OSError("set inheritable sentinel")

    monkeypatch.setattr(bash_module.os, "pipe", tracked_pipe)
    monkeypatch.setattr(bash_module.os, "set_inheritable", fail_inheritable)
    with pytest.raises(OSError, match="set inheritable sentinel"):
        await bash_module._open_owned_pipe()

    assert len(descriptors) == 2
    for fd in descriptors:
        with pytest.raises(OSError):
            os.fstat(fd)


async def test_pipe_setup_fdopen_failure_closes_both_descriptors(monkeypatch: pytest.MonkeyPatch) -> None:
    descriptors: list[int] = []
    real_pipe = os.pipe

    def tracked_pipe() -> tuple[int, int]:
        pair = real_pipe()
        descriptors.extend(pair)
        return pair

    def fail_fdopen(*_args, **_kwargs):
        raise OSError("fdopen sentinel")

    monkeypatch.setattr(bash_module.os, "pipe", tracked_pipe)
    monkeypatch.setattr(bash_module.os, "fdopen", fail_fdopen)
    with pytest.raises(OSError, match="fdopen sentinel"):
        await bash_module._open_owned_pipe()

    assert len(descriptors) == 2
    for fd in descriptors:
        with pytest.raises(OSError):
            os.fstat(fd)


async def test_pipe_transport_setup_failure_closes_both_descriptors(monkeypatch: pytest.MonkeyPatch) -> None:
    descriptors: list[int] = []
    real_pipe = os.pipe
    loop = asyncio.get_running_loop()

    def tracked_pipe() -> tuple[int, int]:
        pair = real_pipe()
        descriptors.extend(pair)
        return pair

    async def fail_connect(*_args, **_kwargs):
        raise RuntimeError("transport sentinel")

    monkeypatch.setattr(bash_module.os, "pipe", tracked_pipe)
    monkeypatch.setattr(loop, "connect_read_pipe", fail_connect)
    with pytest.raises(RuntimeError, match="transport sentinel"):
        await bash_module._open_owned_pipe()

    assert len(descriptors) == 2
    for fd in descriptors:
        with pytest.raises(OSError):
            os.fstat(fd)


async def test_startup_failure_is_structured(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async def fail_spawn(*_args, **_kwargs):
        raise FileNotFoundError("bash missing")

    monkeypatch.setattr(bash_module, "_spawn_process", fail_spawn)
    result, _ = await _run_bash(tmp_path, {"command": "printf never"})

    assert result.ok is False
    assert result.metadata["error_type"] == "ProcessStartError"
    assert "bash missing" in result.content


@pytest.mark.parametrize("failure", [ValueError("invalid spawn"), NotImplementedError("unsupported spawn")])
async def test_all_spawn_failures_are_normalized_and_close_owned_pipes(
    failure: Exception,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owned_pipes: list[Any] = []
    write_fds: list[int] = []
    real_open = bash_module._open_owned_pipe

    async def tracked_open():
        pipe, write_fd = await real_open()
        owned_pipes.append(pipe)
        write_fds.append(write_fd)
        return pipe, write_fd

    async def fail_spawn(*_args, **_kwargs):
        raise failure

    monkeypatch.setattr(bash_module, "_open_owned_pipe", tracked_open)
    monkeypatch.setattr(bash_module, "_spawn_process", fail_spawn)
    result, _ = await _run_bash(tmp_path, {"command": "printf never"})

    assert result.metadata["error_type"] == "ProcessStartError"
    assert len(owned_pipes) == 2
    assert all(pipe.file.closed for pipe in owned_pipes)
    assert all(pipe.transport.is_closing() for pipe in owned_pipes)
    for fd in write_fds:
        with pytest.raises(OSError):
            os.fstat(fd)


async def test_post_spawn_exception_closes_readers_and_pipes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owned_pipes: list[Any] = []
    reader_tasks: list[asyncio.Task[None]] = []
    real_open = bash_module._open_owned_pipe
    real_start_readers = bash_module._start_readers

    async def tracked_open():
        pipe, write_fd = await real_open()
        owned_pipes.append(pipe)
        return pipe, write_fd

    def tracked_start_readers(pipes, root, limit):
        readers, captures = real_start_readers(pipes, root, limit)
        reader_tasks.extend(readers)
        return readers, captures

    async def fail_cleanup(*_args, **_kwargs):
        raise RuntimeError("cleanup sentinel")

    monkeypatch.setattr(bash_module, "_open_owned_pipe", tracked_open)
    monkeypatch.setattr(bash_module, "_start_readers", tracked_start_readers)
    monkeypatch.setattr(bash_module, "_terminate_group", fail_cleanup)
    result, _ = await _run_bash(tmp_path, {"command": "printf partial"})

    assert result.metadata["error_type"] == "RuntimeError"
    assert "cleanup sentinel" in result.content
    assert len(reader_tasks) == 2
    assert all(task.done() for task in reader_tasks)
    assert all(pipe.file.closed for pipe in owned_pipes)
    assert all(pipe.transport.is_closing() for pipe in owned_pipes)


async def test_command_timeout_starts_after_delayed_spawn_handoff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_spawn = bash_module._spawn_process

    async def delayed_spawn(*args, **kwargs):
        await asyncio.sleep(0.2)
        return await real_spawn(*args, **kwargs)

    monkeypatch.setattr(bash_module, "_spawn_process", delayed_spawn)
    started = time.monotonic()
    result, _ = await _run_bash(tmp_path, {"command": "sleep 0.02; printf done", "timeout": 0.05})

    assert result.ok is True
    assert _stdout(result) == "done"
    assert time.monotonic() - started >= 0.2


@pytest.mark.parametrize("setup_stage", ["environment", "spawn-task"])
async def test_pre_spawn_setup_failures_close_owned_resources(
    setup_stage: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owned_pipes: list[Any] = []
    write_fds: list[int] = []
    real_open = bash_module._open_owned_pipe

    async def tracked_open():
        pipe, write_fd = await real_open()
        owned_pipes.append(pipe)
        write_fds.append(write_fd)
        return pipe, write_fd

    def fail_environment(_config):
        raise ValueError("environment sentinel")

    def fail_spawn_task(*_args, **_kwargs):
        raise RuntimeError("spawn-task sentinel")

    monkeypatch.setattr(bash_module, "_open_owned_pipe", tracked_open)
    if setup_stage == "environment":
        monkeypatch.setattr(bash_module, "_command_environment", fail_environment)
    else:
        monkeypatch.setattr(bash_module, "_spawn_process", fail_spawn_task)

    result, _ = await _run_bash(tmp_path, {"command": "printf never"})

    assert result.metadata["error_type"] == "ProcessStartError"
    assert setup_stage in result.content
    assert all(pipe.file.closed for pipe in owned_pipes)
    assert all(pipe.transport.is_closing() for pipe in owned_pipes)
    for fd in write_fds:
        with pytest.raises(OSError):
            os.fstat(fd)


async def test_unsupported_platform_fails_before_pipe_or_spawn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    opened = False

    async def unexpected_pipe():
        nonlocal opened
        opened = True
        raise AssertionError

    monkeypatch.setattr(bash_module, "_supports_process_groups", lambda: False)
    monkeypatch.setattr(bash_module, "_open_owned_pipe", unexpected_pipe)
    result, _ = await _run_bash(tmp_path, {"command": "printf never"})

    assert result.metadata["error_type"] == "UnsupportedPlatform"
    assert opened is False


async def test_cwd_must_exist_be_directory_and_stay_inside_root(tmp_path: Path) -> None:
    file_path = tmp_path / "file.txt"
    file_path.write_text("x", encoding="utf-8")
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    link = tmp_path / "link"
    link.symlink_to(outside, target_is_directory=True)

    missing, _ = await _run_bash(tmp_path, {"command": "pwd", "cwd": "missing"})
    not_directory, _ = await _run_bash(tmp_path, {"command": "pwd", "cwd": "file.txt"})
    parent, _ = await _run_bash(tmp_path, {"command": "pwd", "cwd": ".."})
    absolute, _ = await _run_bash(tmp_path, {"command": "pwd", "cwd": str(outside)})
    symlink, _ = await _run_bash(tmp_path, {"command": "pwd", "cwd": "link"})

    assert missing.metadata["error_type"] == "PathNotFound"
    assert not_directory.metadata["error_type"] == "NotADirectory"
    assert parent.metadata["error_type"] == "PathValidationError"
    assert absolute.metadata["error_type"] == "PathValidationError"
    assert symlink.metadata["error_type"] == "PathValidationError"


@pytest.mark.parametrize(
    ("limit", "head", "tail", "omitted"),
    [
        (1, "a", "", 9),
        (5, "abc", "ij", 5),
        (6, "abc", "hij", 4),
    ],
)
async def test_output_uses_fixed_head_tail_split(
    limit: int,
    head: str,
    tail: str,
    omitted: int,
    tmp_path: Path,
) -> None:
    result, _ = await _run_bash(
        tmp_path,
        {"command": "printf abcdefghij; printf ABCDEFGHIJ >&2"},
        plugin=BashPlugin(max_output_bytes=limit),
    )

    stdout = _stdout(result)
    stderr = _stderr(result)
    assert stdout.startswith(head)
    assert stdout.endswith(tail)
    assert stderr.startswith(head.upper())
    assert stderr.endswith(tail.upper())
    assert f"{omitted} bytes omitted" in stdout
    assert f"{omitted} bytes omitted" in stderr
    assert f"complete stdout saved to {result.metadata['stdout_artifact_path']}" in stdout
    assert f"complete stderr saved to {result.metadata['stderr_artifact_path']}" in stderr
    assert result.metadata["stdout_bytes"] == 10
    assert result.metadata["stderr_bytes"] == 10
    assert result.metadata["stdout_truncated"] is True
    assert result.metadata["stderr_truncated"] is True


async def test_truncated_streams_preserve_complete_bytes_in_separate_artifacts(tmp_path: Path) -> None:
    stdout_bytes = bytes(range(256)) * 2
    stderr_bytes = bytes(reversed(range(256))) * 2
    code = f"import os; os.write(1, {stdout_bytes!r}); os.write(2, {stderr_bytes!r})"

    result, _ = await _run_bash(
        tmp_path,
        {"command": f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}"},
        plugin=BashPlugin(max_output_bytes=17),
    )

    stdout_path = Path(result.metadata["stdout_artifact_path"])
    stderr_path = Path(result.metadata["stderr_artifact_path"])
    assert not stdout_path.is_absolute()
    assert not stderr_path.is_absolute()
    assert stdout_path.parent == Path(".thinharness/outputs")
    assert stderr_path.parent == Path(".thinharness/outputs")
    assert stdout_path != stderr_path
    assert (tmp_path / stdout_path).read_bytes() == stdout_bytes
    assert (tmp_path / stderr_path).read_bytes() == stderr_bytes
    assert result.metadata["stdout_omitted_bytes"] == 495
    assert result.metadata["stderr_omitted_bytes"] == 495
    assert result.metadata["stdout_retained_ranges"] == [[0, 9], [504, 512]]
    assert result.metadata["stderr_retained_ranges"] == [[0, 9], [504, 512]]
    assert f"complete stdout saved to {stdout_path}" in _stdout(result)
    assert f"complete stderr saved to {stderr_path}" in _stderr(result)
    retained = "retained bytes [0, 9) and [504, 512); 495 bytes omitted"
    assert retained in _stdout(result)
    assert retained in _stderr(result)


async def test_large_no_newline_floods_are_drained_and_bounded_per_stream(tmp_path: Path) -> None:
    command = "(head -c 200000 /dev/zero | tr '\\0' x) & (head -c 220000 /dev/zero | tr '\\0' y >&2) & wait"
    result, _ = await _run_bash(tmp_path, {"command": command}, plugin=BashPlugin(max_output_bytes=101))
    assert result.ok is True
    assert result.metadata["stdout_bytes"] == 200_000
    assert result.metadata["stderr_bytes"] == 220_000
    assert result.metadata["stdout_truncated"] is True
    assert result.metadata["stderr_truncated"] is True
    assert len(_stdout(result)) < 500
    assert len(_stderr(result)) < 500
    assert _stdout(result).startswith("x" * 51)
    assert _stdout(result).endswith("x" * 50)
    assert _stderr(result).startswith("y" * 51)
    assert _stderr(result).endswith("y" * 50)
    assert (tmp_path / result.metadata["stdout_artifact_path"]).read_bytes() == b"x" * 200_000
    assert (tmp_path / result.metadata["stderr_artifact_path"]).read_bytes() == b"y" * 220_000


async def test_exact_output_limit_is_not_truncated(tmp_path: Path) -> None:
    result, _ = await _run_bash(
        tmp_path,
        {"command": "printf abcde; printf ABCDE >&2"},
        plugin=BashPlugin(max_output_bytes=5),
    )

    assert _stdout(result) == "abcde"
    assert _stderr(result) == "ABCDE"
    assert result.metadata["stdout_bytes"] == 5
    assert result.metadata["stderr_bytes"] == 5
    assert result.metadata["stdout_truncated"] is False
    assert result.metadata["stderr_truncated"] is False
    assert "stdout_artifact_path" not in result.metadata
    assert "stderr_artifact_path" not in result.metadata
    assert not (tmp_path / ".thinharness").exists()


async def test_artifact_failure_is_explicit_and_preserves_primary_command_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_artifact(_capture) -> None:
        raise OSError("artifact sentinel")

    monkeypatch.setattr(bash_module._StreamCapture, "_start_artifact", fail_artifact)
    succeeded, _ = await _run_bash(
        tmp_path,
        {"command": "printf abcdefghij"},
        plugin=BashPlugin(max_output_bytes=5),
    )
    failed, _ = await _run_bash(
        tmp_path,
        {"command": "printf abcdefghij; exit 7"},
        plugin=BashPlugin(max_output_bytes=5),
    )

    assert succeeded.ok is False
    assert succeeded.metadata["exit_code"] == 0
    assert succeeded.metadata["error_type"] == "OutputArtifactError"
    assert succeeded.metadata["output_artifact_errors"] == {"stdout": "OSError: artifact sentinel"}
    assert "stdout_artifact_path" not in succeeded.metadata
    assert "complete stdout could not be saved (OSError: artifact sentinel)" in _stdout(succeeded)
    assert len(_stdout(succeeded)) < 250
    assert failed.ok is False
    assert failed.metadata["exit_code"] == 7
    assert failed.metadata["error_type"] == "NonZeroExit"
    assert failed.metadata["output_artifact_errors"] == {"stdout": "OSError: artifact sentinel"}
    assert not (tmp_path / ".thinharness").exists()


async def test_invalid_and_split_utf8_decode_with_replacement(tmp_path: Path) -> None:
    code = "import os; os.write(1, b'a\\xe2\\x82\\xacb\\xffc')"
    result, _ = await _run_bash(
        tmp_path,
        {"command": f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}"},
        plugin=BashPlugin(max_output_bytes=5),
    )

    assert "�" in _stdout(result)
    assert result.metadata["stdout_bytes"] == 7
    assert result.metadata["stdout_truncated"] is True
    assert (tmp_path / result.metadata["stdout_artifact_path"]).read_bytes() == b"a\xe2\x82\xacb\xffc"


async def test_normal_exit_cleans_same_group_background_descendant(tmp_path: Path) -> None:
    pgid_file = tmp_path / "pgid"
    child_file = tmp_path / "child"
    command = f"echo $$ > {shlex.quote(str(pgid_file))}; sleep 30 & echo $! > {shlex.quote(str(child_file))}"
    result, _ = await _run_bash(tmp_path, {"command": command})
    pgid = int(pgid_file.read_text())
    child = int(child_file.read_text())

    assert result.ok is True
    assert not _group_exists(pgid)
    with pytest.raises(ProcessLookupError):
        os.kill(child, 0)


async def test_normal_exit_kills_term_ignoring_same_group_descendant(tmp_path: Path) -> None:
    child_file = tmp_path / "child"
    code = (
        "import os,signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"open({str(child_file)!r},'w').write(str(os.getpid())); time.sleep(30)"
    )
    command = (
        f"{shlex.quote(sys.executable)} -c {shlex.quote(code)} & "
        f"while ! test -f {shlex.quote(str(child_file))}; do sleep 0.01; done"
    )
    started = time.monotonic()
    result, _ = await _run_bash(tmp_path, {"command": command})
    child = int(child_file.read_text())

    assert result.ok is True
    assert time.monotonic() - started >= 1
    with pytest.raises(ProcessLookupError):
        os.kill(child, 0)


async def test_cancelling_run_kills_group_and_propagates(tmp_path: Path) -> None:
    pgid_file = tmp_path / "pgid"
    ready_file = tmp_path / "ready"
    session = ScriptedSession(start_turn=_call_turn({
        "command": (
            f"echo $$ > {shlex.quote(str(pgid_file))}; printf abcdefghij; "
            f"touch {shlex.quote(str(ready_file))}; trap '' TERM; while :; do sleep 1; done"
        ),
    }))
    harness = Harness(
        HarnessConfig(root=tmp_path),
        model=ScriptedModel([session]),
        plugins=[BashPlugin(max_output_bytes=5)],
    )
    task = asyncio.create_task(harness.run("go"))
    await _wait_for_file(ready_file)
    pgid = int(pgid_file.read_text())

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=4)

    assert not _group_exists(pgid)
    assert not list((tmp_path / ".thinharness" / "outputs").glob("*"))


async def test_repeated_cancellation_does_not_detach_cleanup(tmp_path: Path) -> None:
    pgid_file = tmp_path / "pgid"
    session = ScriptedSession(start_turn=_call_turn({
        "command": f"echo $$ > {shlex.quote(str(pgid_file))}; trap '' TERM; while :; do sleep 1; done",
    }))
    harness = Harness(HarnessConfig(root=tmp_path), model=ScriptedModel([session]), plugins=[BashPlugin()])
    task = asyncio.create_task(harness.run("go"))
    await _wait_for_file(pgid_file)
    pgid = int(pgid_file.read_text())

    task.cancel()
    await asyncio.sleep(0.1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=4)

    assert not _group_exists(pgid)


async def test_cleanup_exception_does_not_replace_run_cancellation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pgid_file = tmp_path / "pgid"
    owned_pipes: list[Any] = []
    reader_tasks: list[asyncio.Task[None]] = []
    real_open = bash_module._open_owned_pipe
    real_start_readers = bash_module._start_readers
    real_cleanup = bash_module._cleanup_process

    async def tracked_open():
        pipe, write_fd = await real_open()
        owned_pipes.append(pipe)
        return pipe, write_fd

    def tracked_start_readers(pipes, root, limit):
        readers, captures = real_start_readers(pipes, root, limit)
        reader_tasks.extend(readers)
        return readers, captures

    async def failing_cleanup(*args, **kwargs):
        await real_cleanup(*args, **kwargs)
        raise RuntimeError("cancel cleanup sentinel")

    monkeypatch.setattr(bash_module, "_open_owned_pipe", tracked_open)
    monkeypatch.setattr(bash_module, "_start_readers", tracked_start_readers)
    monkeypatch.setattr(bash_module, "_cleanup_process", failing_cleanup)
    session = ScriptedSession(start_turn=_call_turn({
        "command": f"echo $$ > {shlex.quote(str(pgid_file))}; trap '' TERM; while :; do sleep 1; done",
    }))
    harness = Harness(HarnessConfig(root=tmp_path), model=ScriptedModel([session]), plugins=[BashPlugin()])
    task = asyncio.create_task(harness.run("go"))
    await _wait_for_file(pgid_file)
    pgid = int(pgid_file.read_text())

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=4)

    assert not _group_exists(pgid)
    assert all(task.done() for task in reader_tasks)
    assert all(pipe.file.closed for pipe in owned_pipes)
    assert all(pipe.transport.is_closing() for pipe in owned_pipes)


async def test_cancellation_during_spawn_waits_for_handoff_and_cleans_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pgid_file = tmp_path / "pgid"
    entered = asyncio.Event()
    release = asyncio.Event()
    real_spawn = bash_module._spawn_process
    spawned_pids: list[int] = []

    async def delayed_spawn(*args, **kwargs):
        entered.set()
        await release.wait()
        process = await real_spawn(*args, **kwargs)
        spawned_pids.append(process.pid)
        return process

    monkeypatch.setattr(bash_module, "_spawn_process", delayed_spawn)
    session = ScriptedSession(start_turn=_call_turn({
        "command": f"echo $$ > {shlex.quote(str(pgid_file))}; trap '' TERM; while :; do sleep 1; done",
    }))
    harness = Harness(HarnessConfig(root=tmp_path), model=ScriptedModel([session]), plugins=[BashPlugin()])
    task = asyncio.create_task(harness.run("go"))
    await entered.wait()

    task.cancel()
    await asyncio.sleep(0.05)
    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=4)

    assert len(spawned_pids) == 1
    assert not _group_exists(spawned_pids[0])


async def _escaped_descendant_command(tmp_path: Path, *, shell_tail: str = ":") -> tuple[str, Path]:
    pid_file = tmp_path / "escaped-pid"
    ready_file = tmp_path / "escaped-ready"
    code = (
        "import os,time; os.setsid(); "
        f"open({str(pid_file)!r},'w').write(str(os.getpid())); "
        f"open({str(ready_file)!r},'w').write('ready'); time.sleep(30)"
    )
    command = (
        f"{shlex.quote(sys.executable)} -c {shlex.quote(code)} & "
        f"while ! test -f {shlex.quote(str(ready_file))}; do sleep 0.01; done; {shell_tail}"
    )
    return command, pid_file


def _kill_escaped(pid_file: Path) -> None:
    if pid_file.exists():
        try:
            os.kill(int(pid_file.read_text()), signal.SIGKILL)
        except ProcessLookupError:
            pass


async def test_escaped_descendant_pipe_has_bounded_normal_drain(tmp_path: Path) -> None:
    command, pid_file = await _escaped_descendant_command(tmp_path, shell_tail="printf done")
    started = time.monotonic()
    try:
        result, _ = await _run_bash(tmp_path, {"command": command})
    finally:
        _kill_escaped(pid_file)

    assert result.ok is True
    assert _stdout(result) == "done"
    assert time.monotonic() - started < 3


async def test_escaped_descendant_pipe_has_bounded_timeout_drain(tmp_path: Path) -> None:
    command, pid_file = await _escaped_descendant_command(tmp_path, shell_tail="printf before; sleep 30")
    started = time.monotonic()
    try:
        result, _ = await _run_bash(tmp_path, {"command": command, "timeout": 0.1})
    finally:
        _kill_escaped(pid_file)

    assert result.metadata["error_type"] == "Timeout"
    assert "before" in _stdout(result)
    assert time.monotonic() - started < 4


async def test_escaped_descendant_pipe_has_bounded_cancellation_drain(tmp_path: Path) -> None:
    command, pid_file = await _escaped_descendant_command(tmp_path, shell_tail="sleep 30")
    ready = tmp_path / "escaped-ready"
    session = ScriptedSession(start_turn=_call_turn({"command": command}))
    harness = Harness(HarnessConfig(root=tmp_path), model=ScriptedModel([session]), plugins=[BashPlugin()])
    task = asyncio.create_task(harness.run("go"))
    await _wait_for_file(ready)
    started = time.monotonic()
    try:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=4)
    finally:
        _kill_escaped(pid_file)

    assert time.monotonic() - started < 3


async def test_minimal_environment_filters_secrets_and_sets_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("THINHARNESS_SECRET_SENTINEL", "hidden")
    monkeypatch.setenv("BASH_ENV", str(tmp_path / "missing-startup"))
    result, _ = await _run_bash(tmp_path, {"command": "env"})
    environment = dict(line.split("=", 1) for line in _stdout(result).splitlines() if "=" in line)

    assert "THINHARNESS_SECRET_SENTINEL" not in environment
    assert "BASH_ENV" not in environment
    assert "ENV" not in environment
    assert environment["NO_COLOR"] == "1"
    assert environment["TERM"] == "dumb"
    assert environment["PAGER"] == "cat"
    assert environment["GIT_PAGER"] == "cat"
    assert set(environment) <= {
        "PATH", "HOME", "TMPDIR", "TMP", "TEMP", "LANG", "LC_ALL", "LC_CTYPE", "TZ",
        "NO_COLOR", "TERM", "PAGER", "GIT_PAGER", "PWD", "SHLVL", "_",
    }


async def test_minimal_environment_allows_explicit_host_overrides(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("THINHARNESS_SECRET_SENTINEL", "hidden")
    result, _ = await _run_bash(
        tmp_path,
        {"command": "env"},
        plugin=BashPlugin(env={"TERM": "minimal-host-term", "EXPLICIT_MINIMAL": "yes"}),
    )
    environment = dict(line.split("=", 1) for line in _stdout(result).splitlines() if "=" in line)

    assert "THINHARNESS_SECRET_SENTINEL" not in environment
    assert environment["TERM"] == "minimal-host-term"
    assert environment["EXPLICIT_MINIMAL"] == "yes"


async def test_full_environment_is_explicit_and_host_values_override_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("THINHARNESS_INHERITED_SENTINEL", "visible")
    result, _ = await _run_bash(
        tmp_path,
        {"command": "env"},
        plugin=BashPlugin(inherit_env=True, env={"TERM": "host-term", "EXPLICIT": "yes"}),
    )
    environment = dict(line.split("=", 1) for line in _stdout(result).splitlines() if "=" in line)

    assert environment["THINHARNESS_INHERITED_SENTINEL"] == "visible"
    assert environment["TERM"] == "host-term"
    assert environment["EXPLICIT"] == "yes"


async def test_bash_env_and_env_are_removed_unless_explicit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    startup = tmp_path / "startup.sh"
    startup.write_text("printf 'STARTUP_MARKER\\n'", encoding="utf-8")
    monkeypatch.setenv("BASH_ENV", str(startup))
    monkeypatch.setenv("ENV", "inherited-env")

    inherited, _ = await _run_bash(tmp_path, {"command": "env"}, plugin=BashPlugin(inherit_env=True))
    explicit, _ = await _run_bash(
        tmp_path,
        {"command": "env"},
        plugin=BashPlugin(env={"BASH_ENV": str(startup), "ENV": "explicit-env"}),
    )
    inherited_environment = dict(line.split("=", 1) for line in _stdout(inherited).splitlines() if "=" in line)
    explicit_environment = dict(line.split("=", 1) for line in _stdout(explicit).splitlines() if "=" in line)

    assert "BASH_ENV" not in inherited_environment
    assert "ENV" not in inherited_environment
    assert explicit_environment["BASH_ENV"] == str(startup)
    assert explicit_environment["ENV"] == "explicit-env"
    assert _stdout(explicit).startswith("STARTUP_MARKER\n")


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"command": "   \t"},
        {"command": 1},
        {"command": "printf ok", "cwd": 1},
        {"command": "printf ok", "timeout": True},
        {"command": "printf ok", "timeout": "1"},
        {"command": "printf ok", "timeout": float("inf")},
        {"command": "printf ok", "timeout": 0},
        {"command": "printf ok", "env": {"SECRET": "x"}},
    ],
)
def test_model_arguments_use_retryable_strict_validation(tmp_path: Path, arguments: dict[str, Any]) -> None:
    harness = Harness(HarnessConfig(root=tmp_path), model=ScriptedModel([]), plugins=[BashPlugin()])
    output = ToolResult.from_json(call_tool(harness.tools[0], arguments))

    assert output.ok is False
    assert output.metadata["error_type"] == "ValidationError"
    assert output.metadata["retry"] is True


async def test_huge_integer_timeout_is_retryable_through_public_call_paths(tmp_path: Path) -> None:
    arguments = {"command": "printf never", "timeout": 10**400}
    harness = Harness(HarnessConfig(root=tmp_path), model=ScriptedModel([]), plugins=[BashPlugin()])
    direct = ToolResult.from_json(call_tool(harness.tools[0], arguments))
    through_run, _ = await _run_bash(tmp_path, arguments)

    for result in (direct, through_run):
        assert result.ok is False
        assert result.metadata["error_type"] == "ValidationError"
        assert result.metadata["retry"] is True


async def test_approval_uses_normal_top_level_pause_and_resume(tmp_path: Path) -> None:
    first = ScriptedSession(start_turn=_call_turn({"command": "printf approved"}))
    resumed_outputs: list[ToolResult] = []
    resumed = ScriptedSession(
        start_turn=ModelTurn(raw={"unused": True}),
        continue_turn=ModelTurn(text="done", raw={"id": "done"}),
        on_continue=lambda outputs, _tools, _metadata: resumed_outputs.append(ToolResult.from_json(outputs[0].output)),
    )
    harness = Harness(
        HarnessConfig(root=tmp_path),
        model=ScriptedModel([first, resumed]),
        plugins=[BashPlugin(requires_approval=True)],
    )

    paused = await harness.run("go")
    assert paused.stop_reason == "approval_required"
    assert paused.pending_approvals[0].tool_name == "bash"
    result = await harness.resume_approvals(
        paused.resume_state,
        [ApprovalDecision(call_id="call_1", approved=True)],
    )

    assert result.text == "done"
    assert resumed_outputs[0].ok is True
    assert _stdout(resumed_outputs[0]) == "approved"


def test_approval_required_bash_follows_existing_model_and_child_rules(tmp_path: Path) -> None:
    model = ScriptedModel([])
    del model.resume_kind
    with pytest.raises(ValueError, match="resumable model"):
        Harness(
            HarnessConfig(root=tmp_path),
            model=model,
            plugins=[BashPlugin(requires_approval=True)],
        )
    with pytest.raises(ValueError, match="child harnesses"):
        Harness(
            HarnessConfig(root=tmp_path),
            model=ScriptedModel([]),
            plugins=[SubagentsPlugin(agents=[SubAgentConfig(
                name="shell",
                description="Shell child.",
                plugins=[BashPlugin(requires_approval=True)],
            )])],
        )


def test_bash_does_not_implement_child_inheritance() -> None:
    assert not hasattr(BashPlugin(), "for_child")


def test_default_and_named_children_do_not_inherit_bash_and_explicit_child_can_use_it(tmp_path: Path) -> None:
    observed_tools: list[list[str]] = []
    parent = ScriptedSession(
        start_turn=ModelTurn(
            tool_calls=[ModelToolCall(id="call_1", name="subagent", arguments='{"task":"check"}')],
            raw={"id": "parent"},
        ),
    )
    child = ScriptedSession(
        start_turn=ModelTurn(text="child", raw={"id": "child"}),
        on_start=lambda _prompt, _instructions, tools, _metadata, _previous: observed_tools.append([tool["name"] for tool in tools]),
    )
    Harness(
        HarnessConfig(root=tmp_path),
        model=ScriptedModel([parent, child]),
        plugins=[BashPlugin(), SubagentsPlugin()],
    ).run_sync("go")

    assert "bash" not in observed_tools[0]

    observed_tools.clear()
    parent = ScriptedSession(
        start_turn=ModelTurn(
            tool_calls=[ModelToolCall(id="call_2", name="subagent", arguments='{"task":"check","agent":"plain"}')],
            raw={"id": "parent"},
        ),
    )
    child = ScriptedSession(
        start_turn=ModelTurn(text="child", raw={"id": "child"}),
        on_start=lambda _prompt, _instructions, tools, _metadata, _previous: observed_tools.append([tool["name"] for tool in tools]),
    )
    Harness(
        HarnessConfig(root=tmp_path),
        model=ScriptedModel([parent, child]),
        plugins=[BashPlugin(), SubagentsPlugin(agents=[SubAgentConfig(
            name="plain",
            description="Plain child.",
        )])],
    ).run_sync("go")

    assert "bash" not in observed_tools[0]

    observed_tools.clear()
    child_bash_results: list[ToolResult] = []
    parent = ScriptedSession(
        start_turn=ModelTurn(
            tool_calls=[ModelToolCall(id="call_3", name="subagent", arguments='{"task":"check","agent":"shell"}')],
            raw={"id": "parent"},
        ),
    )
    child = ScriptedSession(
        start_turn=_call_turn({"command": "printf child-bash"}, call_id="child_bash_call"),
        continue_turn=ModelTurn(text="child", raw={"id": "child"}),
        on_start=lambda _prompt, _instructions, tools, _metadata, _previous: observed_tools.append([tool["name"] for tool in tools]),
        on_continue=lambda outputs, _tools, _metadata: child_bash_results.append(ToolResult.from_json(outputs[0].output)),
    )
    Harness(
        HarnessConfig(root=tmp_path),
        model=ScriptedModel([parent, child]),
        plugins=[SubagentsPlugin(agents=[SubAgentConfig(
            name="shell",
            description="Shell child.",
            plugins=[BashPlugin()],
        )])],
    ).run_sync("go")

    assert "bash" in observed_tools[0]
    assert len(child_bash_results) == 1
    assert child_bash_results[0].ok is True
    assert _stdout(child_bash_results[0]) == "child-bash"


def test_mixed_batch_containing_bash_runs_sequentially(tmp_path: Path) -> None:
    client = MultiCallClient([("bash", '{"command":"sleep 0.2; printf bash"}'), ("slow", "{}")])
    harness = Harness(
        HarnessConfig(root=tmp_path, model="openai:test-model"),
        model=_fake_openai(client),
        plugins=[BashPlugin()],
        tools=[slow_tool("slow", 0.2)],
    )

    started = time.monotonic()
    harness.run_sync("go")

    assert time.monotonic() - started >= 0.38
    assert [item["call_id"] for item in client.payloads[1]["input"]] == ["call_1", "call_2"]


async def test_bash_uses_normal_hooks_records_and_tracing(tmp_path: Path) -> None:
    hook_calls: list[str] = []
    tracer = ContextFakeTracer()
    result, run_result = await _run_bash(
        tmp_path,
        {"command": "printf integrated"},
        hooks=[
            Hook("before_tool_call", lambda ctx: hook_calls.append(f"before:{ctx.tool_name}")),
            Hook("after_tool_call", lambda ctx: hook_calls.append(f"after:{ctx.tool_name}")),
        ],
        tracing=[TracingOptions(tracer=tracer)],
    )

    assert result.ok is True
    assert hook_calls == ["before:bash", "after:bash"]
    assert ToolResult.from_json(run_result.tool_call_records[0]["output"]).content == result.content
    tool_span = next(span for span in tracer.spans if span.name == "execute_tool bash")
    assert tool_span.attributes["gen_ai.tool.name"] == "bash"
