"""Opt-in bash tool for exploratory workflows."""

from __future__ import annotations

import os
import signal
import subprocess
import tempfile
import time
from pathlib import Path

from pydantic import Field

from .base import (
    Json,
    PathPolicy,
    PathValidationError,
    StrictArgs,
    ToolResult,
    ToolSpec,
    _path_error,
    coerce_args,
)

BASH_DESCRIPTION = "Run one bash command from a workspace-contained cwd. Intended for exploratory workflows; prefer typed tools for production."


class BashArgs(StrictArgs):
    """Arguments for bash."""

    command: str
    cwd: str = "."
    timeout: int = Field(default=10, ge=1, le=120)
    max_chars: int | None = Field(default=None, ge=1, description="Per-stream stdout/stderr output cap, clamped to the tool cap.")


class BashTool:
    """Small, explicitly registered bash command tool."""

    def __init__(
        self,
        root: str | Path = ".",
        *,
        max_tool_chars: int = 40_000,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.cwd_policy = PathPolicy(self.root, ["."], "bash cwd")
        self.max_tool_chars = max_tool_chars

    def spec(self) -> ToolSpec:
        """Return the bash tool spec."""
        return ToolSpec("bash", BASH_DESCRIPTION, BashArgs, self.run, sequential=True)

    def run(self, args: BashArgs | Json) -> ToolResult:
        """Run one bash command from a contained working directory."""
        args = coerce_args(args, BashArgs)
        try:
            cwd = self.cwd_policy.resolve(args.cwd)
        except PathValidationError as exc:
            return _path_error(exc)
        if not cwd.exists():
            return ToolResult(False, f"cwd not found: {self._display(cwd)}", {"error_type": "PathNotFound", "cwd": str(cwd)})
        if not cwd.is_dir():
            return ToolResult(False, f"cwd is not a directory: {self._display(cwd)}", {"error_type": "NotADirectory", "cwd": str(cwd)})

        limit = min(args.max_chars or self.max_tool_chars, self.max_tool_chars)
        start = time.perf_counter()
        with tempfile.TemporaryFile() as stdout_file, tempfile.TemporaryFile() as stderr_file:
            process = subprocess.Popen(
                ["bash", "-c", args.command],
                cwd=cwd,
                stdin=subprocess.DEVNULL,
                stdout=stdout_file,
                stderr=stderr_file,
                start_new_session=True,
            )
            timed_out = False
            try:
                process.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                self._terminate_process_group(process)
            else:
                self._cleanup_process_group(process)
            stdout, stdout_truncated = _read_limited_text(stdout_file, limit)
            stderr, stderr_truncated = _read_limited_text(stderr_file, limit)

        duration = time.perf_counter() - start
        exit_code = process.returncode
        ok = not timed_out and exit_code == 0
        metadata: Json = {
            "exit_code": exit_code,
            "timed_out": timed_out,
            "duration_seconds": round(duration, 3),
            "cwd": str(cwd),
            "stdout_truncated": stdout_truncated,
            "stderr_truncated": stderr_truncated,
        }
        if timed_out:
            metadata["error_type"] = "Timeout"
        elif exit_code != 0:
            metadata["error_type"] = "NonZeroExit"
        return ToolResult(ok, _format_output(stdout, stderr), metadata)

    def _display(self, path: Path) -> str:
        """Return a workspace-relative display path where possible."""
        try:
            return str(path.relative_to(self.root)) or "."
        except ValueError:
            return str(path)

    @staticmethod
    def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
        """Terminate a timed-out process group on POSIX, falling back to kill."""
        BashTool._signal_process_group(process, signal.SIGTERM)
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            pass
        BashTool._signal_process_group(process, signal.SIGKILL)
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            process.kill()
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass

    @staticmethod
    def _cleanup_process_group(process: subprocess.Popen[bytes]) -> None:
        """Best-effort cleanup for background descendants left by bash."""
        if not BashTool._signal_process_group(process, signal.SIGTERM):
            return
        time.sleep(0.05)
        BashTool._signal_process_group(process, signal.SIGKILL)

    @staticmethod
    def _signal_process_group(process: subprocess.Popen[bytes], sig: signal.Signals) -> bool:
        """Signal a process group if it still exists."""
        try:
            os.killpg(process.pid, sig)
            return True
        except ProcessLookupError:
            return False


def _format_output(stdout: str, stderr: str) -> str:
    """Return labeled command output."""
    return f"stdout:\n{stdout}\nstderr:\n{stderr}"


def _read_limited_text(handle, limit: int) -> tuple[str, bool]:
    """Read a bounded amount of UTF-8-ish text from a binary file handle."""
    handle.seek(0)
    max_bytes = limit * 4
    raw = handle.read(max_bytes + 1)
    bytes_truncated = len(raw) > max_bytes
    text = raw[:max_bytes].decode("utf-8", errors="replace")
    text, chars_truncated = _truncate_text(text, limit)
    return text, bytes_truncated or chars_truncated


def _truncate_text(text: str, limit: int) -> tuple[str, bool]:
    """Return bounded output and whether it was truncated."""
    if len(text) <= limit:
        return text, False
    marker = "...[truncated]"
    if limit <= len(marker):
        return marker[:limit], True
    return f"{text[: limit - len(marker)]}{marker}", True
