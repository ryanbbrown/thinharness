"""Explicit bounded local Bash plugin."""

from __future__ import annotations

import asyncio
import math
import os
import signal
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator

from ..tools.base import PathValidationError, StrictArgs, ToolOrigin, ToolResult, ToolSpec, contained_path
from .base import PluginBinding, PluginContext, PluginContribution

_BASH_DESCRIPTION = (
    "Run one non-interactive Bash command. The cwd must be inside the workspace. Each call starts a fresh shell and does not share "
    "shell state with other calls. Background processes in the same process group are terminated on a best-effort basis. timeout is in "
    "seconds and is capped by the host."
)
_ENV_ALLOWLIST = ("PATH", "HOME", "TMPDIR", "TMP", "TEMP", "LANG", "LC_ALL", "LC_CTYPE", "TZ")
_READ_CHUNK_SIZE = 64 * 1024
_TERMINATE_GRACE_SECONDS = 1.0
_FINAL_DRAIN_SECONDS = 1.0
_CANCELLATION_CLEANUP_SECONDS = 4.0


class _BashArgs(StrictArgs):
    """Model arguments for one Bash call."""

    command: str = Field(min_length=1, strict=True)
    cwd: str = Field(default=".", strict=True)
    timeout: float | None = None

    @field_validator("command")
    @classmethod
    def _validate_command(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("command must contain a non-whitespace character")
        return value

    @field_validator("timeout", mode="before")
    @classmethod
    def _validate_timeout(cls, value: object) -> object:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError("timeout must be a real integer or float")
        try:
            converted = float(value)
        except OverflowError as exc:
            raise ValueError("timeout must be finite and greater than zero") from exc
        if not math.isfinite(converted) or converted <= 0:
            raise ValueError("timeout must be finite and greater than zero")
        return converted


@dataclass(frozen=True)
class _BashConfig:
    default_timeout: float
    max_timeout: float
    max_output_bytes: int
    inherit_env: bool
    env_entries: tuple[tuple[str, str], ...]
    requires_approval: bool


class _BashPluginMeta(type):
    """Keep the Bash plugin name fixed on the class hierarchy."""

    def __setattr__(cls, attribute: str, value: object) -> None:
        if attribute == "name":
            raise AttributeError("BashPlugin.name is fixed to 'bash'")
        super().__setattr__(attribute, value)

    def __delattr__(cls, attribute: str) -> None:
        if attribute == "name":
            raise AttributeError("BashPlugin.name is fixed to 'bash'")
        super().__delattr__(attribute)


class BashPlugin(metaclass=_BashPluginMeta):
    """Provide one bounded, non-interactive local Bash tool."""

    name = "bash"
    _config: _BashConfig
    _frozen: bool

    def __init_subclass__(cls) -> None:
        """Reject subclasses that replace the fixed plugin name."""
        super().__init_subclass__()
        if "name" in cls.__dict__:
            raise TypeError("BashPlugin subclasses cannot override the fixed name 'bash'")

    def __setattr__(self, attribute: str, value: object) -> None:
        """Reject configuration changes after construction."""
        if attribute == "name":
            raise AttributeError("BashPlugin.name is fixed to 'bash'")
        if getattr(self, "_frozen", False):
            raise AttributeError("BashPlugin configuration is frozen")
        object.__setattr__(self, attribute, value)

    def __delattr__(self, attribute: str) -> None:
        """Reject configuration deletion after construction."""
        if attribute == "name":
            raise AttributeError("BashPlugin.name is fixed to 'bash'")
        if getattr(self, "_frozen", False):
            raise AttributeError("BashPlugin configuration is frozen")
        object.__delattr__(self, attribute)

    @property
    def env(self) -> dict[str, str]:
        """Return a detached copy of the explicit host environment."""
        return dict(self._config.env_entries)

    def __init__(
        self,
        *,
        default_timeout: float = 30,
        max_timeout: float = 120,
        max_output_bytes: int = 40_000,
        inherit_env: bool = False,
        env: Mapping[str, str] | None = None,
        requires_approval: bool = False,
    ) -> None:
        default = _validate_positive_real("default_timeout", default_timeout)
        maximum = _validate_positive_real("max_timeout", max_timeout)
        if default > maximum:
            raise ValueError("default_timeout must not exceed max_timeout")
        if isinstance(max_output_bytes, bool) or not isinstance(max_output_bytes, int):
            raise TypeError("max_output_bytes must be an integer")
        if max_output_bytes <= 0:
            raise ValueError("max_output_bytes must be greater than zero")
        if not isinstance(inherit_env, bool):
            raise TypeError("inherit_env must be a boolean")
        if not isinstance(requires_approval, bool):
            raise TypeError("requires_approval must be a boolean")
        if env is not None and not isinstance(env, Mapping):
            raise TypeError("env must be a mapping of strings")
        env_entries = tuple(_validate_environment(env).items()) if env is not None else ()
        object.__setattr__(self, "_config", _BashConfig(
            default_timeout=default,
            max_timeout=maximum,
            max_output_bytes=max_output_bytes,
            inherit_env=inherit_env,
            env_entries=env_entries,
            requires_approval=requires_approval,
        ))
        object.__setattr__(self, "_frozen", True)

    def bind(self, context: PluginContext) -> PluginBinding:
        """Build one static tool against the canonical harness root."""
        runner = _BashRunner(context.root, self._config)
        tool = ToolSpec(
            "bash",
            _BASH_DESCRIPTION,
            _BashArgs,
            runner.run,
            sequential=True,
            requires_approval=self._config.requires_approval,
            origin=ToolOrigin(plugin="bash", source="bash"),
        )
        return PluginBinding(static=PluginContribution(tools=(tool,)))


def _validate_positive_real(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be a real integer or float")
    try:
        converted = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be finite and greater than zero") from exc
    if not math.isfinite(converted) or converted <= 0:
        raise ValueError(f"{name} must be finite and greater than zero")
    return converted


def _validate_environment(values: Mapping[Any, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for name, value in values.items():
        if not isinstance(name, str) or not isinstance(value, str):
            raise TypeError("environment names and values must be strings")
        if not name or "=" in name or "\0" in name:
            raise ValueError("environment names must be non-empty and contain neither '=' nor NUL")
        if "\0" in value:
            raise ValueError("environment values must not contain NUL")
        result[name] = value
    return result


@dataclass
class _BoundedBuffer:
    """Retain fixed head and tail byte regions while counting all input."""

    limit: int
    total: int = 0

    def __post_init__(self) -> None:
        self._head_limit = (self.limit + 1) // 2
        self._tail_limit = self.limit // 2
        self._head = bytearray()
        self._tail = bytearray()

    def add(self, data: bytes) -> None:
        self.total += len(data)
        head_needed = self._head_limit - len(self._head)
        if head_needed > 0:
            self._head.extend(data[:head_needed])
            data = data[head_needed:]
        if not data or self._tail_limit == 0:
            return
        if len(data) >= self._tail_limit:
            self._tail[:] = data[-self._tail_limit:]
            return
        overflow = len(self._tail) + len(data) - self._tail_limit
        if overflow > 0:
            del self._tail[:overflow]
        self._tail.extend(data)

    @property
    def truncated(self) -> bool:
        return self.total > self.limit

    def render(self) -> str:
        if not self.truncated:
            return bytes(self._head + self._tail).decode("utf-8", errors="replace")
        omitted = self.total - len(self._head) - len(self._tail)
        marker = f"\n... {omitted} bytes omitted ...\n"
        return self._head.decode("utf-8", errors="replace") + marker + self._tail.decode("utf-8", errors="replace")


@dataclass
class _OwnedPipe:
    """A parent-owned pipe reader and its asyncio transport."""

    reader: asyncio.StreamReader
    transport: asyncio.ReadTransport
    file: Any

    def close(self) -> None:
        self.transport.close()
        self.file.close()


class _BashRunner:
    """Run local processes for one bound Bash plugin."""

    def __init__(self, root: Path, config: _BashConfig) -> None:
        self._root = root
        self._config = config

    async def run(self, args: _BashArgs) -> ToolResult:
        """Run one validated Bash command and return a structured outcome."""
        if not _supports_process_groups():
            return ToolResult(False, "BashPlugin requires POSIX process-group support", {"error_type": "UnsupportedPlatform"})
        try:
            cwd = contained_path(self._root, args.cwd)
        except PathValidationError as exc:
            return ToolResult(False, str(exc), {"error_type": "PathValidationError"})
        if not cwd.exists():
            return ToolResult(False, f"cwd not found: {cwd}", {"error_type": "PathNotFound", "cwd": str(cwd)})
        if not cwd.is_dir():
            return ToolResult(False, f"cwd is not a directory: {cwd}", {"error_type": "NotADirectory", "cwd": str(cwd)})
        timeout = min(args.timeout if args.timeout is not None else self._config.default_timeout, self._config.max_timeout)
        return await self._execute(args.command, cwd, timeout)

    async def _execute(self, command: str, cwd: Path, timeout: float) -> ToolResult:
        pipes: list[_OwnedPipe] = []
        write_fds: list[int] = []
        spawn_task: asyncio.Task[asyncio.subprocess.Process] | None = None
        process: asyncio.subprocess.Process | None = None
        readers: list[asyncio.Task[None]] = []
        started: float | None = None
        try:
            try:
                stdout_pipe, stdout_write = await _open_owned_pipe()
                pipes.append(stdout_pipe)
                write_fds.append(stdout_write)
                stderr_pipe, stderr_write = await _open_owned_pipe()
                pipes.append(stderr_pipe)
                write_fds.append(stderr_write)
                environment = _command_environment(self._config)
                started = time.perf_counter()
                spawn_coroutine = _spawn_process(command, cwd, environment, write_fds[0], write_fds[1])
                try:
                    spawn_task = asyncio.create_task(spawn_coroutine)
                except BaseException:
                    spawn_coroutine.close()
                    raise
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                duration = 0.0 if started is None else time.perf_counter() - started
                return self._start_error(exc, cwd, timeout, duration)

            assert spawn_task is not None
            try:
                process = await asyncio.shield(spawn_task)
            except asyncio.CancelledError:
                process = await _spawn_after_cancellation(spawn_task)
                _close_fds(write_fds)
                write_fds.clear()
                if process is not None:
                    readers, _ = _start_readers(pipes, self._config.max_output_bytes)
                    wait_task = asyncio.create_task(process.wait())
                    _signal_group(process.pid, signal.SIGTERM)
                    cleanup = asyncio.create_task(_cleanup_process(
                        process.pid,
                        process,
                        wait_task,
                        readers,
                        pipes,
                        initial_term_sent=True,
                    ))
                    await _finish_cleanup_despite_cancellation(cleanup)
                else:
                    _close_pipes(pipes)
                raise
            except Exception as exc:
                assert started is not None
                return self._start_error(exc, cwd, timeout, time.perf_counter() - started)
            finally:
                if spawn_task.done():
                    _close_fds(write_fds)
                    write_fds.clear()

            assert process is not None
            assert started is not None
            readers, buffers = _start_readers(pipes, self._config.max_output_bytes)
            wait_task = asyncio.create_task(process.wait())
            timed_out = False
            try:
                try:
                    await asyncio.wait_for(asyncio.shield(wait_task), timeout=timeout)
                except TimeoutError:
                    timed_out = True
                    _signal_group(process.pid, signal.SIGTERM)
                    await _terminate_group(process.pid, initial_term_sent=True)
                    await _join_process(process, wait_task)
                else:
                    await _terminate_group(process.pid)
                await _final_drain(readers, pipes)
            except asyncio.CancelledError:
                _signal_group(process.pid, signal.SIGTERM)
                cleanup = asyncio.create_task(_cleanup_process(
                    process.pid,
                    process,
                    wait_task,
                    readers,
                    pipes,
                    initial_term_sent=True,
                ))
                await _finish_cleanup_despite_cancellation(cleanup)
                raise

            returncode = process.returncode
            duration = time.perf_counter() - started
            metadata: dict[str, Any] = {
                "exit_code": returncode,
                "timed_out": timed_out,
                "duration_seconds": round(duration, 3),
                "cwd": str(cwd),
                "timeout_seconds": timeout,
                "stdout_bytes": buffers[0].total,
                "stderr_bytes": buffers[1].total,
                "stdout_truncated": buffers[0].truncated,
                "stderr_truncated": buffers[1].truncated,
            }
            if returncode is not None and returncode < 0:
                metadata["signal"] = -returncode
            if timed_out:
                metadata["error_type"] = "Timeout"
            elif returncode != 0:
                metadata["error_type"] = "NonZeroExit"
            return ToolResult(
                not timed_out and returncode == 0,
                _format_output(buffers[0].render(), buffers[1].render()),
                metadata,
            )
        finally:
            _close_fds(write_fds)
            _close_pipes(pipes)
            for reader in readers:
                if not reader.done():
                    reader.cancel()
            if readers:
                await asyncio.gather(*readers, return_exceptions=True)
            if spawn_task is not None and not spawn_task.done():
                spawn_task.cancel()
                await asyncio.gather(spawn_task, return_exceptions=True)

    @staticmethod
    def _start_error(exc: BaseException, cwd: Path, timeout: float, duration: float) -> ToolResult:
        return ToolResult(False, f"could not start Bash: {type(exc).__name__}: {exc}", {
            "error_type": "ProcessStartError",
            "timed_out": False,
            "duration_seconds": round(duration, 3),
            "cwd": str(cwd),
            "timeout_seconds": timeout,
        })


def _supports_process_groups() -> bool:
    return os.name == "posix" and callable(getattr(os, "killpg", None))


async def _spawn_process(
    command: str,
    cwd: Path,
    environment: dict[str, str],
    stdout_fd: int,
    stderr_fd: int,
) -> asyncio.subprocess.Process:
    return await asyncio.create_subprocess_exec(
        "bash",
        "-c",
        command,
        cwd=cwd,
        env=environment,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=stdout_fd,
        stderr=stderr_fd,
        start_new_session=True,
    )


def _command_environment(config: _BashConfig) -> dict[str, str]:
    if config.inherit_env:
        environment = dict(os.environ)
    else:
        environment = {name: os.environ[name] for name in _ENV_ALLOWLIST if name in os.environ}
    environment.pop("BASH_ENV", None)
    environment.pop("ENV", None)
    environment.update({"NO_COLOR": "1", "TERM": "dumb", "PAGER": "cat", "GIT_PAGER": "cat"})
    environment.update(config.env_entries)
    return environment


async def _open_owned_pipe() -> tuple[_OwnedPipe, int]:
    read_fd, write_fd = os.pipe()
    file: Any | None = None
    try:
        os.set_inheritable(read_fd, False)
        os.set_inheritable(write_fd, False)
        file = os.fdopen(read_fd, "rb", buffering=0)
        read_fd = -1
        reader = asyncio.StreamReader(limit=_READ_CHUNK_SIZE)
        protocol = asyncio.StreamReaderProtocol(reader)
        transport, _ = await asyncio.get_running_loop().connect_read_pipe(lambda: protocol, file)
        return _OwnedPipe(reader, transport, file), write_fd
    except BaseException:
        if file is not None:
            file.close()
        elif read_fd >= 0:
            os.close(read_fd)
        os.close(write_fd)
        raise


def _start_readers(pipes: list[_OwnedPipe], limit: int) -> tuple[list[asyncio.Task[None]], list[_BoundedBuffer]]:
    buffers = [_BoundedBuffer(limit), _BoundedBuffer(limit)]
    readers = [asyncio.create_task(_read_pipe(pipe.reader, buffer)) for pipe, buffer in zip(pipes, buffers, strict=True)]
    return readers, buffers


async def _read_pipe(reader: asyncio.StreamReader, buffer: _BoundedBuffer) -> None:
    while chunk := await reader.read(_READ_CHUNK_SIZE):
        buffer.add(chunk)


async def _spawn_after_cancellation(task: asyncio.Task[asyncio.subprocess.Process]) -> asyncio.subprocess.Process | None:
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
        except Exception:
            break
    try:
        return task.result()
    except asyncio.CancelledError:
        return None
    except Exception:
        return None


async def _cleanup_process(
    pgid: int,
    process: asyncio.subprocess.Process,
    wait_task: asyncio.Task[int],
    readers: list[asyncio.Task[None]],
    pipes: list[_OwnedPipe],
    *,
    initial_term_sent: bool,
) -> None:
    try:
        await _terminate_group(pgid, initial_term_sent=initial_term_sent)
        await _join_process(process, wait_task)
    finally:
        await _final_drain(readers, pipes)


async def _terminate_group(pgid: int, *, initial_term_sent: bool = False) -> None:
    if not initial_term_sent and not _signal_group(pgid, signal.SIGTERM):
        return
    deadline = asyncio.get_running_loop().time() + _TERMINATE_GRACE_SECONDS
    while _group_exists(pgid):
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            break
        await asyncio.sleep(min(0.05, remaining))
    if _group_exists(pgid):
        _signal_group(pgid, signal.SIGKILL)


async def _join_process(process: asyncio.subprocess.Process, wait_task: asyncio.Task[int]) -> None:
    if not wait_task.done():
        try:
            await asyncio.wait_for(asyncio.shield(wait_task), timeout=_TERMINATE_GRACE_SECONDS)
        except TimeoutError:
            process.kill()
    if not wait_task.done():
        try:
            await asyncio.wait_for(asyncio.shield(wait_task), timeout=_TERMINATE_GRACE_SECONDS)
        except TimeoutError:
            wait_task.cancel()
    await asyncio.gather(wait_task, return_exceptions=True)


async def _final_drain(readers: list[asyncio.Task[None]], pipes: list[_OwnedPipe]) -> None:
    if readers:
        await asyncio.wait(readers, timeout=_FINAL_DRAIN_SECONDS)
    _close_pipes(pipes)
    for reader in readers:
        if not reader.done():
            reader.cancel()
    if readers:
        await asyncio.gather(*readers, return_exceptions=True)


async def _finish_cleanup_despite_cancellation(task: asyncio.Task[None]) -> None:
    deadline = asyncio.get_running_loop().time() + _CANCELLATION_CLEANUP_SECONDS
    while not task.done():
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            task.cancel()
            break
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=remaining)
        except asyncio.CancelledError:
            continue
        except TimeoutError:
            task.cancel()
            break
        except Exception:
            break
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
        except Exception:
            break
    await asyncio.gather(task, return_exceptions=True)


def _signal_group(pgid: int, sig: signal.Signals) -> bool:
    try:
        os.killpg(pgid, sig)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return False


def _group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _close_fds(fds: list[int]) -> None:
    for fd in fds:
        try:
            os.close(fd)
        except OSError:
            pass


def _close_pipes(pipes: list[_OwnedPipe]) -> None:
    for pipe in pipes:
        pipe.close()


def _format_output(stdout: str, stderr: str) -> str:
    if not stdout and not stderr:
        return "(no output)"
    return f"stdout:\n{stdout or '(no output)'}\nstderr:\n{stderr or '(no output)'}"


__all__ = ["BashPlugin"]
