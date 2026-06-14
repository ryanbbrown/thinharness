# Bash Prototype Tool Plan

## Goal

Add a deliberately small, opt-in bash tool for exploratory agent runs. The tool is for prototyping workflow shape before promoting repeated shell logic into typed production tools.

## Assumptions

- The framework opinion changes from "No bash" to "No bash by default."
- The tool should not be part of `DEFAULT_BUILTIN_TOOLS`.
- The tool should not be added to the harness built-in candidate list.
- The tool should be available only through custom tool registration.
- The implementation should avoid sandboxing, approval flows, persistent sessions, PTYs, stdin, shell selection, login-shell behavior, and background execution.
- The tool should run `bash -c`, not a login shell.
- The tool's cwd validation is only cwd containment. It is not a filesystem sandbox: commands can still read or write absolute paths, use network tools, inspect environment variables, and perform any action the host process can perform.

## Proposed Public Shape

Expose a `BashTool` from a new `thinharness.tools.bash` module.

Arguments:

```python
class BashArgs(StrictArgs):
    command: str
    cwd: str = "."
    timeout: int = Field(default=10, ge=1, le=120)
    max_chars: int | None = Field(default=None, ge=1)
```

Tool name: `bash`

Description:

```text
Run one bash command from a workspace-contained cwd. Intended for exploratory workflows; prefer typed tools for production.
```

Result metadata:

- `exit_code`
- `timed_out`
- `duration_seconds`
- `cwd`
- `stdout_truncated`
- `stderr_truncated`

## Implementation

1. Add `thinharness/tools/bash.py`.
   - Define `BashArgs`.
   - Define `BashTool` with `root`, cwd validation, default timeout, and maximum output characters.
   - Use `PathPolicy(root, ["."], "bash cwd")` to ensure the process cwd stays inside the workspace. Do not reuse `read_paths` or `write_paths`; neither maps cleanly to a command that can both read and write.
   - Fail early with a `ToolResult` if the resolved cwd does not exist or is not a directory.
   - Invoke `bash -c` with pipes. Prefer `subprocess.Popen(..., start_new_session=True)` plus `communicate(timeout=...)` so timeout handling can terminate the process group on POSIX before collecting output.
   - Decode stdout and stderr bytes with `errors="replace"` rather than relying on `text=True`, including timeout paths.
   - Return a `ToolResult` containing labeled stdout and stderr, plus metadata.
   - On timeout, return `ok=False`, `timed_out=True`, terminate the process group where possible, and include captured output after termination. Do not attempt streaming or incremental partial-output semantics.
   - Set `ToolSpec(..., sequential=True)` because bash can mutate the workspace or external system state.

2. Wire exports.
   - Export `BashTool` and `BashArgs` from `thinharness.tools.__init__`.
   - Export them from top-level `thinharness.__init__` if consistent with existing public tool exports.

3. Keep registration custom-only.
   - Do not import `BashTool` in `thinharness/core.py`.
   - Do not add `BashTool(...).spec()` to `builtin_candidates`.
   - Do not add `bash` to `DEFAULT_BUILTIN_TOOLS`.
   - `HarnessConfig(builtin_tools=["bash"])` should continue to fail with the existing "unknown builtin tool" error.
   - Callers opt in by passing `tools=[BashTool(root, max_tool_chars=...).spec()]` or by calling `harness.add_tool(BashTool(root).spec())`.
   - Subagents can receive bash only if the host application explicitly passes a custom bash `ToolSpec` into that subagent configuration; `SubAgentConfig(builtin_tools=["bash"])` should not work.

4. Update docs.
   - Revise the README opinion from "No bash" to "No bash by default" while keeping the production warning explicit.
   - State that bash is a prototyping aid and repeated commands should be promoted into typed tools.
   - Update docs source/site output and the `scripts/build_site.py` tag mapping if the opinion title changes.
   - Add a `CHANGELOG.md` Unreleased entry.

## Non-Goals

- No sandbox or permission escalation.
- No command allowlist.
- No persistent shell session.
- No PTY or streaming output.
- No platform-specific shell abstraction.
- No automatic production-mode detection.
- No string-selectable built-in registration.
- No mutation of the existing filesystem tools.
- No command-level filesystem containment beyond cwd validation.

## Verification

Tests:

- `BashTool.spec()` exposes `bash` with the expected description and strict args.
- A successful command returns `ok=True`, exit code `0`, stdout, stderr metadata, and resolved cwd.
- A non-zero command returns `ok=False` with the non-zero exit code and captured stderr.
- A timeout returns `ok=False`, `timed_out=True`, and timeout metadata.
- `cwd` cannot escape the workspace.
- Nonexistent and file-typed `cwd` values return clean tool failures.
- Output exceeding `max_chars` or the constructor/default cap is truncated with metadata.
- `BashTool.spec().sequential is True`.
- A mixed batch containing `bash` runs sequentially.
- `bash` is available through explicit custom tool registration.
- `builtin_tools=["bash"]` fails as an unknown built-in.
- `bash` is not included in default `HarnessConfig()` built-ins.
- A named subagent cannot opt into `builtin_tools=["bash"]`.

Commands:

```bash
uv run pytest tests/test_bash_tool.py tests/test_harness.py tests/test_parallel_tools.py tests/test_subagents.py
uv run ruff check thinharness/tools/bash.py tests/test_bash_tool.py
uv run pyright
uv run python scripts/build_site.py
```

## Expected Size

- Runtime module: 120-180 LOC.
- Integration: 15-30 LOC.
- Tests: 120-200 LOC.
- Docs: 20-40 LOC.

Expected total: 275-450 LOC including tests and docs.
