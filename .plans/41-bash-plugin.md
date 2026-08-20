# Bash plugin — plan v2

Replace the directly registered `BashTool` with an explicit `BashPlugin`. The plugin provides one bounded, non-interactive Bash command tool. It uses the canonical harness root and does not inherit into child harnesses.

This is a clean pre-1.0 break. Do not keep compatibility exports, aliases, fallback registration, or two public Bash interfaces.

## Resolved decisions

1. **Bash is an explicit plugin.** A plain harness has no Bash tool. Callers enable it with `plugins=[BashPlugin()]`.
2. **Bash stays one-shot.** Every call starts a fresh `bash -c` process. Shell variables, functions, aliases, and `cd` do not persist between calls.
3. **The tool is non-interactive.** It has no stdin, PTY, background mode, job tools, or live output stream.
4. **The plugin uses the canonical root.** `BashPlugin` receives `PluginContext.root`; callers cannot configure a second root. A model-selected cwd must resolve under that root.
5. **Cancellation stops the command.** Cancelling the task that awaits `Harness.run()` signals the command's process group, completes bounded cleanup after process handoff despite repeated cancellation requests, and then propagates cancellation. Cancellation during process creation waits for handoff to settle so no process can be orphaned.
6. **Timeouts stop the process group.** The effective timeout covers shell execution after process creation. Process creation is outside this timeout. Timeout sends `SIGTERM`, waits one second, then sends `SIGKILL` if needed. Fixed cleanup periods can make wall-clock time exceed `timeout_seconds`.
7. **Background cleanup is best effort.** After the direct shell exits, the plugin terminates remaining processes in the captured process group before it returns. A process that creates a new session can escape this cleanup, and the operating system can reuse a process-group ID after the shell exits.
8. **Output is bounded while the command runs.** The implementation reads stdout and stderr concurrently in fixed-size chunks into separate bounded head-and-tail buffers. It does not use line reads or write full output to temporary spill files.
9. **The environment is minimal by default.** The process inherits a small host allowlist plus non-interactive defaults. Full environment inheritance is an explicit host choice. Model arguments cannot set environment variables.
10. **The host controls limits.** Plugin configuration owns default timeout, maximum timeout, and per-stream output bytes. The model may request a timeout but cannot exceed the plugin maximum.
11. **Bash is sequential.** A tool batch containing Bash runs in model order without overlapping sibling calls.
12. **Child access is explicit.** `BashPlugin` does not implement `for_child()`. A child that needs Bash must list its own `BashPlugin`, and approval-required Bash remains invalid in children under existing child rules.
13. **Approval is optional.** `BashPlugin(requires_approval=True)` uses the existing top-level approval pause and resume flow. Approval is not a shell-specific prompt or UI.
14. **Local Bash is not a sandbox.** Cwd containment and environment filtering do not restrict absolute paths, network access, host credentials stored in files, or other host authority.
15. **No executor interface yet.** Keep local process execution private until ThinHarness has a real second executor such as a container or remote sandbox.
16. **POSIX only.** The plugin runs `bash` from `PATH` and depends on POSIX process groups. It does not discover Windows Bash installations or fall back to another shell.

## Public interface

Add `thinharness/plugins/bash.py` and export `BashPlugin` from `thinharness.plugins` and `thinharness`:

```python
from thinharness import BashPlugin, Harness, HarnessConfig

harness = Harness(
    HarnessConfig(root="."),
    plugins=[BashPlugin()],
)
```

The constructor is keyword-only:

```python
BashPlugin(
    *,
    default_timeout: float = 30,
    max_timeout: float = 120,
    max_output_bytes: int = 40_000,
    inherit_env: bool = False,
    env: Mapping[str, str] | None = None,
    requires_approval: bool = False,
)
```

Validate configuration when the plugin is constructed:

- `default_timeout` and `max_timeout` must be real `int` or `float` values, not booleans, and must be finite and greater than zero;
- `default_timeout` must not exceed `max_timeout`;
- `max_output_bytes` must be an `int`, not a boolean, and must be greater than zero;
- `inherit_env` and `requires_approval` must be real booleans;
- every environment name and value must be a string;
- environment names must be non-empty and contain neither `=` nor NUL, and values must not contain NUL;
- copy `env` so later caller mutation cannot change a binding.

The plugin name is fixed to `"bash"`. Its configuration is frozen after construction, matching the other first-party plugins. `bind()` performs no filesystem or process I/O and returns one static `ToolSpec` with `ToolOrigin(plugin="bash", source="bash")` after normal plugin composition. The contribution adds no plugin instructions; the tool description contains the model-facing execution rules.

Remove `BashTool` and `BashArgs` from `thinharness.tools` and top-level exports. The replacement argument model is private to `thinharness.plugins.bash`; `BashPlugin` is the only public Bash interface. Remove the public direct-registration form. Keep all process helpers inside `thinharness/plugins/bash.py`; the existing synchronous process users do not justify a shared executor seam.

## Model-facing tool

The tool name is `bash`. Use a private Pydantic argument model based on `StrictArgs`. Here, strict means both `extra="forbid"` and no value coercion: `command` and `cwd` accept only strings, while `timeout` accepts only real integers or floats, rejects booleans and non-finite values, and must be greater than zero.

```python
class _BashArgs(StrictArgs):
    command: str = Field(min_length=1, strict=True)
    cwd: str = Field(default=".", strict=True)
    timeout: float | None = None
```

Use field validators for the numeric rules and for `command`. Reject a command that contains only whitespace during Pydantic validation so the model receives the existing retryable validation result. Do not strip or otherwise change a valid command.

The tool description must state:

- it runs one non-interactive Bash command;
- cwd must be inside the workspace;
- calls do not share shell state;
- same-process-group background processes are terminated on a best-effort basis;
- `timeout` is in seconds and is capped by the host.

Do not expose output limits, environment variables, stdin, shell selection, login mode, background mode, sandbox permissions, or approval in the model schema.

If the model omits `timeout`, use `default_timeout`. If it requests more than `max_timeout`, clamp the effective timeout to `max_timeout` and report the effective value in result metadata.

## Process execution

Use a native async handler and `asyncio.create_subprocess_exec()` rather than a synchronous handler in `asyncio.to_thread()`. The runner owns the stdout and stderr pipe read ends and their asyncio read transports instead of relying on `Process.stdout` and `Process.stderr`. Pass the corresponding write descriptors to `create_subprocess_exec()`. This ownership lets cleanup close the parent read transports when an escaped descendant keeps a write end open.

Execution rules:

- reject unsupported platforms before creating pipes or spawning, through a private capability check that requires `os.name == "posix"` and `os.killpg`; return `UnsupportedPlatform` otherwise;
- resolve cwd under `PluginContext.root` and require an existing directory;
- call `bash -c` with the resolved cwd, filtered environment, `stdin=DEVNULL`, owned stdout and stderr pipe write descriptors, and `start_new_session=True`;
- return a structured `ProcessStartError` if Bash or the process cannot start;
- close every pipe descriptor if pipe setup or process creation fails;
- make process creation a cancellation-safe handoff: keep the spawn task referenced, and if cancellation arrives during `create_subprocess_exec()`, tolerate repeated cancellation while waiting for the spawn task to settle; if it returns a process, use `process.pid` as the process-group ID created by `start_new_session=True`, signal it, complete bounded cleanup, and then propagate cancellation; never leave a spawn or cleanup task detached;
- after handoff, capture `process.pid` as the process-group ID and start one reader task for stdout and one for stderr before waiting for process completion;
- readers use fixed-size `read(n)` chunks, never line-oriented reads;
- the effective timeout starts after process handoff and covers waiting for the direct shell;
- on normal shell exit, attempt to terminate remaining members of the captured process group, then run bounded final drain;
- on timeout, terminate the captured process group, then run bounded final drain and return a timeout result;
- on `asyncio.CancelledError`, synchronously send the first group signal, then run one referenced cleanup task through a re-cancellation-tolerant await loop with a fixed deadline; after cleanup, re-raise cancellation;
- group termination sends `SIGTERM`, waits one second, then sends `SIGKILL` if the group still exists;
- final drain waits up to one second for both readers, then closes both parent read transports, cancels and joins both readers, and returns the partial output;
- close and join every owned transport, descriptor, reader task, spawn task, and cleanup task on every exit path;
- do not expose spawn, termination, or drain cleanup periods as public configuration in this slice.

Process-group cleanup is best effort. It cannot stop a descendant that deliberately creates a new session or otherwise leaves the process group. Signalling a captured group after the direct shell has been reaped also has an unavoidable process-group-ID reuse race. State both limits without calling the tool isolated or promising that all descendants are stopped.

## Environment policy

When `inherit_env=False`, copy only these values when present:

- `PATH`;
- `HOME`;
- `TMPDIR`, `TMP`, and `TEMP`;
- `LANG`, `LC_ALL`, and `LC_CTYPE`;
- `TZ`.

When `inherit_env=True`, start from a copy of `os.environ`.

For both modes:

1. remove inherited `BASH_ENV` and `ENV`;
2. set `NO_COLOR=1`, `TERM=dumb`, `PAGER=cat`, and `GIT_PAGER=cat`;
3. apply the explicit host `env` mapping last, so the host can deliberately replace any value.

This policy reduces accidental environment-secret exposure. It is not a security boundary because commands can still read host files and use other credential sources.

## Output and results

Keep stdout and stderr separate. Each stream gets its own `max_output_bytes` budget.

The private buffer must:

- retain `ceil(max_output_bytes / 2)` bytes from the start and `floor(max_output_bytes / 2)` bytes from the end;
- when `max_output_bytes == 1`, retain the first byte and no tail;
- discard the middle after the cap is reached;
- keep retained stream bytes at or below `max_output_bytes`, not total command output;
- continue draining discarded bytes so the child cannot block on a full pipe;
- decode UTF-8 with replacement after collection;
- for truncated output, insert `\n... {omitted_bytes} bytes omitted ...\n` between the decoded head and tail; the rendered marker does not count against the retained-byte budget;
- return whether truncation occurred and the total bytes seen.

Return model content with labelled `stdout` and `stderr` sections. Preserve partial output for non-zero exits and timeouts. If both streams are empty, return only `(no output)`. If one stream is empty, keep both labelled sections and put `(no output)` in the empty section.

Result metadata contains:

- `exit_code`;
- `signal` when the return code represents a signal;
- `timed_out`;
- `duration_seconds`, measured from the start of spawn handoff through final cleanup;
- resolved `cwd`;
- effective `timeout_seconds`, which covers direct-shell execution after spawn handoff and does not include fixed cleanup periods;
- `stdout_bytes` and `stderr_bytes`;
- `stdout_truncated` and `stderr_truncated`.

Result rules:

- exit code zero returns `ok=True`;
- a non-zero exit returns `ok=False` with `error_type="NonZeroExit"`;
- timeout returns `ok=False` with `error_type="Timeout"`;
- cwd, platform, and startup failures return their specific error type;
- command outcomes preserve their output and do not request a model retry;
- argument validation keeps the existing retryable validation behavior;
- run cancellation propagates and does not become a tool result.

## Behavior contract changes before implementation

After plan review and before implementation, add a Bash Plugin section to `docs/behavior.md` that records:

- explicit plugin composition and no implicit Bash;
- the fixed plugin and tool names;
- canonical root and contained cwd;
- fresh non-interactive calls with no supported background persistence and the documented best-effort cleanup limits;
- timeout, run cancellation, best-effort process-group cleanup, bounded final drain, and cancellation propagation;
- bounded separate head-and-tail output with the fixed split and truncation marker and no spill files;
- minimal versus inherited environment policy;
- sequential execution;
- optional top-level approval;
- no automatic child inheritance;
- POSIX-only support and no shell fallback;
- the fact that local Bash is not a sandbox.

Update only the affected plugin and Bash behavior. Do not change unrelated sections.

## Implementation steps

1. Update the affected behavior contract after plan review.
2. Add the frozen `BashPlugin` configuration and static binding.
3. Replace the synchronous subprocess implementation with the private async process runner, owned pipes, and cancellation-safe spawn handoff.
4. Add bounded per-stream head-and-tail collection and bounded final drain.
5. Add process-group timeout, repeated-cancellation-safe cleanup, normal-exit descendant cleanup, and environment policy.
6. Move Bash registration to the plugin and remove the public `BashTool` interface and exports.
7. Replace `tests/unit/test_bash_tool.py` with plugin, process, output, environment, approval, and inheritance tests.
8. Add architecture guards that keep Bash construction out of core and prevent `BashPlugin` from implementing child inheritance.
9. Update README, reference docs, generated site content, examples, and changelog.
10. Add a live Bash plugin journey and run all validation.

## Tests

Add focused coverage for:

### Plugin composition

- no plugin means no Bash tool;
- `BashPlugin()` contributes one static sequential tool with fixed name and origin;
- binding uses the canonical root and does not create it;
- invalid or mutable constructor values cannot change later bindings;
- string and integer Boolean options, Boolean numeric limits, fractional output limits, and invalid environment entries fail at construction;
- normal duplicate plugin and tool collisions fail;
- direct `BashTool` and `BashArgs` imports and exports are gone;
- the plugin does not inherit into default or named children;
- an explicit child `BashPlugin` works when approval is off;
- approval-required Bash works at the top level, requires a resumable top-level model at harness construction, and fails under existing child approval rules.

### Process lifecycle

- success, non-zero exit, signal exit, timeout, and startup failure;
- timeout escalates from TERM to KILL when the command ignores TERM;
- cancelling an active harness run kills the process group and propagates cancellation promptly;
- cancelling a second time during cleanup does not detach cleanup or leave the process group running;
- cancellation paused during process creation still completes spawn handoff, group cleanup, and cancellation propagation;
- normal shell exit cleans up a same-group background descendant;
- a new-session descendant that holds an output pipe cannot keep normal exit, timeout, or cancellation from returning after bounded drain;
- stdout and stderr are drained concurrently with chunk reads, including simultaneous floods larger than 64 KiB with no newlines;
- unsupported platforms fail before pipe creation or spawn through the private capability-check seam;
- missing and non-directory cwd values fail cleanly;
- cwd cannot escape root through absolute paths, `..`, or symlinks.

### Output

- small stdout and stderr remain separate and unchanged;
- large streams use the defined head/tail split and marker, report total bytes, and keep the marker outside the retained-byte budget;
- one-byte and odd output limits follow the defined split;
- output floods stay within the configured retained-memory bound and create no spill file;
- split multibyte and invalid UTF-8 decode with replacement;
- timeout and non-zero results keep partial output;
- empty output returns `(no output)`.

### Environment

- minimal mode keeps only the named host values plus non-interactive defaults;
- full inheritance is explicit;
- explicit host values override inherited and default values;
- provider API keys are absent in minimal mode;
- inherited `BASH_ENV` and `ENV` are not sourced unless the host explicitly supplies them;
- model arguments cannot provide environment variables;
- string and Boolean model values do not coerce into `timeout`, and whitespace-only commands produce retryable validation results.

### Limits and integration

- default timeout, per-call timeout, and maximum clamping;
- invalid plugin limits fail during construction;
- Bash keeps mixed tool batches sequential;
- hooks, tracing, tool-call records, and structured result envelopes use the normal plugin tool path;
- the end-to-end journey uses `BashPlugin` to run a bounded command and verifies that the parent receives its output.

## Documentation and caller migration

Update:

- `README.md` opinion, feature list, and usage example;
- `docs/docs.md` plugin and Bash sections;
- the hand-written `docs/site/index.html` Bash text;
- README-derived `docs/site/about/index.html` through `scripts/build_site.py`;
- `scripts/build_site.py` opinion-tag mapping if the `No bash by default` heading changes;
- `tests/e2e/README.md` with the new journey;
- `CHANGELOG.md` with the breaking removal of public `BashTool` and `BashArgs` and the new environment, cancellation, and output behavior;
- top-level and plugin exports;
- every tracked test, example, and document that uses `BashTool(...).spec()`.

`READMEV2.md` is an untracked working draft, not a product document for this slice. Do not modify or add it as part of the implementation. `docs/site/explainer/index.html` has no Bash-specific text and is also out of scope unless the implementation changes a statement in that file.

Use this replacement:

```python
# Remove
tools=[BashTool(root=".").spec()]

# Add
plugins=[BashPlugin()]
```

Do not add a compatibility wrapper or deprecation period.

## Validation

Run:

```bash
uv run pytest tests/unit/test_bash_plugin.py tests/unit/test_plugins.py tests/unit/test_harness.py tests/unit/test_subagents.py tests/unit/test_approvals.py tests/unit/test_parallel_tools.py tests/unit/test_architecture.py
uv run pytest tests/unit/test_streaming.py tests/unit/test_tracing.py tests/unit/test_tool_retry.py
uv run pytest
uv run ruff check .
uv run pyright
uv run scripts/build_site.py
uv run scripts/build_site.py --check
uv run --env-file .env python tests/e2e/bash_plugin_journey.py
git diff --check
```

A credential-based end-to-end skip is not a pass. Report it separately.

## Success criteria

- Bash is available only through explicit `BashPlugin` composition.
- Core does not import or construct Bash behavior.
- The plugin uses the canonical root and does not create it during construction or binding.
- Timeout and run cancellation perform bounded best-effort process-group cleanup; repeated cancellation does not detach cleanup.
- Normal exit, timeout, and cancellation return after bounded final drain even when an escaped descendant holds a pipe open.
- Output memory stays bounded while stdout and stderr are drained with chunk reads.
- Minimal environment mode does not expose unrelated host environment variables.
- Bash remains one-shot, non-interactive, sequential, and unavailable to children unless configured there explicitly.
- No background-job interface, persistent shell, PTY, output stream, spill file, shell fallback, command policy, sandbox, or public executor interface is added.
- Focused tests, the full suite, Ruff, Pyright, generated-site checks, and the live journey pass.

## Out of scope

- Background command management.
- Persistent shell state or cwd.
- PTYs, interactive programs, or stdin.
- Live tool-output streaming.
- Windows shell support or shell discovery.
- Command allowlists or denylists.
- Filesystem, network, syscall, or credential sandboxing.
- Remote, container, VM, or provider-native execution.
- Model-selected environment, shell, output limits, or sandbox permissions.
- Full-output spill files.
- Automatic child inheritance.
- A public process or executor interface.
- Compatibility aliases or migrations.
