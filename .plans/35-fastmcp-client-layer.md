# FastMCP client-layer migration — plan v2

Replace ThinHarness's home-grown MCP connection lifecycle with `fastmcp.Client`. Keep ThinHarness responsible for selecting MCP tools and converting them to `ToolSpec` objects. This revision incorporates the single multi-review round recorded in `.reviews/plans/fastmcp-client-layer/`.

This plan does not migrate Semley. It adds the in-process transport support Semley needs so that migration can follow without changing Semley's MCP architecture.

## Goal

ThinHarness currently uses the official `mcp` SDK directly. `thinharness/tools/mcp.py` owns a reference-counted background `ClientSession`, cancellation and shutdown, and three wrappers around the SDK's stdio, SSE, and Streamable HTTP stream factories. The module is 398 lines, with about half devoted to work already provided by the FastMCP client.

After this migration:

- FastMCP owns MCP transport execution, `ClientSession` construction, initialization, and shared-session reference counting. ThinHarness keeps only the small boundary needed to preserve its existing cancellation and bounded-shutdown contract.
- ThinHarness owns tool filtering, name normalization, schema cleanup, `ToolSpec` creation, retry envelopes, tracing metadata, and harness integration.
- Applications can pass an explicit FastMCP `ClientTransport`, including `FastMCPTransport` for an MCP server in the same Python process.
- Existing `MCPServerStdio`, `MCPServerSSE`, and `MCPServerStreamableHTTP` callers continue to work.
- The base `thinharness` import still works without the `mcp` extra.

## Public interface

`MCPServer` becomes a concrete ThinHarness class that accepts a FastMCP client transport as its first argument:

```python
from fastmcp.client.transports import FastMCPTransport
from thinharness import Harness, HarnessConfig, MCPServer

theodosia_server, _upstream, _persister = mount_surface(surface)

harness = Harness(HarnessConfig(
    root=".",
    mcp_servers=[
        MCPServer(
            FastMCPTransport(theodosia_server),
            id="semley",
            include_tools=["step", "reset_session"],
        )
    ],
))
```

The pieces stay explicit:

- `FastMCPTransport` is a FastMCP class. It decides how the client connects to the in-process server.
- `MCPServer` is a ThinHarness class. It decides which discovered tools enter the harness and how their results become `ToolResult` envelopes.
- `HarnessConfig.mcp_servers` remains a list of ThinHarness `MCPServer` objects.

The base class accepts only a FastMCP `ClientTransport`, not a URL, script path, arbitrary server object, or MCP configuration dictionary. Callers choose the transport explicitly. ThinHarness does not copy FastMCP's transport inference interface. Reject other values at construction with a clear `TypeError` once the optional dependency is available.

For the generic class, derive the default ID deterministically from the transport class name using ThinHarness's existing ID normalization. Callers should set `id` when trace attribution needs a domain name. Duplicate IDs keep the existing `-2`, `-3` suffix behavior. The three compatibility classes keep their current command- or URL-derived IDs. Add tests for punctuation-only class names and collisions so the default cannot become empty or unstable.

### Existing compatibility classes

Keep the current constructors and exports:

```python
MCPServerStdio(command, args, env=..., cwd=..., ...)
MCPServerSSE(url, headers=..., ...)
MCPServerStreamableHTTP(url, headers=..., ...)
```

Each class becomes a small wrapper that builds the matching FastMCP transport when the connection opens. Construction remains cheap and does not import the optional MCP dependencies. `MCPServerStdio` sets `keep_alive=False` so the final context exit terminates the child process, matching current ThinHarness ownership. The same wrapper remains reconnectable: entering it after a final exit starts a new subprocess. Do not add `MCPServerFastMCP`; callers compose `MCPServer(FastMCPTransport(server), ...)` directly.

## Dependency decision

Keep MCP optional. Change the `mcp` extra to include both direct dependencies used by ThinHarness:

```toml
mcp = [
    "mcp>=1.24.0,<2.0",
    "fastmcp-slim[client]==3.4.4",
]
```

`fastmcp-slim[client]` provides the client and transport layer without FastMCP's web-server stack. Pin it to the exact version characterized by ThinHarness because FastMCP permits compatibility-affecting changes between minor releases. Upgrade it deliberately with the focused transport and lifecycle suite. Its client extra currently requires `mcp>=1.24.0,<2.0`, so align ThinHarness's direct MCP range with that contract.

An application that creates a modern `fastmcp.FastMCP` server installs the full `fastmcp` package itself. Semley already receives the full package through Theodosia. The slim client can still connect to normal stdio/HTTP servers and to the official SDK's legacy `mcp.server.fastmcp.FastMCP` server when that server package is present.

Keep direct `mcp` in the extra because ThinHarness imports MCP protocol types and errors itself; do not rely on a transitive dependency.

## Behavior to preserve

The refactor must preserve these caller-visible behaviors:

- Connections open lazily on `Harness.connect()` or the first run and stay open until `Harness.aclose()`.
- Multiple runs on one harness reuse one MCP session and one discovered tool snapshot.
- Reusing the same `MCPServer` object in a parent and child harness shares the FastMCP client through its reference-counted context manager. Distinct wrappers remain distinct sessions even if they use the same transport settings.
- A partial multi-server connection failure closes servers that were already opened.
- `include_tools` and `exclude_tools` match original MCP names before prefixing or normalization.
- `tool_prefix`, schema cleanup, normalized-name collision checks, and cross-harness tool collision checks do not change.
- MCP tools remain ordinary `ToolSpec` objects with `kind="mcp"` and `McpToolInfo(server_id, tool_name)` attribution.
- Successful `structuredContent` remains a JSON string. Text, image, audio, embedded-resource, and resource-link blocks keep their current conversion and ordering.
- A protocol-level tool error remains a failed, retryable `ToolResult` with `error_type="MCPToolError"` and `retry=True`.
- Transport/protocol failures remain failed `ToolResult` values with `error_type="MCPError"` when they occur during a tool call. Programming errors still propagate.
- MCP tools and connection details do not enter resume state.
- Subagents inherit MCP servers only through the existing explicit `inherit_mcp_servers` and `mcp_servers` settings.
- Tracing continues to read MCP identity from the `ToolSpec`, so an after-tool hook cannot erase attribution.
- Importing `thinharness` and constructing the three compatibility wrappers still works without the `mcp` extra. Opening a connection without the extra raises `MCPDependencyError` with the existing install hint.

FastMCP's disconnect machinery replaces ThinHarness's private background session runner, but not its observable shutdown contract. Final close remains bounded. If FastMCP suppresses a `CancelledError` while cleaning up, the ThinHarness boundary detects pending caller cancellation after cleanup and re-raises it. Do not recreate a second session runner or reference counter to achieve this.

## Implementation steps

### 1. Record the current behavior before changing the client layer

- Run the focused MCP unit tests before editing.
- Record the exact pass count and every existing skip for `tests/unit/test_mcp.py`, `tests/unit/test_mcp_optional_dependency.py`, and the deterministic MCP end-to-end test.
- Treat the current public tests listed under "Behavior to preserve" as characterization tests. Change a test only when it asserts a private implementation that the refactor deletes.
- Add characterization coverage before the refactor for currently untested conversion paths: `structuredContent`, image, audio, embedded resource, resource link, mixed-block ordering, protocol tool errors, transport drops, and programming errors. Make those tests pass against the current implementation before replacing it.

Verify: `uv run pytest tests/unit/test_mcp.py tests/unit/test_mcp_optional_dependency.py` passes before implementation.

### 2. Add FastMCP to the optional extra

- Use `uv add --optional mcp 'fastmcp-slim[client]==3.4.4' 'mcp>=1.24.0,<2.0'` so the lockfile and `pyproject.toml` record the reviewed contract.
- Confirm the resolved versions support Python 3.11, the repository's minimum Python version.
- Keep FastMCP imports lazy in `thinharness/tools/mcp.py` so the base package import does not require the extra.
- Update `MCPDependencyError` handling so a missing `fastmcp` import gives the same `pip install thinharness[mcp]` hint as a missing `mcp` import.
- Test missing `mcp` and missing `fastmcp` independently; one broad import failure test cannot identify which optional dependency path regressed.

Verify: the lockfile contains compatible `fastmcp-slim` and `mcp` versions; a subprocess with the optional packages hidden can still `import thinharness` and construct each compatibility wrapper.

### 3. Make `MCPServer` own a FastMCP client

- Remove `ABC`, `_SessionState`, `_client_streams()`, `_session_runner()`, `_client_session_type()`, and the ThinHarness-owned background-task lifecycle.
- Make `MCPServer` accept one FastMCP `ClientTransport` and store it without importing FastMCP at module import time.
- Add one private lazy client accessor. On first connection it constructs `fastmcp.Client(transport, timeout=read_timeout, init_timeout=timeout)` and reuses that client for the life of the `MCPServer` object.
- Delegate `MCPServer.__aenter__` to the FastMCP client and keep returning the ThinHarness `MCPServer` so existing harness code does not change.
- Delegate normal final exit to FastMCP, but preserve ThinHarness's caller-cancellation contract at the boundary: complete cleanup within the existing bounded window and re-raise pending `CancelledError` if FastMCP consumed it. Keep this wrapper about cancellation only; do not duplicate FastMCP's connection state.
- Keep `list_tools()` and `call_tool()` safe for standalone use by entering `self` around the operation. FastMCP's reentrant client then shares the already-open harness connection.
- State the ownership rule in the class docstring: once a transport is passed to `MCPServer`, that wrapper owns the FastMCP client and closes its transport. Do not reuse one stateful transport object in several different `MCPServer` wrappers; reuse the same wrapper when a session should be shared.
- Keep current ID resolution inside ThinHarness. Do not use FastMCP's generated client name as trace identity.

The pinned FastMCP `Client` already implements locked, reference-counted, reentrant contexts. Do not add another ThinHarness reference counter. Treat that implementation detail as a dependency risk and prove the required behavior through ThinHarness tests rather than private FastMCP state.

Verify: one wrapper enters one FastMCP client connection; nested and concurrent entries reuse it; the final exit closes it; the same wrapper can reconnect after final exit; cancellation during first connection and final close is bounded and propagates cancellation.

### 4. Rebuild the compatibility classes on FastMCP transports

- `MCPServerStdio` builds `fastmcp.client.transports.StdioTransport` from the existing command, args, environment, and working-directory fields, with `keep_alive=False`.
- `MCPServerSSE` builds `SSETransport` from the existing URL and headers.
- `MCPServerStreamableHTTP` builds `StreamableHttpTransport` from the existing URL and headers.
- Preserve `timeout` as both the MCP initialization limit and the HTTP connection limit. Preserve `read_timeout` as the MCP request, HTTP read, and SSE read limit. Where FastMCP's constructor cannot express separate connect/read values directly, supply its `httpx_client_factory` with an `httpx.Timeout` carrying those values rather than collapsing them into one timeout.
- Pass `env`, `cwd`, and `headers` without reinterpretation. Preserve the current rule for whether a supplied stdio environment augments or replaces the process environment.
- Preserve the current derived IDs exactly.
- Build these transports lazily on first connection so compatibility-wrapper construction remains available without the optional extra.

Verify: current constructor tests remain valid; timeout tests stall connection, initialization, SSE reads, and tool calls separately; the deterministic stdio smoke test lists and calls a tool through FastMCP, observes the child exit on final close, and reconnects with a new child.

### 5. Keep raw MCP tool semantics at the ThinHarness conversion layer

- Use `await client.list_tools()` for discovery. In the pinned API it returns a bare `list[mcp.types.Tool]`, not a result object with a `.tools` attribute.
- Use `await client.call_tool_mcp(name, arguments)` for execution. This returns the raw `mcp.types.CallToolResult`, avoiding FastMCP's hydrated `.data` values and default exception-on-tool-error behavior.
- Keep ThinHarness's existing filter, prefix, schema cleanup, content parsing, and metadata functions.
- Keep `isError=True` mapped to the existing retry envelope.
- Enumerate the pinned dependency's actual failure shapes before editing the exception boundary: MCP protocol errors, FastMCP tool errors, `httpx` transport errors, AnyIO stream closure, timeouts, and exception groups or cause chains containing those errors. Normalize only a known MCP or transport failure. Do not catch a bare `RuntimeError` or `Exception`; a wrapper error is normalized only when its cause chain contains a known transport or protocol failure, so programming bugs still propagate.
- Delete only helpers made obsolete by FastMCP. Do not move ThinHarness policy into FastMCP transforms during this migration.

Verify: parameterized raw-result tests cover `structuredContent`, every supported content-block type, mixed ordering, `isError=True`, a real transport drop, a protocol error, and an injected programming error with the same observable results as before.

### 6. Replace lifecycle fakes with behavior-level integration tests

- Remove tests that directly exercise `_SessionState`, `_session_runner`, or `_client_streams`; those would test deleted implementation.
- Replace the existing `FakeMCPServer` subclass throughout the suite, not only in lifecycle tests: it depends on private methods that disappear. Use small behavior-level fixtures based on the new public constructor, and keep local doubles only for ThinHarness-owned conversion and exception policy.
- Test the public in-process interface with both server families supported by `FastMCPTransport`: modern `fastmcp.FastMCP` and the official SDK's legacy `mcp.server.fastmcp.FastMCP`.
- Make the Semley-shaped integration use modern `fastmcp.FastMCP`, matching Theodosia's current server family. Its server exposes `step`, `reset_session`, and one excluded tool; `MCPServer(FastMCPTransport(server), include_tools=["step", "reset_session"])` discovers only the governed pair and calls `step` through the harness.
- Test connection reuse through observable server lifespan/tool-call state, not private client counters.
- Keep pure unit tests for ThinHarness-owned decisions such as filtering, name normalization, schema cleanup, metadata, and ID collision resolution.
- Keep the stdio integration test because it proves a separate real transport. Do not add tests that only assert ThinHarness instantiated a particular FastMCP class.
- Preserve harness-level tests for partial connect cleanup, close after tool failure, close after cancellation, resume, approvals, tracing, and subagent inheritance.
- Add targeted regressions for nested entry, concurrent entry of the same wrapper, final-exit cancellation, reconnect after final exit, invalid generic transport input, default generic IDs, stdio process exit, and each independent optional dependency failure.

Verify: each retained test would fail if a caller-visible requirement broke; no new test proves only FastMCP's own behavior or asserts private helper order. The new suite does not reach into FastMCP's reference counters or background tasks.

### 7. Update behavior contracts and documentation

After this plan review and before implementation, add an MCP section to `docs/behavior.md` covering:

- lazy connection and bounded cleanup;
- one discovered tool snapshot per harness connection;
- explicit FastMCP transport input and ownership;
- cancellation propagation, reconnectability, and stdio child-process ownership;
- filter/prefix/collision behavior;
- retry and error envelopes;
- explicit subagent inheritance and resume-state exclusion.

Update `docs/docs.md` and the MCP feature text in `README.md`:

- show the generic `MCPServer(FastMCPTransport(server), ...)` interface;
- retain one compatibility-wrapper example;
- explain that `thinharness[mcp]` installs the FastMCP client layer;
- name the exact FastMCP client version and the deliberate-upgrade policy;
- keep prompts, resources, sampling, OAuth, provider-native MCP, and configuration discovery out of ThinHarness's supported surface;
- remove wording that says ThinHarness itself provides the transport implementations.

Add a `CHANGELOG.md` entry naming the new in-process capability and confirming compatibility for the existing constructors.

Verify: documentation examples import real exported names and match tested constructor forms.

### 8. Run the full verification set

Run:

```bash
uv run pytest tests/unit/test_mcp.py tests/unit/test_mcp_optional_dependency.py
uv run pytest
uv run ruff check thinharness/tools/mcp.py thinharness/core.py thinharness/subagents.py tests/unit/test_mcp.py tests/unit/test_mcp_optional_dependency.py tests/e2e/mcp_journey.py
uv run pyright
```

Run the deterministic MCP end-to-end journey if it no longer requires model credentials. If the existing model-backed journey still requires credentials, run it only when the required key is present and report it separately; do not claim it passed when it skipped.

Verify: all required commands exit successfully, no test is skipped unexpectedly, and the final diff removes more MCP lifecycle/transport code than it adds.

## Success criteria

- Semley's in-process Theodosia server can be configured through `MCPServer(FastMCPTransport(server), include_tools=["step", "reset_session"])` without HTTP, stdio, or a manual `ToolSpec` bridge.
- Existing stdio, SSE, and Streamable HTTP ThinHarness configuration remains source-compatible.
- FastMCP owns connection and session lifecycle; ThinHarness no longer contains a second reference-counted MCP session implementation.
- Final close propagates caller cancellation, terminates stdio children, and leaves the wrapper reconnectable.
- ThinHarness's tool filtering, conversion, retry, tracing, resume, and subagent behavior remains unchanged.
- The base install stays independent of MCP dependencies.
- Focused tests, the full test suite, Ruff, and Pyright pass with no unreported skips.

## Out of scope

- Migrating Semley itself or writing its pull request.
- Exposing FastMCP server objects directly; callers use `FastMCPTransport`.
- Accepting URL strings, script paths, `.mcp.json`, or multi-server dictionaries in `MCPServer`.
- Passing a prebuilt `fastmcp.Client`; the first interface accepts `ClientTransport` only.
- MCP prompts, resources, sampling, elicitation, OAuth, task execution, server instructions, or provider-native MCP.
- Dynamic `notifications/tools/list_changed` handling.
- Removing the three existing compatibility classes.

## Review record

Exactly one multi-review round was run against plan v1. Codex, Claude, and GLM reviews are stored in `.reviews/plans/fastmcp-client-layer/`. This v2 accepts their shared findings on version pinning, timeout mapping, cancellation, stdio ownership, public API characterization, optional-dependency coverage, and the scope of test-double replacement. It does not adopt the suggestion to retain ThinHarness's own reference-counted session state: the pinned FastMCP client already provides locked reentrant reference counting, and duplicating it would preserve the very lifecycle layer this migration is intended to remove. No second review round is part of this task.
