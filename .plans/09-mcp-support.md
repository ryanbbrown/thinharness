# Plan: MCP Support (v4)

This revision supersedes v3 (in this file) and folds in `plan-mcp-support-feedback-v3.md`. Written against commit `321bfd4 feat: add conversation resume support`. v1–v3 content is in git history; the v1–v3 feedback docs (`plan-mcp-support-feedback-v{1,2,3}.md`) remain alongside.

## What changed since v3
Decisions taken (user-confirmed):
- **Subagent MCP API restructured.** Drop `mcp_servers: list[MCPServer] | None = None`. Replace with two orthogonal fields:
  - `inherit_mcp_servers: bool = False` — symmetric with the existing `inherit_parent_tools`. When `True`, the child enters each of the parent's MCP server objects via ref-counted `enter_async_context`, reusing the parent's open sessions.
  - `mcp_servers: list[MCPServer] = []` — an additive override list. Identity-deduped against the inherited set (same Python object appearing in both is folded once; different objects with the same `id` field are treated as separate and will trip the existing tool-name collision check).
  Validator rule: a named subagent passes if any of {`builtin_tools` non-empty, `tools` non-empty, `inherit_parent_tools=True`, `inherit_mcp_servers=True`, `mcp_servers` non-empty} holds. `SubAgentConfig(name="x", description="...")` now correctly fails validation again.

Mechanical fixes from v3 feedback:
- **Connect/list failure cleanup (v3 HP-1).** Wrap the connect/discover/validate block in `try/except BaseException: await stack.aclose(); raise` so any partial entry unwinds before the exception propagates.
- **Child config update for MCP (v3 HP-3).** `build_child_harness()` builds the child's `mcp_servers` list explicitly: identity-deduped union of `parent_config.mcp_servers` (if `inherit_mcp_servers=True`) and `child_config.mcp_servers`. Otherwise just `child_config.mcp_servers`. `model_copy(update={"mcp_servers": ...})` always sets the field explicitly so the parent's value never leaks through.
- **Trace attribution from `ToolSpec` (v3 HP-4 / LP-12).** `_traced_call_output` reads `self._tool_map[name].metadata` for `mcp.server.id` / `mcp.tool.name`, not the result envelope. Survives after-tool hooks rewriting the output.
- **No-extra test placement (v3 HP-5).** Module-level `pytest.importorskip("mcp")` on `tests/test_mcp.py` for everything that needs the SDK; move `test_construction_without_extra` to a separate `tests/test_mcp_optional_dependency.py` with no top-level skip.
- **`MCPDependencyError` export (v3 MP-6).** Re-export from `thinharness/__init__.py` alongside the server classes.
- **`run_sync()` one-shot migration (v3 MP-7).** Existing tests that reuse a harness via repeated `run_sync()` calls (notably provider message-leak tests and resume/branching tests) switch to `await harness.run(...)` inside one async lifecycle, or to fresh harness instances per call. This is intentional churn; updating those tests is part of the implementation, not a separate task.
- **Closed-state tests (v3 MP-8).** Tests added for `run()` / `run_sync()` after `aclose()`, `tool_schemas()` after close, and `add_tool()` after close. `add_tool()` after close is intentionally not gated — harmless mutation on a dead object.
- **Consistent MCP result metadata (v3 MP-9).** All `MCPServer.call_tool()` paths return metadata with the stable shape `{source: "mcp", mcp_server_id, mcp_tool_name, error_type?, retry?}`. Success omits `error_type` and `retry`; `isError=True` sets `error_type="MCPToolError"`, `retry=True`; `McpError` sets `error_type="MCPError"` with no `retry`.
- **Stable default `id` (v3 LP-11).** When `id=None`, derive a readable default: `command` for stdio (e.g. `"uvx mcp-server-git"`), `url` for HTTP/SSE. At harness setup, if two servers in `self._mcp_servers` produce the same derived id, append a numeric suffix (`-2`, `-3`, …) for disambiguation.

Skipped per user direction:
- v3 LP-10 (README / docs updates). Skipping for now; can land in a follow-up.

## Overview
Add Model Context Protocol (MCP) client support to thinharness so a `Harness` can be configured with one or more MCP servers and surface their tools alongside built-ins, skills, subagents, and custom tools. Use the official `mcp` Python package via an optional extra (`thinharness[mcp]`); do not reimplement the protocol. Support stdio, SSE, and streamable HTTP.

Each MCP server is a reference-counted async context manager that owns a background `ClientSession` (pydantic-ai's pattern). On first connect, the server is queried for its tool list (a snapshot); tools become `ToolSpec`s appended to `self.tools` / `self._tool_map` and stay there until `aclose()`. No new toolset abstraction — the existing single tool-call path handles MCP tools transparently.

Connections are owned by the `Harness` instance: opened lazily on the first `run()` that needs them (or via explicit `await harness.connect()`), reused across subsequent `run()` calls and resumed runs, and closed in `aclose()`. Mid-run tool list changes (`notifications/tools/list_changed`) are explicitly out of scope for v1.

Scope exclusions for v1: MCP prompts, MCP resources as tools, MCP sampling, OAuth, `.mcp.json` discovery, and `tools/list_changed`.

## References
- `vendor/pydantic-ai/pydantic_ai_slim/pydantic_ai/mcp.py` — closest design match. Reference-counted async context manager with a dedicated background `asyncio.Task` owning the `ClientSession` (anyio cancel-scope discipline). `MCPServerStdio`, `MCPServerSSE`, `MCPServerStreamableHTTP` inherit from an `MCPServer` base. `isError=True` is surfaced as a retryable result; `tool_prefix` namespacing is stripped on dispatch.
- `vendor/openai-agents/src/agents/mcp/server.py` — `_normalize_mcp_name` sanitizer.
- `vendor/strands/src/strands/tools/mcp/mcp_client.py` — prefix + allow/reject filters at connect time.
- `thinharness/tools.py` — `ToolSpec`, `ToolResult`, `_normalize_result`, `_tool_retry_kind`. `metadata.retry is True` drives the harness retry path.
- `thinharness/core.py` — `Harness.__init__` assembles built-ins, skills, subagent tool, custom tools. `Harness.run(prompt, *, resume_from=None, ...)` either starts fresh or calls `model.resume_session(state)`. `aclose()` closes provider HTTP clients when `_owns_model`. `RunStartContext` fires inside the agent span (`core.py:403-410`). `run_sync()` wraps `run()` + `aclose()` in an event loop (`core.py:581-587`).
- `thinharness/providers.py` — `ResumableModel` protocol; sessions implement `continue_with_user_prompt` and `dump_state`. MCP integration is provider-agnostic.
- `thinharness/subagents.py` — `_effective_custom_tools(parent, config)` copies parent tools when `inherit_parent_tools=True`. `run_subagent_tool()` records `effective_tools` before `child.run()` (`subagents.py:125-131`). Named-subagent validator (`subagents.py:54-57`) gains `inherit_mcp_servers` / `mcp_servers` as valid tool sources.
- `thinharness/hooks.py` — `HookRegistry.validate_filters(tool_names=..., agent_names=...)` (`hooks.py:229-237`). v4 removes the tool-name path.

## Steps

### 1. Add the `mcp` Optional Extra; Cheap Construction
- Add `mcp = ["mcp>=1.23.0"]` to `pyproject.toml`'s `[project.optional-dependencies]`.
- `thinharness/mcp.py` imports nothing from `mcp` at module scope. Imports happen inside `__aenter__` / `list_tools` / `call_tool`, wrapped to raise `MCPDependencyError(ImportError)` with `pip install thinharness[mcp]` as the hint. Construction never imports `mcp`.
- Re-export `MCPServer`, `MCPServerStdio`, `MCPServerSSE`, `MCPServerStreamableHTTP`, `MCPError`, and `MCPDependencyError` from `thinharness/__init__.py`.

**Verify**: `import thinharness` succeeds without the extra; `MCPServerStdio(command="x")` succeeds without the extra; `await server.__aenter__()` without the extra raises `MCPDependencyError` with the install hint; `MCPDependencyError` is importable from `thinharness`.

### 2. Define `MCPServer` and Three Transport Subclasses
Create `thinharness/mcp.py`:

```python
class MCPServer(ABC):
    """One MCP server connection that contributes ToolSpecs to a harness."""

    def __init__(
        self,
        *,
        tool_prefix: str | None = None,
        timeout: float = 5.0,
        read_timeout: float = 300.0,
        include_tools: list[str] | None = None,
        exclude_tools: list[str] | None = None,
        id: str | None = None,
    ) -> None: ...

    @property
    def id(self) -> str:
        """Stable readable identifier; falls back to a derived default."""

    @abstractmethod
    async def _client_streams(self) -> AsyncContextManager[tuple[ReadStream, WriteStream]]: ...

    async def __aenter__(self) -> MCPServer: ...
    async def __aexit__(self, *exc) -> None: ...
    async def list_tools(self) -> list[ToolSpec]: ...
    async def call_tool(self, name: str, arguments: Json) -> ToolResult: ...
```

`id` default: `MCPServerStdio` derives `f"{command} {' '.join(args)}"`; `MCPServerSSE` and `MCPServerStreamableHTTP` derive from `url`. The default is computed in the base property; subclasses override the derivation. At harness setup (step 5), if multiple servers in `self._mcp_servers` resolve to the same derived id, the harness appends `-2`, `-3`, … to disambiguate in tracing and metadata.

Subclasses:
- `MCPServerStdio(command: str, args: list[str], *, env: dict[str, str] | None = None, cwd: str | Path | None = None, ...)`
- `MCPServerSSE(url: str, *, headers: dict[str, str] | None = None, ...)`
- `MCPServerStreamableHTTP(url: str, *, headers: dict[str, str] | None = None, ...)`

Each implements `_client_streams()` via the corresponding `mcp.client.{stdio,sse,streamable_http}` factory.

Lifecycle exactly as pydantic-ai does (mcp.py:295-331, 775-827, 829-899):
- `_SessionState` dataclass holds `session_task`, `ready_event`, `stop_event`, `nesting_counter`, `client`, `connect_error`, lock.
- `__aenter__` reference-counted. First entry spawns a background `asyncio.Task` (`_session_runner`) that opens client streams, constructs `mcp.ClientSession`, calls `session.initialize()` under `timeout`, signals `ready_event`, then awaits `stop_event`. Subsequent entries (from any task) bump the counter.
- `__aexit__` decrements. Last exit sets `stop_event`, awaits the session task with a 3-second grace period.
- Anyio cancel scopes from the transports must enter and exit on the same task — the dedicated session task is the fix.

**Before coding**, verify the `mcp.ClientSession` constructor parameter shape against the pinned `mcp>=1.23` package. We do not need to register a `notifications/tools/list_changed` handler for v1 (snapshot only).

**Verify**: an `MCPServerStdio` against the official `mcp` python "everything" server starts, lists tools, calls a tool, shuts down cleanly. Nested `async with` from two concurrent tasks shares one connection; tears down on last exit. Killing the subprocess between calls surfaces `MCPError`. Cancelling the outer task during `__aenter__` cleans up the spawned session task. `MCPServerStdio("uvx", ["mcp-server-git"]).id == "uvx mcp-server-git"`.

### 3. Tool Discovery and Conversion to `ToolSpec`
Inside `MCPServer.list_tools()`:
- `async with self`, call `await self._session.list_tools()`.
- **Filter first**: drop tools whose un-prefixed name matches `exclude_tools` or, when `include_tools` is set, doesn't match it.
- For each surviving tool, construct a `ToolSpec`:
  - `name`: `f"{tool_prefix}_{tool.name}"` if `tool_prefix` else `tool.name`, then run through a sanitizer mirroring OpenAI's `_normalize_mcp_name` (replace non-`[a-zA-Z0-9_-]` with `_`). Detect within-server sanitization collisions and raise an error naming both original MCP tool names.
  - `description`: `tool.description` (may be empty).
  - `parameters`: `_clean_mcp_schema(tool.inputSchema)` — a deep copy with `$schema` removed, top-level `title` removed, and top-level `additionalProperties: False` added only when not already specified. No recursion.
  - `handler`: closure over `(self, tool.name)` calling `await self.call_tool(tool.name, arguments)`. Async.
  - `sequential`: `False`.
  - `metadata`: `{"source": "mcp", "mcp_server_id": self.id, "mcp_tool_name": tool.name}`. Load-bearing for subagent inheritance and trace attribution.
  - `max_retries`: `None` (defer to harness default).

No cache field. No list_changed handler.

**Verify**: filters apply before conversion; prefix and sanitization apply in that order; within-server sanitization collisions raise with both original names; `inputSchema` is not mutated in place; top-level-only `additionalProperties: False`; an already-set `additionalProperties` is preserved.

### 4. Tool Call Dispatch and Result Mapping
Inside `MCPServer.call_tool(name: str, arguments: Json) -> ToolResult`:
- `async with self` to ensure the session is alive.
- `await self._session.call_tool(name, arguments=arguments)` returns `mcp.types.CallToolResult`.
- **Common metadata for every return path**:
  ```python
  base_metadata = {
      "source": "mcp",
      "mcp_server_id": self.id,
      "mcp_tool_name": name,
  }
  ```
- If `result.isError is True`:
  ```python
  ToolResult(
      ok=False,
      content=<joined text from result.content>,
      metadata={**base_metadata, "error_type": "MCPToolError", "retry": True},
  )
  ```
  This is the structured retry envelope path; `_tool_retry_kind()` in `core.py` keys off `metadata.retry is True`. Tests assert `usage.tool_retries[...]` and `stop_reason == "tool_retries_exceeded"`, not exception behavior.
- If `result.structuredContent` is present: `content = json.dumps(result.structuredContent, ensure_ascii=False)`. Preferred path.
- Otherwise iterate `result.content` blocks and concatenate as text:
  - `TextContent` → append `block.text` verbatim. `ToolResult.content` stays a `str`.
  - `ImageContent` / `AudioContent` → stub marker `[image: <mime>]` / `[audio: <mime>]`.
  - `EmbeddedResource` / `ResourceLink` → stub `[resource: <uri>]`. Follow-up can inline via `session.read_resource(uri)`.
- Success: `ToolResult(ok=True, content=<assembled string>, metadata=base_metadata)`.
- Unexpected `McpError`: `ToolResult(ok=False, content=str(exc), metadata={**base_metadata, "error_type": "MCPError"})`. No `retry` field.

**Verify**: every return path's metadata contains `source`, `mcp_server_id`, `mcp_tool_name`; `isError=True` adds `error_type` and `retry=True`; `McpError` adds `error_type="MCPError"` and no `retry`; success has neither `error_type` nor `retry`; a text-content call surfaces text; structured content surfaces as a JSON string; image content degrades to a stub; mid-call server kill surfaces `ok=False` with `error_type="MCPError"`; `ToolResult.content` is always `str`.

### 5. Harness Integration: Config, Lifecycle, `connect()`, `aclose()`
**Config.** Add `mcp_servers: list[MCPServer] = Field(default_factory=list)` to `HarnessConfig`. `arbitrary_types_allowed=True` is already set.

**`Harness.__init__` changes.**
- `self._mcp_servers = list(self.config.mcp_servers)`
- `self._mcp_stack: AsyncExitStack | None = None`
- `self._mcp_connected: bool = False`
- Resolve `id` collisions among `self._mcp_servers` — if multiple servers produce the same derived id, suffix `-2`, `-3`, … by setting an internal `_resolved_id` attribute on the server. Use `_resolved_id` (falling back to `id`) when constructing `ToolSpec.metadata["mcp_server_id"]` in step 3.
- No connection attempted. Construction stays sync.

**`_validate_hook_filters()`.** Stop passing `tool_names` through. `HookRegistry.validate_filters()` keeps `agent_names` validation and removes the `tool_names` path entirely. A hook with a filter naming a non-existent tool now succeeds construction and silently never fires.

**`_ensure_mcp_connected()` (private coroutine, idempotent).**
1. If `self._mcp_connected`, return.
2. If `self._mcp_servers` is empty, set `_mcp_connected = True` and return.
3. Build a local `AsyncExitStack`. **Wrap the rest in `try/except BaseException: await stack.aclose(); raise`** so any partial entry unwinds before the exception propagates.
   - For each `server` in `self._mcp_servers`:
     - `await stack.enter_async_context(server)`
     - `tools = await server.list_tools()`
   - Validate uniqueness across existing `self._tool_map` keys, the synthetic `final_result` tool when `output_mode == "tool"`, and other MCP servers' tools in the same batch. On collision, raise `HarnessError` naming the colliding tool and pointing at `tool_prefix` / `exclude_tools` — the `try/except` will then close the stack before propagating.
   - Append each MCP `ToolSpec` to `self.tools` and `self._tool_map`.
4. Store the stack on `self._mcp_stack`. Set `self._mcp_connected = True`.

**Public `async def connect(self) -> None`.**
- If `self._closed`: raise `HarnessError("harness is closed")`.
- If already connected, return.
- Otherwise call `await self._ensure_mcp_connected()`.
- Idempotent. Does not fire run hooks. Does not open tracing spans. Pure infrastructure setup.

**`Harness.run()` changes.**
- At the very top, before the `_running` guard: `if self._closed: raise HarnessError("harness is closed")`.
- **Remove** `self._closed = False` at the run-state init (`core.py:265`). `_closed` is now set exactly once, by `aclose()`.
- Inside the agent span, after `self.hooks.fire(RunStartContext(...))` and before `prompt_ctx = UserPromptSubmitContext(...)`, await `self._ensure_mcp_connected()`. A failure raised here propagates through the existing exception handling: `agent_span.record_exception(exc)`, `stop_reason="error"`, `RunEndContext` observes it normally.

**`Harness.aclose()` rewrite.**
```python
async def aclose(self) -> None:
    """Close MCP servers (always owned) and the provider HTTP client (only when owned)."""
    if self._closed:
        return
    try:
        if self._mcp_stack is not None:
            await self._mcp_stack.aclose()
            self._mcp_stack = None
            self._mcp_connected = False
        if self._owns_model:
            aclose = getattr(self.model.provider, "aclose", None)
            if aclose is not None:
                await aclose()
    finally:
        self._closed = True
```
- MCP closes unconditionally (no `_owns_model` gate) since the harness always owns its MCP server connections.
- Provider close stays gated on `_owns_model`.
- `_closed` is set in `finally` so a failure mid-teardown still locks the harness out.

**Note on `run_sync()` migration.** With the closed-state checks in place, `run_sync()` becomes structurally one-shot per `Harness` instance. Existing tests that loop on the same harness via `run_sync()` (notably provider message-leak tests and resume/branching tests in `tests/`) need to switch to `await harness.run(...)` inside a single `asyncio.run(...)` or to fresh `Harness` instances per call. This migration is part of the v1 implementation.

**Verify**: a harness with two stdio MCP servers connects both on first `run()`; a second `run()` reuses the connections (assert one initialize per server via the in-process fixture); name collisions across MCP servers, against built-ins/skills/subagents, or against `final_result` (when `output_mode="tool"`) raise during `_ensure_mcp_connected()`, not mid-run; partial connect failure (second server fails initialize) closes the first server before the exception propagates (assert via the in-process fixture's connection counter); `aclose()` tears both servers down; `aclose()` is idempotent; a harness with an injected model (`_owns_model=False`) and MCP servers still tears MCP down; `run()` after `aclose()` raises; `run_sync()` after `aclose()` raises; `connect()` before any `run()` succeeds; `connect()` after `aclose()` raises; two servers with the same derived id get `-2` suffix on the second.

### 6. Resume + MCP
- MCP is **not** part of `resume_state`. `_build_resume_state()` only serializes `session.dump_state()` (`core.py:62-80`); MCP tools come from the harness instance.
- Both fresh and resumed runs go through `_ensure_mcp_connected()` (fast-path after first call). Resumed runs use `session.continue_with_user_prompt(...)` and receive `tools=self.tool_schemas()` — MCP tools are already in `self.tools`.
- Cross-process resume: a fresh `Harness` instance opens its own MCP connections on first run. Tool drift between processes is treated as out of scope.

**Verify**: `resume_state` from run #1 does not contain MCP tool names or server identifiers; resuming run #1 inside run #2 on the same harness sees MCP tools and does not reconnect; resuming in a fresh `Harness` instance reconnects MCP on first run.

### 7. Subagent Inheritance
**`SubAgentConfig` additions.**
```python
class SubAgentConfig(BaseModel):
    ...
    inherit_parent_tools: bool = False  # existing
    inherit_mcp_servers: bool = False   # NEW: symmetric with above
    mcp_servers: list[MCPServer] = []   # NEW: additive override list
```

Semantics:
- `inherit_mcp_servers=True`: child enters each of the parent's MCP server objects via ref-counted `enter_async_context`. Sessions are not re-opened.
- `mcp_servers=[other]`: child opens these servers fresh in the child's MCP stack on the child's first run.
- Both: union. Identity-deduped — if a parent server object also appears in the child's `mcp_servers` list, fold to one entry.
- Identity-different but `id`-equal servers (two separately-constructed `MCPServerStdio` with the same command) are treated as separate; they'll trip the existing tool-name uniqueness check during `_ensure_mcp_connected()` and the user will get the "use `tool_prefix` or `exclude_tools`" error.

**Validator update.** `SubAgentConfig`'s named-subagent validator (`subagents.py:54-57`) accepts as a sufficient tool source any of: `builtin_tools` non-empty, `tools` non-empty, `inherit_parent_tools=True`, `inherit_mcp_servers=True`, or `mcp_servers` non-empty list. `SubAgentConfig(name="x", description="...")` with no other fields fails validation as before.

**Parent-tools copy guard.** `_effective_custom_tools(parent, config)` skips parent tools whose `metadata.get("source") == "mcp"`. Otherwise a parent MCP tool would arrive both as a copied custom tool (named collision) and via `inherit_mcp_servers=True`. The tag is set in step 3.

**`build_child_harness()` child config update.** The `model_copy(update={...})` call always sets `mcp_servers` explicitly so the parent's value cannot leak through:
```python
parent_mcp = self._mcp_servers if config.inherit_mcp_servers else []
combined = list(parent_mcp)
for server in config.mcp_servers:
    if not any(server is existing for existing in combined):
        combined.append(server)
child_overrides = {"mcp_servers": combined, ...}
```
Identity dedup uses `is`. Order: inherited parents first, then explicit child overrides (only those not already present).

**Subagent `effective_tools` timing.** In `run_subagent_tool()` (`subagents.py:125-131`), call `await child.connect()` before collecting `effective_tools = [tool.name for tool in child.tools]`. This ensures MCP tools appear in `AfterSubagentRunContext.tools`, in the subagent `ToolResult.metadata["tools"]`, and in the parent's `subagent.tools` trace attribute.

**Child lifecycle.** Child's `aclose()` decrements ref counts on inherited servers; parent's connection stays alive because the parent's stack still holds a reference.

**Verify**: parent + subagent inheriting via `inherit_mcp_servers=True` open the underlying server process exactly once; child with `inherit_mcp_servers=False` and `mcp_servers=[]` sees no MCP tools; child with `inherit_mcp_servers=True` and `mcp_servers=[other]` sees both parent's tools and `other`'s tools (union); child where the same parent server object appears in both the inherited set and the explicit list folds to one entry (assert via initialize-count); two MCPServer instances with the same `id` but different identity in the same child config trip the collision check; tearing down a child does not kill the parent's session; `inherit_parent_tools=True` does not produce duplicate MCP tool names; `SubAgentConfig(name="x", description="...")` fails validation; `SubAgentConfig(name="x", inherit_mcp_servers=True)` validates; `SubAgentConfig(name="x", mcp_servers=[server])` validates; `effective_tools` includes MCP tool names.

### 8. Tracing
In `_traced_call_output`, after fetching the spec from `self._tool_map[name]`, read `spec.metadata` (not the result envelope) for trace attribution:
```python
spec_metadata = self._tool_map.get(name, ToolSpec(...)).metadata or {}
if spec_metadata.get("source") == "mcp":
    span.set_attributes({
        "mcp.server.id": spec_metadata.get("mcp_server_id"),
        "mcp.tool.name": spec_metadata.get("mcp_tool_name"),
    })
```
This is robust to after-tool hooks rewriting the output envelope. Mirrors the existing subagent attribute pattern (`core.py:622-627`) but reads the spec, not the parsed result.

No new span types. `_ensure_mcp_connected()` runs inside the agent span; connection errors are automatically recorded on `agent_span` via the existing `record_exception` path.

**Verify**: an MCP tool call span carries `mcp.server.id` and `mcp.tool.name` read from the `ToolSpec`; the existing `gen_ai.tool.call.result` attribute still fires; connection failures appear on `agent_span` as recorded exceptions with `stop_reason="error"` on `RunEndContext`; subagent MCP tool calls carry the same attributes; an after-tool hook that rewrites the output envelope's metadata does not lose `mcp.server.id` / `mcp.tool.name` on the span.

### 9. Tests
Two files:

**`tests/test_mcp_optional_dependency.py`** (no module-level skip):
- `test_construction_without_extra`: stub `sys.modules['mcp']` to raise on import; assert `MCPServerStdio(command="x")` constructs cleanly; assert `await server.__aenter__()` raises `MCPDependencyError`. Also assert `MCPDependencyError` is importable from `thinharness`.

**`tests/test_mcp.py`** (module-level `pytest.importorskip("mcp")`). Use the official `mcp` package's in-process `Server` for fixtures.

- `test_stdio_smoke`: tiny stdio server as subprocess (`tests/fixtures/mcp_servers/echo_server.py`). Only test that exercises the subprocess path.
- `test_tool_discovery_snapshot`: list_tools called once during connect; subsequent runs don't re-fetch.
- `test_tool_prefix_filters_and_sanitization`: two in-process servers with the same MCP tool name distinguished via `tool_prefix`; `include_tools` and `exclude_tools` work; sanitizer collision raises with both original names.
- `test_input_schema_not_mutated`: server returns a shared dict; harness's cleaned schema is distinct; mutating the harness copy does not affect the server's dict.
- `test_tool_result_metadata_shape`: success, `isError=True`, and `McpError` paths all carry `{source, mcp_server_id, mcp_tool_name}`; `isError=True` adds `error_type` + `retry=True`; `McpError` adds `error_type` only; success has neither.
- `test_isError_drives_harness_retry`: an `isError=True` result drives `usage.tool_retries`; with the budget exceeded, `stop_reason == "tool_retries_exceeded"`.
- `test_lifecycle_reference_counted`: two tasks enter the same server context; exit one, session still alive; exit the second, torn down.
- `test_collision_detected_before_first_turn`: MCP server exposing `read` (collides with built-in) raises in `_ensure_mcp_connected()` before any model request.
- `test_partial_connect_failure_cleans_up`: first server connects successfully; second fails during `initialize()`; assert the first server's session is torn down before the exception propagates.
- `test_final_result_collision`: with `output_type=SomeModel` and `output_mode="tool"`, an MCP server exposing `final_result` raises during connect.
- `test_hook_filter_on_unknown_tool_does_not_raise`: constructing a harness with a hook filtered to a non-existent tool name succeeds; the hook never fires when a different tool runs.
- `test_connection_reused_across_runs`: in-process server tracks initialize-count; two sequential `run()` calls produce exactly one initialize.
- `test_explicit_connect`: `await harness.connect()` succeeds; subsequent `run()` does not reconnect.
- `test_run_after_aclose_raises`: `aclose()` then `run()` raises `HarnessError("harness is closed")`.
- `test_run_sync_after_aclose_raises`: `run_sync()` called twice on the same instance raises on the second call.
- `test_tool_schemas_after_close`: `tool_schemas()` after `aclose()` still returns the old schemas (intentional; harmless inspectable list).
- `test_add_tool_after_close_allowed`: `add_tool()` after `aclose()` succeeds (no functional effect since the harness can't run).
- `test_aclose_with_injected_model_closes_mcp`: harness created with `model=...` (so `_owns_model=False`) and MCP servers — `aclose()` closes MCP but not the model's HTTP client.
- `test_connect_after_aclose_raises`: `aclose()` then `await harness.connect()` raises.
- `test_resume_with_mcp`: run #1 + run #2 (resumed via `resume_state`) on the same harness both call MCP tools; in-process server sees one initialize; `resume_state` from run #1 does not mention MCP.
- `test_subagent_inherits_mcp_via_flag`: parent + subagent with `inherit_mcp_servers=True` both call MCP tools; underlying server initializes once.
- `test_subagent_overrides_mcp_only`: subagent with `inherit_mcp_servers=False` and `mcp_servers=[other]` sees only `other`'s tools, not parent's.
- `test_subagent_unions_inherit_plus_override`: subagent with `inherit_mcp_servers=True` and `mcp_servers=[other]` sees both parent MCP tools and `other`'s tools.
- `test_subagent_identity_dedup`: parent has server A; child with `inherit_mcp_servers=True` and `mcp_servers=[A]` (same object) folds to one entry — assert one initialize, no collision.
- `test_subagent_id_equal_but_distinct_collides`: two `MCPServerStdio` instances with identical commands but different Python identity in one child config trip the tool-name collision check at child connect time.
- `test_subagent_excludes_inherited_mcp_from_custom_copy`: parent has MCP tool `foo`; subagent with `inherit_parent_tools=True` and `inherit_mcp_servers=True` ends up with exactly one `foo`.
- `test_subagent_mcp_only_validates`: named subagent with only `inherit_mcp_servers=True` validates; same with `mcp_servers=[server]`.
- `test_subagent_empty_fails_validation`: `SubAgentConfig(name="x", description="...")` fails validation; same with `mcp_servers=[]` and nothing else.
- `test_subagent_effective_tools_includes_mcp`: `AfterSubagentRunContext.tools` and the parent `subagent.tools` trace attribute include MCP tool names.
- `test_trace_attribution_survives_after_tool_hook`: an after-tool hook rewrites the result envelope to drop MCP metadata; the span still carries `mcp.server.id` / `mcp.tool.name` because tracing reads from the `ToolSpec`.
- `test_connection_failure_in_run`: an MCP server whose `initialize()` raises — assert `RunStartContext` fires, `RunEndContext.stop_reason == "error"`, `agent_span` records the exception.
- `test_run_teardown_on_exception`: a custom tool raises mid-run; `aclose()` still tears MCP servers down cleanly.
- `test_run_teardown_on_cancellation`: cancel the outer task during a tool call; `aclose()` tears MCP servers down without anyio cancel-scope errors.
- `test_duplicate_derived_id_disambiguated`: two `MCPServerStdio` with the same command but explicit different `id=None` get `-2` suffix on the second; trace attributes use the suffixed id.

Plus extend `tests/test_harness.py` smoke tests: `HarnessConfig(mcp_servers=[])` works without the `mcp` extra installed.

**Test migration**: existing tests that reuse a single `Harness` instance across multiple `run_sync()` calls (the v3 feedback flagged provider message-leak and resume/branching tests) switch to either (a) one `asyncio.run(...)` calling `await harness.run(...)` repeatedly, or (b) fresh harness instances per call. This is part of the implementation diff.

**Verify**: `uv run pytest tests/test_mcp.py tests/test_mcp_optional_dependency.py` passes with the extra installed; `uv run pytest tests/` passes without (MCP tests skip, optional-dependency test runs). No subprocess leaks after stdio tests.

## Out of Scope (Follow-ups)
- **Mid-run tool list changes (`tools/list_changed`).** Register a `ClientSession` notification handler that flags the harness; refresh the spliced `ToolSpec`s between turns. Requires deciding refresh granularity and how to handle a tool the model just called disappearing.
- **MCP resources as agent context.** Future `MCPServer.list_resources()` + `read_resource()`, optionally as an `mcp_resource` built-in tool.
- **MCP prompts.** Map `prompts/list` into the system prompt or as callable prompt templates.
- **MCP sampling.** Wire `Harness.model` into `ClientSession(sampling_callback=...)`. Requires translating MCP message types ↔ `ModelTurn`.
- **OAuth and dynamic auth headers.** v1 takes static `headers`. A `headers_provider` callable plus an OAuth client helper is a clean v2 add.
- **`.mcp.json` discovery.** A `load_mcp_servers(path)` helper mirroring the Claude Desktop / DeepAgents config format with `${VAR}` expansion.
- **Docs / README updates.** Installing `thinharness[mcp]`; transport examples; lazy `connect()` behavior; snapshot-only tool discovery; no MCP content in `resume_state`; `run_sync()` one-shot semantics.
- **Hook filter typo-catching as a warning.** If the absence of construction-time validation bites users, add a one-shot warning the first time a hook's tool-name filter has never matched during a turn.
