# Plan: Subagents Fork-Conversation Mode

## Overview
Fork-conversation mode lets a subagent start its child run with the parent's running conversation state instead of a fresh context. This is useful for "reviewer"-style helpers that need the full task context the parent has already accumulated. It is provider-sensitive plumbing and is deliberately split out from the base subagent plan (`plan-subagents.md`) so that fresh-context subagents can ship cleanly first.

This plan assumes the base subagent plan is already implemented.

## References
- `vendor/pydantic-ai/pydantic_ai_slim/pydantic_ai/agent/abstract.py` run APIs accept `message_history`, `conversation_id`, `instructions`, `toolsets`, and `capabilities`; this supports the split between fresh context and forked context as an explicit per-run concern.
- OpenAI Responses: the cheap path is reusing the parent session's `previous_response_id` for the child's first request.
- Anthropic Messages and OpenRouter: no server-side conversation handle; fork by copying the parent session's accumulated messages list.

## Steps

### 1. Add a Fork Switch to SubAgentConfig
Extend `SubAgentConfig` with a single field:

```python
fork_conversation: bool = False
```

Default is `False` so fork mode is opt-in. The same configuration validation rule applies as in the base plan: `fork_conversation=True` should not conflict with `inherit_parent_tools=True` if we decide they shouldn't be combined; revisit during implementation.

### 2. Introduce a Conversation Fork Abstraction
Add a small data class so the subagent tool handler can fork without reaching into provider-specific internals.

```python
@dataclass
class ConversationFork:
    """Provider-specific state needed to fork a child run."""

    previous_response_id: str | None = None
    messages: list[Json] | None = None
    system: str | None = None
```

Extend `ModelSession` implementations with optional `fork_state() -> ConversationFork` and `start_from_fork(fork: ConversationFork, ...) -> ModelTurn` methods, or add helper functions in `providers.py` if keeping the protocol small is preferred.

Provider mapping:
- OpenAI Responses: `fork_state()` returns the session's current `previous_response_id`; `start_from_fork()` passes it as `previous_response_id` on the child `start(...)`.
- Anthropic Messages: `fork_state()` returns a deep copy of `AnthropicMessagesSession.messages` plus `system`; `start_from_fork()` constructs a child session whose initial messages list is the copied parent history, then appends the subagent task as the next user message.
- OpenRouter: `fork_state()` returns a deep copy of `OpenRouterSession.messages`; `start_from_fork()` constructs a child session whose initial messages list is the copied parent history, then appends the subagent task as the next user message.

### 3. Expose Parent Session to the Subagent Tool Handler
The parent `Harness.run()` currently keeps `session` local to one run. Fork mode needs the live session at tool-call time. Add a private run context field that lives only for the duration of tool execution.

```python
@dataclass
class _RunContext:
    """Mutable state available to tools during one harness run."""

    session: ModelSession
    metadata: Json | None
```

Set `self._run_context` before model start and clear it in `finally`. The subagent tool reads `parent._run_context.session` only when `config.fork_conversation` is true; fresh mode ignores it entirely.

This is a deliberately narrow leak: only the subagent tool, which is a built-in we control, ever touches `_run_context`. Don't expand `ToolHandler` signatures for this — the run context is harness-private state, not part of the public tool contract.

### 4. Branch Child Construction by Fork Mode
In the subagent handler, after the agent config is resolved:

```python
if config.fork_conversation:
    fork = parent._run_context.session.fork_state()
    child = build_child_harness(parent, config, requested_tools)
    result = child.run_from_fork(args.task, fork=fork, metadata=...)
else:
    child = build_child_harness(parent, config, requested_tools)
    result = child.run(args.task, metadata=...)
```

`Harness.run_from_fork(prompt, *, fork, metadata)` is a new entry point that creates a child session via `model.new_session()` and uses `start_from_fork()` instead of `start()` for the first turn. The rest of the tool loop is unchanged.

### 5. Tracing Attributes
Add a trace attribute to the child agent span indicating whether the run used forked context (`subagent.fork_mode = "forked" | "fresh"`). The base plan already adds attributes for inherited-tools vs configured-tools; this is the same idea for the conversation axis.

Record child errors on both the parent `execute_tool subagent` span and the child `invoke_agent subagent.<name>` span exactly as the base plan does.

### 6. Documentation
Add a fork-mode example to the README subagent section:

```python
SubAgentConfig(
    name="reviewer",
    description="Reviews the current conversation state and checks for risks.",
    system_prompt="Review the current task context and return concrete risks.",
    builtin_tools=["read", "search"],
    fork_conversation=True,
)
```

Document the per-provider semantics so callers understand what "forked" means:
- OpenAI: fast, server-side fork via `previous_response_id`.
- Anthropic / OpenRouter: client-side message copy; cost scales with parent history length.

## Verification
- Provider-specific tests confirm: OpenAI child `start()` receives the parent's last response id in fork mode; Anthropic and OpenRouter child payloads include the copied parent messages followed by the new task message.
- A test confirms fresh mode is unaffected: the child's first request contains only the subagent task and the child's system prompt.
- A test confirms `_run_context` is cleared after the parent run regardless of success or exception, so leftover session references never escape past a single run.
- A tracing test asserts `subagent.fork_mode` is set correctly on the child agent span.

## Considerations
- Forking is provider-sensitive. OpenAI's `previous_response_id` makes fork mode cheap, while Anthropic and OpenRouter need copied message lists. Keep this behind a small provider/session method so the subagent tool does not become provider-aware.
- The `_RunContext` should be cleared reliably in `finally` because `Harness` is documented as single-run/single-thread. A leaked session reference would be a footgun for callers reusing the harness.
- Fork mode and `inherit_parent_tools=True` may or may not be combinable. Decide at implementation time whether to allow them together; default toward "yes, they're orthogonal axes" unless tracing or recursion makes that messy.
- A fork-mode child can still emit its own tool calls and even (if the recursion guard is ever lifted) its own subagent calls. The fork is just about conversation state, not the tool universe — keep these axes separate in code and docs.
