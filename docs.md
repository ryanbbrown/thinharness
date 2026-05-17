# ThinHarness Docs

## Continuing a Conversation

`HarnessResult.resume_state` is an opaque, JSON-serializable token that lets callers continue a completed conversation with a new user message.

```python
first = await harness.run("Summarize this repository.")
if first.resume_state is None:
    raise RuntimeError("run cannot be continued")
save_json(first.resume_state)

state = load_json()
second = await harness.run("Now turn that into a checklist.", resume_from=state)
```

The contract:

- Save `result.resume_state` exactly as JSON.
- Pass it back as `resume_from` with the next user message.
- Use the same provider, model, system prompt, and tools as the run that produced it.
- Expect no state after failed, cancelled, partial, or exhausted runs.
- Treat the contents as provider-owned details; do not read or construct them.

`resume_from` is a new-turn API. It means the prior run completed, and the next call appends a new user message to that conversation. It is not a retry mechanism, an interrupted-tool-call recovery mechanism, or a way to continue the assistant's previous response.

`resume_state` is emitted only for clean terminal runs where `stop_reason == "end_turn"` and the provider session can produce a usable continuation token. It is `None` after provider errors, tool errors, hook cancellation, max-turn or max-tool limits, structured-output validation exhaustion, tool retry exhaustion, and structured-output `final_result` tool termination.

Provider behavior differs internally:

- OpenAI Responses stores conversation state server-side. `resume_state` contains the previous response id, and a later resumed call sends that id as `previous_response_id`.
- Anthropic Messages is stateless. `resume_state` contains the full message transcript and grows with the conversation.
- OpenRouter chat completions is stateless. `resume_state` contains the full chat transcript and grows with the conversation.

OpenAI response retention is controlled by the provider. If a stored response is deleted or expires, resuming from that state surfaces as a provider error with the provider's error text.

The same `resume_state` can be reused for sequential branching:

```python
base = await harness.run("Draft three product names.")
one = await harness.run("Make them more formal.", resume_from=base.resume_state)
two = await harness.run("Make them more playful.", resume_from=base.resume_state)
```

For parallel branches, use separate `Harness` instances. A single `Harness` instance still rejects concurrent `run()` calls.

ThinHarness has no separate cross-run message-history parameter. `resume_from` is the supported way to carry prior context across `run()` calls.
