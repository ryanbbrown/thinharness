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

`user_prompt_submit` hooks run for every caller-submitted prompt, including resumed prompts. If a hook adds prompt context on a resumed run, the provider receives the resumed prompt, then hook context, then any harness-owned limit notice.

`resume_state` is emitted only for clean terminal runs where `stop_reason == "end_turn"` and the provider session can produce a usable continuation token. It is `None` after provider errors, tool errors, hook cancellation, max-turn or max-tool limits, structured-output validation exhaustion, tool retry exhaustion, and structured-output `final_result` tool termination.

Provider behavior differs internally:

- OpenAI Responses stores conversation state server-side. `resume_state` contains the previous response id, and a later resumed call sends that id as `previous_response_id`.
- Anthropic Messages is stateless. `resume_state` contains the full message transcript and grows with the conversation.
- OpenRouter chat completions is stateless. `resume_state` contains the full chat transcript and grows with the conversation.

Model-facing limit notices are real provider input, so they may be part of the conversation state behind `resume_state`. Notice text is scoped to "this run", deduping is per `Harness.run(...)`, and a resumed run may emit a notice that is also visible in prior conversation history.

OpenAI response retention is controlled by the provider. If a stored response is deleted or expires, resuming from that state surfaces as a provider error with the provider's error text.

The same `resume_state` can be reused for sequential branching:

```python
base = await harness.run("Draft three product names.")
one = await harness.run("Make them more formal.", resume_from=base.resume_state)
two = await harness.run("Make them more playful.", resume_from=base.resume_state)
```

For parallel branches, use separate `Harness` instances. A single `Harness` instance still rejects concurrent `run()` calls.

ThinHarness has no separate cross-run message-history parameter. `resume_from` is the supported way to carry prior context across `run()` calls.

## Parallel LLM Batches

`parallel_llm` is an opt-in built-in tool for batches of independent one-shot prompts:

```python
harness = Harness(HarnessConfig(
    builtin_tools=["parallel_llm"],
    builtin_parallel_llm_model="openai:gpt-5.2-mini",  # optional; defaults to the parent model
    builtin_parallel_llm_temperature=0,
    parallel_llm_max_prompts=100,
    parallel_llm_max_attempts=4,
))
```

Each batch call is stateless. The per-prompt model session receives `tools=[]`, no memory, no continuation, and no inherited harness system prompt. Pass `system` when the batch needs shared instructions.

The model-facing prompt source is a single `source` object. For inline prompts, use `{"kind": "inline", "prompts": [...]}`. For a JSON prompt file under the read path policy, use `{"kind": "file", "path": "prompts.json"}`.

Use `output_file` when combined results may be large. Inline output returns compact JSON in the normal `ToolResult.content`; file output writes pretty JSON under the workspace path policy and returns only a summary with failed indices.

`max_concurrency` is model-controlled per tool call and only limits in-flight attempts. `parallel_llm_max_prompts` and `parallel_llm_max_attempts` are host-controlled `HarnessConfig` fields for the built-in. Parallel attempts are reported as `model_requests` in the tool payload and metadata; they do not consume `max_model_requests`, while the `parallel_llm` invocation itself still counts as one tool call.

The model-facing arguments do not include model, temperature, or output-schema overrides. The built-in uses `builtin_parallel_llm_model` and `builtin_parallel_llm_temperature` when set, otherwise it falls back to the parent harness model and temperature. Custom `ParallelLlmTool` instances own their model, temperature, API settings, prompt cap, retry budget, and optional structured output schema.

For a custom, renameable version, construct the same tool as a normal `ToolSpec`:

```python
from thinharness import ParallelLlmTool
from pydantic import BaseModel


class InvoiceFields(BaseModel):
    vendor: str
    total: float

extract_tool = ParallelLlmTool(
    name="parallel_extract",
    description="Extract fields from independent chunks.",
    model="openai:gpt-5.2-mini",
    root=".",
    read_paths=["inputs"],
    write_paths=["outputs"],
    max_prompts=50,
    max_attempts=2,
    output_type=InvoiceFields,
    output_mode="auto",
    output_retries=1,
).spec()

harness = Harness(HarnessConfig(root="."), tools=[extract_tool])
```

When `output_type` is set on a custom `ParallelLlmTool`, each successful result entry contains the parsed JSON-compatible value rather than raw text. `output_mode` accepts the same modes as harness structured output: `auto`, `native`, `tool`, `prompted`, and `text`.

For structured custom tools, pass a description that tells the parent model the tool returns validated result values rather than raw text.
