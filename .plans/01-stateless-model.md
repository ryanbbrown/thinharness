# Stateless Model Session Plan

## Goal

Separate reusable model configuration from per-run mutable conversation state.

Today the model adapters store run state directly:

- `OpenAIResponsesModel.previous_response_id`
- `AnthropicMessagesModel.messages`
- `OpenRouterModel.messages`

That makes one model or harness instance unsafe to reuse across concurrent runs. The model object should instead hold durable configuration only: model name, provider, and settings. Each harness run should create a fresh session object that owns conversation state for that run.

## Target Shape

Add a `ModelSession` protocol next to `Model` in `providers.py`:

```python
class Model(Protocol):
    provider: Provider
    model: str

    @property
    def api_key(self) -> str | None: ...

    def new_session(self) -> ModelSession: ...

class ModelSession(Protocol):
    def start(
        self,
        *,
        prompt: str,
        instructions: str,
        tools: list[Json],
        metadata: Json | None = None,
        previous_response_id: str | None = None,
    ) -> ModelTurn: ...

    def continue_with_tools(
        self,
        outputs: list[ToolOutput],
        *,
        tools: list[Json],
        metadata: Json | None = None,
    ) -> ModelTurn: ...
```

`Harness.run()` should call `session = self.model.new_session()` once per run, then use `session.start(...)` and `session.continue_with_tools(...)`.

Tracing can still receive `self.model` for provider/model attributes, since those are stable config values.

## Adapter Changes

Keep these as reusable config classes:

- `OpenAIResponsesModel`
- `AnthropicMessagesModel`
- `OpenRouterModel`

Add session classes:

- `OpenAIResponsesSession`
  - owns `previous_response_id`
  - delegates payload creation and request normalization through model/provider helpers
- `AnthropicMessagesSession`
  - owns `messages`
- `OpenRouterSession`
  - owns `messages`

Each model class gets:

```python
def new_session(self) -> ModelSession:
    return OpenAIResponsesSession(self)
```

The current `start` and `continue_with_tools` methods can either move entirely to the session classes, or remain as compatibility wrappers that create a temporary session. Prefer moving the logic to sessions and, if compatibility is desired, add wrappers with a short deprecation comment.

## OpenAI Responses State

Move `previous_response_id` from `OpenAIResponsesModel` to `OpenAIResponsesSession`.

The session should:

- accept `previous_response_id` on `start`
- send it on the first request when present
- update it from each response id
- include it on later tool-output continuation requests

This preserves current behavior while making the state per-run.

## Chat Model State

Move `messages` from `AnthropicMessagesModel` and `OpenRouterModel` to their session classes.

The session should:

- initialize `messages` in `start`
- append assistant/tool messages during `continue_with_tools`
- never mutate the model object

## Tests

Add focused tests that prove model instances are reusable:

1. Two sequential harness runs with the same `Harness` do not leak Anthropic/OpenRouter messages between runs.
2. Two `model.new_session()` objects can be advanced independently.
3. OpenAI `previous_response_id` is scoped to one session and does not affect a second session.
4. Existing provider loop tests still pass.
5. Pyright stays clean:

```bash
uv run --extra tracing --with pyright pyright thinharness tests
```

Also keep:

```bash
uv run --extra dev pytest -q
uv run python -m compileall -q thinharness tests
```

## Expected Impact

This should be a small structural refactor, mostly in `providers.py` and `core.py`.

The public `Harness` API should not need to change. The main user-visible improvement is that a `Harness` or model object becomes safer to reuse because each run gets isolated model-session state.
