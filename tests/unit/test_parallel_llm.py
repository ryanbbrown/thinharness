from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from fakes import FakeChildHarnessHost
from pydantic import BaseModel, ValidationError

import thinharness.plugins.parallel_llm as parallel_plugin_module
from thinharness import Harness, HarnessConfig, ModelCapabilities, ModelToolCall, ModelTurn, ParallelLlmPlugin, PluginContext, ToolOutput, ToolSpec
from thinharness.providers import ModelSettings, OpenAIProvider, OpenAIResponsesModel, ProviderError
from thinharness.tools.base import _invoke_tool
from thinharness.tools.parallel_llm import (
    DEFAULT_PARALLEL_LLM_INSTRUCTIONS,
    PROMPTS_FILE_ERROR,
    ParallelLlmArgs,
    ParallelLlmTool,
    _atomic_write_json,
)


class BatchProvider:
    name = "OpenAI"

    def __init__(self) -> None:
        self.closed = False

    async def aclose(self) -> None:
        """Record provider shutdown."""
        self.closed = True


class ExtractedPerson(BaseModel):
    """Structured result used by parallel LLM tests."""

    name: str
    age: int


def _inline(prompts: list[str]) -> dict[str, Any]:
    """Build inline-source parallel_llm args."""
    return {"source": {"kind": "inline", "prompts": prompts}}


def _file(path: str) -> dict[str, Any]:
    """Build file-source parallel_llm args."""
    return {"source": {"kind": "file", "path": path}}


class BatchModel:
    def __init__(self, outcomes: list[Any] | None = None, *, delay: float = 0) -> None:
        self.model = "batch-model"
        self.provider = BatchProvider()
        self.api_key = "batch-key"
        self.capabilities = ModelCapabilities()
        self.outcomes = list(outcomes or [])
        self.delay = delay
        self.session_requests = 0
        self.calls: list[dict[str, Any]] = []
        self.in_flight = 0
        self.max_in_flight = 0

    def new_session(self) -> BatchSession:
        """Return a fresh batch session."""
        self.session_requests += 1
        return BatchSession(self)

    async def complete(self, prompt: str, instructions: str, tools: list[dict[str, Any]], structured_output: Any = None) -> ModelTurn:
        """Record one completion and return or raise the scripted outcome."""
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        self.calls.append({"prompt": prompt, "instructions": instructions, "tools": tools, "structured_output": structured_output})
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            outcome = self.outcomes.pop(0) if self.outcomes else f"echo:{prompt}"
            if isinstance(outcome, BaseException):
                raise outcome
            if isinstance(outcome, ModelTurn):
                return outcome
            return ModelTurn(text=str(outcome), raw={"output_text": str(outcome)})
        finally:
            self.in_flight -= 1


class BatchSession:
    def __init__(self, model: BatchModel) -> None:
        self.model = model

    async def start(self, prompt, constants, *, previous_response_id=None, notices=None):
        """Run one batch completion."""
        return await self.model.complete(prompt, constants.instructions, constants.tools, constants.structured_output)

    async def continue_with_tools(self, outputs, constants, *, notices=None):
        """Batch sessions never continue."""
        raise AssertionError("batch session should not continue")

    async def continue_with_user_content(self, text, constants, *, notices=None):
        """Batch sessions never continue."""
        raise AssertionError("batch session should not continue")

    def dump_state(self):
        """Batch sessions are not resumable."""
        return None


class HybridModel(BatchModel):
    def __init__(self) -> None:
        super().__init__()
        self.main_session = MainSession()

    def new_session(self):
        """Return the main loop session first, then batch sessions."""
        self.session_requests += 1
        if self.session_requests == 1:
            return self.main_session
        return BatchSession(self)


class MainSession:
    async def start(self, prompt, constants, *, previous_response_id=None, notices=None):
        """Ask the harness to call parallel_llm."""
        return ModelTurn(
            raw={"id": "first"},
            tool_calls=[
                ModelToolCall(
                    id="call_1",
                    name="parallel_llm",
                    arguments=json.dumps({**_inline(["a", "b"]), "max_concurrency": 2}),
                )
            ],
        )

    async def continue_with_tools(self, outputs: list[ToolOutput], constants, *, notices=None):
        """Finish after receiving the tool output."""
        parsed = json.loads(outputs[0].output)
        payload = json.loads(parsed["content"])
        return ModelTurn(text=f"done:{payload['succeeded']}", raw={"id": "done"})

    async def continue_with_user_content(self, text, constants, *, notices=None):
        """Main session never receives user-text continuations."""
        raise AssertionError("should not continue with user text")

    def dump_state(self):
        """Main session has no resume state."""
        return None


def _parent(
    tmp_path: Path,
    batch_model: BatchModel | None = None,
    *,
    plugin: ParallelLlmPlugin | None = None,
    **config: Any,
) -> Harness:
    """Build a harness parent with the parallel LLM plugin."""
    return Harness(
        HarnessConfig(root=tmp_path, **config),
        model=batch_model or BatchModel(),
        plugins=[plugin or ParallelLlmPlugin()],
    )


async def _call_parallel(parent: Harness, args: dict[str, Any]) -> dict[str, Any]:
    """Invoke parallel_llm through the normal tool envelope."""
    spec = next(tool for tool in parent.tools if tool.name == "parallel_llm")
    output = await _invoke_tool(spec, args)
    parsed = json.loads(output.to_json())
    if parsed["ok"]:
        parsed["payload"] = json.loads(parsed["content"])
    return parsed


async def _call_custom_tool(tool: ParallelLlmTool, args: dict[str, Any]) -> dict[str, Any]:
    """Invoke a custom ParallelLlmTool through the normal tool envelope."""
    output = await _invoke_tool(tool.spec(), args)
    parsed = json.loads(output.to_json())
    if parsed["ok"]:
        parsed["payload"] = json.loads(parsed["content"])
    return parsed


async def test_parallel_llm_inline_prompts_return_compact_ordered_payload(tmp_path: Path) -> None:
    model = BatchModel(outcomes=["second", "first"], delay=0.01)
    parent = _parent(tmp_path, model)

    result = await _call_parallel(parent, {**_inline(["p0", "p1"]), "system": "shared", "max_concurrency": 2})

    assert result["ok"] is True
    assert "\n" not in result["content"]
    assert result["payload"] == {
        "total": 2,
        "succeeded": 2,
        "failed": 0,
        "model_requests": 2,
        "results": [
            {"index": 0, "ok": True, "result": "second"},
            {"index": 1, "ok": True, "result": "first"},
        ],
    }
    assert [call["tools"] for call in model.calls] == [[], []]
    assert [call["instructions"] for call in model.calls] == ["shared", "shared"]


async def test_parallel_llm_file_source_and_output_file(tmp_path: Path) -> None:
    (tmp_path / "prompts.json").write_text(json.dumps(["a", "b"]), encoding="utf-8")
    model = BatchModel(outcomes=["ok", ProviderError("provider error 401: nope", status_code=401)])
    parent = _parent(tmp_path, model)

    result = await _call_parallel(parent, {**_file("prompts.json"), "output_file": "nested/results.json"})

    assert result["payload"] == {
        "total": 2,
        "succeeded": 1,
        "failed": 1,
        "model_requests": 2,
        "output_file": "nested/results.json",
        "failed_indices": [1],
    }
    assert "results" not in result["payload"]
    file_text = (tmp_path / "nested/results.json").read_text(encoding="utf-8")
    assert file_text.endswith("\n")
    file_payload = json.loads(file_text)
    assert file_payload["model_requests"] == 2
    assert file_payload["results"][0] == {"index": 0, "ok": True, "result": "ok"}
    assert "error" not in file_payload["results"][0]
    assert file_payload["results"][1] == {"index": 1, "ok": False, "error": "provider error 401: nope"}
    assert "result" not in file_payload["results"][1]


@pytest.mark.parametrize(
    ("content", "args"),
    [
        ("", _file("prompts.json")),
        ("   ", _file("prompts.json")),
        ('{"prompt":"x"}', _file("prompts.json")),
        ('["x", 1]', _file("prompts.json")),
        ("[]", _file("prompts.json")),
    ],
)
async def test_parallel_llm_rejects_bad_prompt_files(tmp_path: Path, content: str, args: dict[str, Any]) -> None:
    (tmp_path / "prompts.json").write_text(content, encoding="utf-8")
    parent = _parent(tmp_path)

    result = await _call_parallel(parent, args)

    assert result["ok"] is False
    assert result["content"] == PROMPTS_FILE_ERROR


@pytest.mark.parametrize("args", [{}, {"prompts": ["x"], "prompts_file": "prompts.json"}])
def test_parallel_llm_requires_structured_prompt_source(args: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        ParallelLlmArgs.model_validate(args)


def test_parallel_llm_args_normalize_blank_optional_fields_and_reject_model_override() -> None:
    args = ParallelLlmArgs.model_validate({**_inline(["x"]), "output_file": "", "system": ""})

    assert args.source.kind == "inline"
    assert args.source.prompts == ["x"]
    assert args.output_file is None
    assert args.system is None
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ParallelLlmArgs.model_validate({**_inline(["x"]), "model": "openai:gpt-5-mini"})


async def test_parallel_llm_enforces_path_policies_and_prompt_cap(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    (tmp_path / "prompts.json").write_text('["x"]', encoding="utf-8")
    model = BatchModel()
    parent = _parent(
        tmp_path,
        batch_model=model,
        plugin=ParallelLlmPlugin(read_paths=["allowed"], write_paths=["allowed"], max_prompts=1),
    )

    read_result = await _call_parallel(parent, _file("prompts.json"))
    write_result = await _call_parallel(parent, {**_inline(["x"]), "output_file": "outside.json"})
    cap_result = await _call_parallel(parent, _inline(["x", "y"]))

    assert read_result["content"] == "path is outside allowed read paths: prompts.json"
    assert write_result["content"] == "path is outside allowed write paths: outside.json"
    assert cap_result["content"] == "parallel_llm prompts exceed configured limit 1"
    assert model.calls == []


async def test_parallel_llm_does_not_retry_custom_model_sessions(tmp_path: Path) -> None:
    model = BatchModel(outcomes=[ProviderError("provider error 429: slow", status_code=429), "unused"])
    parent = _parent(tmp_path, model)

    result = await _call_parallel(parent, _inline(["x"]))

    assert result["payload"]["results"] == [{"index": 0, "ok": False, "error": "provider error 429: slow"}]
    assert result["payload"]["model_requests"] == 1
    assert model.session_requests == 1


async def test_parallel_llm_concurrency_cap(tmp_path: Path) -> None:
    model = BatchModel(delay=0.02)
    parent = _parent(tmp_path, model)

    await _call_parallel(parent, {**_inline(["a", "b", "c", "d"]), "max_concurrency": 2})

    assert model.max_in_flight <= 2


async def test_parallel_llm_does_not_inherit_parent_system_prompt(tmp_path: Path) -> None:
    model = BatchModel()
    parent = _parent(tmp_path, model, system_prompt="DISTINCTIVE_PARENT_MARKER")

    await _call_parallel(parent, _inline(["x"]))

    assert model.calls[0]["instructions"] == ""


async def test_parallel_llm_text_only_stray_tool_call_is_sparse_failure(tmp_path: Path) -> None:
    model = BatchModel(outcomes=[
        ModelTurn(text="ignored", tool_calls=[ModelToolCall(id="call_lookup", name="lookup", arguments="{}")])
    ])
    parent = _parent(tmp_path, model)

    result = await _call_parallel(parent, {**_inline(["x"]), "max_concurrency": 1})

    assert result["payload"]["results"] == [
        {"index": 0, "ok": False, "error": "parallel_llm does not execute nested tool calls"}
    ]
    assert result["payload"]["model_requests"] == 1


async def test_custom_parallel_llm_prompted_output_parses_json_results(tmp_path: Path) -> None:
    model = BatchModel(outcomes=['{"name":"Ada","age":37}'])
    tool = ParallelLlmTool(
        name="parallel_extract",
        model=model,
        root=tmp_path,
        output_type=ExtractedPerson,
        output_mode="prompted",
    )

    result = await _call_custom_tool(tool, {**_inline(["extract"]), "system": "Return JSON.", "max_concurrency": 1})

    assert result["payload"]["results"] == [{"index": 0, "ok": True, "result": {"name": "Ada", "age": 37}}]
    assert "Return JSON." in model.calls[0]["instructions"]
    assert "JSON Schema" in model.calls[0]["instructions"]


async def test_custom_parallel_llm_invalid_structured_text_returns_failure(tmp_path: Path) -> None:
    model = BatchModel(outcomes=["not json"])
    tool = ParallelLlmTool(
        model=model,
        root=tmp_path,
        output_type=ExtractedPerson,
        output_mode="prompted",
        output_retries=0,
    )

    result = await _call_custom_tool(tool, {**_inline(["extract"]), "max_concurrency": 1})

    assert result["payload"]["succeeded"] == 0
    assert result["payload"]["failed"] == 1
    assert result["payload"]["model_requests"] == 1
    assert result["payload"]["results"][0]["ok"] is False
    assert result["payload"]["results"][0]["error"].startswith("output validation failed:")


async def test_custom_parallel_llm_structured_retry_uses_fresh_session(tmp_path: Path) -> None:
    model = BatchModel(outcomes=["not json", '{"name":"Ada","age":37}'])
    tool = ParallelLlmTool(
        model=model,
        root=tmp_path,
        output_type=ExtractedPerson,
        output_mode="prompted",
        output_retries=1,
    )

    result = await _call_custom_tool(tool, {**_inline(["extract this"]), "max_concurrency": 1})

    assert result["payload"]["results"] == [{"index": 0, "ok": True, "result": {"name": "Ada", "age": 37}}]
    assert result["payload"]["model_requests"] == 2
    assert model.session_requests == 2
    assert "failed structured output validation" in model.calls[1]["prompt"]
    assert "extract this" in model.calls[1]["prompt"]


async def test_custom_parallel_llm_tool_mode_accepts_final_result(tmp_path: Path) -> None:
    model = BatchModel(outcomes=[
        ModelTurn(tool_calls=[ModelToolCall(id="call_final", name="final_result", arguments='{"name":"Ada","age":37}')])
    ])
    tool = ParallelLlmTool(model=model, root=tmp_path, output_type=ExtractedPerson, output_mode="tool")

    result = await _call_custom_tool(tool, {**_inline(["extract"]), "max_concurrency": 1})

    assert result["payload"]["results"] == [{"index": 0, "ok": True, "result": {"name": "Ada", "age": 37}}]
    assert [tool_schema["name"] for tool_schema in model.calls[0]["tools"]] == ["final_result"]


async def test_custom_parallel_llm_tool_mode_rejects_text_without_tool_call(tmp_path: Path) -> None:
    model = BatchModel(outcomes=['{"name":"Ada","age":37}'])
    tool = ParallelLlmTool(
        model=model,
        root=tmp_path,
        output_type=ExtractedPerson,
        output_mode="tool",
        output_retries=0,
    )

    result = await _call_custom_tool(tool, {**_inline(["extract"]), "max_concurrency": 1})

    assert result["payload"]["results"][0]["ok"] is False
    assert "final_result" in result["payload"]["results"][0]["error"]


async def test_custom_parallel_llm_structured_continue_is_sparse_failure(tmp_path: Path) -> None:
    model = BatchModel(outcomes=[
        ModelTurn(tool_calls=[ModelToolCall(id="call_lookup", name="lookup", arguments="{}")])
    ])
    tool = ParallelLlmTool(
        model=model,
        root=tmp_path,
        output_type=ExtractedPerson,
        output_mode="prompted",
        output_retries=1,
    )

    result = await _call_custom_tool(tool, {**_inline(["extract"]), "max_concurrency": 1})

    assert result["payload"]["results"] == [
        {"index": 0, "ok": False, "error": "parallel_llm does not execute nested tool calls"}
    ]
    assert result["payload"]["model_requests"] == 1


async def test_custom_parallel_llm_tool_mode_stray_tool_call_is_sparse_failure(tmp_path: Path) -> None:
    model = BatchModel(outcomes=[
        ModelTurn(tool_calls=[ModelToolCall(id="call_lookup", name="lookup", arguments="{}")])
    ])
    tool = ParallelLlmTool(
        model=model,
        root=tmp_path,
        output_type=ExtractedPerson,
        output_mode="tool",
        output_retries=1,
    )

    result = await _call_custom_tool(tool, {**_inline(["extract"]), "max_concurrency": 1})

    assert result["payload"]["results"] == [
        {"index": 0, "ok": False, "error": "parallel_llm does not execute nested tool calls"}
    ]
    assert result["payload"]["model_requests"] == 1


async def test_custom_parallel_llm_unexpected_final_result_pattern_is_sparse_failure(tmp_path: Path) -> None:
    model = BatchModel(outcomes=[
        ModelTurn(tool_calls=[
            ModelToolCall(id="call_final_1", name="final_result", arguments='{"name":"Ada","age":37}'),
            ModelToolCall(id="call_final_2", name="final_result", arguments='{"name":"Grace","age":85}'),
        ])
    ])
    tool = ParallelLlmTool(
        model=model,
        root=tmp_path,
        output_type=ExtractedPerson,
        output_mode="tool",
        output_retries=1,
    )

    result = await _call_custom_tool(tool, {**_inline(["extract"]), "max_concurrency": 1})

    assert result["payload"]["results"] == [
        {"index": 0, "ok": False, "error": "final_result must be the only tool call in its turn"}
    ]
    assert result["payload"]["model_requests"] == 1


async def test_custom_parallel_llm_native_mode_sends_structured_output(tmp_path: Path) -> None:
    model = BatchModel(outcomes=['{"name":"Ada","age":37}'])
    model.capabilities = ModelCapabilities(supports_json_schema_output=True, default_structured_output_mode="native")
    tool = ParallelLlmTool(model=model, root=tmp_path, output_type=ExtractedPerson, output_mode="native")

    result = await _call_custom_tool(tool, {**_inline(["extract"]), "max_concurrency": 1})

    assert result["payload"]["results"] == [{"index": 0, "ok": True, "result": {"name": "Ada", "age": 37}}]
    assert model.calls[0]["tools"] == []
    assert model.calls[0]["structured_output"].name == "final_result"


async def test_custom_parallel_llm_auto_resolves_model_capabilities(tmp_path: Path) -> None:
    model = BatchModel(outcomes=['{"name":"Ada","age":37}'])
    model.capabilities = ModelCapabilities(supports_json_schema_output=True, default_structured_output_mode="native")
    tool = ParallelLlmTool(model=model, root=tmp_path, output_type=ExtractedPerson)

    result = await _call_custom_tool(tool, {**_inline(["extract"]), "max_concurrency": 1})

    assert result["payload"]["results"][0]["result"] == {"name": "Ada", "age": 37}
    assert model.calls[0]["structured_output"].name == "final_result"


def test_custom_parallel_llm_rejects_unknown_output_mode(tmp_path: Path) -> None:
    invalid_mode: Any = "nativee"

    with pytest.raises(ValueError, match="unknown output_mode"):
        ParallelLlmTool(model=BatchModel(), root=tmp_path, output_type=ExtractedPerson, output_mode=invalid_mode)


async def test_custom_parallel_llm_text_mode_rejects_structured_output_type(tmp_path: Path) -> None:
    model = BatchModel()
    tool = ParallelLlmTool(
        model=model,
        root=tmp_path,
        output_type=ExtractedPerson,
        output_mode="text",
    )

    result = await _invoke_tool(tool.spec(), {**_inline(["extract"]), "max_concurrency": 1})
    envelope = json.loads(result.to_json())

    assert envelope["ok"] is False
    assert envelope["content"] == "text output mode requires output_type=str"
    assert model.calls == []


async def test_custom_parallel_llm_closes_owned_model_when_schema_resolution_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    model = BatchModel()

    def fake_infer_model(model_ref: str, **kwargs: Any) -> BatchModel:
        return model

    monkeypatch.setattr("thinharness.providers.infer_model", fake_infer_model)
    tool = ParallelLlmTool(
        model="openai:gpt-child",
        root=tmp_path,
        output_type=ExtractedPerson,
        output_mode="text",
    )

    result = await _invoke_tool(tool.spec(), {**_inline(["extract"]), "max_concurrency": 1})
    envelope = json.loads(result.to_json())

    assert envelope["ok"] is False
    assert envelope["content"] == "text output mode requires output_type=str"
    assert model.provider.closed is True


async def test_builtin_parallel_llm_stays_text_only_with_json_output(tmp_path: Path) -> None:
    model = BatchModel(outcomes=['{"name":"Ada","age":37}'])
    model.capabilities = ModelCapabilities(supports_json_schema_output=True, default_structured_output_mode="native")
    parent = _parent(tmp_path, model)
    spec = next(tool for tool in parent.tools if tool.name == "parallel_llm")

    result = await _call_parallel(parent, {**_inline(["extract"]), "max_concurrency": 1})

    assert result["payload"]["results"] == [{"index": 0, "ok": True, "result": '{"name":"Ada","age":37}'}]
    assert model.calls[0]["structured_output"] is None
    assert model.calls[0]["tools"] == []
    assert "output_type" not in spec.response_tool()["parameters"]["properties"]


async def test_parallel_llm_cancellation_propagates(tmp_path: Path) -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    class BlockingModel(BatchModel):
        async def complete(self, prompt: str, instructions: str, tools: list[dict[str, Any]], structured_output: Any = None) -> ModelTurn:
            """Block until the test cancels the surrounding task."""
            started.set()
            await release.wait()
            return ModelTurn(text="late")

    parent = _parent(tmp_path, BlockingModel())
    task = asyncio.create_task(_call_parallel(parent, _inline(["x"])))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    release.set()


def test_atomic_write_json_cleans_temp_file_on_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    output_path = tmp_path / "results.json"

    def fail_replace(src: Path, dst: Path) -> None:
        raise RuntimeError("replace failed")

    monkeypatch.setattr("thinharness.tools.parallel_llm.os.replace", fail_replace)
    with pytest.raises(RuntimeError, match="replace failed"):
        _atomic_write_json(output_path, "{}")

    assert not output_path.exists()
    assert list(tmp_path.glob(".results.json.*.tmp")) == []


def test_parallel_llm_tool_validates_request_retry_settings(tmp_path: Path) -> None:
    for kwargs in (
        {"request_retries": -1},
        {"request_retries": 11},
        {"request_retry_backoff": -0.1},
        {"request_retry_backoff": float("inf")},
        {"request_retry_backoff": float("nan")},
    ):
        with pytest.raises(ValueError):
            ParallelLlmTool(model="openai:test", root=tmp_path, **kwargs)

    assert ParallelLlmTool(model="openai:test", root=tmp_path, request_retries=0).request_retries == 0
    assert ParallelLlmTool(model="openai:test", root=tmp_path, request_retries=10).request_retries == 10


async def test_parallel_llm_builtin_provider_uses_one_transport_retry_budget(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, text="temporary", request=request)
        return httpx.Response(200, json={"id": "response", "output_text": "done"}, request=request)

    async def no_sleep(_delay: float) -> None:
        return None

    monkeypatch.setattr("thinharness.providers.asyncio.sleep", no_sleep)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        model = OpenAIResponsesModel(
            "test-model",
            provider=OpenAIProvider(api_key="key", request_retries=1, request_retry_backoff=0, http_client=client),
        )
        tool = ParallelLlmTool(model=model, root=tmp_path)
        result = await _call_custom_tool(tool, _inline(["x"]))

    assert calls == 2
    assert result["payload"]["results"] == [{"index": 0, "ok": True, "result": "done"}]
    assert result["payload"]["model_requests"] == 1


def test_parallel_llm_tool_custom_spec_and_model_resolution(tmp_path: Path) -> None:
    tool = ParallelLlmTool(
        name="parallel_extract",
        description="Extract fields.",
        model="openai:gpt-child",
        root=tmp_path,
        read_paths=["."],
        write_paths=["."],
        api_key="key",
        base_url="https://example.test",
        request_timeout=7,
        request_retries=2,
        request_retry_backoff=0.5,
        temperature=0.3,
        max_tokens=2048,
        effort="medium",
        extra_body={"seed": 1},
    )

    spec = tool.spec()
    model, should_close = tool._resolve_model()

    assert spec.name == "parallel_extract"
    assert spec.description == "Extract fields."
    assert "kind" not in ToolSpec.__dataclass_fields__
    assert isinstance(model, OpenAIResponsesModel)
    assert should_close is True
    assert model.provider.api_key == "key"
    assert model.provider.base_url == "https://example.test"
    assert model.provider.timeout == 7
    assert model.provider.request_retries == 2
    assert model.provider.request_retry_backoff == 0.5
    assert model.settings == ModelSettings(temperature=0.3, max_tokens=2048, effort="medium", extra_body={"seed": 1})


async def test_parallel_llm_plugin_string_model_uses_its_provider_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, Any] = {}
    inferred = BatchModel()

    def fake_infer_model(model_ref: str, **kwargs: Any) -> BatchModel:
        captured["model_ref"] = model_ref
        captured["kwargs"] = kwargs
        return inferred

    monkeypatch.setattr("thinharness.providers.infer_model", fake_infer_model)
    plugin = ParallelLlmPlugin(
        "openai:gpt-cheap",
        api_key="plugin-key",
        base_url="https://example.test",
        max_tokens=4096,
        effort="low",
        request_retries=2,
        request_retry_backoff=0.25,
        temperature=0.2,
    )
    parent = _parent(tmp_path, BatchModel(), plugin=plugin)

    result = await _call_parallel(parent, _inline(["x"]))

    assert result["payload"]["succeeded"] == 1
    assert captured["model_ref"] == "openai:gpt-cheap"
    assert captured["kwargs"]["api_key"] == "plugin-key"
    assert captured["kwargs"]["base_url"] == "https://example.test"
    assert captured["kwargs"]["temperature"] == 0.2
    assert captured["kwargs"]["max_tokens"] == 4096
    assert captured["kwargs"]["effort"] == "low"
    assert captured["kwargs"]["request_retries"] == 2
    assert captured["kwargs"]["request_retry_backoff"] == 0.25
    assert inferred.provider.closed is True


def test_parallel_llm_plugin_composition_and_builtin_migration(tmp_path: Path) -> None:
    default_harness = Harness(HarnessConfig(root=tmp_path / "default"))
    selected_harness = Harness(HarnessConfig(root=tmp_path / "selected"), plugins=[ParallelLlmPlugin()])

    assert "parallel_llm" not in {tool.name for tool in default_harness.tools}
    assert "parallel_llm" in {tool.name for tool in selected_harness.tools}
    assert next(tool for tool in selected_harness.tools if tool.name == "parallel_llm").instructions == DEFAULT_PARALLEL_LLM_INSTRUCTIONS
    with pytest.raises(ValueError, match="SubagentsPlugin"):
        HarnessConfig(root=tmp_path / "bad", builtin_tools=["parallel_llm"])


async def test_parallel_llm_usage_accounting_in_harness_run(tmp_path: Path) -> None:
    model = HybridModel()
    harness = Harness(HarnessConfig(root=tmp_path, max_model_requests=3), model=model, plugins=[ParallelLlmPlugin()])

    result = await harness.run("go")

    assert result.text == "done:2"
    assert result.usage.tool_calls == 1
    assert result.usage.model_requests == 2
    tool_record = json.loads(result.tool_call_records[0]["output"])
    assert tool_record["metadata"]["model_requests"] == 2
    assert [call["prompt"] for call in model.calls] == ["a", "b"]


def test_parallel_llm_plugin_static_contract_and_fixed_names(tmp_path: Path) -> None:
    plugin = ParallelLlmPlugin(description="Batch now.", instructions="Use carefully.")
    harness = Harness(HarnessConfig(root=tmp_path), model=BatchModel(), plugins=[plugin])
    spec = harness.tools[0]

    assert spec.name == "parallel_llm"
    assert spec.description == "Batch now."
    assert spec.instructions == "Use carefully."
    assert "kind" not in ToolSpec.__dataclass_fields__
    assert spec.origin is not None
    assert spec.origin.plugin == "parallel_llm"
    assert spec.origin.source == "parallel_llm"
    with pytest.raises(AttributeError, match="fixed"):
        plugin.name = "other"
    with pytest.raises(AttributeError, match="fixed"):
        ParallelLlmPlugin.name = "other"
    with pytest.raises(TypeError, match="cannot override"):
        class RenamedParallelLlmPlugin(ParallelLlmPlugin):
            name = "other"


@pytest.mark.parametrize(
    "option",
    [
        {"api_key": "key"},
        {"base_url": "https://example.test"},
        {"request_timeout": 1},
        {"request_retries": 1},
        {"request_retry_backoff": 0.1},
        {"temperature": 0.1},
        {"max_tokens": 1},
        {"effort": "low"},
        {"extra_body": {}},
    ],
)
def test_parallel_llm_plugin_rejects_provider_settings_for_borrowed_models(option: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="valid only when"):
        ParallelLlmPlugin(**option)
    with pytest.raises(ValueError, match="valid only when"):
        ParallelLlmPlugin(BatchModel(), **option)


def test_parallel_llm_plugin_rejects_invalid_prompt_cap() -> None:
    with pytest.raises(ValueError, match="max_prompts"):
        ParallelLlmPlugin(max_prompts=0)


def test_parallel_llm_plugin_omits_default_sentinels_and_preserves_explicit_falsey_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict[str, Any]] = []
    real_tool = parallel_plugin_module.ParallelLlmTool

    def capture_tool(**kwargs: Any) -> ParallelLlmTool:
        captured.append(kwargs)
        return real_tool(**kwargs)

    monkeypatch.setattr(parallel_plugin_module, "ParallelLlmTool", capture_tool)
    context = PluginContext(root=tmp_path, model=BatchModel(), child_harnesses=FakeChildHarnessHost())

    ParallelLlmPlugin("openai:default").bind(context)
    ParallelLlmPlugin(
        "openai:explicit",
        api_key="",
        base_url="",
        request_timeout=0,
        request_retries=0,
        request_retry_backoff=0,
        temperature=0,
        max_tokens=1,
        effort="",
        extra_body={},
    ).bind(context)

    provider_names = {
        "api_key",
        "base_url",
        "request_timeout",
        "request_retries",
        "request_retry_backoff",
        "temperature",
        "max_tokens",
        "effort",
        "extra_body",
    }
    assert provider_names.isdisjoint(captured[0])
    assert {name: captured[1][name] for name in provider_names} == {
        "api_key": "",
        "base_url": "",
        "request_timeout": 0,
        "request_retries": 0,
        "request_retry_backoff": 0,
        "temperature": 0,
        "max_tokens": 1,
        "effort": "",
        "extra_body": {},
    }


def test_parallel_llm_constructor_and_public_values_stay_frozen_across_bindings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict[str, Any]] = []
    real_tool = parallel_plugin_module.ParallelLlmTool

    def capture_tool(**kwargs: Any) -> ParallelLlmTool:
        captured.append(kwargs)
        return real_tool(**kwargs)

    monkeypatch.setattr(parallel_plugin_module, "ParallelLlmTool", capture_tool)
    read_paths = ["inputs"]
    write_paths = ["outputs"]
    extra_body = {"nested": {"stable": True}}
    plugin = ParallelLlmPlugin(
        "openai:fixed",
        read_paths=read_paths,
        write_paths=write_paths,
        extra_body=extra_body,
    )
    read_paths[0] = "mutated"
    write_paths[0] = "mutated"
    extra_body["nested"]["stable"] = False
    returned = plugin.extra_body
    returned["nested"]["stable"] = False

    first_model = BatchModel()
    second_model = BatchModel()
    plugin.bind(PluginContext(root=tmp_path / "first", model=first_model, child_harnesses=FakeChildHarnessHost()))
    plugin.for_child().bind(
        PluginContext(root=tmp_path / "second", model=second_model, child_harnesses=FakeChildHarnessHost())
    )

    assert [entry["read_paths"] for entry in captured] == [["inputs"], ["inputs"]]
    assert [entry["write_paths"] for entry in captured] == [["outputs"], ["outputs"]]
    assert [entry["extra_body"] for entry in captured] == [
        {"nested": {"stable": True}},
        {"nested": {"stable": True}},
    ]


async def test_parallel_llm_plugin_reuse_borrows_each_harness_model(tmp_path: Path) -> None:
    plugin = ParallelLlmPlugin()
    first_model = BatchModel(outcomes=["first"])
    second_model = BatchModel(outcomes=["second"])
    first = _parent(tmp_path / "first", first_model, plugin=plugin)
    second = _parent(tmp_path / "second", second_model, plugin=plugin)

    first_result = await _call_parallel(first, _inline(["one"]))
    second_result = await _call_parallel(second, _inline(["two"]))

    assert first_result["payload"]["results"][0]["result"] == "first"
    assert second_result["payload"]["results"][0]["result"] == "second"
    assert [call["prompt"] for call in first_model.calls] == ["one"]
    assert [call["prompt"] for call in second_model.calls] == ["two"]


async def test_parallel_llm_plugin_borrows_explicit_model_without_closing_it(tmp_path: Path) -> None:
    explicit_model = BatchModel(outcomes=["explicit"])
    harness_model = BatchModel()
    harness = _parent(tmp_path, harness_model, plugin=ParallelLlmPlugin(explicit_model))

    result = await _call_parallel(harness, _inline(["x"]))
    await harness.aclose()

    assert result["payload"]["results"][0]["result"] == "explicit"
    assert explicit_model.provider.closed is False
    assert harness_model.provider.closed is False


async def test_parallel_llm_plugin_does_not_close_borrowed_harness_owned_model_during_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inferred = BatchModel(outcomes=["borrowed"])
    monkeypatch.setattr("thinharness.core.infer_model", lambda *_args, **_kwargs: inferred)
    harness = Harness(
        HarnessConfig(root=tmp_path, model="openai:owned"),
        plugins=[ParallelLlmPlugin()],
    )

    result = await _call_parallel(harness, _inline(["x"]))

    assert result["payload"]["results"][0]["result"] == "borrowed"
    assert inferred.provider.closed is False
    await harness.aclose()
    assert inferred.provider.closed is True


def test_parallel_llm_plugin_bind_is_io_free_and_does_not_infer_provider(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plugin = ParallelLlmPlugin("openai:gpt-child", read_paths=["future"], write_paths=["outputs"])

    def fail(*_args, **_kwargs):
        raise AssertionError("I/O or provider inference used during bind")

    monkeypatch.setattr(Path, "resolve", fail)
    monkeypatch.setattr(Path, "exists", fail)
    monkeypatch.setattr(Path, "stat", fail)
    monkeypatch.setattr("thinharness.providers.infer_model", fail)

    binding = plugin.bind(PluginContext(root=tmp_path, model=BatchModel(), child_harnesses=FakeChildHarnessHost()))
    assert binding.static.tools[0].name == "parallel_llm"


async def test_parallel_llm_plugin_string_model_closes_provider_after_request_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inferred = BatchModel(outcomes=[RuntimeError("failed")])
    monkeypatch.setattr("thinharness.providers.infer_model", lambda *_args, **_kwargs: inferred)
    harness = _parent(tmp_path, BatchModel(), plugin=ParallelLlmPlugin("openai:gpt-child"))

    result = await _call_parallel(harness, _inline(["x"]))

    assert result["payload"]["failed"] == 1
    assert inferred.provider.closed is True


async def test_parallel_llm_plugin_string_model_closes_provider_after_schema_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inferred = BatchModel()
    monkeypatch.setattr("thinharness.providers.infer_model", lambda *_args, **_kwargs: inferred)
    monkeypatch.setattr(
        "thinharness.tools.parallel_llm.resolve_output_schema_for_model",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("schema failed")),
    )
    harness = _parent(tmp_path, BatchModel(), plugin=ParallelLlmPlugin("openai:gpt-child"))

    result = await _call_parallel(harness, _inline(["x"]))

    assert result["ok"] is False
    assert result["content"] == "schema failed"
    assert inferred.provider.closed is True


async def test_parallel_llm_plugin_string_model_closes_provider_after_cancellation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = asyncio.Event()

    class BlockingModel(BatchModel):
        async def complete(self, prompt: str, instructions: str, tools: list[dict[str, Any]], structured_output: Any = None) -> ModelTurn:
            started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    inferred = BlockingModel()
    monkeypatch.setattr("thinharness.providers.infer_model", lambda *_args, **_kwargs: inferred)
    harness = _parent(tmp_path, BatchModel(), plugin=ParallelLlmPlugin("openai:gpt-child"))
    task = asyncio.create_task(_call_parallel(harness, _inline(["x"])))
    await started.wait()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert inferred.provider.closed is True
