from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from thinharness import Harness, HarnessConfig, ModelToolCall, ModelTurn, ToolOutput
from thinharness.providers import ModelSettings, OpenAIResponsesModel, ProviderError
from thinharness.tools.base import _invoke_tool
from thinharness.tools.parallel_llm import (
    PROMPTS_FILE_ERROR,
    ParallelLlmArgs,
    ParallelLlmTool,
    _atomic_write_json,
    _is_retryable,
    _retry_delay,
    create_parallel_llm_tool,
)


class BatchProvider:
    name = "OpenAI"


class BatchModel:
    def __init__(self, outcomes: list[Any] | None = None, *, delay: float = 0) -> None:
        self.model = "batch-model"
        self.provider = BatchProvider()
        self.api_key = "batch-key"
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

    async def complete(self, prompt: str, instructions: str, tools: list[dict[str, Any]]) -> ModelTurn:
        """Record one completion and return or raise the scripted outcome."""
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        self.calls.append({"prompt": prompt, "instructions": instructions, "tools": tools})
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            outcome = self.outcomes.pop(0) if self.outcomes else f"echo:{prompt}"
            if isinstance(outcome, BaseException):
                raise outcome
            return ModelTurn(text=str(outcome), raw={"output_text": str(outcome)})
        finally:
            self.in_flight -= 1


class BatchSession:
    def __init__(self, model: BatchModel) -> None:
        self.model = model

    async def start(self, *, prompt, instructions, tools, metadata=None, previous_response_id=None, structured_output=None, notices=None):
        """Run one batch completion."""
        return await self.model.complete(prompt, instructions, tools)

    async def continue_with_tools(self, outputs, *, tools, metadata=None, structured_output=None, notices=None):
        """Batch sessions never continue."""
        raise AssertionError("batch session should not continue")

    async def continue_with_user_message(self, message, *, tools, metadata=None, structured_output=None, notices=None):
        """Batch sessions never continue."""
        raise AssertionError("batch session should not continue")

    async def continue_with_user_prompt(self, *, prompt, instructions, tools, metadata=None, structured_output=None, notices=None):
        """Batch sessions never resume."""
        raise AssertionError("batch session should not resume")

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
    async def start(self, *, prompt, instructions, tools, metadata=None, previous_response_id=None, structured_output=None, notices=None):
        """Ask the harness to call parallel_llm."""
        return ModelTurn(
            raw={"id": "first"},
            tool_calls=[
                ModelToolCall(
                    id="call_1",
                    name="parallel_llm",
                    arguments=json.dumps({"prompts": ["a", "b"], "max_concurrency": 2}),
                )
            ],
        )

    async def continue_with_tools(self, outputs: list[ToolOutput], *, tools, metadata=None, structured_output=None, notices=None):
        """Finish after receiving the tool output."""
        parsed = json.loads(outputs[0].output)
        payload = json.loads(parsed["content"])
        return ModelTurn(text=f"done:{payload['succeeded']}", raw={"id": "done"})

    async def continue_with_user_message(self, message, *, tools, metadata=None, structured_output=None, notices=None):
        """Main session never receives corrections."""
        raise AssertionError("should not correct")

    async def continue_with_user_prompt(self, *, prompt, instructions, tools, metadata=None, structured_output=None, notices=None):
        """Main session never resumes."""
        raise AssertionError("should not resume")

    def dump_state(self):
        """Main session has no resume state."""
        return None


def _parent(tmp_path: Path, batch_model: BatchModel | None = None, **config: Any) -> Harness:
    """Build a harness parent for direct tool tests."""
    return Harness(HarnessConfig(root=tmp_path, **config), model=batch_model or BatchModel())


async def _call_parallel(parent: Harness, args: dict[str, Any]) -> dict[str, Any]:
    """Invoke parallel_llm through the normal tool envelope."""
    spec = create_parallel_llm_tool(parent)
    output = await _invoke_tool(spec, args)
    parsed = json.loads(output)
    if parsed["ok"]:
        parsed["payload"] = json.loads(parsed["content"])
    return parsed


async def test_parallel_llm_inline_prompts_return_compact_ordered_payload(tmp_path: Path) -> None:
    model = BatchModel(outcomes=["second", "first"], delay=0.01)
    parent = _parent(tmp_path, model)

    result = await _call_parallel(parent, {"prompts": ["p0", "p1"], "system": "shared", "max_concurrency": 2})

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


async def test_parallel_llm_prompts_file_and_output_file(tmp_path: Path) -> None:
    (tmp_path / "prompts.json").write_text(json.dumps(["a", "b"]), encoding="utf-8")
    model = BatchModel(outcomes=["ok", ProviderError("provider error 401: nope", status_code=401)])
    parent = _parent(tmp_path, model)

    result = await _call_parallel(parent, {"prompts_file": "prompts.json", "output_file": "nested/results.json"})

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
        ("", {"prompts_file": "prompts.json"}),
        ("   ", {"prompts_file": "prompts.json"}),
        ('{"prompt":"x"}', {"prompts_file": "prompts.json"}),
        ('["x", 1]', {"prompts_file": "prompts.json"}),
        ("[]", {"prompts_file": "prompts.json"}),
    ],
)
async def test_parallel_llm_rejects_bad_prompts_files(tmp_path: Path, content: str, args: dict[str, Any]) -> None:
    (tmp_path / "prompts.json").write_text(content, encoding="utf-8")
    parent = _parent(tmp_path)

    result = await _call_parallel(parent, args)

    assert result["ok"] is False
    assert result["content"] == PROMPTS_FILE_ERROR


@pytest.mark.parametrize("args", [{}, {"prompts": ["x"], "prompts_file": "prompts.json"}])
def test_parallel_llm_requires_exactly_one_prompt_source(args: dict[str, Any]) -> None:
    with pytest.raises(ValidationError, match="exactly one"):
        ParallelLlmArgs.model_validate(args)


def test_parallel_llm_args_normalize_blank_optional_paths_and_reject_model_override() -> None:
    args = ParallelLlmArgs.model_validate({"prompts": ["x"], "prompts_file": "", "output_file": "", "system": ""})

    assert args.prompts == ["x"]
    assert args.prompts_file is None
    assert args.output_file is None
    assert args.system is None
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ParallelLlmArgs.model_validate({"prompts": ["x"], "model": "openai:gpt-5-mini"})


async def test_parallel_llm_enforces_path_policies_and_prompt_cap(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    (tmp_path / "prompts.json").write_text('["x"]', encoding="utf-8")
    model = BatchModel()
    parent = _parent(
        tmp_path,
        batch_model=model,
        read_paths=["allowed"],
        write_paths=["allowed"],
        parallel_llm_max_prompts=1,
    )

    read_result = await _call_parallel(parent, {"prompts_file": "prompts.json"})
    write_result = await _call_parallel(parent, {"prompts": ["x"], "output_file": "outside.json"})
    cap_result = await _call_parallel(parent, {"prompts": ["x", "y"]})

    assert read_result["content"] == "path is outside allowed read paths: prompts.json"
    assert write_result["content"] == "path is outside allowed write paths: outside.json"
    assert cap_result["content"] == "parallel_llm prompts exceed configured limit 1"
    assert model.calls == []


async def test_parallel_llm_retries_retryable_errors_and_counts_attempts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr("thinharness.tools.parallel_llm._sleep_retry", fake_sleep)
    monkeypatch.setattr("thinharness.tools.parallel_llm._retry_delay", lambda attempt: float(attempt + 1))
    model = BatchModel(outcomes=[
        ProviderError("provider error 429: slow", status_code=429),
        ProviderError("provider request failed: offline"),
        "ok",
    ])
    parent = _parent(tmp_path, model, parallel_llm_max_attempts=3)

    result = await _call_parallel(parent, {"prompts": ["x"]})

    assert result["payload"]["results"] == [{"index": 0, "ok": True, "result": "ok"}]
    assert result["payload"]["model_requests"] == 3
    assert result["metadata"]["model_requests"] == 3
    assert sleeps == [1.0, 2.0]
    assert model.session_requests == 3


async def test_parallel_llm_fast_fails_non_retryable_errors(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    async def fail_sleep(delay: float) -> None:
        raise AssertionError("should not sleep")

    monkeypatch.setattr("thinharness.tools.parallel_llm._sleep_retry", fail_sleep)
    model = BatchModel(outcomes=[ProviderError("provider error 401: auth", status_code=401)])
    parent = _parent(tmp_path, model, parallel_llm_max_attempts=3)

    result = await _call_parallel(parent, {"prompts": ["x"]})

    assert result["payload"]["results"] == [{"index": 0, "ok": False, "error": "provider error 401: auth"}]
    assert model.session_requests == 1


async def test_parallel_llm_concurrency_cap(tmp_path: Path) -> None:
    model = BatchModel(delay=0.02)
    parent = _parent(tmp_path, model)

    await _call_parallel(parent, {"prompts": ["a", "b", "c", "d"], "max_concurrency": 2})

    assert model.max_in_flight <= 2


async def test_parallel_llm_does_not_inherit_parent_system_prompt(tmp_path: Path) -> None:
    model = BatchModel()
    parent = _parent(tmp_path, model, system_prompt="DISTINCTIVE_PARENT_MARKER")

    await _call_parallel(parent, {"prompts": ["x"]})

    assert model.calls[0]["instructions"] == ""


async def test_parallel_llm_cancellation_propagates(tmp_path: Path) -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    class BlockingModel(BatchModel):
        async def complete(self, prompt: str, instructions: str, tools: list[dict[str, Any]]) -> ModelTurn:
            """Block until the test cancels the surrounding task."""
            started.set()
            await release.wait()
            return ModelTurn(text="late")

    parent = _parent(tmp_path, BlockingModel())
    task = asyncio.create_task(_call_parallel(parent, {"prompts": ["x"]}))
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


def test_retry_classification_and_delay_bounds() -> None:
    assert _is_retryable(ProviderError("provider error 429", status_code=429))
    assert _is_retryable(ProviderError("provider error 503", status_code=503))
    assert _is_retryable(ProviderError("provider request failed: offline"))
    assert not _is_retryable(ProviderError("provider error 401", status_code=401))
    assert not _is_retryable(ProviderError("OPENAI_API_KEY is required for OpenAI"))
    for attempt in range(3):
        delay = _retry_delay(attempt)
        assert 2**attempt <= delay <= 2**attempt * 1.25


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
        temperature=0.3,
        extra_body={"seed": 1},
    )

    spec = tool.spec()
    model, should_close = tool._resolve_model()

    assert spec.name == "parallel_extract"
    assert spec.description == "Extract fields."
    assert isinstance(model, OpenAIResponsesModel)
    assert should_close is True
    assert model.provider.api_key == "key"
    assert model.provider.base_url == "https://example.test"
    assert model.provider.timeout == 7
    assert model.settings == ModelSettings(temperature=0.3, extra_body={"seed": 1})


async def test_builtin_parallel_llm_model_and_temperature_are_host_configured(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, Any] = {}
    inferred = BatchModel()

    def fake_infer_model(model_ref: str, **kwargs: Any) -> BatchModel:
        captured["model_ref"] = model_ref
        captured["kwargs"] = kwargs
        return inferred

    monkeypatch.setattr("thinharness.providers.infer_model", fake_infer_model)
    parent = _parent(
        tmp_path,
        BatchModel(),
        api_key="parent-key",
        base_url="https://example.test",
        builtin_parallel_llm_model="openai:gpt-cheap",
        builtin_parallel_llm_temperature=0.2,
    )

    result = await _call_parallel(parent, {"prompts": ["x"]})

    assert result["payload"]["succeeded"] == 1
    assert captured["model_ref"] == "openai:gpt-cheap"
    assert captured["kwargs"]["api_key"] == "parent-key"
    assert captured["kwargs"]["base_url"] == "https://example.test"
    assert captured["kwargs"]["temperature"] == 0.2


def test_parallel_llm_builtin_selection(tmp_path: Path) -> None:
    default_harness = Harness(HarnessConfig(root=tmp_path / "default"))
    selected_harness = Harness(HarnessConfig(root=tmp_path / "selected", builtin_tools=["parallel_llm"]))

    assert "parallel_llm" not in {tool.name for tool in default_harness.tools}
    assert "parallel_llm" in {tool.name for tool in selected_harness.tools}
    with pytest.raises(ValueError, match="parallel_llm"):
        Harness(HarnessConfig(root=tmp_path / "bad", builtin_tools=["not_a_tool"]))


async def test_parallel_llm_usage_accounting_in_harness_run(tmp_path: Path) -> None:
    model = HybridModel()
    harness = Harness(HarnessConfig(root=tmp_path, builtin_tools=["parallel_llm"], max_model_requests=3), model=model)

    result = await harness.run("go")

    assert result.text == "done:2"
    assert result.usage.tool_calls == 1
    assert result.usage.model_requests == 2
    tool_record = json.loads(result.tool_call_records[0]["output"])
    assert tool_record["metadata"]["model_requests"] == 2
    assert [call["prompt"] for call in model.calls] == ["a", "b"]
