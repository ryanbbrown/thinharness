"""Parallel one-shot LLM completion tool."""

from __future__ import annotations

import asyncio
import json
import os
import random
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import Field, model_validator

from ..defaults import DEFAULT_PARALLEL_LLM_DESCRIPTION, DEFAULT_PARALLEL_LLM_INSTRUCTIONS
from ..output import (
    OUTPUT_MODES,
    OutputMode,
    OutputSchema,
    OutputSpec,
    OutputTurnDecision,
    resolve_output_schema_for_model,
    resolve_turn_output,
    structured_instructions,
    structured_retry_prompt,
)
from .base import Json, PathPolicy, PathValidationError, StrictArgs, ToolResult, ToolSpec, coerce_args

if TYPE_CHECKING:
    from ..core import Harness
    from ..providers import Model, ProviderError


RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})
PROMPTS_FILE_ERROR = "source.path must point to a non-empty JSON array of strings"


class InlinePromptSource(StrictArgs):
    """Inline prompts for parallel one-shot LLM completions."""

    kind: Literal["inline"]
    prompts: list[str] = Field(description="Inline prompt batch.")


class FilePromptSource(StrictArgs):
    """Prompt file source for parallel one-shot LLM completions."""

    kind: Literal["file"]
    path: str = Field(description="Path to a JSON array of strings.")


PromptSource = Annotated[InlinePromptSource | FilePromptSource, Field(discriminator="kind")]


class ParallelLlmArgs(StrictArgs):
    """Arguments for parallel one-shot LLM completions."""

    source: PromptSource = Field(description="Prompt source. Use kind='inline' for inline prompts or kind='file' for a prompt file.")
    system: str | None = Field(default=None, description="Shared instructions for every prompt. The parent harness system prompt is not inherited.")
    output_file: str | None = Field(default=None, description="Optional workspace path for full JSON results. Inline content returns only a summary.")
    max_concurrency: int = Field(default=8, ge=1, le=32, description="Maximum in-flight prompt completions.")

    @model_validator(mode="before")
    @classmethod
    def normalize_blank_optional_strings(cls, data: object) -> object:
        """Treat blank optional string fields as omitted."""
        if not isinstance(data, dict):
            return data
        normalized = dict(data)
        for field in ("system", "output_file"):
            if normalized.get(field) == "":
                normalized[field] = None
        return normalized


class ParallelLlmTool:
    """Configurable normal ToolSpec wrapper for parallel one-shot LLM calls."""

    def __init__(
        self,
        *,
        model: Model | str,
        model_ref: str | None = None,
        root: str | Path = ".",
        name: str = "parallel_llm",
        description: str = DEFAULT_PARALLEL_LLM_DESCRIPTION,
        instructions: str | None = None,
        read_paths: list[str | Path] | None = None,
        write_paths: list[str | Path] | None = None,
        max_prompts: int = 100,
        max_attempts: int = 4,
        api_key: str | None = None,
        base_url: str | None = None,
        request_timeout: int = 120,
        temperature: float | None = None,
        extra_body: dict[str, Any] | None = None,
        output_type: OutputSpec | None = None,
        output_mode: OutputMode = "auto",
        output_retries: int = 1,
    ) -> None:
        self.name = name
        self.description = description
        self.instructions = instructions
        self.root = Path(root).expanduser().resolve()
        self.read_policy = PathPolicy(self.root, read_paths, "read")
        self.write_policy = PathPolicy(self.root, write_paths, "write")
        if max_prompts < 1:
            raise ValueError("max_prompts must be >= 1")
        if max_attempts < 1 or max_attempts > 10:
            raise ValueError("max_attempts must be between 1 and 10")
        if output_retries < 0:
            raise ValueError("output_retries must be >= 0")
        if output_mode not in OUTPUT_MODES:
            raise ValueError(f"unknown output_mode: {output_mode}")
        self.max_prompts = max_prompts
        self.max_attempts = max_attempts
        self.output_type: OutputSpec | None = output_type
        self.output_mode: OutputMode = output_mode
        self.output_retries = output_retries
        self._model = model if not isinstance(model, str) else None
        self.model_ref = model if isinstance(model, str) else model_ref
        self.api_key = api_key
        self.base_url = base_url
        self.request_timeout = request_timeout
        self.temperature = temperature
        self.extra_body = extra_body or {}

    def spec(self) -> ToolSpec:
        """Return this parallel LLM tool as a normal ToolSpec."""
        async def handler(raw_args: ParallelLlmArgs | Json) -> ToolResult:
            """Run a batch of independent prompts in parallel."""
            args = coerce_args(raw_args, ParallelLlmArgs)
            try:
                return await self.run(args)
            except PathValidationError as exc:
                return ToolResult(False, str(exc), {"error_type": "PathValidationError"})
            except ValueError as exc:
                return ToolResult(False, str(exc), {"error_type": "ValueError"})

        return ToolSpec(
            self.name,
            self.description,
            ParallelLlmArgs,
            handler,
            metadata={"framework_tool": "parallel_llm"},
            instructions=self.instructions,
            background="model",
        )

    async def run(self, args: ParallelLlmArgs) -> ToolResult:
        """Run the parallel LLM batch and return either results or a file summary."""
        from ..providers import ProviderError

        prompts = _load_prompts(args, self.read_policy)
        if not prompts:
            raise ValueError("prompts must be non-empty")
        if len(prompts) > self.max_prompts:
            raise ValueError(f"{self.name} prompts exceed configured limit {self.max_prompts}")

        output_path = self.write_policy.resolve(args.output_file) if args.output_file is not None else None
        batch_model, should_close = self._resolve_model()
        try:
            output_schema = resolve_output_schema_for_model(batch_model, self.output_type, self.output_mode)
            sem = asyncio.Semaphore(args.max_concurrency)
            instructions = structured_instructions(args.system or "", output_schema)
            tools = output_schema.synthetic_tools() if output_schema is not None else []
            structured_output = output_schema.structured_output_request() if output_schema is not None else None
            model_requests = 0

            async def request_turn(request_prompt: str):
                """Run one provider request with provider-level retry."""
                nonlocal model_requests
                for attempt in range(self.max_attempts):
                    try:
                        async with sem:
                            session = batch_model.new_session()
                            model_requests += 1
                            return await session.start(
                                prompt=request_prompt,
                                instructions=instructions,
                                tools=tools,
                                structured_output=structured_output,
                            )
                    except ProviderError as exc:
                        if attempt == self.max_attempts - 1 or not _is_retryable(exc):
                            raise
                        await _sleep_retry(_retry_delay(attempt))
                raise AssertionError("unreachable parallel_llm provider retry loop exit")

            async def run_one(index: int, prompt: str) -> Json:
                """Run one prompt with retry and return a sparse result entry."""
                request_prompt = prompt
                try:
                    for output_attempt in range(self.output_retries + 1):
                        try:
                            turn = await request_turn(request_prompt)
                        except ProviderError as exc:
                            return {"index": index, "ok": False, "error": str(exc)}
                        decision = resolve_turn_output(turn, output_schema)
                        if decision.kind == "final":
                            return _success_entry(index, decision.text, decision.output, output_schema)
                        if decision.kind in {"retry_user_message", "retry_tool_output"}:
                            if output_attempt == self.output_retries:
                                return _failure_entry(index, decision)
                            assert decision.error is not None, "structured-output retry requires validation error text"
                            request_prompt = structured_retry_prompt(prompt, decision.error)
                            continue
                        if decision.kind == "continue":
                            return {"index": index, "ok": False, "error": "parallel_llm does not execute nested tool calls"}
                        return {"index": index, "ok": False, "error": decision.unexpected_message}
                except Exception as exc:
                    return {"index": index, "ok": False, "error": f"{type(exc).__name__}: {exc}"}
                raise AssertionError("unreachable parallel_llm output retry loop exit")

            results = await asyncio.gather(*(run_one(index, prompt) for index, prompt in enumerate(prompts)))
        finally:
            if should_close:
                aclose = getattr(batch_model.provider, "aclose", None)
                if aclose is not None:
                    await aclose()

        payload = _batch_payload(results, model_requests)
        metadata: Json = {
            "total": payload["total"],
            "succeeded": payload["succeeded"],
            "failed": payload["failed"],
            "model_requests": model_requests,
        }
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            file_text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
            _atomic_write_json(output_path, file_text)
            summary = {key: payload[key] for key in ("total", "succeeded", "failed", "model_requests")}
            summary["output_file"] = args.output_file
            failed_indices = [item["index"] for item in results if item.get("ok") is False]
            if failed_indices:
                summary["failed_indices"] = failed_indices
            content = json.dumps(summary, ensure_ascii=False, separators=(",", ":"))
            metadata["output_file"] = args.output_file
            return ToolResult(True, content, metadata=metadata)

        content = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        return ToolResult(True, content, metadata=metadata)

    def _resolve_model(self) -> tuple[Model, bool]:
        """Return the configured model or create a fresh configured model."""
        from ..providers import infer_model

        if self._model is not None:
            return self._model, False
        if self.model_ref is None:
            raise ValueError("model reference is required")
        return infer_model(
            self.model_ref,
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.request_timeout,
            temperature=self.temperature,
            extra_body=self.extra_body,
        ), True


def create_parallel_llm_tool(parent: Harness) -> ToolSpec:
    """Create the built-in parallel LLM tool from a parent harness."""
    from ..providers import parse_model_ref, provider_prefix

    model: Model | str = parent.config.builtin_parallel_llm_model or parent.model
    model_ref = parent.config.builtin_parallel_llm_model or parent.model_ref
    api_key = parent.config.api_key
    base_url = parent.config.base_url
    if parent.config.builtin_parallel_llm_model is not None:
        builtin_provider, _ = parse_model_ref(parent.config.builtin_parallel_llm_model)
        parent_provider = provider_prefix(getattr(getattr(parent.model, "provider", None), "name", ""))
        if builtin_provider != parent_provider:
            api_key = None
            base_url = None
    description = DEFAULT_PARALLEL_LLM_DESCRIPTION
    instructions = DEFAULT_PARALLEL_LLM_INSTRUCTIONS
    if parent.config.tool_execution == "sequential":
        description = description.replace(
            " For large independent batches, `_background: true` is available when it lets other work continue.",
            "",
        )
        instructions = instructions.replace(
            "\n- For large independent batches, background mode is available; default to synchronous unless it is clearly useful.",
            "",
        )
    return ParallelLlmTool(
        model=model,
        model_ref=model_ref,
        root=parent.root,
        description=description,
        read_paths=parent.config.read_paths,
        write_paths=parent.config.write_paths,
        max_prompts=parent.config.parallel_llm_max_prompts,
        max_attempts=parent.config.parallel_llm_max_attempts,
        instructions=instructions,
        api_key=api_key,
        base_url=base_url,
        request_timeout=parent.config.request_timeout,
        temperature=parent.config.builtin_parallel_llm_temperature
        if parent.config.builtin_parallel_llm_temperature is not None
        else parent.config.temperature,
        extra_body=parent.config.extra_body,
    ).spec()


def _load_prompts(args: ParallelLlmArgs, read_policy: PathPolicy) -> list[str]:
    """Load inline prompts or parse a prompt file under the read policy."""
    if args.source.kind == "inline":
        return args.source.prompts
    path = read_policy.resolve(args.source.path)
    try:
        raw = path.read_text(encoding="utf-8")
        parsed = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(PROMPTS_FILE_ERROR) from exc
    if not isinstance(parsed, list) or not parsed or not all(isinstance(item, str) for item in parsed):
        raise ValueError(PROMPTS_FILE_ERROR)
    return parsed


def _batch_payload(results: list[Json], model_requests: int) -> Json:
    """Build the ordered batch payload."""
    succeeded = sum(1 for item in results if item.get("ok") is True)
    failed = len(results) - succeeded
    return {
        "total": len(results),
        "succeeded": succeeded,
        "failed": failed,
        "model_requests": model_requests,
        "results": results,
    }


def _success_entry(index: int, text: str, value: Any, output_schema: OutputSchema | None) -> Json:
    """Build one successful result entry."""
    if output_schema is None or output_schema.mode == "text":
        return {"index": index, "ok": True, "result": text}
    return {"index": index, "ok": True, "result": output_schema.adapter.dump_python(value, mode="json")}


def _failure_entry(index: int, decision: OutputTurnDecision) -> Json:
    """Build one structured-output validation failure entry."""
    assert decision.error is not None, "structured-output validation failure requires error text"
    return {"index": index, "ok": False, "error": f"output validation failed: {decision.error}"}


def _is_retryable(exc: ProviderError) -> bool:
    """Return whether a provider error should be retried."""
    if exc.status_code is None:
        return str(exc).startswith("provider request failed:")
    return exc.status_code in RETRYABLE_STATUS


def _retry_delay(attempt: int) -> float:
    """Return seconds to wait before the next attempt."""
    base = 2 ** attempt
    return base + random.uniform(0, base * 0.25)


async def _sleep_retry(delay: float) -> None:
    """Sleep for the computed retry delay."""
    await asyncio.sleep(delay)


def _atomic_write_json(output_path: Path, file_text: str) -> None:
    """Write file_text to output_path atomically via a same-dir temp file."""
    tmp = tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        delete=False,
    )
    temp_path = Path(tmp.name)
    try:
        with tmp:
            tmp.write(file_text)
        os.replace(temp_path, output_path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise
