"""Parallel one-shot LLM completion tool."""

from __future__ import annotations

import asyncio
import json
import os
import random
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import Field, model_validator

from .base import Json, PathPolicy, PathValidationError, StrictArgs, ToolResult, ToolSpec, coerce_args

if TYPE_CHECKING:
    from ..core import Harness
    from ..providers import Model, ProviderError


RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})
PROMPTS_FILE_ERROR = "prompts_file must be a non-empty JSON array of strings"
DEFAULT_PARALLEL_LLM_DESCRIPTION = (
    "Run N independent prompts as one-shot LLM completions in parallel. Each call is stateless: no tools, no memory, no continuation "
    "- only the model's text response is returned. Use this when you have a batch of independent prompts (classify, summarize, translate). "
    "Pass exactly one prompt source: either prompts or prompts_file. Do not include the unused prompt source field. For multi-step work, use the "
    "subagent tool instead. For large batches, pass output_file and read it back rather than receiving full results inline. If you need the parent "
    "harness system prompt, include the relevant instructions in system; it is not inherited automatically. The tool's model is host-configured "
    "and cannot be changed by tool arguments."
)


class ParallelLlmArgs(StrictArgs):
    """Arguments for parallel one-shot LLM completions."""

    prompts: list[str] | None = Field(default=None, description="Inline prompt batch. Use this or prompts_file, never both.")
    prompts_file: str | None = Field(default=None, description="Path to a JSON array of strings. Use this or prompts, never both.")
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
        for field in ("prompts_file", "system", "output_file"):
            if normalized.get(field) == "":
                normalized[field] = None
        return normalized

    @model_validator(mode="after")
    def validate_prompt_source(self) -> ParallelLlmArgs:
        """Require exactly one prompt source."""
        if (self.prompts is None) == (self.prompts_file is None):
            raise ValueError("exactly one of prompts or prompts_file must be set")
        return self


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
        read_paths: list[str | Path] | None = None,
        write_paths: list[str | Path] | None = None,
        max_prompts: int = 100,
        max_attempts: int = 4,
        api_key: str | None = None,
        base_url: str | None = None,
        request_timeout: int = 120,
        temperature: float | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self.root = Path(root).expanduser().resolve()
        self.read_policy = PathPolicy(self.root, read_paths, "read")
        self.write_policy = PathPolicy(self.root, write_paths, "write")
        if max_prompts < 1:
            raise ValueError("max_prompts must be >= 1")
        if max_attempts < 1 or max_attempts > 10:
            raise ValueError("max_attempts must be between 1 and 10")
        self.max_prompts = max_prompts
        self.max_attempts = max_attempts
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
        sem = asyncio.Semaphore(args.max_concurrency)
        instructions = args.system if args.system is not None else ""
        model_requests = 0

        async def run_one(index: int, prompt: str) -> Json:
            """Run one prompt with retry and return a sparse result entry."""
            nonlocal model_requests
            try:
                for attempt in range(self.max_attempts):
                    try:
                        async with sem:
                            session = batch_model.new_session()
                            model_requests += 1
                            turn = await session.start(prompt=prompt, instructions=instructions, tools=[])
                            return {"index": index, "ok": True, "result": turn.text}
                    except ProviderError as exc:
                        if attempt == self.max_attempts - 1 or not _is_retryable(exc):
                            return {"index": index, "ok": False, "error": str(exc)}
                        await _sleep_retry(_retry_delay(attempt))
            except Exception as exc:
                return {"index": index, "ok": False, "error": f"{type(exc).__name__}: {exc}"}
            raise AssertionError("unreachable parallel_llm retry loop exit")

        try:
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
    return ParallelLlmTool(
        model=model,
        model_ref=model_ref,
        root=parent.root,
        read_paths=parent.config.read_paths,
        write_paths=parent.config.write_paths,
        max_prompts=parent.config.parallel_llm_max_prompts,
        max_attempts=parent.config.parallel_llm_max_attempts,
        api_key=api_key,
        base_url=base_url,
        request_timeout=parent.config.request_timeout,
        temperature=parent.config.builtin_parallel_llm_temperature
        if parent.config.builtin_parallel_llm_temperature is not None
        else parent.config.temperature,
        extra_body=parent.config.extra_body,
    ).spec()


def _load_prompts(args: ParallelLlmArgs, read_policy: PathPolicy) -> list[str]:
    """Load inline prompts or parse a prompts_file under the read policy."""
    if args.prompts is not None:
        return args.prompts
    assert args.prompts_file is not None
    path = read_policy.resolve(args.prompts_file)
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
