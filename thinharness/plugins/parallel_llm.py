"""Parallel LLM plugin."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..tools.parallel_llm import (
    DEFAULT_PARALLEL_LLM_DESCRIPTION,
    DEFAULT_PARALLEL_LLM_INSTRUCTIONS,
    ParallelLlmTool,
)
from .base import PluginBinding, PluginContext, PluginContribution

if TYPE_CHECKING:
    from ..providers import Model


class _ParallelLlmPluginMeta(type):
    """Keep the parallel LLM plugin name fixed on the class hierarchy."""

    def __setattr__(cls, attribute: str, value: object) -> None:
        if attribute == "name":
            raise AttributeError("ParallelLlmPlugin.name is fixed to 'parallel_llm'")
        super().__setattr__(attribute, value)

    def __delattr__(cls, attribute: str) -> None:
        if attribute == "name":
            raise AttributeError("ParallelLlmPlugin.name is fixed to 'parallel_llm'")
        super().__delattr__(attribute)


class ParallelLlmPlugin(metaclass=_ParallelLlmPluginMeta):
    """Expose one root-scoped text-only parallel completion tool."""

    name = "parallel_llm"

    def __init_subclass__(cls) -> None:
        """Reject subclasses that replace the fixed plugin name."""
        super().__init_subclass__()
        if "name" in cls.__dict__:
            raise TypeError("ParallelLlmPlugin subclasses cannot override the fixed name 'parallel_llm'")

    def __setattr__(self, attribute: str, value: object) -> None:
        """Reject instance changes to the fixed plugin name."""
        if attribute == "name":
            raise AttributeError("ParallelLlmPlugin.name is fixed to 'parallel_llm'")
        super().__setattr__(attribute, value)

    def __init__(
        self,
        model: Model | str | None = None,
        *,
        description: str = DEFAULT_PARALLEL_LLM_DESCRIPTION,
        instructions: str | None = DEFAULT_PARALLEL_LLM_INSTRUCTIONS,
        read_paths: Sequence[str | Path] | None = None,
        write_paths: Sequence[str | Path] | None = None,
        max_prompts: int = 100,
        api_key: str | None = None,
        base_url: str | None = None,
        request_timeout: int | None = None,
        request_retries: int | None = None,
        request_retry_backoff: float | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        effort: str | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        if max_prompts < 1:
            raise ValueError("max_prompts must be >= 1")
        provider_options = {
            "api_key": api_key,
            "base_url": base_url,
            "request_timeout": request_timeout,
            "request_retries": request_retries,
            "request_retry_backoff": request_retry_backoff,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "effort": effort,
            "extra_body": extra_body,
        }
        if not isinstance(model, str):
            supplied = next((name for name, value in provider_options.items() if value is not None), None)
            if supplied is not None:
                raise ValueError(f"{supplied} is valid only when ParallelLlmPlugin model is a string")

        self.model = model
        self.description = description
        self.instructions = instructions
        self.read_paths = tuple(read_paths) if read_paths is not None else None
        self.write_paths = tuple(write_paths) if write_paths is not None else None
        self.max_prompts = max_prompts
        self.api_key = api_key
        self.base_url = base_url
        self.request_timeout = request_timeout
        self.request_retries = request_retries
        self.request_retry_backoff = request_retry_backoff
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.effort = effort
        self.extra_body = dict(extra_body) if extra_body is not None else None

    def bind(self, context: PluginContext) -> PluginBinding:
        """Build the static tool with the canonical root and resolved model."""
        model = context.model if self.model is None else self.model
        provider_options: dict[str, Any] = {}
        for name in (
            "api_key",
            "base_url",
            "request_timeout",
            "request_retries",
            "request_retry_backoff",
            "temperature",
            "max_tokens",
            "effort",
            "extra_body",
        ):
            value = getattr(self, name)
            if value is not None:
                provider_options[name] = value
        tool = ParallelLlmTool(
            model=model,
            root=context.root,
            description=self.description,
            instructions=self.instructions,
            read_paths=list(self.read_paths) if self.read_paths is not None else None,
            write_paths=list(self.write_paths) if self.write_paths is not None else None,
            max_prompts=self.max_prompts,
            _root_is_resolved=True,
            **provider_options,
        )
        return PluginBinding(static=PluginContribution(tools=(tool.spec(),)))


__all__ = ["ParallelLlmPlugin"]
