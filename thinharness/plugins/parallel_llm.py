"""Parallel LLM plugin."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
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


@dataclass(frozen=True)
class _ParallelLlmConfig:
    model: Model | str | None
    description: str
    instructions: str | None
    read_paths: tuple[str | Path, ...] | None
    write_paths: tuple[str | Path, ...] | None
    max_prompts: int
    api_key: str | None
    base_url: str | None
    request_timeout: int | None
    request_retries: int | None
    request_retry_backoff: float | None
    temperature: float | None
    max_tokens: int | None
    effort: str | None
    extra_body: dict[str, Any] | None


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
    _config: _ParallelLlmConfig
    _frozen: bool

    def __init_subclass__(cls) -> None:
        """Reject subclasses that replace the fixed plugin name."""
        super().__init_subclass__()
        if "name" in cls.__dict__:
            raise TypeError("ParallelLlmPlugin subclasses cannot override the fixed name 'parallel_llm'")

    def __setattr__(self, attribute: str, value: object) -> None:
        """Reject configuration changes after construction."""
        if attribute == "name":
            raise AttributeError("ParallelLlmPlugin.name is fixed to 'parallel_llm'")
        if getattr(self, "_frozen", False):
            raise AttributeError("ParallelLlmPlugin configuration is frozen")
        object.__setattr__(self, attribute, value)

    def __delattr__(self, attribute: str) -> None:
        """Reject configuration deletion after construction."""
        if attribute == "name":
            raise AttributeError("ParallelLlmPlugin.name is fixed to 'parallel_llm'")
        if getattr(self, "_frozen", False):
            raise AttributeError("ParallelLlmPlugin configuration is frozen")
        object.__delattr__(self, attribute)

    def __getattr__(self, attribute: str) -> Any:
        """Expose immutable values or copies from the constructor snapshot."""
        config = object.__getattribute__(self, "_config")
        if not hasattr(config, attribute):
            raise AttributeError(attribute)
        value = getattr(config, attribute)
        return deepcopy(value) if attribute == "extra_body" else value

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
        object.__setattr__(self, "_config", _ParallelLlmConfig(
            model=model,
            description=description,
            instructions=instructions,
            read_paths=tuple(read_paths) if read_paths is not None else None,
            write_paths=tuple(write_paths) if write_paths is not None else None,
            max_prompts=max_prompts,
            api_key=api_key,
            base_url=base_url,
            request_timeout=request_timeout,
            request_retries=request_retries,
            request_retry_backoff=request_retry_backoff,
            temperature=temperature,
            max_tokens=max_tokens,
            effort=effort,
            extra_body=deepcopy(extra_body) if extra_body is not None else None,
        ))
        object.__setattr__(self, "_frozen", True)

    def for_child(self) -> ParallelLlmPlugin:
        """Reuse the frozen settings and resolve a borrowed model at child bind time."""
        return self

    def bind(self, context: PluginContext) -> PluginBinding:
        """Build the static tool with the canonical root and resolved model."""
        config = self._config
        model = context.model if config.model is None else config.model
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
            value = getattr(config, name)
            if value is not None:
                provider_options[name] = deepcopy(value)
        tool = ParallelLlmTool(
            model=model,
            root=context.root,
            description=config.description,
            instructions=config.instructions,
            read_paths=list(config.read_paths) if config.read_paths is not None else None,
            write_paths=list(config.write_paths) if config.write_paths is not None else None,
            max_prompts=config.max_prompts,
            _root_is_resolved=True,
            **provider_options,
        )
        return PluginBinding(static=PluginContribution(tools=(tool.spec(),)))


__all__ = ["ParallelLlmPlugin"]
