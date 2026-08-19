"""Built-in plugin contracts and adapters."""

from .base import Plugin, PluginBinding, PluginConnector, PluginContext, PluginContribution, ToolOrigin
from .filesystem import FilesystemPlugin
from .mcp import MCPPlugin
from .parallel_llm import ParallelLlmPlugin
from .skills import SkillsPlugin

__all__ = [
    "FilesystemPlugin",
    "MCPPlugin",
    "ParallelLlmPlugin",
    "SkillsPlugin",
    "Plugin",
    "PluginBinding",
    "PluginConnector",
    "PluginContext",
    "PluginContribution",
    "ToolOrigin",
]
