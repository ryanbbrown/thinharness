"""Built-in plugin contracts and adapters."""

from .base import Plugin, PluginBinding, PluginConnector, PluginContext, PluginContribution, ToolOrigin
from .filesystem import FilesystemPlugin
from .mcp import MCPPlugin

__all__ = [
    "FilesystemPlugin",
    "MCPPlugin",
    "Plugin",
    "PluginBinding",
    "PluginConnector",
    "PluginContext",
    "PluginContribution",
    "ToolOrigin",
]
