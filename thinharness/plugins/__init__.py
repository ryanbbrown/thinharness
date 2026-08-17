"""Built-in plugin contracts and adapters."""

from .base import Plugin, PluginBinding, PluginConnector, PluginContext, PluginContribution
from .filesystem import FilesystemPlugin

__all__ = [
    "FilesystemPlugin",
    "Plugin",
    "PluginBinding",
    "PluginConnector",
    "PluginContext",
    "PluginContribution",
]
