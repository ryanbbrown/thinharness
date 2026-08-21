"""Built-in plugin contracts and adapters."""

from ..children import ChildHarnessHost, ChildHarnessOutcome, ChildHarnessRequest
from .base import ChildInheritablePlugin, Plugin, PluginBinding, PluginConnector, PluginContext, PluginContribution, ToolOrigin
from .bash import BashPlugin
from .filesystem import FilesystemPlugin
from .mcp import MCPPlugin
from .parallel_llm import ParallelLlmPlugin
from .skills import SkillsPlugin
from .subagents import DEFAULT_SUBAGENT_NAME, SubAgentArgs, SubAgentConfig, SubagentsPlugin

__all__ = [
    "BashPlugin",
    "ChildHarnessHost",
    "ChildHarnessOutcome",
    "ChildHarnessRequest",
    "ChildInheritablePlugin",
    "DEFAULT_SUBAGENT_NAME",
    "FilesystemPlugin",
    "MCPPlugin",
    "ParallelLlmPlugin",
    "SkillsPlugin",
    "SubAgentArgs",
    "SubAgentConfig",
    "SubagentsPlugin",
    "Plugin",
    "PluginBinding",
    "PluginConnector",
    "PluginContext",
    "PluginContribution",
    "ToolOrigin",
]
