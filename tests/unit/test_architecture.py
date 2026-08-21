from __future__ import annotations

import ast
from dataclasses import fields
from pathlib import Path

import thinharness
import thinharness.providers as providers
from thinharness import Harness, HarnessConfig, ModelTurn, OpenAIProvider, OpenAIResponsesModel, PluginBinding


def test_private_plugin_runtime_contracts_stay_narrow() -> None:
    """The public binding excludes agent catalogs and the runtime scope stays private."""
    root = Path(__file__).resolve().parents[2]
    package_source = (root / "thinharness" / "__init__.py").read_text(encoding="utf-8")

    assert [field.name for field in fields(PluginBinding)] == ["static", "connect"]
    assert "_ToolRuntimeScope" not in package_source


def test_core_has_no_bash_imports_or_construction() -> None:
    """Core stays independent from Bash process behavior."""
    root = Path(__file__).resolve().parents[2]
    core_source = (root / "thinharness" / "core.py").read_text(encoding="utf-8")
    tools_init = (root / "thinharness" / "tools" / "__init__.py").read_text(encoding="utf-8")
    plugin_source = (root / "thinharness" / "plugins" / "bash.py").read_text(encoding="utf-8")

    assert "BashPlugin" not in core_source
    assert "plugins.bash" not in core_source
    assert "tools.bash" not in core_source
    assert "create_subprocess" not in core_source
    assert not (root / "thinharness" / "tools" / "bash.py").exists()
    assert "BashTool" not in tools_init
    assert "BashArgs" not in tools_init
    assert "class BashPlugin" in plugin_source
    assert "create_subprocess_exec" in plugin_source
    assert "def for_child" not in plugin_source
    assert "Executor" not in plugin_source


def test_core_has_no_mcp_imports_or_lifecycle_state() -> None:
    """Core stays independent from MCP composition and lifecycle details."""
    core_path = Path(__file__).resolve().parents[2] / "thinharness" / "core.py"
    source = core_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_modules = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module is not None}
    imported_modules.update(alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names)

    assert not any(module.endswith(("tools.mcp", "plugins.mcp")) for module in imported_modules)
    for forbidden in ("MCPServer", "mcp_servers", "_mcp_", "_open_mcp_tools"):
        assert forbidden not in source


def test_core_has_no_subagent_feature_imports_or_identifiers() -> None:
    """Core depends only on the neutral child host and authoritative provenance."""
    root = Path(__file__).resolve().parents[2]
    core_path = root / "thinharness" / "core.py"
    source = core_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    imported_modules.update(
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}

    assert not any(module.endswith(("subagents", "plugins.subagents")) for module in imported_modules)
    forbidden_names = {
        "SubAgentConfig",
        "SubagentsPlugin",
        "SubAgentArgs",
        "DEFAULT_SUBAGENT_NAME",
        "create_subagent_tool",
        "build_child_harness",
        "_select_builtin_tools",
        "builtin_tools",
        "subagent_hooks",
    }
    assert forbidden_names.isdisjoint(names | attributes)
    assert not (root / "thinharness" / "subagents.py").exists()
    assert "PluginContext(root=self.root, model=self.model, child_harnesses=child_harnesses)" in source
    assert "harness=self" not in source.split("PluginContext(", 1)[1].split(")", 1)[0]


def test_subagents_plugin_owns_configuration_and_delegation_tool() -> None:
    """Feature vocabulary and tool construction stay in the plugin module."""
    root = Path(__file__).resolve().parents[2]
    source = (root / "thinharness" / "plugins" / "subagents.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    classes = {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
    string_values = {node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)}

    assert {"SubagentsPlugin", "SubAgentConfig", "SubAgentArgs"} <= classes
    assert "subagent" in string_values
    assert "register_delegation_tool" in source


def test_child_inheritance_and_delegation_detection_are_structural() -> None:
    """Automatic inheritance and tracing do not use plugin or tool name guesses."""
    root = Path(__file__).resolve().parents[2]
    children_source = (root / "thinharness" / "children.py").read_text(encoding="utf-8")
    execution_source = (root / "thinharness" / "tool_execution.py").read_text(encoding="utf-8")
    execution_tree = ast.parse(execution_source)

    assert "isinstance(plugin, ChildInheritablePlugin)" in children_source
    comparisons = [node for node in ast.walk(execution_tree) if isinstance(node, ast.Compare)]
    compared_strings = {
        comparator.value
        for node in comparisons
        for comparator in node.comparators
        if isinstance(comparator, ast.Constant) and isinstance(comparator.value, str)
    }
    assert "subagent" not in compared_strings
    assert "composition.delegation" in execution_source


def test_core_has_no_skills_or_parallel_llm_implementation_details() -> None:
    """Core stays independent from skills and parallel LLM composition."""
    core_path = Path(__file__).resolve().parents[2] / "thinharness" / "core.py"
    source = core_path.read_text(encoding="utf-8")

    forbidden = (
        "tools.skills",
        "SkillRegistry",
        "skills_dir",
        "selected_skills",
        "_skills_enabled",
        "prompt_summary",
        "tools.parallel_llm",
        "create_parallel_llm_tool",
        "builtin_parallel_llm",
        "parallel_llm_max_prompts",
    )
    for token in forbidden:
        assert token not in source

    migration_source = (core_path.parent / "_migration.py").read_text(encoding="utf-8")
    for field_name in (
        "skills_dir",
        "selected_skills",
        "read_paths",
        "write_paths",
        "builtin_parallel_llm_model",
        "builtin_parallel_llm_temperature",
        "parallel_llm_max_prompts",
    ):
        assert f'"{field_name}"' in migration_source


def test_provider_implementations_have_focused_package_ownership() -> None:
    """Each built-in provider owns its model, session, transport, and wire dialect."""
    root = Path(__file__).resolve().parents[2]
    provider_root = root / "thinharness" / "providers"

    assert not (root / "thinharness" / "providers.py").exists()
    assert {path.name for path in provider_root.glob("*.py")} == {
        "__init__.py",
        "anthropic.py",
        "base.py",
        "openai.py",
        "openrouter.py",
        "transcript.py",
        "transport.py",
    }
    assert providers.OpenAIProvider.__module__ == "thinharness.providers.openai"
    assert providers.OpenAIResponsesModel.__module__ == "thinharness.providers.openai"
    assert providers.OpenAIResponsesSession.__module__ == "thinharness.providers.openai"
    assert providers.AnthropicProvider.__module__ == "thinharness.providers.anthropic"
    assert providers.AnthropicMessagesModel.__module__ == "thinharness.providers.anthropic"
    assert providers.AnthropicMessagesSession.__module__ == "thinharness.providers.anthropic"
    assert providers.OpenRouterProvider.__module__ == "thinharness.providers.openrouter"
    assert providers.OpenRouterModel.__module__ == "thinharness.providers.openrouter"
    assert providers.OpenRouterSession.__module__ == "thinharness.providers.openrouter"

    for shared_name in ("base.py", "transport.py", "transcript.py"):
        source = (provider_root / shared_name).read_text(encoding="utf-8")
        assert "from .openai" not in source
        assert "from .anthropic" not in source
        assert "from .openrouter" not in source


def test_provider_public_imports_resolve_from_package_and_top_level() -> None:
    """The package exports its API explicitly and keeps all top-level provider imports."""
    assert all(getattr(providers, name) is not None for name in providers.__all__)
    private_owner_helpers = {
        "_anthropic_thinking_on_by_default",
        "_is_retryable_status",
        "_openai_supports_encrypted_reasoning",
        "_retry_after_seconds",
        "_retry_delay",
        "_validate_retry_settings",
    }
    assert private_owner_helpers.isdisjoint(vars(providers))
    top_level_provider_names = {
        "AnthropicMessagesModel",
        "AnthropicProvider",
        "Model",
        "ModelCapabilities",
        "ModelNotice",
        "ModelSession",
        "ModelSettings",
        "ModelToolCall",
        "ModelTurn",
        "OpenAIProvider",
        "OpenAIResponsesModel",
        "OpenRouterModel",
        "OpenRouterProvider",
        "Provider",
        "RequestConstants",
        "StructuredOutputRequest",
        "TokenUsage",
        "ToolOutput",
        "infer_model",
        "parse_model_ref",
    }
    assert top_level_provider_names <= set(thinharness.__all__)
    for name in top_level_provider_names:
        assert getattr(thinharness, name) is getattr(providers, name)


async def test_external_model_and_custom_builtin_transport_are_injectable(tmp_path: Path) -> None:
    """Applications can inject custom model and transport implementations without source edits."""
    class ExternalSession:
        async def start(self, prompt, constants, *, previous_response_id=None, notices=None):
            return ModelTurn(text="external model")

        async def continue_with_tools(self, outputs, constants, *, notices=None):
            raise AssertionError("external model must finish on its first turn")

        async def continue_with_user_content(self, content, constants, *, notices=None):
            raise AssertionError("external model must finish on its first turn")

        def dump_state(self):
            return None

    class ExternalModel:
        model = "external-model"
        api_key = None
        provider = type("ExternalProvider", (), {"name": "External"})()

        def new_session(self):
            return ExternalSession()

    external_result = await Harness(HarnessConfig(root=tmp_path), model=ExternalModel()).run("hello")
    assert external_result.text == "external model"

    class CustomOpenAITransport(OpenAIProvider):
        async def create_response(self, payload):
            assert payload["model"] == "custom-transport-model"
            return {"id": "resp_custom", "output_text": "custom transport"}

    transport = CustomOpenAITransport(api_key="test-key")
    model = OpenAIResponsesModel("custom-transport-model", provider=transport)
    transport_result = await Harness(HarnessConfig(root=tmp_path), model=model).run("hello")

    assert model.provider is transport
    assert transport_result.text == "custom transport"
