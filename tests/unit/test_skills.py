from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from fakes import FakeChildHarnessHost, ScriptedModel

from thinharness import Harness, HarnessConfig, PluginBinding, PluginContext, PluginContribution, SkillRegistry, SkillsPlugin, ToolSpec


def test_skill_registry_reads_and_runs_skill(tmp_path: Path) -> None:
    skill = tmp_path / "skills" / "demo"
    (skill / "scripts").mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: demo\ndescription: Demo skill\n---\nBody", encoding="utf-8")
    script = skill / "scripts" / "echo.py"
    script.write_text("import sys\nprint('hi', *sys.argv[1:])\n", encoding="utf-8")
    registry = SkillRegistry(tmp_path / "skills")
    assert "demo - Demo skill" in registry.prompt_summary()
    read = registry.skill_read({"skill_name": "demo"})
    assert read.ok
    assert "SKILL.md" in read.content
    run = registry.skill_run({"skill_name": "demo", "script": "scripts/echo.py", "args": ["there"]})
    assert run.ok
    assert "hi there" in run.content
    assert run.metadata["cmd"][:2] == ["uv", "run"]


def test_skill_run_runs_shell_cli_without_executable_bit(tmp_path: Path) -> None:
    skill = tmp_path / "skills" / "demo"
    (skill / "scripts").mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: demo\n---\nBody", encoding="utf-8")
    (skill / "scripts" / "tool.sh").write_text('printf "shell:%s:%s\\n" "$1" "$2"\n', encoding="utf-8")
    registry = SkillRegistry(tmp_path / "skills")

    run = registry.skill_run({"skill_name": "demo", "script": "scripts/tool.sh", "args": ["convert", "--strict"]})

    assert run.ok
    assert "shell:convert:--strict" in run.content
    assert run.metadata["cmd"][:2] == ["bash", str(skill / "scripts" / "tool.sh")]


@pytest.mark.parametrize(
    ("script_name", "expected_prefix"),
    [
        ("tool.js", ["node"]),
        ("tool.mjs", ["node"]),
        ("tool.go", ["go", "run"]),
        ("tool", []),
    ],
)
def test_skill_run_selects_script_runner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, script_name: str, expected_prefix: list[str]) -> None:
    skill = tmp_path / "skills" / "demo"
    (skill / "scripts").mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: demo\n---\nBody", encoding="utf-8")
    script = skill / "scripts" / script_name
    script.write_text("placeholder\n", encoding="utf-8")
    registry = SkillRegistry(tmp_path / "skills")
    captured: dict[str, list[str]] = {}

    def run(command, **kwargs):
        captured["command"] = command
        return subprocess.CompletedProcess(command, 0, stdout="ok\n")

    monkeypatch.setattr("subprocess.run", run)

    result = registry.skill_run({"skill_name": "demo", "script": f"scripts/{script_name}", "args": ["subcommand", "--flag"]})

    assert result.ok
    assert captured["command"] == [*expected_prefix, str(script), "subcommand", "--flag"]


def test_skill_run_timeout_returns_structured_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    skill = tmp_path / "skills" / "demo"
    (skill / "scripts").mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: demo\n---\nBody", encoding="utf-8")
    (skill / "scripts" / "slow.py").write_text("print('slow')\n", encoding="utf-8")
    registry = SkillRegistry(tmp_path / "skills")

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(kwargs.get("args", "python"), timeout=1)

    monkeypatch.setattr("subprocess.run", timeout)
    result = registry.skill_run({"skill_name": "demo", "script": "scripts/slow.py", "timeout": 1})

    assert not result.ok
    assert result.content == "skill script timed out after 1s"
    assert result.metadata["timeout"] == 1


def test_skill_registry_aggregates_dirs_and_filters_selected_skills(tmp_path: Path) -> None:
    alpha = tmp_path / "a" / "alpha"
    beta = tmp_path / "b" / "beta"
    alpha.mkdir(parents=True)
    beta.mkdir(parents=True)
    (alpha / "SKILL.md").write_text("---\nname: alpha\ndescription: Alpha skill\n---\nAlpha", encoding="utf-8")
    (beta / "SKILL.md").write_text("---\nname: beta\ndescription: Beta skill\n---\nBeta", encoding="utf-8")

    registry = SkillRegistry([tmp_path / "a", tmp_path / "b"], selected_skills=["beta"])

    assert list(registry.skills) == ["beta"]
    assert "beta - Beta skill" in registry.prompt_summary()
    assert "alpha - Alpha skill" not in registry.prompt_summary()

def test_skill_registry_rejects_duplicate_skill_names(tmp_path: Path) -> None:
    first = tmp_path / "first" / "demo"
    second = tmp_path / "second" / "demo"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    (first / "SKILL.md").write_text("---\nname: demo\n---\nFirst", encoding="utf-8")
    (second / "SKILL.md").write_text("---\nname: demo\n---\nSecond", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate skill name: demo"):
        SkillRegistry([tmp_path / "first", tmp_path / "second"])


def _write_skill(root: Path, name: str, body: str = "Body", *, description: str = "Demo skill") -> Path:
    skill = root / name
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n{body}",
        encoding="utf-8",
    )
    return skill


@pytest.mark.parametrize(
    "kwargs",
    [
        {"skills_dir": set()},
        {"skills_dir": []},
        {"skills_dir": ["skills"], "tools": set()},
        {"skills_dir": ["skills"], "tools": []},
        {"skills_dir": ["skills"], "tools": ["skill_read", "skill_read"]},
        {"skills_dir": ["skills"], "tools": ["missing"]},
    ],
)
def test_skills_plugin_rejects_invalid_ordered_inputs(kwargs) -> None:
    kwargs.setdefault("tools", ["skill_read"])
    with pytest.raises((TypeError, ValueError)):
        SkillsPlugin(**kwargs)


def test_skills_plugin_discovers_and_selects_at_construction(tmp_path: Path) -> None:
    _write_skill(tmp_path / "skills", "alpha", description="Alpha")
    _write_skill(tmp_path / "skills" / "nested", "beta", description="Beta")

    plugin = SkillsPlugin(tmp_path / "skills", selected_skills=["beta"], tools=["skill_run", "skill_read"])
    harness = Harness(HarnessConfig(root=tmp_path / "workspace"), model=ScriptedModel([]), plugins=[plugin])

    assert list(plugin.registry.skills) == ["beta"]
    assert [tool.name for tool in harness.tools] == ["skill_run", "skill_read"]
    assert [tool.origin.plugin for tool in harness.tools if tool.origin] == ["skills", "skills"]
    assert "beta - Beta" in harness.system_instructions()
    assert "alpha - Alpha" not in harness.system_instructions()
    assert harness.tools[0].sequential is True
    assert harness.tools[1].sequential is False



def test_skills_plugin_empty_catalog_has_no_contribution_and_missing_selection_fails(tmp_path: Path) -> None:
    plugin = SkillsPlugin(tmp_path / "missing", tools=["skill_read"])
    harness = Harness(HarnessConfig(root=tmp_path), model=ScriptedModel([]), plugins=[plugin])

    assert harness.tools == []
    assert "Available skills" not in harness.system_instructions()
    with pytest.raises(ValueError, match="unknown selected skill"):
        SkillsPlugin(tmp_path / "missing", selected_skills=["absent"], tools=["skill_read"])



def test_skills_plugin_catalog_is_frozen_but_discovered_content_and_scripts_are_live(tmp_path: Path) -> None:
    skill = _write_skill(tmp_path / "skills", "demo", "Old body")
    scripts = skill / "scripts"
    scripts.mkdir()
    script = scripts / "live.sh"
    script.write_text("printf old\n", encoding="utf-8")
    plugin = SkillsPlugin(tmp_path / "skills", tools=["skill_read", "skill_run"])

    (skill / "SKILL.md").write_text("---\nname: demo\ndescription: Changed metadata\n---\nNew body", encoding="utf-8")
    script.write_text("printf new\n", encoding="utf-8")
    _write_skill(tmp_path / "skills", "added")
    harness = Harness(HarnessConfig(root=tmp_path), model=ScriptedModel([]), plugins=[plugin])
    by_name = {tool.name: tool for tool in harness.tools}

    read_result = by_name["skill_read"].handler(by_name["skill_read"].parse_args({"skill_name": "demo"}))
    run_result = by_name["skill_run"].handler(
        by_name["skill_run"].parse_args({"skill_name": "demo", "script": "scripts/live.sh"})
    )
    assert "New body" in read_result.content
    assert "new" in run_result.content
    assert "added" not in plugin.registry.skills
    assert "Changed metadata" not in harness.system_instructions()
    assert "added" in SkillsPlugin(tmp_path / "skills", tools=["skill_read"]).registry.skills



def test_skills_plugin_relative_paths_use_cwd_and_reuse_one_catalog(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    process_dir = tmp_path / "process"
    _write_skill(process_dir / "skills", "demo")
    process_dir.mkdir(exist_ok=True)
    monkeypatch.chdir(process_dir)
    plugin = SkillsPlugin("skills", tools=["skill_read"])

    first = Harness(HarnessConfig(root=tmp_path / "one"), model=ScriptedModel([]), plugins=[plugin])
    second = Harness(HarnessConfig(root=tmp_path / "two"), model=ScriptedModel([]), plugins=[plugin])

    assert plugin.registry.skills["demo"].root == process_dir / "skills" / "demo"
    assert first.plugins[0] is second.plugins[0] is plugin
    assert first.tools[0].handler.__self__ is second.tools[0].handler.__self__ is plugin.registry



def test_skills_plugin_summary_wording_and_plugin_order(tmp_path: Path) -> None:
    _write_skill(tmp_path / "skills", "demo")

    class InstructionPlugin:
        def __init__(self, name: str, instruction: str) -> None:
            self.name = name
            self.instruction = instruction

        def bind(self, context: PluginContext) -> PluginBinding:
            return PluginBinding(static=PluginContribution(instructions=(self.instruction,)))

    harness = Harness(
        HarnessConfig(root=tmp_path, system_prompt="base"),
        model=ScriptedModel([]),
        plugins=[
            InstructionPlugin("before", "before marker"),
            SkillsPlugin(tmp_path / "skills", tools=["skill_run"]),
            InstructionPlugin("after", "after marker"),
        ],
        tools=[ToolSpec("direct", "direct", {"type": "object", "properties": {}}, lambda _args: "ok", instructions="tool marker")],
    )
    instructions = harness.system_instructions()

    assert "call skill_read" not in instructions
    assert instructions.index("before marker") < instructions.index("Available skills:") < instructions.index("after marker")
    assert instructions.index("after marker") < instructions.index("tool marker")
    assert instructions.count("Available skills:") == 1



def test_skills_plugin_name_is_fixed_and_collisions_are_atomic(tmp_path: Path) -> None:
    _write_skill(tmp_path / "skills", "demo")
    plugin = SkillsPlugin(tmp_path / "skills", tools=["skill_read"])

    with pytest.raises(AttributeError, match="fixed"):
        plugin.name = "other"
    with pytest.raises(AttributeError, match="fixed"):
        SkillsPlugin.name = "other"
    with pytest.raises(TypeError, match="cannot override"):
        class RenamedSkillsPlugin(SkillsPlugin):
            name = "other"

    duplicate = ToolSpec("skill_read", "duplicate", {"type": "object", "properties": {}}, lambda _args: "ok")
    with pytest.raises(ValueError, match="duplicate tool name"):
        Harness(HarnessConfig(root=tmp_path), model=ScriptedModel([]), plugins=[plugin], tools=[duplicate])
    with pytest.raises(ValueError, match="duplicate plugin name: skills"):
        Harness(
            HarnessConfig(root=tmp_path),
            model=ScriptedModel([]),
            plugins=[plugin, SkillsPlugin(tmp_path / "skills", tools=["skill_run"])],
        )



def test_skills_plugin_bind_is_io_free_after_construction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_skill(tmp_path / "skills", "demo")
    plugin = SkillsPlugin(tmp_path / "skills", tools=["skill_read"])
    model = ScriptedModel([])

    def fail(*_args, **_kwargs):
        raise AssertionError("filesystem metadata used during bind")

    monkeypatch.setattr(Path, "resolve", fail)
    monkeypatch.setattr(Path, "exists", fail)
    monkeypatch.setattr(Path, "stat", fail)

    binding = plugin.bind(PluginContext(root=tmp_path, model=model, child_harnesses=FakeChildHarnessHost()))
    assert binding.static.tools[0].name == "skill_read"
