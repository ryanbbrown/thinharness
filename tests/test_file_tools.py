from __future__ import annotations

import json
import subprocess
from pathlib import Path

from thinharness.tools.filesystem import FileTools


def test_file_tools_read_write_edit_and_list(tmp_path: Path) -> None:
    tools = FileTools(tmp_path)
    assert tools.write({"path": "notes/todo.txt", "content": "one\ntwo\n"}).ok
    read = tools.read({"path": "notes/todo.txt", "offset": 2, "limit": 1})
    assert read.ok
    assert "2\ttwo" in read.content
    edit = tools.edit({"path": "notes/todo.txt", "old_string": "two", "new_string": "TWO"})
    assert edit.ok
    listed = tools.list_files({"path": ".", "recursive": True})
    assert "notes/todo.txt" in listed.content

def test_file_tools_large_reads_require_and_stream_bounded_range(tmp_path: Path) -> None:
    path = tmp_path / "large.txt"
    path.write_text("one\ntwo\nthree\nfour\nfive\n", encoding="utf-8")
    tools = FileTools(tmp_path, max_read_bytes=10)

    unbounded = tools.read({"path": "large.txt"})
    assert not unbounded.ok
    assert "pass offset and limit" in unbounded.content

    bounded = tools.read({"path": "large.txt", "offset": 3, "limit": 2})
    assert bounded.ok
    assert "3\tthree" in bounded.content
    assert "4\tfour" in bounded.content
    assert "one" not in bounded.content
    assert bounded.metadata["total_lines"] is None

def test_file_tools_reject_path_escape(tmp_path: Path) -> None:
    tools = FileTools(tmp_path)
    read = tools.read({"path": "../outside.txt"})
    assert not read.ok
    assert read.metadata["error_type"] == "PathValidationError"
    write = tools.write({"path": "/tmp/outside.txt", "content": "no"})
    assert not write.ok
    assert write.metadata["error_type"] == "PathValidationError"

def test_file_tools_enforce_read_and_write_paths(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "docs").mkdir()
    (tmp_path / "src" / "app.py").write_text("def Target():\n    pass\n", encoding="utf-8")
    (tmp_path / "tests" / "test_app.py").write_text("Target()\n", encoding="utf-8")
    (tmp_path / "docs" / "note.md").write_text("Target\n", encoding="utf-8")
    tools = FileTools(tmp_path, read_paths=["src", "tests"], write_paths=["src"])

    assert tools.read({"path": "src/app.py"}).ok
    blocked_read = tools.read({"path": "docs/note.md"})
    assert not blocked_read.ok
    assert blocked_read.metadata["error_type"] == "PathValidationError"

    assert tools.write({"path": "src/generated.py", "content": "ok\n"}).ok
    blocked_write = tools.write({"path": "tests/generated.py", "content": "no\n"})
    assert not blocked_write.ok
    assert blocked_write.metadata["error_type"] == "PathValidationError"

    search = tools.search({"query": "Target"})
    assert search.ok
    assert "src/app.py" in search.content
    assert "tests/test_app.py" in search.content
    assert "docs/note.md" not in search.content

def test_file_tools_validate_glob_selectors(tmp_path: Path) -> None:
    tools = FileTools(tmp_path)
    for result in [
        tools.search({"query": "x", "path_glob": "../*.py"}),
        tools.list_files({"path": ".", "glob": "../*"}),
        tools.glob({"path": ".", "pattern": "/tmp/*"}),
        tools.jsonl_search({"path_glob": "src/../../*.jsonl"}),
    ]:
        assert not result.ok
        assert result.metadata["error_type"] == "PathValidationError"

def test_search_ranks_and_formats_agent_results(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "src" / "app.py").write_text("def HandleRequest():\n    return 'ok'\n", encoding="utf-8")
    (tmp_path / "tests" / "test_app.py").write_text("from src.app import HandleRequest\n", encoding="utf-8")
    tools = FileTools(tmp_path)
    result = tools.search({"query": "HandleRequest", "max_files": 5})
    assert result.ok
    assert "summary:" in result.content
    assert "best_next_step: read src/app.py around line 1" in result.content
    assert result.content.index("src/app.py") < result.content.index("tests/test_app.py")
    assert "why: definition, source" in result.content
    assert result.metadata["cmd"] == ["rg", "--json", "--", "HandleRequest", "."]

def test_search_no_matches_has_refinement_hint(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("hello\n", encoding="utf-8")
    result = FileTools(tmp_path).search({"query": "MissingThing", "path_glob": "**/*.py"})
    assert result.ok
    assert "No matches found." in result.content
    assert "scope: glob=**/*.py" in result.content

def test_search_reports_ripgrep_errors(tmp_path: Path) -> None:
    result = FileTools(tmp_path).search({"query": "["})
    assert not result.ok
    assert "ripgrep failed" in result.content
    assert result.metadata["returncode"] not in (0, 1)

def test_search_line_preview_limit_is_search_only(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("target = '" + ("x" * 40) + "'\n", encoding="utf-8")
    result = FileTools(tmp_path, max_search_line_chars=12).search({"query": "target"})
    assert result.ok
    assert "target = 'xx..." in result.content

def test_search_excludes_and_priority_are_configurable(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "vendor").mkdir()
    (tmp_path / "custom_low").mkdir()
    (tmp_path / "src" / "app.py").write_text("def Target():\n    pass\n", encoding="utf-8")
    (tmp_path / "vendor" / "lib.py").write_text("def Target():\n    pass\n", encoding="utf-8")
    (tmp_path / "custom_low" / "lib.py").write_text("def Target():\n    pass\n", encoding="utf-8")

    excluded = FileTools(tmp_path, search_exclude_globs=["vendor/**"]).search({"query": "Target"})
    assert excluded.ok
    assert "vendor/lib.py" not in excluded.content
    assert excluded.metadata["cmd"][:4] == ["rg", "--json", "--glob", "!vendor/**"]

    ranked = FileTools(tmp_path, search_low_priority_dirs=["custom_low"]).search({"query": "Target"})
    assert ranked.ok
    assert "custom_low/lib.py\n  why: definition, low-priority" in ranked.content
    assert "vendor/lib.py\n  why: definition, source" in ranked.content

def test_search_timeout_returns_structured_result(tmp_path: Path, monkeypatch) -> None:
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(kwargs.get("args", "rg"), timeout=1)

    monkeypatch.setattr("subprocess.run", timeout)
    result = FileTools(tmp_path).search({"query": "Target", "timeout": 1})

    assert not result.ok
    assert result.content == "ripgrep timed out after 1s"
    assert result.metadata["timeout"] == 1

def test_jsonl_search_filters_projects_and_formats(tmp_path: Path) -> None:
    rows = [
        {"id": 1, "user": {"name": "alice", "tags": ["admin", "ops"]}, "msg": "login ok"},
        {"id": 2, "user": {"name": "bob", "tags": ["user"]}, "msg": "login ok"},
        {"id": 3, "user": {"name": "carol", "tags": ["admin"]}, "msg": "login fail"},
    ]
    data = tmp_path / "events.jsonl"
    data.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    result = FileTools(tmp_path).jsonl_search({
        "path_glob": "*.jsonl",
        "fields": {"user.name": 0, "msg": 4},
        "where": [
            {"field": "user.tags[0]", "op": "eq", "value": "admin"},
            {"field": 'user["name"]', "op": "regex", "value": "^[ac]"},
        ],
    })
    assert result.ok, result.content
    assert "rows_matched: 2" in result.content
    assert 'events.jsonl:1: {"user.name": "alice", "msg": "logi…"}' in result.content
    assert 'events.jsonl:3: {"user.name": "carol", "msg": "logi…"}' in result.content
    assert "bob" not in result.content

def test_jsonl_search_uses_ripgrep_prefilter(tmp_path: Path) -> None:
    data = tmp_path / "events.jsonl"
    data.write_text(
        '{"id":1,"msg":"login ok"}\n{"id":2,"msg":"logout ok"}\n{"id":3,"msg":"login fail"}\n',
        encoding="utf-8",
    )
    result = FileTools(tmp_path).jsonl_search({
        "query": "login",
        "path_glob": "*.jsonl",
        "where": [{"field": "msg", "op": "contains", "value": "fail"}],
        "fields": {"id": 0},
    })
    assert result.ok
    assert "rows_matched: 1" in result.content
    assert 'events.jsonl:3: {"id": 3}' in result.content

def test_jsonl_search_reports_ripgrep_errors(tmp_path: Path) -> None:
    result = FileTools(tmp_path).jsonl_search({"query": "[", "path_glob": "*.jsonl"})
    assert not result.ok
    assert "ripgrep failed" in result.content

def test_jsonl_search_limits_display_without_losing_counts(tmp_path: Path) -> None:
    data = tmp_path / "events.jsonl"
    data.write_text(
        "\n".join(json.dumps({"id": i, "msg": "hit"}) for i in range(1, 5)) + "\n",
        encoding="utf-8",
    )

    result = FileTools(tmp_path).jsonl_search({"path_glob": "*.jsonl", "max_matches_per_file": 2})

    assert result.ok
    assert "rows_matched: 4" in result.content
    assert 'events.jsonl:1: {"id": 1, "msg": "hit"}' in result.content
    assert 'events.jsonl:2: {"id": 2, "msg": "hit"}' in result.content
    assert "events.jsonl:3" not in result.content
    assert "... 2 more row(s) in events.jsonl" in result.content

def test_jsonl_search_timeout_returns_structured_result(tmp_path: Path, monkeypatch) -> None:
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(kwargs.get("args", "rg"), timeout=1)

    monkeypatch.setattr("subprocess.run", timeout)
    result = FileTools(tmp_path).jsonl_search({"query": "hit", "path_glob": "*.jsonl", "timeout": 1})

    assert not result.ok
    assert result.content == "ripgrep timed out after 1s"
    assert result.metadata["timeout"] == 1
