from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from thinharness.tools.filesystem import FileTools, PathValidationError, SearchArgs
from thinharness.tools.jsonl import JsonlSearchArgs


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

def test_search_groups_document_results_by_path_and_line(tmp_path: Path) -> None:
    (tmp_path / "claims").mkdir()
    (tmp_path / "policies").mkdir()
    (tmp_path / "policies" / "refunds.md").write_text(
        "Refunds are available within 30 days.\nNo receipt required.\nRefund exceptions require manager approval.\n",
        encoding="utf-8",
    )
    (tmp_path / "claims" / "customer_1024.txt").write_text(
        "Customer requests a refund for a damaged item.\nPrevious refund was denied due to missing receipt.\n",
        encoding="utf-8",
    )
    tools = FileTools(tmp_path)
    result = tools.search({"query": "Refund|refund", "max_files": 5})
    assert result.ok
    assert "summary:" in result.content
    assert "scope: all readable files" in result.content
    assert "files: 2 total, 2 shown" in result.content
    assert "matches: 4 shown, 0 omitted" in result.content
    assert result.content.index("claims/customer_1024.txt") < result.content.index("policies/refunds.md")
    assert "  1: Customer requests a refund for a damaged item." in result.content
    assert "  3: Refund exceptions require manager approval." in result.content
    assert "why:" not in result.content
    assert "buckets:" not in result.content
    assert "definition_candidates:" not in result.content
    assert "best_next_step:" not in result.content
    assert result.metadata["cmd"] == ["rg", "--json", "--", "Refund|refund", "."]

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

def test_search_excludes_are_configurable(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "vendor").mkdir()
    (tmp_path / "src" / "app.py").write_text("def Target():\n    pass\n", encoding="utf-8")
    (tmp_path / "vendor" / "lib.py").write_text("def Target():\n    pass\n", encoding="utf-8")

    excluded = FileTools(tmp_path, search_exclude_globs=["vendor/**"]).search({"query": "Target"})
    assert excluded.ok
    assert "vendor/lib.py" not in excluded.content
    assert excluded.metadata["cmd"][:4] == ["rg", "--json", "--glob", "!vendor/**"]

def test_search_reports_omitted_matches_and_files(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("hit one\nhit two\nhit three\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("hit four\n", encoding="utf-8")

    result = FileTools(tmp_path).search({"query": "hit", "max_files": 1, "max_matches_per_file": 2})

    assert result.ok
    assert "matches: 2 shown, 2 omitted" in result.content
    assert "a.txt\n  1: hit one\n  2: hit two\n  ... 1 more match(es)" in result.content
    assert "note: 1 more file(s) omitted" in result.content

def test_search_timeout_returns_structured_result(tmp_path: Path, monkeypatch) -> None:
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(kwargs.get("args", "rg"), timeout=1)

    monkeypatch.setattr("subprocess.run", timeout)
    result = FileTools(tmp_path).search({"query": "Target", "timeout": 1})

    assert not result.ok
    assert result.content == "ripgrep timed out after 1s"
    assert result.metadata["timeout"] == 1

def test_search_treats_ripgrep_rc2_as_partial_success_with_matches(tmp_path: Path, monkeypatch) -> None:
    def partial(*args, **kwargs):
        stdout = "\n".join([
            _rg_match("docs/refunds.md", 4, "Refund request is pending."),
            "rg: ./restricted: Permission denied",
        ])
        return subprocess.CompletedProcess(args[0], 2, stdout=stdout)

    monkeypatch.setattr("subprocess.run", partial)
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "refunds.md").write_text("Refund request is pending.\n", encoding="utf-8")
    result = FileTools(tmp_path).search({"query": "Refund"})

    assert result.ok
    assert "docs/refunds.md" in result.content
    assert result.metadata["returncode"] == 2
    assert result.metadata["warning"] == "ripgrep returned 2; showing parsed partial matches"
    assert "Permission denied" in result.metadata["warning_excerpt"]

def test_search_sorts_ripgrep_matches_by_line_number(tmp_path: Path, monkeypatch) -> None:
    def out_of_order(*args, **kwargs):
        stdout = "\n".join([
            _rg_match("notes.txt", 5, "hit five"),
            _rg_match("notes.txt", 2, "hit two"),
            _rg_match("notes.txt", 8, "hit eight"),
        ])
        return subprocess.CompletedProcess(args[0], 0, stdout=stdout)

    monkeypatch.setattr("subprocess.run", out_of_order)
    (tmp_path / "notes.txt").write_text("hit\n", encoding="utf-8")
    result = FileTools(tmp_path).search({"query": "hit"})

    assert result.ok
    assert result.content.index("  2: hit two") < result.content.index("  5: hit five")
    assert result.content.index("  5: hit five") < result.content.index("  8: hit eight")

def test_search_keeps_ripgrep_rc2_fatal_without_matches(tmp_path: Path, monkeypatch) -> None:
    def failed(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 2, stdout="rg: regex parse error")

    monkeypatch.setattr("subprocess.run", failed)
    result = FileTools(tmp_path).search({"query": "["})

    assert not result.ok
    assert "ripgrep failed (rc=2)" in result.content

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
    assert "events.jsonl" in result.content
    assert '  1: {"user.name": "alice", "msg": "logi…"}' in result.content
    assert '  3: {"user.name": "carol", "msg": "logi…"}' in result.content
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
    assert '  3: {"id": 3}' in result.content

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
    assert '  1: {"id": 1, "msg": "hit"}' in result.content
    assert '  2: {"id": 2, "msg": "hit"}' in result.content
    assert "  3:" not in result.content
    assert "... 2 more row(s)" in result.content

def test_jsonl_search_timeout_returns_structured_result(tmp_path: Path, monkeypatch) -> None:
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(kwargs.get("args", "rg"), timeout=1)

    monkeypatch.setattr("subprocess.run", timeout)
    result = FileTools(tmp_path).jsonl_search({"query": "hit", "path_glob": "*.jsonl", "timeout": 1})

    assert not result.ok
    assert result.content == "ripgrep timed out after 1s"
    assert result.metadata["timeout"] == 1

def test_jsonl_search_preserves_partial_ripgrep_warning_metadata(tmp_path: Path, monkeypatch) -> None:
    def partial(*args, **kwargs):
        stdout = "\n".join([
            _rg_match("events.jsonl", 1, '{"id":1,"msg":"login ok","secret":"hidden"}'),
            "rg: ./restricted: Permission denied",
        ])
        return subprocess.CompletedProcess(args[0], 2, stdout=stdout)

    monkeypatch.setattr("subprocess.run", partial)
    (tmp_path / "events.jsonl").write_text('{"id":1,"msg":"login ok","secret":"hidden"}\n', encoding="utf-8")
    result = FileTools(tmp_path).jsonl_search({"query": "login", "path_glob": "*.jsonl", "fields": {"id": 0}})

    assert result.ok
    assert '  1: {"id": 1}' in result.content
    assert "secret" not in result.content
    assert result.metadata["returncode"] == 2
    assert result.metadata["warning"] == "ripgrep returned 2; showing parsed partial matches"
    assert "secret" not in json.dumps(result.metadata)

def test_spill_output_uses_thinharness_directory_and_read_guidance(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("\n".join(f"hit {i}" for i in range(100)), encoding="utf-8")
    tools = FileTools(tmp_path, max_tool_chars=120)

    result = tools.search({"query": "hit"})

    assert result.ok
    assert result.metadata["truncated"] is True
    assert result.metadata["saved_to_display"].startswith(".thinharness/outputs/search-")
    assert Path(result.metadata["saved_to"]).exists()
    assert "Read the saved output with read(path=\".thinharness/outputs/search-" in result.content
    assert "offset=1, limit=400" in result.content

def test_spilled_output_can_be_read_with_restricted_read_paths(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "notes.txt").write_text("\n".join(f"hit {i}" for i in range(100)), encoding="utf-8")
    tools = FileTools(tmp_path, read_paths=["docs"], max_tool_chars=120)

    result = tools.search({"query": "hit"})
    read = tools.read({"path": result.metadata["saved_to_display"], "offset": 1, "limit": 8})

    assert read.ok
    assert "notes.txt" in read.content

def test_spilled_output_can_be_read_from_custom_output_dir_with_restricted_read_paths(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "notes.txt").write_text("\n".join(f"hit {i}" for i in range(100)), encoding="utf-8")
    tools = FileTools(tmp_path, output_dir="artifacts/tool-output", read_paths=["docs"], max_tool_chars=120)

    result = tools.search({"query": "hit"})
    read = tools.read({"path": result.metadata["saved_to_display"], "offset": 1, "limit": 8})

    assert result.metadata["saved_to_display"].startswith("artifacts/tool-output/search-")
    assert read.ok
    assert "notes.txt" in read.content

def test_output_dir_must_stay_under_workspace_root(tmp_path: Path) -> None:
    with pytest.raises(PathValidationError):
        FileTools(tmp_path, output_dir="../outside")

def test_restricted_read_paths_do_not_allow_arbitrary_output_dir_files(tmp_path: Path) -> None:
    output_dir = tmp_path / ".thinharness" / "outputs"
    output_dir.mkdir(parents=True)
    (output_dir / "manual.txt").write_text("not generated\n", encoding="utf-8")
    (tmp_path / "docs").mkdir()
    tools = FileTools(tmp_path, read_paths=["docs"])

    result = tools.read({"path": ".thinharness/outputs/manual.txt"})

    assert not result.ok
    assert result.metadata["error_type"] == "PathValidationError"

def test_spill_read_access_rejects_root_and_symlink_escapes(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside.txt"
    outside.write_text("outside\n", encoding="utf-8")
    try:
        output_dir = tmp_path / ".thinharness" / "outputs"
        output_dir.mkdir(parents=True)
        (output_dir / "escape.txt").symlink_to(outside)
        tools = FileTools(tmp_path, read_paths=["docs"])

        root_escape = tools.read({"path": "../outside.txt"})
        symlink_escape = tools.read({"path": ".thinharness/outputs/escape.txt"})

        assert not root_escape.ok
        assert root_escape.metadata["error_type"] == "PathValidationError"
        assert not symlink_escape.ok
        assert symlink_escape.metadata["error_type"] == "PathValidationError"
    finally:
        outside.unlink(missing_ok=True)

def test_missing_search_roots_return_successful_no_match_metadata(tmp_path: Path) -> None:
    tools = FileTools(tmp_path, read_paths=["missing"])

    search = tools.search({"query": "hit"})
    jsonl = tools.jsonl_search({"query": "hit"})

    assert search.ok
    assert search.metadata["returncode"] == 1
    assert jsonl.ok
    assert jsonl.metadata["returncode"] == 1

def test_raised_search_display_defaults() -> None:
    assert SearchArgs(query="hit").max_files == 50
    assert SearchArgs(query="hit").max_matches_per_file == 10
    assert JsonlSearchArgs().max_files == 100
    assert JsonlSearchArgs().max_matches_per_file == 25

def test_raised_search_display_defaults_show_more_than_old_limits(tmp_path: Path) -> None:
    for index in range(11):
        (tmp_path / f"doc_{index:02}.txt").write_text("hit\n", encoding="utf-8")
    (tmp_path / "events.jsonl").write_text(
        "\n".join(json.dumps({"id": index, "msg": "hit"}) for index in range(4)) + "\n",
        encoding="utf-8",
    )
    tools = FileTools(tmp_path)

    search = tools.search({"query": "hit", "path_glob": "*.txt"})
    jsonl = tools.jsonl_search({"path_glob": "*.jsonl"})

    assert "files: 11 total, 11 shown" in search.content
    assert "matches: 11 shown, 0 omitted" in search.content
    assert "rows_matched: 4" in jsonl.content
    assert '  4: {"id": 3, "msg": "hit"}' in jsonl.content

def test_gitignore_ignores_thinharness_outputs() -> None:
    ignore = Path(".gitignore").read_text(encoding="utf-8")

    assert ".thinharness/" in ignore

def _rg_match(path: str, line_number: int, line_text: str) -> str:
    return json.dumps({
        "type": "match",
        "data": {
            "path": {"text": path},
            "line_number": line_number,
            "lines": {"text": f"{line_text}\n"},
        },
    })
