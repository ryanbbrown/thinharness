from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError

from thinharness.defaults import (
    DEFAULT_EDIT_DESCRIPTION,
    DEFAULT_GLOB_DESCRIPTION,
    DEFAULT_JSONL_SEARCH_DESCRIPTION,
    DEFAULT_LIST_DESCRIPTION,
    DEFAULT_READ_DESCRIPTION,
    DEFAULT_SEARCH_DESCRIPTION,
    DEFAULT_WRITE_DESCRIPTION,
)
from thinharness.tools.base import StrictArgs, tool_parameters
from thinharness.tools.filesystem import FileTools, PathValidationError, SearchArgs
from thinharness.tools.jsonl import JsonlSearchArgs


def test_file_tool_descriptions_use_defaults(tmp_path: Path) -> None:
    descriptions = {tool.name: tool.description for tool in FileTools(tmp_path).specs()}

    assert descriptions == {
        "read": DEFAULT_READ_DESCRIPTION,
        "write": DEFAULT_WRITE_DESCRIPTION,
        "edit": DEFAULT_EDIT_DESCRIPTION,
        "search": DEFAULT_SEARCH_DESCRIPTION,
        "list": DEFAULT_LIST_DESCRIPTION,
        "glob": DEFAULT_GLOB_DESCRIPTION,
        "jsonl_search": DEFAULT_JSONL_SEARCH_DESCRIPTION,
    }

def test_tool_parameters_preserves_title_field_while_stripping_schema_titles() -> None:
    class TitledArgs(StrictArgs):
        title: str
        path: str

    schema = tool_parameters(TitledArgs)

    assert "title" in schema["properties"]
    assert schema["properties"]["title"] == {"type": "string"}
    assert "title" in schema["required"]
    assert schema.get("title") is None

def test_file_tools_read_write_edit_and_list(tmp_path: Path) -> None:
    tools = FileTools(tmp_path)
    assert tools.write({"path": "notes/todo.txt", "content": "one\ntwo\n"}).ok
    read = tools.read({"path": "notes/todo.txt", "offset": 2, "limit": 1})
    assert read.ok
    assert "2\ttwo" in read.content
    edit = tools.edit({"edits": [{"path": "notes/todo.txt", "old_string": "two", "new_string": "TWO"}]})
    assert edit.ok
    listed = tools.list_files({"path": ".", "recursive": True})
    assert "notes/todo.txt" in listed.content

def test_file_tools_edit_batches_across_files_without_cross_file_state(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("shared\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("shared\n", encoding="utf-8")
    tools = FileTools(tmp_path)

    result = tools.edit({
        "edits": [
            {"path": "a.txt", "old_string": "shared", "new_string": "alpha"},
            {"path": "b.txt", "old_string": "shared", "new_string": "beta"},
        ],
    })

    assert result.ok, result.content
    assert result.metadata["applied"] == 2
    assert result.metadata["failed"] == 0
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "alpha\n"
    assert (tmp_path / "b.txt").read_text(encoding="utf-8") == "beta\n"

def test_file_tools_edit_applies_same_file_edits_in_order(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("one\n", encoding="utf-8")
    tools = FileTools(tmp_path)

    result = tools.edit({
        "edits": [
            {"path": "notes.txt", "old_string": "one", "new_string": "two"},
            {"path": "notes.txt", "old_string": "two", "new_string": "three"},
        ],
    })

    assert result.ok
    assert (tmp_path / "notes.txt").read_text(encoding="utf-8") == "three\n"

def test_file_tools_edit_reports_partial_failures_and_continues(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("one\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("two\n", encoding="utf-8")
    tools = FileTools(tmp_path)

    result = tools.edit({
        "edits": [
            {"path": "a.txt", "old_string": "one", "new_string": "ONE"},
            {"path": "b.txt", "old_string": "missing", "new_string": "MISSING"},
            {"path": "b.txt", "old_string": "two", "new_string": "TWO"},
        ],
    })

    assert not result.ok
    assert result.metadata["applied"] == 2
    assert result.metadata["failed"] == 1
    assert "1. ok a.txt: replaced 1 occurrence(s)" in result.content
    assert "2. FAILED b.txt: old_string not found" in result.content
    assert "3. ok b.txt: replaced 1 occurrence(s)" in result.content
    assert result.metadata["results"][1]["error"] == "old_string not found"
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "ONE\n"
    assert (tmp_path / "b.txt").read_text(encoding="utf-8") == "TWO\n"

def test_file_tools_edit_reports_when_all_items_fail(tmp_path: Path) -> None:
    tools = FileTools(tmp_path)

    result = tools.edit({
        "edits": [
            {"path": "missing-a.txt", "old_string": "one", "new_string": "ONE"},
            {"path": "missing-b.txt", "old_string": "two", "new_string": "TWO"},
        ],
    })

    assert not result.ok
    assert result.metadata["applied"] == 0
    assert result.metadata["failed"] == 2
    assert "1. FAILED missing-a.txt: file not found" in result.content
    assert "2. FAILED missing-b.txt: file not found" in result.content

def test_file_tools_edit_reports_ambiguity_introduced_by_earlier_edit(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("first\nsecond\n", encoding="utf-8")
    tools = FileTools(tmp_path)

    result = tools.edit({
        "edits": [
            {"path": "notes.txt", "old_string": "first", "new_string": "second"},
            {"path": "notes.txt", "old_string": "second", "new_string": "done"},
        ],
    })

    assert not result.ok
    assert result.metadata["applied"] == 1
    assert result.metadata["results"][1]["matches"] == 2
    assert "old_string appears 2 times" in result.metadata["results"][1]["error"]
    assert (tmp_path / "notes.txt").read_text(encoding="utf-8") == "second\nsecond\n"

def test_file_tools_edit_supports_per_item_all(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("x x y\n", encoding="utf-8")
    tools = FileTools(tmp_path)

    result = tools.edit({
        "edits": [
            {"path": "notes.txt", "old_string": "x", "new_string": "z", "all": True},
        ],
    })

    assert result.ok
    assert result.metadata["results"][0]["replacements"] == 2
    assert (tmp_path / "notes.txt").read_text(encoding="utf-8") == "z z y\n"

def test_file_tools_edit_expected_replacements_is_count_assertion_per_item(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("x x\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("one\n", encoding="utf-8")
    (tmp_path / "c.txt").write_text("q q\n", encoding="utf-8")
    tools = FileTools(tmp_path)

    mismatch = tools.edit({
        "edits": [
            {"path": "a.txt", "old_string": "x", "new_string": "z", "all": True, "expected_replacements": 3},
            {"path": "b.txt", "old_string": "one", "new_string": "two"},
        ],
    })
    assert not mismatch.ok
    assert mismatch.metadata["applied"] == 1
    assert mismatch.metadata["results"][0]["matches"] == 2
    assert "expected 3 replacement(s), found 2" in mismatch.content
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "x x\n"
    assert (tmp_path / "b.txt").read_text(encoding="utf-8") == "two\n"

    ambiguity = tools.edit({
        "edits": [
            {"path": "a.txt", "old_string": "x", "new_string": "z", "expected_replacements": 2},
        ],
    })
    assert not ambiguity.ok
    assert ambiguity.metadata["results"][0]["matches"] == 2
    assert "old_string appears 2 times" in ambiguity.content
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "x x\n"

    matched_all = tools.edit({
        "edits": [
            {"path": "c.txt", "old_string": "q", "new_string": "r", "all": True, "expected_replacements": 2},
        ],
    })
    assert matched_all.ok
    assert matched_all.metadata["results"][0]["replacements"] == 2
    assert (tmp_path / "c.txt").read_text(encoding="utf-8") == "r r\n"

def test_file_tools_edit_path_and_file_errors_are_per_item(tmp_path: Path) -> None:
    (tmp_path / "ok.txt").write_text("one\n", encoding="utf-8")
    (tmp_path / "folder").mkdir()
    tools = FileTools(tmp_path, write_paths=["ok.txt", "missing.txt", "folder"])

    result = tools.edit({
        "edits": [
            {"path": "../blocked.txt", "old_string": "x", "new_string": "y"},
            {"path": "missing.txt", "old_string": "x", "new_string": "y"},
            {"path": "folder", "old_string": "x", "new_string": "y"},
            {"path": "ok.txt", "old_string": "one", "new_string": "two"},
        ],
    })

    assert not result.ok
    assert result.metadata["applied"] == 1
    assert result.metadata["failed"] == 3
    assert result.metadata["results"][0]["error_type"] == "PathValidationError"
    assert result.metadata["results"][1]["error"] == "file not found"
    assert result.metadata["results"][2]["error"] == "path is not a file"
    assert result.metadata["results"][2]["error_type"] == "PathTypeError"
    assert (tmp_path / "ok.txt").read_text(encoding="utf-8") == "two\n"

def test_file_tools_edit_reports_os_errors_per_item_and_continues(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "fail.txt").write_text("one\n", encoding="utf-8")
    (tmp_path / "ok.txt").write_text("two\n", encoding="utf-8")
    original_write_text = Path.write_text

    def write_text(path: Path, data: str, *args: object, **kwargs: object) -> int:
        if path == tmp_path / "fail.txt":
            raise PermissionError("denied")
        return original_write_text(path, data, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", write_text)
    tools = FileTools(tmp_path)

    result = tools.edit({
        "edits": [
            {"path": "fail.txt", "old_string": "one", "new_string": "ONE"},
            {"path": "ok.txt", "old_string": "two", "new_string": "TWO"},
        ],
    })

    assert not result.ok
    assert result.metadata["applied"] == 1
    assert result.metadata["failed"] == 1
    assert result.metadata["results"][0]["error_type"] == "PermissionError"
    assert result.metadata["results"][1]["replacements"] == 1
    assert (tmp_path / "fail.txt").read_text(encoding="utf-8") == "one\n"
    assert (tmp_path / "ok.txt").read_text(encoding="utf-8") == "TWO\n"

def test_file_tools_edit_rejects_empty_old_string_per_item(tmp_path: Path) -> None:
    (tmp_path / "ok.txt").write_text("one\n", encoding="utf-8")
    tools = FileTools(tmp_path)

    result = tools.edit({"edits": [{"path": "ok.txt", "old_string": "", "new_string": "two"}]})

    assert not result.ok
    assert result.metadata["applied"] == 0
    assert result.metadata["results"][0]["path"] == "ok.txt"
    assert result.metadata["results"][0]["error"] == "old_string must not be empty"
    assert (tmp_path / "ok.txt").read_text(encoding="utf-8") == "one\n"

def test_file_tools_edit_rejects_empty_list_and_old_flat_shape(tmp_path: Path) -> None:
    tools = FileTools(tmp_path)

    with pytest.raises(ValidationError):
        tools.edit({"edits": []})
    with pytest.raises(ValidationError):
        tools.edit({"path": "notes.txt", "old_string": "one", "new_string": "two"})

def test_file_tools_edit_provider_schema_inlines_nested_operation(tmp_path: Path) -> None:
    edit_schema = next(tool.response_tool() for tool in FileTools(tmp_path).specs() if tool.name == "edit")
    parameters = edit_schema["parameters"]

    assert not _contains_key(parameters, "$ref")
    assert not _contains_key(parameters, "$defs")
    assert "edits" in parameters["required"]
    assert parameters["properties"]["edits"]["type"] == "array"
    assert parameters["properties"]["edits"]["minItems"] == 1
    operation_schema = parameters["properties"]["edits"]["items"]
    assert operation_schema["additionalProperties"] is False

def test_file_tools_read_without_limit_reads_through_eof_under_hard_caps(tmp_path: Path) -> None:
    (tmp_path / "many.txt").write_text("\n".join(f"line {i}" for i in range(450)), encoding="utf-8")
    tools = FileTools(tmp_path, max_read_chars=20_000)

    result = tools.read({"path": "many.txt"})

    assert result.ok
    assert result.metadata["returned_lines"] == 450
    assert "450\tline 449" in result.content
    assert "more lines available" not in result.content

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
    read_schema = next(tool.response_tool() for tool in tools.specs() if tool.name == "read")
    read_properties = read_schema["parameters"]["properties"]
    assert "limit" not in read_schema["parameters"]["required"]
    assert "default" not in read_properties["limit"]

    search_schema = next(tool.response_tool() for tool in tools.specs() if tool.name == "search")
    search_properties = search_schema["parameters"]["properties"]
    assert "path" in search_properties
    assert "path_glob" not in search_properties
    jsonl_schema = next(tool.response_tool() for tool in tools.specs() if tool.name == "jsonl_search")
    jsonl_properties = jsonl_schema["parameters"]["properties"]
    assert "path" in jsonl_properties
    assert "path_glob" not in jsonl_properties

    for result in [
        tools.search({"query": "x", "path": "../src"}),
        tools.list_files({"path": ".", "glob": "../*"}),
        tools.glob({"path": ".", "pattern": "/tmp/*"}),
        tools.jsonl_search({"path": "src/../../*.jsonl"}),
    ]:
        assert not result.ok
        assert result.metadata["error_type"] == "PathValidationError"

def test_jsonl_search_schema_exposes_range_and_field_search_arguments(tmp_path: Path) -> None:
    jsonl_schema = next(tool.response_tool() for tool in FileTools(tmp_path).specs() if tool.name == "jsonl_search")
    jsonl_schema_text = json.dumps(jsonl_schema)

    for token in ["gt", "gte", "lt", "lte", "number", "date", "field_searches", "context_lines", "max_line_chars"]:
        assert f'"{token}"' in jsonl_schema_text

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
    (tmp_path / "claims").mkdir()
    (tmp_path / "claims" / "a.txt").write_text("hello\n", encoding="utf-8")
    result = FileTools(tmp_path).search({"query": "MissingThing", "path": "claims"})
    assert result.ok
    assert "No matches found." in result.content
    assert "scope: path=claims" in result.content

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
        "path": "*.jsonl",
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
        "path": "*.jsonl",
        "where": [{"field": "msg", "op": "contains", "value": "fail"}],
        "fields": {"id": 0},
    })
    assert result.ok
    assert "rows_matched: 1" in result.content
    assert '  3: {"id": 3}' in result.content

def test_jsonl_search_non_range_where_operators_after_compile_refactor(tmp_path: Path) -> None:
    rows = [
        {"id": 1, "status": "open", "owner": "alice", "priority": "high"},
        {"id": 2, "status": "closed", "owner": None, "priority": "low"},
        {"id": 3, "status": "open", "priority": "medium"},
    ]
    (tmp_path / "events.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    tools = FileTools(tmp_path)

    ne = tools.jsonl_search({
        "path": "*.jsonl",
        "where": [{"field": "status", "op": "ne", "value": "closed"}],
        "fields": {"id": 0},
    })
    in_filter = tools.jsonl_search({
        "path": "*.jsonl",
        "where": [{"field": "priority", "op": "in", "values": ["high", "medium"]}],
        "fields": {"id": 0},
    })
    exists = tools.jsonl_search({
        "path": "*.jsonl",
        "where": [{"field": "owner", "op": "exists"}],
        "fields": {"id": 0},
    })

    assert ne.ok, ne.content
    assert "rows_matched: 2" in ne.content
    assert '  1: {"id": 1}' in ne.content
    assert '  3: {"id": 3}' in ne.content
    assert '"id": 2' not in ne.content
    assert in_filter.ok, in_filter.content
    assert "rows_matched: 2" in in_filter.content
    assert '  1: {"id": 1}' in in_filter.content
    assert '  3: {"id": 3}' in in_filter.content
    assert '"id": 2' not in in_filter.content
    assert exists.ok, exists.content
    assert "rows_matched: 1" in exists.content
    assert '  1: {"id": 1}' in exists.content
    assert '"id": 2' not in exists.content
    assert '"id": 3' not in exists.content

def test_jsonl_search_path_accepts_recursive_directory(tmp_path: Path) -> None:
    nested = tmp_path / "logs" / "nested"
    nested.mkdir(parents=True)
    (nested / "events.jsonl").write_text('{"id":1,"msg":"hit"}\n', encoding="utf-8")
    (tmp_path / "logs" / "notes.txt").write_text('{"id":2,"msg":"hit"}\n', encoding="utf-8")

    result = FileTools(tmp_path).jsonl_search({"path": "logs", "fields": {"id": 0}})

    assert result.ok
    assert "scope: path=logs" in result.content
    assert "logs/nested/events.jsonl" in result.content
    assert '  1: {"id": 1}' in result.content
    assert "notes.txt" not in result.content

def test_jsonl_search_broad_glob_skips_non_jsonl_files(tmp_path: Path) -> None:
    (tmp_path / "events.jsonl").write_text('{"id":1,"msg":"hit"}\n', encoding="utf-8")
    (tmp_path / "notes.txt").write_text('{"id":2,"msg":"hit"}\nnot json\n', encoding="utf-8")

    result = FileTools(tmp_path).jsonl_search({"query": "hit", "path": "*", "fields": {"id": 0}})

    assert result.ok
    assert "events.jsonl" in result.content
    assert '  1: {"id": 1}' in result.content
    assert "notes.txt" not in result.content
    assert "json_parse_errors" not in result.content

def test_jsonl_search_reports_ripgrep_errors(tmp_path: Path) -> None:
    result = FileTools(tmp_path).jsonl_search({"query": "[", "path": "*.jsonl"})
    assert not result.ok
    assert "ripgrep failed" in result.content

def test_jsonl_search_limits_display_without_losing_counts(tmp_path: Path) -> None:
    data = tmp_path / "events.jsonl"
    data.write_text(
        "\n".join(json.dumps({"id": i, "msg": "hit"}) for i in range(1, 5)) + "\n",
        encoding="utf-8",
    )

    result = FileTools(tmp_path).jsonl_search({"path": "*.jsonl", "max_matches_per_file": 2})

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
    result = FileTools(tmp_path).jsonl_search({"query": "hit", "path": "*.jsonl", "timeout": 1})

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
    result = FileTools(tmp_path).jsonl_search({"query": "login", "path": "*.jsonl", "fields": {"id": 0}})

    assert result.ok
    assert '  1: {"id": 1}' in result.content
    assert "secret" not in result.content
    assert result.metadata["returncode"] == 2
    assert result.metadata["warning"] == "ripgrep returned 2; showing parsed partial matches"
    assert "secret" not in json.dumps(result.metadata)

def test_jsonl_search_field_search_returns_matching_internal_lines(tmp_path: Path) -> None:
    (tmp_path / "states.jsonl").write_text(
        json.dumps({
            "state_index": 11,
            "url": "https://example.test",
            "accessibility_tree": "\n".join([
                "[a1] button 'Save'",
                "[a2] menuitem 'Edit personal filters'",
                "[a3] menuitem '-- None --'",
                "[a4] menuitem 'Incident Portal'",
            ]),
        })
        + "\n",
        encoding="utf-8",
    )

    result = FileTools(tmp_path).jsonl_search({
        "path": "states.jsonl",
        "where": [{"field": "state_index", "op": "eq", "value": "11"}],
        "fields": {"state_index": 0, "url": 0},
        "field_searches": [{"field": "accessibility_tree", "query": "Incident|-- None --|Edit personal filters", "regex": True}],
    })

    assert result.ok, result.content
    assert "field_searches: accessibility_tree" in result.content
    assert '  1: {"state_index": 11, "url": "https://example.test"}' in result.content
    assert "    accessibility_tree matches:" in result.content
    assert "      2: [a2] menuitem 'Edit personal filters'" in result.content
    assert "      3: [a3] menuitem '-- None --'" in result.content
    assert "      4: [a4] menuitem 'Incident Portal'" in result.content
    assert "Save" not in result.content

def test_jsonl_search_field_search_without_fields_does_not_render_whole_row(tmp_path: Path) -> None:
    (tmp_path / "states.jsonl").write_text(
        json.dumps({"id": 1, "blob": "alpha\nneedle\nomega", "secret": "do not print"}) + "\n",
        encoding="utf-8",
    )

    result = FileTools(tmp_path).jsonl_search({"path": "*.jsonl", "field_searches": [{"field": "blob", "query": "needle"}]})

    assert result.ok, result.content
    assert '  1: {}' in result.content
    assert "      2: needle" in result.content
    assert "secret" not in result.content
    assert "alpha" not in result.content

def test_jsonl_search_field_search_top_level_query_remains_row_prefilter(tmp_path: Path) -> None:
    (tmp_path / "states.jsonl").write_text(
        "\n".join([
            json.dumps({"id": 1, "kind": "candidate", "blob": "target internal line"}),
            json.dumps({"id": 2, "kind": "other", "blob": "target internal line"}),
        ])
        + "\n",
        encoding="utf-8",
    )

    result = FileTools(tmp_path).jsonl_search({
        "query": "candidate",
        "path": "*.jsonl",
        "fields": {"id": 0},
        "field_searches": [{"field": "blob", "query": "target"}],
    })

    assert result.ok, result.content
    assert "rows_matched: 1" in result.content
    assert '  1: {"id": 1}' in result.content
    assert '"id": 2' not in result.content

def test_jsonl_search_field_search_miss_with_fields_keeps_output_compact(tmp_path: Path) -> None:
    (tmp_path / "states.jsonl").write_text(json.dumps({"id": 1, "blob": "alpha"}) + "\n", encoding="utf-8")

    result = FileTools(tmp_path).jsonl_search({
        "path": "*.jsonl",
        "fields": {"id": 0},
        "field_searches": [{"field": "blob", "query": "needle"}],
    })

    assert result.ok, result.content
    assert '  1: {"id": 1}' in result.content
    assert "blob matches" not in result.content

def test_jsonl_search_field_search_multiple_and_duplicate_searches_render_in_order(tmp_path: Path) -> None:
    (tmp_path / "states.jsonl").write_text(
        json.dumps({"id": 1, "body": "alpha\nbeta\nALPHA", "thought": "first\nfilters\nlast"}) + "\n",
        encoding="utf-8",
    )

    result = FileTools(tmp_path).jsonl_search({
        "path": "*.jsonl",
        "fields": {"id": 0},
        "field_searches": [
            {"field": "body", "query": "alpha"},
            {"field": "thought", "query": "filters"},
            {"field": "body", "query": "beta"},
        ],
    })

    assert result.ok, result.content
    first = result.content.index("    body matches #1 (query='alpha'):")
    second = result.content.index("    thought matches:")
    third = result.content.index("    body matches #2 (query='beta'):")
    assert first < second < third
    assert "      1: alpha" in result.content
    assert "      3: ALPHA" in result.content
    assert "      2: beta" in result.content

def test_jsonl_search_field_search_duplicate_same_query_labels_are_distinguishable(tmp_path: Path) -> None:
    (tmp_path / "states.jsonl").write_text(json.dumps({"id": 1, "body": "alpha\nmiddle\nalpha"}) + "\n", encoding="utf-8")

    result = FileTools(tmp_path).jsonl_search({
        "path": "*.jsonl",
        "fields": {"id": 0},
        "field_searches": [
            {"field": "body", "query": "alpha", "context_lines": 0},
            {"field": "body", "query": "alpha", "context_lines": 1},
        ],
    })

    assert result.ok, result.content
    assert "    body matches #1 (query='alpha'):" in result.content
    assert "    body matches #2 (query='alpha'):" in result.content

def test_jsonl_search_field_search_context_merges_and_limits_matches(tmp_path: Path) -> None:
    (tmp_path / "states.jsonl").write_text(
        json.dumps({"id": 1, "blob": "\n".join(["before", "target one", "middle", "target two", "after", "target three"])}) + "\n",
        encoding="utf-8",
    )

    result = FileTools(tmp_path).jsonl_search({
        "path": "*.jsonl",
        "field_searches": [{"field": "blob", "query": "target", "context_lines": 1, "max_matches": 2}],
    })

    assert result.ok, result.content
    assert "      1: before" in result.content
    assert "      2: target one" in result.content
    assert "      3: middle" in result.content
    assert "      4: target two" in result.content
    assert "      5: after" in result.content
    assert "target three" not in result.content
    assert "      ... 1 more match(es)" in result.content
    assert result.content.count("      3: middle") == 1

def test_jsonl_search_field_search_does_not_count_context_visible_match_as_omitted(tmp_path: Path) -> None:
    (tmp_path / "states.jsonl").write_text(json.dumps({"id": 1, "blob": "target one\ntarget two\nlast"}) + "\n", encoding="utf-8")

    result = FileTools(tmp_path).jsonl_search({
        "path": "*.jsonl",
        "field_searches": [{"field": "blob", "query": "target", "context_lines": 1, "max_matches": 1}],
    })

    assert result.ok, result.content
    assert "      1: target one" in result.content
    assert "      2: target two" in result.content
    assert "more match(es)" not in result.content

def test_jsonl_search_field_search_truncates_and_respects_case_sensitive(tmp_path: Path) -> None:
    (tmp_path / "states.jsonl").write_text(json.dumps({"id": 1, "blob": "Needle abcdef\nneedle ghijkl"}) + "\n", encoding="utf-8")

    insensitive = FileTools(tmp_path).jsonl_search({
        "path": "*.jsonl",
        "field_searches": [{"field": "blob", "query": "needle", "max_line_chars": 10}],
    })
    sensitive = FileTools(tmp_path).jsonl_search({
        "path": "*.jsonl",
        "field_searches": [{"field": "blob", "query": "needle", "case_sensitive": True}],
    })

    assert insensitive.ok, insensitive.content
    assert "      1: Needle abc…" in insensitive.content
    assert "      2: needle ghi…" in insensitive.content
    assert sensitive.ok, sensitive.content
    assert "Needle" not in sensitive.content
    assert "      2: needle ghijkl" in sensitive.content

def test_jsonl_search_field_search_string_miss_without_fields_renders_plain_none(tmp_path: Path) -> None:
    (tmp_path / "states.jsonl").write_text(json.dumps({"id": 1, "blob": "alpha"}) + "\n", encoding="utf-8")

    result = FileTools(tmp_path).jsonl_search({"path": "*.jsonl", "field_searches": [{"field": "blob", "query": "needle"}]})

    assert result.ok, result.content
    assert "  1: {}\n    blob matches: none" in result.content

def test_jsonl_search_field_search_regex_is_case_insensitive_by_default(tmp_path: Path) -> None:
    (tmp_path / "states.jsonl").write_text(json.dumps({"id": 1, "blob": "Incident Portal"}) + "\n", encoding="utf-8")

    result = FileTools(tmp_path).jsonl_search({
        "path": "*.jsonl",
        "field_searches": [{"field": "blob", "query": "incident", "regex": True}],
    })

    assert result.ok, result.content
    assert "      1: Incident Portal" in result.content

def test_jsonl_search_field_search_regex_respects_case_sensitive(tmp_path: Path) -> None:
    (tmp_path / "states.jsonl").write_text(json.dumps({"id": 1, "blob": "Needle\nneedle"}) + "\n", encoding="utf-8")

    result = FileTools(tmp_path).jsonl_search({
        "path": "*.jsonl",
        "field_searches": [{"field": "blob", "query": "needle", "regex": True, "case_sensitive": True}],
    })

    assert result.ok, result.content
    assert "Needle" not in result.content
    assert "      2: needle" in result.content

def test_jsonl_search_field_search_invalid_regex_and_field_path_fail_clearly(tmp_path: Path) -> None:
    tools = FileTools(tmp_path)
    (tmp_path / "states.jsonl").write_text(json.dumps({"id": 1, "blob": "needle"}) + "\n", encoding="utf-8")

    invalid_regex = tools.jsonl_search({"path": "*.jsonl", "field_searches": [{"field": "blob", "query": "[", "regex": True}]})
    invalid_path = tools.jsonl_search({"path": "*.jsonl", "field_searches": [{"field": "blob[", "query": "needle"}]})

    assert not invalid_regex.ok
    assert invalid_regex.content.startswith("invalid field_search regex:")
    assert not invalid_path.ok
    assert invalid_path.content.startswith("invalid field path:")

def test_jsonl_search_field_search_missing_and_non_string_fields_are_compact(tmp_path: Path) -> None:
    (tmp_path / "states.jsonl").write_text(
        "\n".join([
            json.dumps({"id": 1, "blob": 123}),
            json.dumps({"id": 2, "blob": None}),
            json.dumps({"id": 3}),
        ])
        + "\n",
        encoding="utf-8",
    )

    result = FileTools(tmp_path).jsonl_search({"path": "*.jsonl", "field_searches": [{"field": "blob", "query": "needle"}]})

    assert result.ok, result.content
    assert "rows_matched: 3" in result.content
    assert "  1: {}\n    blob matches: none (non-string field)" in result.content
    assert "  2: {}\n    blob matches: none (null field)" in result.content
    assert "  3: {}\n    blob matches: none (missing field)" in result.content

def test_jsonl_search_field_search_suppresses_miss_notes_when_sibling_matches(tmp_path: Path) -> None:
    (tmp_path / "states.jsonl").write_text(json.dumps({"id": 1, "body": "needle", "other": 123}) + "\n", encoding="utf-8")

    result = FileTools(tmp_path).jsonl_search({
        "path": "*.jsonl",
        "field_searches": [
            {"field": "body", "query": "needle"},
            {"field": "other", "query": "needle"},
        ],
    })

    assert result.ok, result.content
    assert "    body matches:" in result.content
    assert "other matches" not in result.content

def test_jsonl_search_field_search_single_line_field_with_context(tmp_path: Path) -> None:
    (tmp_path / "states.jsonl").write_text(json.dumps({"id": 1, "body": "needle"}) + "\n", encoding="utf-8")

    result = FileTools(tmp_path).jsonl_search({
        "path": "*.jsonl",
        "field_searches": [{"field": "body", "query": "needle", "context_lines": 3}],
    })

    assert result.ok, result.content
    assert "      1: needle" in result.content
    assert "      0:" not in result.content

def test_jsonl_search_field_search_combines_with_range_where_filters(tmp_path: Path) -> None:
    rows = [
        {"id": 1, "score": 2, "blob": "target line"},
        {"id": 2, "score": "bad", "blob": "target line"},
        {"id": 3, "score": 0, "blob": "target line"},
    ]
    (tmp_path / "states.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    result = FileTools(tmp_path).jsonl_search({
        "path": "*.jsonl",
        "where": [{"field": "score", "op": "gt", "value": "1", "type": "number"}],
        "fields": {"id": 0},
        "field_searches": [{"field": "blob", "query": "target"}],
    })

    assert result.ok, result.content
    assert "rows_matched: 1" in result.content
    assert "compare_warnings: 1 row(s) had non-comparable values" in result.content
    assert result.metadata["compare_warnings"] == 1
    assert '  1: {"id": 1}' in result.content
    assert "      1: target line" in result.content

def test_jsonl_search_number_equality_filters_match_json_numbers_only(tmp_path: Path) -> None:
    rows = [
        {"id": 1, "state_index": 1},
        {"id": 2, "state_index": "1"},
        {"id": 3, "state_index": True},
        {"id": 4, "state_index": 2},
        {"id": 5, "state_index": 1.0},
        {"id": 6},
    ]
    (tmp_path / "events.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    tools = FileTools(tmp_path)

    eq = tools.jsonl_search({
        "path": "*.jsonl",
        "where": [{"field": "state_index", "op": "eq", "value": "1", "type": "number"}],
        "fields": {"id": 0},
    })
    ne = tools.jsonl_search({
        "path": "*.jsonl",
        "where": [{"field": "state_index", "op": "ne", "value": "1", "type": "number"}],
        "fields": {"id": 0},
    })

    assert eq.ok, eq.content
    assert "where: state_index eq '1' (number)" in eq.content
    assert "rows_matched: 2" in eq.content
    assert '  1: {"id": 1}' in eq.content
    assert '  5: {"id": 5}' in eq.content
    assert '"id": 2' not in eq.content
    assert eq.metadata["compare_warnings"] == 3
    assert ne.ok, ne.content
    assert "rows_matched: 1" in ne.content
    assert '  4: {"id": 4}' in ne.content
    assert '"id": 1' not in ne.content

def test_jsonl_search_number_range_filters_match_json_numbers_only(tmp_path: Path) -> None:
    rows = [
        {"id": 1, "score": 9.5},
        {"id": 2, "score": 8},
        {"id": 3, "score": "9.5"},
        {"id": 4, "score": True},
        {"id": 5, "score": float("nan")},
        {"id": 6, "score": -3},
        {"id": 7},
        {"id": 8, "score": None},
        {"id": 9, "score": {"nested": 9}},
        {"id": 10, "score": [9]},
    ]
    (tmp_path / "events.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    result = FileTools(tmp_path).jsonl_search({
        "path": "*.jsonl",
        "where": [{"field": "score", "op": "gt", "value": "8", "type": "number"}],
        "fields": {"id": 0},
    })

    assert result.ok, result.content
    assert "scope: path=*.jsonl" in result.content
    assert "where: score gt '8' (number)" in result.content
    assert "rows_matched: 1" in result.content
    assert '  1: {"id": 1}' in result.content
    assert '"id": 2' not in result.content
    assert result.metadata["compare_warnings"] == 7
    assert "compare_warnings: 7 row(s) had non-comparable values" in result.content

def test_jsonl_search_number_range_strictness_and_negative_targets(tmp_path: Path) -> None:
    (tmp_path / "events.jsonl").write_text(
        "\n".join(json.dumps({"id": index, "score": score}) for index, score in enumerate([-2, -1, 0], start=1)) + "\n",
        encoding="utf-8",
    )
    tools = FileTools(tmp_path)

    gt = tools.jsonl_search({
        "path": "*.jsonl",
        "where": [{"field": "score", "op": "gt", "value": "-1", "type": "number"}],
        "fields": {"id": 0},
    })
    lt = tools.jsonl_search({
        "path": "*.jsonl",
        "where": [{"field": "score", "op": "lt", "value": "-1", "type": "number"}],
        "fields": {"id": 0},
    })

    assert gt.ok
    assert "rows_matched: 1" in gt.content
    assert '  3: {"id": 3}' in gt.content
    assert lt.ok
    assert "rows_matched: 1" in lt.content
    assert '  1: {"id": 1}' in lt.content
    assert "compare_warnings" not in gt.content
    assert "compare_warnings" not in gt.metadata

def test_jsonl_search_number_range_filters_compare_large_integers_exactly(tmp_path: Path) -> None:
    target = 9_007_199_254_740_993
    rows = [
        {"id": 1, "counter": target - 1},
        {"id": 2, "counter": target},
        {"id": 3, "counter": target + 1},
    ]
    (tmp_path / "events.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    tools = FileTools(tmp_path)

    cases = [
        ("gt", [3]),
        ("gte", [2, 3]),
        ("lt", [1]),
        ("lte", [1, 2]),
    ]
    for op, expected_ids in cases:
        result = tools.jsonl_search({
            "path": "*.jsonl",
            "where": [{"field": "counter", "op": op, "value": str(target), "type": "number"}],
            "fields": {"id": 0},
        })

        assert result.ok, result.content
        assert f"rows_matched: {len(expected_ids)}" in result.content
        for expected_id in expected_ids:
            assert f'{{"id": {expected_id}}}' in result.content
        for unexpected_id in {1, 2, 3} - set(expected_ids):
            assert f'{{"id": {unexpected_id}}}' not in result.content

def test_jsonl_search_rejects_invalid_range_filter_definitions_before_scanning(tmp_path: Path) -> None:
    tools = FileTools(tmp_path)
    bad_filters = [
        [{"field": "score", "op": "gt", "value": "NaN", "type": "number"}],
        [{"field": "score", "op": "gt", "value": "Infinity", "type": "number"}],
        [{"field": "score", "op": "gt", "value": "-Infinity", "type": "number"}],
        [{"field": "score", "op": "gt", "value": "", "type": "number"}],
        [{"field": "score", "op": "gt", "value": "1"}],
        [{"field": "score", "op": "gt", "value": "1", "type": "number", "values": ["1"]}],
        [{"field": "score", "op": "contains", "value": "1", "type": "number"}],
        [{"field": "published_at", "op": "lt", "value": "not-a-date", "type": "date"}],
    ]

    for where in bad_filters:
        result = tools.jsonl_search({"path": "*.jsonl", "where": where})
        assert not result.ok
        assert result.content.startswith("invalid where filter:"), where

def test_jsonl_search_date_range_filters_use_declared_date_semantics(tmp_path: Path) -> None:
    rows = [
        {"id": 1, "published_at": "2026-06-11"},
        {"id": 2, "published_at": "2026-06-12"},
        {"id": 3, "published_at": "2026-06-12T15:30:00Z"},
        {"id": 4, "published_at": "2026-06-12T23:30:00-04:00"},
        {"id": 5, "published_at": "2026-06-13"},
    ]
    (tmp_path / "events.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    tools = FileTools(tmp_path)

    lte = tools.jsonl_search({
        "path": "*.jsonl",
        "where": [{"field": "published_at", "op": "lte", "value": "2026-06-12", "type": "date"}],
        "fields": {"id": 0},
    })
    strict_gt = tools.jsonl_search({
        "path": "*.jsonl",
        "where": [{"field": "published_at", "op": "gt", "value": "2026-06-12", "type": "date"}],
        "fields": {"id": 0},
    })

    assert lte.ok, lte.content
    assert "where: published_at lte '2026-06-12' (date)" in lte.content
    assert "rows_matched: 4" in lte.content
    assert '  3: {"id": 3}' in lte.content
    assert '  4: {"id": 4}' in lte.content
    assert strict_gt.ok
    assert "rows_matched: 1" in strict_gt.content
    assert '  5: {"id": 5}' in strict_gt.content

def test_jsonl_search_datetime_range_filters_compare_like_awareness(tmp_path: Path) -> None:
    rows = [
        {"id": 1, "published_at": "2026-06-12T08:00:00"},
        {"id": 2, "published_at": "2026-06-12T10:00:00"},
        {"id": 3, "published_at": "2026-06-12T15:00:00Z"},
        {"id": 4, "published_at": "not-a-date"},
    ]
    (tmp_path / "events.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    result = FileTools(tmp_path).jsonl_search({
        "path": "*.jsonl",
        "where": [{"field": "published_at", "op": "gte", "value": "2026-06-12T09:00:00", "type": "date"}],
        "fields": {"id": 0},
    })

    assert result.ok, result.content
    assert "rows_matched: 1" in result.content
    assert '  2: {"id": 2}' in result.content
    assert result.metadata["compare_warnings"] == 2

def test_jsonl_search_range_warning_counts_candidate_rows_once(tmp_path: Path) -> None:
    rows = [
        {"id": 1, "kind": "keep", "score": "bad", "other": "bad"},
        {"id": 2, "kind": "skip", "score": "bad", "other": "bad"},
        {"id": 3, "kind": "keep", "score": 2, "other": "bad"},
    ]
    (tmp_path / "events.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    result = FileTools(tmp_path).jsonl_search({
        "path": "*.jsonl",
        "where": [
            {"field": "kind", "op": "eq", "value": "keep"},
            {"field": "score", "op": "gte", "value": "1", "type": "number"},
            {"field": "other", "op": "gte", "value": "1", "type": "number"},
        ],
        "fields": {"id": 0},
    })

    assert result.ok, result.content
    assert "rows_matched: 0" in result.content
    assert result.metadata["compare_warnings"] == 2

def test_jsonl_search_compare_warnings_coexist_with_ripgrep_warnings(tmp_path: Path, monkeypatch) -> None:
    def partial(*args, **kwargs):
        stdout = "\n".join([
            _rg_match("events.jsonl", 1, '{"id":1,"msg":"login","score":"bad"}'),
            "rg: ./restricted: Permission denied",
        ])
        return subprocess.CompletedProcess(args[0], 2, stdout=stdout)

    monkeypatch.setattr("subprocess.run", partial)
    (tmp_path / "events.jsonl").write_text('{"id":1,"msg":"login","score":"bad"}\n', encoding="utf-8")

    result = FileTools(tmp_path).jsonl_search({
        "query": "login",
        "path": "*.jsonl",
        "where": [{"field": "score", "op": "gte", "value": "1", "type": "number"}],
    })

    assert result.ok, result.content
    assert result.metadata["warning"] == "ripgrep returned 2; showing parsed partial matches"
    assert result.metadata["compare_warnings"] == 1

def test_spill_output_uses_thinharness_directory_and_read_guidance(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("\n".join(f"hit {i}" for i in range(100)), encoding="utf-8")
    tools = FileTools(tmp_path, max_tool_chars=120)

    result = tools.search({"query": "hit"})

    assert result.ok
    assert result.metadata["truncated"] is True
    assert result.metadata["saved_to_display"].startswith(".thinharness/outputs/search-")
    assert Path(result.metadata["saved_to"]).exists()
    assert "Read the saved output with read(path=\".thinharness/outputs/search-" in result.content
    assert "offset=1)" in result.content

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
    assert SearchArgs(query="hit").path == "."
    assert JsonlSearchArgs().path == "."
    assert SearchArgs(query="hit").max_files == 50
    assert SearchArgs(query="hit").max_matches_per_file == 10
    assert JsonlSearchArgs().max_files == 100
    assert JsonlSearchArgs().max_matches_per_file == 25

def test_raised_search_display_defaults_show_more_than_old_limits(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    for index in range(11):
        (docs / f"doc_{index:02}.txt").write_text("hit\n", encoding="utf-8")
    (tmp_path / "events.jsonl").write_text(
        "\n".join(json.dumps({"id": index, "msg": "hit"}) for index in range(4)) + "\n",
        encoding="utf-8",
    )
    tools = FileTools(tmp_path)

    search = tools.search({"query": "hit", "path": "docs"})
    jsonl = tools.jsonl_search({"path": "*.jsonl"})

    assert "files: 11 total, 11 shown" in search.content
    assert "matches: 11 shown, 0 omitted" in search.content
    assert "rows_matched: 4" in jsonl.content
    assert '  4: {"id": 3, "msg": "hit"}' in jsonl.content

def test_gitignore_ignores_thinharness_outputs() -> None:
    ignore = Path(".gitignore").read_text(encoding="utf-8")

    assert ".thinharness/" in ignore

def _contains_key(value: object, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(_contains_key(item, key) for item in value.values())
    if isinstance(value, list):
        return any(_contains_key(item, key) for item in value)
    return False

def _rg_match(path: str, line_number: int, line_text: str) -> str:
    return json.dumps({
        "type": "match",
        "data": {
            "path": {"text": path},
            "line_number": line_number,
            "lines": {"text": f"{line_text}\n"},
        },
    })
