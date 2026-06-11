from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.web_research_report.agent import (
    ExaFetchSourcesArgs,
    ExaSearchSourcesArgs,
    ExaTools,
    PrepareSourceNotePromptsArgs,
    SelectedSource,
    _source_audit_hook,
    build_harness,
)


class FakeResponse:
    def __init__(self, payload: dict[str, Any], *, status_code: int = 200) -> None:
        self.payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> dict[str, Any]:
        return self.payload


class FakeAsyncClient:
    payloads: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []

    def __init__(self, **_kwargs: Any) -> None:
        pass

    async def __aenter__(self) -> FakeAsyncClient:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        return None

    async def post(self, url: str, *, headers: dict[str, str], json: dict[str, Any]) -> FakeResponse:
        self.calls.append({"url": url, "headers": headers, "json": json})
        return FakeResponse(self.payloads.pop(0))


def test_exa_args_enforce_bounds() -> None:
    with pytest.raises(ValueError):
        ExaSearchSourcesArgs(queries=[], num_results_per_query=5)
    with pytest.raises(ValueError):
        ExaSearchSourcesArgs(queries=["q"], num_results_per_query=11)
    with pytest.raises(ValueError):
        ExaFetchSourcesArgs(sources=[], text_max_characters=12_000)
    with pytest.raises(ValueError):
        ExaFetchSourcesArgs(sources=[SelectedSource(source_id="s001", url="https://example.com")], text_max_characters=500)


async def test_exa_search_sources_writes_manifest_and_results(tmp_path: Path) -> None:
    FakeAsyncClient.calls = []
    FakeAsyncClient.payloads = [
        {
            "requestId": "req-1",
            "costDollars": 0.01,
            "results": [
                {
                    "title": "A",
                    "url": "https://a.example",
                    "publishedDate": "2026-01-01T00:00:00.000Z",
                    "author": "Ann",
                    "id": "exa-a",
                    "highlights": ["one"],
                    "score": 0.8,
                }
            ],
        },
        {"requestId": "req-2", "results": [{"title": "B", "url": "https://b.example", "id": "exa-b"}]},
    ]
    tools = ExaTools(tmp_path, api_key="exa-key", client_factory=FakeAsyncClient)

    result = await tools.search_sources(ExaSearchSourcesArgs(queries=["q1", "q2"], num_results_per_query=2))

    assert result.ok is True
    compact = json.loads(result.content)
    assert compact["query_count"] == 2
    assert compact["result_count"] == 2
    assert compact["request_ids"] == ["req-1", "req-2"]
    rows = (tmp_path / "outputs/sources/search_001/results.jsonl").read_text(encoding="utf-8").splitlines()
    first = json.loads(rows[0])
    assert first["source_id"] == "s001"
    assert first["query_id"] == "q001"
    assert first["highlights"] == ["one"]
    manifest = json.loads((tmp_path / "outputs/sources/search_001/manifest.json").read_text(encoding="utf-8"))
    assert manifest["queries"][0]["status"] == "success"
    assert FakeAsyncClient.calls[0]["headers"]["x-api-key"] == "exa-key"
    assert FakeAsyncClient.calls[0]["json"]["contents"] == {"highlights": True}


async def test_exa_fetch_sources_preserves_source_ids_and_statuses(tmp_path: Path) -> None:
    FakeAsyncClient.calls = []
    FakeAsyncClient.payloads = [
        {
            "requestId": "fetch-1",
            "costDollars": 0.02,
            "statuses": [
                {"url": "https://a.example", "status": "success", "source": "cached"},
                {"url": "https://b.example", "status": "error", "error": "blocked"},
            ],
            "results": [
                {"title": "A", "url": "https://a.example", "publishedDate": "2026-01-01T00:00:00.000Z", "text": "body", "highlights": ["h"]}
            ],
        }
    ]
    tools = ExaTools(tmp_path, api_key="exa-key", client_factory=FakeAsyncClient)

    result = await tools.fetch_sources(
        ExaFetchSourcesArgs(
            sources=[
                SelectedSource(source_id="s010", url="https://a.example"),
                SelectedSource(source_id="s011", url="https://b.example"),
            ],
            text_max_characters=3_000,
            highlights_query="query",
        )
    )

    assert result.ok is True
    compact = json.loads(result.content)
    assert compact["status_counts"] == {"success": 1, "error": 1}
    assert compact["failed_urls"] == ["https://b.example"]
    row = json.loads((tmp_path / "outputs/sources/fetch_001/documents.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert row["source_id"] == "s010"
    assert row["document_id"] == "d001"
    assert row["source"] == "cached"
    assert (tmp_path / "outputs/sources/fetch_001/documents/d001.json").exists()
    assert FakeAsyncClient.calls[0]["json"]["text"] == {"maxCharacters": 3_000}
    assert FakeAsyncClient.calls[0]["json"]["highlights"] == {"query": "query"}


def test_prepare_source_note_prompts_writes_full_document_record(tmp_path: Path) -> None:
    (tmp_path / "outputs/sources/fetch_001/documents").mkdir(parents=True)
    (tmp_path / "outputs/source_note_context.md").write_text("# Context\nFocus on adoption signals.", encoding="utf-8")
    document = {
        "document_id": "d001",
        "title": "Title",
        "url": "https://example.com",
        "text": "Full fetched document text.",
    }
    (tmp_path / "outputs/sources/fetch_001/documents/d001.json").write_text(json.dumps(document), encoding="utf-8")
    tools = ExaTools(tmp_path, api_key="exa-key")

    result = tools.prepare_source_note_prompts(
        PrepareSourceNotePromptsArgs(
            context_path="outputs/source_note_context.md",
            document_paths=["outputs/sources/fetch_001/documents/d001.json"],
        )
    )

    assert result.ok is True
    compact = json.loads(result.content)
    assert compact["prompt_count"] == 1
    prompts = json.loads((tmp_path / "outputs/source_note_prompts.json").read_text(encoding="utf-8"))
    assert len(prompts) == 1
    assert "Context:" in prompts[0]
    assert "Focus on adoption signals." in prompts[0]
    assert "Source record:" in prompts[0]
    assert '"text": "Full fetched document text."' in prompts[0]


def test_source_audit_hook_writes_exa_tool_summary(tmp_path: Path) -> None:
    output = {
        "ok": True,
        "content": json.dumps({
            "query_count": 2,
            "result_count": 4,
            "request_ids": ["req-1", "req-2"],
            "cost": 0.03,
            "manifest_path": "outputs/sources/search_001/manifest.json",
            "results_path": "outputs/sources/search_001/results.jsonl",
        }),
    }
    ctx = SimpleNamespace(
        tool_name="exa_search_sources",
        output=json.dumps(output),
        harness=SimpleNamespace(root=tmp_path),
    )

    _source_audit_hook(ctx)

    rows = (tmp_path / "outputs/source_audit.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1
    row = json.loads(rows[0])
    assert row["tool"] == "exa_search_sources"
    assert row["query_count"] == 2
    assert row["result_count"] == 4
    assert row["request_ids"] == ["req-1", "req-2"]
    assert row["cost"] == 0.03
    assert row["output_paths"] == {
        "manifest_path": "outputs/sources/search_001/manifest.json",
        "results_path": "outputs/sources/search_001/results.jsonl",
    }


def test_build_harness_uses_custom_parallel_llm_without_name_collision(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXA_API_KEY", "exa-key")
    harness = build_harness(tmp_path, model="openrouter:deepseek/deepseek-v4-pro")

    tool_names = [tool.name for tool in harness.tools]
    assert tool_names.count("parallel_llm") == 1
    assert "exa_search_sources" in tool_names
    assert "exa_fetch_sources" in tool_names
    assert "prepare_source_note_prompts" in tool_names
    assert "edit" in tool_names
    assert "subagent" in tool_names


def test_build_harness_uses_default_trace_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / "home"
    root = tmp_path / "example"
    monkeypatch.delenv("THINHARNESS_DISABLE_LOCAL_TRACING", raising=False)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("EXA_API_KEY", "exa-key")

    harness = build_harness(root, model="openrouter:deepseek/deepseek-v4-pro")

    assert harness.local_tracing is not None
    trace_dir = harness.local_tracing.trace_dir
    assert str(trace_dir).startswith(str(home / ".thinharness" / "traces"))
    assert not str(trace_dir).startswith(str(root))


def test_build_harness_keeps_critical_path_tools_synchronous(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXA_API_KEY", "exa-key")
    harness = build_harness(tmp_path, model="openrouter:deepseek/deepseek-v4-pro")

    schemas = {schema["name"]: schema for schema in harness.tool_schemas()}

    assert "_background" not in schemas["parallel_llm"]["parameters"]["properties"]
    assert "_background" not in schemas["subagent"]["parameters"]["properties"]
