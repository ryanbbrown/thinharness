from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import tempfile
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field

from thinharness import Harness, HarnessConfig, Hook, ParallelLlmTool, PathPolicy, PathValidationError, SubAgentConfig, ToolResult, ToolSpec

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = os.getenv("WEB_RESEARCH_REPORT_MODEL", "openrouter:deepseek/deepseek-v4-pro")
CURRENT_DATE = os.getenv("WEB_RESEARCH_REPORT_CURRENT_DATE", "June 9, 2026")
TARGET_QUERY_COUNT = os.getenv("WEB_RESEARCH_REPORT_TARGET_QUERIES", "8")
TARGET_SOURCE_COUNT = os.getenv("WEB_RESEARCH_REPORT_TARGET_SOURCES", "8")
DEFAULT_PROMPT = (
    "Prepare a market landscape brief for a product strategy team evaluating AI-powered support quality monitoring tools "
    "for mid-market B2B SaaS companies. Identify buyer pain, vendor categories, adoption signals, risks, and recommended next research steps."
)
EXA_BASE_URL = "https://api.exa.ai"


class SourcePoint(BaseModel):
    point: str
    evidence_excerpt: str


class SourceNote(BaseModel):
    document_id: str
    title: str
    url: str
    source_type: Literal["vendor_page", "news", "analyst_post", "docs", "case_study", "forum", "other"]
    summary: str
    useful_points: list[SourcePoint]
    numbers_or_dates: list[str]
    limitations: list[str]
    citation_worthy: bool


class ReportReceipt(BaseModel):
    completed: bool
    report_path: str
    source_count: int
    citation_issues: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    next_action: str


class ExaSearchSourcesArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    queries: list[str] = Field(min_length=1, max_length=16, description="Search queries to run as one batched research step.")
    num_results_per_query: int = Field(default=5, ge=1, le=10)
    search_type: Literal["auto", "keyword", "neural"] = "auto"
    include_domains: list[str] | None = None
    exclude_domains: list[str] | None = None
    start_published_date: str | None = None
    end_published_date: str | None = None


class SelectedSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str
    url: str


class ExaFetchSourcesArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sources: list[SelectedSource] = Field(min_length=1, max_length=20)
    text_max_characters: int = Field(default=12_000, ge=1_000, le=30_000)
    highlights_query: str | None = None


class PrepareSourceNotePromptsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_paths: list[str] = Field(min_length=1, max_length=20)
    research_brief_path: str = "outputs/research_plan.md"
    output_file: str = "outputs/source_note_prompts.json"


class CompactSourceNotesArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    batch_paths: list[str] = Field(default_factory=lambda: ["outputs/source_notes_batch.json"], min_length=1, max_length=3)
    output_file: str = "outputs/source_notes_compact.md"
    max_points_per_source: int = Field(default=2, ge=1, le=4)


class ExaTools:
    def __init__(
        self,
        root: Path,
        *,
        api_key: str | None = None,
        base_url: str = EXA_BASE_URL,
        timeout: int = 90,
        client_factory: Callable[..., Any] = httpx.AsyncClient,
    ) -> None:
        self.root = root
        self.api_key = api_key or os.getenv("EXA_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.client_factory = client_factory

    def specs(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                "exa_search_sources",
                (
                    "Run a batched Exa search. Provide 8-12 targeted queries for serious research. "
                    "The tool writes outputs/sources/search_N/results.jsonl and returns compact paths and top URL pairs."
                ),
                ExaSearchSourcesArgs,
                self.search_sources,
            ),
            ToolSpec(
                "exa_fetch_sources",
                (
                    "Fetch full Exa contents for selected search results. Pass source_id/url objects selected from saved search results. "
                    "The tool writes outputs/sources/fetch_N/documents.jsonl, per-document JSON files, and a manifest with Exa statuses."
                ),
                ExaFetchSourcesArgs,
                self.fetch_sources,
            ),
            ToolSpec(
                "prepare_source_note_prompts",
                (
                    "Build outputs/source_note_prompts.json from fetched per-document JSON files. "
                    "Use this deterministic helper after exa_fetch_sources and before parallel_llm."
                ),
                PrepareSourceNotePromptsArgs,
                self.prepare_source_note_prompts,
                sequential=True,
            ),
            ToolSpec(
                "compact_source_notes",
                (
                    "Parse one or more parallel_llm source-note batch JSON files and write a compact markdown evidence file. "
                    "Use this before drafting so the agent does not need to read large batch JSON payloads."
                ),
                CompactSourceNotesArgs,
                self.compact_source_notes,
                sequential=True,
            ),
        ]

    async def search_sources(self, args: ExaSearchSourcesArgs) -> ToolResult:
        if not self.api_key:
            return ToolResult(False, "EXA_API_KEY is required for exa_search_sources", {"error_type": "MissingApiKey"})

        run_name, folder = self._next_folder("search")
        manifest_path = folder / "manifest.json"
        results_path = folder / "results.jsonl"
        queries = list(args.queries)

        async with self.client_factory(timeout=self.timeout, follow_redirects=True) as client:
            tasks = [
                self._post_exa(client, "/search", _search_payload(args, query))
                for query in queries
            ]
            responses = await asyncio.gather(*tasks, return_exceptions=True)

        rows: list[dict[str, Any]] = []
        query_entries: list[dict[str, Any]] = []
        source_counter = 1
        total_cost = 0.0
        request_ids: list[str] = []

        for query_index, (query, response) in enumerate(zip(queries, responses, strict=True), start=1):
            query_id = f"q{query_index:03d}"
            if isinstance(response, BaseException):
                query_entries.append({"query_id": query_id, "query": query, "status": "error", "error": _safe_error(response), "result_count": 0})
                continue
            request_id = _response_request_id(response)
            cost = _response_cost(response)
            if request_id:
                request_ids.append(request_id)
            total_cost += cost or 0.0
            results = response.get("results") or []
            query_entries.append(
                {
                    "query_id": query_id,
                    "query": query,
                    "status": "success",
                    "result_count": len(results),
                    "request_id": request_id,
                    "cost": cost,
                }
            )
            for rank, result in enumerate(results, start=1):
                rows.append(
                    {
                        "search_run": run_name,
                        "query_id": query_id,
                        "query": query,
                        "rank": rank,
                        "source_id": f"s{source_counter:03d}",
                        "title": result.get("title"),
                        "url": result.get("url"),
                        "published_date": result.get("publishedDate") or result.get("published_date"),
                        "author": result.get("author"),
                        "exa_id": result.get("id"),
                        "highlights": _normalize_highlights(result.get("highlights")),
                        "score": result.get("score"),
                    }
                )
                source_counter += 1

        manifest = {
            "search_run": run_name,
            "created_at": _now(),
            "request": args.model_dump(),
            "query_count": len(queries),
            "result_count": len(rows),
            "queries": query_entries,
            "request_ids": request_ids,
            "cost": total_cost,
        }
        _write_json(manifest_path, manifest)
        _write_jsonl(results_path, rows)

        failures = [entry for entry in query_entries if entry["status"] != "success"]
        compact = {
            "folder_path": _rel(folder, self.root),
            "manifest_path": _rel(manifest_path, self.root),
            "results_path": _rel(results_path, self.root),
            "query_count": len(queries),
            "result_count": len(rows),
            "request_ids": request_ids,
            "cost": total_cost,
            "query_failures": failures,
            "top_results": [{"title": row["title"], "url": row["url"], "source_id": row["source_id"]} for row in rows[:8]],
        }
        ok = bool(rows)
        return ToolResult(ok, json.dumps(compact, ensure_ascii=False, separators=(",", ":")), {"query_failures": len(failures)})

    async def fetch_sources(self, args: ExaFetchSourcesArgs) -> ToolResult:
        if not self.api_key:
            return ToolResult(False, "EXA_API_KEY is required for exa_fetch_sources", {"error_type": "MissingApiKey"})

        run_name, folder = self._next_folder("fetch")
        manifest_path = folder / "manifest.json"
        documents_path = folder / "documents.jsonl"
        documents_folder = folder / "documents"
        documents_folder.mkdir(parents=True, exist_ok=True)
        sources = list(args.sources)
        urls = [source.url for source in sources]
        source_by_url = {source.url: source.source_id for source in sources}
        payload: dict[str, Any] = {"urls": urls, "text": {"maxCharacters": args.text_max_characters}}
        if args.highlights_query:
            payload["highlights"] = {"query": args.highlights_query}

        try:
            async with self.client_factory(timeout=self.timeout, follow_redirects=True) as client:
                response = await self._post_exa(client, "/contents", payload)
        except Exception as exc:
            manifest = {
                "fetch_run": run_name,
                "created_at": _now(),
                "request": args.model_dump(),
                "status": "error",
                "error": _safe_error(exc),
                "status_counts": {"error": len(urls)},
                "failed_urls": urls,
            }
            _write_json(manifest_path, manifest)
            _write_jsonl(documents_path, [])
            return ToolResult(False, json.dumps({"manifest_path": _rel(manifest_path, self.root), "error": _safe_error(exc)}, separators=(",", ":")))

        request_id = _response_request_id(response)
        cost = _response_cost(response)
        statuses = _normalize_statuses(response.get("statuses"), urls)
        status_by_url = {status["url"]: status for status in statuses}
        rows: list[dict[str, Any]] = []
        document_files: list[str] = []

        for document_index, result in enumerate(response.get("results") or [], start=1):
            url = str(result.get("url") or (urls[document_index - 1] if document_index - 1 < len(urls) else ""))
            status = status_by_url.get(url, {"status": "success", "url": url})
            row = {
                "fetch_run": run_name,
                "document_id": f"d{document_index:03d}",
                "source_id": source_by_url.get(url) or _source_id_for_position(sources, document_index),
                "title": result.get("title"),
                "url": url,
                "published_date": result.get("publishedDate") or result.get("published_date"),
                "status": status.get("status") or "success",
                "source": status.get("source") or result.get("source"),
                "text": result.get("text") or "",
                "highlights": _normalize_highlights(result.get("highlights")),
            }
            rows.append(row)
            document_path = documents_folder / f"{row['document_id']}.json"
            _write_json(document_path, row)
            document_files.append(_rel(document_path, self.root))

        status_counts = Counter(status.get("status") or "unknown" for status in statuses)
        result_urls = {str(row["url"]) for row in rows}
        failed_urls = [
            status["url"]
            for status in statuses
            if status.get("status") not in {None, "success"} or status["url"] not in result_urls
        ]
        manifest = {
            "fetch_run": run_name,
            "created_at": _now(),
            "request": args.model_dump(),
            "result_count": len(rows),
            "statuses": statuses,
            "status_counts": dict(status_counts),
            "failed_urls": failed_urls,
            "request_id": request_id,
            "cost": cost,
            "documents_path": _rel(documents_path, self.root),
            "document_files": document_files,
        }
        _write_json(manifest_path, manifest)
        _write_jsonl(documents_path, rows)

        compact = {
            "folder_path": _rel(folder, self.root),
            "manifest_path": _rel(manifest_path, self.root),
            "documents_path": _rel(documents_path, self.root),
            "document_files": document_files,
            "status_counts": dict(status_counts),
            "failed_urls": failed_urls,
            "request_id": request_id,
            "cost": cost,
        }
        return ToolResult(bool(rows), json.dumps(compact, ensure_ascii=False, separators=(",", ":")), {"failed_urls": len(failed_urls)})

    def prepare_source_note_prompts(self, args: PrepareSourceNotePromptsArgs) -> ToolResult:
        read_policy = PathPolicy(self.root, ["outputs"], "read")
        write_policy = PathPolicy(self.root, ["outputs"], "write")
        try:
            brief_path = read_policy.resolve(args.research_brief_path)
            output_path = write_policy.resolve(args.output_file)
            document_paths = [read_policy.resolve(path) for path in args.document_paths]
        except PathValidationError as exc:
            return ToolResult(False, str(exc), {"error_type": "PathValidationError"})
        if not brief_path.exists():
            return ToolResult(False, f"research brief not found: {args.research_brief_path}", {"error_type": "FileNotFound"})

        research_brief = brief_path.read_text(encoding="utf-8", errors="replace")
        prompts: list[str] = []
        document_ids: list[str] = []
        for document_path in document_paths:
            if not document_path.exists():
                return ToolResult(False, f"document not found: {_rel(document_path, self.root)}", {"error_type": "FileNotFound"})
            try:
                document = json.loads(document_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                return ToolResult(False, f"invalid document JSON at {_rel(document_path, self.root)}: {exc}", {"error_type": "JSONDecodeError"})
            if not isinstance(document, dict):
                return ToolResult(False, f"document JSON is not an object: {_rel(document_path, self.root)}", {"error_type": "InvalidDocument"})
            document_ids.append(str(document.get("document_id") or document_path.stem))
            prompts.append(_source_note_prompt(research_brief, document))

        _write_json(output_path, prompts)
        compact = {
            "output_file": _rel(output_path, self.root),
            "prompt_count": len(prompts),
            "document_ids": document_ids,
        }
        return ToolResult(True, json.dumps(compact, ensure_ascii=False, separators=(",", ":")))

    def compact_source_notes(self, args: CompactSourceNotesArgs) -> ToolResult:
        read_policy = PathPolicy(self.root, ["outputs"], "read")
        write_policy = PathPolicy(self.root, ["outputs"], "write")
        try:
            batch_paths = [read_policy.resolve(path) for path in args.batch_paths]
            output_path = write_policy.resolve(args.output_file)
        except PathValidationError as exc:
            return ToolResult(False, str(exc), {"error_type": "PathValidationError"})

        notes: list[dict[str, Any]] = []
        failures = 0
        for batch_path in batch_paths:
            if not batch_path.exists():
                return ToolResult(False, f"batch not found: {_rel(batch_path, self.root)}", {"error_type": "FileNotFound"})
            try:
                batch = json.loads(batch_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                return ToolResult(False, f"invalid batch JSON at {_rel(batch_path, self.root)}: {exc}", {"error_type": "JSONDecodeError"})
            if not isinstance(batch, dict):
                return ToolResult(False, f"batch JSON is not an object: {_rel(batch_path, self.root)}", {"error_type": "InvalidBatch"})
            for item in batch.get("results") or []:
                if not isinstance(item, dict) or item.get("ok") is not True:
                    failures += 1
                    continue
                result = item.get("result")
                if isinstance(result, dict):
                    notes.append(result)
                else:
                    failures += 1

        lines = ["# Compact Source Notes", ""]
        citation_worthy = 0
        for index, note in enumerate(notes, start=1):
            if note.get("citation_worthy"):
                citation_worthy += 1
            lines.extend([
                f"## {index}. {note.get('title') or 'Untitled'}",
                f"- URL: {note.get('url') or ''}",
                f"- Document ID: {note.get('document_id') or ''}",
                f"- Source type: {note.get('source_type') or 'other'}",
                f"- Citation worthy: {bool(note.get('citation_worthy'))}",
                f"- Summary: {_one_line(note.get('summary'), 700)}",
            ])
            numbers = note.get("numbers_or_dates")
            if isinstance(numbers, list) and numbers:
                lines.append(f"- Numbers/dates/signals: {_one_line('; '.join(str(item) for item in numbers[:8]), 700)}")
            points = note.get("useful_points")
            if isinstance(points, list) and points:
                lines.append("- Useful points:")
                for point in points[: args.max_points_per_source]:
                    if isinstance(point, dict):
                        text = point.get("point") or ""
                        excerpt = point.get("evidence_excerpt") or ""
                        lines.append(f"  - {_one_line(text, 300)} Evidence: {_one_line(excerpt, 300)}")
            limitations = note.get("limitations")
            if isinstance(limitations, list) and limitations:
                lines.append(f"- Limitations: {_one_line('; '.join(str(item) for item in limitations[:4]), 500)}")
            lines.append("")

        _atomic_write(output_path, "\n".join(lines).rstrip() + "\n")
        compact = {
            "output_file": _rel(output_path, self.root),
            "source_count": len(notes),
            "citation_worthy_count": citation_worthy,
            "failed_results": failures,
        }
        return ToolResult(True, json.dumps(compact, ensure_ascii=False, separators=(",", ":")))

    async def _post_exa(self, client: Any, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        response = await client.post(
            f"{self.base_url}{path}",
            headers={"x-api-key": self.api_key, "Content-Type": "application/json"},
            json=payload,
        )
        response.raise_for_status()
        parsed = response.json()
        if not isinstance(parsed, dict):
            raise ValueError("Exa response was not a JSON object")
        return parsed

    def _next_folder(self, kind: Literal["search", "fetch"]) -> tuple[str, Path]:
        sources_root = self.root / "outputs" / "sources"
        sources_root.mkdir(parents=True, exist_ok=True)
        pattern = re.compile(rf"^{kind}_(\d{{3}})$")
        existing = []
        for child in sources_root.iterdir():
            match = pattern.match(child.name)
            if match:
                existing.append(int(match.group(1)))
        run_name = f"{kind}_{max(existing, default=0) + 1:03d}"
        folder = sources_root / run_name
        folder.mkdir(parents=True, exist_ok=False)
        return run_name, folder


SYSTEM_PROMPT = (
    "You are a web research report agent.\n"
    "\n"
    f"Current date: {CURRENT_DATE}. Use this date for the report metadata; do not use stale dates.\n"
    "\n"
    "Your job is to produce a concise, source-grounded markdown research report for the user's question. "
    "The user's prompt supplies the topic; do not assume a fixed domain or fixed report sections.\n"
    "\n"
    "Work from saved artifacts, not memory. Search results and fetched web pages should be written to workspace files, "
    "then read back before you rely on them. Keep source-backed facts separate from your interpretation.\n"
    "\n"
    "Required workflow:\n"
    "1. Write outputs/research_plan.md before searching. The plan should define the report sections or research questions "
    "implied by the user request, explain why each matters, and list the initial search queries you intend to run.\n"
    f"2. Use exa_search_sources for the initial query set. Use about {TARGET_QUERY_COUNT} well-targeted queries.\n"
    "3. Read or search the Exa results JSONL before selecting URLs to fetch. Before fetching, write outputs/source_selection.md "
    "with selected source_id/url pairs and a short rationale. This artifact is required.\n"
    "4. Fetch full contents only for selected high-value URLs with exa_fetch_sources. Do not fetch every search result "
    f"automatically. For this first public trace, select {TARGET_SOURCE_COUNT} high-value sources.\n"
    "5. Call prepare_source_note_prompts with the fetched per-document JSON paths returned by exa_fetch_sources. Do not hand-write "
    "or summarize the prompt file yourself; the helper must include one full source record per prompt.\n"
    "6. Use parallel_llm to create one source note per fetched document. Use "
    "source={\"kind\":\"file\",\"path\":\"outputs/source_note_prompts.json\"} and write the batch artifact to "
    "outputs/source_notes_batch.json.\n"
    "7. Call compact_source_notes on outputs/source_notes_batch.json and read outputs/source_notes_compact.md before drafting. "
    "Do not read the full source_notes_batch.json unless resolving a specific discrepancy.\n"
    "8. Skip follow-up search for the default trace unless the compact evidence has zero independent adoption or funding signals; "
    "preserve uncertainty instead of expanding scope.\n"
    "9. Write outputs/draft_report.md, then ask the citation_critic subagent to review the draft against outputs/source_notes_compact.md. "
    "Do not ask the critic to read large batch JSON files unless there is a specific citation discrepancy.\n"
    "10. Revise useful issues from the critic and write the final markdown report to outputs/market_landscape_report.md.\n"
    "\n"
    "Evidence standards:\n"
    "- Cite sources by title and URL.\n"
    "- Use vendor-authored sources carefully; they can support claims about product positioning, but not broad market "
    "conclusions by themselves.\n"
    "- Prefer concrete numbers, dates, named customers, product capabilities, case studies, documentation, analyst writing, "
    "reputable news, and primary sources.\n"
    "- Mark uncertainty when evidence is thin, stale, contradictory, or dominated by vendor claims.\n"
    "- Do not invent citations or cite sources you did not save.\n"
    "\n"
    "The final report should include an executive summary, scope and method, key findings, evidence notes, risks or "
    "counterarguments, recommended next steps, and a source list. Keep it credible but readable, roughly 1,200-1,800 words "
    "for the first public trace.\n"
    "\n"
    "When finished, submit the structured receipt with completed, report_path, source_count, citation_issues, open_questions, "
    "and next_action."
)


CITATION_CRITIC_PROMPT = (
    "You are a citation and evidence critic.\n"
    "\n"
    "You do not perform new web research. You review the draft report against the saved source-note batch and source "
    "documents provided by the parent agent.\n"
    "\n"
    "Your job is to identify evidence problems that should be fixed before publication:\n"
    "- Important claims that lack a citation.\n"
    "- Citations that do not appear to support the claim.\n"
    "- Claims that rely only on vendor-authored sources when independent support is needed.\n"
    "- Overstated conclusions relative to the evidence.\n"
    "- Missing counterevidence, uncertainty, or caveats.\n"
    "- Source quality issues such as stale content, unclear authorship, weak excerpts, or irrelevant pages.\n"
    "\n"
    "Be specific and actionable. Reference the draft section, claim, source title or URL, and the recommended fix. "
    "If the draft is mostly sound, say so and list only the highest-value improvements.\n"
    "\n"
    "Return a markdown review with these sections:\n"
    "- Blocking issues: severe support or citation problems.\n"
    "- Advisory issues: useful improvements that do not block the report.\n"
    "- Strong areas: claims or sections that are well-supported.\n"
    "- Suggested revision plan: concise steps for the parent agent."
)


def build_harness(root: Path, *, model: str = DEFAULT_MODEL) -> Harness:
    exa_tools = ExaTools(root)
    output_mode = "tool" if model.startswith("openrouter:") else "native"
    parallel_tool = ParallelLlmTool(
        name="parallel_llm",
        description=(
            "Run independent source-note extraction prompts in parallel. "
            "Use source={kind:'file', path:'outputs/source_note_prompts.json'} and output_file='outputs/source_notes_batch.json'."
        ),
        instructions=(
            "Return only the requested source note. Do not use facts outside the prompt. "
            "Prefer concise, concrete evidence and preserve exact names, numbers, and dates."
        ),
        model=model,
        root=root,
        read_paths=["outputs"],
        write_paths=["outputs"],
        max_prompts=20,
        max_attempts=4,
        request_timeout=240,
        temperature=0,
        extra_body=_model_extra_body(model),
        output_type=SourceNote,
        output_mode="tool" if model.startswith("openrouter:") else "native",
        output_retries=2,
    ).spec()
    return Harness(
        HarnessConfig(
            root=root,
            model=model,
            system_prompt=SYSTEM_PROMPT,
            builtin_tools=["read", "write", "search", "list", "glob", "jsonl_search", "subagent"],
            output_type=ReportReceipt,
            output_mode=output_mode,
            output_retries=2,
            tool_retries=2,
            max_model_requests=64,
            max_tool_calls=96,
            local_trace_dir=root / "outputs" / "traces",
            read_paths=["outputs"],
            write_paths=["outputs"],
            max_read_chars=80_000,
            max_tool_chars=80_000,
            tool_execution="sequential",
            request_timeout=240,
            temperature=0,
            extra_body=_model_extra_body(model),
            subagents=[
                SubAgentConfig(
                    name="citation_critic",
                    description="Citation and evidence critic for saved draft reports.",
                    system_prompt=CITATION_CRITIC_PROMPT,
                    builtin_tools=["read", "search"],
                    max_model_requests=10,
                    max_tool_calls=20,
                    output_retries=1,
                    tool_retries=1,
                )
            ],
        ),
        tools=[*exa_tools.specs(), parallel_tool],
        hooks=[Hook("after_tool_call", _source_audit_hook)],
    )


def run_report(prompt: str = DEFAULT_PROMPT, *, root: Path = EXAMPLE_ROOT, model: str = DEFAULT_MODEL, validate: bool = True) -> dict[str, Any]:
    root = root.resolve()
    (root / "outputs").mkdir(parents=True, exist_ok=True)
    before_trace_files = set((root / "outputs" / "traces").glob("**/*.jsonl"))
    harness = build_harness(root, model=model)
    result = harness.run_sync(prompt, metadata={"example": "web_research_report", "conversation_id": "web_research_report"})
    trace_files = sorted(set((root / "outputs" / "traces").glob("**/*.jsonl")) - before_trace_files)
    validation = validate_outputs(root, result, trace_files) if validate else {"validated": False}
    payload = {
        "model": model,
        "stop_reason": result.stop_reason,
        "usage": {
            "model_requests": result.usage.model_requests,
            "tool_calls": result.usage.tool_calls,
            "cancelled_tool_calls": result.usage.cancelled_tool_calls,
        },
        "tools": [record["call"]["name"] for record in result.tool_call_records if "call" in record],
        "trace_files": [_rel(path, root) for path in trace_files],
        "receipt": result.output.model_dump() if isinstance(result.output, ReportReceipt) else None,
        "validation": validation,
    }
    _write_json(root / "outputs" / "run_summary.json", payload)
    return payload


def validate_outputs(root: Path, result: Any, trace_files: list[Path]) -> dict[str, Any]:
    if result.stop_reason != "end_turn":
        raise AssertionError(f"run stopped with {result.stop_reason}")
    if not isinstance(result.output, ReportReceipt):
        raise AssertionError(f"run did not return ReportReceipt: {result.output!r}")
    if not result.output.completed:
        raise AssertionError("receipt completed=false")

    required_files = [
        "outputs/research_plan.md",
        "outputs/source_selection.md",
        "outputs/source_note_prompts.json",
        "outputs/source_notes_batch.json",
        "outputs/source_notes_compact.md",
        "outputs/draft_report.md",
        "outputs/market_landscape_report.md",
    ]
    missing = [path for path in required_files if not (root / path).exists()]
    if missing:
        raise AssertionError(f"missing required artifacts: {missing}")

    search_results = sorted((root / "outputs" / "sources").glob("search_*/results.jsonl"))
    fetch_results = sorted((root / "outputs" / "sources").glob("fetch_*/documents.jsonl"))
    if not search_results:
        raise AssertionError("no Exa search results were written")
    if not fetch_results:
        raise AssertionError("no Exa fetch documents were written")

    source_notes = json.loads((root / "outputs" / "source_notes_batch.json").read_text(encoding="utf-8"))
    if source_notes.get("succeeded", 0) <= 0 or source_notes.get("failed", 0) != 0:
        raise AssertionError(f"source note batch did not fully succeed: {source_notes.get('succeeded')} succeeded, {source_notes.get('failed')} failed")

    report = (root / "outputs" / "market_landscape_report.md").read_text(encoding="utf-8")
    if "http://" not in report and "https://" not in report:
        raise AssertionError("final report does not include source URLs")
    if len(report.split()) < 700:
        raise AssertionError("final report looks too short for the requested landscape brief")

    tool_names = [record["call"]["name"] for record in result.tool_call_records if "call" in record]
    required_tools = {"exa_search_sources", "exa_fetch_sources", "parallel_llm", "subagent"}
    missing_tools = sorted(required_tools - set(tool_names))
    if missing_tools:
        raise AssertionError(f"required tools were not called: {missing_tools}")
    if not trace_files:
        raise AssertionError("no trace files were produced")
    trace_text = "\n".join(path.read_text(encoding="utf-8") for path in trace_files)
    prompts = json.loads((root / "outputs" / "source_note_prompts.json").read_text(encoding="utf-8"))
    if not isinstance(prompts, list) or not prompts or not all(isinstance(prompt, str) for prompt in prompts):
        raise AssertionError("source_note_prompts.json is not a non-empty JSON array of strings")
    if not all("Source record:" in prompt and '"text"' in prompt for prompt in prompts):
        raise AssertionError("source-note prompts do not include full source records with text")

    for expected in [
        "invoke_agent",
        "execute_tool exa_search_sources",
        "execute_tool prepare_source_note_prompts",
        "execute_tool compact_source_notes",
        "execute_tool parallel_llm",
        "execute_tool subagent",
    ]:
        if expected not in trace_text:
            raise AssertionError(f"trace missing {expected!r}")
    return {
        "validated": True,
        "search_runs": len(search_results),
        "fetch_runs": len(fetch_results),
        "source_notes_succeeded": source_notes["succeeded"],
        "report_path": "outputs/market_landscape_report.md",
        "trace_files": [_rel(path, root) for path in trace_files],
    }


def _source_audit_hook(ctx: Any) -> None:
    if ctx.tool_name not in {"exa_search_sources", "exa_fetch_sources"}:
        return
    parsed = _parse_tool_output(getattr(ctx, "output", None))
    if not parsed:
        return
    content = parsed.get("content")
    if isinstance(content, str):
        try:
            content = json.loads(content)
        except json.JSONDecodeError:
            content = {"raw": content}
    status_counts = content.get("status_counts") if isinstance(content, dict) else None
    audit_row = {
        "created_at": _now(),
        "tool": ctx.tool_name,
        "ok": parsed.get("ok"),
        "query_count": content.get("query_count") if isinstance(content, dict) else None,
        "url_count": sum(status_counts.values()) if isinstance(status_counts, dict) else None,
        "result_count": content.get("result_count") if isinstance(content, dict) else None,
        "status_counts": status_counts,
        "request_ids": content.get("request_ids") if isinstance(content, dict) else None,
        "request_id": content.get("request_id") if isinstance(content, dict) else None,
        "cost": content.get("cost") if isinstance(content, dict) else None,
        "output_paths": {
            key: value
            for key, value in (content.items() if isinstance(content, dict) else [])
            if key.endswith("_path") or key.endswith("_file")
        },
    }
    path = ctx.harness.root / "outputs" / "source_audit.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(audit_row, ensure_ascii=False, default=str) + "\n")


def _model_extra_body(model: str) -> dict[str, Any]:
    if not model.startswith("openrouter:"):
        return {}
    raw_max_tokens = os.getenv("WEB_RESEARCH_REPORT_MAX_TOKENS", "4096")
    try:
        max_tokens = int(raw_max_tokens)
    except ValueError:
        max_tokens = 4096
    return {"max_tokens": max_tokens}


def _search_payload(args: ExaSearchSourcesArgs, query: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "query": query,
        "numResults": args.num_results_per_query,
        "type": args.search_type,
        "contents": {"highlights": True},
    }
    if args.include_domains:
        payload["includeDomains"] = args.include_domains
    if args.exclude_domains:
        payload["excludeDomains"] = args.exclude_domains
    if args.start_published_date:
        payload["startPublishedDate"] = args.start_published_date
    if args.end_published_date:
        payload["endPublishedDate"] = args.end_published_date
    return payload


def _source_note_prompt(research_brief: str, document: dict[str, Any]) -> str:
    return (
        "Research brief:\n"
        f"{research_brief.strip()}\n\n"
        "Source record:\n"
        f"{json.dumps(document, ensure_ascii=False, indent=2, default=str)}\n\n"
        "Create a compact source note using only this source.\n\n"
        "Return one source note with:\n"
        "- document_id\n"
        "- title\n"
        "- url\n"
        "- source_type: vendor_page | news | analyst_post | docs | case_study | forum | other\n"
        "- summary: 2-4 sentences\n"
        "- useful_points: list of {point, evidence_excerpt}\n"
        "- numbers_or_dates: list of concrete numbers, dates, customer names, vendor names, or adoption signals\n"
        "- limitations: list of bias, uncertainty, missing context, or reasons not to over-weight this source\n"
        "- citation_worthy: boolean\n\n"
        "Rules:\n"
        "- Do not infer beyond the source text.\n"
        "- Prefer concrete evidence over marketing claims.\n"
        "- Preserve exact numbers and named entities when present.\n"
        "- If the source is vendor-authored, say so in limitations."
    )


def _one_line(value: Any, limit: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return f"{text[: max(limit - 3, 0)]}..."


def _normalize_highlights(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        highlights = []
        for item in value:
            if isinstance(item, str):
                highlights.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("highlight") or item.get("content")
                if text:
                    highlights.append(str(text))
            else:
                highlights.append(str(item))
        return highlights
    return [str(value)]


def _normalize_statuses(value: Any, urls: list[str]) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return [{"url": url, "status": "success"} for url in urls]
    statuses: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if isinstance(item, dict):
            status = dict(item)
            status["url"] = str(status.get("url") or (urls[index] if index < len(urls) else ""))
            status["status"] = str(status.get("status") or "success")
            statuses.append(status)
        elif index < len(urls):
            statuses.append({"url": urls[index], "status": str(item)})
    seen = {status["url"] for status in statuses}
    for url in urls:
        if url not in seen:
            statuses.append({"url": url, "status": "unknown"})
    return statuses


def _source_id_for_position(sources: list[SelectedSource], document_index: int) -> str | None:
    index = document_index - 1
    if index < 0 or index >= len(sources):
        return None
    return sources[index].source_id


def _response_request_id(response: dict[str, Any]) -> str | None:
    for key in ("requestId", "request_id", "requestID"):
        value = response.get(key)
        if value:
            return str(value)
    return None


def _response_cost(response: dict[str, Any]) -> float | None:
    for key in ("costDollars", "cost", "cost_dollars"):
        value = response.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    return None


def _safe_error(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}"
    api_key = os.getenv("EXA_API_KEY")
    if api_key:
        text = text.replace(api_key, "[redacted]")
    return text


def _parse_tool_output(output: Any) -> dict[str, Any] | None:
    if isinstance(output, str):
        try:
            parsed = json.loads(output)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    if isinstance(output, dict):
        return output
    return None


def _write_json(path: Path, value: Any) -> None:
    _atomic_write(path, json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    _atomic_write(path, "".join(json.dumps(row, ensure_ascii=False, default=str) + "\n" for row in rows))


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False)
    tmp_path = Path(tmp.name)
    try:
        with tmp:
            tmp.write(text)
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _rel(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the web research report example agent.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--root", type=Path, default=EXAMPLE_ROOT)
    parser.add_argument("--no-validate", action="store_true")
    args = parser.parse_args(argv)

    provider = args.model.split(":", 1)[0]
    provider_env = {"openrouter": "OPENROUTER_API_KEY", "openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}.get(provider)
    missing = [name for name in [provider_env, "EXA_API_KEY"] if name and not os.getenv(name)]
    if missing:
        raise SystemExit(f"{', '.join(missing)} required; run with uv run --env-file .env python examples/web_research_report/agent.py")

    payload = run_report(args.prompt, root=args.root, model=args.model, validate=not args.no_validate)
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    sys.path.insert(0, str(REPO_ROOT))
    main()
