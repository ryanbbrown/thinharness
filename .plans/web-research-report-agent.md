# Web Research Report Agent Plan

## Decision

Build one general `web_research_report` example agent. The agent is not market-specific; the run prompt supplies the topic. The first trace should use a market landscape task because it is realistic, web-heavy, and not finance-oriented:

```text
Prepare a market landscape brief for a product strategy team evaluating AI-powered support quality monitoring tools for mid-market B2B SaaS companies. Identify buyer pain, vendor categories, adoption signals, risks, and recommended next research steps.
```

The main product is a markdown report. The main agent's final structured output, if used, is only a small status receipt. Structured output is still useful inside `parallel_llm` for validated source-note extraction.

## Architecture

Use the main agent for orchestration. Do not split research tracks across subagents. ThinHarness subagents are blocking today, and the public transcript should stay readable. The only predefined subagent is `citation_critic`, which runs after the draft and provides advisory review.

Use batched Exa search. The reference repos all do this in some form: `open_deep_research` passes multiple queries into its Tavily search tool, `dzhng-deep-research` generates a breadth of queries and runs them concurrently, and `gpt-researcher` generates subqueries and gathers context concurrently. ThinHarness should expose the same pattern as a single tool call that accepts multiple queries and writes one coherent result set.

Use `parallel_llm` only after full documents are fetched. It should not plan research or search the web. Configure the example's `parallel_llm` tool with a `SourceNote` structured output schema and keep the model-facing name `parallel_llm`. It receives one fetched document per prompt plus the research brief, then writes the batch payload to `outputs/source_notes_batch.json`.

## Workflow

1. Write `outputs/research_plan.md`.
   The plan names the report sections/research questions implied by the user request. It also lists the initial Exa query batch. For the market landscape prompt, sections may include buyer pain, vendor landscape, adoption evidence, risks, and next research steps, but those are derived from the prompt rather than hardcoded in the system prompt.

2. Run initial batched search.
   Call `exa_search_sources` once with roughly 8-12 queries. Each query should be tied in the plan text to a research question/report section, but the tool does not need a `track_id` argument.

3. Triage search results.
   Read or search `outputs/sources/search_001/results.jsonl`. Choose a bounded set of URLs for full fetch, usually 8-14 for the public trace. Write a short selection note to `outputs/source_selection.md`.

4. Fetch full documents.
   Call `exa_fetch_sources` once with the selected URLs. It writes `documents.jsonl` plus a manifest with Exa statuses. Do not fetch every search result automatically.

5. Extract source notes.
   Create `outputs/source_note_prompts.json`, one prompt string per fetched document. Each prompt includes the research brief, one document record, and the source-note instructions below. Call `parallel_llm` with `source={"kind":"file","path":"outputs/source_note_prompts.json"}` and `output_file="outputs/source_notes_batch.json"`. The batch JSON contains one structured source note per successful result.

6. Optional follow-up search.
   After reading source notes, the agent may run one follow-up batched `exa_search_sources` call with up to 6 queries if important report sections are weak or contradictory. This is not required for every run; it exists to prevent shallow reports when the first batch misses.

7. Draft and critique.
   Write `outputs/draft_report.md`, then call `citation_critic` with the draft path, source notes batch path, and relevant source document paths. The critic is advisory: it should identify citation gaps, weak support, overclaims, vendor bias, and missing counterevidence.

8. Finalize.
   Revise based on useful critic feedback and write `outputs/market_landscape_report.md`.

## Exa Tools

These tools are example-local, not framework-level tools.

### `exa_search_sources`

Arguments:

```python
class ExaSearchSourcesArgs(StrictArgs):
    queries: list[str]  # 1-16 queries
    num_results_per_query: int = 5  # 1-10
    search_type: Literal["auto", "keyword", "neural"] = "auto"
    include_domains: list[str] | None = None
    exclude_domains: list[str] | None = None
    start_published_date: str | None = None
    end_published_date: str | None = None
```

Behavior:

- Calls `POST https://api.exa.ai/search` once per query, concurrently inside the tool.
- Requests highlights only: `contents: {"highlights": true}`.
- Writes `outputs/sources/search_N/manifest.json`.
- Writes `outputs/sources/search_N/results.jsonl`.
- Returns compact JSON with folder path, manifest path, results path, query count, result count, request IDs, cost, and the top title/URL pairs.

`results.jsonl` row shape:

```json
{
  "search_run": "search_001",
  "query_id": "q001",
  "query": "AI customer support QA software mid-market SaaS",
  "rank": 1,
  "source_id": "s001",
  "title": "Example title",
  "url": "https://example.com/page",
  "published_date": "2026-05-01T00:00:00.000Z",
  "author": null,
  "exa_id": "...",
  "highlights": ["..."],
  "score": null
}
```

### `exa_fetch_sources`

Arguments:

```python
class SelectedSource(BaseModel):
    source_id: str
    url: str

class ExaFetchSourcesArgs(StrictArgs):
    sources: list[SelectedSource]  # 1-20 selected search results
    text_max_characters: int = 12000
    highlights_query: str | None = None
```

Behavior:

- Calls `POST https://api.exa.ai/contents`.
- Requests top-level `text: {"maxCharacters": text_max_characters}`.
- Requests top-level `highlights` only when `highlights_query` is provided.
- Writes `outputs/sources/fetch_N/manifest.json`.
- Writes `outputs/sources/fetch_N/documents.jsonl`.
- Carries caller-provided `source_id` through to each fetched document row. `source_id` is generated by `exa_search_sources`, not by Exa.
- Preserves Exa per-URL `statuses`; HTTP success does not imply every URL succeeded.
- Returns compact JSON with folder path, manifest path, documents path, status counts, failed URLs, request ID, and cost.

`documents.jsonl` row shape:

```json
{
  "fetch_run": "fetch_001",
  "document_id": "d001",
  "source_id": "s001",
  "title": "Example title",
  "url": "https://example.com/page",
  "published_date": "2026-05-01T00:00:00.000Z",
  "status": "success",
  "source": "cached",
  "text": "...",
  "highlights": []
}
```

## Source Notes

`parallel_llm` receives a JSON prompts file, one prompt per fetched document. The prompts are created by the main agent. The tool call should use the implemented prompt-source shape:

```json
{
  "source": {"kind": "file", "path": "outputs/source_note_prompts.json"},
  "output_file": "outputs/source_notes_batch.json",
  "max_concurrency": 8
}
```

Configure this example's `parallel_llm` ToolSpec with a structured `SourceNote` output type so each successful batch result contains a validated source-note object.

```python
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
```

Each prompt should use this shape:

```text
Research brief:
<brief from outputs/research_plan.md>

Source record:
<one JSON object from documents.jsonl>

Create a compact source note using only this source.

Return one source note with:
- document_id
- title
- url
- source_type: vendor_page | news | analyst_post | docs | case_study | forum | other
- summary: 2-4 sentences
- useful_points: list of {point, evidence_excerpt}
- numbers_or_dates: list of concrete numbers, dates, customer names, vendor names, or adoption signals
- limitations: list of bias, uncertainty, missing context, or reasons not to over-weight this source
- citation_worthy: boolean

Rules:
- Do not infer beyond the source text.
- Prefer concrete evidence over marketing claims.
- Preserve exact numbers and named entities when present.
- If the source is vendor-authored, say so in limitations.
```

The source-note step should produce `outputs/source_notes_batch.json`. That file is a `parallel_llm` batch payload with `total`, `succeeded`, `failed`, `model_requests`, and `results`. The main agent reads this file before drafting the report and uses only successful source-note results.

## Main Agent System Prompt

```text
You are a web research report agent.

Your job is to produce a concise, source-grounded markdown research report for the user's question. The user's prompt supplies the topic; do not assume a fixed domain or fixed report sections.

Work from saved artifacts, not memory. Search results and fetched web pages should be written to workspace files, then read back before you rely on them. Keep source-backed facts separate from your interpretation.

Required workflow:
1. Write outputs/research_plan.md before searching. The plan should define the report sections or research questions implied by the user request, explain why each matters, and list the initial search queries you intend to run.
2. Use batched Exa search for the initial query set. Prefer 8-12 well-targeted queries for a serious report.
3. Read or search the Exa results JSONL before selecting URLs to fetch.
4. Fetch full contents only for selected high-value URLs. Do not fetch every search result automatically.
5. Use parallel_llm to create one source note per fetched document. Include the research brief and exactly one fetched document in each prompt. Use the `source={"kind":"file","path":"outputs/source_note_prompts.json"}` argument shape and write the batch artifact to outputs/source_notes_batch.json.
6. If source notes reveal major gaps, you may run one follow-up batched search with up to 6 additional queries, then fetch and note any selected follow-up sources.
7. Draft the report, then ask the citation_critic subagent to review the draft against the saved source notes and source documents.
8. Revise useful issues from the critic and write the final markdown report under outputs/.

Evidence standards:
- Cite sources by title and URL.
- Use vendor-authored sources carefully; they can support claims about product positioning, but not broad market conclusions by themselves.
- Prefer concrete numbers, dates, named customers, product capabilities, case studies, documentation, analyst writing, reputable news, and primary sources.
- Mark uncertainty when evidence is thin, stale, contradictory, or dominated by vendor claims.
- Do not invent citations or cite sources you did not save.

Output:
- The final report should be markdown.
- Include an executive summary, scoped findings, evidence notes, risks or counterarguments, recommended next steps, and a source list.
- If structured output is required, return only a compact receipt with completed, report_path, source_count, citation_issues, open_questions, and next_action.
```

## `citation_critic` System Prompt

```text
You are a citation and evidence critic.

You do not perform new web research. You review the draft report against the saved source-note batch and source documents provided by the parent agent.

Your job is to identify evidence problems that should be fixed before publication:
- Important claims that lack a citation.
- Citations that do not appear to support the claim.
- Claims that rely only on vendor-authored sources when independent support is needed.
- Overstated conclusions relative to the evidence.
- Missing counterevidence, uncertainty, or caveats.
- Source quality issues such as stale content, unclear authorship, weak excerpts, or irrelevant pages.

Be specific and actionable. Reference the draft section, claim, source title or URL, and the recommended fix. If the draft is mostly sound, say so and list only the highest-value improvements.

Return a markdown review with these sections:
- Blocking issues: severe support or citation problems.
- Advisory issues: useful improvements that do not block the report.
- Strong areas: claims or sections that are well-supported.
- Suggested revision plan: concise steps for the parent agent.
```

## Final Report Shape

The first trace should write `outputs/market_landscape_report.md` with:

- Executive summary.
- Scope and method.
- Key findings organized by the report sections derived in the research plan.
- Evidence table or evidence notes for the most important claims.
- Risks, counterarguments, and uncertainty.
- Recommended next research steps.
- Sources with title and URL.

The report should be long enough to be credible but short enough for the HTML trace viewer. Target roughly 1,200-1,800 words for the first public trace.

## Hooks

Hooks are optional and audit-only. Source persistence belongs in the Exa tools.

If used, add an after-tool hook for Exa tools that appends `outputs/source_audit.jsonl` with:

- tool name
- query count or URL count
- result count or status counts
- request IDs
- cost
- output paths

Do not use hooks for research policy, source selection, citation validation, or business conclusions.

## Reference Repo Mapping

The selected architecture intentionally borrows only the useful parts of the reference agents:

- From `vendor/open_deep_research`: research brief -> focused research -> compressed notes -> final report, plus a critic/verifier pass. We are not copying the long procedural prompts or `think_tool`.
- From `vendor/dzhng-deep-research`: bounded breadth/depth thinking and concise learnings. We are using one planned initial query batch plus at most one follow-up batch rather than recursive deep research.
- From `vendor/gpt-researcher`: explicit source tracking, source curation, progress/cost logging, and final report discipline. We are not copying its broad retriever/provider abstraction surface.

## Implementation Steps

1. Add example-local Exa argument models and async handlers.
2. Add a fresh `web_research_report` example entrypoint and runner.
3. Configure only the tools this example needs: file tools, batched Exa tools, subagent support, and a structured `ParallelLlmTool(...).spec()` exposed as `parallel_llm`.
4. Add a predefined `citation_critic` subagent with plain markdown output.
5. Use a tiny structured output receipt only if useful; the markdown report remains the product.
6. Run only this example first.
7. Inspect the trace, source artifacts, source notes, critic review, and final report manually.
8. Tune prompt/tool descriptions only if the trace is too procedural, too shallow, or too hard to read.
9. Regenerate the HTML transcript viewer once the trace is good.

## Validation Criteria

The example is ready when:

- Search is batched and writes `results.jsonl`.
- Fetch writes `documents.jsonl` and preserves Exa statuses.
- The agent reads/searches saved JSONL before making source selections.
- `parallel_llm` creates validated source notes from full fetched documents and writes `source_notes_batch.json`.
- The critic subagent reviews the draft against saved evidence.
- The final report cites real saved sources by title and URL.
- The transcript remains readable.
- The system prompt guides behavior without hardcoding the market landscape topic.
- The report is credible enough that it does not look like a toy recipe.
