"""Shared defaults for thinharness."""

DEFAULT_SYSTEM_PROMPT = """You are a filesystem automation agent working inside the workspace root.

Use search to find relevant text, filenames, and repeated patterns.
Use read to inspect files before editing.
Use edit for targeted replacements and write for creating or replacing files.
Start narrow, broaden only if needed, and use offset/limit when a file is large or you only need a known section.
Prefer batching independent tool calls in one assistant turn. When several reads, searches, listings, or other inspections do not
depend on each other's results, emit them together instead of waiting between calls.
When making several independent, non-overlapping edits to the same file, emit multiple edit calls in the same assistant turn;
only read between edits when a later edit depends on the result of an earlier one.

When finished, respond concisely with what changed and any verification run."""

DEFAULT_READ_DESCRIPTION = "Read a UTF-8 text file with line numbers and optional offset/limit."
DEFAULT_WRITE_DESCRIPTION = "Create, overwrite, or append to a UTF-8 text file under the workspace root."
DEFAULT_EDIT_DESCRIPTION = "Replace exact text in a UTF-8 file. old_string must be unique unless all=true."
DEFAULT_SEARCH_DESCRIPTION = "Search readable workspace files with ripgrep and return compact grouped path/line matches."
DEFAULT_LIST_DESCRIPTION = "List a directory or glob files under the workspace root."
DEFAULT_GLOB_DESCRIPTION = "Find files by glob pattern under the workspace root."
DEFAULT_JSONL_SEARCH_DESCRIPTION = "Search JSONL files by file, directory, or glob path with optional ripgrep prefilter plus structured field/where filtering."

DEFAULT_READ_INSTRUCTIONS = """read usage:
- Use read when you know the file path and need line-numbered context.
- Omit offset and limit to read from the start through the harness character cap.
- Use offset and limit to inspect large files or targeted chunks.
- Use max_chars only when you need a smaller returned excerpt."""

DEFAULT_WRITE_INSTRUCTIONS = """write usage:
- Use write to create or replace an entire file, or append when append=true.
- For targeted changes to an existing file, prefer edit over rewriting the whole file.
- Include path and complete content at the top level of the tool arguments."""

DEFAULT_EDIT_INSTRUCTIONS = """edit usage:
- Use edit for targeted replacements in existing UTF-8 files.
- old_string must match the file text exactly and should include enough surrounding context to be unique.
- If old_string appears multiple times, add more context or set all=true only when every occurrence should change.
- Include path, old_string, and new_string at the top level of the tool arguments."""

DEFAULT_SEARCH_INSTRUCTIONS = """search usage:
- Use search to find text or regex matches in readable workspace files.
- Pass query for the text or regex to find.
- Pass path to scope search to a file or directory; omit path to search all readable roots.
- Use file_type only for ripgrep file types such as py, js, or md."""

DEFAULT_LIST_INSTRUCTIONS = """list usage:
- Use list to inspect a known directory or file path.
- Use glob to filter entries inside that path.
- Set recursive=true only when you need nested entries."""

DEFAULT_GLOB_INSTRUCTIONS = """glob usage:
- Use glob to discover files by filename pattern before reading or searching them.
- Pass path for the directory to search within and pattern for the glob, such as **/*.py.
- Use include_dirs=true only when directory matches matter."""

DEFAULT_JSONL_SEARCH_INSTRUCTIONS = """jsonl_search usage:
- Use jsonl_search for saved JSONL files when you need structured rows or selected fields.
- Pass path to scope to a JSONL file, a directory of JSONL files, or a glob; omit path to search all readable JSONL files.
- Pass query for an optional ripgrep prefilter before field and where filtering.
- Use fields to return only the keys needed for the next decision."""

DEFAULT_PARALLEL_LLM_DESCRIPTION = (
    "Run N independent prompts as one-shot LLM completions in parallel. Each call is stateless: no tools, no memory, no continuation "
    "- only the model's text response is returned. Use this when you have a batch of independent prompts (classify, summarize, translate). "
    "Choose one prompt source with source.kind: inline uses source.prompts, file uses source.path. For multi-step work, use the "
    "subagent tool instead. For large batches, pass output_file and read it back rather than receiving full results inline. If you need the parent "
    "harness system prompt, include the relevant instructions in system; it is not inherited automatically. The tool's model is host-configured "
    "and cannot be changed by tool arguments. For large independent batches, `_background: true` is available when it lets other work continue."
)
DEFAULT_PARALLEL_LLM_INSTRUCTIONS = """parallel_llm usage:
- Use only for independent one-shot prompts, not multi-step work.
- It does not inherit the parent system prompt; put shared instructions in system when needed.
- For inline prompts, pass source={"kind":"inline","prompts":[...]}.
- For a prompt file, pass source={"kind":"file","path":"prompts.json"}.
- Do not add unused prompt source fields or placeholder values.
- For large or structured batches, set output_file and read that file afterward.
- For large independent batches, background mode is available; default to synchronous unless it is clearly useful."""
