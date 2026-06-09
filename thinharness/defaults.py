"""Shared defaults for thinharness."""

DEFAULT_SYSTEM_PROMPT = """You are a filesystem automation agent working inside the workspace root.

Use search to find relevant text, filenames, and repeated patterns.
Use read to inspect bounded file sections before editing.
Use edit for targeted replacements and write for creating or replacing files.
Start narrow, broaden only if needed, and prefer bounded reads over full-file reads.

When finished, respond concisely with what changed and any verification run."""

DEFAULT_READ_DESCRIPTION = "Read a UTF-8 text file with line numbers, offset, and limit."
DEFAULT_WRITE_DESCRIPTION = "Create, overwrite, or append to a UTF-8 text file under the workspace root."
DEFAULT_EDIT_DESCRIPTION = "Replace exact text in a UTF-8 file. old_string must be unique unless all=true."
DEFAULT_SEARCH_DESCRIPTION = "Search readable workspace files with ripgrep and return compact grouped path/line matches."
DEFAULT_LIST_DESCRIPTION = "List a directory or glob files under the workspace root."
DEFAULT_GLOB_DESCRIPTION = "Find files by glob pattern under the workspace root."
DEFAULT_JSONL_SEARCH_DESCRIPTION = "Search JSONL files: optional ripgrep prefilter plus structured field/where filtering. Default scope is **/*.jsonl."

DEFAULT_PARALLEL_LLM_DESCRIPTION = (
    "Run N independent prompts as one-shot LLM completions in parallel. Each call is stateless: no tools, no memory, no continuation "
    "- only the model's text response is returned. Use this when you have a batch of independent prompts (classify, summarize, translate). "
    "Choose one prompt source with source.kind: inline uses source.prompts, file uses source.path. For multi-step work, use the "
    "subagent tool instead. For large batches, pass output_file and read it back rather than receiving full results inline. If you need the parent "
    "harness system prompt, include the relevant instructions in system; it is not inherited automatically. The tool's model is host-configured "
    "and cannot be changed by tool arguments."
)
DEFAULT_PARALLEL_LLM_INSTRUCTIONS = """parallel_llm usage:
- Use only for independent one-shot prompts, not multi-step work.
- It does not inherit the parent system prompt; put shared instructions in system when needed.
- For inline prompts, pass source={"kind":"inline","prompts":[...]}.
- For a prompt file, pass source={"kind":"file","path":"prompts.json"}.
- Do not add unused prompt source fields or placeholder values.
- For large or structured batches, set output_file and read that file afterward."""
