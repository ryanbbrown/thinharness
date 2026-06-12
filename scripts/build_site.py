from __future__ import annotations

import argparse
import difflib
import html
import re
from html.parser import HTMLParser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
README = REPO_ROOT / "README.md"
SITE_DIR = REPO_ROOT / "docs" / "site"
ABOUT = SITE_DIR / "about.html"
GITHUB_ROOT = "https://github.com/ryanbbrown/thinharness"


def section(markdown: str, title: str) -> str:
    pattern = re.compile(rf"^## {re.escape(title)}\n(?P<body>.*?)(?=^## |\Z)", re.M | re.S)
    match = pattern.search(markdown)
    if not match:
        raise ValueError(f"README section not found: {title}")
    return match.group("body").strip()


def inline_markdown(text: str) -> str:
    placeholders: list[str] = []

    def hold(value: str) -> str:
        placeholders.append(value)
        return f"\0{len(placeholders) - 1}\0"

    text = re.sub(r"`([^`]+)`", lambda m: hold(f"<code>{html.escape(m.group(1))}</code>"), text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", lambda m: m.group(1), text)
    escaped = html.escape(text, quote=False)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)
    for index, value in enumerate(placeholders):
        escaped = escaped.replace(f"\0{index}\0", value)
    return escaped


def paragraphs(markdown: str) -> list[str]:
    return [block.replace("\n", " ") for block in markdown.split("\n\n") if block.strip()]


def slug_for_opinion(title: str) -> str:
    known_tags = {
        "Purpose-built agents, not universal agents": "purpose_built",
        "No bash by default": "no_bash",
        "Skills are tools, not auto-discovery": "skills",
        "Search is a top priority": "search",
        "Parallel LLM calls, built in": "parallel_llm",
        "Background tools are simple": "background_tools",
        "Three providers, no matrix": "providers",
        "No compaction": "no_compaction",
        "No deployment layer": "no_deployment",
    }
    if title in known_tags:
        return known_tags[title]
    return re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_") or "opinion"


def opinions_from_readme(markdown: str) -> list[tuple[str, str, str]]:
    body = section(markdown, "Opinions")
    result: list[tuple[str, str, str]] = []
    for match in re.finditer(r"\*\*([^*]+?)\.\*\* ([\s\S]*?)(?=\n\n\*\*|\Z)", body):
        title = match.group(1)
        text = match.group(2).strip().replace("\n", " ")
        if title == "Three providers, no matrix" and not text.endswith("."):
            text += "."
        result.append((slug_for_opinion(title), title, inline_markdown(text)))
    return result


def features_from_readme(markdown: str) -> list[tuple[str, str]]:
    body = section(markdown, "Features")
    result: list[tuple[str, str]] = []
    for match in re.finditer(r"^- \*\*([^*]+):\*\* (.+)$", body, re.M):
        title, text = match.groups()
        if title == "Limit notices":
            text = text.replace(
                "Notices are harness-owned model input, not hooks or callbacks; parent and child runs compute them from their own local budgets.",
                "Harness-owned model input — parent and child runs compute them from their own local budgets.",
            )
        result.append((title, sentence_case(inline_markdown(text))))
    return result


def sentence_case(text: str) -> str:
    if not text:
        return text
    return text[0].upper() + text[1:]


class TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[dict[str, str]]] = []
        self._row: list[dict[str, str]] | None = None
        self._cell: dict[str, str] | None = None
        self._text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "tr":
            self._row = []
        elif tag in {"td", "th"} and self._row is not None:
            self._cell = {"text": "", "img": ""}
            self._text = []
        elif tag == "img" and self._cell is not None:
            attrs_dict = dict(attrs)
            self._cell["img"] = attrs_dict.get("src") or ""
        elif tag == "br" and self._cell is not None:
            self._text.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self._cell is not None and self._row is not None:
            self._cell["text"] = normalize_table_text("".join(self._text))
            self._row.append(self._cell)
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if self._row:
                self.rows.append(self._row)
            self._row = None

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._text.append(data)


def normalize_table_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()


def readme_table_rows(markdown: str) -> list[list[dict[str, str]]]:
    table_match = re.search(r"<table>.*?</table>", markdown, re.S)
    if not table_match:
        raise ValueError("README comparison table not found")
    parser = TableParser()
    parser.feed(table_match.group(0))
    return parser.rows


def mark(value: str) -> str:
    return {
        "✅": '<i class="mark y">●</i>',
        "❌": '<i class="mark n">○</i>',
        "⚠️": '<i class="mark p">◑</i>',
    }[value]


def library_cell(cell: dict[str, str]) -> str:
    name = cell["text"]
    superscript = ""
    match = re.search(r"(\d+)$", name)
    if match and name not in {"3"}:
        name = name[: -len(match.group(1))].strip()
        superscript = f"<sup>{match.group(1)}</sup>"
    display_name = html.escape(name).replace("Claude Agent SDK", "Claude Agent SDK").replace("OpenAI Agents SDK", "OpenAI Agents SDK")
    if name == "ThinHarness":
        return '<div class="lib-cell"><img class="thinharness-table-logo" src="assets/ThinHarness.svg" alt="">ThinHarness</div>'
    if name == "Agno":
        return '<div class="lib-cell">Agno</div>'
    img = cell["img"]
    return f'<div class="lib-cell"><img src="{html.escape(img, quote=True)}" alt="">{display_name}{superscript}</div>'


def comparison_table(markdown: str) -> str:
    rows = readme_table_rows(markdown)
    body_rows = []
    for row in rows[1:]:
        library = row[0]
        css = ' class="me"' if library["text"] == "ThinHarness" else ""
        marks = "".join(f"<td>{mark(cell['text'])}</td>" for cell in row[2:])
        body_rows.append(
            f"""              <tr{css}>
                <td class="lib">{library_cell(library)}</td>
                <td class="loc-n">{html.escape(row[1]["text"])}</td>
                {marks}
              </tr>"""
        )
    return "\n".join(body_rows)


def code_highlight(code: str) -> str:
    escaped = html.escape(code)
    escaped = re.sub(r"\b(import|from|async def|async with|as|await)\b", r'<span class="k">\1</span>', escaped)
    escaped = re.sub(r'(&quot;[^&]+?&quot;)', r'<span class="s">\1</span>', escaped)
    return escaped.replace("&quot;", '"')


def fenced_code(markdown: str, language: str) -> str:
    match = re.search(rf"```{language}\n(.*?)\n```", markdown, re.S)
    if not match:
        raise ValueError(f"{language} code block not found")
    return match.group(1)


def render_about(markdown: str) -> str:
    why = paragraphs(section(markdown, "Why this exists").split("<!--", 1)[0].strip())
    install = section(markdown, "Install")
    install_line = fenced_code(install, "bash")
    install_command, install_comment = [part.strip() for part in install_line.split("#", 1)]
    use = section(markdown, "Use")
    use_code = code_highlight(fenced_code(use, "python"))
    use_paragraph = paragraphs(use.split("```", 2)[2].strip())[0]
    tracing = paragraphs(section(markdown, "Tracing"))

    opinion_items = "\n".join(
        f'          <div class="op-item"><div class="tag">{tag}</div><h3>{html.escape(title)}</h3><p>{body}</p></div>'
        for tag, title, body in opinions_from_readme(markdown)
    )
    feature_items = "\n".join(
        f'          <div class="f"><div class="ft">{html.escape(title)}</div><p>{body}</p></div>' for title, body in features_from_readme(markdown)
    )
    comparison_scope_note = (
        "Framework-only LOC. Each row strips non-framework code (platform/deployment, voice/realtime, eval suites, "
        "UI/CLI, wire protocols) from the upstream package. Provider implementations stay in."
    )
    table_summary = "Table focuses on features that differentiate the harnesses. All listed also support MCP, lifecycle hooks, and multi-turn conversations."
    retry_footnote = (
        'Tool retries: a documented primitive (e.g. Pydantic AI\'s <code>ModelRetry</code>) that lets tools signal "model passed '
        'bad args — retry with this feedback," distinct from generic exception propagation.'
    )
    install_line_html = (
        f'<div class="install-line"><span><span class="p">$</span> {html.escape(install_command)}</span>'
        f'<span class="c">&nbsp;&nbsp;# {html.escape(install_comment)}</span></div>'
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>About — ThinHarness</title>
<link rel="stylesheet" href="assets/site.css">
<link rel="icon" type="image/svg+xml" href="assets/favicon.svg">
<link rel="apple-touch-icon" href="assets/apple-touch-icon.png">
</head>
<body class="th page-about">
  <header class="site-header">
    <div class="site-header__inner">
      <a class="site-logo" href="index.html"><img src="assets/ThinHarness.svg" alt="ThinHarness"></a>
      <nav class="site-nav">
        <a href="index.html">~/home</a>
        <a href="about.html" class="is-active">~/about</a>
        <a href="explainer.html">~/explainer</a>
        <a href="examples.html">~/examples</a>
        <a href="{GITHUB_ROOT}" class="ext">github&nbsp;↗</a>
      </nav>
    </div>
  </header>

  <div class="wrap">
    <div class="doc">
      <div class="doc-hero">
        <img class="bigmark" src="assets/ThinHarness.svg" alt="ThinHarness">
        <p class="tag">A minimal, opinionated agent harness — <b>focused scope, readable core, easy to fork.</b></p>
        <div class="badges">
          <a class="ci" href="{GITHUB_ROOT}/actions/workflows/ci.yml"><span class="b-dot"></span>CI · passing</a>
          <a class="lic" href="{GITHUB_ROOT}/blob/main/LICENSE"><span class="b-dot"></span>License · MIT</a>
          <a class="pypi" href="https://pypi.org/project/thinharness/"><span class="b-dot"></span>PyPI · thinharness</a>
        </div>
      </div>

      <nav class="toc">
        <a href="#why">why</a>
        <a href="#comparison">comparison</a>
        <a href="#opinions">opinions</a>
        <a href="#install">install</a>
        <a href="#use">use</a>
        <a href="#features">features</a>
        <a href="#tracing">tracing</a>
        <a href="#status">status</a>
        <a href="#license">license</a>
      </nav>

      <section id="why">
        <div class="eyebrow">// why this exists</div>
        <h2>Why this exists</h2>
        <p>{inline_markdown(why[0])}</p>
        <p>{inline_markdown(why[1])}</p>
      </section>

      <section id="comparison">
        <div class="eyebrow">// loc comparison</div>
        <h2>How small, exactly</h2>
        <p class="table-note">{comparison_scope_note}</p>
        <div class="legend">
          <span><i class="mark y">●</i> yes</span>
          <span><i class="mark p">◑</i> partial</span>
          <span><i class="mark n">○</i> no</span>
        </div>
        <div class="table-wrap">
          <table class="loc">
            <thead>
              <tr>
                <th class="lib">Library</th>
                <th>LOC<sup>1</sup></th>
                <th>Tool<br>retries<sup>2</sup></th>
                <th>Sub-<br>agents</th>
                <th>Structured<br>output</th>
                <th>Skills</th>
                <th>FS<br>tools</th>
                <th>OTel<br>tracing</th>
              </tr>
            </thead>
            <tbody>
{comparison_table(markdown)}
            </tbody>
          </table>
        </div>
        <p class="table-note table-note-summary">{table_summary}</p>
        <div class="footnotes">
          <p><strong>1.</strong> LOC excludes anything that is not the core agent harness framework. See raw README source comments for exact commands.</p>
          <p><strong>2.</strong> {retry_footnote}</p>
          <p><strong>3.</strong> Claude Agent SDK shells out to the Claude Code CLI binary, which is 200k+ LOC.</p>
          <p><strong>4.</strong> deepagents is a thin wrapper over LangChain/LangGraph; effective import surface is ≈105k LOC.</p>
        </div>
        <p class="source-note"><span class="source-link">See <code>docs/table.md</code> for per-cell rationale and how the LOC numbers are measured.</span></p>
      </section>

      <section id="opinions">
        <div class="eyebrow">// opinions</div>
        <h2>Opinions</h2>
        <p>ThinHarness has opinions. They are the reason it stays small.</p>
        <div class="op-list">
{opinion_items}
        </div>
      </section>

      <section id="install">
        <div class="eyebrow">// install</div>
        <h2>Install</h2>
        {install_line_html}
        <p class="req">Requires Python 3.11+.</p>
      </section>

      <section id="use">
        <div class="eyebrow">// use</div>
        <h2>Use</h2>
        <pre><code>{use_code}</code></pre>
        <p>{inline_markdown(use_paragraph)}</p>
      </section>

      <section id="features">
        <div class="eyebrow">// features</div>
        <h2>Features</h2>
        <div class="feat">
{feature_items}
        </div>
      </section>

      <section id="tracing">
        <div class="eyebrow">// tracing</div>
        <h2>Tracing</h2>
        <p>{inline_markdown(tracing[0])}</p>
        <div class="callout"><p>{inline_markdown(tracing[1])}</p></div>
      </section>

      <section id="status">
        <div class="eyebrow">// status</div>
        <h2>Status</h2>
        <p>{inline_markdown(paragraphs(section(markdown, "Status"))[0])}</p>
      </section>

      <section id="license">
        <div class="eyebrow">// license</div>
        <h2>License</h2>
        <p>MIT. See <code>LICENSE</code>.</p>
      </section>
    </div>
  </div>

  <footer class="site-footer">
    <div class="site-footer__inner">
      <div class="mono-mark"><span class="dot"></span>thinharness · MIT · pre-1.0 · python 3.11+</div>
      <div class="links">
        <a href="{GITHUB_ROOT}">github</a>
        <a href="https://pypi.org/project/thinharness/">pypi</a>
        <a href="index.html">home</a>
      </div>
    </div>
  </footer>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ThinHarness docs pages from repository sources.")
    parser.add_argument("--check", action="store_true", help="Fail if generated output differs from docs/site/about.html without writing it.")
    args = parser.parse_args()

    generated = render_about(README.read_text(encoding="utf-8"))
    if args.check:
        current = ABOUT.read_text(encoding="utf-8")
        if current != generated:
            diff = difflib.unified_diff(
                current.splitlines(),
                generated.splitlines(),
                fromfile=str(ABOUT),
                tofile="generated about.html",
                lineterm="",
            )
            print("\n".join(diff))
            raise SystemExit(1)
        print(f"{ABOUT} is up to date")
        return

    ABOUT.write_text(generated, encoding="utf-8")
    print(f"wrote {ABOUT}")


if __name__ == "__main__":
    main()
