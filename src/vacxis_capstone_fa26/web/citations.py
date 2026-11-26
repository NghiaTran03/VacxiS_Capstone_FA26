"""Render compact source references without altering stored answers or memory."""

from html import escape
from html.parser import HTMLParser
import re
from urllib.parse import urlencode

from markdown_it import MarkdownIt


class _SourceText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)


def _excerpt(text: str) -> str:
    reader = _SourceText()
    reader.feed(MarkdownIt("commonmark", {"html": True}).enable("table").render(text))
    reader.close()
    return " ".join(" ".join(reader.parts).split())


def format_sources(answer: str, sources: list[dict]) -> str:
    """Replace known bracketed chunk IDs with numbered, escaped source chips."""
    indexed = {source["chunk_id"]: source for source in reversed(sources) if source.get("chunk_id")}
    ordered_ids = list(dict.fromkeys(source["chunk_id"] for source in sources if source.get("chunk_id")))
    if not ordered_ids:
        return answer
    used_ids = [cid for cid in ordered_ids if indexed[cid].get("doc_id") not in {"", "lightrag_kg", None}]
    chips = {}
    for number, cid in enumerate(ordered_ids, 1):
        source = indexed[cid]
        title = source.get("doc_title") or "Tài liệu vắc xin"
        preview = "\n".join((title, source.get("header") or "Đoạn trích dẫn", _excerpt(source.get("text") or "")))
        attributes = (
            f'class="vacxis-citation" title="{escape(preview, quote=True)}" '
            f'aria-label="{escape(f"Nguồn {number}: {title}", quote=True)}"'
        )
        if cid in used_ids:
            parameters = {"doc": source["doc_id"], "chunk": cid}
            if len(used_ids) > 1:
                parameters["used"] = used_ids
            href = "/documents?" + urlencode(parameters, doseq=True) + "#citation"
            chips[cid] = f'<a {attributes} href="{escape(href, quote=True)}" target="_blank" rel="noopener noreferrer">{number}</a>'
        else:
            chips[cid] = f'<span {attributes} tabindex="0">{number}</span>'

    id_pattern = re.compile(r"(?<![\w-])(?:" + "|".join(re.escape(cid) for cid in ordered_ids) + r")(?![\w-])")
    referenced = set()

    def replace(match):
        groups = [match.group(1)]
        if match.group(2) is not None:
            groups.append(match.group(2))
        # Two adjacent known groups are citations; other pairs may be reference links.
        if any(not id_pattern.search(group) for group in groups):
            return match.group(0)
        replacements = []
        for group in groups:
            ids = list(dict.fromkeys(id_pattern.findall(group)))
            referenced.update(ids)
            replacements.extend(chips[cid] for cid in ids)
            unknown = id_pattern.sub("", group).strip(" ,;|\t")
            if unknown:
                replacements.append(f"[{unknown}]")
        return " ".join(replacements)

    # Markdown block maps protect fenced and indented code. Inline code stays literal.
    lines = answer.splitlines(keepends=True)
    protected = set()
    for token in MarkdownIt("commonmark").parse(answer):
        if token.type in {"fence", "code_block", "html_block"} and token.map:
            protected.update(range(*token.map))
    rendered = []
    for index, line in enumerate(lines):
        if index in protected or re.match(r"^ {0,3}\[[^\]]+\]:", line):
            rendered.append(line)
            continue
        parts = re.split(r"(`+[^`\n]*`+)", line)
        rendered.append("".join(
            part if offset % 2 else re.sub(r"\[([^\]\n]+)\](?:\[([^\]\n]*)\])?(?!\()", replace, part)
            for offset, part in enumerate(parts)
        ))
    result = "".join(rendered)
    remaining = [chips[cid] for cid in ordered_ids if cid not in referenced]
    if remaining:
        result += "\n\nNguồn: " + " ".join(remaining)
    return result
