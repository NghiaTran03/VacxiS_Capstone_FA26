"""Render the indexed document and navigate to a cited chunk."""

from html import escape, unescape
from html.parser import HTMLParser
from pathlib import Path
import re
from urllib.parse import urlencode

import pyarrow.compute as pc
import pyarrow.parquet as pq

from ..generation.prompts import get_chunk_field
from ..indexing.discovery import discover_documents
from ..indexing.parser import parse_markdown_to_sections
from ..indexing.schemas import classify_category


FIELDS = ("doc_id", "chunk_id", "doc_title", "header", "text")


class _DocumentHTML(HTMLParser):
    """Keep document formatting and HTML tables while excluding executable markup."""

    tags = {
        "p", "br", "hr", "b", "strong", "i", "em", "s", "u", "sub", "sup",
        "h1", "h2", "h3", "h4", "h5", "h6", "ul", "ol", "li", "blockquote",
        "pre", "code", "table", "thead", "tbody", "tfoot", "tr", "th", "td", "caption", "a",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag not in self.tags:
            self.parts.append(escape(self.get_starttag_text()))
            return
        safe_attrs = []
        for key, value in attrs:
            if tag in {"th", "td"} and key in {"colspan", "rowspan"} and value and value.isdecimal():
                safe_attrs.append(f'{key}="{escape(value, quote=True)}"')
            if tag == "a" and key == "href" and value and value.startswith(("#", "https://", "http://")):
                safe_attrs.append(f'href="{escape(value, quote=True)}" rel="noopener noreferrer"')
        attributes = " " + " ".join(safe_attrs) if safe_attrs else ""
        self.parts.append(f"<{tag}{attributes}>")

    def handle_endtag(self, tag):
        if tag in {"br", "hr"}:
            return
        self.parts.append(f"</{tag}>" if tag in self.tags else escape(f"</{tag}>"))

    def handle_data(self, data):
        self.parts.append(escape(data))


def _render_markdown(text: str) -> str:
    from markdown_it import MarkdownIt

    cleaner = _DocumentHTML()
    cleaner.feed(MarkdownIt("commonmark", {"html": True}).enable("table").render(text))
    cleaner.close()
    return "".join(cleaner.parts)


class _DocumentText(HTMLParser):
    """Collect visible text for aligning indexed excerpts with original sections."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data):
        self.parts.append(data)


def _comparison_text(text: str) -> str:
    text = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("[CHỦ ĐỀ:"))
    reader = _DocumentText()
    reader.feed(_render_markdown(text))
    reader.close()
    return "".join("".join(reader.parts).split()).casefold()


def _covered_categories(sections: list[tuple[str, str]], chunk: dict) -> set[str]:
    """Map a unique exact excerpt span to every original category it overlaps."""
    ranges = []
    source = ""
    for category, body in sections:
        start = len(source)
        source += _comparison_text(body)
        ranges.append((category, start, len(source)))
    excerpt = _comparison_text(chunk["text"])
    start = source.find(excerpt) if excerpt else -1
    if start < 0 or source.find(excerpt, start + 1) >= 0:
        return {chunk["category"]}
    end = start + len(excerpt)
    return {category for category, left, right in ranges if left < end and start < right}


def _heading_text(text: str) -> str:
    """Normalize heading wrappers for locating parser headings in original lines."""
    text = unescape(re.sub(r"<[^>]*>", "", text))
    text = re.sub(r"^\s*#{1,6}\s+", "", text)
    text = text.strip().removesuffix(":").strip().strip("*").strip()
    return " ".join(text.removesuffix(":").split())


def _original_sections(text: str) -> list[tuple[str, str]]:
    """Use existing heading classification while preserving every source line."""
    from markdown_it import MarkdownIt

    lines = text.splitlines(keepends=True)
    code_lines = set()
    html_lines = set()
    table_lines = set()
    table_depth = 0
    for index, line in enumerate(lines):
        tags = re.findall(r"</?table\b[^>]*>", line, flags=re.IGNORECASE)
        if table_depth or any(not tag.startswith("</") for tag in tags):
            table_lines.add(index)
        for tag in tags:
            table_depth = max(0, table_depth + (-1 if tag.startswith("</") else 1))
    for token in MarkdownIt("commonmark", {"html": True}).parse(text):
        if token.map and token.type in {"fence", "code_block"}:
            code_lines.update(range(*token.map))
        if token.map and token.type == "html_block":
            html_lines.update(range(*token.map))
    headings = [
        _heading_text(line) if index not in code_lines and index not in table_lines else None
        for index, line in enumerate(lines)
    ]
    boundaries = {0: "thông tin chung"}
    cursor = 0
    for section in parse_markdown_to_sections(text):
        target = _heading_text(section["header"])
        position = next((index for index in range(cursor, len(lines)) if headings[index] == target), None)
        if position is None:
            continue
        category = classify_category(section["header"], section.get("parent_header", ""))
        boundaries[position] = category
        cursor = position + 1
    for position, line in enumerate(lines):
        if position in code_lines or position in html_lines or position in table_lines:
            continue
        inline_heading = re.match(r"^([A-ZÀ-ỸĐ][A-ZÀ-ỸĐ\s]+):", line.strip())
        if inline_heading:
            category = classify_category(inline_heading[1])
            if category != "thông tin chung":
                boundaries.setdefault(position, category)
    ordered = sorted(boundaries.items())
    sections = []
    for index, (start, category) in enumerate(ordered):
        end = ordered[index + 1][0] if index + 1 < len(ordered) else len(lines)
        body = "".join(lines[start:end])
        if body.strip():
            sections.append((category, body))
    return sections


def citation(chunk) -> dict[str, str]:
    return {field: str(get_chunk_field(chunk, field) or "") for field in FIELDS}


class Documents:
    def __init__(self, path: Path, *, project_root: Path | None = None) -> None:
        self.path = path
        self.choices = []
        if path.is_file():
            rows = pq.read_table(path, columns=["doc_id", "doc_title"])
            titles = dict(zip(rows["doc_id"].to_pylist(), rows["doc_title"].to_pylist(), strict=True))
            self.choices = sorted(
                ((title or doc_id, doc_id) for doc_id, title in titles.items()),
                key=lambda choice: choice[0].casefold(),
            )
        self.indexed_ids = {doc_id for _, doc_id in self.choices}
        self.originals = {}
        root = project_root or Path.cwd()
        if (root / "75_vaccine_names_mapping.json").is_file() or (root / "vaccine_name_mapping.json").is_file():
            self.originals = {doc.doc_id: doc for doc in discover_documents(root)}
        titles = {doc_id: title for title, doc_id in self.choices}
        for doc in self.originals.values():
            titles.setdefault(doc.doc_id, doc.doc_title)
        self.choices = sorted(
            ((title, doc_id) for doc_id, title in titles.items()), key=lambda choice: choice[0].casefold(),
        )
        self.ids = set(titles)

    def rows(self, doc_id: str) -> list[dict]:
        if doc_id not in self.ids or not self.path.is_file():
            return []
        columns = list(FIELDS) + [name for name in ("category", "parent_header") if name in pq.read_schema(self.path).names]
        table = pq.read_table(self.path, columns=columns, filters=[("doc_id", "=", doc_id)])
        rows = table.take(pc.sort_indices(table, sort_keys=[("chunk_id", "ascending")])).to_pylist()
        for row in rows:
            row["category"] = row.get("category") or classify_category(row["header"], row.get("parent_header") or "")
        return rows

    def _categories(self, doc_id: str, rows: list[dict], cited: dict | None, used_chunks: list[str]) -> str:
        """Group full source documents and attach an exact citation category marker."""
        groups: dict[str, list[str]] = {}
        original = self.originals.get(doc_id)
        evidence = list(rows)
        used_ids = set(used_chunks)
        if cited:
            used_ids.add(cited["chunk_id"])
        used_categories = set()
        if original:
            related = [doc for doc in self.originals.values() if doc.vaccine_name.casefold() == original.vaccine_name.casefold()]
            for document in related:
                if document.doc_id != doc_id:
                    evidence.extend(self.rows(document.doc_id))
                source_sections = []
                for file in document.md_files:
                    for category, body in _original_sections(file.read_text(encoding="utf-8")):
                        source_sections.append((category, body))
                        groups.setdefault(category, []).append(
                            f'<article><h3 class="source-file">{escape(file.name)}</h3>{_render_markdown(body)}</article>'
                        )
                for row in evidence:
                    if row["doc_id"] == document.doc_id and row["chunk_id"] in used_ids:
                        used_categories.update(_covered_categories(source_sections, row))
        else:
            for row in rows:
                groups.setdefault(row["category"], []).append(
                    f'<article><h3>{escape(row["header"])}</h3><div class="excerpt">{escape(row["text"])}</div></article>'
                )
                if row["chunk_id"] in used_ids:
                    used_categories.add(row["category"])
        for row in evidence:
            if row["chunk_id"] in used_ids and row["category"] not in groups:
                groups[row["category"]] = [f'<div class="excerpt">{escape(row["text"])}</div>']
        title = original.doc_title if original else next(title for title, identifier in self.choices if identifier == doc_id)
        categories = []
        for category, bodies in groups.items():
            used = category in used_categories
            marker = '<span class="used-marker" aria-label="Được dùng trong câu trả lời">✓</span>' if used else ""
            opened = " open" if used else ""
            categories.append(
                f'<details class="category" data-category="{escape(category, quote=True)}"{opened}>'
                f'<summary>{escape(category.capitalize())}{marker}</summary>'
                f'<div class="category-body">{"".join(bodies)}</div></details>'
            )
        return (
            f'<details class="vaccine" open><summary>{escape(title)}</summary>'
            f'{"".join(categories)}</details>'
        )

    def render(self, doc_id: str | None, source: dict[str, str] | None = None) -> str:
        rows = []
        if doc_id in self.ids and self.path.is_file():
            rows = self.rows(doc_id)
        if source and not any(
            row["chunk_id"] == source["chunk_id"] and row["text"] == source["text"] for row in rows
        ):
            rows = [source]
        if not rows:
            return "<p>Chọn một tài liệu hoặc bấm vào nguồn trong câu trả lời.</p>"

        title = escape(rows[0]["doc_title"] or doc_id or "Tài liệu vắc xin")
        sections = [f"<h2>{title}</h2>"]
        for row in rows:
            target = ' data-cited-chunk tabindex="-1"' if source and row["chunk_id"] == source["chunk_id"] else ""
            header = escape(row["header"] or row["chunk_id"])
            body = escape(row["text"] or "")
            sections.append(f"<section{target}><h3>{header}</h3><div>{body}</div></section>")
        return "<article class='vacxis-document'>" + "".join(sections) + "</article>"

    def page(self, doc_id: str | None, chunk_id: str | None, used_chunks: list[str] | None = None) -> str:
        """Show an exact citation alongside complete original Markdown documents."""
        catalogue = []
        for title, document_id in self.choices:
            url = "/documents?" + urlencode({"doc": document_id})
            selected = ' aria-current="page"' if document_id == doc_id else ""
            catalogue.append(f'<a href="{escape(url, quote=True)}"{selected}>{escape(title)}</a>')
        content = '<h1>Tài liệu vắc xin</h1><p>Chọn một tài liệu trong danh sách để đọc.</p>'
        if doc_id in self.ids:
            rows = self.rows(doc_id)
            cited = next((row for row in rows if row["chunk_id"] == chunk_id), None)
            content = ""
            if cited:
                content += (
                    '<section id="citation" class="citation" tabindex="-1"><small>ĐOẠN ĐƯỢC TRÍCH DẪN</small>'
                    f'<h2>{escape(cited["header"] or cited["doc_title"])}</h2>'
                    f'<div class="excerpt">{escape(cited["text"])}</div></section>'
                )
            if doc_id not in self.originals:
                content += '<p>Chưa có tệp nguồn tương ứng. Nội dung đã lập chỉ mục hiển thị bên dưới.</p>'
            content += (
                '<div class="document-search" role="search" aria-label="Tìm trong toàn bộ tài liệu vắc xin">'
                '<label for="document-search">Tìm trong tài liệu</label><div class="search-controls">'
                '<input id="document-search" type="search" placeholder="Nhập từ khóa…" autocomplete="off">'
                '<button id="search-previous" type="button" disabled aria-label="Kết quả trước">↑</button>'
                '<button id="search-next" type="button" disabled aria-label="Kết quả tiếp theo">↓</button>'
                '</div><span id="search-status" role="status" aria-live="polite"></span></div>'
                f'<div id="vaccine-content">{self._categories(doc_id, rows, cited, used_chunks or [])}</div>'
            )
        css = Path(__file__).with_name("documents.css").read_text(encoding="utf-8")
        script = Path(__file__).with_name("documents.js").read_text(encoding="utf-8")
        return (
            '<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
            '<title>Tài liệu vắc xin · VacxiS</title><style>'
            f'{css}'
            '</style><header><div><strong>VacxiS</strong><span>Thư viện tài liệu vắc xin</span></div>'
            '<label class="theme-control" for="document-theme">Giao diện'
            '<select id="document-theme"><option value="system">Theo hệ thống</option>'
            '<option value="light">Sáng</option><option value="dark">Tối</option></select></label></header>'
            f'<div class="layout"><nav aria-label="Danh sách tài liệu">{"".join(catalogue)}</nav><main>{content}</main></div>'
            f'<script>{script}</script></html>'
        )


def create_server(blocks, documents: Documents, css: str):
    """Mount Gradio and the public document reader on the same origin."""
    from fastapi import FastAPI, Query
    from fastapi.responses import HTMLResponse
    import gradio as gr

    app = FastAPI()

    @app.get("/documents", response_class=HTMLResponse)
    def document_page(doc: str | None = None, chunk: str | None = None, used: list[str] = Query(default=[])):
        if doc is not None and doc not in documents.ids:
            return HTMLResponse("Không tìm thấy tài liệu.", status_code=404)
        if chunk is not None and not any(row["chunk_id"] == chunk for row in documents.rows(doc)):
            return HTMLResponse("Không tìm thấy đoạn trích dẫn trong tài liệu này.", status_code=404)
        return HTMLResponse(documents.page(doc, chunk, used))

    return gr.mount_gradio_app(app, blocks, path="/", css=css, theme=gr.themes.Monochrome(), footer_links=[])
