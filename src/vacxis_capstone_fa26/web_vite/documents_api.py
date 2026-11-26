"""JSON document serialization reusing the core document parser and sanitizer."""

from __future__ import annotations

from typing import Any

from ..web.citations import _excerpt
from ..web.documents import (
    Documents,
    _covered_categories,
    _original_sections,
    _render_markdown,
    citation,
)


def enrich_citation(chunk: Any) -> dict[str, str]:
    """Serialize a cited chunk with a plain-text preview excerpt for tooltips."""
    base = citation(chunk)
    base["excerpt"] = _excerpt(base["text"])
    return base


def list_documents(documents: Documents) -> list[dict[str, Any]]:
    """Return sorted vaccine catalogue metadata for the React document browser."""
    return [
        {
            "doc_id": doc_id,
            "title": title,
            "indexed": doc_id in documents.indexed_ids,
            "has_original": doc_id in documents.originals,
        }
        for title, doc_id in documents.choices
    ]


def get_document_payload(
    documents: Documents,
    doc_id: str,
    chunk_id: str | None = None,
    used_chunks: list[str] | None = None,
) -> dict[str, Any]:
    """Build structured category sections and cited-chunk metadata for a vaccine."""
    if doc_id not in documents.ids:
        raise KeyError("document")

    rows = documents.rows(doc_id)
    cited: dict[str, Any] | None = None
    if chunk_id is not None:
        cited = next((row for row in rows if row["chunk_id"] == chunk_id), None)
        if cited is None:
            raise KeyError("chunk")

    groups: dict[str, list[dict[str, str]]] = {}
    original = documents.originals.get(doc_id)
    evidence = list(rows)
    used_ids = set(used_chunks or [])
    if cited is not None:
        used_ids.add(cited["chunk_id"])
    used_categories: set[str] = set()

    if original is not None:
        related = [
            doc
            for doc in documents.originals.values()
            if doc.vaccine_name.casefold() == original.vaccine_name.casefold()
        ]
        for document in related:
            if document.doc_id != doc_id:
                evidence.extend(documents.rows(document.doc_id))
            source_sections: list[tuple[str, str]] = []
            for file in document.md_files:
                for category, body in _original_sections(file.read_text(encoding="utf-8")):
                    source_sections.append((category, body))
                    groups.setdefault(category, []).append({
                        "source_file": file.name,
                        "html": _render_markdown(body),
                    })
            for row in evidence:
                if row["doc_id"] == document.doc_id and row["chunk_id"] in used_ids:
                    used_categories.update(_covered_categories(source_sections, row))
    else:
        for row in rows:
            groups.setdefault(row["category"], []).append({
                "header": row["header"],
                "text": row["text"],
            })
            if row["chunk_id"] in used_ids:
                used_categories.add(row["category"])

    for row in evidence:
        if row["chunk_id"] in used_ids and row["category"] not in groups:
            groups[row["category"]] = [{"header": row["header"], "text": row["text"]}]

    title = (
        original.doc_title
        if original is not None
        else next(label for label, identifier in documents.choices if identifier == doc_id)
    )
    categories = [
        {
            "name": category,
            "label": category.capitalize(),
            "used": category in used_categories,
            "items": items,
        }
        for category, items in groups.items()
    ]
    cited_payload = (
        {
            "doc_id": cited["doc_id"],
            "chunk_id": cited["chunk_id"],
            "doc_title": cited["doc_title"],
            "header": cited["header"],
            "text": cited["text"],
        }
        if cited is not None
        else None
    )
    return {
        "doc_id": doc_id,
        "title": title,
        "has_original": original is not None,
        "cited": cited_payload,
        "categories": categories,
    }
