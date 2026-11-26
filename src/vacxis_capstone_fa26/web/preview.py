"""Provide explicitly labelled sample interactions for UI preview without AI services."""

import re
from html.parser import HTMLParser
from time import sleep

from ..generation.schemas import GeneratedAnswer
from ..query import ClarifyPlan
from .documents import Documents


class _ExcerptText(HTMLParser):
    """Extract readable text from source tables for sample chat responses."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data):
        self.parts.append(data)


def preview_responder(documents: Documents):
    names = [
        (title.removeprefix("Hướng dẫn sử dụng Vaccine "), doc_id)
        for title, doc_id in documents.choices if doc_id in documents.indexed_ids
    ]

    def respond(query, *, history=(), clarifications=(), on_progress):
        on_progress("planning", None)
        sleep(0.7)
        current = query + " " + " ".join(turn.answer for turn in clarifications)
        matched = next((doc_id for name, doc_id in names if name.casefold() in current.casefold()), None)
        if matched is None:
            matched = next((
                doc_id for message in reversed(history) if message.role == "user"
                for name, doc_id in names if name.casefold() in message.content.casefold()
            ), None)
        if matched is None:
            return ClarifyPlan("Bạn đang hỏi về vắc xin nào?")
        on_progress("searching", None)
        sleep(0.8)
        rows = documents.rows(matched)
        words = set(re.findall(r"\w+", current.casefold())) - {"có", "và", "của", "là", "cho", "xin", "vắc"}
        rows.sort(key=lambda row: len(words & set(re.findall(r"\w+", (row["header"] + " " + row["text"]).casefold()))), reverse=True)
        on_progress("reranking", None)
        sleep(0.8)
        sources = rows[:2]
        on_progress("generating", len(sources))
        sleep(1)
        excerpts = [row["text"].partition("\n")[2] or row["text"] for row in sources]
        answer = "**Phản hồi mẫu để xem trước giao diện.**\n\nCác đoạn tài liệu liên quan để bạn mở và đối chiếu\n\n"
        snippets = []
        for source, text in zip(sources, excerpts, strict=True):
            extractor = _ExcerptText()
            extractor.feed(text)
            extractor.close()
            snippet = " ".join(" ".join(extractor.parts).split())[:350]
            snippet = re.sub(r"([\\`*_{}\[\]()<>#!|])", r"\\\1", snippet)
            snippets.append(f"> {snippet}… [{source['chunk_id']}]")
        answer += "\n\n".join(snippets)
        return GeneratedAnswer(query=query, answer=answer, cited_chunks=sources, model="ui-preview")

    return respond
