"""Render source text and rebuild records for the benchmark."""

import json
import re

from markdown_it import MarkdownIt
from markupsafe import Markup
import nh3


STATUS = {"pending": "Chưa duyệt", "draft": "Đang duyệt", "flagged": "Cần xem lại", "verified": "Đã xác nhận"}
ANSWERABILITY = {
    "ANSWERABLE": "Tài liệu trả lời đầy đủ",
    "PARTIALLY_ANSWERABLE": "Tài liệu trả lời một phần",
    "UNANSWERABLE": "Tài liệu chưa đủ để trả lời",
}
COMPLETENESS = {"ANSWERABLE": "COMPLETE", "PARTIALLY_ANSWERABLE": "PARTIAL", "UNANSWERABLE": "NOT_ASSESSED"}
PROFESSIONS = {"doctor": "Bác sĩ", "nurse": "Điều dưỡng", "other": "Khác"}
MARKDOWN = MarkdownIt("commonmark", {"html": True, "breaks": True}).enable("table")
TAGS = {"p", "br", "hr", "b", "i", "strong", "em", "sup", "sub", "code", "pre", "blockquote", "ul", "ol", "li",
        "h1", "h2", "h3", "h4", "h5", "h6", "table", "caption", "thead", "tbody", "tfoot", "tr", "td", "th"}
ATTRIBUTES = {"td": {"rowspan", "colspan"}, "th": {"rowspan", "colspan", "scope"}, "ol": {"start"}}
SOURCE_HEADING = re.compile(r"^\[CHỦ ĐỀ: (.+?)\] > \[MỤC: (.+?)\]\s*(?:\n|$)")
PAGE_BREAK = re.compile(r"(?im)^([ \t]*\d{1,4}[ \t]*\n(?:[ \t]*\n)*)?---[ \t]*TRANG[ \t]+(\d{1,4})[ \t]*---[ \t]*$")
CITATION = re.compile(r"\[(?:CHUNK_ID(?:\s*:\s*|\s+))?([A-Za-z0-9_-]+_c_\d+)\]", re.IGNORECASE)


def reviewer_label(reviewer):
    if not reviewer["full_name"]:
        return f"Người duyệt {reviewer['id']}"
    title = PROFESSIONS.get(reviewer["profession"]) if reviewer["profession"] != "other" else None
    return f"{title} {reviewer['full_name']}" if title else reviewer["full_name"]


def remove_citations(answer):
    return CITATION.sub("", answer).strip()


def score_three(record):
    return [chunk for chunk in (record.get("RELEVANT_CHUNKS") or []) if chunk.get("RELEVANCE_SCORE") == 3]


def _simple_math(match):
    source = match.group(1)
    for command, glyph in ((r"\le", "≤ "), (r"\ge", "≥ "), (r"\%", "%")):
        source = source.replace(command, glyph)
    return source if re.fullmatch(r"[\dnN\s≤≥=.,%+\-<>]+", source) else match.group(0)


def _page_label(match):
    previous, current = match.groups()
    keep = previous if previous and int(previous.strip()) != int(current) - 1 else ""
    return f"{keep}\n\n**Trang {current}**\n\n"


def render_chunk(chunk):
    raw = chunk["TEXT"]
    body = raw.replace("\r\n", "\n")
    heading = SOURCE_HEADING.match(body)
    subject, section = heading.groups() if heading else ("Đoạn trích", "")
    if heading:
        body = body[heading.end():]
    if not re.search(r"<[a-zA-Z][^>]*>", body):
        body = PAGE_BREAK.sub(_page_label, body)
        body = re.sub(r"\$([^$\n]+)\$", _simple_math, body)
        lines = []
        for line in body.splitlines():
            if len(line) >= 400 and not line.lstrip().startswith(("- ", "* ", "+ ", "|", ">")):
                line = re.sub(r"(?<=[.!?])[ \t]+(?=[A-ZÀ-ỸĐ])", "\n\n", line)
            if re.match(r"^(Rất thường gặp|Thường gặp|Ít gặp|Hiếm gặp|Rất hiếm gặp):", line):
                label, rest = line.split(":", 1)
                line = f"- **{label}:**{rest}"
            lines.append(line)
        body = "\n".join(lines)
    safe = nh3.clean(MARKDOWN.render(body), tags=TAGS, attributes=ATTRIBUTES)
    return {"id": chunk["CHUNK_ID"], "subject": subject, "section": section, "raw": raw, "html": Markup(safe)}


def export_record(row):
    record = json.loads(row["original"])
    answer = row["answer"]
    label = row["answerability"]
    evidence_ids = set(json.loads(row["evidence_ids"]))
    chunks = record.get("RELEVANT_CHUNKS") or []
    selected = [chunk["CHUNK_ID"] for chunk in score_three(record) if chunk["CHUNK_ID"] in evidence_ids]
    same_answer = remove_citations(record["REFERENCE_ANSWER"]) == answer
    original_cites = set(CITATION.findall(record["REFERENCE_ANSWER"]))
    if same_answer and original_cites <= evidence_ids and record.get("ANSWERABILITY") == label:
        answer = CITATION.sub(r"[\1]", record["REFERENCE_ANSWER"])
    elif selected:
        answer = f"{answer} " + " ".join(f"[{identifier}]" for identifier in selected)
    docs = {identifier.rsplit("_c_", 1)[0] for identifier in selected}
    claims = []
    if same_answer and record.get("ANSWERABILITY") == label:
        claims = [claim for claim in (record.get("CLAIM_EVIDENCE_MAP") or []) if claim.get("chunk_id") in evidence_ids]
    record.update({
        "REFERENCE_ANSWER": answer,
        "ANSWERABILITY": label,
        "EVIDENCE_COMPLETENESS": COMPLETENESS[label],
        "RELEVANT_CHUNKS": [chunk for chunk in chunks if chunk.get("RELEVANCE_SCORE", 0) < 3
                            or chunk["CHUNK_ID"] in evidence_ids] if selected else [],
        "RELEVANT_DOCS": [doc for doc in (record.get("RELEVANT_DOCS") or []) if doc in docs],
        "CLAIM_EVIDENCE_MAP": claims,
        "VERIFIED": True,
        "ANNOTATOR_IDS": list(dict.fromkeys([*(record.get("ANNOTATOR_IDS") or []), row["reviewer_id"]])),
    })
    return record
