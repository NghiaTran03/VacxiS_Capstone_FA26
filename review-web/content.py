"""Render source text and rebuild records for the benchmark."""

import json
import re

from markdown_it import MarkdownIt
from markupsafe import Markup
import nh3

from dataset_format import as_golden, context_title

STATUS = {
    "pending": "Chưa duyệt",
    "draft": "Đang duyệt",
    "flagged": "Cần xem lại",
    "verified": "Đã xác nhận",
    "submitted": "Đã nộp",
    "needs_revision": "Cần sửa lại",
    "approved": "Đạt kiểm định",
    "rejected": "Báo lỗi trả lại",
}
ADMIN_STATUSES = {"pending": "Chưa duyệt", "draft": "Đang duyệt", "flagged": "Cần xem lại", "verified": "Đã xác nhận"}
BATCH_LABELS = {
    "calibration": "Hiệu chỉnh (Calibration)",
    "overlap": "Đo đồng thuận (Overlap)",
    "independent": "Chấm độc lập",
    "primary": "Chấm chính (Primary)",
    "auditor": "Kiểm định chéo (Audit)",
}
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


def review_content(record):
    golden = as_golden(record)
    if golden is None:
        return record["REFERENCE_ANSWER"], record.get("ANSWERABILITY", "UNANSWERABLE"), score_three(record)
    metadata = golden["additional_metadata"]
    sources = [
        {"CHUNK_ID": identifier, "TEXT": text}
        for identifier, text in zip(metadata["source_chunk_ids"], golden["context"], strict=True)
    ]
    return golden["expected_output"], metadata["answerability"], sources


def review_data(row):
    return json.loads(row.get("review_data") or "{}")


def changed_content(record, query, answer, label, evidence_ids):
    reference, original_label, sources = review_content(record)
    original_query = record["input"] if is_native(record) else record["QUERY"]
    baseline = {source["CHUNK_ID"] for source in sources} if original_label != "UNANSWERABLE" else set()
    return (query.strip() != original_query.strip() or answer != remove_citations(reference)
            or label != original_label or set(evidence_ids) != baseline)


def is_native(record):
    return "input" in record and "additional_metadata" in record


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
    if heading is None:
        prefix, _, rest = body.partition("\n")
        if prefix.startswith("[") and prefix.endswith("]"):
            heading = SOURCE_HEADING.match(rest)
            if heading:
                body = rest
    subject, section = heading.groups() if heading else ("", "")
    if heading:
        section = section.replace(" | ", " · ")
        if section == "CHỐNG CHỈ ĐỊNH":
            section = "Chống chỉ định"
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
    safe = safe.replace(
        "<table>", '<div class="source-table" role="region" aria-label="Bảng trong tài liệu" tabindex="0"><table>'
    ).replace("</table>", "</table></div>")
    return {"id": chunk["CHUNK_ID"], "subject": subject, "section": section, "raw": raw, "html": Markup(safe)}


def review_fields(row, original, same_answer):
    data = review_data(row)
    reviewer_note = row["notes"].strip()
    notes = [note for note in (original.get("VERIFICATION_NOTES"), reviewer_note) if note]
    if not same_answer:
        notes.append(f"Câu trả lời sau chỉnh sửa:\n{row['answer']}")
    annotator = reviewer_label(row) if row["full_name"] else row["reviewer_id"]
    annotators = row.get("annotator_labels") or [annotator]
    result = data.get("status")
    if result is None and row.get("status") == "verified":
        source = json.loads(row["original"])
        query = data.get("query") or source.get("input", source.get("QUERY"))
        result = "EDITED" if changed_content(source, query, row["answer"], row["answerability"],
                                             json.loads(row["evidence_ids"])) else "APPROVED"
    fields = {
        "VERIFIED": result != "REJECTED",
        "ANNOTATOR_IDS": list(dict.fromkeys([*(original.get("ANNOTATOR_IDS") or []), *annotators])),
        "ADJUDICATED": not same_answer or bool(data.get("question_reason")) or bool(reviewer_note) or bool(row.get("adjudicated")),
        "VERIFICATION_NOTES": "\n\n".join(notes) or None,
        "status": result,
        "annotator_notes": {
            "lí do sửa câu hỏi": data.get("question_reason") or None,
            "lí do sửa câu trả lời": data.get("answer_reason") or None,
            "ghi chú cho nhóm đồ án": reviewer_note or None,
        },
    }
    return fields


def export_golden(row, golden):
    metadata = golden["additional_metadata"]
    data = review_data(row)
    evidence_ids = set(json.loads(row["evidence_ids"]))
    selected = [(identifier, text) for identifier, text in
                zip(metadata["source_chunk_ids"], golden["context"], strict=True) if identifier in evidence_ids]
    doc_ids = {identifier.rsplit("_c_", 1)[0] for identifier, _ in selected}
    doc_titles = {context_title(text) for _, text in selected}
    same_answer = remove_citations(golden["expected_output"]) == row["answer"]
    extra_meta = {}
    if row.get("review_batch"):
        extra_meta["review_batch"] = row["review_batch"]
    if row.get("audited"):
        extra_meta["audited"] = True
    return dict(golden,
                input=data.get("query") or golden["input"],
                expected_output=golden["expected_output"] if same_answer else row["answer"],
                context=[text for _, text in selected],
                additional_metadata={
                    **metadata,
                    "answerability": row["answerability"],
                    "source_chunk_ids": [identifier for identifier, _ in selected],
                    "doc_ids": [doc for doc in metadata["doc_ids"] if doc in doc_ids],
                    "doc_titles": [title for title in metadata["doc_titles"] if title in doc_titles],
                    **extra_meta,
                    **review_fields(row, metadata, same_answer),
                })


def export_record(row):
    record = json.loads(row["original"])
    golden = as_golden(record)
    if golden is not None:
        return export_golden(row, golden)
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
        "QUERY": review_data(row).get("query") or record["QUERY"],
        "REFERENCE_ANSWER": answer,
        "ANSWERABILITY": label,
        "EVIDENCE_COMPLETENESS": COMPLETENESS[label],
        "RELEVANT_CHUNKS": [chunk for chunk in chunks if chunk.get("RELEVANCE_SCORE", 0) < 3
                            or chunk["CHUNK_ID"] in evidence_ids] if selected else [],
        "RELEVANT_DOCS": [doc for doc in (record.get("RELEVANT_DOCS") or []) if doc in docs],
        "CLAIM_EVIDENCE_MAP": claims,
        **review_fields(row, record, same_answer),
    })
    return record
