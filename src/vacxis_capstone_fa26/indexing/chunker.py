"""Semantic chunking: merge short sections, split long ones, preserve tables."""
from .schemas import classify_category, detect_content_type


def _count_tokens(text: str) -> int:
    """Rough word-based token count for Vietnamese text."""
    return len(text.split())


def create_medical_chunks(
    sections: list[dict],
    min_tokens: int = 80,
    max_tokens: int = 600,
) -> list[dict]:
    """
    Merge short sections (< min_tokens) into the next section,
    split sections exceeding max_tokens, and preserve standalone tables.

    Each output chunk contains:
        header, parent_header, content, category, content_type, has_table
    """
    if not sections:
        return []

    chunks: list[dict] = []
    buffer_sections: list[dict] = []
    buffer_tokens = 0

    def flush_buffer():
        nonlocal buffer_sections, buffer_tokens
        if not buffer_sections:
            return

        # Pick dominant section by token count
        dominant = max(buffer_sections, key=lambda s: _count_tokens(s["content"]))

        # Check for specific medical category among all buffered sections
        chosen_cat = "thông tin chung"
        for s in sorted(buffer_sections, key=lambda x: _count_tokens(x["content"]), reverse=True):
            cat = classify_category(s["header"], s.get("parent_header", ""))
            if cat != "thông tin chung":
                chosen_cat = cat
                break

        combined_content = "\n\n".join(s["content"] for s in buffer_sections)
        combined_header = " | ".join(dict.fromkeys(s["header"] for s in buffer_sections))
        has_table = any(s.get("has_table", False) for s in buffer_sections)

        chunks.append({
            "header": dominant["header"] if len(buffer_sections) == 1 else combined_header,
            "parent_header": dominant.get("parent_header", ""),
            "content": combined_content,
            "category": chosen_cat,
            "content_type": detect_content_type(combined_content, has_table),
            "has_table": has_table,
        })

        buffer_sections = []
        buffer_tokens = 0

    for section in sections:
        sec_tokens = _count_tokens(section["content"])

        # Tables are kept intact as standalone semantic units
        if section.get("has_table", False):
            flush_buffer()
            chunks.append({
                "header": section["header"],
                "parent_header": section.get("parent_header", ""),
                "content": section["content"],
                "category": classify_category(section["header"], section.get("parent_header", "")),
                "content_type": "table" if sec_tokens > 0 else "text",
                "has_table": True,
            })
            continue

        if sec_tokens > max_tokens:
            flush_buffer()
            _split_long_section(section, max_tokens, chunks)
        elif buffer_tokens + sec_tokens < min_tokens:
            buffer_sections.append(section)
            buffer_tokens += sec_tokens
        else:
            buffer_sections.append(section)
            buffer_tokens += sec_tokens
            flush_buffer()

    flush_buffer()
    return chunks


def _split_long_section(
    section: dict,
    max_tokens: int,
    output: list[dict],
):
    """Split a long section by paragraph boundaries without breaking tables."""
    content = section["content"]
    paragraphs = content.split("\n\n")

    current_parts: list[str] = []
    current_tokens = 0
    sub_idx = 0

    for para in paragraphs:
        para_tokens = _count_tokens(para)

        # Do not break markdown tables even if they exceed max_tokens
        if para_tokens > max_tokens and not para.strip().startswith("|"):
            if current_parts:
                _emit_sub_chunk(section, "\n\n".join(current_parts), sub_idx, output)
                sub_idx += 1
                current_parts = []
                current_tokens = 0
            _emit_sub_chunk(section, para, sub_idx, output)
            sub_idx += 1
            continue

        if current_tokens + para_tokens > max_tokens and current_parts:
            _emit_sub_chunk(section, "\n\n".join(current_parts), sub_idx, output)
            sub_idx += 1
            current_parts = []
            current_tokens = 0

        current_parts.append(para)
        current_tokens += para_tokens

    if current_parts:
        _emit_sub_chunk(section, "\n\n".join(current_parts), sub_idx, output)


def _emit_sub_chunk(section: dict, content: str, sub_idx: int, output: list[dict]):
    """Emit a sub-chunk produced from splitting a long section."""
    has_table = "|" in content and content.count("|") > 4
    output.append({
        "header": section["header"],
        "parent_header": section.get("parent_header", "") if sub_idx == 0 else section["header"],
        "content": content,
        "category": classify_category(section["header"], section.get("parent_header", "")),
        "content_type": detect_content_type(content, has_table),
        "has_table": has_table,
    })
