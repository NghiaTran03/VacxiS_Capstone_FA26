"""Semantic chunking: merge short sections, split long ones, preserve tables."""
from bisect import bisect_right
from collections.abc import Callable, Iterator

from .schemas import classify_category, detect_content_type

DEFAULT_CHUNK_OVERLAP_TOKENS = 50


def _count_tokens(text: str) -> int:
    """Rough word-based token count for Vietnamese text."""
    return len(text.split())


def create_medical_chunks(
    sections: list[dict],
    min_tokens: int = 80,
    max_tokens: int = 600,
    chunk_overlap_tokens: int = DEFAULT_CHUNK_OVERLAP_TOKENS,
    *,
    embedding_prefix: Callable[[str], str] | None = None,
) -> list[dict]:
    """
    Merge short sections (< min_tokens) into the next section,
    split sections exceeding max_tokens, and preserve standalone tables.

    Limits and overlap use the same word-based estimate. When provided,
    embedding_prefix returns the prefix for the emitted chunk's header.

    Each output chunk contains:
        header, parent_header, content, category, content_type, has_table
    """
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    if chunk_overlap_tokens < 0:
        raise ValueError("chunk_overlap_tokens must be nonnegative")

    chunks: list[dict] = []
    buffer_sections: list[dict] = []
    buffer_tokens = 0

    def content_budget(header: str) -> int:
        prefix_tokens = _count_tokens(embedding_prefix(header)) if embedding_prefix else 0
        return max_tokens - prefix_tokens

    def merged_header(sections: list[dict]) -> str:
        return " | ".join(dict.fromkeys(s["header"] for s in sections))

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
        combined_header = merged_header(buffer_sections)
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
        if not sec_tokens:
            continue
        budget = content_budget(section["header"])
        if budget < 1:
            raise ValueError(f"Embedding prefix leaves no content budget for {section['header']!r}")
        blocks = _split_section_into_blocks(section["content"]) if "|" in section["content"] else []
        has_table = section.get("has_table", False) or any(kind == "table" for kind, _ in blocks)

        # Tables: if section exceeds max_tokens, split it while preserving table blocks;
        # otherwise keep intact as a standalone chunk.
        if has_table:
            flush_buffer()
            if sec_tokens > budget:
                _split_long_section(section, blocks, budget, chunk_overlap_tokens, chunks)
            else:
                chunks.append({
                    "header": section["header"],
                    "parent_header": section.get("parent_header", ""),
                    "content": section["content"],
                    "category": classify_category(section["header"], section.get("parent_header", "")),
                    "content_type": (
                        "table" if len(blocks) == 1 and blocks[0][0] == "table"
                        else detect_content_type(section["content"], True)
                    ),
                    "has_table": True,
                })
            continue

        if sec_tokens > budget:
            flush_buffer()
            _split_long_section(section, blocks, budget, chunk_overlap_tokens, chunks)
            continue

        if buffer_sections and buffer_tokens + sec_tokens > content_budget(
            merged_header([*buffer_sections, section])
        ):
            flush_buffer()
        buffer_sections.append(section)
        buffer_tokens += sec_tokens
        if buffer_tokens >= min_tokens:
            flush_buffer()

    flush_buffer()
    return chunks


def _table_cells(line: str) -> list[str]:
    """Split on unescaped pipes and ignore optional outer pipes."""
    line = line.strip()
    if "|" not in line:
        return []
    cells: list[str] = []
    start = 0
    escaped = False
    for index, char in enumerate(line):
        if char == "|" and not escaped:
            if index:
                cells.append(line[start:index])
            start = index + 1
        escaped = char == "\\" and not escaped
    if start and start < len(line):
        cells.append(line[start:])
    return cells


def _has_table_header(header: str, separator: str) -> bool:
    rules = _table_cells(separator)
    if not rules or len(rules) != len(_table_cells(header)):
        return False
    for rule in rules:
        dashes = rule.strip().removeprefix(":").removesuffix(":")
        if not dashes or dashes.strip("-"):
            return False
    return True


def _split_section_into_blocks(content: str) -> list[tuple[str, str]]:
    """Keep pipe tables and surrounding text in their original order."""
    lines = content.splitlines(keepends=True)
    blocks: list[tuple[str, str]] = []
    start = 0
    kind = "text"

    def append_block(end: int) -> None:
        block = "".join(lines[start:end])
        block = block.rstrip("\r\n") if kind == "table" else block.strip()
        if block:
            blocks.append((kind, block))

    for index, line in enumerate(lines):
        is_table = line.lstrip().startswith("|") or (
            bool(_table_cells(line)) and (
                kind == "table" or (
                    index + 1 < len(lines) and _has_table_header(line, lines[index + 1])
                )
            )
        )
        line_kind = "table" if is_table else "text"
        if line_kind != kind:
            append_block(index)
            start = index
            kind = line_kind
    append_block(len(lines))
    return blocks


def _split_text_paragraphs(text: str, max_tokens: int, overlap: int) -> Iterator[str]:
    """Prefer paragraph or line endings, with word boundaries as the fallback."""
    if _count_tokens(text) <= max_tokens:
        yield text
        return

    starts: list[int] = []
    line_ends: list[int] = []
    paragraph_ends: list[int] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        words = line.split()
        cursor = offset
        for word in words:
            start = text.index(word, cursor)
            starts.append(start)
            cursor = start + len(word)
        if words:
            line_ends.append(len(starts))
        elif starts:
            paragraph_ends.append(len(starts))
        offset += len(line)
    starts.append(len(text))

    end = 0
    overlap = min(overlap, max_tokens - 1)
    while end < len(starts) - 1:
        start = max(0, end - overlap)
        limit = min(start + max_tokens, len(starts) - 1)
        if limit < len(starts) - 1:
            for boundaries in (paragraph_ends, line_ends):
                boundary = bisect_right(boundaries, limit) - 1
                if boundary >= 0 and boundaries[boundary] > end:
                    limit = boundaries[boundary]
                    break
        yield text[starts[start]:starts[limit]].rstrip()
        end = limit


def _split_table_row(row: str, column_count: int, max_tokens: int) -> Iterator[str]:
    """Continue an oversized row in the original column slots, in cell order."""
    cells = _table_cells(row)
    column_count = max(column_count, len(cells))
    if cells and not any(cell.strip() for cell in cells):
        yield "|" * (column_count + 1)
        return
    if not cells:
        raise ValueError("Cannot split an oversized table row without cell delimiters")
    # Under a tight budget, the header supplies the implied empty trailing cells.
    compact = max_tokens < 3
    cell_budget = max_tokens if compact else max_tokens - 2
    for column, cell in enumerate(cells):
        if not cell.strip():
            continue
        left = "|" * (column + 1)
        for part in _split_text_paragraphs(cell.strip(), cell_budget, 0):
            if compact:
                yield f"{left}{part}"
            else:
                yield f"{left} {part} {'|' * (column_count - column)}"


def _split_table(content: str, max_tokens: int, overlap: int) -> Iterator[str]:
    """Pack complete rows with a repeated header and a suffix of complete rows."""
    if _count_tokens(content) <= max_tokens:
        yield content
        return

    lines = content.splitlines()
    has_header = len(lines) > 1 and _has_table_header(lines[0], lines[1])
    header = "\n".join(lines[:2]) + "\n" if has_header else ""
    column_count = len(_table_cells(lines[0])) if has_header else 0
    row_budget = max_tokens - _count_tokens(header)
    if row_budget < 1:
        raise ValueError("max_tokens cannot fit the Markdown table header and row content")

    rows: list[tuple[str, int]] = []
    row_tokens = 0
    for row in lines[2 if has_header else 0:]:
        size = _count_tokens(row)
        if row_tokens + size > row_budget:
            if rows:
                yield header + "\n".join(line for line, _ in rows)
            keep = len(rows)
            row_tokens = 0
            # Choose the suffix closest to the target that leaves space for new content.
            while keep:
                candidate = row_tokens + rows[keep - 1][1]
                if candidate + size > row_budget or abs(overlap - candidate) >= abs(overlap - row_tokens):
                    break
                keep -= 1
                row_tokens = candidate
            rows = rows[keep:]
        if size > row_budget:
            for part in _split_table_row(row, column_count, row_budget):
                yield header + part
        else:
            rows.append((row, size))
            row_tokens += size
    if rows:
        yield header + "\n".join(line for line, _ in rows)


def _split_long_section(
    section: dict,
    blocks: list[tuple[str, str]],
    max_tokens: int,
    overlap: int,
    output: list[dict],
):
    """Split each block in order, keeping overlap inside that block."""
    sub_idx = 0
    for kind, content in blocks or [("text", section["content"])]:
        has_table = kind == "table"
        split = _split_table if has_table else _split_text_paragraphs
        for part in split(content, max_tokens, overlap):
            _emit_sub_chunk(section, part, sub_idx, has_table, output)
            sub_idx += 1


def _emit_sub_chunk(section: dict, content: str, sub_idx: int, has_table: bool, output: list[dict]):
    """Emit a sub-chunk produced from splitting a long section."""
    output.append({
        "header": section["header"],
        "parent_header": section.get("parent_header", "") if sub_idx == 0 else section["header"],
        "content": content,
        "category": classify_category(section["header"], section.get("parent_header", "")),
        "content_type": "table" if has_table else "text",
        "has_table": has_table,
    })

