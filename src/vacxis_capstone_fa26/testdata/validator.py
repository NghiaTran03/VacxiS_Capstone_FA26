"""Post-write validation for the generated dataset.

The validator is read-only: it inspects the produced query list and emits a
: class:`ValidationReport` summarising hard failures (exit non-zero) and soft
warnings. The output JSON schema is unchanged from the original pipeline.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable

from .chunks import Chunk


@dataclass
class ValidationReport:
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.failures

    def render(self) -> str:
        lines: list[str] = []
        lines.append(
            f"Validation: {'OK' if self.ok else f'FAIL ({len(self.failures)})'}; "
            f"warnings={len(self.warnings)}; stats={self.stats}"
        )
        for w in self.warnings:
            lines.append(f"  WARN  {w}")
        for f in self.failures:
            lines.append(f"  FAIL  {f}")
        return "\n".join(lines)


def _chunks_by_id(chunks: Iterable[Chunk]) -> dict[str, Chunk]:
    return {c.chunk_id: c for c in chunks}


def validate(
    queries: list[dict],
    chunks: list[Chunk],
    by_category: dict[str, list[Chunk]],
    expected_per_type: int,
    expected_types: list[str],
) -> ValidationReport:
    rpt = ValidationReport()
    by_id = _chunks_by_id(chunks)
    type_counts: Counter[str] = Counter()

    for q in queries:
        qid = q.get("QUERY_ID", "<missing>")
        qtype = q.get("QUERY_TYPE")
        type_counts[qtype] += 1

        relevant = q.get("RELEVANT_CHUNKS", [])
        answerability = q.get("ANSWERABILITY")

        if answerability == "UNANSWERABLE":
            if relevant:
                rpt.failures.append(
                    f"{qid} UNANSWERABLE but RELEVANT_CHUNKS not empty"
                )
            continue

        if not relevant:
            rpt.failures.append(
                f"{qid} {answerability} but RELEVANT_CHUNKS is empty"
            )
            continue

        primary = relevant[0]
        cid = primary.get("CHUNK_ID")
        if cid not in by_id:
            rpt.failures.append(f"{qid} references unknown CHUNK_ID {cid!r}")
            continue

        text = primary.get("TEXT", "")
        if text != by_id[cid].text:
            rpt.warnings.append(
                f"{qid} TEXT in RELEVANT_CHUNKS does not match chunks.parquet"
            )

        spans = primary.get("EVIDENCE_SPANS", [])
        if answerability == "ANSWERABLE" and not spans:
            rpt.failures.append(
                f"{qid} ANSWERABLE but EVIDENCE_SPANS is empty on primary chunk"
            )
        for sp in spans:
            start, end = sp.get("START"), sp.get("END")
            if start is None or end is None or start < 0 or end > len(text) or start >= end:
                rpt.failures.append(
                    f"{qid} EVIDENCE_SPAN out of bounds: {sp!r} for text of {len(text)} chars"
                )

        # MULTI_HOP / INTERACTION should pull a second doc; other types should
        # not (per the strict pairing rules).
        if q.get("QUERY_COMPLEXITY") == "MULTI_HOP":
            docs = q.get("RELEVANT_DOCS", [])
            if qtype == "INTERACTION":
                if len(docs) < 2:
                    rpt.failures.append(
                        f"{qid} INTERACTION multi-hop needs >= 2 RELEVANT_DOCS, got {docs}"
                    )
            else:
                if len(docs) != 1:
                    rpt.failures.append(
                        f"{qid} {qtype} multi-hop should keep RELEVANT_DOCS==1, got {docs}"
                    )

        for d in relevant[1:]:
            if d.get("CHUNK_ID") not in by_id:
                rpt.failures.append(
                    f"{qid} distractor references unknown CHUNK_ID {d.get('CHUNK_ID')!r}"
                )

        # Validate REFERENCE_ANSWER if present
        ref_ans = q.get("REFERENCE_ANSWER")
        if ref_ans is not None:
            if not isinstance(ref_ans, str) or not ref_ans.strip():
                rpt.failures.append(f"{qid} REFERENCE_ANSWER is present but empty or not a string")
            elif answerability == "ANSWERABLE":
                citations = re.findall(r"\[([a-zA-Z0-9_\-]+)\]", ref_ans)
                relevant_cids = {c.get("CHUNK_ID") for c in relevant}
                for cite in citations:
                    if cite not in relevant_cids and cite not in by_id:
                        rpt.warnings.append(
                            f"{qid} citation [{cite}] in REFERENCE_ANSWER not in RELEVANT_CHUNKS"
                        )

        # Validate CLAIM_EVIDENCE_MAP if present
        claim_map = q.get("CLAIM_EVIDENCE_MAP")
        if claim_map:
            relevant_cids = {c.get("CHUNK_ID") for c in relevant}
            for cl in claim_map:
                if not isinstance(cl, dict) or not cl.get("claim"):
                    rpt.warnings.append(f"{qid} malformed claim entry: {cl!r}")
                elif cl.get("chunk_id") and cl.get("chunk_id") not in relevant_cids:
                    rpt.warnings.append(
                        f"{qid} claim chunk_id {cl.get('chunk_id')!r} not in RELEVANT_CHUNKS"
                    )

    for t in expected_types:
        if type_counts.get(t, 0) == 0:
            rpt.failures.append(f"QUERY_TYPE {t} produced 0 queries")

        if expected_per_type > 0 and type_counts.get(t, 0) < expected_per_type:
            rpt.warnings.append(
                f"QUERY_TYPE {t} produced {type_counts.get(t, 0)} of {expected_per_type} requested"
            )

    rpt.stats = {
        "queries": len(queries),
        "by_type": dict(type_counts),
        "chunks": len(chunks),
    }
    return rpt
