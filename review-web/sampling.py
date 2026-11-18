"""Stratified batch allocation and Fleiss' Kappa inter-annotator agreement metrics."""

from collections import Counter, defaultdict
import json
import random

from dataset_format import as_golden

ANSWERABILITY_LABELS = ("ANSWERABLE", "PARTIALLY_ANSWERABLE", "UNANSWERABLE")

# Exact stratified plan for the 300-question benchmark:
# stratum -> (calibration_count, overlap_count, independent_per_expert, audit_per_expert)
EXACT_300_PLAN = {
    ("ANSWERABLE", "SINGLE_HOP", "PATIENT_STYLE"): (3, 6, 18, 0),
    ("ANSWERABLE", "SINGLE_HOP", "CLINICAL_STYLE"): (2, 4, 12, 0),
    ("ANSWERABLE", "MULTI_HOP", "PATIENT_STYLE"): (2, 3, 11, 0),
    ("ANSWERABLE", "MULTI_HOP", "CLINICAL_STYLE"): (1, 3, 7, 0),
    ("ANSWERABLE", "TABLE_REASONING", "PATIENT_STYLE"): (1, 3, 7, 3),
    ("ANSWERABLE", "TABLE_REASONING", "CLINICAL_STYLE"): (1, 1, 5, 2),
    ("PARTIALLY_ANSWERABLE", "SINGLE_HOP", "PATIENT_STYLE"): (1, 2, 8, 2),
    ("PARTIALLY_ANSWERABLE", "SINGLE_HOP", "CLINICAL_STYLE"): (1, 2, 5, 1),
    ("PARTIALLY_ANSWERABLE", "MULTI_HOP", "PATIENT_STYLE"): (1, 3, 4, 1),
    ("PARTIALLY_ANSWERABLE", "MULTI_HOP", "CLINICAL_STYLE"): (1, 1, 3, 1),
    ("PARTIALLY_ANSWERABLE", "TABLE_REASONING", "PATIENT_STYLE"): (1, 1, 3, 3),
    ("PARTIALLY_ANSWERABLE", "TABLE_REASONING", "CLINICAL_STYLE"): (0, 1, 2, 2),
}


def question_stratum(record):
    """Extract the 3-dimensional stratum tuple from a question record."""
    golden = as_golden(record)
    if golden is not None:
        meta = golden["additional_metadata"]
        return (
            meta.get("answerability") or "UNKNOWN",
            meta.get("complexity") or "UNKNOWN",
            meta.get("query_style") or "UNKNOWN",
        )
    return (
        record.get("ANSWERABILITY") or "UNKNOWN",
        record.get("QUERY_COMPLEXITY") or "UNKNOWN",
        record.get("QUERY_STYLE") or "UNKNOWN",
    )


def format_stratum(stratum):
    return "|".join(stratum)


def _hamilton_counts(weights, target):
    """Allocate integer counts summing to target proportionally via largest remainder."""
    total_weight = sum(weights.values())
    if target <= 0 or total_weight <= 0:
        return {key: 0 for key in weights}
    raw = {key: (value / total_weight) * target for key, value in weights.items()}
    counts = {key: int(val) for key, val in raw.items()}
    remainder = target - sum(counts.values())
    if remainder > 0:
        ranked = sorted(weights, key=lambda key: (raw[key] - counts[key], weights[key], key), reverse=True)
        for key in ranked[:remainder]:
            counts[key] += 1
    return counts


def _allocate_exact_300(strata, rng):
    """Allocate the 300-question benchmark using the exact stratified design."""
    allocations = []
    for stratum, (calib_n, overlap_n, indep_each, audit_each) in EXACT_300_PLAN.items():
        qids = list(strata[stratum])
        rng.shuffle(qids)
        idx = 0
        for qid in qids[idx : idx + calib_n]:
            allocations.append(_row(qid, "calibration", stratum))
        idx += calib_n
        for qid in qids[idx : idx + overlap_n]:
            allocations.append(_row(qid, "overlap", stratum))
        idx += overlap_n
        for slot in (1, 2, 3):
            slot_qids = qids[idx : idx + indep_each]
            idx += indep_each
            audit_set = set(slot_qids[:audit_each])
            for qid in slot_qids:
                allocations.append(_row(qid, "independent", stratum, slot, qid in audit_set))
    return allocations


def _audit_priority(stratum):
    """Rank independent items for cross-audit: both > PARTIALLY_ANSWERABLE or TABLE_REASONING > rest."""
    answerability, complexity, _ = stratum
    is_partial = answerability == "PARTIALLY_ANSWERABLE"
    is_table = complexity == "TABLE_REASONING"
    if is_partial and is_table:
        return 0
    if is_partial or is_table:
        return 1
    return 2


def _allocate_general(strata, total, rng):
    """Fallback proportional stratified allocation for datasets not matching the 300-item distribution."""
    if total < 5:
        raise ValueError("Cần ít nhất 5 câu hỏi để chia gói chuẩn khoa học cho 3 chuyên gia.")
    # Target ~5% calibration, ~10% overlap, and a multiple-of-3 remainder for independent work
    calib_n = max(1, round(total * 0.05))
    overlap_n = max(1, round(total * 0.10))
    while total - calib_n - overlap_n < 3 or (total - calib_n - overlap_n) % 3 != 0:
        if calib_n + overlap_n > total - 3:
            if overlap_n > 1:
                overlap_n -= 1
            else:
                calib_n = max(1, calib_n - 1)
        else:
            overlap_n += 1

    weights = {stratum: len(qids) for stratum, qids in strata.items()}
    calib_counts = _hamilton_counts(weights, calib_n)
    rem_after_calib = {stratum: weights[stratum] - calib_counts[stratum] for stratum in weights}
    overlap_counts = _hamilton_counts(rem_after_calib, overlap_n)

    allocations = []
    indep_pool = []
    for stratum in sorted(strata):
        qids = list(strata[stratum])
        rng.shuffle(qids)
        c_n = calib_counts[stratum]
        o_n = overlap_counts[stratum]
        for qid in qids[:c_n]:
            allocations.append(_row(qid, "calibration", stratum))
        for qid in qids[c_n : c_n + o_n]:
            allocations.append(_row(qid, "overlap", stratum))
        for qid in qids[c_n + o_n :]:
            indep_pool.append((stratum, qid))

    slot_items = {1: [], 2: [], 3: []}
    for index, (stratum, qid) in enumerate(indep_pool):
        slot = (index % 3) + 1
        slot_items[slot].append((stratum, qid))

    for slot, items in slot_items.items():
        audit_target = min(len(items), max(1, round(len(items) * (15 / 85))))
        ranked = sorted(items, key=lambda pair: (_audit_priority(pair[0]), rng.random()))
        audit_qids = {qid for _, qid in ranked[:audit_target]}
        for stratum, qid in items:
            allocations.append(_row(qid, "independent", stratum, slot, qid in audit_qids))
    return allocations


def _row(question_id, batch_type, stratum, primary_slot=None, requires_audit=False):
    auditor_slot = ((primary_slot % 3) + 1) if (primary_slot and requires_audit) else None
    return {
        "question_id": question_id,
        "batch_type": batch_type,
        "stratum": format_stratum(stratum),
        "primary_slot": primary_slot,
        "auditor_slot": auditor_slot,
        "requires_audit": 1 if requires_audit else 0,
    }


def allocate_scientific_batches(questions, *, seed=42):
    """Partition questions into stratified calibration, overlap, and 3x balanced independent+audit batches.

    Args:
        questions: Sequence of tuples or dicts containing ``id`` and ``original`` (JSON string or dict).
        seed: Deterministic RNG seed for sampling within each stratum.
    """
    strata = defaultdict(list)
    order = {}
    for index, item in enumerate(questions):
        qid = item["id"]
        raw = item["original"]
        record = json.loads(raw) if isinstance(raw, str) else raw
        strata[question_stratum(record)].append(qid)
        order[qid] = index

    rng = random.Random(seed)
    matches_300 = (
        len(order) == 300
        and set(strata.keys()) == set(EXACT_300_PLAN.keys())
        and all(len(strata[key]) == c + o + 3 * i for key, (c, o, i, _) in EXACT_300_PLAN.items())
    )
    rows = _allocate_exact_300(strata, rng) if matches_300 else _allocate_general(strata, len(order), rng)
    rows.sort(key=lambda row: order[row["question_id"]])
    return rows


def kappa_interpretation(kappa):
    """Map Fleiss' Kappa to standard Landis & Koch (1977) agreement labels."""
    if kappa is None:
        return "Chưa đủ dữ liệu"
    if kappa < 0.0:
        return "Không đồng thuận (< 0.00)"
    if kappa <= 0.20:
        return "Rất thấp (0.00–0.20)"
    if kappa <= 0.40:
        return "Trung bình yếu (0.21–0.40)"
    if kappa <= 0.60:
        return "Khá (0.41–0.60)"
    if kappa <= 0.80:
        return "Tốt / Đáng tin cậy (0.61–0.80)"
    return "Rất tốt / Xuất sắc (0.81–1.00)"


def _fleiss_kappa_matrix(counts_matrix, num_raters):
    """Compute Fleiss' Kappa from an N x k rating-count matrix with constant raters per subject."""
    n_items = len(counts_matrix)
    if n_items == 0 or num_raters < 2:
        return None
    total_ratings = n_items * num_raters
    n_categories = len(counts_matrix[0])
    p_j = [sum(row[col] for row in counts_matrix) / total_ratings for col in range(n_categories)]
    p_e = sum(p * p for p in p_j)
    denom_per_item = num_raters * (num_raters - 1)
    p_bar = sum(sum(c * (c - 1) for c in row) / denom_per_item for row in counts_matrix) / n_items
    if abs(1.0 - p_e) <= 1e-12:
        return 1.0 if p_bar >= 1.0 - 1e-12 else 0.0
    return (p_bar - p_e) / (1.0 - p_e)


def _jaccard(left, right):
    if not left and not right:
        return 1.0
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def compute_fleiss_kappa(overlap_total, overlap_reviews, question_chunks):
    """Calculate Fleiss' Kappa metrics for completed 3-rater overlap questions.

    Args:
        overlap_total: Total number of overlap questions in the dataset.
        overlap_reviews: Rows from ``expert_reviews`` where ``role = 'overlap'``.
        question_chunks: Mapping ``question_id -> list[chunk_id]`` of available source chunks.
    """
    by_question = defaultdict(list)
    for row in overlap_reviews:
        if row["status"] == "submitted":
            by_question[row["question_id"]].append(row)

    completed_qids = [qid for qid, items in by_question.items() if len(items) == 3]
    if not completed_qids:
        return {
            "total": overlap_total,
            "completed": 0,
            "answerability_kappa": None,
            "answerability_level": kappa_interpretation(None),
            "answerability_exact_rate": None,
            "evidence_kappa": None,
            "evidence_level": kappa_interpretation(None),
            "evidence_exact_rate": None,
            "mean_jaccard": None,
            "total_chunks_evaluated": 0,
        }

    label_index = {label: idx for idx, label in enumerate(ANSWERABILITY_LABELS)}
    ans_matrix = []
    chunk_matrix = []
    ans_exact = 0
    ev_exact = 0
    jaccard_scores = []

    for qid in completed_qids:
        raters = by_question[qid]
        counts = [0] * len(ANSWERABILITY_LABELS)
        ev_sets = []
        for item in raters:
            counts[label_index[item["answerability"]]] += 1
            ev_sets.append(set(json.loads(item["evidence_ids"] or "[]")))
        ans_matrix.append(counts)
        if max(counts) == 3:
            ans_exact += 1
        if ev_sets[0] == ev_sets[1] == ev_sets[2]:
            ev_exact += 1
        pair_jaccards = (
            _jaccard(ev_sets[0], ev_sets[1]),
            _jaccard(ev_sets[1], ev_sets[2]),
            _jaccard(ev_sets[0], ev_sets[2]),
        )
        jaccard_scores.append(sum(pair_jaccards) / 3.0)

        candidate_chunks = list(dict.fromkeys(question_chunks.get(qid) or []))
        for extra in ev_sets:
            for cid in extra:
                if cid not in candidate_chunks:
                    candidate_chunks.append(cid)
        for cid in candidate_chunks:
            picked = sum(1 for s in ev_sets if cid in s)
            chunk_matrix.append([picked, 3 - picked])

    n_done = len(completed_qids)
    ans_kappa = _fleiss_kappa_matrix(ans_matrix, 3)
    ev_kappa = _fleiss_kappa_matrix(chunk_matrix, 3) if chunk_matrix else None
    return {
        "total": overlap_total,
        "completed": n_done,
        "answerability_kappa": ans_kappa,
        "answerability_level": kappa_interpretation(ans_kappa),
        "answerability_exact_rate": ans_exact / n_done,
        "evidence_kappa": ev_kappa,
        "evidence_level": kappa_interpretation(ev_kappa),
        "evidence_exact_rate": ev_exact / n_done,
        "mean_jaccard": sum(jaccard_scores) / n_done,
        "total_chunks_evaluated": len(chunk_matrix),
    }


def summarize_allocation(allocations):
    """Build per-batch and per-slot summary statistics from allocation rows."""
    by_batch = Counter(row["batch_type"] for row in allocations)
    by_slot = {}
    for slot in (1, 2, 3):
        primary = [row for row in allocations if row["primary_slot"] == slot]
        auditor = [row for row in allocations if row["auditor_slot"] == slot]
        by_slot[slot] = {
            "calibration": by_batch["calibration"],
            "overlap": by_batch["overlap"],
            "primary": len(primary),
            "audit": len(auditor),
            "total_turns": by_batch["calibration"] + by_batch["overlap"] + len(primary) + len(auditor),
        }
    return {"by_batch": dict(by_batch), "by_slot": by_slot}
