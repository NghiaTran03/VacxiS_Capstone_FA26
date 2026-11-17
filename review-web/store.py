"""Persist assignments, user preferences, and verified revisions."""

import argparse
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import sqlite3


BASE_DIR = Path(__file__).resolve().parent
SCOPES = {"live": "Duyệt thật", "test": "Thử nghiệm nội bộ"}
CONSENT_VERSION = 1
QUESTION_ID = re.compile(r"[A-Za-z0-9_-]{1,100}")


class ReviewError(ValueError):
    pass


class ReviewConflict(ReviewError):
    pass


def timestamp(when=None):
    return (when or datetime.now(UTC)).isoformat(timespec="seconds")


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def database_location():
    url = os.environ.get("DATABASE_URL")
    if url:
        if not url.startswith(("postgresql://", "postgres://")):
            raise RuntimeError("DATABASE_URL must be a PostgreSQL connection string.")
        return url
    if os.environ.get("VERCEL") or os.environ.get("REVIEW_PUBLIC_URL", "").startswith("https://"):
        raise RuntimeError("Set DATABASE_URL for the hosted app.")
    return os.environ.get("REVIEW_DB_PATH", str(BASE_DIR / "review.sqlite3"))


class Connection:
    def __init__(self, raw, postgres):
        self.raw = raw
        self.postgres = postgres

    def execute(self, sql, params=()):
        return self.raw.execute(sql.replace("?", "%s") if self.postgres else sql, params)

    def one(self, sql, params=()):
        row = self.execute(sql, params).fetchone()
        return dict(row) if row is not None else None

    def all(self, sql, params=()):
        return [dict(row) for row in self.execute(sql, params)]

    def locked(self, sql, params=()):
        return self.one(sql + (" FOR UPDATE" if self.postgres else ""), params)


@contextmanager
def connect(location, *, write=False):
    postgres = location.startswith(("postgresql://", "postgres://"))
    if postgres:
        import psycopg
        from psycopg.rows import dict_row

        raw = psycopg.connect(location, row_factory=dict_row, connect_timeout=8)
    else:
        raw = sqlite3.connect(location, timeout=8)
        raw.row_factory = sqlite3.Row
        raw.execute("PRAGMA foreign_keys = ON")
    try:
        if write and not postgres:
            raw.execute("BEGIN IMMEDIATE")
        yield Connection(raw, postgres)
        raw.commit()
    except BaseException:
        raw.rollback()
        raise
    finally:
        raw.close()


def import_dataset(location, path):
    contents = path.read_bytes()
    source_hash = hashlib.sha256(contents).hexdigest()
    records = json.loads(contents.decode("utf-8-sig"))
    if not isinstance(records, list) or not records:
        raise ReviewError("Bộ dữ liệu phải là một danh sách câu hỏi.")
    ids = set()
    for record in records:
        if not isinstance(record, dict):
            raise ReviewError("Mỗi câu hỏi phải là một bản ghi JSON.")
        identifier = record.get("QUERY_ID", "")
        if not isinstance(identifier, str) or not QUESTION_ID.fullmatch(identifier) or identifier in ids:
            raise ReviewError("QUERY_ID phải hợp lệ và duy nhất.")
        if any(not isinstance(record.get(field), str) for field in ("QUERY", "REFERENCE_ANSWER")):
            raise ReviewError(f"{identifier} thiếu nội dung câu hỏi hoặc câu trả lời.")
        ids.add(identifier)

    with connect(location, write=True) as db:
        for statement in (BASE_DIR / "schema.sql").read_text(encoding="utf-8").split(";"):
            if statement.strip():
                db.execute(statement)
        db.execute("INSERT INTO dataset VALUES (1, ?) ON CONFLICT (id) DO NOTHING", (source_hash,))
        existing = db.locked("SELECT source_hash FROM dataset WHERE id = 1")
        if existing["source_hash"] != source_hash:
            raise ReviewError("Cơ sở dữ liệu đã chứa một phiên bản khác. Hãy dùng cơ sở dữ liệu mới.")
        for position, record in enumerate(records):
            identifier = record["QUERY_ID"]
            db.execute(
                "INSERT INTO questions VALUES (?, ?, ?, ?) ON CONFLICT (id) DO NOTHING",
                (identifier, position, record["QUERY"], json.dumps(record, ensure_ascii=False)),
            )
            for scope in SCOPES:
                db.execute(
                    "INSERT INTO review_items (scope, question_id) VALUES (?, ?) "
                    "ON CONFLICT (scope, question_id) DO NOTHING",
                    (scope, identifier),
                )
    return len(records)


def issue_link(db, reviewer_id, referral_id=None, *, token=None):
    token = token or secrets.token_urlsafe(32)
    db.execute(
        "UPDATE invitations SET consumed_at = ? WHERE reviewer_id = ? AND consumed_at IS NULL",
        (timestamp(), reviewer_id),
    )
    inserted = db.execute(
        "INSERT INTO invitations (token_hash, reviewer_id, referral_id, expires_at) VALUES (?, ?, ?, ?) "
        "ON CONFLICT (token_hash) DO NOTHING",
        (digest(token), reviewer_id, referral_id, timestamp(datetime.now(UTC) + timedelta(days=7))),
    ).rowcount
    if inserted != 1:
        raise ReviewError("Hãy thử tạo lại lời mời.")
    return token


def reserve_operation(db, operation):
    if not isinstance(operation, str) or not re.fullmatch(r"[a-f0-9]{32}", operation):
        raise ReviewError("Hãy tải lại trang quản lý rồi thử lại.")
    changed = db.execute("INSERT INTO admin_operations VALUES (?) ON CONFLICT (id) DO NOTHING", (operation,)).rowcount
    if changed != 1:
        raise ReviewConflict("Thao tác này đã được xử lý. Hãy tải lại trang quản lý.")


def release_unstarted(db, reviewer_id):
    changed = db.execute(
        "UPDATE review_items SET reviewer_id = NULL WHERE reviewer_id = ? AND status = 'pending' AND revision = 0",
        (reviewer_id,),
    ).rowcount
    if not changed:
        raise ReviewError("Người duyệt này không còn câu chưa bắt đầu.")
    if not db.one("SELECT question_id FROM review_items WHERE reviewer_id = ? LIMIT 1", (reviewer_id,)):
        db.execute("UPDATE reviewers SET access_version = access_version + 1 WHERE id = ?", (reviewer_id,))
        person = db.one("SELECT * FROM reviewers WHERE id = ?", (reviewer_id,))
        if person["scope"] == "live" and person["consent"] == "accepted":
            update_profile(db, person, "withdrawn", {})
        db.execute("UPDATE invitations SET consumed_at = ? WHERE reviewer_id = ? AND consumed_at IS NULL",
                   (timestamp(), reviewer_id))
    return changed


def invite(db, scope, count, can_refer, referral_id, *, token=None):
    if scope not in SCOPES or not 1 <= count <= 460:
        raise ReviewError("Chọn lượt duyệt và số câu từ 1 đến 460.")
    db.locked("SELECT source_hash FROM dataset WHERE id = 1")
    if referral_id:
        changed = db.execute(
            "UPDATE referrals SET handled_at = ? WHERE id = ? AND handled_at IS NULL",
            (timestamp(), referral_id),
        ).rowcount
        if scope != "live" or changed != 1:
            raise ReviewError("Đề xuất này không còn chờ duyệt.")
    available = db.all(
        "SELECT r.question_id FROM review_items r JOIN questions q ON r.question_id = q.id "
        "WHERE r.scope = ? AND r.reviewer_id IS NULL ORDER BY q.position LIMIT ?",
        (scope, count),
    )
    if len(available) != count:
        raise ReviewError("Lượt duyệt này còn ít câu chưa phân công hơn số đã chọn.")
    reviewer_id = "R-" + secrets.token_hex(8).upper()
    db.execute(
        "INSERT INTO reviewers (id, scope, can_refer) VALUES (?, ?, ?)",
        (reviewer_id, scope, int(can_refer and scope == "live")),
    )
    for row in available:
        db.execute(
            "UPDATE review_items SET reviewer_id = ? WHERE scope = ? AND question_id = ?",
            (reviewer_id, scope, row["question_id"]),
        )
    return issue_link(db, reviewer_id, referral_id, token=token)


def accept_invite(db, token):
    token_hash = digest(token)
    invitation = db.one("SELECT reviewer_id FROM invitations WHERE token_hash = ?", (token_hash,))
    if not invitation:
        return None
    reviewer_id = invitation["reviewer_id"]
    db.locked("SELECT id FROM reviewers WHERE id = ?", (reviewer_id,))
    now = timestamp()
    changed = db.execute(
        "UPDATE invitations SET consumed_at = ? WHERE token_hash = ? "
        "AND consumed_at IS NULL AND expires_at > ?",
        (now, token_hash, now),
    ).rowcount
    if changed != 1:
        return None
    db.execute("UPDATE reviewers SET access_version = access_version + 1 WHERE id = ?", (reviewer_id,))
    return db.one("SELECT id, access_version FROM reviewers WHERE id = ?", (reviewer_id,))


def update_profile(db, person, decision, profile):
    now = timestamp()
    changed = db.execute(
        "UPDATE reviewers SET consent = ?, consent_at = ?, consent_version = ?, full_name = ?, "
        "profession = ?, workplace = ?, profile_revision = profile_revision + 1 "
        "WHERE id = ? AND scope = 'live' AND profile_revision = ?",
        (decision, now, CONSENT_VERSION, profile.get("full_name"), profile.get("profession"),
         profile.get("workplace"), person["id"], person["profile_revision"]),
    ).rowcount
    if changed != 1:
        raise ReviewConflict("Lựa chọn chia sẻ thông tin vừa thay đổi. Hãy tải lại trang.")


def record_event(db, item, actor_id, now):
    snapshot = {field: item[field] for field in ("answer", "answerability", "evidence_ids", "notes", "verified_at")}
    db.execute(
        "INSERT INTO review_events VALUES (?, ?, ?, ?, ?, ?, ?)",
        (item["scope"], item["question_id"], item["revision"], actor_id, item["status"], now,
         json.dumps(snapshot, ensure_ascii=False)),
    )


def save_review(db, item, reviewer_id, revision, status, answer, answerability, evidence_ids, notes):
    now = timestamp()
    if item["status"] == "verified" or item["revision"] != revision:
        raise ReviewConflict("Câu này vừa được cập nhật. Hãy mở lại để xem bản mới.")
    updated = dict(item, status=status, revision=revision + 1, answer=answer, answerability=answerability,
                   evidence_ids=json.dumps(evidence_ids), notes=notes, verified_at=now if status == "verified" else None)
    changed = db.execute(
        "UPDATE review_items SET status = ?, revision = ?, answer = ?, answerability = ?, "
        "evidence_ids = ?, notes = ?, verified_at = ? "
        "WHERE scope = ? AND question_id = ? AND revision = ? AND reviewer_id = ?",
        (status, updated["revision"], answer, answerability, updated["evidence_ids"], notes, updated["verified_at"],
         item["scope"], item["question_id"], revision, reviewer_id),
    ).rowcount
    if changed != 1:
        raise ReviewConflict("Câu này vừa được cập nhật. Hãy mở lại để xem bản mới.")
    record_event(db, updated, reviewer_id, now)


def reopen(db, scope, question_id, revision):
    item = db.locked("SELECT * FROM review_items WHERE scope = ? AND question_id = ?", (scope, question_id))
    if not item or item["status"] not in {"flagged", "verified"} or item["revision"] != revision:
        raise ReviewConflict("Trạng thái câu hỏi đã thay đổi. Hãy tải lại trang.")
    updated = dict(item, status="draft", revision=revision + 1, verified_at=None)
    db.execute(
        "UPDATE review_items SET status = 'draft', revision = ?, verified_at = NULL "
        "WHERE scope = ? AND question_id = ?",
        (updated["revision"], scope, question_id),
    )
    record_event(db, updated, "admin", timestamp())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Import the VacxiS review dataset into the configured database.")
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    try:
        count = import_dataset(database_location(), args.dataset)
    except (ReviewError, OSError) as error:
        parser.exit(1, f"{error}\n")
    print(f"Đã chuẩn bị {count} câu cho hai lượt duyệt.")
