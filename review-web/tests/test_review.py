"""Exercise the review workflow, access boundaries, and benchmark export."""

import json
import os
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import create_app
from content import render_chunk
from store import ReviewError, connect, import_dataset, invite


PASSWORD = "test-admin-password-for-review-web"
SOURCE_FILE = Path(__file__).resolve().parents[2] / "test_dataset.json"
FIRST = "/review/VAX_Q_0001"
WITH_TABLE = "/review/VAX_Q_0003"


def field(response, name):
    return re.search(rf'name="{name}" value="([A-Za-z0-9_.-]+)"', response.text).group(1)


def csrf(response):
    return field(response, "csrf")


def credentials(response):
    code = re.search(r'id="invite-code"[^>]* value="([A-Z0-9-]+)"', response.text).group(1)
    url = re.search(r'id="invite-url"[^>]* value="([^"]+)"', response.text).group(1)
    return code, url


class ReviewWorkflow(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.location = str(Path(self.temp.name) / "review.sqlite3")
        dataset = json.loads(SOURCE_FILE.read_text(encoding="utf-8-sig"))[:3]
        fixture = Path(self.temp.name) / "fixture.json"
        fixture.write_text(json.dumps(dataset, ensure_ascii=False), encoding="utf-8")
        import_dataset(self.location, fixture)
        self.open_app()

    def open_app(self):
        with patch.dict(os.environ, {"REVIEW_ADMIN_PASSWORD": PASSWORD}, clear=False):
            self.app = create_app(self.location)
        self.admin = TestClient(self.app, follow_redirects=False)
        login = self.admin.get("/admin/login")
        self.admin_csrf = csrf(login)
        self.assertEqual(self.admin_post("/admin/login", {"password": PASSWORD}).status_code, 303)
        self.admin_csrf = csrf(self.admin.get("/admin"))

    def admin_post(self, path, data):
        return self.admin.post(path, data={"csrf": self.admin_csrf, **data})

    def create_invite(self, scope, count, **extra):
        operation = field(self.admin.get(f"/admin?scope={scope}"), "operation")
        response = self.admin_post("/admin/invite", {"scope": scope, "count": count, "operation": operation, **extra})
        self.assertEqual(response.status_code, 200, response.text)
        return response

    def make_invite(self, scope="live", count=3, **extra):
        _, url = credentials(self.create_invite(scope, count, **extra))
        return url.rsplit("#", 1)[-1]

    def join(self, token, expected=303):
        client = TestClient(self.app, follow_redirects=False)
        response = self.open_invite(client, token, action="enter")
        if response.status_code == 200 and expected == 303:
            response = self.complete_invite(client, response, "skip")
        self.assertEqual(response.status_code, expected)
        return client

    def open_invite(self, client, token, action="open"):
        return client.post("/join", data={"token": token, "action": action, "csrf": csrf(client.get("/join"))})

    def complete_invite(self, client, page, action, **extra):
        data = {"csrf": csrf(page), "token": field(page, "token"), "action": action, **extra}
        if action in {"skip", "share"}:
            data["permission_intent"] = field(page, "permission_intent")
        return client.post("/join", data=data)

    def review(self, client, path, action, revision=0, **extra):
        data = {"answer": "Câu trả lời mẫu", "answerability": "UNANSWERABLE", "notes": "",
                "csrf": csrf(client.get(path)), "revision": revision, "action": action, **extra}
        return client.post(path, data=data)

    def test_deepeval_export_and_upgrade_preserve_review(self):
        temp = Path(self.temp.name)
        self.location = str(temp / "eval.sqlite3")
        contexts = [
            "[Tài liệu A | Chống chỉ định]\n[CHỦ ĐỀ: Vắc xin A] > [MỤC: CHỐNG CHỈ ĐỊNH]\nĐoạn đối chiếu A",
            "[Tài liệu B | Chống chỉ định]\n[CHỦ ĐỀ: Vắc xin B] > [MỤC: CHỐNG CHỈ ĐỊNH]\nĐoạn đối chiếu B",
        ]
        golden = {
            "input": "Câu hỏi DeepEval", "expected_output": "Đáp án gốc", "context": contexts,
            "additional_metadata": {
                "query_id": "eval_0001", "answerability": "ANSWERABLE", "complexity": "easy",
                "multi_hop_type": None, "query_style": "formal", "source_chunk_ids": ["a_c_0000", "b_c_0000"],
                "doc_ids": ["a", "b"], "doc_titles": ["Tài liệu A", "Tài liệu B"],
            },
        }
        converted = [{
            "QUERY_ID": "eval_0001", "QUERY": golden["input"], "REFERENCE_ANSWER": golden["expected_output"],
            "ANSWERABILITY": "ANSWERABLE", "QUERY_COMPLEXITY": "easy", "MULTI_HOP_TYPE": None,
            "QUERY_STYLE": "formal", "RELEVANT_DOCS": ["a", "b"], "ANNOTATOR_IDS": ["GEN"],
            "RELEVANT_CHUNKS": [{"CHUNK_ID": identifier, "TEXT": text, "RELEVANCE_SCORE": 3}
                                for identifier, text in zip(["a_c_0000", "b_c_0000"], contexts, strict=True)],
        }]
        old_path = temp / "converted.json"
        old_path.write_text(json.dumps(converted), encoding="utf-8")
        import_dataset(self.location, old_path)
        self.open_app()
        self.assertEqual(self.admin.get("/admin/export/live").json(), {"metadata": {"total_generated": 1}, "goldens": []})

        client = self.join(self.make_invite(count=1))
        path = "/review/eval_0001"
        page = client.get(path)
        self.assertIn("Đáp án gốc", page.text)
        self.assertIn("Đoạn đối chiếu B", page.text)
        self.assertIn('Đoạn trích trong tài liệu (2)', page.text)
        self.assertIn('<a href="#source-1"><strong>Vắc xin A</strong> · Chống chỉ định</a>', page.text)
        self.assertIn('<a href="#source-2"><strong>Vắc xin B</strong> · Chống chỉ định</a>', page.text)
        self.assertLess(page.text.index('href="#source-2"'), page.text.index('id="source-1"'))
        self.assertIn('<details class="disclosure source" open>', page.text)
        self.assertIn('<summary id="source-2"><strong>Vắc xin B</strong> · Chống chỉ định</summary>', page.text)
        self.assertIn('<div class="source-text"><p>Đoạn đối chiếu B</p>', page.text)
        self.assertIn('<pre>[Tài liệu B | Chống chỉ định]', page.text)
        self.assertEqual(page.text.count('href="#source-index"'), 2)
        self.assertEqual(self.review(client, path, "confirm", answerability="ANSWERABLE", evidence_id="b_c_0000",
                                     notes="Ghi chú cho nhóm").status_code, 303)
        response = self.admin.get("/admin/export/live")
        [exported] = response.json()["goldens"]
        self.assertEqual(set(exported), set(golden))
        self.assertNotIn("RELEVANCE_SCORE", response.text)
        self.assertEqual(exported["input"], golden["input"])
        self.assertEqual(exported["expected_output"], "Câu trả lời mẫu")
        self.assertEqual(exported["context"], contexts[1:])
        metadata = exported["additional_metadata"]
        self.assertEqual(metadata["source_chunk_ids"], ["b_c_0000"])
        self.assertEqual(metadata["doc_ids"], ["b"])
        self.assertEqual(metadata["doc_titles"], ["Tài liệu B"])
        self.assertTrue(metadata["VERIFIED"])
        self.assertTrue(metadata["ADJUDICATED"])
        self.assertEqual(metadata["VERIFICATION_NOTES"], "Ghi chú cho nhóm\n\nCâu trả lời sau chỉnh sửa:\nCâu trả lời mẫu")

        prepared_path = temp / "prepared.json"
        source_metadata = {"total_generated": 1, "model": "test-model"}
        unrelated = dict(golden, input="Câu hỏi khác")
        prepared_path.write_text(json.dumps({"metadata": source_metadata, "goldens": [unrelated]}), encoding="utf-8")
        with self.assertRaises(ReviewError):
            import_dataset(self.location, prepared_path, upgrade_eval=True)
        self.assertEqual(self.admin.get("/admin/export/live").json(), response.json())

        prepared_path.write_text(json.dumps({"metadata": source_metadata, "goldens": [golden]}), encoding="utf-8")
        with self.assertRaises(ReviewError):
            import_dataset(self.location, prepared_path)
        import_dataset(self.location, prepared_path, upgrade_eval=True)
        import_dataset(self.location, prepared_path)
        self.assertIn("Đoạn đối chiếu B", client.get(path).text)
        self.assertEqual(self.admin.get("/admin/export/live").json(), {"metadata": source_metadata, "goldens": [exported]})

    def test_invitation_can_be_entered_on_home_or_opened_as_link_once(self):
        home = TestClient(self.app, follow_redirects=False).get("/")
        self.assertIn('name="token"', home.text)
        self.assertIn('href="/admin/login"', home.text)
        self.assertEqual(self.admin.get("/admin/login").headers["location"], "/admin")

        issued = self.create_invite("live", 1)
        code, url = credentials(issued)
        self.assertRegex(code, r"^[0-9A-HJKMNP-TV-Z]{4}(?:-[0-9A-HJKMNP-TV-Z]{4}){2}$")
        self.assertIn("Sao chép lời mời", issued.text)
        entered = code.lower().replace("-", " ").translate(str.maketrans({"0": "o", "1": "l"}))
        client = self.join(f"  {entered}  ")
        self.assertEqual(client.get(FIRST).status_code, 200)
        self.join(url, expected=400)

        other_code, other_url = credentials(self.create_invite("live", 1))
        other = self.join(other_url)
        self.assertEqual(other.get(FIRST).status_code, 404)
        self.join(other_code, expected=400)

    def test_reissued_invitation_replaces_previous_access_and_legacy_links_work(self):
        issued = self.create_invite("live", 1)
        previous_code, previous_url = credentials(issued)
        with connect(self.location) as db:
            reviewer = db.one("SELECT id FROM reviewers WHERE scope = 'live'")
        operation = re.findall(r'name="operation" value="([a-f0-9]{32})"', issued.text)[-1]
        replacement = self.admin_post("/admin/reissue", {"reviewer_id": reviewer["id"], "operation": operation})
        self.assertEqual(replacement.status_code, 200)
        code, _ = credentials(replacement)
        self.join(previous_code, expected=400)
        self.join(previous_url, expected=400)
        self.assertEqual(self.join(code).get(FIRST).status_code, 200)
        with connect(self.location, write=True) as db:
            legacy_token = invite(db, "test", 1, False, None)
        self.assertIn("Thử nghiệm nội bộ", self.join(legacy_token).get(FIRST).text)

    def test_live_permission_profile_and_withdrawal(self):
        token = self.make_invite()
        client = TestClient(self.app, follow_redirects=False)
        with patch("app.CONSENT_VERSION", 1):
            old_notice = self.open_invite(client, token)
        self.assertEqual(self.complete_invite(client, old_notice, "share", full_name="Tên nhập ở bản cũ").status_code, 409)
        notice = self.open_invite(client, token)
        self.assertEqual(notice.status_code, 200)
        self.assertIn("Nhóm đồ án xin phép", notice.text)
        self.assertIn("Bỏ qua và bắt đầu duyệt", notice.text)
        self.assertIn('name="full_name"', notice.text)
        self.assertNotIn('/admin/login', notice.text)
        self.assertEqual(client.get(FIRST).status_code, 401)
        self.assertEqual(self.admin.get("/admin/export/live").json(), [])

        result = self.complete_invite(client, notice, "share", full_name="Nguyễn Văn A", profession="doctor",
                                      workplace="Đơn vị kiểm thử")
        self.assertEqual(result.status_code, 303)
        self.assertEqual(result.headers["location"], FIRST)
        self.assertNotIn('name="full_name"', client.get(FIRST).text)
        self.assertEqual(self.review(client, FIRST, "confirm").status_code, 303)
        self.assertIn("Bác sĩ Nguyễn Văn A", client.get(FIRST).text)
        response = self.admin.get("/admin/export/live")
        self.assertNotIn("Đơn vị kiểm thử", response.text)
        self.assertEqual(response.headers["cache-control"], "no-store")
        [record] = response.json()
        self.assertTrue(record["VERIFIED"])
        self.assertEqual(record["ANNOTATOR_IDS"][-1], "Bác sĩ Nguyễn Văn A")
        self.assertTrue(record["ADJUDICATED"])
        self.assertEqual(record["VERIFICATION_NOTES"], "Câu trả lời sau chỉnh sửa:\nCâu trả lời mẫu")
        self.assertEqual(record["RELEVANT_CHUNKS"], [])

        with connect(self.location, write=True) as db:
            db.execute("UPDATE reviewers SET consent_version = 1 WHERE scope = 'live'")
        [legacy] = self.admin.get("/admin/export/live").json()
        self.assertTrue(legacy["ANNOTATOR_IDS"][-1].startswith("R-"))
        privacy = client.get("/quyen-rieng-tu")
        self.assertIn("Đơn vị kiểm thử", privacy.text)
        renewal = {"csrf": csrf(privacy), "profile_revision": field(privacy, "profile_revision"), "action": "share",
                   "full_name": "Nguyễn Văn A", "profession": "doctor", "workplace": "Đơn vị kiểm thử"}
        self.assertEqual(client.post("/quyen-rieng-tu", data=renewal).status_code, 409)
        renewal["consent_version"] = field(privacy, "consent_version")
        self.assertEqual(client.post("/quyen-rieng-tu", data=renewal).status_code, 303)
        [renewed] = self.admin.get("/admin/export/live").json()
        self.assertEqual(renewed["ANNOTATOR_IDS"][-1], "Bác sĩ Nguyễn Văn A")

        privacy = client.get("/quyen-rieng-tu")
        profile_revision = field(privacy, "profile_revision")
        profile_action = {"csrf": csrf(privacy), "profile_revision": profile_revision}
        self.assertEqual(client.post("/quyen-rieng-tu", data={**profile_action, "action": "withdraw"}).status_code, 303)
        stale = client.post("/quyen-rieng-tu", data={**profile_action, "action": "share", "full_name": "Bản cũ"})
        self.assertEqual(stale.status_code, 409)
        self.assertNotIn("Nguyễn Văn A", self.admin.get("/admin/history/live/VAX_Q_0001").text)
        self.assertIn("Người duyệt R-", client.get(FIRST).text)
        [withdrawn] = self.admin.get("/admin/export/live").json()
        self.assertTrue(withdrawn["ANNOTATOR_IDS"][-1].startswith("R-"))

    def test_test_scope_verifies_without_profile_and_never_exports_live(self):
        token = self.make_invite("test")
        client = TestClient(self.app, follow_redirects=False)
        opened = self.open_invite(client, token)
        self.assertNotIn('name="full_name"', opened.text)
        self.assertNotIn('/admin/login', opened.text)
        self.assertEqual(self.complete_invite(client, opened, "begin").status_code, 303)
        self.assertIn("Thử nghiệm nội bộ", client.get(FIRST).text)
        result = self.review(client, FIRST, "confirm", full_name="Dữ liệu giả", profession="doctor")
        self.assertEqual(result.status_code, 303)
        self.assertNotIn('name="full_name"', client.get("/quyen-rieng-tu").text)
        self.assertEqual(self.admin.get("/admin/export/live").json(), [])
        self.assertEqual(len(self.admin.get("/admin/export/test").json()), 1)
        with connect(self.location) as db:
            self.assertIsNone(db.one("SELECT consent FROM reviewers WHERE scope = 'test'")["consent"])

    def test_assignment_link_reuse_revision_conflicts_and_reopen(self):
        token = self.make_invite(count=1)
        client = self.join(token)
        self.join(token, expected=400)
        self.assertEqual(client.get(WITH_TABLE).status_code, 404)
        self.assertEqual(client.get("/admin/export/live").status_code, 403)
        self.assertEqual(client.post(FIRST, data={"action": "confirm"}).status_code, 403)
        self.assertEqual(self.review(client, FIRST, "draft").status_code, 303)
        self.assertEqual(self.review(client, FIRST, "draft").status_code, 409)
        self.assertEqual(self.review(client, FIRST, "confirm", revision=1).status_code, 303)
        history = self.admin.get("/admin/history/live/VAX_Q_0001")
        self.assertIn("Lịch sử duyệt", history.text)
        response = self.admin_post("/admin/reopen", {"scope": "live", "identifier": "VAX_Q_0001", "revision": 2})
        self.assertEqual(response.status_code, 303)
        self.assertIn("Đang duyệt", client.get(FIRST).text)
        self.assertEqual(self.admin.get("/admin/export/live").json(), [])

    def test_return_confirmed_question_to_unassigned_pool(self):
        client = self.join(self.make_invite(count=1))
        self.assertEqual(self.review(client, FIRST, "confirm", answer="Bản duyệt nhầm", notes="Ghi chú thử").status_code, 303)
        self.assertEqual(len(self.admin.get("/admin/export/live").json()), 1)
        history_path = "/admin/history/live/VAX_Q_0001"
        self.assertIn("Trả về chưa phân công", self.admin.get(history_path).text)

        recovery = {"scope": "live", "identifier": "VAX_Q_0001", "revision": 1, "action": "unassign"}
        self.assertEqual(self.admin_post("/admin/reopen", recovery).status_code, 303)
        self.assertEqual(self.admin_post("/admin/reopen", recovery).status_code, 409)
        self.assertEqual(self.admin.get("/admin/export/live").json(), [])
        self.assertIn("3 câu chưa phân công", self.admin.get("/admin").text)
        self.assertEqual(client.get(FIRST).status_code, 404)
        self.assertIn("Bản duyệt nhầm", self.admin.get(history_path).text)

        new_reviewer = self.join(self.make_invite(count=1))
        question = new_reviewer.get(FIRST)
        self.assertEqual(question.status_code, 200)
        self.assertNotIn("Bản duyệt nhầm", question.text)
        self.assertNotIn("Ghi chú thử", question.text)
        with connect(self.location) as db:
            assigned = db.one("SELECT reviewer_id FROM review_items WHERE scope = 'live' AND question_id = 'VAX_Q_0001'")
        dashboard = self.admin.get("/admin")
        operations = re.findall(r'name="operation" value="([a-f0-9]{32})"', dashboard.text)
        released = self.admin_post("/admin/release", {"reviewer_id": assigned["reviewer_id"], "operation": operations[-1]})
        self.assertEqual(released.status_code, 200)
        self.assertIn("3 câu chưa phân công", released.text)

    def test_table_preview_and_edited_export_keep_evidence_in_sync(self):
        client = self.join(self.make_invite())
        page = client.get(WITH_TABLE)
        self.assertIn("<table>", page.text)
        self.assertIn('rowspan="2"', page.text)
        self.assertEqual(page.text.count('name="evidence_id"'), 1)
        chunk_id = "gardasil_9_001310178700_c_0009"
        review = {"answerability": "PARTIALLY_ANSWERABLE", "evidence_id": chunk_id}
        self.assertEqual(self.review(client, WITH_TABLE, "confirm", answerability="PARTIALLY_ANSWERABLE").status_code, 422)
        result = self.review(client, WITH_TABLE, "confirm", notes="Cần lưu ý cách diễn đạt.", **review)
        self.assertEqual(result.status_code, 303)
        exported = self.admin.get("/admin/export/live")
        self.assertIn('\n\t\t"QUERY_ID":', exported.text)
        self.assertIn('Câu trả lời mẫu', exported.text)
        [record] = exported.json()
        self.assertEqual(record["REFERENCE_ANSWER"], f"Câu trả lời mẫu [{chunk_id}]")
        self.assertEqual(record["CLAIM_EVIDENCE_MAP"], [])
        self.assertEqual(record["RELEVANT_DOCS"], ["gardasil_9_001310178700"])
        self.assertTrue(record["ADJUDICATED"])
        self.assertEqual(record["VERIFICATION_NOTES"],
                         "Cần lưu ý cách diễn đạt.\n\nCâu trả lời sau chỉnh sửa:\nCâu trả lời mẫu")

    def test_export_distinguishes_confirmation_from_notes_without_answer_edits(self):
        client = self.join(self.make_invite(count=1))
        with connect(self.location) as db:
            original = json.loads(db.one("SELECT original FROM questions WHERE id = 'VAX_Q_0001'")["original"])
        answer = original["REFERENCE_ANSWER"]
        self.assertEqual(self.review(client, FIRST, "confirm", answer=answer, notes="  ").status_code, 303)
        [confirmed] = self.admin.get("/admin/export/live").json()
        self.assertTrue(confirmed["VERIFIED"])
        self.assertFalse(confirmed["ADJUDICATED"])
        self.assertIsNone(confirmed["VERIFICATION_NOTES"])

        reopened = self.admin_post("/admin/reopen", {"scope": "live", "identifier": "VAX_Q_0001", "revision": 1})
        self.assertEqual(reopened.status_code, 303)
        result = self.review(client, FIRST, "confirm", revision=2, answer=answer, notes="Đối chiếu xong.")
        self.assertEqual(result.status_code, 303)
        [annotated] = self.admin.get("/admin/export/live").json()
        self.assertTrue(annotated["ADJUDICATED"])
        self.assertEqual(annotated["VERIFICATION_NOTES"], "Đối chiếu xong.")

    def test_referral_needs_admin_and_assigns_a_distinct_account(self):
        requester = self.join(self.make_invite(count=1, can_refer="yes"))
        consent = requester.get("/gioi-thieu")
        self.assertEqual(consent.status_code, 200)
        self.assertEqual(requester.post("/gioi-thieu", data={"csrf": csrf(consent), "label": "reviewer-2"}).status_code, 303)
        with connect(self.location) as db:
            referral = db.one("SELECT id FROM referrals")
        colleague = self.join(self.make_invite(count=1, referral_id=referral["id"]))
        self.assertEqual(colleague.get(FIRST).status_code, 404)
        self.assertEqual(colleague.get("/gioi-thieu").status_code, 403)

    def test_retaking_permission_in_another_tab_preserves_opt_out(self):
        token = self.make_invite()
        client = TestClient(self.app, follow_redirects=False)
        other = TestClient(self.app, follow_redirects=False)
        early = self.open_invite(client, token)
        late = self.open_invite(other, token)
        result = self.complete_invite(client, early, "skip")
        self.assertEqual(result.status_code, 303)
        self.assertEqual(result.headers["location"], FIRST)
        stale = self.complete_invite(other, late, "share", full_name="Bản cũ")
        self.assertEqual(stale.status_code, 400)
        self.assertEqual(self.admin.get("/admin/export/live").json(), [])
        with connect(self.location) as db:
            self.assertEqual(db.one("SELECT consent, full_name FROM reviewers"), {"consent": "declined", "full_name": None})

    def test_opening_live_invitation_keeps_admin_session_until_continue(self):
        token = self.make_invite()
        welcome = self.admin.get("/join")
        self.assertNotIn('/admin/login', welcome.text)
        notice = self.open_invite(self.admin, token)
        self.assertIn("chuyển phiên đăng nhập", notice.text)
        self.assertNotIn('/admin/login', notice.text)
        self.assertEqual(self.admin.get("/admin").status_code, 200)
        self.assertEqual(self.complete_invite(self.admin, notice, "skip").status_code, 303)
        self.assertEqual(self.admin.get("/admin").status_code, 403)

    def test_legacy_live_session_gets_permission_before_its_next_review(self):
        client = self.join(self.make_invite())
        with connect(self.location, write=True) as db:
            db.execute("UPDATE reviewers SET consent = NULL")
        path = "/review/onboarding/start"
        self.assertEqual(client.get(FIRST).headers["location"], path)
        start = client.get(path)
        self.assertIn('name="full_name"', start.text)
        old_form = client.post(FIRST, data={"csrf": csrf(start), "action": "confirm", "revision": 0,
                                            "answer": "Câu trả lời mẫu", "answerability": "UNANSWERABLE"})
        self.assertEqual(old_form.headers["location"], path)
        self.assertEqual(self.admin.get("/admin/export/live").json(), [])
        submitted = client.post(path, data={"csrf": csrf(start), "action": "skip",
                                           "permission_intent": field(start, "permission_intent")})
        self.assertEqual(submitted.headers["location"], FIRST)
        self.assertEqual(client.get(FIRST).status_code, 200)

    def test_invite_retry_does_not_reserve_again_and_release_revokes_link(self):
        operation = field(self.admin.get("/admin"), "operation")
        payload = {"scope": "live", "count": 1, "operation": operation}
        invited = self.admin_post("/admin/invite", payload)
        token = re.search(r'/join#([A-Za-z0-9_-]{43})', invited.text).group(1)
        self.assertEqual(self.admin_post("/admin/invite", payload).status_code, 422)
        with connect(self.location) as db:
            [person] = db.all("SELECT id FROM reviewers")
        operations = re.findall(r'name="operation" value="([a-f0-9]{32})"', invited.text)
        self.assertEqual(self.admin_post("/admin/release", {"reviewer_id": person["id"], "operation": operations[-1]}).status_code, 200)
        self.join(token, expected=400)
        other = self.join(self.make_invite())
        self.assertEqual(other.get(FIRST).status_code, 200)

        profile = other.get("/quyen-rieng-tu")
        saved = other.post("/quyen-rieng-tu", data={"csrf": csrf(profile), "action": "share",
                           "profile_revision": field(profile, "profile_revision"), "full_name": "Tên sẽ được xóa",
                           "consent_version": field(profile, "consent_version")})
        self.assertEqual(saved.status_code, 303)
        with connect(self.location) as db:
            active = db.one("SELECT id FROM reviewers WHERE consent = 'accepted'")
        operations = re.findall(r'name="operation" value="([a-f0-9]{32})"', self.admin.get("/admin").text)
        self.assertEqual(self.admin_post("/admin/release", {"reviewer_id": active["id"], "operation": operations[-1]}).status_code, 200)
        self.assertEqual(other.get(FIRST).status_code, 401)
        with connect(self.location) as db:
            self.assertEqual(db.one("SELECT consent, full_name FROM reviewers WHERE id = ?", (active["id"],)),
                             {"consent": "withdrawn", "full_name": None})


class SourceRendering(unittest.TestCase):
    def test_plain_text_repairs_are_display_only_and_html_is_sanitized(self):
        page_text = "8\n\n--- TRANG 9 ---\n\nThường gặp: văn bản ($\\le4,5\\%$)."
        display = render_chunk({"CHUNK_ID": "sample", "TEXT": page_text})
        self.assertIn("Trang 9", display["html"])
        self.assertIn("≤ 4,5%", display["html"])
        self.assertEqual(display["raw"], page_text)
        source = '<table><tr><td rowspan="2" style="display:none">Nguồn</td></tr></table>'
        sanitized = render_chunk({"CHUNK_ID": "sample", "TEXT": source})["html"]
        self.assertIn('rowspan="2"', sanitized)
        self.assertNotIn("style=", sanitized)


if __name__ == "__main__":
    unittest.main()
