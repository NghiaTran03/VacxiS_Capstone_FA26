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
from store import connect, import_dataset, invite


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
        self.assertNotIn("Nguyễn Văn A", response.text)
        self.assertNotIn("Đơn vị kiểm thử", response.text)
        self.assertEqual(response.headers["cache-control"], "no-store")
        [record] = response.json()
        self.assertTrue(record["VERIFIED"])
        self.assertTrue(record["ANNOTATOR_IDS"][-1].startswith("R-"))
        self.assertEqual(record["RELEVANT_CHUNKS"], [])

        privacy = client.get("/quyen-rieng-tu")
        self.assertIn("Đơn vị kiểm thử", privacy.text)
        profile_revision = field(privacy, "profile_revision")
        profile_action = {"csrf": csrf(privacy), "profile_revision": profile_revision}
        self.assertEqual(client.post("/quyen-rieng-tu", data={**profile_action, "action": "withdraw"}).status_code, 303)
        stale = client.post("/quyen-rieng-tu", data={**profile_action, "action": "share", "full_name": "Bản cũ"})
        self.assertEqual(stale.status_code, 409)
        self.assertNotIn("Nguyễn Văn A", self.admin.get("/admin/history/live/VAX_Q_0001").text)
        self.assertIn("Người duyệt R-", client.get(FIRST).text)

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

    def test_table_preview_and_edited_export_keep_evidence_in_sync(self):
        client = self.join(self.make_invite())
        page = client.get(WITH_TABLE)
        self.assertIn("<table>", page.text)
        self.assertIn('rowspan="2"', page.text)
        self.assertEqual(page.text.count('name="evidence_id"'), 1)
        chunk_id = "gardasil_9_001310178700_c_0009"
        review = {"answerability": "PARTIALLY_ANSWERABLE", "evidence_id": chunk_id}
        self.assertEqual(self.review(client, WITH_TABLE, "confirm", answerability="PARTIALLY_ANSWERABLE").status_code, 422)
        result = self.review(client, WITH_TABLE, "confirm", **review)
        self.assertEqual(result.status_code, 303)
        exported = self.admin.get("/admin/export/live")
        self.assertIn('\n\t\t"QUERY_ID":', exported.text)
        self.assertIn('Câu trả lời mẫu', exported.text)
        [record] = exported.json()
        self.assertEqual(record["REFERENCE_ANSWER"], f"Câu trả lời mẫu [{chunk_id}]")
        self.assertEqual(record["CLAIM_EVIDENCE_MAP"], [])
        self.assertEqual(record["RELEVANT_DOCS"], ["gardasil_9_001310178700"])

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
                           "profile_revision": field(profile, "profile_revision"), "full_name": "Tên sẽ được xóa"})
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
