"""VacxiS reviewer and administrator web routes."""

from base64 import urlsafe_b64encode
import hmac
import json
import os
import re
import secrets
import time
from typing import Annotated
from urllib.parse import urlsplit

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from itsdangerous import URLSafeSerializer
from starlette.datastructures import FormData
from starlette.middleware.sessions import SessionMiddleware

from content import ANSWERABILITY, PROFESSIONS, STATUS, export_record, remove_citations, render_chunk, reviewer_label, score_three
from store import (
    BASE_DIR, SCOPES, ReviewConflict, ReviewError, accept_invite, connect, database_location, invite, issue_link,
    release_unstarted, reopen, reserve_operation, save_review, timestamp, update_profile,
)


INVITE_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
INVITE_TOKEN = re.compile(r"[A-Za-z0-9_-]{43}")
INVITE_CODE = re.compile(rf"[{INVITE_ALPHABET}]{{12}}")
INVITE_ALIASES = str.maketrans({"O": "0", "I": "1", "L": "1"})


def token_for_code(code, key):
    signed = hmac.digest(key, f"vacxis-invite-code-v1:{code}".encode(), "sha256")
    return urlsafe_b64encode(signed).decode().rstrip("=")


def new_invitation(key):
    code = "".join(secrets.choice(INVITE_ALPHABET) for _ in range(12))
    return "-".join((code[:4], code[4:8], code[8:])), token_for_code(code, key)


def invitation_token(credential, key):
    if not isinstance(credential, str) or len(credential) > 2048:
        return None
    value = credential.strip()
    token = value.rsplit("#", 1)[-1]
    if INVITE_TOKEN.fullmatch(token):
        return token
    code = re.sub(r"[-\s]", "", value).upper().translate(INVITE_ALIASES)
    if INVITE_CODE.fullmatch(code):
        return token_for_code(code, key)
    return None


def redirect(path):
    return RedirectResponse(path, status_code=303)


def equal(left, right):
    return isinstance(left, str) and isinstance(right, str) and secrets.compare_digest(left.encode(), right.encode())


async def checked_form(request: Request):
    form = await request.form(max_fields=30, max_files=0)
    if not equal(form.get("csrf"), request.session.get("csrf")):
        raise HTTPException(403, "Phiên làm việc hết hạn. Hãy tải lại trang rồi thử lại.")
    return form


PostForm = Annotated[FormData, Depends(checked_form)]


def integer(value, message):
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ReviewError(message) from None


def profile_from(form):
    profile = {field: str(form.get(field, "")).strip() or None for field in ("full_name", "profession", "workplace")}
    if profile["profession"] and profile["profession"] not in PROFESSIONS:
        raise ReviewError("Hãy chọn một vai trò chuyên môn trong danh sách.")
    if len(profile["full_name"] or "") > 120 or len(profile["workplace"] or "") > 200:
        raise ReviewError("Họ tên hoặc nơi công tác dài hơn giới hạn của biểu mẫu.")
    return profile


def reviewer(db, request, *, lock=False):
    identifier = request.session.get("reviewer_id")
    if not identifier:
        return None
    select = db.locked if lock else db.one
    person = select("SELECT * FROM reviewers WHERE id = ?", (identifier,))
    if person and person["access_version"] == request.session.get("access_version"):
        return person
    request.session.clear()
    return None


def require_reviewer(db, request, *, lock=False):
    person = reviewer(db, request, lock=lock)
    if not person:
        raise HTTPException(401, "Hãy mở lời mời hoặc liên hệ nhóm đồ án để đăng nhập lại.")
    return person


def is_admin(request):
    return request.session.get("admin_until", 0) > time.time()


def require_admin(request):
    if not is_admin(request):
        raise HTTPException(403, "Trang này dành cho nhóm đồ án.")


def get_item(db, person, identifier):
    item = db.one(
        "SELECT r.*, q.query, q.original FROM review_items r JOIN questions q ON r.question_id = q.id "
        "WHERE r.scope = ? AND r.reviewer_id = ? AND r.question_id = ?",
        (person["scope"], person["id"], identifier),
    )
    if not item:
        raise HTTPException(404, "Câu hỏi này chưa được giao cho bạn.")
    return item


def next_item(db, person, excluded=""):
    return db.one(
        "SELECT r.question_id FROM review_items r JOIN questions q ON r.question_id = q.id "
        "WHERE r.scope = ? AND r.reviewer_id = ? AND r.question_id <> ? AND r.status IN ('draft', 'pending') "
        "ORDER BY CASE WHEN r.status = 'draft' THEN 0 ELSE 1 END, q.position LIMIT 1",
        (person["scope"], person["id"], excluded),
    )


def create_app(location=None):
    deployed = bool(os.environ.get("VERCEL"))
    location = location or database_location()
    secret = os.environ.get("REVIEW_SECRET_KEY", "")
    password = os.environ.get("REVIEW_ADMIN_PASSWORD", "")
    vercel_host = os.environ.get("VERCEL_BRANCH_URL") if os.environ.get("VERCEL_ENV") == "preview" else os.environ.get("VERCEL_PROJECT_PRODUCTION_URL")
    public_url = (os.environ.get("REVIEW_PUBLIC_URL") or (f"https://{vercel_host}" if vercel_host else "http://127.0.0.1:8000")).rstrip("/")
    origin = urlsplit(public_url)
    if not origin.hostname or origin.path or origin.query or origin.fragment or origin.username or origin.password:
        raise RuntimeError("REVIEW_PUBLIC_URL must be an HTTP(S) origin without a path.")
    if origin.scheme != "https" and (deployed or origin.scheme != "http" or origin.hostname not in {"127.0.0.1", "localhost"}):
        raise RuntimeError("Use HTTPS outside localhost.")
    if (secret and len(secret) < 32) or (password and len(password) < 24):
        raise RuntimeError("Use a REVIEW_SECRET_KEY of at least 32 characters and a REVIEW_ADMIN_PASSWORD of at least 24 characters.")
    if origin.scheme == "https" and (not secret or not password or secret == password):
        raise RuntimeError("Set distinct REVIEW_SECRET_KEY and REVIEW_ADMIN_PASSWORD values.")

    secret = secret or secrets.token_urlsafe(48)
    invitation_key = secret.encode()
    consent_signer = URLSafeSerializer(secret, salt="vacxis-consent-v1")
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(SessionMiddleware, secret_key=secret,
                       https_only=origin.scheme == "https", max_age=14 * 24 * 60 * 60, session_cookie="vacxis_review")
    if not deployed:
        app.mount("/assets", StaticFiles(directory=BASE_DIR / "public" / "assets"), name="assets")
    templates = Jinja2Templates(directory=BASE_DIR / "templates")

    def page(request, name, *, code=200, **context):
        return templates.TemplateResponse(request=request, name=f"{name}.html", status_code=code, context={
            "csrf": request.session.setdefault("csrf", secrets.token_urlsafe(32)),
            "admin": is_admin(request), "statuses": STATUS, "scopes": SCOPES, "professions": PROFESSIONS,
            "labels": ANSWERABILITY, "reviewer_label": reviewer_label,
            "signed_in": bool(request.session.get("reviewer_id")) or is_admin(request), **context,
        })

    @app.middleware("http")
    async def protect(request, call_next):
        if request.method == "POST" and (
            request.headers.get("sec-fetch-site") == "cross-site"
            or not request.headers.get("content-length", "").isdigit()
            or int(request.headers["content-length"]) > 131072
        ):
            response = HTMLResponse("Yêu cầu không hợp lệ hoặc quá dài.", status_code=400)
        else:
            response = await call_next(request)
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Robots-Tag"] = "noindex, nofollow"
        response.headers["Content-Security-Policy"] = (
            "default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self'; form-action 'self'; "
            "base-uri 'none'; frame-ancestors 'none'"
        )
        if not request.url.path.startswith("/assets/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(HTTPException)
    async def http_error(request, error):
        return page(request, "error", code=error.status_code, message=error.detail)

    @app.get("/")
    def home(request: Request):
        if is_admin(request):
            return redirect("/admin")
        if request.session.get("reviewer_id"):
            return redirect("/review")
        return page(request, "welcome")

    @app.get("/join")
    def join_page(request: Request):
        return page(request, "welcome", joining=True)

    @app.post("/join")
    def join(request: Request, form: PostForm):
        token = invitation_token(form.get("token"), invitation_key)
        person = None
        if token:
            with connect(location, write=True) as db:
                person = accept_invite(db, token)
        if not person:
            return page(request, "welcome", joining=True, code=400,
                        error="Hãy kiểm tra mã hoặc liên kết. Nếu lời mời đã dùng hoặc hết hạn, hãy xin nhóm đồ án cấp lại.")
        request.session.clear()
        request.session.update(reviewer_id=person["id"], access_version=person["access_version"])
        return redirect("/review")

    @app.get("/admin/login")
    def admin_login_page(request: Request):
        if is_admin(request):
            return redirect("/admin")
        return page(request, "welcome", login=True)

    @app.post("/admin/login")
    def admin_login(request: Request, form: PostForm):
        if not password or not equal(form.get("password"), password):
            return page(request, "welcome", login=True, code=403, error="Mật khẩu quản trị không đúng.")
        request.session.clear()
        request.session["admin_until"] = int(time.time()) + 8 * 60 * 60
        return redirect("/admin")

    @app.post("/logout")
    def logout(request: Request, form: PostForm):
        if request.session.get("reviewer_id"):
            with connect(location, write=True) as db:
                db.execute("UPDATE reviewers SET access_version = access_version + 1 WHERE id = ? AND access_version = ?",
                           (request.session["reviewer_id"], request.session.get("access_version")))
        request.session.clear()
        return redirect("/")

    @app.get("/review")
    def review_list(request: Request, status: str = "", q: str = ""):
        with connect(location) as db:
            person = require_reviewer(db, request)
            rows = db.all(
                "SELECT r.question_id, r.status, q.query FROM review_items r JOIN questions q ON q.id = r.question_id "
                "WHERE r.scope = ? AND r.reviewer_id = ? ORDER BY q.position",
                (person["scope"], person["id"]),
            )
            next_row = next_item(db, person)
        search = q.strip()[:120].casefold()
        filtered = [row for row in rows if (not status or row["status"] == status)
                    and (search in row["query"].casefold() or search in row["question_id"].casefold())]
        return page(request, "queue", person=person, rows=filtered, total=len(rows),
                    done=sum(row["status"] == "verified" for row in rows), next_row=next_row,
                    status=status, search=q[:120], message=request.session.pop("message", ""))

    def permission_intent(person, item):
        return consent_signer.dumps([person["id"], person["access_version"], person["profile_revision"],
                                     item["question_id"], item["revision"]])

    def question_page(request, person, item, *, form=None, ask_profile=False, error=None, code=200):
        original = json.loads(item["original"])
        sources = score_three(original)
        answer = item["answer"] if item["answer"] is not None else remove_citations(original["REFERENCE_ANSWER"])
        label = item["answerability"] or original.get("ANSWERABILITY", "UNANSWERABLE")
        selected = json.loads(item["evidence_ids"]) if item["evidence_ids"] is not None else [c["CHUNK_ID"] for c in sources]
        values = {"answer": answer, "answerability": label, "evidence_ids": selected, "notes": item["notes"]}
        if form is not None:
            values = {field: str(form.get(field, "")) for field in ("answer", "answerability", "notes")}
            values["evidence_ids"] = form.getlist("evidence_id")
        profile_values = {field: form.get(field, "") for field in ("full_name", "profession", "workplace")} if form else {}
        return page(request, "question", person=person, item=item, values=values,
                    original_answer=remove_citations(original["REFERENCE_ANSWER"]),
                    sources=[render_chunk(chunk) for chunk in sources], ask_profile=ask_profile,
                    intent=permission_intent(person, item) if ask_profile else None,
                    profile_values=profile_values, error=error, code=code,
                    message=request.session.pop("message", ""))

    @app.get("/review/{identifier}")
    def question(request: Request, identifier: str):
        with connect(location) as db:
            person = require_reviewer(db, request)
            item = get_item(db, person, identifier)
        return question_page(request, person, item)

    @app.post("/review/{identifier}")
    def review_save(request: Request, identifier: str, form: PostForm):
        action = form.get("action")
        ask_profile = False
        try:
            with connect(location, write=True) as db:
                person = require_reviewer(db, request, lock=True)
                item = get_item(db, person, identifier)
                revision = integer(form.get("revision"), "Tải lại trang trước khi lưu.")
                if item["revision"] != revision or item["status"] == "verified":
                    raise ReviewConflict("Câu này vừa được cập nhật. Hãy mở lại để xem bản mới.")
                actions = {"draft": "draft", "flag": "flagged", "confirm": "verified",
                           "confirm_skip": "verified", "confirm_share": "verified"}
                if action not in actions:
                    raise ReviewError("Hãy chọn một thao tác duyệt.")
                status = actions[action]
                answer = remove_citations(str(form.get("answer", "")))
                notes = str(form.get("notes", "")).strip()
                label = form.get("answerability")
                if len(answer) > 20000 or len(notes) > 3000 or label not in ANSWERABILITY:
                    raise ReviewError("Kiểm tra độ dài câu trả lời, ghi chú và mức độ tài liệu trả lời được.")
                choices = set(form.getlist("evidence_id"))
                original = json.loads(item["original"])
                selected = [chunk["CHUNK_ID"] for chunk in score_three(original) if chunk["CHUNK_ID"] in choices]
                if label == "UNANSWERABLE":
                    selected = []
                if status == "flagged" and not notes:
                    raise ReviewError("Hãy ghi lý do cần xem lại.")
                if status == "verified" and (not answer or (label != "UNANSWERABLE" and not selected)):
                    raise ReviewError("Hãy điền câu trả lời và chọn đoạn làm căn cứ nếu tài liệu trả lời được.")
                ask_profile = status == "verified" and person["scope"] == "live" and person["consent"] is None
                if ask_profile and action == "confirm":
                    return question_page(request, person, item, form=form, ask_profile=True)
                if action in {"confirm_skip", "confirm_share"} and (
                    not ask_profile or not equal(form.get("permission_intent"), permission_intent(person, item))
                ):
                    return question_page(request, person, item, form=form, ask_profile=ask_profile, code=409,
                                         error="Lựa chọn chia sẻ thông tin đã thay đổi. Hãy kiểm tra và xác nhận lại.")
                if ask_profile:
                    decision = "accepted" if action == "confirm_share" else "declined"
                    update_profile(db, person, decision, profile_from(form) if decision == "accepted" else {})
                save_review(db, item, person["id"], revision, status, answer, label, selected, notes)
                following = next_item(db, person, identifier) if status == "verified" else None
        except ReviewConflict as error:
            return question_page(request, person, item, error=str(error), code=409)
        except ReviewError as error:
            return question_page(request, person, item, form=form, ask_profile=ask_profile, error=str(error), code=422)
        request.session["message"] = f"{STATUS[status]} · {identifier}"
        if following:
            return redirect(f"/review/{following['question_id']}")
        return redirect("/review" if status == "verified" else f"/review/{identifier}")

    @app.get("/quyen-rieng-tu")
    def privacy(request: Request):
        with connect(location) as db:
            person = reviewer(db, request) if request.session.get("reviewer_id") else None
        return page(request, "privacy", person=person, email=os.environ.get("REVIEW_PRIVACY_EMAIL", ""),
                    message=request.session.pop("message", ""))

    @app.post("/quyen-rieng-tu")
    def privacy_update(request: Request, form: PostForm):
        with connect(location, write=True) as db:
            person = require_reviewer(db, request, lock=True)
            if person["scope"] != "live":
                raise HTTPException(403, "Lượt thử nghiệm không có hồ sơ thông tin cá nhân.")
            if str(form.get("profile_revision", "")) != str(person["profile_revision"]):
                return page(request, "privacy", person=person, code=409, email=os.environ.get("REVIEW_PRIVACY_EMAIL", ""),
                            error="Hồ sơ vừa thay đổi. Hãy xem lại thông tin hiện tại trước khi lưu.")
            action = form.get("action")
            if action == "withdraw":
                update_profile(db, person, "withdrawn", {})
                request.session["message"] = "Đã xóa thông tin cá nhân khỏi hồ sơ. Bạn vẫn có thể duyệt bằng mã người duyệt."
            elif action == "share":
                try:
                    update_profile(db, person, "accepted", profile_from(form))
                except ReviewError as error:
                    return page(request, "privacy", person=person, error=str(error), code=422,
                                email=os.environ.get("REVIEW_PRIVACY_EMAIL", ""))
                request.session["message"] = "Đã lưu thông tin theo lựa chọn của bạn."
            else:
                raise HTTPException(400, "Thao tác không hợp lệ.")
        return redirect("/quyen-rieng-tu")

    @app.get("/gioi-thieu")
    def refer_page(request: Request):
        with connect(location) as db:
            person = require_reviewer(db, request)
        if not person["can_refer"] or person["scope"] != "live":
            raise HTTPException(403, "Tài khoản này chưa được giao giới thiệu đồng nghiệp.")
        return page(request, "referral", person=person)

    @app.post("/gioi-thieu")
    def refer(request: Request, form: PostForm):
        label = str(form.get("label", "")).strip()
        with connect(location, write=True) as db:
            person = require_reviewer(db, request, lock=True)
            if not person["can_refer"] or person["scope"] != "live":
                raise HTTPException(403, "Tài khoản này chưa được giao giới thiệu đồng nghiệp.")
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", label):
                return page(request, "referral", person=person, code=422,
                            error="Nhập mã tối đa 40 ký tự, gồm chữ không dấu, số hoặc dấu gạch.")
            db.execute("INSERT INTO referrals (id, referred_by, label, created_at) VALUES (?, ?, ?, ?)",
                       (secrets.token_hex(16), person["id"], label, timestamp()))
        request.session["message"] = "Đã gửi đề xuất cho nhóm đồ án."
        return redirect("/review")

    def admin_page(request, scope, *, status="", invitation=None, error=None):
        with connect(location) as db:
            rows = db.all(
                "SELECT r.question_id, r.status, r.revision, r.reviewer_id, r.verified_at, q.query, "
                "u.id, u.full_name, u.profession FROM review_items r JOIN questions q ON q.id = r.question_id "
                "LEFT JOIN reviewers u ON u.id = r.reviewer_id WHERE r.scope = ? ORDER BY q.position", (scope,),
            )
            people = db.all(
                "SELECT u.id, u.full_name, u.profession, u.workplace FROM reviewers u WHERE u.scope = ? "
                "AND EXISTS (SELECT 1 FROM review_items r WHERE r.reviewer_id = u.id) ORDER BY u.id", (scope,),
            )
            referrals = db.all("SELECT * FROM referrals WHERE handled_at IS NULL ORDER BY created_at") if scope == "live" else []
        code, token = invitation or (None, None)
        return page(request, "admin", scope=scope, total=len(rows), people=people, referrals=referrals,
                    available=sum(not row["reviewer_id"] for row in rows),
                    done=sum(row["status"] == "verified" for row in rows), status=status,
                    rows=[row for row in rows if not status or row["status"] == status],
                    invite_operation=secrets.token_hex(16), reviewer_operation=secrets.token_hex(16),
                    invite_url=f"{public_url}/join#{token}" if token else None, invite_code=code,
                    public_url=public_url, error=error, code=422 if error else 200,
                    message=request.session.pop("message", ""))

    @app.get("/admin")
    def admin_dashboard(request: Request, scope: str = "live", status: str = ""):
        require_admin(request)
        if scope not in SCOPES:
            raise HTTPException(404, "Lượt duyệt không tồn tại.")
        return admin_page(request, scope, status=status)

    @app.post("/admin/invite")
    def admin_invite(request: Request, form: PostForm):
        require_admin(request)
        scope = form.get("scope", "live")
        if scope not in SCOPES:
            raise HTTPException(400, "Lượt duyệt không hợp lệ.")
        try:
            with connect(location, write=True) as db:
                reserve_operation(db, form.get("operation"))
                count = integer(form.get("count"), "Nhập số câu cần phân công.")
                code, token = new_invitation(invitation_key)
                invite(db, scope, count, form.get("can_refer") == "yes", form.get("referral_id") or None, token=token)
        except ReviewError as error:
            return admin_page(request, scope, error=str(error))
        return admin_page(request, scope, invitation=(code, token))

    @app.post("/admin/reissue")
    @app.post("/admin/release")
    def manage_assignments(request: Request, form: PostForm):
        require_admin(request)
        reviewer_id = form.get("reviewer_id", "")
        invitation = None
        try:
            with connect(location, write=True) as db:
                person = db.locked("SELECT scope FROM reviewers WHERE id = ?", (reviewer_id,))
                if not person:
                    raise HTTPException(404, "Không tìm thấy người duyệt.")
                reserve_operation(db, form.get("operation"))
                if request.url.path == "/admin/release":
                    count = release_unstarted(db, reviewer_id)
                    request.session["message"] = f"Đã thu hồi {count} câu chưa bắt đầu để phân công lại."
                else:
                    if not db.one("SELECT question_id FROM review_items WHERE reviewer_id = ? LIMIT 1", (reviewer_id,)):
                        raise ReviewError("Người duyệt này không còn câu được giao.")
                    code, token = new_invitation(invitation_key)
                    issue_link(db, reviewer_id, token=token)
                    invitation = (code, token)
        except ReviewError as error:
            return admin_page(request, person["scope"], error=str(error))
        return admin_page(request, person["scope"], invitation=invitation)

    @app.get("/admin/history/{scope}/{identifier}")
    def history(request: Request, scope: str, identifier: str):
        require_admin(request)
        with connect(location) as db:
            item = db.one("SELECT r.*, q.query FROM review_items r JOIN questions q ON q.id = r.question_id "
                          "WHERE r.scope = ? AND r.question_id = ?", (scope, identifier))
            if not item:
                raise HTTPException(404, "Không tìm thấy câu hỏi.")
            history_rows = db.all(
                "SELECT e.*, u.id, u.full_name, u.profession FROM review_events e LEFT JOIN reviewers u ON u.id = e.actor_id "
                "WHERE e.scope = ? AND e.question_id = ? ORDER BY e.revision DESC", (scope, identifier),
            )
        for entry in history_rows:
            entry["snapshot"] = json.loads(entry["snapshot"])
        return page(request, "history", scope=scope, item=item, events=history_rows)

    @app.post("/admin/reopen")
    def admin_reopen(request: Request, form: PostForm):
        require_admin(request)
        scope, identifier = form.get("scope", ""), form.get("identifier", "")
        try:
            with connect(location, write=True) as db:
                reopen(db, scope, identifier, integer(form.get("revision"), "Hãy tải lại trang."))
        except ReviewError as error:
            raise HTTPException(409, str(error)) from error
        return redirect(f"/admin/history/{scope}/{identifier}")

    @app.get("/admin/export/{scope}")
    def export(request: Request, scope: str):
        require_admin(request)
        if scope not in SCOPES:
            raise HTTPException(404, "Lượt duyệt không tồn tại.")
        with connect(location) as db:
            rows = db.all(
                "SELECT r.*, q.original FROM review_items r JOIN questions q ON q.id = r.question_id "
                "WHERE r.scope = ? AND r.status = 'verified' ORDER BY q.position", (scope,),
            )
        filename = "test_dataset_verified.json" if scope == "live" else "internal_review_test_only.json"
        return JSONResponse([export_record(row) for row in rows],
                            headers={"Content-Disposition": f'attachment; filename="{filename}"'})

    return app


app = create_app()
