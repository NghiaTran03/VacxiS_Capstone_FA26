"""Tests for the FastAPI + Vite backend module (vacxis_capstone_fa26.web_vite)."""

from __future__ import annotations

from dataclasses import asdict
import json
import logging
from time import sleep
from unittest.mock import MagicMock, Mock

from fastapi.testclient import TestClient
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

from vacxis_capstone_fa26 import web_vite
from vacxis_capstone_fa26.generation.schemas import GeneratedAnswer
from vacxis_capstone_fa26.llm import LLMError
from vacxis_capstone_fa26.query import Clarification, ClarifyPlan, ConversationMessage
from vacxis_capstone_fa26.web.documents import Documents, FIELDS
from vacxis_capstone_fa26.web_vite.server import (
    build_progress_record,
    conversation_history,
    create_api_app,
)
from vacxis_capstone_fa26.web_vite.storage import ConversationStore


def _parse_sse(body: str) -> list[dict]:
    events = []
    for line in body.splitlines():
        if line.startswith("data: "):
            events.append(json.loads(line[6:]))
    return events


@pytest.fixture
def documents(tmp_path):
    folder = tmp_path / "CLEANED" / "HUNG" / "001"
    folder.mkdir(parents=True)
    (tmp_path / "75_vaccine_names_mapping.json").write_text(
        json.dumps({"001": "Vax A", "002": "Vax B"}),
        encoding="utf-8",
    )
    (folder / "insert.md").write_text(
        "# Full product insert\n\nFirst section.\n\n## Last section\n\nUnindexed original text.\n\n"
        "## Thành phần\n\nThành phần từ tài liệu gốc.\n\n## Chỉ định\n\nNội dung chỉ định.\n\n"
        '<table onclick="alert(1)"><tr><td colspan="2">Original table cell</td></tr></table>\n\n'
        "<script>alert(1)</script>",
        encoding="utf-8",
    )
    (folder / "additional.md").write_text("# Additional source\n\nSecond original file.", encoding="utf-8")
    other = tmp_path / "CLEANED" / "HUNG" / "002"
    other.mkdir()
    (other / "insert.md").write_text("# Another vaccine\n\nOther product.", encoding="utf-8")

    parquet = tmp_path / "chunks.parquet"
    rows = [
        dict(
            zip(
                FIELDS,
                ("vax_a_001", f"vax_a_001_c_{index:04d}", "Hướng dẫn sử dụng Vaccine Vax A", "Section", text),
                strict=True,
            )
        )
        for index, text in enumerate(["First section.", "Cited text <script>bad</script>"])
    ]
    rows[0].update(category="thông tin chung", parent_header="")
    rows[1].update(category="chỉ định", header="Chỉ định", parent_header="")
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, parquet)
    assert pc.count(pq.read_table(parquet)["chunk_id"]).as_py() == 2
    return Documents(parquet, project_root=tmp_path)


def test_conversation_store_crud_validation_and_visible_count(tmp_path):
    store = ConversationStore(tmp_path / "conversations")
    assert store.list_conversations() == []

    created = store.create_conversation()
    assert created["title"] == "Cuộc trò chuyện mới"
    assert created["pending"] is None
    assert created["messages"] == []

    created["title"] = "Verorab"
    created["messages"] = [
        {"role": "user", "content": "Hỏi về Verorab"},
        build_progress_record([("planning", None), ("generating", 1)], outcome="answered", elapsed=1.2),
        {"role": "assistant", "content": "Thông tin Verorab"},
    ]
    store.save_conversation(created)

    listed = store.list_conversations()
    assert len(listed) == 1
    assert listed[0]["id"] == created["id"]
    assert listed[0]["title"] == "Verorab"
    assert listed[0]["message_count"] == 2

    with pytest.raises(ValueError):
        store.get_conversation("../escape")
    with pytest.raises(ValueError):
        store.delete_conversation("bad/id")

    assert store.delete_conversation(created["id"]) is True
    assert store.delete_conversation(created["id"]) is False
    assert store.get_conversation(created["id"]) is None


def test_chat_stream_emits_progress_and_commits_with_enriched_sources(tmp_path, documents):
    store = ConversationStore(tmp_path / "conversations")
    source_row = documents.rows("vax_a_001")[1]

    def run_turn(query, **options):
        options["on_progress"]("planning", None)
        sleep(0.3)
        options["on_progress"]("searching", None)
        options["on_progress"]("reranking", None)
        options["on_progress"]("generating", 1)
        return GeneratedAnswer(query=query, answer="Trả lời [vax_a_001_c_0001]", cited_chunks=[source_row])

    app = create_api_app(run_turn, documents, store)
    with TestClient(app) as client:
        response = client.post("/api/chat/stream", json={"conversation_id": None, "content": "Vax A chỉ định gì?"})
        assert response.status_code == 200
        events = _parse_sse(response.text)

    progress_events = [item for item in events if item["type"] == "progress"]
    assert len(progress_events) >= 3
    assert any(item["elapsed"] > 0 for item in progress_events)

    complete = events[-1]
    assert complete["type"] == "complete"
    assert complete["outcome"] == "answered"
    saved = complete["conversation"]
    assert saved["title"] == "Vax A chỉ định gì?"
    assert saved["pending"] is None
    assert len(saved["messages"]) == 3
    assert saved["messages"][1]["is_progress"] is True
    assert saved["messages"][1]["metadata"]["title"].startswith("Đã trả lời trong ")
    assert saved["messages"][2]["sources"][0]["chunk_id"] == "vax_a_001_c_0001"
    assert "excerpt" in saved["messages"][2]["sources"][0]
    assert "<script>" not in saved["messages"][2]["sources"][0]["excerpt"]

    assert [m.content for m in conversation_history(saved["messages"])] == [
        "Vax A chỉ định gì?",
        "Trả lời [vax_a_001_c_0001]",
    ]


def test_clarification_flow_reuses_pre_question_snapshot_until_answered(tmp_path, documents):
    store = ConversationStore(tmp_path / "conversations")
    conv = store.create_conversation()
    conv["messages"] = [
        {"role": "user", "content": "Bé 14 tháng hỏi về Verorab."},
        build_progress_record([("planning", None)], outcome="answered", elapsed=0.5),
        {"role": "assistant", "content": "Thông tin Verorab.", "sources": [{"doc_id": "d", "text": "x"}]},
    ]
    store.save_conversation(conv)
    expected_snapshot = conversation_history(conv["messages"])

    run_turn = Mock(
        side_effect=[
            ClarifyPlan("Bạn hỏi bé hay mẹ?"),
            ClarifyPlan("Bạn hỏi liều nào?"),
            GeneratedAnswer("Tiêm được không?", "Câu trả lời cuối."),
        ]
    )
    app = create_api_app(run_turn, documents, store)
    with TestClient(app) as client:
        r1 = _parse_sse(
            client.post("/api/chat/stream", json={"conversation_id": conv["id"], "content": "Tiêm được không?"}).text
        )[-1]
        assert r1["outcome"] == "clarified"
        pending = r1["conversation"]["pending"]
        assert pending == {
            "query": "Tiêm được không?",
            "question": "Bạn hỏi bé hay mẹ?",
            "clarifications": [],
            "history": [asdict(item) for item in expected_snapshot],
        }
        assert r1["conversation"]["messages"][-1] == {
            "role": "assistant",
            "content": "Bạn hỏi bé hay mẹ?",
        }

        r2 = _parse_sse(
            client.post("/api/chat/stream", json={"conversation_id": conv["id"], "content": "Bé"}).text
        )[-1]
        assert r2["outcome"] == "clarified"
        assert run_turn.call_args.args == ("Tiêm được không?",)
        assert run_turn.call_args.kwargs["history"] == expected_snapshot
        assert run_turn.call_args.kwargs["clarifications"] == [Clarification("Bạn hỏi bé hay mẹ?", "Bé")]

        r3 = _parse_sse(
            client.post("/api/chat/stream", json={"conversation_id": conv["id"], "content": "Liều đầu"}).text
        )[-1]
        assert r3["outcome"] == "answered"
        assert r3["conversation"]["pending"] is None
        assert run_turn.call_args.kwargs["history"] == expected_snapshot
        assert run_turn.call_args.kwargs["clarifications"] == [
            Clarification("Bạn hỏi bé hay mẹ?", "Bé"),
            Clarification("Bạn hỏi liều nào?", "Liều đầu"),
        ]


@pytest.mark.parametrize(
    ("error", "expected_message"),
    [
        (LLMError("provider failed"), "Dịch vụ trả lời chưa sẵn sàng. Hãy kiểm tra cấu hình OpenRouter."),
        (RuntimeError("retrieval failed"), "Tra cứu gặp lỗi. Hãy kiểm tra Qdrant và các model rồi thử lại."),
    ],
)
def test_failed_stream_does_not_commit_or_create_empty_conversation(tmp_path, documents, error, expected_message):
    store = ConversationStore(tmp_path / "conversations")

    def fail_turn(query, **options):
        options["on_progress"]("planning", None)
        raise error

    app = create_api_app(fail_turn, documents, store)
    with TestClient(app) as client:
        response = client.post("/api/chat/stream", json={"conversation_id": None, "content": "Câu hỏi lỗi"})
        events = _parse_sse(response.text)
        assert events[-1]["type"] == "error"
        assert events[-1]["message"] == expected_message
        assert store.list_conversations() == []

        existing = store.create_conversation()
        existing["messages"] = [{"role": "user", "content": "Cũ"}, {"role": "assistant", "content": "Đáp"}]
        store.save_conversation(existing)

        response2 = client.post("/api/chat/stream", json={"conversation_id": existing["id"], "content": "Hỏi tiếp"})
        events2 = _parse_sse(response2.text)
        assert events2[-1]["type"] == "error"
        reloaded = store.get_conversation(existing["id"])
        assert len(reloaded["messages"]) == 2


def test_chat_validation_rejects_empty_and_over_limit(tmp_path, documents):
    store = ConversationStore(tmp_path / "conversations")
    app = create_api_app(Mock(), documents, store)
    with TestClient(app) as client:
        assert client.post("/api/chat/stream", json={"conversation_id": None, "content": "   "}).status_code == 400
        too_long = client.post("/api/chat/stream", json={"conversation_id": None, "content": "a" * 2001})
        assert too_long.status_code == 400
        assert "2.000" in too_long.json()["detail"]


def test_documents_api_and_spa_route_preserve_sanitization_and_404s(tmp_path, documents):
    store = ConversationStore(tmp_path / "conversations")
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<!doctype html><div id='root'></div>", encoding="utf-8")

    app = create_api_app(Mock(), documents, store, frontend_dist=dist)
    with TestClient(app) as client:
        catalogue = client.get("/api/documents").json()["documents"]
        doc_ids = {item["doc_id"] for item in catalogue}
        assert {"vax_a_001", "vax_b_002"}.issubset(doc_ids)

        detail_resp = client.get(
            "/api/documents/vax_a_001",
            params=[("chunk", "vax_a_001_c_0001"), ("used", "vax_a_001_c_0000"), ("used", "vax_a_001_c_0001")],
        )
        assert detail_resp.status_code == 200
        detail = detail_resp.json()
        assert detail["has_original"] is True
        assert detail["cited"]["chunk_id"] == "vax_a_001_c_0001"
        assert detail["cited"]["text"] == "Cited text <script>bad</script>"

        by_name = {cat["name"]: cat for cat in detail["categories"]}
        assert by_name["thông tin chung"]["used"] is True
        assert by_name["chỉ định"]["used"] is True
        assert by_name["thành phần"]["used"] is False

        joined_html = "".join(
            item.get("html", "")
            for cat in detail["categories"]
            for item in cat["items"]
        )
        assert "Unindexed original text." in joined_html
        assert "Second original file." in joined_html
        assert '<td colspan="2">Original table cell</td>' in joined_html
        assert "<script>alert(1)</script>" not in joined_html
        assert "onclick=" not in joined_html

        assert client.get("/api/documents/unknown").status_code == 404
        assert client.get("/api/documents/vax_a_001?chunk=wrong").status_code == 404
        assert client.get("/documents?doc=unknown").status_code == 404
        assert client.get("/documents?doc=vax_a_001&chunk=wrong").status_code == 404
        assert client.get("/documents?doc=vax_b_002").status_code == 200
        assert client.get("/").status_code == 200


def test_cli_parse_args_and_main_wiring(monkeypatch, capsys, tmp_path):
    args = web_vite.parse_args([])
    assert args.host == "127.0.0.1"
    assert args.port == 8001
    assert args.preview is False
    assert args.verbose is True
    assert web_vite.parse_args(["-v"]).verbose is True
    assert web_vite.parse_args(["--verbose"]).verbose is True
    assert web_vite.parse_args(["--no-verbose"]).verbose is False

    for removed_flag in ("--share", "--data-dir", "--top-k", "--skip-rerank"):
        with pytest.raises(SystemExit):
            web_vite.parse_args([removed_flag])

    pipeline = MagicMock()
    pipeline.__enter__.return_value = pipeline
    pipeline.run_turn.return_value = "answer"
    pipeline_factory = Mock(return_value=pipeline)
    planner = Mock()
    uvicorn_run = Mock()

    monkeypatch.setattr(web_vite, "VacxisRAGPipeline", pipeline_factory)
    monkeypatch.setattr(web_vite, "QueryPlanner", Mock(return_value=planner))
    monkeypatch.setattr(web_vite, "validate_config", Mock(return_value="model"))
    monkeypatch.setattr(web_vite, "Documents", Mock(return_value=Mock(ids=set(), indexed_ids=set())))
    monkeypatch.setattr(web_vite.uvicorn, "run", uvicorn_run)

    captured_run_turn = {}

    def fake_create_api(run_turn, documents, store, *, preview, frontend_dist):
        assert not preview
        assert store.data_dir == tmp_path.resolve() / "data" / "web_vite_conversations"
        assert frontend_dist == tmp_path.resolve() / "web-vite" / "dist"
        captured_run_turn["fn"] = run_turn
        assert run_turn("question") == "answer"
        return "fake-app"

    monkeypatch.setattr(web_vite, "create_api_app", fake_create_api)
    try:
        assert web_vite.main(["--project-root", str(tmp_path)]) == 0
    finally:
        web_vite.configure_logging(False)

    config = pipeline_factory.call_args.kwargs["retrieval_config"]
    assert config.mode == "hybrid_lightrag"
    assert config.collection_name == "VacxiS_Capstone_FA26_table_dual"
    assert config.lightrag_fusion == "rrf_rerank"
    assert pipeline_factory.call_args.kwargs["query_planner"] is planner
    stdout = capsys.readouterr().out
    assert "Retriever: hybrid_lightrag" in stdout
    assert "Verbose server logging: enabled" in stdout
    uvicorn_run.assert_called_once_with("fake-app", host="127.0.0.1", port=8001)
    pipeline.__exit__.assert_called_once()


def test_configure_logging_toggles_handler_without_duplicates():
    root_logger = logging.getLogger()
    pipeline_logger = logging.getLogger(web_vite.PLANNER_PIPELINE_LOGGER)
    original_root_level = root_logger.level
    try:
        web_vite.configure_logging(False)
        web_vite.configure_logging(True)
        web_vite.configure_logging(True)
        tagged = [
            handler
            for handler in root_logger.handlers
            if getattr(handler, web_vite._VERBOSE_HANDLER_ATTR, False)
        ]
        assert root_logger.level == logging.INFO
        assert pipeline_logger.level == logging.DEBUG
        assert len(tagged) == 1
        assert tagged[0].level == logging.DEBUG

        web_vite.configure_logging(False)
        tagged_after = [
            handler
            for handler in root_logger.handlers
            if getattr(handler, web_vite._VERBOSE_HANDLER_ATTR, False)
        ]
        assert root_logger.level == original_root_level
        assert pipeline_logger.level == logging.NOTSET
        assert tagged_after == []
    finally:
        web_vite.configure_logging(False)
        root_logger.setLevel(original_root_level)


def test_preview_with_empty_corpus_exits_cleanly(monkeypatch):
    monkeypatch.setattr(web_vite, "Documents", Mock(return_value=Mock(indexed_ids=set())))
    with pytest.raises(SystemExit, match="Preview requires an indexed --chunks file"):
        web_vite.main(["--preview"])
