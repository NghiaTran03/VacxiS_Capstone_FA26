"""Cover elapsed progress, compact clarification and full-source document links."""

from copy import deepcopy
from html import unescape
import json
import re
from time import sleep
from urllib.parse import parse_qs, urlsplit
from unittest.mock import Mock

from fastapi.testclient import TestClient
import gradio as gr
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

from vacxis_capstone_fa26.generation.schemas import GeneratedAnswer
from vacxis_capstone_fa26.query import ClarifyPlan
from vacxis_capstone_fa26.web import EMPTY_HISTORY, _active, _conversation_history, _display, _new_conversation, _submit
from vacxis_capstone_fa26.web.documents import Documents, FIELDS, _covered_categories, _original_sections, create_server
from vacxis_capstone_fa26.web.preview import preview_responder


@pytest.fixture
def documents(tmp_path):
    folder = tmp_path / "CLEANED" / "HUNG" / "001"
    folder.mkdir(parents=True)
    (tmp_path / "75_vaccine_names_mapping.json").write_text(
        json.dumps({"001": "Vax A", "002": "Vax B"}), encoding="utf-8",
    )
    (folder / "insert.md").write_text(
        '# Full product insert\n\nFirst section.\n\n## Last section\n\nUnindexed original text.\n\n'
        '## Thành phần\n\nThành phần từ tài liệu gốc.\n\n## Chỉ định\n\nNội dung chỉ định.\n\n'
        '<table onclick="alert(1)"><tr><td colspan="2">Original table cell</td></tr></table>\n\n'
        '<script>alert(1)</script>',
        encoding="utf-8",
    )
    (folder / "additional.md").write_text("# Additional source\n\nSecond original file.", encoding="utf-8")
    other = tmp_path / "CLEANED" / "HUNG" / "002"
    other.mkdir()
    (other / "insert.md").write_text("# Another vaccine\n\nOther product.", encoding="utf-8")
    parquet = tmp_path / "chunks.parquet"
    rows = [dict(zip(FIELDS, (
        "vax_a_001", f"vax_a_001_c_{index:04d}", "Hướng dẫn sử dụng Vaccine Vax A", "Section", text,
    ), strict=True)) for index, text in enumerate(["First section.", "Cited text <script>bad</script>"])]
    rows[0].update(category="thông tin chung", parent_header="")
    rows[1].update(category="chỉ định", header="Chỉ định", parent_header="")
    pq.write_table(pa.Table.from_pylist(rows), parquet)
    return Documents(parquet, project_root=tmp_path)


def test_document_page_contains_all_original_files_and_exact_cited_excerpt(documents):
    html = documents.page("vax_a_001", "vax_a_001_c_0001")
    assert 'id="citation"' in html
    assert "Cited text &lt;script&gt;bad&lt;/script&gt;" in html
    assert "Unindexed original text." in html
    assert "Second original file." in html
    assert "<script>alert(1)</script>" not in html
    assert "<script>bad</script>" not in html
    assert '<td colspan="2">Original table cell</td>' in html
    assert "onclick=" not in html
    assert "Other product." not in html
    assert ('Hướng dẫn sử dụng Vaccine Vax B', 'vax_b_002') in documents.choices
    assert "Other product." in documents.page("vax_b_002", None)


def test_categories_are_collapsible_and_only_cited_category_is_marked_open(documents):
    html = documents.page("vax_a_001", "vax_a_001_c_0001")
    assert 'data-category="chỉ định" open' in html
    assert 'data-category="thành phần" open' not in html
    assert 'data-category="thành phần"' in html
    assert html.count('class="used-marker"') == 1
    assert '<summary>Chỉ định<span class="used-marker"' in html
    assert 'id="vaccine-content"' in html
    assert 'id="document-search" type="search"' in html
    assert 'id="search-next"' in html
    plain = documents.page("vax_a_001", None)
    assert 'class="used-marker"' not in plain
    assert 'class="category" data-category="chỉ định" open' not in plain


def test_all_categories_cited_in_one_answer_are_marked_on_each_source_link(documents):
    sources = documents.rows("vax_a_001")
    message = _display([{"role": "assistant", "content": "Answer", "sources": sources}], "session")[0]
    assert unescape(message["content"]).count("used=vax_a_001_c_0000&used=vax_a_001_c_0001") == 2
    html = documents.page("vax_a_001", "vax_a_001_c_0001", [row["chunk_id"] for row in sources])
    assert html.count('class="used-marker"') == 2
    assert 'data-category="thông tin chung" open' in html
    assert 'data-category="chỉ định" open' in html
    assert 'data-category="thành phần" open' not in html


def test_excerpt_spanning_general_and_ingredients_opens_both_categories(documents):
    source = documents.originals["vax_a_001"].md_files[1].read_text(encoding="utf-8")
    excerpt = source.split("## Chỉ định")[0]
    rows = pq.read_table(documents.path).to_pylist()
    rows[0]["text"] = "[CHỦ ĐỀ: Vax A] > [MỤC: Thông tin chung]\n" + excerpt
    pq.write_table(pa.Table.from_pylist(rows), documents.path)
    html = documents.page("vax_a_001", "vax_a_001_c_0000")
    assert 'data-category="thông tin chung" open' in html
    assert 'data-category="thành phần" open' in html
    assert 'data-category="chỉ định" open' not in html
    assert html.count('class="used-marker"') == 2
    assert "Thành phần từ tài liệu gốc." in html


def test_repeated_excerpt_does_not_mark_unrelated_categories():
    sections = [("thành phần", "Repeated text."), ("bảo quản", "Repeated text.")]
    chunk = {"text": "Repeated text.", "category": "bảo quản"}
    assert _covered_categories(sections, chunk) == {"bảo quản"}


def test_html_and_markdown_table_excerpts_align_to_the_same_source_category():
    sections = [("chỉ định", "<table><tr><th>Name</th><th>Value</th></tr><tr><td>Item</td><td>Contents</td></tr></table>")]
    chunk = {"text": "| Name | Value |\n| --- | --- |\n| Item | Contents |", "category": "thông tin chung"}
    assert _covered_categories(sections, chunk) == {"chỉ định"}


def test_category_grouping_keeps_all_original_lines_including_inline_headings():
    source = (
        'Vaccine name\n\nTHÀNH PHẦN: trong 1 liều\nOriginal ingredients.\n\n'
        '<table><tr><td>\nCHỈ ĐỊNH\n\nCHỈ ĐỊNH: table cell content\n</td></tr></table>\n\n'
        '2/8\n\n## CHỈ ĐỊNH\nFull indication text.\n'
    )
    sections = _original_sections(source)
    assert "".join(body for _, body in sections) == source
    assert [category for category, _ in sections] == ["thông tin chung", "thành phần", "chỉ định"]
    assert '<table><tr><td>\nCHỈ ĐỊNH\n' in sections[1][1]
    assert '</td></tr></table>' in sections[1][1]


def test_bold_heading_followed_by_colon_keeps_its_category():
    source = '**THÀNH PHẦN**:\n\nOriginal ingredient text.\n\n**CHỈ ĐỊNH:**\n\nOriginal indication text.\n'
    sections = _original_sections(source)
    assert [category for category, _ in sections] == ["thành phần", "chỉ định"]
    assert "".join(body for _, body in sections) == source


def test_legacy_corpus_without_category_uses_header_classification(documents):
    table = pq.read_table(documents.path).drop(["category", "parent_header"])
    pq.write_table(table, documents.path)
    assert documents.rows("vax_a_001")[1]["category"] == "chỉ định"
    assert 'data-category="chỉ định" open' in documents.page("vax_a_001", "vax_a_001_c_0001")


def test_sources_resolve_on_mounted_server_and_invalid_citations_return_404(documents):
    source = documents.rows("vax_a_001")[1]
    messages = _display([{"role": "assistant", "content": "Answer", "sources": [source]}], "session")
    url = unescape(re.search(r'href="([^"]+)"', messages[0]["content"]).group(1))
    params = parse_qs(urlsplit(url).query)
    assert params == {"doc": [source["doc_id"]], "chunk": [source["chunk_id"]]}
    assert urlsplit(url).fragment == "citation"
    with gr.Blocks(analytics_enabled=False) as blocks:
        gr.Markdown("VacxiS")
    with TestClient(create_server(blocks, documents, "")) as client:
        response = client.get(url)
        assert response.status_code == 200
        assert "Unindexed original text." in response.text
        assert client.get("/documents?doc=unknown").status_code == 404
        assert client.get("/documents?doc=vax_a_001&chunk=wrong").status_code == 404
        assert client.get("/documents?doc=vax_b_002").status_code == 200
        assert client.get("/").status_code == 200


def test_live_elapsed_time_and_final_duration_are_excluded_from_planner_memory():
    def run_turn(query, **options):
        options["on_progress"]("planning", None)
        sleep(0.6)
        return GeneratedAnswer(query, "Answer")

    outputs = list(_submit("Question", deepcopy(EMPTY_HISTORY), run_turn))
    live_titles = [output[3][-1]["metadata"]["title"] for output in outputs[:-1]]
    assert any(title.startswith("Đang xử lý · ") and "0,0 giây" not in title for title in live_titles)
    messages = _active(outputs[-1][0])["messages"]
    assert messages[-2]["metadata"]["title"].startswith("Đã trả lời trong ")
    assert messages[-2]["metadata"]["status"] == "done"
    assert [message.content for message in _conversation_history(messages)] == ["Question", "Answer"]


def test_clarification_is_only_a_short_assistant_message():
    result = list(_submit(
        "Bé tiêm loại này được không?", deepcopy(EMPTY_HISTORY),
        Mock(return_value=ClarifyPlan("Bạn đang hỏi về vắc xin nào?")),
    ))[-1]
    messages = _active(result[0])["messages"]
    assert len(messages) == 2
    assert messages[-1] == {"role": "assistant", "content": "Bạn đang hỏi về vắc xin nào?"}
    assert _active(result[0])["pending"]["query"] == "Bé tiêm loại này được không?"


def test_preview_escapes_truncated_tables_before_appending_citation_links(monkeypatch):
    monkeypatch.setattr("vacxis_capstone_fa26.web.preview.sleep", lambda _: None)
    documents = Mock(
        choices=[("Hướng dẫn sử dụng Vaccine Verorab", "verorab")], indexed_ids={"verorab"},
    )
    documents.rows.return_value = [{
        "doc_id": "verorab", "chunk_id": "chunk", "doc_title": "Verorab", "header": "Table",
        "text": "Context\n| Column | Value |\n| --- | --- |\n| First | Contents |\n<table><tr><td>HTML cell</td></tr></table>",
    }]
    result = preview_responder(documents)("Verorab", on_progress=lambda *args: None)
    assert "\\| Column \\| Value \\|" in result.answer
    assert "HTML cell" in result.answer
    assert "<table>" not in result.answer
    assert len(result.cited_chunks) == 1


def test_new_conversation_is_created_on_click_and_followups_keep_one_history_entry():
    original = deepcopy(EMPTY_HISTORY)
    new = _new_conversation(original)
    saved = new[0]
    chat_id = saved["active"]
    assert original == EMPTY_HISTORY
    assert len(saved["conversations"]) == 1
    assert _active(saved) == {"title": "Cuộc trò chuyện mới", "messages": [], "pending": None}
    assert new[3:] == ([], "")
    assert new[2]["value"] == chat_id
    assert _new_conversation(saved)[0] == saved

    responder = Mock(return_value=GeneratedAnswer("query", "answer"))
    first = list(_submit("Verorab là gì?", saved, responder))[-1][0]
    followup = list(_submit("Loại này có tác dụng phụ gì?", first, responder))[-1][0]
    assert followup["active"] == chat_id
    assert len(followup["conversations"]) == 1
    assert _active(followup)["title"] == "Verorab là gì?"
    assert sum(message["role"] == "user" for message in _active(followup)["messages"]) == 2

    second = _new_conversation(followup)[0]
    assert len(second["conversations"]) == 2
    assert second["active"] != chat_id
    assert second["conversations"][chat_id] == _active(followup)
    assert _active(second)["messages"] == []
