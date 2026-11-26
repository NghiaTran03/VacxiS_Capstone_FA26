"""Cover stable citation numbering, safe previews and unchanged planner history."""

from copy import deepcopy
from html.parser import HTMLParser

from vacxis_capstone_fa26.web import _conversation_history, _display
from vacxis_capstone_fa26.web.citations import format_sources


class Anchors(HTMLParser):
    def __init__(self, html):
        super().__init__()
        self.links = []
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.links.append(dict(attrs))


def sources():
    return [
        {"chunk_id": "vax_c_0001", "doc_id": "vax", "doc_title": "Vaccine A", "header": "Liều dùng", "text": "First excerpt."},
        {"chunk_id": "vax_c_0002", "doc_id": "vax", "doc_title": "Vaccine A", "header": "Tương tác", "text": "Second excerpt."},
    ]


def test_inline_references_repeat_the_same_number_and_preserve_markdown():
    answer = "**Heading**\n\nFirst [vax_c_0001]. Again [vax_c_0001]. Both [vax_c_0001, vax_c_0002]."
    result = format_sources(answer, sources())
    assert result.startswith("**Heading**")
    assert result.count('>1</a>') == 3
    assert result.count('>2</a>') == 1
    assert "Nguồn:" not in result
    links = Anchors(result).links
    assert links[0]["href"] == links[1]["href"]
    assert links[0]["title"] == "Vaccine A\nLiều dùng\nFirst excerpt."
    assert all(link["target"] == "_blank" for link in links)


def test_preview_attributes_escape_untrusted_source_content():
    source = sources()[0] | {"doc_title": '\" onclick=\"alert(1)<img>', "text": '<table><tr><td>Cell</td></tr></table><script>bad()</script>'}
    result = format_sources("Answer [vax_c_0001]", [source])
    link = Anchors(result).links[0]
    assert "onclick" not in link
    assert "<table>" not in result and "<script>" not in result
    assert "Cell" in link["title"]


def test_code_and_unknown_ids_are_not_rewritten_and_prefix_ids_do_not_match():
    answer = "`[vax_c_0001]`\n\n```text\n[vax_c_0002]\n```\n\n[unknown_c_0001] [vax_c_00010]"
    result = format_sources(answer, sources())
    assert result.startswith(answer)
    assert result.count('class="vacxis-citation"') == 2
    assert "Nguồn:" in result


def test_graph_context_has_preview_without_a_broken_document_link():
    source = sources()[0] | {"chunk_id": "lightrag_kg_context", "doc_id": "lightrag_kg"}
    result = format_sources("Graph evidence [lightrag_kg_context]", [source])
    assert 'tabindex="0"' in result
    assert "href=" not in result


def test_display_does_not_change_saved_answer_or_planner_memory():
    messages = [{"role": "assistant", "content": "Answer [vax_c_0001]", "sources": sources()}]
    original = deepcopy(messages)
    displayed = _display(messages, "chat")
    assert displayed[0]["content"] != messages[0]["content"]
    assert messages == original
    assert _conversation_history(messages)[0].content == "Answer [vax_c_0001]"


def test_duplicate_sources_do_not_add_another_number():
    result = format_sources("No inline citation.", [sources()[0], sources()[0]])
    assert result.count('class="vacxis-citation"') == 1


def test_adjacent_citations_are_numbered_and_unknown_ids_are_preserved():
    result = format_sources("Adjacent [vax_c_0001][vax_c_0002]. Mixed [vax_c_0001, unknown_c_0003].", sources())
    assert result.count('>1</a>') == 2
    assert result.count('>2</a>') == 1
    assert "[unknown_c_0003]" in result
    assert "Nguồn:" not in result


def test_existing_markdown_links_and_reference_links_stay_literal():
    answer = "[vax_c_0001](https://example.com) [label][vax_c_0002]\n\n[vax_c_0002]: https://example.com"
    result = format_sources(answer, sources())
    assert result.startswith(answer)
