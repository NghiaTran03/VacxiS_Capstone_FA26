"""Gradio entry point for the local VaxiS demo."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict
import logging
import os
from pathlib import Path
from queue import Empty, SimpleQueue
from time import perf_counter
from uuid import uuid4

import gradio as gr

from ..generation import VacxisRAGPipeline
from ..llm import LLMError, validate_config
from ..query import Clarification, ClarifyPlan, ConversationMessage, QueryPlanner
from ..retrieval.schemas import RetrievalConfig
from .documents import Documents, citation, create_server
from .citations import format_sources

logger = logging.getLogger(__name__)

EMPTY_HISTORY = {"active": None, "conversations": {}}
STAGES = {
    "planning": ("Đang phân tích câu hỏi…", "Đã phân tích câu hỏi"),
    "searching": ("Đang tìm tài liệu liên quan…", "Đã tìm tài liệu liên quan"),
    "reranking": ("Đang xếp hạng lại các đoạn tài liệu…", "Đã xếp hạng lại các đoạn tài liệu"),
    "generating": ("Đang soạn câu trả lời…", "Đã soạn câu trả lời"),
}

SOURCE_EVENTS = """
document.addEventListener('click', (event) => {
    const link = event.target.closest('#vacxis-chat a[href^="/documents?"]');
    if (!link) return;
    link.target = '_blank';
    link.rel = 'noopener noreferrer';
}, true);
""" + Path(__file__).with_name("citations.js").read_text(encoding="utf-8")

CSS = """
.gradio-container { width: 100% !important; max-width: 1320px !important; margin: auto; }
.gradio-container > .main { width: 100%; max-width: none !important; padding: 12px 20px; }
#vacxis-shell { display: grid !important; grid-template-columns: 230px minmax(0, 1fr); gap: 24px; align-items: start; }
#vacxis-sidebar { min-width: 0 !important; padding: 16px; border-radius: 12px; background: var(--background-fill-secondary); }
#vacxis-main { min-width: 0 !important; }
#vacxis-conversations .wrap { display: flex; flex-direction: column; gap: 4px; }
#vacxis-conversations label { border: 0; background: transparent; padding: 9px 10px; border-radius: 7px; }
#vacxis-conversations label.selected { background: var(--background-fill-primary); }
#vacxis-conversations label:focus-within { outline: 2px solid var(--body-text-color-subdued); }
#vacxis-conversations input[type=radio] { position: absolute; opacity: 0; width: 1px; height: 1px; }
#vacxis-conversations label span { white-space: normal; overflow-wrap: anywhere; }
#vacxis-source-events { display: none; }
#vacxis-chat { border: 0; }
#vacxis-chat .vacxis-citation { display: inline-flex; align-items: center; justify-content: center; min-width: 20px; height: 20px; padding: 0 5px; margin: 0 2px; border-radius: 6px; font-size: 11px; line-height: 1; vertical-align: super; text-decoration: none; color: var(--body-text-color); background: var(--background-fill-primary); border: 1px solid var(--border-color-primary); cursor: pointer; }
#vacxis-chat .vacxis-citation:focus-visible { outline: 2px solid var(--body-text-color); outline-offset: 2px; }
.vacxis-source-preview { position: fixed; z-index: 10000; width: min(360px, calc(100vw - 24px)); box-sizing: border-box; padding: 18px; border: 1px solid var(--border-color-primary); border-radius: 14px; background: var(--background-fill-secondary); color: var(--body-text-color); box-shadow: 0 12px 36px #0004; font-size: 13px; line-height: 1.5; }
.vacxis-source-preview .source-title { display: block; font-size: 14px; }
.vacxis-source-preview .source-heading { color: var(--body-text-color-subdued); margin: 6px 0 10px; }
.vacxis-source-preview .source-excerpt { max-height: 190px; overflow: auto; white-space: pre-wrap; }
.vacxis-source-preview > a, .vacxis-source-preview > span { display: block; margin-top: 12px; text-align: right; color: var(--body-text-color); font-size: 12px; }
#vacxis-composer { display: grid; grid-template-columns: minmax(0, 1fr) auto; align-items: end; gap: 8px; border: 1px solid var(--border-color-primary); border-radius: 14px; padding: 10px; }
#vacxis-composer > * { min-width: 0 !important; }
#vacxis-composer textarea { border: 0; box-shadow: none; }
#vacxis-input-area { gap: 6px; }
#vacxis-disclaimer .html-container { padding: 0; }
.vacxis-disclaimer { display: block; margin: 2px 4px 0; font-size: 12px; line-height: 1.5; color: var(--body-text-color-subdued); }
.vacxis-brand { padding: 8px 4px; }
.vacxis-brand strong { font-size: 26px; letter-spacing: -.8px; }
.vacxis-brand span { font-size: 13px; color: var(--body-text-color-subdued); margin-left: 12px; }
.vacxis-library { display: block; padding: 12px 0; text-decoration: none; color: var(--body-text-color); }
.vacxis-preview { padding: 10px 14px; border-radius: 8px; background: var(--background-fill-secondary); font-size: 13px; }
@media (max-width: 640px) {
    .gradio-container > .main { padding: 10px; }
    #vacxis-shell { grid-template-columns: minmax(0, 1fr); gap: 12px; }
    #vacxis-sidebar { padding: 12px; }
    #vacxis-conversations .wrap { max-height: 140px; overflow: auto; }
}
"""


def _active(history: dict) -> dict | None:
    return history["conversations"].get(history["active"])


def _history_picker(history: dict):
    choices = [(chat["title"], chat_id) for chat_id, chat in reversed(history["conversations"].items())]
    return gr.update(choices=choices, value=history["active"])


def _new_conversation(saved: dict):
    """Select a blank conversation immediately and reuse an untouched draft."""
    updated = deepcopy(saved)
    active = _active(updated)
    if active is None or active["messages"]:
        chat_id = str(uuid4())
        updated["conversations"][chat_id] = {
            "title": "Cuộc trò chuyện mới", "messages": [], "pending": None,
        }
        updated["active"] = chat_id
    return updated, updated, _history_picker(updated), [], ""


def _display(messages: list[dict], chat_id: str | None) -> list[dict]:
    display = []
    for message in messages:
        text = message["content"]
        if message["role"] == "assistant":
            text = format_sources(text, message.get("sources", []))
        display.append({"role": message["role"], "content": text, "metadata": message.get("metadata", {})})
    return display


def _progress(events: list[tuple[str, int | None]], outcome: str | None = None, elapsed: float = 0) -> dict:
    lines = []
    for index, (stage, count) in enumerate(events):
        complete = index < len(events) - 1 or outcome in {"answered", "clarified"}
        if count is not None:
            lines.append(f"- ✓ Đã chọn {count} đoạn tài liệu để trả lời")
        pending_text, completed_text = STAGES[stage]
        lines.append(f"- {'✓' if complete else '◉'} {completed_text if complete else pending_text}")
    if outcome == "clarified":
        lines.append("- Cần làm rõ câu hỏi")
    if outcome == "failed":
        lines.append("- Quá trình trả lời bị gián đoạn")
    seconds = f"{elapsed:.1f}".replace(".", ",")
    title = "Quá trình trả lời"
    if outcome == "answered":
        title = f"Đã trả lời trong {seconds} giây"
    elif outcome is None:
        title = f"Đang xử lý · {seconds} giây"
    return {
        "role": "assistant",
        "content": "\n".join(lines),
        "metadata": {"title": title, "status": "done" if outcome else "pending"},
        "is_progress": True,
    }


def _conversation_history(messages: list[dict]) -> tuple[ConversationMessage, ...]:
    """Extract plain text without progress records or citation metadata."""
    return tuple(
        ConversationMessage(message["role"], message["content"])
        for message in messages
        if message.get("role") in {"user", "assistant"}
        and isinstance(message.get("content"), str) and message["content"].strip()
        and not message.get("is_progress")
        and message.get("metadata", {}).get("title") != "Quá trình trả lời"
    )


def _submit(text: str, saved: dict, run_turn):
    """Preview a turn and commit browser history only after a successful result."""
    text = text.strip()
    if not text or len(text) > 2000:
        if text:
            gr.Warning("Câu hỏi có thể dài tối đa 2.000 ký tự.")
        yield (gr.skip(),) * 6
        return

    updated = deepcopy(saved)
    active = _active(updated)
    if active is None:
        chat_id = str(uuid4())
        active = {"title": text[:65], "messages": [], "pending": None}
        updated["conversations"][chat_id] = active
        updated["active"] = chat_id
    messages = active["messages"]
    if not messages:
        active["title"] = text[:65]
    pending = active["pending"]
    snapshot = (
        tuple(ConversationMessage(**message) for message in pending.get("history", []))
        if pending else _conversation_history(messages)
    )
    messages.append({"role": "user", "content": text})
    original = pending["query"] if pending else text
    clarifications = pending["clarifications"] + [{"question": pending["question"], "answer": text}] if pending else []
    events = []
    started = perf_counter()

    def preview():
        return gr.skip(), gr.skip(), gr.skip(), _display(messages + [_progress(events, elapsed=perf_counter() - started)], updated["active"]), gr.update(value="", interactive=False), gr.update(interactive=False)

    yield preview()
    notices = SimpleQueue()
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            run_turn, original, clarifications=[Clarification(**turn) for turn in clarifications],
            history=snapshot,
            on_progress=lambda stage, count: notices.put((stage, count)),
        )
        future.add_done_callback(lambda _: notices.put(None))
        while True:
            try:
                notice = notices.get(timeout=0.5)
            except Empty:
                yield preview()
                continue
            if notice is None:
                break
            events.append(notice)
            yield preview()
        try:
            result = future.result()
        except Exception as exc:
            if isinstance(exc, LLMError):
                logger.warning("Chat model unavailable: %s", exc)
                gr.Warning("Dịch vụ trả lời chưa sẵn sàng. Hãy kiểm tra cấu hình OpenRouter.")
            else:
                logger.error("Chat turn failed (%s). Check retrieval services and model settings.", type(exc).__name__)
                gr.Warning("Tra cứu gặp lỗi. Hãy kiểm tra Qdrant và các model rồi thử lại.")
            previous = _active(saved)
            yield gr.skip(), gr.skip(), gr.skip(), _display(
                (previous["messages"] if previous else []) + [_progress(events, "failed", perf_counter() - started)], saved["active"],
            ), gr.update(value=text, interactive=True), gr.update(interactive=True)
            return

    if isinstance(result, ClarifyPlan):
        active["pending"] = {
            "query": original, "clarifications": clarifications,
            "question": result.clarification_question,
            "history": [asdict(message) for message in snapshot],
        }
        messages.append({"role": "assistant", "content": result.clarification_question})
    else:
        active["pending"] = None
        messages.extend([_progress(events, "answered", perf_counter() - started), {
            "role": "assistant", "content": result.answer, "sources": [citation(chunk) for chunk in result.cited_chunks],
        }])
    yield updated, updated, _history_picker(updated), _display(messages, updated["active"]), gr.update(value="", interactive=True), gr.update(interactive=True)


def build_app(run_turn, documents: Documents, *, preview: bool = False) -> gr.Blocks:
    with gr.Blocks(title="VaxiS", fill_width=True) as app:
        browser_history = gr.BrowserState(
            EMPTY_HISTORY, storage_key="vacxis-conversations-v1", secret=os.getenv("VACXIS_HISTORY_SECRET"),
        )
        history = gr.State(EMPTY_HISTORY)
        gr.HTML('<header class="vacxis-brand"><strong>VaxiS<span>Trợ lý thông tin vắc xin</span></strong></header>', apply_default_css=False)
        if preview:
            gr.HTML('<p class="vacxis-preview">Bản xem trước giao diện · Phản hồi mẫu từ tài liệu có sẵn. Pipeline AI chưa được kết nối.</p>', apply_default_css=False)
        with gr.Row(elem_id="vacxis-shell"):
            with gr.Column(scale=0, min_width=0, elem_id="vacxis-sidebar"):
                new_chat = gr.Button("Cuộc trò chuyện mới")
                gr.HTML('<a class="vacxis-library" href="/documents" target="_blank" rel="noopener noreferrer">Tài liệu vắc xin ↗</a>', apply_default_css=False)
                picker = gr.Radio(label="Các cuộc trò chuyện", choices=[], container=False, elem_id="vacxis-conversations")
            with gr.Column(scale=1, min_width=0, elem_id="vacxis-main"):
                chat = gr.Chatbot(
                    elem_id="vacxis-chat", height="60vh", show_label=False,
                    allow_tags=False, group_consecutive_messages=False, buttons=["copy", "copy_all"],
                    placeholder="Bạn muốn tìm hiểu về vắc xin nào?",
                )
                gr.HTML("", js_on_load=SOURCE_EVENTS, elem_id="vacxis-source-events")
                with gr.Column(elem_id="vacxis-input-area", min_width=0):
                    with gr.Row(elem_id="vacxis-composer"):
                        question = gr.Textbox(label="Câu hỏi", show_label=False, placeholder="Hỏi VaxiS về vắc xin…", scale=6, min_width=0, elem_id="vacxis-question")
                        send = gr.Button("Gửi ↑", variant="primary", scale=1, min_width=64)
                    gr.HTML(
                        '<small class="vacxis-disclaimer">* Thông tin này chỉ nhằm mục đích tham khảo. '
                        'Để được tư vấn hoặc chẩn đoán y tế, hãy tham khảo ý kiến của chuyên gia. '
                        'Câu trả lời của AI có thể chứa thông tin không chính xác.</small>',
                        apply_default_css=False, elem_id="vacxis-disclaimer",
                    )

        def restore(saved: dict):
            active = _active(saved)
            return saved, _history_picker(saved), _display(active["messages"] if active else [], saved["active"])

        def select_chat(chat_id: str | None, saved: dict):
            updated = {**saved, "active": chat_id if chat_id in saved["conversations"] else None}
            active = _active(updated)
            return updated, updated, _display(active["messages"] if active else [], updated["active"]), ""

        def submit(text: str, saved: dict):
            yield from _submit(text, saved, run_turn)

        app.load(restore, browser_history, [history, picker, chat], api_visibility="private")
        conversation_events = {"concurrency_id": "vacxis-conversation", "api_visibility": "private"}
        picker.input(select_chat, [picker, history], [history, browser_history, chat, question], **conversation_events)
        new_chat.click(
            _new_conversation, history, [history, browser_history, picker, chat, question],
            show_progress="hidden", **conversation_events,
        ).then(fn=None, js="() => { document.querySelector('#vacxis-question textarea')?.focus(); }", queue=False)
        gr.on(
            [question.submit, send.click], submit, [question, history],
            [history, browser_history, picker, chat, question, send], show_progress="hidden", **conversation_events,
        )
    return app.queue()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Select the main Table-aware + LightRAG architecture for the web demo."""
    from ..cli import add_retrieval_arguments

    parser = argparse.ArgumentParser(description="Run the local VaxiS chat UI")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link for chat and documents")
    parser.add_argument("--chunks", type=Path, default=Path("chunks_storage/chunks.parquet"))
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--top-k", type=int, default=RetrievalConfig.top_k)
    parser.add_argument("--skip-rerank", action="store_true")
    parser.add_argument("--preview", action="store_true", help="Preview UI using sample extractive responses without model services")
    add_retrieval_arguments(parser)
    parser.set_defaults(
        retriever_mode="hybrid_lightrag",
        collection_name="VacxiS_Capstone_FA26_table_dual",
        lightrag_fusion="rrf_rerank",
    )
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535 or args.top_k < 1:
        parser.error("--port must be between 1 and 65535, and --top-k must be positive.")
    return args


def main(argv: list[str] | None = None) -> int:
    from ..cli import retrieval_config

    args = parse_args(argv)

    with ExitStack() as resources:
        pipeline = None
        retrieval = retrieval_config(args)
        if not args.preview:
            print(f"* Retriever: {retrieval.mode} | collection: {retrieval.collection_name}", flush=True)
            if retrieval.mode in {"lightrag", "hybrid_lightrag"}:
                print(
                    f"* LightRAG mode: {retrieval.lightrag_mode} | workspace: {retrieval.lightrag_workspace}",
                    flush=True,
                )
            if retrieval.mode == "hybrid_lightrag":
                print(f"* LightRAG fusion: {retrieval.lightrag_fusion}", flush=True)

        def run_turn(query: str, **options):
            nonlocal pipeline
            if pipeline is None:
                validate_config()
                planner = QueryPlanner(model=validate_config(planner=True))
                pipeline = resources.enter_context(VacxisRAGPipeline(retrieval_config=retrieval, query_planner=planner))
            return pipeline.run_turn(query, **options)

        project_root = args.project_root.resolve()
        chunks_path = args.chunks if args.chunks.is_absolute() else project_root / args.chunks
        documents = Documents(chunks_path, project_root=project_root)
        if args.preview:
            if not documents.indexed_ids:
                raise SystemExit("Preview requires an indexed --chunks file with vaccine documents.")
            from .preview import preview_responder
            run_turn = preview_responder(documents)
        app = build_app(run_turn, documents, preview=args.preview)
        from .sharing import run_server
        run_server(create_server(app, documents, CSS), port=args.port, share=args.share)
    return 0
