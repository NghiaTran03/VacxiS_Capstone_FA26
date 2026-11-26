"""FastAPI application serving REST/SSE APIs and the built Vite frontend."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
import logging
import os
from pathlib import Path
from queue import Empty, SimpleQueue
from time import perf_counter
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..llm import LLMError
from ..query import Clarification, ClarifyPlan, ConversationMessage
from ..web.documents import Documents
from .documents_api import enrich_citation, get_document_payload, list_documents
from .storage import DEFAULT_TITLE, ConversationStore, validate_conversation_id

logger = logging.getLogger(__name__)

MAX_QUESTION_LENGTH = 2000
DEFAULT_CORS_ORIGINS = (
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
)
STAGES = {
    "planning": ("Đang phân tích câu hỏi…", "Đã phân tích câu hỏi"),
    "searching": ("Đang tìm tài liệu liên quan…", "Đã tìm tài liệu liên quan"),
    "reranking": ("Đang xếp hạng lại các đoạn tài liệu…", "Đã xếp hạng lại các đoạn tài liệu"),
    "generating": ("Đang soạn câu trả lời…", "Đã soạn câu trả lời"),
}


class ChatStreamRequest(BaseModel):
    """Payload for streaming a chat turn."""

    conversation_id: str | None = None
    content: str


def conversation_history(messages: list[dict[str, Any]]) -> tuple[ConversationMessage, ...]:
    """Extract plain user and assistant text without progress records or citations."""
    return tuple(
        ConversationMessage(message["role"], message["content"])
        for message in messages
        if message.get("role") in {"user", "assistant"}
        and isinstance(message.get("content"), str)
        and message["content"].strip()
        and not message.get("is_progress")
        and message.get("metadata", {}).get("title") != "Quá trình trả lời"
    )


def build_progress_record(
    events: Sequence[tuple[str, int | None]],
    outcome: str | None = None,
    elapsed: float = 0.0,
) -> dict[str, Any]:
    """Format progress events into the same structure used by the chat UI."""
    lines: list[str] = []
    serialized_events: list[list[Any]] = []
    for index, (stage, count) in enumerate(events):
        serialized_events.append([stage, count])
        complete = index < len(events) - 1 or outcome in {"answered", "clarified"}
        if count is not None:
            lines.append(f"- ✓ Đã chọn {count} đoạn tài liệu để trả lời")
        pending_text, completed_text = STAGES.get(stage, (stage, stage))
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
        "progress": {
            "events": serialized_events,
            "outcome": outcome,
            "elapsed": round(elapsed, 2),
        },
    }


def _sse_frame(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _resolve_cors_origins(cors_origins: Sequence[str] | None) -> list[str]:
    if cors_origins is not None:
        return list(cors_origins)
    raw = os.getenv("VACXIS_CORS_ORIGINS", "")
    extra = [item.strip() for item in raw.split(",") if item.strip()]
    return list(dict.fromkeys([*DEFAULT_CORS_ORIGINS, *extra]))


def _missing_build_response() -> HTMLResponse:
    return HTMLResponse(
        "<!doctype html><html lang='vi'><meta charset='utf-8'>"
        "<title>VaxiS Web Vite</title><body style='font-family:system-ui;padding:32px'>"
        "<h1>Chưa build frontend Vite</h1>"
        "<p>Chạy <code>cd web-vite &amp;&amp; npm install &amp;&amp; npm run build</code> "
        "hoặc dùng Vite dev server tại <code>http://localhost:5173</code>.</p></body></html>",
        status_code=200,
    )


def create_api_app(
    run_turn: Callable[..., Any],
    documents: Documents,
    store: ConversationStore,
    *,
    preview: bool = False,
    frontend_dist: Path | None = None,
    cors_origins: Sequence[str] | None = None,
) -> FastAPI:
    """Create the FastAPI server exposing REST/SSE APIs and the Vite SPA."""
    app = FastAPI(title="VaxiS Web Vite API")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_resolve_cors_origins(cors_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "service": "VaxiS Web Vite API", "preview": preview}

    @app.get("/api/config")
    def config() -> dict[str, Any]:
        return {"preview": preview, "max_question_length": MAX_QUESTION_LENGTH}

    @app.get("/api/conversations")
    def list_conversations() -> list[dict[str, Any]]:
        return store.list_conversations()

    @app.post("/api/conversations")
    def create_conversation() -> dict[str, Any]:
        return store.create_conversation()

    @app.get("/api/conversations/{conversation_id}")
    def get_conversation(conversation_id: str) -> dict[str, Any]:
        try:
            conversation = store.get_conversation(conversation_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if conversation is None:
            raise HTTPException(status_code=404, detail="Không tìm thấy cuộc trò chuyện.")
        return conversation

    @app.delete("/api/conversations/{conversation_id}")
    def delete_conversation(conversation_id: str) -> dict[str, Any]:
        try:
            deleted = store.delete_conversation(conversation_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if not deleted:
            raise HTTPException(status_code=404, detail="Không tìm thấy cuộc trò chuyện.")
        return {"deleted": True, "id": conversation_id}

    @app.post("/api/chat/stream")
    async def chat_stream(request: ChatStreamRequest) -> StreamingResponse:
        text = request.content.strip()
        if not text:
            raise HTTPException(status_code=400, detail="Vui lòng nhập câu hỏi.")
        if len(text) > MAX_QUESTION_LENGTH:
            raise HTTPException(status_code=400, detail="Câu hỏi có thể dài tối đa 2.000 ký tự.")

        existing: dict[str, Any] | None = None
        if request.conversation_id is not None:
            try:
                validate_conversation_id(request.conversation_id)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            existing = store.get_conversation(request.conversation_id)
            if existing is None:
                raise HTTPException(status_code=404, detail="Không tìm thấy cuộc trò chuyện.")

        base_messages = list(existing["messages"]) if existing is not None else []
        pending = existing.get("pending") if existing is not None else None
        snapshot = (
            tuple(ConversationMessage(**item) for item in pending.get("history", []))
            if pending
            else conversation_history(base_messages)
        )
        original_query = pending["query"] if pending else text
        clarifications = (
            [*pending.get("clarifications", []), {"question": pending["question"], "answer": text}]
            if pending
            else []
        )

        async def event_generator():
            events: list[tuple[str, int | None]] = []
            started = perf_counter()
            notices: SimpleQueue[tuple[str, int | None] | None] = SimpleQueue()

            yield _sse_frame({
                "type": "progress",
                "events": [],
                "elapsed": 0.0,
            })

            executor = ThreadPoolExecutor(max_workers=1)
            try:
                future = executor.submit(
                    run_turn,
                    original_query,
                    clarifications=[Clarification(**turn) for turn in clarifications],
                    history=snapshot,
                    on_progress=lambda stage, count: notices.put((stage, count)),
                )
                future.add_done_callback(lambda _: notices.put(None))

                while True:
                    try:
                        notice = await asyncio.to_thread(notices.get, True, 0.25)
                    except Empty:
                        yield _sse_frame({
                            "type": "progress",
                            "events": [[stage, count] for stage, count in events],
                            "elapsed": round(perf_counter() - started, 2),
                        })
                        continue
                    if notice is None:
                        break
                    events.append(notice)
                    yield _sse_frame({
                        "type": "progress",
                        "events": [[stage, count] for stage, count in events],
                        "elapsed": round(perf_counter() - started, 2),
                    })

                elapsed = perf_counter() - started
                try:
                    result = future.result()
                except Exception as exc:
                    if isinstance(exc, LLMError):
                        logger.warning("Chat model unavailable: %s", exc)
                        message = "Dịch vụ trả lời chưa sẵn sàng. Hãy kiểm tra cấu hình OpenRouter."
                    else:
                        logger.error(
                            "Chat turn failed (%s). Check retrieval services and model settings.",
                            type(exc).__name__,
                        )
                        message = "Tra cứu gặp lỗi. Hãy kiểm tra Qdrant và các model rồi thử lại."
                    yield _sse_frame({
                        "type": "error",
                        "message": message,
                        "events": [[stage, count] for stage, count in events],
                        "elapsed": round(elapsed, 2),
                    })
                    return
            finally:
                executor.shutdown(wait=False)

            if request.conversation_id is not None:
                conversation = store.get_conversation(request.conversation_id)
                if conversation is None:
                    conversation = {
                        "id": request.conversation_id,
                        "title": text[:65] or DEFAULT_TITLE,
                        "pending": None,
                        "messages": [],
                    }
            else:
                conversation = {
                    "id": str(uuid4()),
                    "title": text[:65] or DEFAULT_TITLE,
                    "pending": None,
                    "messages": [],
                }

            messages = conversation["messages"]
            if not messages:
                conversation["title"] = text[:65] or DEFAULT_TITLE
            messages.append({"role": "user", "content": text})

            if isinstance(result, ClarifyPlan):
                conversation["pending"] = {
                    "query": original_query,
                    "clarifications": clarifications,
                    "question": result.clarification_question,
                    "history": [asdict(item) for item in snapshot],
                }
                messages.append({
                    "role": "assistant",
                    "content": result.clarification_question,
                })
                outcome = "clarified"
            else:
                conversation["pending"] = None
                progress_record = build_progress_record(events, outcome="answered", elapsed=elapsed)
                assistant_record = {
                    "role": "assistant",
                    "content": result.answer,
                    "sources": [enrich_citation(chunk) for chunk in result.cited_chunks],
                }
                messages.extend([progress_record, assistant_record])
                outcome = "answered"

            saved = store.save_conversation(conversation)
            yield _sse_frame({
                "type": "complete",
                "outcome": outcome,
                "conversation": saved,
                "elapsed": round(elapsed, 2),
            })

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/api/documents")
    def api_list_documents() -> dict[str, Any]:
        return {"documents": list_documents(documents)}

    @app.get("/api/documents/{doc_id}")
    def api_get_document(
        doc_id: str,
        chunk: str | None = None,
        used: list[str] = Query(default=[]),
    ) -> dict[str, Any]:
        try:
            return get_document_payload(documents, doc_id, chunk_id=chunk, used_chunks=used)
        except KeyError as exc:
            if exc.args[0] == "document":
                raise HTTPException(status_code=404, detail="Không tìm thấy tài liệu.") from exc
            raise HTTPException(
                status_code=404,
                detail="Không tìm thấy đoạn trích dẫn trong tài liệu này.",
            ) from exc

    dist_dir = frontend_dist.resolve() if frontend_dist is not None else None
    index_html = dist_dir / "index.html" if dist_dir is not None else None

    if dist_dir is not None and (dist_dir / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=dist_dir / "assets"), name="vite-assets")

    @app.get("/documents", response_model=None)
    def documents_spa(doc: str | None = None, chunk: str | None = None):
        if doc is not None and doc not in documents.ids:
            return HTMLResponse("Không tìm thấy tài liệu.", status_code=404)
        if chunk is not None:
            if doc is None or not any(row["chunk_id"] == chunk for row in documents.rows(doc)):
                return HTMLResponse("Không tìm thấy đoạn trích dẫn trong tài liệu này.", status_code=404)
        if index_html is not None and index_html.is_file():
            return FileResponse(index_html)
        return _missing_build_response()

    @app.get("/", response_model=None)
    def root_spa():
        if index_html is not None and index_html.is_file():
            return FileResponse(index_html)
        return _missing_build_response()

    @app.get("/{asset_name:path}", response_model=None)
    def static_or_spa(asset_name: str):
        if asset_name.startswith("api/") or asset_name == "api":
            raise HTTPException(status_code=404, detail="Not found")
        if dist_dir is not None:
            candidate = (dist_dir / asset_name).resolve()
            try:
                candidate.relative_to(dist_dir)
            except ValueError:
                raise HTTPException(status_code=404, detail="Not found")
            if candidate.is_file():
                return FileResponse(candidate)
        if index_html is not None and index_html.is_file():
            return FileResponse(index_html)
        return _missing_build_response()

    return app
