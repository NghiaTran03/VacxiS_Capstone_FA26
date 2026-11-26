"""Server-side JSON storage for VaxiS web-vite conversations."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import threading
from typing import Any
from uuid import uuid4

DEFAULT_TITLE = "Cuộc trò chuyện mới"
_VALID_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_conversation_id(conversation_id: str) -> str:
    """Ensure a conversation ID is safe to use as a filename stem."""
    if not isinstance(conversation_id, str) or not _VALID_ID_PATTERN.fullmatch(conversation_id):
        raise ValueError("Invalid conversation ID.")
    return conversation_id


def count_visible_messages(messages: list[dict[str, Any]]) -> int:
    """Count real user and assistant messages, excluding progress records."""
    return sum(
        1
        for message in messages
        if message.get("role") in {"user", "assistant"} and not message.get("is_progress")
    )


class ConversationStore:
    """Thread-safe JSON file store with one file per conversation."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = Path(data_dir)
        self._lock = threading.Lock()

    def _ensure_dir(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, conversation_id: str) -> Path:
        safe_id = validate_conversation_id(conversation_id)
        return self.data_dir / f"{safe_id}.json"

    def _write_atomic(self, path: Path, payload: dict[str, Any]) -> None:
        self._ensure_dir()
        temp_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            temp_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            os.replace(temp_path, path)
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise

    def create_conversation(
        self,
        conversation_id: str | None = None,
        *,
        title: str = DEFAULT_TITLE,
    ) -> dict[str, Any]:
        """Create and persist a blank conversation."""
        cid = validate_conversation_id(conversation_id) if conversation_id is not None else str(uuid4())
        now = _utc_now_iso()
        conversation = {
            "id": cid,
            "created_at": now,
            "updated_at": now,
            "title": title,
            "pending": None,
            "messages": [],
        }
        with self._lock:
            self._write_atomic(self._path(cid), conversation)
        return deepcopy(conversation)

    def get_conversation(self, conversation_id: str) -> dict[str, Any] | None:
        """Load a single conversation by ID, returning None if it does not exist."""
        path = self._path(conversation_id)
        with self._lock:
            if not path.is_file():
                return None
            return json.loads(path.read_text(encoding="utf-8"))

    def save_conversation(self, conversation: dict[str, Any]) -> dict[str, Any]:
        """Persist an updated conversation payload atomically."""
        cid = validate_conversation_id(conversation["id"])
        now = _utc_now_iso()
        payload = deepcopy(conversation)
        payload.setdefault("created_at", now)
        payload["updated_at"] = now
        payload.setdefault("title", DEFAULT_TITLE)
        payload.setdefault("pending", None)
        payload.setdefault("messages", [])
        with self._lock:
            self._write_atomic(self._path(cid), payload)
        return deepcopy(payload)

    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation file if present."""
        path = self._path(conversation_id)
        with self._lock:
            if not path.is_file():
                return False
            path.unlink()
            return True

    def list_conversations(self) -> list[dict[str, Any]]:
        """Return conversation metadata sorted from newest to oldest."""
        with self._lock:
            if not self.data_dir.is_dir():
                return []
            items: list[dict[str, Any]] = []
            for path in self.data_dir.glob("*.json"):
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                if not isinstance(data, dict) or not data.get("id"):
                    continue
                messages = data.get("messages") if isinstance(data.get("messages"), list) else []
                created_at = str(data.get("created_at") or "")
                updated_at = str(data.get("updated_at") or created_at)
                items.append({
                    "id": str(data["id"]),
                    "created_at": created_at,
                    "updated_at": updated_at,
                    "title": str(data.get("title") or DEFAULT_TITLE),
                    "message_count": count_visible_messages(messages),
                })
        items.sort(key=lambda item: (item["updated_at"], item["created_at"]), reverse=True)
        return items
