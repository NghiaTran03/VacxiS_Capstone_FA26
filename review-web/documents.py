"""Read the packaged source documents without a runtime Parquet dependency."""

from functools import lru_cache
import json
from pathlib import Path


@lru_cache(maxsize=1)
def packaged_documents():
    path = Path(__file__).resolve().parent / "data" / "documents.json"
    documents = {}
    for row in json.loads(path.read_text(encoding="utf-8")):
        documents.setdefault(row["doc_id"], []).append(row)
    return documents
