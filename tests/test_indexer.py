"""Unit tests for vacxis_capstone_fa26.indexing.indexer."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from qdrant_client import models

from vacxis_capstone_fa26.indexing.indexer import (
    convert_sparse_to_qdrant,
    init_qdrant_collection,
    load_chunks_from_parquet,
)


def test_convert_sparse_to_qdrant_empty():
    """Test convert_sparse_to_qdrant with None or empty dict."""
    res_none = convert_sparse_to_qdrant(None)
    assert isinstance(res_none, models.SparseVector)
    assert res_none.indices == []
    assert res_none.values == []

    res_empty = convert_sparse_to_qdrant({})
    assert isinstance(res_empty, models.SparseVector)
    assert res_empty.indices == []
    assert res_empty.values == []


def test_convert_sparse_to_qdrant_values():
    """Test convert_sparse_to_qdrant with string token ids and weights."""
    sample_weights = {
        "101": 0.45,
        "2045": 1.25,
        "309": 0.08,
    }
    sparse_vec = convert_sparse_to_qdrant(sample_weights)
    assert isinstance(sparse_vec, models.SparseVector)
    assert sparse_vec.indices == [101, 2045, 309]
    assert [round(v, 2) for v in sparse_vec.values] == [0.45, 1.25, 0.08]


def test_init_qdrant_collection_creation():
    """Test init_qdrant_collection creates collection if not existing."""
    mock_client = MagicMock()
    mock_collection = MagicMock()
    mock_collection.name = "Other_Collection"
    mock_client.get_collections.return_value.collections = [mock_collection]

    init_qdrant_collection(mock_client, collection_name="VacxiS_Capstone_FA26", recreate=True)

    mock_client.delete_collection.assert_not_called()
    mock_client.create_collection.assert_called_once()
    call_kwargs = mock_client.create_collection.call_args.kwargs
    assert call_kwargs["collection_name"] == "VacxiS_Capstone_FA26"
    assert "dense" in call_kwargs["vectors_config"]
    assert "sparse" in call_kwargs["sparse_vectors_config"]


def test_init_qdrant_collection_recreate():
    """Test init_qdrant_collection deletes existing collection when recreate=True."""
    mock_client = MagicMock()
    mock_collection = MagicMock()
    mock_collection.name = "VacxiS_Capstone_FA26"
    mock_client.get_collections.return_value.collections = [mock_collection]

    init_qdrant_collection(mock_client, collection_name="VacxiS_Capstone_FA26", recreate=True)

    mock_client.delete_collection.assert_called_once_with("VacxiS_Capstone_FA26")
    mock_client.create_collection.assert_called_once()


def test_load_chunks_from_parquet(tmp_path: Path):
    """Test loading chunks from parquet using PyArrow."""
    test_parquet = tmp_path / "test_chunks.parquet"

    schema = pa.schema([
        ("chunk_id", pa.string()),
        ("doc_id", pa.string()),
        ("doc_title", pa.string()),
        ("category", pa.string()),
        ("parent_header", pa.string()),
        ("header", pa.string()),
        ("content_type", pa.string()),
        ("text", pa.string()),
        ("metadata", pa.struct([
            ("has_table", pa.bool_()),
            ("source_file", pa.string()),
        ])),
    ])

    data = {
        "chunk_id": ["chk_1"],
        "doc_id": ["doc_1"],
        "doc_title": ["Doc 1"],
        "category": ["chỉ định"],
        "parent_header": [""],
        "header": ["Section 1"],
        "content_type": ["text"],
        "text": ["Content text"],
        "metadata": [{"has_table": False, "source_file": "file.md"}],
    }

    table = pa.Table.from_pydict(data, schema=schema)
    pq.write_table(table, str(test_parquet))

    records = load_chunks_from_parquet(test_parquet)
    assert len(records) == 1
    assert records[0]["chunk_id"] == "chk_1"
    assert records[0]["category"] == "chỉ định"
    assert records[0]["metadata"]["source_file"] == "file.md"
