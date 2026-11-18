"""Package full Parquet documents referenced by a review dataset."""

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq
import pyarrow.compute as pc
import pyarrow as pa


def prepare_documents(dataset, parquet):
    payload = json.loads(dataset.read_text(encoding="utf-8-sig"))
    identifiers = {doc for golden in payload["goldens"] for doc in golden["additional_metadata"]["doc_ids"]}
    table = pq.read_table(parquet, columns=["chunk_id", "doc_id", "doc_title", "category", "header", "text"])
    table = table.filter(pc.is_in(table["doc_id"], value_set=pa.array(sorted(identifiers))))
    rows = table.to_pylist()
    missing = identifiers - {row["doc_id"] for row in rows}
    if missing:
        raise ValueError(f"Missing documents: {', '.join(sorted(missing))}")
    for row in rows:
        row["position"] = int(row["chunk_id"].rsplit("_c_", 1)[1])
        for key in ("doc_title", "category", "header", "text"):
            row[key] = row[key] or ""
    return sorted(rows, key=lambda row: (row["doc_id"], row["position"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("parquet", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    records = prepare_documents(args.dataset, args.parquet)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared {len(records)} chunks in {args.output}")
