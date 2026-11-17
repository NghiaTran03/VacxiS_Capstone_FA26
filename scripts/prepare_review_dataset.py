"""Convert DeepEval goldens into the VacxiS review import format."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
import pyarrow.compute as pc


def load_context_index(path):
    columns = ["chunk_id", "doc_id", "doc_title", "category", "header", "text"]
    table = pq.read_table(path, columns=columns)
    index = defaultdict(set)
    for row in table.filter(pc.is_valid(table["text"])).to_pylist():
        text = row["text"].strip()
        if not text:
            continue
        header = " | ".join(row[key] for key in ("doc_title", "category", "header") if row[key])
        context = f"[{header}]\n{text}" if header else text
        index[context].add((row["chunk_id"], row["doc_id"]))
    return index


def convert_goldens(payload, index):
    records = []
    for golden in payload["goldens"]:
        metadata = golden["additional_metadata"]
        query_id = metadata["query_id"]
        chunks = []
        documents = []
        contexts = golden["context"]
        if not isinstance(contexts, list) or not contexts:
            raise ValueError(f"{query_id}: expected a nonempty list of source contexts.")
        for position, context in enumerate(contexts, start=1):
            matches = index.get(context, set())
            if len(matches) != 1:
                raise ValueError(f"{query_id}: source context {position} matched {len(matches)} chunks.")
            chunk_id, doc_id = next(iter(matches))
            chunks.append({"CHUNK_ID": chunk_id, "TEXT": context, "RELEVANCE_SCORE": 3})
            documents.append(doc_id)
        records.append({
            "QUERY_ID": query_id,
            "QUERY": golden["input"],
            "REFERENCE_ANSWER": golden["expected_output"],
            "ANSWERABILITY": metadata["answerability"],
            "QUERY_COMPLEXITY": metadata["complexity"],
            "QUERY_STYLE": metadata["query_style"],
            "MULTI_HOP_TYPE": metadata["multi_hop_type"],
            "RELEVANT_DOCS": list(dict.fromkeys(documents)),
            "RELEVANT_CHUNKS": chunks,
            "VERIFIED": False,
            "ANNOTATOR_IDS": ["GEN"],
            "ADJUDICATED": False,
            "VERIFICATION_NOTES": None,
        })
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("goldens", type=Path)
    parser.add_argument("chunks", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        payload = json.loads(args.goldens.read_text(encoding="utf-8-sig"))
        records = convert_goldens(payload, load_context_index(args.chunks))
        content = json.dumps(records, ensure_ascii=False, allow_nan=False, indent="\t")
        with args.output.open("x", encoding="utf-8", newline="\n") as output:
            output.write(content + "\n")
    except (OSError, KeyError, TypeError, ValueError) as error:
        parser.exit(1, f"{error}\n")
    print(f"Prepared {len(records)} questions in {args.output}")


if __name__ == "__main__":
    main()
