"""Read the DeepEval format and recognize the previous eval import."""

import re


def is_golden(record):
    return {"input", "expected_output", "context", "additional_metadata"} <= record.keys()


def context_title(text):
    headline = text.split("\n", 1)[0]
    if headline.startswith("[") and headline.endswith("]"):
        return headline[1:-1].split(" | ", 1)[0]
    return None


def as_golden(record):
    if is_golden(record):
        return record
    if not (re.fullmatch(r"eval_\d+", record.get("QUERY_ID", ""))
            and {"QUERY_STYLE", "QUERY_COMPLEXITY", "MULTI_HOP_TYPE", "RELEVANT_CHUNKS"} <= record.keys()):
        return None
    chunks = record["RELEVANT_CHUNKS"]
    contexts = [chunk["TEXT"] for chunk in chunks]
    titles = [title for text in contexts if (title := context_title(text))]
    return {
        "input": record["QUERY"],
        "expected_output": record["REFERENCE_ANSWER"],
        "context": contexts,
        "additional_metadata": {
            "query_id": record["QUERY_ID"],
            "answerability": record["ANSWERABILITY"],
            "complexity": record["QUERY_COMPLEXITY"],
            "multi_hop_type": record["MULTI_HOP_TYPE"],
            "query_style": record["QUERY_STYLE"],
            "source_chunk_ids": [chunk["CHUNK_ID"] for chunk in chunks],
            "doc_ids": record.get("RELEVANT_DOCS") or [],
            "doc_titles": list(dict.fromkeys(titles)),
        },
    }
