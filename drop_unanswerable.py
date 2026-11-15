"""Throwaway script: drop UNANSWERABLE records from test_dataset_masked.json.

Writes the surviving (ANSWERABLE + PARTIALLY_ANSWERABLE) records to
test_dataset_masked_no_unanswerable.json and prints the counts.
"""
import json
from collections import Counter
from pathlib import Path

IN = Path("test_dataset_masked.json")
OUT = Path("test_dataset_masked_no_unanswerable.json")

records = json.loads(IN.read_text(encoding="utf-8"))
kept = [r for r in records if r.get("ANSWERABILITY") != "UNANSWERABLE"]

OUT.write_text(json.dumps(kept, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

print(f"{IN.name}: {len(records)} records")
print("dropped UNANSWERABLE:", len(records) - len(kept))
print("kept by label:", dict(Counter(r["ANSWERABILITY"] for r in kept)))
print(f"wrote {OUT.name}")
