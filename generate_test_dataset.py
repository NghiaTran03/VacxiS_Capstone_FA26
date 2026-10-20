"""Generate a golden evaluation dataset for the Vaccine RAG system.

Thin CLI shim around :mod:`vacxis_capstone_fa26.testdata.cli`. Run with::

    uv run generate_test_dataset.py
    uv run generate_test_dataset.py --per-type 8 --out test_data.json
    uv run generate_test_dataset.py --seed 42 --distractors 3
    uv run generate_test_dataset.py --refine-spans   # opt-in LLM span upgrade
"""
from __future__ import annotations

from vacxis_capstone_fa26.testdata.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
