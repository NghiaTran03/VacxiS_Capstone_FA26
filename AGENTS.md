# Agent Directive

## Environment & Runtime (Python)
- **Tooling Mandate**: You must strictly use `uv` for all Python operations. Do not use standard `python3`, `python`, `pip`, `venv`, `conda`, or `poetry` unless explicitly instructed otherwise.
- **Execution Rules**:
  - Run scripts using: `uv run <script_name>.py` or `uv run python -c ...`
  - Request install new dependencies if not found using: `uv add <package>` or `uv pip install <package>`
- **Performance Priority**: Leverage `uv`'s caching mechanisms to ensure ultra-fast execution of pipelines and autonomous tasks.

### Agent Skill: Parquet Data Operations

**Trigger:** Reading, querying, filtering, aggregating, or writing `.parquet` files.

**Core Rules:**
1. **Mandatory Imports:**
   Always use PyArrow for native Parquet handling:
   ```python
   import pyarrow.parquet as pq
   import pyarrow.compute as pc
   ```

## Coding & Output Style
- Do not use separator lines, banner dividers, or rows of equal signs (=====).
- Keep the code, terminal and chat outputs clean and concise.
