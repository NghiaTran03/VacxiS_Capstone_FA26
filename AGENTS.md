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


## Strict Execution Lifecycle & Transition Gates

Every code modification task must strictly adhere to the sequential lifecycle below. Skipping or reordering steps is strictly prohibited:

1. **Phase 1: Implementation**
   - Apply source code edits according to the user's requirements and prompt constraints.

2. **Phase 2: Code Review Gate (MANDATORY)**
   - **Trigger:** Immediately upon completing code modifications, BEFORE running test suites and BEFORE writing any walkthrough/summary.
   - **Action:** Delegate execution to the `code-reviewer` agent (defined in `.agents/agents/code-reviewer/agent.md`).
   - **Inspection Scope:**
     * Run `git status --short` and `git diff` (including `--cached` if staged).
     * Verify 100% diff alignment against the initial prompt to detect omissions, unintended changes, or scope creep.
     * Perform deep static analysis for logic flaws, edge cases, regression risks, broken contracts, and stray debug code.
   - **Gate Rule:**
     * If verdict is **`[BLOCKED]`**: Halt forward progress immediately. Return to Phase 1 to patch the flagged issues, then re-trigger Phase 2.
     * Proceed to Phase 3 ONLY when verdict is explicitly marked **`[PASSED]`**.

3. **Phase 3: Automated Testing & Verification**
   - Execute linters, type checkers, and relevant unit/integration test suites.
   - If tests fail due to implementation bugs, loop back to Phase 1.

4. **Phase 4: Summary & Walkthrough**
   - **Entry Condition:** Phase 2 is **`[PASSED]`** AND Phase 3 tests pass cleanly.
   - Synthesize changes, explain implementation details, and produce the final user-facing summary walkthrough.

---

### Non-Negotiable Directives
- **Zero-Bypass Policy:** Never invoke test runners, generate summary artifacts, or declare task completion without an explicit `[PASSED]` verdict from the `code-reviewer` agent.
- **Context Preservation:** When handing off to `code-reviewer`, pass the exact verbatim user prompt along with relevant issue/ticket context to serve as the ground truth.

---

## Coding & Output Style
- Code comment, docstring must be written in English.
- Do not use separator lines, banner dividers, or rows of equal signs (=====).
- Keep the code, terminal and chat outputs clean and concise.
