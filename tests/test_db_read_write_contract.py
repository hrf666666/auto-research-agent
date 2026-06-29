"""L3 anti-regression contract test: SQLite table/column health.

Purpose
-------
Prevent "orphan" data-flow breakages from ever silently re-appearing. Three
classes of bug caused real outages in this codebase:

1. A table with DDL but no INSERT and no SELECT (`roadmap_history` — pure
   zombie, removed 2026-06-29).
2. A table with a SELECT reader but no reachable INSERT writer
   (`get_recent_failures` — dead reader, removed 2026-06-29).
3. A column removed from a table while SELECT sites still referenced it
   (`experiments.dead_end` migration — would raise OperationalError).

This test asserts, at two granularities:

* **Table-level:** every table created by the schema has BOTH an INSERT writer
  AND a SELECT reader in `core/`. Tables explicitly marked KNOWN-BROKEN or
  INTENTIONAL in `docs/DATA_CONTRACT.md` are exempted (with a documented root
  cause).

* **Column-level:** every column referenced in `SELECT ... FROM <table>` and
  `INSERT INTO <table> (...)` statements must actually exist in that table's
  real DDL (queried via `pragma_table_info`). This catches the dead_end class
  of bug: if someone removes a column but leaves a SELECT referencing it, the
  query would OperationalError at runtime — this test fails at test time
  instead.

The set of "real tables" is taken from a live `MemoryManager` instance (truth
source), NOT a hardcoded list — so newly added tables are automatically
covered.
"""

import io
import re
import sqlite3
import tempfile
import tokenize
from pathlib import Path

import pytest

from core.memory import MemoryManager

REPO_ROOT = Path(__file__).resolve().parent.parent
CORE_DIR = REPO_ROOT / "core"
DATA_CONTRACT = REPO_ROOT / "docs" / "DATA_CONTRACT.md"

# ──────────────────────────────────────────────────────────────────────
# Helpers: extract SQL tables/columns from core/ source
# ──────────────────────────────────────────────────────────────────────

_INSERT_TABLE_RE = re.compile(
    r"INSERT\s+(?:OR\s+\w+\s+)?INTO\s+([a-zA-Z_][a-zA-Z0-9_]*)", re.IGNORECASE
)
_FROM_TABLE_RE = re.compile(
    r"(?:FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*)\b", re.IGNORECASE
)
_INSERT_COLS_RE = re.compile(
    r"INSERT\s+(?:OR\s+\w+\s+)?INTO\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(([^)]*)\)",
    re.IGNORECASE | re.DOTALL,
)
_SELECT_COLS_RE = re.compile(
    r"SELECT\s+(.+?)\s+FROM\s+([a-zA-Z_][a-zA-Z0-9_]*)\b",
    re.IGNORECASE | re.DOTALL,
)
_SQL_KW = re.compile(r"\b(SELECT|INSERT|FROM|WHERE|UPDATE|DELETE)\b", re.IGNORECASE)


def _iter_sql_literals(text: str):
    """Yield Python string-literal values from `text` that look like SQL.

    Uses tokenize() to correctly extract every string literal regardless of
    quoting style (triple, single, inline, concatenated), then keeps only
    those containing a SQL keyword. This excludes English prose and import
    statements reliably.
    """
    try:
        toks = tokenize.generate_tokens(io.StringIO(text).readline)
        for tok in toks:
            if tok.type == tokenize.STRING:
                # Evaluate the literal to get its raw value. Fall back to the
                # source text if eval fails (e.g. f-strings — those still
                # contain SQL text we want to scan).
                src = tok.string
                try:
                    val = ast_eval_string(src)
                except Exception:
                    val = src
                if val and _SQL_KW.search(val):
                    yield val
    except tokenize.TokenizeError:
        return


def ast_eval_string(src: str) -> str:
    """Safely get the text value of a string-literal token source."""
    import ast
    # Only handle plain string prefixes; for f-strings return the source minus
    # the prefix/quotes so embedded SQL text is still scanned.
    p = src
    for pref in ("f", "F", "r", "R", "b", "B", "u", "U", "rb", "Rb", "rf", "Rf"):
        if p.startswith(pref) and len(p) > len(pref) and p[len(pref)] in "\"'":
            # f-string: return inner source (may contain {expr} but SQL text
            # outside braces is what we want).
            inner = p[len(pref):]
            quote = inner[0]
            if inner.endswith(quote) and len(inner) >= 2:
                return inner[1:-1]
            return p
    return ast.literal_eval(src)


def _collect_sql_from_core():
    """Return (insert_tables:set, select_tables:set, table_col_refs:dict).

    `table_col_refs` maps lowercase table name -> {
        'select_cols': set of columns seen in SELECT <list> FROM <t>,
        'insert_cols': set of columns seen in INSERT INTO <t> (list),
    }. Columns are lowercased. `*` and function calls are filtered out.
    """
    insert_tables = set()
    select_tables = set()
    col_refs = {}

    def _ensure(t):
        col_refs.setdefault(t, {"select_cols": set(), "insert_cols": set()})
        return col_refs[t]

    for src_file in sorted(CORE_DIR.glob("*.py")):
        text = src_file.read_text()
        for sql in _iter_sql_literals(text):
            for m in _INSERT_TABLE_RE.finditer(sql):
                insert_tables.add(m.group(1).lower())
            for m in _FROM_TABLE_RE.finditer(sql):
                select_tables.add(m.group(1).lower())
            # INSERT columns
            for m in _INSERT_COLS_RE.finditer(sql):
                t = m.group(1).lower()
                cols_str = m.group(2)
                cols = _parse_col_list(cols_str)
                _ensure(t)["insert_cols"].update(cols)
            # SELECT columns
            for m in _SELECT_COLS_RE.finditer(sql):
                cols_str = m.group(1)
                t = m.group(2).lower()
                cols = _parse_col_list(cols_str)
                _ensure(t)["select_cols"].update(cols)

    return insert_tables, select_tables, col_refs


def _parse_col_list(cols_str: str) -> set:
    """Parse a comma-separated column list, dropping *, aggregates, literals."""
    out = set()
    for raw in cols_str.split(","):
        c = raw.strip()
        if not c:
            continue
        # Drop aggregates/functions/literals: anything containing '(' or "'",
        # or starting with a digit.
        if "(" in c or "'" in c or '"' in c:
            continue
        if c == "*" or c.endswith(".*"):
            continue
        # Take the last identifier after a dot / AS / space.
        c = re.split(r"\s+(?:as|AS)\s+", c)[-1]
        c = c.split(".")[-1].strip()
        if c and re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", c):
            out.add(c.lower())
    return out


# ──────────────────────────────────────────────────────────────────────
# Helpers: DATA_CONTRACT.md exemptions
# ──────────────────────────────────────────────────────────────────────

def _parse_known_broken_tables() -> set:
    """Tables marked KNOWN-BROKEN in docs/DATA_CONTRACT.md (lowercased)."""
    if not DATA_CONTRACT.exists():
        return set()
    text = DATA_CONTRACT.read_text()
    broken = set()
    # Each table section starts with "### N. `table_name`".
    sections = re.split(r"(?=^### \d+\. `)", text, flags=re.MULTILINE)
    for sec in sections:
        m = re.match(r"### \d+\. `([a-zA-Z_][a-zA-Z0-9_]*)`", sec)
        if m and re.search(r"Status\s*\|\s*`?KNOWN-BROKEN`?", sec):
            broken.add(m.group(1).lower())
    return broken


def _parse_removed_tables() -> set:
    """Tables marked as REMOVED in docs/DATA_CONTRACT.md (lowercased)."""
    if not DATA_CONTRACT.exists():
        return set()
    text = DATA_CONTRACT.read_text()
    removed = set()
    for m in re.finditer(r"### \d+\. `([a-zA-Z_][a-zA-Z0-9_]*)`\s*—\s*REMOVED", text):
        removed.add(m.group(1).lower())
    return removed


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def live_tables():
    """Instantiate a real MemoryManager and return {table_name: set(columns)}.

    Uses a temp project/workspace so no real DB is touched. This is the truth
    source for which tables actually exist and what columns they have at
    runtime.

    We exercise the two runtime DDL side-effects so the column snapshot matches
    what a real run produces:
      * `scan_experiment_facts()` -> fact_scanner `_ensure_table` creates the
        `experiment_facts` table.
      * `log_dead_end(...)` -> runtime `ALTER TABLE memory_entries ADD COLUMN
        failure_category` (a known DDL gap documented in DATA_CONTRACT.md).
    Without these, the snapshot would miss experiment_facts entirely and miss
    failure_category — both real-at-runtime artifacts.
    """
    tmp = Path(tempfile.mkdtemp())
    mm = MemoryManager(project_dir=tmp, workspace=tmp)
    # Trigger runtime table/column creation.
    try:
        mm.scan_experiment_facts()
    except Exception:
        pass  # no outputs/ dir in tmp workspace; _ensure_table still ran
    try:
        mm.log_dead_end("probe entry to materialize failure_category column")
    except Exception:
        pass
    tables = {}
    with sqlite3.connect(str(mm.db_path)) as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        for (name,) in rows:
            cols = {
                r[1].lower()
                for r in conn.execute(f"PRAGMA table_info({name})").fetchall()
            }
            tables[name.lower()] = cols
    return tables


@pytest.fixture(scope="module")
def collected():
    """Static SQL scan of core/."""
    return _collect_sql_from_core()


# ──────────────────────────────────────────────────────────────────────
# Table-level contract
# ──────────────────────────────────────────────────────────────────────

class TestTableLevelContract:
    """Every real table must have both an INSERT writer and a SELECT reader.

    Catches:
    - zombie tables (DDL only, no read/write) like the removed roadmap_history
    - dead-reader tables (SELECT but no INSERT)
    """

    def test_every_table_has_writer_and_reader(self, live_tables, collected):
        insert_tables, select_tables, _ = collected
        broken = _parse_known_broken_tables()
        missing_rw = []
        for table in sorted(live_tables):
            has_write = table in insert_tables
            has_read = table in select_tables
            if has_write and has_read:
                continue
            if table in broken:
                # KNOWN-BROKEN tables may legitimately be single-sided; the
                # DATA_CONTRACT.md documents the root cause.
                continue
            missing_rw.append(
                f"  {table}: INSERT={'yes' if has_write else 'NO'}, "
                f"SELECT={'yes' if has_read else 'NO'}"
            )
        assert not missing_rw, (
            "Tables without a complete read/write pair (and not marked "
            "KNOWN-BROKEN in docs/DATA_CONTRACT.md):\n" + "\n".join(missing_rw)
        )

    def test_removed_tables_do_not_reappear(self, live_tables):
        """Tables marked REMOVED in DATA_CONTRACT.md must not exist in the schema."""
        removed = _parse_removed_tables()
        resurrected = sorted(set(live_tables) & removed)
        assert not resurrected, (
            "These tables were marked REMOVED but re-appeared in the schema: "
            + ", ".join(resurrected)
        )

    def test_known_broken_tables_are_documented(self, live_tables):
        """Every KNOWN-BROKEN entry must still exist in the schema.

        If a KNOWN-BROKEN table was actually fixed/removed, its entry must be
        cleaned from DATA_CONTRACT.md so the exemption list stays accurate.
        """
        broken = _parse_known_broken_tables()
        stale = sorted(broken - set(live_tables))
        assert not stale, (
            "Tables marked KNOWN-BROKEN in DATA_CONTRACT.md no longer exist "
            "in the schema — remove their entry: " + ", ".join(stale)
        )


# ──────────────────────────────────────────────────────────────────────
# Column-level contract
# ──────────────────────────────────────────────────────────────────────

class TestColumnLevelContract:
    """Every column referenced in SQL must exist in the table's real DDL.

    Catches the dead_end class of bug: a column dropped from a table while
    SELECT/INSERT sites still reference it — which raises OperationalError at
    runtime.
    """

    def test_select_columns_exist_in_ddl(self, live_tables, collected):
        _, _, col_refs = collected
        broken = _parse_known_broken_tables()
        problems = []
        for table, refs in sorted(col_refs.items()):
            if table not in live_tables:
                continue  # not a managed table (false positive from prose)
            if table in broken:
                continue
            real_cols = live_tables[table]
            for col in sorted(refs["select_cols"]):
                if col not in real_cols:
                    problems.append(
                        f"  SELECT references {table}.{col} but column does not exist "
                        f"(real cols: {sorted(real_cols)})"
                    )
        assert not problems, (
            "SELECT statements reference non-existent columns "
            "(would raise OperationalError at runtime):\n" + "\n".join(problems)
        )

    def test_insert_columns_exist_in_ddl(self, live_tables, collected):
        _, _, col_refs = collected
        broken = _parse_known_broken_tables()
        problems = []
        for table, refs in sorted(col_refs.items()):
            if table not in live_tables:
                continue
            if table in broken:
                continue
            real_cols = live_tables[table]
            for col in sorted(refs["insert_cols"]):
                if col not in real_cols:
                    problems.append(
                        f"  INSERT references {table}.{col} but column does not exist "
                        f"(real cols: {sorted(real_cols)})"
                    )
        assert not problems, (
            "INSERT statements reference non-existent columns:\n" + "\n".join(problems)
        )

    def test_dead_end_migrated_from_experiments(self, live_tables):
        """Regression guard: experiments.dead_end must stay removed.

        The dead_end column was migrated to memory_entries. If it silently
        re-appears in experiments, the B9 gate (which reads memory_entries)
        would again diverge from any experiments-based reader. This test
        pins the migration in place.
        """
        assert "dead_end" not in live_tables["experiments"], (
            "experiments.dead_end re-appeared — it was removed as part of the "
            "dead_end→memory_entries migration (see docs/DATA_CONTRACT.md). "
            "Either keep it removed or update the contract + B9 gate."
        )
        # And the B9 gate's source of truth (memory_entries) must remain.
        assert "memory_entries" in live_tables
