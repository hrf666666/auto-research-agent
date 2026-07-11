"""Tests for failure_category DDL migration and dead-end persistence."""
from __future__ import annotations

import sqlite3

from core.memory import MemoryManager


def test_fresh_db_has_failure_category_column(tmp_path):
    (tmp_path / "PROJECT_BRIEF.md").write_text("# brief")
    mm = MemoryManager(project_dir=tmp_path, workspace=tmp_path)

    with sqlite3.connect(str(mm.db_path)) as conn:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(memory_entries)").fetchall()}

    assert "failure_category" in cols


def test_dead_end_with_category_persists(tmp_path):
    (tmp_path / "PROJECT_BRIEF.md").write_text("# brief")
    mm = MemoryManager(project_dir=tmp_path, workspace=tmp_path)

    mm.log_dead_end("FFT fails on sparse data", cycle=5, failure_category="hypothesis_wrong")

    entries = mm.get_dead_ends_by_category()
    assert len(entries) == 1
    assert entries[0]["failure_category"] == "hypothesis_wrong"
    assert entries[0]["cycle"] == 5


def test_dead_end_invalid_category_empties(tmp_path):
    (tmp_path / "PROJECT_BRIEF.md").write_text("# brief")
    mm = MemoryManager(project_dir=tmp_path, workspace=tmp_path)

    mm.log_dead_end("bad approach", cycle=1, failure_category="totally_bogus")

    entries = mm.get_dead_ends_by_category()
    assert len(entries) == 1
    assert entries[0]["failure_category"] == ""


def test_old_db_without_column_gets_migrated(tmp_path):
    (tmp_path / "PROJECT_BRIEF.md").write_text("# brief")
    db_path = tmp_path / "experiment_history.db"

    # Create a DB without failure_category (simulating pre-v12)
    with sqlite3.connect(str(db_path)) as conn:
        conn.executescript("""
            CREATE TABLE experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cycle INTEGER NOT NULL,
                timestamp REAL NOT NULL,
                action TEXT NOT NULL DEFAULT '',
                hypothesis TEXT NOT NULL DEFAULT '',
                success_criteria TEXT NOT NULL DEFAULT '',
                agent_type TEXT NOT NULL DEFAULT '',
                task_summary TEXT NOT NULL DEFAULT '',
                experiment_launched INTEGER NOT NULL DEFAULT 0,
                pid INTEGER,
                log_file TEXT NOT NULL DEFAULT '',
                verify_pass INTEGER,
                verify_fail INTEGER,
                verify_warnings INTEGER,
                verify_diagnosis TEXT NOT NULL DEFAULT '',
                metrics_json TEXT NOT NULL DEFAULT '{}',
                milestone TEXT NOT NULL DEFAULT '',
                decision TEXT NOT NULL DEFAULT '',
                active_problem TEXT NOT NULL DEFAULT '',
                module_failure TEXT NOT NULL DEFAULT '',
                duration_seconds REAL,
                notes TEXT NOT NULL DEFAULT ''
            );
            CREATE TABLE memory_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                entry_type TEXT NOT NULL,
                content TEXT NOT NULL,
                cycle INTEGER,
                in_llm_context INTEGER NOT NULL DEFAULT 0
            );
        """)

    # Now open with MemoryManager — migration should add the column
    mm = MemoryManager(project_dir=tmp_path, workspace=tmp_path)

    with sqlite3.connect(str(db_path)) as conn:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(memory_entries)").fetchall()}

    assert "failure_category" in cols
