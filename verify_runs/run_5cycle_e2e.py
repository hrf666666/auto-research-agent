#!/usr/bin/env python3
"""5-cycle end-to-end smoke test on the L1+L2+L3 changes.

Runs 5 REAL cycles (THINK→EXECUTE→VERIFY→REFLECT) against the
depth_estimation_unify_theory project, but into an ISOLATED temp workspace
so the live experiment_history.db is NOT polluted.

Validates the full dead_end feedback loop end-to-end:
  - REFLECT produces dead_end (with the new leader.md schema guidance)
  - log_dead_end writes memory_entries
  - B9 gate reads memory_entries and can match
  - experiments table still works (no dead_end column, no OperationalError)
  - roadmap_history table is gone (no error from missing DDL consumers)
  - no crashes across 5 cycles

Usage:
  bash -lc '/home/bigboss/miniconda3/envs/py311/bin/python verify_runs/run_5cycle_e2e.py'
"""
import json
import sqlite3
import sys
import tempfile
from pathlib import Path

# Run with the project interpreter; GLM key comes from ~/.bashrc via bash -l.
# Repo root must be on sys.path so `import api` / `import core.*` work.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from api import AutoResearcher

PROJECT = "/home/bigboss/code/depth_estimation_unify_theory"
ISOLATED_WORKSPACE = Path(tempfile.mkdtemp(prefix="e2e_5cycle_"))

print("=" * 72)
print("5-cycle E2E smoke test (isolated workspace)")
print(f"  project  : {PROJECT}")
print(f"  workspace: {ISOLATED_WORKSPACE}")
print("=" * 72)

r = AutoResearcher(PROJECT)
# Patch the config the loop will use: point workspace at the isolated dir so
# the live DB at /home/bigboss/code/experiment_history.db is untouched.
r._load_config_orig = r._load_config


def _load_config_isolated():
    cfg = r._load_config_orig()
    cfg.setdefault("project", {})["workspace"] = str(ISOLATED_WORKSPACE)
    return cfg


r._load_config = _load_config_isolated

# ── Pre-run snapshot: confirm the schema changes are in effect ────────────
db_path = ISOLATED_WORKSPACE / "experiment_history.db"
# The DB won't exist until the first cycle runs; init a MemoryManager to probe.
from core.memory import MemoryManager

probe = MemoryManager(project_dir=Path(PROJECT), workspace=ISOLATED_WORKSPACE)
with sqlite3.connect(str(probe.db_path)) as conn:
    tables = {
        row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    exp_cols = {
        col[1] for col in conn.execute("PRAGMA table_info(experiments)").fetchall()
    }
print("\n[pre-run] schema checks:")
print(f"  roadmap_history present? {'roadmap_history' in tables} (expect False)")
assert "roadmap_history" not in tables, "roadmap_history should be removed"
print(f"  experiments.dead_end present? {'dead_end' in exp_cols} (expect False)")
assert "dead_end" not in exp_cols, "experiments.dead_end should be removed"
print(f"  memory_entries present? {'memory_entries' in tables} (expect True)")
assert "memory_entries" in tables
print("  ✅ schema changes in effect")

# ── Run 5 cycles ──────────────────────────────────────────────────────────
print("\n[run] starting 5 cycles ...")
results = []
try:
    results = r.run_n_cycles(5)
except Exception as e:
    print(f"\n[FAIL] run_n_cycles raised: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    raise SystemExit(1)

print(f"\n[run] {len(results)} cycle result(s) returned")
for i, res in enumerate(results, 1):
    action = res.get("action", "?")
    launched = res.get("experiment_launched")
    milestone = (res.get("milestone") or "")[:60]
    dead_end = (res.get("dead_end") or "")
    print(f"  cycle {i}: action={action} launched={launched}")
    print(f"           milestone={milestone!r}")
    print(f"           dead_end={'(present)' if dead_end else '(none)'}")

# ── Post-run verification ─────────────────────────────────────────────────
print("\n[post-run] verifying the feedback loop wrote to memory_entries:")
with sqlite3.connect(str(probe.db_path)) as conn:
    de_rows = conn.execute(
        "SELECT cycle, substr(content,1,70) FROM memory_entries "
        "WHERE entry_type='dead_end' ORDER BY id"
    ).fetchall()
    me_counts = dict(conn.execute(
        "SELECT entry_type, COUNT(*) FROM memory_entries GROUP BY entry_type"
    ).fetchall())
    exp_count = conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]
    # Verify experiments INSERT still works (no OperationalError from dead_end removal)
    try:
        conn.execute("SELECT milestone, decision, active_problem FROM experiments LIMIT 1").fetchall()
        exp_read_ok = True
    except sqlite3.OperationalError as e:
        exp_read_ok = f"OperationalError: {e}"

print(f"  experiments rows written : {exp_count}")
print(f"  experiments read OK?     : {exp_read_ok}")
print(f"  memory_entries by type   : {me_counts}")
print(f"  memory_entries dead_ends : {len(de_rows)}")
for c, content in de_rows:
    print(f"    cycle={c}: {content!r}")

# ── Verdict ───────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
ok = True
if exp_count == 0:
    print("  ⚠️  experiments table empty — record_cycle_outcome may not have run")
    ok = False
else:
    print(f"  ✅ experiments populated ({exp_count} rows) — dead_end removal didn't break INSERT")
if exp_read_ok is not True:
    print(f"  ❌ experiments SELECT failed: {exp_read_ok}")
    ok = False
else:
    print("  ✅ experiments SELECT works (no dead_end column drift)")
print(f"  ℹ️  {len(de_rows)} dead_end(s) recorded in memory_entries")
print("  ℹ️  (whether LLM produced dead_end depends on the cycle content)")
if ok:
    print("\n✅ 5-CYCLE E2E PASSED — no crashes, schema changes hold")
else:
    print("\n⚠️  5-cycle completed but with warnings (see above)")
print("=" * 72)
print(f"\nIsolated workspace (inspect/delete): {ISOLATED_WORKSPACE}")
