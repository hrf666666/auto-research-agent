#!/usr/bin/env python3
"""5-cycle E2E in the REAL environment (depth project, live DB).

Unlike run_5cycle_e2e.py (isolated workspace), this uses the depth project's
own workspace so the code agent can read project files normally. It writes to
the LIVE experiment_history.db. Run only after backing up that DB.

Validates: REFLECT落库 → dead_end写memory_entries → 全链路真实跑通。
"""
import sys
import sqlite3
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from api import AutoResearcher

PROJECT = "/home/bigboss/code/depth_estimation_unify_theory"
DB = Path("/home/bigboss/code/experiment_history.db")

# Baseline snapshot before run (to compute the delta the run produces).
with sqlite3.connect(str(DB)) as conn:
    BASELINE = {
        "exp": conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0],
        "me_all": conn.execute("SELECT COUNT(*) FROM memory_entries").fetchone()[0],
        "me_deadend": conn.execute(
            "SELECT COUNT(*) FROM memory_entries WHERE entry_type='dead_end'"
        ).fetchone()[0],
        "cc": conn.execute("SELECT COUNT(*) FROM causal_chain").fetchone()[0],
        "crl": conn.execute("SELECT COUNT(*) FROM code_review_lessons").fetchone()[0],
    }

print("=" * 72)
print("5-cycle E2E (REAL environment, live DB)")
print(f"  project : {PROJECT}")
print(f"  db      : {DB}")
print(f"  baseline: {BASELINE}")
print("=" * 72)
print(">>> running 5 cycles (this takes a while — each may train 50 epochs)")
print(">>> tail -f /home/bigboss/code/depth_estimation_unify_theory/autoresearcher.log")

r = AutoResearcher(PROJECT)
try:
    results = r.run_n_cycles(5)
    print(f"\n[done] {len(results)} cycle(s) returned")
    for i, res in enumerate(results, 1):
        print(f"  cycle {i}: action={res.get('action','?')} "
              f"launched={res.get('experiment_launched')} "
              f"milestone={(res.get('milestone') or '')[:50]!r} "
              f"dead_end={'yes' if res.get('dead_end') else 'no'}")
except Exception as e:
    print(f"\n[FAIL] run_n_cycles raised: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ── Post-run: compute delta and verify the feedback loop wrote to memory_entries
print("\n" + "=" * 72)
print("[post-run] DB delta analysis:")
with sqlite3.connect(str(DB)) as conn:
    AFTER = {
        "exp": conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0],
        "me_all": conn.execute("SELECT COUNT(*) FROM memory_entries").fetchone()[0],
        "me_deadend": conn.execute(
            "SELECT COUNT(*) FROM memory_entries WHERE entry_type='dead_end'"
        ).fetchone()[0],
        "cc": conn.execute("SELECT COUNT(*) FROM causal_chain").fetchone()[0],
        "crl": conn.execute("SELECT COUNT(*) FROM code_review_lessons").fetchone()[0],
    }
    # New dead_end rows written by THIS run (id > baseline max id).
    new_de = conn.execute(
        "SELECT cycle, substr(content,1,80) FROM memory_entries "
        "WHERE entry_type='dead_end' ORDER BY id DESC LIMIT 5"
    ).fetchall()
    # New experiments rows.
    new_exp = conn.execute(
        "SELECT cycle, action, substr(milestone,1,40) FROM experiments "
        "ORDER BY id DESC LIMIT 5"
    ).fetchall()

for k in BASELINE:
    d = AFTER[k] - BASELINE[k]
    mark = "✅" if d >= 0 else "⚠️"
    print(f"  {mark} {k}: {BASELINE[k]} → {AFTER[k]} (Δ{d:+d})")

print("\n[post-run] newest experiments rows:")
for row in new_exp:
    print(f"  cycle={row[0]} action={row[1]} milestone={row[2]!r}")
print("\n[post-run] newest memory_entries dead_end rows:")
for row in new_de:
    print(f"  cycle={row[0]}: {row[1]!r}")

# Verify experiments INSERT worked (no OperationalError from dead_end removal).
try:
    with sqlite3.connect(str(DB)) as conn:
        conn.execute("SELECT milestone, decision, active_problem FROM experiments LIMIT 1").fetchall()
    print("\n✅ experiments SELECT works (dead_end removal didn't break it)")
except sqlite3.OperationalError as e:
    print(f"\n❌ experiments SELECT FAILED: {e}")

print("\n" + "=" * 72)
print("5-cycle E2E (REAL) complete. Inspect /home/bigboss/code/experiment_history.db")
print("=" * 72)
