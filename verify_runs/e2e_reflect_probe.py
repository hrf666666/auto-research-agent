"""End-to-end REFLECT probe (Reform v21 step-0+1 verification).

This bypasses the 10h training by constructing a REAL execute_result (pointing
at a real training log that actually exists on disk) and calling loop._reflect()
directly. Everything else is REAL:
  - real AgentDispatcher (real GLM API call for the reflection)
  - the PATCHED parser (task-aware schema)
  - real memory layer (real SQLite writes)
  - real fact_scanner lookup

Only the execute_result is constructed (because we killed the training at epoch 2).
The log_file points at a REAL log that exists, so fact_spine / log parsing see
real bytes.

WHAT WE'RE CHECKING (the whole point of step 0+1):
  1. GLM produces a VALID REFLECT JSON (proves "LLM writes prose" was wrong)
  2. the PATCHED parser ACCEPTS it (proves the action-key bug is fixed)
  3. dead_end / causal_link / lesson are PERSISTED to memory (proves the fields
     that were always dropped now survive)
  4. fact_spine fallback does NOT fire (REFLECT succeeded)
"""
from __future__ import annotations
import os, sys, json, time

# Must run from agent dir so `core` imports + config resolve.
AGENT = "/home/bigboss/code/auto_research_agent"
PROJ = "/home/bigboss/code/depth_estimation_unify_theory"
sys.path.insert(0, AGENT)
os.chdir(AGENT)

import yaml
from core.loop import ResearchLoop

# --- load real config (same path the real loop uses) ---
with open(os.path.join(AGENT, "config.yaml")) as f:
    config = yaml.safe_load(f)
config.setdefault("agent", {})["max_cycles"] = 1

# --- snapshot causal_chain / dead_end BEFORE ---
import sqlite3
DB = os.path.join(PROJ, "experiment_history.db")
con = sqlite3.connect(DB); cur = con.cursor()
def cnt(sql):
    cur.execute(sql); return cur.fetchone()[0]
before = {
    "causal_chain": cnt("SELECT COUNT(*) FROM causal_chain"),
    "code_review_lessons": cnt("SELECT COUNT(*) FROM code_review_lessons"),
    "dead_end_nonempty": cnt("SELECT COUNT(*) FROM memory_entries WHERE entry_type = 'dead_end'"),
}
print("=" * 72)
print("PRE-REFLECT SNAPSHOT:")
for k, v in before.items():
    print("  " + k + " = " + str(v))
print("=" * 72)
con.close()

# --- build a REAL loop via real __init__ (correctly wires dispatcher/memory/etc) ---
# We use the real constructor so all internal objects are initialized exactly
# as in production. We just never call .run(); we call ._reflect() directly.
loop = ResearchLoop(config=config, project_dir=PROJ)
loop.cycle_count = 11
# _reflect needs these to be set (they are by __init__, but be explicit):
assert loop.memory is not None, "memory not initialized"
assert loop.dispatcher is not None, "dispatcher not initialized"

# Real execute_result: log_file points at a REAL log on disk.
REAL_LOG = os.path.join(PROJ, "logs/v30_80ep_train.log")
execute_result = {
    "log_file": REAL_LOG,
    "experiment_launched": True,
    "final_metrics": {
        "val_mae": 0.230560,
        "mae_lambertian": 0.307155,
        "mae_non_lambertian": 0.435303,
        "mae_urban": 0.208417,
        "epoch": 2,
    },
    "training_metrics": {"val_mae": 0.230560},
}

print("\nCalling loop._reflect() with REAL GLM (task=reflect)...")
print("  log_file = " + REAL_LOG)
print("  (waiting on GLM API for the reflection...)\n")
t0 = time.time()

try:
    result = loop._reflect(execute_result, verify_report=None)
except Exception as e:
    print("!!! _reflect raised: " + repr(e))
    import traceback; traceback.print_exc()
    result = {}

elapsed = time.time() - t0
print("=" * 72)
print("POST-REFLECT (elapsed %.1fs)" % elapsed)
print("=" * 72)

# --- what did the parser/LLM actually produce? ---
print("\n[1] REFLECT result dict (what the parser accepted):")
for k in ("milestone", "decision", "dead_end", "active_problem", "causal_link", "lesson"):
    v = result.get(k)
    shown = repr(v)[:120] if v else repr(v)
    print("  " + k + " = " + shown)
src = result.get("_milestone_source", "<none — came from LLM, NOT fallback>")
print("  _milestone_source = " + str(src))

# --- snapshot AFTER ---
con = sqlite3.connect(DB); cur = con.cursor()
after = {
    "causal_chain": cnt("SELECT COUNT(*) FROM causal_chain"),
    "code_review_lessons": cnt("SELECT COUNT(*) FROM code_review_lessons"),
    "dead_end_nonempty": cnt("SELECT COUNT(*) FROM memory_entries WHERE entry_type = 'dead_end'"),
}
print("\n[2] DB DELTA (did fields actually persist?):")
all_good = True
for k in before:
    d = after[k] - before[k]
    mark = "✅ +1" if d > 0 else ("— (no change)" if d == 0 else "⚠️ " + str(d))
    if k in ("causal_chain",) and d == 0:
        all_good = False
    print("  " + k + ": " + str(before[k]) + " → " + str(after[k]) + "  " + mark)

# --- verdict ---
print("\n" + "=" * 72)
print("VERDICT:")
fell_back = (src == "fact_spine_fallback")
print("  REFLECT succeeded (LLM JSON accepted by patched parser): " + ("❌ NO (fell back)" if fell_back else "✅ YES"))
print("  causal_link persisted (was ALWAYS 0 before fix):          " + ("✅ YES" if after["causal_chain"] > before["causal_chain"] else "❌ NO"))
print("=" * 72)
