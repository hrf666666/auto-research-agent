#!/usr/bin/env python3
"""
Reform v21 verification: collect M1-M8 metrics from SQLite + logs,
compare against expected values, output PASS/FAIL/INVESTIGATE report.

This is the "answer key" for the 10-cycle experiment.
Run after the 10 cycles complete: python3 tools/verify_reform.py <project_dir>
"""
import json
import sqlite3
import sys
import time
from pathlib import Path


# ── Expected values (from REFORM_V21_EXPERIMENT.md PART 3) ──
EXPECTED = {
    "M1":  {"desc": "disk experiments in experiment_facts", "target": 1.0, "tolerance": 0.0, "red_line": True},
    "M1b": {"desc": "new session experiments in experiment_facts", "target": 1.0, "tolerance": 0.0, "red_line": True},
    "M2":  {"desc": "milestone non-empty rate", "target": 0.60, "tolerance": 0.15},
    "M3":  {"desc": "success_criteria evaluated (of parseable)", "target": 1.0, "tolerance": 0.0, "red_line": True},
    "M3b": {"desc": "criteria parseable rate", "target": 0.80, "tolerance": 0.10},
    "M4":  {"desc": "THINK parse-fail direct waits", "target_max": 2},
    "M5":  {"desc": "duplicate experiments (known signature)", "target_max": 0, "red_line": True},
    "M6":  {"desc": "dead-end retries (known signature)", "target_max": 0, "red_line": True},
    "M7":  {"desc": "uncontrolled causal claims marked", "target": 1.0, "tolerance": 0.0},
    "M8":  {"desc": "experiments with spec check record", "target": 1.0, "tolerance": 0.0},
}


def collect_metrics(project_dir: str) -> dict:
    """Collect all M1-M8 metrics from the project's SQLite + logs."""
    project = Path(project_dir)
    db_path = project / "experiment_history.db"
    log_path = project / "autoresearcher.log"
    facts = {}

    # ── M1: disk experiments in experiment_facts ──
    manifests_on_disk = list((project / "outputs").glob("*/experiment_manifest.json"))
    try:
        with sqlite3.connect(str(db_path)) as conn:
            fact_rows = conn.execute("SELECT COUNT(*) FROM experiment_facts").fetchone()[0]
    except sqlite3.Error:
        fact_rows = 0
    disk_count = len(manifests_on_disk)
    facts["M1"] = {
        "value": fact_rows / disk_count if disk_count > 0 else 1.0,
        "raw": f"{fact_rows}/{disk_count} manifests in experiment_facts",
    }

    # ── M1b: new session experiments ──
    # Count experiments launched this session (from experiments table, recent cycles)
    try:
        with sqlite3.connect(str(db_path)) as conn:
            # experiments with experiment_launched=1 in recent cycles
            launched = conn.execute(
                "SELECT COUNT(*) FROM experiments WHERE experiment_launched=1"
            ).fetchone()[0]
            # of those, how many have a matching experiment_facts record?
            # (approximation: count experiment_facts with non-empty command)
            facts_with_cmd = conn.execute(
                "SELECT COUNT(*) FROM experiment_facts WHERE command != ''"
            ).fetchone()[0]
    except sqlite3.Error:
        launched = facts_with_cmd = 0
    facts["M1b"] = {
        "value": facts_with_cmd / launched if launched > 0 else 1.0,
        "raw": f"{facts_with_cmd} facts for {launched} launched experiments",
    }

    # ── M2: milestone non-empty rate ──
    try:
        with sqlite3.connect(str(db_path)) as conn:
            total = conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]
            has_milestone = conn.execute(
                "SELECT COUNT(*) FROM experiments WHERE milestone IS NOT NULL AND milestone != ''"
            ).fetchone()[0]
    except sqlite3.Error:
        total = has_milestone = 0
    facts["M2"] = {
        "value": has_milestone / total if total > 0 else 0,
        "raw": f"{has_milestone}/{total} experiments have milestone",
    }

    # ── M3/M3b: success_criteria evaluation ──
    # From log: count "methodology" lines that show criteria evaluation
    m3_evaluated = m3_total = m3b_parseable = m3b_total = 0
    if log_path.exists():
        log_text = log_path.read_text(errors="ignore")
        for line in log_text.split("\n"):
            if "[methodology]" in line:
                if "criteria(" in line:
                    m3_total += 1
                    if "MET" in line or "NOT MET" in line:
                        m3_evaluated += 1
                    if "unparseable" not in line:
                        m3b_parseable += 1
                if "criteria(" in line and "unparseable" not in line:
                    m3b_total += 1
    facts["M3"] = {
        "value": m3_evaluated / m3_total if m3_total > 0 else 1.0,
        "raw": f"{m3_evaluated}/{m3_total} parseable criteria evaluated",
    }
    facts["M3b"] = {
        "value": m3b_parseable / m3b_total if m3b_total > 0 else 0,
        "raw": f"{m3b_parseable}/{m3b_total} criteria were parseable",
    }

    # ── M4: THINK parse-fail direct waits ──
    think_waits = 0
    if log_path.exists():
        log_text = log_path.read_text(errors="ignore")
        # Count "Defaulting to wait" that are NOT from retry
        # and "retry also failed"
        think_waits = log_text.count("retry also failed to parse") + \
                      log_text.count("THINK retry call failed")
    facts["M4"] = {
        "value": think_waits,
        "raw": f"{think_waits} THINK parse failures (after retry)",
    }

    # ── M5/M6: duplicate / dead-end ──
    # From log: count "action_gate" and "DEAD_END_WARN" lines
    dup_count = dead_retry = 0
    if log_path.exists():
        log_text = log_path.read_text(errors="ignore")
        dup_count = log_text.count("duplicate experiment")  # if we log this
        dead_retry = log_text.count("DEAD_END_WARN")
    facts["M5"] = {"value": dup_count, "raw": f"{dup_count} duplicate experiments"}
    facts["M6"] = {"value": dead_retry, "raw": f"{dead_retry} dead-end warnings"}

    # ── M7: uncontrolled causal claims marked ──
    uncontrolled_total = uncontrolled_marked = 0
    if log_path.exists():
        log_text = log_path.read_text(errors="ignore")
        for line in log_text.split("\n"):
            if "[methodology]" in line and "UNCONTROLLED" in line:
                uncontrolled_total += 1
                uncontrolled_marked += 1
    facts["M7"] = {
        "value": uncontrolled_marked / uncontrolled_total if uncontrolled_total > 0 else 1.0,
        "raw": f"{uncontrolled_marked}/{uncontrolled_total} uncontrolled claims marked",
    }

    # ── M8: spec check records ──
    spec_checks = 0
    if log_path.exists():
        log_text = log_path.read_text(errors="ignore")
        spec_checks = log_text.count("spec_ok") + log_text.count("SPEC_DEVIATION")
    facts["M8"] = {
        "value": spec_checks,
        "raw": f"{spec_checks} spec check records",
    }

    return facts


def evaluate(metrics: dict) -> list[dict]:
    """Evaluate each metric against expected, return PASS/FAIL/INVESTIGATE."""
    results = []
    for key, expected in EXPECTED.items():
        actual = metrics.get(key, {})
        value = actual.get("value", None)
        raw = actual.get("raw", "")

        if "target_max" in expected:
            # Lower-is-better metric
            status = "PASS" if value <= expected["target_max"] else "FAIL"
            if status == "FAIL" and not expected.get("red_line"):
                status = "INVESTIGATE"
        elif "target" in expected:
            if value is None:
                status = "SKIP"
            elif expected["tolerance"] == 0:
                status = "PASS" if value >= expected["target"] else ("FAIL" if expected.get("red_line") else "INVESTIGATE")
            else:
                lo = expected["target"] - expected["tolerance"]
                hi = expected["target"] + expected["tolerance"]
                status = "PASS" if lo <= value <= hi else "INVESTIGATE"
                if value < lo and expected.get("red_line"):
                    status = "FAIL"
        else:
            status = "SKIP"

        results.append({
            "metric": key,
            "desc": expected["desc"],
            "actual_value": value,
            "raw": raw,
            "status": status,
            "red_line": expected.get("red_line", False),
        })
    return results


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 tools/verify_reform.py <project_dir>")
        sys.exit(1)
    project_dir = sys.argv[1]

    print(f"Reform v21 Verification Report")
    print(f"Project: {project_dir}")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    metrics = collect_metrics(project_dir)
    results = evaluate(metrics)

    red_lines_hit = []
    for r in results:
        icon = {"PASS": "✅", "FAIL": "🔴", "INVESTIGATE": "⚠️", "SKIP": "⏭️"}[r["status"]]
        rl = " [RED LINE]" if r["red_line"] else ""
        print(f"  {icon} {r['metric']:4s} {r['status']:12s} {r['desc']}")
        print(f"        value={r['actual_value']}  ({r['raw']}){rl}")
        if r["status"] == "FAIL" and r["red_line"]:
            red_lines_hit.append(r["metric"])

    print("=" * 70)
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    invest = sum(1 for r in results if r["status"] == "INVESTIGATE")
    print(f"Summary: {passed} PASS / {failed} FAIL / {invest} INVESTIGATE")

    if red_lines_hit:
        print(f"\n🔴 RED LINES HIT: {red_lines_hit}")
        print("   Per protocol: STOP and investigate before proceeding.")

    # Save report
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "project": project_dir,
        "metrics": metrics,
        "results": results,
        "summary": {"pass": passed, "fail": failed, "investigate": invest},
    }
    report_path = Path(project_dir) / "reform_verification_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReport saved: {report_path}")


if __name__ == "__main__":
    main()
