r"""Deterministic fact scanner — records experiment facts from disk.

PHASE 1 of Reform v21. This module is the "fact spine": it reads experiment
artifacts that ALREADY EXIST on disk (experiment_manifest.json + train.log) and
records structured facts into SQLite. It does NOT depend on:

  - Any LLM call (unlike REFLECT)
  - The agent process being alive (unlike monitor callbacks)
  - Any regex re-derivation from free text (the D1 anti-pattern)

It only depends on FILES BEING PRESENT. Files survive reboots; agent processes
don't. This is why V30's 50-epoch results vanished from memory (the agent was
killed by a reboot before REFLECT could run) — but the train.log and manifest
survived on disk. This scanner reads those survivors.

Called at cycle start (loop.py) and on demand. Idempotent: output_dir is the
primary key, repeated scans are INSERT-OR-IGNORE.

Design rule (Reform v21, Ground 2): this module records FACTS only
(what happened: metrics, loss trend, best epoch). It does NOT record
INTERPRETATIONS (why it happened, is the method dead). Interpretations stay
with the LLM in REFLECT.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from core.training_log_parser import (
    classify_loss_trend,
    extract_metrics,
    has_nan_loss,
    load_training_log,
    parse_loss_series,
)

logger = logging.getLogger("autoresearcher.fact_scanner")


def _ensure_table(conn: sqlite3.Connection) -> None:
    """Create experiment_facts table if not exists. Idempotent."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS experiment_facts (
            output_dir TEXT PRIMARY KEY,
            manifest_timestamp TEXT NOT NULL DEFAULT '',
            command TEXT NOT NULL DEFAULT '',
            pid INTEGER,
            gpu TEXT NOT NULL DEFAULT '',
            log_file TEXT NOT NULL DEFAULT '',
            log_exists INTEGER NOT NULL DEFAULT 0,
            log_lines INTEGER NOT NULL DEFAULT 0,
            loss_first REAL,
            loss_last REAL,
            loss_count INTEGER NOT NULL DEFAULT 0,
            loss_trend TEXT NOT NULL DEFAULT '',
            has_nan_loss INTEGER NOT NULL DEFAULT 0,
            metrics_json TEXT NOT NULL DEFAULT '{}',
            best_metric_name TEXT NOT NULL DEFAULT '',
            best_metric_value REAL,
            best_epoch INTEGER,
            scanned_at REAL NOT NULL DEFAULT 0,
            scan_source TEXT NOT NULL DEFAULT 'disk_scan'
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_facts_best ON experiment_facts(best_metric_value)"
    )
    conn.commit()


def _pick_best_metric(metrics: dict[str, float]) -> tuple[str, float | None]:
    """Pick the single best scalar metric from extracted metrics.

    Priority order (lower-is-better MAE convention):
      val_mae > val_mae_overall > best_val_mae > mae_overall >
      mae > first val_* > first mae* > first metric

    Returns (name, value). (\"\", None) if no metrics found.
    Mirrors loop.py's _extract_current_metric fallback chain but centralized.
    """
    if not metrics:
        return "", None
    priority = [
        "val_mae",
        "val_mae_overall",
        "best_val_mae",
        "mae_overall",
        "mae",
    ]
    for name in priority:
        if name in metrics:
            return name, metrics[name]
    # Fallback: first val_* then first mae* then first anything
    for name in metrics:
        if name.startswith("val_"):
            return name, metrics[name]
    for name in metrics:
        if name.startswith("mae"):
            return name, metrics[name]
    # Last resort
    name = next(iter(metrics))
    return name, metrics[name]


def _trend_label(trend) -> str:
    """Convert LossTrend dataclass to a short label string."""
    if trend.is_nan:
        return "nan"
    if trend.is_diverging:
        return "diverging"
    if trend.is_increasing:
        return "increasing"
    if trend.is_plateaued:
        return "plateaued"
    if trend.is_decreasing:
        return "decreasing"
    if trend.has_data:
        return "insufficient"
    return "empty"


def _find_best_epoch(log_text: str, best_metric_name: str) -> int | None:
    """Try to find the epoch number where best metric was achieved.

    Looks for 'best' markers or scans for the metric value's first occurrence.
    Returns None if can't determine.
    """
    if not log_text or not best_metric_name:
        return None
    # Look for explicit "best" lines: "best at epoch 38" / "best_val_mae=... epoch=38"
    best_epoch_re_patterns = [
        r"best[^a-z]*epoch[^0-9]*(\d+)",
        r"epoch[^0-9]*(\d+)[^a-z]*best",
        r"best[^0-9]*(\d+)",
    ]
    import re

    for pat in best_epoch_re_patterns:
        m = re.search(pat, log_text, re.IGNORECASE)
        if m:
            try:
                ep = int(m.group(1))
                if 0 < ep < 100000:  # sanity bound
                    return ep
            except ValueError:
                continue
    return None


def scan_single(
    manifest_path: Path, conn: sqlite3.Connection
) -> dict[str, Any] | None:
    """Scan one experiment directory, write facts to DB. Returns the record or None.

    Idempotent: if output_dir already in table, INSERT OR IGNORE skips it.
    """
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        logger.debug(f"Skipping unreadable manifest {manifest_path}: {e}")
        return None

    output_dir = str(manifest_path.parent)
    command = manifest.get("command", "")
    manifest_ts = manifest.get("timestamp", "")
    pid = manifest.get("pid")
    gpu = manifest.get("gpu", "")
    log_file = manifest.get("log_file", "")

    # Locate train.log — prefer manifest's log_file, fall back to standard names
    workspace = manifest_path.parent.parent  # outputs/<dir> -> outputs/
    log_path = None
    if log_file:
        candidate = Path(log_file)
        if candidate.exists():
            log_path = candidate
    if log_path is None:
        for name in ("train.log", "training.log", "output.log"):
            candidate = manifest_path.parent / name
            if candidate.exists():
                log_path = candidate
                break

    log_exists = 1 if (log_path and log_path.exists()) else 0

    # Idempotency with staleness detection (Phase 1 fix, obstacle 1):
    # If this output_dir is already recorded AND the train.log hasn't changed
    # since last scan, skip (pure idempotent). But if train.log is NEWER than
    # the recorded scan (e.g. code agent reused the directory and re-ran
    # training), re-parse and REPLACE the stale record.
    log_mtime = log_path.stat().st_mtime if (log_path and log_path.exists()) else 0
    existing_scanned_at = 0
    try:
        row = conn.execute(
            "SELECT scanned_at FROM experiment_facts WHERE output_dir = ?",
            (output_dir,),
        ).fetchone()
        if row:
            existing_scanned_at = row[0] or 0
    except sqlite3.Error:
        pass

    if existing_scanned_at > 0 and log_mtime <= existing_scanned_at:
        # Record exists and log hasn't changed — true idempotent skip
        return None

    log_text = load_training_log(log_path) if log_path else ""
    log_lines = log_text.count("\n") if log_text else 0

    # Extract facts from log
    losses = parse_loss_series(log_text)
    trend = classify_loss_trend(losses)
    nan_loss = has_nan_loss(log_text)
    metrics = extract_metrics(log_text)
    best_name, best_val = _pick_best_metric(metrics)
    best_epoch = _find_best_epoch(log_text, best_name)

    record = {
        "output_dir": output_dir,
        "manifest_timestamp": manifest_ts,
        "command": command,
        "pid": pid,
        "gpu": gpu,
        "log_file": str(log_path) if log_path else "",
        "log_exists": log_exists,
        "log_lines": log_lines,
        "loss_first": trend.first if trend.has_data else None,
        "loss_last": trend.last if trend.has_data else None,
        "loss_count": trend.count,
        "loss_trend": _trend_label(trend),
        "has_nan_loss": 1 if nan_loss else 0,
        "metrics_json": json.dumps(metrics, ensure_ascii=False),
        "best_metric_name": best_name,
        "best_metric_value": best_val,
        "best_epoch": best_epoch,
        "scanned_at": time.time(),
        "scan_source": "disk_scan",
    }

    try:
        conn.execute(
            """
            INSERT OR REPLACE INTO experiment_facts (
                output_dir, manifest_timestamp, command, pid, gpu,
                log_file, log_exists, log_lines, loss_first, loss_last,
                loss_count, loss_trend, has_nan_loss, metrics_json,
                best_metric_name, best_metric_value, best_epoch,
                scanned_at, scan_source
            ) VALUES (
                :output_dir, :manifest_timestamp, :command, :pid, :gpu,
                :log_file, :log_exists, :log_lines, :loss_first, :loss_last,
                :loss_count, :loss_trend, :has_nan_loss, :metrics_json,
                :best_metric_name, :best_metric_value, :best_epoch,
                :scanned_at, :scan_source
            )
            """,
            record,
        )
        conn.commit()
    except sqlite3.Error as e:
        logger.warning(f"Failed to insert fact for {output_dir}: {e}")
        return None

    return record


def scan_all(
    workspace: Path, db_path: Path, rescan: bool = False
) -> dict[str, int]:
    """Scan all experiment directories under workspace/outputs.

    Args:
        workspace: project root (contains outputs/ dir)
        db_path: path to experiment_history.db
        rescan: if True, drop and re-insert existing rows (for re-parsing
                logs that grew after first scan). Default False = idempotent.

    Returns:
        {"scanned": N, "inserted": M, "skipped": K, "errors": E}
    """
    with sqlite3.connect(str(db_path)) as conn:
        _ensure_table(conn)

        if rescan:
            conn.execute("DELETE FROM experiment_facts")
            conn.commit()

        outputs_dir = workspace / "outputs"
        if not outputs_dir.exists():
            return {"scanned": 0, "inserted": 0, "skipped": 0, "errors": 0}

        manifests = sorted(outputs_dir.glob("*/experiment_manifest.json"))
        scanned = inserted = skipped = errors = 0

        for manifest_path in manifests:
            scanned += 1
            # scan_single handles its own idempotency: it checks mtime and
            # returns None if the record exists AND log hasn't changed.
            # No pre-check here — let scan_single decide.
            result = scan_single(manifest_path, conn)
            if result is not None:
                inserted += 1
            else:
                skipped += 1  # None = idempotent skip (log unchanged)

        logger.info(
            f"fact_scanner: scanned={scanned} inserted={inserted} "
            f"skipped={skipped} errors={errors}"
        )
        return {
            "scanned": scanned,
            "inserted": inserted,
            "skipped": skipped,
            "errors": errors,
        }


def get_facts_for_output_dir(
    db_path: Path, output_dir: str
) -> dict[str, Any] | None:
    """Retrieve the fact record for a given output_dir. Returns dict or None."""
    with sqlite3.connect(str(db_path)) as conn:
        _ensure_table(conn)
        row = conn.execute(
            "SELECT * FROM experiment_facts WHERE output_dir = ?", (output_dir,)
        ).fetchone()
        if not row:
            return None
        cols = [d[0] for d in conn.execute(
            "SELECT * FROM experiment_facts LIMIT 0"
        ).description]
        return dict(zip(cols, row))


def get_all_facts(db_path: Path) -> list[dict[str, Any]]:
    """Retrieve all fact records, newest scan first."""
    with sqlite3.connect(str(db_path)) as conn:
        _ensure_table(conn)
        rows = conn.execute(
            "SELECT * FROM experiment_facts ORDER BY scanned_at DESC"
        ).fetchall()
        cols = [d[0] for d in conn.execute(
            "SELECT * FROM experiment_facts LIMIT 0"
        ).description]
        return [dict(zip(cols, r)) for r in rows]
