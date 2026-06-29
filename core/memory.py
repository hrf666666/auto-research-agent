"""
AutoResearcher Two-Tier Memory System

Maintains a constant-size memory regardless of how long the agent runs:
- Tier 1 (PROJECT_BRIEF.md): Frozen reference, never modified by the agent
- Tier 2 (MEMORY_LOG.md): Rolling log with auto-compaction (for LLM context)
- Tier 3 (experiment_history.db): SQLite database with full experiment history

Total LLM context budget: ~5000 chars (~1500 tokens) — always.
Database has no size limit — all history is preserved.
"""

import time
import json
import sqlite3
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("autoresearcher.memory")

# Numerical safety epsilon
_EPS = 1e-8


class MemoryManager:
    """Two-tier memory with automatic compaction + SQLite history.

    The key insight: long-running agents accumulate context that grows
    without bound, leading to degraded performance and ballooning costs.
    This system caps memory at a fixed budget by:
    - Keeping milestones (key results) in a priority queue, oldest dropped first
    - Keeping only the N most recent decisions
    - Never modifying the frozen project brief

    The SQLite database stores complete experiment history that survives
    MEMORY_LOG.md compaction. This allows:
    - Looking up past experiment results by cycle number
    - Tracking metric trends across all experiments
    - Never losing dead_end information even after compaction
    """

    def __init__(
        self,
        project_dir: Path,
        brief_max: int = 3000,
        log_max: int = 4000,
        milestone_max: int = 1200,
        max_recent: int = 15,
        workspace: Path = None,
        method_keywords: dict = None,
        domain_keys: list = None,
    ):
        self.project_dir = Path(project_dir)
        self.brief_path = self.project_dir / "PROJECT_BRIEF.md"
        # Use workspace path for log if provided, otherwise fall back to
        # project_dir/MEMORY_LOG.md (not hardcoded workspace/ subdirectory)
        if workspace:
            self.log_path = Path(workspace) / "MEMORY_LOG.md"
            self.db_path = Path(workspace) / "experiment_history.db"
        else:
            self.log_path = self.project_dir / "MEMORY_LOG.md"
            self.db_path = self.project_dir / "experiment_history.db"
        self.brief_max = brief_max
        self.log_max = log_max
        self.milestone_max = milestone_max
        self.max_recent = max_recent

        # Configurable domain keywords (override for different projects)
        self.method_keywords = method_keywords or self._default_method_keywords()
        self.domain_keys = domain_keys or self._infer_domain_keys()

        # Ensure log file exists
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.log_path.exists():
            self._init_log()

        # Initialize SQLite database
        self._init_db()

    @staticmethod
    def _default_method_keywords() -> dict:
        """Generic method keyword patterns — reusable across domains."""
        return {
            "epi": ["epi", "epinet", "epipolar"],
            "fft": ["fft", "fourier", "frequency", "spectral"],
            "attention": ["attention", "self-attention", "cross-attention"],
            "conv3d": ["conv3d", "3d conv", "volumetric"],
            "resnet": ["resnet", "backbone", "pretrained"],
            "contrastive": ["contrastive", "simclr", "infonce"],
            "dropout": ["dropout", "regulariz"],
        }

    def _infer_domain_keys(self) -> list:
        """Dynamically infer domain metric keys from DATASET_MANIFEST.json."""
        manifest_path = self.project_dir / "DATASET_MANIFEST.json"
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text())
                domains = set()
                for ds_info in manifest.get("datasets", {}).values():
                    ds_type = ds_info.get("type", "")
                    if ds_type:
                        domains.add(ds_type)
                if domains:
                    # Build metric keys like MAE_{Domain}
                    return [f"MAE_{d.replace('-', '_').replace(' ', '_')}" for d in sorted(domains)]
            except Exception:
                pass
        # Generic fallback — no project-specific names
        return []

    # ─────────────────────────────────────────────────
    # SQLite Database (Tier 3 — full history, no limit)
    # ─────────────────────────────────────────────────

    def _init_db(self):
        """Initialize SQLite experiment history database."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS experiments (
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

CREATE INDEX IF NOT EXISTS idx_cycle ON experiments(cycle);
                CREATE INDEX IF NOT EXISTS idx_launched ON experiments(experiment_launched);

                CREATE TABLE IF NOT EXISTS memory_entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    entry_type TEXT NOT NULL,  -- milestone, decision, dead_end, active_problem, major_event
                    content TEXT NOT NULL,
                    cycle INTEGER,
                    in_llm_context INTEGER NOT NULL DEFAULT 0
                );

                CREATE INDEX IF NOT EXISTS idx_entry_type ON memory_entries(entry_type);

                CREATE TABLE IF NOT EXISTS pareto_matrix (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cycle INTEGER NOT NULL,
                    method TEXT NOT NULL,
                    domain TEXT NOT NULL,
                    mae REAL,
                    experiment_type TEXT NOT NULL DEFAULT 'full',
                    timestamp REAL NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_pareto_method ON pareto_matrix(method);
                CREATE INDEX IF NOT EXISTS idx_pareto_domain ON pareto_matrix(domain);

                CREATE TABLE IF NOT EXISTS causal_chain (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cycle INTEGER NOT NULL,
                    design_decision TEXT NOT NULL,
                    architectural_property TEXT NOT NULL,
                    metric_affected TEXT NOT NULL,
                    expected_effect TEXT NOT NULL DEFAULT '',
                    actual_effect TEXT NOT NULL DEFAULT '',
                    verified INTEGER NOT NULL DEFAULT 0,
                    timestamp REAL NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_causal_cycle ON causal_chain(cycle);

                CREATE TABLE IF NOT EXISTS experiment_value (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cycle INTEGER NOT NULL,
                    hypothesis TEXT NOT NULL,
                    expected_improvement REAL,
                    prior_probability REAL,
                    information_value REAL,
                    actual_improvement REAL,
                    was_correct INTEGER,
                    timestamp REAL NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_expval_cycle ON experiment_value(cycle);

                CREATE TABLE IF NOT EXISTS code_review_lessons (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    cycle INTEGER NOT NULL,
                    severity TEXT NOT NULL DEFAULT 'MEDIUM',
                    category TEXT NOT NULL DEFAULT '',
                    pattern TEXT NOT NULL,
                    description TEXT NOT NULL,
                    file_pattern TEXT NOT NULL DEFAULT '',
                    code_snippet TEXT NOT NULL DEFAULT '',
                    fix_suggestion TEXT NOT NULL DEFAULT '',
                    evidence TEXT NOT NULL DEFAULT '',
                    hit_count INTEGER NOT NULL DEFAULT 0,
                    last_hit_cycle INTEGER,
                    source TEXT NOT NULL DEFAULT 'auto'
                );

                CREATE INDEX IF NOT EXISTS idx_lesson_pattern ON code_review_lessons(pattern);
                CREATE INDEX IF NOT EXISTS idx_lesson_severity ON code_review_lessons(severity);
                CREATE INDEX IF NOT EXISTS idx_lesson_category ON code_review_lessons(category);
            """)

    def record_cycle_outcome(self, cycle: int, think_result: dict,
                              execute_result: dict, reflect_result: dict,
                              verify_report: dict = None, duration: float = None):
        """Record complete cycle outcome to SQLite database.

        This captures ALL information that might be useful later, even if
        MEMORY_LOG.md compaction drops it from LLM context.
        """
        verify_dict = verify_report or {}
        metrics = execute_result.get("final_metrics", {})
        if isinstance(metrics, str):
            try:
                metrics = json.loads(metrics)
            except (json.JSONDecodeError, TypeError):
                metrics = {"raw": metrics}

        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                INSERT INTO experiments (
                    cycle, timestamp, action, hypothesis, success_criteria,
                    agent_type, task_summary, experiment_launched, pid, log_file,
                    verify_pass, verify_fail, verify_warnings, verify_diagnosis,
                    metrics_json, milestone, decision,
                    active_problem, module_failure, duration_seconds, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                cycle,
                time.time(),
                think_result.get("action", ""),
                think_result.get("hypothesis", "")[:500],
                think_result.get("success_criteria", "")[:500],
                think_result.get("agent", ""),
                think_result.get("task", "")[:1000],
                1 if execute_result.get("experiment_launched") else 0,
                execute_result.get("pid"),
                execute_result.get("log_file", ""),
                verify_dict.get("passed", 0),
                verify_dict.get("failed", 0),
                verify_dict.get("warnings", 0),
                "; ".join(str(d) for d in (verify_dict.get("diagnosis") or []))[:1000],
                json.dumps(metrics, ensure_ascii=False),
                (reflect_result.get("milestone") or "")[:500],
                (reflect_result.get("decision") or "")[:500],
                (reflect_result.get("active_problem") or "")[:500],
                (reflect_result.get("module_failure") or "")[:500],
                duration,
                (execute_result.get("response") or "")[:500],
            ))

    def _record_memory_entry(self, entry_type: str, content: str, cycle: int = None):
        """Record a memory entry to SQLite."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                INSERT INTO memory_entries (timestamp, entry_type, content, cycle, in_llm_context)
                VALUES (?, ?, ?, ?, 1)
            """, (time.time(), entry_type, content, cycle))

    def get_experiment_history(self, limit: int = 20) -> list[dict]:
        """Get recent experiment history from SQLite."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT cycle, timestamp, action, hypothesis, success_criteria,
                       experiment_launched, pid, verify_pass, verify_fail,
                       metrics_json, milestone, decision, active_problem
                FROM experiments
                ORDER BY cycle DESC
                LIMIT ?
            """, (limit,)).fetchall()
            return [dict(r) for r in rows]

    def get_dead_ends_full(self) -> list[str]:
        """Get ALL dead ends from SQLite (survives MEMORY_LOG compaction)."""
        with sqlite3.connect(str(self.db_path)) as conn:
            rows = conn.execute("""
                SELECT content FROM memory_entries
                WHERE entry_type = 'dead_end'
                ORDER BY timestamp ASC
            """).fetchall()
            return [r[0] for r in rows]

    def get_dead_ends_by_category(self, category: str = None) -> list[dict]:
        """Get dead ends grouped by failure_category (v12).

        Returns structured dead end entries with their categories, enabling
        the system to distinguish hypothesis failures from method inadequacies.

        Args:
            category: If provided, filter to this specific category.
                One of: 'hypothesis_wrong', 'implementation_bug',
                'insufficient_experiment', 'method_inadequacy'.
        """
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            if category:
                rows = conn.execute("""
                    SELECT content, failure_category, timestamp, cycle
                    FROM memory_entries
                    WHERE entry_type = 'dead_end' AND failure_category = ?
                    ORDER BY timestamp ASC
                """, (category,)).fetchall()
            else:
                rows = conn.execute("""
                    SELECT content, failure_category, timestamp, cycle
                    FROM memory_entries
                    WHERE entry_type = 'dead_end'
                    ORDER BY timestamp ASC
                """).fetchall()
            return [dict(r) for r in rows]

    def get_method_inadequacy_count(self) -> int:
        """Count dead ends categorized as 'method_inadequacy' (v12).

        These are dead ends where the analysis method was too narrow,
        not the hypothesis itself being wrong. High count suggests the
        direction should be retried with broader analysis.
        """
        with sqlite3.connect(str(self.db_path)) as conn:
            try:
                count = conn.execute("""
                    SELECT COUNT(*) FROM memory_entries
                    WHERE entry_type = 'dead_end' AND failure_category = 'method_inadequacy'
                """).fetchone()[0]
                return count
            except Exception:
                # Column may not exist yet (pre-v12 database)
                return 0

    def get_summary_stats(self) -> dict:
        """Get aggregate statistics for the research session."""
        with sqlite3.connect(str(self.db_path)) as conn:
            try:
                total = conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]
                launched = conn.execute("SELECT COUNT(*) FROM experiments WHERE experiment_launched = 1").fetchone()[0]
                dead_ends = conn.execute("SELECT COUNT(*) FROM memory_entries WHERE entry_type = 'dead_end'").fetchone()[0]

                # Extract best/worst metric from metrics_json across all experiments
                best_metric = None
                worst_metric = None
                rows = conn.execute(
                    "SELECT metrics_json FROM experiments WHERE experiment_launched = 1 AND metrics_json != '{}'"
                ).fetchall()
                for (mj,) in rows:
                    try:
                        m = json.loads(mj)
                        # Try metric keys from config, fallback to common defaults
                        val = None
                        metric_keys = getattr(self, '_metric_keys', None)
                        if not metric_keys:
                            metric_keys = self.domain_keys or ("val_MAE", "val_MAE_overall", "best_val_MAE", "val_mae", "MAE_overall")
                        for key in metric_keys:
                            if key in m:
                                val = float(m[key])
                                break
                        if val is not None:
                            if best_metric is None or val < best_metric:
                                best_metric = val
                            if worst_metric is None or val > worst_metric:
                                worst_metric = val
                    except (json.JSONDecodeError, TypeError, ValueError):
                        pass

                return {
                    "total_cycles": total,
                    "experiments_launched": launched,
                    "dead_ends_count": dead_ends,
                    "launch_rate": launched / max(total, 1),
                    "best_metric": best_metric,
                    "worst_metric": worst_metric,
                }
            except Exception:
                return {"total_cycles": 0}

    # ── Pareto Frontier Tracking ──

    def record_pareto_entry(self, cycle: int, method: str, domain: str,
                            mae: float, experiment_type: str = "full"):
        """Record a method×domain result for Pareto frontier analysis."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                INSERT INTO pareto_matrix (cycle, method, domain, mae, experiment_type, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (cycle, method, domain, mae, experiment_type, time.time()))

    def get_pareto_matrix(self) -> dict:
        """Get method×domain MAE matrix from all experiments.

        Returns: {method: {domain: best_mae}} — best MAE per method per domain.
        """
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT method, domain, MIN(mae) as best_mae, COUNT(*) as attempts
                FROM pareto_matrix
                WHERE mae IS NOT NULL
                GROUP BY method, domain
                ORDER BY method, domain
            """).fetchall()

            matrix = {}
            for row in rows:
                method = row["method"]
                if method not in matrix:
                    matrix[method] = {}
                matrix[method][row["domain"]] = {
                    "best_mae": round(row["best_mae"], 4),
                    "attempts": row["attempts"],
                }
            return matrix

    def get_pareto_frontier(self) -> dict:
        """Get the Pareto-optimal methods per domain.

        A method is Pareto-optimal if it is the best for at least one domain.
        """
        matrix = self.get_pareto_matrix()
        if not matrix:
            return {"frontier": [], "dominated": [], "matrix": matrix}

        all_domains = set()
        for method_data in matrix.values():
            all_domains.update(method_data.keys())

        # Find best method per domain
        best_per_domain = {}
        for domain in all_domains:
            best_method = None
            best_mae = float('inf')
            for method, domain_data in matrix.items():
                if domain in domain_data and domain_data[domain]["best_mae"] < best_mae:
                    best_mae = domain_data[domain]["best_mae"]
                    best_method = method
            best_per_domain[domain] = {"method": best_method, "mae": best_mae}

        # Pareto-optimal: methods that are best for at least one domain
        frontier_methods = {v["method"] for v in best_per_domain.values() if v["method"]}
        dominated_methods = set(matrix.keys()) - frontier_methods

        return {
            "frontier": list(frontier_methods),
            "dominated": list(dominated_methods),
            "best_per_domain": best_per_domain,
            "matrix": matrix,
        }

    # ── Causal Chain Tracking ──

    def record_causal_chain_entry(self, cycle: int, design_decision: str,
                                   metric_affected: str = "", expected_effect: str = "",
                                   actual_effect: str = "", verified: int = 0):
        """Record a causal link between a design decision and a metric outcome.

        Fed by the REFLECT phase's 'causal_link' field. Consumed by
        get_causal_history() which injects into THINK context.
        """
        if not design_decision:
            return
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                INSERT INTO causal_chain
                    (cycle, design_decision, architectural_property,
                     metric_affected, expected_effect, actual_effect, verified, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (cycle, design_decision[:500], "", metric_affected[:200],
                  expected_effect[:500], actual_effect[:500], verified, time.time()))

    def get_causal_history(self, limit: int = 20) -> list[dict]:
        """Get recent causal chain entries for reasoning."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT cycle, design_decision, architectural_property,
                       metric_affected, expected_effect, actual_effect, verified
                FROM causal_chain
                ORDER BY cycle DESC
                LIMIT ?
            """, (limit,)).fetchall()
            return [dict(r) for r in rows]

    # ── Experiment Value of Information ──

    def record_experiment_value(self, cycle: int, hypothesis: str,
                                 expected_improvement: float = None,
                                 actual_improvement: float = None,
                                 was_correct: int = None):
        """Record the value-of-information for one experiment.

        Tracks whether a hypothesis prediction matched reality. Consumed by
        get_experiment_calibration() which feeds StrategyConstraintEngine
        to learn which hypothesis types tend to work.
        """
        if not hypothesis:
            return
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                INSERT INTO experiment_value
                    (cycle, hypothesis, expected_improvement, prior_probability,
                     information_value, actual_improvement, was_correct, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (cycle, hypothesis[:500], expected_improvement, None, None,
                  actual_improvement, was_correct, time.time()))

    def get_experiment_calibration(self) -> dict:
        """Get calibration data: how often do hypotheses actually work?

        Returns calibration stats to help the agent learn which hypotheses
        are more likely to succeed.
        """
        with sqlite3.connect(str(self.db_path)) as conn:
            try:
                total = conn.execute(
                    "SELECT COUNT(*) FROM experiment_value WHERE was_correct IS NOT NULL"
                ).fetchone()[0]
                correct = conn.execute(
                    "SELECT COUNT(*) FROM experiment_value WHERE was_correct = 1"
                ).fetchone()[0]

                avg_expected = conn.execute(
                    "SELECT AVG(expected_improvement) FROM experiment_value"
                ).fetchone()[0] or 0
                avg_actual = conn.execute(
                    "SELECT AVG(actual_improvement) FROM experiment_value WHERE actual_improvement IS NOT NULL"
                ).fetchone()[0] or 0

                return {
                    "total_hypotheses": total,
                    "correct_hypotheses": correct,
                    "accuracy": correct / max(total, 1),
                    "avg_expected_improvement": round(avg_expected, 4),
                    "avg_actual_improvement": round(avg_actual, 4),
                    "overconfidence_ratio": round(avg_expected / max(avg_actual, _EPS), 2),
                }
            except Exception:
                return {"total_hypotheses": 0}

    def get_low_value_experiments(self, limit: int = 5) -> list[dict]:
        """Get experiments previously assessed as low value (VOI < 0.01).

        Phase 1: surfaces directions the agent already evaluated as unlikely
        to help, so it doesn't waste cycles repeating them.
        """
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    """SELECT hypothesis, information_value as voi,
                              prior_probability as prior, expected_improvement, cycle
                       FROM experiment_value
                       WHERE information_value < 0.01
                       ORDER BY cycle DESC LIMIT ?""",
                    (limit,)
                ).fetchall()
                return [dict(r) for r in rows]
        except Exception:
            return []

    def _query_best(self, conn, metric_key: str) -> float | None:
        """Query best metric value for a specific key from metrics_json."""
        best = None
        rows = conn.execute(
            "SELECT metrics_json FROM experiments WHERE metrics_json IS NOT NULL"
        ).fetchall()
        for (mj,) in rows:
            try:
                d = json.loads(mj) if isinstance(mj, str) else mj
                if isinstance(d, dict) and metric_key in d:
                    try:
                        v = float(d[metric_key])
                        if best is None or v < best:
                            best = v
                    except (ValueError, TypeError):
                        continue
            except (json.JSONDecodeError, TypeError):
                continue
        return best

    def get_method_domain_effect_matrix(self) -> dict:
        """Build method→domain→effect matrix from structured data.

        Replaces regex-based pattern detection in _build_cross_experiment_insights.
        Uses actual metrics from experiments table instead of text matching.

        Note: dead_end tracking was migrated to the memory_entries table (see
        B9 gate / log_dead_end). The experiments table no longer carries a
        dead_end column. success_rate/dead_ends below are kept as baseline
        fields (1.0 / 0) so downstream dict consumers (api/tools) don't break;
        they are no longer driven by per-cycle dead-end data.
        """
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT cycle, hypothesis, metrics_json, experiment_launched,
                       milestone
                FROM experiments
                WHERE experiment_launched = 1 AND metrics_json != '{}'
                ORDER BY cycle ASC
            """).fetchall()

            # Extract method names from hypothesis text
            import re
            method_keywords = self.method_keywords
            domain_keys = self.domain_keys

            matrix = {}  # method → {domain: [mae_values]}
            for row in rows:
                # Detect method
                hypothesis_lower = (row["hypothesis"] or "").lower()
                methods_found = []
                for mk, keywords in method_keywords.items():
                    for kw in keywords:
                        if kw in hypothesis_lower:
                            methods_found.append(mk)
                            break

                if not methods_found:
                    continue

                # Extract domain metrics
                try:
                    metrics = json.loads(row["metrics_json"])
                except (json.JSONDecodeError, TypeError):
                    continue

                for method in methods_found:
                    if method not in matrix:
                        matrix[method] = {}
                    for dk in domain_keys:
                        if dk in metrics:
                            domain_name = dk.replace("MAE_", "")
                            if domain_name not in matrix[method]:
                                matrix[method][domain_name] = []
                            try:
                                mae_val = float(metrics[dk])
                                matrix[method][domain_name].append({
                                    "mae": mae_val,
                                    "cycle": row["cycle"],
                                })
                            except (TypeError, ValueError):
                                pass

            # Summarize: best MAE per method per domain + success rate.
            # Note: dead_end tracking moved to memory_entries; these per-domain
            # dead_ends/success_rate fields are kept as baseline (0 / 1.0) so
            # downstream dict consumers don't KeyError.
            summary = {}
            for method, domains in matrix.items():
                summary[method] = {}
                for domain, entries in domains.items():
                    maes = [e["mae"] for e in entries]
                    # dead_end tracking moved to memory_entries (see docstring);
                    # these two fields are baseline placeholders, not driven by
                    # per-cycle dead-end data.
                    dead_ends = 0
                    summary[method][domain] = {
                        "best_mae": round(min(maes), 4),
                        "avg_mae": round(sum(maes) / len(maes), 4),
                        "attempts": len(entries),
                        "dead_ends": dead_ends,
                        "success_rate": 1.0,
                    }

            return summary

    def get_stuck_domains_structured(self, threshold: float = 0.30) -> list[dict]:
        """Find domains where no method has achieved MAE below threshold."""
        matrix = self.get_method_domain_effect_matrix()
        if not matrix:
            return []

        all_domains = set()
        for method_data in matrix.values():
            all_domains.update(method_data.keys())

        stuck = []
        for domain in all_domains:
            best_mae = min(
                (matrix[m][domain]["best_mae"] for m in matrix if domain in matrix[m]),
                default=float('inf')
            )
            if best_mae >= threshold:
                methods_tried = [
                    m for m in matrix if domain in matrix[m]
                ]
                stuck.append({
                    "domain": domain,
                    "best_mae": round(best_mae, 4),
                    "methods_tried": methods_tried,
                    "total_attempts": sum(
                        matrix[m][domain]["attempts"] for m in methods_tried
                    ),
                    "implication": (
                        f"{len(methods_tried)} methods tried, none below MAE {threshold}. "
                        f"Structural problem: core assumption violated."
                    ),
                })
        return stuck

    def get_brief(self) -> str:
        """Return the frozen project brief (Tier 1)."""
        if self.brief_path.exists():
            content = self.brief_path.read_text()
            return content[: self.brief_max]
        return ""

    def get_log(self) -> str:
        """Return the rolling memory log (Tier 2)."""
        if self.log_path.exists():
            return self.log_path.read_text()
        return ""

    def log_structured_result(self, cycle: int, metric_key: str,
                              metric_value: float, method: str = "",
                              status: str = "success"):
        """Phase 1: Write a structured quantitative result line to MEMORY_LOG.

        This is the SYSTEM-written anchor that guarantees quantitative results
        reach the LLM's text memory channel. Previously, val_MAE=0.184 existed
        only in SQLite but never in MEMORY_LOG unless the LLM happened to write
        it in a free-text milestone. Now the system writes it deterministically.

        Format: [Cycle N] metric=val method=X status=Y
        Compatible with _parse_log (starts with '[').
        """
        sections = self._parse_log()
        method_str = f" method={method}" if method else ""
        try:
            mv = f"{float(metric_value):.6f}"
        except (TypeError, ValueError):
            mv = str(metric_value)[:20]
        line = f"[Cycle {cycle}] {metric_key}={mv}{method_str} status={status}"
        sections["milestones"].append(line)
        self._write_log(sections)

    def log_milestone(self, entry: str, cycle: int = None):
        """Add a key result milestone. Auto-compacts if over budget."""
        sections = self._parse_log()
        timestamp = time.strftime("%m-%d %H:%M")
        sections["milestones"].append(f"[{timestamp}] {entry}")
        self._record_memory_entry("milestone", entry, cycle)

        # Compact: drop oldest milestones if over char budget
        while self._section_size(sections["milestones"]) > self.milestone_max and len(sections["milestones"]) > 1:
            sections["milestones"].pop(0)

        self._write_log(sections)

    def log_decision(self, entry: str, cycle: int = None):
        """Add a recent decision. Auto-compacts to keep only last N."""
        sections = self._parse_log()
        timestamp = time.strftime("%m-%d %H:%M")
        sections["decisions"].append(f"[{timestamp}] {entry}")
        self._record_memory_entry("decision", entry, cycle)

        # Compact: keep only last N entries
        if len(sections["decisions"]) > self.max_recent:
            sections["decisions"] = sections["decisions"][-self.max_recent :]

        self._write_log(sections)

    def log_dead_end(self, entry: str, cycle: int = None, failure_category: str = ""):
        """Add a dead end (failed approach). Dead ends are NEVER deleted to prevent repeating mistakes.

        Args:
            entry: Description of the failed approach and WHY it failed.
            cycle: Cycle number when the dead end occurred.
            failure_category: One of 'hypothesis_wrong', 'implementation_bug',
                'insufficient_experiment', 'method_inadequacy'. Used for
                structured retrieval and preventing false-negative dead ends.
        """
        sections = self._parse_log()
        timestamp = time.strftime("%m-%d %H:%M")
        # Prepend category tag for structured retrieval
        if failure_category:
            tagged_entry = f"[{failure_category}] [{timestamp}] {entry}"
        else:
            tagged_entry = f"[{timestamp}] {entry}"
        sections["dead_ends"].append(tagged_entry)
        # Record to SQLite with category
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                INSERT INTO memory_entries (timestamp, entry_type, content, cycle, in_llm_context)
                VALUES (?, ?, ?, ?, 1)
            """, (time.time(), "dead_end", tagged_entry, cycle))
            # Also update category if column exists (v12 migration)
            try:
                conn.execute("""
                    ALTER TABLE memory_entries ADD COLUMN failure_category TEXT NOT NULL DEFAULT ''
                """)
            except Exception:
                pass  # Column already exists
            if failure_category:
                conn.execute("""
                    UPDATE memory_entries SET failure_category = ? WHERE content = ? AND entry_type = 'dead_end'
                """, (failure_category, tagged_entry))
        self._write_log(sections)

    def log_active_problem(self, entry: str, cycle: int = None):
        """Add an active problem. Resolved problems should be moved to dead ends."""
        sections = self._parse_log()
        timestamp = time.strftime("%m-%d %H:%M")
        sections["active_problems"].append(f"[{timestamp}] {entry}")
        self._record_memory_entry("active_problem", entry, cycle)
        self._write_log(sections)

    def log_major_event(self, entry: str, cycle: int = None):
        """Log a major research event (e.g., paper research breakthrough, paradigm shift).

        Major events are stored in milestones (which are preserved during compaction)
        AND duplicated into decisions with a ★ prefix for easy scanning.
        This ensures they are never lost even if decisions are trimmed.
        """
        sections = self._parse_log()
        timestamp = time.strftime("%m-%d %H:%M")
        # Add to milestones — these are preserved during compaction
        sections["milestones"].append(f"★ [{timestamp}] PAPER_RESEARCH: {entry}")
        # Also add to decisions for visibility, with ★ prefix
        sections["decisions"].append(f"★ [{timestamp}] PAPER_RESEARCH: {entry}")
        # Record to SQLite
        self._record_memory_entry("major_event", f"PAPER_RESEARCH: {entry}", cycle)

        # Routine decisions are trimmed, but ★-prefixed major events are preserved
        if len(sections["decisions"]) > self.max_recent:
            # Keep recent major events (★) + the most recent routine decisions
            major = [d for d in sections["decisions"] if d.startswith("★")]
            routine = [d for d in sections["decisions"] if not d.startswith("★")]
            # Cap major events to half of budget, routine gets the rest
            max_major = max(self.max_recent // 2, 3)
            if len(major) > max_major:
                major = major[-max_major:]  # Keep only the most recent major events
            remaining = max(0, self.max_recent - len(major))
            routine = routine[-remaining:]
            sections["decisions"] = major + routine

        self._write_log(sections)

    # ─────────────────────────────────────────────────
    # Code Review Lessons (Knowledge Base)
    # ─────────────────────────────────────────────────

    def record_code_review_lesson(self, cycle: int, category: str,
                                    pattern: str, description: str,
                                    fix_suggestion: str = ""):
        """Record a reusable code/architecture lesson from a cycle.

        Fed by the REFLECT phase's 'lesson' field. Consumed by
        get_code_review_lessons() and query_memory(type='lessons').
        """
        if not description:
            return
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                INSERT INTO code_review_lessons
                    (timestamp, cycle, category, pattern, description,
                     fix_suggestion, source)
                VALUES (?, ?, ?, ?, ?, ?, 'reflect')
            """, (time.time(), cycle, category[:100], pattern[:200],
                  description[:1000], fix_suggestion[:500]))

    def get_code_review_lessons(self, severity: str = None, category: str = None, limit: int = 30) -> list[dict]:
        """Retrieve code review lessons, optionally filtered."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            query = "SELECT * FROM code_review_lessons WHERE 1=1"
            params = []
            if severity:
                # Severity is TEXT ("HIGH"/"MEDIUM"/"LOW") — use IN clause
                sev_rank = {"HIGH": 2, "MEDIUM": 1, "LOW": 0}
                min_rank = sev_rank.get(severity, 1)
                allowed = [s for s, r in sev_rank.items() if r >= min_rank]
                placeholders = ",".join("?" * len(allowed))
                query += f" AND severity IN ({placeholders})"
                params.extend(allowed)
            if category:
                query += " AND category = ?"
                params.append(category)
            query += " ORDER BY hit_count DESC, timestamp DESC LIMIT ?"
            params.append(limit)
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]

    def search_relevant_lessons(self, code_content: str, limit: int = 10) -> list[dict]:
        """Find lessons relevant to the given code content.

        Uses pattern matching against the code to find historically-relevant
        mistakes. This is the key method for context injection.
        """
        if not code_content:
            return []

        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            # Cap at 100 most-relevant candidates to avoid loading entire table
            rows = conn.execute(
                "SELECT * FROM code_review_lessons ORDER BY hit_count DESC, timestamp DESC LIMIT 100"
            ).fetchall()

            scored = []
            code_lower = code_content.lower()
            for row in rows:
                lesson = dict(row)
                pattern = lesson.get("pattern", "").lower()
                cat = lesson.get("category", "").lower()

                # Require at least one keyword match for relevance
                pattern_match = bool(pattern and pattern in code_lower)
                cat_match = bool(cat and cat in code_lower)
                if not pattern_match and not cat_match:
                    continue

                score = 0
                if pattern_match:
                    score += 10
                if cat_match:
                    score += 5
                # hit_count as tiebreaker (scaled down to not dominate)
                score += min(lesson["hit_count"], 10)
                sev_boost = {"HIGH": 5, "MEDIUM": 2, "LOW": 0}
                score += sev_boost.get(lesson.get("severity", "MEDIUM"), 0)

                lesson["_relevance_score"] = score
                scored.append(lesson)

            scored.sort(key=lambda x: x["_relevance_score"], reverse=True)
            return scored[:limit]

    def format_lessons_for_context(self, lessons: list[dict], max_chars: int = 2000) -> str:
        """Format lessons as a compact string for LLM context injection."""
        if not lessons:
            return ""

        lines = ["## Code Review Lessons (PAST MISTAKES TO AVOID)", ""]
        used = len(lines[0]) + len(lines[1])

        for lesson in lessons:
            sev = lesson.get("severity", "MEDIUM")
            hits = lesson.get("hit_count", 1)
            pattern = lesson.get("pattern", "?")
            desc = lesson.get("description", "")
            fix = lesson.get("fix_suggestion", "")

            entry = f"- [{sev}] (hit {hits}x) **{pattern}**: {desc[:200]}"
            if fix:
                entry += f" → Fix: {fix[:150]}"

            if used + len(entry) > max_chars:
                break
            lines.append(entry)
            used += len(entry)

        lines.append("")
        lines.append(
            "**IMPORTANT**: These are mistakes the agent has made before. "
            "Check the current code for these patterns BEFORE writing any code."
        )
        return "\n".join(lines)

    def _init_log(self):
        """Create initial empty memory log."""
        content = "# Memory Log\n\n## Key Results\n\n## Dead Ends\n\n## Active Problems\n\n## Recent Decisions\n"
        # Atomic write to prevent corruption on crash
        tmp_path = self.log_path.with_suffix(".tmp")
        tmp_path.write_text(content)
        tmp_path.replace(self.log_path)

    def _parse_log(self) -> dict:
        """Parse MEMORY_LOG.md into sections."""
        content = self.get_log()
        sections = {"milestones": [], "dead_ends": [], "active_problems": [], "decisions": []}

        current_section = None
        for line in content.split("\n"):
            line_stripped = line.strip()
            if line_stripped == "## Key Results":
                current_section = "milestones"
            elif line_stripped == "## Dead Ends":
                current_section = "dead_ends"
            elif line_stripped == "## Active Problems":
                current_section = "active_problems"
            elif line_stripped == "## Recent Decisions":
                current_section = "decisions"
            elif line_stripped and current_section and (line_stripped.startswith("[") or line_stripped.startswith("★")):
                sections[current_section].append(line_stripped)

        return sections

    def _write_log(self, sections: dict):
        """Write sections back to MEMORY_LOG.md."""
        content = self._build_content(sections)

        # Final safety check: total log must fit budget
        if len(content) > self.log_max:
            # Compression priority: summarize old entries instead of deleting.
            # This preserves knowledge while fitting the budget.
            # dead_ends are NEVER trimmed — only compressed if very long.
            for section_name in ("milestones", "decisions", "active_problems"):
                if len(content) > self.log_max and len(sections[section_name]) > 3:
                    sections[section_name] = self._compress_section(sections[section_name])
                    content = self._build_content(sections)
            # dead_ends: only compress, never delete
            if len(content) > self.log_max and len(sections["dead_ends"]) > 5:
                sections["dead_ends"] = self._compress_section(sections["dead_ends"])
                content = self._build_content(sections)
            # Fallback: if still too large, trim oldest entries (NOT dead_ends)
            for section_name in ("milestones", "decisions", "active_problems"):
                while len(content) > self.log_max and len(sections[section_name]) > 1:
                    sections[section_name].pop(0)
                    content = self._build_content(sections)

        # Atomic write to prevent corruption on crash
        tmp_path = self.log_path.with_suffix(".tmp")
        tmp_path.write_text(content)
        tmp_path.replace(self.log_path)

    def _build_content(self, sections: dict) -> str:
        lines = ["# Memory Log", "", "## Key Results"]
        lines.extend(sections.get("milestones", []))
        lines.append("")
        lines.append("## Dead Ends")
        lines.extend(sections.get("dead_ends", []))
        lines.append("")
        lines.append("## Active Problems")
        lines.extend(sections.get("active_problems", []))
        lines.append("")
        lines.append("## Recent Decisions")
        lines.extend(sections.get("decisions", []))
        lines.append("")
        return "\n".join(lines)

    def _section_size(self, entries: list) -> int:
        return sum(len(e) for e in entries)

    def _compress_section(self, entries: list) -> list:
        """Compress a section by summarizing old entries instead of deleting them.

        Keeps the most recent entries intact and compresses older ones into
        a single summary line. This preserves knowledge while fitting the budget.
        """
        if len(entries) <= 3:
            return entries

        recent = entries[-3:]
        old = entries[:-3]

        # Extract key themes from old entries
        themes = []
        for entry in old:
            # Take first 60 chars of each old entry as a theme
            theme = entry[:60].rstrip()
            if theme and theme not in themes:
                themes.append(theme)

        if themes:
            compressed = f"[Historical {len(old)} entries: {'; '.join(themes[:3])}]"
            return [compressed] + recent
        return recent

    # ── Phase 1 (Reform v21): Deterministic fact spine ──

    def scan_experiment_facts(self, rescan: bool = False) -> dict[str, int]:
        """Scan all experiment manifests + logs on disk, record structured facts.

        This is the fact spine: it reads files that survived any reboot/crash
        and records what happened — independent of any LLM or agent process.
        Idempotent (output_dir is the primary key). Safe to call every cycle.

        Returns {"scanned": N, "inserted": M, "skipped": K, "errors": E}.
        """
        from core.fact_scanner import scan_all

        return scan_all(self.project_dir, self.db_path, rescan=rescan)

    def get_experiment_facts(self, limit: int = 20) -> list[dict]:
        """Retrieve experiment facts (newest scan first). For THINK context + gates."""
        from core.fact_scanner import get_all_facts

        return get_all_facts(self.db_path)[:limit]

    def get_fact_for_output_dir(self, output_dir: str) -> dict | None:
        """Retrieve the fact record for a specific experiment output_dir."""
        from core.fact_scanner import get_facts_for_output_dir

        return get_facts_for_output_dir(self.db_path, output_dir)
