"""Tests for Phase 3 (Reform v21): Methodology gates.

Three fact-layer gates that check scientific methodology deterministically:
  1. Falsification: success_criteria predicate evaluation (pure math)
  2. Control-coverage: causal claim without control → mark inconclusive
  3. Dead-end signature: structured method matching (replaces 16-word keywords)

All gates operate on FACTS only. None change think_result["action"].
The boundary (fact vs interpretation) is tested explicitly in each suite.
"""
from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.methodology_gates import (
    evaluate_falsification,
    check_control_coverage,
    check_dead_end_signature,
    run_all_gates,
    _normalize_metric_name,
    _build_method_signature,
    MethodologyVerdict,
)


# ═══════════════════════════════════════════════════════════════════════
# Gate 1: Falsification
# ═══════════════════════════════════════════════════════════════════════

class TestFalsificationGate:
    """success_criteria predicate evaluation — the core of M3/M3b."""

    def test_simple_met(self):
        """val_MAE < 0.15, actual 0.10 → MET."""
        r = evaluate_falsification("val_MAE < 0.15", {"val_mae": 0.10})
        assert r.parseable is True
        assert r.criteria_met is True
        assert r.threshold == 0.15

    def test_simple_not_met(self):
        """val_MAE < 0.15, actual 0.20 → NOT MET."""
        r = evaluate_falsification("val_MAE < 0.15", {"val_mae": 0.20})
        assert r.parseable is True
        assert r.criteria_met is False

    def test_le_operator(self):
        r = evaluate_falsification("val_MAE <= 0.15", {"val_mae": 0.15})
        assert r.criteria_met is True

    def test_greater_than(self):
        r = evaluate_falsification("psnr > 20.0", {"psnr": 25.5})
        assert r.criteria_met is True

    def test_equal(self):
        r = evaluate_falsification("accuracy == 0.95", {"accuracy": 0.95})
        assert r.criteria_met is True

    def test_naming_chaos_val_MAE_vs_val_mae(self):
        """THE key test: criteria says 'val_MAE', metrics key is 'val_mae'."""
        r = evaluate_falsification("val_MAE < 0.15", {"val_mae": 0.10})
        assert r.parseable is True
        assert r.actual_value == 0.10
        assert r.criteria_met is True

    def test_naming_chaos_overall_variants(self):
        """'overall_MAE' in criteria vs 'mae_overall' in metrics."""
        r = evaluate_falsification("overall_MAE < 0.20", {"mae_overall": 0.18})
        assert r.actual_value == 0.18
        assert r.criteria_met is True

    def test_naming_chaos_lambertian(self):
        """'Lambertian_MAE' vs 'mae_lambertian'."""
        r = evaluate_falsification("Lambertian_MAE < 0.16", {"mae_lambertian": 0.164})
        assert r.actual_value == 0.164
        assert r.criteria_met is False  # 0.164 >= 0.16

    def test_metric_not_found(self):
        """criteria references a metric not in results → criteria_met=None (honest)."""
        r = evaluate_falsification("val_MAE < 0.15", {"rmse": 0.5})
        assert r.parseable is True
        assert r.criteria_met is None
        assert "not found" in r.detail

    def test_unparseable_qualitative(self):
        """Qualitative criteria → unparseable, criteria_met=None (NOT False)."""
        r = evaluate_falsification("verify energy guidance reduces error", {})
        assert r.parseable is False
        assert r.criteria_met is None
        assert "unparseable" in r.detail

    def test_empty_criteria(self):
        r = evaluate_falsification("", {"val_mae": 0.1})
        assert r.parseable is False
        assert r.criteria_met is None

    def test_none_criteria(self):
        r = evaluate_falsification(None, {"val_mae": 0.1})
        assert r.parseable is False

    def test_no_metrics(self):
        """Parseable criteria but no metrics at all → None (honest)."""
        r = evaluate_falsification("val_MAE < 0.15", {})
        assert r.parseable is True
        assert r.criteria_met is None

    def test_v30_scenario(self):
        """The actual V30 case: Lambertian 0.364 vs target 0.16."""
        r = evaluate_falsification(
            "Lambertian_MAE < 0.16",
            {"mae_lambertian": 0.364, "val_mae": 0.144, "mae_urban": 0.093},
        )
        assert r.parseable is True
        assert r.criteria_met is False  # 0.364 >> 0.16
        assert r.actual_value == 0.364


# ═══════════════════════════════════════════════════════════════════════
# Gate 2: Control-coverage
# ═══════════════════════════════════════════════════════════════════════

class TestControlCoverageGate:
    """causal claim without control → marked inconclusive (NOT forced)."""

    @pytest.fixture
    def db_with_experiments(self, tmp_path):
        """Temp DB with experiment_facts table and some experiments."""
        db = tmp_path / "test.db"
        with sqlite3.connect(str(db)) as conn:
            conn.execute("""
                CREATE TABLE experiment_facts (
                    output_dir TEXT PRIMARY KEY,
                    command TEXT NOT NULL DEFAULT '',
                    metrics_json TEXT NOT NULL DEFAULT '{}'
                )
            """)
            # Insert: main experiment + an ablation + an unrelated one
            conn.execute(
                "INSERT INTO experiment_facts (output_dir, command) VALUES (?, ?)",
                ("/proj/outputs/v30", "python train_v30.py --energy_guided"),
            )
            conn.execute(
                "INSERT INTO experiment_facts (output_dir, command) VALUES (?, ?)",
                ("/proj/outputs/v30_ablation", "python train_v30.py --no_energy_guided"),
            )
            conn.execute(
                "INSERT INTO experiment_facts (output_dir, command) VALUES (?, ?)",
                ("/proj/outputs/exp2", "python train_other.py"),
            )
            conn.commit()
        return db

    def test_causal_without_control_marked_inconclusive(self, tmp_path):
        """causal claim, NO control run of same method in DB → inconclusive.

        Uses a clean DB with only the main experiment (no ablation).
        """
        db = tmp_path / "no_control.db"
        with sqlite3.connect(str(db)) as conn:
            conn.execute("""
                CREATE TABLE experiment_facts (
                    output_dir TEXT PRIMARY KEY,
                    command TEXT NOT NULL DEFAULT '',
                    metrics_json TEXT NOT NULL DEFAULT '{}'
                )
            """)
            # Only the main experiment, no ablation/control/baseline
            conn.execute(
                "INSERT INTO experiment_facts (output_dir, command) VALUES (?, ?)",
                ("/proj/outputs/v30_main", "python train_v30.py --energy_guided"),
            )
            conn.commit()
        r = check_control_coverage(
            "causal", "cost_volume", db, "/proj/outputs/v30_main"
        )
        assert r.needs_control is True
        assert r.marked_inconclusive is True
        assert r.control_exists is False

    def test_causal_with_control_found(self, db_with_experiments):
        """causal claim, ablation exists → control_exists=True, NOT inconclusive."""
        r = check_control_coverage(
            "causal", "cost_volume", db_with_experiments, "/proj/outputs/v30"
        )
        assert r.needs_control is True
        assert r.control_exists is True
        assert r.marked_inconclusive is False

    def test_non_causal_no_check_needed(self, db_with_experiments):
        """correlational claim → no control needed."""
        r = check_control_coverage(
            "correlational", "cost_volume", db_with_experiments, "/proj/outputs/exp2"
        )
        assert r.needs_control is False
        assert r.marked_inconclusive is False

    def test_null_claim_no_check(self, db_with_experiments):
        r = check_control_coverage(
            "null", "cost_volume", db_with_experiments, "/proj/outputs/exp2"
        )
        assert r.needs_control is False

    def test_none_claim_treated_as_null(self, db_with_experiments):
        r = check_control_coverage(
            None, "cost_volume", db_with_experiments, "/proj/outputs/exp2"
        )
        assert r.needs_control is False

    def test_does_not_force_action(self, db_with_experiments):
        """BOUNDARY TEST: gate result is a FACT, it must NOT include any
        'force_action' or 'block' field. Interpretation stays with LLM."""
        r = check_control_coverage(
            "causal", "cost_volume", db_with_experiments, "/proj/outputs/exp2"
        )
        d = r.to_dict()
        assert "force_action" not in d
        assert "block" not in d
        assert "action" not in d


# ═══════════════════════════════════════════════════════════════════════
# Gate 3: Dead-end signature
# ═══════════════════════════════════════════════════════════════════════

class TestDeadEndSignatureGate:
    """Structured method matching replaces 16-word keyword substring match."""

    @pytest.fixture
    def db_with_dead_ends(self, tmp_path):
        """Temp DB with memory_entries table containing dead_end entries.

        dead_end source of truth is memory_entries (entry_type='dead_end'),
        not the experiments table — see docs/DATA_CONTRACT.md.
        """
        db = tmp_path / "test.db"
        with sqlite3.connect(str(db)) as conn:
            conn.execute("""
                CREATE TABLE memory_entries (
                    id INTEGER PRIMARY KEY,
                    timestamp REAL NOT NULL DEFAULT 0,
                    entry_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    cycle INTEGER,
                    in_llm_context INTEGER NOT NULL DEFAULT 0
                )
            """)
            # Phase 1 (material classification) confirmed dead.
            # content carries the method text the gate matches on.
            conn.execute(
                "INSERT INTO memory_entries (entry_type, content, cycle) VALUES (?, ?, ?)",
                ("dead_end", "FFT spectral material classification: FFT classification dead, AUC 0.557", 1),
            )
            conn.execute(
                "INSERT INTO memory_entries (entry_type, content, cycle) VALUES (?, ?, ?)",
                ("dead_end", "FFT approach fails on non-Lambertian", 2),
            )
            conn.commit()
        return db

    def test_dead_end_matched(self, db_with_dead_ends):
        """Method 'fft' matches recorded dead ends."""
        think = {"method": "fft", "task": "try FFT again", "hypothesis": ""}
        r = check_dead_end_signature(think, {"mae": 0.3}, db_with_dead_ends)
        assert r.is_known_dead_end is True
        assert len(r.matched_dead_ends) >= 1

    def test_new_method_no_match(self, db_with_dead_ends):
        """New method 'cost_volume' has no dead-end history → not matched (honest)."""
        think = {
            "method": "cost_volume",
            "task": "energy-guided cost volume",
            "hypothesis": "energy guidance reduces Lambertian error",
        }
        r = check_dead_end_signature(think, {"mae_lambertian": 0.3}, db_with_dead_ends)
        assert r.is_known_dead_end is False
        assert len(r.matched_dead_ends) == 0

    def test_empty_dead_end_history(self, tmp_path):
        """No dead ends in DB → no match, honest 'no data'."""
        db = tmp_path / "empty.db"
        with sqlite3.connect(str(db)) as conn:
            conn.execute("""
                CREATE TABLE memory_entries (
                    id INTEGER PRIMARY KEY, cycle INTEGER,
                    entry_type TEXT NOT NULL, content TEXT NOT NULL,
                    timestamp REAL NOT NULL DEFAULT 0,
                    in_llm_context INTEGER NOT NULL DEFAULT 0
                )
            """)
            conn.commit()
        think = {"method": "anything", "task": "", "hypothesis": ""}
        r = check_dead_end_signature(think, {}, db)
        assert r.is_known_dead_end is False
        assert "no dead ends" in r.detail

    def test_does_not_block_action(self, db_with_dead_ends):
        """BOUNDARY TEST: dead-end gate WARNS, must NOT block action."""
        think = {"method": "fft", "task": "", "hypothesis": ""}
        r = check_dead_end_signature(think, {}, db_with_dead_ends)
        d = r.to_dict()
        assert "force_action" not in d
        assert "block" not in d
        # It warns (is_known_dead_end=True) but doesn't dictate action
        assert r.is_known_dead_end is True

    def test_method_extraction_from_hypothesis(self, db_with_dead_ends):
        """When 'method' field absent, extract from hypothesis text."""
        think = {
            "task": "train model",
            "hypothesis": "FFT-based frequency analysis improves accuracy",
        }
        sig = _build_method_signature(think, {})
        assert "fft" in sig

    def test_dead_end_e2e_memory_entries_to_gate(self, tmp_path):
        """E2E: a dead_end row in memory_entries is picked up by the gate.

        Verifies the L1 migration: dead_end data lives ONLY in memory_entries
        (the experiments table has no dead_end column anymore), and the B9
        gate reads it from there. This is the regression guard for the
        migration — if someone re-adds a dead_end column to experiments and
        reads from it, this test still asserts the memory_entries path works.
        """
        db = tmp_path / "e2e.db"
        with sqlite3.connect(str(db)) as conn:
            conn.execute("""
                CREATE TABLE memory_entries (
                    id INTEGER PRIMARY KEY, timestamp REAL NOT NULL DEFAULT 0,
                    entry_type TEXT NOT NULL, content TEXT NOT NULL,
                    cycle INTEGER, in_llm_context INTEGER NOT NULL DEFAULT 0
                )
            """)
            # Simulate what log_dead_end writes: a tagged content string.
            conn.execute(
                "INSERT INTO memory_entries (entry_type, content, cycle) VALUES (?, ?, ?)",
                ("dead_end", "[hypothesis_wrong] [06-29 12:00] GAN-based depth refinement is a dead end: increases MAE", 7),
            )
            conn.commit()
        think = {"method": "gan", "task": "retry GAN refinement", "hypothesis": ""}
        r = check_dead_end_signature(think, {"mae": 0.5}, db)
        assert r.is_known_dead_end is True
        assert any("GAN" in m or "gan" in m for m in r.matched_dead_ends)


# ═══════════════════════════════════════════════════════════════════════
# Combined: run_all_gates
# ═══════════════════════════════════════════════════════════════════════

class TestRunAllGates:
    """End-to-end: all three gates run together on a realistic scenario."""

    @pytest.fixture
    def full_db(self, tmp_path):
        db = tmp_path / "full.db"
        with sqlite3.connect(str(db)) as conn:
            # experiment_facts table
            conn.execute("""
                CREATE TABLE experiment_facts (
                    output_dir TEXT PRIMARY KEY,
                    command TEXT NOT NULL DEFAULT '',
                    metrics_json TEXT NOT NULL DEFAULT '{}'
                )
            """)
            conn.execute(
                "INSERT INTO experiment_facts (output_dir, command, metrics_json) VALUES (?,?,?)",
                (
                    "/proj/outputs/v30",
                    "python train_v30.py --energy_guided",
                    json.dumps({"val_mae": 0.144, "mae_lambertian": 0.364, "mae_urban": 0.093}),
                ),
            )
            # experiments table (with one dead end)
            conn.execute("""
                CREATE TABLE experiments (
                    id INTEGER PRIMARY KEY, cycle INTEGER,
                    hypothesis TEXT DEFAULT '', dead_end TEXT DEFAULT ''
                )
            """)
            conn.execute(
                "INSERT INTO experiments (cycle, hypothesis, dead_end) VALUES (1, 'fft', 'fft dead')"
            )
            conn.commit()
        return db

    def test_v30_full_scenario(self, full_db):
        """The V30 case: criteria not met + causal + no control + not dead end."""
        think = {
            "success_criteria": "Lambertian_MAE < 0.16",
            "claim_type": "causal",
            "method": "cost_volume",
            "task": "energy-guided cost volume",
            "hypothesis": "energy guidance reduces Lambertian error",
        }
        execute = {"log_file": "/proj/outputs/v30/train.log"}
        verdict = run_all_gates(think, execute, full_db, "/proj/outputs/v30")

        # Falsification: Lambertian 0.364 >= 0.16 → NOT MET
        assert verdict.falsification.criteria_met is False
        assert verdict.falsification.actual_value == 0.364

        # Control: causal but no ablation (only main run) → inconclusive
        assert verdict.control_coverage.marked_inconclusive is True

        # Dead end: cost_volume not in dead ends → no match
        assert verdict.dead_end.is_known_dead_end is False

        # Summary should mention criteria NOT MET and UNCONTROLLED
        s = verdict.summary()
        assert "NOT MET" in s
        assert "UNCONTROLLED" in s

    def test_summary_when_criteria_met(self, full_db):
        think = {
            "success_criteria": "val_MAE < 0.20",
            "claim_type": "null",
            "method": "cost_volume",
            "task": "", "hypothesis": "",
        }
        execute = {"log_file": "/proj/outputs/v30/train.log"}
        verdict = run_all_gates(think, execute, full_db, "/proj/outputs/v30")
        assert verdict.falsification.criteria_met is True  # 0.144 < 0.20
        assert "MET" in verdict.summary()

    def test_no_experiment_ran(self, full_db):
        """No log_file → gates handle gracefully."""
        think = {"success_criteria": "val_MAE < 0.15", "claim_type": "null"}
        verdict = run_all_gates(think, {}, full_db)
        # No metrics → falsification can't evaluate
        assert verdict.falsification.criteria_met is None
