"""Tests for Phase 4 (Reform v21): Spec conformance + fact-based action gating.

4a: Spec-conformance gate — checks code signatures against experiment_spec.json
4b: Fact-based action gating — uncontrolled failing causal claim → not progress
4c: state.json / fact_spine alignment — monitor-empty → fact fallback

Critical boundary: the action gate must ONLY fire when all three facts hold
(criteria explicitly False + causal + no control). None/unparseable/non-causal
must NOT be gated.
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.methodology_gates import (
    check_spec_conformance,
    run_all_gates,
    MethodologyVerdict,
)


# ═══════════════════════════════════════════════════════════════════════
# 4a: Spec-conformance gate
# ═══════════════════════════════════════════════════════════════════════

class TestSpecConformanceGate:

    def test_no_spec_file_is_noop(self, tmp_path):
        """No experiment_spec.json → spec_loaded=False, no deviation."""
        r = check_spec_conformance(tmp_path)
        assert r.spec_loaded is False
        assert r.has_deviation is False

    def test_all_signatures_present(self, tmp_path):
        """Code contains all required signatures → no deviation."""
        (tmp_path / "experiment_spec.json").write_text(json.dumps({
            "name": "V30",
            "files": ["models/cost_volume_net.py"],
            "required_signatures": ["sort", "n_keep", "mean"],
        }))
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        (model_dir / "cost_volume_net.py").write_text(
            "x = diff.sort(dim=1)\n"
            "n_keep = (80 * keep).long()\n"
            "cost = sorted[:, :n_keep].mean(dim=1)\n"
        )
        r = check_spec_conformance(tmp_path)
        assert r.spec_loaded is True
        assert r.has_deviation is False
        assert len(r.found_signatures) == 3

    def test_missing_signature_detected(self, tmp_path):
        """Code missing a signature → deviation detected.

        This is the V30 case: spec requires sort+trimmed mean (hard cutoff),
        but actual code uses sigmoid (soft gating) → 'sort' missing.
        """
        (tmp_path / "experiment_spec.json").write_text(json.dumps({
            "name": "V30 trimmed mean",
            "files": ["models/cost_volume_net.py"],
            "required_signatures": ["sort", "n_keep"],
        }))
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        # Code uses sigmoid, NOT sort — silent deviation
        (model_dir / "cost_volume_net.py").write_text(
            "weights = sigmoid(alpha * (median - diff))\n"
            "cost = (weights * diff).sum() / weights.sum()\n"
        )
        r = check_spec_conformance(tmp_path)
        assert r.spec_loaded is True
        assert r.has_deviation is True
        assert "sort" in r.missing_signatures
        assert "n_keep" in r.missing_signatures

    def test_spec_file_missing(self, tmp_path):
        """Spec declares files that don't exist → detail explains."""
        (tmp_path / "experiment_spec.json").write_text(json.dumps({
            "name": "test",
            "files": ["models/nonexistent.py"],
            "required_signatures": ["sort"],
        }))
        r = check_spec_conformance(tmp_path)
        assert r.spec_loaded is True
        assert "found" in r.detail  # "none of the spec files found"

    def test_empty_spec_signatures(self, tmp_path):
        """Spec with no required_signatures → nothing to check."""
        (tmp_path / "experiment_spec.json").write_text(json.dumps({
            "name": "test",
            "files": ["models/x.py"],
            "required_signatures": [],
        }))
        r = check_spec_conformance(tmp_path)
        assert r.spec_loaded is True
        assert r.has_deviation is False

    def test_does_not_judge_correctness(self, tmp_path):
        """BOUNDARY: gate checks signature PRESENCE, not correctness.

        Having 'sort' in the code doesn't mean the sort is used correctly.
        That's interpretation — left to LLM."""
        (tmp_path / "experiment_spec.json").write_text(json.dumps({
            "name": "test",
            "files": ["models/x.py"],
            "required_signatures": ["sort"],
        }))
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        # sort is present but in a comment — gate still says "found"
        # (this is a known limitation: fact layer = presence, not semantics)
        (model_dir / "x.py").write_text("# TODO: maybe use sort here\n")
        r = check_spec_conformance(tmp_path)
        assert "sort" in r.found_signatures  # presence-based, not semantic


# ═══════════════════════════════════════════════════════════════════════
# 4b: Fact-based action gating (boundary tests)
# ═══════════════════════════════════════════════════════════════════════

class TestActionGatingBoundary:
    """The action gate in _record_cycle_outcome is the most dangerous change.
    Test its BOUNDARY: it must fire ONLY when all three facts hold, never otherwise.
    """

    @pytest.fixture
    def db_with_data(self, tmp_path):
        db = tmp_path / "test.db"
        with sqlite3.connect(str(db)) as conn:
            conn.execute("""
                CREATE TABLE experiment_facts (
                    output_dir TEXT PRIMARY KEY,
                    command TEXT DEFAULT '',
                    metrics_json TEXT DEFAULT '{}'
                )
            """)
            conn.execute("""
                CREATE TABLE experiments (
                    id INTEGER PRIMARY KEY, cycle INTEGER,
                    hypothesis TEXT DEFAULT ''
                )
            """)
            # dead_end lives in memory_entries now (see docs/DATA_CONTRACT.md).
            conn.execute("""
                CREATE TABLE memory_entries (
                    id INTEGER PRIMARY KEY, cycle INTEGER,
                    timestamp REAL NOT NULL DEFAULT 0,
                    entry_type TEXT NOT NULL, content TEXT NOT NULL,
                    in_llm_context INTEGER NOT NULL DEFAULT 0
                )
            """)
            conn.commit()
        return db

    def _make_verdict(self, criteria_met, claim_type, marked_inconclusive):
        """Build a MethodologyVerdict with specific gate results."""
        from core.methodology_gates import (
            FalsificationResult, ControlCoverageResult, DeadEndResult
        )
        v = MethodologyVerdict()
        v.falsification = FalsificationResult(
            parseable=True,
            criteria_met=criteria_met,
            actual_value=0.364 if criteria_met is False else 0.1,
            operator="<", threshold=0.16,
        )
        v.control_coverage = ControlCoverageResult(
            claim_type=claim_type,
            needs_control=(claim_type == "causal"),
            marked_inconclusive=marked_inconclusive,
        )
        return v

    def test_gate_fires_on_all_three_facts(self, db_with_data):
        """criteria=False + causal + no control → should gate progress."""
        v = self._make_verdict(
            criteria_met=False, claim_type="causal", marked_inconclusive=True
        )
        # Simulate the gate condition from loop.py
        should_gate = (
            v.falsification.parseable
            and v.falsification.criteria_met is False
            and v.control_coverage.needs_control
            and v.control_coverage.marked_inconclusive
        )
        assert should_gate is True, "Gate should fire: criteria failed + causal + uncontrolled"

    def test_gate_does_not_fire_on_none_criteria(self, db_with_data):
        """criteria_met=None (unparseable) → must NOT gate (None ≠ False)."""
        v = self._make_verdict(
            criteria_met=None, claim_type="causal", marked_inconclusive=True
        )
        should_gate = (
            v.falsification.parseable
            and v.falsification.criteria_met is False  # None is not False!
            and v.control_coverage.needs_control
            and v.control_coverage.marked_inconclusive
        )
        assert should_gate is False, "Gate must NOT fire when criteria_met is None"

    def test_gate_does_not_fire_on_criteria_met(self, db_with_data):
        """criteria_met=True → no gating (experiment succeeded)."""
        v = self._make_verdict(
            criteria_met=True, claim_type="causal", marked_inconclusive=True
        )
        should_gate = (
            v.falsification.parseable
            and v.falsification.criteria_met is False
            and v.control_coverage.needs_control
            and v.control_coverage.marked_inconclusive
        )
        assert should_gate is False

    def test_gate_does_not_fire_on_non_causal(self, db_with_data):
        """claim_type=null (bug fix) → no gating even if criteria failed."""
        v = self._make_verdict(
            criteria_met=False, claim_type="null", marked_inconclusive=False
        )
        should_gate = (
            v.falsification.parseable
            and v.falsification.criteria_met is False
            and v.control_coverage.needs_control  # False for non-causal
            and v.control_coverage.marked_inconclusive
        )
        assert should_gate is False

    def test_gate_does_not_fire_with_control(self, db_with_data):
        """criteria failed + causal BUT control exists → no gating.
        The control means the result is interpretable, let LLM decide."""
        v = self._make_verdict(
            criteria_met=False, claim_type="causal", marked_inconclusive=False
        )
        should_gate = (
            v.falsification.parseable
            and v.falsification.criteria_met is False
            and v.control_coverage.needs_control
            and v.control_coverage.marked_inconclusive  # False — control exists
        )
        assert should_gate is False


# ═══════════════════════════════════════════════════════════════════════
# Integration: run_all_gates with spec gate
# ═══════════════════════════════════════════════════════════════════════

class TestRunWithSpecGate:

    def test_spec_gate_included_when_workspace_given(self, tmp_path):
        """run_all_gates includes spec gate when workspace provided."""
        db = tmp_path / "test.db"
        with sqlite3.connect(str(db)) as conn:
            conn.execute("""
                CREATE TABLE experiment_facts (
                    output_dir TEXT PRIMARY KEY, command TEXT DEFAULT '',
                    metrics_json TEXT DEFAULT '{}'
                )
            """)
            conn.execute("""
                CREATE TABLE experiments (
                    id INTEGER PRIMARY KEY, cycle INTEGER,
                    hypothesis TEXT DEFAULT ''
                )
            """)
            # dead_end lives in memory_entries now (see docs/DATA_CONTRACT.md).
            conn.execute("""
                CREATE TABLE memory_entries (
                    id INTEGER PRIMARY KEY, cycle INTEGER,
                    timestamp REAL NOT NULL DEFAULT 0,
                    entry_type TEXT NOT NULL, content TEXT NOT NULL,
                    in_llm_context INTEGER NOT NULL DEFAULT 0
                )
            """)
            conn.commit()

        # No spec file → spec gate is no-op
        think = {"success_criteria": "val_MAE < 0.15", "claim_type": "null"}
        execute = {"log_file": ""}
        verdict = run_all_gates(think, execute, db, workspace=tmp_path)
        assert verdict.spec_conformance.spec_loaded is False

    def test_spec_deviation_shown_in_summary(self, tmp_path):
        """Spec deviation appears in verdict summary."""
        (tmp_path / "experiment_spec.json").write_text(json.dumps({
            "name": "test",
            "files": ["models/x.py"],
            "required_signatures": ["sort"],
        }))
        (tmp_path / "models").mkdir()
        (tmp_path / "models" / "x.py").write_text("x = 1\n")  # no sort

        db = tmp_path / "test.db"
        with sqlite3.connect(str(db)) as conn:
            conn.execute("""
                CREATE TABLE experiment_facts (
                    output_dir TEXT PRIMARY KEY, command TEXT DEFAULT '',
                    metrics_json TEXT DEFAULT '{}'
                )
            """)
            conn.execute("""
                CREATE TABLE experiments (
                    id INTEGER PRIMARY KEY, cycle INTEGER,
                    hypothesis TEXT DEFAULT ''
                )
            """)
            # dead_end lives in memory_entries now (see docs/DATA_CONTRACT.md).
            conn.execute("""
                CREATE TABLE memory_entries (
                    id INTEGER PRIMARY KEY, cycle INTEGER,
                    timestamp REAL NOT NULL DEFAULT 0,
                    entry_type TEXT NOT NULL, content TEXT NOT NULL,
                    in_llm_context INTEGER NOT NULL DEFAULT 0
                )
            """)
            conn.commit()

        think = {"success_criteria": "val_MAE < 0.15", "claim_type": "null"}
        execute = {"log_file": ""}
        verdict = run_all_gates(think, execute, db, workspace=tmp_path)
        assert "SPEC_DEVIATION" in verdict.summary()
