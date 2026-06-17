"""Tests for the v20.1 memory table completion: causal_chain, experiment_value,
code_review_lessons write methods, pareto method extraction, and the full
write→read→consume loop.

These tests verify the tables are not just "wired" but actually produce data
that the existing read methods can consume.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from core.memory import MemoryManager


@pytest.fixture
def mem():
    """Fresh MemoryManager backed by a temp project dir."""
    tmp = Path(tempfile.mkdtemp())
    return MemoryManager(project_dir=tmp, workspace=tmp)


# ═════════════════════════════════════════════════════════════
# 1. causal_chain: write → read → THINK consumption
# ═════════════════════════════════════════════════════════════
class TestCausalChain:
    def test_write_then_read(self, mem):
        """record_causal_chain_entry writes, get_causal_history reads back."""
        mem.record_causal_chain_entry(
            cycle=1,
            design_decision="Added FFT branch to handle high-frequency detail",
            metric_affected="val_MAE",
            expected_effect="MAE should decrease on non-Lambertian scenes",
            actual_effect="MAE improved 0.35→0.28 on non-Lambertian",
            verified=1,
        )
        history = mem.get_causal_history(limit=10)
        assert len(history) == 1
        assert "FFT" in history[0]["design_decision"]
        assert history[0]["verified"] == 1

    def test_empty_design_decision_is_noop(self, mem):
        """Empty/null design_decision should not insert."""
        mem.record_causal_chain_entry(cycle=1, design_decision="")
        mem.record_causal_chain_entry(cycle=1, design_decision=None)
        assert mem.get_causal_history() == []

    def test_multiple_entries_ordered_by_cycle_desc(self, mem):
        for cycle in [1, 2, 3]:
            mem.record_causal_chain_entry(
                cycle=cycle, design_decision=f"decision at cycle {cycle}"
            )
        history = mem.get_causal_history()
        assert len(history) == 3
        # Most recent first
        assert history[0]["cycle"] == 3
        assert history[-1]["cycle"] == 1

    def test_long_text_truncated(self, mem):
        """Very long design_decision should be truncated, not crash."""
        long_text = "x" * 1000
        mem.record_causal_chain_entry(cycle=1, design_decision=long_text)
        history = mem.get_causal_history()
        assert len(history[0]["design_decision"]) <= 500


# ═════════════════════════════════════════════════════════════
# 2. experiment_value: write → calibration stats
# ═════════════════════════════════════════════════════════════
class TestExperimentValue:
    def test_write_then_calibration(self, mem):
        """record_experiment_value writes, get_experiment_calibration reads stats."""
        for cycle, correct in [(1, 1), (2, 0), (3, 1), (4, 1)]:
            mem.record_experiment_value(
                cycle=cycle,
                hypothesis=f"hypothesis {cycle}",
                actual_improvement=0.05 * cycle,
                was_correct=correct,
            )
        cal = mem.get_experiment_calibration()
        assert cal["total_hypotheses"] == 4
        # 3 correct out of 4 = 75%
        assert cal["accuracy"] == 0.75

    def test_empty_hypothesis_is_noop(self, mem):
        mem.record_experiment_value(cycle=1, hypothesis="")
        cal = mem.get_experiment_calibration()
        assert cal["total_hypotheses"] == 0

    def test_calibration_consumed_by_constraint_engine(self, mem):
        """The calibration data should reach StrategyConstraintEngine's threshold.

        constraint_engine checks `if total >= 3` before generating calibration
        rules. Verify we can produce enough data to cross that threshold.
        """
        for i in range(5):
            mem.record_experiment_value(
                cycle=i, hypothesis=f"h{i}", was_correct=1,
            )
        cal = mem.get_experiment_calibration()
        assert cal["total_hypotheses"] >= 3  # crosses the constraint engine threshold


# ═════════════════════════════════════════════════════════════
# 3. code_review_lessons: write → query_memory consumption
# ═════════════════════════════════════════════════════════════
class TestCodeReviewLessons:
    def test_write_then_read(self, mem):
        mem.record_code_review_lesson(
            cycle=5,
            category="bug_pattern",
            pattern="shape_mismatch",
            description="Data loader returned 4 channels but model expects 5",
            fix_suggestion="Add channel padding in __getitem__",
        )
        lessons = mem.get_code_review_lessons()
        assert len(lessons) == 1
        assert "channels" in lessons[0]["description"]
        assert lessons[0]["source"] == "reflect"

    def test_empty_description_is_noop(self, mem):
        mem.record_code_review_lesson(cycle=1, category="x", pattern="y", description="")
        assert mem.get_code_review_lessons() == []

    def test_multiple_lessons_accumulate(self, mem):
        for i in range(3):
            mem.record_code_review_lesson(
                cycle=i, category="insight", pattern=f"p{i}",
                description=f"lesson {i}",
            )
        lessons = mem.get_code_review_lessons()
        assert len(lessons) == 3


# ═════════════════════════════════════════════════════════════
# 4. pareto_matrix: method_name extraction (non-degenerate)
# ═════════════════════════════════════════════════════════════
class TestMethodExtraction:
    """_extract_method_name must produce non-empty method labels so the
    Pareto matrix doesn't collapse to a single key."""

    def test_extracts_known_keyword(self):
        from core.loop import ResearchLoop
        result = ResearchLoop._extract_method_name(
            None,
            {"task": "Add FFT feature engineering", "hypothesis": "frequency helps"},
        )
        assert result == "fft"

    def test_extracts_from_hypothesis(self):
        from core.loop import ResearchLoop
        result = ResearchLoop._extract_method_name(
            None,
            {"task": "train model", "hypothesis": "gradient_boosting ensemble will reduce MAE"},
        )
        assert result == "gradient_boosting"

    def test_fallback_is_experiment_not_empty(self):
        from core.loop import ResearchLoop
        result = ResearchLoop._extract_method_name(
            None,
            {"task": "fix data loader bug", "hypothesis": "typo in channel count"},
        )
        assert result == "experiment"  # NOT empty string
        assert result != ""

    def test_space_replaced_with_underscore(self):
        from core.loop import ResearchLoop
        result = ResearchLoop._extract_method_name(
            None,
            {"task": "implement cost volume approach", "hypothesis": ""},
        )
        assert result == "cost_volume"


# ═════════════════════════════════════════════════════════════
# 5. pareto_matrix: non-degenerate write + read
# ═════════════════════════════════════════════════════════════
class TestParetoNonDegenerate:
    def test_different_methods_produce_matrix(self, mem):
        """Two different methods should produce a 2-key matrix, not collapse."""
        mem.record_pareto_entry(cycle=1, method="fft", domain="lambertian", mae=0.35)
        mem.record_pareto_entry(cycle=2, method="gcd", domain="lambertian", mae=0.30)
        matrix = mem.get_pareto_matrix()
        # get_pareto_matrix returns {method: {domain: best_mae}} directly
        assert "fft" in matrix
        assert "gcd" in matrix
        assert len(matrix) == 2

    def test_empty_method_name_is_degenerate(self, mem):
        """Empty method names collapse the matrix — this is the bug we fixed."""
        mem.record_pareto_entry(cycle=1, method="", domain="x", mae=0.3)
        mem.record_pareto_entry(cycle=2, method="", domain="y", mae=0.4)
        matrix = mem.get_pareto_matrix()
        # With empty method, everything collapses to one key ""
        assert len(matrix) == 1
        assert "" in matrix
