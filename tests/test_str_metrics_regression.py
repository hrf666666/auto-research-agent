"""Regression test: metrics values as strings (the bug found in 10-cycle run).

Root cause: monitor._extract_metrics historically did metrics[k] = str(v),
converting floats to strings. This broke:
  - falsification gate: str < float TypeError
  - _derive_factual_milestone: :.4f format on str ValueError

Fix: monitor keeps floats + consumers coerce with float(). These tests
ensure the fix holds even if str values leak through from any path.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.methodology_gates import evaluate_falsification, _find_metric_value


class TestStringMetricsRegression:
    """metrics dict with STRING values must not crash any gate."""

    def test_falsification_with_str_metrics(self):
        """val_MAE < 0.15 where metric value is '0.10' (str) → must work."""
        metrics = {"val_mae": "0.10"}  # str, not float!
        r = evaluate_falsification("val_MAE < 0.15", metrics)
        assert r.parseable is True
        assert r.criteria_met is True
        assert r.actual_value == 0.10

    def test_falsification_with_str_metrics_not_met(self):
        """val_MAE < 0.15 where metric value is '0.20' (str) → NOT MET."""
        r = evaluate_falsification("val_MAE < 0.15", {"val_mae": "0.20"})
        assert r.criteria_met is False

    def test_find_metric_value_coerces_str(self):
        """_find_metric_value must return float even for str input."""
        assert _find_metric_value("val_mae", {"val_mae": "0.15"}) == 0.15
        assert _find_metric_value("val_mae", {"val_mae": 0.15}) == 0.15
        assert _find_metric_value("val_mae", {"val_mae": 0.15}) == _find_metric_value(
            "val_mae", {"val_mae": "0.15"}
        )

    def test_find_metric_value_handles_garbage(self):
        """Non-numeric string → None (not crash)."""
        assert _find_metric_value("val_mae", {"val_mae": "N/A"}) is None
        assert _find_metric_value("val_mae", {"val_mae": None}) is None

    def test_v30_scenario_with_str_metrics(self):
        """The EXACT scenario that crashed in the 10-cycle run:
        Lambertian_MAE < 0.16 with str metrics → must evaluate, not crash."""
        metrics = {
            "val_mae": "0.144",
            "mae_lambertian": "0.364",
            "mae_urban": "0.093",
        }
        r = evaluate_falsification("Lambertian_MAE < 0.16", metrics)
        assert r.parseable is True
        assert r.criteria_met is False
        assert r.actual_value == 0.364


class TestDeriveMilestoneStrMetrics:
    """_derive_factual_milestone must handle str metrics in execute_result fallback."""

    def test_str_metric_in_execute_result_fallback(self):
        from core.loop import ResearchLoop

        loop = ResearchLoop.__new__(ResearchLoop)
        loop.memory = MagicMock()
        loop.memory.get_fact_for_output_dir.return_value = None
        loop.workspace = Path("/tmp")

        # execute_result with STRING metrics (as monitor produced them)
        result = loop._derive_factual_milestone({
            "final_metrics": {"val_mae": "0.151899"},  # str!
        })
        assert "[FACT]" in result
        assert "0.1519" in result  # formatted as float, not crashed

    def test_float_metric_still_works(self):
        from core.loop import ResearchLoop

        loop = ResearchLoop.__new__(ResearchLoop)
        loop.memory = MagicMock()
        loop.memory.get_fact_for_output_dir.return_value = None
        loop.workspace = Path("/tmp")

        result = loop._derive_factual_milestone({
            "final_metrics": {"val_mae": 0.15},  # float
        })
        assert "[FACT]" in result
        assert "0.1500" in result
