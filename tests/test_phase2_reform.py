"""Tests for Phase 2 (Reform v21): REFLECT fact-spine fallback + THINK retry.

Core invariant: when REFLECT's LLM produces no parseable JSON, the experiment
is NOT forgotten — its facts survive via the fact spine (Phase 1) and
_derive_factual_milestone. This is the direct fix for V30's amnesia.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def loop_with_mock_memory(tmp_path):
    """A ResearchLoop instance with a mocked MemoryManager.

    We don't run a real loop — we just need the _derive_factual_milestone
    method, which reads from self.memory.get_fact_for_output_dir.
    """
    from core.loop import ResearchLoop

    # Create a minimal ResearchLoop without full __init__ (it needs config,
    # API keys, etc). We bind only what _derive_factual_milestone needs.
    loop = ResearchLoop.__new__(ResearchLoop)
    loop.memory = MagicMock()
    # By default, fact lookup returns None (no record)
    loop.memory.get_fact_for_output_dir.return_value = None
    loop.workspace = tmp_path
    return loop


class TestDeriveFactualMilestone:
    """Test _derive_factual_milestone — the Phase 2b fact-spine fallback."""

    def test_empty_when_no_experiment_ran(self, loop_with_mock_memory):
        """No log_file, no metrics → empty milestone (nothing to record)."""
        result = loop_with_mock_memory._derive_factual_milestone({})
        assert result == ""

    def test_empty_when_no_facts_anywhere(self, loop_with_mock_memory):
        """log_file present but fact_scanner has no record, no metrics → empty."""
        loop_with_mock_memory.memory.get_fact_for_output_dir.return_value = None
        result = loop_with_mock_memory._derive_factual_milestone(
            {"log_file": "/some/path/outputs/exp1/train.log"}
        )
        assert result == ""

    def test_derives_from_fact_spine_record(self, loop_with_mock_memory):
        """Fact spine has a record → milestone derived from it."""
        loop_with_mock_memory.memory.get_fact_for_output_dir.return_value = {
            "best_metric_name": "val_mae",
            "best_metric_value": 0.1440,
            "best_epoch": 38,
            "loss_trend": "decreasing",
            "metrics_json": json.dumps({
                "val_mae": 0.144,
                "mae_lambertian": 0.364,
                "mae_urban": 0.093,
            }),
        }
        result = loop_with_mock_memory._derive_factual_milestone(
            {"log_file": "/proj/outputs/v30/train.log"}
        )
        assert "[FACT]" in result
        assert "val_mae=0.1440" in result
        assert "epoch 38" in result
        assert "decreasing" in result
        assert "mae_lambertian" in result  # per-domain metrics included

    def test_derives_without_epoch(self, loop_with_mock_memory):
        """Fact record without best_epoch → milestone still derived."""
        loop_with_mock_memory.memory.get_fact_for_output_dir.return_value = {
            "best_metric_name": "val_mae_overall",
            "best_metric_value": 0.163,
            "best_epoch": None,
            "loss_trend": "decreasing",
            "metrics_json": json.dumps({"val_mae_overall": 0.163}),
        }
        result = loop_with_mock_memory._derive_factual_milestone(
            {"log_file": "/proj/outputs/exp/train.log"}
        )
        assert "[FACT]" in result
        assert "val_mae_overall=0.1630" in result
        assert "epoch" not in result  # no epoch claimed

    def test_fallback_to_execute_result_metrics(self, loop_with_mock_memory):
        """No fact_spine record, but execute_result has metrics → use those."""
        loop_with_mock_memory.memory.get_fact_for_output_dir.return_value = None
        result = loop_with_mock_memory._derive_factual_milestone({
            "log_file": "/proj/outputs/exp/train.log",
            "final_metrics": {"val_mae": 0.15, "mae_lambertian": 0.37},
        })
        assert "[FACT]" in result
        assert "val_mae=0.1500" in result

    def test_fallback_picks_priority_metric(self, loop_with_mock_memory):
        """When multiple metrics exist, picks by priority order."""
        loop_with_mock_memory.memory.get_fact_for_output_dir.return_value = None
        result = loop_with_mock_memory._derive_factual_milestone({
            "final_metrics": {"rmse": 0.5, "val_mae": 0.15, "mae_overall": 0.2},
        })
        # val_mae has highest priority
        assert "val_mae" in result
        assert "0.1500" in result

    def test_handles_fact_spine_exception_gracefully(self, loop_with_mock_memory):
        """If fact lookup raises, falls back to execute_result, doesn't crash."""
        loop_with_mock_memory.memory.get_fact_for_output_dir.side_effect = RuntimeError("DB locked")
        result = loop_with_mock_memory._derive_factual_milestone({
            "log_file": "/proj/outputs/exp/train.log",
            "final_metrics": {"val_mae": 0.15},
        })
        # Should have fallen back to execute_result, not crashed
        assert "[FACT]" in result
        assert "val_mae" in result


class TestReflectFallback:
    """Test that _reflect uses the fact-spine fallback when LLM fails."""

    def test_reflect_failure_records_factual_milestone(self, loop_with_mock_memory, monkeypatch):
        """When REFLECT LLM returns empty shell, milestone is derived from facts.

        This is THE test for V30's amnesia fix: REFLECT fails → facts still recorded.
        """
        loop = loop_with_mock_memory

        # Mock dispatch_leader to simulate REFLECT failure (empty shell,
        # exactly what _parse_leader_response returns on unparseable input)
        mock_dispatcher = MagicMock()
        mock_dispatcher.dispatch_leader.return_value = {
            "milestone": "",
            "decision": "Reflect failed",
            "dead_end": None,
            "active_problem": None,
        }
        loop.dispatcher = mock_dispatcher
        loop.context_pruner = MagicMock()
        loop.context_pruner.prune.side_effect = lambda ctx, tier: ctx
        loop.cycle_count = 5

        # Fact spine has the V30 record
        loop.memory.get_fact_for_output_dir.return_value = {
            "best_metric_name": "val_mae",
            "best_metric_value": 0.144,
            "best_epoch": 38,
            "loss_trend": "decreasing",
            "metrics_json": json.dumps({"val_mae": 0.144}),
        }
        loop.memory.get_brief.return_value = ""
        loop.memory.get_log.return_value = ""

        # Run _reflect with a completed experiment
        execute_result = {
            "log_file": "/proj/outputs/v30/train.log",
            "experiment_launched": True,
        }
        result = loop._reflect(execute_result, verify_report=None)

        # THE INVARIANT: even though REFLECT LLM failed, milestone is non-empty
        assert result["milestone"] != "", (
            "REFLECT failure must not result in empty milestone — "
            "facts should survive via fact spine"
        )
        assert "[FACT]" in result["milestone"]
        assert "val_mae=0.1440" in result["milestone"]
        assert result.get("_milestone_source") == "fact_spine_fallback"

        # Reform v21 step-1 honesty invariant: a fallback MUST mark itself as a
        # DEGRADED reflection, never disguise itself as a normal one. The old
        # code used a vague "Recorded from fact spine" decision that masked
        # parser/LLM failures. The decision must now carry [REFLECT-FAILED] so a
        # regression is visible to humans and downstream, not silently papered
        # over. (See test_reflect_parser_fix.py for the success path.)
        assert "[REFLECT-FAILED]" in result.get("decision", ""), (
            "fallback decision must loudly mark itself as degraded reflection, "
            "not disguise a failure as success"
        )
        # And cognitive fields that the LLM didn't produce must be NULL, not
        # carried over from any stale state — a fallback cannot fabricate a
        # dead_end / causal_link it doesn't have.
        assert result.get("dead_end") is None
        assert result.get("causal_link") is None

        # And it was actually logged to memory
        loop.memory.log_milestone.assert_called_once()
        logged_milestone = loop.memory.log_milestone.call_args[0][0]
        assert "[FACT]" in logged_milestone

    def test_reflect_success_uses_llm_milestone(self, loop_with_mock_memory):
        """When REFLECT LLM succeeds, its milestone is used (no fallback)."""
        loop = loop_with_mock_memory
        mock_dispatcher = MagicMock()
        llm_milestone = "V30 energy-guided cost volume achieved breakthrough on urban scenes"
        mock_dispatcher.dispatch_leader.return_value = {
            "milestone": llm_milestone,
            "decision": "Lambertian needs more work",
            "dead_end": None,
            "active_problem": "Lambertian MAE 0.364 far from target 0.16",
        }
        loop.dispatcher = mock_dispatcher
        loop.context_pruner = MagicMock()
        loop.context_pruner.prune.side_effect = lambda ctx, tier: ctx
        loop.cycle_count = 5
        loop.memory.get_brief.return_value = ""
        loop.memory.get_log.return_value = ""

        execute_result = {"log_file": "/proj/outputs/v30/train.log"}
        result = loop._reflect(execute_result, verify_report=None)

        # LLM's milestone is used, NOT the fact-spine fallback
        assert result["milestone"] == llm_milestone
        assert "_milestone_source" not in result
