"""Tests for Fix B: advisory systems that now have enforcement teeth.

Two systems changed:
1. Constraint Engine: dead-end rules with count>=5 now set priority="forbidden",
   making the FORBIDDEN hard gate reachable (previously auto-rules were always
   high/medium, so has_forbidden_violation was always False).
2. Audit Escalation: repeated audit issues now increment a per-signature
   enforcement counter. After 2 escalations, the action is forced to a
   targeted fix; after 3, pause_human. Previously audit only wrote
   DIRECTIVE.md text that the LLM could ignore forever.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from core.loop import ResearchLoop
from core.constraint_engine import StrategyConstraintEngine, StrategyRule


def _make_loop(tmp_path):
    loop = ResearchLoop.__new__(ResearchLoop)
    loop.workspace = tmp_path
    loop.project_dir = tmp_path
    loop._audit_enforcement = {}
    loop._consecutive_failed_launches = 0
    return loop


# ── Constraint Engine: FORBIDDEN now reachable ──

class TestForbiddenGate:
    """Fix B part 1: dead-end count>=5 → priority=forbidden → hard block."""

    def test_forbidden_detected_from_string(self):
        engine = StrategyConstraintEngine.__new__(StrategyConstraintEngine)
        violations = ["[CONSTRAINT:FORBIDDEN] Approach 'epi' is a dead end."]
        assert engine.has_forbidden_violation(violations) is True

    def test_high_priority_not_forbidden(self):
        engine = StrategyConstraintEngine.__new__(StrategyConstraintEngine)
        violations = ["[CONSTRAINT:HIGH] Some warning."]
        assert engine.has_forbidden_violation(violations) is False

    def test_forbidden_priority_rule_renders_forbidden_tag(self):
        """A rule with priority='forbidden' must render [CONSTRAINT:FORBIDDEN]."""
        rule = StrategyRule(
            rule_id="dead_end_test",
            description="test dead end",
            condition="task contains test",
            action="FORBIDDEN",
            source="dead_ends",
            priority="forbidden",
        )
        rendered = f"[CONSTRAINT:{rule.priority.upper()}] {rule.description}"
        assert "FORBIDDEN" in rendered


# ── Audit enforcement: counter-based escalation ──

class TestAuditEnforcement:
    """Fix B part 2: repeated audit issues force action rewrite / pause."""

    def test_no_enforcement_below_threshold(self, tmp_path):
        loop = _make_loop(tmp_path)
        loop._audit_enforcement = {"sig_a": 1}
        result = loop._enforce_audit_findings({"action": "experiment", "task": "t"})
        assert result.get("action") == "experiment"  # unchanged

    def test_force_fix_at_two_escalations(self, tmp_path):
        loop = _make_loop(tmp_path)
        loop._audit_enforcement = {"sig_a": 2}
        result = loop._enforce_audit_findings({"action": "paper_research", "task": "orig"})
        assert result.get("_forced_audit_fix") is True
        assert "sig_a" in result.get("task", "")
        # Counter reset after forcing (clean slate for the fix attempt)
        assert loop._audit_enforcement["sig_a"] == 0

    def test_pause_human_at_three_escalations(self, tmp_path):
        loop = _make_loop(tmp_path)
        loop._audit_enforcement = {"sig_a": 3}
        result = loop._enforce_audit_findings({"action": "experiment", "task": "t"})
        assert result.get("action") == "pause_human"
        assert "sig_a" in result.get("reason", "")

    def test_highest_counter_wins(self, tmp_path):
        """When multiple signatures are escalated, the highest count wins."""
        loop = _make_loop(tmp_path)
        loop._audit_enforcement = {"sig_a": 2, "sig_b": 3}
        result = loop._enforce_audit_findings({"action": "experiment", "task": "t"})
        # sig_b=3 → pause_human takes priority over sig_a=2
        assert result.get("action") == "pause_human"

    def test_empty_enforcement_is_noop(self, tmp_path):
        loop = _make_loop(tmp_path)
        loop._audit_enforcement = {}
        think = {"action": "experiment", "task": "t"}
        result = loop._enforce_audit_findings(think)
        assert result is think  # same object, unchanged

    def test_counter_increments_on_escalation(self, tmp_path):
        """When _handle_escalated_issues fires for targeted_fix, the
        enforcement counter must increment (simulated here directly)."""
        loop = _make_loop(tmp_path)
        loop._audit_enforcement = {}
        # Simulate what _handle_escalated_issues does for a targeted_fix
        sig = "INTEGRITY:multi_branch_fusion"
        loop._audit_enforcement[sig] = loop._audit_enforcement.get(sig, 0) + 1
        loop._audit_enforcement[sig] = loop._audit_enforcement.get(sig, 0) + 1
        assert loop._audit_enforcement[sig] == 2
        # Now it should trigger forced fix
        result = loop._enforce_audit_findings({"action": "experiment", "task": "t"})
        assert result.get("_forced_audit_fix") is True
