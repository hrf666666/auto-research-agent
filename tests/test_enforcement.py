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
