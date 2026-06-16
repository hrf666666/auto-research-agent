"""Tests for the Signal Arbitration System (v18).

The arbiter is the single decision point that replaces 8+ independent signal
injection paths. Tests cover:
1. CRITICAL signals → forced_action (bypasses LLM)
2. Budget allocation (total context <= budget_chars)
3. Signal deduplication
4. Deferred signal persistence (backlog survives to next cycle)
5. tier-1 keys always included regardless of budget
"""
from __future__ import annotations

import pytest

from core.signal_arbiter import SignalArbiter, Signal, CycleDirective


class TestForcedAction:
    """CRITICAL signals from enforcement subsystems produce forced_action."""

    def test_critical_launch_failure_forces_fix(self):
        arb = SignalArbiter(budget_chars=10000)
        arb.begin_cycle()
        arb.add_signal("launch", "research_roadmap", "roadmap",
                       severity="CRITICAL", forced_action="forced_fix",
                       forced_task="Must launch now", forced_reason="2 failures")
        d = arb.arbitrate("think")
        assert d.forced_action == "forced_fix"
        assert "Must launch now" in d.forced_task

    def test_critical_pause_human_overrides_forced_fix(self):
        """When multiple CRITICAL signals fire, pause_human wins."""
        arb = SignalArbiter(budget_chars=10000)
        arb.begin_cycle()
        arb.add_signal("launch", "research_roadmap", "r",
                       severity="CRITICAL", forced_action="forced_fix",
                       forced_task="fix", forced_reason="2 failures")
        arb.add_signal("audit", "phase_focus", "p",
                       severity="CRITICAL", forced_action="pause_human",
                       forced_task="", forced_reason="3 audit escalations")
        d = arb.arbitrate("think")
        assert d.forced_action == "pause_human"

    def test_no_critical_means_no_forced_action(self):
        arb = SignalArbiter(budget_chars=10000)
        arb.begin_cycle()
        arb.add_signal("domain", "domain_knowledge", "DK", severity="WARNING")
        arb.add_signal("memory", "brief", "brief", severity="INFO")
        d = arb.arbitrate("think")
        assert d.forced_action is None

    def test_forced_action_context_is_minimal(self):
        """When a CRITICAL fires, context should be tier-1 only."""
        arb = SignalArbiter(budget_chars=10000)
        arb.begin_cycle()
        arb.add_signal("core", "brief", "B")
        arb.add_signal("core", "cycle", 5)
        arb.add_signal("domain", "domain_knowledge", "DK")
        arb.add_signal("launch", "research_roadmap", "r",
                       severity="CRITICAL", forced_action="forced_fix",
                       forced_task="t", forced_reason="r")
        d = arb.arbitrate("think")
        assert "brief" in d.context   # tier-1
        assert "cycle" in d.context   # tier-1
        # domain_knowledge is tier-2, should NOT be in forced context
        assert "domain_knowledge" not in d.context


class TestBudgetAllocation:
    """Context total must not exceed budget_chars."""

    def test_budget_respected(self):
        arb = SignalArbiter(budget_chars=2000)
        arb.begin_cycle()
        arb.add_signal("core", "brief", "B" * 300)
        arb.add_signal("core", "cycle", 1)
        arb.add_signal("core", "workspace_dir", "/w")
        arb.add_signal("core", "memory_log", "")
        arb.add_signal("domain", "domain_knowledge", "D" * 500)
        arb.add_signal("plan", "architecture_plan_summary", "P" * 300)
        arb.add_signal("memory", "pareto_frontier", "PF" * 2000)
        d = arb.arbitrate("think")
        assert d.forced_action is None
        assert d.context_char_count <= 2100  # small overhead allowed

    def test_tier1_always_included_regardless_of_budget(self):
        """Even with a tiny budget, tier-1 keys (brief, cycle, workspace_dir)
        must be in the context."""
        arb = SignalArbiter(budget_chars=50)
        arb.begin_cycle()
        arb.add_signal("core", "brief", "short")
        arb.add_signal("core", "cycle", 1)
        arb.add_signal("core", "workspace_dir", "/w")
        d = arb.arbitrate("think")
        assert "brief" in d.context
        assert "cycle" in d.context

    def test_low_priority_deferred_when_budget_tight(self):
        arb = SignalArbiter(budget_chars=200)
        arb.begin_cycle()
        arb.add_signal("core", "brief", "B" * 150)
        arb.add_signal("domain", "domain_knowledge", "D" * 200)
        d = arb.arbitrate("think")
        # brief is tier-1, always included. domain_knowledge is tier-2, deferred.
        assert "brief" in d.context
        assert len(d.deferred) >= 1


class TestDedup:
    """Same source+key signals merge, keeping highest severity."""

    def test_duplicate_merged(self):
        arb = SignalArbiter(budget_chars=10000)
        arb.begin_cycle()
        arb.add_signal("audit", "phase_focus", "v1", severity="INFO")
        arb.add_signal("audit", "phase_focus", "v2", severity="WARNING")
        d = arb.arbitrate("think")
        # Only one phase_focus in context (deduped)
        assert d.context.get("phase_focus") in ("v1", "v2")

    def test_different_sources_not_merged(self):
        arb = SignalArbiter(budget_chars=10000)
        arb.begin_cycle()
        arb.add_signal("audit", "phase_focus", "audit_focus")
        arb.add_signal("roadmap", "phase_focus", "roadmap_focus")
        d = arb.arbitrate("think")
        # Both are different sources, so both are separate signals
        # but they write to the same context key, so last-write-wins in dict
        assert "phase_focus" in d.context


class TestBacklogPersistence:
    """Deferred signals survive to the next cycle."""

    def test_deferred_survives_to_backlog(self):
        arb = SignalArbiter(budget_chars=100)
        arb.begin_cycle()
        arb.add_signal("core", "brief", "B" * 80)
        arb.add_signal("domain", "domain_knowledge", "D" * 200, severity="WARNING")
        d1 = arb.arbitrate("think")
        assert len(d1.deferred) >= 1
        assert len(arb._backlog) >= 1
        # escalation_count is 0 at first deferral; increments when carried
        # into the next cycle's arbitrate() call.
        assert arb._backlog[0].escalation_count == 0

        # Next cycle: with a larger budget, the backlog signal gets included
        arb2 = SignalArbiter(budget_chars=10000)
        arb2.load_backlog(arb.get_backlog())
        arb2.begin_cycle()
        arb2.add_signal("core", "brief", "B")
        d2 = arb2.arbitrate("think")
        # Signal was consumed from backlog (either included or re-deferred)
        # With 10000 budget it should be included now
        assert "domain_knowledge" in d2.context

    def test_backlog_serialize_deserialize(self):
        """Backlog can be persisted to state.json and restored."""
        arb = SignalArbiter(budget_chars=100)
        arb.begin_cycle()
        arb.add_signal("core", "brief", "B" * 80)
        arb.add_signal("domain", "domain_knowledge", "D" * 200, severity="WARNING")
        arb.arbitrate("think")
        assert len(arb._backlog) >= 1

        # Serialize
        data = arb.get_backlog()
        assert len(data) >= 1
        assert data[0]["key"] == "domain_knowledge"

        # Restore in new arbiter
        arb2 = SignalArbiter(budget_chars=10000)
        arb2.load_backlog(data)
        assert len(arb2._backlog) >= 1

    def test_clear_resolved(self):
        arb = SignalArbiter(budget_chars=100)
        arb.begin_cycle()
        arb.add_signal("core", "brief", "B" * 80)
        arb.add_signal("domain", "domain_knowledge", "D" * 200, severity="WARNING")
        arb.arbitrate("think")
        assert len(arb._backlog) >= 1
        arb.clear_resolved("domain_knowledge")
        assert len(arb._backlog) == 0


class TestMultipleCycles:
    """Multi-cycle simulation with realistic signal patterns."""

    def test_normal_cycle_no_critical(self):
        """A normal cycle with domain knowledge + plan + memory → no forced action."""
        arb = SignalArbiter(budget_chars=8000)
        arb.begin_cycle()
        arb.add_signal("core", "brief", "Depth estimation project")
        arb.add_signal("core", "memory_log", "Cycle 5: trained EPI model, MAE=0.28")
        arb.add_signal("core", "cycle", 5)
        arb.add_signal("core", "workspace_dir", "/proj/workspace")
        arb.add_signal("domain", "domain_knowledge", "EPI is effective for Lambertian scenes")
        arb.add_signal("plan", "architecture_plan_summary", "Use dual-mask EPI decomposition")
        arb.add_signal("memory", "session_stats",
                       {"total_cycles": 5, "experiments_launched": 3, "launch_rate": 0.6})
        arb.add_signal("memory", "hypothesis_calibration", "prior=0.3")
        d = arb.arbitrate("think")
        assert d.forced_action is None
        assert d.context_char_count > 0
        assert d.context_char_count <= 8100

    def test_cycle_with_launch_failure_escalation(self):
        """Simulate 2 failed launches → CRITICAL → forced_fix."""
        arb = SignalArbiter(budget_chars=8000)
        arb.begin_cycle()
        arb.add_signal("core", "brief", "proj")
        arb.add_signal("core", "cycle", 8)
        arb.add_signal("launch", "research_roadmap", "roadmap",
                       severity="CRITICAL", forced_action="forced_fix",
                       forced_task="Launch the experiment NOW",
                       forced_reason="2 consecutive failed launches")
        d = arb.arbitrate("think")
        assert d.forced_action == "forced_fix"
        assert "Launch the experiment" in d.forced_task
