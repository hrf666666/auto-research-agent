"""Tests for the v20 structural cleanup: verify_summary (P2 compliance),
default provider fix, Anthropic dead-path removal, and model_analyzer
value-judgment deletion.

These tests pin the structural changes so regressions are caught immediately.
"""
from __future__ import annotations

import inspect
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from core.agents import AgentDispatcher
from core.tools import ToolRegistry
from core.verifier import VerifyCheck, VerifyReport


@pytest.fixture
def registry():
    return ToolRegistry(Path("/tmp/_none"), config={})


# ═════════════════════════════════════════════════════════════
# 1. Default provider + Anthropic removal (改动3)
# ═════════════════════════════════════════════════════════════
class TestDefaultProvider:
    def test_default_provider_is_glm(self):
        """Default must be glm_token_plan, not anthropic."""
        sig = inspect.signature(AgentDispatcher.__init__)
        assert sig.parameters["provider"].default == "glm_token_plan"

    def test_default_model_is_auto(self):
        sig = inspect.signature(AgentDispatcher.__init__)
        assert sig.parameters["model"].default == "auto"

    def test_call_anthropic_removed(self):
        """The dead Anthropic code path must be gone."""
        assert not hasattr(AgentDispatcher, "_call_anthropic")

    def test_openai_fallback_still_present(self):
        """OpenAI fallback branch must survive (only Anthropic was removed)."""
        src = inspect.getsource(AgentDispatcher._call_llm)
        assert 'self.provider == "openai"' in src
        # Anthropic branch must be gone
        assert '_call_anthropic' not in src


# ═════════════════════════════════════════════════════════════
# 2. generate_diagnostic removal (改动4b)
# ═════════════════════════════════════════════════════════════
class TestDiagnosticToolRemoved:
    def test_not_in_code_toolkit(self, registry):
        names = [t["name"] for t in registry.get_tools_for("code")]
        assert "generate_diagnostic" not in names

    def test_handler_not_registered(self, registry):
        out = json.loads(registry.execute_tool("generate_diagnostic", {}))
        assert "error" in out  # unknown tool

    def test_fact_tools_survive(self, registry):
        """Fact-extraction tools must still be available."""
        names = [t["name"] for t in registry.get_tools_for("code")]
        assert "analyze_model" in names
        assert "probe_model" in names
        assert "design_ablation" in names

    def test_value_judgment_methods_gone(self, registry):
        """The 4 value-judgment methods must be deleted from the mixin."""
        assert not hasattr(registry, "_analyze_domain_assumptions")
        assert not hasattr(registry, "_diagnose_results_vs_architecture")
        assert not hasattr(registry, "_analyze_idea_architecture_alignment")
        assert not hasattr(registry, "_analyze_decoder_adequacy")

    def test_fact_extraction_methods_survive(self, registry):
        """Fact-extraction methods must still be present."""
        assert hasattr(registry, "_analyze_model_ast")
        assert hasattr(registry, "_analyze_data_flow")
        assert hasattr(registry, "_detect_information_bottlenecks")
        assert hasattr(registry, "_analyze_gradient_paths")
        assert hasattr(registry, "_analyze_structural_soundness")


# ═════════════════════════════════════════════════════════════
# 3. _verify_summary (P2 compliance — 改动2)
# ═════════════════════════════════════════════════════════════
class TestVerifySummary:
    """The full 30+ check verify_report must NOT be injected into context.
    Only a compact summary reaches the LLM (P2: query, not inject)."""

    def test_method_exists(self):
        from core.loop import ResearchLoop
        assert hasattr(ResearchLoop, "_verify_summary")

    def test_empty_report_returns_empty_dict(self):
        from core.loop import ResearchLoop
        # Build a minimal mock — ResearchLoop.__init__ is heavy, so we
        # call the unbound method with a dummy self.
        result = ResearchLoop._verify_summary(None, None)
        assert result == {}

    def test_summary_has_bounded_fields(self):
        from core.loop import ResearchLoop
        report = VerifyReport(cycle=1)
        report.checks = [
            VerifyCheck(name="c1", category="data", status="pass", detail="ok", severity="info"),
            VerifyCheck(name="c2", category="module", status="fail", detail="broken", severity="critical"),
            VerifyCheck(name="c3", category="module", status="warn", detail="iffy", severity="medium"),
        ]
        report.diagnosis = ["d1", "d2", "d3", "d4", "d5"]

        result = ResearchLoop._verify_summary(None, report)
        assert result["passed"] == 1
        assert result["failed"] == 1
        assert result["warned"] == 1
        assert len(result["critical_failures"]) == 1
        assert result["critical_failures"][0] == "broken"
        # Top diagnoses capped at 3
        assert len(result["top_diagnoses"]) == 3

    def test_no_checks_list_in_summary(self):
        """The summary must NOT contain the full checks list."""
        from core.loop import ResearchLoop
        report = VerifyReport(cycle=1)
        report.checks = [
            VerifyCheck(name=f"c{i}", category="test", status="pass", detail="x", severity="info")
            for i in range(30)
        ]
        result = ResearchLoop._verify_summary(None, report)
        assert "checks" not in result
        assert "total_checks" not in result


# ═════════════════════════════════════════════════════════════
# 4. Orphan construction removal (改动1)
# ═════════════════════════════════════════════════════════════
class TestOrphanRemoval:
    def test_obsidian_not_imported_in_loop(self):
        import core.loop as loop_mod
        src = inspect.getsource(loop_mod)
        assert "from .obsidian import" not in src
        assert "ObsidianExporter" not in src

    def test_strategy_engine_not_constructed_in_loop(self):
        """StrategyConstraintEngine is used in tools.py (its own __new__),
        not constructed in loop.py."""
        import core.loop as loop_mod
        src = inspect.getsource(loop_mod)
        assert "StrategyConstraintEngine" not in src
        assert "self.strategy_engine" not in src

    def test_context_pruner_still_in_loop(self):
        import core.loop as loop_mod
        src = inspect.getsource(loop_mod)
        assert "ContextPruner" in src
        assert "self.context_pruner" in src

    def test_obsidian_file_still_exists(self):
        """obsidian.py has a standalone CLI — only the loop mount was removed."""
        assert Path("core/obsidian.py").exists()


# ═════════════════════════════════════════════════════════════
# 5. context_keys cleanup (改动5)
# ═════════════════════════════════════════════════════════════
class TestContextKeysClean:
    def test_no_broken_comment_residue(self):
        """Comment lines should not contain dangling tier=2), fragments.

        Legitimate 'tier=2),' appears at end of ContextKey(...) definition lines.
        Broken residue appears INSIDE comment lines (# ...) — those are what we
        check for here.
        """
        import core.context_keys as ck
        for line in inspect.getsource(ck).splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                # Comment lines must not have dangling tier=N), fragments
                assert "tier=" not in stripped, f"Broken comment residue: {stripped}"
                assert "PREVIOUSLY DROPPED" not in stripped, f"Stale comment: {stripped}"
        # The removed dead keys must not appear anywhere
        full_src = inspect.getsource(ck)
        assert "experiment_evaluation" not in full_src

    def test_reflect_keys_still_registered(self):
        from core.context_keys import get_keys_for_phase
        reflect_names = [k.name for k in get_keys_for_phase("reflect")]
        assert "experiment_result" in reflect_names
        assert "verify_diagnosis" in reflect_names
        assert "verify_failed_modules" in reflect_names
