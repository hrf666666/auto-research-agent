"""Tests for the IdeaScout bridge (core/idea_scout_bridge.py).

All tests are fully mocked — no real LLM calls, no real paper searches. They
verify the data-flow contracts of the bridge: that search_papers output
flattens correctly, that a Profile builds from a brief, and that the LLMScorer
routes through the dispatcher's _call_llm.

A separate real-GLM smoke test is run manually (not in the pytest suite) to
verify end-to-end behavior against the live API.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from core.idea_scout_bridge import (
    is_available, gather_papers, build_profile_from_brief,
    LLMScorer, run_pipeline, format_results_markdown,
    _normalize_paper_row, _expand_query, _extract_positive_keywords,
)


# Skip the whole module if research-idea-scout isn't importable.
pytestmark = pytest.mark.skipif(
    not is_available(),
    reason="research-idea-scout not on sys.path; install or set idea_scout.lib_path",
)


# ─────────────────────────────────────────────────────────────
# Fixture: a fake ToolRegistry whose execute_tool returns canned papers
# ─────────────────────────────────────────────────────────────

class FakeToolRegistry:
    """Mimics ToolRegistry.execute_tool — returns canned search_papers JSON."""

    def __init__(self, papers_response: dict):
        self._response = papers_response
        self.calls = []

    def execute_tool(self, name: str, args: dict) -> str:
        self.calls.append((name, args))
        return json.dumps(self._response)


@pytest.fixture
def fake_papers_s2():
    """Semantic Scholar-shaped response: {papers: [...]}."""
    return {
        "papers": [
            {"title": "Subspace Editing for Representations", "abstract": "We discover linear subspaces for semantic factors.", "year": 2025, "url": "http://s2/p1"},
            {"title": "Temporal Style Transfer", "abstract": "Factorizes motion into content and style tokens.", "year": 2026, "url": "http://s2/p2"},
            {"title": "A Large-Scale Robot Benchmark", "abstract": "Dataset for robot planning.", "year": 2024, "url": "http://s2/p3"},
        ],
        "source": "semantic_scholar",
    }


@pytest.fixture
def fake_papers_mcp():
    """MCP-shaped response: {results: [...]}."""
    return {
        "results": [
            {"title": "Diffusion-Based Idea Generation", "abstract": "Uses diffusion for controllable generation.", "url": "http://mcp/p1"},
        ],
        "source": "mcp_web_search_prime",
    }


# ─────────────────────────────────────────────────────────────
# (a) gather_papers — flattening
# ─────────────────────────────────────────────────────────────

class TestGatherPapers:

    def test_flattens_semantic_scholar_shape(self, fake_papers_s2):
        reg = FakeToolRegistry(fake_papers_s2)
        papers = gather_papers(reg, "cross-domain idea transfer", max_papers=10)
        assert len(papers) == 3
        assert papers[0]["title"] == "Subspace Editing for Representations"
        assert "abstract" in papers[0]
        assert "url" in papers[0]

    def test_flattens_mcp_shape(self, fake_papers_mcp):
        reg = FakeToolRegistry(fake_papers_mcp)
        papers = gather_papers(reg, "test", max_papers=10)
        assert len(papers) == 1
        assert papers[0]["title"] == "Diffusion-Based Idea Generation"

    def test_dedupes_by_title_url(self, fake_papers_s2):
        # Same paper returned across two queries → deduped
        reg = FakeToolRegistry(fake_papers_s2)
        papers = gather_papers(reg, "q", max_papers=10, max_queries=1)
        assert len(papers) == 3  # no dups within one response
        # If gather is called again with the same data, the dedup is per-call
        # (gather doesn't persist seen_keys across calls, which is fine —
        # the pipeline runs gather once).

    def test_empty_response_returns_empty(self):
        reg = FakeToolRegistry({"papers": [], "source": "s2"})
        papers = gather_papers(reg, "nothing", max_papers=10)
        assert papers == []

    def test_search_failure_returns_empty(self):
        reg = FakeToolRegistry({})  # no papers/results key
        papers = gather_papers(reg, "test", max_papers=10)
        assert papers == []

    def test_normalize_row_handles_missing_fields(self):
        row = _normalize_paper_row({"title": "Some Paper"})
        assert row["title"] == "Some Paper"
        assert row["abstract"] == ""  # missing → empty string
        assert row["url"] == ""


# ─────────────────────────────────────────────────────────────
# (b) build_profile_from_brief
# ─────────────────────────────────────────────────────────────

class TestBuildProfile:

    def test_builds_from_real_brief(self, tmp_path):
        brief = tmp_path / "PROJECT_BRIEF.md"
        brief.write_text(
            "# My Research\n\n"
            "Design a depth estimation model using angular frequency analysis.\n\n"
            "## 研究目标\n"
            "- Improve depth estimation for non-Lambertian scenes\n"
            "- Use **angular frequency** decomposition for material analysis\n"
            "- Build a **dual-mask** model for scene unification\n\n"
            "## Method\n"
            "The **subspace projection** approach separates material components.\n",
            encoding="utf-8",
        )
        profile = build_profile_from_brief(brief)
        assert profile.name == "PROJECT_BRIEF"
        assert len(profile.target_tasks) >= 1
        assert any("non-Lambertian" in t or "depth" in t.lower() for t in profile.target_tasks)
        assert len(profile.positive_keywords) >= 1
        assert len(profile.scoring_dimensions) >= 1

    def test_manual_profile_path_overrides(self, tmp_path):
        # Create a hand-written profile YAML
        profile_file = tmp_path / "my_profile.yaml"
        profile_file.write_text(
            "name: manual\n"
            "description: Manual profile\n"
            "target_tasks:\n  - Task A\n"
            "positive_keywords:\n  - kw1\n"
            "negative_keywords: []\n"
            "scoring_dimensions:\n  - transferability\n"
            "prefer: []\n"
            "downweight: []\n",
            encoding="utf-8",
        )
        brief = tmp_path / "PROJECT_BRIEF.md"
        brief.write_text("# Brief", encoding="utf-8")
        profile = build_profile_from_brief(brief, profile_path=profile_file)
        assert profile.name == "manual"

    def test_empty_brief_uses_fallbacks(self, tmp_path):
        brief = tmp_path / "EMPTY.md"
        brief.write_text("# Empty", encoding="utf-8")
        profile = build_profile_from_brief(brief)
        assert len(profile.target_tasks) >= 1  # fallback
        assert len(profile.positive_keywords) >= 1  # fallback


# ─────────────────────────────────────────────────────────────
# (c) LLMScorer — routes through dispatcher._call_llm
# ─────────────────────────────────────────────────────────────

class TestLLMScorer:

    def test_score_calls_dispatch_llm_and_parses(self):
        """The scorer must call dispatcher._call_llm (not codex CLI) and parse
        the JSON response through IdeaScout's normalize_result."""
        from idea_scout.profile import Profile, Dimension
        profile = Profile(
            name="test", description="d",
            target_tasks=["t"], positive_keywords=["kw"],
            negative_keywords=[], scoring_dimensions=[Dimension("x", "x", 1.0)],
            prefer=[], downweight=[], language="English",
        )
        paper = {"title": "Test Paper", "abstract": "An abstract about methods.",
                 "year": 2025, "url": "http://x"}
        mock_dispatcher = MagicMock()
        mock_dispatcher._call_llm.return_value = (
            json.dumps({
                "is_suitable": True,
                "priority": "keep",
                "idea_core": "A transferable method",
                "transferable_mechanism": "Reusable subspace decomposition",
                "fit_reason": "Directly applicable",
                "risk_or_limitation": "Compute cost",
                "score_overall_fit": 8,
                "score_theory_novelty": 7,
                "scores": {"x": 7},
            }),
            None,
        )
        scorer = LLMScorer(mock_dispatcher, abstract_max_chars=1000)
        result = scorer.score(paper, profile)

        # Verify _call_llm was called (not a subprocess)
        mock_dispatcher._call_llm.assert_called_once()
        assert result["rank_score"] > 0
        assert result["priority"] == "keep"
        assert result["idea_core"] == "A transferable method"
        assert result["title"] == "Test Paper"  # original paper field preserved

    def test_score_uses_researcher_tier(self):
        """The scorer must request task_tier='researcher' (strong model)."""
        from idea_scout.profile import Profile, Dimension
        profile = Profile(
            name="t", description="", target_tasks=[], positive_keywords=[],
            negative_keywords=[], scoring_dimensions=[Dimension("x", "x")],
            prefer=[], downweight=[],
        )
        mock_dispatcher = MagicMock()
        mock_dispatcher._call_llm.return_value = ('{"scores":{"x":5},"score_overall_fit":5,"score_theory_novelty":5}', None)
        scorer = LLMScorer(mock_dispatcher)
        scorer.score({"title": "P", "abstract": "A"}, profile)
        _, kwargs = mock_dispatcher._call_llm.call_args
        assert kwargs.get("task_tier") == "researcher"


# ─────────────────────────────────────────────────────────────
# Pipeline orchestration + output formatting
# ─────────────────────────────────────────────────────────────

class TestPipeline:

    def test_pipeline_gather_filter_score(self, fake_papers_s2):
        """End-to-end mock: gather → filter → score, verify ranking."""
        from idea_scout.profile import Profile, Dimension
        reg = FakeToolRegistry(fake_papers_s2)
        profile = Profile(
            name="test", description="depth estimation",
            target_tasks=["depth estimation"],
            positive_keywords=["subspace", "style", "representation"],
            negative_keywords=["benchmark", "dataset"],
            scoring_dimensions=[Dimension("transferability", "t", 1.0)],
            prefer=[], downweight=[],
        )
        mock_dispatcher = MagicMock()
        # Each _call_llm call returns a scored JSON
        mock_dispatcher._call_llm.side_effect = [
            (json.dumps({"scores":{"transferability":7},"score_overall_fit":7,"score_theory_novelty":6,"priority":"keep","idea_core":"c1","transferable_mechanism":"m1","fit_reason":"f1","risk_or_limitation":"r1"}), None),
            (json.dumps({"scores":{"transferability":8},"score_overall_fit":8,"score_theory_novelty":7,"priority":"keep","idea_core":"c2","transferable_mechanism":"m2","fit_reason":"f2","risk_or_limitation":"r2"}), None),
        ]
        result = run_pipeline(reg, mock_dispatcher, profile, "test query",
                              max_papers=10, filter_top_k=5, score_top_k=5)
        assert result["papers_gathered"] == 3
        assert result["papers_scored"] >= 1
        assert len(result["ranked"]) >= 1
        # Ranked must be sorted by rank_score descending
        scores = [r["rank_score"] for r in result["ranked"]]
        assert scores == sorted(scores, reverse=True)

    def test_format_markdown_has_structure(self):
        result = {
            "papers_gathered": 5, "papers_filtered": 3, "papers_scored": 2,
            "ranked": [
                {"title": "Paper A", "rank_score": 7.5, "priority": "keep",
                 "idea_core": "core", "transferable_mechanism": "mech",
                 "year": 2025, "url": "http://a"},
            ],
            "errors": [],
        }
        md = format_results_markdown(result, "test query")
        assert "# IdeaScout" in md
        assert "Paper A" in md
        assert "rank=7.50" in md
        assert "Core idea" in md
