"""Tests for the explore_citations tool wiring (P1: tool-as-contract).

Verifies the tool is correctly registered, exposed to the researcher agent,
and degrades gracefully on bad input. Network calls are mocked so the test
is hermetic.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from core.tools import ToolRegistry


@pytest.fixture
def registry():
    return ToolRegistry(Path("/tmp/_none"), config={})


class TestToolRegistration:
    """The tool must be reachable through every public surface."""

    def test_schema_exists_and_is_complete(self, registry):
        schema = registry._tool_explore_citations
        assert schema["name"] == "explore_citations"
        assert "seed" in schema["input_schema"]["properties"]
        assert schema["input_schema"]["required"] == ["seed"]
        # description must mention what makes this tool different from search
        assert "citation graph" in schema["description"].lower()

    def test_researcher_gets_the_tool(self, registry):
        names = [t["name"] for t in registry.get_tools_for("researcher")]
        assert "explore_citations" in names

    def test_handler_dispatch_resolves(self, registry):
        """execute_tool must route explore_citations to the executor."""
        out = json.loads(registry.execute_tool("explore_citations", {"seed": ""}))
        assert "error" in out  # empty seed → graceful error, not a crash

    def test_unknown_tool_still_errors(self, registry):
        out = json.loads(registry.execute_tool("not_a_real_tool", {}))
        assert "error" in out


class TestSeedResolution:
    """_oa_resolve_seed must normalize every accepted seed form."""

    def test_bare_openalex_id_passes_through(self, registry):
        assert registry._oa_resolve_seed("W2626778328") == "W2626778328"

    def test_openalex_url_strips_to_id(self, registry):
        assert registry._oa_resolve_seed("https://openalex.org/W2626778328") == "W2626778328"

    def test_empty_seed_returns_none(self, registry):
        assert registry._oa_resolve_seed("") is None
        assert registry._oa_resolve_seed("   ") is None


class TestInvertedAbstract:
    """OpenAlex stores abstracts as inverted indexes; we reconstruct them."""

    def test_reconstructs_simple_abstract(self, registry):
        inv = {"Attention": [0], "is": [1], "all": [2], "you": [3], "need": [4]}
        assert registry._invert_abstract(inv) == "Attention is all you need"

    def test_none_or_empty_returns_empty(self, registry):
        assert registry._invert_abstract(None) == ""
        assert registry._invert_abstract({}) == ""

    def test_handles_duplicate_positions(self, registry):
        # Multiple words can occupy the same position slot in OpenAlex
        inv = {"Hello": [0, 2], "world": [1]}
        result = registry._invert_abstract(inv)
        assert "Hello" in result and "world" in result


class TestWorkToCompact:
    """The compact record shape is the contract the LLM consumes."""

    def test_flattens_minimal_work(self, registry):
        work = {
            "id": "https://openalex.org/W123",
            "title": "Test Paper",
            "publication_year": 2024,
            "primary_location": {"source": {"display_name": "NeurIPS"}},
            "cited_by_count": 42,
            "authorships": [{"author": {"display_name": "Alice"}}],
            "abstract_inverted_index": None,
        }
        compact = registry._oa_work_to_compact(work)
        assert compact["openalex_id"] == "W123"
        assert compact["title"] == "Test Paper"
        assert compact["year"] == 2024
        assert compact["venue"] == "NeurIPS"
        assert compact["cited_by_count"] == 42
        assert compact["authors"] == ["Alice"]
        assert compact["abstract"] == ""
