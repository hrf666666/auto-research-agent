"""Tests for the explore_citations tool wiring and logic (P1: tool-as-contract).

Covers:
  - Tool registration (schema, researcher toolkit, handler dispatch)
  - Seed resolution (all accepted forms + failure modes)
  - Inverted-abstract reconstruction (stability, edge cases)
  - OpenAlex record → compact flattening (DOI normalization, missing fields)
  - Network methods (mocked: _oa_fetch_id 404/error, _oa_fetch_work, _oa_fetch_forward)
  - Full _exec_explore_citations orchestration (backward+forward, empty refs, resolve fail)

All network calls are mocked via monkeypatch so the suite is hermetic.
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


# ─────────────────────────────────────────────────────────────
# Helpers: build realistic OpenAlex work payloads
# ─────────────────────────────────────────────────────────────
def _work(wid: str = "W123", *, title="T", year=2024, refs=None, doi="10.65215/abc", cited=5, abstract=None):
    """Build a minimal-but-realistic OpenAlex work dict."""
    return {
        "id": f"https://openalex.org/{wid}",
        "doi": f"https://doi.org/{doi}" if doi else None,
        "title": title,
        "display_name": title,
        "publication_year": year,
        "primary_location": {"source": {"display_name": "NeurIPS"}},
        "cited_by_count": cited,
        "authorships": [{"author": {"display_name": "Alice"}}, {"author": {"display_name": "Bob"}}],
        "abstract_inverted_index": abstract,
        "referenced_works": refs or [],
        "ids": {"openalex": f"https://openalex.org/{wid}"},
    }


# ═════════════════════════════════════════════════════════════
# 1. Tool registration contract
# ═════════════════════════════════════════════════════════════
class TestToolRegistration:
    def test_schema_exists_and_is_complete(self, registry):
        schema = registry._tool_explore_citations
        assert schema["name"] == "explore_citations"
        props = schema["input_schema"]["properties"]
        assert "seed" in props
        assert "per_direction" in props
        assert schema["input_schema"]["required"] == ["seed"]
        # Description must convey what makes this tool different from search
        assert "citation graph" in schema["description"].lower()

    def test_researcher_gets_the_tool(self, registry):
        names = [t["name"] for t in registry.get_tools_for("researcher")]
        assert "explore_citations" in names

    def test_leader_does_not_get_search_tool_directly(self, registry):
        """Leader delegates to researcher; it doesn't call explore_citations itself."""
        names = [t["name"] for t in registry.get_tools_for("leader") if isinstance(t, dict)]
        assert "explore_citations" not in names

    def test_handler_dispatch_routes_correctly(self, registry):
        out = json.loads(registry.execute_tool("explore_citations", {"seed": ""}))
        assert "error" in out

    def test_unknown_tool_still_errors(self, registry):
        out = json.loads(registry.execute_tool("not_a_real_tool", {}))
        assert "error" in out

    def test_list_args_handled_gracefully(self, registry):
        """execute_tool guards against args-as-list, a common LLM mistake."""
        out = json.loads(registry.execute_tool("explore_citations", [{"seed": "W1"}]))
        assert "error" in out  # not a crash


# ═════════════════════════════════════════════════════════════
# 2. Seed resolution (_oa_resolve_seed)
# ═════════════════════════════════════════════════════════════
class TestSeedResolution:
    def test_bare_openalex_id_passes_through(self, registry):
        assert registry._oa_resolve_seed("W2626778328") == "W2626778328"

    def test_openalex_url_strips_to_id(self, registry):
        assert registry._oa_resolve_seed("https://openalex.org/W2626778328") == "W2626778328"

    def test_openalex_url_with_trailing_slash(self, registry):
        assert registry._oa_resolve_seed("https://openalex.org/W2626778328/") == "W2626778328"

    def test_empty_seed_returns_none(self, registry):
        assert registry._oa_resolve_seed("") is None
        assert registry._oa_resolve_seed("   ") is None
        assert registry._oa_resolve_seed(None) is None

    def test_doi_resolves_via_network(self, registry):
        """Real DOI resolution hits the network — patch _oa_fetch_id."""
        with patch.object(registry, "_oa_fetch_id", return_value="W999") as m:
            assert registry._oa_resolve_seed("10.65215/abc") == "W999"
            m.assert_called_once_with("doi:10.65215/abc")

    def test_doi_with_prefix(self, registry):
        with patch.object(registry, "_oa_fetch_id", return_value="W999"):
            assert registry._oa_resolve_seed("doi:10.65215/abc") == "W999"

    def test_doi_url_form(self, registry):
        with patch.object(registry, "_oa_fetch_id", return_value="W999") as m:
            assert registry._oa_resolve_seed("https://doi.org/10.65215/abc") == "W999"
            m.assert_called_once_with("doi:10.65215/abc")

    def test_arxiv_resolves_via_datacite_doi(self, registry):
        with patch.object(registry, "_oa_fetch_id", return_value="W4390723197") as m:
            assert registry._oa_resolve_seed("arXiv:2401.04088") == "W4390723197"
            m.assert_called_once_with("doi:10.48550/arXiv.2401.04088")

    def test_arxiv_url_form(self, registry):
        with patch.object(registry, "_oa_fetch_id", return_value="W1"):
            assert registry._oa_resolve_seed("https://arxiv.org/abs/2401.04088") == "W1"

    def test_arxiv_pdf_url_form(self, registry):
        with patch.object(registry, "_oa_fetch_id", return_value="W1") as m:
            assert registry._oa_resolve_seed("https://arxiv.org/pdf/2401.04088") == "W1"
            m.assert_called_once_with("doi:10.48550/arXiv.2401.04088")

    def test_arxiv_strips_pdf_extension(self, registry):
        with patch.object(registry, "_oa_fetch_id", return_value="W1") as m:
            registry._oa_resolve_seed("https://arxiv.org/pdf/2401.04088.pdf")
            assert "2401.04088.pdf" not in m.call_args[0][0]

    def test_doi_resolution_failure_returns_none(self, registry):
        with patch.object(registry, "_oa_fetch_id", return_value=None):
            assert registry._oa_resolve_seed("10.xxx/nonexistent") is None

    def test_arxiv_resolution_failure_returns_none(self, registry):
        with patch.object(registry, "_oa_fetch_id", return_value=None):
            assert registry._oa_resolve_seed("arXiv:1706.03762") is None

    def test_garbage_input_returns_none(self, registry):
        assert registry._oa_resolve_seed("garbage_id_xyz") is None


# ═════════════════════════════════════════════════════════════
# 3. _oa_fetch_id (network, mocked at urllib level)
# ═════════════════════════════════════════════════════════════
class TestFetchId:
    def test_returns_wid_on_success(self, registry):
        fake_resp = {"id": "https://openalex.org/W12345"}
        with patch("urllib.request.urlopen") as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(fake_resp).encode()
            assert registry._oa_fetch_id("doi:10.xxx") == "W12345"

    def test_returns_none_on_404(self, registry):
        import urllib.error
        err = urllib.error.HTTPError("url", 404, "Not Found", {}, None)
        with patch("urllib.request.urlopen", side_effect=err):
            assert registry._oa_fetch_id("doi:10.xxx") is None

    def test_returns_none_on_500(self, registry):
        import urllib.error
        err = urllib.error.HTTPError("url", 500, "Server Error", {}, None)
        with patch("urllib.request.urlopen", side_effect=err):
            assert registry._oa_fetch_id("doi:10.xxx") is None

    def test_returns_none_on_network_error(self, registry):
        with patch("urllib.request.urlopen", side_effect=OSError("timeout")):
            assert registry._oa_fetch_id("doi:10.xxx") is None

    def test_returns_none_on_non_w_id_payload(self, registry):
        """If the API returns a malformed id, reject it."""
        fake_resp = {"id": "not-a-wid"}
        with patch("urllib.request.urlopen") as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(fake_resp).encode()
            assert registry._oa_fetch_id("doi:10.xxx") is None


# ═════════════════════════════════════════════════════════════
# 4. _oa_fetch_work (network, mocked)
# ═════════════════════════════════════════════════════════════
class TestFetchWork:
    def test_returns_work_dict_on_success(self, registry):
        work = _work("W1")
        with patch("urllib.request.urlopen") as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(work).encode()
            result = registry._oa_fetch_work("W1")
            assert result == work

    def test_strips_url_prefix_from_id(self, registry):
        with patch("urllib.request.urlopen") as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = b'{}'
            registry._oa_fetch_work("https://openalex.org/W123")
            called_url = mock_open.call_args[0][0].full_url
            assert "/W123?" in called_url

    def test_returns_none_on_network_error(self, registry):
        with patch("urllib.request.urlopen", side_effect=OSError("timeout")):
            assert registry._oa_fetch_work("W1") is None


# ═════════════════════════════════════════════════════════════
# 5. _oa_fetch_forward (network, mocked)
# ═════════════════════════════════════════════════════════════
class TestFetchForward:
    def test_returns_results_list_on_success(self, registry):
        payload = {"results": [_work("W1"), _work("W2")]}
        with patch("urllib.request.urlopen") as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(payload).encode()
            result = registry._oa_fetch_forward("W123", 5)
            assert len(result) == 2

    def test_returns_empty_on_network_error(self, registry):
        with patch("urllib.request.urlopen", side_effect=OSError("timeout")):
            assert registry._oa_fetch_forward("W123", 5) == []

    def test_returns_empty_on_no_results(self, registry):
        with patch("urllib.request.urlopen") as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = b'{"results": []}'
            assert registry._oa_fetch_forward("W123", 5) == []

    def test_clamps_limit_to_range(self, registry):
        """per_direction is clamped to [1, 10]."""
        with patch("urllib.request.urlopen") as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = b'{"results": []}'
            registry._oa_fetch_forward("W1", 0)   # too small
            registry._oa_fetch_forward("W1", 99)  # too large
            assert mock_open.call_count == 2


# ═════════════════════════════════════════════════════════════
# 6. Inverted-abstract reconstruction
# ═════════════════════════════════════════════════════════════
class TestInvertAbstract:
    def test_reconstructs_simple_abstract(self, registry):
        inv = {"Attention": [0], "is": [1], "all": [2], "you": [3], "need": [4]}
        assert registry._invert_abstract(inv) == "Attention is all you need"

    def test_none_or_empty_returns_empty(self, registry):
        assert registry._invert_abstract(None) == ""
        assert registry._invert_abstract({}) == ""

    def test_non_dict_returns_empty(self, registry):
        assert registry._invert_abstract("not a dict") == ""
        assert registry._invert_abstract([1, 2, 3]) == ""

    def test_stable_sort_preserves_same_position_order(self, registry):
        """Multiple words at the same position keep insertion order (not alphabetical)."""
        # 'cats' and 'dogs' both at position 0 — insertion order must be preserved
        inv = {"cats": [0], "dogs": [0], "are": [1], "great": [2]}
        result = registry._invert_abstract(inv)
        # 'cats' was inserted first, must come before 'dogs'
        assert result.index("cats") < result.index("dogs")


# ═════════════════════════════════════════════════════════════
# 7. _oa_work_to_compact (record flattening)
# ═════════════════════════════════════════════════════════════
class TestWorkToCompact:
    def test_flattens_complete_work(self, registry):
        work = _work("W123", title="Deep Learning", year=2023, doi="10.65215/abc", cited=42)
        c = registry._oa_work_to_compact(work)
        assert c["openalex_id"] == "W123"
        assert c["title"] == "Deep Learning"
        assert c["year"] == 2023
        assert c["venue"] == "NeurIPS"
        assert c["cited_by_count"] == 42
        assert c["authors"] == ["Alice", "Bob"]
        assert c["doi"] == "10.65215/abc"

    def test_strips_https_doi_prefix(self, registry):
        c = registry._oa_work_to_compact({"doi": "https://doi.org/10.xxx/yyy", "id": "https://openalex.org/W1"})
        assert c["doi"] == "10.xxx/yyy"

    def test_strips_http_doi_prefix(self, registry):
        """FIX5: http (not https) prefix must also be stripped."""
        c = registry._oa_work_to_compact({"doi": "http://doi.org/10.xxx/yyy", "id": "https://openalex.org/W1"})
        assert c["doi"] == "10.xxx/yyy"

    def test_strips_dx_doi_prefix(self, registry):
        c = registry._oa_work_to_compact({"doi": "http://dx.doi.org/10.xxx/yyy", "id": "https://openalex.org/W1"})
        assert c["doi"] == "10.xxx/yyy"

    def test_handles_missing_fields(self, registry):
        c = registry._oa_work_to_compact({})
        assert c["title"] == ""
        assert c["year"] is None
        assert c["venue"] == ""
        assert c["cited_by_count"] == 0
        assert c["authors"] == []
        assert c["abstract"] == ""

    def test_truncates_long_abstract(self, registry):
        inv = {"word": list(range(2000))}  # very long
        c = registry._oa_work_to_compact({"abstract_inverted_index": inv, "id": "https://openalex.org/W1"})
        assert len(c["abstract"]) <= 1500

    def test_limits_authors_to_eight(self, registry):
        work = {"authorships": [{"author": {"display_name": f"A{i}"}} for i in range(20)], "id": "https://openalex.org/W1"}
        c = registry._oa_work_to_compact(work)
        assert len(c["authors"]) == 8


# ═════════════════════════════════════════════════════════════
# 8. _exec_explore_citations (full orchestration, all mocked)
# ═════════════════════════════════════════════════════════════
class TestExecExploreCitations:
    def test_empty_seed_returns_graceful_error(self, registry):
        out = json.loads(registry._exec_explore_citations(""))
        assert "error" in out

    def test_unresolvable_seed_returns_error(self, registry):
        with patch.object(registry, "_oa_resolve_seed", return_value=None):
            out = json.loads(registry._exec_explore_citations("garbage"))
            assert "error" in out
            assert "resolve" in out["error"].lower() or "Could not" in out["error"]

    def test_seed_fetch_failure_returns_error(self, registry):
        with patch.object(registry, "_oa_resolve_seed", return_value="W1"), \
             patch.object(registry, "_oa_fetch_work", return_value=None):
            out = json.loads(registry._exec_explore_citations("W1"))
            assert "error" in out

    def test_full_success_backward_and_forward(self, registry):
        """Seed has references (backward) + papers cite it (forward)."""
        seed = _work("W1", refs=["https://openalex.org/W2", "https://openalex.org/W3"])
        backward1 = _work("W2", title="Old Paper")
        backward2 = _work("W3", title="Older Paper")
        forward1 = _work("W4", title="New Paper")
        work_map = {"W1": seed, "W2": backward1, "W3": backward2}

        def fake_fetch_work(wid):
            return work_map.get(wid.split("/")[-1])

        with patch.object(registry, "_oa_resolve_seed", return_value="W1"), \
             patch.object(registry, "_oa_fetch_work", side_effect=fake_fetch_work), \
             patch.object(registry, "_oa_fetch_forward", return_value=[forward1]):
            out = json.loads(registry._exec_explore_citations("W1", per_direction=5))
            assert out["seed"]["title"] == "T"
            assert len(out["backward"]) == 2
            assert out["backward"][0]["title"] == "Old Paper"
            assert len(out["forward"]) == 1
            assert out["forward"][0]["title"] == "New Paper"
            assert out["counts"] == {"backward": 2, "forward": 1}
            assert out["backward_note"] == ""  # refs existed

    def test_empty_references_sets_backward_note(self, registry):
        """FIX6: when seed has no referenced_works, explain why backward is empty."""
        seed = _work("W1", refs=[])  # preprint with no refs in OpenAlex
        with patch.object(registry, "_oa_resolve_seed", return_value="W1"), \
             patch.object(registry, "_oa_fetch_work", return_value=seed), \
             patch.object(registry, "_oa_fetch_forward", return_value=[]):
            out = json.loads(registry._exec_explore_citations("W1"))
            assert out["backward"] == []
            assert "referenced_works" in out["backward_note"]
            assert "preprint" in out["backward_note"]

    def test_per_direction_string_does_not_crash(self, registry):
        """FIX3: LLM may send per_direction as a string; must not raise ValueError."""
        seed = _work("W1", refs=[])
        with patch.object(registry, "_oa_resolve_seed", return_value="W1"), \
             patch.object(registry, "_oa_fetch_work", return_value=seed), \
             patch.object(registry, "_oa_fetch_forward", return_value=[]):
            out = json.loads(registry._exec_explore_citations("W1", per_direction="five"))
            assert "error" not in out  # fell back to default, did not crash

    def test_per_direction_negative_clamped(self, registry):
        seed = _work("W1", refs=[])
        with patch.object(registry, "_oa_resolve_seed", return_value="W1"), \
             patch.object(registry, "_oa_fetch_work", return_value=seed), \
             patch.object(registry, "_oa_fetch_forward", return_value=[]):
            out = json.loads(registry._exec_explore_citations("W1", per_direction=-3))
            assert "error" not in out

    def test_backward_filters_non_w_ref_urls(self, registry):
        """Malformed ref URLs that don't match W\\d+ are skipped."""
        seed = _work("W1", refs=["https://openalex.org/W2", "https://example.com/garbage"])
        work_map = {"W1": seed, "W2": _work("W2", title="Ref")}

        def fake_fetch_work(wid):
            return work_map.get(wid.split("/")[-1])

        with patch.object(registry, "_oa_resolve_seed", return_value="W1"), \
             patch.object(registry, "_oa_fetch_work", side_effect=fake_fetch_work), \
             patch.object(registry, "_oa_fetch_forward", return_value=[]):
            out = json.loads(registry._exec_explore_citations("W1"))
            # Only W2 should be fetched; example.com/garbage filtered out
            assert len(out["backward"]) == 1
            assert out["backward"][0]["openalex_id"] == "W2"
