"""Bridge between auto_research_agent and the research-idea-scout toolkit.

This module adapts IdeaScout's cross-domain idea-discovery pipeline so it runs
*inside* auto_research_agent's paper_research phase, reusing this agent's own
ProviderRouter (failover + quota cooldown) instead of IdeaScout's hardcoded
codex CLI backend.

Three responsibilities:
  (a) ``gather_papers``  — call this agent's ``search_papers`` tool, flatten the
      ``{papers:[...]}`` wrapper into the per-row list IdeaScout's filter expects.
  (b) ``build_profile_from_brief`` — auto-generate an IdeaScout ``Profile`` from
      PROJECT_BRIEF.md (+ memory dead-ends) so users don't hand-write YAML.
  (c) ``LLMScorer`` — wrap ``AgentDispatcher._call_llm`` so IdeaScout's
      ``make_prompt`` / ``extract_json_object`` / ``normalize_result`` run through
      this agent's provider failover instead of ``codex exec``.

Design principle: IdeaScout is referenced as a library (sys.path), NOT copied.
If ``idea_scout`` cannot be imported, the bridge degrades gracefully and the
caller falls back to the original researcher-agent path.
"""
from __future__ import annotations

import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("autoresearcher.idea_scout")

# ── Locate and import the research-idea-scout package ──
_DEFAULT_SCOUT_PATHS = [
    Path.home() / "code" / "research-idea-scout",
    Path("/home/bigboss/code/research-idea-scout"),
]

_SCOUT_AVAILABLE = False
try:
    from idea_scout.profile import (  # type: ignore
        Profile, Dimension, load_profile, profile_to_prompt_block,
    )
    from idea_scout.filter_candidates import filter_rows, score_rule_based  # type: ignore
    from idea_scout.codex_idea_score import (  # type: ignore
        make_prompt, extract_json_object, normalize_result,
    )
    from idea_scout.io_utils import clean_text  # type: ignore
    _SCOUT_AVAILABLE = True
except ImportError:
    # Try adding the scout path to sys.path and retry once.
    for _p in _DEFAULT_SCOUT_PATHS:
        if _p.is_dir() and str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
            break
    try:
        from idea_scout.profile import (  # type: ignore
            Profile, Dimension, load_profile, profile_to_prompt_block,
        )
        from idea_scout.filter_candidates import filter_rows, score_rule_based  # type: ignore
        from idea_scout.codex_idea_score import (  # type: ignore
            make_prompt, extract_json_object, normalize_result,
        )
        from idea_scout.io_utils import clean_text  # type: ignore
        _SCOUT_AVAILABLE = True
    except ImportError:
        logger.info(
            "research-idea-scout not found on sys.path; idea_scout integration "
            "disabled. Set idea_scout.lib_path in config or install the package."
        )


def is_available() -> bool:
    """True iff the research-idea-scout package is importable."""
    return _SCOUT_AVAILABLE


# ─────────────────────────────────────────────────────────────────
# (a) Paper gathering — flatten search_papers output to IdeaScout rows
# ─────────────────────────────────────────────────────────────────

def gather_papers(tools_registry, query: str, max_papers: int = 50,
                  max_queries: int = 3) -> list[dict]:
    """Call this agent's search_papers tool and flatten to IdeaScout row dicts.

    ``search_papers`` returns a JSON string shaped either ``{papers:[...]}``
    (Semantic Scholar / DuckDuckGo) or ``{results:[...]}`` (MCP). We normalize
    both to a flat list of ``{title, abstract, year, venue, url}`` dicts — the
    minimum schema IdeaScout's filter expects.

    Venue is often absent from search results (only get_paper returns it); we
    leave it empty rather than issuing N extra get_paper calls (cost), since
    filter_candidates tolerates a missing venue (paper_key falls back to sha1).
    """
    rows: list[dict] = []
    seen_keys: set[str] = set()

    # search_papers accepts a single query; we try a few reformulations if the
    # first returns too few results.
    queries = _expand_query(query, max_queries)
    for q in queries:
        if len(rows) >= max_papers:
            break
        try:
            raw = tools_registry.execute_tool(
                "search_papers", {"query": q, "limit": max_papers - len(rows)}
            )
            data = json.loads(raw) if isinstance(raw, str) else raw
        except Exception as e:
            logger.warning(f"search_papers failed for '{q[:60]}': {e}")
            continue

        candidates = data.get("papers") or data.get("results") or []
        for p in candidates:
            if not isinstance(p, dict):
                continue
            row = _normalize_paper_row(p)
            if not row.get("title"):
                continue
            key = f"{row.get('title','')[:80]}::{row.get('url','')}"
            if key in seen_keys:
                continue
            seen_keys.add(key)
            rows.append(row)
            if len(rows) >= max_papers:
                break

    logger.info(f"gather_papers: collected {len(rows)} papers from {len(queries)} queries")
    return rows


def _normalize_paper_row(p: dict) -> dict:
    """Coerce a search result into IdeaScout's minimum schema."""
    return {
        "title": clean_text(p.get("title"), 400) if _SCOUT_AVAILABLE else str(p.get("title", ""))[:400],
        "abstract": clean_text(p.get("abstract"), 4000) if _SCOUT_AVAILABLE else str(p.get("abstract", ""))[:4000],
        "year": p.get("year"),
        "venue": p.get("venue") or p.get("publicationVenue", {}).get("name", "") if isinstance(p.get("publicationVenue"), dict) else p.get("venue", ""),
        "url": p.get("url") or p.get("pdf_url", ""),
        "citation_count": p.get("citationCount"),
    }


def _expand_query(query: str, max_queries: int) -> list[str]:
    """Yield up to max_queries search variants.

    The task string from THINK is often a long directive; we extract the core
    research terms and also try the raw first sentence.
    """
    queries = []
    # First sentence as a natural-language query
    first_sentence = re.split(r'[.\n]', query, 1)[0].strip()
    if first_sentence:
        queries.append(first_sentence[:200])
    # Key terms (words longer than 4 chars, deduped, top 8)
    words = re.findall(r'\b[a-zA-Z]{5,}\b', query.lower())
    unique = list(dict.fromkeys(words))[:8]
    if unique:
        queries.append(" ".join(unique))
    if len(queries) < max_queries and first_sentence:
        queries.append(first_sentence[:120])
    return queries[:max_queries] or [query[:200]]


# ─────────────────────────────────────────────────────────────────
# (b) Profile generation — from PROJECT_BRIEF + memory
# ─────────────────────────────────────────────────────────────────

# A balanced default scoring dimension set, derived from IdeaScout's examples.
_DEFAULT_DIMENSIONS = [
    Dimension(key="transferability", description="How directly the core idea transfers to the target task", weight=1.0),
    Dimension(key="mechanism_clarity", description="Whether the paper exposes a clear, reusable mechanism (not just a benchmark)", weight=0.8),
    Dimension(key="feasibility", description="How feasible it is to adapt this idea given typical compute and data constraints", weight=0.7),
] if _SCOUT_AVAILABLE else []


def build_profile_from_brief(brief_path: Path, memory=None,
                             profile_path: Optional[Path] = None) -> "Profile":
    """Build an IdeaScout Profile from the project brief.

    If ``profile_path`` is given and exists, load it directly (manual override).
    Otherwise parse PROJECT_BRIEF.md to extract:
      - description: the first paragraph after the title
      - target_tasks: lines under a "研究目标"/"Goal"/"Objective" heading
      - positive_keywords: domain-mechanism terms (extracted from the brief)
      - negative_keywords: from memory dead-ends if available
    """
    if not _SCOUT_AVAILABLE:
        raise RuntimeError("research-idea-scout not available; cannot build profile")

    if profile_path and Path(profile_path).exists():
        logger.info(f"Loading IdeaScout profile from {profile_path}")
        return load_profile(profile_path)

    brief_p = Path(brief_path)
    brief_text = brief_p.read_text(encoding="utf-8") if brief_p.exists() else ""

    description = _extract_description(brief_text)
    target_tasks = _extract_target_tasks(brief_text)
    positive_keywords = _extract_positive_keywords(brief_text)
    negative_keywords = _extract_negative_keywords(memory)

    # Fallbacks so the profile is never empty
    if not target_tasks:
        target_tasks = ["Discover transferable research ideas from other fields."]
    if not positive_keywords:
        positive_keywords = ["representation learning", "deep learning", "neural network"]

    profile = Profile(
        name=brief_p.stem if brief_p.name != "." else "auto_generated",
        description=description or "Auto-generated from PROJECT_BRIEF.md",
        target_tasks=target_tasks,
        positive_keywords=positive_keywords,
        negative_keywords=negative_keywords,
        scoring_dimensions=_DEFAULT_DIMENSIONS,
        prefer=[
            "Transferable mechanisms rather than surface topic similarity.",
            "Papers with reusable modeling ideas, representation designs, or objectives.",
        ],
        downweight=[
            "Generic benchmark, dataset, or survey papers without a transferable mechanism.",
            "Papers only related by keyword overlap but offering no reusable idea.",
        ],
    )
    logger.info(
        f"Built IdeaScout profile '{profile.name}': "
        f"{len(target_tasks)} tasks, {len(positive_keywords)} pos keywords, "
        f"{len(negative_keywords)} neg keywords"
    )
    return profile


def _extract_description(text: str) -> str:
    """First non-heading paragraph after the title."""
    lines = text.splitlines()
    for line in lines[1:]:
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        return s[:500]
    return ""


def _extract_target_tasks(text: str) -> list[str]:
    """Lines under a goal/objective/研究目标 heading."""
    tasks: list[str] = []
    in_section = False
    for line in text.splitlines():
        s = line.strip()
        if re.match(r'^#{1,3}\s*(研究目标|Goal|Objective|Target|目标)', s, re.I):
            in_section = True
            continue
        if in_section:
            if re.match(r'^#{1,3}\s', s):  # next heading
                break
            if s.startswith(("-", "*", "•")) or re.match(r'^\d+\.', s):
                task = re.sub(r'^[-*•\d.\s]+', '', s).strip()
                if task and len(task) > 5:
                    tasks.append(task[:200])
    return tasks[:5]


def _extract_positive_keywords(text: str) -> list[str]:
    """Extract domain-mechanism terms from the brief.

    Strategy: find frequently-occurring technical multi-word phrases and
    single domain terms. We look for bold/emphasized terms (**...**) and
    capitalized technical terms, plus a curated stopword filter.
    """
    keywords: list[str] = []
    # Bold terms (**...**) are usually emphasized mechanisms
    for m in re.finditer(r'\*\*([^*]{3,40})\*\*', text):
        term = m.group(1).strip().lower()
        if not _is_stopword_term(term):
            keywords.append(term)
    # Dedupe preserving order
    seen = set()
    unique = []
    for kw in keywords:
        if kw not in seen:
            seen.add(kw)
            unique.append(kw)
    return unique[:20]


def _extract_negative_keywords(memory) -> list[str]:
    """Derive negative keywords from memory dead-ends (failed approaches)."""
    if memory is None:
        return ["benchmark", "survey", "dataset only"]
    try:
        dead_ends = memory.get_dead_ends() if hasattr(memory, "get_dead_ends") else []
        kws = []
        for de in dead_ends[:10]:
            content = de.get("content", "") if isinstance(de, dict) else str(de)
            # Extract method-name-like tokens
            for m in re.finditer(r'\b([a-z]{4,}(?:_[a-z]+)?)\b', content.lower()):
                kws.append(m.group(1))
        return list(dict.fromkeys(kws))[:15] or ["benchmark", "survey"]
    except Exception:
        return ["benchmark", "survey", "dataset only"]


_STOPWORDS = frozenset({
    "the", "and", "for", "with", "that", "this", "from", "using", "via",
    "based", "model", "method", "approach", "paper", "propose", "proposed",
    "towards", "toward", "through", "between", "within", "where", "which",
    "these", "those", "their", "there", "been", "were", "they", "them",
})


def _is_stopword_term(term: str) -> bool:
    words = term.split()
    return all(w.lower() in _STOPWORDS for w in words) or len(term) < 4


# ─────────────────────────────────────────────────────────────────
# (c) LLM Scorer — use ProviderRouter instead of codex CLI
# ─────────────────────────────────────────────────────────────────

class LLMScorer:
    """Score papers via the agent's own LLM provider (failover-aware).

    Replaces IdeaScout's ``run_codex`` subprocess call with a direct
    ``_call_llm`` invocation, inheriting the full provider failover chain
    and quota-cooldown logic (Phase 1) for free.
    """

    def __init__(self, dispatcher, abstract_max_chars: int = 3000):
        self.dispatcher = dispatcher
        self.abstract_max_chars = abstract_max_chars

    def score(self, paper: dict, profile: "Profile") -> dict:
        """Score one paper. Returns the normalized result dict (with rank_score).

        Raises on persistent failure so the caller can skip the paper.
        """
        if not _SCOUT_AVAILABLE:
            raise RuntimeError("research-idea-scout not available")

        prompt = make_prompt(paper, profile, self.abstract_max_chars)
        # Use the researcher tier (strong model) — idea scoring needs reasoning.
        response_text, _trace = self.dispatcher._call_llm(
            system="You are a research idea scout. Score paper transferability. Return ONLY JSON.",
            messages=[{"role": "user", "content": prompt}],
            tools=None,
            max_turns=1,
            task_tier="researcher",
        )
        raw = extract_json_object(response_text)
        result = normalize_result(raw, profile)
        # Carry the original paper fields alongside the scores
        out = dict(paper)
        out.update(result)
        return out


# ─────────────────────────────────────────────────────────────────
# Orchestrator — the full gather → filter → score pipeline
# ─────────────────────────────────────────────────────────────────

def run_pipeline(
    tools_registry,
    dispatcher,
    profile: "Profile",
    query: str,
    max_papers: int = 50,
    filter_top_k: int = 20,
    score_top_k: int = 10,
    abstract_max_chars: int = 3000,
) -> dict:
    """Run the full IdeaScout pipeline and return a structured result.

    Returns a dict with:
      - ``papers_gathered``: int
      - ``papers_filtered``: int
      - ``papers_scored``: int
      - ``ranked``: list of scored paper dicts (sorted by rank_score desc)
      - ``errors``: list of per-paper scoring errors
    """
    if not _SCOUT_AVAILABLE:
        raise RuntimeError("research-idea-scout not available; cannot run pipeline")

    # 1. Gather
    papers = gather_papers(tools_registry, query, max_papers=max_papers)
    if not papers:
        return {"papers_gathered": 0, "papers_filtered": 0, "papers_scored": 0,
                "ranked": [], "errors": ["No papers gathered"]}

    # 2. Filter (rule-based)
    keep, _reject, summary = filter_rows(
        papers, profile, target_total=filter_top_k, min_score=0.5,
    )
    logger.info(f"IdeaScout filter: {summary['kept']} kept, {summary['rejected']} rejected")

    # 3. Score (LLM)
    scorer = LLMScorer(dispatcher, abstract_max_chars=abstract_max_chars)
    scored = []
    errors = []
    for i, paper in enumerate(keep[:filter_top_k], 1):
        try:
            result = scorer.score(paper, profile)
            scored.append(result)
            logger.info(
                f"IdeaScout score {i}/{min(len(keep), filter_top_k)}: "
                f"rank={result.get('rank_score')} priority={result.get('priority')} "
                f"| {str(result.get('title',''))[:60]}"
            )
            if len(scored) >= score_top_k:
                break
        except Exception as e:
            errors.append(f"{str(paper.get('title',''))[:60]}: {e}")
            logger.warning(f"IdeaScout score failed for '{str(paper.get('title',''))[:60]}': {e}")

    # 4. Rank
    scored.sort(key=lambda r: float(r.get("rank_score", 0)), reverse=True)

    return {
        "papers_gathered": len(papers),
        "papers_filtered": len(keep),
        "papers_scored": len(scored),
        "ranked": scored,
        "errors": errors,
    }


def format_results_markdown(pipeline_result: dict, query: str) -> str:
    """Format the pipeline output as a Markdown report for the workspace."""
    ranked = pipeline_result.get("ranked", [])
    lines = [
        f"# IdeaScout Cross-Domain Idea Discovery",
        f"",
        f"**Query**: {query[:200]}",
        f"**Papers gathered**: {pipeline_result.get('papers_gathered', 0)}",
        f"**Papers filtered**: {pipeline_result.get('papers_filtered', 0)}",
        f"**Papers scored**: {pipeline_result.get('papers_scored', 0)}",
        f"**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"",
    ]
    if pipeline_result.get("errors"):
        lines.append(f"**Scoring errors**: {len(pipeline_result['errors'])}")
        lines.append("")

    for i, paper in enumerate(ranked, 1):
        rank = paper.get("rank_score", 0)
        priority = paper.get("priority", "?")
        lines.append(f"## {i}. [{priority.upper()}] rank={rank:.2f} — {paper.get('title', '?')[:100]}")
        lines.append("")
        meta = []
        if paper.get("year"):
            meta.append(str(paper["year"]))
        if paper.get("venue"):
            meta.append(str(paper["venue"]))
        if paper.get("url"):
            meta.append(f"[link]({paper['url']})")
        if meta:
            lines.append(f"**{' | '.join(meta)}**")
        lines.append("")
        if paper.get("idea_core"):
            lines.append(f"**Core idea**: {paper['idea_core']}")
        if paper.get("transferable_mechanism"):
            lines.append(f"**Transferable mechanism**: {paper['transferable_mechanism']}")
        if paper.get("fit_reason"):
            lines.append(f"**Fit reason**: {paper['fit_reason']}")
        if paper.get("risk_or_limitation"):
            lines.append(f"**Risk**: {paper['risk_or_limitation']}")
        # Per-dimension scores
        scores = {k: v for k, v in paper.items() if k.startswith("score_") and isinstance(v, (int, float))}
        if scores:
            score_str = " | ".join(f"{k.replace('score_','')}: {v:.1f}" for k, v in scores.items())
            lines.append(f"\n*Scores*: {score_str}")
        lines.append("")

    return "\n".join(lines)
