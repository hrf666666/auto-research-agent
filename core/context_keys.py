"""
Central registry of all context injection keys — the SINGLE SOURCE OF TRUTH.

Every key injected into the Leader's context dict is defined here with:
- name:       The exact dict key (injection, pruning, and serialization all read this)
- phase:      "think" or "reflect"
- description: Human-readable note
- serializer: A function (value, context) -> str | None that formats the value
              into a prompt section. Returns None to skip. This is what makes
              _format_leader_input register-driven instead of hardcoded.
- tier:       Pruning priority (1=always, 2=situational, 3=conditional, 4=rare)
- required:   If True, warn on injection failure

This registry replaces three previously-independent hardcoded lists:
  1. The inline `context["..."] = ...` blocks in loop.py (injection side)
  2. ContextPruner's TIER_1/2/3/4 sets (pruning side)
  3. _format_leader_input's manual `if context.get(...)` blocks (serialization)

Adding a new context key now requires exactly ONE change: add a ContextKey
entry here. Injection, pruning, and serialization all pick it up automatically.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional


# Type alias: a serializer takes (value, full_context) and returns a prompt
# section string, or None to skip.
Serializer = Callable[[Any, dict], Optional[str]]


# ── Serializer helpers (reusable across keys) ──

def _text_section(heading: str, max_chars: int = 2000) -> Serializer:
    """Serialize a string value under a heading, with truncation."""
    def serialize(value, _ctx):
        if not value:
            return None
        s = str(value)
        if len(s) > max_chars:
            s = s[:max_chars] + "\n... (truncated)"
        return f"## {heading}\n{s}\n"
    return serialize


def _json_section(heading: str, max_chars: int = 3000) -> Serializer:
    """Serialize a dict/list value as indented JSON under a heading."""
    import json
    def serialize(value, _ctx):
        if not value:
            return None
        s = json.dumps(value, indent=2, ensure_ascii=False, default=str)
        if len(s) > max_chars:
            s = s[:max_chars] + "\n... (truncated)"
        return f"## {heading}\n{s}\n"
    return serialize


def _list_section(heading: str, max_items: int = 5, prefix: str = "-") -> Serializer:
    """Serialize a list of strings as bullet points."""
    def serialize(value, _ctx):
        if not value or not isinstance(value, (list, tuple)):
            return None
        lines = [f"## {heading}"]
        for item in value[:max_items]:
            lines.append(f"{prefix} {item}")
        return "\n".join(lines) + "\n" if len(lines) > 1 else None
    return serialize


# ── Custom serializers that need special formatting ──

def _serialize_session_stats(value, _ctx):
    if not value or not isinstance(value, dict) or value.get("total_cycles", 0) == 0:
        return None
    lines = ["## Session Statistics",
             f"- Total cycles: {value.get('total_cycles', '?')}",
             f"- Experiments launched: {value.get('experiments_launched', '?')} "
             f"({value.get('launch_rate', 0)*100:.0f}%)" if isinstance(value.get('launch_rate'), (int, float)) else "",
             f"- Dead ends: {value.get('dead_ends_count', '?')}"]
    return "\n".join(l for l in lines if l) + "\n"


def _serialize_fabrication_warning(value, ctx):
    if not value:
        return None
    parts = ["## LLM FABRICATION DETECTED",
             "**The Code agent CLAIMED actions it did NOT perform.**",
             "**Do NOT trust claims from the previous EXECUTE phase.**"]
    details = ctx.get("fabrication_details", [])
    for d in details[:5]:
        parts.append(f"- {d}")
    return "\n".join(parts) + "\n"


def _serialize_visual_analysis(value, _ctx):
    if not value or not isinstance(value, dict) or not value.get("triggered"):
        return None
    sev = {"critical": "CRITICAL", "warning": "WARNING", "info": "INFO"}.get(
        value.get("severity", "info"), "INFO")
    return (
        f"## {sev}: VISUAL ANALYSIS DIAGNOSIS\n"
        f"Analyzed {value.get('images_analyzed', '?')} prediction images.\n"
    )


@dataclass(frozen=True)
class ContextKey:
    """Definition of a single context injection key."""
    name: str
    phase: str           # "think" or "reflect"
    description: str
    serializer: Serializer = field(default=None)
    tier: int = 3        # 1=always, 2=situational, 3=conditional, 4=rare
    required: bool = False


# ─────────────────────────────────────────────────────────────
# THINK phase context keys
# ─────────────────────────────────────────────────────────────

THINK_KEYS = [
    # ── Core (tier 1, always present) ──
    ContextKey("brief", "think", "PROJECT_BRIEF.md content",
               serializer=_text_section("Project Brief", 3000), tier=1),
    ContextKey("memory_log", "think", "MEMORY_LOG.md recent entries",
               serializer=_text_section("Memory Log", 4000), tier=1),
    ContextKey("cycle", "think", "Current cycle number",
               serializer=lambda v, _c: f"## Cycle: {v}\n" if v else None, tier=1),
    ContextKey("workspace_dir", "think", "Path to workspace directory",
               serializer=lambda v, _c: (
                   f"## Working Directory (CRITICAL)\n"
                   f"The code agent's working directory is: `{v}`\n"
                   f"All file paths must be relative to this directory.\n"
               ) if v else None, tier=1),
    ContextKey("directive", "think", "Human directive from DIRECTIVE.md",
               serializer=_text_section("Human Directive (HIGHEST PRIORITY)", 2000), tier=1),
    ContextKey("persistent_constraints", "think", "Project-level hard rules",
               serializer=_text_section("Persistent Constraints", 1000), tier=1),

    # ── Session intelligence (tier 2) ──
    ContextKey("session_stats", "think", "SQLite summary: cycles, experiments, dead ends",
               serializer=_serialize_session_stats, tier=2),
    ContextKey("recent_failures", "think", "Recent experiment failures (last 3)",
               serializer=_list_section("Recent Failure Patterns", 3), tier=2),
    ContextKey("code_review_lessons", "think", "Past mistakes from knowledge base",
               serializer=_text_section("Code Review Lessons", 1500), tier=2),
    ContextKey("relevant_code_review_lessons", "think", "Targeted lessons for current task",
               serializer=_text_section("Relevant Lessons", 1500), tier=2),

    # ── Domain knowledge (tier 2) ──
    ContextKey("domain_knowledge", "think",
               "Method-property mappings, domain compatibility, method assumptions",
               serializer=_text_section("Domain Knowledge", 2500), tier=2),
    ContextKey("data_constraints", "think", "Top-level data constraints from PROJECT_BRIEF",
               serializer=_text_section("Data Constraints", 1000), tier=2),
    ContextKey("cross_experiment_insights", "think",
               "Meta-patterns: dominant methods, hypothesis accuracy, calibration",
               serializer=_text_section("Cross-Experiment Insights", 2000), tier=2),

    # ── Experiment intelligence (tier 2) ──
    ContextKey("pareto_frontier", "think", "Pareto-optimal methods per domain",
               serializer=_text_section("Pareto Frontier", 1500), tier=2),
    ContextKey("causal_history", "think", "Past design decisions with verified effects",
               serializer=_text_section("Causal History", 1500), tier=2),
    ContextKey("hypothesis_calibration", "think", "Historical hypothesis accuracy",
               serializer=_text_section("Hypothesis Calibration", 800), tier=2),
    ContextKey("experiment_value_warn", "think",
               "Previously assessed low-value directions (VOI < 0.01)",
               serializer=_text_section("Low-Value Directions (avoid)", 600), tier=3),

    # ── Direction control (tier 3) ──
    ContextKey("data_scarcity_warning", "think", "Warning when < 10 training samples",
               serializer=_text_section("Data Scarcity Warning", 500), tier=3),
    ContextKey("method_inadequacy_retry_prompt", "think", "Retry guidance for inadequate methods",
               serializer=_text_section("Method Inadequacy Retry", 1000), tier=3),

    # ── Dataset understanding (tier 3) ──
    ContextKey("dataset_manifest_summary", "think", "DATASET_MANIFEST.json summary",
               serializer=_text_section("Dataset Manifest", 1500), tier=3),
]

# ─────────────────────────────────────────────────────────────
# REFLECT phase context keys
# ─────────────────────────────────────────────────────────────

REFLECT_KEYS = [
    # ── Core (tier 1) ──
    ContextKey("brief", "reflect", "PROJECT_BRIEF.md content",
               serializer=_text_section("Project Brief", 3000), tier=1),
    ContextKey("memory_log", "reflect", "MEMORY_LOG.md recent entries",
               serializer=_text_section("Memory Log", 4000), tier=1),
    ContextKey("cycle", "reflect", "Current cycle number",
               serializer=lambda v, _c: f"## Cycle: {v}\n" if v else None, tier=1),
    ContextKey("workspace_dir", "reflect", "Path to workspace directory",
               serializer=lambda v, _c: (
                   f"## Working Directory\n`{v}`\n"
               ) if v else None, tier=1),
    ContextKey("persistent_constraints", "reflect", "Project-level hard rules",
               serializer=_text_section("Persistent Constraints", 1000), tier=1),

    # ── Experiment result + VERIFY (tier 1) ──
    ContextKey("experiment_result", "reflect", "Full execute_result dict",
               serializer=_json_section("Experiment Result", 4000), tier=1),
    ContextKey("verify_diagnosis", "reflect", "List of VERIFY diagnosis strings",
               serializer=_list_section("VERIFY Report — Module Diagnosis", 8), tier=1),
    ContextKey("verify_failed_modules", "reflect", "Modules that failed VERIFY",
               serializer=lambda v, _c: (
                   f"\nFailed modules: {', '.join(v)}\n" if v else None
               ), tier=1),

    # ── Anti-deception (tier 2) ──
    ContextKey("llm_fabrication_detected", "reflect", "True if fabrication detected",
               serializer=_serialize_fabrication_warning, tier=2),
    ContextKey("fabrication_details", "reflect", "List of fabrication evidence",
               serializer=_list_section("Fabrication Evidence", 5), tier=2),

    # ── Visual analysis (tier 2) ──
    ContextKey("visual_analysis", "reflect", "VisualAnalysisResult dict (when triggered)",
               serializer=_serialize_visual_analysis, tier=2),
    ContextKey("visual_analysis_diagnosis", "reflect", "Visual diagnosis strings",
               serializer=_list_section("Visual Findings", 5), tier=2),
    ContextKey("visual_analysis_actions", "reflect", "Recommended actions from visual analysis",
               serializer=_list_section("Recommended Actions (Visual)", 5, prefix="1."), tier=2),

]


# ── Lookup helpers ──

THINK_KEY_NAMES = {k.name for k in THINK_KEYS}
REFLECT_KEY_NAMES = {k.name for k in REFLECT_KEYS}
ALL_KEY_NAMES = THINK_KEY_NAMES | REFLECT_KEY_NAMES

# Organized by (phase, tier) for the pruning logic
_KEY_INDEX: dict[tuple[str, int], list[ContextKey]] = {}
for _k in THINK_KEYS + REFLECT_KEYS:
    _KEY_INDEX.setdefault((_k.phase, _k.tier), []).append(_k)


def get_keys_for_phase(phase: str) -> list[ContextKey]:
    """Return all registered keys for a phase, in tier order."""
    return [k for k in (THINK_KEYS if phase == "think" else REFLECT_KEYS)]


def get_serializer(name: str, phase: str) -> Optional[Serializer]:
    """Look up the serializer for a key by name + phase."""
    for k in (THINK_KEYS if phase == "think" else REFLECT_KEYS):
        if k.name == name:
            return k.serializer
    return None


def serialize_context(context: dict, phase: str) -> str:
    """Serialize a context dict into a prompt string, register-driven.

    Iterates keys in tier order, calling each key's serializer. This is the
    SINGLE serialization path — replacing _format_leader_input's hardcoded
    if-blocks. Adding a key to the registry automatically makes it appear
    in the prompt.
    """
    parts = []
    keys = get_keys_for_phase(phase)
    for key in keys:
        value = context.get(key.name)
        if value is None:
            continue
        if key.serializer is None:
            # Keys without a serializer are internal (e.g., experiment_result
            # is read directly by the old code path during transition)
            continue
        section = key.serializer(value, context)
        if section:
            parts.append(str(section))
    return "\n".join(parts)


def validate_context(context: dict, phase: str) -> list[str]:
    """Validate a context dict against the registry. Returns list of warnings."""
    warnings = []
    expected = THINK_KEY_NAMES if phase == "think" else REFLECT_KEY_NAMES
    for key in context:
        if key not in expected and key not in ALL_KEY_NAMES:
            warnings.append(f"Unknown context key '{key}' in {phase} phase")
    required = {k.name for k in (THINK_KEYS if phase == "think" else REFLECT_KEYS) if k.required}
    for key in required:
        if key not in context:
            warnings.append(f"Missing required context key '{key}' in {phase} phase")
    return warnings
