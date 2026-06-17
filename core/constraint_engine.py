"""
AutoResearcher Constraint Engine — Tool-level safety + context pruning.

1. StrategyConstraintEngine: Converts historical dead-end patterns into rules.
   Used inside launch_experiment (tools.py) to block re-running known dead ends
   before they waste GPU time.
2. ContextPruner: Limits context injection per phase to keep prompts bounded.

P3 (referee-not-player): constraints fire inside tools as safety contracts,
not as overrides on the LLM's decisions.
"""

import ast
import json
import re
import math
import logging
import subprocess
import sys
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

logger = logging.getLogger("autoresearcher.constraint_engine")


# ──────────────────────────────────────────────────────────
# 1. Strategy Constraint Engine
# ──────────────────────────────────────────────────────────

@dataclass
class StrategyRule:
    """A constraint rule derived from historical patterns."""
    rule_id: str
    description: str
    condition: str  # Human-readable condition
    action: str     # What to do when triggered
    source: str     # Where this rule came from (e.g., "causal_history", "calibration")
    priority: str = "medium"  # critical | high | medium | low
    trigger_count: int = 0     # How many times this rule has been triggered


class StrategyConstraintEngine:
    """Convert historical patterns into executable constraint rules.

    Reads from:
    - causal_history: Past design decisions with verified effects
    - hypothesis_calibration: Historical hypothesis accuracy
    - dead_ends: Approaches that failed
    - experiment history: Method×domain Pareto frontier

    Generates rules like:
    - "If method X was tried and failed in domain Y, do NOT propose X again for Y"
    - "If last 3 hypotheses had <30% accuracy, force paper research before next experiment"
    - "If method X is Pareto-dominated by method Y, do NOT propose X"
    """

    def __init__(self, project_dir: Path, workspace: Path, config: dict = None):
        self.project_dir = Path(project_dir)
        self.workspace = Path(workspace)
        self._config = config or {}
        sbx = self._config.get("sandbox", {})
        self.subprocess_timeout: int = sbx.get("subprocess_timeout", 120)
        self._rules: list[StrategyRule] = []
        self._rules_loaded = False

    def _load_rules(self):
        """Load or generate constraint rules from historical data."""
        if self._rules_loaded:
            return
        self._rules_loaded = True

        # Load rules from workspace if they were saved
        rules_path = self.workspace / "STRATEGY_RULES.json"
        if rules_path.exists():
            try:
                data = json.loads(rules_path.read_text())
                for rd in data.get("rules", []):
                    self._rules.append(StrategyRule(**{k: v for k, v in rd.items()
                                                        if k in StrategyRule.__dataclass_fields__}))
                return
            except Exception as e:
                logger.debug(f"Failed to load strategy rules: {e}")

    def generate_rules_from_history(self, memory) -> list[StrategyRule]:
        """Generate constraint rules from historical experiment data."""
        self._load_rules()
        new_rules = []

        # Rule source 1: Hypothesis calibration → confidence constraint
        try:
            calibration = memory.get_experiment_calibration()
            total = calibration.get("total_hypotheses", 0)
            if total >= 3:
                accuracy = calibration.get("accuracy_rate", 0.5)
                if accuracy < 0.3:
                    new_rules.append(StrategyRule(
                        rule_id="low_hypothesis_accuracy",
                        description=(
                            f"Historical hypothesis accuracy is only {accuracy:.0%} across "
                            f"{total} experiments. The agent is making poor predictions."
                        ),
                        condition="hypothesis_accuracy < 0.3",
                        action=(
                            "MANDATORY: Before proposing ANY experiment, the agent MUST:\n"
                            "1. Cite specific evidence for the hypothesis\n"
                            "2. Identify what assumption the hypothesis relies on\n"
                            "3. Propose a MINIMAL test that could falsify the hypothesis\n"
                            "4. Get at least one independent data point before committing GPU hours"
                        ),
                        source="hypothesis_calibration",
                        priority="high",
                    ))
                elif accuracy < 0.5:
                    new_rules.append(StrategyRule(
                        rule_id="moderate_hypothesis_accuracy",
                        description=f"Hypothesis accuracy is {accuracy:.0%} — below random chance.",
                        condition="hypothesis_accuracy < 0.5",
                        action=(
                            "Before proposing an experiment, the agent must explain WHY this "
                            "hypothesis is different from past failed ones. What new evidence "
                            "or insight justifies trying again?"
                        ),
                        source="hypothesis_calibration",
                        priority="medium",
                    ))
        except Exception as e:
            logger.debug(f"Calibration rule generation skipped: {e}")

        # Rule source 2: Dead ends → forbidden approaches
        try:
            dead_ends = memory.get_dead_ends_full()[:20]
            if dead_ends:
                # Group dead ends by approach keywords
                approach_failures: dict[str, int] = {}
                for de in dead_ends:
                    text = de if isinstance(de, str) else str(de)
                    for keyword in self._extract_approach_keywords(text):
                        approach_failures[keyword] = approach_failures.get(keyword, 0) + 1

                for approach, count in approach_failures.items():
                    if count >= 3:
                        new_rules.append(StrategyRule(
                            rule_id=f"dead_end_{approach}",
                            description=(
                                f"Approach '{approach}' has been recorded as a dead end "
                                f"{count} times. It consistently fails."
                            ),
                            condition=f"task contains '{approach}'",
                            action=(
                                f"FORBIDDEN: Do NOT propose any experiment involving '{approach}'. "
                                f"It has failed {count} times. If you believe the situation has "
                                f"changed, you must explicitly justify why this time is different."
                            ),
                            source="dead_ends",
                            # Fix B: 5+ failures = hard block (forbidden), not just "high".
                            # Previously this was always "high"/"medium", making
                            # has_forbidden_violation unreachable for auto-generated rules.
                            priority="forbidden" if count >= 5 else ("high" if count >= 3 else "medium"),
                        ))
        except Exception as e:
            logger.debug(f"Dead-end rule generation skipped: {e}")

        # Rule source 3: Pareto frontier → method elimination
        try:
            pareto = memory.get_pareto_frontier()
            matrix = pareto.get("matrix", {})
            if matrix:
                dominated_methods = self._find_dominated_methods(matrix)
                if dominated_methods:
                    methods_str = ", ".join(dominated_methods[:5])
                    new_rules.append(StrategyRule(
                        rule_id="pareto_dominated",
                        description=(
                            f"Methods {methods_str} are Pareto-dominated by other methods "
                            f"in ALL domains. They are strictly worse."
                        ),
                        condition=f"proposed method is in [{methods_str}]",
                        action=(
                            f"AVOID: These methods are strictly dominated: {methods_str}. "
                            f"Do NOT propose them unless you have a fundamentally new variant "
                            f"that addresses the specific weakness that made them dominated."
                        ),
                        source="pareto_frontier",
                        priority="medium",
                    ))
        except Exception as e:
            logger.debug(f"Pareto rule generation skipped: {e}")

        # v16.1: Preserve human-written rules (source="human")
        # v16 bug: self._rules = new_rules deleted all human rules after first REFLECT
        human_rules = [r for r in self._rules if r.source == "human"]
        self._rules = human_rules + new_rules
        self._save_rules()

        return new_rules

    def _extract_approach_keywords(self, text: str) -> list[str]:
        """Extract approach keywords from dead end text."""
        approach_keywords = [
            "edge", "loss", "pretrain", "attention", "transformer",
            "gnn", "resnet", "unet", "conv3d", "lstm",
            "fft", "dct", "wavelet", "frequency",
            "augment", "mixup", "cutout", "dropout",
        ]
        text_lower = text.lower()
        return [kw for kw in approach_keywords if kw in text_lower]

    def _find_dominated_methods(self, matrix: dict) -> list[str]:
        """Find methods that are Pareto-dominated in all domains."""
        if not matrix:
            return []

        # matrix: {method: {domain: mae}}
        methods = list(matrix.keys())
        if len(methods) < 2:
            return []

        dominated = []
        for method in methods:
            method_scores = matrix[method]
            is_dominated = False
            for other in methods:
                if other == method:
                    continue
                other_scores = matrix[other]
                # Check if 'other' is strictly better in ALL shared domains
                shared_domains = set(method_scores.keys()) & set(other_scores.keys())
                if shared_domains:
                    all_better = all(
                        other_scores.get(d, float('inf')) < method_scores.get(d, float('inf'))
                        for d in shared_domains
                    )
                    if all_better:
                        is_dominated = True
                        break
            if is_dominated:
                dominated.append(method)

        return dominated

    def _save_rules(self):
        """Save generated rules to workspace for persistence."""
        rules_path = self.workspace / "STRATEGY_RULES.json"
        try:
            data = {
                "rules": [
                    {
                        "rule_id": r.rule_id,
                        "description": r.description,
                        "condition": r.condition,
                        "action": r.action,
                        "source": r.source,
                        "priority": r.priority,
                        "trigger_count": r.trigger_count,
                    }
                    for r in self._rules
                ]
            }
            rules_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        except Exception as e:
            logger.debug(f"Failed to save strategy rules: {e}")

    def check_constraints(self, think_result: dict, memory) -> list[str]:
        """Check proposed action against all active constraints. Returns violation messages."""
        self._load_rules()

        # Regenerate rules if none loaded
        if not self._rules:
            self.generate_rules_from_history(memory)

        violations = []
        task_text = (think_result.get("task", "") + " " + think_result.get("hypothesis", "")).lower()

        for rule in self._rules:
            triggered = False

            # Dead end rules
            if rule.source == "dead_ends":
                keywords = self._extract_approach_keywords(task_text)
                rule_keywords = self._extract_approach_keywords(rule.description.lower())
                if set(keywords) & set(rule_keywords):
                    triggered = True

            # Calibration rules
            elif rule.source == "hypothesis_calibration":
                if "low_hypothesis_accuracy" in rule.rule_id:
                    triggered = think_result.get("action") == "experiment"
                elif "moderate_hypothesis_accuracy" in rule.rule_id:
                    triggered = think_result.get("action") == "experiment"

            # Pareto rules
            elif rule.source == "pareto_frontier":
                if any(m.lower() in task_text for m in rule.description.lower().split(",")):
                    triggered = True

            if triggered:
                rule.trigger_count += 1
                violations.append(
                    f"[CONSTRAINT:{rule.priority.upper()}] {rule.description}\n"
                    f"ACTION REQUIRED: {rule.action}"
                )

        if violations:
            logger.warning(f"StrategyConstraintEngine: {len(violations)} constraint violation(s)")

        return violations

    def has_forbidden_violation(self, violations: list[str]) -> bool:
        """Check if any violation is a FORBIDDEN (hard block) rule."""
        return any("[CONSTRAINT:FORBIDDEN]" in v.upper() for v in violations)

    def get_constraint_prompt(self, violations: list[str]) -> str:
        """Format constraint violations as a prompt for the Leader."""
        if not violations:
            return ""
        return (
            "STRATEGY CONSTRAINT VIOLATIONS DETECTED:\n"
            "The proposed action violates learned constraints from past experiments.\n"
            "You MUST address each violation before proceeding:\n\n"
            + "\n\n".join(violations)
            + "\n\nIf you proceed despite these constraints, you MUST justify why this time is different."
        )


# v16.1: QuickBenchmark, AdaptiveThresholds, ImplementationTracker removed
# (dead modules, ~770 lines deleted)


# ──────────────────────────────────────────────────────────
# 2. Context Pruner
# ──────────────────────────────────────────────────────────

class ContextPruner:
    """Limit context injection to most relevant keys per cycle.

    v18: Now register-driven — tier sets are derived from context_keys.py
    (the single source of truth) instead of maintaining a divergent copy.
    Also enforces a character budget (not just a key-count limit), so a
    few large keys can't starve out important small ones.

    The tier system here mirrors context_keys.py's `tier` field:
    tier=1 (always) → tier=2 (situational) → tier=3 (conditional) → tier=4 (rare)
    """

    MAX_KEYS = 18  # v18: raised slightly since budget is now character-based
    MAX_CHARS = 12000  # v18: hard character budget on serialized context

    def __init__(self):
        # Derive tier sets from the registry (single source of truth).
        # This eliminates the dual-tier-system bug where ContextPruner's
        # hardcoded sets disagreed with context_keys.py.
        from .context_keys import THINK_KEYS, REFLECT_KEYS
        self._tier_sets = {1: set(), 2: set(), 3: set(), 4: set()}
        for ck in THINK_KEYS + REFLECT_KEYS:
            self._tier_sets.setdefault(ck.tier, set()).add(ck.name)

    @property
    def TIER_1_ALWAYS(self):
        return self._tier_sets.get(1, set())

    @property
    def TIER_2_SITUATIONAL(self):
        return self._tier_sets.get(2, set())

    @property
    def TIER_3_CONDITIONAL(self):
        return self._tier_sets.get(3, set())

    @property
    def TIER_4_RARE(self):
        return self._tier_sets.get(4, set())

    def prune(self, context: dict, phase: str) -> dict:
        """Select most relevant context keys, dropping low-priority ones.

        v18: Now uses both a key-count limit AND a character budget.
        Tier-1 keys are always included regardless of budget. Remaining keys
        are included by tier until either MAX_KEYS or MAX_CHARS is reached.
        """
        if len(context) <= self.MAX_KEYS:
            # Still check character budget
            pruned = dict(context)
        else:
            pruned = {}
            # Tier 1: Always include
            for key in self.TIER_1_ALWAYS:
                if key in context:
                    pruned[key] = context[key]
            # Tier 2-4: Include if present and under key budget
            for tier in (2, 3, 4):
                for key in self._tier_sets.get(tier, set()):
                    if key in context and len(pruned) < self.MAX_KEYS:
                        pruned[key] = context[key]
            # Catch-all: remaining non-empty keys
            for key, value in context.items():
                if key not in pruned and len(pruned) < self.MAX_KEYS:
                    if value is not None and value != "" and value != {} and value != []:
                        pruned[key] = value

        # v18: Character budget enforcement — drop lowest-tier keys if over
        from .context_keys import serialize_context, get_keys_for_phase
        registry = {k.name: k for k in get_keys_for_phase(phase)}
        total_chars = len(serialize_context(pruned, phase))
        if total_chars > self.MAX_CHARS:
            # Drop keys in reverse tier order (tier 4 first, then 3...)
            for tier in (4, 3, 2):
                for key in list(pruned.keys()):
                    ck = registry.get(key)
                    if ck and ck.tier == tier:
                        del pruned[key]
                        total_chars = len(serialize_context(pruned, phase))
                        if total_chars <= self.MAX_CHARS:
                            break
                if total_chars <= self.MAX_CHARS:
                    break

        dropped = len(context) - len(pruned)
        if dropped > 0:
            logger.debug(
                f"ContextPruner: {len(context)} -> {len(pruned)} keys "
                f"({dropped} dropped, {total_chars} chars)"
            )

        return pruned
