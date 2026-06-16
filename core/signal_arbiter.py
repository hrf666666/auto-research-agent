"""Signal Arbitration System (v18) — the single decision arbiter.

ARCHITECTURE: Previously, 8+ subsystems independently injected signals into
the agent's context, task string, and directive files. Later injections
overrode earlier ones; signals could be lost if the LLM ignored them; there
was no priority arbitration or context budget enforcement.

The SignalArbiter replaces this with a unified pipeline:

    collect_signals()  ->  arbitrate()  ->  CycleDirective

KEY INVARIANTS:
  1. CRITICAL signals (from enforcement counters: launch failures, audit
     escalations, forbidden constraints) produce forced_action — the LLM
     never sees them as advisory; the loop acts directly.
  2. Deferred signals (budget-excluded) persist to next cycle's backlog.
     The CALLING subsystem decides whether to re-submit at higher severity
     (e.g., _consecutive_failed_launches increments independently).
  3. Total context <= budget_chars. Allocation is greedy by (severity, tier).
  4. ContextPruner is obsolete — the arbiter IS the pruner, but budget-aware.

SEVERITY MODEL:
  - CRITICAL: set by enforcement subsystems (launch counter >=2, audit
    counter >=2, forbidden constraint). Produces forced_action.
  - WARNING: set by subsystems tracking concerning patterns (stagnation,
    data scarcity, calibration drift). Always included in context if budget
    allows; never forced.
  - INFO: background context (domain knowledge, architecture plan, memory
    stats). Included if budget allows; first to be deferred.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from .context_keys import serialize_context, get_keys_for_phase

logger = logging.getLogger("autoresearcher.arbiter")

SEVERITY_ORDER = {"CRITICAL": 3, "WARNING": 2, "INFO": 1}
SEVERITY_LEVELS = ["INFO", "WARNING", "CRITICAL"]


@dataclass
class Signal:
    """A single signal from one subsystem."""
    source: str
    key: str
    severity: str
    content: Any
    forced_action: Optional[str] = None
    forced_task: Optional[str] = None
    forced_reason: Optional[str] = None
    escalation_count: int = 0

    @property
    def sort_key(self):
        return (SEVERITY_ORDER.get(self.severity, 0), self.escalation_count)


@dataclass
class CycleDirective:
    """The arbiter's decision for one cycle."""
    forced_action: Optional[str] = None
    forced_task: Optional[str] = None
    forced_reason: Optional[str] = None
    context: dict = field(default_factory=dict)
    deferred: list = field(default_factory=list)
    context_char_count: int = 0


class SignalArbiter:
    """Collects, prioritizes, and budgets all subsystem signals per cycle."""

    def __init__(self, budget_chars: int = 12000):
        self.budget_chars = budget_chars
        self._backlog = []
        self._pending = []

    def begin_cycle(self):
        self._pending = []

    def add_signal(self, source, key, content, severity="INFO",
                   forced_action=None, forced_task=None, forced_reason=None):
        sig = Signal(source=source, key=key, content=content, severity=severity,
                     forced_action=forced_action, forced_task=forced_task,
                     forced_reason=forced_reason)
        self._pending.append(sig)

    def get_backlog(self):
        return [{"source": s.source, "key": s.key, "content": s.content,
                 "severity": s.severity, "escalation_count": s.escalation_count}
                for s in self._backlog]

    def load_backlog(self, data):
        self._backlog = [Signal(source=d["source"], key=d["key"], content=d["content"],
                                severity=d["severity"], escalation_count=d.get("escalation_count", 0))
                         for d in data]

    def arbitrate(self, phase="think"):
        all_signals = list(self._pending)
        for back_sig in self._backlog:
            all_signals.append(self._escalate(back_sig))
        self._backlog.clear()

        all_signals = self._dedup(all_signals)

        critical = [s for s in all_signals if s.forced_action and s.severity == "CRITICAL"]
        if critical:
            pause = [s for s in critical if s.forced_action == "pause_human"]
            winner = max(pause or critical, key=lambda s: s.sort_key)
            logger.warning(
                "SIGNAL ARBITER: forced_action=%s from %s (severity=%s, escalation=%d)",
                winner.forced_action, winner.source, winner.severity,
                winner.escalation_count)
            context = self._build_minimal_context(all_signals, phase)
            return CycleDirective(
                forced_action=winner.forced_action,
                forced_task=winner.forced_task,
                forced_reason=winner.forced_reason or f"{winner.source} requires action",
                context=context)

        context, deferred = self._allocate_budget(all_signals, phase)
        self._backlog = deferred
        char_count = len(serialize_context(context, phase))
        logger.info("SIGNAL ARBITER: %d signals, %d keys (%d chars), %d deferred",
                    len(all_signals), len(context), char_count, len(deferred))
        return CycleDirective(context=context, deferred=deferred,
                              context_char_count=char_count)

    def _escalate(self, sig):
        """Carry a backlog signal into the next cycle with incremented count.

        Severity is NOT auto-derived — the calling subsystem (launch counter,
        audit counter, etc.) tracks its own enforcement and re-submits at the
        appropriate severity. The backlog just ensures budget-deferred signals
        survive to the next cycle rather than being silently dropped.
        """
        return Signal(source=sig.source, key=sig.key, content=sig.content,
                      severity=sig.severity,
                      forced_action=sig.forced_action,
                      forced_task=sig.forced_task,
                      forced_reason=sig.forced_reason,
                      escalation_count=sig.escalation_count + 1)


    def _dedup(self, signals):
        by_key = {}
        for sig in signals:
            k = (sig.source, sig.key)
            if k not in by_key:
                by_key[k] = sig
            else:
                existing = by_key[k]
                if SEVERITY_ORDER.get(sig.severity, 0) > SEVERITY_ORDER.get(existing.severity, 0):
                    by_key[k] = sig
                by_key[k].escalation_count = max(existing.escalation_count,
                                                  sig.escalation_count)
        return list(by_key.values())

    def _allocate_budget(self, signals, phase):
        ranked = sorted(signals, key=lambda s: s.sort_key, reverse=True)
        registry_keys = {k.name: k for k in get_keys_for_phase(phase)}
        context = {}
        included = []
        deferred = []
        current_chars = 0

        for sig in ranked:
            ck = registry_keys.get(sig.key)
            if ck and ck.serializer:
                section = ck.serializer(sig.content, context)
                est_size = len(str(section)) if section else 0
            else:
                est_size = len(str(sig.content)) if sig.content else 0

            is_tier1 = ck and ck.tier == 1
            if is_tier1 or current_chars + est_size <= self.budget_chars:
                context[sig.key] = sig.content
                included.append(sig)
                current_chars += est_size
            else:
                deferred.append(sig)

        return context, deferred

    def _build_minimal_context(self, all_signals, phase):
        registry_keys = {k.name: k for k in get_keys_for_phase(phase)}
        context = {}
        for sig in all_signals:
            ck = registry_keys.get(sig.key)
            if ck and ck.tier == 1:
                context[sig.key] = sig.content
        return context

    def clear_resolved(self, key):
        self._backlog = [s for s in self._backlog if s.key != key]
