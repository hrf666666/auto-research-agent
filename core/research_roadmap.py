"""
Research Roadmap State Machine (v15)

The core research methodology layer that enforces structured research:
  总体 → 分支 → 细节 → 理论验证 → 模块实验 → 集成

Key principles:
1. Module-level phase tracking (not global linear) — each module has its own phase
2. Hard gates: code-level enforcement, not just prompt injection
3. Multi-method verification: 3-6 independent attempts (min 3, max 7) before marking dead_end
4. Alignment detection: real-time check that THINK output matches ROADMAP

Architecture coupling:
- Reads: PROJECT_BRIEF.md, idea_planner output (dict)
- Writes: workspace/RESEARCH_ROADMAP.md (human & LLM readable)
- Consumed by: loop.py (_think context injection, post-THINK alignment check)
- Records to: memory.py (via log_roadmap_update)
"""

import json
import re
import time
import logging
from enum import Enum
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

logger = logging.getLogger("autoresearcher.roadmap")


# ── Phase Enumeration ──

class ModulePhase(str, Enum):
    """Phase of a single research module.

    Each module progresses independently through these phases.
    The global research phase is determined by the slowest module.
    """
    THEORY_VERIFICATION = "theory_verification"   # Phase 0: validate assumptions with data analysis
    MODULE_DESIGN = "module_design"               # Phase 1: design module architecture
    MODULE_VALIDATION = "module_validation"        # Phase 2: implement + validate independently
    INTEGRATED = "integrated"                      # Phase 3: integrated into full pipeline
    DEAD_END = "dead_end"                          # Module failed after exhausting attempts


class GlobalPhase(str, Enum):
    """Global research phase, derived from all modules' states."""
    THEORY_VERIFICATION = "theory_verification"
    MODULE_IMPLEMENTATION = "module_implementation"
    INTEGRATION = "integration"
    OPTIMIZATION = "optimization"
    FAILED = "failed"  # All modules or core modules are dead ends


# ── Data Structures ──

@dataclass
class VerificationAttempt:
    """A single verification attempt for a module's theoretical assumption."""
    method: str                  # Description of the verification method
    result: str = "pending"      # "pending" | "passed" | "failed" | "inconclusive"
    evidence: str = ""           # What was observed
    cycle: int = 0               # Cycle number when this was executed

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "result": self.result,
            "evidence": self.evidence,
            "cycle": self.cycle,
        }


@dataclass
class ModuleEntry:
    """A single research module tracked in the roadmap.

    This is the atomic unit of research progress tracking.
    Each module must be independently verified before integration.
    """
    name: str                                     # Unique module identifier
    description: str                              # What this module does (from idea_planner)
    theoretical_assumptions: list[str] = field(default_factory=list)
    verification_methods: list[str] = field(default_factory=list)  # Suggested ≥3 methods
    attempts: list[VerificationAttempt] = field(default_factory=list)
    phase: ModulePhase = ModulePhase.THEORY_VERIFICATION
    dependencies: list[str] = field(default_factory=list)  # Module names this depends on
    is_core: bool = True                          # Core modules must pass for idea to be viable
    dead_end_reason: str = ""                     # Why this module was marked as dead_end
    last_updated_cycle: int = 0

    # Progress tracking
    design_description: str = ""                  # Module architecture description
    implementation_file: str = ""                 # Path to implementation file
    validation_metrics: dict = field(default_factory=dict)  # Module-level metrics

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "theoretical_assumptions": self.theoretical_assumptions,
            "verification_methods": self.verification_methods,
            "attempts": [a.to_dict() for a in self.attempts],
            "phase": self.phase.value,
            "dependencies": self.dependencies,
            "is_core": self.is_core,
            "dead_end_reason": self.dead_end_reason,
            "last_updated_cycle": self.last_updated_cycle,
            "design_description": self.design_description,
            "implementation_file": self.implementation_file,
            "validation_metrics": self.validation_metrics,
        }

    @property
    def is_verified(self) -> bool:
        """Check if at least one verification attempt passed."""
        return any(a.result == "passed" for a in self.attempts)

    @property
    def exhausted_attempts(self) -> bool:
        """Check if all suggested verification methods have been tried."""
        tried = {a.method for a in self.attempts if a.result != "pending"}
        return len(tried) >= max(len(self.verification_methods), 3)

    @property
    def failed_attempt_count(self) -> int:
        return sum(1 for a in self.attempts if a.result == "failed")


# ── Research Roadmap Manager ──

class ResearchRoadmap:
    """Manages the research roadmap: module decomposition, phase tracking, alignment.

    Lifecycle:
    1. generate_from_brief() — called once at Cycle 1, creates ROADMAP from PROJECT_BRIEF
    2. get_phase_context() — called every cycle in _think(), injects phase constraints
    3. check_alignment() — called after _think() returns, detects deviation
    4. update_from_cycle_outcome() — called in _record_cycle_outcome(), updates module status

    The ROADMAP file (workspace/RESEARCH_ROADMAP.md) is the source of truth.
    It's written in a format that both humans and LLMs can read and modify.
    """

    def __init__(self, workspace: Path):
        self.workspace = Path(workspace)
        self.roadmap_path = self.workspace / "RESEARCH_ROADMAP.md"
        self.modules: list[ModuleEntry] = []
        self.idea_summary: str = ""
        self.global_phase: GlobalPhase = GlobalPhase.THEORY_VERIFICATION
        self._last_loaded_mtime: float = 0.0
        self._deviation_count: int = 0  # Consecutive deviations

    # ── ROADMAP File I/O ──

    def load(self) -> bool:
        """Load ROADMAP from file. Returns True if loaded successfully."""
        if not self.roadmap_path.exists():
            return False

        try:
            content = self.roadmap_path.read_text(encoding="utf-8")
            self._parse_markdown(content)
            self._last_loaded_mtime = self.roadmap_path.stat().st_mtime
            self._compute_global_phase()
            return True
        except Exception as e:
            logger.warning(f"Failed to load ROADMAP: {e}")
            return False

    def save(self):
        """Save current ROADMAP state to markdown file."""
        content = self._render_markdown()
        self.roadmap_path.write_text(content, encoding="utf-8")
        self._last_loaded_mtime = self.roadmap_path.stat().st_mtime
        logger.info(f"ROADMAP saved: {len(self.modules)} modules, phase={self.global_phase.value}")

    def reload_if_changed(self) -> bool:
        """Reload if file was modified externally (by Leader agent)."""
        if not self.roadmap_path.exists():
            return False
        current_mtime = self.roadmap_path.stat().st_mtime
        if current_mtime != self._last_loaded_mtime:
            return self.load()
        return False

    # ── Generation from PROJECT_BRIEF ──

    def generate_from_brief(self, project_brief_path: Path, arch_plan: dict = None) -> dict:
        """Generate initial ROADMAP from PROJECT_BRIEF and (optionally) idea_planner output.

        This is called once at Cycle 1. It creates the module decomposition.

        Args:
            project_brief_path: Path to PROJECT_BRIEF.md
            arch_plan: Optional output from IdeaPlanner.plan() with module specs

        Returns:
            dict with "status" and "modules" keys
        """
        if not project_brief_path.exists():
            return {"status": "error", "message": f"PROJECT_BRIEF not found: {project_brief_path}"}

        brief_content = project_brief_path.read_text(encoding="utf-8")

        # Extract idea summary from PROJECT_BRIEF
        self.idea_summary = self._extract_idea_summary(brief_content)

        # Build modules from arch_plan if available
        if arch_plan and "modules" in arch_plan:
            self.modules = self._modules_from_arch_plan(arch_plan)
        else:
            # Fallback: create a single module from the idea
            self.modules = self._modules_from_brief(brief_content)

        # Ensure each module has 3-6 verification methods (min 3, max 7)
        for mod in self.modules:
            while len(mod.verification_methods) < 3:
                idx = len(mod.verification_methods) + 1
                mod.verification_methods.append(
                    f"Verification method {idx}: [TO BE DEFINED by agent based on module assumptions]"
                )
            # Cap at 7 methods
            if len(mod.verification_methods) > 7:
                mod.verification_methods = mod.verification_methods[:7]

        self.global_phase = GlobalPhase.THEORY_VERIFICATION
        self.save()

        logger.info(
            f"ROADMAP generated: {len(self.modules)} modules, "
            f"core={sum(1 for m in self.modules if m.is_core)}"
        )
        return {
            "status": "ok",
            "modules": [m.to_dict() for m in self.modules],
            "global_phase": self.global_phase.value,
        }

    # ── Phase Context Injection ──

    def get_phase_context(self, cycle: int) -> str:
        """Generate context string for _think() injection.

        This is the PRIMARY control mechanism: it tells the Leader what phase
        we're in and what the current module requires.
        """
        # Reload in case agent modified the file
        self.reload_if_changed()

        current_modules = self._get_active_modules()
        if not current_modules:
            if all(m.phase == ModulePhase.DEAD_END for m in self.modules):
                return (
                    "⚠️ RESEARCH ROADMAP — ALL MODULES DEAD END\n"
                    "All research modules have been marked as dead ends after exhaustive attempts.\n"
                    "MANDATORY: Perform a SYSTEMATIC FAILURE ANALYSIS:\n"
                    "1. Is the core idea fundamentally flawed? (physical assumptions violated?)\n"
                    "2. Is the architecture approach wrong? (wrong decomposition?)\n"
                    "3. Is it a resource/implementation issue? (data, compute, library?)\n"
                    "Report findings and propose next action (paper_research or project pivot)."
                )
            return ""  # All modules completed, no constraint needed

        # Build phase-specific context
        lines = [
            f"📋 RESEARCH ROADMAP — Phase: {self.global_phase.value} | Cycle {cycle}",
            f"Idea: {self.idea_summary[:200]}",
            "",
            "ACTIVE MODULES (must be addressed before proceeding):",
        ]

        for mod in current_modules:
            lines.append(f"")
            lines.append(f"  【{mod.name}】 ({mod.phase.value})")
            lines.append(f"  Purpose: {mod.description[:150]}")
            if mod.theoretical_assumptions:
                lines.append(f"  Assumptions to verify:")
                for i, a in enumerate(mod.theoretical_assumptions, 1):
                    verified = any(attempt.result == "passed" for attempt in mod.attempts
                                   if any(kw in attempt.method.lower() for kw in a.lower().split()[:3]))
                    mark = "✓" if verified else "○"
                    lines.append(f"    {mark} {i}. {a[:100]}")
            if mod.attempts:
                passed = sum(1 for a in mod.attempts if a.result == "passed")
                failed = sum(1 for a in mod.attempts if a.result == "failed")
                lines.append(f"  Attempts: {passed} passed, {failed} failed, {len(mod.attempts)} total")
                # Show latest attempt
                latest = mod.attempts[-1]
                lines.append(f"  Latest: {latest.method[:80]} → {latest.result} (cycle {latest.cycle})")

            # Phase-specific requirements
            if mod.phase == ModulePhase.THEORY_VERIFICATION:
                remaining_methods = self._get_remaining_methods(mod)
                if remaining_methods:
                    lines.append(f"  ⚡ NEXT: Verify assumption using one of:")
                    for m in remaining_methods[:7]:  # Show all methods (max 7)
                        lines.append(f"     - {m[:100]}")
                else:
                    lines.append(f"  ⚡ All methods tried but none passed. Consider new verification approaches.")

            elif mod.phase == ModulePhase.MODULE_DESIGN:
                lines.append(f"  ⚡ NEXT: Design module architecture based on verified assumptions.")

            elif mod.phase == ModulePhase.MODULE_VALIDATION:
                lines.append(f"  ⚡ NEXT: Implement module and validate independently (isolated test).")

        # Add phase-specific constraint
        lines.append("")
        if self.global_phase == GlobalPhase.THEORY_VERIFICATION:
            lines.append("🚫 PHASE CONSTRAINT: You MUST NOT start model training or full pipeline experiments.")
            lines.append("   Current phase requires DATA ANALYSIS experiments to verify theoretical assumptions.")
            lines.append("   Only propose: data analysis, statistical tests, visualization, small-scale probing experiments.")
        elif self.global_phase == GlobalPhase.MODULE_IMPLEMENTATION:
            lines.append("🚫 PHASE CONSTRAINT: You MUST implement and validate modules INDEPENDENTLY.")
            lines.append("   Do NOT integrate modules or run full pipeline training yet.")
            lines.append("   Only propose: module implementation, unit tests, isolated validation experiments.")
        elif self.global_phase == GlobalPhase.INTEGRATION:
            lines.append("✅ Integration phase: modules validated, you may now integrate and train.")

        return "\n".join(lines)

    # ── Alignment Check ──

    def check_alignment(self, think_result: dict) -> dict:
        """Check if the THINK output aligns with current ROADMAP requirements.

        Called AFTER dispatch_leader() returns, BEFORE _apply_no_progress_fallback().

        Returns:
            {
                "aligned": bool,
                "current_modules": list[str],  # Names of active modules
                "deviation_type": str,  # "phase_violation" | "off_roadmap" | ""
                "correction_prompt": str,  # Prompt to inject if misaligned
            }
        """
        self.reload_if_changed()

        active_modules = self._get_active_modules()
        if not active_modules:
            return {
                "aligned": True,
                "current_modules": [],
                "deviation_type": "",
                "correction_prompt": "",
            }

        action = think_result.get("action", "")
        task = think_result.get("task", "").lower()
        active_names = [m.name.lower() for m in active_modules]

        # Check 1: Phase violation — training during theory_verification
        if self.global_phase == GlobalPhase.THEORY_VERIFICATION:
            if action == "experiment" and self._is_training_task(task):
                self._deviation_count += 1
                correction = self._build_phase_violation_correction(active_modules)
                return {
                    "aligned": False,
                    "current_modules": [m.name for m in active_modules],
                    "deviation_type": "phase_violation",
                    "correction_prompt": correction,
                }

        # Check 2: Off-roadmap — task doesn't relate to any active module
        if action == "experiment" and active_names:
            related = self._is_task_related(task, active_modules)
            if not related:
                self._deviation_count += 1
                correction = self._build_off_roadmap_correction(active_modules)
                return {
                    "aligned": False,
                    "current_modules": [m.name for m in active_modules],
                    "deviation_type": "off_roadmap",
                    "correction_prompt": correction,
                }

        # Aligned — reset deviation count
        self._deviation_count = 0
        return {
            "aligned": True,
            "current_modules": [m.name for m in active_modules],
            "deviation_type": "",
            "correction_prompt": "",
        }

    # ── Cycle Outcome Update ──

    def update_from_cycle_outcome(self, think_result: dict, reflect_result: dict, cycle: int):
        """Update ROADMAP based on cycle outcome. Called in _record_cycle_outcome().

        This is the feedback loop: experiment results → module status updates.
        """
        changed = False

        # Extract relevant info from results
        task = think_result.get("task", "")
        action = think_result.get("action", "")
        milestone = reflect_result.get("milestone", "")
        dead_end = reflect_result.get("dead_end", "")
        decision = reflect_result.get("decision", "")

        # Update module status based on results
        for mod in self.modules:
            if mod.phase in (ModulePhase.DEAD_END, ModulePhase.INTEGRATED):
                continue

            # Check if this cycle's task relates to this module
            if not self._is_task_related(task.lower(), [mod]):
                continue

            mod.last_updated_cycle = cycle

            # Milestone: module made progress (but not if also a dead_end)
            if milestone and not dead_end:
                if mod.phase == ModulePhase.THEORY_VERIFICATION:
                    # Check if milestone indicates assumption verification
                    if any(kw in milestone.lower() for kw in ["verified", "validated", "confirmed", "passed"]):
                        mod.attempts.append(VerificationAttempt(
                            method=self._extract_method_from_task(task),
                            result="passed",
                            evidence=milestone[:300],
                            cycle=cycle,
                        ))
                        # Check if theory verification phase can advance
                        if mod.is_verified:
                            mod.phase = ModulePhase.MODULE_DESIGN
                            logger.info(f"ROADMAP: {mod.name} advanced to MODULE_DESIGN")
                            changed = True

                elif mod.phase == ModulePhase.MODULE_DESIGN:
                    # Design completed — advance to validation
                    if any(kw in milestone.lower() for kw in ["designed", "implemented", "coded", "built"]):
                        mod.design_description = milestone[:500]
                        mod.phase = ModulePhase.MODULE_VALIDATION
                        logger.info(f"ROADMAP: {mod.name} advanced to MODULE_VALIDATION")
                        changed = True

                elif mod.phase == ModulePhase.MODULE_VALIDATION:
                    if any(kw in milestone.lower() for kw in ["validated", "working", "passed", "converged"]):
                        mod.phase = ModulePhase.INTEGRATED
                        logger.info(f"ROADMAP: {mod.name} advanced to INTEGRATED")
                        changed = True

            # Dead end: module failed
            if dead_end:
                method = self._extract_method_from_task(task)
                existing = [a for a in mod.attempts if a.method == method]
                if not existing:
                    mod.attempts.append(VerificationAttempt(
                        method=method,
                        result="failed",
                        evidence=dead_end[:300],
                        cycle=cycle,
                    ))

                # Check if we should mark as dead_end
                if self._should_mark_dead_end(mod):
                    mod.phase = ModulePhase.DEAD_END
                    mod.dead_end_reason = (
                        f"Exhausted {len(mod.attempts)} verification attempts. "
                        f"Last failure: {dead_end[:200]}"
                    )
                    logger.warning(f"ROADMAP: {mod.name} marked as DEAD_END")
                    changed = True

            # Decision-based updates
            if decision and "module_design" in decision.lower() and mod.phase == ModulePhase.MODULE_DESIGN:
                mod.design_description = decision[:500]
                mod.phase = ModulePhase.MODULE_VALIDATION
                logger.info(f"ROADMAP: {mod.name} advanced to MODULE_VALIDATION")
                changed = True

        if changed:
            self._compute_global_phase()
            self.save()

    # ── Module Status Management ──

    def get_current_module(self) -> Optional[ModuleEntry]:
        """Get the module that should be worked on next (highest priority)."""
        active = self._get_active_modules()
        if not active:
            return None

        # Priority: core modules first, then by phase order
        core = [m for m in active if m.is_core]
        if core:
            return core[0]
        return active[0]

    def update_module_status(self, module_name: str, phase: str, cycle: int,
                             attempt: dict = None, design_desc: str = "") -> bool:
        """Manually update a module's status (for human directive or agent REFLECT).

        Returns True if module was found and updated.
        """
        for mod in self.modules:
            if mod.name == module_name:
                try:
                    mod.phase = ModulePhase(phase)
                except ValueError:
                    logger.warning(f"Invalid phase: {phase}")
                    return False

                mod.last_updated_cycle = cycle

                if attempt:
                    mod.attempts.append(VerificationAttempt(**attempt))

                if design_desc:
                    mod.design_description = design_desc

                self._compute_global_phase()
                self.save()
                return True

        return False

    def reset_deviation_count(self):
        """Reset deviation counter. Called by loop.py after hard gate enforcement."""
        self._deviation_count = 0

    @property
    def active_module_names(self) -> list[str]:
        """Public accessor for active module names (non-dead_end, non-integrated)."""
        return [m.name for m in self._get_active_modules()]

    @property
    def is_theory_verification_phase(self) -> bool:
        """Check if global phase is theory_verification with active modules."""
        return (
            self.global_phase == GlobalPhase.THEORY_VERIFICATION
            and len(self._get_active_modules()) > 0
        )

    def get_status_summary(self) -> dict:
        """Get a concise status summary for logging."""
        return {
            "global_phase": self.global_phase.value,
            "modules": {
                m.name: {
                    "phase": m.phase.value,
                    "attempts": len(m.attempts),
                    "verified": m.is_verified,
                    "is_core": m.is_core,
                }
                for m in self.modules
            },
            "deviation_count": self._deviation_count,
        }

    # ── Internal Helpers ──

    def _get_active_modules(self) -> list[ModuleEntry]:
        """Get modules that still need work (not dead_end, not integrated)."""
        active = [m for m in self.modules
                  if m.phase not in (ModulePhase.DEAD_END, ModulePhase.INTEGRATED)]
        # Sort: core first, then by phase order
        phase_order = {
            ModulePhase.THEORY_VERIFICATION: 0,
            ModulePhase.MODULE_DESIGN: 1,
            ModulePhase.MODULE_VALIDATION: 2,
        }
        active.sort(key=lambda m: (not m.is_core, phase_order.get(m.phase, 99)))
        return active

    def _compute_global_phase(self):
        """Derive global phase from all modules' states."""
        active = self._get_active_modules()
        if not active:
            # Check if all are dead ends
            if all(m.phase == ModulePhase.DEAD_END for m in self.modules):
                self.global_phase = GlobalPhase.FAILED
            else:
                # All modules integrated — optimization phase
                self.global_phase = GlobalPhase.OPTIMIZATION
            return

        # Global phase = the earliest phase among active modules
        if any(m.phase == ModulePhase.THEORY_VERIFICATION for m in active):
            self.global_phase = GlobalPhase.THEORY_VERIFICATION
        elif any(m.phase in (ModulePhase.MODULE_DESIGN, ModulePhase.MODULE_VALIDATION) for m in active):
            self.global_phase = GlobalPhase.MODULE_IMPLEMENTATION
        else:
            self.global_phase = GlobalPhase.INTEGRATION

    def _is_training_task(self, task: str) -> bool:
        """Detect if a task description involves model training.

        Uses compound patterns to avoid false positives from words like 'loss'
        appearing in analysis contexts (e.g., 'information loss', 'lossless').
        """
        # Strong indicators: almost certainly training
        strong_indicators = [
            "train the", "training", "train model", "train a", "trainer",
            "epoch", "batch_size", "batch size", "optimizer", "learning rate",
            "fine-tun", "finetun", "backprop", "gradient descent",
            "full pipeline", "end-to-end", "e2e", "training loop",
            "训练", "微调",
        ]
        if any(kw in task for kw in strong_indicators):
            return True

        # Weak indicators: only match if combined with model context
        weak_indicators = ["loss function", "loss curve", "minimize loss",
                           "fit(", ".fit(", "model.fit"]
        return any(kw in task for kw in weak_indicators)

    def _is_task_related(self, task: str, modules: list[ModuleEntry]) -> bool:
        """Check if a task description relates to any of the given modules.

        Uses a tiered matching strategy:
        1. Module name (strongest signal)
        2. Module name sub-tokens (e.g., 'freq' from 'freq_analyzer')
        3. Key words from description/assumptions
        """
        task_lower = task.lower()

        for mod in modules:
            # Check module name (full match)
            if mod.name.lower() in task_lower:
                return True

            # Check module name sub-tokens (e.g., 'freq' from 'freq_analyzer')
            name_tokens = [t for t in re.split(r"[_\s]+", mod.name.lower()) if len(t) > 2]
            if sum(1 for t in name_tokens if t in task_lower) >= 2:
                return True

            # Check key words from description (lower threshold for longer words)
            desc_words = [w.lower() for w in mod.description.split() if len(w) > 3]
            if desc_words and sum(1 for w in desc_words if w in task_lower) >= max(2, len(desc_words) // 3):
                return True

            # Check assumptions
            for assumption in mod.theoretical_assumptions:
                words = [w.lower() for w in assumption.split() if len(w) > 3]
                if words and sum(1 for w in words if w in task_lower) >= max(2, len(words) // 3):
                    return True

        return False

    def _get_remaining_methods(self, mod: ModuleEntry) -> list[str]:
        """Get verification methods not yet tried."""
        tried_methods = {a.method for a in mod.attempts}
        return [m for m in mod.verification_methods if m not in tried_methods]

    def _should_mark_dead_end(self, mod: ModuleEntry) -> bool:
        """Determine if a module should be marked as dead_end.

        Criteria:
        - At least 3 independent verification methods attempted (up to 7)
        - All attempts failed or inconclusive
        - No single method was marked as method_inadequacy retryable
        """
        if len(mod.attempts) < 3:
            return False

        failed = [a for a in mod.attempts if a.result == "failed"]
        inconclusive = [a for a in mod.attempts if a.result == "inconclusive"]

        # If all attempts are failed/inconclusive and we have ≥3
        if len(failed) + len(inconclusive) == len(mod.attempts) and len(mod.attempts) >= 3:
            return True

        # Safety valve: if ≥7 attempts with mixed results but no clear path forward,
        # mark as dead_end to prevent infinite stalling
        if len(mod.attempts) >= 7 and not mod.is_verified:
            return True

        return False

    def _extract_method_from_task(self, task: str) -> str:
        """Extract a short description of the verification method from task."""
        # Take first 150 chars as method description
        return task[:150].strip().replace("\n", " ")

    def _extract_idea_summary(self, brief_content: str) -> str:
        """Extract the core idea summary from PROJECT_BRIEF."""
        # Look for the first substantial paragraph or "## Core Idea" section
        lines = brief_content.split("\n")
        summary_lines = []
        capture = False
        for line in lines:
            if any(h in line.lower() for h in ["## idea", "## core", "## objective", "## goal", "## overview"]):
                capture = True
                continue
            if capture:
                if line.startswith("##") and summary_lines:
                    break
                if line.strip():
                    summary_lines.append(line.strip())
        if summary_lines:
            return " ".join(summary_lines)[:500]

        # Fallback: first non-empty paragraph
        for line in lines:
            line = line.strip()
            if line and not line.startswith("#") and len(line) > 20:
                return line[:500]
        return "No idea summary extracted from PROJECT_BRIEF"

    def _modules_from_arch_plan(self, arch_plan: dict) -> list[ModuleEntry]:
        """Create ModuleEntry list from IdeaPlanner output."""
        modules = []
        plan_modules = arch_plan.get("modules", [])

        for pm in plan_modules:
            mod = ModuleEntry(
                name=pm.get("name", "unnamed_module"),
                description=pm.get("function", ""),
                theoretical_assumptions=pm.get("assumptions", []),
                verification_methods=pm.get("verification_methods", []),
                dependencies=pm.get("dependencies", []),
                is_core=True,  # All planner-generated modules are core by default
            )
            modules.append(mod)

        # If no modules from planner, create from idea components
        if not modules:
            for i, comp in enumerate(arch_plan.get("idea_components", [])):
                mod = ModuleEntry(
                    name=comp.get("name", f"component_{i+1}"),
                    description=comp.get("description", ""),
                    theoretical_assumptions=comp.get("assumptions", []),
                    is_core=True,
                )
                modules.append(mod)

        return modules

    def _modules_from_brief(self, brief_content: str) -> list[ModuleEntry]:
        """Fallback: create a single module from PROJECT_BRIEF content."""
        return [
            ModuleEntry(
                name="core_idea",
                description=f"Core research idea: {self.idea_summary[:200]}",
                theoretical_assumptions=["Extract assumptions from PROJECT_BRIEF and verify"],
                verification_methods=[
                    "Data analysis: examine dataset properties relevant to the assumption",
                    "Statistical test: validate key distributions or correlations in data",
                    "Small-scale probing: run minimal experiment to test feasibility",
                ],
                is_core=True,
            )
        ]

    def _build_phase_violation_correction(self, active_modules: list[ModuleEntry]) -> str:
        """Build correction prompt for phase violation (e.g., training during theory phase)."""
        mod_names = ", ".join(m.name for m in active_modules)
        return (
            f"⚠️ ROADMAP PHASE VIOLATION DETECTED (deviation #{self._deviation_count})\n\n"
            f"Current phase: {self.global_phase.value}\n"
            f"Active modules: {mod_names}\n\n"
            f"You proposed a TRAINING experiment, but the current phase requires "
            f"THEORETICAL VERIFICATION of assumptions first.\n\n"
            f"MANDATORY CORRECTION:\n"
            f"1. Identify which active module's assumption needs verification\n"
            f"2. Design a DATA ANALYSIS experiment (no training) to test that assumption\n"
            f"3. The experiment should produce evidence that the assumption holds or doesn't\n\n"
            f"Example: Instead of training a model, analyze the dataset to check if "
            f"the physical/mathematical assumption is supported by the data."
        )

    def _build_off_roadmap_correction(self, active_modules: list[ModuleEntry]) -> str:
        """Build correction prompt for off-roadmap deviation."""
        mod_list = "\n".join(f"  - {m.name}: {m.description[:100]}" for m in active_modules)
        return (
            f"⚠️ ROADMAP DEVIATION DETECTED (deviation #{self._deviation_count})\n\n"
            f"Your proposed experiment does not relate to any active ROADMAP module.\n\n"
            f"Active modules that need attention:\n{mod_list}\n\n"
            f"MANDATORY: Your next experiment MUST address one of these modules.\n"
            f"Do NOT work on unrelated optimizations or architecture changes."
        )

    # ── Markdown Serialization ──

    def _render_markdown(self) -> str:
        """Render ROADMAP as human/LLM-readable markdown."""
        lines = [
            "# Research Roadmap",
            "",
            f"> Auto-generated by ResearchRoadmap v15",
            f"> Global Phase: **{self.global_phase.value}**",
            "",
            f"## Idea Summary",
            "",
            self.idea_summary,
            "",
            f"## Module Status Overview",
            "",
        ]

        # Summary table
        lines.append("| Module | Phase | Verified | Attempts | Core |")
        lines.append("|--------|-------|----------|----------|------|")
        for m in self.modules:
            v = "✓" if m.is_verified else "○"
            lines.append(f"| {m.name} | {m.phase.value} | {v} | {len(m.attempts)} | {'Yes' if m.is_core else 'No'} |")
        lines.append("")

        # Detailed module sections
        lines.append("## Module Details")
        lines.append("")

        for m in self.modules:
            lines.append(f"### {m.name}")
            lines.append(f"- **Phase**: {m.phase.value}")
            lines.append(f"- **Description**: {m.description}")
            lines.append(f"- **Core**: {'Yes' if m.is_core else 'No'}")

            if m.theoretical_assumptions:
                lines.append(f"- **Theoretical Assumptions**:")
                for a in m.theoretical_assumptions:
                    lines.append(f"  - {a}")

            if m.verification_methods:
                lines.append(f"- **Verification Methods** (need 3-6, max 7):")
                for method in m.verification_methods:
                    tried = any(attempt.method == method for attempt in m.attempts)
                    mark = "→" if not tried else "✗"
                    lines.append(f"  - [{mark}] {method}")

            if m.attempts:
                lines.append(f"- **Verification Attempts**:")
                for a in m.attempts:
                    lines.append(f"  - Cycle {a.cycle}: [{a.result}] {a.method[:100]}")
                    if a.evidence:
                        lines.append(f"    Evidence: {a.evidence[:150]}")

            if m.design_description:
                lines.append(f"- **Design**: {m.design_description[:300]}")

            if m.dead_end_reason:
                lines.append(f"- **Dead End Reason**: {m.dead_end_reason}")

            if m.dependencies:
                lines.append(f"- **Dependencies**: {', '.join(m.dependencies)}")

            if m.validation_metrics:
                lines.append(f"- **Validation Metrics**: {json.dumps(m.validation_metrics)}")

            lines.append(f"- **Last Updated**: Cycle {m.last_updated_cycle}")
            lines.append("")

        lines.append("## Research Methodology Rules")
        lines.append("")
        lines.append("1. **No skipping phases**: Each module must complete theory verification before implementation")
        lines.append("2. **3-6 method verification**: A module needs 3-6 independent verification attempts (max 7) before marking dead_end")
        lines.append("3. **Data analysis first**: Theory verification means data analysis, NOT training")
        lines.append("4. **Systematic failure analysis**: When a module fails, analyze whether it's architecture or idea issue")
        lines.append("5. **Independent validation**: Each module must be validated in isolation before integration")
        lines.append("")

        return "\n".join(lines)

    def _parse_markdown(self, content: str):
        """Parse ROADMAP markdown back into data structures."""
        self.modules = []
        self.idea_summary = ""

        # Parse idea summary
        idea_match = re.search(r"## Idea Summary\s*\n\s*\n(.*?)(?=\n## )", content, re.DOTALL)
        if idea_match:
            self.idea_summary = idea_match.group(1).strip()

        # Parse global phase from header
        phase_match = re.search(r"Global Phase: \*\*(\w+)\*\*", content)
        if phase_match:
            try:
                self.global_phase = GlobalPhase(phase_match.group(1))
            except ValueError:
                self.global_phase = GlobalPhase.THEORY_VERIFICATION

        # Parse module sections
        module_sections = re.split(r"### (.+?)$", content, flags=re.MULTILINE)
        # module_sections: [preamble, name1, body1, name2, body2, ...]
        for i in range(1, len(module_sections), 2):
            name = module_sections[i]
            body = module_sections[i + 1] if i + 1 < len(module_sections) else ""

            mod = ModuleEntry(name=name, description="")

            # Parse phase
            phase_m = re.search(r"\*\*Phase\*\*:\s*(\w+)", body)
            if phase_m:
                try:
                    mod.phase = ModulePhase(phase_m.group(1))
                except ValueError:
                    mod.phase = ModulePhase.THEORY_VERIFICATION

            # Parse description
            desc_m = re.search(r"\*\*Description\*\*:\s*(.+)", body)
            if desc_m:
                mod.description = desc_m.group(1).strip()

            # Parse core
            core_m = re.search(r"\*\*Core\*\*:\s*(Yes|No)", body)
            if core_m:
                mod.is_core = core_m.group(1) == "Yes"

            # Parse assumptions
            assumptions = re.findall(r"- \*\*Theoretical Assumptions\*\*:\s*\n((?:  - .+\n?)+)", body)
            if assumptions:
                mod.theoretical_assumptions = [
                    a.strip().lstrip("- ").strip()
                    for a in assumptions[0].strip().split("\n")
                    if a.strip()
                ]

            # Parse verification methods
            methods_section = re.findall(r"- \*\*Verification Methods\*\*.*?:\s*\n((?:  - .+\n?)+)", body)
            if methods_section:
                mod.verification_methods = []
                for line in methods_section[0].strip().split("\n"):
                    line = line.strip()
                    if line.startswith("-"):
                        # Remove [→] or [✗] prefix
                        clean = re.sub(r"^\[[→✗]\]\s*", "", line.lstrip("- ").strip())
                        mod.verification_methods.append(clean)

            # Parse attempts (allow hyphenated/underscore result values, preserve evidence)
            attempts_section = re.findall(r"- \*\*Verification Attempts\*\*:\s*\n((?:  - .+\n?(?:    .+\n?)*))+", body)
            if attempts_section:
                for attempt_block in attempts_section:
                    lines_list = attempt_block.strip().split("\n")
                    for idx, line in enumerate(lines_list):
                        attempt_m = re.match(r"\s*- Cycle (\d+): \[([\w_-]+)\] (.+)", line)
                        if attempt_m:
                            # Look for evidence on the next line
                            evidence = ""
                            if idx + 1 < len(lines_list):
                                ev_m = re.match(r"\s+Evidence:\s*(.+)", lines_list[idx + 1])
                                if ev_m:
                                    evidence = ev_m.group(1).strip()
                            mod.attempts.append(VerificationAttempt(
                                method=attempt_m.group(3).strip(),
                                result=attempt_m.group(2),
                                evidence=evidence,
                                cycle=int(attempt_m.group(1)),
                            ))

            # Parse dead end reason
            dead_m = re.search(r"\*\*Dead End Reason\*\*:\s*(.+)", body)
            if dead_m:
                mod.dead_end_reason = dead_m.group(1).strip()

            # Parse design
            design_m = re.search(r"\*\*Design\*\*:\s*(.+)", body)
            if design_m:
                mod.design_description = design_m.group(1).strip()

            # Parse dependencies
            deps_m = re.search(r"\*\*Dependencies\*\*:\s*(.+)", body)
            if deps_m:
                mod.dependencies = [d.strip() for d in deps_m.group(1).split(",") if d.strip()]

            # Parse last updated
            updated_m = re.search(r"\*\*Last Updated\*\*:\s*Cycle (\d+)", body)
            if updated_m:
                mod.last_updated_cycle = int(updated_m.group(1))

            self.modules.append(mod)

        logger.info(f"ROADMAP loaded: {len(self.modules)} modules from file")
