"""
AutoResearcher Core Loop

The autonomous THINK → EXECUTE → VERIFY → REFLECT cycle that drives experiments 24/7.

Pipeline:
  THINK   → Analyze state, form hypothesis, plan experiment
  EXECUTE → Dispatch worker agent to implement and run experiment
  VERIFY  → Reverse-engineer whether each module actually worked
  REFLECT → Evaluate results, update memory, decide next action
"""

import os
import re
import sys
import math
import time
import json
import signal
import argparse
import logging
from pathlib import Path
from typing import Optional

from .memory import MemoryManager
from .monitor import ExperimentMonitor
from .agents import AgentDispatcher
from .obsidian import ObsidianExporter
from .tools import ToolRegistry
from .verifier import ExperimentVerifier
from .visual_analyzer import VisualAnalyzer
from .domain_knowledge import DomainKnowledgeMixin
from .constraint_engine import (
    StrategyConstraintEngine,
    ContextPruner,
)
from .simulation_sandbox import SimulationSandbox

logger = logging.getLogger("autoresearcher")

# Numerical safety epsilon
_EPS = 1e-8


class ResearchLoop(DomainKnowledgeMixin):
    """Main autonomous research loop.

    Implements the THINK → EXECUTE → VERIFY → REFLECT cycle:
    - THINK: Analyze state, form hypothesis, plan experiment
    - EXECUTE: Dispatch worker agent to implement and run experiment
    - VERIFY: Reverse-engineer whether each module actually worked
    - REFLECT: Evaluate results, update memory, decide next action
    """

    def __init__(self, config: dict, project_dir: str):
        self.config = config
        self.project_dir = Path(project_dir).resolve()
        self.workspace = self.project_dir / config.get("project", {}).get("workspace", "workspace")
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.state_path = self.workspace / "state.json"

        # Ensure file logging is always set up, regardless of how the loop is started
        self._setup_file_logging()

        # Core components
        self.memory = MemoryManager(
            project_dir=self.project_dir,
            brief_max=config.get("memory", {}).get("brief_max_chars", 3000),
            log_max=config.get("memory", {}).get("log_max_chars", 2000),
            milestone_max=config.get("memory", {}).get("milestone_max_chars", 1200),
            max_recent=config.get("memory", {}).get("max_recent_entries", 15),
            workspace=self.workspace,
        )
        self.monitor = ExperimentMonitor(
            poll_interval=config.get("monitor", {}).get("poll_interval", 900),
            zero_llm=config.get("monitor", {}).get("zero_llm", True),
            max_runtime_hours=config.get("monitor", {}).get("max_runtime_hours", None),
        )
        agent_config = config.get("agent", {})
        model = agent_config.get("model", "auto")
        provider = agent_config.get("provider", "glm_token_plan")
        
        logger.info(f"Agent Configuration: provider={provider}, model={model}")
        logger.debug(f"Full agent config: {agent_config}")
        
        self.tools = ToolRegistry(self.workspace, memory=self.memory)
        self.dispatcher = AgentDispatcher(
            model=model,
            provider=provider,
            max_steps=agent_config.get("max_steps_per_cycle", 3),
            tools=self.tools,
        )
        self.obsidian = ObsidianExporter(config=config, project_dir=self.project_dir)

        # VERIFY phase: module-level result verification
        # v16.1: Use hardcoded thresholds (AdaptiveThresholds removed)
        self.verifier = ExperimentVerifier(
            project_dir=self.project_dir,
            workspace=self.workspace,
            thresholds={"severe_degradation": 0.35, "improvement_threshold": 0.005},
        )

        # VISUAL ANALYSIS: inference + multimodal diagnosis when training is stuck
        self.visual_analyzer = VisualAnalyzer(
            project_dir=self.project_dir,
            workspace=self.workspace,
            config=config,
            tools_registry=self.tools,
        )

        # State
        self.cycle_count = self._load_cycle_counter()
        self.max_cycles = config.get("agent", {}).get("max_cycles", -1)
        self.cooldown = config.get("agent", {}).get("cooldown_interval", 300)
        self.no_progress_fallback_threshold = config.get("agent", {}).get("no_progress_fallback_threshold", 3)
        self._running = True
        self._no_progress_streak = 0
        self._last_no_progress_signature = ""
        self._consecutive_wait_count = 0
        self._hard_gate_consecutive_blocks = 0  # track HARD GATE dead loops
        self._max_consecutive_waits = config.get("agent", {}).get("max_consecutive_waits", 3)
        # Repeated issue tracking for error escalation
        # Uses sliding window: issue_signature → list of recent cycle numbers
        self._audit_issue_history: dict[str, list[int]] = {}  # sig → [cycle_num, ...]
        self._audit_escalation_threshold = config.get("agent", {}).get("audit_escalation_threshold", 3)
        self._audit_sliding_window = 10  # How many recent cycles to consider

        # ── Metric-based progress tracking (for visual analysis trigger) ──
        # Visual analysis should fire when METRICS stop improving,
        # not just when experiments fail to launch.
        self._best_metric_ever: float = float('inf')  # Best val_MAE ever seen
        self._metric_no_progress_streak: int = 0       # Cycles since last metric improvement
        self._visual_trigger_threshold: int = (config or {}).get(
            "visual_analysis", {}
        ).get("trigger_threshold", 5)  # Sync with VisualAnalyzer default
        self._consecutive_audit_directives: int = 0     # Track audit death loop (Fix 5)

        # ── Fix 1: Output quality awareness ──
        # Track per-domain metrics to detect when agent produces "successful bad results"
        self._best_domain_metrics: dict[str, float] = {}  # domain → best MAE

        # ── Fix 3: Strategic abandonment ──
        # Track whether the agent is stuck in a research direction
        self._direction_change_threshold: int = 3     # Force paper research after N stagnations

        # ── v14: Architecture-level stagnation (independent of direction stagnation) ──
        # Tracks whether the agent is stuck patching the SAME architecture.
        # Unlike direction stagnation, this is NOT reset by paper_research —
        # only reset when a genuinely different architecture is detected.
        self._architecture_stagnation_threshold: int = 5    # Trigger architecture switch after N cycles
        self._architecture_survey_done: bool = False        # Whether architecture survey has been completed
        self._architecture_survey_path = self.workspace / "ARCHITECTURE_SURVEY.md"

        # ── Fix 2: Infrastructure degradation ──
        self._infra_failure_streak: int = 0  # Consecutive infrastructure failures
        self._infra_degradation_threshold: int = 3  # Skip VERIFY after N infra failures

        # ── Phase 4: Failed-launch forced re-dispatch ──
        # When THINK plans an experiment but EXECUTE never launches it
        # (convergence_failed or experiment_launched=False without a tool
        # error), this counter tracks consecutive failures. After 2, the next
        # cycle's action is forced to a 'fix + launch' task; after 3, the loop
        # pauses for human intervention instead of burning more quota.
        # Monotonic — only resets to 0 on a genuine launch.
        self._consecutive_failed_launches: int = 0

        # ── Fix B: Audit enforcement counters ──
        self._audit_enforcement: dict[str, int] = {}

        # ── v18: Signal Arbitration System ──
        # The single decision arbiter that collects enforcement signals from
        # all subsystems (audit, constraint, launch, stagnation) and produces
        # a CycleDirective: either a forced_action (bypassing the LLM) or a
        # budgeted context for normal THINK. Replaces the scattered advisory
        # directive files and the positional enforcement chain.
        from .signal_arbiter import SignalArbiter
        arbiter_budget = self.config.get("context_budget", 12000)
        self.arbiter = SignalArbiter(budget_chars=arbiter_budget)

        # Phase 2: deterministic garbage collector
        from .garbage_collector import GarbageCollector
        self._gc = GarbageCollector(self.project_dir)

        # ── Constraint Engine (v10 → v16.1): LLM behavior control ──
        # v16.1: Removed PlannerChecker, QuickBenchmark, AdaptiveThresholds, ImplementationTracker
        self.strategy_engine = StrategyConstraintEngine(self.project_dir, self.workspace)
        self.context_pruner = ContextPruner()

        # ── Simulation Sandbox (v11): Model evaluation engine ──
        self.sandbox = SimulationSandbox(self.project_dir, self.workspace, config=config)

        # ── Research Roadmap (v15): Structured research methodology ──
        from .research_roadmap import ResearchRoadmap
        self.roadmap = ResearchRoadmap(self.workspace)
        self._roadmap_initialized = False  # Set True after first generate_from_brief()
        self._phase_violation_count = 0    # Consecutive phase violations in THINK

        # Graceful shutdown
        try:
            signal.signal(signal.SIGTERM, self._handle_signal)
            signal.signal(signal.SIGINT, self._handle_signal)
        except ValueError:
            logger.warning("Signals not available (not running in main thread)")

    def _setup_file_logging(self):
        """Ensure autoresearcher log file exists regardless of how the loop is started.

        When launched via `python -m core.loop`, main() already sets up FileHandler.
        But when created directly (e.g., by Claude Code skill), logging would only go
        to console. This method ensures a FileHandler is always attached to the
        'autoresearcher' root logger.
        """
        log_path = self.project_dir / "autoresearcher.log"

        # Check if a FileHandler for our log file already exists on any logger
        resolved = log_path.resolve()
        for logger_candidate in (logging.getLogger("autoresearcher"), logging.getLogger()):
            for handler in logger_candidate.handlers:
                if isinstance(handler, logging.FileHandler):
                    try:
                        if Path(handler.baseFilename) == resolved or Path(handler.baseFilename).resolve() == resolved:
                            return  # Already set up
                    except Exception:
                        pass

        # Add FileHandler
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s")
        )
        logging.getLogger().addHandler(file_handler)
        logger.info(f"File logging enabled: {log_path}")

    def run(self):
        """Main entry point. Runs the THINK → EXECUTE → VERIFY → REFLECT loop."""
        logger.info(f"AutoResearcher starting | project={self.project_dir} | cycle={self.cycle_count}")

        # v18: Restore deferred signals from the previous run's state.json
        restored_state = self._load_state()
        if restored_state.get("signal_backlog"):
            self.arbiter.load_backlog(restored_state["signal_backlog"])
            logger.info(f"Restored {len(self.arbiter._backlog)} deferred signals from state.json")

        while self._running:
            # Phase 1: Stop when all goals are achieved
            if self.cycle_count > 1:
                try:
                    if self._goal_achieved():
                        self._running = False
                        self.memory.log_milestone(
                            "🎯 ALL TARGETS ACHIEVED. Agent stopping. "
                            "Review outputs/ for final results."
                        )
                        break
                except Exception:
                    pass  # goal check failure should never block the loop
            if self.max_cycles > 0 and self.cycle_count >= self.max_cycles:
                logger.info(f"Reached max cycles ({self.max_cycles}). Stopping.")
                break

            self.cycle_count += 1
            logger.info(f"=== Cycle {self.cycle_count} ===")
            # Save counter immediately at cycle start for crash recovery
            self._save_cycle_counter()

            try:
                # Keep leader context bounded to one cycle.
                self.dispatcher.reset_leader_history()

                # Check for human directive
                directive = self._consume_directive()
                self._update_state(
                    {
                        "cycle": self.cycle_count,
                        "status": "planning",
                        "updated_at": time.time(),
                        "last_directive": directive or "",
                    }
                )

                # MANDATORY HALT: If directive contains ⛔ STOP, force the
                # agent to execute only the directive's tasks (no training).
                if directive and "⛔ STOP" in directive:
                    logger.info("⛔ MANDATORY HALT detected in directive. Forcing directive-only execution.")
                    think_result = {
                        "action": "experiment",
                        "agent": "code",
                        "task": (
                            f"⛔ MANDATORY HALT — You MUST NOT launch any training.\n\n"
                            f"Execute the following tasks IN ORDER. Do NOT skip any task.\n\n"
                            f"--- HUMAN DIRECTIVE ---\n{directive}\n--- END DIRECTIVE ---\n\n"
                            f"Complete ALL tasks listed in the directive before doing anything else."
                        ),
                    }
                else:
                    # DATASET UNDERSTANDING: First cycle or when manifest missing
                    if self.cycle_count == 1 or not self._dataset_manifest_exists():
                        logger.info("DATASET UNDERSTANDING phase — scanning data/ directory")
                        self._run_dataset_understanding()

                    # ── ROADMAP INIT (v15): Generate research roadmap on first cycle ──
                    if not self._roadmap_initialized:
                        self._init_research_roadmap()

                    # THINK: Analyze and plan
                    think_result = self._think(directive)

                    # ── v18: Signal Arbitration (THE single enforcement layer) ──
                    # Replaces the former 7-layer chain:
                    #   _check_phase_blocked → _enforce_roadmap_alignment →
                    #   _apply_no_progress_fallback → _enforce_launch_after_failure →
                    #   _enforce_audit_findings → _arbitrate_cycle
                    # Now: arbiter collects all enforcement signals (launch
                    # failures, audit escalations, forbidden constraints) and
                    # produces one CycleDirective. Code-scan gates (phase,
                    # roadmap) still run independently because they inspect
                    # source files, not counters.
                    directive_v18 = self._arbitrate_cycle(think_result)
                    if directive_v18.forced_action:
                        think_result = {
                            "action": directive_v18.forced_action,
                            "task": directive_v18.forced_task or think_result.get("task", ""),
                            "reason": directive_v18.forced_reason or "",
                        }
                        logger.info(
                            f"SIGNAL ARBITER override: action={directive_v18.forced_action}"
                        )
                    elif think_result.get("action") == "experiment":
                        # Code-scan gates (kept — they inspect files, not counters)
                        blocked, block_reason = self._check_phase_blocked(think_result)
                        if blocked:
                            logger.warning(f"PHASE GATE BLOCKED: {block_reason[:200]}")
                            ps = self._load_phase_status()
                            current_phase = ps.get("phases", {}).get(ps.get("current_phase", ""), {})
                            focus = current_phase.get("focus_methods", ["data_analysis"])
                            think_result["action"] = "paper_research"
                            think_result["task"] = (
                                f"{block_reason}\n\n"
                                f"Research alternative approaches using these methods: {', '.join(focus)}.\n"
                            )
                            self.memory.log_decision(f"[PHASE GATE] Blocked: {block_reason[:150]}")
                        think_result = self._enforce_roadmap_alignment(think_result)

                # ── Phase 4: PAUSE-HUMAN — stop the loop and surface for inspection ──
                # Triggered by _enforce_launch_after_failure after 3 consecutive
                # failed launches. Stops burning quota on a stuck pattern and
                # requires human intervention to resume.
                if think_result.get("action") == "pause_human":
                    logger.error(
                        f"⛔ PAUSE-HUMAN: {think_result.get('reason', 'no reason given')}"
                    )
                    self._update_state({
                        "cycle": self.cycle_count,
                        "status": "pause_human",
                        "updated_at": time.time(),
                        "pause_reason": think_result.get("reason", ""),
                        "suggested_next_step": (
                            "Inspect the agent's recent cycles to understand why "
                            "launch_experiment is never called. Common causes: "
                            "(1) the code agent's turn budget is exhausted by "
                            "exploration, (2) the training script has an error "
                            "the agent can't fix, (3) a phase gate is blocking "
                            "training. Resume with a directive after fixing."
                        ),
                    })
                    self.memory.log_decision(
                        f"PAUSE-HUMAN: {think_result.get('reason', '')}"
                    )
                    self._running = False
                    break

                if think_result.get("action") == "wait":
                    self._consecutive_wait_count += 1
                    logger.info(
                        f"THINK decided to wait ({self._consecutive_wait_count}/{self._max_consecutive_waits})."
                    )

                    # If no experiment is running and we've waited too many times,
                    # force the agent to take action instead of idling.
                    if (
                        self._consecutive_wait_count >= self._max_consecutive_waits
                        and not self.monitor.has_active_experiments()
                    ):
                        reason = (
                            f"Forced experiment: {self._consecutive_wait_count} consecutive waits "
                            "with no active experiments. Agent must propose a concrete experiment."
                        )
                        logger.warning(reason)
                        self.memory.log_decision(reason)
                        # Override wait → experiment
                        think_result = {
                            "action": "experiment",
                            "reason": reason,
                            "agent": "code",
                            "task": (
                                "The agent has been idling with no active experiments. "
                                "You MUST propose and launch a concrete experiment now. "
                                "Read workspace/MEMORY_LOG.md for context and propose a "
                                "specific improvement to try."
                            ),
                        }
                    else:
                        self._update_state(
                            {
                                "cycle": self.cycle_count,
                                "status": "waiting",
                                "updated_at": time.time(),
                                "suggested_next_step": think_result.get("reason", ""),
                            }
                        )
                        self._smart_cooldown()
                        continue

                # PAPER RESEARCH: Execute deep literature search instead of experiment
                if think_result.get("action") == "paper_research":
                    self._consecutive_wait_count = 0
                    logger.info("PAPER RESEARCH triggered — executing deep literature search.")
                    self._update_state(
                        {
                            "cycle": self.cycle_count,
                            "status": "paper_research",
                            "updated_at": time.time(),
                        }
                    )
                    execute_result = self._execute_paper_research(think_result)

                    # VERIFY: Check paper research produced useful output
                    verify_report = self._verify(self.cycle_count, think_result, execute_result)
                    execute_result["verify_report"] = verify_report.to_dict()

                    # REFLECT on research findings (no training to monitor)
                    reflect_result = self._reflect(execute_result, verify_report=verify_report)
                    self._update_state(
                        {
                            "cycle": self.cycle_count,
                            "updated_at": time.time(),
                            "last_milestone": reflect_result.get("milestone", ""),
                            "last_decision": reflect_result.get("decision", ""),
                            "suggested_next_step": reflect_result.get("decision", "")
                            or reflect_result.get("reason", ""),
                        }
                    )
                    self._record_cycle_outcome(think_result, execute_result, reflect_result,
                                                verify_report_dict=verify_report.to_dict() if verify_report else None)
                    self._refresh_obsidian(reflect_result=reflect_result, directive=directive)
                    self._gc.run()  # Phase 2: deterministic GC
                    # Post-reflect code review: learn from mistakes
                    self._post_reflect_code_review(execute_result, reflect_result, verify_report)
                    # Paper research is meaningful work — persist cycle counter
                    self._save_cycle_counter()
                    continue

                # ARCHITECTURE SWITCH (v14): Execute architecture switch instead of experiment
                # This is triggered when the architecture stagnation threshold is reached.
                # The agent researches alternative architectures AND starts implementing.
                if think_result.get("action") == "architecture_switch":
                    self._consecutive_wait_count = 0
                    logger.info(
                        f"ARCHITECTURE SWITCH triggered — researching alternatives to "
                        f"'{self._current_architecture_name}'."
                    )
                    self._update_state(
                        {
                            "cycle": self.cycle_count,
                            "status": "architecture_switch",
                            "updated_at": time.time(),
                        }
                    )
                    # Architecture switch is dispatched as paper_research (uses researcher agent)
                    # but with a specific architecture-switch task.
                    execute_result = self._execute_paper_research(think_result)

                    verify_report = self._verify(self.cycle_count, think_result, execute_result)
                    execute_result["verify_report"] = verify_report.to_dict()

                    reflect_result = self._reflect(execute_result, verify_report=verify_report)
                    self._update_state(
                        {
                            "cycle": self.cycle_count,
                            "updated_at": time.time(),
                            "last_milestone": reflect_result.get("milestone", ""),
                            "last_decision": reflect_result.get("decision", ""),
                            "suggested_next_step": reflect_result.get("decision", "")
                            or reflect_result.get("reason", ""),
                        }
                    )
                    # Record as paper_research for outcome tracking purposes
                    think_result_for_record = dict(think_result)
                    think_result_for_record["action"] = "paper_research"
                    self._record_cycle_outcome(
                        think_result_for_record, execute_result, reflect_result,
                        verify_report_dict=verify_report.to_dict() if verify_report else None
                    )
                    self._refresh_obsidian(reflect_result=reflect_result, directive=directive)
                    self._gc.run()  # Phase 2: deterministic GC
                    self._post_reflect_code_review(execute_result, reflect_result, verify_report)
                    self._save_cycle_counter()
                    continue

                # ── GATE PIPELINE (v12.4): ordered priority ──
                # Gate 1 (PRE-VERIFY):  critical preconditions (synthetic data, missing data, broken imports)
                # Gate 2 (CODE REVIEW): architectural / code defects
                # Gate 3 (FALSIFIABILITY): hypothesis quality (soft gate, never blocks)
                #
                # A hard-gate (full rewrite of think_result) causes subsequent gates to
                # skip entirely, preventing one gate from overwriting another's output.
                _gate_blocked = False  # set to True when any hard-gate fires

                # ── Gate 1: PRE-VERIFY ──
                pre_verify_report = self._pre_verify(self.cycle_count, think_result)
                critical_pre_issues = pre_verify_report.critical_failures
                if critical_pre_issues:
                    issues_text = "; ".join(c.detail for c in critical_pre_issues)
                    logger.warning(
                        f"PRE-VERIFY blocked execution: {issues_text}"
                    )
                    think_result = {
                        "action": "experiment",
                        "agent": "code",
                        "task": (
                            f"⛔ PRE-VERIFY BLOCKED EXPERIMENT\n\n"
                            f"The following CRITICAL issues must be fixed BEFORE any training:\n\n"
                            + "\n".join(f"- {c.detail}" for c in critical_pre_issues)
                            + "\n\n## Mandatory Actions:\n"
                            "1. Fix ALL issues listed above\n"
                            "2. Ensure training scripts use the project's real dataset class (NOT synthetic/random data)\n"
                            "3. Run a 2-step dry-run to verify data loads correctly\n"
                            "4. Do NOT launch real training until pre-verify passes\n"
                        ),
                    }
                    _gate_blocked = True

                # ── Gate 2: PRE-EXECUTE CODE REVIEW (v12.3+) ──
                # Two-phase code review BEFORE training:
                #   Phase 1: Zero-LLM regex checks (fast, free, catches known anti-patterns)
                #   Phase 2: LLM semantic review (catches logic bugs, design flaws)
                if not _gate_blocked and think_result.get("action") == "experiment":
                    code_review_warnings = self._pre_execute_code_review(think_result)

                    if not code_review_warnings:
                        # Code review passed — reset dead-loop counter
                        self._hard_gate_consecutive_blocks = 0
                    else:
                        high_issues = [w for w in code_review_warnings if w["severity"] == "HIGH"]

                        if high_issues:
                            self._hard_gate_consecutive_blocks += 1
                            
                            # v16.1: Phase-aware downgrade — only downgrade if phase is VALIDATED
                            # v16 bug: 3 consecutive blocks → downgrade allowed training during Phase 1
                            # Fix: Check phase status before downgrading
                            ps = self._load_phase_status()
                            current_phase = ps.get("phases", {}).get(ps.get("current_phase", ""), {})
                            phase_status = current_phase.get("status", "PENDING")
                            phase_validated = (phase_status == "VALIDATED")
                            
                            # ── Anti-deadloop: after 2 consecutive HARD blocks, downgrade to SOFT ──
                            # BUT ONLY if phase is VALIDATED (training is legitimate)
                            if self._hard_gate_consecutive_blocks > 2 and phase_validated:
                                logger.warning(
                                    f"HARD GATE downgraded to SOFT after "
                                    f"{self._hard_gate_consecutive_blocks} consecutive blocks — "
                                    f"phase is VALIDATED, training is legitimate"
                                )
                                self._hard_gate_consecutive_blocks = 0  # reset
                                # Fall through to SOFT GATE below
                            elif self._hard_gate_consecutive_blocks > 2 and not phase_validated:
                                # v16.1: Phase not validated — do NOT downgrade, keep blocking
                                logger.warning(
                                    f"HARD GATE NOT downgraded: phase '{ps.get('current_phase')}' "
                                    f"status is {phase_status}, not VALIDATED. "
                                    f"Continuing to block training (streak={self._hard_gate_consecutive_blocks})"
                                )
                                self._hard_gate_consecutive_blocks = 2  # cap at 2 to avoid overflow
                            else:
                                # ── HARD GATE: HIGH severity blocks execution entirely ──
                                logger.warning(
                                    f"PRE-EXECUTE CODE REVIEW HARD GATE: {len(high_issues)} HIGH issue(s), "
                                    f"blocking execution (streak={self._hard_gate_consecutive_blocks})"
                                )
                                think_result = {
                                    "action": "experiment",
                                    "agent": "code",
                                    "task": (
                                        f"⛔ PRE-EXECUTE CODE REVIEW BLOCKED TRAINING\n\n"
                                        f"The following CRITICAL architectural issues must be fixed "
                                        f"BEFORE any training:\n\n"
                                        + "\n".join(
                                            f"- [{w['severity']}] {w['detail']}"
                                            for w in high_issues
                                        )
                                        + "\n\n## Mandatory Actions:\n"
                                        "1. Fix ALL HIGH severity issues listed above\n"
                                        "2. Verify the model file is syntactically correct (can import)\n"
                                        "3. Re-run will auto-check after fixes\n"
                                        "4. Do NOT launch real training until code review passes\n"
                                    ),
                                }
                                _gate_blocked = True
                        if not _gate_blocked and code_review_warnings:
                            # ── SOFT GATE: MEDIUM/LOW issues injected as mandatory fix ──
                            review_prompt = (
                                "PRE-EXECUTE CODE REVIEW (v12.3) — ISSUES DETECTED:\n\n"
                                + "\n".join(
                                    f"- [{w['severity']}] {w['detail']}"
                                    for w in code_review_warnings
                                )
                                + "\n\nYou MUST address these issues BEFORE launching training. "
                                + "Fix the model architecture, then re-verify. "
                                + "Do NOT proceed with training until these are resolved.\n\n"
                            )
                            think_result["task"] = review_prompt + think_result.get("task", "")
                            think_result["_soft_gate_injected"] = True  # skip falsifiability to avoid task bloat
                            logger.warning(
                                f"PRE-EXECUTE CODE REVIEW: {len(code_review_warnings)} issue(s) injected into task"
                            )

                # ── Gate 3: FALSIFIABLE HYPOTHESIS CHECK ──
                # Always runs (soft gate). Skipped when a hard-gate already fired to
                # avoid wrapping the fix-task in hypothesis boilerplate.
                if not _gate_blocked and not think_result.get("_soft_gate_injected"):
                    hypothesis = think_result.get("hypothesis", "")
                    success_criteria = think_result.get("success_criteria", "")

                    non_falsifiable_warning = None
                    if not hypothesis or len(hypothesis.strip()) < 10:
                        non_falsifiable_warning = (
                            "NO HYPOTHESIS: The experiment has no stated hypothesis. "
                            "Every experiment MUST state what it expects to learn and what would prove it wrong."
                        )
                    elif "improve" in hypothesis.lower() and "if" not in hypothesis.lower():
                        non_falsifiable_warning = (
                            f"NON-FALSIFIABLE HYPOTHESIS: '{hypothesis[:100]}' is vague. "
                            f"A hypothesis must be structured as: 'If we change X, then Y should improve "
                            f"because Z. If Y does NOT improve (or gets worse), the hypothesis is wrong.' "
                            f"State the SPECIFIC change, the EXPECTED effect, and the FAILURE condition."
                        )
                    elif not success_criteria or len(success_criteria.strip()) < 10:
                        non_falsifiable_warning = (
                            "NO SUCCESS CRITERIA: Without concrete success criteria, you cannot "
                            "determine whether the experiment succeeded or failed. "
                            "Example: 'worst_domain_MAE < 0.30' (pass) vs '>= 0.30' (fail)."
                        )

                    if non_falsifiable_warning:
                        logger.warning(f"FALSIFIABILITY CHECK: {non_falsifiable_warning}")
                        falsify_prefix = (
                            f"FALSIFIABILITY: {non_falsifiable_warning}\n"
                            f"Add HYPOTHESIS/SUCCESS/FAILURE comments to training script.\n\n"
                        )
                        think_result["task"] = falsify_prefix + think_result.get("task", "")

                # EXECUTE: Run the plan
                self._consecutive_wait_count = 0
                execute_result = self._execute(think_result)

                # Phase 4: update the consecutive-failed-launch counter so the
                # next cycle's THINK can force a re-dispatch or pause_human.
                self._update_launch_counter(execute_result)

                if execute_result.get("experiment_launched"):
                    self._update_state(
                        {
                            "cycle": self.cycle_count,
                            "status": "running",
                            "pid": execute_result.get("pid"),
                            "log_file": execute_result.get("log_file", ""),
                            "started_at": time.time(),
                            "updated_at": time.time(),
                        }
                    )
                    # Monitor experiment (zero LLM cost)
                    monitor_result = self._monitor_experiment(execute_result)
                    execute_result["training_logs"] = monitor_result.get("log_tail", "")
                    execute_result["final_metrics"] = monitor_result.get("metrics", {})
                    self._update_state(
                        {
                            "status": "completed",
                            "pid": execute_result.get("pid"),
                            "log_file": execute_result.get("log_file", ""),
                            "updated_at": time.time(),
                            "last_training_logs": monitor_result.get("log_tail", ""),
                            "last_metrics": monitor_result.get("metrics", {}),
                            "elapsed_hours": monitor_result.get("elapsed_hours"),
                        }
                    )

                # VERIFY: Reverse-engineer whether each module actually worked
                verify_report = self._verify(self.cycle_count, think_result, execute_result)
                execute_result["verify_report"] = verify_report.to_dict()

                # VISUAL ANALYSIS: When METRICS stop improving for N consecutive cycles
                # (OR when experiments keep failing to launch), run inference → multimodal
                # visual analysis to diagnose WHY the model is failing.
                # This catches problems invisible to numeric metrics alone
                # (e.g., uniform depth maps, domain collapse, structural failures).
                visual_analysis_result = None

                # Fix 3 (结果分析): Force visual analysis when domain-specific metrics
                # are severely degraded — the agent MUST look at its own outputs.
                domain_metrics = (execute_result.get("final_metrics") or {})
                force_visual = False
                severe_threshold = 0.35  # v16.1: hardcoded (AdaptiveThresholds removed)
                for key, val in domain_metrics.items():
                    if key.startswith("MAE_"):
                        try:
                            val = float(val)
                            if val > severe_threshold:
                                force_visual = True
                                logger.warning(
                                    f"FORCE VISUAL ANALYSIS: {key} = {val:.4f} > {severe_threshold:.4f}. "
                                    f"Agent must visually inspect predictions."
                                )
                                break
                        except (TypeError, ValueError):
                            pass

                # Use MAX of launch-streak and metric-streak so either condition triggers analysis.
                effective_streak = max(self._no_progress_streak, self._metric_no_progress_streak)
                if force_visual or self.visual_analyzer.should_trigger(effective_streak):
                    logger.warning(
                        f"VISUAL ANALYSIS TRIGGERED: effective streak={effective_streak} "
                        f"(launch={self._no_progress_streak}, metric={self._metric_no_progress_streak}), "
                        f"threshold={self.visual_analyzer.trigger_threshold}. "
                        f"Running inference + multimodal diagnosis..."
                    )
                    visual_analysis_result = self.visual_analyzer.analyze(
                        no_progress_streak=effective_streak,
                        experiment_info={
                            "cycle": self.cycle_count,
                            "streak": effective_streak,
                            "best_metric_ever": self._best_metric_ever,
                            "current_metric": (execute_result.get("final_metrics") or {}).get(
                                "val_MAE", execute_result.get("final_metrics", {}).get(
                                    "val_MAE_overall", "N/A")
                            ),
                            "model": think_result.get("task", "")[:200],
                        },
                    )
                    if visual_analysis_result.triggered and visual_analysis_result.diagnosis:
                        logger.info(
                            f"Visual analysis found {len(visual_analysis_result.diagnosis)} issue(s), "
                            f"severity={visual_analysis_result.severity}"
                        )
                        # Log diagnosis to memory so it persists across cycles
                        for diag in visual_analysis_result.diagnosis[:3]:
                            desc = diag.get("description", str(diag)) if isinstance(diag, dict) else str(diag)
                            self.memory.log_active_problem(
                                f"[VISUAL Cycle {self.cycle_count}] {desc[:300]}"
                            )
                        if visual_analysis_result.recommended_actions:
                            self.memory.log_decision(
                                f"[VISUAL] Actions: {'; '.join(visual_analysis_result.recommended_actions[:3])}"
                            )

                # REFLECT: Evaluate and update (now with VERIFY diagnosis)
                reflect_result = self._reflect(
                    execute_result, verify_report=verify_report,
                    visual_analysis_result=visual_analysis_result,
                )
                self._update_state(
                    {
                        "cycle": self.cycle_count,
                        "updated_at": time.time(),
                        "last_milestone": reflect_result.get("milestone", ""),
                        "last_decision": reflect_result.get("decision", ""),
                        "suggested_next_step": reflect_result.get("decision")
                        or reflect_result.get("reason")
                        or reflect_result.get("task", ""),
                        "last_error": "",
                    }
                )

                # ── v16: Update phase status from experiment results ──
                # Auto-compare experiment results against phase targets
                try:
                    final_metrics = execute_result.get("final_metrics") or {}
                    if final_metrics:
                        self._update_phase_status_from_results(final_metrics)
                except Exception as e:
                    logger.debug(f"Phase status update skipped: {e}")
                self._record_cycle_outcome(think_result, execute_result, reflect_result,
                                            verify_report_dict=verify_report.to_dict())

                self._refresh_obsidian(reflect_result=reflect_result, directive=directive)

                # Post-reflect code review: learn from mistakes
                self._post_reflect_code_review(execute_result, reflect_result, verify_report)

                # Only count as meaningful cycle if experiment was launched or progress was made
                if execute_result.get("experiment_launched") or reflect_result.get("milestone"):
                    self._save_cycle_counter()

                # AUTO CODE-CLEANUP: Check trigger conditions after each cycle
                self._gc.run()  # Phase 2: deterministic GC

                # AUDIT ESCALATION: Check if VERIFY failures are recurring
                audit_issues = [
                    f"[{c.category}] {c.name}: {c.detail}"
                    for c in verify_report.all_failures
                ]
                escalated_issues = self._check_audit_escalation(audit_issues)
                if escalated_issues:
                    self._handle_escalated_issues(escalated_issues)

                # SMART CIRCUIT BREAKER: If pre-verify AND verify both have
                # critical failures, force paper_research next cycle instead
                # of repeating the same failed experiment pattern.
                if (
                    critical_pre_issues
                    and verify_report.critical_failures
                    and self._no_progress_streak >= 2
                ):
                    reason = (
                        f"SMART CIRCUIT BREAKER: Pre-verify AND verify both have critical "
                        f"failures for {self._no_progress_streak} consecutive cycles. "
                        f"Forcing paper research to find new approaches."
                    )
                    logger.warning(reason)
                    self.memory.log_decision(reason)
                    # Write a directive for the next cycle to do paper research
                    directive_path = self.workspace / "DIRECTIVE.md"
                    directive_path.write_text(
                        f"🔴 CIRCUIT BREAKER TRIGGERED\n\n"
                        f"The agent has been stuck for {self._no_progress_streak} cycles "
                        f"with critical infrastructure failures.\n\n"
                        f"DO NOT attempt another experiment. Instead:\n"
                        f"1. Read the current code and identify ALL issues\n"
                        f"2. Read MEMORY_LOG.md for dead ends and active problems\n"
                        f"3. Search for papers on the specific failing component\n"
                        f"4. Write a comprehensive diagnosis to workspace/diagnosis.md\n"
                    )

            except Exception as e:
                err_msg = str(e)
                logger.error(f"Cycle {self.cycle_count} failed: {e}", exc_info=True)
                self.memory.log_decision(f"Cycle {self.cycle_count} error: {err_msg[:200]}")
                self._update_state(
                    {
                        "cycle": self.cycle_count,
                        "status": "error",
                        "updated_at": time.time(),
                        "last_error": err_msg[:500],
                    }
                )
                # v12.1: Don't backoff for quota errors — retrying won't help
                is_quota_error = (
                    "insufficient_quota" in err_msg
                    or "quota" in err_msg.lower()
                )
                if is_quota_error:
                    logger.warning(
                        f"API quota exhausted — pausing for 30 min before retry. "
                        f"Cycle state preserved for resumption."
                    )
                    # Save cycle state for resumption
                    self._save_cycle_counter()
                    # Quota recovery: wait longer (30 min) then retry instead of giving up
                    self._update_state({
                        "cycle": self.cycle_count,
                        "status": "quota_recovery",
                        "updated_at": time.time(),
                        "last_error": err_msg[:500],
                    })
                    time.sleep(1800)  # 30 min cooldown for quota recovery
                    continue  # Retry the cycle instead of breaking
                self._cooldown_after_error()

        logger.info("AutoResearcher stopped.")

    def _think(self, directive: Optional[str] = None) -> dict:
        """THINK phase: analyze current state and plan next experiment."""
        logger.info("THINK phase starting...")

        context = {
            "brief": self.memory.get_brief(),
            "memory_log": self.memory.get_log(),
            "cycle": self.cycle_count,
            "directive": directive,
            "workspace_dir": str(self.workspace),
        }

        # ── v16.1: PERSISTENT CONSTRAINTS ──
        # Load project-level hard constraints (read-only, agent cannot modify)
        persistent_constraints_path = self.project_dir / "PERSISTENT_CONSTRAINTS.md"
        if persistent_constraints_path.exists():
            try:
                constraints_text = persistent_constraints_path.read_text().strip()
                if constraints_text:
                    context["persistent_constraints"] = (
                        "PROJECT-LEVEL PERSISTENT CONSTRAINTS (HARD RULES):\n"
                        + constraints_text
                    )
            except Exception as e:
                logger.warning(f"Failed to load PERSISTENT_CONSTRAINTS.md: {e}")

        # ── Phase 1: Goal Progress ──
        # Inject current best metrics vs targets so the LLM always knows how
        # close it is to the goal. Previously there was no goal-tracking —
        # the agent achieved val_MAE=0.184 (target < 0.20) in cycle 1 but
        # had no idea it was already close.
        try:
            goal_text = self._build_goal_progress()
            if goal_text:
                context["goal_progress"] = goal_text
        except Exception as e:
            logger.warning(f"Failed to inject goal progress: {e}")

        # Inject dataset manifest (if available) so Leader knows data quality issues
        manifest_path = self.workspace / "DATASET_MANIFEST.json"
        if manifest_path.exists():
            try:
                manifest_data = json.loads(manifest_path.read_text())
                # Only inject the summary parts — not the full 4000-line manifest
                summary = {}
                if "training_recommendations" in manifest_data:
                    summary["training_recommendations"] = manifest_data["training_recommendations"]
                if "data_quality_issues" in manifest_data:
                    summary["data_quality_issues"] = manifest_data["data_quality_issues"]
                if "total_trainable" in manifest_data:
                    summary["total_trainable"] = manifest_data["total_trainable"]
                # Per-dataset scene counts
                ds_counts = {}
                for ds_name, ds_info in manifest_data.get("datasets", {}).items():
                    ds_counts[ds_name] = {
                        "type": ds_info.get("type", "unknown"),
                        "train": ds_info.get("train_scenes", 0),
                        "val": ds_info.get("val_scenes", 0),
                        "native_resolution": ds_info.get("native_resolution"),
                    }
                summary["datasets"] = ds_counts
                context["dataset_manifest_summary"] = summary
            except Exception as e:
                logger.warning(f"Failed to inject dataset manifest: {e}")

        # Inject session statistics from SQLite
        try:
            stats = self.memory.get_summary_stats()
            if stats.get("total_cycles", 0) > 0:
                context["session_stats"] = stats
                context["recent_failures"] = self.memory.get_recent_failures(count=3)
        except Exception as e:
            logger.warning(f"Failed to inject session stats: {e}")

        # ── DOMAIN KNOWLEDGE INJECTION ──
        # Extract method-hypothesis mappings from PROJECT_BRIEF + dead ends
        # This gives the Leader the PHYSICAL PRINCIPLES behind each method
        domain_kb = self._build_domain_knowledge()
        if domain_kb:
            context["domain_knowledge"] = domain_kb
            # Inject data constraints as top-level prompt
            if domain_kb.get("data_constraints"):
                context["data_constraints"] = domain_kb["data_constraints"]
                scarce_domains = [
                    c for c in domain_kb["data_constraints"]
                    if c.get("type") == "data_scarcity"
                ]
                if scarce_domains:
                    context["data_scarcity_warning"] = (
                        "DATA SCARCITY WARNING:\n"
                        + "\n".join(f"- {c['detail']}\n  Fix: {c['recommendation']}" for c in scarce_domains)
                        + "\n\nYou MUST NOT propose architecture changes for data-scarce domains. "
                        + "Focus on data augmentation, transfer learning, or accepting the limitation."
                    )

        # ── IDEA GUARDIAN CHECK (every 5 cycles) ──
        if self.cycle_count > 0 and self.cycle_count % 5 == 0:
            context["idea_guardian_check"] = (
                f"IDEA GUARDIAN CHECK (Cycle {self.cycle_count}):\n"
                "This is a MANDATORY direction alignment check. You MUST:\n"
                "1. Re-read PROJECT_BRIEF phase goals — which phase should we be in?\n"
                "2. List how many cycles contributed to the CORE IDEA (not incremental tuning)\n"
                "3. Rate: Core idea implementation (0-10), Phase completion (0-10), Data-first verification (0-10)\n"
                "4. If ANY score < 5, propose a COURSE CORRECTION — not another training run\n"
                "5. Have you verified the data supports the core idea BEFORE building models?\n"
                "If not, the next experiment MUST be a DATA ANALYSIS experiment, not model training."
            )

        # ── DIRECTION CIRCUIT BREAKER ──
        # Force direction re-evaluation when stagnation is detected
        # v15: ROADMAP has priority — if ROADMAP says we're still verifying theory,
        # "direction change" is irrelevant; the agent should verify the current module's assumptions.
        _roadmap_active = (
            self._roadmap_initialized
            and self.roadmap.is_theory_verification_phase
        )

        # ── v14: ARCHITECTURE CIRCUIT BREAKER ──
        # When the same architecture has been patched for too many cycles without
        # improvement, force the agent to SWITCH to a completely different architecture.
        # v15: During theory_verification, architecture switching is premature.
        # ── v14: ARCHITECTURE SURVEY GATE ──
        # In early cycles (1-2), force an architecture survey before committing to any model.
        # This prevents the agent from blindly using PROJECT_BRIEF's suggested baseline.
        if not self._architecture_survey_done and self.cycle_count <= 2:
            if not self._architecture_survey_path.exists():
                context["architecture_survey_gate"] = (
                    "ARCHITECTURE SURVEY GATE (v14): This is an early cycle and no architecture "
                    "survey has been completed yet.\n\n"
                    "BEFORE committing to any baseline architecture, you MUST:\n"
                    "1. Search for at least 3 different architectures/methods for this task\n"
                    "2. For each candidate, analyze:\n"
                    "   - Core ASSUMPTION (e.g., Lambertian, smooth, regular grid)\n"
                    "   - Data requirements vs. what's actually available\n"
                    "   - Computational feasibility given available resources\n"
                    "   - Published performance on similar tasks\n"
                    "3. Write the survey to workspace/ARCHITECTURE_SURVEY.md\n"
                    "4. ONLY THEN select the best architecture based on evidence\n\n"
                    "Do NOT use any architecture just because it's mentioned in PROJECT_BRIEF.\n"
                    "PROJECT_BRIEF provides context, NOT architectural decisions."
                )

        # ── CROSS-EXPERIMENT KNOWLEDGE INTEGRATION ──
        # Connect dead ends across experiments to identify meta-patterns
        cross_exp = self._build_cross_experiment_insights()
        if cross_exp:
            context["cross_experiment_insights"] = cross_exp

        # ── v12: METHOD INADEQUACY RE-AWAKENING ──
        # If previous dead ends were categorized as 'method_inadequacy' (the analysis
        # method was too narrow, not the hypothesis being wrong), inject a prompt
        # encouraging the Leader to retry with broader analysis instead of abandoning.
        try:
            mi_count = self.memory.get_method_inadequacy_count()
            if mi_count > 0:
                mi_entries = self.memory.get_dead_ends_by_category("method_inadequacy")
                mi_summaries = [e.get("content", "")[:120] for e in mi_entries[-3:]]
                context["method_inadequacy_retry_prompt"] = (
                    f"METHOD INADEQUACY RE-AWAKENING (v12):\n"
                    f"You have {mi_count} dead_end(s) categorized as 'method_inadequacy'.\n"
                    f"These are NOT hypothesis failures — the analysis method was too narrow.\n"
                    f"Recent entries:\n"
                    + "\n".join(f"  - {s}" for s in mi_summaries)
                    + "\n\n"
                    f"Consider RETRYING these directions with broader analysis:\n"
                    f"- Use at least 3 independent feature families\n"
                    f"- Include non-frequency methods (gradients, symmetry, entropy, view consistency)\n"
                    f"- Check DC-dominance before relying on frequency-domain results\n"
                    f"Only abandon a direction after ≥3 independent methods ALL show no signal."
                )
        except Exception as e:
            logger.debug(f"Method inadequacy check skipped: {e}")

        # ── v12.1: PENDING DEGRADED REFLECT ──
        # If the previous cycle's REFLECT was degraded (API quota exhausted),
        # inject a reminder so the Leader can revisit those results.
        degraded_note_path = self.workspace / ".degraded_reflect_pending"
        if degraded_note_path.exists():
            try:
                degraded_info = json.loads(degraded_note_path.read_text())
                context["degraded_reflect_pending"] = (
                    f"PRIOR CYCLE INCOMPLETE REFLECT (v12.1):\n"
                    f"Cycle {degraded_info.get('cycle', '?')} REFLECT was degraded "
                    f"(API quota exhausted). Results were preserved but NOT fully analyzed.\n"
                    f"Metrics: {degraded_info.get('metrics_summary', 'N/A')}\n"
                    f"VERIFY: {degraded_info.get('verify_status', 'N/A')}\n"
                    f"{'Milestone recorded (partial).' if degraded_info.get('has_milestone') else ''}"
                    f"{'Dead end recorded (partial).' if degraded_info.get('has_dead_end') else ''}\n"
                    f"→ You SHOULD briefly review the last cycle's results before planning new work.\n"
                    f"→ If the last cycle was successful, continue from there.\n"
                    f"→ If it failed, diagnose and fix before proceeding."
                )
                # Consume the note after one injection
                degraded_note_path.unlink()
            except Exception:
                pass

        # ── ARCHITECTURE PLAN INJECTION (Phase 2+) ──
        # When transitioning from analysis to model building, provide the Leader
        # with a pre-computed architecture plan so they can dispatch informed tasks.
        # Also re-generate when a new model file appears (detected by mtime).
        brief_path = self.workspace / "PROJECT_BRIEF.md"
        should_plan = False
        if brief_path.exists():
            if self.cycle_count <= 1:
                should_plan = True
            elif not getattr(self, '_last_architecture_plan', None):
                # No plan yet — generate one
                should_plan = True
            else:
                # Re-generate if a model file was modified after last plan
                last_plan_time = getattr(self, '_last_plan_time', 0)
                models_dir = self.project_dir / "models"
                if models_dir.exists():
                    newest_model = max(
                        (f.stat().st_mtime for f in models_dir.glob("*.py")),
                        default=0,
                    )
                    if newest_model > last_plan_time:
                        should_plan = True

        if should_plan:
            try:
                from .idea_planner import IdeaPlanner
                planner = IdeaPlanner(self.workspace)
                arch_plan = planner.plan()
                if "error" not in arch_plan:
                    self._last_architecture_plan = arch_plan
                    self._last_plan_time = time.time()
                    context["architecture_plan"] = arch_plan
                    score = arch_plan.get("alignment_score", 0)
                    context["architecture_plan_summary"] = (
                        f"Pre-computed architecture plan available (alignment={score}/10). "
                        f"Use `plan_model` tool for detailed view. "
                        f"Plan has {len(arch_plan.get('modules', []))} modules, "
                        f"fusion={arch_plan.get('fusion_strategy', {}).get('method', 'N/A')}, "
                        f"{len(arch_plan.get('risks', []))} risks identified."
                    )
                    # v16.1: ImplementationTracker removed
            except Exception as e:
                logger.warning(f"Architecture plan generation failed: {e}")

        # ── PARETO FRONTIER INJECTION ──
        # Show which methods are Pareto-optimal for which domains
        try:
            pareto = self.memory.get_pareto_frontier()
            if pareto.get("matrix"):
                context["pareto_frontier"] = pareto
        except Exception as e:
            logger.warning(f"Failed to inject pareto frontier: {e}")

        # ── CAUSAL CHAIN HISTORY ──
        # Show past design decisions and their actual effects.
        # Phase 1 fix: inject ALL causal links (not just verified), marking
        # which are verified. Previously the verified filter made this always
        # empty (0/102 links verified), leaving the LLM blind to history.
        try:
            causal_history = self.memory.get_causal_history(limit=10)
            if causal_history:
                # Format with verification status so LLM can judge confidence
                lines = []
                for c in causal_history:
                    decision = c.get("design_decision", "?")
                    expected = c.get("expected_effect", "?")
                    actual = c.get("actual_effect")
                    verified = "✓ verified" if c.get("verified") else "? unverified"
                    actual_str = f" → actual: {actual}" if actual else ""
                    lines.append(f"- [{verified}] {decision}: expected {expected}{actual_str}")
                context["causal_history"] = "\n".join(lines)
        except Exception as e:
            logger.warning(f"Failed to inject causal history: {e}")

        # ── CODE REVIEW LESSONS INJECTION ──
        # Inject past mistakes into THINK context so the agent learns from them.
        # This is the key mechanism that makes the knowledge base actually used.
        try:
            # Get all HIGH/MEDIUM lessons, format for context
            all_lessons = self.memory.get_code_review_lessons(severity="MEDIUM", limit=20)
            if all_lessons:
                lesson_text = self.memory.format_lessons_for_context(all_lessons, max_chars=1500)
                if lesson_text:
                    context["code_review_lessons"] = lesson_text

            # Also search for lessons relevant to the current project code.
            # Phase 1 fix: when models/ doesn't exist, fall back to searching
            # using the current task text instead of skipping entirely.
            # Previously: no models/ dir → 0 relevant lessons injected.
            search_text = None
            model_dir = self.project_dir / "models"
            if model_dir.exists():
                try:
                    model_files = sorted(model_dir.glob("*.py"), key=lambda f: f.stat().st_mtime, reverse=True)
                    if model_files:
                        latest = model_files[0]
                        mtime = latest.stat().st_mtime
                        if (not hasattr(self, '_cached_model_mtime') or
                                self._cached_model_mtime != mtime):
                            self._cached_model_content = latest.read_text()
                            self._cached_model_mtime = mtime
                        search_text = self._cached_model_content
                except Exception:
                    pass
            # Fallback: use the most recent memory log entries (last cycle's
            # decisions) for keyword matching. This gives the lessons context
            # about what the agent is currently working on.
            if search_text is None:
                mem_log = self.memory.get_log()
                # Use last 500 chars of memory log as search context
                search_text = mem_log[-500:] if mem_log else ""
            if search_text:
                relevant = self.memory.search_relevant_lessons(search_text, limit=5)
                if relevant:
                    context["relevant_code_review_lessons"] = (
                        self.memory.format_lessons_for_context(relevant, max_chars=1000)
                    )
        except Exception as e:
            logger.warning(f"Failed to inject code review lessons: {e}")

        # ── EXPERIMENT CALIBRATION ──
        # Help the agent learn from past hypothesis accuracy
        try:
            calibration = self.memory.get_experiment_calibration()
            if calibration.get("total_hypotheses", 0) >= 3:
                context["hypothesis_calibration"] = calibration
        except Exception as e:
            logger.warning(f"Failed to inject hypothesis calibration: {e}")

        # ── EXPERIMENT VALUE: warn about low-value directions ──
        # Phase 1: inject previously-assessed low-VOI directions so the LLM
        # knows which paths have already been evaluated as unlikely to help.
        try:
            low_voi = self.memory.get_low_value_experiments(limit=5) if hasattr(self.memory, 'get_low_value_experiments') else []
            if low_voi:
                lines = [f"- {v.get('hypothesis','?')[:80]} (VOI={v.get('voi',0):.3f})"
                         for v in low_voi]
                context["experiment_value_warn"] = (
                    "Previously assessed as low-value:\n" + "\n".join(lines)
                )
        except Exception as e:
            logger.warning(f"Failed to inject experiment value: {e}")

        # v16.1: ImplementationTracker and AdaptiveThresholds removed
        # (dead modules, context keys removed)

        # v16.1: sandbox_design_guidance removed from context (context key reduction)

        # ── RESEARCH ROADMAP (v15): Inject phase constraints ──
        # This is the PRIMARY control mechanism: tells Leader what phase and module to work on.
        try:
            roadmap_ctx = self.roadmap.get_phase_context(self.cycle_count)
            if roadmap_ctx:
                context["research_roadmap"] = roadmap_ctx
        except Exception as e:
            logger.warning(f"ROADMAP context injection failed: {e}")

        # ── PHASE FOCUS (v16): State-driven structured thinking ──
        # Replace open-ended "what should we do next?" with structured
        # "here's the gap, propose how to close it" — constrains the answer space.
        phase_focus = self._build_phase_focus()
        if phase_focus:
            context["phase_focus"] = phase_focus

        # ── CONTEXT PRUNING (v10) ──
        # Limit context to most relevant keys to prevent LLM confusion
        context = self.context_pruner.prune(context, "think")

        result = self.dispatcher.dispatch_leader(
            task="think",
            context=context,
        )

        # ── STRATEGY CONSTRAINT CHECK (v10 → v16 hard gate) ──
        # Check proposed action against learned constraints.
        # FORBIDDEN violations → hard block (redirect to data_analysis).
        # Non-FORBIDDEN violations → silently logged (no context bloat).
        if result.get("action") == "experiment":
            violations = self.strategy_engine.check_constraints(result, self.memory)
            if violations:
                if self.strategy_engine.has_forbidden_violation(violations):
                    # HARD GATE: FORBIDDEN constraint → block experiment
                    blocked_msg = self.strategy_engine.get_constraint_prompt(violations)
                    logger.warning(f"⛔ FORBIDDEN constraint blocked experiment")
                    result["action"] = "paper_research"
                    result["agent"] = "researcher"
                    result["task"] = (
                        f"⛔ BLOCKED: Proposed experiment violates FORBIDDEN constraint(s).\n"
                        f"{blocked_msg}\n\n"
                        f"Research alternative approaches that avoid the forbidden methods. "
                        f"Focus on approaches compatible with the current research phase."
                    )
                    self.memory.log_decision(
                        f"[BLOCKED v16] Experiment blocked by FORBIDDEN constraint"
                    )
                else:
                    # Non-FORBIDDEN: just log, don't inject into context (reduces bloat)
                    logger.info(f"Strategy: {len(violations)} non-FORBIDDEN constraint(s) noted")
                    self.memory.log_decision(
                        f"[CONSTRAINT] {len(violations)} non-FORBIDDEN constraint(s) noted"
                    )

        # ── EXPERIMENT VALUE OF INFORMATION (VOI) ──
        # Estimate the value of the proposed experiment before running it.
        # This helps the agent learn to prioritize high-value experiments.
        if result.get("action") == "experiment" and result.get("hypothesis"):
            self._estimate_experiment_value(result)

        # ── CAUSAL CHAIN RECORDING ──
        # Record the design decision → architectural property → expected metric link
        if result.get("action") == "experiment" and result.get("hypothesis"):
            self._record_causal_chain_from_think(result)

        logger.info(f"THINK result: action={result.get('action', 'unknown')}")

        # Validate context keys against registry
        try:
            from .context_keys import validate_context
            key_warnings = validate_context(context, "think")
            for w in key_warnings:
                logger.debug(f"Context key: {w}")
        except Exception:
            pass
        return result

    def _execute(self, plan: dict) -> dict:
        """EXECUTE phase: implement and run the planned experiment."""
        logger.info("EXECUTE phase starting...")

        agent_type = plan.get("agent", "code")
        task_description = plan.get("task", "")

        # v16.1: scope_prefix removed (pure text injection ineffective against LLM)

        result = self.dispatcher.dispatch_worker(
            agent_type=agent_type,
            task=task_description,
            tools=self.tools.get_tools_for(agent_type),
        )

        return result

    def _execute_paper_research(self, plan: dict) -> dict:
        """EXECUTE phase: run deep paper research.

        Two paths:
          - If ``idea_scout.enabled`` is True in config AND the
            research-idea-scout library is available, run the cross-domain
            idea-discovery pipeline (gather → filter → score) and return a
            structured ranked list.
          - Otherwise, dispatch to the 'researcher' agent (web search + paper
            tools) as before — unchanged behavior.
        """
        scout_cfg = self.config.get("idea_scout", {})
        if scout_cfg.get("enabled", False):
            try:
                from core.idea_scout_bridge import is_available as scout_available
                if scout_available():
                    return self._execute_idea_scout(plan, scout_cfg)
                logger.warning(
                    "idea_scout.enabled=True but research-idea-scout library not "
                    "found. Falling back to researcher agent."
                )
            except Exception as e:
                logger.warning(
                    f"IdeaScout pipeline failed ({e}). Falling back to researcher agent."
                )

        logger.info("PAPER RESEARCH EXECUTE phase starting (researcher agent)...")

        task_description = plan.get(
            "task",
            "Execute the /paper-research skill. Read skills/paper-research/SKILL.md for instructions.",
        )

        result = self.dispatcher.dispatch_worker(
            agent_type="researcher",
            task=task_description,
            tools=self.tools.get_tools_for("researcher"),
        )

        # Mark as paper_research so REFLECT knows how to handle it
        result["is_paper_research"] = True
        return result

    def _execute_idea_scout(self, plan: dict, scout_cfg: dict) -> dict:
        """Run the IdeaScout cross-domain idea-discovery pipeline.

        1. Build a Profile from PROJECT_BRIEF (+ memory dead-ends).
        2. Gather papers via this agent's search_papers tool.
        3. Rule-filter candidates (fast keyword pruning).
        4. LLM-score survivors for transferability (via ProviderRouter).
        5. Write a ranked Markdown report to the workspace.
        Returns a result dict with ``idea_scout_ranked`` for VERIFY/REFLECT.
        """
        from core.idea_scout_bridge import (
            build_profile_from_brief, run_pipeline, format_results_markdown,
        )

        task = plan.get("task", "")
        logger.info(f"IDEA SCOUT pipeline starting for: {task[:100]}")

        # 1. Profile
        brief_path = self.project_dir / "PROJECT_BRIEF.md"
        profile_path = scout_cfg.get("profile_path", "")
        profile = build_profile_from_brief(
            brief_path, memory=self.memory,
            profile_path=profile_path or None,
        )

        # 2-4. Pipeline (gather → filter → score)
        pipeline_result = run_pipeline(
            tools_registry=self.tools,
            dispatcher=self.dispatcher,
            profile=profile,
            query=task,
            max_papers=scout_cfg.get("max_papers", 50),
            filter_top_k=scout_cfg.get("filter_top_k", 20),
            score_top_k=scout_cfg.get("score_top_k", 10),
            abstract_max_chars=scout_cfg.get("abstract_max_chars", 3000),
        )

        # 5. Write report
        report = format_results_markdown(pipeline_result, task)
        date_str = time.strftime("%Y-%m-%d")
        report_path = self.workspace / f"idea_scout_results_{date_str}.md"
        report_path.write_text(report, encoding="utf-8")
        logger.info(f"IdeaScout report written to {report_path}")

        # Build the execute_result dict (consumed by VERIFY/REFLECT)
        ranked = pipeline_result.get("ranked", [])
        top_ideas = [
            {
                "title": p.get("title", ""),
                "rank_score": p.get("rank_score", 0),
                "priority": p.get("priority", ""),
                "idea_core": p.get("idea_core", ""),
                "transferable_mechanism": p.get("transferable_mechanism", ""),
                "url": p.get("url", ""),
            }
            for p in ranked[:5]  # top-5 for the summary
        ]
        return {
            "agent": "idea_scout",
            "is_paper_research": True,
            "response": report[:2000],  # truncated for context
            "idea_scout_ranked": pipeline_result,
            "top_ideas": top_ideas,
            "report_path": str(report_path),
            "papers_scored": pipeline_result.get("papers_scored", 0),
        }

    def _monitor_experiment(self, execute_result: dict) -> dict:
        """Monitor running experiment with ZERO LLM calls."""
        pid = execute_result.get("pid")
        log_file = execute_result.get("log_file")

        if not pid:
            return {"status": "no_pid"}

        # Register PID with monitor so has_active_experiments() / has_completed_experiments() work.
        # ToolRegistry._exec_launch_experiment starts the process but doesn't register it
        # with the monitor, so we manually register here.
        state = self._load_state()
        self.monitor.register_experiment(
            pid=pid,
            log_file=log_file,
            command=execute_result.get("tool_trace", {}).get("launch_facts", {}).get("command", ""),
            start_time=state.get("started_at"),
        )

        start_time = state.get("started_at")

        logger.info(f"Monitoring experiment PID={pid}, log={log_file}")
        return self.monitor.wait_for_completion(
            pid=pid,
            log_file=log_file,
            notify=self.config.get("monitor", {}).get("notify_on_complete", True),
            start_time=start_time,
        )

    def _verify(self, cycle: int, think_result: dict, execute_result: dict):
        """VERIFY phase: reverse-engineer whether each module actually worked.

        Fix 2: Infrastructure degradation — if infrastructure failures are
        recurring, skip VERIFY to avoid wasting cycles on non-experimental issues.
        """
        # Fix 2: Check if we should skip VERIFY due to infrastructure degradation
        if self._infra_failure_streak >= self._infra_degradation_threshold:
            logger.warning(
                f"VERIFY SKIPPED: {self._infra_failure_streak} consecutive infrastructure "
                f"failures. Agent proceeds with experiment — infrastructure issues "
                f"will be logged but not block progress."
            )
            # Still run VERIFY but don't escalate failures
            verify_report = self.verifier.verify(
                cycle=cycle,
                think_result=think_result,
                execute_result=execute_result,
            )
            # Reset streak if no new infra failures this cycle
            has_new_infra = any(
                c.category == "infrastructure" for c in verify_report.all_failures
            )
            if not has_new_infra:
                self._infra_failure_streak = 0
            return verify_report

        logger.info("VERIFY phase starting...")
        verify_report = self.verifier.verify(
            cycle=cycle,
            think_result=think_result,
            execute_result=execute_result,
        )

        # Track infrastructure failures for degradation
        infra_failed = any(
            c.category == "infrastructure" for c in verify_report.all_failures
        )
        if infra_failed:
            self._infra_failure_streak += 1
        else:
            self._infra_failure_streak = 0  # Reset on success

        # Log summary to memory for REFLECT context
        if verify_report.has_failures:
            failed_modules = ", ".join(verify_report.failed_modules) or "unknown"
            self.memory.log_decision(
                f"[VERIFY Cycle {cycle}] {len(verify_report.all_failures)} issue(s) "
                f"in module(s): {failed_modules}. "
                + "; ".join(verify_report.diagnosis[:2])
            )
        else:
            logger.info(f"VERIFY Cycle {cycle}: all checks passed")

        return verify_report

    def _pre_verify(self, cycle: int, think_result: dict):
        """PRE-VERIFY phase: check preconditions BEFORE EXECUTE.

        Catches problems like synthetic data, missing data directories,
        broken imports — before wasting GPU hours.
        Also runs SimulationSandbox GPU safety check (v11).
        """
        logger.info("PRE-VERIFY phase starting...")
        pre_report = self.verifier.pre_verify(cycle=cycle, think_result=think_result)

        # v11: Sandbox GPU safety check — prevent OOM before training
        if think_result.get("action") == "experiment":
            model_path = self._extract_model_path_from_task(think_result)
            if model_path:
                try:
                    # Save snapshot before EXECUTE modifies the model
                    self.sandbox.save_snapshot(model_path, self.cycle_count)

                    # Layer 0: Feasibility check (shape + GPU memory)
                    feasibility = self.sandbox.check_feasibility(model_path)
                    if not feasibility.feasible:
                        pre_report.checks.append(VerifyCheck(
                            name="sandbox_model_infeasible",
                            category="system",
                            status="fail",
                            detail=f"Model cannot run: {feasibility.error}",
                            severity="critical",
                            module_path="simulation_sandbox",
                        ))
                    elif feasibility.warnings:
                        pre_report.checks.append(VerifyCheck(
                            name="sandbox_warnings",
                            category="system",
                            status="warn",
                            detail="; ".join(feasibility.warnings[:3]),
                            severity="medium",
                            module_path="simulation_sandbox",
                        ))
                    logger.info(
                        f"Sandbox feasibility: feasible={feasibility.feasible}, "
                        f"params={feasibility.total_params:,}, "
                        f"gpu_peak={feasibility.gpu_memory_peak_mb:.0f}MB, "
                        f"max_batch={feasibility.max_safe_batch_size}"
                    )
                except Exception as e:
                    logger.debug(f"Sandbox feasibility check skipped: {e}")

        return pre_report

    def _extract_model_path_from_task(self, think_result: dict) -> str:
        """Extract model file path from think result task description."""
        task = think_result.get("task", "")
        # Look for model path patterns in the task description
        patterns = [
            r"models/[\w/\-\.]+\.py",
            r"(?:model_path|model_file|model)\s*[:=]\s*['\"]([\w/\-\.]+\.py)['\"]",
        ]
        for pattern in patterns:
            match = re.search(pattern, task)
            if match:
                path = match.group(1) if match.lastindex else match.group(0)
                # Ensure the path starts with "models/" for consistency
                if not path.startswith("models/"):
                    path = f"models/{path}"
                return path

        # Fallback: find most recently modified model file
        models_dir = self.project_dir / "models"
        if models_dir.exists():
            files = sorted(models_dir.glob("*.py"), key=lambda f: f.stat().st_mtime, reverse=True)
            if files:
                return f"models/{files[0].name}"
        return ""

    def _pre_execute_code_review(self, think_result: dict) -> list:
        """v12.3: Two-phase code review BEFORE training.

        Phase 1: Zero-LLM regex checks (fast, free, catches known anti-patterns)
          - Checks both model code AND training scripts
          - HIGH severity -> hard-block execution
        Phase 2: LLM semantic review (catches logic bugs, design flaws)
          - Uses cheap fast model for architecture sanity check
          - Only runs when Phase 1 found no HIGH issues (to avoid wasted LLM tokens)

        Returns a list of warnings with severity levels.
        """
        warnings = []

        # -- Phase 1: Regex-based structural checks --
        model_rel_path = self._extract_model_path_from_task(think_result)
        model_content = ""
        if model_rel_path:
            model_path = self.project_dir / model_rel_path
            if model_path.exists():
                try:
                    model_content = model_path.read_text()
                except Exception:
                    pass

        if not model_content:
            # Still check training script even without model code
            train_script_content = self._find_training_script_content(think_result)
            if train_script_content:
                warnings.extend(self._regex_code_review_train_script(
                    train_script_content, ""
                ))
            return warnings

        # Regex checks on model code
        warnings.extend(self._regex_code_review_model(model_content))

        # Regex checks on training script
        train_script_content = self._find_training_script_content(think_result)
        if train_script_content:
            warnings.extend(self._regex_code_review_train_script(
                train_script_content, model_content
            ))

        # -- Phase 1c: Cross-file duplicate class detection --
        dup_warnings = self._check_duplicate_classes()
        warnings.extend(dup_warnings)

        # -- Phase 2: LLM semantic review --
        # Only run if Phase 1 found no HIGH issues (otherwise already blocked)
        has_high = any(w["severity"] == "HIGH" for w in warnings)
        if not has_high and model_content:
            llm_warnings = self._llm_code_review(model_content, train_script_content)
            warnings.extend(llm_warnings)

        return warnings

    @staticmethod
    def _strip_comments_and_strings(content: str) -> str:
        """Remove comments and string literals so regex checks only match real code."""
        # Phase 1: Remove multi-line strings first (operates on full content)
        cleaned = re.sub(r'"""[\s\S]*?"""', '', content)
        cleaned = re.sub(r"'''[\s\S]*?'''", '', cleaned)
        # Phase 2: Line-by-line comment and single-line string removal
        lines = []
        for line in cleaned.splitlines():
            stripped = line.lstrip()
            if stripped.startswith('#'):
                continue
            # Remove inline comments (simple heuristic — doesn't handle '#' inside strings)
            code_part = stripped.split('#')[0] if '#' in stripped else stripped
            # Remove remaining single-line strings
            code_part = re.sub(r'"[^"]*"', '', code_part)
            code_part = re.sub(r"'[^']*'", '', code_part)
            lines.append(code_part)
        return '\n'.join(lines)

    def _regex_code_review_model(self, content: str) -> list:
        """Phase 1a: Regex-based structural checks on model code."""
        warnings = []
        code_only = self._strip_comments_and_strings(content)

        # -- Check 1: Routing/Fusion without auxiliary supervision --
        has_routing = bool(re.search(
            r"routing|router|route_weight|w_epi|w_defocus|gate_weight|gate_network|gating|fusion_weight",
            code_only, re.IGNORECASE
        ))
        has_aux_loss = bool(re.search(
            r"aux_loss|aux_weight|auxiliary|routing_loss|gate_loss",
            code_only, re.IGNORECASE
        ))
        if has_routing and not has_aux_loss:
            warnings.append({
                "severity": "HIGH",
                "detail": (
                    "Model has routing/fusion mechanism but no auxiliary loss for routing. "
                    "Without explicit supervision, routing weights will not differentiate. "
                    "Add aux_loss with domain-specific routing targets (e.g. "
                    "Lambertian->[1,0], NL->[0,1])."
                ),
            })

        # -- Check 2: Input channel information asymmetry --
        conv_inputs = re.findall(r"Conv2d\((\d+),", code_only)
        if len(conv_inputs) >= 3:
            inputs_int = [int(c) for c in conv_inputs if int(c) > 1]
            if inputs_int:
                ratio = max(inputs_int) / min(inputs_int)
                if ratio > 10:
                    warnings.append({
                        "severity": "MEDIUM",
                        "detail": (
                            f"Input channel information asymmetry detected: "
                            f"branches range from {min(inputs_int)}ch to {max(inputs_int)}ch "
                            f"({ratio:.0f}x ratio). The low-channel branch may not "
                            f"have enough information to learn useful features."
                        ),
                    })

        # -- Check 3: Router with only 1x1 conv (no spatial context) --
        # Use per-line matching to avoid cross-block false positives.
        # Only flag when router class/method DEFINES a 1x1 conv (not just mentions it).
        has_router_1x1 = False
        for line in code_only.splitlines():
            stripped = line.strip()
            if re.search(r"(?:router|routing|gate)", stripped, re.IGNORECASE):
                if re.search(r"Conv2d\(\d+,\s*\d+.*?kernel_size\s*=\s*1", stripped):
                    has_router_1x1 = True
                    break
        if has_router_1x1:
            warnings.append({
                "severity": "MEDIUM",
                "detail": (
                    "Router uses only 1x1 convolutions -- no spatial context. "
                    "This produces pixel-independent routing, causing spatial "
                    "artifacts in the output. Consider using 3x3 conv or adding "
                    "spatial smoothing after routing weights."
                ),
            })

        # -- Check 4: Weighted fusion without baseline comparison --
        has_weighted_fusion = bool(re.search(
            r"w_\w+\s*\*\s*\w+_feature|weighted.*sum|weighted.*fusion",
            content, re.IGNORECASE,
        ))
        if has_weighted_fusion and has_routing:
            has_skip = bool(re.search(
                r"skip|residual.*baseline|identity|epi_only",
                content, re.IGNORECASE,
            ))
            if not has_skip:
                warnings.append({
                    "severity": "LOW",
                    "detail": (
                        "Weighted fusion of EPI + Defocus without skip/baseline "
                        "connection. If defocus branch produces noise, there is no "
                        "fallback to EPI-only prediction. Consider adding a "
                        "skip connection from EPI features to decoder."
                    ),
                })

        # -- Check 5: Extreme channel compression in routing path --
        channel_seq = re.findall(
            r"(?:Conv2d|Linear)\((\d+),\s*(\d+)", content
        )
        for i in range(len(channel_seq) - 1):
            out1 = int(channel_seq[i][1])
            in2 = int(channel_seq[i + 1][0])
            if out1 == in2 and out1 > 0:
                reduction_ratio = int(channel_seq[i][0]) / out1
                if reduction_ratio > 10:
                    pos = content.find(channel_seq[i][0])
                    surrounding = content[max(0, pos - 300): pos + 300]
                    if re.search(r"router|routing|gate", surrounding, re.IGNORECASE):
                        warnings.append({
                            "severity": "MEDIUM",
                            "detail": (
                                f"Extreme channel compression ({int(channel_seq[i][0])}→{out1}) "
                                f"near routing/gate module. This may lose too much information "
                                f"for the router to make meaningful decisions. "
                                f"Consider using a wider intermediate dimension."
                            ),
                        })

        # -- Check 6: Import inside function body (hot-path performance) --
        in_function = False
        func_indent = 0
        for line in content.splitlines():
            stripped = line.lstrip()
            indent = len(line) - len(stripped)
            if re.match(r"def\s", stripped):
                in_function = True
                func_indent = indent
            elif in_function:
                if indent <= func_indent and stripped and not stripped.startswith("#"):
                    if re.match(r"(def |class |@)", stripped):
                        in_function = False
                        if re.match(r"def\s", stripped):
                            in_function = True
                            func_indent = indent
                    elif re.match(r"import\s", stripped) or re.match(r"from\s+\S+\s+import", stripped):
                        warnings.append({
                            "severity": "MEDIUM",
                            "detail": (
                                "Import statement inside function body detected. "
                                "Imports in hot-path functions (e.g. __getitem__, forward) "
                                "add sys.modules lookup overhead on every call and may "
                                "hide circular dependencies. Move imports to module top level."
                            ),
                        })
                        break

        # -- Check 7: Silent error fallback to random/default data --
        if re.search(
            r"except\s+.*:.*(?:torch\.rand|np\.random|random\.random|np\.randn)",
            code_only, re.IGNORECASE,
        ):
            warnings.append({
                "severity": "HIGH",
                "detail": (
                    "Exception handler falls back to random data (torch.rand / np.random). "
                    "This silently injects noise into training data, corrupting model "
                    "learning without any visible error. Either raise the exception, "
                    "return None (let DataLoader skip), or use zero/mean fill instead."
                ),
            })
        elif re.search(
            r"except\s+(?:Exception|BaseException).*:\s*\n"
            r"[\s\S]*?(?:torch\.rand|np\.random|random\.random|np\.randn)",
            code_only, re.IGNORECASE,
        ):
            warnings.append({
                "severity": "HIGH",
                "detail": (
                    "Broad exception handler falls back to random data (torch.rand / np.random). "
                    "This silently injects noise into training, corrupting model learning. "
                    "Replace with raise or return None to skip the sample."
                ),
            })

        return warnings

    def _regex_code_review_train_script(self, script_content: str, model_content: str) -> list:
        """Phase 1b: Regex-based checks on training script."""
        warnings = []

        # -- Check T1: Routing target majority trivial --
        all_targets = re.findall(
            r"(?:target|label|routing_target|gate_target)\s*[=:]\s*\[([^\]]+)\]",
            script_content, re.IGNORECASE,
        )
        if all_targets:
            trivial_count = sum(1 for t in all_targets if "0.5" in t)
            if trivial_count > len(all_targets) / 2:
                warnings.append({
                    "severity": "HIGH",
                    "detail": (
                        f"Routing target design flaw: {trivial_count}/{len(all_targets)} "
                        f"domain routing targets are [0.5, 0.5] (trivial). "
                        f"This means the router gets no useful gradient signal for "
                        f"{trivial_count} domains -- it cannot learn to differentiate. "
                        f"Use domain-specific targets like [1,0] vs [0,1]."
                    ),
                })

        # -- Check T2: aux_weight too low relative to main loss scale --
        aux_weight_match = re.search(
            r"aux_weight\s*[=:]\s*([0-9.]+)", script_content, re.IGNORECASE,
        )
        if aux_weight_match:
            aux_weight = float(aux_weight_match.group(1))
            if aux_weight < 0.05:
                warnings.append({
                    "severity": "HIGH",
                    "detail": (
                        f"aux_weight={aux_weight} is very low (< 0.05). "
                        f"When main loss >> aux loss, this means aux gradient is "
                        f"effectively zero. The routing/attention module cannot learn. "
                        f"Recommended: aux_weight >= 0.1, or use a separate optimizer."
                    ),
                })
            elif aux_weight < 0.1:
                if re.search(r"router|routing|gate", model_content, re.IGNORECASE):
                    warnings.append({
                        "severity": "MEDIUM",
                        "detail": (
                            f"aux_weight={aux_weight} may be too low for complex routing. "
                            f"If aux loss doesn't decrease during training, increase to "
                            f"0.2-0.5 or use gradient scaling."
                        ),
                    })

        # -- Check T3: Training data leakage or no validation split --
        if not re.search(r"val_split|val_dataset|validation|test_split", script_content, re.IGNORECASE):
            warnings.append({
                "severity": "MEDIUM",
                "detail": (
                    "Training script has no visible validation split. "
                    "Without validation, you cannot detect overfitting. "
                    "Add a proper train/val split."
                ),
            })

        # -- Check T4: Missing __main__ guard --
        script_clean = self._strip_comments_and_strings(script_content)
        if not re.search(r'if\s+__name__\s*==\s*["\']__main__["\']', script_clean):
            has_multiprocess = bool(re.search(
                r"num_workers|multiprocessing|DataLoader", script_clean, re.IGNORECASE,
            ))
            severity = "MEDIUM" if has_multiprocess else "LOW"
            detail = (
                "Training script missing `if __name__ == '__main__':` guard. "
                "Without this, DataLoader with num_workers > 0 will cause infinite "
                "spawn on Windows and may cause issues on Linux with fork strategy. "
                "Wrap main training logic in __main__ guard."
            )
            if not has_multiprocess:
                detail = (
                    "Training script missing `if __name__ == '__main__':` guard. "
                    "This is a best practice for all training scripts."
                )
            warnings.append({"severity": severity, "detail": detail})

        # -- Check T5: Hardcoded absolute paths --
        abs_path_matches = re.findall(
            r'["\'](/home/|/root/|/data/|/mnt/|/opt/)[^"\']+["\']',
            script_clean,
        )
        if abs_path_matches:
            warnings.append({
                "severity": "MEDIUM",
                "detail": (
                    f"Hardcoded absolute path(s) detected: {abs_path_matches[:3]}. "
                    f"Absolute paths break portability across machines. "
                    f"Use argparse, config files, or os.path relative paths instead."
                ),
            })

        # -- Check T6: Using sys.argv without argparse --
        has_sys_argv = bool(re.search(r"sys\.argv", script_clean))
        has_argparse = bool(re.search(r"argparse|ArgumentParser", script_clean))
        if has_sys_argv and not has_argparse:
            warnings.append({
                "severity": "LOW",
                "detail": (
                    "Training script uses sys.argv directly instead of argparse. "
                    "This makes hyperparameter management fragile and error-prone. "
                    "Consider using argparse for cleaner CLI interface."
                ),
            })

        return warnings

    def _check_duplicate_classes(self) -> list:
        """Phase 1c: Detect classes defined identically across multiple model files.

        Scans models/ and scripts/ for class definitions. If the same class name
        appears in 2+ files, emits a MEDIUM warning — this usually indicates
        copy-paste duplication that should be refactored into a shared module.
        """
        warnings = []
        class_locations = {}  # class_name -> [file_paths]

        search_dirs = [self.project_dir / "models", self.project_dir / "scripts"]
        for search_dir in search_dirs:
            if not search_dir.exists():
                continue
            for py_file in search_dir.glob("*.py"):
                if py_file.name.startswith("_") and py_file.name == "__init__.py":
                    continue
                try:
                    content = py_file.read_text()
                except Exception:
                    continue
                for match in re.finditer(r"^class\s+(\w+)\s*[(\[:]", content, re.MULTILINE):
                    cname = match.group(1)
                    # Skip small utility classes unlikely to be duplicated
                    skip_names = ("Test", "Config", "Error", "Exception", "Enum",
                                  "Dataset", "DataLoader", "Sampler", "Transform")
                    if any(s in cname for s in skip_names):
                        continue
                    rel = str(py_file.relative_to(self.project_dir))
                    class_locations.setdefault(cname, []).append(rel)

        for cname, files in class_locations.items():
            if len(files) >= 3:
                warnings.append({
                    "severity": "MEDIUM",
                    "detail": (
                        f"Class '{cname}' is defined in {len(files)} files: {files[:5]}. "
                        f"Consider extracting it into a shared module (e.g. models/components.py) "
                        f"to avoid copy-paste drift and ease maintenance."
                    ),
                })
        return warnings

    def _find_training_script_content(self, think_result: dict) -> str:
        """Find and read the training script content from task description or project."""
        task = think_result.get("task", "")

        script_patterns = [
            r"scripts/[\w/]+\.py",
            r"train[\w_]*\.py",
        ]
        for pattern in script_patterns:
            match = re.search(pattern, task)
            if match:
                script_path = self.project_dir / match.group(0)
                if script_path.exists():
                    try:
                        return script_path.read_text()
                    except Exception:
                        pass

        # Fallback: find most recently modified training script
        for search_dir in [self.project_dir / "scripts", self.project_dir]:
            if not search_dir.exists():
                continue
            scripts = sorted(
                search_dir.glob("train*.py"),
                key=lambda f: f.stat().st_mtime,
                reverse=True,
            )
            if scripts:
                try:
                    return scripts[0].read_text()
                except Exception:
                    pass

        return ""

    @staticmethod
    def _extract_key_code_segments(content: str, budget: int = 5000) -> str:
        """Extract the most semantically important code segments within a char budget.

        Priority order (highest first):
        1. forward() method — the data flow / computation graph
        2. __init__ of the main model class — architecture definition
        3. Loss / criterion / routing weight computation functions
        4. Remaining code (tail, to catch helper methods)

        For each block we keep a configurable budget.  If the total is still
        under *budget* after extracting the priority blocks, we append lines
        from the tail of the file (where helper utilities and small modules
        tend to live) until the budget is exhausted.
        """
        lines = content.splitlines()
        total_lines = len(lines)

        # Helper: extract a contiguous block identified by its start line.
        def _block_from(start_idx: int) -> tuple[list[str], int]:
            """Return (block_lines, end_idx) — stops at next def/class at same or lower indent."""
            block = [lines[start_idx]]
            base_indent = len(lines[start_idx]) - len(lines[start_idx].lstrip())
            end = start_idx + 1
            while end < total_lines:
                line = lines[end]
                # Blank lines and comments are always included
                if not line.strip() or line.strip().startswith("#"):
                    block.append(line)
                    end += 1
                    continue
                cur_indent = len(line) - len(line.lstrip())
                # A new def/class at same or lower indent ends this block
                if cur_indent <= base_indent and re.match(r"\s*(def |class )", line):
                    break
                block.append(line)
                end += 1
            return block, end

        # --- Collect priority blocks ---
        segments: list[tuple[int, list[str]]] = []  # (priority, lines)

        forward_idx = None
        init_idx = None
        loss_indices: list[int] = []

        for i, line in enumerate(lines):
            stripped = line.strip()
            if re.match(r"def forward\s*\(", stripped):
                forward_idx = i
            elif re.match(r"class \w+.*Model|class \w+.*Net|class \w+.*Network", stripped):
                # Skip test/debug/utility classes that aren't the main model
                skip_keywords = ("Test", "Debug", "Dummy", "Loader", "Helper", "Mock",
                                 "Sampler", "Dataset", "DataLoader", "Transform")
                class_name = stripped.split("class ")[1].split("(")[0].split(":")[0].strip()
                if any(kw in class_name for kw in skip_keywords):
                    continue
                # Find __init__ inside this class
                for j in range(i, min(i + 80, total_lines)):
                    if re.match(r"\s+def __init__\s*\(", lines[j]):
                        init_idx = j
                        break
            elif re.match(r"def .*(?:loss|criterion|compute_weight|routing_weight|gate_weight)", stripped, re.IGNORECASE):
                loss_indices.append(i)

        if forward_idx is not None:
            blk, _ = _block_from(forward_idx)
            segments.append((0, blk))  # highest priority
        if init_idx is not None:
            blk, _ = _block_from(init_idx)
            segments.append((1, blk))
        for li in loss_indices:
            blk, _ = _block_from(li)
            segments.append((2, blk))

        # Sort by priority, then assemble within budget
        segments.sort(key=lambda s: s[0])
        result_parts: list[str] = []
        used = 0
        seen_line_sets: set[int] = set()

        for _pri, blk in segments:
            # Deduplicate: skip blocks we already included
            blk_start_hash = hash(blk[0]) if blk else 0
            if blk_start_hash in seen_line_sets:
                continue
            seen_line_sets.add(blk_start_hash)

            block_text = "\n".join(blk)
            if used + len(block_text) + 3 > budget:
                # Partially include — take as many lines as fit
                remaining = budget - used - 3
                if remaining > 50:
                    partial = "\n".join(blk[: remaining // max(1, len(blk[0]) + 1)])
                    result_parts.append(partial + "\n# ... (truncated)")
                    used += len(partial) + 20
                break
            result_parts.append(block_text)
            used += len(block_text) + 1

        # If still under budget, append tail lines
        if used < budget:
            tail_text = "\n".join(lines[-max(total_lines // 5, 20):])
            remaining = budget - used
            if len(tail_text) > remaining:
                tail_text = tail_text[:remaining] + "\n# ... (tail truncated)"
            if tail_text.strip():
                result_parts.append("# --- Additional code (tail) ---\n" + tail_text)

        assembled = "\n\n".join(result_parts)
        # Final safety truncate
        if len(assembled) > budget:
            assembled = assembled[:budget] + "\n# ... (overall truncated)"
        return assembled

    def _llm_code_review(self, model_content: str, train_script_content: str) -> list:
        """Phase 2: LLM semantic code review using fast model.

        Uses smart code extraction to fit the most important code segments
        (forward(), __init__(), loss functions) within the token budget,
        instead of naive head-truncation that misses critical logic.

        Now also injects code_review_lessons from the knowledge base so
        the LLM checks for past mistakes.
        """
        warnings = []
        try:
            # Smart extraction: prioritize forward() > __init__() > loss fns > tail
            model_snippet = self._extract_key_code_segments(model_content, budget=5000)
            train_snippet = (
                self._extract_key_code_segments(train_script_content, budget=3000)
                if train_script_content else "(not found)"
            )

            review_prompt = (
                "You are a deep learning architecture reviewer. Review the following model code "
                "and training script for CRITICAL design flaws. Focus ONLY on issues that would "
                "cause training to fail silently (model trains but key mechanisms don't work).\n\n"
                "Common critical patterns to check:\n"
                "1. Loss function doesn't match the model's objective (e.g., BCE for multi-class routing)\n"
                "2. Gradient disconnection: detach() or stop_gradient in wrong place\n"
                "3. Information bottleneck: layer dimensions too small for task\n"
                "4. Dead modules: parameters that receive no gradient (e.g., unused forward path)\n"
                "5. Data processing: wrong normalization, missing augmentation for specific branch\n"
                "6. Target/label construction bugs: wrong shape, wrong values, type mismatch\n\n"
            )

            # ── Inject knowledge base lessons ──
            # Search for lessons relevant to the current code
            try:
                relevant_lessons = self.memory.search_relevant_lessons(model_content, limit=5)
                if relevant_lessons:
                    lesson_text = self.memory.format_lessons_for_context(relevant_lessons, max_chars=800)
                    review_prompt += (
                        "KNOWN PAST MISTAKES (CHECK FOR THESE IN THE CODE):\n"
                        + lesson_text + "\n\n"
                    )
            except Exception:
                pass

            review_prompt += (
                "MODEL CODE:\n```python\n" + model_snippet + "\n```\n\n"
                "TRAINING SCRIPT:\n```python\n" + train_snippet + "\n```\n\n"
                "Respond in this EXACT format (one issue per line):\n"
                "SEVERITY|category|description\n"
                "Where SEVERITY is HIGH/MEDIUM/LOW, category is loss/gradient/arch/data/other.\n"
                "If no issues found, respond with: OK|none|No critical issues detected\n"
                "Maximum 5 issues."
            )

            # Use fast model tier for code review (cheap)
            # IMPORTANT: task_tier must NOT be in STRONG_MODEL_TASKS to use the cheap model.
            # "code" is in STRONG_MODEL_TASKS, so we use "review" instead.
            response_text, _trace = self.dispatcher._call_llm(
                system="You are a concise code reviewer. Only report REAL issues.",
                messages=[{"role": "user", "content": review_prompt}],
                tools=None,
                max_turns=1,
                task_tier="review",  # fast model (not in STRONG_MODEL_TASKS)
            )

            # Parse structured response
            for line in response_text.strip().split("\n"):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("|", 2)
                if len(parts) == 3:
                    severity, category, detail = parts
                    severity = severity.strip().upper()
                    if severity not in ("HIGH", "MEDIUM", "LOW"):
                        continue
                    if detail.strip().lower().startswith("no "):
                        continue
                    warnings.append({
                        "severity": severity,
                        "detail": f"[LLM Review:{category.strip()}] {detail.strip()}",
                    })

        except Exception as e:
            logger.warning(f"LLM code review skipped (non-critical): {e}")
            # LLM review failure should NOT block execution

        return warnings

    def _post_reflect_code_review(self, execute_result: dict, reflect_result: dict,
                                   verify_report=None):
        """Post-REFLECT code review: learn from mistakes and record lessons.

        This is the knowledge-extraction step that turns failures into reusable
        knowledge base entries. It runs AFTER every REFLECT phase, extracting:

        1. VERIFY failures → architectural lessons (pattern, fix, evidence)
        2. Dead-end decisions → anti-pattern lessons
        3. Module failures → specific bug patterns

        Lessons are stored in code_review_lessons table and automatically
        injected into future THINK and code-review phases.
        """
        try:
            # ── 1. Extract lessons from VERIFY failures ──
            if verify_report and hasattr(verify_report, 'all_failures'):
                for check in verify_report.all_failures:
                    if hasattr(check, 'name') and hasattr(check, 'detail'):
                        self._extract_lesson_from_verify_failure(check)

            # ── 2. Extract lessons from reflect dead-end ──
            dead_end = reflect_result.get("dead_end", "")
            if dead_end:
                self._extract_lesson_from_dead_end(dead_end)

            # ── 3. Extract lessons from module failure ──
            module_failure = reflect_result.get("module_failure", "")
            if module_failure:
                self._extract_lesson_from_module_failure(module_failure)

            # ── 4. Run LLM-based lesson extraction for experiment failures ──
            # When experiment failed but no clear lesson was extracted above,
            # use LLM to analyze the failure and extract a lesson.
            if (not execute_result.get("experiment_launched") or
                    reflect_result.get("dead_end")):
                self._llm_extract_lesson(execute_result, reflect_result)

        except Exception as e:
            logger.warning(f"Post-reflect code review failed (non-critical): {e}")

    def _extract_lesson_from_verify_failure(self, check):
        """Convert a VERIFY check failure into a code review lesson."""
        name = getattr(check, 'name', '')
        detail = getattr(check, 'detail', '')
        severity = getattr(check, 'severity', 'medium').upper()
        category = getattr(check, 'category', 'other')

        if not detail:
            return

        # Map verify category to lesson category
        cat_map = {
            "data": "data",
            "model": "architecture",
            "training": "training",
            "output": "output",
            "metric": "metric",
        }
        lesson_cat = cat_map.get(category, category)

        # Extract a pattern from the check name
        pattern = name.replace("_", " ").lower() if name else "unknown_failure"

        self.memory.record_code_review_lesson(
            cycle=self.cycle_count,
            severity=severity if severity in ("HIGH", "MEDIUM", "LOW") else "MEDIUM",
            category=lesson_cat,
            pattern=pattern,
            description=detail[:500],
            evidence=f"VERIFY failure in cycle {self.cycle_count}: {name}",
            source="verify",
        )

    def _extract_lesson_from_dead_end(self, dead_end: str):
        """Extract a lesson from a dead-end decision."""
        if not dead_end or len(dead_end) < 20:
            return

        # Use first significant words as pattern to avoid merging all dead ends
        words = re.findall(r'[a-zA-Z_]{4,}', dead_end[:100])
        pattern = "_".join(words[:4]).lower() if words else "dead_end_approach"

        self.memory.record_code_review_lesson(
            cycle=self.cycle_count,
            severity="MEDIUM",
            category="strategy",
            pattern=pattern,
            description=dead_end[:500],
            evidence=f"Dead end in cycle {self.cycle_count}",
            source="reflect",
        )

    def _extract_lesson_from_module_failure(self, module_failure: str):
        """Extract a lesson from a module failure."""
        if not module_failure or len(module_failure) < 10:
            return

        # Try to categorize the module failure
        mf_lower = module_failure.lower()
        category = "other"
        if any(kw in mf_lower for kw in ["import", "module not found", "attribute"]):
            category = "import"
        elif any(kw in mf_lower for kw in ["shape", "dimension", "size mismatch", "channel"]):
            category = "architecture"
        elif any(kw in mf_lower for kw in ["nan", "inf", "overflow", "underflow"]):
            category = "numerical"
        elif any(kw in mf_lower for kw in ["gradient", "backward", "loss"]):
            category = "gradient"
        elif any(kw in mf_lower for kw in ["data", "dataset", "loader", "dataloader"]):
            category = "data"

        # Use first significant words as pattern to preserve distinctiveness
        words = re.findall(r'[a-zA-Z_]{3,}', module_failure[:100])
        pattern = "_".join(words[:4]).lower() if words else f"{category}_failure"

        self.memory.record_code_review_lesson(
            cycle=self.cycle_count,
            severity="HIGH",
            category=category,
            pattern=pattern,
            description=module_failure[:500],
            evidence=f"Module failure in cycle {self.cycle_count}",
            source="verify",
        )

    def _llm_extract_lesson(self, execute_result: dict, reflect_result: dict):
        """Use LLM to analyze a failed experiment and extract a reusable lesson.

        Only runs when other extraction methods didn't find specific lessons.
        Uses the fast model to keep costs low.
        """
        response = reflect_result.get("response", "")
        dead_end = reflect_result.get("dead_end", "")
        if not response and not dead_end:
            return

        # Don't extract lessons every cycle — only on failures
        if not dead_end and execute_result.get("experiment_launched"):
            return

        combined_text = f"Dead end: {dead_end}\nReflect: {response[:500]}"
        if len(combined_text) < 50:
            return

        try:
            lesson_prompt = (
                "Analyze this failed experiment and extract ONE reusable code review lesson.\n"
                "Focus on the ROOT CAUSE — what code pattern caused the failure?\n\n"
                f"FAILURE CONTEXT:\n{combined_text[:1000]}\n\n"
                "Respond in this EXACT format:\n"
                "PATTERN|CATEGORY|SEVERITY|DESCRIPTION|FIX_SUGGESTION\n\n"
                "Where:\n"
                "- PATTERN: a short code pattern to watch for (e.g., 'softmax_without_temperature')\n"
                "- CATEGORY: architecture|gradient|data|training|numerical|strategy\n"
                "- SEVERITY: HIGH|MEDIUM|LOW\n"
                "- DESCRIPTION: what went wrong and why (1-2 sentences)\n"
                "- FIX_SUGGESTION: how to prevent this (1 sentence)\n\n"
                "If no clear lesson can be extracted, respond with: SKIP|none|LOW|No lesson|N/A"
            )

            response_text, _trace = self.dispatcher._call_llm(
                system="You extract concise code review lessons from failed experiments.",
                messages=[{"role": "user", "content": lesson_prompt}],
                tools=None,
                max_turns=1,
                task_tier="review",
            )

            # Handle possible JSON-wrapped response from tool-based APIs
            try:
                parsed = json.loads(response_text)
                if isinstance(parsed, dict) and "content" in parsed:
                    response_text = parsed["content"]
                elif isinstance(parsed, list) and parsed:
                    response_text = parsed[0].get("content", response_text)
            except (json.JSONDecodeError, TypeError):
                pass

            # Parse the structured response
            line = response_text.strip().split("\n")[0].strip()
            parts = line.split("|", 4)
            if len(parts) == 5:
                pattern, category, severity, description, fix = parts
                if pattern.strip().upper() == "SKIP":
                    return
                self.memory.record_code_review_lesson(
                    cycle=self.cycle_count,
                    severity=severity.strip().upper(),
                    category=category.strip(),
                    pattern=pattern.strip(),
                    description=description.strip()[:500],
                    fix_suggestion=fix.strip()[:300],
                    source="llm_reflect",
                )

        except Exception as e:
            logger.debug(f"LLM lesson extraction skipped: {e}")

    def _reflect(self, execute_result: dict, verify_report=None, visual_analysis_result=None) -> dict:
        """REFLECT phase: evaluate results and update memory.

        Now receives VERIFY report with module-level diagnosis. The Leader
        MUST address verify failures before drawing conclusions about the
        experiment's success or failure.

        Also receives VisualAnalysisResult when training is stuck — the Leader
        can use multimodal image-based diagnosis to understand WHY the model fails.
        """
        logger.info("REFLECT phase starting...")

        context = {
            "brief": self.memory.get_brief(),
            "memory_log": self.memory.get_log(),
            "experiment_result": execute_result,
            "cycle": self.cycle_count,
            "workspace_dir": str(self.workspace),
        }

        # ── v16: Gap-closing reflection ──
        # Inject phase gap context so REFLECT compares results against targets
        phase_focus = self._build_phase_focus()
        if phase_focus:
            context["phase_focus"] = phase_focus

        # Inject VERIFY diagnosis so Leader knows what actually worked/failed
        if verify_report:
            context["verify_report"] = verify_report.to_dict()
            if verify_report.has_failures:
                context["verify_diagnosis"] = verify_report.diagnosis
                context["verify_failed_modules"] = verify_report.failed_modules

            # ── ANTI-DECEPTION: Flag LLM fabrication if detected ──
            fabrication_checks = [
                c for c in verify_report.checks
                if c.name in ("llm_fabrication", "pid_trace_mismatch")
                and c.status == "fail"
            ]
            if fabrication_checks:
                context["llm_fabrication_detected"] = True
                context["fabrication_details"] = [
                    f"[{c.severity.upper()}] {c.name}: {c.detail}"
                    for c in fabrication_checks
                ]
                logger.error(
                    f"ANTI-DECEPTION: LLM fabrication detected in cycle "
                    f"{self.cycle_count}: {[c.detail for c in fabrication_checks]}"
                )
                # Auto-log as active problem so it persists across cycles
                self.memory.log_active_problem(
                    f"LLM fabrication detected: {[c.detail[:100] for c in fabrication_checks]}. "
                    f"The Code agent claimed actions it did not perform. "
                    f"Previous cycle results are UNRELIABLE."
                )

        # ── Inject DATASET QUALITY diagnosis ──
        # Let the Leader know when metrics are statistically unreliable
        if verify_report and verify_report.dataset_issues:
            ds_issues = verify_report.dataset_issues
            context["dataset_quality_issues"] = ds_issues.get("issues", [])
            context["dataset_val_counts"] = ds_issues.get("val_counts", {})
            context["dataset_train_counts"] = ds_issues.get("train_counts", {})
            context["dataset_quality_prompt"] = (
                "DATASET QUALITY WARNING:\n"
                f"Validation scene counts: {ds_issues.get('val_counts', {})}\n"
                f"Training scene counts: {ds_issues.get('train_counts', {})}\n"
                f"Issues: {'; '.join(ds_issues.get('issues', []))}\n\n"
                "YOU MUST:\n"
                "1. If any domain has < 3 validation scenes, the MAE for that domain "
                "is STATISTICALLY UNRELIABLE — do NOT treat it as a real signal.\n"
                "2. If a metric is based on 1 scene, any change < 0.1 is noise — "
                "do NOT celebrate 'improvements' or panic about 'degradation'.\n"
                "3. Consider whether the dataset split needs to be fixed before "
                "continuing experiments — fixing data is often more important "
                "than tuning models.\n"
                "4. If you find a dataset problem, log it as an active problem "
                "and suggest a fix."
            )
            logger.warning(
                f"Injecting dataset quality issues into REFLECT: "
                f"{len(ds_issues.get('issues', []))} issues found"
            )

        # Inject VISUAL ANALYSIS diagnosis (when available)
        if visual_analysis_result and visual_analysis_result.triggered:
            context["visual_analysis"] = visual_analysis_result.to_dict()
            if visual_analysis_result.diagnosis:
                va_diags = []
                for d in visual_analysis_result.diagnosis:
                    if isinstance(d, dict):
                        va_diags.append(f"[{d.get('category','?')}/{d.get('confidence','?')}] {d.get('description', '')[:300]}")
                    else:
                        va_diags.append(str(d)[:300])
                context["visual_analysis_diagnosis"] = va_diags
                logger.info(
                    f"Injecting visual analysis into REFLECT: "
                    f"{len(va_diags)} findings, severity={visual_analysis_result.severity}"
                )
            if visual_analysis_result.recommended_actions:
                context["visual_analysis_actions"] = visual_analysis_result.recommended_actions

        # ── Fix 3 (结果分析): Inject cross-domain analysis prompt ──
        # Force the Leader to analyze WHY different domains perform differently.
        # This prevents the agent from only reporting "MAE went up/down" without
        # understanding the structural reasons.
        final_metrics = execute_result.get("final_metrics") or {}
        domain_maes = {}
        # Dynamically discover per-domain metrics (MAE_{Domain}, val_MAE_{Domain}, etc.)
        domain_keys = getattr(self.memory, 'domain_keys', [])
        all_metric_keys = set(domain_keys)
        # Also scan final_metrics for any key matching MAE_* or *_MAE pattern
        for key in final_metrics:
            if re.match(r"(MAE_|.*_MAE)", key):
                all_metric_keys.add(key)
        for key in all_metric_keys:
            if key in final_metrics:
                try:
                    domain_maes[key] = float(final_metrics[key])
                except (TypeError, ValueError):
                    pass

        if domain_maes:
            # Detect worst domain and its gap from best (generic, no hardcoded names)
            mae_values = {k: v for k, v in domain_maes.items()
                          if v is not None and math.isfinite(v)}
            if mae_values:
                best_domain = min(mae_values, key=mae_values.get)
                worst_domain = max(mae_values, key=mae_values.get)
                domain_gap = mae_values[worst_domain] - mae_values[best_domain]

                context["domain_analysis_prompt"] = (
                    "CROSS-DOMAIN ANALYSIS REQUIRED:\n"
                    f"Current results: {domain_maes}\n"
                    f"Best overall MAE: {self._best_metric_ever:.4f}\n"
                    f"Domain breakdown: {mae_values}\n"
                    f"Worst domain: {worst_domain} (MAE={mae_values[worst_domain]:.4f}), "
                    f"Best domain: {best_domain} (MAE={mae_values[best_domain]:.4f})\n"
                    f"Domain gap: {domain_gap:.4f}\n\n"
                    "You MUST answer these questions:\n"
                    "1. WHY does the model perform differently across domains? What is the ROOT CAUSE?\n"
                    "2. Is there a domain where performance is severely degraded? "
                    "If yes, what does this tell you about the METHOD'S fundamental assumptions?\n"
                    "3. Could the method's core assumption be VIOLATED in the "
                    "worst-performing domain? If so, incremental tuning will NOT help — you need a "
                    "fundamentally different approach.\n"
                    "4. What is the estimated CEILING of the current approach? If you've been iterating "
                    "for 3+ cycles without improvement in a domain, the method may have reached its limit.\n"
                    "5. Should you STOP pursuing the current direction and search for a fundamentally "
                    "different method? Justify your answer."
                )

                # ── RESULT-TO-ARCHITECTURE FEEDBACK ──
                # When domain gap is large (> 0.10), force structural analysis
                if domain_gap > 0.10:
                    context["architecture_feedback_prompt"] = (
                        "RESULT-TO-ARCHITECTURE FEEDBACK (MANDATORY):\n"
                        f"The domain gap is {domain_gap:.4f} — this is LARGE and indicates a "
                        f"STRUCTURAL problem, not a tuning problem.\n\n"
                        f"Worst domain: {worst_domain} = {mae_values[worst_domain]:.4f}\n"
                        f"Best domain: {best_domain} = {mae_values[best_domain]:.4f}\n\n"
                        "You MUST follow this reasoning chain:\n"
                        "1. IDENTIFY: What architectural component processes the input for the worst domain?\n"
                        "2. ASSUMPTION: What physical assumption does that component encode?\n"
                        "3. VERIFY: Is that assumption valid for the worst domain's data characteristics?\n"
                        "   - Example: a component assuming smooth input, violated by noisy data.\n"
                        "   - FFT → assumes frequency patterns are stable. Violated by noise/aliasing.\n"
                        "   - Mean pool → assumes all views equally informative. Violated when some views are occluded.\n"
                        "   - Conv3D → assumes regular input structure. Violated when patterns are irregular.\n"
                        "4. DIAGNOSE: If the assumption is violated, the component is fundamentally unsuitable.\n"
                        "5. FIX: Design an alternative that does NOT rely on the violated assumption.\n\n"
                        "CRITICAL: Do NOT propose incremental changes (loss weights, data augmentation, "
                        "learning rate) for a structural problem. These will NOT fix the root cause.\n"
                        "Instead, use the Code agent's analyze_model or probe_model tool to inspect "
                        "the architecture before proposing changes."
                    )

        # ── Fix 1 (实验设计): Inject hypothesis validation prompt ──
        # When a domain is severely degraded, force the agent to verify
        # whether the method's core assumptions hold in that domain.
        self._inject_training_curve_analysis(context, execute_result)

        # ── EXPERIMENT EVALUATOR INJECTION ──
        # Post-experiment evaluation: plan vs result, failure diagnosis, iteration guidance
        try:
            from .experiment_evaluator import ExperimentEvaluator
            evaluator = ExperimentEvaluator(
                self.project_dir, self.workspace,
                thresholds={"severe_degradation": 0.35, "improvement_threshold": 0.005},
            )

            # Get the architecture plan (from previous think or cached)
            arch_plan = getattr(self, '_last_architecture_plan', None)
            if not arch_plan:
                # Try to get from the plan file or re-generate
                plan_path = self.workspace / "ARCHITECTURE_PLAN.json"
                if plan_path.exists():
                    arch_plan = json.loads(plan_path.read_text())
                else:
                    from .idea_planner import IdeaPlanner
                    planner = IdeaPlanner(self.workspace)
                    brief_path = self.workspace / "PROJECT_BRIEF.md"
                    if brief_path.exists():
                        arch_plan = planner.plan()

            if arch_plan:
                eval_result = evaluator.evaluate(
                    experiment_results=execute_result,
                    architecture_plan=arch_plan,
                    model_path=execute_result.get("model_path", ""),
                )
                context["experiment_evaluation"] = eval_result

                # If there are critical/high diagnoses, inject them prominently
                critical_guidance = [
                    g for g in eval_result.get("iteration_guidance", [])
                    if g.get("priority") in ("critical", "high")
                ]
                if critical_guidance:
                    context["iteration_guidance_prompt"] = (
                        "EXPERIMENT EVALUATION — MANDATORY NEXT STEPS:\n"
                        + "\n".join(
                            f"[{g['priority'].upper()}] {g['action']}: {g['expected_improvement']}"
                            for g in critical_guidance[:3]
                        )
                        + "\n\nYou MUST address these issues before launching the next experiment. "
                        + "Do NOT repeat the same training configuration."
                    )
        except Exception as e:
            logger.warning(f"Experiment evaluator failed: {e}")

        # ── INDEPENDENT ASSESSMENT INJECTION ──
        # If the independent probe found anomalies during VERIFY, inject them
        if verify_report and verify_report.independent_assessment:
            ind_assess = verify_report.independent_assessment
            if ind_assess.get("anomaly_detected"):
                context["independent_assessment_warning"] = (
                    f"INDEPENDENT THIRD-PARTY ASSESSMENT WARNING:\n"
                    f"An independent probe (separate from your model's evaluation code) "
                    f"detected: {ind_assess.get('detail', 'unknown')}\n"
                    f"Agreement score: {ind_assess.get('agreement_score', 0):.2f}/1.0\n"
                    f"Confidence: {ind_assess.get('confidence', 'low')}\n\n"
                    f"Your reported metrics may be UNRELIABLE. The probe suggests the model's "
                    f"outputs are not what the metrics claim. Investigate the output quality "
                    f"before trusting the metrics."
                )

        # v16.1: PlannerChecker and QuickBenchmark removed (dead modules, scores 2/10 and 1/10)

        # ── v11: SIMULATION SANDBOX — full model evaluation ──
        # Run A/B comparison + internal behavior + scaling guidance
        if execute_result.get("experiment_launched") or execute_result.get("final_metrics"):
            try:
                model_path = execute_result.get("model_path", "") or self._extract_model_path_from_task(
                    {"task": str(execute_result.get("tool_trace", ""))}
                )
                if model_path:
                    # Find snapshot from before this cycle
                    snapshot_before = self.sandbox.find_previous_snapshot(self.cycle_count)
                    model_before = str(snapshot_before) if snapshot_before else ""

                    # Find checkpoint
                    ckpt_path = ""
                    for candidate in [
                        self.project_dir / "checkpoints" / "best_model.pth",
                        self.project_dir / "outputs",
                    ]:
                        if candidate.is_file():
                            ckpt_path = str(candidate.relative_to(self.project_dir))
                            break
                        elif candidate.is_dir():
                            pths = list(candidate.glob("**/best_model.pth"))
                            if pths:
                                ckpt_path = str(pths[0].relative_to(self.project_dir))
                                break

                    # Run full evaluation (all 5 layers)
                    sandbox_report = self.sandbox.full_evaluation(
                        cycle=self.cycle_count,
                        model_path=model_path,
                        model_path_before=model_before,
                        checkpoint_path=ckpt_path,
                        target_gpu_mb=self.sandbox.target_gpu_mb,
                        project_brief_path="PROJECT_BRIEF.md",
                    )

                    # Format and inject sandbox report into context
                    sandbox_prompt = self.sandbox.format_report_prompt(sandbox_report)
                    if sandbox_prompt:
                        context["sandbox_evaluation"] = sandbox_prompt

                    logger.info(
                        f"Sandbox: feasible={sandbox_report.feasible}, "
                        f"judgment={sandbox_report.judgment.get('modification_verdict', 'N/A')}, "
                        f"dead={sandbox_report.internal_behavior.get('dead_modules', [])}"
                    )

                    # Cache verdict for next THINK phase
                    try:
                        verdict_cache = self.workspace / "_sandbox_last_verdict.json"
                        cache_data = {
                            "modification_verdict": sandbox_report.judgment.get("modification_verdict", ""),
                            "effective_modules": sandbox_report.judgment.get("effective_modules", []),
                            "ineffective_modules": sandbox_report.judgment.get("ineffective_modules", []),
                            "scalable_modules": [m["name"] for m in sandbox_report.scaling.get("scalable_modules", [])],
                            "bottleneck_modules": [b.get("location", "") for b in sandbox_report.scaling.get("bottlenecks", [])],
                            "recommended_actions": [
                                sandbox_report.judgment.get("recommendation", ""),
                                sandbox_report.scaling.get("recommendation", ""),
                            ],
                        }
                        verdict_cache.write_text(json.dumps(cache_data, ensure_ascii=False))
                    except Exception:
                        pass

            except Exception as e:
                logger.debug(f"Sandbox evaluation skipped: {e}")

        # v16.1: ImplementationTracker removed (dead module)

        # ── v12: ANALYSIS EXPERIMENT REFLECTION ──
        # When the experiment was a data analysis (no training), inject specialized
        # reflection prompts that force the Leader to evaluate method coverage.
        is_analysis = (
            not execute_result.get("experiment_launched", False)
            and not execute_result.get("is_paper_research", False)
            and execute_result.get("response", "")  # has output
        )
        if is_analysis:
            # Count feature families from output (heuristic: check for known patterns)
            response_text = execute_result.get("response", "") or ""
            analysis_output = str(execute_result.get("output", "")) or response_text

            # Inject analysis-specific reflection prompt
            context["analysis_reflection_prompt"] = (
                "ANALYSIS EXPERIMENT REFLECTION (v12):\n"
                "This was a DATA ANALYSIS experiment, not model training. You MUST evaluate:\n\n"
                "1. METHOD COVERAGE: How many INDEPENDENT analysis methods were used?\n"
                "   - 1 method → INSUFFICIENT (cannot conclude direction is infeasible)\n"
                "   - 2 methods → WEAK (need at least 1 more)\n"
                "   - 3+ methods → ADEQUATE (can draw conclusions)\n\n"
                "2. FEATURE DIVERSITY: Are the features measuring DIFFERENT physical properties?\n"
                "   - Example: FFT energy ratios at 3 frequency bands = 1 family, not 3\n"
                "   - Example: FFT shape + spatial gradient + view consistency = 3 families\n\n"
                "3. DC-DOMINANCE: If using frequency-domain methods, what fraction of energy is DC?\n"
                "   - DC > 90% → frequency energy ratios are degenerate (useless for discrimination)\n"
                "   - In this case, you MUST try non-frequency methods before concluding\n\n"
                "4. CORRECT FAILURE CATEGORY:\n"
                "   - If < 3 independent methods tried and all show no signal → method_inadequacy\n"
                "   - If ≥ 3 independent methods tried and ALL show no signal → hypothesis_wrong\n"
                "   - If ANY method shows Cohen's d > 0.8 → direction HAS potential\n\n"
                "5. CRITICAL: Do NOT extrapolate 'method X doesn't work' to 'the entire direction doesn't work'.\n"
                "   Example: 'FFT energy ratios cannot discriminate materials' ≠ 'no angular feature can discriminate materials'"
            )

            # Check for method-inadequacy dead ends in recent history
            method_inadequacy_count = self.memory.get_method_inadequacy_count()
            if method_inadequacy_count > 0:
                context["method_inadequacy_history"] = (
                    f"WARNING: {method_inadequacy_count} previous dead_end(s) were categorized as "
                    f"'method_inadequacy'. This means the ANALYSIS METHOD was too narrow, "
                    f"not the hypothesis being wrong. Consider retrying with broader analysis "
                    f"before abandoning this direction."
                )

        # ── v12.2: TRAINING EXPERIMENT ARCHITECTURE REFLECTION ──
        # When the experiment was a training run (not analysis), inject
        # specialized reflection prompts for architecture-level issues.
        is_training = (
            execute_result.get("experiment_launched", False)
            and not execute_result.get("is_paper_research", False)
        )
        if is_training:
            # Check for routing-related VERIFY issues from Layer 12
            verify_report_dict = verify_report.to_dict() if verify_report else {}
            routing_issues = [
                c for c in verify_report_dict.get("checks", [])
                if c.get("name") in ("routing_differentiation", "aux_loss_convergence",
                                     "domain_regression")
                and c.get("status") in ("fail", "warn")
            ]

            if routing_issues:
                context["training_architecture_reflection_prompt"] = (
                    "TRAINING ARCHITECTURE REFLECTION (v12.2):\n"
                    "VERIFY detected architectural convergence issues. You MUST evaluate:\n\n"
                    "1. ROUTING/FUSION CONVERGENCE:\n"
                    "   - Did routing weights differentiate across domains?\n"
                    "   - If all domains have ~50/50 weights, the router is NOT learning.\n"
                    "   - Possible causes: aux_weight too low, routing target [0.5,0.5]\n"
                    "     for majority class, router input lacks discriminative info.\n\n"
                    "2. AUX LOSS CONVERGENCE:\n"
                    "   - Is aux_loss actually decreasing across epochs?\n"
                    "   - If aux_loss is flat, the auxiliary module receives no useful gradient.\n"
                    "   - Consider: higher aux_weight, separate optimizer for router,\n"
                    "     or pre-training the router with material classification GT.\n\n"
                    "3. PER-DOMAIN REGRESSION:\n"
                    "   - Did ANY domain get WORSE compared to the baseline?\n"
                    "   - A domain regressing >20% means the new mechanism is HARMFUL for it.\n"
                    "   - The new component may need a domain-specific on/off switch.\n\n"
                    "4. CORRECT FAILURE CATEGORY:\n"
                    "   - If routing weights did not differentiate → implementation_bug\n"
                    "     (the architecture cannot learn what it's supposed to)\n"
                    "   - If overall MAE improved but specific domains regressed →\n"
                    "     method_inadequacy (the approach helps some domains but hurts others)\n"
                    "   - Do NOT classify as hypothesis_wrong unless ≥3 independent\n"
                    "     architecture variants all fail the same way.\n\n"
                    f"VERIFY issues:\n"
                    + "\n".join(
                        f"  - [{c.get('severity','?')}] {c.get('detail', '')[:200]}"
                        for c in routing_issues[:5]
                    )
                )
                logger.info(
                    f"Injecting training architecture reflection: "
                    f"{len(routing_issues)} VERIFY issues"
                )

        # ── v10: STRATEGY CONSTRAINT ENGINE — generate rules from history ──
        try:
            self.strategy_engine.generate_rules_from_history(self.memory)
        except Exception as e:
            logger.debug(f"Strategy rule generation skipped: {e}")

        # ── v10: CONTEXT PRUNING ──
        context = self.context_pruner.prune(context, "reflect")

        # ── v12.1: REFLECT with quota-exhaustion fallback ──
        # If the LLM call fails (e.g. insufficient_quota, all providers down),
        # use a rule-based degraded reflect instead of losing the entire cycle's
        # EXECUTE + VERIFY results.
        try:
            result = self.dispatcher.dispatch_leader(
                task="reflect",
                context=context,
            )
        except (RuntimeError, Exception) as reflect_err:
            err_msg = str(reflect_err)
            is_quota_error = (
                "insufficient_quota" in err_msg
                or "All providers failed" in err_msg
                or "429" in err_msg
                or "quota" in err_msg.lower()
            )
            if is_quota_error:
                logger.warning(
                    f"REFLECT LLM call failed (quota/API error): {err_msg[:200]}. "
                    f"Using degraded rule-based reflect to preserve cycle results."
                )
                result = self._degraded_reflect(
                    execute_result, verify_report, context
                )
            else:
                raise

        # Update memory based on reflection
        if result.get("milestone"):
            self.memory.log_milestone(result["milestone"])
        if result.get("decision"):
            self.memory.log_decision(result["decision"])
        if result.get("dead_end"):
            failure_cat = result.get("failure_category", "")
            self.memory.log_dead_end(result["dead_end"], failure_category=failure_cat)
        if result.get("active_problem"):
            self.memory.log_active_problem(result["active_problem"])

        # Paper research: always log as major event
        if execute_result.get("is_paper_research") and result.get("milestone"):
            self.memory.log_major_event(result["milestone"])

        # Validate context keys against registry
        try:
            from .context_keys import validate_context
            key_warnings = validate_context(context, "reflect")
            for w in key_warnings:
                logger.debug(f"Context key: {w}")
        except Exception:
            pass

        return result

    def _degraded_reflect(
        self, execute_result: dict, verify_report, context: dict
    ) -> dict:
        """Rule-based fallback when REFLECT's LLM call fails (e.g. quota exhausted).

        Instead of losing the entire cycle's EXECUTE + VERIFY results, this method
        generates a basic reflection from available structured data:
        - VERIFY report (module-level pass/fail)
        - Experiment results (metrics, training logs)
        - Memory log (dead ends, active problems)

        The output follows the same JSON schema as Leader REFLECT so downstream
        code (_record_cycle_outcome, _update_state) works unchanged.
        """
        import json as _json

        # ── 1. Extract key facts from available data ──
        experiment_launched = execute_result.get("experiment_launched", False)
        is_paper_research = execute_result.get("is_paper_research", False)
        response_text = execute_result.get("response", "") or ""
        output_text = str(execute_result.get("output", "")) or response_text

        # VERIFY status
        verify_pass = 0
        verify_fail = 0
        verify_warnings = []
        if verify_report:
            verify_pass = verify_report.pass_count if hasattr(verify_report, 'pass_count') else 0
            verify_fail = verify_report.fail_count if hasattr(verify_report, 'fail_count') else 0
            if hasattr(verify_report, 'all_failures'):
                verify_warnings = [
                    f"[{c.category}] {c.name}: {c.detail}" for c in verify_report.all_failures
                ]

        # ── 2. Extract metrics from training logs ──
        metrics_summary = ""
        training_logs = execute_result.get("training_log", "") or ""
        if not training_logs:
            training_logs = self._load_state().get("last_training_logs", "")

        # Try to find best val_MAE
        best_mae = None
        mae_matches = re.findall(r"(?:val_MAE|Best val_MAE)[:\s=]+([\d.]+)", training_logs)
        if mae_matches:
            best_mae = min(float(m) for m in mae_matches)

        # Try to find per-domain MAE
        domain_maes = {}
        for domain in ["Lambertian", "Non-Lambertian", "Urban", "Mixed", "light_field_4d", "Overall"]:
            patterns = [
                rf"{domain}[^)]*?MAE[=:]\s*([\d.]+)",
                rf"MAE_{domain}[=:]\s*([\d.]+)",
            ]
            for pat in patterns:
                m = re.search(pat, training_logs)
                if m:
                    domain_maes[domain] = float(m.group(1))
                    break

        if best_mae is not None:
            metrics_summary = f"Best val_MAE={best_mae:.4f}"
        if domain_maes:
            metrics_summary += " | " + " ".join(
                f"{k}={v:.4f}" for k, v in domain_maes.items()
            )

        # ── 3. Extract accuracy from analysis results ──
        analysis_info = ""
        acc_match = re.search(r"(?:accuracy|Accuracy)[\s:=]+([\d.]+)%?", output_text)
        auc_match = re.search(r"(?:AUC|auc)[\s:=]+([\d.]+)", output_text)
        if acc_match:
            analysis_info += f" Accuracy={acc_match.group(1)}%"
        if auc_match:
            analysis_info += f" AUC={auc_match.group(1)}"

        # ── 4. Build decision ──
        if verify_fail > 0:
            verify_status = f"VERIFY: {verify_pass} passed, {verify_fail} FAILED"
        else:
            verify_status = f"VERIFY: all {verify_pass} checks passed"

        if experiment_launched:
            if best_mae is not None:
                decision = (
                    f"[DEGRADED REFLECT — API quota exhausted] "
                    f"Cycle {self.cycle_count}: Experiment completed. {metrics_summary}. "
                    f"{verify_status}. "
                    f"LLM REFLECT unavailable — results preserved for next cycle."
                )
            else:
                decision = (
                    f"[DEGRADED REFLECT — API quota exhausted] "
                    f"Cycle {self.cycle_count}: Experiment launched but metrics extraction failed. "
                    f"{verify_status}. "
                    f"LLM REFLECT unavailable — review results manually."
                )
        elif is_paper_research:
            decision = (
                f"[DEGRADED REFLECT — API quota exhausted] "
                f"Cycle {self.cycle_count}: Paper research completed. "
                f"LLM REFLECT unavailable — findings preserved."
            )
        elif output_text:
            # Analysis experiment
            decision = (
                f"[DEGRADED REFLECT — API quota exhausted] "
                f"Cycle {self.cycle_count}: Data analysis completed.{analysis_info} "
                f"{verify_status}. "
                f"LLM REFLECT unavailable — results preserved for next cycle."
            )
        else:
            decision = (
                f"[DEGRADED REFLECT — API quota exhausted] "
                f"Cycle {self.cycle_count}: No experiment output available. "
                f"{verify_status}."
            )

        # ── 5. Detect success/failure heuristically ──
        milestone = ""
        dead_end = ""
        failure_category = ""
        active_problem = ""

        if experiment_launched and best_mae is not None:
            # Heuristic: check if MAE improved vs. known baselines
            # We don't have the exact target, so just report the result
            milestone = (
                f"[Degraded] Cycle {self.cycle_count} experiment: {metrics_summary}. "
                f"Full analysis deferred (API quota)."
            )
        elif not experiment_launched and output_text:
            # Analysis experiment — check if results suggest hypothesis confirmed
            if any(kw in output_text.lower() for kw in ["hypothesis confirmed", "recommendation: hypothesis confirmed", "feasible"]):
                milestone = (
                    f"[Degraded] Cycle {self.cycle_count} analysis: hypothesis appears confirmed. "
                    f"{analysis_info}. Full analysis deferred."
                )
            elif any(kw in output_text.lower() for kw in ["not supported", "failed", "infeasible"]):
                # Check if this might be method_inadequacy
                if verify_warnings:
                    dead_end = (
                        f"[Degraded] Cycle {self.cycle_count} analysis: negative result. "
                        f"Could not verify method coverage (LLM unavailable). "
                        f"Categorize as method_inadequacy pending full REFLECT."
                    )
                    failure_category = "method_inadequacy"
                else:
                    dead_end = (
                        f"[Degraded] Cycle {self.cycle_count} analysis: negative result. "
                        f"Full analysis deferred."
                    )

        if verify_fail > 0:
            active_problem = (
                f"[Degraded REFLECT Cycle {self.cycle_count}] "
                f"VERIFY found {verify_fail} failure(s): {'; '.join(verify_warnings[:3])}. "
                f"LLM REFLECT was unavailable — investigate in next cycle."
            )

        # ── 6. Write a directive hint for next cycle ──
        # If we degraded, tell the next cycle to re-reflect on this one
        degraded_note_path = self.workspace / ".degraded_reflect_pending"
        try:
            degraded_note_path.write_text(_json.dumps({
                "cycle": self.cycle_count,
                "reason": "api_quota_exhausted",
                "metrics_summary": metrics_summary,
                "verify_status": verify_status,
                "has_milestone": bool(milestone),
                "has_dead_end": bool(dead_end),
            }))
        except Exception:
            pass

        # ── 7. Return in Leader REFLECT format ──
        return {
            "decision": decision,
            "milestone": milestone,
            "dead_end": dead_end,
            "failure_category": failure_category,
            "active_problem": active_problem,
            "reason": "LLM REFLECT unavailable (API quota exhausted)",
            "task": "",
            "_degraded": True,
        }

    def _refresh_obsidian(self, reflect_result: dict, directive: Optional[str]):
        if not self.obsidian.is_enabled():
            return
        self.obsidian.refresh_dashboard(memory=self.memory, cycle_count=self.cycle_count)
        self.obsidian.append_daily_entry(
            memory=self.memory,
            cycle_count=self.cycle_count,
            event_type="cycle_complete",
            reflection=reflect_result,
            directive=directive,
        )

    def _plan_signature(self, plan: dict) -> str:
        """Build a stable signature for repeated-plan detection."""
        normalized = {
            "action": plan.get("action", ""),
            "agent": plan.get("agent", ""),
            "task": " ".join(plan.get("task", "").split())[:300],
            "hypothesis": " ".join(plan.get("hypothesis", "").split())[:200],
        }
        return json.dumps(normalized, sort_keys=True, ensure_ascii=True)

    # v18 Phase 3: __extract_direction_signature removed (0 triggers in production, research decision)
    # ── Known architecture names for architecture-level detection (v14) ──
    _ARCHITECTURE_PATTERNS = {
        "epi": ["epi", "epinet", "epipolar", "epi_net", "epi slope", "epi branch"],
        "unet": ["unet", "u-net", "u_net", "unet_decoder", "unet_encoder"],
        "transformer": ["transformer", "vit", "attention_is_all", "self_attention"],
        "cnn": ["resnet", "vgg", "mobilenet", "efficientnet", "densenet", "inception"],
        "graph": ["gnn", "graph", "gcn", "gat", "message_passing"],
        "lfnet": ["lfnet", "lf_net", "lfanet", "lf_network"],
        "oacc": ["oacc", "occlusion_aware", "occlusion-aware"],
        "mvsnet": ["mvsnet", "mvs_net", "multi_view_stereo"],
        "dpt": ["dpt", "dense_prediction_transformer"],
        "adaspike": ["adaspike", "spike", "spiking"],
    }

    # v18 Phase 3: __extract_architecture_name removed (0 triggers in production, research decision)
    # v18 Phase 3: __analyze_architecture_dead_ends removed (0 triggers in production, research decision)
    # ── v16: Phase-Gated State-Driven Architecture ──

    # v16.1: _build_scope_prefix removed (pure text injection ineffective against LLM)

    def _load_phase_status(self, force_reload: bool = False) -> dict:
        """Load PHASE_STATUS.json — the single source of truth for phase state.

        Results are cached per cycle. Use force_reload=True after _save_phase_status.
        """
        if not force_reload and hasattr(self, '_phase_status_cache') and self._phase_status_cache is not None:
            return self._phase_status_cache
        path = self.workspace / "PHASE_STATUS.json"
        if path.exists():
            try:
                self._phase_status_cache = json.loads(path.read_text())
                return self._phase_status_cache
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Failed to load PHASE_STATUS.json: {e}")
        self._phase_status_cache = {}
        return {}

    def _save_phase_status(self, phase_status: dict):
        """Persist PHASE_STATUS.json."""
        path = self.workspace / "PHASE_STATUS.json"
        try:
            path.write_text(json.dumps(phase_status, indent=2, ensure_ascii=False))
            self._phase_status_cache = phase_status  # Keep cache in sync
        except OSError as e:
            logger.warning(f"Failed to save PHASE_STATUS.json: {e}")

    def _build_phase_focus(self) -> str:
        """Build a compact phase focus prompt (5-7 lines) for state-driven THINK.

        Instead of asking LLM "what should we do next?" (open-ended),
        we ask "here's the gap to the target, propose how to close it" (structured).
        This constrains the answer space and naturally prevents deviation.
        """
        ps = self._load_phase_status()
        if not ps:
            return ""

        current_id = ps.get("current_phase", "")
        phases = ps.get("phases", {})
        current = phases.get(current_id, {})
        if not current:
            return ""

        status = current.get("status", "PENDING")
        results = current.get("results", {})
        targets = current.get("targets", {})
        focus = current.get("focus_methods", [])
        name = current.get("name", current_id)

        # Phase already validated → allow free exploration
        if status == "VALIDATED":
            return ""

        # Phase FAILED → force replan
        if status == "FAILED":
            return (
                f"⛔ PHASE GATE — Phase '{name}' has FAILED.\n"
                f"Results: {results}\nTargets: {targets}\n"
                f"You MUST research alternative approaches or analyze why current approach failed.\n"
                f"DO NOT proceed to next phase."
            )

        # Phase PARTIAL/PENDING → structured gap-closing prompt
        gap = {}
        for metric, target in targets.items():
            actual = results.get(metric)
            if actual is not None and target > 0:
                gap[metric] = round(target - actual, 4)

        gap_str = ", ".join(f"{k}: {v:+.4f}" for k, v in gap.items()) if gap else "no data yet — run baseline first"
        results_str = ", ".join(f"{k}: {v:.4f}" for k, v in results.items()) if results else "no experiments run yet"
        targets_str = ", ".join(f"{k}: ≥{v:.4f}" for k, v in targets.items()) if targets else "see PROJECT_BRIEF"
        focus_str = ", ".join(focus[:6]) if focus else "see PROJECT_BRIEF"

        if not results:
            instruction = "Run a baseline experiment using focus methods to establish initial results."
        else:
            instruction = "Propose a concrete experiment to close the gap using focus methods above."

        return (
            f"PHASE FOCUS: Phase '{name}' (status={status})\n"
            f"Target: {targets_str}\n"
            f"Current: {results_str}\n"
            f"Gap to close: {gap_str}\n"
            f"Focus methods: {focus_str}\n"
            f"{instruction}"
        )

    def _check_phase_blocked(self, think_result: dict) -> tuple[bool, str]:
        """v16.1: Check if proposed action violates current phase blocked_patterns.
        
        CRITICAL CHANGE from v16: Scan ACTUAL CODE FILES instead of task text.
        v16 failed because LLM-generated task descriptions use abstract language
        that doesn't match blocked_patterns keywords (train, epoch, Conv, etc.).
        
        This method reuses the file-location logic from PRE-EXECUTE CODE REVIEW,
        which successfully blocked 6 training attempts by checking actual code.

        Returns (is_blocked, reason).
        """
        ps = self._load_phase_status()
        if not ps:
            return False, ""

        current_id = ps.get("current_phase", "")
        phases = ps.get("phases", {})
        current = phases.get(current_id, {})
        if not current:
            return False, ""

        status = current.get("status", "PENDING")
        # Only block if phase is not yet VALIDATED
        if status == "VALIDATED":
            return False, ""

        blocked_patterns = current.get("blocked_patterns", [])
        if not blocked_patterns:
            return False, ""

        # v16.1: Scan ACTUAL CODE FILES, not task text
        code_to_check = ""
        files_checked = []
        
        # 1. Check model file (reuse _extract_model_path_from_task logic)
        model_rel_path = self._extract_model_path_from_task(think_result)
        if model_rel_path:
            model_path = self.project_dir / model_rel_path
            if model_path.exists():
                try:
                    code_to_check += model_path.read_text()
                    files_checked.append(model_rel_path)
                except Exception:
                    pass
        
        # 2. Check training script (reuse _find_training_script_content logic)
        train_script_content = self._find_training_script_content(think_result)
        if train_script_content:
            code_to_check += "\n" + train_script_content
            files_checked.append("training_script")
        
        # No code files found (pure analysis task) → allow
        if not code_to_check:
            return False, ""
        
        # Scan code against blocked patterns
        matched_patterns = []
        for pattern in blocked_patterns:
            try:
                if re.search(pattern, code_to_check, re.IGNORECASE):
                    matched_patterns.append(pattern)
            except re.error:
                logger.warning(f"Invalid blocked_pattern skipped: {pattern}")
                continue
        
        if matched_patterns:
            name = current.get("name", current_id)
            focus = current.get("focus_methods", [])
            return True, (
                f"⛔ PHASE GATE v2 BLOCKED: Phase '{name}' status is {status}. "
                f"Code files contain blocked patterns.\n"
                f"Files checked: {', '.join(files_checked)}\n"
                f"Blocked patterns matched: {', '.join(matched_patterns[:3])}\n"
                f"Focus on: {', '.join(focus[:5])}"
            )

        return False, ""

    def _update_phase_status_from_results(self, results: dict):
        """After REFLECT: auto-compare experiment results with phase targets.

        Updates PHASE_STATUS.json status: VALIDATED / PARTIAL / FAILED.
        """
        ps = self._load_phase_status()
        if not ps:
            return

        current_id = ps.get("current_phase", "")
        phases = ps.get("phases", {})
        current = phases.get(current_id, {})
        if not current:
            return

        targets = current.get("targets", {})
        if not targets:
            return

        # Determine metric direction: higher_is_better for accuracy/auc,
        # lower_is_better for error/mae/rmse/loss
        def _is_higher_better(metric_name: str) -> bool:
            lower_keywords = ("error", "mae", "rmse", "loss", "mse", "cost")
            return not any(kw in metric_name.lower() for kw in lower_keywords)

        # Extract relevant metrics from results
        new_results = {}
        all_met = True
        any_improved = False
        for metric, target in targets.items():
            # Match priority: exact key > val_ prefixed > case-insensitive
            actual = None
            # 1. Exact match
            if metric in results and isinstance(results[metric], (int, float)):
                actual = float(results[metric])
            else:
                # 2. Prefixed match (e.g., val_auc, test_auc) — case-insensitive
                for prefix in ("val_", "test_", "best_"):
                    prefixed = f"{prefix}{metric}"
                    for key, val in results.items():
                        if key.lower() == prefixed.lower() and isinstance(val, (int, float)):
                            actual = float(val)
                            break
                    if actual is not None:
                        break
            if actual is None:
                # 3. Case-insensitive match as last resort
                for key, val in results.items():
                    if metric.lower() == key.lower() and isinstance(val, (int, float)):
                        actual = float(val)
                        break

            if actual is not None:
                new_results[metric] = round(actual, 4)
                hib = _is_higher_better(metric)
                met = (actual >= target) if hib else (actual <= target)
                if not met:
                    all_met = False
                # Check if improved from previous
                prev = current.get("results", {}).get(metric)
                if prev is None or ((actual > prev) if hib else (actual < prev)):
                    any_improved = True

        if not new_results:
            return

        # Update results
        current["results"] = new_results
        current["attempts"] = current.get("attempts", 0) + 1
        current["last_updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")

        # Determine status
        if all_met:
            current["status"] = "VALIDATED"
            logger.info(f"✅ PHASE GATE PASSED: {current_id} meets all criteria!")
            # Auto-advance to next phase
            phase_ids = list(phases.keys())
            idx = phase_ids.index(current_id) if current_id in phase_ids else -1
            if idx + 1 < len(phase_ids):
                next_id = phase_ids[idx + 1]
                next_phase = phases[next_id]
                if next_phase.get("status") == "BLOCKED":
                    next_phase["status"] = "PENDING"
                    ps["current_phase"] = next_id
                    logger.info(f"→ Auto-advancing to {next_id}: {next_phase.get('name', '')}")
        else:
            # Check if close (PARTIAL) or far (FAILED)
            worst_ratio = 1.0
            for metric, target in targets.items():
                if metric in new_results and target > 0:
                    hib = _is_higher_better(metric)
                    actual = new_results[metric]
                    # For higher_is_better: ratio = actual/target (e.g. 0.85/0.90 = 0.94)
                    # For lower_is_better: ratio = target/actual (e.g. 0.10/0.15 = 0.67)
                    if hib:
                        ratio = actual / target
                    else:
                        ratio = target / actual if actual > 0 else 0
                    worst_ratio = min(worst_ratio, ratio)
            if worst_ratio < 0.4:
                current["status"] = "FAILED"
                logger.warning(f"❌ PHASE GATE FAILED: {current_id} far below targets (worst ratio={worst_ratio:.2f})")
            else:
                current["status"] = "PARTIAL"

        phases[current_id] = current
        ps["phases"] = phases
        self._save_phase_status(ps)

    # ── v15: Research Roadmap Integration ──

    def _init_research_roadmap(self):
        """Initialize the research roadmap from PROJECT_BRIEF on first cycle.

        Called once; subsequent cycles read the persisted file.
        """
        try:
            # Try loading existing ROADMAP first
            if self.roadmap.load():
                logger.info(f"ROADMAP loaded from file: {len(self.roadmap.modules)} modules")
                self._roadmap_initialized = True
                return

            # Generate from PROJECT_BRIEF + architecture plan
            brief_path = self.workspace / "PROJECT_BRIEF.md"
            if not brief_path.exists():
                # Try project root
                brief_path = self.project_dir / "PROJECT_BRIEF.md"

            arch_plan = getattr(self, '_last_architecture_plan', None)
            result = self.roadmap.generate_from_brief(brief_path, arch_plan=arch_plan)

            if result.get("status") == "ok":
                self._roadmap_initialized = True
                self.memory.log_decision(
                    f"[ROADMAP v15] Research roadmap generated: "
                    f"{len(self.roadmap.modules)} modules, "
                    f"phase={self.roadmap.global_phase.value}"
                )
                logger.info(
                    f"ROADMAP initialized: {result.get('global_phase')}, "
                    f"{len(result.get('modules', []))} modules"
                )
            else:
                logger.warning(f"ROADMAP generation failed: {result.get('message', 'unknown')}")
        except Exception as e:
            logger.warning(f"ROADMAP init failed: {e}")

    def _enforce_roadmap_alignment(self, think_result: dict) -> dict:
        """Check THINK output against ROADMAP and correct deviations.

        v16 simplification: The phase gate already blocks based on PHASE_STATUS.
        This method now only handles roadmap-level deviations (not phase violations).
        Single hard gate: misaligned → redirect immediately (no 3-strike warming).
        """
        try:
            alignment = self.roadmap.check_alignment(think_result)

            if alignment["aligned"]:
                self._phase_violation_count = 0
                return think_result

            # Misaligned — single hard gate
            deviation_type = alignment["deviation_type"]
            current_modules = alignment.get("current_modules", [])
            self._phase_violation_count += 1

            logger.warning(
                f"ROADMAP DEVIATION (#{self._phase_violation_count}): "
                f"type={deviation_type}, modules={current_modules}, "
                f"task={think_result.get('task', '')[:100]}"
            )

            self.memory.log_decision(
                f"[ROADMAP v16] Deviation: {deviation_type}. "
                f"Active modules: {', '.join(current_modules[:3])}"
            )

            # v16: Single hard gate — force paper_research on ANY deviation
            # (PHASE GATE already handles the common case; this catches roadmap-level drift)
            if self._phase_violation_count >= 2:
                logger.warning("ROADMAP: Forcing paper_research after 2 consecutive deviations")
                self._phase_violation_count = 0
                self.roadmap.reset_deviation_count()
                return {
                    "action": "paper_research",
                    "agent": "paper_researcher",
                    "task": (
                        "ROADMAP DEVIATION DETECTED — repeated deviations from the plan.\n\n"
                        "You MUST research methods to verify the following modules:\n"
                        + "\n".join(f"  - {m}" for m in current_modules)
                        + "\n\nFind published methods or techniques to validate the theoretical "
                        "assumptions underlying each module BEFORE implementing them."
                    ),
                    "reason": (
                        f"Forced paper_research: repeated deviations from ROADMAP. "
                        f"Agent keeps proposing {deviation_type} "
                        f"instead of addressing active modules."
                    ),
                }

            # 1st deviation: inject correction into result (one chance)
            correction = alignment.get("correction_prompt", "")
            if correction:
                original_task = think_result.get("task", "")
                think_result["_roadmap_alignment_warning"] = correction
                think_result["task"] = (
                    f"{original_task}\n\n"
                    f"---\n{correction}\n---"
                )

            return think_result

        except Exception as e:
            logger.warning(f"ROADMAP alignment check failed: {e}")
            return think_result  # Fail open — don't block on errors

    # v18 Phase 3: __apply_no_progress_fallback removed (0 triggers in production, research decision)
    # ─────────────────────────────────────────────────────────────
    # Phase 4: Failed-launch forced re-dispatch
    # ─────────────────────────────────────────────────────────────
    def _arbitrate_cycle(self, think_result: dict):
        """v18: Collect enforcement signals from all subsystems and let the
        SignalArbiter produce a unified CycleDirective.

        This is the SINGLE enforcement decision point. It replaces the
        scattered positional enforcement chain (Phase 4 + Fix B) with one
        priority-ordered arbitration. Signals are collected from:
        - Launch failures (_consecutive_failed_launches)
        - Audit escalations (_audit_enforcement)
        - Constraint engine (forbidden violations)
        - Human directives
        """
        self.arbiter.begin_cycle()

        # Signal: launch failures → CRITICAL if >= 2
        if self._consecutive_failed_launches >= 2:
            action = "pause_human" if self._consecutive_failed_launches >= 3 else "forced_fix"
            self.arbiter.add_signal(
                source="launch", key="research_roadmap",
                content="Launch enforcement active",
                severity="CRITICAL", forced_action=action,
                forced_task=self._enforce_launch_after_failure(think_result).get("task"),
                forced_reason=f"{self._consecutive_failed_launches} consecutive failed launches")

        # Signal: audit escalations → CRITICAL if >= 2
        for sig, count in self._audit_enforcement.items():
            if count >= 2:
                action = "pause_human" if count >= 3 else "forced_fix"
                self.arbiter.add_signal(
                    source="audit", key="phase_focus",
                    content=f"Audit issue: {sig}",
                    severity="CRITICAL", forced_action=action,
                    forced_task=self._enforce_audit_findings(think_result).get("task"),
                    forced_reason=f"Audit '{sig}' escalated {count} times")

        # Signal: constraint violations → CRITICAL if forbidden
        try:
            violations = self.strategy_engine.check_constraints(think_result, self.memory)
            if self.strategy_engine.has_forbidden_violation(violations):
                self.arbiter.add_signal(
                    source="constraint", key="persistent_constraints",
                    content="; ".join(violations),
                    severity="CRITICAL", forced_action="forced_fix",
                    forced_task=think_result.get("task", ""),
                    forced_reason="FORBIDDEN constraint violation")
        except Exception:
            pass

        # Signal: human directive (highest priority, always WARNING at minimum)
        directive = self._consume_directive if hasattr(self, '_last_directive') else None

        return self.arbiter.arbitrate("think")

    def _update_launch_counter(self, execute_result: dict):
        """Update the consecutive-failed-launch counter after EXECUTE.

        A genuine launch (experiment_launched=True) resets the counter to 0.
        A failed launch (convergence_failed or experiment_launched=False on an
        experiment plan, absent a tool error) increments it. The counter is
        monotonic — it never self-resets on firing, only on a real launch.
        """
        if execute_result.get("experiment_launched"):
            self._consecutive_failed_launches = 0
        elif execute_result.get("convergence_failed") or \
                (not execute_result.get("launch_error")
                 and not execute_result.get("deception_detected")
                 and not execute_result.get("experiment_launched")):
            self._consecutive_failed_launches += 1

    def _enforce_launch_after_failure(self, think_result: dict) -> dict:
        """Force action when consecutive failed launches stack up.

        Returns the (possibly rewritten) think_result:
          - < 2 failures: no change (the cycle proceeds normally).
          - 2 failures: rewrite the action/task to a forced 'fix + launch'
            dispatch, carrying the specific failure reason so the code agent
            knows what went wrong.
          - ≥ 3 failures: rewrite to pause_human — stop burning quota and
            surface the problem for human intervention.
        """
        n = self._consecutive_failed_launches
        if n < 2:
            return think_result

        if n >= 3:
            logger.error(
                f"⚠️  PAUSE-HUMAN: {n} consecutive cycles planned an experiment "
                f"but EXECUTE never launched one. Stopping to avoid burning "
                f"more quota. Last task: {str(think_result.get('task',''))[:100]}"
            )
            return {
                "action": "pause_human",
                "reason": (
                    f"{n} consecutive failed launches. The agent repeatedly "
                    f"plans an experiment but never calls launch_experiment. "
                    f"This is likely an infrastructure or prompt issue requiring "
                    f"human inspection."
                ),
                "task": think_result.get("task", ""),
            }

        # 2 failures: force a targeted 'fix + launch' re-dispatch.
        logger.warning(
            f"🔄 FORCED RE-DISPATCH: {n} consecutive failed launches. "
            f"Forcing a 'fix + launch' task this cycle."
        )
        original_task = think_result.get("task", "")
        return {
            "action": "experiment",
            "reason": (
                f"Forced re-dispatch after {n} failed launches. The previous "
                f"cycle(s) planned training but launch_experiment was never called."
            ),
            "task": (
                f"CRITICAL — PREVIOUS LAUNCH FAILED.\n\n"
                f"Last cycle you were asked to run an experiment but you did NOT "
                f"call launch_experiment. This is your final chance before the "
                f"system pauses for human intervention.\n\n"
                f"Original task: {original_task[:300]}\n\n"
                f"MANDATORY STEPS (do NOT deviate):\n"
                f"1. Verify the training script exists and is correct (one read_file).\n"
                f"2. Run a 2-step dry-run to confirm it works (one run_shell).\n"
                f"3. Call launch_experiment(command=..., log_file=...) IMMEDIATELY.\n"
                f"4. Do NOT explore further. Do NOT call read_file/list_files more "
                f"than once each. CONVERGE NOW.\n\n"
                f"If you cannot launch, explicitly report why in your response — "
                f"do NOT silently skip the launch."
            ),
            "_forced_redispatch": True,
        }

    def _enforce_audit_findings(self, think_result: dict) -> dict:
        """Fix B: Force action when repeated audit issues go uncorrected.

        Same enforcement model as _enforce_launch_after_failure: per-signature
        monotonic counter. If an audit issue has been escalated (DIRECTIVE
        written) but the agent still hasn't fixed it after 2 cycles, force
        the action to a targeted fix task. After 3, pause for human
        intervention. This replaces the old "advisory-only DIRECTIVE.md that
        the LLM ignores forever" pattern.

        Resets a signature's counter only when the issue stops recurring for
        2 consecutive cycles (grace period), not on fire.
        """
        if not self._audit_enforcement:
            return think_result

        for sig, n in sorted(self._audit_enforcement.items(), key=lambda x: -x[1]):
            if n >= 3:
                logger.error(
                    f"⛔ PAUSE-HUMAN: audit issue '{sig}' escalated {n} times "
                    f"without resolution. Stopping for human inspection."
                )
                self._audit_enforcement[sig] = 0  # reset on pause
                return {
                    "action": "pause_human",
                    "reason": (
                        f"Audit issue '{sig}' has been escalated {n} times. "
                        f"The agent cannot resolve this automatically — likely "
                        f"an infrastructure or prompt-level issue. Human "
                        f"inspection required."
                    ),
                    "task": think_result.get("task", ""),
                }
            if n >= 2:
                logger.warning(
                    f"🔄 FORCED FIX: audit issue '{sig}' escalated {n} times. "
                    f"Forcing a targeted fix task."
                )
                self._audit_enforcement[sig] = 0  # give the forced fix a clean slate
                return {
                    "action": think_result.get("action", "paper_research"),
                    "reason": f"Forced fix for recurring audit issue '{sig}'.",
                    "task": (
                        f"CRITICAL — RECURRING AUDIT ISSUE (escalated {n} times).\n\n"
                        f"Issue signature: {sig}\n"
                        f"This issue has persisted across multiple cycles despite "
                        f"directives. You MUST diagnose and fix it NOW:\n"
                        f"1. Identify the root cause (not the symptom)\n"
                        f"2. Apply the minimal fix\n"
                        f"3. Verify the fix resolves the issue\n"
                        f"4. Report what was wrong and what you changed\n\n"
                        f"Original task: {think_result.get('task', '')[:200]}\n"
                    ),
                    "_forced_audit_fix": True,
                }
        return think_result



    def _parse_goals_from_brief(self) -> list[dict]:
        """Parse target metrics from PROJECT_BRIEF. Shared by goal_progress and goal_achieved."""
        import re as _re
        brief = self.memory.get_brief()
        targets = []
        for m in _re.finditer(
            r'(?:(Lambertian|Non.Lambertian|Mixed|Urban|整体|overall)\s+\S*\s+)?'
            r'(?:val_)?(MAE|mae)\s*(?:<|>|<=|>=)\s*([0-9.]+)',
            brief, _re.IGNORECASE
        ):
            sub = m.group(1) or ""
            sub_map = {"lambertian": "Lambertian", "non-lambertian": "NonLambertian",
                       "non lambertian": "NonLambertian", "mixed": "Mixed",
                       "urban": "Urban", "整体": "overall", "overall": "overall"}
            sub_key = sub_map.get(sub.lower().strip(), "") if sub else ""
            full_key = f"val_MAE_{sub_key}" if sub_key else "val_MAE"
            try:
                val = float(m.group(3))
                if not any(t["key"] == full_key and t["target"] == val for t in targets):
                    targets.append({"key": full_key, "target": val})
            except ValueError:
                continue
        return targets

    def _build_goal_progress(self) -> str:
        """Phase 1: Build a goal progress string from PROJECT_BRIEF + SQLite.

        Extracts target metrics from the brief (e.g., "val_MAE < 0.20"),
        queries SQLite for the best achieved metric, and formats a progress
        report showing achieved vs unachieved targets.
        """
        targets = self._parse_goals_from_brief()

        if not targets:
            return ""

        # Query best metrics from SQLite
        lines = ["Target metrics progress:"]
        for t in targets[:5]:  # cap at 5 targets
            key = t["key"]
            target = t["target"]
            try:
                best = self.memory.get_best_metric(key)
            except Exception:
                best = None

            if best is not None:
                achieved = best < target  # lower is better for MAE-like metrics
                marker = "✅" if achieved else "❌"
                gap = best - target
                lines.append(f"  {marker} {key}: best={best:.4f} target<{target} gap={gap:+.4f}")
            else:
                lines.append(f"  ⬜ {key}: target<{target} (no result yet)")

        return "\n".join(lines)

    def _goal_achieved(self) -> bool:
        """Phase 1: Check if ALL target metrics from PROJECT_BRIEF are met.

        Only checks targets that have real data in SQLite. Sub-domain targets
        (e.g. val_MAE_Lambertian) that have no dedicated metric stored are
        skipped (not treated as unmet), because get_best_metric's fallback to
        val_MAE would use the wrong comparison (overall vs sub-domain target).
        """
        targets = self._parse_goals_from_brief()

        if not targets:
            return False  # no targets parsed → don't stop

        checked = 0
        for t in targets:
            key = t["key"]
            target = t["target"]
            # Only check the overall val_MAE — sub-domain metrics aren't
            # stored separately in SQLite. Checking them with val_MAE fallback
            # would compare overall vs sub-domain target (wrong comparison).
            if key != "val_MAE" and key != "val_MAE_overall":
                continue  # skip sub-domain targets (no data to verify)
            try:
                best = self.memory._query_best_raw(key)
            except Exception:
                best = None
            if best is None:
                continue  # no data → don't block on this target
            checked += 1
            if best >= target:
                return False  # overall target not met

        if checked == 0:
            return False  # nothing verifiable → don't stop
        logger.info("🎯 GOALS ACHIEVED (verifiable targets) — agent will stop.")
        return True

    def _record_cycle_outcome(self, think_result: dict, execute_result: dict, reflect_result: dict,
                              verify_report_dict: dict = None):
        """Track whether repeated cycles are producing real progress.

        Also records complete cycle outcome to SQLite database for
        permanent experiment history (survives MEMORY_LOG.md compaction).
        """
        # Record to SQLite database
        try:
            self.memory.record_cycle_outcome(
                cycle=self.cycle_count,
                think_result=think_result,
                execute_result=execute_result,
                reflect_result=reflect_result,
                verify_report=verify_report_dict,
            )
        except Exception as e:
            logger.warning(f"Failed to record cycle outcome to SQLite: {e}")

        # ── ROADMAP UPDATE (v15): Feed cycle outcome back to roadmap ──
        try:
            self.roadmap.update_from_cycle_outcome(
                think_result=think_result,
                reflect_result=reflect_result,
                cycle=self.cycle_count,
            )
        except Exception as e:
            logger.debug(f"ROADMAP update from cycle outcome skipped: {e}")

        if think_result.get("action") == "paper_research":
            # Paper research is always considered progress — it generates new knowledge
            self._no_progress_streak = 0
            self._last_no_progress_signature = ""
            self._metric_no_progress_streak = 0
            self._infra_failure_streak = 0
            # v14: Do NOT reset _architecture_stagnation_count — paper research alone
            # does not change the underlying architecture being used.
            # Only a real architecture switch (detected in _extract_architecture_name) resets it.
            return

        if think_result.get("action") != "experiment":
            if think_result.get("action") != "wait":
                self._no_progress_streak = 0
                self._last_no_progress_signature = ""
                if reflect_result.get("milestone"):
                    self._metric_no_progress_streak = 0
            return

        signature = self._plan_signature(think_result)
        made_progress = bool(
            execute_result.get("experiment_launched")
            or execute_result.get("final_metrics")
            or reflect_result.get("milestone")
        )

        # ── Metric-based progress tracking (Fix 1: visual analysis trigger) ──
        final_metrics = execute_result.get("final_metrics") or {}
        current_metric = None
        # v18 Phase 4: metric keys from config goals, fallback to defaults
        goal_metrics = self.config.get("goals", {}).get("metrics", [])
        metric_keys = tuple(g.get("key", "val_MAE") for g in goal_metrics) if goal_metrics else \
                      ("val_MAE", "val_MAE_overall", "best_val_MAE", "val_mae")
        for key in metric_keys:
            if key in final_metrics:
                try:
                    current_metric = float(final_metrics[key])
                except (TypeError, ValueError):
                    pass
                break
        # System deterministically writes a quantitative result line to
        # MEMORY_LOG.md. Previously, quantitative results (val_MAE=0.184)
        # only existed in SQLite but never reached MEMORY_LOG (the LLM's
        # text-based memory channel) unless the LLM happened to include
        # the number in its free-text milestone. Now the system guarantees
        # the number is always there.
        #
        # Fix: when monitor's final_metrics is empty (60% of experiments),
        # re-extract from the training log using the shared parser.
        if current_metric is None:
            # Try to extract from training log text
            log_file = execute_result.get("log_file", "")
            training_logs = execute_result.get("training_logs", "")
            from .training_log_parser import extract_metrics
            for source in (training_logs, log_file):
                if not source:
                    continue
                # If it's a file path, read it; if it's text, use directly
                log_text = source
                if isinstance(source, str) and len(source) < 500 and Path(source).exists():
                    try:
                        log_text = Path(source).read_text(errors="ignore")
                    except Exception:
                        continue
                parsed = extract_metrics(str(log_text))
                for mkey in ("val_mae", "val_mae_overall", "best_val_mae"):
                    if mkey in parsed:
                        try:
                            current_metric = float(parsed[mkey])
                        except (ValueError, TypeError):
                            pass
                        break
                if current_metric is not None:
                    break

        if current_metric is not None:
            method = self._extract_method_from_task(think_result.get("task", ""))
            status = "success" if made_progress else "inconclusive"
            try:
                self.memory.log_structured_result(
                    cycle=self.cycle_count,
                    metric_key="val_MAE",
                    metric_value=current_metric,
                    method=method,
                    status=status,
                )
            except Exception as e:
                logger.debug(f"Structured metric record skipped: {e}")

        # ── Fix 1: Output quality awareness ──
        # Detect domain-specific degradation (e.g., one domain's metric much worse than overall)
        domain_metrics = {}
        for key in final_metrics:
            if key.startswith("MAE_"):
                try:
                    domain_metrics[key] = float(final_metrics[key])
                except (TypeError, ValueError):
                    pass

        quality_degraded = False
        for domain_key, domain_val in domain_metrics.items():
            if domain_key in self._best_domain_metrics:
                best_val = self._best_domain_metrics[domain_key]
                # Degradation: > 10% worse than best for that domain
                # Only flag if best_val > 0.01 to avoid false positives near zero
                if best_val > 0.01 and domain_val > best_val * 1.10:
                    quality_degraded = True
                    logger.warning(
                        f"QUALITY DEGRADATION: {domain_key} = {domain_val:.4f} "
                        f"vs best = {best_val:.4f} ({(domain_val/best_val - 1)*100:.1f}% worse)"
                    )
                    break
            # Always update best — first occurrence or improvement
            if domain_key not in self._best_domain_metrics or domain_val < self._best_domain_metrics[domain_key]:
                self._best_domain_metrics[domain_key] = domain_val

        # ── PARETO MATRIX RECORDING ──
        # Record method×domain results for cross-experiment Pareto frontier tracking
        if domain_metrics:
            method_name = self._extract_method_from_task(think_result.get("task", ""))
            exp_type = "pilot" if think_result.get("pilot_experiment") else "full"
            for dk, dv in domain_metrics.items():
                domain_name = dk.replace("MAE_", "")
                try:
                    self.memory.record_pareto_entry(
                        cycle=self.cycle_count,
                        method=method_name,
                        domain=domain_name,
                        mae=dv,
                        experiment_type=exp_type,
                    )
                except Exception as e:
                    logger.debug(f"Pareto recording failed: {e}")

        # ── CAUSAL CHAIN UPDATE ──
        # Update actual effect for any causal links from this cycle's hypothesis
        if domain_metrics and think_result.get("hypothesis"):
            for dk, dv in domain_metrics.items():
                try:
                    self.memory.update_causal_actual(
                        cycle=self.cycle_count,
                        metric_affected=dk,
                        actual_effect=f"MAE={dv:.4f}",
                    )
                except Exception as e:
                    logger.debug(f"Causal actual update skipped: {e}")
        # Compare actual improvement with expected improvement
        if current_metric is not None and self._best_metric_ever < float('inf'):
            actual_improvement = self._best_metric_ever - current_metric
            was_correct = actual_improvement > 0
            try:
                self.memory.update_experiment_value_actual(
                    cycle=self.cycle_count,
                    actual_improvement=actual_improvement,
                    was_correct=was_correct,
                )
            except Exception as e:
                logger.debug(f"Experiment value update skipped: {e}")
        else:
            # v18 Phase 3: Direction/architecture stagnation tracking removed.
        # These counters had 0 triggers in 23 cycles of production. The LLM
        # should decide to change direction based on experiment history
        # (Phase 1 knowledge loop), not system-enforced stagnation detection.
            task_text = think_result.get("task", "")[:200]

        # Check architecture survey completion
        if not self._architecture_survey_done and self._architecture_survey_path.exists():
            self._architecture_survey_done = True
            logger.info("ARCHITECTURE SURVEY completed — survey file detected.")

        # ── Metric tracking (existing logic) ──
        if current_metric is not None:
            if not math.isfinite(current_metric):
                logger.warning(f"METRIC INVALID: {current_metric} — skipping metric tracking")
            else:
                improvement_threshold = 0.005  # v16.1: hardcoded (AdaptiveThresholds removed)
                if current_metric < (self._best_metric_ever * (1 + improvement_threshold)):
                    if current_metric < self._best_metric_ever:
                        logger.info(
                            f"METRIC IMPROVEMENT: {current_metric:.4f} < "
                            f"prev_best={self._best_metric_ever:.4f}"
                        )
                    self._best_metric_ever = min(self._best_metric_ever, current_metric)
                    self._metric_no_progress_streak = 0
                    self._consecutive_audit_directives = 0
                else:
                    self._metric_no_progress_streak += 1
                    logger.info(
                        f"METRIC NO PROGRESS: {current_metric:.4f} >= "
                        f"best={self._best_metric_ever:.4f} (streak={self._metric_no_progress_streak})"
                    )
        elif made_progress:
            pass

        if made_progress:
            self._no_progress_streak = 0
            self._last_no_progress_signature = ""
            return

        if signature == self._last_no_progress_signature:
            self._no_progress_streak += 1
        else:
            self._last_no_progress_signature = signature
            self._no_progress_streak = 1

    def _smart_cooldown(self):
        """Poll at short intervals instead of fixed long wait."""
        logger.info(f"Smart cooldown: polling every 60s for {self.cooldown}s")
        elapsed = 0
        while elapsed < self.cooldown and self._running:
            sleep_time = min(60, self.cooldown - elapsed)
            time.sleep(sleep_time)
            elapsed += sleep_time

            # Check if any experiment just finished
            if self.monitor.has_completed_experiments():
                logger.info("Experiment completed during cooldown. Waking up.")
                return

    def _cooldown_after_error(self):
        """Back off after an error to prevent burn loops."""
        backoff = min(self.cooldown * 2, 1800)  # Max 30 min
        logger.warning(f"Error backoff: waiting {backoff}s")
        # Check _running flag during backoff to allow graceful shutdown
        elapsed = 0
        while elapsed < backoff and self._running:
            sleep_chunk = min(60, backoff - elapsed)
            time.sleep(sleep_chunk)
            elapsed += sleep_chunk


    def _check_audit_escalation(self, issues: list[str]) -> list[dict]:
        """Check if any audit issue has repeated enough times to warrant escalation.

        Uses a sliding window approach: if an issue appeared in >= threshold
        of the last `sliding_window` cycles, it escalates. This prevents the
        agent from cycling through L1 fixes without ever escalating.

        Returns list of escalated issue dicts with:
        - signature: issue type key (e.g. "UNREGISTERED:Non-lambertian")
        - count: how many times this issue appeared in the sliding window
        - issue: the full issue text
        - action: recommended escalation action
        """
        import re

        escalated = []
        current_signatures = set()

        for issue in issues:
            # Extract a stable signature from the issue text
            sig_match = re.match(r"^(\w+):\s*(\S+)", issue)
            if sig_match:
                sig = f"{sig_match.group(1)}:{sig_match.group(2)}"
            else:
                sig = issue[:40].strip()

            current_signatures.add(sig)

            # Append current cycle number
            history = self._audit_issue_history.setdefault(sig, [])
            history.append(self.cycle_count)

            # Trim to sliding window
            cutoff = self.cycle_count - self._audit_sliding_window
            self._audit_issue_history[sig] = [c for c in history if c > cutoff]
            count = len(self._audit_issue_history[sig])

            if count >= self._audit_escalation_threshold:
                escalated.append({
                    "signature": sig,
                    "count": count,
                    "issue": issue,
                    "action": self._determine_escalation_action(sig, count),
                })

        # Remove signatures not seen this cycle (no longer recurring)
        for sig in list(self._audit_issue_history.keys()):
            if sig not in current_signatures:
                # Keep history but it will naturally age out of the sliding window
                pass

        return escalated

    def _determine_escalation_action(self, signature: str, count: int) -> str:
        """Determine what action to take for an escalating issue.

        Escalation levels:
        - Level 1 (count >= threshold): Log warning + targeted fix directive
        - Level 2 (count >= threshold * 2): Force error-handler skill
        - Level 3 (count >= threshold * 3): Pause agent + write directive file
        - Level 4 (count >= threshold * 4): Mark as unfixable dead_end
        """
        t = self._audit_escalation_threshold
        if count >= t * 4:
            return "mark_unfixable"
        elif count >= t * 3:
            return "pause_and_directive"
        elif count >= t * 2:
            return "force_error_handler"
        else:
            return "targeted_fix"

    def _handle_escalated_issues(self, escalated_issues: list[dict]):
        """Take action on repeatedly failing audit checks.

        Fix 5: Prevents audit death loops by capping consecutive directives.
        After MAX_CONSECUTIVE_AUDIT_DIRECTIVES (default 5) for the same or any
        audit issue, auto-clears the directive and lets the agent continue working.
        Infrastructure issues should not block experimental progress indefinitely.
        """
        MAX_CONSECUTIVE_AUDIT_DIRECTIVES = 5

        for item in escalated_issues:
            sig = item["signature"]
            count = item["count"]
            action = item["action"]
            issue = item["issue"]

            # Fix 5: Audit death loop protection — if too many consecutive directives,
            # clear the directive and let the agent proceed with real work.
            if action in ("targeted_fix", "force_error_handler"):
                self._consecutive_audit_directives += 1
                if self._consecutive_audit_directives >= MAX_CONSECUTIVE_AUDIT_DIRECTIVES:
                    logger.error(
                        f"⛔ AUDIT DEATH LOOP DETECTED: {self._consecutive_audit_directives} "
                        f"consecutive audit directives (latest: '{sig}', count={count}). "
                        f"AUTO-CLEARING directive to unblock agent. "
                        f"This infrastructure issue will be logged but will not block progress."
                    )
                    # Clear directive file to unblock next cycle
                    directive_path = self.workspace / "DIRECTIVE.md"
                    if directive_path.exists():
                        directive_path.unlink()
                        logger.info(f"Cleared {directive_path} to break audit death loop")
                    # Log as known issue, not a dead end (it may resolve itself)
                    self.memory.log_active_problem(
                        f"[AUDIT-LOOP] Issue '{sig}' appeared {count} times across "
                        f"{self._consecutive_audit_directives} directives. "
                        f"Auto-unblocked. Agent continues — this is a known infrastructure "
                        f"limitation, NOT an experiment blocker."
                    )
                    self._consecutive_audit_directives = 0
                    # Remove this issue from escalation history so it doesn't re-trigger immediately
                    self._audit_issue_history.pop(sig, None)
                    continue
            else:
                # Non-directive actions (pause, mark_unfixable) don't increment counter
                pass

            if action == "targeted_fix":
                # Level 1: Log a focused fix suggestion and inject into next THINK
                logger.warning(
                    f"🔄 AUDIT ESCALATION L1: '{sig}' appeared {count} times. "
                    f"Injecting targeted fix into next cycle."
                )
                self._inject_error_directive(sig, count, issue, level=1)
                # Fix B: also increment the enforcement counter so repeated
                # audit issues get real teeth (forced action rewrite / pause).
                self._audit_enforcement[sig] = self._audit_enforcement.get(sig, 0) + 1

            elif action == "force_error_handler":
                # Level 2: Force the error-handler skill
                logger.warning(
                    f"🔴 AUDIT ESCALATION L2: '{sig}' appeared {count} times. "
                    f"Forcing error-handler skill."
                )
                self._inject_error_directive(sig, count, issue, level=2)
                self._audit_enforcement[sig] = self._audit_enforcement.get(sig, 0) + 1

            elif action == "pause_and_directive":
                # Level 3: Pause the agent and write a human directive
                logger.error(
                    f"⛔ AUDIT ESCALATION L3: '{sig}' appeared {count} times. "
                    f"Agent is stuck. Writing human directive and pausing."
                )
                self._write_stuck_directive(sig, count, issue)
                self._running = False
                return

            elif action == "mark_unfixable":
                # Level 4: Record as dead_end and stop escalating
                logger.error(
                    f"🗑️ AUDIT ESCALATION L4: '{sig}' appeared {count} times. "
                    f"Marking as unfixable dead_end."
                )
                self.memory.log_dead_end(
                    f"AUDIT ISSUE MARKED UNFIXABLE after {count} attempts: {sig}. "
                    f"Agent will stop trying to fix this automatically."
                )
                self._audit_issue_history.pop(sig, None)

    def _inject_error_directive(self, signature: str, count: int, issue: str, level: int):
        """Write a targeted fix directive that the next cycle will pick up."""
        directive_path = self.workspace / "DIRECTIVE.md"

        # Try to inline the error-handler skill content so the code agent
        # doesn't have to find the file (workspace ≠ skills directory)
        skill_content = ""
        skill_path = Path(__file__).parent.parent / "skills" / "error-handler" / "SKILL.md"
        if skill_path.exists():
            skill_content = skill_path.read_text()

        if level == 1:
            prefix = "⚠️ TARGETED FIX REQUIRED"
            instruction = (
                f"The audit issue '{signature}' has appeared {count} consecutive cycles.\n"
                f"Full issue: {issue}\n\n"
                f"You MUST fix this EXACT issue before doing anything else.\n"
                f"Do NOT attempt training until this audit passes.\n"
            )
        else:  # level 2
            prefix = "🔴 ERROR-HANDLER SKILL REQUIRED"
            if skill_content:
                instruction = (
                    f"The audit issue '{signature}' has appeared {count} consecutive cycles.\n"
                    f"The agent has FAILED to fix this automatically {count} times.\n\n"
                    f"Follow the error-handler workflow below IN FULL:\n\n"
                    f"--- ERROR-HANDLER SKILL (inline) ---\n{skill_content}\n--- END SKILL ---\n\n"
                    f"Do NOT skip any step. Do NOT attempt training until the audit passes.\n\n"
                    f"Full issue: {issue}\n"
                )
            else:
                instruction = (
                    f"The audit issue '{signature}' has appeared {count} consecutive cycles.\n"
                    f"The agent has FAILED to fix this automatically {count} times.\n\n"
                    f"## Mandatory Diagnostic Steps\n"
                    f"1. Identify the EXACT error (read the error text above)\n"
                    f"2. Read the files involved in the error\n"
                    f"3. Diagnose the root cause (one-line diagnosis)\n"
                    f"4. Validate your diagnosis with a diagnostic command\n"
                    f"5. Apply the MINIMAL fix\n"
                    f"6. Verify the fix works\n"
                    f"7. Report what was wrong and what was fixed\n\n"
                    f"Do NOT skip any step. Do NOT attempt training until the audit passes.\n\n"
                    f"Full issue: {issue}\n"
                )

        content = f"{prefix}\n\n{instruction}"
        directive_path.write_text(content)
        logger.info(f"Error directive written to {directive_path}")

    def _write_stuck_directive(self, signature: str, count: int, issue: str):
        """Write a CRITICAL stuck notification for human review."""
        stuck_path = self.workspace / "AGENT_STUCK.md"
        content = (
            f"# ⛔ Agent Stuck — Human Intervention Required\n\n"
            f"**Time:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"**Cycle:** {self.cycle_count}\n"
            f"**Issue:** {signature}\n"
            f"**Recurring for:** {count} consecutive cycles\n\n"
            f"## Full Error\n```\n{issue}\n```\n\n"
            f"## What the Agent Tried\n"
            f"See workspace/MEMORY_LOG.md for recent decisions.\n\n"
            f"## Suggested Human Actions\n"
            f"1. Read the files involved in the error\n"
            f"2. Fix the root cause manually\n"
            f"3. Delete this file and restart the agent\n"
        )
        stuck_path.write_text(content)
        logger.error(f"Agent stuck notification written to {stuck_path}")
        self.memory.log_decision(
            f"⛔ AGENT STUCK: '{signature}' recurred {count} times. "
            "Paused for human intervention. See workspace/AGENT_STUCK.md"
        )


    def _dataset_manifest_exists(self) -> bool:
        """Check if DATASET_MANIFEST.json exists, is valid, and not too old (<=7 days).

        Skips re-running dataset understanding if a valid manifest already exists,
        saving ~40 tool calls per restart.
        """
        manifest = self.workspace / "DATASET_MANIFEST.json"
        if not manifest.exists():
            return False
        try:
            data = json.loads(manifest.read_text())
            datasets = data.get("datasets", {})
            if not datasets:
                return False
            # Check manifest age — re-scan if older than 7 days
            import time
            age_seconds = time.time() - manifest.stat().st_mtime
            max_age_seconds = 7 * 24 * 3600  # 7 days
            if age_seconds > max_age_seconds:
                logger.info(f"DATASET_MANIFEST.json is {age_seconds/86400:.1f} days old, will re-scan")
                return False
            return True
        except (json.JSONDecodeError, OSError):
            return False

    def _run_dataset_understanding(self):
        """Run dataset-understanding skill to validate data/ directory.

        This dispatches a code agent to:
        1. Scan all subdirectories in data/
        2. Validate GT formats for each dataset
        3. Verify unified_lf_dataset.py consistency
        4. Produce/update DATASET_MANIFEST.json
        """
        import json

        manifest_path = self.workspace / "DATASET_MANIFEST.json"
        existing_manifest = {}
        if manifest_path.exists():
            try:
                existing_manifest = json.loads(manifest_path.read_text())
            except Exception:
                pass

        task = (
            "DATASET UNDERSTANDING — mandatory first-cycle validation\n\n"
            "Read the full skill instructions: skills/dataset-understanding/SKILL.md\n\n"
            "You MUST write a Python inspection script and RUN it to scan data/.\n"
            "Do NOT manually guess or copy old manifests.\n\n"
            "The script must:\n"
            "1. List all subdirectories in data/\n"
            "2. For EACH dataset, for EACH scene:\n"
            "   - Find input image pattern and count views\n"
            "   - Find GT file (try gt_*.pfm, *_disparity.npy, disp_*.npy, depth/*.png, Depth/*.png)\n"
            "   - Load GT and validate: 2D shape, continuous values, NOT all-zero, NOT class IDs, NOT r_map.npy\n"
            "   - Record shape and value range\n"
            "   - Record native resolution of input images (H, W)\n"
            "3. Exclude scenes with no valid GT (empty depth/, all-zero, r_map.npy only)\n"
            "4. Assign train/val splits based on directory structure\n"
            "5. Add training_recommendations section:\n"
            "   - target_size for unified resolution across all datasets\n"
            "   - exclude_datasets for broken/unloadable datasets (h5 format etc.)\n"
            "   - Reason: different datasets have different native resolutions\n"
            "6. Write results to workspace/DATASET_MANIFEST.json\n"
            "7. Smoke test: from datasets import main_dataset_class; load 1 sample from each type\n"
            "8. DELETE the inspection script after success\n\n"
            "CRITICAL ISSUES TO WATCH FOR:\n"
            "- r_map.npy = reflectance (NOT depth). Scenes with ONLY r_map.npy have NO depth GT.\n"
            "- uint8 [1,11] = semantic labels (NOT depth)\n"
            "- h5 format files CANNOT be loaded by PIL/numpy → exclude entire dataset\n"
            "- Different datasets will have DIFFERENT native resolutions (e.g. 512x512 vs 926x926 vs 480x640)\n"
            "  Training MUST use target_size to unify these, or batching will crash!\n"
            "- The model's SharedViewEncoder dynamically infers view count from input channels,\n"
            "  so different num_views is OK as long as views are flattened to (B, V*C, H, W).\n"
            "\n"
            "MANDATORY DATA QUALITY CHECKS (Step 3.6 in SKILL.md):\n"
            "You MUST detect and report these problems in a 'data_quality_issues' section:\n"
            "1. Dataset imbalance: if any domain has < 10% of total scenes → HIGH severity\n"
            "2. Missing val split for a domain: if any domain type has 0 val scenes → CRITICAL\n"
            "   (e.g. a domain having only train scenes means you CANNOT evaluate it)\n"
            "3. GT duplication: if multiple scenes share the same GT file → MEDIUM severity\n"
            "4. GT range inconsistency: if max GT values differ > 3x across datasets → HIGH severity\n"
            "5. GT shape mismatch: if some GT arrays are (H,W) and others (1,H,W) → HIGH severity\n"
            "Each issue MUST include severity, detail, and a concrete recommendation.\n"
            "If issues are found, the training script MUST address them (domain-balanced sampling,\n"
            "split adjustments, deduplication, normalization).\n"
            f"Existing manifest path: {manifest_path}\n"
        )

        logger.info("Running dataset understanding scan...")
        result = self.dispatcher.dispatch_worker(
            agent_type="code",
            task=task,
            tools=self.tools.get_tools_for("code"),
        )
        logger.info(f"Dataset understanding completed: {str(result)[:300]}")

        # Verify the manifest was created/updated
        if manifest_path.exists():
            logger.info("DATASET_MANIFEST.json verified after scan")
        else:
            logger.warning("DATASET_MANIFEST.json not found after scan — dataset understanding may have failed")

    def _consume_directive(self) -> Optional[str]:
        """Read and consume directive files if present.

        Priority: HUMAN_DIRECTIVE.md (manual) > DIRECTIVE.md (auto error-handler)
        """
        # Check human directive first
        directive_path = self.workspace / "HUMAN_DIRECTIVE.md"
        if directive_path.exists():
            content = directive_path.read_text().strip()
            if content:
                # Archive the directive
                archive_dir = self.workspace / "directive_archive"
                archive_dir.mkdir(exist_ok=True)
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                # Use uuid-style suffix to avoid collision when cycles run <1s apart
                import uuid
                unique_suffix = uuid.uuid4().hex[:6]
                directive_path.rename(archive_dir / f"directive_{timestamp}_{unique_suffix}.md")
                logger.info(f"Consumed human directive: {content[:100]}...")
                return content

        # Check auto error-handler directive
        auto_directive_path = self.workspace / "DIRECTIVE.md"
        if auto_directive_path.exists():
            content = auto_directive_path.read_text().strip()
            if content:
                # Archive and consume
                archive_dir = self.workspace / "directive_archive"
                archive_dir.mkdir(exist_ok=True)
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                import uuid
                unique_suffix = uuid.uuid4().hex[:6]
                auto_directive_path.rename(archive_dir / f"auto_directive_{timestamp}_{unique_suffix}.md")
                logger.info(f"Consumed error-handler directive: {content[:100]}...")
                return content

        return None

    def _load_cycle_counter(self) -> int:
        counter_file = self.workspace / ".cycle_counter"
        if counter_file.exists():
            return int(counter_file.read_text().strip())
        return 0

    def _save_cycle_counter(self):
        counter_file = self.workspace / ".cycle_counter"
        counter_file.write_text(str(self.cycle_count))

    def _load_state(self) -> dict:
        if self.state_path.exists():
            try:
                return json.loads(self.state_path.read_text())
            except json.JSONDecodeError:
                return {}
        return {}

    def _update_state(self, updates: dict):
        state = self._load_state()
        state.update(updates)
        # v18: persist the signal arbiter's backlog so deferred signals
        # survive process restarts (previously they were in-memory only).
        state["signal_backlog"] = self.arbiter.get_backlog()
        # Atomic write: write to temp file first, then rename
        # This prevents state.json corruption if the process crashes mid-write
        tmp_path = self.state_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(state, indent=2))
        tmp_path.replace(self.state_path)

    def _handle_signal(self, signum, frame):
        logger.info(f"Received signal {signum}. Initiating graceful shutdown.")
        self._running = False
        if self.tools:
            self.tools.shutdown()

    # ── Domain Knowledge & Cross-Experiment Integration ──


    # Domain knowledge methods inherited from DomainKnowledgeMixin (see domain_knowledge.py)
    # Includes: _build_domain_knowledge, _build_cross_experiment_insights

    def _extract_method_from_task(self, task_text: str) -> str:
        """Extract the primary method name from a task description for Pareto tracking."""
        task_lower = task_text.lower()
        method_keywords = {
            "attention": ["attention", "angular attention", "self-attention"],
            "fft": ["fft", "fourier", "frequency", "angular_freq"],
            "epi": ["epi", "epinet", "epipolar"],
            "pcgrad": ["pcgrad", "gradient projection"],
            "domain_heads": ["domain_heads", "domain-specific head", "domain head"],
            "conv3d": ["conv3d", "3d conv"],
            "resnet": ["resnet", "backbone", "pretrained"],
            "dropout": ["dropout"],
            "ema": ["ema", "exponential moving average"],
        }
        for method, keywords in method_keywords.items():
            for kw in keywords:
                if kw in task_lower:
                    return method
        # If no method keyword found, extract first meaningful word combo
        words = task_text.split()[:5]
        return "_".join(w.lower() for w in words if len(w) > 3)[:30] or "unknown"

    def _estimate_experiment_value(self, think_result: dict):
        """Estimate the value of information for the proposed experiment.

        Uses a simple heuristic model:
        - VOI = expected_improvement × prior_probability
        - expected_improvement: how much the metric is expected to improve
        - prior_probability: based on calibration history

        Low-VOI experiments get a warning injected into the task.
        """
        hypothesis = think_result.get("hypothesis", "")
        success_criteria = think_result.get("success_criteria", "")

        # Estimate expected improvement from success criteria or hypothesis
        import re
        improvement_matches = re.findall(
            r'(?:mae|loss|error)[:\s]*<?\s*([0-9.]+)', hypothesis + " " + success_criteria,
            re.IGNORECASE
        )
        expected_improvement = 0.05  # Default assumption
        if improvement_matches:
            try:
                target = float(improvement_matches[0])
                # Improvement = distance from current best to target
                if self._best_metric_ever < float('inf'):
                    expected_improvement = max(0, self._best_metric_ever - target)
                else:
                    expected_improvement = target * 0.1  # Rough estimate
            except (TypeError, ValueError):
                pass

        # Prior probability from calibration
        try:
            calibration = self.memory.get_experiment_calibration()
            prior = calibration.get("accuracy", 0.5)
        except Exception:
            prior = 0.5  # Uninformative prior

        # Floor: never let prior drop below 0.1 — even with bad calibration,
        # a novel hypothesis might work. Without this floor, a streak of
        # failures drives prior to 0 and labels ALL experiments as LOW VOI.
        prior = max(prior, 0.1)

        # If the method has been tried before and failed, lower prior
        method_name = self._extract_method_from_task(think_result.get("task", ""))
        try:
            effect_matrix = self.memory.get_method_domain_effect_matrix()
            if method_name in effect_matrix:
                for domain, stats in effect_matrix[method_name].items():
                    if stats.get("success_rate", 1.0) < 0.3:
                        prior *= 0.5  # Penalize methods with low success rate
        except Exception as e:
            logger.debug(f"VOI method stats lookup skipped: {e}")
        prior = max(prior, 0.05)

        voi = expected_improvement * prior

        # Record for future calibration
        try:
            self.memory.record_experiment_value(
                cycle=self.cycle_count,
                hypothesis=hypothesis[:500],
                expected_improvement=expected_improvement,
                prior_probability=prior,
                information_value=voi,
            )
        except Exception as e:
            logger.debug(f"VOI recording skipped: {e}")
        if voi < 0.005 and prior < 0.3:
            logger.warning(
                f"LOW VOI EXPERIMENT: voi={voi:.4f}, prior={prior:.2f}, "
                f"expected_improvement={expected_improvement:.4f}. "
                f"This experiment is unlikely to produce useful information."
            )
            # Inject warning and strongly suggest paper_research instead
            think_result["task"] = (
                f"⚠️ LOW VALUE EXPERIMENT (VOI={voi:.4f}, success probability={prior:.0%})\n"
                f"This experiment has low estimated value based on past calibration. "
                f"STRONGLY consider switching to paper_research to find a NEW approach, or "
                f"choose a hypothesis you have NOT tried before.\n"
                f"If you must proceed with this experiment, run a pilot (2-3 epochs) first "
                f"to quickly validate the hypothesis.\n\n"
                f"--- ORIGINAL TASK ---\n{think_result.get('task', '')}"
            )

        # ── PILOT EXPERIMENT RECOMMENDATION ──
        # For high-risk hypotheses (prior < 0.4), recommend a pilot run
        if prior < 0.4 and not think_result.get("pilot_experiment"):
            think_result["pilot_recommended"] = True
            think_result["task"] = (
                f"📊 PILOT RECOMMENDED: Success probability={prior:.0%}. "
                f"Before full training, consider running a 2-epoch pilot to validate.\n"
                f"To run pilot: add '--epochs 2' or similar to the training command.\n\n"
                f"--- TASK ---\n{think_result.get('task', '')}"
            )

    def _record_causal_chain_from_think(self, think_result: dict):
        """Extract and record causal links from the THINK result.

        Parses the hypothesis to extract:
        design_decision → architectural_property → metric_affected
        """
        hypothesis = think_result.get("hypothesis", "")
        task = think_result.get("task", "")
        combined = hypothesis + " " + task

        import re

        # Extract design decisions from hypothesis
        # Common patterns: "add X", "remove X", "replace X with Y", "increase X"
        decision_patterns = [
            (r'(?:add|introduce|incorporate)\s+(\w+)', "add"),
            (r'(?:remove|delete|drop)\s+(\w+)', "remove"),
            (r'(?:replace|swap)\s+(\w+)\s+(?:with|by)\s+(\w+)', "replace"),
            (r'(?:increase|raise)\s+(\w+)', "increase"),
            (r'(?:decrease|reduce|lower)\s+(\w+)', "decrease"),
            (r'(?:use|adopt|apply)\s+(\w+)', "use"),
        ]

        design_decision = ""
        for pattern, action_type in decision_patterns:
            match = re.search(pattern, combined, re.IGNORECASE)
            if match:
                design_decision = f"{action_type}: {match.group(0)}"
                break

        if not design_decision:
            return

        # Extract metric affected
        metric_patterns = [
            r'(?:MAE|mae)(?:_\w+)?',
            r'(?:loss|Loss)',
            r'(?:accuracy|acc)',
            r'(?:Non.Lambertian|Lambertian|Mixed)',
        ]
        metrics_affected = []
        for mp in metric_patterns:
            found = re.findall(mp, combined)
            metrics_affected.extend(found[:1])

        if not metrics_affected:
            metrics_affected = ["overall_MAE"]

        # Extract architectural property
        arch_keywords = {
            "attention": "attention_mechanism",
            "conv": "convolution_layer",
            "resnet": "backbone",
            "epi": "epi_processing",
            "fft": "frequency_processing",
            "loss": "loss_function",
            "lr": "learning_rate",
            "dropout": "regularization",
            "batch": "batch_processing",
            "head": "prediction_head",
        }
        arch_property = "unknown"
        combined_lower = combined.lower()
        for kw, prop in arch_keywords.items():
            if kw in combined_lower:
                arch_property = prop
                break

        # Record
        for metric in metrics_affected:
            try:
                self.memory.record_causal_link(
                    cycle=self.cycle_count,
                    design_decision=design_decision,
                    architectural_property=arch_property,
                    metric_affected=metric,
                    expected_effect=hypothesis[:200],
                )
            except Exception as e:
                logger.debug(f"Causal chain recording skipped: {e}")

    def _inject_training_curve_analysis(self, context: dict, execute_result: dict):
        """Inject training curve analysis from VERIFY into REFLECT context.

        This provides the Leader with curve-level diagnostics beyond final metrics:
        overfitting point, oscillation, convergence speed, plateau.
        """
        log_file = execute_result.get("log_file", "")
        if not log_file:
            return

        log_path = self.project_dir / log_file
        if not log_path.exists():
            return

        try:
            log_text = log_path.read_text(errors="ignore")
        except Exception:
            return

        import re
        loss_values = re.findall(r"loss[=:\s]+([0-9.]+)", log_text, re.IGNORECASE)
        if len(loss_values) < 10:
            return

        floats = [fv for v in loss_values if (fv := float(v)) > 0]
        if len(floats) < 10:
            return

        # Build curve summary
        n = len(floats)
        first_10pct = sum(floats[:max(n//10, 1)]) / max(n//10, 1)
        last_10pct = sum(floats[-max(n//10, 1):]) / max(n//10, 1)
        total_decrease = floats[0] - floats[-1]
        decrease_pct = total_decrease / max(floats[0], _EPS)

        curve_summary = (
            f"TRAINING CURVE ANALYSIS ({n} loss values):\n"
            f"- Initial loss: {floats[0]:.4f}\n"
            f"- Final loss: {floats[-1]:.4f}\n"
            f"- Total decrease: {total_decrease:.4f} ({decrease_pct:.1%})\n"
            f"- First 10% avg: {first_10pct:.4f}\n"
            f"- Last 10% avg: {last_10pct:.4f}\n"
        )

        # Detect late-stage plateau
        if n >= 20:
            last_quarter = floats[-n//4:]
            q_max = max(last_quarter)
            q_min = min(last_quarter)
            q_range = (q_max - q_min) / max(q_min, _EPS)
            if q_range < 0.01:
                curve_summary += (
                    f"- Late-stage plateau: loss barely changed in last {n//4} steps "
                    f"(range: {q_range:.4f}). Model has likely converged.\n"
                )

        # Detect if loss is still decreasing at end
        if n >= 10:
            last_10 = floats[-10:]
            recent_decrease = last_10[0] - last_10[-1]
            if recent_decrease > last_10[0] * 0.01:
                curve_summary += (
                    f"- Still improving: loss decreased {recent_decrease:.4f} in last 10 steps. "
                    f"More training epochs may help.\n"
                )

        context["training_curve_analysis"] = curve_summary

        # ── v12.2: Per-domain MAE trend + Aux loss analysis ──
        # Parse training_log.json for structured per-domain metrics
        self._inject_structured_metrics(context, execute_result)

    def _inject_structured_metrics(self, context: dict, execute_result: dict):
        """v12.2: Parse training_log.json for per-domain MAE trends and aux loss."""
        # Find training_log.json
        log_json = None
        for pattern in ["outputs/*/training_log.json", "outputs/training_log.json"]:
            candidates = list(self.project_dir.glob(pattern))
            if candidates:
                latest = max(candidates, key=lambda p: p.stat().st_mtime)
                try:
                    log_json = json.loads(latest.read_text())
                    break
                except Exception:
                    continue

        if not log_json:
            return

        epochs = log_json.get("epochs", [])
        if len(epochs) < 2:
            return

        # ── Per-domain MAE trend ──
        domain_trends = {}
        for ep in epochs:
            for key, val in ep.items():
                if key.startswith("MAE_") and not key.endswith("_count") and isinstance(val, (int, float)):
                    domain = key.replace("MAE_", "")
                    domain_trends.setdefault(domain, []).append(float(val))

        if domain_trends:
            trend_lines = ["PER-DOMAIN MAE TREND:"]
            for domain, values in sorted(domain_trends.items()):
                if len(values) >= 2:
                    direction = "↓" if values[-1] < values[0] else "↑"
                    change = values[-1] - values[0]
                    trend_lines.append(
                        f"  {domain}: {' → '.join(f'{v:.4f}' for v in values)} "
                        f"({direction} {abs(change):.4f})"
                    )
                    # Flag domains that are getting worse
                    if change > 0.05:
                        trend_lines.append(
                            f"    ⚠️ {domain} is GETTING WORSE (+{change:.4f})"
                        )
            context["per_domain_mae_trend"] = "\n".join(trend_lines)

        # ── Aux loss trend ──
        aux_values = []
        for ep in epochs:
            for key in ["train_aux_loss", "aux_loss"]:
                if key in ep and isinstance(ep[key], (int, float)):
                    aux_values.append(float(ep[key]))
                    break

        if len(aux_values) >= 2:
            aux_change = abs(aux_values[-1] - aux_values[0]) / max(abs(aux_values[0]), _EPS)
            aux_line = (
                f"AUX LOSS TREND: {' → '.join(f'{v:.6f}' for v in aux_values)} "
                f"(change: {aux_change:.2%})"
            )
            if aux_change < 0.01:
                aux_line += " ⚠️ FLAT — auxiliary module NOT learning"
            context["aux_loss_trend"] = aux_line


def main():
    parser = argparse.ArgumentParser(description="AutoResearcher - Autonomous ML Experiment Agent")
    parser.add_argument("--project", type=str, required=True, help="Path to project directory")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file path")
    parser.add_argument("--max-cycles", type=int, default=None, help="Override max cycles")
    parser.add_argument("--gpu", type=str, default=None, help="GPU device(s) to use")
    parser.add_argument("--check", action="store_true", help="Verify installation and exit")

    args = parser.parse_args()

    if args.check:
        print("AutoResearcher installation check:")
        print(f"  Python: {sys.version}")
        print(f"  Project: {args.project}")
        print("  Status: OK")
        return

    # Load config
    import yaml
    config_path = Path(args.project).expanduser().resolve() / args.config
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
        logger.info(f"Config loaded from: {config_path}")
    else:
        logger.warning(f"Config file not found: {config_path}, using defaults")
        config = {}

    if args.max_cycles is not None:
        config.setdefault("agent", {})["max_cycles"] = args.max_cycles

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(Path(args.project) / "autoresearcher.log"),
        ],
    )

    # Run
    loop = ResearchLoop(config=config, project_dir=args.project)
    loop.run()


if __name__ == "__main__":
    main()
