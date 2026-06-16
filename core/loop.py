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

                    # ── ROADMAP INIT (v15): Generate research roadmap on first cycle ──
                    if not self._roadmap_initialized:
                        pass

                    # THINK: Analyze and plan
                    think_result = self._think(directive)

                    if think_result.get("action") == "experiment":
                        think_result = self._enforce_launch_after_failure(think_result)
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
                        pass
                except Exception as e:
                    logger.debug(f"Phase status update skipped: {e}")
                self._record_cycle_outcome(think_result, execute_result, reflect_result,
                                            verify_report_dict=verify_report.to_dict())

                self._refresh_obsidian(reflect_result=reflect_result, directive=directive)

                # Post-reflect code review: learn from mistakes

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
        # ── CAUSAL CHAIN RECORDING ──
        # Record the design decision → architectural property → expected metric link
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
        result["is_paper_research"] = True
        return result


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

