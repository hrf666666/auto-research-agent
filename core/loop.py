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
    PlannerChecker,
    StrategyConstraintEngine,
    QuickBenchmark,
    AdaptiveThresholds,
    ImplementationTracker,
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

        # Adaptive thresholds (initialized early — used by verifier and evaluator)
        self.adaptive_thresholds = AdaptiveThresholds(self.memory)

        # VERIFY phase: module-level result verification
        self.verifier = ExperimentVerifier(
            project_dir=self.project_dir,
            workspace=self.workspace,
            thresholds=self.adaptive_thresholds.get_thresholds(),
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
        self._quality_alert_streak: int = 0  # Consecutive cycles with quality degradation

        # ── Fix 3: Strategic abandonment ──
        # Track whether the agent is stuck in a research direction
        self._current_direction_signature: str = ""  # Hash of current research direction
        self._direction_stagnation_count: int = 0     # Cycles without improvement in current direction
        self._direction_change_threshold: int = 3     # Force paper research after N stagnations

        # ── Fix 2: Infrastructure degradation ──
        self._infra_failure_streak: int = 0  # Consecutive infrastructure failures
        self._infra_degradation_threshold: int = 3  # Skip VERIFY after N infra failures

        # ── Constraint Engine (v10): LLM behavior control ──
        self.planner_checker = PlannerChecker(self.project_dir, self.workspace)
        self.strategy_engine = StrategyConstraintEngine(self.project_dir, self.workspace)
        self.quick_benchmark = QuickBenchmark(self.project_dir, self.workspace, config=config)
        self.impl_tracker = ImplementationTracker(self.workspace)
        self.context_pruner = ContextPruner()

        # ── Simulation Sandbox (v11): Model evaluation engine ──
        self.sandbox = SimulationSandbox(self.project_dir, self.workspace, config=config)

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
            if self.max_cycles > 0 and self.cycle_count >= self.max_cycles:
                logger.info(f"Reached max cycles ({self.max_cycles}). Stopping.")
                break

            self.cycle_count += 1
            logger.info(f"=== Cycle {self.cycle_count} ===")
            # NOTE: _save_cycle_counter is deferred — only saved when the cycle
            # produces meaningful work (experiment launched or paper research done).

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

                    # THINK: Analyze and plan
                    think_result = self._think(directive)
                    think_result = self._apply_no_progress_fallback(think_result, directive)

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
                    self._auto_code_cleanup(execute_result, reflect_result)
                    # Paper research is meaningful work — persist cycle counter
                    self._save_cycle_counter()
                    continue

                # PRE-VERIFY: Check critical preconditions BEFORE executing
                # This catches problems like synthetic data, missing data, broken imports
                # before wasting GPU hours on doomed experiments.
                pre_verify_report = self._pre_verify(self.cycle_count, think_result)

                # ── FALSIFIABLE HYPOTHESIS CHECK ──
                # Force the experiment to have a falsifiable hypothesis.
                # If the hypothesis cannot be proven wrong, the experiment is not scientific.
                hypothesis = think_result.get("hypothesis", "")
                success_criteria = think_result.get("success_criteria", "")
                task_text = think_result.get("task", "")

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
                    # Don't block, but inject as a strong warning into the task
                    think_result["task"] = (
                        f"⚠️ FALSIFIABILITY WARNING: {non_falsifiable_warning}\n\n"
                        f"--- ORIGINAL TASK ---\n{think_result.get('task', '')}\n\n"
                        f"--- MANDATORY ADDITION ---\n"
                        f"Before starting: Write down your HYPOTHESIS, SUCCESS CRITERIA, and "
                        f"FAILURE CONDITION as comments at the top of your training script.\n"
                        f"Format: # HYPOTHESIS: If X then Y because Z\n"
                        f"        # SUCCESS: metric < threshold\n"
                        f"        # FAILURE: metric >= threshold (hypothesis wrong)\n"
                    )

                critical_pre_issues = pre_verify_report.critical_failures
                if critical_pre_issues:
                    issues_text = "; ".join(c.detail for c in critical_pre_issues)
                    logger.warning(
                        f"PRE-VERIFY blocked execution: {issues_text}"
                    )
                    # Convert pre-verify failures into a fix task instead of executing
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

                # EXECUTE: Run the plan
                self._consecutive_wait_count = 0
                execute_result = self._execute(think_result)

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
                severe_threshold = self.adaptive_thresholds.get_thresholds().get("severe_degradation", 0.35)
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
                self._record_cycle_outcome(think_result, execute_result, reflect_result,
                                            verify_report_dict=verify_report.to_dict())

                self._refresh_obsidian(reflect_result=reflect_result, directive=directive)

                # Only count as meaningful cycle if experiment was launched or progress was made
                if execute_result.get("experiment_launched") or reflect_result.get("milestone"):
                    self._save_cycle_counter()

                # AUTO CODE-CLEANUP: Check trigger conditions after each cycle
                self._auto_code_cleanup(execute_result, reflect_result)

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
                logger.error(f"Cycle {self.cycle_count} failed: {e}", exc_info=True)
                self.memory.log_decision(f"Cycle {self.cycle_count} error: {str(e)[:200]}")
                self._update_state(
                    {
                        "cycle": self.cycle_count,
                        "status": "error",
                        "updated_at": time.time(),
                        "last_error": str(e)[:500],
                    }
                )
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
        if self._direction_stagnation_count >= self._direction_change_threshold:
            context["direction_circuit_breaker"] = (
                f"DIRECTION CIRCUIT BREAKER TRIGGERED: {self._direction_stagnation_count} "
                f"cycles without progress on current direction.\n"
                f"STOP and re-read PROJECT_BRIEF. You MUST propose a FUNDAMENTALLY different approach.\n"
                f"Record the current direction as a dead end before proceeding."
            )

        # ── CROSS-EXPERIMENT KNOWLEDGE INTEGRATION ──
        # Connect dead ends across experiments to identify meta-patterns
        cross_exp = self._build_cross_experiment_insights()
        if cross_exp:
            context["cross_experiment_insights"] = cross_exp

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
                    # v10: Update implementation tracker with planned modules
                    self.impl_tracker.update_from_plan(arch_plan, self.cycle_count)
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
        # Show past design decisions and their actual effects
        try:
            causal_history = self.memory.get_causal_history(limit=10)
            if causal_history:
                verified = [c for c in causal_history if c.get("verified")]
                if verified:
                    context["causal_history"] = verified
        except Exception as e:
            logger.warning(f"Failed to inject causal history: {e}")

        # ── EXPERIMENT CALIBRATION ──
        # Help the agent learn from past hypothesis accuracy
        try:
            calibration = self.memory.get_experiment_calibration()
            if calibration.get("total_hypotheses", 0) >= 3:
                context["hypothesis_calibration"] = calibration
        except Exception as e:
            logger.warning(f"Failed to inject hypothesis calibration: {e}")

        # ── IMPLEMENTATION PROGRESS TRACKER (v10) ──
        # Show Leader which planned modules are still pending
        impl_prompt = self.impl_tracker.get_progress_prompt()
        if impl_prompt:
            context["implementation_progress"] = impl_prompt

        # ── ADAPTIVE THRESHOLDS (v10) ──
        # Inject calibrated thresholds for experiment evaluation
        thresholds = self.adaptive_thresholds.get_thresholds()
        if thresholds.get("calibrated"):
            context["adaptive_thresholds"] = (
                f"ADAPTIVE THRESHOLDS (calibrated from project history):\n"
                f"- Domain gap critical: > {thresholds['domain_gap_critical']:.4f}\n"
                f"- Domain gap high: > {thresholds['domain_gap_high']:.4f}\n"
                f"- Metric degradation: > {thresholds['metric_degradation_pct']:.0%}\n"
                f"- Improvement threshold: > {thresholds['improvement_threshold']:.4f}\n"
            )

        # ── v11: SANDBOX SCALING GUIDANCE ──
        # Inject previous cycle's sandbox verdict to guide model design
        try:
            sandbox_cache = self.workspace / "_sandbox_last_verdict.json"
            if sandbox_cache.exists():
                last_verdict = json.loads(sandbox_cache.read_text())
                if last_verdict.get("recommended_actions"):
                    context["sandbox_design_guidance"] = (
                        "SANDBOX DESIGN GUIDANCE (from last evaluation):\n"
                        + "\n".join(f"- {a}" for a in last_verdict["recommended_actions"][:5])
                        + f"\n\nScalable modules: {last_verdict.get('scalable_modules', [])}"
                        + f"\nBottleneck modules: {last_verdict.get('bottleneck_modules', [])}"
                    )
        except Exception:
            pass

        # ── CONTEXT PRUNING (v10) ──
        # Limit context to most relevant keys to prevent LLM confusion
        context = self.context_pruner.prune(context, "think")

        result = self.dispatcher.dispatch_leader(
            task="think",
            context=context,
        )

        # ── STRATEGY CONSTRAINT CHECK (v10) ──
        # Check proposed action against learned constraints
        if result.get("action") == "experiment":
            violations = self.strategy_engine.check_constraints(result, self.memory)
            if violations:
                constraint_prompt = self.strategy_engine.get_constraint_prompt(violations)
                logger.warning(f"Strategy constraint violations: {len(violations)}")
                # Inject constraint warnings back into result for the leader
                result["_constraint_warnings"] = constraint_prompt
                self.memory.log_decision(
                    f"[CONSTRAINT] {len(violations)} strategy constraint(s) triggered for proposed experiment"
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

        result = self.dispatcher.dispatch_worker(
            agent_type=agent_type,
            task=task_description,
            tools=self.tools.get_tools_for(agent_type),
        )

        return result

    def _execute_paper_research(self, plan: dict) -> dict:
        """EXECUTE phase: run deep paper research via researcher agent.

        Paper research dispatches to the 'researcher' agent (not 'code') so it
        gets web search + paper tools instead of training tools.
        """
        logger.info("PAPER RESEARCH EXECUTE phase starting...")

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
            r"models/[\w/]+\.py",
            r"model.*?['\"]([\w/]+\.py)['\"]",
        ]
        for pattern in patterns:
            match = re.search(pattern, task)
            if match:
                return match.group(0) if "/" in match.group(0) else match.group(1)

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
        if self._quality_alert_streak >= 2:
            context["hypothesis_validation_prompt"] = (
                "HYPOTHESIS VALIDATION REQUIRED:\n"
                "Your method has been producing severely degraded results for multiple cycles. "
                "Before proposing another experiment, you MUST:\n"
                "1. State the CORE ASSUMPTION of your current method (e.g., 'EPI slope encodes depth').\n"
                "2. Identify which domain(s) violate this assumption.\n"
                "3. If the assumption is violated, incremental improvements (loss weights, "
                "data augmentation, hyperparameters) will NOT help. You need a NEW method.\n"
                "4. Propose a method that does NOT rely on the violated assumption.\n"
            )

        # ── TRAINING CURVE ANALYSIS INJECTION ──
        # Provide curve-level diagnostics to augment the Leader's reflection
        self._inject_training_curve_analysis(context, execute_result)

        # ── EXPERIMENT EVALUATOR INJECTION ──
        # Post-experiment evaluation: plan vs result, failure diagnosis, iteration guidance
        try:
            from .experiment_evaluator import ExperimentEvaluator
            evaluator = ExperimentEvaluator(
                self.project_dir, self.workspace,
                thresholds=self.adaptive_thresholds.get_thresholds(),
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

        # ── v10: PLANNER CHECKER — Plan vs Implementation compliance ──
        # Verify Code Agent faithfully executed the architecture plan
        arch_plan = getattr(self, '_last_architecture_plan', None)
        if arch_plan and execute_result.get("experiment_launched"):
            try:
                compliance = self.planner_checker.check_plan_compliance(
                    cycle=self.cycle_count,
                    architecture_plan=arch_plan,
                    execute_result=execute_result,
                )
                if compliance.warnings:
                    context["plan_compliance_warning"] = "\n".join(compliance.warnings)
                    logger.warning(
                        f"Plan compliance: score={compliance.compliance_score:.0%}, "
                        f"missing={compliance.missing_modules}, risk={compliance.fabrication_risk}"
                    )
                # Update implementation tracker
                self.impl_tracker.update_from_compliance(compliance)
            except Exception as e:
                logger.warning(f"PlannerChecker failed: {e}")

        # ── v10: QUICK BENCHMARK — validate reported metrics ──
        # Run 5 validation samples to catch metric fabrication
        final_metrics = execute_result.get("final_metrics") or {}
        if final_metrics and execute_result.get("experiment_launched"):
            try:
                bench_result = self.quick_benchmark.run(
                    model_path=execute_result.get("model_path", ""),
                    checkpoint_path="",
                    val_data_path="",
                    reported_metrics=final_metrics,
                    max_samples=5,
                )
                if bench_result.run and bench_result.anomaly:
                    context["quick_benchmark_warning"] = (
                        f"QUICK BENCHMARK ANOMALY:\n"
                        f"Independent verification on {bench_result.num_samples} samples found:\n"
                        f"{bench_result.anomaly_detail}\n\n"
                        f"DO NOT trust the reported metrics. Investigate the evaluation pipeline."
                    )
                    logger.error(f"QuickBenchmark anomaly: {bench_result.anomaly_detail}")
                    self.memory.log_active_problem(
                        f"[BENCHMARK] Metric fabrication suspected: {bench_result.anomaly_detail[:200]}"
                    )
                elif bench_result.run and bench_result.discrepancy is not None:
                    logger.info(
                        f"QuickBenchmark OK: avg={bench_result.avg_metric:.4f}, "
                        f"reported={bench_result.reported_metric:.4f}, "
                        f"disc={bench_result.discrepancy:.4f}"
                    )
            except Exception as e:
                logger.debug(f"QuickBenchmark skipped: {e}")

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

        # ── v10: IMPLEMENTATION PROGRESS ──
        impl_prompt = self.impl_tracker.get_progress_prompt()
        if impl_prompt:
            context["implementation_progress"] = impl_prompt

        # ── v10: STRATEGY CONSTRAINT ENGINE — generate rules from history ──
        try:
            self.strategy_engine.generate_rules_from_history(self.memory)
        except Exception as e:
            logger.debug(f"Strategy rule generation skipped: {e}")

        # ── v10: CONTEXT PRUNING ──
        context = self.context_pruner.prune(context, "reflect")

        result = self.dispatcher.dispatch_leader(
            task="reflect",
            context=context,
        )

        # Update memory based on reflection
        if result.get("milestone"):
            self.memory.log_milestone(result["milestone"])
        if result.get("decision"):
            self.memory.log_decision(result["decision"])
        if result.get("dead_end"):
            self.memory.log_dead_end(result["dead_end"])
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

    def _extract_direction_signature(self, task_text: str) -> str:
        """Extract the research direction from a task description.

        Unlike _plan_signature (which detects identical plans), this extracts
        the high-level research direction to detect strategic stagnation.
        E.g., 'add edge loss' and 'increase edge loss weight' are different
        plans but the SAME direction (edge-aware training).
        """
        import hashlib
        # Extract key methodological terms (heuristic: keywords related to approach)
        direction_keywords = [
            "edge", "loss", "pretrain", "backbone", "resnet", "epi",
            "angular", "conv", "stride", "view", "direction", "stream",
            "attention", "transformer", "gnn", "groupnorm", "batchnorm",
            "lambertian", "non-lambertian", "mixed", "domain",
            "augment", "crop", "flip", "rotate", "scale",
            "lr", "scheduler", "adam", "sgd", "epoch", "batch",
            "disparity", "depth", "pfm", "gt",
        ]
        # Normalize: replace hyphens with spaces so "edge-aware" → "edge aware"
        normalized = task_text.lower().replace("-", " ").replace("_", " ")
        words = normalized.split()
        direction_words = [w for w in words if w in direction_keywords]
        # Use first 5 direction keywords as signature
        sig = " ".join(direction_words[:5])
        if not sig:
            sig = task_text[:100]
        return hashlib.md5(sig.encode()).hexdigest()[:12]

    def _apply_no_progress_fallback(self, think_result: dict, directive: Optional[str]) -> dict:
        """Back off if the same experiment plan keeps repeating without progress.

        When stuck, redirect to paper research instead of just waiting —
        this forces the agent to seek new knowledge rather than idle.

        Fix 1: Also force paper research when output quality degrades repeatedly.
        Fix 3: Force paper research when direction stagnation is detected.
        """
        if directive or self.no_progress_fallback_threshold <= 0:
            return think_result

        if think_result.get("action") != "experiment":
            return think_result

        # Fix 1: Quality degradation fallback
        if self._quality_alert_streak >= 2:
            reason = (
                f"QUALITY FALLBACK: {self._quality_alert_streak} consecutive cycles with "
                f"degraded domain metrics. Agent must diagnose why results are getting worse."
            )
            logger.warning(reason)
            self.memory.log_decision(reason)
            return {
                "action": "paper_research",
                "reason": reason,
                "decision": reason,
                "agent": "researcher",
                "task": (
                    "QUALITY DIAGNOSIS — experiments are producing worse results.\n\n"
                    "1. Read MEMORY_LOG.md for recent experiment results and failures\n"
                    "2. Analyze WHY the agent's experiments are producing degraded metrics\n"
                    "3. Search for papers that address the specific failure mode\n"
                    "4. Write a diagnosis and recommended new direction to workspace/\n"
                ),
            }

        # Fix 3: Direction stagnation fallback
        if self._direction_stagnation_count >= self._direction_change_threshold:
            reason = (
                f"DIRECTION FALLBACK: Same research direction for "
                f"{self._direction_stagnation_count} cycles without improvement. "
                f"Agent must find a fundamentally different approach."
            )
            logger.warning(reason)
            self.memory.log_decision(reason)
            return {
                "action": "paper_research",
                "reason": reason,
                "decision": reason,
                "agent": "researcher",
                "task": (
                    "DIRECTION CHANGE — agent is stuck in a research rut.\n\n"
                    "The agent has been trying the same approach for multiple cycles\n"
                    "without any metric improvement. This suggests the direction itself\n"
                    "may be fundamentally flawed.\n\n"
                    "1. Read MEMORY_LOG.md to understand what has been tried\n"
                    "2. Search for papers proposing FUNDAMENTALLY DIFFERENT methods\n"
                    "3. Do NOT suggest incremental improvements to the current approach\n"
                    "4. Propose a completely new research direction with specific implementation plan\n"
                ),
            }

        signature = self._plan_signature(think_result)
        if (
            self._no_progress_streak >= self.no_progress_fallback_threshold
            and signature == self._last_no_progress_signature
        ):
            reason = (
                f"Fallback: {self._no_progress_streak} no-progress cycles on same plan. "
                "Forcing paper research to seek breakthrough methods from literature."
            )
            logger.warning(reason)
            self.memory.log_decision(reason)
            return {
                "action": "paper_research",
                "reason": reason,
                "decision": reason,
                "agent": "researcher",
                "task": (
                    "EMERGENCY PAPER RESEARCH — agent is stuck repeating the same failed experiment.\n\n"
                    "Execute the /paper-research skill immediately. The full instructions are in:\n"
                    "skills/paper-research/SKILL.md\n\n"
                    "Read that file and follow the Phase 1→2→3→4 workflow precisely.\n"
                    "This is a MAJOR EVENT. Log results and update MEMORY_LOG.md accordingly."
                ),
            }

        return think_result

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

        if think_result.get("action") == "paper_research":
            # Paper research is always considered progress — it generates new knowledge
            self._no_progress_streak = 0
            self._last_no_progress_signature = ""
            self._metric_no_progress_streak = 0
            self._direction_stagnation_count = 0  # Reset direction stagnation
            self._infra_failure_streak = 0
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
        for key in ("val_MAE", "val_MAE_overall", "best_val_MAE", "val_mae"):
            if key in final_metrics:
                try:
                    current_metric = float(final_metrics[key])
                except (TypeError, ValueError):
                    pass
                break

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
            if quality_degraded:
                self._quality_alert_streak += 1
                if self._quality_alert_streak >= 2:
                    logger.warning(
                        f"QUALITY ALERT: {self._quality_alert_streak} consecutive cycles with "
                        f"degraded domain metrics. Forcing visual analysis + paper research."
                    )
                    self.memory.log_decision(
                        f"[QUALITY] {self._quality_alert_streak} cycles of degraded quality. "
                        f"Agent must diagnose root cause before next experiment."
                    )
            else:
                self._quality_alert_streak = 0
        else:
            self._quality_alert_streak = 0

        # ── Fix 3: Strategic abandonment (direction stagnation detection) ──
        # Extract research direction from the experiment task/hypothesis
        task_text = think_result.get("task", "")[:200]
        direction_sig = self._extract_direction_signature(task_text)
        if direction_sig != self._current_direction_signature:
            # New direction — reset counter
            self._current_direction_signature = direction_sig
            self._direction_stagnation_count = 0
            logger.info(f"NEW DIRECTION: '{direction_sig[:80]}'")
        else:
            # Same direction — check if metrics improved
            if current_metric is not None and current_metric < self._best_metric_ever:
                self._direction_stagnation_count = 0  # Improvement in current direction
            else:
                self._direction_stagnation_count += 1

        if self._direction_stagnation_count >= self._direction_change_threshold:
            logger.warning(
                f"DIRECTION STAGNATION: Same direction for {self._direction_stagnation_count} "
                f"cycles without improvement. Forcing paper research for new direction."
            )
            self.memory.log_decision(
                f"[DIRECTION] Stagnant for {self._direction_stagnation_count} cycles on: "
                f"'{direction_sig[:120]}'. Agent must seek fundamentally different approaches."
            )
            # Reset to allow new direction after paper research
            self._direction_stagnation_count = 0
            self._current_direction_signature = ""

        # ── Metric tracking (existing logic) ──
        if current_metric is not None:
            if not math.isfinite(current_metric):
                logger.warning(f"METRIC INVALID: {current_metric} — skipping metric tracking")
            else:
                improvement_threshold = self.adaptive_thresholds.get_thresholds().get("improvement_threshold", 0.005)
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

    def _auto_code_cleanup(self, execute_result: dict, reflect_result: dict):
        """Automatically trigger code-cleanup when conditions are met.

        Trigger conditions:
        1. Root .py files > 15
        2. logs/ or outputs/ has > 10 unarchived log files
        3. outputs/ has > 10 experiment directories
        4. archive/ has > 20 experiments (needs pruning)
        5. scripts/ has > 10 .py files (naming pollution / stale scripts)
        6. Experiment failed or hit major bug (MUST trigger immediately)
        """
        project = self.project_dir
        should_cleanup = False
        reasons = []

        # Condition 1: Root .py files > 15
        root_py_count = len(list(project.glob("*.py")))
        if root_py_count > 15:
            should_cleanup = True
            reasons.append(f"Root .py files: {root_py_count} (> 15)")

        # Condition 2: logs/ or outputs/ has > 10 unarchived files
        logs_dir = project / "logs"
        outputs_dir = project / "outputs"
        log_count = 0
        if logs_dir.exists():
            log_count = len(list(logs_dir.rglob("*.log"))) + len(list(logs_dir.rglob("*.csv")))
        if outputs_dir.exists():
            log_count += len(list(outputs_dir.rglob("*.log"))) + len(list(outputs_dir.rglob("*.csv")))
        if log_count > 10:
            should_cleanup = True
            reasons.append(f"Unarchived log files: {log_count} (> 10)")

        # Condition 3: outputs/ has too many experiment dirs
        if outputs_dir.exists():
            exp_dirs = [d for d in outputs_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
            if len(exp_dirs) > 10:
                should_cleanup = True
                reasons.append(f"Output experiment dirs: {len(exp_dirs)} (> 10)")

        # Condition 4: archive/ is bloated
        archive_dir = project / "archive" / "experiments"
        if archive_dir.exists():
            archive_count = len([d for d in archive_dir.iterdir() if d.is_dir()])
            if archive_count > 20:
                should_cleanup = True
                reasons.append(f"Archive experiments: {archive_count} (> 20)")

        # Condition 5: scripts/ has too many files (naming pollution)
        scripts_dir = project / "scripts"
        if scripts_dir.exists():
            script_count = len([f for f in scripts_dir.iterdir() if f.suffix == ".py"])
            if script_count > 10:
                should_cleanup = True
                reasons.append(f"scripts/ files: {script_count} (> 10, likely stale scripts)")

        # Condition 6: Experiment failed
        if execute_result.get("response", "").startswith('{"error"'):
            should_cleanup = True
            reasons.append("Experiment API error — cleanup to prevent raw log accumulation")

        if not execute_result.get("experiment_launched") and execute_result.get("agent") == "code":
            if self._no_progress_streak >= 4:
                should_cleanup = True
                reasons.append(f"Code agent failed to launch experiment for {self._no_progress_streak} consecutive cycles")

        if not should_cleanup:
            logger.debug("Code-cleanup conditions not met. Skipping.")
            return

        logger.info(f"Auto code-cleanup triggered: {'; '.join(reasons)}")

        # Dispatch code agent to execute code-cleanup
        cleanup_task = (
            "AUTO CODE-CLEANUP TRIGGERED\n\n"
            f"Reasons: {'; '.join(reasons)}\n\n"
            "You have MAXIMUM 10 tool calls. Be EFFICIENT.\n\n"
            "Execute the following cleanup steps:\n\n"
            "## Step 1: Clean up outputs/ (2 tool calls max)\n"
            "For experiments in outputs/ that are NOT the latest/best:\n"
            "- Keep only best_checkpoint.pt (delete final_checkpoint.pt)\n"
            "- Delete dry-run experiments entirely (dirs starting with 'dry_' or 'dryrun_')\n"
            "- Delete superseded experiments (not current or previous cycle)\n"
            "Example:\n"
            "```bash\n"
            "find outputs/ -name 'final_checkpoint.pt' -delete\n"
            "rm -rf outputs/dry_* outputs/dryrun_*\n"
            "```\n\n"
            "## Step 2: Prune archive/ (2 tool calls max)\n"
            "- Delete ALL .pt checkpoint files (too large to keep)\n"
            "- Delete dirs without SUMMARY.md (no useful info)\n"
            "- If archive has > 20 entries, delete oldest 50%\n"
            "```bash\n"
            "find archive/ -name '*.pt' -delete\n"
            "find archive/experiments/ -mindepth 1 -maxdepth 1 -type d "
            "| while read d; do [ -f \"$d/SUMMARY.md\" ] || rm -rf \"$d\"; done\n"
            "```\n\n"
            "## Step 3: Clean up scripts/ (2 tool calls max)\n"
            "Delete scripts that violate naming convention or are obsolete:\n"
            "- One-time diagnostics (test_*.py, audit_*.py, diagnose_forward.py)\n"
            "- Superseded versions (*_v2.py, *_fix.py when newer version exists)\n"
            "- Scripts importing models that no longer exist\n"
            "- Keep: train_*.py (one per model), eval_*.py, inference_*.py, dry_run.py\n"
            "```bash\n"
            "rm -f scripts/test_*.py scripts/audit_*.py scripts/diagnose_forward.py\n"
            "```\n\n"
            "## Step 4: Clean up root .py files (1 tool call max)\n"
            "Remove obsolete .py files in project root.\n\n"
            "CRITICAL RULES:\n"
            "- NEVER delete: models/, datasets/, scripts/, data/, DATASET_MANIFEST.json, config.yaml\n"
            "- NEVER delete any .npy or .png in data/\n"
            "- Use run_shell for batch operations\n"
            "- Maximum 10 tool calls total\n"
            "- Git history is the backup\n"
            "- Checkpoints (.pt files) are the #1 disk hog — always clean them first"
        )

        cleanup_result = self.dispatcher.dispatch_worker(
            agent_type="code",
            task=cleanup_task,
            tools=self.tools.get_tools_for("code"),
            max_turns_override=12,  # Limit cleanup to avoid wasting 40+ turns
        )
        logger.info(f"Auto code-cleanup completed: {str(cleanup_result)[:200]}")

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

            elif action == "force_error_handler":
                # Level 2: Force the error-handler skill
                logger.warning(
                    f"🔴 AUDIT ESCALATION L2: '{sig}' appeared {count} times. "
                    f"Forcing error-handler skill."
                )
                self._inject_error_directive(sig, count, issue, level=2)

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
            # Inject warning but don't block
            think_result["task"] = (
                f"⚠️ LOW VALUE EXPERIMENT (VOI={voi:.4f}, success probability={prior:.0%})\n"
                f"This experiment has low estimated value based on past calibration. "
                f"Consider whether a DIFFERENT hypothesis would be more informative.\n"
                f"If proceeding, consider running a pilot experiment (2-3 epochs) first "
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
