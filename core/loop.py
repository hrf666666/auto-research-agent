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
import math
import time
import json
import signal
import logging
from pathlib import Path
from typing import Optional

from .memory import MemoryManager
from .monitor import ExperimentMonitor
from .agents import AgentDispatcher
from .obsidian import ObsidianExporter
from .tools import ToolRegistry
from .verifier import ExperimentVerifier, VerifyCheck
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
        self.no_progress_fallback_threshold = config.get("agent", {}).get("no_progress_fallback_threshold", 3)
        self._running = True
        self._no_progress_streak = 0
        self._last_no_progress_signature = ""
        self._consecutive_wait_count = 0
        self._max_consecutive_waits = config.get("agent", {}).get("max_consecutive_waits", 3)
        # Repeated issue tracking for error escalation
        # Uses sliding window: issue_signature → list of recent cycle numbers

        # ── Metric-based progress tracking (for visual analysis trigger) ──
        # Visual analysis should fire when METRICS stop improving,
        # not just when experiments fail to launch.
        self._best_metric_ever: float = float('inf')  # Best val_MAE ever seen
        self._metric_no_progress_streak: int = 0       # Cycles since last metric improvement
        self._visual_trigger_threshold: int = (config or {}).get(
            "visual_analysis", {}
        ).get("trigger_threshold", 5)  # Sync with VisualAnalyzer default

        # ── Fix 1: Output quality awareness ──
        # Track per-domain metrics to detect when agent produces "successful bad results"
        self._best_domain_metrics: dict[str, float] = {}  # domain → best MAE

        # ── Fix 3: Strategic abandonment ──
        # Track whether the agent is stuck in a research direction

        # ── v14: Architecture-level stagnation (independent of direction stagnation) ──
        # Tracks whether the agent is stuck patching the SAME architecture.
        # Unlike direction stagnation, this is NOT reset by paper_research —
        # only reset when a genuinely different architecture is detected.
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


        # Phase 2: deterministic garbage collector
        from .garbage_collector import GarbageCollector
        self._gc = GarbageCollector(self.project_dir)

        # ── Constraint Engine (v10 → v16.1): LLM behavior control ──
        # v16.1: Removed PlannerChecker, QuickBenchmark, AdaptiveThresholds, ImplementationTracker
        self.strategy_engine = StrategyConstraintEngine(self.project_dir, self.workspace)
        self.context_pruner = ContextPruner()

        # ── Simulation Sandbox (v11): Model evaluation engine ──
        self.sandbox = SimulationSandbox(self.project_dir, self.workspace, config=config)

        # ── Research Roadmap (v15): Structured research methodology ──  # Set True after first generate_from_brief()    # Consecutive phase violations in THINK

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

    def run(self, directive: str = ""):
        """Main entry point. Runs the THINK → EXECUTE → VERIFY → REFLECT loop."""
        logger.info(f"AutoResearcher starting | project={self.project_dir} | cycle={self.cycle_count}")

        while self._running:
            # Phase 1: Stop when all goals are achieved
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
                    if self.cycle_count == 1:
                        logger.info("DATASET UNDERSTANDING phase — scanning data/ directory")

                    # ── ROADMAP INIT (v15): Generate research roadmap on first cycle ──

                    # THINK: Analyze and plan
                    think_result = self._think(directive)


                # ── Phase 4: PAUSE-HUMAN — stop the loop and surface for inspection ──
                # failed launches. Stops burning quota on a stuck pattern and
                # requires human intervention to resume.

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
                        # v18: System is referee — record but don't override.
                        # LLM sees idle warning in memory and self-corrects.
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
                    self._gc.run()  # Phase 2: deterministic GC
                    # Post-reflect code review: learn from mistakes
                    # Paper research is meaningful work — persist cycle counter
                    self._save_cycle_counter()
                    continue

                # ARCHITECTURE SWITCH (v14): Execute architecture switch instead of experiment
                # This is triggered when the architecture stagnation threshold is reached.
                # The agent researches alternative architectures AND starts implementing.
                # Gate 1: PRE-VERIFY — referee only. Detect issues, tell LLM, let LLM decide.
                pre_verify_report = self._pre_verify(self.cycle_count, think_result)
                critical_pre_issues = pre_verify_report.critical_failures
                if critical_pre_issues:
                    issues_text = "; ".join(c.detail for c in critical_pre_issues)
                    logger.warning(f"PRE-VERIFY found issues: {issues_text}")
                    # Record as active problem — LLM will see it in next THINK's memory_log
                    self.memory.log_active_problem(f"PRE-VERIFY: {issues_text}")
                    # Inject into current context so LLM sees it immediately
                    think_result["pre_verify_warning"] = issues_text

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
                except Exception as e:
                    logger.debug(f"Phase status update skipped: {e}")
                self._record_cycle_outcome(think_result, execute_result, reflect_result,
                                            verify_report_dict=verify_report.to_dict())

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

    def _think(self, directive: str = "") -> dict:
        """THINK phase: Leader decides what to do next based on context."""
        context = {}
        context["brief"] = self.memory.get_brief()
        context["memory_log"] = self.memory.get_log()
        context["cycle"] = self.cycle_count
        context["workspace_dir"] = str(self.workspace)



        # Causal history (what design decisions led to what effects)
        try:
            causal = self.memory.get_causal_history(limit=10)
            if causal:
                lines = []
                for c in causal:
                    verified = "✓" if c.get("verified") else "?"
                    decision = c.get("design_decision", "?")
                    expected = c.get("expected_effect", "?")
                    actual = c.get("actual_effect")
                    actual_str = f" → {actual}" if actual else ""
                    lines.append(f"- [{verified}] {decision}: expected {expected}{actual_str}")
                context["causal_history"] = "\n".join(lines)
        except Exception:
            pass

        # Domain knowledge
        try:
            domain_kb = self._build_domain_knowledge()
            if domain_kb:
                context["domain_knowledge"] = domain_kb
                if isinstance(domain_kb, dict) and domain_kb.get("data_constraints"):
                    context["data_constraints"] = domain_kb["data_constraints"]
        except Exception:
            pass


        # Persistent constraints
        constraints_path = self.project_dir / "PERSISTENT_CONSTRAINTS.md"
        if constraints_path.exists():
            try:
                text = constraints_path.read_text().strip()
                if text:
                    context["persistent_constraints"] = text
            except Exception:
                pass

        # Context pruning
        context = self.context_pruner.prune(context, "think")

        # Inject any pre-verify warnings from previous cycle
        # (referee tells player what's wrong, player decides what to do)
        if directive and "PRE-VERIFY" in str(directive):
            context["pre_verify_warning"] = directive

        result = self.dispatcher.dispatch_leader(task="think", context=context)


        logger.info(f"THINK result: action={result.get('action', 'unknown')}")
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
        """EXECUTE phase: dispatch researcher agent for paper/web research."""
        task_description = plan.get("task", "Research existing approaches.")
        result = self.dispatcher.dispatch_worker(
            agent_type="researcher",
            task=task_description,
            tools=self.tools.get_tools_for("researcher"),
        )
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

    def _reflect(self, execute_result: dict, verify_report) -> dict:
        """REFLECT phase: Leader evaluates results and records learnings."""
        context = {}
        context["brief"] = self.memory.get_brief()
        context["memory_log"] = self.memory.get_log()
        context["cycle"] = self.cycle_count
        context["workspace_dir"] = str(self.workspace)
        context["experiment_result"] = execute_result

        # VERIFY report
        if verify_report:
            context["verify_report"] = verify_report.to_dict()
            if verify_report.diagnosis:
                context["verify_diagnosis"] = verify_report.diagnosis
            if verify_report.failed_modules:
                context["verify_failed_modules"] = verify_report.failed_modules

        # Anti-deception
        if execute_result.get("deception_detected"):
            context["llm_fabrication_detected"] = True
            context["fabrication_details"] = execute_result.get("deception_detail", [])

        # Context pruning
        context = self.context_pruner.prune(context, "reflect")

        try:
            result = self.dispatcher.dispatch_leader(task="reflect", context=context)
        except Exception as e:
            logger.warning(f"REFLECT LLM call failed: {e}")
            result = {"milestone": "", "decision": "Reflect failed", "dead_end": None,
                      "active_problem": None}

        # Record to memory
        if result.get("milestone"):
            self.memory.log_milestone(result["milestone"], cycle=self.cycle_count)
        if result.get("decision"):
            self.memory.log_decision(result["decision"])
        if result.get("dead_end"):
            self.memory.log_dead_end(result["dead_end"])
        if result.get("active_problem"):
            self.memory.log_active_problem(result["active_problem"])

        logger.info(f"REFLECT result: milestone={'yes' if result.get('milestone') else 'no'}")
        return result


    def _update_launch_counter(self, execute_result: dict):
        """Update the consecutive-failed-launch counter after EXECUTE."""
        if execute_result.get("experiment_launched"):
            self._consecutive_failed_launches = 0
        elif execute_result.get("convergence_failed") or \
                (not execute_result.get("launch_error")
                 and not execute_result.get("deception_detected")
                 and not execute_result.get("experiment_launched")):
            self._consecutive_failed_launches += 1


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

        signature = think_result.get("task", "")[:100]
        made_progress = bool(
            execute_result.get("experiment_launched")
            or execute_result.get("final_metrics")
            or reflect_result.get("milestone")
        )

        # ── Metric-based progress tracking (Fix 1: visual analysis trigger) ──
        final_metrics = execute_result.get("final_metrics") or {}
        current_metric = None
        # v18: dynamically extract the best available metric from results
        # (not hardcoded to val_MAE — supports accuracy, MAE, PSNR, etc.)
        if final_metrics:
            # Try config-defined keys first, then any *_MAE, then any numeric value
            config_keys = [g.get("key") for g in self.config.get("goals", {}).get("metrics", [])]
            search_keys = config_keys + ["val_MAE", "val_mae", "accuracy", "acc", "PSNR", "psnr"]
            for key in search_keys:
                if key in final_metrics:
                    try:
                        current_metric = float(final_metrics[key])
                        break
                    except (TypeError, ValueError):
                        continue
            # Fallback: take first numeric value
            if current_metric is None:
                for key, val in final_metrics.items():
                    try:
                        current_metric = float(val)
                        break
                    except (TypeError, ValueError):
                        continue
        metric_keys = ()  # no longer used
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
            method = ""
            status = "success" if made_progress else "inconclusive"
            metric_key_used = "metric"  # simplified
            try:
                self.memory.log_structured_result(
                    cycle=self.cycle_count,
                    metric_key=metric_key_used,
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
            method_name = ""  # method extraction removed
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
        # Compare actual improvement with expected improvement
                    # v18: improvement tracking simplified

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

