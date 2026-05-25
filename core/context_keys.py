"""
Central registry of all context injection keys used in THINK and REFLECT phases.

Every key injected into the Leader's context dict is defined here with:
- name: The exact dict key (must match between injection and prompt templates)
- phase: "think" or "reflect"
- description: What this key contains, for maintainer reference
- required: Whether the system should warn if injection fails

Usage in loop.py:
    from .context_keys import THINK_KEYS, REFLECT_KEYS, ContextKey

In agent prompts (leader.md, code_agent.md):
    Keys are referenced as {key_name} in the prompt template.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ContextKey:
    """Definition of a single context injection key."""
    name: str
    phase: str  # "think" or "reflect"
    description: str
    required: bool = False  # If True, warn on injection failure


# ── THINK phase context keys ──

THINK_KEYS = [
    # Core context (always present)
    ContextKey("brief", "think", "PROJECT_BRIEF.md content (truncated)"),
    ContextKey("memory_log", "think", "MEMORY_LOG.md recent entries"),
    ContextKey("cycle", "think", "Current cycle number"),
    ContextKey("directive", "think", "Human directive from DIRECTIVE.md (if any)"),
    ContextKey("workspace_dir", "think", "Path to workspace directory"),

    # Dataset understanding
    ContextKey("dataset_manifest_summary", "think",
               "DATASET_MANIFEST.json summary: training recommendations, quality issues, per-dataset counts"),

    # Session statistics
    ContextKey("session_stats", "think",
               "SQLite summary: total_cycles, best_metric, experiment_count"),
    ContextKey("recent_failures", "think",
               "Recent experiment failures from SQLite (last 3)"),

    # Domain knowledge
    ContextKey("domain_knowledge", "think",
               "Method-property mappings, domain compatibility, method assumptions from domain_knowledge.py"),
    ContextKey("data_constraints", "think",
               "Top-level data constraints extracted from PROJECT_BRIEF"),
    ContextKey("data_scarcity_warning", "think",
               "Warning when any domain has < 10 training samples (blocks architecture proposals)"),

    # Direction control
    ContextKey("idea_guardian_check", "think",
               "Injected every 5 cycles: mandatory direction alignment check"),
    ContextKey("direction_circuit_breaker", "think",
               "Injected when direction stagnation count exceeds threshold"),

    # Cross-experiment insights
    ContextKey("cross_experiment_insights", "think",
               "Meta-patterns across experiments: dominant methods, hypothesis accuracy, calibration"),

    # Architecture plan
    ContextKey("architecture_plan", "think",
               "Full IdeaPlanner 9-phase plan dict (when available)"),
    ContextKey("architecture_plan_summary", "think",
               "One-line summary of architecture plan for quick scanning"),

    # Experiment intelligence
    ContextKey("pareto_frontier", "think",
               "Pareto-optimal methods per domain from SQLite pareto_matrix"),
    ContextKey("causal_history", "think",
               "Past design decisions with verified actual effects"),
    ContextKey("hypothesis_calibration", "think",
               "Historical hypothesis accuracy and confidence calibration"),

    # Constraint engine (v10)
    ContextKey("adaptive_thresholds", "think",
               "Calibrated diagnostic thresholds from project history"),
    ContextKey("implementation_progress", "think",
               "ImplementationTracker: pending modules and completion rate"),

    # Simulation sandbox (v11)
    ContextKey("sandbox_design_guidance", "think",
               "Sandbox scaling guidance from previous cycle's model evaluation"),

    # Research roadmap (v15)
    ContextKey("research_roadmap", "think",
               "ResearchRoadmap: module decomposition, phase constraints, active module requirements"),
]

# ── REFLECT phase context keys ──

REFLECT_KEYS = [
    # Core context (always present)
    ContextKey("brief", "reflect", "PROJECT_BRIEF.md content (truncated)"),
    ContextKey("memory_log", "reflect", "MEMORY_LOG.md recent entries"),
    ContextKey("experiment_result", "reflect", "Full execute_result dict from EXECUTE phase"),
    ContextKey("cycle", "reflect", "Current cycle number"),
    ContextKey("workspace_dir", "reflect", "Path to workspace directory"),

    # VERIFY report
    ContextKey("verify_report", "reflect",
               "Full VerifyReport dict (all 10 layers)"),
    ContextKey("verify_diagnosis", "reflect",
               "List of diagnosis strings from VerifyReport"),
    ContextKey("verify_failed_modules", "reflect",
               "List of module names that failed VERIFY"),

    # Anti-deception
    ContextKey("llm_fabrication_detected", "reflect",
               "True if VERIFY detected LLM fabrication (claimed actions not performed)"),
    ContextKey("fabrication_details", "reflect",
               "List of fabrication evidence strings"),

    # Dataset quality
    ContextKey("dataset_quality_issues", "reflect",
               "List of dataset quality issue strings from VERIFY"),
    ContextKey("dataset_val_counts", "reflect",
               "Per-domain validation scene counts"),
    ContextKey("dataset_train_counts", "reflect",
               "Per-domain training scene counts"),
    ContextKey("dataset_quality_prompt", "reflect",
               "Mandatory dataset reliability guidance prompt"),

    # Visual analysis
    ContextKey("visual_analysis", "reflect",
               "VisualAnalysisResult dict (when triggered)"),
    ContextKey("visual_analysis_diagnosis", "reflect",
               "List of visual diagnosis strings"),
    ContextKey("visual_analysis_actions", "reflect",
               "List of recommended actions from visual analysis"),

    # Domain analysis
    ContextKey("domain_analysis_prompt", "reflect",
               "Cross-domain metric comparison with mandatory analysis questions"),
    ContextKey("architecture_feedback_prompt", "reflect",
               "Result-to-architecture feedback when domain gap > 0.10"),

    # Hypothesis validation
    ContextKey("hypothesis_validation_prompt", "reflect",
               "Forced hypothesis validation when quality_alert_streak >= 2"),

    # Training curve
    ContextKey("training_curve_analysis", "reflect",
               "Training curve diagnostics: overfitting, oscillation, convergence speed, plateau"),

    # Experiment evaluation (v9)
    ContextKey("experiment_evaluation", "reflect",
               "ExperimentEvaluator output: plan_vs_result, failure_diagnoses, iteration_guidance"),
    ContextKey("iteration_guidance_prompt", "reflect",
               "Priority-sorted mandatory next steps from ExperimentEvaluator"),

    # Independent assessment (v9)
    ContextKey("independent_assessment_warning", "reflect",
               "Third-party probe anomaly warning from IndependentProbe"),

    # Constraint engine (v10)
    ContextKey("plan_compliance_warning", "reflect",
               "PlannerChecker: warnings when Code Agent didn't implement planned modules"),
    ContextKey("quick_benchmark_warning", "reflect",
               "QuickBenchmark: anomaly warning when reported metrics don't match actual computation"),
    ContextKey("implementation_progress", "reflect",
               "ImplementationTracker: pending modules and completion rate"),

    # Simulation sandbox (v11)
    ContextKey("sandbox_evaluation", "reflect",
               "Full sandbox evaluation: feasibility, design A/B, reference + internal behavior, verdict"),
]

# ── Lookup helpers ──

THINK_KEY_NAMES = {k.name for k in THINK_KEYS}
REFLECT_KEY_NAMES = {k.name for k in REFLECT_KEYS}
ALL_KEY_NAMES = THINK_KEY_NAMES | REFLECT_KEY_NAMES


def validate_context(context: dict, phase: str) -> list[str]:
    """Validate a context dict against the registry. Returns list of warnings."""
    warnings = []
    expected = THINK_KEY_NAMES if phase == "think" else REFLECT_KEY_NAMES
    required = {k.name for k in (THINK_KEYS if phase == "think" else REFLECT_KEYS) if k.required}

    # Check for unknown keys
    for key in context:
        if key not in expected:
            warnings.append(f"Unknown context key '{key}' in {phase} phase")

    # Check for missing required keys
    for key in required:
        if key not in context:
            warnings.append(f"Missing required context key '{key}' in {phase} phase")

    return warnings
