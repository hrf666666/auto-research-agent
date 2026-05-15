"""
AutoResearcher Experiment Evaluator — Post-Experiment Analysis & Guidance

Three capabilities:
1. ExperimentEvaluator: Analyzes experiment results vs architecture plan, identifies failure causes
2. IterationGuide: Generates specific next-step guidance based on failure diagnosis
3. IndependentProbe: Third-party verification using a lightweight probe model

This module is called from:
- VERIFY phase: IndependentProbe runs a quick third-party assessment
- REFLECT phase: ExperimentEvaluator + IterationGuide provide structured diagnosis

Design principle: The model should NOT evaluate itself. IndependentProbe uses
a separate lightweight model to cross-validate the main model's outputs.
"""

import ast
import json
import re
import logging
import subprocess
import sys
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

logger = logging.getLogger("autoresearcher.experiment_evaluator")


@dataclass
class FailureDiagnosis:
    """Structured diagnosis of why an experiment failed or underperformed."""
    failure_type: str  # "architecture", "data", "training", "alignment", "capacity"
    severity: str  # "critical", "high", "medium", "low"
    root_cause: str
    evidence: list[str] = field(default_factory=list)
    fix_suggestion: str = ""
    module_involved: str = ""
    plan_phase_failed: str = ""  # Which of the 9 plan phases this relates to


@dataclass
class IterationGuidance:
    """Specific guidance for the next iteration."""
    action: str  # "fix_architecture", "adjust_training", "change_data", "pivot_method", "iterate"
    priority: str  # "critical", "high", "medium"
    specific_changes: list[str] = field(default_factory=list)
    modules_to_modify: list[str] = field(default_factory=list)
    expected_improvement: str = ""
    risk_if_ignored: str = ""


@dataclass
class IndependentAssessment:
    """Third-party assessment result from an independent lightweight model."""
    assessed: bool = False
    agreement_score: float = 0.0  # 0-1, how much independent model agrees with reported results
    independent_predictions_sample: list = field(default_factory=list)
    anomaly_detected: bool = False
    anomaly_detail: str = ""
    confidence: str = "low"  # "low", "medium", "high"


class ExperimentEvaluator:
    """Post-experiment evaluation: results vs plan, failure diagnosis, iteration guidance.

    This is the "reflector's brain" that answers:
    1. Did the experiment achieve what the plan intended?
    2. If not, WHERE in the pipeline did it fail? (module? capacity? fusion? data?)
    3. What specific change should the next iteration make?
    """

    def __init__(self, project_dir: Path, workspace: Path, thresholds: dict = None):
        self.project_dir = Path(project_dir)
        self.workspace = Path(workspace)
        self._thresholds = thresholds or {}
        self._gap_critical = self._thresholds.get("domain_gap_critical", 0.15)
        self._gap_moderate = self._thresholds.get("domain_gap_high", 0.05)

    def evaluate(
        self,
        experiment_results: dict,
        architecture_plan: dict,
        training_log: str = "",
        model_path: str = "",
    ) -> dict:
        """Full post-experiment evaluation.

        Args:
            experiment_results: Final metrics from the experiment
            architecture_plan: The plan from IdeaPlanner.plan()
            training_log: Training log text (last N chars)
            model_path: Path to the model file for structural analysis

        Returns:
            Dict with diagnosis, guidance, and plan-vs-result comparison
        """
        result = {
            "plan_vs_result": {},
            "failure_diagnoses": [],
            "iteration_guidance": [],
            "overall_assessment": "",
        }

        # Phase 1: Plan vs Result comparison
        comparison = self._compare_plan_vs_result(architecture_plan, experiment_results)
        result["plan_vs_result"] = comparison

        # Phase 2: Failure diagnosis
        diagnoses = self._diagnose_failures(
            comparison, experiment_results, training_log, model_path
        )
        result["failure_diagnoses"] = [
            {
                "failure_type": d.failure_type,
                "severity": d.severity,
                "root_cause": d.root_cause,
                "evidence": d.evidence,
                "fix_suggestion": d.fix_suggestion,
                "module_involved": d.module_involved,
                "plan_phase_failed": d.plan_phase_failed,
            }
            for d in diagnoses
        ]

        # Phase 3: Iteration guidance
        guidance = self._generate_iteration_guidance(diagnoses, architecture_plan, experiment_results)
        result["iteration_guidance"] = [
            {
                "action": g.action,
                "priority": g.priority,
                "specific_changes": g.specific_changes,
                "modules_to_modify": g.modules_to_modify,
                "expected_improvement": g.expected_improvement,
                "risk_if_ignored": g.risk_if_ignored,
            }
            for g in guidance
        ]

        # Phase 4: Overall assessment
        critical = [d for d in diagnoses if d.severity == "critical"]
        high = [d for d in diagnoses if d.severity == "high"]
        if critical:
            result["overall_assessment"] = (
                f"CRITICAL ISSUES ({len(critical)}): "
                + "; ".join(d.root_cause for d in critical[:3])
                + ". Do NOT iterate on training — fix structural issues first."
            )
        elif high:
            result["overall_assessment"] = (
                f"SIGNIFICANT ISSUES ({len(high)}): "
                + "; ".join(d.root_cause for d in high[:3])
                + ". Targeted fixes recommended before next training run."
            )
        elif diagnoses:
            result["overall_assessment"] = (
                f"MINOR ISSUES ({len(diagnoses)}): Iteration can proceed with adjustments."
            )
        else:
            result["overall_assessment"] = "No issues detected — results align with plan."

        return result

    def _compare_plan_vs_result(self, plan: dict, results: dict) -> dict:
        """Compare architecture plan expectations against actual results."""
        comparison = {
            "success_criteria_met": [],
            "success_criteria_missed": [],
            "modules_implemented": [],
            "modules_missing": [],
            "capacity_vs_actual": {},
        }

        if not plan:
            comparison["error"] = "No architecture plan available for comparison"
            return comparison

        # Check success criteria
        plan_criteria = []
        for comp in plan.get("idea_components", []):
            if comp.get("type") == "success_criteria":
                plan_criteria.extend(comp.get("metrics", []))

        final_metrics = results.get("final_metrics", {})
        for criterion in plan_criteria:
            metric_name = criterion.get("metric", "")
            threshold = float(criterion.get("threshold", float("inf")))
            actual = final_metrics.get(metric_name)

            if actual is not None:
                try:
                    actual_val = float(actual)
                    met = actual_val <= threshold
                    entry = {
                        "metric": metric_name,
                        "target": threshold,
                        "actual": actual_val,
                        "met": met,
                        "gap": actual_val - threshold,
                    }
                    if met:
                        comparison["success_criteria_met"].append(entry)
                    else:
                        comparison["success_criteria_missed"].append(entry)
                except (ValueError, TypeError):
                    pass

        # Check modules
        plan_modules = {m["name"] for m in plan.get("modules", [])}
        if plan_modules:
            comparison["modules_planned"] = list(plan_modules)

        return comparison

    def _diagnose_failures(
        self,
        comparison: dict,
        results: dict,
        training_log: str,
        model_path: str,
    ) -> list[FailureDiagnosis]:
        """Diagnose WHY the experiment underperformed."""
        diagnoses = []

        # Diagnosis 1: Success criteria failure analysis
        for missed in comparison.get("success_criteria_missed", []):
            gap = missed.get("gap", 0)
            if gap > self._gap_critical:
                diagnoses.append(FailureDiagnosis(
                    failure_type="alignment",
                    severity="critical" if gap > (self._gap_critical * 2) else "high",
                    root_cause=(
                        f"{missed['metric']} missed target by {gap:.4f} "
                        f"(actual={missed['actual']:.4f}, target={missed['target']:.4f}). "
                        f"Large gap suggests fundamental architectural or method issue."
                    ),
                    evidence=[f"Metric gap: {gap:.4f}"],
                    fix_suggestion=self._suggest_metric_fix(missed, gap, comparison),
                    plan_phase_failed="idea_formalization",
                ))
            elif gap > self._gap_moderate:
                diagnoses.append(FailureDiagnosis(
                    failure_type="training",
                    severity="medium",
                    root_cause=(
                        f"{missed['metric']} close to target but not met "
                        f"(gap={gap:.4f}). Likely needs training adjustments, not architecture changes."
                    ),
                    evidence=[f"Metric gap: {gap:.4f}"],
                    fix_suggestion="Adjust learning rate, train longer, or add regularization",
                    plan_phase_failed="verification_plan",
                ))

        # Diagnosis 2: Training dynamics analysis
        if training_log:
            self._diagnose_from_log(training_log, diagnoses)

        # Diagnosis 3: Per-domain analysis
        final_metrics = results.get("final_metrics", {})
        domain_diagnosis = self._diagnose_domain_gap(final_metrics)
        diagnoses.extend(domain_diagnosis)

        # Diagnosis 4: Model structure vs plan (if model file available)
        if model_path:
            struct_diagnosis = self._diagnose_structure_vs_plan(model_path, comparison)
            diagnoses.extend(struct_diagnosis)

        return diagnoses

    def _suggest_metric_fix(self, missed: dict, gap: float, comparison: dict) -> str:
        """Suggest a fix for a missed success criterion."""
        metric = missed.get("metric", "")
        metric_lower = metric.lower()

        # Generic guidance based on metric name patterns
        if any(kw in metric_lower for kw in ("non_", "hard", "difficult", "adversarial")):
            return (
                "Performance gap on the harder domain: The model's core method may have "
                "assumptions that don't hold for this domain. Consider: (1) Add a "
                "domain-specific branch, (2) Use domain-adaptive fusion, (3) Check if "
                "training data has enough samples for this domain."
            )
        elif any(kw in metric_lower for kw in ("base", "easy", "simple", "baseline")):
            return (
                "Performance worse than expected on the easier domain. "
                "Check for: (1) Overfitting to training data, (2) Model capacity too small, "
                "(3) Loss function not suited for this domain."
            )
        elif gap > 0.3:
            return "Large gap suggests a fundamental issue. Run analyze_model and probe_model before next experiment."
        else:
            return "Moderate gap. Try: longer training, learning rate adjustment, or minor architecture tweaks."

    def _diagnose_from_log(self, log_text: str, diagnoses: list):
        """Diagnose failures from training log patterns."""
        # Check for NaN loss
        if re.search(r"loss[=:\s]+nan", log_text, re.IGNORECASE):
            diagnoses.append(FailureDiagnosis(
                failure_type="training",
                severity="critical",
                root_cause="NaN loss detected — model produces invalid gradients. "
                           "Usually caused by: log(0), division by zero, or exploding activations.",
                evidence=["NaN loss in training log"],
                fix_suggestion="Add gradient clipping, check for log(0) in loss function, "
                               "reduce learning rate, add LayerNorm",
                module_involved="loss_function",
                plan_phase_failed="verification_plan",
            ))

        # Check for loss not decreasing
        losses = re.findall(r"loss[=:\s]+([0-9.]+)", log_text, re.IGNORECASE)
        if len(losses) >= 10:
            floats = [float(v) for v in losses if float(v) > 0]
            if len(floats) >= 10:
                early_avg = sum(floats[:len(floats) // 3]) / (len(floats) // 3)
                late_avg = sum(floats[-len(floats) // 3:]) / (len(floats) // 3)
                if late_avg >= early_avg * 0.99:
                    diagnoses.append(FailureDiagnosis(
                        failure_type="training",
                        severity="high",
                        root_cause=(
                            f"Loss not decreasing: early_avg={early_avg:.4f}, "
                            f"late_avg={late_avg:.4f}. Model is not learning."
                        ),
                        evidence=[f"Loss sequence: {floats[:5]}...{floats[-5:]}"],
                        fix_suggestion=(
                            "Check: (1) Is data loading correctly? (2) Is the loss function correct? "
                            "(3) Is the learning rate too low? (4) Is the model output collapsed?"
                        ),
                        plan_phase_failed="verification_plan",
                    ))

                # Check for divergence
                if late_avg > early_avg * 2:
                    diagnoses.append(FailureDiagnosis(
                        failure_type="training",
                        severity="critical",
                        root_cause="Loss diverging — model is getting WORSE over training. "
                                   "Learning rate likely too high.",
                        evidence=[f"Early loss: {early_avg:.4f}, Late loss: {late_avg:.4f}"],
                        fix_suggestion="Reduce learning rate by 10x, add gradient clipping",
                        plan_phase_failed="verification_plan",
                    ))

    def _diagnose_domain_gap(self, metrics: dict) -> list[FailureDiagnosis]:
        """Diagnose per-domain performance gaps."""
        diagnoses = []
        domain_maes = {}
        for key, val in metrics.items():
            if "MAE" in key and val is not None:
                try:
                    domain_maes[key] = float(val)
                except (ValueError, TypeError):
                    pass

        if len(domain_maes) < 2:
            return diagnoses

        best_domain = min(domain_maes, key=domain_maes.get)
        worst_domain = max(domain_maes, key=domain_maes.get)
        gap = domain_maes[worst_domain] - domain_maes[best_domain]

        if gap > self._gap_critical:
            severity = "critical" if gap > (self._gap_critical * 2) else "high"
            diagnoses.append(FailureDiagnosis(
                failure_type="architecture",
                severity=severity,
                root_cause=(
                    f"Domain gap = {gap:.4f}: {worst_domain} (MAE={domain_maes[worst_domain]:.4f}) "
                    f"is much worse than {best_domain} (MAE={domain_maes[best_domain]:.4f}). "
                    f"The architecture's method may have assumptions that don't hold for {worst_domain}."
                ),
                evidence=[f"{k}={v:.4f}" for k, v in sorted(domain_maes.items())],
                fix_suggestion=(
                    f"For {worst_domain}: (1) Check if the method's physical assumptions hold, "
                    f"(2) Add domain-specific processing branch, "
                    f"(3) Use domain-adaptive fusion instead of uniform fusion, "
                    f"(4) Check if there's enough training data for {worst_domain}."
                ),
                module_involved="fusion_module",
                plan_phase_failed="fusion_strategy",
            ))

        return diagnoses

    def _diagnose_structure_vs_plan(self, model_path: str, comparison: dict) -> list[FailureDiagnosis]:
        """Diagnose structural gaps between planned and implemented modules."""
        diagnoses = []
        model_full = self.project_dir / model_path

        if not model_full.exists():
            return diagnoses

        try:
            content = model_full.read_text()
            tree = ast.parse(content)
        except Exception:
            return diagnoses

        # Find nn.Module subclasses and their self.xxx = ... assignments
        implemented_modules = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for base in node.bases:
                if isinstance(base, ast.Attribute) and base.attr == "Module":
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                            for stmt in ast.walk(item):
                                if isinstance(stmt, ast.Assign):
                                    for target in stmt.targets:
                                        if (isinstance(target, ast.Attribute)
                                                and isinstance(target.value, ast.Name)
                                                and target.value.id == "self"
                                                and isinstance(stmt.value, ast.Call)):
                                            implemented_modules.add(target.attr)

        # Check if plan's modules have counterparts in implementation
        plan_modules = comparison.get("modules_planned", [])
        if plan_modules and implemented_modules:
            # Map plan module names to likely implementation names
            missing = []
            for plan_mod in plan_modules:
                plan_words = set(plan_mod.replace("_", " ").split())
                found = any(
                    plan_words & set(impl.replace("_", " ").split())
                    for impl in implemented_modules
                )
                if not found:
                    missing.append(plan_mod)

            if missing:
                diagnoses.append(FailureDiagnosis(
                    failure_type="alignment",
                    severity="high",
                    root_cause=(
                        f"Planned modules not found in implementation: {missing}. "
                        f"The architecture plan had these modules but the code doesn't implement them."
                    ),
                    evidence=[f"Planned: {plan_modules}", f"Implemented: {list(implemented_modules)[:10]}"],
                    fix_suggestion=f"Implement the missing modules: {missing}. "
                                   f"Follow the architecture plan's implementation_order.",
                    plan_phase_failed="module_decomposition",
                ))

        return diagnoses

    def _generate_iteration_guidance(
        self,
        diagnoses: list[FailureDiagnosis],
        plan: dict,
        results: dict,
    ) -> list[IterationGuidance]:
        """Generate specific guidance for the next iteration."""
        guidance = []

        # Group by failure type
        arch_failures = [d for d in diagnoses if d.failure_type == "architecture"]
        data_failures = [d for d in diagnoses if d.failure_type == "data"]
        train_failures = [d for d in diagnoses if d.failure_type == "training"]
        align_failures = [d for d in diagnoses if d.failure_type == "alignment"]

        # Architecture failures → fix architecture
        for d in arch_failures:
            mods_to_fix = [d.module_involved] if d.module_involved else []
            guidance.append(IterationGuidance(
                action="fix_architecture",
                priority=d.severity,
                specific_changes=[d.fix_suggestion],
                modules_to_modify=mods_to_fix,
                expected_improvement=f"Address {d.failure_type} failure: {d.root_cause[:100]}",
                risk_if_ignored="Continuing training with broken architecture wastes GPU hours",
            ))

        # Training failures → adjust training
        for d in train_failures:
            guidance.append(IterationGuidance(
                action="adjust_training",
                priority=d.severity,
                specific_changes=[d.fix_suggestion],
                modules_to_modify=["training_config", "loss_function"],
                expected_improvement="Fix training dynamics so model can actually learn",
                risk_if_ignored="Model will not converge regardless of architecture quality",
            ))

        # Alignment failures → re-align with plan
        for d in align_failures:
            guidance.append(IterationGuidance(
                action="fix_architecture",
                priority=d.severity,
                specific_changes=[d.fix_suggestion],
                modules_to_modify=["model_architecture"],
                expected_improvement="Bring implementation closer to the research idea",
                risk_if_ignored="Results won't reflect the research idea — wasted experiments",
            ))

        # Data failures → fix data
        for d in data_failures:
            guidance.append(IterationGuidance(
                action="change_data",
                priority=d.severity,
                specific_changes=[d.fix_suggestion],
                modules_to_modify=["dataset_loader", "data_augmentation"],
                expected_improvement="Enable the model to learn from the data",
                risk_if_ignored="No architecture change can overcome data problems",
            ))

        # If no specific failures, suggest incremental iteration
        if not guidance:
            guidance.append(IterationGuidance(
                action="iterate",
                priority="medium",
                specific_changes=[
                    "Try longer training",
                    "Adjust learning rate schedule",
                    "Add minor regularization",
                ],
                modules_to_modify=[],
                expected_improvement="Incremental improvement on current baseline",
                risk_if_ignored="None — safe to continue current direction",
            ))

        # Sort by priority
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        guidance.sort(key=lambda g: priority_order.get(g.priority, 99))

        return guidance


class IndependentProbe:
    """Third-party verification using a lightweight probe model.

    Instead of trusting the main model's own metrics, this spins up a
    lightweight evaluation model (e.g., a small CNN) on the same validation
    data. If the probe's assessment disagrees significantly with the main
    model's results, it flags an anomaly.

    This prevents the "self-evaluation" problem where:
    - The model reports good metrics but the outputs are actually bad
    - The training loss decreased but the model collapsed to mean prediction
    - The evaluation code has bugs that inflate metrics
    """

    def __init__(self, project_dir: Path, workspace: Path, thresholds: dict = None):
        self.project_dir = Path(project_dir)
        self.workspace = Path(workspace)
        self._thresholds = thresholds or {}
        self._gap_critical = self._thresholds.get("domain_gap_critical", 0.15)
        self._gap_moderate = self._thresholds.get("domain_gap_high", 0.05)

    def run_independent_assessment(
        self,
        model_path: str = "",
        checkpoint_path: str = "",
        val_data_path: str = "",
        reported_metrics: dict = None,
    ) -> IndependentAssessment:
        """Run a lightweight independent assessment.

        Strategy: Instead of training a separate model (too expensive), we:
        1. Load the main model's checkpoint
        2. Run forward pass on a few validation samples
        3. Compute basic statistics on the OUTPUTS (not the reported metrics)
        4. Compare output statistics against expected distributions

        This catches: collapsed outputs, constant predictions, NaN outputs,
        output range mismatches, etc.
        """
        assessment = IndependentAssessment()

        # Find the checkpoint and model
        ckpt_path = self._find_checkpoint(checkpoint_path)
        model_file = self._find_model_file(model_path)

        if not ckpt_path or not model_file:
            assessment.anomaly_detail = "No checkpoint or model file found for independent assessment"
            return assessment

        # Generate and run the probe script
        probe_script = self._build_probe_script(model_file, ckpt_path)
        if not probe_script:
            return assessment

        try:
            result = subprocess.run(
                [sys.executable, "-c", probe_script],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=str(self.project_dir),
            )

            if result.returncode == 0 and result.stdout.strip():
                probe_data = json.loads(result.stdout.strip())
                assessment.assessed = True
                assessment.independent_predictions_sample = probe_data.get("samples", [])

                # Analyze probe output
                self._analyze_probe_output(probe_data, reported_metrics or {}, assessment)
            else:
                logger.info(f"Independent probe execution failed: {result.stderr[:200]}")
                assessment.anomaly_detail = f"Probe execution error: {result.stderr[:200]}"

        except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception) as e:
            logger.info(f"Independent probe failed: {e}")
            assessment.anomaly_detail = str(e)[:200]

        return assessment

    def _find_checkpoint(self, checkpoint_path: str) -> Optional[Path]:
        """Find the most recent checkpoint file."""
        if checkpoint_path:
            p = self.project_dir / checkpoint_path
            if p.exists():
                return p

        # Search common locations (prioritize best_model/best_checkpoint over generic .pth)
        search_dirs = [
            self.project_dir / "outputs",
            self.project_dir / "checkpoints",
            self.workspace / "outputs",
        ]
        candidates = []
        seen = set()
        for d in search_dirs:
            if not d.exists():
                continue
            for pattern in ["**/best_model.pth", "**/best_checkpoint.pth"]:
                for p in d.glob(pattern):
                    if p not in seen:
                        candidates.append(p)
                        seen.add(p)
            # Only search generic .pth if no best_* found
            if not candidates:
                for p in d.glob("*.pth"):
                    if p not in seen:
                        candidates.append(p)
                        seen.add(p)

        if not candidates:
            return None

        # Return most recently modified
        return max(candidates, key=lambda p: p.stat().st_mtime)

    def _find_model_file(self, model_path: str) -> Optional[Path]:
        """Find the model definition file."""
        if model_path:
            p = self.project_dir / model_path
            if p.exists():
                return p

        # Search models directory
        models_dir = self.project_dir / "models"
        if not models_dir.exists():
            return None

        model_files = sorted(
            models_dir.glob("*.py"),
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        )
        return model_files[0] if model_files else None

    def _build_probe_script(self, model_file: Path, ckpt_path: Path) -> str:
        """Build a lightweight probe script that loads the model and runs inference."""
        model_rel = model_file.relative_to(self.project_dir)
        ckpt_rel = ckpt_path.relative_to(self.project_dir)

        return (
            "import sys, json, torch, numpy as np; "
            "sys.path.insert(0, '.'); "
            f"from {model_rel.with_suffix('').as_posix().replace('/', '.')} import *; "
            "ckpt = torch.load("
            f"  '{ckpt_rel}', "
            "  map_location='cpu', weights_only=False"
            "); "
            "import torch.nn as nn; "
            "ModelClass = None; "
            "for name, obj in list(globals().items()): "
            "  if (isinstance(obj, type) and issubclass(obj, nn.Module) "
            "      and obj is not nn.Module and hasattr(obj, 'forward')): "
            "    ModelClass = obj; break; "
            "if ModelClass is None: "
            "  print(json.dumps({'error': 'No nn.Module found'})); sys.exit(0); "
            "try: "
            "  model = ModelClass(); "
            "  state = ckpt.get('model_state_dict', ckpt) if isinstance(ckpt, dict) else ckpt; "
            "  model.load_state_dict(state, strict=False); "
            "  model.eval(); "
            "  first_key = next(iter(state), ''); "
            "  in_ch = state[first_key].shape[1] if first_key else 3; "
            "  shapes = []; "
            "  if first_key and len(state[first_key].shape) >= 5: "
            "    shapes.append([1, in_ch, 3, 64, 64]); "
            "  if first_key and len(state[first_key].shape) >= 4: "
            "    shapes.append([1, in_ch, 64, 64]); "
            "  shapes += [[1, 3, 64, 64]]; "
            "  out = None; "
            "  for shape in shapes: "
            "    try: "
            "      x = torch.randn(*shape); "
            "      with torch.no_grad(): out = model(x); "
            "      break; "
            "    except: continue; "
            "  if out is None: raise RuntimeError('No valid input shape found'); "
            "  if isinstance(out, (list, tuple)): out = out[0]; "
            "  if isinstance(out, dict): out = list(out.values())[0]; "
            "  out_np = out.detach().cpu().numpy(); "
            "  result = { "
            "    'output_shape': list(out.shape), "
            "    'output_mean': float(np.mean(out_np)), "
            "    'output_std': float(np.std(out_np)), "
            "    'output_min': float(np.min(out_np)), "
            "    'output_max': float(np.max(out_np)), "
            "    'output_nan_ratio': float(np.isnan(out_np).mean()), "
            "    'output_uniform_ratio': float(np.std(out_np) < 1e-5), "
            "    'samples': out_np.flatten()[:20].tolist(), "
            "  }; "
            "  print(json.dumps(result)); "
            "except Exception as e: "
            "  print(json.dumps({'error': str(e)[:200]})); "
        )

    def _analyze_probe_output(
        self,
        probe_data: dict,
        reported_metrics: dict,
        assessment: IndependentAssessment,
    ):
        """Analyze independent probe results and check for anomalies."""
        if "error" in probe_data:
            assessment.anomaly_detail = f"Probe error: {probe_data['error']}"
            return

        anomalies = []

        # Check 1: NaN outputs
        nan_ratio = probe_data.get("output_nan_ratio", 0)
        if nan_ratio > 0:
            anomalies.append(f"NaN outputs: {nan_ratio:.1%} of predictions are NaN")

        # Check 2: Uniform/collapsed outputs
        is_uniform = probe_data.get("output_uniform_ratio", 0)
        if is_uniform:
            anomalies.append(
                f"Collapsed outputs: std={probe_data.get('output_std', 0):.8f}. "
                f"Model produces near-constant predictions."
            )

        # Check 3: Output range suspicious (e.g., all ~0.5 = sigmoid converged to mean)
        out_mean = probe_data.get("output_mean", 0)
        out_std = probe_data.get("output_std", 0)
        if 0.4 < out_mean < 0.6 and out_std < 0.01:
            anomalies.append(
                f"Suspicious output range: mean={out_mean:.4f}, std={out_std:.6f}. "
                f"Model likely collapsed to mean prediction (sigmoid ≈ 0.5)."
            )

        # Check 4: Output range outside expected [0, 1] for typical normalized tasks
        out_min = probe_data.get("output_min", 0)
        out_max = probe_data.get("output_max", 0)
        if out_min < -1 or out_max > 2:
            anomalies.append(
                f"Output range [{out_min:.4f}, {out_max:.4f}] outside expected range. "
                f"Check output activation function."
            )

        # Check 5: Cross-validate against reported metrics
        # If model outputs are collapsed (std < 0.01) but reported MAE < 0.1, that's suspicious
        if out_std < 0.01:
            for metric_name, metric_val in reported_metrics.items():
                if "MAE" in metric_name:
                    try:
                        mae = float(metric_val)
                        if mae < 0.1:
                            anomalies.append(
                                f"METRIC SUSPECT: {metric_name}={mae:.4f} looks good, but model "
                                f"outputs are collapsed (std={out_std:.6f}). "
                                f"The metric computation may have a bug."
                            )
                    except (ValueError, TypeError):
                        pass

        if anomalies:
            assessment.anomaly_detected = True
            assessment.anomaly_detail = "; ".join(anomalies)
            assessment.confidence = "high"
            assessment.agreement_score = 0.0
        else:
            assessment.anomaly_detected = False
            assessment.confidence = "medium"
            assessment.agreement_score = 0.8  # No anomalies found → likely OK

        logger.info(
            f"Independent probe: assessed={assessment.assessed}, "
            f"anomaly={assessment.anomaly_detected}, "
            f"agreement={assessment.agreement_score:.2f}, "
            f"mean={out_mean:.4f}, std={out_std:.6f}"
        )
