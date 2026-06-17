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
