"""
AutoResearcher Simulation Sandbox — Pre-Training Model Validation & A/B Evaluation

Five-layer evaluation system:
  Layer 0: Feasibility check — can the model even run? (shape/OOM/crash)
  Layer 1: Design comparison — A(before) vs B(after) structural analysis
  Layer 2a: Reference-based evaluation — vs GT metrics + vs before metrics
  Layer 2b: Reference-free evaluation — internal behavior (module activity/contribution/gradient)
  Layer 3: Synthesis judgment — comprehensive assessment + project intent alignment
  Layer 4: Scaling guidance — which modules can grow, bottleneck analysis, GPU budget

Design principle: Every modification must be quantitatively justified.
The sandbox runs actual PyTorch code in a subprocess — no guessing.
"""

import json
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

logger = logging.getLogger("autoresearcher.sandbox")


# ──────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────

@dataclass
class FeasibilityReport:
    """Layer 0: Can the model run at all?"""
    feasible: bool = False
    forward_ok: bool = False
    backward_ok: bool = False
    output_shape: list = field(default_factory=list)
    layer_shapes: dict = field(default_factory=dict)  # layer_name → shape
    total_params: int = 0
    trainable_params: int = 0
    gpu_memory_peak_mb: float = 0.0
    max_safe_batch_size: int = 0
    error: str = ""
    warnings: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


@dataclass
class DesignComparison:
    """Layer 1: Structural A/B comparison."""
    params_before: int = 0
    params_after: int = 0
    params_delta: int = 0
    params_delta_pct: float = 0.0
    modules_before: dict = field(default_factory=dict)  # name → param_count
    modules_after: dict = field(default_factory=dict)
    new_modules: list = field(default_factory=list)
    removed_modules: list = field(default_factory=list)
    bottleneck_before: str = ""
    bottleneck_after: str = ""
    max_compress_ratio_before: float = 0.0
    max_compress_ratio_after: float = 0.0
    design_verdict: str = ""  # "improved" / "neutral" / "degraded"
    design_notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


@dataclass
class ReferenceEvaluation:
    """Layer 2a: Metrics vs GT and vs before."""
    mae_before_avg: Optional[float] = None
    mae_after_avg: Optional[float] = None
    mae_delta: Optional[float] = None
    per_sample: list = field(default_factory=list)  # {sample, gt, before, after, delta}
    param_efficiency_before: Optional[float] = None  # MAE per 1K params
    param_efficiency_after: Optional[float] = None
    domain_breakdown: dict = field(default_factory=dict)
    verdict: str = ""  # "significant_improvement" / "marginal" / "degradation"

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


@dataclass
class InternalBehaviorReport:
    """Layer 2b: Module-level behavior analysis without GT."""
    module_activity: dict = field(default_factory=dict)
    # name → {dead_ratio, output_std, output_mean, status: "active"|"weak"|"dead"}
    module_contribution: dict = field(default_factory=dict)
    # name → {output_shift_pct, contribution: "high"|"medium"|"low"|"none"}
    gradient_health: dict = field(default_factory=dict)
    # name → {grad_norm, grad_ratio_vs_backbone, status: "normal"|"weak"|"starved"}
    parameter_utilization: dict = field(default_factory=dict)
    # name → {weight_std, weight_entropy, learning_signal: "strong"|"weak"|"none"}
    gradient_balance: dict = field(default_factory=dict)
    # {imbalance_ratio, strongest_module, weakest_module}
    data_flow_trace: list = field(default_factory=list)
    # [{layer, input_shape, output_shape}]
    verdict: str = ""
    dead_modules: list = field(default_factory=list)
    healthy_modules: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


@dataclass
class SynthesisJudgment:
    """Layer 3: Comprehensive judgment combining all layers."""
    modification_verdict: str = ""  # "effective" / "partial" / "ineffective" / "harmful"
    effective_modules: list = field(default_factory=list)
    ineffective_modules: list = field(default_factory=list)
    harmful_modules: list = field(default_factory=list)
    project_alignment: str = ""  # "aligned" / "partially_aligned" / "misaligned"
    alignment_reason: str = ""
    recommendation: str = ""
    confidence: str = "low"  # "low" | "medium" | "high"

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


@dataclass
class ScalingGuidance:
    """Layer 4: Where can the model grow?"""
    scalable_modules: list = field(default_factory=list)
    # [{name, current_params, suggested_growth, reason}]
    bottlenecks: list = field(default_factory=list)
    # [{location, type, detail}]
    gpu_budget: dict = field(default_factory=dict)
    # {current_mb, headroom_mb, max_batch_at_current, max_batch_after_scaling}
    recommendation: str = ""

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


# ──────────────────────────────────────────────────────────
# Simulation Sandbox
# ──────────────────────────────────────────────────────────

class SimulationSandbox:
    """Isolated sandbox for model validation, A/B comparison, and scaling guidance.

    All model execution happens in subprocesses — the main loop is never at risk.
    Each layer produces structured data that feeds into subsequent layers.
    """

    def __init__(self, project_dir: Path, workspace: Path, config: dict = None):
        self.project_dir = Path(project_dir)
        self.workspace = Path(workspace)
        self.snapshot_dir = self.workspace / "model_snapshots"
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self._config = config or {}
        sbx = self._config.get("sandbox", {})
        self.target_gpu_mb: int = sbx.get("gpu_memory_mb", 24000)
        self.default_input_shape: str = json.dumps(sbx.get("default_input_shape", [1, 3, 64, 64]))
        self.subprocess_timeout: int = sbx.get("subprocess_timeout", 120)
        self.inference_timeout: int = sbx.get("inference_timeout", 120)
        self.feasibility_timeout: int = sbx.get("feasibility_timeout", 90)

    # ── Layer 0: Feasibility ──

    def check_feasibility(
        self,
        model_path: str,
        input_shape: str = "",
        target_gpu_mb: int = 0,
        timeout: int = 0,
    ) -> FeasibilityReport:
        """Can the model instantiate + forward + backward without crashing?"""
        if target_gpu_mb <= 0:
            target_gpu_mb = self.target_gpu_mb
        if timeout <= 0:
            timeout = self.feasibility_timeout
        report = FeasibilityReport()
        model_file = self._resolve_model(model_path)
        if not model_file:
            report.error = f"Model file not found: {model_path}"
            return report

        # Build and run the feasibility script
        script = self._build_feasibility_script(model_file, input_shape, target_gpu_mb)
        result = self._run_subprocess(script, timeout)

        if not result:
            report.error = "Subprocess failed or timed out"
            return report

        if "error" in result:
            report.error = result["error"]
            if "shape" in result.get("error", "").lower():
                report.warnings.append("Shape mismatch — model cannot process the given input")
            return report

        report.feasible = result.get("forward_ok", False)
        report.forward_ok = result.get("forward_ok", False)
        report.backward_ok = result.get("backward_ok", False)
        report.output_shape = result.get("output_shape", [])
        report.layer_shapes = result.get("layer_shapes", {})
        report.total_params = result.get("total_params", 0)
        report.trainable_params = result.get("trainable_params", 0)
        report.gpu_memory_peak_mb = result.get("gpu_memory_peak_mb", 0)
        report.max_safe_batch_size = result.get("max_safe_batch_size", 0)
        report.warnings = result.get("warnings", [])

        if report.feasible:
            logger.info(
                f"Sandbox feasibility OK: params={report.total_params:,}, "
                f"output_shape={report.output_shape}, "
                f"gpu_peak={report.gpu_memory_peak_mb:.0f}MB"
            )
        else:
            logger.warning(f"Sandbox feasibility FAILED: {report.error}")

        return report

    # ── Layer 1: Design Comparison ──

    # ── Layer 2b: Internal Behavior ──

    # ── Layer 2a: Reference Evaluation ──

    # v18: evaluate_vs_reference removed (Layer 2a dead chain)
    # ── Layer 4: Scaling Guidance ──

    # ── Full Pipeline ──


