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

import ast
import json
import math
import re
import sys
import logging
import subprocess
import shutil
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


@dataclass
class SandboxReport:
    """Complete sandbox evaluation report."""
    cycle: int = 0
    model_path: str = ""
    feasible: bool = False
    feasibility: dict = field(default_factory=dict)
    design_comparison: dict = field(default_factory=dict)
    reference_evaluation: dict = field(default_factory=dict)
    internal_behavior: dict = field(default_factory=dict)
    judgment: dict = field(default_factory=dict)
    scaling: dict = field(default_factory=dict)
    summary: str = ""

    def to_dict(self) -> dict:
        return {
            "cycle": self.cycle,
            "model_path": self.model_path,
            "feasible": self.feasible,
            "feasibility": self.feasibility,
            "design_comparison": self.design_comparison,
            "reference_evaluation": self.reference_evaluation,
            "internal_behavior": self.internal_behavior,
            "judgment": self.judgment,
            "scaling": self.scaling,
            "summary": self.summary,
        }


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

    def compare_design(
        self,
        model_path_before: str,
        model_path_after: str,
    ) -> DesignComparison:
        """Compare structural design of before/after models."""
        comp = DesignComparison()

        info_before = self._extract_model_info(model_path_before)
        info_after = self._extract_model_info(model_path_after)

        comp.params_before = info_before.get("total_params", 0)
        comp.params_after = info_after.get("total_params", 0)
        comp.params_delta = comp.params_after - comp.params_before
        comp.params_delta_pct = (
            (comp.params_delta / comp.params_before * 100)
            if comp.params_before > 0 else 0
        )

        comp.modules_before = info_before.get("module_params", {})
        comp.modules_after = info_after.get("module_params", {})

        # Find new/removed modules
        names_before = set(comp.modules_before.keys())
        names_after = set(comp.modules_after.keys())
        comp.new_modules = sorted(names_after - names_before)
        comp.removed_modules = sorted(names_before - names_after)

        # Analyze information flow bottlenecks
        comp.bottleneck_before = info_before.get("max_bottleneck", "unknown")
        comp.bottleneck_after = info_after.get("max_bottleneck", "unknown")
        comp.max_compress_ratio_before = info_before.get("max_compress_ratio", 0)
        comp.max_compress_ratio_after = info_after.get("max_compress_ratio", 0)

        # Design verdict
        notes = []
        if comp.params_delta_pct > 50:
            notes.append(f"参数量增长{comp.params_delta_pct:.0f}%，需要证明有效性")
        if comp.max_compress_ratio_after > comp.max_compress_ratio_before * 1.5:
            notes.append(
                f"信息瓶颈恶化: compress ratio {comp.max_compress_ratio_before:.1f}:1 → "
                f"{comp.max_compress_ratio_after:.1f}:1"
            )
        if comp.new_modules:
            notes.append(f"新增模块: {comp.new_modules}")

        comp.design_notes = notes

        if comp.params_delta <= 0:
            comp.design_verdict = "neutral"
        elif comp.params_delta > 0 and comp.max_compress_ratio_after <= comp.max_compress_ratio_before:
            comp.design_verdict = "improved"
        else:
            comp.design_verdict = "needs_validation"

        return comp

    # ── Layer 2b: Internal Behavior ──

    def analyze_internal_behavior(
        self,
        model_path: str,
        checkpoint_path: str = "",
        input_shape: str = "",
        timeout: int = 120,
    ) -> InternalBehaviorReport:
        """Reference-free analysis: module activity, contribution, gradient health."""
        report = InternalBehaviorReport()

        model_file = self._resolve_model(model_path)
        if not model_file:
            return report

        script = self._build_behavior_script(model_file, checkpoint_path, input_shape)
        result = self._run_subprocess(script, timeout)

        if not result or "error" in result:
            logger.debug(f"Internal behavior analysis failed: {result}")
            return report

        # Parse module activity
        for mod_stat in result.get("activation_stats", []):
            name = mod_stat.get("name", "")
            dead_ratio = mod_stat.get("dead_ratio", 0)
            output_std = mod_stat.get("std", 0)
            output_mean = mod_stat.get("mean", 0)

            if dead_ratio > 0.9:
                status = "dead"
            elif dead_ratio > 0.5 or output_std < 1e-4:
                status = "weak"
            else:
                status = "active"

            report.module_activity[name] = {
                "dead_ratio": dead_ratio,
                "output_std": output_std,
                "output_mean": output_mean,
                "status": status,
            }

        # Parse gradient health
        backbone_grad = 0
        for g_stat in result.get("gradient_stats", []):
            name = g_stat.get("name", "")
            grad_norm = g_stat.get("grad_norm", 0)
            # Use first significant gradient as backbone reference
            if grad_norm > backbone_grad:
                backbone_grad = grad_norm

        for g_stat in result.get("gradient_stats", []):
            name = g_stat.get("name", "")
            grad_norm = g_stat.get("grad_norm", 0)
            ratio = grad_norm / backbone_grad if backbone_grad > 0 else 0

            if ratio < 0.001:
                status = "starved"
            elif ratio < 0.01:
                status = "weak"
            else:
                status = "normal"

            report.gradient_health[name] = {
                "grad_norm": grad_norm,
                "grad_ratio_vs_backbone": round(ratio, 4),
                "status": status,
            }

        # Gradient balance
        grad_balance = result.get("gradient_balance", {})
        if grad_balance:
            report.gradient_balance = grad_balance

        # Parameter utilization
        for p_stat in result.get("parameter_stats", []):
            name = p_stat.get("name", "")
            weight_std = p_stat.get("std", 0)
            if weight_std < 1e-5:
                signal = "none"
            elif weight_std < 1e-3:
                signal = "weak"
            else:
                signal = "strong"
            report.parameter_utilization[name] = {
                "weight_std": weight_std,
                "num_params": p_stat.get("num_params", 0),
                "learning_signal": signal,
            }

        # Data flow trace
        report.data_flow_trace = result.get("data_flow_trace", [])

        # Categorize modules
        for name, info in report.module_activity.items():
            if info["status"] == "dead":
                report.dead_modules.append(name)
            elif info["status"] == "active":
                # Check gradient too
                grad_info = report.gradient_health.get(name, {})
                if grad_info.get("status") in ("starved", "weak"):
                    report.dead_modules.append(name)
                else:
                    report.healthy_modules.append(name)

        if report.dead_modules:
            report.verdict = (
                f"DEAD MODULES DETECTED: {report.dead_modules}. "
                f"These modules are not contributing to the model."
            )
        elif report.healthy_modules:
            report.verdict = "All analyzed modules are active and receiving gradients."

        return report

    # ── Layer 2a: Reference Evaluation ──

    # v18: evaluate_vs_reference removed (Layer 2a dead chain)
    def synthesize_judgment(
        self,
        design: DesignComparison,
        ref_eval: ReferenceEvaluation,
        behavior: InternalBehaviorReport,
        project_brief_path: str = "",
    ) -> SynthesisJudgment:
        """Combine all layers into a comprehensive judgment."""
        judgment = SynthesisJudgment()

        # Categorize new modules
        for mod_name in design.new_modules:
            activity = behavior.module_activity.get(mod_name, {})
            grad = behavior.gradient_health.get(mod_name, {})
            is_dead = (
                activity.get("status") in ("dead", "weak")
                or grad.get("status") in ("starved", "weak")
            )
            if is_dead:
                judgment.ineffective_modules.append(mod_name)
            else:
                judgment.effective_modules.append(mod_name)

        # Overall verdict
        has_improvement = ref_eval.verdict in ("significant_improvement", "marginal_improvement")
        has_dead = len(judgment.ineffective_modules) > 0
        has_degradation = ref_eval.verdict == "degradation"

        if has_degradation:
            judgment.modification_verdict = "harmful"
        elif has_improvement and not has_dead:
            judgment.modification_verdict = "effective"
        elif has_improvement and has_dead:
            judgment.modification_verdict = "partial"
            judgment.recommendation = (
                f"部分模块有效({judgment.effective_modules})，部分无效({judgment.ineffective_modules})。"
                f"建议移除无效模块，将资源分配给有效模块。"
            )
        elif not has_improvement and has_dead:
            judgment.modification_verdict = "ineffective"
        else:
            judgment.modification_verdict = "neutral"

        # Project alignment
        brief_text = ""
        brief_path = self.project_dir / project_brief_path if project_brief_path else self.workspace / "PROJECT_BRIEF.md"
        if brief_path.exists():
            try:
                brief_text = brief_path.read_text()[:2000].lower()
            except Exception:
                pass

        if brief_text:
            # Check if modification addresses core goals mentioned in brief
            core_keywords = self._extract_core_keywords(brief_text)
            task_keywords = set()
            for mod_name in design.new_modules:
                task_keywords.update(mod_name.replace("_", " ").split())
            overlap = core_keywords & task_keywords
            if overlap:
                judgment.project_alignment = "aligned"
                judgment.alignment_reason = f"修改涉及项目核心目标关键词: {overlap}"
            elif design.new_modules:
                judgment.project_alignment = "partially_aligned"
                judgment.alignment_reason = "新增模块未直接对应PROJECT_BRIEF核心目标"
            else:
                judgment.project_alignment = "aligned"
        else:
            judgment.project_alignment = "unknown"

        # Recommendation
        if not judgment.recommendation:
            if judgment.modification_verdict == "effective":
                judgment.recommendation = "修改有效，可以保留并考虑扩展有效模块"
            elif judgment.modification_verdict == "harmful":
                judgment.recommendation = "修改有害，建议回退到修改前版本"
            elif judgment.modification_verdict == "ineffective":
                judgment.recommendation = (
                    f"修改无效: {judgment.ineffective_modules}未激活。"
                    f"检查输入连接和梯度通路。"
                )

        # Confidence
        if ref_eval.mae_delta is not None and behavior.healthy_modules:
            judgment.confidence = "high"
        elif behavior.healthy_modules or ref_eval.mae_delta is not None:
            judgment.confidence = "medium"
        else:
            judgment.confidence = "low"

        return judgment

    # ── Layer 4: Scaling Guidance ──

    def generate_scaling_guidance(
        self,
        behavior: InternalBehaviorReport,
        feasibility: FeasibilityReport,
        design: DesignComparison,
        target_gpu_mb: int = 0,
    ) -> ScalingGuidance:
        """Where can the model grow? What are the bottlenecks?"""
        if target_gpu_mb <= 0:
            target_gpu_mb = self.target_gpu_mb
        guidance = ScalingGuidance()

        # Identify scalable modules: active + good gradient + good parameter utilization
        for name, info in behavior.module_activity.items():
            if info["status"] != "active":
                continue
            grad = behavior.gradient_health.get(name, {})
            if grad.get("status") not in ("normal",):
                continue
            param = behavior.parameter_utilization.get(name, {})
            if param.get("learning_signal") == "none":
                continue

            current_params = param.get("num_params", 0)
            suggested = int(current_params * 1.5)
            guidance.scalable_modules.append({
                "name": name,
                "current_params": current_params,
                "suggested_growth": suggested,
                "reason": "模块活跃、梯度正常、参数利用率高，适合扩展",
            })

        # Identify bottlenecks
        if design.max_compress_ratio_after > 8:
            guidance.bottlenecks.append({
                "location": "融合层/信息瓶颈",
                "type": "information_compression",
                "detail": f"最大压缩比 {design.max_compress_ratio_after:.1f}:1，"
                          f"可能导致信息丢失。考虑增加通道或加skip connection。",
            })

        if len(behavior.dead_modules) > 0:
            guidance.bottlenecks.append({
                "location": str(behavior.dead_modules),
                "type": "dead_modules",
                "detail": f"这些模块未激活，占用参数但不起作用。移除可释放资源。",
            })

        # GPU budget
        current_mb = feasibility.gpu_memory_peak_mb
        if current_mb > 0 and target_gpu_mb > 0:
            headroom = target_gpu_mb - current_mb
            current_batch = max(1, feasibility.max_safe_batch_size) if feasibility.max_safe_batch_size > 0 else 1
            guidance.gpu_budget = {
                "current_mb": round(current_mb, 0),
                "target_mb": target_gpu_mb,
                "headroom_mb": round(headroom, 0),
                "max_batch_at_current": current_batch,
                "estimated_max_batch_at_target": max(1, int(current_batch * target_gpu_mb / current_mb)) if current_mb > 0 else 0,
            }
            if headroom < 500:
                guidance.bottlenecks.append({
                    "location": "GPU显存",
                    "type": "memory",
                    "detail": f"仅剩{headroom:.0f}MB余量，大规模扩展可能OOM。"
                              f"考虑gradient checkpointing或mixed precision。",
                })

        # Recommendation
        if guidance.scalable_modules:
            names = [m["name"] for m in guidance.scalable_modules[:3]]
            guidance.recommendation = (
                f"可扩展模块: {names}。"
                f"移除死模块({behavior.dead_modules[:3]})可释放资源给活跃模块。"
            )
        elif behavior.dead_modules:
            guidance.recommendation = (
                f"当前无可扩展模块，但有{len(behavior.dead_modules)}个死模块。"
                f"先清理死模块再评估扩展方向。"
            )

        return guidance

    # ── Full Pipeline ──


