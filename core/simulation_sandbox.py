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

    def evaluate_vs_reference(
        self,
        model_path_before: str,
        model_path_after: str,
        checkpoint_path: str,
        val_data_dir: str = "",
        gt_dir: str = "",
        max_samples: int = 10,
    ) -> ReferenceEvaluation:
        """Reference-based evaluation: compare before/after on val samples vs GT."""
        report = ReferenceEvaluation()

        # Find validation samples
        val_samples = self._find_val_samples(val_data_dir, max_samples)
        if not val_samples:
            report.verdict = "no_val_samples"
            return report

        model_after = self._resolve_model(model_path_after)
        model_before = self._resolve_model(model_path_before)

        if not model_after:
            report.verdict = "model_not_found"
            return report

        # Run inference on both models
        after_result = self._run_inference(
            model_after, checkpoint_path, val_samples, gt_dir
        )
        before_result = None
        if model_before and model_before.exists():
            before_result = self._run_inference(
                model_before, "", val_samples, gt_dir
            )

        if not after_result:
            report.verdict = "inference_failed"
            return report

        # Aggregate results
        mae_before_list = []
        mae_after_list = []

        for sample_id in after_result.get("per_sample", {}):
            after_mae = after_result["per_sample"][sample_id].get("mae")
            if after_mae is not None:
                mae_after_list.append(after_mae)

            if before_result:
                before_mae = before_result.get("per_sample", {}).get(sample_id, {}).get("mae")
                if before_mae is not None:
                    mae_before_list.append(before_mae)

        if mae_after_list:
            report.mae_after_avg = sum(mae_after_list) / len(mae_after_list)
        if mae_before_list:
            report.mae_before_avg = sum(mae_before_list) / len(mae_before_list)

        if report.mae_before_avg is not None and report.mae_after_avg is not None:
            report.mae_delta = report.mae_after_avg - report.mae_before_avg

            # Parameter efficiency
            info_after = self._extract_model_info(model_path_after)
            info_before = self._extract_model_info(model_path_before)
            params_after = info_after.get("total_params", 1)
            params_before = info_before.get("total_params", 1)
            report.param_efficiency_after = report.mae_after_avg / (params_after / 1000)
            report.param_efficiency_before = report.mae_before_avg / (params_before / 1000)

            # Verdict
            improvement_pct = abs(report.mae_delta) / report.mae_before_avg * 100 if report.mae_before_avg > 0 else 0
            if report.mae_delta < -0.02:
                report.verdict = "significant_improvement"
            elif report.mae_delta < 0:
                report.verdict = "marginal_improvement"
            elif report.mae_delta < 0.01:
                report.verdict = "marginal"
            else:
                report.verdict = "degradation"

        return report

    # ── Layer 3: Synthesis ──

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

    def full_evaluation(
        self,
        cycle: int,
        model_path: str,
        model_path_before: str = "",
        checkpoint_path: str = "",
        input_shape: str = "",
        target_gpu_mb: int = 0,
        val_data_dir: str = "",
        gt_dir: str = "",
        project_brief_path: str = "",
    ) -> SandboxReport:
        """Run all 5 layers and produce a complete report."""
        if target_gpu_mb <= 0:
            target_gpu_mb = self.target_gpu_mb
        if not input_shape:
            input_shape = self.default_input_shape
        report = SandboxReport(cycle=cycle, model_path=model_path)

        # Layer 0: Feasibility
        feasibility = self.check_feasibility(model_path, input_shape, target_gpu_mb)
        report.feasibility = feasibility.to_dict()
        report.feasible = feasibility.feasible

        if not feasibility.feasible:
            report.summary = (
                f"模型不可行: {feasibility.error}。无法进行后续评价。"
            )
            return report

        # Layer 1: Design comparison (if before model exists)
        design = DesignComparison()
        if model_path_before:
            before_file = self._resolve_model(model_path_before)
            if before_file and before_file.exists():
                design = self.compare_design(model_path_before, model_path)
        report.design_comparison = design.to_dict()

        # Layer 2b: Internal behavior (always possible)
        behavior = self.analyze_internal_behavior(
            model_path, checkpoint_path, input_shape
        )
        report.internal_behavior = behavior.to_dict()

        # Layer 2a: Reference evaluation (if val data exists)
        ref_eval = ReferenceEvaluation()
        if val_data_dir:
            ref_eval = self.evaluate_vs_reference(
                model_path_before or "", model_path,
                checkpoint_path, val_data_dir, gt_dir,
            )
        report.reference_evaluation = ref_eval.to_dict()

        # Layer 3: Synthesis
        judgment = self.synthesize_judgment(
            design, ref_eval, behavior, project_brief_path
        )
        report.judgment = judgment.to_dict()

        # Layer 4: Scaling guidance
        scaling = self.generate_scaling_guidance(
            behavior, feasibility, design, target_gpu_mb
        )
        report.scaling = scaling.to_dict()

        # Summary
        parts = []
        if judgment.modification_verdict != "effective":
            parts.append(f"修改判定: {judgment.modification_verdict}")
        if behavior.dead_modules:
            parts.append(f"死模块: {behavior.dead_modules[:3]}")
        if judgment.effective_modules:
            parts.append(f"有效模块: {judgment.effective_modules[:3]}")
        if scaling.scalable_modules:
            parts.append(f"可扩展: {[m['name'] for m in scaling.scalable_modules[:3]]}")
        report.summary = " | ".join(parts) if parts else "评价完成"

        return report

    # ── Model Snapshot Management ──

    def save_snapshot(self, model_path: str, cycle: int):
        """Save a snapshot of the model file for before/after comparison."""
        src = self._resolve_model(model_path)
        if not src or not src.exists():
            return
        dst = self.snapshot_dir / f"cycle_{cycle:03d}_{src.name}"
        try:
            shutil.copy2(src, dst)
            logger.debug(f"Model snapshot saved: {dst}")
        except Exception as e:
            logger.debug(f"Snapshot save failed: {e}")

    def find_previous_snapshot(self, current_cycle: int, model_name: str = "") -> Optional[Path]:
        """Find the most recent snapshot before current cycle."""
        snapshots = sorted(self.snapshot_dir.glob("cycle_*_*.py"), reverse=True)
        for snap in snapshots:
            # Parse cycle number from filename
            match = re.match(r"cycle_(\d+)_", snap.name)
            if match:
                snap_cycle = int(match.group(1))
                if snap_cycle < current_cycle:
                    if not model_name or snap.name.endswith(model_name):
                        return snap
        return None

    # ── Prompt Generation ──

    def format_report_prompt(self, report: SandboxReport) -> str:
        """Format sandbox report as a prompt for the Leader."""
        if not report.feasible:
            return (
                f"SANDBOX: 模型不可行\n"
                f"原因: {report.feasibility.get('error', 'unknown')}\n"
                f"必须修复模型代码后才能训练。"
            )

        lines = ["SANDBOX 模型评价报告:\n"]

        # Feasibility summary
        feas = report.feasibility
        lines.append(f"[可行性] ✅ 参数={feas.get('total_params', 0):,}, "
                      f"输出shape={feas.get('output_shape', [])}, "
                      f"GPU峰值={feas.get('gpu_memory_peak_mb', 0):.0f}MB")

        # Design
        design = report.design_comparison
        if design.get("params_delta", 0) != 0:
            lines.append(f"[设计] 参数变化: {design['params_delta']:+,} ({design['params_delta_pct']:+.1f}%)")
            if design.get("new_modules"):
                lines.append(f"  新增模块: {design['new_modules']}")
            for note in design.get("design_notes", []):
                lines.append(f"  ⚠ {note}")

        # Reference evaluation
        ref = report.reference_evaluation
        if ref.get("verdict") and ref["verdict"] != "no_val_samples":
            lines.append(f"[有参考评价] 判定: {ref.get('verdict', 'N/A')}")
            if ref.get("mae_delta") is not None:
                delta = ref["mae_delta"]
                direction = "改善" if delta < 0 else "恶化"
                lines.append(f"  MAE变化: {delta:+.4f} ({direction})")
            if ref.get("param_efficiency_before") and ref.get("param_efficiency_after"):
                lines.append(f"  参数效率: {ref['param_efficiency_before']:.6f} → {ref['param_efficiency_after']:.6f} (MAE/1K参数)")

        # Internal behavior
        behavior = report.internal_behavior
        if behavior.get("dead_modules"):
            lines.append(f"[无参考评价] ❌ 死模块: {behavior['dead_modules']}")
            for mod_name in behavior["dead_modules"]:
                act = behavior.get("module_activity", {}).get(mod_name, {})
                grad = behavior.get("gradient_health", {}).get(mod_name, {})
                lines.append(f"  {mod_name}: dead_ratio={act.get('dead_ratio', 0):.0%}, "
                              f"grad_ratio={grad.get('grad_ratio_vs_backbone', 0):.4f}")
        if behavior.get("healthy_modules"):
            lines.append(f"  ✅ 活跃模块: {behavior['healthy_modules']}")

        # Judgment
        judg = report.judgment
        if judg.get("modification_verdict"):
            lines.append(f"[综合判定] {judg['modification_verdict']} "
                          f"(置信度: {judg.get('confidence', 'low')})")
        if judg.get("recommendation"):
            lines.append(f"  建议: {judg['recommendation']}")
        if judg.get("project_alignment"):
            lines.append(f"  项目初衷对齐: {judg['project_alignment']}")

        # Scaling
        scaling = report.scaling
        if scaling.get("scalable_modules"):
            names = [m["name"] for m in scaling["scalable_modules"]]
            lines.append(f"[扩展指引] 可扩展: {names}")
        if scaling.get("bottlenecks"):
            for bn in scaling["bottlenecks"]:
                lines.append(f"  ⚠ 瓶颈: {bn.get('detail', '')}")

        return "\n".join(lines)

    # ──────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────

    def _resolve_model(self, model_path: str) -> Optional[Path]:
        if not model_path:
            return None
        p = self.project_dir / model_path
        if p.exists():
            return p
        # Try models/ directory
        models_dir = self.project_dir / "models"
        if models_dir.exists():
            for f in models_dir.glob("*.py"):
                if model_path in str(f) or f.stem == model_path:
                    return f
            # Do NOT silently fallback to a random model — return None
            logger.warning(
                f"Sandbox: cannot resolve model '{model_path}' in models/ directory. "
                f"Available: {[f.name for f in models_dir.glob('*.py')]}"
            )
        return None

    def _run_subprocess(self, script: str, timeout: int = 90) -> Optional[dict]:
        """Run a Python script in subprocess and parse JSON output."""
        try:
            proc = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(self.project_dir),
            )
            if proc.returncode == 0 and proc.stdout.strip():
                text = proc.stdout.strip()
                # Find JSON in output
                start = text.find("{")
                if start >= 0:
                    return json.loads(text[start:])
            elif proc.stderr:
                return {"error": proc.stderr[-500:]}
        except subprocess.TimeoutExpired:
            return {"error": f"Subprocess timed out after {timeout}s"}
        except (json.JSONDecodeError, Exception) as e:
            return {"error": str(e)[:300]}
        return None

    def _find_val_samples(self, val_data_dir: str, max_samples: int) -> list[Path]:
        """Find validation sample files."""
        val_dir = self.project_dir / val_data_dir if val_data_dir else None
        if not val_dir or not val_dir.exists():
            # Try common locations
            for candidate in [
                self.project_dir / "data" / "val",
                self.project_dir / "data" / "validation",
            ]:
                if candidate.exists():
                    val_dir = candidate
                    break
        if not val_dir:
            return []

        samples = []
        for ext in ("*.png", "*.jpg", "*.npy", "*.pt", "*.pth", "*.pfm"):
            samples.extend(val_dir.rglob(ext))
            if len(samples) >= max_samples:
                break
        return samples[:max_samples]

    def _run_inference(self, model_path: Path, checkpoint_path: str,
                       val_samples: list, gt_dir: str) -> Optional[dict]:
        """Run inference on val samples. Returns per-sample metrics."""
        script = self._build_inference_script(model_path, checkpoint_path, val_samples, gt_dir)
        return self._run_subprocess(script, timeout=self.inference_timeout)

    def _extract_model_info(self, model_path: str) -> dict:
        """Extract structural info from model file (AST-based, no subprocess)."""
        info = {"total_params": 0, "module_params": {}, "max_bottleneck": "unknown", "max_compress_ratio": 0}
        model_file = self._resolve_model(model_path)
        if not model_file:
            return info

        try:
            content = model_file.read_text()
            tree = ast.parse(content)
        except Exception:
            return info

        # Find nn.Module subclasses and their self.xxx = ... assignments
        module_params = {}
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            is_module = False
            for base in node.bases:
                if isinstance(base, ast.Attribute) and base.attr == "Module":
                    is_module = True
                elif isinstance(base, ast.Name) and base.id == "nn":
                    is_module = True
            if not is_module:
                continue

            param_count = 0
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                    for stmt in ast.walk(item):
                        if isinstance(stmt, ast.Call):
                            # Rough parameter estimation from calls
                            call_name = ""
                            if isinstance(stmt.func, ast.Attribute):
                                call_name = stmt.func.attr
                            elif isinstance(stmt.func, ast.Name):
                                call_name = stmt.func.id

                            if call_name in ("Conv2d", "Conv3d", "ConvTranspose2d"):
                                # Extract channel args
                                args = stmt.args
                                if len(args) >= 2:
                                    try:
                                        in_ch = self._get_const(args[0], 64)
                                        out_ch = self._get_const(args[1], 64)
                                        k = self._get_const(args[2], 3) if len(args) > 2 else 3
                                        # Conv3d uses 3D kernel, others use 2D
                                        if call_name == "Conv3d":
                                            kd = self._get_const(args[3], k) if len(args) > 3 else k
                                            param_count += in_ch * out_ch * k * k * kd
                                        else:
                                            param_count += in_ch * out_ch * k * k
                                    except Exception:
                                        param_count += 1000
                            elif call_name in ("Linear",):
                                if len(stmt.args) >= 2:
                                    try:
                                        in_f = self._get_const(stmt.args[0], 256)
                                        out_f = self._get_const(stmt.args[1], 256)
                                        param_count += in_f * out_f + out_f
                                    except Exception:
                                        param_count += 1000

            if param_count > 0:
                module_params[node.name] = param_count

        info["module_params"] = module_params
        info["total_params"] = sum(module_params.values())

        # Estimate bottleneck (module with smallest output relative to its input)
        if module_params:
            min_mod = min(module_params, key=module_params.get)
            info["max_bottleneck"] = min_mod

        return info

    def _get_const(self, node, default=0):
        """Extract constant value from AST node."""
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -self._get_const(node.operand, default)
        return default

    def _extract_core_keywords(self, brief_text: str) -> set[str]:
        """Extract core goal keywords from PROJECT_BRIEF."""
        # Look for emphasized/goal sections
        keywords = set()
        # Common research goal indicators
        goal_patterns = [
            r"goal[:\s]+([^\n.]+)",
            r"objective[:\s]+([^\n.]+)",
            r"target[:\s]+([^\n.]+)",
            r"improve\s+(\w+)",
            r"reduce\s+(\w+)",
            r"unif\w*\s+(\w+)",
            r"cross[_\s]?domain",
        ]
        for pat in goal_patterns:
            for match in re.finditer(pat, brief_text, re.IGNORECASE):
                words = match.group(1 if match.lastindex else 0).split()
                for w in words:
                    if len(w) > 3:
                        keywords.add(w.lower())
        return keywords

    # ── Script Builders ──

    def _build_feasibility_script(self, model_file: Path, input_shape: str, target_gpu_mb: int) -> str:
        """Build subprocess script for Layer 0 feasibility check."""
        model_rel = model_file.relative_to(self.project_dir)
        parent_pkg = str(model_rel.parent).replace("/", ".").replace("\\", ".") if model_rel.parent != Path(".") else ""
        import_stmt = f"from {parent_pkg}.{model_rel.stem} import *" if parent_pkg else f"import {model_rel.stem}"

        # Try to infer shape from state_dict if not provided
        default_shape = json.loads(self.default_input_shape)
        shape_code = f'shape = {default_shape}'
        if input_shape:
            try:
                shape = json.loads(input_shape)
                shape_code = f'shape = {shape}'
            except Exception:
                pass

        use_cuda = target_gpu_mb > 0
        cuda_setup = """
import torch
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
if device.startswith('cuda'):
    torch.cuda.reset_peak_memory_stats()
""" if use_cuda else """
import torch
device = 'cpu'
"""
        memory_code = """
    peak_mb = torch.cuda.max_memory_allocated() / (1024*1024) if device.startswith('cuda') else 0
    # Estimate max batch size
    if peak_mb > 0 and peak_mb < TARGET_GPU_MB:
        est_per_sample = peak_mb / 1  # batch=1
        max_batch = max(1, int(TARGET_GPU_MB * 0.85 / est_per_sample))
    else:
        max_batch = 1
""" if use_cuda else """
    peak_mb = 0
    max_batch = 1
"""

        return f'''
import sys, json, traceback
sys.path.insert(0, '{self.project_dir}')
{cuda_setup}
import torch.nn as nn

TARGET_GPU_MB = {target_gpu_mb}
{shape_code}

result = {{"forward_ok": False, "backward_ok": False, "warnings": []}}

try:
    {import_stmt}
except Exception as e:
    result["error"] = f"Import failed: {{e}}"
    print(json.dumps(result))
    sys.exit(0)

# Find model class
import importlib
spec = importlib.util.spec_from_file_location('model_mod', '{model_file}')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

model_cls = None
for attr_name in dir(mod):
    attr = getattr(mod, attr_name)
    if isinstance(attr, type) and issubclass(attr, torch.nn.Module) and attr is not torch.nn.Module:
        model_cls = attr
        break

if model_cls is None:
    result["error"] = "No nn.Module found"
    print(json.dumps(result))
    sys.exit(0)

try:
    model = model_cls()
except Exception as e:
    result["error"] = f"Instantiation failed: {{e}}"
    print(json.dumps(result))
    sys.exit(0)

model.to(device)
result["total_params"] = sum(p.numel() for p in model.parameters())
result["trainable_params"] = sum(p.numel() for p in model.parameters() if p.requires_grad)

# Hook for shape tracing
layer_shapes = {{}}
def shape_hook(name):
    def hook(module, input, output):
        if isinstance(output, torch.Tensor):
            layer_shapes[name] = list(output.shape)
    return hook

for name, module in model.named_modules():
    if isinstance(module, (nn.Conv2d, nn.Conv3d, nn.Linear, nn.BatchNorm2d)):
        module.register_forward_hook(shape_hook(name))

# Forward pass
x = torch.randn(*shape).to(device)
try:
    with torch.no_grad():
        output = model(x)
    result["forward_ok"] = True
    if isinstance(output, torch.Tensor):
        result["output_shape"] = list(output.shape)
    elif isinstance(output, (tuple, list)) and len(output) > 0 and isinstance(output[0], torch.Tensor):
        result["output_shape"] = list(output[0].shape)
    result["layer_shapes"] = layer_shapes
except Exception as e:
    result["error"] = f"Forward failed: {{e}}"
    print(json.dumps(result))
    sys.exit(0)
{memory_code}

# Backward pass
model.train()
x2 = torch.randn(*shape).to(device)
x2.requires_grad_(True)
try:
    out2 = model(x2)
    if isinstance(out2, torch.Tensor):
        out2.mean().backward()
        result["backward_ok"] = True
    elif isinstance(out2, (tuple, list)) and isinstance(out2[0], torch.Tensor):
        out2[0].mean().backward()
        result["backward_ok"] = True
except Exception as e:
    result["warnings"].append(f"Backward failed: {{str(e)[:100]}}")

result["gpu_memory_peak_mb"] = peak_mb
result["max_safe_batch_size"] = max_batch

print(json.dumps(result))
'''

    def _build_behavior_script(self, model_file: Path, checkpoint_path: str, input_shape: str) -> str:
        """Build subprocess script for Layer 2b internal behavior analysis."""
        model_rel = model_file.relative_to(self.project_dir)
        parent_pkg = str(model_rel.parent).replace("/", ".").replace("\\", ".") if model_rel.parent != Path(".") else ""
        import_stmt = f"from {parent_pkg}.{model_rel.stem} import *" if parent_pkg else f"import {model_rel.stem}"

        default_shape = json.loads(self.default_input_shape)
        shape_code = f'shape = {default_shape}'
        if input_shape:
            try:
                shape = json.loads(input_shape)
                shape_code = f'shape = {shape}'
            except Exception:
                pass

        ckpt_code = ""
        if checkpoint_path:
            ckpt_path = self.project_dir / checkpoint_path
            if ckpt_path.exists():
                ckpt_code = f"""
ckpt = torch.load('{ckpt_path}', map_location='cpu', weights_only=False)
if isinstance(ckpt, dict):
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
    elif 'state_dict' in ckpt:
        model.load_state_dict(ckpt['state_dict'], strict=False)
else:
    model.load_state_dict(ckpt, strict=False)
"""

        return f'''
import sys, json, traceback
sys.path.insert(0, '{self.project_dir}')
import torch
import torch.nn as nn
import numpy as np

{shape_code}
activations = {{}}
gradients = {{}}

def hook_fn(name):
    def forward_hook(module, input, output):
        if isinstance(output, torch.Tensor):
            activations[name] = output.detach()
        elif isinstance(output, (tuple, list)) and len(output) > 0 and isinstance(output[0], torch.Tensor):
            activations[name] = output[0].detach()
    return forward_hook

def grad_hook_fn(name):
    def backward_hook(module, grad_input, grad_output):
        if isinstance(grad_output, (tuple,)) and len(grad_output) > 0 and grad_output[0] is not None:
            gradients[name] = grad_output[0].detach()
    return backward_hook

result = {{}}

try:
    {import_stmt}

    spec = __import__('importlib').util.spec_from_file_location('mod', '{model_file}')
    m = __import__('importlib').util.module_from_spec(spec)
    spec.loader.exec_module(m)

    model_cls = None
    for a in dir(m):
        attr = getattr(m, a)
        if isinstance(attr, type) and issubclass(attr, torch.nn.Module) and attr is not torch.nn.Module:
            model_cls = attr
            break
    if not model_cls:
        result["error"] = "No nn.Module found"
        print(json.dumps(result))
        sys.exit(0)

    model = model_cls()
    {ckpt_code}

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.Conv3d)):
            module.register_forward_hook(hook_fn(name))
            module.register_full_backward_hook(grad_hook_fn(name))

    x = torch.randn(*shape)
    with torch.no_grad():
        output = model(x)

    # Data flow trace
    data_flow = []
    for name, act in activations.items():
        data_flow.append({{"layer": name, "output_shape": list(act.shape)}})
    result["data_flow_trace"] = data_flow

    # Activation stats
    act_stats = []
    for name, act in activations.items():
        flat = act.flatten().float()
        if flat.numel() == 0:
            continue
        s = {{
            "name": name,
            "shape": list(act.shape),
            "mean": round(flat.mean().item(), 6),
            "std": round(flat.std().item(), 6),
            "min": round(flat.min().item(), 6),
            "max": round(flat.max().item(), 6),
            "dead_ratio": round((flat.abs() < 1e-6).float().mean().item(), 4),
        }}
        act_stats.append(s)
    result["activation_stats"] = act_stats

    # Backward pass for gradient analysis
    model.train()
    activations.clear()
    gradients.clear()
    # Hooks are still registered from above — no need to re-register

    x2 = torch.randn(*shape)
    out2 = model(x2)
    if isinstance(out2, torch.Tensor):
        out2.mean().backward()

    grad_stats = []
    for name, grad in gradients.items():
        flat = grad.flatten().float()
        if flat.numel() == 0:
            continue
        g = {{
            "name": name,
            "grad_norm": round(flat.norm().item(), 6),
            "grad_mean": round(flat.mean().item(), 8),
            "grad_std": round(flat.std().item(), 8),
        }}
        grad_stats.append(g)
    result["gradient_stats"] = grad_stats

    # Gradient balance
    if len(grad_stats) >= 2:
        norms = [(g["name"], g["grad_norm"]) for g in grad_stats if g["grad_norm"] > 0]
        if norms:
            mx_name, mx_norm = max(norms, key=lambda x: x[1])
            mn_name, mn_norm = min(norms, key=lambda x: x[1])
            if mx_norm > 0 and mn_norm > 0:
                ratio = mx_norm / mn_norm
                result["gradient_balance"] = {{
                    "strongest_module": mx_name,
                    "strongest_norm": round(mx_norm, 6),
                    "weakest_module": mn_name,
                    "weakest_norm": round(mn_norm, 6),
                    "imbalance_ratio": round(ratio, 1),
                }}

    # Parameter stats
    param_stats = []
    for name, p in model.named_parameters():
        if p.requires_grad:
            flat = p.data.flatten().float()
            if flat.numel() == 0:
                continue
            param_stats.append({{
                "name": name,
                "num_params": p.numel(),
                "std": round(flat.std().item(), 6),
                "mean": round(flat.mean().item(), 6),
            }})
    result["parameter_stats"] = param_stats

except Exception as e:
    result["error"] = traceback.format_exc()[-500:]

print(json.dumps(result))
'''

    def _build_inference_script(self, model_file: Path, checkpoint_path: str,
                                 val_samples: list, gt_dir: str) -> str:
        """Build inference script for Layer 2a reference evaluation."""
        model_rel = model_file.relative_to(self.project_dir)
        parent_pkg = str(model_rel.parent).replace("/", ".").replace("\\", ".") if model_rel.parent != Path(".") else ""
        import_stmt = f"from {parent_pkg}.{model_rel.stem} import *" if parent_pkg else f"import {model_rel.stem}"

        sample_paths = [str(s) for s in val_samples[:10]]

        ckpt_code = ""
        if checkpoint_path:
            ckpt_path = self.project_dir / checkpoint_path
            if ckpt_path.exists():
                ckpt_code = f"""
ckpt = torch.load('{ckpt_path}', map_location='cpu', weights_only=False)
if isinstance(ckpt, dict):
    if 'model_state_dict' in ckpt: model.load_state_dict(ckpt['model_state_dict'], strict=False)
    elif 'state_dict' in ckpt: model.load_state_dict(ckpt['state_dict'], strict=False)
"""

        gt_dir_str = str(self.project_dir / gt_dir) if gt_dir else ""

        return f'''
import sys, json, os
from pathlib import Path
sys.path.insert(0, '{self.project_dir}')
import torch
import numpy as np

result = {{"per_sample": {{}}}}
GT_DIR = '{gt_dir_str}'

def _find_gt_for(sample_path):
    """Try to find GT file corresponding to a sample."""
    sp = Path(sample_path)
    stem = sp.stem
    if not GT_DIR:
        return None
    gt_base = Path(GT_DIR)
    if not gt_base.exists():
        return None
    # Try exact name match with common GT extensions
    for ext in ['.npy', '.pt', '.pth', '.pfm', '.png', '.jpg']:
        candidate = gt_base / (stem + ext)
        if candidate.exists():
            return str(candidate)
    # Try in subdirectories
    for candidate in gt_base.rglob(stem + '.*'):
        if candidate.suffix in ('.npy', '.pt', '.pth', '.pfm', '.png', '.jpg'):
            return str(candidate)
    return None

def _load_array(path):
    """Load an array from various file formats."""
    if path.endswith('.npy'):
        return np.load(path).astype(np.float32)
    elif path.endswith('.pt') or path.endswith('.pth'):
        return torch.load(path, map_location='cpu', weights_only=True).numpy().astype(np.float32)
    elif path.endswith('.pfm'):
        # PFM loader
        with open(path, 'rb') as f:
            header = f.readline().decode().strip()
            dims = f.readline().decode().strip()
            scale = float(f.readline().decode().strip())
            if header == 'PF':
                raise ValueError("Color PFM not supported")
            w, h = map(int, dims.split())
            data = np.frombuffer(f.read(), dtype=np.float32 if scale > 0 else np.float32)
            return data.reshape(h, w).astype(np.float32)
    elif path.endswith('.png') or path.endswith('.jpg'):
        from PIL import Image
        return np.array(Image.open(path)).astype(np.float32) / 255.0
    return None

try:
    {import_stmt}
    spec = __import__('importlib').util.spec_from_file_location('mod', '{model_file}')
    m = __import__('importlib').util.module_from_spec(spec)
    spec.loader.exec_module(m)

    model_cls = None
    for a in dir(m):
        attr = getattr(m, a)
        if isinstance(attr, type) and issubclass(attr, torch.nn.Module) and attr is not torch.nn.Module:
            model_cls = attr
            break
    if not model_cls:
        result["error"] = "No model class"
        print(json.dumps(result))
        sys.exit(0)

    model = model_cls()
    {ckpt_code}
    model.eval()

    # Run on each sample
    for sp in {sample_paths}:
        try:
            data = _load_array(sp)
            if data is None:
                continue
            x = torch.from_numpy(data).float()
            if x.dim() == 2:
                x = x.unsqueeze(0).unsqueeze(0)  # H,W -> 1,1,H,W
            elif x.dim() == 3:
                x = x.unsqueeze(0)  # C,H,W -> 1,C,H,W
            # else already 4D+

            with torch.no_grad():
                out = model(x)
            if isinstance(out, (tuple, list)):
                out = out[0]
            if isinstance(out, dict):
                out = list(out.values())[0]

            sample_result = {{
                "output_mean": float(out.mean()),
                "output_std": float(out.std()),
                "output_min": float(out.min()),
                "output_max": float(out.max()),
            }}

            # Try to compute MAE vs GT
            gt_path = _find_gt_for(sp)
            if gt_path:
                gt = _load_array(gt_path)
                if gt is not None:
                    pred = out.detach().cpu().numpy().squeeze()
                    gt = gt.squeeze()
                    # Resize prediction to GT shape if needed
                    if pred.shape != gt.shape:
                        try:
                            from PIL import Image as PILImage
                            pred_resized = np.array(PILImage.fromarray(pred).resize(
                                (gt.shape[1], gt.shape[0]), PILImage.BILINEAR
                            ))
                            mae = float(np.abs(pred_resized - gt).mean())
                        except Exception:
                            mae = float(np.abs(pred.flatten()[:gt.size] - gt.flatten()[:pred.size]).mean())
                    else:
                        mae = float(np.abs(pred - gt).mean())
                    sample_result["mae"] = round(mae, 6)

            result["per_sample"][sp] = sample_result
        except Exception as e:
            result["per_sample"][sp] = {{"error": str(e)[:100]}}

except Exception as e:
    result["error"] = str(e)[:300]

print(json.dumps(result))
'''
