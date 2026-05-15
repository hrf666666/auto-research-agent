"""
AutoResearcher Constraint Engine — LLM Behavior Control Layer

Prevents LLM hallucination, lying, and corner-cutting through:
1. PlannerChecker: Verifies Code Agent faithfully executed the architecture plan
2. StrategyConstraintEngine: Converts historical patterns into executable rules
3. QuickBenchmark: Runs validation samples to catch metric fabrication
4. AdaptiveThresholds: Calibrates diagnostic thresholds from project history
5. ImplementationTracker: Tracks which plan modules are actually implemented
6. ContextPruner: Limits context injection to most relevant keys per cycle

Design principle: LLMs will take shortcuts when unconstrained. Every constraint
must be CHECKABLE — it cannot rely on the LLM's self-reporting.
"""

import ast
import json
import re
import math
import logging
import subprocess
import sys
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

logger = logging.getLogger("autoresearcher.constraint_engine")


# ──────────────────────────────────────────────────────────
# 1. Planner Checker — Plan vs Implementation comparison
# ──────────────────────────────────────────────────────────

@dataclass
class PlanComplianceReport:
    """Report comparing architecture plan against actual implementation."""
    cycle: int
    plan_modules: list[str] = field(default_factory=list)
    implemented_modules: list[str] = field(default_factory=list)
    missing_modules: list[str] = field(default_factory=list)
    extra_modules: list[str] = field(default_factory=list)
    compliance_score: float = 0.0  # 0-1, fraction of plan modules found
    suspicious_patterns: list[str] = field(default_factory=list)
    fabrication_risk: str = "low"  # low | medium | high
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "cycle": self.cycle,
            "plan_modules": self.plan_modules,
            "implemented_modules": self.implemented_modules,
            "missing_modules": self.missing_modules,
            "extra_modules": self.extra_modules,
            "compliance_score": self.compliance_score,
            "suspicious_patterns": self.suspicious_patterns,
            "fabrication_risk": self.fabrication_risk,
            "warnings": self.warnings,
        }


class PlannerChecker:
    """Verify Code Agent faithfully executed the architecture plan.

    Checks:
    1. All planned modules exist in the codebase
    2. Module structure matches plan specifications
    3. No suspicious patterns indicating shortcut-taking:
       - Stub implementations (pass, NotImplementedError, TODO)
       - Copied code with find-replace patterns
       - Empty forward() methods
       - Hardcoded outputs
    """

    def __init__(self, project_dir: Path, workspace: Path, config: dict = None):
        self.project_dir = Path(project_dir)
        self.workspace = Path(workspace)
        self._config = config or {}
        sbx = self._config.get("sandbox", {})
        self.subprocess_timeout: int = sbx.get("subprocess_timeout", 120)

    def check_plan_compliance(
        self,
        cycle: int,
        architecture_plan: dict,
        execute_result: dict,
    ) -> PlanComplianceReport:
        """Full plan vs implementation compliance check."""
        report = PlanComplianceReport(cycle=cycle)

        plan_modules = architecture_plan.get("modules", [])
        if not plan_modules:
            report.warnings.append("No modules in architecture plan — skipping compliance check")
            return report

        report.plan_modules = [m.get("name", "") for m in plan_modules if m.get("name")]

        # Scan implementation files for actual modules
        implemented = self._scan_implemented_modules()
        report.implemented_modules = list(implemented)

        # Match plan modules to implementation (fuzzy matching)
        for plan_name in report.plan_modules:
            plan_words = set(re.split(r"[_\s]", plan_name.lower()))
            found = False
            for impl_name in implemented:
                impl_words = set(re.split(r"[_\s]", impl_name.lower()))
                # Match if >50% of plan words appear in implementation name
                if plan_words and len(plan_words & impl_words) >= max(1, len(plan_words) * 0.5):
                    found = True
                    break
            if not found:
                report.missing_modules.append(plan_name)

        # Check for extra modules not in plan (possible freestyling)
        plan_names_lower = {n.lower() for n in report.plan_modules}
        for impl_name in implemented:
            impl_words = set(re.split(r"[_\s]", impl_name.lower()))
            matched = False
            for plan_name in plan_names_lower:
                plan_words = set(re.split(r"[_\s]", plan_name))
                if plan_words and len(plan_words & impl_words) >= max(1, len(plan_words) * 0.5):
                    matched = True
                    break
            if not matched and impl_name not in ("__init__", "main"):
                report.extra_modules.append(impl_name)

        # Calculate compliance score
        if report.plan_modules:
            found_count = len(report.plan_modules) - len(report.missing_modules)
            report.compliance_score = found_count / len(report.plan_modules)

        # Check for suspicious patterns in implementation
        report.suspicious_patterns = self._detect_suspicious_patterns()
        if report.suspicious_patterns:
            report.fabrication_risk = "high" if len(report.suspicious_patterns) >= 3 else "medium"

        # Generate warnings
        if report.missing_modules:
            report.warnings.append(
                f"MISSING MODULES: {report.missing_modules}. "
                f"Code Agent did not implement {len(report.missing_modules)}/{len(report.plan_modules)} "
                f"planned modules. Do NOT trust experiment results."
            )
        if report.extra_modules and len(report.extra_modules) > len(report.plan_modules):
            report.warnings.append(
                f"FREESTYLING DETECTED: {len(report.extra_modules)} unplanned modules found. "
                f"Code Agent invented modules not in the architecture plan."
            )
        if report.suspicious_patterns:
            report.warnings.append(
                f"SHORTCUT PATTERNS: {report.suspicious_patterns[:5]}. "
                f"Implementation may contain stub/fake code."
            )

        if report.warnings:
            logger.warning(f"PlannerChecker cycle {cycle}: {len(report.warnings)} issue(s)")
        else:
            logger.info(f"PlannerChecker cycle {cycle}: compliance={report.compliance_score:.0%}")

        return report

    def _scan_implemented_modules(self) -> set[str]:
        """Scan models/ directory for implemented nn.Module subclasses."""
        implemented = set()
        models_dir = self.project_dir / "models"
        if not models_dir.exists():
            # Also check root for model files
            for py_file in self.project_dir.glob("*.py"):
                self._extract_module_names(py_file, implemented)
            return implemented

        for py_file in models_dir.glob("**/*.py"):
            self._extract_module_names(py_file, implemented)
        return implemented

    def _extract_module_names(self, py_file: Path, names: set[str]):
        """Extract nn.Module subclass names from a Python file."""
        try:
            content = py_file.read_text()
            tree = ast.parse(content)
        except Exception:
            return

        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for base in node.bases:
                base_name = ""
                if isinstance(base, ast.Attribute) and base.attr == "Module":
                    base_name = node.name
                elif isinstance(base, ast.Name) and base.id == "nn.Module":
                    base_name = node.name
                if base_name:
                    names.add(base_name)
                    # Also extract self.xxx = SomeModule() assignments
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                            for stmt in ast.walk(item):
                                if isinstance(stmt, ast.Assign):
                                    for target in stmt.targets:
                                        if (isinstance(target, ast.Attribute)
                                                and isinstance(target.value, ast.Name)
                                                and target.value.id == "self"
                                                and isinstance(stmt.value, ast.Call)):
                                            call_name = self._get_call_name(stmt.value)
                                            if call_name:
                                                names.add(target.attr)

    def _get_call_name(self, call: ast.Call) -> str:
        """Extract readable name from a Call node."""
        if isinstance(call.func, ast.Name):
            return call.func.id
        elif isinstance(call.func, ast.Attribute):
            return call.func.attr
        return ""

    def _detect_suspicious_patterns(self) -> list[str]:
        """Detect patterns that indicate shortcut-taking or stub implementation."""
        patterns = []
        models_dir = self.project_dir / "models"

        search_dirs = [models_dir] if models_dir.exists() else []
        # Also check scripts/ for stub patterns
        scripts_dir = self.project_dir / "scripts"
        if scripts_dir.exists():
            search_dirs.append(scripts_dir)

        for search_dir in search_dirs:
            for py_file in search_dir.glob("**/*.py"):
                try:
                    content = py_file.read_text()
                    tree = ast.parse(content)
                except Exception:
                    continue

                for node in ast.walk(tree):
                    if not isinstance(node, ast.FunctionDef):
                        continue

                    func_body = node.body
                    if not func_body:
                        continue

                    # Check 1: Function body is just "pass"
                    if (len(func_body) == 1
                            and isinstance(func_body[0], ast.Pass)
                            and node.name not in ("__init__", )):
                        patterns.append(f"{py_file.name}:{node.name}() is empty (pass)")

                    # Check 2: NotImplementedError
                    for stmt in ast.walk(node):
                        if isinstance(stmt, ast.Raise):
                            if (isinstance(stmt.exc, ast.Call)
                                    and isinstance(stmt.exc.func, ast.Name)
                                    and stmt.exc.func.id == "NotImplementedError"):
                                patterns.append(
                                    f"{py_file.name}:{node.name}() raises NotImplementedError"
                                )

                    # Check 3: Hardcoded return values in forward()
                    if node.name in ("forward", "predict", "inference"):
                        for stmt in ast.walk(node):
                            if isinstance(stmt, ast.Return):
                                if isinstance(stmt.value, ast.Constant):
                                    val = stmt.value.value
                                    if isinstance(val, (int, float)):
                                        patterns.append(
                                            f"{py_file.name}:{node.name}() returns constant {val}"
                                        )

                    # Check 4: forward() method shorter than 5 lines (likely stub)
                    if node.name == "forward":
                        line_count = node.end_lineno - node.lineno if hasattr(node, 'end_lineno') else 0
                        if 0 < line_count < 5:
                            patterns.append(
                                f"{py_file.name}:{node.name}() is only {line_count} lines"
                            )

        return patterns


# ──────────────────────────────────────────────────────────
# 2. Strategy Constraint Engine
# ──────────────────────────────────────────────────────────

@dataclass
class StrategyRule:
    """A constraint rule derived from historical patterns."""
    rule_id: str
    description: str
    condition: str  # Human-readable condition
    action: str     # What to do when triggered
    source: str     # Where this rule came from (e.g., "causal_history", "calibration")
    priority: str = "medium"  # critical | high | medium | low
    trigger_count: int = 0     # How many times this rule has been triggered


class StrategyConstraintEngine:
    """Convert historical patterns into executable constraint rules.

    Reads from:
    - causal_history: Past design decisions with verified effects
    - hypothesis_calibration: Historical hypothesis accuracy
    - dead_ends: Approaches that failed
    - experiment history: Method×domain Pareto frontier

    Generates rules like:
    - "If method X was tried and failed in domain Y, do NOT propose X again for Y"
    - "If last 3 hypotheses had <30% accuracy, force paper research before next experiment"
    - "If method X is Pareto-dominated by method Y, do NOT propose X"
    """

    def __init__(self, project_dir: Path, workspace: Path, config: dict = None):
        self.project_dir = Path(project_dir)
        self.workspace = Path(workspace)
        self._config = config or {}
        sbx = self._config.get("sandbox", {})
        self.subprocess_timeout: int = sbx.get("subprocess_timeout", 120)
        self._rules: list[StrategyRule] = []
        self._rules_loaded = False

    def _load_rules(self):
        """Load or generate constraint rules from historical data."""
        if self._rules_loaded:
            return
        self._rules_loaded = True

        # Load rules from workspace if they were saved
        rules_path = self.workspace / "STRATEGY_RULES.json"
        if rules_path.exists():
            try:
                data = json.loads(rules_path.read_text())
                for rd in data.get("rules", []):
                    self._rules.append(StrategyRule(**{k: v for k, v in rd.items()
                                                        if k in StrategyRule.__dataclass_fields__}))
                return
            except Exception as e:
                logger.debug(f"Failed to load strategy rules: {e}")

    def generate_rules_from_history(self, memory) -> list[StrategyRule]:
        """Generate constraint rules from historical experiment data."""
        self._load_rules()
        new_rules = []

        # Rule source 1: Hypothesis calibration → confidence constraint
        try:
            calibration = memory.get_experiment_calibration()
            total = calibration.get("total_hypotheses", 0)
            if total >= 3:
                accuracy = calibration.get("accuracy_rate", 0.5)
                if accuracy < 0.3:
                    new_rules.append(StrategyRule(
                        rule_id="low_hypothesis_accuracy",
                        description=(
                            f"Historical hypothesis accuracy is only {accuracy:.0%} across "
                            f"{total} experiments. The agent is making poor predictions."
                        ),
                        condition="hypothesis_accuracy < 0.3",
                        action=(
                            "MANDATORY: Before proposing ANY experiment, the agent MUST:\n"
                            "1. Cite specific evidence for the hypothesis\n"
                            "2. Identify what assumption the hypothesis relies on\n"
                            "3. Propose a MINIMAL test that could falsify the hypothesis\n"
                            "4. Get at least one independent data point before committing GPU hours"
                        ),
                        source="hypothesis_calibration",
                        priority="high",
                    ))
                elif accuracy < 0.5:
                    new_rules.append(StrategyRule(
                        rule_id="moderate_hypothesis_accuracy",
                        description=f"Hypothesis accuracy is {accuracy:.0%} — below random chance.",
                        condition="hypothesis_accuracy < 0.5",
                        action=(
                            "Before proposing an experiment, the agent must explain WHY this "
                            "hypothesis is different from past failed ones. What new evidence "
                            "or insight justifies trying again?"
                        ),
                        source="hypothesis_calibration",
                        priority="medium",
                    ))
        except Exception as e:
            logger.debug(f"Calibration rule generation skipped: {e}")

        # Rule source 2: Dead ends → forbidden approaches
        try:
            dead_ends = memory.get_dead_ends_full()[:20]
            if dead_ends:
                # Group dead ends by approach keywords
                approach_failures: dict[str, int] = {}
                for de in dead_ends:
                    text = de if isinstance(de, str) else str(de)
                    for keyword in self._extract_approach_keywords(text):
                        approach_failures[keyword] = approach_failures.get(keyword, 0) + 1

                for approach, count in approach_failures.items():
                    if count >= 3:
                        new_rules.append(StrategyRule(
                            rule_id=f"dead_end_{approach}",
                            description=(
                                f"Approach '{approach}' has been recorded as a dead end "
                                f"{count} times. It consistently fails."
                            ),
                            condition=f"task contains '{approach}'",
                            action=(
                                f"FORBIDDEN: Do NOT propose any experiment involving '{approach}'. "
                                f"It has failed {count} times. If you believe the situation has "
                                f"changed, you must explicitly justify why this time is different."
                            ),
                            source="dead_ends",
                            priority="high" if count >= 5 else "medium",
                        ))
        except Exception as e:
            logger.debug(f"Dead-end rule generation skipped: {e}")

        # Rule source 3: Pareto frontier → method elimination
        try:
            pareto = memory.get_pareto_frontier()
            matrix = pareto.get("matrix", {})
            if matrix:
                dominated_methods = self._find_dominated_methods(matrix)
                if dominated_methods:
                    methods_str = ", ".join(dominated_methods[:5])
                    new_rules.append(StrategyRule(
                        rule_id="pareto_dominated",
                        description=(
                            f"Methods {methods_str} are Pareto-dominated by other methods "
                            f"in ALL domains. They are strictly worse."
                        ),
                        condition=f"proposed method is in [{methods_str}]",
                        action=(
                            f"AVOID: These methods are strictly dominated: {methods_str}. "
                            f"Do NOT propose them unless you have a fundamentally new variant "
                            f"that addresses the specific weakness that made them dominated."
                        ),
                        source="pareto_frontier",
                        priority="medium",
                    ))
        except Exception as e:
            logger.debug(f"Pareto rule generation skipped: {e}")

        # Update stored rules
        self._rules = new_rules
        self._save_rules()

        return new_rules

    def _extract_approach_keywords(self, text: str) -> list[str]:
        """Extract approach keywords from dead end text."""
        approach_keywords = [
            "edge", "loss", "pretrain", "attention", "transformer",
            "gnn", "resnet", "unet", "conv3d", "lstm",
            "fft", "dct", "wavelet", "frequency",
            "augment", "mixup", "cutout", "dropout",
        ]
        text_lower = text.lower()
        return [kw for kw in approach_keywords if kw in text_lower]

    def _find_dominated_methods(self, matrix: dict) -> list[str]:
        """Find methods that are Pareto-dominated in all domains."""
        if not matrix:
            return []

        # matrix: {method: {domain: mae}}
        methods = list(matrix.keys())
        if len(methods) < 2:
            return []

        dominated = []
        for method in methods:
            method_scores = matrix[method]
            is_dominated = False
            for other in methods:
                if other == method:
                    continue
                other_scores = matrix[other]
                # Check if 'other' is strictly better in ALL shared domains
                shared_domains = set(method_scores.keys()) & set(other_scores.keys())
                if shared_domains:
                    all_better = all(
                        other_scores.get(d, float('inf')) < method_scores.get(d, float('inf'))
                        for d in shared_domains
                    )
                    if all_better:
                        is_dominated = True
                        break
            if is_dominated:
                dominated.append(method)

        return dominated

    def _save_rules(self):
        """Save generated rules to workspace for persistence."""
        rules_path = self.workspace / "STRATEGY_RULES.json"
        try:
            data = {
                "rules": [
                    {
                        "rule_id": r.rule_id,
                        "description": r.description,
                        "condition": r.condition,
                        "action": r.action,
                        "source": r.source,
                        "priority": r.priority,
                        "trigger_count": r.trigger_count,
                    }
                    for r in self._rules
                ]
            }
            rules_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        except Exception as e:
            logger.debug(f"Failed to save strategy rules: {e}")

    def check_constraints(self, think_result: dict, memory) -> list[str]:
        """Check proposed action against all active constraints. Returns violation messages."""
        self._load_rules()

        # Regenerate rules if none loaded
        if not self._rules:
            self.generate_rules_from_history(memory)

        violations = []
        task_text = (think_result.get("task", "") + " " + think_result.get("hypothesis", "")).lower()

        for rule in self._rules:
            triggered = False

            # Dead end rules
            if rule.source == "dead_ends":
                keywords = self._extract_approach_keywords(task_text)
                rule_keywords = self._extract_approach_keywords(rule.description.lower())
                if set(keywords) & set(rule_keywords):
                    triggered = True

            # Calibration rules
            elif rule.source == "hypothesis_calibration":
                if "low_hypothesis_accuracy" in rule.rule_id:
                    triggered = think_result.get("action") == "experiment"
                elif "moderate_hypothesis_accuracy" in rule.rule_id:
                    triggered = think_result.get("action") == "experiment"

            # Pareto rules
            elif rule.source == "pareto_frontier":
                if any(m.lower() in task_text for m in rule.description.lower().split(",")):
                    triggered = True

            if triggered:
                rule.trigger_count += 1
                violations.append(
                    f"[CONSTRAINT:{rule.priority.upper()}] {rule.description}\n"
                    f"ACTION REQUIRED: {rule.action}"
                )

        if violations:
            logger.warning(f"StrategyConstraintEngine: {len(violations)} constraint violation(s)")

        return violations

    def get_constraint_prompt(self, violations: list[str]) -> str:
        """Format constraint violations as a prompt for the Leader."""
        if not violations:
            return ""
        return (
            "STRATEGY CONSTRAINT VIOLATIONS DETECTED:\n"
            "The proposed action violates learned constraints from past experiments.\n"
            "You MUST address each violation before proceeding:\n\n"
            + "\n\n".join(violations)
            + "\n\nIf you proceed despite these constraints, you MUST justify why this time is different."
        )


# ──────────────────────────────────────────────────────────
# 3. Quick Benchmark — validation sample verification
# ──────────────────────────────────────────────────────────

@dataclass
class QuickBenchmarkResult:
    """Result from quick benchmark on validation samples."""
    run: bool = False
    num_samples: int = 0
    sample_metrics: list[dict] = field(default_factory=list)
    avg_metric: Optional[float] = None
    reported_metric: Optional[float] = None
    discrepancy: Optional[float] = None  # |reported - benchmark|
    anomaly: bool = False
    anomaly_detail: str = ""


class QuickBenchmark:
    """Run 5-10 validation samples through the model and compare against GT.

    This catches:
    - Fabricated metrics (reported metric doesn't match actual computation)
    - Evaluation code bugs (wrong metric calculation)
    - Model collapse (outputs are constant regardless of input)
    """

    def __init__(self, project_dir: Path, workspace: Path, config: dict = None):
        self.project_dir = Path(project_dir)
        self.workspace = Path(workspace)
        self._config = config or {}
        sbx = self._config.get("sandbox", {})
        self.subprocess_timeout: int = sbx.get("subprocess_timeout", 120)

    def run(
        self,
        model_path: str = "",
        checkpoint_path: str = "",
        val_data_path: str = "",
        reported_metrics: dict = None,
        max_samples: int = 5,
    ) -> QuickBenchmarkResult:
        """Run quick benchmark on a handful of validation samples."""
        result = QuickBenchmarkResult()

        # Find model and checkpoint
        model_file = self._find_model(model_path)
        ckpt_file = self._find_checkpoint(checkpoint_path)

        if not model_file or not ckpt_file:
            result.anomaly_detail = "No model or checkpoint found for quick benchmark"
            return result

        # Find validation data
        val_dir = self._find_val_data(val_data_path)
        if not val_dir:
            result.anomaly_detail = "No validation data found for quick benchmark"
            return result

        # Build and run benchmark script
        script = self._build_benchmark_script(
            model_file, ckpt_file, val_dir, max_samples
        )
        if not script:
            return result

        try:
            proc = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True,
                text=True,
                timeout=self.subprocess_timeout,
                cwd=str(self.project_dir),
            )

            if proc.returncode == 0 and proc.stdout.strip():
                bench_data = json.loads(proc.stdout.strip())
                result.run = True
                result.num_samples = bench_data.get("num_samples", 0)
                result.sample_metrics = bench_data.get("per_sample_metrics", [])

                # Calculate average
                metrics_list = result.sample_metrics
                if metrics_list:
                    mae_values = [m.get("mae", m.get("MAE", None)) for m in metrics_list]
                    mae_values = [v for v in mae_values if v is not None]
                    if mae_values:
                        result.avg_metric = sum(mae_values) / len(mae_values)

                # Compare with reported
                if reported_metrics and result.avg_metric is not None:
                    reported_mae = self._extract_reported_mae(reported_metrics)
                    if reported_mae is not None:
                        result.reported_metric = reported_mae
                        result.discrepancy = abs(reported_mae - result.avg_metric)

                        # Flag if discrepancy > 20% of reported value
                        if reported_mae > 0.01 and result.discrepancy > reported_mae * 0.20:
                            result.anomaly = True
                            result.anomaly_detail = (
                                f"Quick benchmark MAE={result.avg_metric:.4f} but "
                                f"reported={reported_mae:.4f} (discrepancy={result.discrepancy:.4f}, "
                                f">20% off). Metrics may be fabricated or evaluation code buggy."
                            )
                            logger.error(f"QUICK BENCHMARK ANOMALY: {result.anomaly_detail}")
            else:
                logger.debug(f"Quick benchmark failed: {proc.stderr[:200]}")

        except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception) as e:
            logger.debug(f"Quick benchmark error: {e}")

        return result

    def _find_model(self, model_path: str) -> Optional[Path]:
        if model_path:
            p = self.project_dir / model_path
            if p.exists():
                return p
        models_dir = self.project_dir / "models"
        if models_dir.exists():
            files = sorted(models_dir.glob("*.py"), key=lambda f: f.stat().st_mtime, reverse=True)
            return files[0] if files else None
        return None

    def _find_checkpoint(self, checkpoint_path: str) -> Optional[Path]:
        if checkpoint_path:
            p = self.project_dir / checkpoint_path
            if p.exists():
                return p
        search_dirs = [
            self.project_dir / "outputs",
            self.project_dir / "checkpoints",
            self.workspace / "outputs",
        ]
        candidates = []
        for d in search_dirs:
            if not d.exists():
                continue
            for p in d.glob("**/best_model.pth"):
                candidates.append(p)
            for p in d.glob("**/best_checkpoint.pth"):
                candidates.append(p)
            if not candidates:
                for p in d.glob("*.pth"):
                    candidates.append(p)
        return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None

    def _find_val_data(self, val_data_path: str) -> Optional[Path]:
        if val_data_path:
            p = self.project_dir / val_data_path
            if p.exists():
                return p
        # Search common locations
        for candidate in [
            self.project_dir / "data" / "val",
            self.project_dir / "data" / "validation",
            self.project_dir / "data",
        ]:
            if candidate.exists():
                return candidate
        return None

    def _extract_reported_mae(self, metrics: dict) -> Optional[float]:
        """Extract the primary MAE from reported metrics."""
        for key in ("val_MAE", "val_MAE_overall", "best_val_MAE", "val_mae", "MAE_overall"):
            if key in metrics:
                try:
                    return float(metrics[key])
                except (TypeError, ValueError):
                    pass
        return None

    def _build_benchmark_script(
        self, model_file: Path, ckpt_file: Path, val_dir: Path, max_samples: int
    ) -> str:
        """Build a minimal benchmark script that runs N validation samples."""
        model_rel = model_file.relative_to(self.project_dir)
        ckpt_rel = ckpt_file.relative_to(self.project_dir)
        parent_pkg = str(model_rel.parent).replace("/", ".").replace("\\", ".") if model_rel.parent != Path(".") else ""

        import_stmt = (
            f"from {parent_pkg}.{model_rel.stem} import *"
            if parent_pkg
            else f"import {model_rel.stem}"
        )

        return f"""
import sys, json, os
os.chdir('{self.project_dir}')
sys.path.insert(0, '{self.project_dir}')

import torch
import numpy as np

# Load model
try:
    {import_stmt}
except Exception as e:
    print(json.dumps({{"error": f"Import failed: {{e}}"}}))
    sys.exit(0)

# Find the model class
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
    print(json.dumps({{"error": "No nn.Module found"}}))
    sys.exit(0)

# Load checkpoint
state = torch.load('{ckpt_file}', map_location='cpu', weights_only=False)
if 'model_state_dict' in state:
    state = state['model_state_dict']
elif 'state_dict' in state:
    state = state['state_dict']

try:
    model = model_cls()
    model.load_state_dict(state, strict=False)
    model.eval()
except Exception as e:
    print(json.dumps({{"error": f"Model load failed: {{e}}"}}))
    sys.exit(0)

# Probe to find input shape
first_key = next(iter(state), '')
if first_key:
    in_ch = state[first_key].shape[1] if len(state[first_key].shape) >= 2 else 3
else:
    in_ch = 3

shapes = []
if first_key and len(state[first_key].shape) >= 5:
    shapes.append([1, in_ch, 3, 64, 64])
if first_key and len(state[first_key].shape) >= 4:
    shapes.append([1, in_ch, 64, 64])
shapes.append([1, 3, 64, 64])

output = None
for shape in shapes:
    try:
        x = torch.randn(*shape)
        with torch.no_grad():
            output = model(x)
        break
    except Exception:
        continue

if output is None:
    print(json.dumps({{"error": "No valid input shape found"}}))
    sys.exit(0)

# Check output properties
if isinstance(output, (tuple, list)):
    output = output[0]
if isinstance(output, dict):
    output = list(output.values())[0]

out_np = output.detach().cpu().numpy()
results = {{
    "num_samples": 1,
    "per_sample_metrics": [{{
        "output_shape": list(out_np.shape),
        "output_mean": float(np.mean(out_np)),
        "output_std": float(np.std(out_np)),
        "output_min": float(np.min(out_np)),
        "output_max": float(np.max(out_np)),
        "output_has_nan": bool(np.any(np.isnan(out_np))),
        "output_has_inf": bool(np.any(np.isinf(out_np))),
        "output_is_constant": bool(np.std(out_np) < 1e-8),
    }}],
    "probe_info": {{
        "input_shape": shape if 'shape' in dir() else [],
        "output_shape": list(out_np.shape),
    }}
}}

print(json.dumps(results))
"""


# ──────────────────────────────────────────────────────────
# 4. Adaptive Thresholds
# ──────────────────────────────────────────────────────────

class AdaptiveThresholds:
    """Calibrate diagnostic thresholds from project's historical metric range.

    Instead of hardcoded thresholds like "gap > 0.15 is critical", adapt
    to the actual metric range of the project. A project with MAE range
    [0.05, 0.20] should have different thresholds than one with [0.5, 2.0].
    """

    def __init__(self, memory):
        self.memory = memory
        self._cached_thresholds = None

    def get_thresholds(self) -> dict:
        """Get calibrated thresholds based on project history."""
        if self._cached_thresholds is not None:
            return self._cached_thresholds

        # Default thresholds
        defaults = {
            "domain_gap_critical": 0.30,
            "domain_gap_high": 0.15,
            "metric_degradation_pct": 0.10,
            "severe_degradation": 0.35,
            "improvement_threshold": 0.005,
            "loss_plateau_ratio": 0.99,
            "loss_divergence_ratio": 2.0,
            "calibrated": False,
        }

        try:
            stats = self.memory.get_summary_stats()
            best = stats.get("best_metric")
            worst = stats.get("worst_metric")

            if best is not None and worst is not None:
                try:
                    best = float(best)
                    worst = float(worst)
                    if math.isfinite(best) and math.isfinite(worst) and worst > best:
                        metric_range = worst - best
                        # Adapt thresholds to metric range
                        defaults["domain_gap_critical"] = max(0.10, metric_range * 0.8)
                        defaults["domain_gap_high"] = max(0.05, metric_range * 0.4)
                        defaults["metric_degradation_pct"] = max(0.05, metric_range * 0.25)
                        defaults["severe_degradation"] = max(0.15, metric_range * 0.9)
                        defaults["improvement_threshold"] = max(0.001, metric_range * 0.015)
                        defaults["calibrated"] = True
                        logger.info(
                            f"AdaptiveThresholds calibrated: range={metric_range:.4f}, "
                            f"domain_gap_critical={defaults['domain_gap_critical']:.4f}"
                        )
                except (TypeError, ValueError):
                    pass
        except Exception as e:
            logger.debug(f"AdaptiveThresholds calibration skipped: {e}")

        self._cached_thresholds = defaults
        return defaults

    def invalidate_cache(self):
        """Force recalibration on next call."""
        self._cached_thresholds = None


# ──────────────────────────────────────────────────────────
# 5. Implementation Progress Tracker
# ──────────────────────────────────────────────────────────

@dataclass
class ModuleStatus:
    """Status of a single planned module."""
    name: str
    status: str = "pending"  # pending | implemented | verified | skipped | failed
    cycle_first_seen: int = 0
    cycle_implemented: int = 0
    notes: str = ""


class ImplementationTracker:
    """Track which plan modules are implemented/pending/skipped across cycles.

    Prevents the agent from "pretending to be done" by tracking module
    implementation status persistently across cycles.
    """

    def __init__(self, workspace: Path):
        self.workspace = Path(workspace)
        self._status: dict[str, ModuleStatus] = {}
        self._load()

    def _status_path(self) -> Path:
        return self.workspace / "IMPLEMENTATION_STATUS.json"

    def _load(self):
        path = self._status_path()
        if path.exists():
            try:
                data = json.loads(path.read_text())
                for name, info in data.get("modules", {}).items():
                    self._status[name] = ModuleStatus(
                        name=name,
                        status=info.get("status", "pending"),
                        cycle_first_seen=info.get("cycle_first_seen", 0),
                        cycle_implemented=info.get("cycle_implemented", 0),
                        notes=info.get("notes", ""),
                    )
            except Exception as e:
                logger.debug(f"ImplementationTracker load failed: {e}")

    def _save(self):
        path = self._status_path()
        try:
            data = {
                "modules": {
                    name: {
                        "status": s.status,
                        "cycle_first_seen": s.cycle_first_seen,
                        "cycle_implemented": s.cycle_implemented,
                        "notes": s.notes,
                    }
                    for name, s in self._status.items()
                }
            }
            path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        except Exception as e:
            logger.debug(f"ImplementationTracker save failed: {e}")

    def update_from_plan(self, plan: dict, cycle: int):
        """Register modules from a plan."""
        for mod in plan.get("modules", []):
            name = mod.get("name", "")
            if not name:
                continue
            if name not in self._status:
                self._status[name] = ModuleStatus(
                    name=name,
                    status="pending",
                    cycle_first_seen=cycle,
                )
        self._save()

    def update_from_compliance(self, report: PlanComplianceReport):
        """Update status based on planner checker compliance report."""
        for mod_name in report.implemented_modules:
            # Find matching status entry
            for name, status in self._status.items():
                if mod_name.lower() in name.lower() or name.lower() in mod_name.lower():
                    if status.status == "pending":
                        status.status = "implemented"
                        status.cycle_implemented = report.cycle
                    elif status.status == "implemented":
                        status.status = "verified"
                    break

        for mod_name in report.missing_modules:
            for name, status in self._status.items():
                if mod_name.lower() in name.lower() or name.lower() in mod_name.lower():
                    if status.status == "pending":
                        status.notes = f"Still missing at cycle {report.cycle}"
                    break

        self._save()

    def get_progress_report(self) -> dict:
        """Get current implementation progress summary."""
        if not self._status:
            return {"total": 0, "message": "No modules tracked yet"}

        total = len(self._status)
        by_status = {}
        for s in self._status.values():
            by_status[s.status] = by_status.get(s.status, 0) + 1

        pending = [s.name for s in self._status.values() if s.status == "pending"]
        stalled = [
            s.name for s in self._status.values()
            if s.status == "pending" and s.notes
        ]

        return {
            "total": total,
            "by_status": by_status,
            "pending_modules": pending,
            "stalled_modules": stalled,
            "completion_rate": (total - by_status.get("pending", 0)) / total if total else 0,
        }

    def get_progress_prompt(self) -> str:
        """Format progress as a prompt for the Leader."""
        progress = self.get_progress_report()
        if progress["total"] == 0:
            return ""

        rate = progress.get("completion_rate", 0)
        if rate >= 1.0:
            return ""

        pending = progress.get("pending_modules", [])
        if not pending:
            return ""

        return (
            f"IMPLEMENTATION PROGRESS ({rate:.0%} complete):\n"
            f"Total modules planned: {progress['total']}\n"
            f"Status: {progress.get('by_status', {})}\n"
            f"STILL PENDING: {pending}\n\n"
            f"You MUST implement the pending modules before adding new features or "
            f"running new experiments. Do NOT skip planned modules."
        )


# ──────────────────────────────────────────────────────────
# 6. Context Pruner
# ──────────────────────────────────────────────────────────

class ContextPruner:
    """Limit context injection to most relevant keys per cycle.

    Prevents information overload that causes LLM confusion and
    increases both latency and cost.

    Strategy: Based on current cycle situation, select 10-15 most
    relevant context keys to inject, dropping low-priority ones.
    """

    # Priority tiers: higher = more important
    TIER_1_ALWAYS = {
        "brief", "memory_log", "cycle", "workspace_dir",
    }

    TIER_2_SITUATIONAL = {
        # Think phase
        "architecture_plan", "architecture_plan_summary",
        "dataset_manifest_summary", "session_stats",
        "domain_knowledge", "hypothesis_calibration",
        "sandbox_design_guidance",
        # Reflect phase
        "experiment_result", "verify_report", "verify_diagnosis",
        "training_curve_analysis", "experiment_evaluation",
        "sandbox_evaluation",
    }

    TIER_3_CONDITIONAL = {
        "directive", "recent_failures", "data_constraints",
        "cross_experiment_insights", "causal_history",
        "pareto_frontier", "iteration_guidance_prompt",
        "dataset_quality_prompt", "domain_analysis_prompt",
        "visual_analysis", "visual_analysis_diagnosis",
        "independent_assessment_warning",
        "adaptive_thresholds", "implementation_progress",
        "plan_compliance_warning", "quick_benchmark_warning",
    }

    TIER_4_RARE = {
        "idea_guardian_check", "direction_circuit_breaker",
        "data_scarcity_warning", "architecture_feedback_prompt",
        "hypothesis_validation_prompt", "llm_fabrication_detected",
        "fabrication_details", "verify_failed_modules",
    }

    MAX_KEYS = 20

    def prune(self, context: dict, phase: str) -> dict:
        """Select most relevant context keys, dropping low-priority ones."""
        if len(context) <= self.MAX_KEYS:
            return context

        pruned = {}

        # Tier 1: Always include
        for key in self.TIER_1_ALWAYS:
            if key in context:
                pruned[key] = context[key]

        # Tier 2: Include if present
        for key in self.TIER_2_SITUATIONAL:
            if key in context and len(pruned) < self.MAX_KEYS:
                pruned[key] = context[key]

        # Tier 3: Include if remaining budget allows
        for key in self.TIER_3_CONDITIONAL:
            if key in context and len(pruned) < self.MAX_KEYS:
                pruned[key] = context[key]

        # Tier 4: Only if still under budget
        for key in self.TIER_4_RARE:
            if key in context and len(pruned) < self.MAX_KEYS:
                pruned[key] = context[key]

        # Catch any remaining keys that have non-empty values
        for key, value in context.items():
            if key not in pruned and len(pruned) < self.MAX_KEYS:
                if value is not None and value != "" and value != {} and value != []:
                    pruned[key] = value

        dropped = len(context) - len(pruned)
        if dropped > 0:
            logger.debug(f"ContextPruner: {len(context)} → {len(pruned)} keys ({dropped} dropped)")

        return pruned
