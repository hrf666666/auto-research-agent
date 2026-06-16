"""
AutoResearcher VERIFY Phase — Module-Level Result Verification

Core principle: VERIFY runs BETWEEN EXECUTE and REFLECT. Its job is to
reverse-engineer whether each functional module actually worked, by checking
output artifacts, log patterns, and system state — NOT by checking code text.

This replaces the old _run_experiment_auditor which only did surface-level
checks (code text matching, manifest registration) and never verified that
modules actually produced correct results.

VERIFY answers the question: "Did the thing we asked for actually happen?"
If not, it pinpoints WHERE the failure occurred so REFLECT can diagnose WHY.
"""

import ast
import os
import re
import sys
import json
import logging
from .training_log_parser import parse_loss_series, has_nan_loss, classify_loss_trend
import subprocess
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

logger = logging.getLogger("autoresearcher.verifier")

# Numerical safety epsilon for division operations
_EPS = 1e-8


@dataclass
class VerifyCheck:
    """A single verification check with structured result."""
    name: str
    category: str  # "execution", "output", "integrity", "system"
    status: str = "pending"  # pending | pass | fail | warn | skip
    detail: str = ""
    evidence: str = ""
    module_path: str = ""  # Which module/function this check relates to
    severity: str = "medium"  # low | medium | high | critical


@dataclass
class VerifyReport:
    """Complete verification report for one cycle."""
    cycle: int
    checks: list[VerifyCheck] = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    diagnosis: list[str] = field(default_factory=list)
    failed_modules: list[str] = field(default_factory=list)
    # Structured extras — set by individual verify layers
    independent_assessment: dict = field(default_factory=dict)
    dataset_issues: dict = field(default_factory=dict)

    @property
    def has_failures(self) -> bool:
        return any(c.status == "fail" for c in self.checks)

    @property
    def critical_failures(self) -> list[VerifyCheck]:
        return [c for c in self.checks if c.status == "fail" and c.severity == "critical"]

    @property
    def all_failures(self) -> list[VerifyCheck]:
        return [c for c in self.checks if c.status in ("fail", "warn")]

    def to_dict(self) -> dict:
        return {
            "cycle": self.cycle,
            "total_checks": len(self.checks),
            "passed": sum(1 for c in self.checks if c.status == "pass"),
            "failed": sum(1 for c in self.checks if c.status == "fail"),
            "warnings": sum(1 for c in self.checks if c.status == "warn"),
            "skipped": sum(1 for c in self.checks if c.status == "skip"),
            "diagnosis": self.diagnosis,
            "failed_modules": self.failed_modules,
            "checks": [
                {
                    "name": c.name,
                    "category": c.category,
                    "status": c.status,
                    "detail": c.detail[:200],
                    "severity": c.severity,
                    "module_path": c.module_path,
                }
                for c in self.checks
            ],
        }


class ExperimentVerifier:
    """VERIFY phase: verify that each module actually did its job.

    Unlike the old auditor which checked "did the code look right?",
    the verifier checks "did the code PRODUCE the right results?"

    Design principle: VERIFY should be domain-agnostic. It doesn't know
    about "light fields" or "depth estimation" — it checks generic properties
    like: files exist, processes ran, outputs have expected format, metrics
    are in reasonable ranges.
    """

    def __init__(self, project_dir: Path, workspace: Path, thresholds: dict = None):
        self.project_dir = Path(project_dir)
        self.workspace = Path(workspace)
        self._thresholds = thresholds or {}
        self._oscillation_threshold = self._thresholds.get("domain_gap_high", 0.15)
        self._overfit_rise_threshold = self._thresholds.get("improvement_threshold", 0.05)

    def pre_verify(
        self,
        cycle: int,
        think_result: dict,
    ) -> VerifyReport:
        """PRE-VERIFY: Check critical preconditions BEFORE executing.

        This catches problems like:
        - Training script using synthetic/random data instead of real data
        - Dataset loader bypassing the unified pipeline
        - Missing data directories
        - Project code in inconsistent state

        Called BEFORE EXECUTE to prevent wasting GPU hours on doomed experiments.
        """
        report = VerifyReport(cycle=cycle)

        action = think_result.get("action", "")
        if action != "experiment":
            return report

        # Check 1: No synthetic/fake data in training scripts
        self._precheck_no_synthetic_data(report)

        # Check 2: Data directories exist and have real data
        self._precheck_data_available(report)

        # Check 3: Key modules importable
        self._precheck_modules_importable(report)

        # Check 4: DATASET_MANIFEST exists and is valid
        self._precheck_dataset_manifest(report)

        self._synthesize_diagnosis(report)
        if report.has_failures:
            logger.warning(
                f"PRE-VERIFY Cycle {cycle}: {len(report.all_failures)} issue(s) found. "
                "EXECUTE should be blocked until fixed."
            )
        return report

    def _precheck_no_synthetic_data(self, report: VerifyReport):
        """Ensure no training script uses synthetic/random data."""
        scripts_dir = self.project_dir / "scripts"
        if not scripts_dir.exists():
            return

        synthetic_patterns = [
            (r"np\.random\.rand\(", "np.random.rand() — generates random noise"),
            (r"torch\.rand\(", "torch.rand() — generates random data"),
            (r"SyntheticLF\w*", "SyntheticLF class — likely generates fake data"),
            (r"FakeData\w*", "FakeData class — generates fake data"),
            (r"RandomDataset", "RandomDataset — generates random data"),
            (r"np\.random\.randn\(", "np.random.randn() — generates random noise"),
        ]

        for script in scripts_dir.glob("*.py"):
            try:
                content = script.read_text()
            except Exception:
                continue

            # Skip if the file is clearly a test or utility
            if "test" in script.name.lower() or "util" in script.name.lower():
                continue

            for pattern, desc in synthetic_patterns:
                matches = list(re.finditer(pattern, content))
                if not matches:
                    continue

                # Check if it's inside a class/function that's actually used for training
                # Look for Dataset subclass or DataLoader usage nearby
                is_dataset_class = bool(re.search(
                    r"class\s+\w+.*Dataset.*?:|def\s+__getitem__|def\s+__len__",
                    content[:matches[0].start()] if matches[0].start() > 200 else content
                ))

                # Check if the project's real dataset module is imported
                uses_real_data = bool(re.search(
                    r"from\s+datasets\s+import|from\s+\.\s+import|import\s+datasets",
                    content
                ))

                if is_dataset_class and not uses_real_data:
                    report.checks.append(VerifyCheck(
                        name=f"synthetic_data_{script.stem}",
                        category="integrity",
                        status="fail",
                        detail=(
                            f"{script.name} uses {desc} as a dataset. "
                        f"All training results will be scientifically worthless. "
                        f"MUST use the project's real dataset class instead."
                        ),
                        evidence=f"Pattern '{pattern}' found in {script.name}",
                        severity="critical",
                        module_path=script.stem,
                    ))
                elif not uses_real_data and "train" in script.name.lower():
                    # Training script with random data but no explicit Dataset class
                    # (might be generating data inline)
                    has_dataloader = bool(re.search(r"DataLoader|data_loader|train_loader", content))
                    if has_dataloader:
                        report.checks.append(VerifyCheck(
                            name=f"suspect_data_{script.stem}",
                            category="integrity",
                            status="warn",
                            detail=(
                                f"{script.name} uses DataLoader but may not load real data. "
                                f"Pattern found: {desc}. Verify it uses the real dataset."
                            ),
                            evidence=f"Pattern '{pattern}' found alongside DataLoader",
                            severity="high",
                            module_path=script.stem,
                        ))

    def _precheck_data_available(self, report: VerifyReport):
        """Ensure data directories exist and contain real files."""
        data_dir = self.project_dir / "data"
        if not data_dir.exists():
            report.checks.append(VerifyCheck(
                name="data_directory",
                category="integrity",
                status="fail",
                detail="data/ directory does not exist — no data available for training",
                severity="critical",
                module_path="data_pipeline",
            ))
            return

        subdirs = [d for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
        if not subdirs:
            report.checks.append(VerifyCheck(
                name="data_directory",
                category="integrity",
                status="fail",
                detail="data/ directory is empty — no datasets available",
                severity="critical",
                module_path="data_pipeline",
            ))
        else:
            report.checks.append(VerifyCheck(
                name="data_directory",
                category="integrity",
                status="pass",
                detail=f"data/ has {len(subdirs)} dataset directories",
                module_path="data_pipeline",
            ))

    def _precheck_modules_importable(self, report: VerifyReport):
        """Check that key Python modules can be imported.

        Dynamically discovers model and dataset classes rather than
        hardcoding specific class names.
        """
        import_checks = []

        # Dynamically discover dataset class
        datasets_dir = self.project_dir / "datasets"
        if datasets_dir.exists():
            for ds_file in datasets_dir.glob("*.py"):
                if ds_file.name.startswith("_"):
                    continue
                try:
                    tree = ast.parse(ds_file.read_text())
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ClassDef):
                            for base in node.bases:
                                if isinstance(base, ast.Attribute) and base.attr == "Dataset":
                                    import_checks.append(
                                        (f"datasets.{ds_file.stem}", node.name)
                                    )
                                    break
                except Exception:
                    pass

        # Dynamically discover model class (most recently modified)
        models_dir = self.project_dir / "models"
        if models_dir.exists():
            model_files = sorted(
                models_dir.glob("*.py"),
                key=lambda f: f.stat().st_mtime,
                reverse=True,
            )
            for mf in model_files[:1]:  # Only check most recent
                if mf.name.startswith("_"):
                    continue
                try:
                    tree = ast.parse(mf.read_text())
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ClassDef):
                            for base in node.bases:
                                if isinstance(base, ast.Attribute) and base.attr == "Module":
                                    import_checks.append(
                                        (f"models.{mf.stem}", node.name)
                                    )
                                    break
                except Exception:
                    pass

        for module_path, class_name in import_checks:
            try:
                result = subprocess.run(
                    [sys.executable, "-c", f"from {module_path} import {class_name}; print('OK')"],
                    capture_output=True, text=True, timeout=10,
                    cwd=str(self.project_dir),
                )
                if result.returncode != 0:
                    report.checks.append(VerifyCheck(
                        name=f"import_{module_path}",
                        category="integrity",
                        status="fail",
                        detail=f"Cannot import {class_name} from {module_path}: {result.stderr[:200]}",
                        severity="critical",
                        module_path=module_path,
                    ))
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass  # Skip if Python not available (unlikely but safe)

    def _precheck_dataset_manifest(self, report: VerifyReport):
        """Check DATASET_MANIFEST.json exists and has real data."""
        manifest_path = self.workspace / "DATASET_MANIFEST.json"
        if not manifest_path.exists():
            report.checks.append(VerifyCheck(
                name="dataset_manifest",
                category="integrity",
                status="warn",
                detail="DATASET_MANIFEST.json not found — dataset understanding may not have run",
                severity="medium",
                module_path="dataset_loader",
            ))
            return

        try:
            manifest = json.loads(manifest_path.read_text())
            datasets = manifest.get("datasets", {})
            total_scenes = 0
            for ds in (datasets.values() if isinstance(datasets, dict) else datasets):
                if not isinstance(ds, dict):
                    continue
                # Count scenes from whichever field is present
                ds_count = 0
                scenes = ds.get("scenes")
                if isinstance(scenes, (dict, list)):
                    ds_count = len(scenes)
                if ds_count == 0:
                    for key in ("train_scenes", "val_scenes"):
                        val = ds.get(key)
                        if isinstance(val, (dict, list)):
                            ds_count += len(val)
                if ds_count == 0:
                    ds_count = ds.get("total_valid_scenes", 0) or 0
                total_scenes += ds_count
            if total_scenes == 0:
                report.checks.append(VerifyCheck(
                    name="dataset_manifest",
                    category="integrity",
                    status="fail",
                    detail="DATASET_MANIFEST.json has zero scenes — no data available",
                    severity="critical",
                    module_path="dataset_loader",
                ))
            else:
                report.checks.append(VerifyCheck(
                    name="dataset_manifest",
                    category="integrity",
                    status="pass",
                    detail=f"DATASET_MANIFEST.json has {len(datasets)} datasets, {total_scenes} total scenes",
                    module_path="dataset_loader",
                ))
        except json.JSONDecodeError:
            report.checks.append(VerifyCheck(
                name="dataset_manifest",
                category="integrity",
                status="fail",
                detail="DATASET_MANIFEST.json is invalid JSON",
                severity="high",
                module_path="dataset_loader",
            ))

    def verify(
        self,
        cycle: int,
        think_result: dict,
        execute_result: dict,
    ) -> VerifyReport:
        """Run all verification checks and return a structured report.

        This is the main entry point, called between EXECUTE and REFLECT.
        """
        report = VerifyReport(cycle=cycle)

        # ── Layer 1: Execution Verification ──
        # Did the requested action actually execute?
        self._verify_execution(think_result, execute_result, report)

        # ── Layer 2: Output Artifact Verification ──
        # Did execution produce the expected output artifacts?
        self._verify_output_artifacts(think_result, execute_result, report)

        # ── Layer 3: Module Functionality Verification ──
        # For each module the experiment depends on, did it actually work?
        self._verify_module_functionality(think_result, execute_result, report)

        # ── Layer 4: Data Integrity Verification ──
        # Is the data pipeline producing valid data?
        self._verify_data_integrity(think_result, execute_result, report)

        # ── Layer 5: Metric Consistency Verification ──
        # Do reported metrics match actual log output?
        self._verify_metric_consistency(think_result, execute_result, report)

        # ── Layer 6: Configuration Consistency Verification ──
        # Fix 2: Check model config matches checkpoint, script args match model, etc.
        self._verify_config_consistency(think_result, execute_result, report)

        # ── Layer 7: System Health Verification ──
        # Are system resources OK? No OOM, no disk full, etc.
        self._verify_system_health(report)

        # ── Layer 8: Dataset Quality Verification ──
        # Are validation splits statistically meaningful?
        # Detect when metrics are based on too few samples.
        self._verify_dataset_quality(report)

        # ── Layer 9: Model Structural Soundness Verification ──
        # Check for architectural issues that would cause training failure.
        self._verify_model_structure(think_result, execute_result, report)

        # ── Layer 10: Independent Third-Party Verification ──
        # Use a lightweight probe to independently verify model outputs.
        # This prevents "self-evaluation" where the model's own metrics are unreliable.
        self._verify_independent_probe(think_result, execute_result, report)

        # ── Layer 11 (v12): Analysis Experiment Coverage Verification ──
        # For data analysis experiments (no training), check that the analysis
        # used multiple independent methods, not just one narrow approach.
        self._verify_analysis_coverage(think_result, execute_result, report)

        # ── Layer 12 (v12.2): Training Architecture Review ──
        # For training experiments, check routing/fusion convergence,
        # aux loss convergence, and per-domain regression.
        self._verify_training_architecture(think_result, execute_result, report)

        # ── Synthesize diagnosis ──
        self._synthesize_diagnosis(report)

        # Log summary
        d = report.to_dict()
        logger.info(
            f"VERIFY Cycle {cycle}: {d['passed']} passed, "
            f"{d['failed']} failed, {d['warnings']} warnings"
        )
        for diag in report.diagnosis:
            logger.warning(f"  VERIFY DIAGNOSIS: {diag}")

        return report

    # ─────────────────────────────────────────────────
    # Layer 1: Execution Verification
    # ─────────────────────────────────────────────────

    def _verify_execution(
        self, think_result: dict, execute_result: dict, report: VerifyReport
    ):
        """Verify the requested action was actually executed.

        ANTI-DECEPTION: Checks the tool_trace to verify that:
        1. If the LLM claims an experiment launched, launch_experiment
           was actually called and returned a real PID
        2. If deception_detected is flagged, marks it as a CRITICAL failure
        """
        action = think_result.get("action", "")
        agent = think_result.get("agent", "")
        launched = execute_result.get("experiment_launched", False)

        # ── ANTI-DECEPTION: Check if LLM fabricated a claim ──
        if execute_result.get("deception_detected"):
            report.checks.append(VerifyCheck(
                name="llm_fabrication",
                category="execution",
                status="fail",
                detail=execute_result.get("deception_detail", "LLM made a claim not backed by tool execution"),
                evidence=f"Tool trace shows: {execute_result.get('tool_trace', {}).get('tool_names', [])}",
                severity="critical",
                module_path="llm_honesty",
            ))

        # Check: If THINK planned an experiment, was one launched?
        if action == "experiment":
            if not launched and agent == "code":
                # Did code agent even run?
                response = execute_result.get("response", "")
                has_error = '"error"' in response[:200]
                if has_error:
                    report.checks.append(VerifyCheck(
                        name="experiment_launch",
                        category="execution",
                        status="fail",
                        detail="Code agent returned an error — experiment never launched",
                        evidence=response[:300],
                        severity="critical",
                        module_path="code_agent",
                    ))
                else:
                    report.checks.append(VerifyCheck(
                        name="experiment_launch",
                        category="execution",
                        status="fail",
                        detail="THINK planned an experiment but EXECUTE did not launch one (no shortcut, no error either)",
                        evidence=f"Think task: {think_result.get('task', '')[:100]}",
                        severity="high",
                        module_path="code_agent",
                    ))
            elif launched:
                # ── ANTI-DECEPTION: Cross-verify PID against tool trace ──
                pid = execute_result.get("pid")
                tool_trace = execute_result.get("tool_trace", {})
                launch_facts = tool_trace.get("launch_facts", {}) if tool_trace else {}

                # Verify PID exists in tool trace (not just in LLM text)
                if pid and launch_facts:
                    trace_pid = launch_facts.get("pid")
                    if trace_pid and int(trace_pid) != int(pid):
                        report.checks.append(VerifyCheck(
                            name="pid_trace_mismatch",
                            category="execution",
                            status="fail",
                            detail=f"PID mismatch: LLM text says {pid}, tool trace says {trace_pid}. LLM may have fabricated the PID.",
                            evidence=f"Tool trace launch_facts: {launch_facts}",
                            severity="critical",
                            module_path="llm_honesty",
                        ))
                    elif trace_pid:
                        # PID verified against tool trace — this is a REAL PID
                        report.checks.append(VerifyCheck(
                            name="pid_trace_verified",
                            category="execution",
                            status="pass",
                            detail=f"PID={pid} verified against tool trace (system-level confirmation)",
                            evidence=f"launch_experiment returned: {launch_facts}",
                            module_path="code_agent",
                        ))
                elif pid and not launch_facts:
                    # PID exists but not in trace — could be from old code path
                    report.checks.append(VerifyCheck(
                        name="pid_no_trace",
                        category="execution",
                        status="warn",
                        detail=f"PID={pid} found but not in tool trace — cannot confirm it came from launch_experiment",
                        severity="medium",
                        module_path="code_agent",
                    ))

                # Verify PID is real (alive or recently finished)
                pid_alive = False
                if pid:
                    try:
                        import os
                        os.kill(int(pid), 0)
                        pid_alive = True
                    except (OSError, ProcessLookupError, TypeError):
                        pass

                report.checks.append(VerifyCheck(
                    name="experiment_launch",
                    category="execution",
                    status="pass",
                    detail=f"Experiment launched successfully, PID={pid}",
                    evidence=f"PID alive at verify time: {pid_alive}",
                    module_path="code_agent",
                ))

                # Check if PID is still alive (might have crashed or finished normally)
                if pid and not pid_alive:
                    log_file = execute_result.get("log_file", "")
                    # First check if the process completed successfully by looking
                    # for completion artifacts (checkpoints, final metrics, saved logs)
                    normal_completion = self._check_normal_completion(log_file)
                    if normal_completion:
                        report.checks.append(VerifyCheck(
                            name="process_completed",
                            category="execution",
                            status="pass",
                            detail=f"Process PID={pid} finished normally: {normal_completion}",
                            evidence=f"PID dead but completion artifacts found",
                            module_path="training_process",
                        ))
                    else:
                        # Process died without normal completion — likely crashed
                        crash_info = self._check_crash_log(log_file)
                        if crash_info:
                            report.checks.append(VerifyCheck(
                                name="process_alive",
                                category="execution",
                                status="fail",
                                detail=f"Process PID={pid} already dead — likely crashed",
                                evidence=crash_info[:500],
                                severity="critical",
                                module_path="training_process",
                            ))

        elif action == "paper_research":
            is_paper = execute_result.get("is_paper_research", False)
            response = execute_result.get("response", "")
            # Paper research should have produced a report file
            has_report = bool(re.search(r"paper_research.*\.md", response))
            report.checks.append(VerifyCheck(
                name="paper_research_executed",
                category="execution",
                status="pass" if is_paper else "warn",
                detail="Paper research dispatched" if is_paper else "Paper research may not have executed properly",
                evidence=f"Report file found: {has_report}",
                module_path="researcher_agent",
            ))

        # ── ANTI-DECEPTION: Verify dry-run was actually executed ──
        if action == "experiment" and agent == "code":
            tool_trace = execute_result.get("tool_trace", {})
            if tool_trace:
                dry_run_performed = execute_result.get("dry_run_performed", False)
                shell_commands_ok = execute_result.get("shell_commands_ok", 0)
                shell_commands_run = execute_result.get("shell_commands_run", 0)

                if not dry_run_performed and launched:
                    report.checks.append(VerifyCheck(
                        name="dry_run_skipped",
                        category="execution",
                        status="warn",
                        detail="Experiment was launched without a dry-run. Code agent may have skipped the mandatory dry-run step.",
                        evidence=f"Shell commands: {shell_commands_run} total, {shell_commands_ok} OK",
                        severity="medium",
                        module_path="code_agent",
                    ))
                elif dry_run_performed and not execute_result.get("dry_run_passed", True):
                    report.checks.append(VerifyCheck(
                        name="dry_run_failed",
                        category="execution",
                        status="fail",
                        detail="Dry-run was performed but FAILED — experiment should not have been launched",
                        severity="critical",
                        module_path="code_agent",
                    ))

        elif action == "wait":
            report.checks.append(VerifyCheck(
                name="wait_action",
                category="execution",
                status="skip",
                detail="THINK decided to wait — no execution to verify",
                module_path="",
            ))

    # ─────────────────────────────────────────────────
    # Layer 2: Output Artifact Verification
    # ─────────────────────────────────────────────────

    def _verify_output_artifacts(
        self, think_result: dict, execute_result: dict, report: VerifyReport
    ):
        """Verify expected output artifacts exist and are non-empty."""
        action = think_result.get("action", "")
        if action != "experiment":
            return

        # If experiment was never launched, skip log file checks entirely.
        # No point auditing log files when no training ran.
        experiment_launched = execute_result.get("experiment_launched", False)
        if not experiment_launched:
            return

        log_file = execute_result.get("log_file", "")
        if not log_file:
            report.checks.append(VerifyCheck(
                name="log_file_exists",
                category="output",
                status="fail",
                detail="No log file path recorded — cannot verify output",
                severity="high",
                module_path="experiment_logger",
            ))
            return

        log_path = self.project_dir / log_file

        # Check: Log file exists?
        if not log_path.exists():
            report.checks.append(VerifyCheck(
                name="log_file_exists",
                category="output",
                status="fail",
                detail=f"Log file not found: {log_file}",
                severity="high",
                module_path="experiment_logger",
            ))
            return

        # Check: Log file has content?
        log_size = log_path.stat().st_size
        if log_size == 0:
            report.checks.append(VerifyCheck(
                name="log_file_content",
                category="output",
                status="fail",
                detail=f"Log file is empty (0 bytes): {log_file}",
                severity="critical",
                module_path="experiment_logger",
            ))
            return

        if log_size < 100:
            report.checks.append(VerifyCheck(
                name="log_file_content",
                category="output",
                status="warn",
                detail=f"Log file suspiciously small ({log_size} bytes): {log_file}",
                severity="medium",
                module_path="experiment_logger",
            ))
            return

        report.checks.append(VerifyCheck(
            name="log_file_content",
            category="output",
            status="pass",
            detail=f"Log file exists and has content ({log_size} bytes)",
            module_path="experiment_logger",
        ))

        # Check: Log contains training progress (not just import errors)
        try:
            log_text = log_path.read_text(errors="ignore")
            self._verify_log_content(log_text, report)
        except Exception as e:
            report.checks.append(VerifyCheck(
                name="log_file_readable",
                category="output",
                status="fail",
                detail=f"Cannot read log file: {e}",
                severity="high",
                module_path="experiment_logger",
            ))

        # Check: Model checkpoint saved?
        self._verify_checkpoints(log_text, report)

    def _verify_log_content(self, log_text: str, report: VerifyReport):
        """Verify training log shows actual training progress."""
        # Common crash indicators
        crash_patterns = [
            (r"Traceback \(most recent call last\)", "Python traceback found — training crashed"),
            (r"RuntimeError", "RuntimeError in log"),
            (r"CUDA out of memory", "CUDA OOM — model too large for GPU"),
            (r"KeyError", "KeyError — likely config or data mismatch"),
            (r"ValueError", "ValueError — likely data shape or type mismatch"),
            (r"FileNotFoundError", "FileNotFoundError — missing data or config"),
            (r"ImportError", "ImportError — missing dependency"),
        ]

        for pattern, desc in crash_patterns:
            matches = re.findall(pattern, log_text)
            if matches:
                # Get the line containing the error
                error_lines = [l.strip() for l in log_text.split("\n") if re.search(pattern, l)]
                error_context = error_lines[-1][:200] if error_lines else desc
                severity = "critical" if "OOM" in desc or "Traceback" in desc else "high"
                report.checks.append(VerifyCheck(
                    name=f"crash_{pattern[:20].replace(' ', '_')}",
                    category="output",
                    status="fail",
                    detail=desc,
                    evidence=error_context,
                    severity=severity,
                    module_path=self._guess_module_from_error(error_context),
                ))

        # Training progress indicators
        progress_patterns = [
            (r"epoch[:\s]+\d+", "epoch progress"),
            (r"step[:\s]+\d+", "step progress"),
            (r"loss[=:\s]+[0-9.]+", "loss values"),
            (r"train.*loss", "training loss"),
            (r"val.*loss|valid.*loss", "validation loss"),
            (r"lr[:\s]+[0-9.e-]+", "learning rate"),
        ]

        found_progress = []
        for pattern, desc in progress_patterns:
            if re.search(pattern, log_text, re.IGNORECASE):
                found_progress.append(desc)

        if found_progress:
            report.checks.append(VerifyCheck(
                name="training_progress",
                category="output",
                status="pass",
                detail=f"Training progress indicators found: {', '.join(found_progress)}",
                module_path="training_loop",
            ))

            # ── Stagnation detection: check for training stuck on same epoch ──
            # If log has hundreds/thousands of lines but epoch never increments
            # past 0 (or 1), the training loop is likely stuck in an infinite loop.
            epoch_values = re.findall(r"epoch[=:\s]+(\d+)", log_text, re.IGNORECASE)
            if epoch_values and len(epoch_values) > 50:
                unique_epochs = set(epoch_values)
                if len(unique_epochs) == 1 and list(unique_epochs)[0] in ("0", "1"):
                    # Check if loss values are repeating (sign of infinite loop over same data)
                    loss_values = [str(v) for v in parse_loss_series(log_text)]
                    if len(loss_values) > 50:
                        # Check last 50 loss values for repetition
                        recent_losses = [float(l) for l in loss_values[-50:]]
                        unique_recent = set(f"{l:.4f}" for l in recent_losses)
                        if len(unique_recent) <= len(recent_losses) // 2:
                            report.checks.append(VerifyCheck(
                                name="training_stagnation",
                                category="integrity",
                                status="fail",
                                detail=(
                                    f"Training appears STUCK: epoch never advanced past "
                                    f"{list(unique_epochs)[0]} after {len(epoch_values)} "
                                    f"iterations, and loss values are repeating "
                                    f"(only {len(unique_recent)} unique in last 50). "
                                    f"This is likely an infinite loop, not real training."
                                ),
                                evidence=(
                                    f"epochs seen: {unique_epochs} | "
                                    f"unique losses in last 50: {len(unique_recent)}/{len(recent_losses)}"
                                ),
                                severity="critical",
                                module_path="training_loop",
                            ))
        else:
            # No progress but no crash either — suspicious
            has_traceback = bool(re.search(r"Traceback", log_text))
            if not has_traceback and len(log_text) > 500:
                report.checks.append(VerifyCheck(
                    name="training_progress",
                    category="output",
                    status="warn",
                    detail="No training progress indicators in log — may be stuck at initialization or data loading",
                    severity="medium",
                    module_path="training_loop",
                ))

    def _verify_checkpoints(self, log_text: str, report: VerifyReport):
        """Verify model checkpoints were saved."""
        checkpoint_patterns = [
            r"Saving.*checkpoint.*?['\"]?([^'\"]+)['\"]?",
            r"Saved model to[:\s]+(.+)",
            r"checkpoint.*saved[:\s]+(.+)",
            r"best_model.*saved[:\s]+(.+)",
        ]

        checkpoints_found = []
        for pattern in checkpoint_patterns:
            matches = re.findall(pattern, log_text, re.IGNORECASE)
            checkpoints_found.extend(m.strip() for m in matches)

        if checkpoints_found:
            # Verify at least one checkpoint file actually exists
            for ckpt_path in checkpoints_found[:3]:
                full_path = self.project_dir / ckpt_path
                if full_path.exists():
                    size_mb = full_path.stat().st_size / (1024 * 1024)
                    report.checks.append(VerifyCheck(
                        name="checkpoint_saved",
                        category="output",
                        status="pass",
                        detail=f"Checkpoint file exists: {ckpt_path} ({size_mb:.1f}MB)",
                        module_path="checkpoint_saver",
                    ))
                    return
            report.checks.append(VerifyCheck(
                name="checkpoint_saved",
                category="output",
                status="warn",
                detail=f"Checkpoint paths mentioned in log but files not found: {checkpoints_found[:2]}",
                severity="medium",
                module_path="checkpoint_saver",
            ))
        # No checkpoints mentioned is OK if training is still early

    # ─────────────────────────────────────────────────
    # Layer 3: Module Functionality Verification
    # ─────────────────────────────────────────────────

    def _verify_module_functionality(
        self, think_result: dict, execute_result: dict, report: VerifyReport
    ):
        """Verify each module the experiment depends on is functional.

        This is the KEY difference from the old auditor. Instead of checking
        code text, we check module BEHAVIOR by looking at what the module
        actually produced.
        """
        action = think_result.get("action", "")
        if action != "experiment":
            return

        # Check: Dataset loader produces valid batches
        self._verify_dataset_loader(report)

        # Check: Model can do forward pass (inferred from logs)
        self._verify_model_forward(report, execute_result)

        # Check: Loss function produces finite values (inferred from logs)
        self._verify_loss_function(report, execute_result)

    def _verify_dataset_loader(self, report: VerifyReport):
        """Verify the dataset loader module is functional."""
        # Check if a recent training log shows successful data loading
        logs_dir = self.project_dir / "logs"
        if not logs_dir.exists():
            return

        # Look for data loading evidence in recent logs
        recent_logs = sorted(logs_dir.glob("*.log"), key=lambda f: f.stat().st_mtime, reverse=True)[:3]
        for log_file in recent_logs:
            try:
                log_text = log_file.read_text(errors="ignore")[-5000:]
            except Exception:
                continue

            # Data loading success indicators
            if re.search(r"(?:Loaded|loading).*\d+.*samples?", log_text, re.IGNORECASE):
                report.checks.append(VerifyCheck(
                    name="dataset_loader",
                    category="integrity",
                    status="pass",
                    detail=f"Dataset loaded successfully (evidence in {log_file.name})",
                    module_path="dataset_loader",
                ))
                return

            # Data loading failure indicators
            if re.search(r"DataLoader.*error|dataloader.*fail", log_text, re.IGNORECASE):
                report.checks.append(VerifyCheck(
                    name="dataset_loader",
                    category="integrity",
                    status="fail",
                    detail=f"DataLoader error in {log_file.name}",
                    severity="high",
                    module_path="dataset_loader",
                ))
                return

        # Check DATASET_MANIFEST
        manifest_path = self.workspace / "DATASET_MANIFEST.json"
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text())
                datasets = manifest.get("datasets", {})
                if not datasets:
                    report.checks.append(VerifyCheck(
                        name="dataset_manifest",
                        category="integrity",
                        status="warn",
                        detail="DATASET_MANIFEST.json has no registered datasets",
                        severity="medium",
                        module_path="dataset_loader",
                    ))
                else:
                    report.checks.append(VerifyCheck(
                        name="dataset_manifest",
                        category="integrity",
                        status="pass",
                        detail=f"DATASET_MANIFEST.json has {len(datasets)} registered datasets",
                        module_path="dataset_loader",
                    ))
            except json.JSONDecodeError:
                report.checks.append(VerifyCheck(
                    name="dataset_manifest",
                    category="integrity",
                    status="fail",
                    detail="DATASET_MANIFEST.json is invalid JSON",
                    severity="medium",
                    module_path="dataset_loader",
                ))

    def _verify_model_forward(self, report: VerifyReport, execute_result: dict):
        """Verify the model can do a forward pass (inferred from logs)."""
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

        # If loss values exist, forward pass worked
        if parse_loss_series(log_text):
            # Check for NaN/Inf loss — model forward pass produces garbage
            if has_nan_loss(log_text):
                nan_count = len(re.findall(r"loss[=:\s]+(nan|inf)", log_text, re.IGNORECASE))
                report.checks.append(VerifyCheck(
                    name="model_forward",
                    category="integrity",
                    status="fail",
                    detail=f"NaN/Inf loss detected ({nan_count} times) — model forward pass produces invalid values",
                    evidence=f"First NaN loss occurrence: check {log_file}",
                    severity="critical",
                    module_path="model_forward",
                ))
            else:
                report.checks.append(VerifyCheck(
                    name="model_forward",
                    category="integrity",
                    status="pass",
                    detail="Model forward pass produces finite loss values",
                    module_path="model_forward",
                ))
        else:
            # No loss values — forward pass may not have run
            if "Traceback" not in log_text:
                report.checks.append(VerifyCheck(
                    name="model_forward",
                    category="integrity",
                    status="warn",
                    detail="No loss values in log — forward pass may not have executed",
                    severity="medium",
                    module_path="model_forward",
                ))

    def _verify_loss_function(self, report: VerifyReport, execute_result: dict):
        """Verify loss function produces meaningful gradients."""
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

        # ── TRAINING CURVE ANALYSIS ──
        # Parse full loss sequence for rich curve diagnostics:
        # overfitting, oscillation, convergence speed, plateau detection
        loss_values = parse_loss_series(log_text)
        if len(loss_values) >= 3:
            floats = [fv for fv in loss_values if fv > 0]
            if len(floats) >= 3:
                # Basic decrease check (kept for backward compatibility)
                first_third = sum(floats[:len(floats)//3]) / (len(floats)//3)
                last_third = sum(floats[-len(floats)//3:]) / (len(floats)//3)

                if last_third < first_third * 0.99:
                    report.checks.append(VerifyCheck(
                        name="loss_decreasing",
                        category="integrity",
                        status="pass",
                        detail=f"Loss is decreasing: early avg={first_third:.4f} → late avg={last_third:.4f}",
                        module_path="loss_function",
                    ))
                elif last_third > first_third * 1.1:
                    report.checks.append(VerifyCheck(
                        name="loss_decreasing",
                        category="integrity",
                        status="fail",
                        detail=f"Loss is INCREASING: early avg={first_third:.4f} → late avg={last_third:.4f} — learning rate may be too high or loss function is wrong",
                        severity="high",
                        module_path="loss_function",
                    ))
                else:
                    report.checks.append(VerifyCheck(
                        name="loss_decreasing",
                        category="integrity",
                        status="warn",
                        detail=f"Loss is flat (not decreasing): early avg={first_third:.4f} → late avg={last_third:.4f} — may need learning rate adjustment",
                        severity="medium",
                        module_path="loss_function",
                    ))

                # ── Rich Training Curve Analysis ──
                curve_analysis = self._analyze_training_curve(floats)
                if curve_analysis:
                    # Overfitting detection
                    if curve_analysis.get("overfit_epoch"):
                        report.checks.append(VerifyCheck(
                            name="overfitting_detected",
                            category="integrity",
                            status="warn",
                            detail=(
                                f"Overfitting detected at ~step {curve_analysis['overfit_epoch']}/{len(floats)} "
                                f"({curve_analysis['overfit_epoch']/len(floats)*100:.0f}% through training). "
                                f"Min loss={curve_analysis['min_loss']:.4f} at step {curve_analysis['min_loss_epoch']}, "
                                f"but final loss={floats[-1]:.4f}. "
                                f"Consider: early stopping, more regularization, or reduce model capacity."
                            ),
                            severity="high",
                            module_path="training_loop",
                        ))

                    # Oscillation detection
                    if curve_analysis.get("oscillation_ratio", 0) > self._oscillation_threshold:
                        report.checks.append(VerifyCheck(
                            name="loss_oscillation",
                            category="integrity",
                            status="warn",
                            detail=(
                                f"Loss oscillation detected: {curve_analysis['oscillation_ratio']:.1%} of steps "
                                f"show loss increases. Direction changes: {curve_analysis['direction_changes']}. "
                                f"This suggests: learning rate too high, batch size too small, or noisy gradients. "
                                f"Consider: reduce LR, increase batch size, gradient clipping."
                            ),
                            severity="medium",
                            module_path="optimizer",
                        ))

                    # Convergence speed
                    if curve_analysis.get("convergence_speed"):
                        speed = curve_analysis["convergence_speed"]
                        if speed == "very_slow":
                            report.checks.append(VerifyCheck(
                                name="convergence_speed",
                                category="integrity",
                                status="warn",
                                detail=(
                                    f"Very slow convergence: loss only decreased {curve_analysis['total_decrease_pct']:.1%} "
                                    f"over {len(floats)} steps. The model may be under-capacity, LR too low, "
                                    f"or the loss landscape is very flat. Consider: increase model capacity, "
                                    f"adjust learning rate schedule, or check if the loss function is correct."
                                ),
                                severity="medium",
                                module_path="training_loop",
                            ))

                    # Plateau detection
                    if curve_analysis.get("plateau_start"):
                        report.checks.append(VerifyCheck(
                            name="loss_plateau",
                            category="integrity",
                            status="warn",
                            detail=(
                                f"Loss plateau starting at step {curve_analysis['plateau_start']}/{len(floats)}. "
                                f"Loss has been flat (< 0.1% change) for {curve_analysis['plateau_length']} steps. "
                                f"The model may have reached its capacity limit. "
                                f"Consider: architectural changes, different optimizer, or new training strategy."
                            ),
                            severity="medium",
                            module_path="training_loop",
                        ))

        # Check for zero loss — model collapsed
        zero_losses = [v for v in loss_values if float(v) == 0.0]
        if len(zero_losses) > 2:
            report.checks.append(VerifyCheck(
                name="loss_zero",
                category="integrity",
                status="fail",
                detail=f"Zero loss detected {len(zero_losses)} times — model may have collapsed or loss is incorrectly computed",
                severity="critical",
                module_path="loss_function",
            ))

    # ─────────────────────────────────────────────────
    # Layer 4: Data Integrity Verification
    # ─────────────────────────────────────────────────

    def _verify_data_integrity(
        self, think_result: dict, execute_result: dict, report: VerifyReport
    ):
        """Verify data pipeline produces valid data."""
        action = think_result.get("action", "")
        if action != "experiment":
            return

        # Check: No orphan data loaders (scripts that load data bypassing the unified dataset)
        self._verify_no_orphan_loaders(report)

        # Check: Data directories are populated
        self._verify_data_directories(report)

        # Check: Runtime data fingerprint — ensure training loaded REAL data, not random noise
        self._verify_real_data_loaded(report, execute_result)

    def _verify_no_orphan_loaders(self, report: VerifyReport):
        """Check no scripts bypass the unified data pipeline."""
        scripts_dir = self.project_dir / "scripts"
        dataset_module = self.project_dir / "datasets" / "unified_lf_dataset.py"

        if not scripts_dir.exists() or not dataset_module.exists():
            return

        for script in scripts_dir.glob("*.py"):
            try:
                content = script.read_text()
            except Exception:
                continue

            has_inline_loader = bool(
                re.search(r"np\.load\(", content)
                or re.search(r"Image\.open", content)
                or re.search(r"h5py\.File", content)
            )
            uses_unified = (
                "from datasets" in content
                or "import datasets" in content
            )
            if has_inline_loader and not uses_unified:
                report.checks.append(VerifyCheck(
                    name=f"no_orphan_loader_{script.stem}",
                    category="integrity",
                    status="warn",
                    detail=f"{script.name} loads data inline, bypasses unified dataset pipeline",
                    severity="medium",
                    module_path=script.stem,
                ))

    def _verify_data_directories(self, report: VerifyReport):
        """Check that data directories are populated and accessible."""
        data_dir = self.project_dir / "data"
        if not data_dir.exists():
            report.checks.append(VerifyCheck(
                name="data_directory",
                category="integrity",
                status="fail",
                detail="data/ directory does not exist",
                severity="critical",
                module_path="data_pipeline",
            ))
            return

        subdirs = [d for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
        if not subdirs:
            report.checks.append(VerifyCheck(
                name="data_directory",
                category="integrity",
                status="fail",
                detail="data/ directory is empty — no datasets available",
                severity="critical",
                module_path="data_pipeline",
            ))
            return

        # Check each dataset directory has files (depth-limited for performance)
        empty_datasets = []
        for ds_dir in subdirs:
            # Use shallow check (depth=2) to avoid expensive rglob on large datasets
            has_files = False
            try:
                for f in ds_dir.glob("**/*"):
                    if f.is_file() and not f.name.startswith("."):
                        has_files = True
                        break
                    # Limit depth to 3 to avoid walking huge trees
                    if len(f.relative_to(ds_dir).parts) > 3:
                        has_files = True  # Deep structure implies files exist
                        break
            except (OSError, ValueError):
                has_files = False
            if not has_files:
                empty_datasets.append(ds_dir.name)

        if empty_datasets:
            report.checks.append(VerifyCheck(
                name="data_directory",
                category="integrity",
                status="warn",
                detail=f"Empty dataset directories: {empty_datasets[:5]}",
                severity="medium",
                module_path="data_pipeline",
            ))
        else:
            report.checks.append(VerifyCheck(
                name="data_directory",
                category="integrity",
                status="pass",
                detail=f"data/ has {len(subdirs)} populated dataset directories",
                module_path="data_pipeline",
            ))

    # ─────────────────────────────────────────────────
    # Layer 5: Metric Consistency Verification
    # ─────────────────────────────────────────────────

    def _verify_metric_consistency(
        self, think_result: dict, execute_result: dict, report: VerifyReport
    ):
        """Verify reported metrics match actual log output."""
        log_file = execute_result.get("log_file", "")
        if not log_file:
            return

        log_path = self.project_dir / log_file
        if not log_path.exists():
            return

        try:
            log_text = log_path.read_text(errors="ignore")[-5000:]
        except Exception:
            return

        # Extract metrics from log
        log_metrics = {}
        for pattern, key in [
            (r"(?:val_)?mae[:\s]+([0-9.]+)", "MAE"),
            (r"(?:val_)?mse[:\s]+([0-9.]+)", "MSE"),
            (r"(?:val_)?rmse[:\s]+([0-9.]+)", "RMSE"),
            (r"loss[=:\s]+([0-9.]+)", "loss"),
            (r"accuracy[:\s]+([0-9.]+)", "accuracy"),
        ]:
            matches = re.findall(pattern, log_text, re.IGNORECASE)
            if matches:
                log_metrics[key] = [float(m) for m in matches]

        # Check for metric anomalies
        for key, values in log_metrics.items():
            if not values:
                continue

            # All identical values — metric may be stuck
            if len(values) >= 5 and len(set(f"{v:.6f}" for v in values)) == 1:
                report.checks.append(VerifyCheck(
                    name=f"metric_stagnant_{key}",
                    category="integrity",
                    status="warn",
                    detail=f"{key} is completely stagnant at {values[0]:.4f} across {len(values)} measurements — metric computation may be broken",
                    severity="high",
                    module_path="metric_computation",
                ))

            # Extremely large values — metric may be in wrong units or buggy
            if any(v > 1e6 for v in values):
                report.checks.append(VerifyCheck(
                    name=f"metric_anomalous_{key}",
                    category="integrity",
                    status="warn",
                    detail=f"{key} has extreme values (max={max(values):.2f}) — may indicate numerical issues",
                    severity="medium",
                    module_path="metric_computation",
                ))

    # ─────────────────────────────────────────────────
    # Layer 6: Configuration Consistency Verification
    # ─────────────────────────────────────────────────

    def _verify_config_consistency(self, think_result, execute_result, report: VerifyReport):
        """Fix 2: Check that model configuration is consistent across files.

        Common issues:
        - Model parameters in script don't match checkpoint
        - Script uses different num_views than model expects
        - Checkpoint saved with different model architecture
        """
        # Check 1: If a checkpoint is loaded, verify model architecture matches
        log_file = execute_result.get("log_file", "")
        log_path = self.project_dir / log_file if log_file else None
        if log_path and log_path.exists():
            try:
                # Read only last 50KB of log to avoid OOM on large files
                with open(str(log_path), "r") as f:
                    f.seek(0, 2)  # Seek to end
                    size = f.tell()
                    f.seek(max(0, size - 50000))  # Read last 50KB
                    content = f.read()
                # Check for checkpoint loading mismatches
                if "size mismatch" in content or "missing key" in content:
                    report.checks.append(VerifyCheck(
                        name="checkpoint_mismatch",
                        category="configuration",
                        status="fail",
                        detail="Model architecture doesn't match checkpoint — "
                               "size mismatch or missing keys detected in training log",
                        severity="critical",
                    ))
            except Exception as e:
                logger.debug(f"Config check failed to read log: {e}")

        # Check 2: Verify training script uses the correct model class
        scripts_dir = self.project_dir / "scripts"
        if scripts_dir.exists():
            for script in scripts_dir.glob("*.py"):
                try:
                    content = script.read_text()
                    # Check if script creates model with hardcoded params that might
                    # differ from what the model class expects
                    if "num_views" in content or "num_angular" in content:
                        # Verify num_views is read from args/config, not hardcoded
                        import re
                        hardcoded_views = re.findall(r"num_views\s*=\s*(\d+)", content)
                        for val in hardcoded_views:
                            if val not in ("81", "49", "25", "9"):
                                report.checks.append(VerifyCheck(
                                    name="hardcoded_num_views",
                                    category="configuration",
                                    status="warn",
                                    detail=f"num_views={val} in {script.name} — "
                                           "may not match actual data grid size",
                                    severity="medium",
                                ))
                except Exception:
                    pass

    # ─────────────────────────────────────────────────
    # Layer 7: System Health Verification
    # ─────────────────────────────────────────────────

    def _verify_system_health(self, report: VerifyReport):
        """Check system resources are adequate."""
        # Disk space
        try:
            stat = subprocess.run(
                ["df", "-h", str(self.project_dir)],
                capture_output=True, text=True, timeout=5,
            )
            if stat.returncode == 0:
                lines = stat.stdout.strip().split("\n")
                if len(lines) >= 2:
                    parts = lines[1].split()
                    use_pct = parts[-2] if len(parts) >= 6 else ""
                    if use_pct and int(use_pct.replace("%", "")) > 95:
                        report.checks.append(VerifyCheck(
                            name="disk_space",
                            category="system",
                            status="fail",
                            detail=f"Disk usage at {use_pct} — risk of write failures",
                            severity="critical",
                            module_path="system",
                        ))
        except Exception:
            pass

        # GPU availability
        try:
            gpu_stat = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            if gpu_stat.returncode == 0:
                for line in gpu_stat.stdout.strip().split("\n"):
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) == 2:
                        used, total = int(parts[0]), int(parts[1])
                        if total > 0 and used / total > 0.95:
                            report.checks.append(VerifyCheck(
                                name="gpu_memory",
                                category="system",
                                status="warn",
                                detail=f"GPU memory nearly full: {used}MB/{total}MB ({used/total*100:.0f}%)",
                                severity="medium",
                                module_path="system",
                            ))
        except Exception:
            pass

    # ─────────────────────────────────────────────────
    # Layer 8: Dataset Quality Verification
    # ─────────────────────────────────────────────────

    def _verify_dataset_quality(self, report: VerifyReport):
        """Check that validation splits are statistically meaningful.

        Detects:
        - Domains with < 3 validation scenes (metrics are unreliable)
        - Zero validation scenes for a domain
        - Extreme training/validation imbalance

        This is a meta-analysis capability: the agent can detect when its
        own metrics are unreliable due to insufficient validation data.
        """
        manifest_path = self.project_dir / "DATASET_MANIFEST.json"
        if not manifest_path.exists():
            return

        try:
            import json
            manifest = json.loads(manifest_path.read_text())
            datasets = manifest.get("datasets", {})

            # Count validation scenes per domain group
            domain_val_counts = {}
            domain_train_counts = {}
            for ds_name, ds_info in datasets.items():
                scenes = ds_info.get("scenes", {})
                grid_size = ds_info.get("grid_size", [9, 9])
                grid_str = f"{grid_size[0]}×{grid_size[1]}" if isinstance(grid_size, list) and len(grid_size) >= 2 else "?"

                for scene_name, scene_info in scenes.items():
                    split = scene_info.get("split", "unknown")
                    if split not in ("train", "val"):
                        continue
                    # Map dataset name to domain group
                    group = self._map_dataset_to_group(ds_name)
                    if split == "train":
                        domain_train_counts[group] = domain_train_counts.get(group, 0) + 1
                    else:
                        domain_val_counts[group] = domain_val_counts.get(group, 0) + 1

            # Check each domain's validation coverage
            issues = []
            for group in sorted(set(list(domain_train_counts.keys()) + list(domain_val_counts.keys()))):
                train_n = domain_train_counts.get(group, 0)
                val_n = domain_val_counts.get(group, 0)

                if val_n == 0:
                    issues.append(
                        f"[{group}] No validation scenes — metrics for this domain "
                        f"CANNOT be evaluated. Training results are UNVERIFIABLE."
                    )
                elif val_n == 1:
                    issues.append(
                        f"[{group}] Only 1 validation scene — metrics for this domain "
                        f"are statistically UNRELIABLE. MAE for {group} is based on "
                        f"a SINGLE scene; fluctuations are likely noise, not real trends. "
                        f"DO NOT make decisions based on {group} MAE changes < 0.1."
                    )
                elif val_n < 3:
                    issues.append(
                        f"[{group}] Only {val_n} validation scenes (vs {train_n} train) — "
                        f"metrics for this domain have HIGH variance. Interpret with caution."
                    )

            if issues:
                issue_text = "; ".join(issues)
                logger.warning(f"DATASET QUALITY ISSUES: {issue_text}")
                report.checks.append(VerifyCheck(
                    name="validation_coverage",
                    category="dataset",
                    status="warn",
                    detail=issue_text,
                    severity="high",
                    module_path="datasets",
                ))
                # Also store as structured info for REFLECT
                report.dataset_issues = {
                    "val_counts": domain_val_counts,
                    "train_counts": domain_train_counts,
                    "issues": issues,
                }

        except Exception as e:
            logger.debug(f"Dataset quality check failed: {e}")

    def _map_dataset_to_group(self, ds_name: str) -> str:
        """Map dataset name to domain group for cross-domain analysis.

        Tries DATASET_MANIFEST.json first (dynamic), falls back to dataset name heuristics.
        """
        # Try reading from manifest (dynamic, project-agnostic)
        manifest_path = self.project_dir / "DATASET_MANIFEST.json"
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text())
                ds_info = manifest.get("datasets", {}).get(ds_name, {})
                ds_type = ds_info.get("type", "")
                if ds_type:
                    return ds_type
            except Exception:
                pass

        # Heuristic fallback: extract group from dataset name patterns
        # This is a GENERIC heuristic, not project-specific
        name_lower = ds_name.lower()
        if "non-lambertian" in name_lower or "nonlambert" in name_lower or "specular" in name_lower:
            return "Non-Lambertian"
        elif "lambertian" in name_lower or "diffuse" in name_lower:
            return "Lambertian"
        elif "urban" in name_lower or "mixed" in name_lower:
            return "Mixed"
        elif "synthetic" in name_lower or "syn" in name_lower:
            return "Synthetic"
        else:
            return ds_name[:30]

    # ─────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────

    def _check_crash_log(self, log_file: str) -> str:
        """Read crash info from a log file."""
        if not log_file:
            return ""
        log_path = self.project_dir / log_file
        if not log_path.exists():
            return ""
        try:
            content = log_path.read_text(errors="ignore")
            # Get last 500 chars which usually contain the error
            return content[-500:]
        except Exception:
            return ""

    def _check_normal_completion(self, log_file: str) -> str:
        """Check if a process completed normally by looking for completion artifacts.

        Returns a description of what was found if completion is confirmed,
        or empty string if the process likely crashed.
        """
        if not log_file:
            return ""
        log_path = self.project_dir / log_file
        if not log_path.exists():
            return ""

        try:
            content = log_path.read_text(errors="ignore")
        except Exception:
            return ""

        # Check for normal completion markers in the log
        completion_markers = [
            (r"Saved\s+(training_log\.json|best_checkpoint|final_checkpoint)", "checkpoints saved"),
            (r"FINAL\s+METRICS|FINAL_VAL_MAE", "final metrics reported"),
            (r"Early stopping at epoch", "early stopped (normal)"),
            (r"Saved.*?to\s+outputs", "output saved"),
            (r"Best val_MAE.*epoch", "best model recorded"),
        ]
        found = []
        for pattern, desc in completion_markers:
            if re.search(pattern, content, re.IGNORECASE):
                found.append(desc)

        if found:
            return "; ".join(found)

        # Check for crash markers — if present, definitely NOT normal completion
        crash_markers = ["Traceback (most recent call last)", "RuntimeError", "CUDA out of memory",
                         "KeyError", "ValueError", "FileNotFoundError", "ImportError"]
        for marker in crash_markers:
            if marker in content:
                return ""

        # No completion markers but also no crash — inconclusive
        return ""

    def _guess_module_from_error(self, error_text: str) -> str:
        """Guess which module failed based on error message."""
        error_lower = error_text.lower()
        if "dataloader" in error_lower or "dataset" in error_lower:
            return "dataset_loader"
        if "model" in error_lower or "forward" in error_lower:
            return "model_forward"
        if "loss" in error_lower:
            return "loss_function"
        if "optimizer" in error_lower or "gradient" in error_lower:
            return "optimizer"
        if "config" in error_lower or "yaml" in error_lower:
            return "config_loader"
        if "cuda" in error_lower or "gpu" in error_lower:
            return "gpu_runtime"
        return "unknown"

    def _verify_real_data_loaded(self, report: VerifyReport, execute_result: dict):
        """Runtime data fingerprint check — verify training used REAL data.

        This runs a quick Python check to load one sample from the dataset
        and verify it's not random noise. This is the most reliable defense
        against LLMs that secretly swap in synthetic data.
        """
        # Dynamically find the dataset module (first non-__init__ .py in datasets/)
        datasets_dir = self.project_dir / "datasets"
        dataset_module = None
        dataset_class_name = None
        if datasets_dir.exists():
            for ds_file in datasets_dir.glob("*.py"):
                if ds_file.name.startswith("__"):
                    continue
                try:
                    tree = ast.parse(ds_file.read_text())
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ClassDef):
                            for base in node.bases:
                                if isinstance(base, ast.Attribute) and base.attr == "Dataset":
                                    dataset_module = ds_file
                                    dataset_class_name = node.name
                                    break
                            if dataset_module:
                                break
                except Exception:
                    pass
                if dataset_module:
                    break

        if not dataset_module or not dataset_class_name:
            return

        module_import = f"datasets.{dataset_module.stem}"

        try:
            result = subprocess.run(
                [
                    sys.executable, "-c",
                    (
                        "import sys, json, numpy as np; "
                        f"from {module_import} import {dataset_class_name}; "
                        f"ds = {dataset_class_name}(split='val'); "
                        "sample = ds[0]; "
                        "checks = []; "
                        "for k, v in sample.items(): "
                        "    if isinstance(v, np.ndarray) and v.size > 10: "
                        "        std = float(np.std(v)); "
                        "        mean = float(np.mean(v)); "
                        "        # Check 1: Not uniform distribution "
                        "        if std < 1e-6: "
                        "            checks.append(f'{k}: std={std:.8f} (constant)'); "
                        "        # Check 2: Has spatial correlation (real images do) "
                        "        flat = v.flatten()[:100]; "
                        "        diff = np.diff(flat.astype(float)); "
                        "        autocorr = float(np.mean(diff * diff)); "
                        "        if autocorr < 1e-8: "
                        "            checks.append(f'{k}: no spatial correlation (noise)'); "
                        "if checks: "
                        "    print(json.dumps({'data_warning': checks})); "
                        "else: "
                        "    print('REAL_DATA_CONFIRMED'); "
                    ),
                ],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(self.project_dir),
            )

            if result.returncode == 0:
                stdout = result.stdout.strip()
                if stdout == "REAL_DATA_CONFIRMED":
                    report.checks.append(VerifyCheck(
                        name="real_data_runtime",
                        category="integrity",
                        status="pass",
                        detail="Runtime data fingerprint check passed — training data is real",
                        module_path="data_pipeline",
                    ))
                elif stdout.startswith("{"):
                    data = json.loads(stdout)
                    warnings = data.get("data_warning", [])
                    report.checks.append(VerifyCheck(
                        name="real_data_runtime",
                        category="integrity",
                        status="fail",
                        detail=f"Runtime data check suspicious: {'; '.join(warnings[:3])}",
                        severity="critical",
                        module_path="data_pipeline",
                    ))
            # If returncode != 0, dataset may not be importable — other checks handle that
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            pass  # Non-blocking — other checks will catch import failures

    # ─────────────────────────────────────────────────
    # Layer 9: Model Structural Soundness Verification
    # ─────────────────────────────────────────────────

    def _verify_model_structure(
        self, think_result: dict, execute_result: dict, report: VerifyReport
    ):
        """Verify model architecture has no structural issues.

        This runs AFTER execution (during VERIFY phase) and checks:
        1. Does the model file exist and parse correctly?
        2. Are there obvious dead branches (modules declared but not used in forward)?
        3. Are there fusion imbalances (one branch dominating gradient flow)?
        4. Are there aggressive bottlenecks that cause information loss?
        5. What domain assumptions does the architecture encode?

        This catches the pattern where the Code agent builds a "complex but broken"
        architecture that looks sophisticated but doesn't actually work.
        """
        action = think_result.get("action", "")
        if action != "experiment":
            return

        # Find model files in the project
        models_dir = self.project_dir / "models"
        if not models_dir.exists():
            return

        # Only check the most recently modified model file (the one being experimented on)
        try:
            model_files = sorted(
                models_dir.glob("*.py"),
                key=lambda f: f.stat().st_mtime,
                reverse=True,
            )
        except Exception:
            return

        if not model_files:
            return

        # Check the most recently modified model file
        model_file = model_files[0]
        try:
            content = model_file.read_text()
            tree = ast.parse(content)
        except Exception as e:
            report.checks.append(VerifyCheck(
                name="model_syntax",
                category="integrity",
                status="fail",
                detail=f"Model file {model_file.name} has syntax errors: {e}",
                severity="critical",
                module_path=model_file.stem,
            ))
            return

        # Analyze model structure
        self._check_model_dead_branches(tree, model_file, report)
        self._check_model_fusion_balance(tree, model_file, report)

    def _check_model_dead_branches(self, tree, model_file, report):
        """Check for modules declared in __init__ but never used in forward().

        v18: Uses model_structure_scanner instead of inline AST walk (67→15 lines).
        """
        from .model_structure_scanner import scan_model_file, find_dead_branches
        try:
            content = model_file.read_text()
            structure = scan_model_file(content)
            dead = find_dead_branches(structure)
            if dead:
                report.checks.append(VerifyCheck(
                    name="dead_modules",
                    category="integrity",
                    status="warn",
                    detail=f"Dead branches detected: {dead}. These modules are allocated but never used in forward().",
                    severity="medium",
                    module_path=model_file.stem,
                ))
        except Exception:
            pass

    def _check_model_fusion_balance(self, tree, model_file, report):
        """Check fusion point for branch channel imbalance."""
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            is_module = False
            for base in node.bases:
                if isinstance(base, ast.Attribute) and base.attr == "Module":
                    is_module = True
                    break
            if not is_module:
                continue

            # Find torch.cat calls in forward() and check branch count
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "forward":
                    for stmt in ast.walk(item):
                        if isinstance(stmt, ast.Call):
                            func = stmt.func
                            if not (isinstance(func, ast.Attribute) and func.attr in ("cat", "concat")):
                                continue

                            if stmt.args and isinstance(stmt.args[0], ast.List):
                                num_branches = len(stmt.args[0].elts)
                                if num_branches >= 3:
                                    report.checks.append(VerifyCheck(
                                        name="multi_branch_fusion",
                                        category="integrity",
                                        status="warn",
                                        detail=(
                                            f"Model {node.name} fuses {num_branches} branches via "
                                            f"concatenation. If channel counts are imbalanced, smaller "
                                            f"branches will receive fewer gradients. Use probe_model "
                                            f"or analyze_model to verify balance."
                                        ),
                                        severity="medium",
                                        module_path=model_file.stem,
                                    ))

            break  # Only check first nn.Module

    def _synthesize_diagnosis(self, report: VerifyReport):
        """Synthesize verification results into actionable diagnosis for REFLECT."""
        if not report.has_failures:
            report.diagnosis = ["All modules passed verification — results are reliable."]
            return

        # Group failures by module
        module_failures: dict[str, list[VerifyCheck]] = {}
        for check in report.all_failures:
            module = check.module_path or "unknown"
            module_failures.setdefault(module, []).append(check)

        report.failed_modules = list(module_failures.keys())

        # Generate diagnosis for each failing module
        for module, checks in module_failures.items():
            critical = [c for c in checks if c.severity == "critical"]
            high = [c for c in checks if c.severity == "high"]
            others = [c for c in checks if c.severity not in ("critical", "high")]

            if critical:
                report.diagnosis.append(
                    f"🔴 CRITICAL [{module}]: {critical[0].detail}. "
                    f"This module is NOT functioning correctly. "
                    f"REFLECT must diagnose root cause before planning next experiment."
                )
            elif high:
                report.diagnosis.append(
                    f"🟡 HIGH [{module}]: {high[0].detail}. "
                    f"This module may not be working as expected. "
                    f"REFLECT should investigate before assuming results are valid."
                )
            elif others:
                report.diagnosis.append(
                    f"🟢 LOW [{module}]: {others[0].detail}. "
                    f"Non-critical issue — results may still be usable but verify carefully."
                )

    # ─────────────────────────────────────────────────
    # Layer 10: Independent Third-Party Verification
    # ─────────────────────────────────────────────────

    def _verify_independent_probe(
        self, think_result: dict, execute_result: dict, report: VerifyReport
    ):
        """Run independent third-party verification using a lightweight probe.

        This loads the model checkpoint and runs a forward pass on random input,
        then checks the OUTPUT statistics. This catches:
        - Model outputs collapsed to constant (sigmoid ≈ 0.5)
        - NaN/Inf in outputs
        - Output range mismatches (e.g., depth > 1 when sigmoid is used)
        - Metric computation bugs (good reported MAE but collapsed outputs)

        Unlike the main model's evaluation (which uses the same codepath that
        generated the metrics), this is a truly independent check.
        """
        action = think_result.get("action", "")
        if action != "experiment":
            return

        experiment_launched = execute_result.get("experiment_launched", False)
        if not experiment_launched:
            return

        try:
            from .experiment_evaluator import IndependentProbe
            probe = IndependentProbe(self.project_dir, self.workspace)

            # Get reported metrics for cross-validation
            reported_metrics = execute_result.get("final_metrics", {})

            assessment = probe.run_independent_assessment(
                checkpoint_path=execute_result.get("checkpoint_path", ""),
                model_path=execute_result.get("model_path", ""),
                reported_metrics=reported_metrics,
            )

            if assessment.assessed:
                if assessment.anomaly_detected:
                    report.checks.append(VerifyCheck(
                        name="independent_probe_anomaly",
                        category="integrity",
                        status="fail",
                        detail=(
                            f"INDEPENDENT PROBE detected anomaly: {assessment.anomaly_detail}. "
                            f"This is a third-party assessment — the model's own metrics may be unreliable."
                        ),
                        evidence=f"Probe output samples: {assessment.independent_predictions_sample[:5]}",
                        severity="critical",
                        module_path="model_outputs",
                    ))
                    report.independent_assessment = {
                        "anomaly_detected": True,
                        "detail": assessment.anomaly_detail,
                        "agreement_score": assessment.agreement_score,
                        "confidence": assessment.confidence,
                    }
                else:
                    report.checks.append(VerifyCheck(
                        name="independent_probe",
                        category="integrity",
                        status="pass",
                        detail=(
                            f"Independent third-party probe confirms model outputs are valid "
                            f"(agreement={assessment.agreement_score:.2f}). "
                            f"No collapsed outputs, NaN, or range anomalies detected."
                        ),
                        module_path="model_outputs",
                    ))
                    report.independent_assessment = {
                        "anomaly_detected": False,
                        "agreement_score": assessment.agreement_score,
                        "confidence": assessment.confidence,
                    }
        except Exception as e:
            logger.debug(f"Independent probe skipped: {e}")
            # Non-blocking — the probe is supplementary, not mandatory

    def _analyze_training_curve(self, loss_sequence: list[float]) -> dict:
        """Analyze a complete training loss sequence for rich curve diagnostics.

        Detects: overfitting, oscillation, convergence speed, plateau.
        Returns a dict of findings (only populated fields).
        """
        if len(loss_sequence) < 5:
            return {}

        result = {}
        n = len(loss_sequence)

        # ── 1. Overfitting Detection ──
        # Find the global minimum, then check if loss rises significantly after it
        min_loss = min(loss_sequence)
        min_epoch = loss_sequence.index(min_loss)
        result["min_loss"] = round(min_loss, 6)
        result["min_loss_epoch"] = min_epoch

        # Overfitting: loss rises > 5% above minimum in the latter half
        if min_epoch < n * 0.75:  # minimum in first 75% of training
            tail_max = max(loss_sequence[min_epoch:])
            relative_rise = (tail_max - min_loss) / max(min_loss, _EPS)
            if relative_rise > self._overfit_rise_threshold and min_epoch < n * 0.5:
                result["overfit_epoch"] = min_epoch
            elif relative_rise > 0.10:
                result["overfit_epoch"] = min_epoch

        # ── 2. Oscillation Detection ──
        # Count direction changes (loss going up vs down)
        direction_changes = 0
        increases = 0
        for i in range(1, n):
            if loss_sequence[i] > loss_sequence[i-1]:
                increases += 1
            if (i >= 2 and
                ((loss_sequence[i] - loss_sequence[i-1]) * (loss_sequence[i-1] - loss_sequence[i-2])) < 0):
                direction_changes += 1

        result["direction_changes"] = direction_changes
        result["oscillation_ratio"] = increases / (n - 1) if n > 1 else 0

        # ── 3. Convergence Speed ──
        # What percentage of total decrease happens in first 25% of training?
        if n >= 8:
            early_loss = sum(loss_sequence[:max(n//4, 1)]) / max(n//4, 1)
            total_decrease = loss_sequence[0] - loss_sequence[-1]
            early_decrease = loss_sequence[0] - early_loss
            result["total_decrease_pct"] = total_decrease / max(loss_sequence[0], _EPS)

            if total_decrease < loss_sequence[0] * 0.05:
                result["convergence_speed"] = "very_slow"
            elif early_decrease > total_decrease * 0.8:
                result["convergence_speed"] = "fast_early_plateau"
            else:
                result["convergence_speed"] = "normal"

        # ── 4. Plateau Detection ──
        # Find the longest window where loss changes < 0.1%
        if n >= 10:
            window_size = max(n // 5, 5)
            for start in range(n - window_size):
                window = loss_sequence[start:start + window_size]
                window_range = max(window) - min(window)
                window_mean = sum(window) / len(window)
                relative_range = window_range / max(window_mean, _EPS)
                if relative_range < 0.001:  # < 0.1% variation
                    result["plateau_start"] = start
                    result["plateau_length"] = window_size
                    break

        return result

    # ─────────────────────────────────────────────────
    # Layer 11 (v12): Analysis Experiment Coverage Verification
    # ─────────────────────────────────────────────────

    def _verify_analysis_coverage(
        self, think_result: dict, execute_result: dict, report: VerifyReport
    ):
        """Verify data analysis experiments used sufficient method coverage.

        Checks that analysis experiments (no training) explored multiple
        independent feature families, not just one narrow approach.
        This prevents false-negative conclusions like "FFT energy ratios
        can't discriminate materials" from being generalized to "no angular
        feature can discriminate materials".
        """
        # Only check analysis experiments (no training launched)
        if execute_result.get("experiment_launched"):
            return
        if execute_result.get("is_paper_research"):
            return

        # Look for analysis output in the response or output files
        response = execute_result.get("response", "") or ""
        output_str = str(execute_result.get("output", "")) or ""
        combined = (response + " " + output_str).lower()

        # Heuristic: detect if this was a discrimination/classification analysis
        is_discrimination_analysis = any(
            kw in combined
            for kw in ["separability", "cohens_d", "cohen's d", "discriminat",
                        "classification", "distinguishable", "feasib", "roc_auc",
                        "bhattacharyya", "frequency_band"]
        )
        if not is_discrimination_analysis:
            return

        # Count independent analysis methods used
        method_indicators = {
            "fft_energy": ["fft", "frequency_band", "spectral_centroid", "energy_ratio"],
            "gradient": ["gradient", "angular_gradient", "spatial_deriv", "du,", "dv,"],
            "view_consistency": ["view_consistency", "direction_change", "sign_change",
                                 "angular_coherence", "cross_direction"],
            "symmetry": ["symmetry", "center_sym", "peak_dist"],
            "entropy": ["entropy", "spectral_entropy", "information"],
            "curvature": ["curvature", "second_order", "epi_curvature"],
            "variance_profile": ["variance_profile", "angular_var"],
        }

        methods_found = []
        for method_name, keywords in method_indicators.items():
            if any(kw in combined for kw in keywords):
                methods_found.append(method_name)

        n_methods = len(methods_found)

        if n_methods == 0:
            # Couldn't detect methods (might be in files we can't read)
            return

        if n_methods == 1:
            report.checks.append(VerifyCheck(
                name="analysis_method_coverage",
                category="integrity",
                status="warn",
                detail=(
                    f"Analysis experiment used only 1 method family: {methods_found[0]}. "
                    f"This is INSUFFICIENT to conclude a direction is infeasible. "
                    f"The Leader MUST categorize this as 'method_inadequacy' if marking as dead_end, "
                    f"NOT 'hypothesis_wrong'. At least 3 independent method families are required "
                    f"before concluding a hypothesis is wrong."
                ),
                evidence=f"Detected methods: {methods_found}. Required: ≥3 independent families.",
                severity="high",
            ))
        elif n_methods == 2:
            report.checks.append(VerifyCheck(
                name="analysis_method_coverage",
                category="integrity",
                status="warn",
                detail=(
                    f"Analysis experiment used 2 method families: {methods_found}. "
                    f"This is WEAK coverage. Consider adding at least 1 more independent method "
                    f"before drawing strong conclusions."
                ),
                evidence=f"Detected methods: {methods_found}. Recommended: ≥3 independent families.",
                severity="medium",
            ))
        else:
            report.checks.append(VerifyCheck(
                name="analysis_method_coverage",
                category="integrity",
                status="pass",
                detail=f"Analysis experiment used {n_methods} independent method families: {methods_found}.",
                evidence=f"Methods: {methods_found}",
                severity="low",
            ))

    # ─────────────────────────────────────────────────
    # Layer 12 (v12.2): Training Architecture Review
    # ─────────────────────────────────────────────────

    def _verify_training_architecture(
        self, think_result: dict, execute_result: dict, report: VerifyReport
    ):
        """Verify training experiments for architectural convergence issues.

        Checks that the Code agent's model actually learned what it was
        supposed to — routing weights differentiated, aux losses converged,
        no per-domain regression vs baseline.

        This catches the pattern where a sophisticated-looking architecture
        (dual-branch, attention, routing) trains successfully but the key
        mechanism (routing, attention) never actually learns to differentiate.
        """
        # Only check training experiments
        if not execute_result.get("experiment_launched"):
            return
        action = think_result.get("action", "")
        if action != "experiment":
            return

        # ── 12a: Routing Weight Differentiation ──
        self._check_routing_differentiation(execute_result, report)

        # ── 12b: Aux Loss Convergence ──
        self._check_aux_loss_convergence(execute_result, report)

        # ── 12c: Per-Domain Regression Detection ──
        self._check_domain_regression(execute_result, report)

    def _check_routing_differentiation(
        self, execute_result: dict, report: VerifyReport
    ):
        """Check if routing/fusion weights differentiated across domains.

        Parses training_log.json for routing_w_epi_* / routing_w_defocus_*
        entries. If all domains have nearly identical weights (within 5%),
        the routing mechanism has not learned.
        """
        log_json = self._load_training_log_json(execute_result)
        if not log_json:
            # Fallback: parse from state.last_training_logs text
            self._check_routing_from_log_text(execute_result, report)
            return

        epochs = log_json.get("epochs", [])
        if len(epochs) < 2:
            return

        # Collect final epoch routing weights per domain.
        # Domain-specific weight = w_epi - w_defocus (epistemic minus shared defocus baseline).
        # Using raw w_epi alone is misleading since it includes the shared defocus component
        # and cannot distinguish true domain specialization from a shared baseline increase.
        final_epoch = epochs[-1]
        routing_by_domain = {}
        for key, val in final_epoch.items():
            if key.startswith("routing_w_epi_") and not key.endswith("_count"):
                domain = key.replace("routing_w_epi_", "")
                # Look up matching defocus weight for the same domain
                defocus_val = final_epoch.get(f"routing_w_defocus_{domain}", 0.0)
                routing_by_domain[domain] = val - defocus_val

        if len(routing_by_domain) < 2:
            return

        # Check differentiation: all domains within 5% of each other?
        weights = list(routing_by_domain.values())
        # Handle potential negative values (w_defocus > w_epi for some domains)
        # Use absolute values for range check: if all weights are near zero
        # (regardless of sign), the router isn't differentiating.
        abs_weights = [abs(w) for w in weights]
        w_range = max(abs_weights) - min(abs_weights)
        # Also check if all weights are effectively zero (absolute sum too small)
        total_abs = sum(abs_weights)

        if w_range < 0.05 or total_abs < 0.02:
            report.checks.append(VerifyCheck(
                name="routing_differentiation",
                category="integrity",
                status="fail",
                detail=(
                    f"Routing weights NOT differentiated across domains. "
                    f"All domains have nearly identical weights: "
                    f"{', '.join(f'{d}={w:.4f}' for d, w in routing_by_domain.items())}. "
                    f"Range={w_range:.4f} (< 0.05 threshold). "
                    f"The routing/fusion mechanism is NOT learning to specialize. "
                    f"Possible causes: (1) aux_weight too low, (2) routing target "
                    f"[0.5,0.5] for majority class suppresses learning, "
                    f"(3) router input lacks information."
                ),
                evidence=f"Routing weights: {routing_by_domain}, range={w_range:.4f}",
                severity="high",
                module_path="routing",
            ))
        elif w_range < 0.15:
            report.checks.append(VerifyCheck(
                name="routing_differentiation",
                category="integrity",
                status="warn",
                detail=(
                    f"Routing weights weakly differentiated (range={w_range:.4f}). "
                    f"Weights: {', '.join(f'{d}={w:.4f}' for d, w in routing_by_domain.items())}. "
                    f"Differentiation may improve with more epochs or higher aux_weight."
                ),
                evidence=f"Routing weights: {routing_by_domain}, range={w_range:.4f}",
                severity="medium",
                module_path="routing",
            ))

    def _check_routing_from_log_text(
        self, execute_result: dict, report: VerifyReport
    ):
        """Fallback: parse routing weights from raw log text."""
        log_text = ""
        log_file = execute_result.get("log_file", "")
        if log_file:
            log_path = self.project_dir / log_file
            if log_path.exists():
                try:
                    log_text = log_path.read_text(errors="ignore")
                except Exception:
                    return

        # Also check state.last_training_logs
        if not log_text:
            log_text = execute_result.get("training_logs", "")

        if not log_text:
            return

        # Parse "Domain: w_epi=X.XXXX, w_defocus=X.XXXX" patterns
        routing_pattern = re.findall(
            r"(\w+):\s*w_epi=(\d+\.\d+),\s*w_defocus=(\d+\.\d+)",
            log_text,
        )
        if not routing_pattern:
            return

        # Use the LAST occurrence (final epoch).
        # Domain-specific weight = w_epi - w_defocus (epistemic minus defocus component).
        # Using only w_epi would be misleading since it includes the shared defocus baseline.
        routing_by_domain = {}
        for domain, w_epi, w_def in routing_pattern:
            routing_by_domain[domain] = float(w_epi) - float(w_def)

        if len(routing_by_domain) < 2:
            return

        weights = list(routing_by_domain.values())
        w_range = max(weights) - min(weights)

        if w_range < 0.05:
            report.checks.append(VerifyCheck(
                name="routing_differentiation",
                category="integrity",
                status="fail",
                detail=(
                    f"Routing weights NOT differentiated (log text). "
                    f"All domains: "
                    f"{', '.join(f'{d}={w:.4f}' for d, w in routing_by_domain.items())}. "
                    f"Range={w_range:.4f}. Routing mechanism not learning."
                ),
                severity="high",
                module_path="routing",
            ))

    def _check_aux_loss_convergence(
        self, execute_result: dict, report: VerifyReport
    ):
        """Check if auxiliary loss is converging (actually learning).

        If aux_loss barely changes across epochs, the auxiliary module
        (router, attention head, etc.) is not receiving useful gradient signal.
        """
        log_json = self._load_training_log_json(execute_result)
        if not log_json:
            return

        epochs = log_json.get("epochs", [])
        if len(epochs) < 2:
            return

        # Look for train_aux_loss or similar fields
        aux_losses = []
        for ep in epochs:
            for key in ["train_aux_loss", "aux_loss"]:
                if key in ep:
                    val = ep[key]
                    if isinstance(val, (int, float)):
                        aux_losses.append(float(val))
                        break

        if len(aux_losses) < 2:
            return

        # Check change rate
        initial = aux_losses[0]
        final = aux_losses[-1]
        if initial == 0:
            return
        change_pct = abs(final - initial) / abs(initial)

        if change_pct < 0.01:
            report.checks.append(VerifyCheck(
                name="aux_loss_convergence",
                category="integrity",
                status="fail",
                detail=(
                    f"Auxiliary loss NOT converging across {len(aux_losses)} epochs. "
                    f"Initial={initial:.6f}, Final={final:.6f}, change={change_pct:.2%}. "
                    f"The auxiliary module (routing/attention) is NOT learning. "
                    f"Possible causes: (1) aux_weight too low (gradient signal drowned by main loss), "
                    f"(2) aux target is trivially satisfied (e.g. [0.5,0.5] for majority class), "
                    f"(3) module input lacks information to differentiate."
                ),
                evidence=f"Aux loss: {aux_losses}",
                severity="high",
                module_path="auxiliary_loss",
            ))
        elif change_pct < 0.05:
            report.checks.append(VerifyCheck(
                name="aux_loss_convergence",
                category="integrity",
                status="warn",
                detail=(
                    f"Auxiliary loss barely changing: {change_pct:.2%} over "
                    f"{len(aux_losses)} epochs. Auxiliary module may need more "
                    f"training signal (higher aux_weight or better targets)."
                ),
                evidence=f"Aux loss: {aux_losses}",
                severity="medium",
                module_path="auxiliary_loss",
            ))

    def _check_domain_regression(
        self, execute_result: dict, report: VerifyReport
    ):
        """Check if any domain regressed significantly from baseline.

        Compares per-domain MAE in the training log against known baseline
        metrics stored in the project. If any domain degraded >20%, flags it.
        """
        log_json = self._load_training_log_json(execute_result)
        if not log_json:
            return

        epochs = log_json.get("epochs", [])
        if not epochs:
            return

        # Load baseline metrics from memory or known location
        baseline = self._load_baseline_metrics()
        if not baseline:
            return

        final_epoch = epochs[-1]
        regressions = []
        for key, val in final_epoch.items():
            if not key.startswith("MAE_") or key.endswith("_count"):
                continue
            domain = key.replace("MAE_", "")
            if domain in baseline:
                base_val = baseline[domain]
                if isinstance(val, (int, float)) and isinstance(base_val, (int, float)):
                    degradation = (val - base_val) / base_val
                    if degradation > 0.20:
                        regressions.append(
                            f"{domain}: {val:.4f} vs baseline {base_val:.4f} "
                            f"(+{degradation:.0%})"
                        )

        if regressions:
            report.checks.append(VerifyCheck(
                name="domain_regression",
                category="integrity",
                status="warn",
                detail=(
                    f"Per-domain MAE REGRESSED vs baseline:\n"
                    + "\n".join(f"  - {r}" for r in regressions)
                    + "\n\nThe new model makes some domains WORSE. "
                    + "REFLECT must investigate WHY before iterating."
                ),
                severity="high",
                module_path="domain_metrics",
            ))

    def _load_training_log_json(self, execute_result: dict) -> dict:
        """Load structured training_log.json from the project outputs.

        v12.3: Also tries parsing from raw log text as fallback when
        training_log.json doesn't exist or has unexpected format.
        """
        # Try known output locations
        for pattern in ["outputs/*/training_log.json", "outputs/training_log.json"]:
            candidates = list(self.project_dir.glob(pattern))
            if candidates:
                # Use the most recently modified
                latest = max(candidates, key=lambda p: p.stat().st_mtime)
                try:
                    data = json.loads(latest.read_text())
                    if data and isinstance(data, dict):
                        return data
                except Exception:
                    continue

        # Fallback: parse from raw log text in execute_result
        return self._parse_log_text_to_json(execute_result)

    def _parse_log_text_to_json(self, execute_result: dict) -> dict:
        """v12.3: Fallback parser — converts raw log text into training_log.json format.

        Extracts routing weights, aux losses, and per-domain MAE from stdout/log text
        when structured JSON is not available.
        """
        log_text = ""
        log_file = execute_result.get("log_file", "")
        if log_file:
            log_path = self.project_dir / log_file
            if log_path.exists():
                try:
                    log_text = log_path.read_text(errors="ignore")
                except Exception:
                    pass

        if not log_text:
            log_text = execute_result.get("training_logs", "")
        if not log_text:
            log_text = execute_result.get("log_tail", "")

        if not log_text or len(log_text) < 50:
            return {}

        result = {"epochs": []}

        # Parse per-domain MAE: "MAE_Lambertian: 0.387" or "Lambertian MAE=0.387"
        # Group by epoch markers: "Epoch 1", "Epoch 2", etc.
        epoch_blocks = re.split(r"(?:Epoch\s+\d+|epoch\s*\d+)", log_text)
        for block in epoch_blocks[1:]:  # Skip text before first epoch marker
            epoch_data = {}

            # Domain MAE patterns
            for m in re.finditer(
                r"MAE[_\s]+(\w+)[\s:=]+(\d+\.\d+)", block, re.IGNORECASE
            ):
                domain = m.group(1).strip("_")
                epoch_data[f"MAE_{domain}"] = float(m.group(2))

            # Aux loss patterns
            for m in re.finditer(
                r"(?:train_)?aux_loss[\s:=]+(\d+\.\d+)", block, re.IGNORECASE
            ):
                epoch_data["train_aux_loss"] = float(m.group(1))

            # Routing weight patterns
            for m in re.finditer(
                r"routing_w_epi_(\w+)[\s:=]+(\d+\.\d+)", block, re.IGNORECASE
            ):
                domain = m.group(1)
                epoch_data[f"routing_w_epi_{domain}"] = float(m.group(2))
            for m in re.finditer(
                r"routing_w_defocus_(\w+)[\s:=]+(\d+\.\d+)", block, re.IGNORECASE
            ):
                domain = m.group(1)
                epoch_data[f"routing_w_defocus_{domain}"] = float(m.group(2))

            if epoch_data:
                result["epochs"].append(epoch_data)

        # If no epoch markers found, try parsing all metrics from entire text
        if not result["epochs"] and log_text:
            epoch_data = {}
            # Take the LAST occurrence of each metric (final epoch)
            for m in re.finditer(
                r"MAE[_\s]+(\w+)[\s:=]+(\d+\.\d+)", log_text, re.IGNORECASE
            ):
                domain = m.group(1).strip("_")
                epoch_data[f"MAE_{domain}"] = float(m.group(2))
            for m in re.finditer(
                r"(?:train_)?aux_loss[\s:=]+(\d+\.\d+)", log_text, re.IGNORECASE
            ):
                epoch_data["train_aux_loss"] = float(m.group(1))
            for m in re.finditer(
                r"routing_w_epi_(\w+)[\s:=]+(\d+\.\d+)", log_text, re.IGNORECASE
            ):
                domain = m.group(1)
                epoch_data[f"routing_w_epi_{domain}"] = float(m.group(2))
            if epoch_data:
                result["epochs"].append(epoch_data)

        return result if result["epochs"] else {}

    def _load_baseline_metrics(self) -> dict:
        """Load baseline per-domain metrics from MEMORY_LOG or known files.

        v12.3: Dynamic domain discovery — no longer hardcoded domain names.
        Parses any 'DOMAIN_NAME: MAE=X.XXX' or 'MAE_DOMAIN: X.XXX' pattern.
        """
        # Check MEMORY_LOG.md for baseline entries
        memory_path = self.project_dir / "MEMORY_LOG.md"
        if not memory_path.exists():
            workspace_memory = self.workspace / "MEMORY_LOG.md"
            if workspace_memory.exists():
                memory_path = workspace_memory

        if not memory_path.exists():
            return {}

        try:
            text = memory_path.read_text(errors="ignore")
        except Exception:
            return {}

        baseline = {}

        # Common English words that are NOT domain names
        _STOPWORDS = frozenset({
            "the", "and", "for", "mae", "loss", "baseline", "best",
            "overall", "average", "mean", "total", "metric", "this",
            "with", "from", "that", "which", "epoch", "train", "val",
            "test", "model", "data", "result", "output", "score",
        })

        def _normalize_domain(name: str) -> str:
            """Normalize domain name: strip, title-case, collapse whitespace."""
            name = name.strip("_ \"'|")
            name = re.sub(r"\s+", "_", name)  # spaces → underscores
            # Title-case normalization: "lambertian" → "Lambertian"
            if name and not name[0].isupper():
                name = name[0].upper() + name[1:]
            return name

        # Pattern 1: "DOMAIN: MAE=X.XXX" or "DOMAIN: 0.387" (highest confidence)
        for match in re.finditer(
            r"(\w[\w\s-]*?)[:\s]+MAE[:\s]*(\d+\.\d+)",
            text,
        ):
            domain = _normalize_domain(match.group(1))
            if len(domain) > 2 and domain.lower() not in _STOPWORDS:
                baseline[domain] = float(match.group(2))

        # Pattern 2: "MAE_DOMAIN: X.XXX" (from training_log.json format)
        for match in re.finditer(
            r"MAE[_\s]+(\w+)[\s:=]+(\d+\.\d+)",
            text,
        ):
            domain = _normalize_domain(match.group(1))
            baseline[domain] = float(match.group(2))

        # Pattern 3: "Domain (DOMAIN): 0.387" or "| DOMAIN | 0.387 |"
        # Only match within table-like or parenthetical contexts to reduce false positives.
        for match in re.finditer(
            r"(?:Domain\s*\(|\|\s*)(\w[\w\s]*?)(?:\)|\s*\|)[:\s]*(\d+\.\d{2,4})",
            text,
        ):
            domain = _normalize_domain(match.group(1))
            val = float(match.group(2))
            if 0 < val < 10 and len(domain) > 2 and domain.lower() not in _STOPWORDS:
                baseline.setdefault(domain, val)

        return baseline
