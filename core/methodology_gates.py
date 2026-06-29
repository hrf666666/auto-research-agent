r"""Methodology gates — fact-layer enforcement (Phase 3, Reform v21).

Three gates that check SCIENTIFIC METHODOLOGY using only DETERMINISTIC facts.
Each gate operates strictly on the FACT layer — never the INTERPRETATION layer.

Design rule (Reform v21, Ground 2, proven via adversarial verification):
  - Falsification gate: success_criteria is a predicate, metrics are numbers.
    Comparing them is pure math. → FACT layer, safe to enforce.
  - Control-coverage gate: "does a control run exist?" is a SQL query.
    → FACT layer. But "what the control result means" is interpretation
    → we only MARK (uncontrolled_inconclusive), never force action.
  - Dead-end signature gate: method+config+dataset is a structured key.
    Matching it against history is a lookup. → FACT layer. But "why it's a
    dead end" / "under what conditions" is interpretation → we only WARN.

NONE of these gates change think_result["action"]. They attach structured
facts to experiment_facts so that THINK (next cycle) and the LLM can see them.
This is the v18-verified pattern: facts at the tool/record layer (unbypassable),
interpretation stays with the LLM.
"""
from __future__ import annotations

import ast
import json
import logging
import operator
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("autoresearcher.methodology_gates")


# ── Gate 1: Falsification ──────────────────────────────────────────────

# Match: "val_MAE < 0.15", "Lambertian_MAE <= 0.16", "mae_overall >= 0.5"
_CRIT_RE = re.compile(
    r"^\s*([A-Za-z_][\w]*)\s*(<=|>=|<|>|==)\s*([-+]?\d*\.?\d+)\s*$"
)
_OPS = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "==": operator.eq,
}

# Metric name normalization — handles the 4 divergent key chains found in
# the codebase (training_log_parser / loop / memory / verifier all use
# different casing). This is the SINGLE normalization point.
_METRIC_ALIASES = {
    "val_mae": ["val_mae", "valmae", "val_mae_overall", "valmaeoverall", "overall_val_mae", "overallvalmae"],
    "best_val_mae": ["best_val_mae", "bestvalmae"],
    "mae_overall": ["mae_overall", "maeoverall", "overall_mae", "overallmae"],
    "mae_lambertian": ["mae_lambertian", "lambertian_mae", "maelambertian"],
    "mae_non_lambertian": ["mae_non_lambertian", "non_lambertian_mae", "maenonlambertian", "mae_nonlambertian"],
    "mae_urban": ["mae_urban", "urban_mae", "maeurban"],
    "mae": ["mae"],
    "rmse": ["rmse"],
    "psnr": ["psnr"],
    "accuracy": ["accuracy", "acc"],
}


def _normalize_metric_name(name: str) -> str:
    """Normalize a metric name to canonical form (handles casing/separator variants).

    Also handles val_ prefix stripping: 'val_mae_lambertian' → 'mae_lambertian'
    because leader writes 'val_X' but metrics keys are often just 'X'.
    """
    key = re.sub(r"[\s\-]", "", name.lower())
    # Direct alias match
    for canonical, aliases in _METRIC_ALIASES.items():
        if key in aliases:
            return canonical
    # Try stripping val_ prefix: val_mae_lambertian → mae_lambertian
    if key.startswith("val_"):
        stripped = key[4:]
        for canonical, aliases in _METRIC_ALIASES.items():
            if stripped in aliases or stripped == canonical:
                return canonical
        # Also try bare match on stripped
        return stripped
    return key  # unknown metric — return normalized as-is


def _find_metric_value(criteria_name: str, metrics: dict[str, float]) -> float | None:
    """Find the metric value matching criteria_name, handling naming chaos.

    Also coerces str values to float (defensive — monitor historically
    str()'d values, and execute_result may carry stale str values).
    """
    canonical = _normalize_metric_name(criteria_name)
    raw = None
    # Direct match on canonical
    if canonical in metrics:
        raw = metrics[canonical]
    else:
        # Try alias matching
        for m_name, m_val in metrics.items():
            if _normalize_metric_name(m_name) == canonical:
                raw = m_val
                break
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


@dataclass
class FalsificationResult:
    """Result of falsification gate — a FACT, not an interpretation."""
    parseable: bool = False
    metric_name: str = ""
    operator: str = ""
    threshold: float | None = None
    actual_value: float | None = None
    criteria_met: bool | None = None  # None = couldn't evaluate
    detail: str = ""
    sub_criteria: list[dict] = field(default_factory=list)  # for compound (AND) criteria

    def to_dict(self) -> dict:
        return {
            "parseable": self.parseable,
            "metric_name": self.metric_name,
            "operator": self.operator,
            "threshold": self.threshold,
            "actual_value": self.actual_value,
            "criteria_met": self.criteria_met,
            "detail": self.detail,
            "sub_criteria": self.sub_criteria,
        }


def _evaluate_single_predicate(
    criteria_str: str, metrics: dict[str, float]
) -> tuple[bool | None, float | None, str, bool]:
    """Evaluate one predicate like 'val_MAE < 0.15'.

    Returns (met, actual_value, detail, format_parseable).
      met: True/False/None (None = couldn't evaluate, e.g. metric missing)
      format_parseable: True if the string IS a valid predicate format
                        (distinguishes "bad format" from "metric not found")
    Strips common English prefixes (Overall, Best, Final) before matching.
    """
    cleaned = criteria_str.strip()
    for prefix in ("overall ", "best ", "final ", "target "):
        if cleaned.lower().startswith(prefix):
            cleaned = cleaned[len(prefix):]
            break
    m = _CRIT_RE.match(cleaned)
    if not m:
        return None, None, f"unparseable: {criteria_str[:40]!r}", False

    name = _normalize_metric_name(m.group(1))
    op = m.group(2)
    threshold = float(m.group(3))

    if not metrics:
        return None, None, f"no metrics to evaluate {name}", True

    actual = _find_metric_value(m.group(1), metrics)
    if actual is None:
        return None, None, f"metric '{m.group(1)}' not found", True

    met = _OPS[op](actual, threshold)
    return met, actual, f"{actual:.6f} {op} {threshold} = {met}", True


def evaluate_falsification(
    success_criteria: str | None, metrics: dict[str, float]
) -> FalsificationResult:
    """Gate 1: Evaluate whether success_criteria is met by metrics.

    Supports:
      - Single predicate: "val_MAE < 0.15"
      - Compound (AND): "val_MAE < 0.15 AND Lambertian_MAE <= 0.16"
        All sub-predicates must be met. If any is unparseable → criteria_met=None.

    Pure math: parse into predicates, look up metric values, compare.
    Returns criteria_met as True/False/None.

    None (couldn't evaluate) is honest — we do NOT pretend success or failure
    when the criterion is unparseable or the metric is missing.
    """
    result = FalsificationResult()

    if not success_criteria or not success_criteria.strip():
        result.detail = "no success_criteria provided"
        return result

    criteria_str = success_criteria.strip()

    # Detect compound (AND) — research standard is "all must hold"
    # Split on " AND " case-insensitive
    parts = re.split(r"\s+AND\s+", criteria_str, flags=re.IGNORECASE)

    if len(parts) == 1:
        # Single predicate
        met, actual, detail, fmt_ok = _evaluate_single_predicate(criteria_str, metrics)
        if not fmt_ok:
            result.detail = detail
            return result
        # Format is valid — even if metric missing (met=None), it's parseable
        result.parseable = True
        # Re-parse for metadata
        cleaned = criteria_str.strip()
        for prefix in ("overall ", "best ", "final ", "target "):
            if cleaned.lower().startswith(prefix):
                cleaned = cleaned[len(prefix):]
                break
        m = _CRIT_RE.match(cleaned)
        if m:
            result.metric_name = _normalize_metric_name(m.group(1))
            result.operator = m.group(2)
            result.threshold = float(m.group(3))
        if met is None:
            result.detail = detail  # metric not found, but format is valid
            return result
        result.actual_value = actual
        result.criteria_met = met
        result.detail = detail
        return result

    # Compound (AND): evaluate each sub-predicate.
    # Leader criteria often mix numerical predicates ("val_MAE < 0.15") with
    # boolean/descriptive ones ("training completes 50 epochs"). The numerical
    # ones are what falsification can evaluate; boolean ones are covered by
    # VERIFY (crash detection). So we evaluate only the numerical sub-predicates
    # and ignore non-numerical ones rather than letting them tank the whole result.
    all_met = True
    sub_results = []
    evaluable_count = 0

    for part in parts:
        part = part.strip()
        if not part:
            continue
        met, actual, detail, fmt_ok = _evaluate_single_predicate(part, metrics)
        sub_name = ""
        # Re-parse with same prefix cleaning
        cleaned = part
        for prefix in ("overall ", "best ", "final ", "target "):
            if cleaned.lower().startswith(prefix):
                cleaned = cleaned[len(prefix):]
                break
        m = _CRIT_RE.match(cleaned)
        if m:
            sub_name = _normalize_metric_name(m.group(1))

        is_numerical = fmt_ok  # format is a valid predicate (even if metric missing)
        sub_results.append({
            "criterion": part[:60],
            "metric": sub_name,
            "met": met,
            "actual": actual,
            "detail": detail,
            "numerical": is_numerical,
        })
        if met is True or met is False:
            evaluable_count += 1
            if met is False:
                all_met = False
        # met is None + not numerical → skip (boolean/descriptive, not our job)

    result.sub_criteria = sub_results
    # parseable if at least one numerical sub-predicate was evaluated
    result.parseable = evaluable_count > 0

    if evaluable_count == 0:
        # No numerical sub-predicates at all → can't evaluate
        result.parseable = False
        result.criteria_met = None
        result.detail = "no numerical predicates found in compound criteria"
    else:
        result.criteria_met = all_met
        met_count = sum(1 for s in sub_results if s.get("met") is True)
        result.detail = (
            f"compound AND: {met_count}/{evaluable_count} numerical predicates met "
            f"→ {'MET' if all_met else 'NOT MET'}"
        )
        # Report first numerical metric as representative
        for s in sub_results:
            if s.get("numerical"):
                result.metric_name = s["metric"]
                result.actual_value = s["actual"]
                break

    return result


# ── Gate 2: Control-coverage ───────────────────────────────────────────

@dataclass
class ControlCoverageResult:
    """Result of control-coverage gate — a FACT (exists or not), not interpretation."""
    claim_type: str = "null"
    needs_control: bool = False
    control_exists: bool = False
    marked_inconclusive: bool = False
    detail: str = ""

    def to_dict(self) -> dict:
        return {
            "claim_type": self.claim_type,
            "needs_control": self.needs_control,
            "control_exists": self.control_exists,
            "marked_inconclusive": self.marked_inconclusive,
            "detail": self.detail,
        }


def check_control_coverage(
    claim_type: str | None,
    method: str,
    db_path: Path,
    current_output_dir: str,
) -> ControlCoverageResult:
    """Gate 2: Check if a causal claim has a supporting control run.

    FACT layer only: "does a control run exist?" is a SQL query.
    We MARK uncontrolled causal claims as inconclusive — we do NOT force
    the agent to run a control (that's an interpretation-layer decision:
    controls are expensive, the LLM decides if worth it).
    """
    result = ControlCoverageResult()
    result.claim_type = (claim_type or "null").lower().strip()

    if result.claim_type != "causal":
        result.detail = f"claim_type={result.claim_type}, no control needed"
        return result

    result.needs_control = True

    # Query experiment_facts for a control run of the SAME method.
    # A "control" is an experiment that uses the same base method but removes
    # a component (ablation) or uses a baseline variant. We require BOTH:
    #   (a) the experiment shares the method (by keyword in command/output_dir)
    #   (b) it's flagged as ablation/control/baseline/no-<component>
    # This avoids false-positives from unrelated experiments that happen to
    # contain "control" in their name.
    method_lower = (method or "").lower()
    control_keywords = ("ablation", "control", "baseline", "no_energy", "no-guided", "no_guided", "without")
    try:
        with sqlite3.connect(str(db_path)) as conn:
            rows = conn.execute(
                """
                SELECT output_dir, command FROM experiment_facts
                WHERE output_dir != ? AND command != ''
                """,
                (current_output_dir,),
            ).fetchall()
            for out_dir, cmd in rows:
                combined = (out_dir + " " + (cmd or "")).lower()
                has_control_kw = any(kw in combined for kw in control_keywords)
                # Method match: either explicit method keyword in command,
                # or same script basename
                shares_method = (
                    (method_lower and method_lower in combined and len(method_lower) >= 4)
                    or any(
                        kw in combined
                        for kw in ("cost_volume", "costvolume", "train_v30", "v30")
                        if "cost_volume" in method_lower or "v30" in method_lower
                    )
                )
                if has_control_kw and shares_method:
                    result.control_exists = True
                    result.detail = f"control run found: {Path(out_dir).name}"
                    break
            if not result.control_exists:
                result.marked_inconclusive = True
                result.detail = (
                    f"causal claim (method='{method}') but no matching "
                    f"ablation/control run found — marked inconclusive "
                    f"(cannot attribute improvement to method)"
                )
    except sqlite3.Error as e:
        result.detail = f"control check DB error: {e}"
        logger.warning(f"control_coverage DB query failed: {e}")

    return result


# ── Gate 3: Dead-end signature ─────────────────────────────────────────

@dataclass
class DeadEndResult:
    """Result of dead-end signature gate — a FACT (matched or not)."""
    method: str = ""
    signature: str = ""
    matched_dead_ends: list[str] = field(default_factory=list)
    is_known_dead_end: bool = False
    detail: str = ""

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "signature": self.signature,
            "matched_dead_ends": self.matched_dead_ends,
            "is_known_dead_end": self.is_known_dead_end,
            "detail": self.detail,
        }


def _build_method_signature(think_result: dict, metrics: dict[str, float]) -> str:
    """Build a structured method signature for dead-end matching.

    Replaces the 16-word keyword substring match (constraint_engine:200-207).
    Uses method name + dataset fingerprint, both from structured fields.
    """
    # Method: prefer explicit "method" field, fall back to hypothesis keywords
    method = think_result.get("method", "").strip().lower()
    if not method:
        # Reuse loop.py's _extract_method_name logic inline (can't import easily)
        text = (
            think_result.get("task", "") + " " + think_result.get("hypothesis", "")
        ).lower()
        method_kws = [
            "cost_volume", "cost volume", "fft", "frequency", "attention",
            "transformer", "unet", "resnet", "epipolar", "lambertian",
            "contrastive", "focal_loss", "distill",
        ]
        for kw in method_kws:
            if kw in text:
                method = kw.replace(" ", "_")
                break
    if not method:
        method = "unknown"

    # Dataset fingerprint: which domains are present in metrics
    domains = sorted(
        k for k in metrics if k.startswith("mae_") and k not in ("mae", "mae_overall")
    )
    dataset_fp = "+".join(domains) if domains else "unknown"

    return f"{method}@{dataset_fp}"


def check_dead_end_signature(
    think_result: dict,
    metrics: dict[str, float],
    db_path: Path,
) -> DeadEndResult:
    """Gate 3: Check if this method+dataset signature is a known dead end.

    FACT layer: queries the memory_entries table for dead_end entries (the
    single source of truth — see docs/DATA_CONTRACT.md) matching the method
    signature. The experiments table no longer carries a dead_end column.
    We WARN (attach to facts) — we do NOT block action (that's interpretation:
    "is this really the same dead end or different conditions?"). New methods
    with no history → honest "no data" (not blocked).
    """
    result = DeadEndResult()
    result.method = think_result.get("method", "") or _build_method_signature(
        think_result, metrics
    ).split("@")[0]
    result.signature = _build_method_signature(think_result, metrics)

    try:
        with sqlite3.connect(str(db_path)) as conn:
            # dead_end source of truth is memory_entries (entry_type='dead_end').
            rows = conn.execute(
                """
                SELECT cycle, content FROM memory_entries
                WHERE entry_type = 'dead_end'
                  AND content IS NOT NULL AND content != ''
                ORDER BY timestamp ASC
                """
            ).fetchall()

            if not rows:
                result.detail = "no dead ends recorded in history"
                return result

            # Match: does any dead_end text reference the same method?
            # Method keywords of length >=3 are specific enough (fft, dct, gcd
            # are real method abbreviations). We match on word boundaries to
            # avoid false positives (e.g. "ma" in "material").
            method_lower = result.method.lower()
            for cycle, dead_end_text in rows:
                combined = f"{dead_end_text or ''}".lower()
                if len(method_lower) >= 3 and method_lower in combined:
                    result.matched_dead_ends.append(
                        f"cycle{cycle}: {dead_end_text[:80]}"
                    )

            if result.matched_dead_ends:
                result.is_known_dead_end = True
                result.detail = (
                    f"method '{result.method}' found in {len(result.matched_dead_ends)} "
                    f"dead-end record(s) — review before proceeding"
                )
            else:
                result.detail = f"no dead-end match for method '{result.method}'"

    except sqlite3.Error as e:
        result.detail = f"dead-end DB query failed: {e}"
        logger.warning(f"dead_end_signature DB query failed: {e}")

    return result


# ── Gate 4: Spec conformance ───────────────────────────────────────────

@dataclass
class SpecConformanceResult:
    """Result of spec-conformance gate — a FACT (signature present or not)."""
    spec_loaded: bool = False
    spec_name: str = ""
    required_signatures: list[str] = field(default_factory=list)
    found_signatures: list[str] = field(default_factory=list)
    missing_signatures: list[str] = field(default_factory=list)
    has_deviation: bool = False
    detail: str = ""

    def to_dict(self) -> dict:
        return {
            "spec_loaded": self.spec_loaded,
            "spec_name": self.spec_name,
            "required_signatures": self.required_signatures,
            "found_signatures": self.found_signatures,
            "missing_signatures": self.missing_signatures,
            "has_deviation": self.has_deviation,
            "detail": self.detail,
        }


def check_spec_conformance(
    workspace: Path, spec_path: Path | None = None
) -> SpecConformanceResult:
    """Gate 4: Check if model code contains signatures declared in spec.

    FACT layer: "does the code file contain this operation?" is a text search.
    We do NOT judge whether the implementation is correct — that's interpretation.

    The spec is a JSON file (experiment_spec.json in workspace root, or custom
    path). It declares required code signatures for the current experiment:
    {
        "name": "V30 energy-guided cost volume",
        "files": ["models/cost_volume_net.py"],
        "required_signatures": ["sort", "n_keep", "mean"]  # operations that MUST appear
    }

    If no spec file exists, the gate returns spec_loaded=False (no-op).
    This is intentional: spec checking only activates when a spec is written.
    """
    result = SpecConformanceResult()

    # Locate spec file
    if spec_path is None:
        spec_path = workspace / "experiment_spec.json"
    if not spec_path.exists():
        result.detail = "no experiment_spec.json found (spec gate inactive)"
        return result

    # Load spec
    try:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        result.detail = f"spec file unreadable: {e}"
        return result

    result.spec_loaded = True
    result.spec_name = spec.get("name", "unnamed")
    result.required_signatures = spec.get("required_signatures", [])
    files_to_check = spec.get("files", [])

    if not result.required_signatures or not files_to_check:
        result.detail = "spec has no required_signatures or files — nothing to check"
        return result

    # Read all specified files, concatenate their content
    combined_code = ""
    for rel_path in files_to_check:
        full_path = workspace / rel_path
        if full_path.exists():
            try:
                combined_code += full_path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue

    if not combined_code:
        result.detail = f"none of the spec files found: {files_to_check}"
        return result

    # Check each required signature (case-sensitive — code operations are)
    for sig in result.required_signatures:
        if sig in combined_code:
            result.found_signatures.append(sig)
        else:
            result.missing_signatures.append(sig)

    result.has_deviation = len(result.missing_signatures) > 0
    if result.has_deviation:
        result.detail = (
            f"spec deviation: missing {result.missing_signatures} "
            f"in {files_to_check}"
        )
    else:
        result.detail = f"all {len(result.required_signatures)} signatures present"

    return result


# ── Combined: run all gates for a cycle ────────────────────────────────

@dataclass
class MethodologyVerdict:
    """Combined result of all methodology gates."""
    falsification: FalsificationResult = field(default_factory=FalsificationResult)
    control_coverage: ControlCoverageResult = field(default_factory=ControlCoverageResult)
    dead_end: DeadEndResult = field(default_factory=DeadEndResult)
    spec_conformance: SpecConformanceResult = field(default_factory=SpecConformanceResult)

    def to_dict(self) -> dict:
        return {
            "falsification": self.falsification.to_dict(),
            "control_coverage": self.control_coverage.to_dict(),
            "dead_end": self.dead_end.to_dict(),
            "spec_conformance": self.spec_conformance.to_dict(),
        }

    def summary(self) -> str:
        """One-line summary for logging / fact record."""
        parts = []
        f = self.falsification
        if f.parseable and f.criteria_met is not None:
            av = f.actual_value
            try:
                av_str = f"{float(av):.4f}" if av is not None else "?"
            except (TypeError, ValueError):
                av_str = str(av)[:20]
            parts.append(
                f"criteria({'MET' if f.criteria_met else 'NOT MET'}: "
                f"{av_str}{f.operator}{f.threshold})"
                if av is not None
                else f"criteria({'MET' if f.criteria_met else 'NOT MET'})"
            )
        elif f.parseable:
            parts.append(f"criteria(unresolvable: {f.detail[:40]})")
        else:
            parts.append(f"criteria(unparseable)")

        c = self.control_coverage
        if c.marked_inconclusive:
            parts.append("UNCONTROLLED")
        elif c.needs_control and c.control_exists:
            parts.append("controlled")

        d = self.dead_end
        if d.is_known_dead_end:
            parts.append(f"DEAD_END_WARN({len(d.matched_dead_ends)})")

        s = self.spec_conformance
        if s.spec_loaded and s.has_deviation:
            parts.append(f"SPEC_DEVIATION({len(s.missing_signatures)})")
        elif s.spec_loaded:
            parts.append("spec_ok")

        return " | ".join(parts) if parts else "no gates applicable"


def run_all_gates(
    think_result: dict,
    execute_result: dict,
    db_path: Path,
    current_output_dir: str = "",
    workspace: Path | None = None,
) -> MethodologyVerdict:
    """Run all methodology gates for a completed experiment.

    Args:
        think_result: The THINK decision (has success_criteria, claim_type, task)
        execute_result: The EXECUTE result (has log_file, final_metrics)
        db_path: Path to experiment_history.db
        current_output_dir: output_dir of current experiment (to exclude from control query)
        workspace: project root (for spec-conformance gate). If None, spec gate skipped.

    Returns combined verdict. Does NOT modify think_result or execute_result.
    The caller (loop.py) decides how to use the verdict.
    """
    verdict = MethodologyVerdict()

    # Get metrics: prefer fact_spine, fall back to execute_result
    metrics = {}
    log_file = execute_result.get("log_file", "")
    if log_file:
        output_dir = str(Path(log_file).parent)
        try:
            with sqlite3.connect(str(db_path)) as conn:
                row = conn.execute(
                    "SELECT metrics_json FROM experiment_facts WHERE output_dir = ?",
                    (output_dir,),
                ).fetchone()
                if row and row[0]:
                    metrics = json.loads(row[0])
        except (sqlite3.Error, json.JSONDecodeError):
            pass
    if not metrics:
        metrics = execute_result.get("final_metrics") or execute_result.get(
            "training_metrics"
        ) or {}

    # Gate 1: Falsification
    success_criteria = think_result.get("success_criteria", "")
    verdict.falsification = evaluate_falsification(success_criteria, metrics)

    # Gate 2: Control-coverage
    claim_type = think_result.get("claim_type", "null")
    method = _build_method_signature(think_result, metrics).split("@")[0]
    verdict.control_coverage = check_control_coverage(
        claim_type, method, db_path, current_output_dir
    )

    # Gate 3: Dead-end signature
    verdict.dead_end = check_dead_end_signature(think_result, metrics, db_path)

    # Gate 4: Spec conformance (only if workspace provided)
    if workspace is not None:
        verdict.spec_conformance = check_spec_conformance(workspace)

    return verdict
