r"""Shared primitives for parsing training logs — single source of truth.

Previously, `re.findall(r"loss[=:\s]+([0-9.]+)", ...)` was copy-pasted across
4 files (verifier.py ×2, experiment_evaluator.py ×1, loop.py ×1) with 3
divergent "is loss decreasing?" thresholds (0.99×, 1.1×, 2.0×, 10% windows).
This module provides ONE parser + ONE trend classifier so all consumers agree.

Also consolidates the NaN/Inf loss detection (previously a separate regex)
and metric extraction into a single `load_training_log` entry point.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# ── Regexes (defined once, used everywhere) ──

_LOSS_RE = re.compile(r"loss[=:\s]+([0-9.]+)", re.IGNORECASE)
_LOSS_NAN_RE = re.compile(r"loss[=:\s]+(nan|inf)", re.IGNORECASE)
# Common metric patterns: val_MAE=0.25, MAE_overall: 0.31, rmse 0.12
_METRIC_RE = re.compile(
    r"(val_[A-Za-z_]+|MAE[A-Za-z_]*|rmse|mse|psnr|accuracy|acc)[=:\s]+([0-9.]+)",
    re.IGNORECASE,
)


@dataclass
class LossTrend:
    """Classification of a loss series — the single agreed-upon verdict."""
    has_data: bool = False
    is_nan: bool = False
    is_decreasing: bool = False
    is_increasing: bool = False
    is_plateaued: bool = False
    is_diverging: bool = False
    first: float = 0.0
    last: float = 0.0
    count: int = 0


def parse_loss_series(log_text: str) -> list[float]:
    """Extract all numeric loss values from log text. Single source."""
    if not log_text:
        return []
    return [float(m) for m in _LOSS_RE.findall(log_text)]


def has_nan_loss(log_text: str) -> bool:
    """True if the log contains loss=nan or loss=inf."""
    return bool(_LOSS_NAN_RE.search(log_text or ""))


def classify_loss_trend(losses: list[float]) -> LossTrend:
    """Classify a loss series using a SINGLE, consistent threshold set.

    Replaces the 3 divergent implementations:
    - verifier.py: decreasing if last_third < first_third * 0.99; increasing if > 1.1
    - experiment_evaluator.py: not decreasing if late >= early * 0.99; diverging if > 2.0
    - loop.py: plateau if range < 0.01 over 10% windows

    Unified thresholds (documented here, not scattered):
    - Need ≥4 points to classify.
    - Compare first-third average vs last-third average.
    - Decreasing: last < first * 0.99
    - Plateaued: |last - first| / max(first, eps) < 0.01
    - Diverging: last > first * 2.0
    - Increasing (not diverging): last > first * 1.1
    """
    if not losses or len(losses) < 4:
        return LossTrend(has_data=bool(losses), count=len(losses))

    third = max(1, len(losses) // 3)
    early = sum(losses[:third]) / third
    late = sum(losses[-third:]) / third

    eps = 1e-8
    trend = LossTrend(has_data=True, first=losses[0], last=losses[-1],
                      count=len(losses))

    if late > early * 2.0:
        trend.is_diverging = True
    elif late > early * 1.1:
        trend.is_increasing = True
    elif abs(late - early) / max(early, eps) < 0.01:
        trend.is_plateaued = True
    elif late < early * 0.99:
        trend.is_decreasing = True

    return trend


def extract_metrics(log_text: str) -> dict[str, float]:
    """Extract named metrics (val_MAE, MAE_overall, etc.) from log text."""
    if not log_text:
        return {}
    metrics = {}
    for m in _METRIC_RE.finditer(log_text):
        name = m.group(1).lower()
        try:
            metrics[name] = float(m.group(2))
        except ValueError:
            continue
    return metrics


def load_training_log(log_path: Optional[Path]) -> str:
    """Read a training log file, returning empty string on any error.

    Consolidates the repeated try/except/read_text pattern.
    """
    if not log_path:
        return ""
    p = Path(log_path)
    if not p.exists() or not p.is_file():
        return ""
    try:
        return p.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""
