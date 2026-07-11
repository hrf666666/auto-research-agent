"""Shared metric key and goal comparison helpers."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MetricSpec:
    key: str
    target: float
    direction: str = "lower"


def canonicalize_metric_key(key: str) -> str:
    """Normalize metric keys to lowercase snake-ish form."""
    return re.sub(r"[^a-z0-9]+", "_", str(key).strip().lower()).strip("_")


def canonicalize_metrics(metrics: dict | None) -> dict[str, Any]:
    if not isinstance(metrics, dict):
        return {}
    return {canonicalize_metric_key(k): v for k, v in metrics.items()}


def is_domain_metric(key: str) -> bool:
    k = canonicalize_metric_key(key)
    if k in {"mae", "val_mae", "mae_overall", "val_mae_overall", "best_val_mae"}:
        return False
    if k.endswith("_count"):
        return False
    return k.startswith("mae_") or k.startswith("val_mae_")


def domain_name_from_metric(key: str) -> str:
    k = canonicalize_metric_key(key)
    if k.startswith("val_mae_"):
        return k[len("val_mae_"):]
    if k.startswith("mae_"):
        return k[len("mae_"):]
    return k


def parse_metric_specs(config: dict | None) -> list[MetricSpec]:
    specs: list[MetricSpec] = []
    for item in ((config or {}).get("goals", {}) or {}).get("metrics", []) or []:
        try:
            key = canonicalize_metric_key(item.get("key", ""))
            if not key:
                continue
            target = float(item.get("target"))
            direction = str(item.get("direction", "lower")).lower()
            if direction not in {"lower", "higher"}:
                continue
            specs.append(MetricSpec(key=key, target=target, direction=direction))
        except (TypeError, ValueError):
            continue
    return specs


def metric_value(metrics: dict | None, key: str):
    cmetrics = canonicalize_metrics(metrics)
    return cmetrics.get(canonicalize_metric_key(key))


def is_finite_number(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def metric_improved(new_value, old_value, direction: str, rel_threshold: float = 0.0) -> bool:
    if old_value is None:
        return True
    try:
        new = float(new_value)
        old = float(old_value)
    except (TypeError, ValueError):
        return False
    if direction == "higher":
        return new > old * (1 + rel_threshold)
    return new < old * (1 - rel_threshold)


def best_metric(current, new_value, direction: str):
    if current is None:
        return float(new_value)
    return max(float(current), float(new_value)) if direction == "higher" else min(float(current), float(new_value))


def goal_achieved(value, spec: MetricSpec) -> bool:
    if not is_finite_number(value):
        return False
    v = float(value)
    return v >= spec.target if spec.direction == "higher" else v <= spec.target
