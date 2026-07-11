"""Tests for core/metrics.py — direction-aware comparison and canonical keys."""
from __future__ import annotations

from core.metrics import (
    MetricSpec,
    canonicalize_metric_key,
    canonicalize_metrics,
    is_domain_metric,
    domain_name_from_metric,
    parse_metric_specs,
    metric_improved,
    best_metric,
    goal_achieved,
)


class TestCanonicalKey:
    def test_uppercase_normalized(self):
        assert canonicalize_metric_key("MAE_Lambertian") == "mae_lambertian"

    def test_spaces_replaced(self):
        assert canonicalize_metric_key("val MAE") == "val_mae"

    def test_canonicalize_metrics_dict(self):
        result = canonicalize_metrics({"MAE_Lambertian": 0.1, "Val_MAE": 0.2})
        assert result == {"mae_lambertian": 0.1, "val_mae": 0.2}


class TestDomainMetric:
    def test_mae_lambertian_is_domain(self):
        assert is_domain_metric("MAE_Lambertian") is True

    def test_val_mae_lambertian_is_domain(self):
        assert is_domain_metric("val_MAE_Lambertian") is True

    def test_plain_mae_not_domain(self):
        assert is_domain_metric("MAE") is False

    def test_mae_overall_not_domain(self):
        assert is_domain_metric("MAE_overall") is False

    def test_count_excluded(self):
        assert is_domain_metric("MAE_Lambertian_count") is False

    def test_domain_name_extraction(self):
        assert domain_name_from_metric("MAE_Lambertian") == "lambertian"
        assert domain_name_from_metric("val_MAE_Urban") == "urban"


class TestDirection:
    def test_lower_improved(self):
        assert metric_improved(0.1, 0.2, "lower") is True

    def test_lower_not_improved(self):
        assert metric_improved(0.3, 0.2, "lower") is False

    def test_higher_improved(self):
        assert metric_improved(0.9, 0.8, "higher") is True

    def test_higher_not_improved(self):
        assert metric_improved(0.7, 0.8, "higher") is False

    def test_first_value_always_improves(self):
        assert metric_improved(0.5, None, "lower") is True


class TestBestMetric:
    def test_lower_takes_min(self):
        assert best_metric(0.2, 0.1, "lower") == 0.1

    def test_higher_takes_max(self):
        assert best_metric(0.8, 0.9, "higher") == 0.9

    def test_first_value(self):
        assert best_metric(None, 0.5, "lower") == 0.5


class TestGoalAchieved:
    def test_lower_meets_target(self):
        spec = MetricSpec(key="val_mae", target=0.15, direction="lower")
        assert goal_achieved(0.10, spec) is True

    def test_lower_equal_target(self):
        spec = MetricSpec(key="val_mae", target=0.15, direction="lower")
        assert goal_achieved(0.15, spec) is True

    def test_lower_misses_target(self):
        spec = MetricSpec(key="val_mae", target=0.15, direction="lower")
        assert goal_achieved(0.20, spec) is False

    def test_higher_meets_target(self):
        spec = MetricSpec(key="accuracy", target=0.9, direction="higher")
        assert goal_achieved(0.95, spec) is True

    def test_higher_misses_target(self):
        spec = MetricSpec(key="accuracy", target=0.9, direction="higher")
        assert goal_achieved(0.85, spec) is False

    def test_nan_never_achieved(self):
        spec = MetricSpec(key="val_mae", target=0.15, direction="lower")
        assert goal_achieved(float("nan"), spec) is False


class TestParseSpecs:
    def test_parse_valid_config(self):
        config = {"goals": {"metrics": [
            {"key": "val_MAE", "target": 0.15, "direction": "lower"},
            {"key": "accuracy", "target": 0.9, "direction": "higher"},
        ]}}
        specs = parse_metric_specs(config)
        assert len(specs) == 2
        assert specs[0].key == "val_mae"
        assert specs[1].direction == "higher"

    def test_empty_config(self):
        assert parse_metric_specs({}) == []

    def test_invalid_direction_skipped(self):
        config = {"goals": {"metrics": [
            {"key": "x", "target": 1.0, "direction": "sideways"},
        ]}}
        assert parse_metric_specs(config) == []
