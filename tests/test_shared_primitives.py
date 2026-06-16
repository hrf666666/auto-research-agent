"""Tests for shared primitives (Fix C): training_log_parser + model_structure_scanner.

These replace 15+ duplicated AST-walk copies and 4 divergent loss-parsers with
one tested implementation each. The key behavioral fix: _estimate_params now
handles keyword arguments (modern PyTorch style), fixing the "0 params detected"
bug that made analyze_model produce garbage.
"""
from __future__ import annotations

import pytest

from core.training_log_parser import (
    parse_loss_series, has_nan_loss, classify_loss_trend,
    extract_metrics, load_training_log, LossTrend,
)
from core.model_structure_scanner import (
    scan_model_file, find_dead_branches, ModelStructure,
)


# ── Training log parser ──

class TestLossParsing:

    def test_parse_loss_series(self):
        log = "epoch 1 loss=0.5\nepoch 2 loss=0.3\nepoch 3 loss=0.2"
        losses = parse_loss_series(log)
        assert losses == [0.5, 0.3, 0.2]

    def test_parse_empty(self):
        assert parse_loss_series("") == []
        assert parse_loss_series("no loss here") == []

    def test_nan_detection(self):
        assert has_nan_loss("loss=nan at epoch 5") is True
        assert has_nan_loss("loss=inf") is True
        assert has_nan_loss("loss=0.5") is False

    def test_decreasing_trend(self):
        losses = [1.0, 0.8, 0.6, 0.4, 0.2, 0.1]
        trend = classify_loss_trend(losses)
        assert trend.is_decreasing is True
        assert trend.is_diverging is False

    def test_diverging_trend(self):
        losses = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
        trend = classify_loss_trend(losses)
        assert trend.is_diverging is True

    def test_plateau_trend(self):
        losses = [0.5, 0.501, 0.499, 0.5, 0.502, 0.498]
        trend = classify_loss_trend(losses)
        assert trend.is_plateaued is True

    def test_insufficient_data(self):
        trend = classify_loss_trend([0.5, 0.3])
        assert trend.has_data is True
        assert trend.is_decreasing is False  # can't classify with <4 points

    def test_metrics_extraction(self):
        log = "val_MAE=0.25\nMAE_overall: 0.31\nrmse 0.12\nacc=0.85"
        metrics = extract_metrics(log)
        assert "val_mae" in metrics
        assert metrics["val_mae"] == 0.25
        assert "mae_overall" in metrics

    def test_parser_matches_verifier_regex(self):
        """Phase 1 change 6: training_log_parser must produce the same loss
        values as verifier's inline regex (the one being replaced)."""
        log = "epoch 1 loss=0.50\nepoch 2 loss=0.35\nepoch 3 loss=nan"
        # Old verifier regex
        import re
        old_result = [float(x) for x in re.findall(r"loss[=:\s]+([0-9.]+)", log, re.IGNORECASE)]
        # New shared parser
        new_result = parse_loss_series(log)
        assert old_result == new_result, f"Mismatch: old={old_result} new={new_result}"

    def test_parser_nan_detection_matches_verifier(self):
        """has_nan_loss must match verifier's inline nan/inf regex."""
        log = "epoch 5 loss=nan\nepoch 6 loss=inf"
        assert has_nan_loss(log) is True
        log2 = "epoch 5 loss=0.5"
        assert has_nan_loss(log2) is False


# ── Model structure scanner (the kwargs fix) ──

class TestModelStructureScanner:

    def test_basic_conv_model(self):
        """Positional args (old style): Conv2d(3, 64, 3)"""
        source = """
import torch.nn as nn
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3)
        self.fc = nn.Linear(64, 10)
    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x
"""
        structure = scan_model_file(source)
        assert len(structure.modules) == 1
        assert structure.modules[0].name == "SimpleModel"
        assert structure.total_estimated_params > 0

    def test_kwargs_model_fixes_zero_params_bug(self):
        """Keyword args (modern style): Conv2d(in_channels=3, out_channels=64, kernel_size=3)

        Before Fix C, this returned 0 params (the bug). Now it must return >0.
        """
        source = """
import torch.nn as nn
class ModernModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3)
    def forward(self, x):
        return self.conv(x)
"""
        structure = scan_model_file(source)
        assert structure.total_estimated_params > 0, (
            "kwargs-style Conv2d still returns 0 params — the bug is not fixed!"
        )

    def test_dead_branch_detection(self):
        """A module assigned in __init__ but never used in forward is a dead branch."""
        source = """
import torch.nn as nn
class ModelWithDeadBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.used = nn.Conv2d(3, 16, 3)
        self.dead = nn.Linear(16, 10)  # never referenced in forward
    def forward(self, x):
        return self.used(x)
"""
        structure = scan_model_file(source)
        dead = find_dead_branches(structure)
        assert any("dead" in d for d in dead)
        assert not any("used" in d for d in dead)

    def test_parse_error_handled(self):
        structure = scan_model_file("def broken(:")
        assert structure.parse_error is not None
        assert structure.modules == []

    def test_non_module_class_ignored(self):
        source = """
class NotAModel:
    def __init__(self):
        self.x = 42
"""
        structure = scan_model_file(source)
        assert len(structure.modules) == 0

    def test_multiple_layers_counted(self):
        source = """
import torch.nn as nn
class MultiLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc = nn.Linear(64, 10)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return self.fc(x)
"""
        structure = scan_model_file(source)
        # conv1: 3*32*9=864, conv2: 32*64*9=18432, fc: 64*10+10=650
        assert structure.total_estimated_params > 19000

    def test_embedding_counted(self):
        source = """
import torch.nn as nn
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(1000, 128)
    def forward(self, x):
        return self.emb(x)
"""
        structure = scan_model_file(source)
        assert structure.total_estimated_params >= 128000  # 1000*128
