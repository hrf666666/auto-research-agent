"""Additional tool safety contract tests."""
from __future__ import annotations

import json

from core.tools import ToolRegistry


def test_research_loop_config_dry_run_gate_rejects_without_receipt(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "train_model.py").write_text("print('train')")
    registry = ToolRegistry(tmp_path, config={"safety": {"mandatory_dry_run": True}})

    out = json.loads(registry.execute_tool("launch_experiment", {
        "command": "python scripts/train_model.py",
        "log_file": "outputs/run/train.log",
    }))

    assert "error" in out
    assert "Dry-run required" in out["error"]


def test_run_shell_dry_run_receipt_allows_launch(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "train_model.py").write_text("import sys\nprint('ok')\n")
    registry = ToolRegistry(tmp_path, config={"safety": {"mandatory_dry_run": True}})

    dry = json.loads(registry.execute_tool("run_shell", {
        "command": "python scripts/train_model.py --dry_run",
        "timeout": 10,
    }))
    assert dry["returncode"] == 0
    launched = json.loads(registry.execute_tool("launch_experiment", {
        "command": "python scripts/train_model.py",
        "log_file": "outputs/run/train.log",
    }))

    assert "pid" in launched
    proc = registry.get_launched_process(launched["pid"])
    if proc:
        proc.wait(timeout=10)


def test_run_shell_blocks_redirect_to_protected_file(tmp_path):
    registry = ToolRegistry(tmp_path)

    out = json.loads(registry.execute_tool("run_shell", {"command": "printf x > state.json"}))

    assert "error" in out
    assert "protected" in out["error"]


def test_run_shell_blocks_rm_project_brief(tmp_path):
    registry = ToolRegistry(tmp_path)

    out = json.loads(registry.execute_tool("run_shell", {"command": "rm PROJECT_BRIEF.md"}))

    assert "error" in out
    assert "protected" in out["error"]


def test_analyze_model_rejects_manifest_absolute_outside_workspace(tmp_path):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "m.py").write_text("import torch.nn as nn\nclass M(nn.Module):\n    def forward(self,x): return x\n")
    outside = tmp_path.parent / "outside_manifest.json"
    outside.write_text("{}")
    registry = ToolRegistry(tmp_path)

    out = json.loads(registry.execute_tool("analyze_model", {
        "model_path": "models/m.py",
        "dataset_manifest": str(outside),
    }))

    assert "error" in out
    assert "relative to workspace" in out["error"] or "escapes workspace" in out["error"]
