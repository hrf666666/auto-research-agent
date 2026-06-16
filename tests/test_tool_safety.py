"""Tests for Phase 2: tool-level safety contracts (P1).

write_file and launch_experiment should enforce safety rules themselves,
not rely on an external SafetyGuard. This tests the naming/path checks
and the dry-run gate.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from core.tools import ToolRegistry


@pytest.fixture
def registry(tmp_path):
    """ToolRegistry rooted at a temp workspace with standard dirs."""
    (tmp_path / "scripts").mkdir()
    (tmp_path / "tools").mkdir()
    (tmp_path / "models").mkdir()
    (tmp_path / "datasets").mkdir()
    (tmp_path / "outputs").mkdir()
    return ToolRegistry(tmp_path, memory=None)


class TestWriteFileNamingSafety:
    """Phase 2 change 1: write_file enforces naming conventions."""

    def test_rejects_root_py_file(self, registry):
        """Python files in the workspace root must be rejected."""
        result = json.loads(registry.execute_tool(
            "write_file", {"path": "rogue_model.py", "content": "x = 1"}
        ))
        assert "error" in result
        assert "scripts/" in result["error"] or "tools/" in result["error"]

    def test_accepts_scripts_dir_py(self, registry):
        """Python files in scripts/ should be allowed."""
        result = json.loads(registry.execute_tool(
            "write_file", {"path": "scripts/train_model.py", "content": "x = 1"}
        ))
        assert "error" not in result or "success" in result.get("status", "").lower()

    def test_accepts_tools_dir_py(self, registry):
        """Python files in tools/ should be allowed."""
        result = json.loads(registry.execute_tool(
            "write_file", {"path": "tools/debug_helper.py", "content": "x = 1"}
        ))
        assert "error" not in result or "success" in result.get("status", "").lower()

    def test_train_script_must_be_in_scripts(self, registry):
        """train_*.py outside scripts/ should be rejected (naming rule, not protected dir)."""
        # Use outputs/ (not protected) to test the naming rule specifically
        result = json.loads(registry.execute_tool(
            "write_file", {"path": "outputs/train_rogue.py", "content": "x = 1"}
        ))
        assert "error" in result
        assert "scripts" in result["error"].lower()

    def test_debug_script_allowed_in_tools(self, registry):
        """debug_* in tools/ should be allowed."""
        result = json.loads(registry.execute_tool(
            "write_file", {"path": "tools/debug_shapes.py", "content": "x = 1"}
        ))
        assert "error" not in result or "success" in result.get("status", "").lower()

    def test_error_message_is_actionable(self, registry):
        """Error message should tell the LLM what to do, not just 'rejected'."""
        result = json.loads(registry.execute_tool(
            "write_file", {"path": "rogue.py", "content": "x = 1"}
        ))
        if "error" in result:
            # Error should contain a suggestion about where to write
            assert "scripts" in result["error"] or "tools" in result["error"]

    def test_non_py_files_not_affected(self, registry):
        """Non-.py files (configs, logs, etc.) should not be subject to naming rules."""
        result = json.loads(registry.execute_tool(
            "write_file", {"path": "config.json", "content": "{}"}
        ))
        assert "error" not in result or "naming" not in result.get("error", "").lower()
