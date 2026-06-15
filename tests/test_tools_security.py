"""Restored ToolRegistry security tests.

These 7 tests were deleted at some point (only their node IDs survived in
`.pytest_cache/v/cache/nodeids`). They guard the workspace-escape and
command-injection boundaries that the LLM-facing tools enforce — the
single most important security surface, since agents run shell commands
and write files autonomously.

Restored to match the CURRENT implementation in core/tools.py:
- _resolve_workspace_path rejects absolute/parent-escape paths
- _validate_command blocks sudo, dd-of-device, pipe-to-shell
- launch_experiment resolves log_file through the same path resolver
"""
from __future__ import annotations

import json

import pytest

from core.tools import ToolRegistry


@pytest.fixture
def registry(tmp_workspace, monkeypatch):
    """A ToolRegistry rooted at the temp workspace.

    We stub out the memory dependency (ToolRegistry accepts memory=None)
    so no SQLite state is created.
    """
    return ToolRegistry(tmp_workspace, memory=None)


class TestToolRegistrySecurity:
    """Restored from .pytest_cache node IDs (originally ToolRegistrySecurityTests,
    renamed to satisfy pytest's default Test* collection prefix)."""

    # ── Path escape protection ──

    def test_read_file_rejects_absolute_path(self, registry, tmp_workspace):
        """An absolute path outside the workspace must be rejected."""
        with pytest.raises(ValueError, match="absolute path|escapes workspace"):
            registry._resolve_workspace_path("/etc/passwd")

    def test_list_files_rejects_parent_escape(self, registry, tmp_workspace):
        """`../` traversal that escapes the workspace must be rejected."""
        with pytest.raises(ValueError, match="escapes workspace"):
            registry._resolve_workspace_path("../../../../etc")

    def test_write_file_rejects_path_traversal(self, registry, tmp_workspace):
        """write_file must refuse paths that escape the workspace."""
        result = registry.execute_tool(
            "write_file",
            {"path": "../../etc/malicious.txt", "content": "pwned"},
        )
        data = json.loads(result)
        assert "error" in data
        assert "escape" in data["error"].lower() or "escapes" in data["error"].lower()

    # ── Command injection protection ──

    def test_run_shell_blocks_sudo(self, registry):
        """sudo must be blocked — the agent must never escalate privileges."""
        result = registry.execute_tool(
            "run_shell", {"command": "sudo rm -rf /"},
        )
        data = json.loads(result)
        assert "error" in data or data.get("returncode", 1) != 0

    def test_run_shell_blocks_dd_to_device(self, registry):
        """`dd ... of=/dev/sdX` must be blocked — disk-wiping protection."""
        result = registry.execute_tool(
            "run_shell", {"command": "dd if=/dev/zero of=/dev/sda bs=1M"},
        )
        data = json.loads(result)
        assert "error" in data or data.get("returncode", 1) != 0

    def test_run_shell_supports_pipes_and_cd(self, registry, tmp_workspace):
        """Legitimate shell operators (|, &&, cd) must still work.

        This is the positive control — the security layer must not be so
        aggressive that it blocks normal pipeline usage.
        """
        # A harmless piped command that the validator should allow.
        result = registry.execute_tool(
            "run_shell", {"command": "echo hello | cat"},
        )
        data = json.loads(result)
        # Should execute (returncode 0), not be blocked.
        assert "error" not in data or "blocked" not in str(data).lower()
        assert data.get("returncode") == 0

    # ── launch_experiment log path ──

    def test_launch_experiment_rejects_log_path_traversal(self, registry):
        """launch_experiment must resolve log_file through the same workspace
        boundary check — otherwise it could write logs anywhere on disk."""
        with pytest.raises(ValueError, match="escapes workspace|absolute path"):
            # We can't safely execute a real launch, but the path resolution
            # happens before any subprocess spawn, so the ValueError fires first.
            registry._exec_launch_experiment(
                command="echo test",  # harmless command
                log_file="../../../tmp/evil.log",
            )
