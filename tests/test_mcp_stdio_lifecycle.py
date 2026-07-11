"""MCP stdio lifecycle tests."""
from __future__ import annotations

import subprocess
import sys

from core.tools import ToolRegistry


def test_readline_with_timeout_returns_none_on_no_data():
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(2)"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    try:
        assert ToolRegistry._readline_with_timeout(proc.stdout, 0.05) is None
    finally:
        proc.kill()
        proc.wait(timeout=5)


def test_mcp_stdio_call_timeout_cleans_session(tmp_path):
    registry = ToolRegistry(tmp_path)
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time,sys; time.sleep(10)"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    registry._mcp_sessions["fake"] = {
        "transport": "stdio",
        "proc": proc,
        "next_id": 1,
        "lock": __import__("threading").RLock(),
    }

    result = registry._mcp_stdio_call("fake", "tool", {}, timeout=0.05)

    assert result is None
    assert "fake" not in registry._mcp_sessions
    assert proc.poll() is not None
