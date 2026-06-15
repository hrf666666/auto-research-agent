"""Shared test fixtures and sys.path setup for the auto_research_agent test suite.

This conftest makes `core` importable from the tests without requiring an
installed package, and provides lightweight mocks for the heaviest
collaborators (the LLM call, the tool registry) so unit tests can exercise
the dispatcher / loop logic without hitting the network or GPU.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

import pytest

# Make the repo root importable so `import core.agents` works from tests/.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ─────────────────────────────────────────────────────────────
# ToolTrace builder — the anti-deception record underpinning most tests
# ─────────────────────────────────────────────────────────────

def make_trace(calls: list[tuple[str, dict, object]]) -> "ToolTrace":
    """Build a ToolTrace from a compact list of (name, args, result).

    `result` may be a dict (auto-json-encoded), a str (used verbatim), or
    None (empty result). This mirrors how real tool results are recorded.
    """
    from core.agents import ToolTrace

    trace = ToolTrace()
    for name, args, result in calls:
        if result is None:
            result_str = ""
        elif isinstance(result, str):
            result_str = result
        else:
            result_str = json.dumps(result)
        trace.record(name, args, result_str)
    return trace


@pytest.fixture
def make_tool_trace():
    """Fixture exposing the make_trace helper to tests."""
    return make_trace


@pytest.fixture
def empty_trace():
    from core.agents import ToolTrace
    return ToolTrace()


# ─────────────────────────────────────────────────────────────
# Fake LLM exception types — stand-ins for openai/zai SDK errors
# ─────────────────────────────────────────────────────────────

class FakeAPIStatusError(Exception):
    """Minimal stand-in for openai.APIStatusError / zai.APIStatusError.

    The real SDKs set ``self.status_code``; we replicate just that so the
    error classifier (which reads ``getattr(exc, "status_code")``) works
    without the SDK installed.
    """

    def __init__(self, message: str = "", status_code: Optional[int] = None,
                 body: Optional[dict] = None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body or {}
        # Some SDKs expose the raw response body via .response / .error
        if body is not None and "error" in body:
            self.error = body["error"]
        else:
            self.error = {}


@pytest.fixture
def fake_api_error():
    return FakeAPIStatusError


# ─────────────────────────────────────────────────────────────
# Mock AgentDispatcher — no real LLM calls, controllable traces
# ─────────────────────────────────────────────────────────────

class MockDispatcher:
    """A drop-in for AgentDispatcher that returns canned worker results.

    Tests push canned (agent_type -> result_dict) mappings; dispatch_worker
    returns them verbatim. This lets cycle-state tests drive THINK→EXECUTE
    without any LLM dependency.
    """

    def __init__(self, provider: str = "glm_token_plan", model: str = "auto"):
        self.provider = provider
        self.model = model
        self._canned: dict[str, list[dict]] = {}
        self.dispatch_calls: list[tuple[str, str]] = []
        self._leader_history: list = []

    def set_worker_result(self, agent_type: str, result: dict,
                          copies: int = 1):
        """Queue `copies` canned results for the given agent type."""
        self._canned.setdefault(agent_type, []).extend([result] * copies)

    def dispatch_worker(self, agent_type: str, task: str,
                        tools: list = None, max_turns_override: int = None) -> dict:
        self.dispatch_calls.append((agent_type, task))
        queue = self._canned.get(agent_type, [])
        if queue:
            return queue.pop(0)
        # Default: nothing launched
        return {"agent": agent_type, "response": "", "experiment_launched": False}

    def dispatch_leader(self, task: str, context: dict) -> dict:
        queue = self._canned.get("leader", [])
        if queue:
            return queue.pop(0)
        return {"action": "experiment", "reason": "mock leader"}

    def reset_leader_history(self):
        self._leader_history = []


@pytest.fixture
def mock_dispatcher():
    return MockDispatcher()


# ─────────────────────────────────────────────────────────────
# Temporary workspace — for ToolRegistry / file-writing tests
# ─────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_workspace(tmp_path):
    """A throwaway workspace dir; cleaned up automatically by pytest."""
    (tmp_path / "scripts").mkdir()
    (tmp_path / "outputs").mkdir()
    (tmp_path / "models").mkdir()
    return tmp_path
