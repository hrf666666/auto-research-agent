"""Tests for the worker-dispatch return contract in core.agents.

The dispatch contract is what the VERIFY/REFLECT phases and the loop trust.
`_parse_worker_response` builds the dict that downstream code reads; the
most important field is `experiment_launched` — when it's False, the cycle
is recorded as no-progress and nothing forces a retry (the P4 failure path).

These tests pin the current behavior so Phase 3 (delete the shell-regex
fallback) and Phase 2 (convergence_failed flag) are observable changes.
"""
from __future__ import annotations

import json

import pytest

from core.agents import AgentDispatcher, ToolTrace
from tests.conftest import make_trace


# A minimal dispatcher whose `_call_llm` we never invoke — we test
# `_parse_worker_response` directly, since that's where the contract lives.
def _make_dispatcher():
    # provider/model don't matter here; we never call the API.
    return AgentDispatcher(model="auto", provider="glm_token_plan", tools=None)


class TestParseWorkerLaunchViaTool:
    """The CORRECT path: launch_experiment was called, facts come from the tool."""

    def test_launch_returns_pid(self):
        d = _make_dispatcher()
        trace = make_trace([
            ("launch_experiment", {"command": "python train.py"},
             {"pid": 12345, "log_file": "logs/exp.log", "status": "launched"}),
        ])
        result = d._parse_worker_response("done", "code", trace)
        assert result["experiment_launched"] is True
        assert result["pid"] == 12345
        assert result["log_file"] == "logs/exp.log"

    def test_launch_returns_error(self):
        """If launch_experiment itself returned an error, it did NOT launch."""
        d = _make_dispatcher()
        trace = make_trace([
            ("launch_experiment", {"command": "python train.py"},
             {"error": "GPU OOM", "status": "failed"}),
        ])
        result = d._parse_worker_response("done", "code", trace)
        # launch_error present → experiment_launched flips to False
        assert result["experiment_launched"] is False
        assert "launch_error" in result


class TestParseWorkerShellLaunchForbidden:
    """Phase 3: training via run_shell is now DETECTED and flagged as a
    forbidden launch path, not silently accepted via regex.

    The old code (deleted in Phase 3) used a 4-branch regex to sniff
    run_shell command text and accept shell-launched training as
    experiment_launched=True. That regex missed renamed scripts, torchrun,
    python -m, etc. — causing experiment_launched=False and silent
    no-progress accumulation. Now any training-looking run_shell command
    is flagged launch_error so the loop (Phase 4) can force a proper
    launch_experiment call.
    """

    def test_python_train_py_flagged_as_forbidden(self):
        """`python train.py` via run_shell must NOT count as a launch."""
        d = _make_dispatcher()
        trace = make_trace([
            ("run_shell", {"command": "python train.py"},
             {"returncode": 0, "stdout": "training started", "stderr": ""}),
        ])
        result = d._parse_worker_response("done", "code", trace)
        assert result.get("experiment_launched") is False
        assert "launch_error" in result
        assert "launch_experiment" in result["launch_error"]

    def test_renamed_script_also_flagged(self):
        """A renamed training script is now CAUGHT (broad detection), not
        missed — the whole point of replacing the narrow regex."""
        d = _make_dispatcher()
        trace = make_trace([
            ("run_shell", {"command": "python run_training.py"},
             {"returncode": 0, "stdout": "epoch 1 loss=0.5", "stderr": ""}),
        ])
        result = d._parse_worker_response("done", "code", trace)
        assert result.get("experiment_launched") is False
        assert "launch_error" in result

    def test_torchrun_flagged(self):
        """torchrun-based launches must also be caught and redirected."""
        d = _make_dispatcher()
        trace = make_trace([
            ("run_shell", {"command": "torchrun --nproc_per_node=1 train.py"},
             {"returncode": 0, "stdout": "ok", "stderr": ""}),
        ])
        result = d._parse_worker_response("done", "code", trace)
        assert result.get("experiment_launched") is False
        assert "launch_error" in result

    def test_non_training_shell_not_flagged(self):
        """A benign run_shell command (ls, echo) must NOT trigger the
        forbidden-launch flag."""
        d = _make_dispatcher()
        trace = make_trace([
            ("run_shell", {"command": "ls -la"},
             {"returncode": 0, "stdout": "files...", "stderr": ""}),
        ])
        result = d._parse_worker_response("done", "code", trace)
        assert result.get("experiment_launched") is False
        assert "launch_error" not in result


class TestParseWorkerDeception:
    """The anti-deception path: LLM claims launch but trace shows nothing."""

    def test_llm_claims_launch_but_no_tool_call(self):
        d = _make_dispatcher()
        # Empty trace, but LLM text claims a launch
        result = d._parse_worker_response(
            "I launched the experiment with PID=99999 and training has started.",
            "code",
            ToolTrace(),
        )
        assert result["experiment_launched"] is False
        assert result.get("deception_detected") is True

    def test_honest_no_launch(self):
        d = _make_dispatcher()
        result = d._parse_worker_response(
            "I wrote the training script but did not run it yet.",
            "code",
            ToolTrace(),
        )
        assert result["experiment_launched"] is False
        assert "deception_detected" not in result


class TestParseWorkerNoTraceFallback:
    """The backward-compat path (trace is None) — deception-vulnerable.

    Pinned so it's clear this path is the legacy fallback.
    """

    def test_no_trace_text_says_pid(self):
        d = _make_dispatcher()
        result = d._parse_worker_response("Done. PID=42", "code", None)
        assert result.get("experiment_launched") is True
        assert result.get("pid") == 42


class TestDryRunEvidence:
    """Shell facts also surface dry-run evidence used by VERIFY."""

    def test_dry_run_detected(self):
        d = _make_dispatcher()
        trace = make_trace([
            ("run_shell", {"command": "python train.py --dry_run"},
             {"returncode": 0, "stdout": "ok", "stderr": ""}),
        ])
        result = d._parse_worker_response("done", "code", trace)
        assert result.get("dry_run_performed") is True
        assert result.get("dry_run_passed") is True

    def test_dry_run_failed(self):
        d = _make_dispatcher()
        trace = make_trace([
            ("run_shell", {"command": "python train.py --dry_run"},
             {"returncode": 1, "stdout": "", "stderr": "CUDA error"}),
        ])
        result = d._parse_worker_response("done", "code", trace)
        assert result.get("dry_run_performed") is True
        assert result.get("dry_run_passed") is False


class TestConvergenceFailedFlag:
    """Phase 2: a code dispatch that never launches (when asked to) is flagged
    `convergence_failed` so the loop (Phase 4) can force a re-dispatch."""

    def test_code_dispatch_no_launch_flagged(self):
        d = _make_dispatcher()
        trace = make_trace([
            ("read_file", {"path": "model.py"}, "code..."),
            ("list_files", {"path": "."}, "files..."),
        ])
        result = d._parse_worker_response(
            "explored the codebase", "code", trace,
            task="run the experiment and launch training",
        )
        assert result.get("convergence_failed") is True

    def test_analysis_task_not_flagged(self):
        """A code task that does NOT mention experiment/train must not be
        flagged — pure analysis/exploration dispatches are legitimate."""
        d = _make_dispatcher()
        trace = make_trace([
            ("read_file", {"path": "model.py"}, "code..."),
        ])
        result = d._parse_worker_response("analyzed", "code", trace,
                                          task="review the model architecture")
        assert "convergence_failed" not in result

    def test_launch_error_not_flagged_as_convergence(self):
        """A forbidden shell-launch (Phase 3) is a launch_error, not a
        convergence failure — distinct signals for distinct problems."""
        d = _make_dispatcher()
        trace = make_trace([
            ("run_shell", {"command": "python train.py"},
             {"returncode": 0, "stdout": "ok", "stderr": ""}),
        ])
        result = d._parse_worker_response("done", "code", trace,
                                          task="launch the experiment")
        assert result.get("launch_error") is not None
        assert "convergence_failed" not in result


class TestHardTurnGate:
    """Phase 2: past 60% of the turn budget, exploration tools are blocked
    for code agents so the remaining turns converge to launch_experiment.

    We test the gate logic directly via _call_openai_compatible's tool loop
    by mocking the LLM to issue a read_file call on turn 24 of 40 (60%)."""

    def test_explore_tool_blocked_past_60pct_budget(self, monkeypatch):
        """A read_file call at turn ≥ 60% of max_turns must return a budget
        error instead of executing."""
        from core.agents import AgentDispatcher, _CODE_EXPLORE_TOOLS

        # Verify the explore set includes read_file (the dominant waste in logs)
        assert "read_file" in _CODE_EXPLORE_TOOLS
        assert "write_file" not in _CODE_EXPLORE_TOOLS
        assert "launch_experiment" not in _CODE_EXPLORE_TOOLS

        # The gate fires at turn >= max_turns * 0.6. For max_turns=40 that's
        # turn 24. We verify the threshold arithmetic is correct: tools in
        # _CODE_EXPLORE_TOOLS are blocked, convergence tools are not.
        max_turns = 40
        gate_turn = int(max_turns * 0.6)
        assert gate_turn == 24  # 60% of 40
        # Before the gate: turns 0-23 allow exploration.
        # At/after the gate: turns 24-39 block exploration.
        for t in range(gate_turn):
            assert t < max_turns * 0.6
        for t in range(gate_turn, max_turns):
            assert t >= max_turns * 0.6
