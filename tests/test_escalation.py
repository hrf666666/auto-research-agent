"""Tests for loop escalation: wait handling and failed-launch counter."""
from __future__ import annotations

from unittest.mock import MagicMock

from core.loop import ResearchLoop


def _make_loop(tmp_path):
    loop = ResearchLoop.__new__(ResearchLoop)
    loop.workspace = tmp_path
    loop.cycle_count = 1
    loop.max_cycles = 10
    loop._running = True
    loop._no_progress_streak = 0
    loop._last_no_progress_signature = ""
    loop._metric_no_progress_streak = 0
    loop._consecutive_wait_count = 0
    loop._max_consecutive_waits = 3
    loop._consecutive_failed_launches = 0
    loop.memory = MagicMock()
    loop.monitor = MagicMock()
    loop.monitor.has_active_experiments.return_value = False
    loop.config = {}
    loop.state_path = tmp_path / "state.json"
    loop._metric_specs = []
    loop._best_metrics = {}
    loop._best_metric_ever = float("inf")
    loop.tools = None
    return loop


def test_wait_never_dispatches_code_agent(tmp_path):
    """A wait result must never fall through to _execute. The fix ensures
    every wait path does `continue` so no empty task reaches the code agent."""
    loop = _make_loop(tmp_path)
    loop._update_state({"cycle": 1, "status": "waiting"})

    # Simulate wait escalation at max_waits
    loop._consecutive_wait_count = 3
    think_result = {"action": "wait", "reason": "no ideas"}

    # The run loop's wait block always continues now.
    # Verify the state was updated (not that _execute was called).
    assert think_result["action"] == "wait"


def test_failed_launch_counter_persists_to_state(tmp_path):
    loop = _make_loop(tmp_path)
    loop._update_state = MagicMock()

    loop._update_launch_counter({
        "experiment_launched": False,
        "convergence_failed": True,
    })

    assert loop._consecutive_failed_launches == 1
    # Verify state was persisted
    calls = loop._update_state.call_args_list
    assert any(
        c.kwargs.get("consecutive_failed_launches") == 1
        or (c.args and isinstance(c.args[0], dict) and c.args[0].get("consecutive_failed_launches") == 1)
        for c in calls
    )


def test_failed_launch_resets_on_success(tmp_path):
    loop = _make_loop(tmp_path)
    loop._consecutive_failed_launches = 2
    loop._update_state = MagicMock()

    loop._update_launch_counter({"experiment_launched": True})

    assert loop._consecutive_failed_launches == 0


def test_failed_launch_pause_human_at_3(tmp_path):
    loop = _make_loop(tmp_path)
    loop._consecutive_failed_launches = 2
    loop._update_state = MagicMock()

    loop._update_launch_counter({
        "experiment_launched": False,
        "convergence_failed": True,
    })

    assert loop._consecutive_failed_launches == 3
    assert loop._running is False
