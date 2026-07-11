"""Experiment monitor terminal status tests."""
from __future__ import annotations

import subprocess
import sys

from core.monitor import ExperimentMonitor


def test_monitor_reports_completed_for_exit_zero(tmp_path):
    log = tmp_path / "train.log"
    proc = subprocess.Popen([sys.executable, "-c", "print('val_MAE=0.1')"], stdout=log.open("w"), stderr=subprocess.STDOUT)
    mon = ExperimentMonitor(poll_interval=0, max_runtime_hours=1)
    mon.register_experiment(proc.pid, str(log), process=proc)

    result = mon.wait_for_completion(proc.pid, str(log), notify=False, process=proc)

    assert result["status"] == "completed"
    assert result["exit_code"] == 0
    assert result["timed_out"] is False


def test_monitor_reports_failed_for_nonzero_exit(tmp_path):
    log = tmp_path / "train.log"
    proc = subprocess.Popen([sys.executable, "-c", "raise SystemExit(3)"], stdout=log.open("w"), stderr=subprocess.STDOUT)
    mon = ExperimentMonitor(poll_interval=0, max_runtime_hours=1)
    mon.register_experiment(proc.pid, str(log), process=proc)

    result = mon.wait_for_completion(proc.pid, str(log), notify=False, process=proc)

    assert result["status"] == "failed"
    assert result["exit_code"] == 3


def test_monitor_reports_timed_out(tmp_path):
    log = tmp_path / "train.log"
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(10)"], stdout=log.open("w"), stderr=subprocess.STDOUT, start_new_session=True)
    mon = ExperimentMonitor(poll_interval=0, max_runtime_hours=0.000001)
    mon.register_experiment(proc.pid, str(log), process=proc)

    result = mon.wait_for_completion(proc.pid, str(log), notify=False, start_time=0, process=proc)

    assert result["status"] == "timed_out"
    assert result["timed_out"] is True
