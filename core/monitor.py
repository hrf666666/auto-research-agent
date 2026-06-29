"""
AutoResearcher Experiment Monitor

The key innovation: ZERO LLM calls during experiment training.

While your model trains (hours/days), the monitor only does:
- Process alive check (kill -0 PID)
- Log file tail read
- GPU utilization check

This means running AutoResearcher 24/7 costs the same as running it
only during the THINK and REFLECT phases.
"""

import os
import time
import signal
import subprocess
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("autoresearcher.monitor")


class ExperimentMonitor:
    """Zero-LLM experiment monitoring.

    Design principle: During training, the agent is effectively "sleeping"
    at zero cost. It only wakes up (calls LLM) when training completes
    and results need analysis.
    """

    # Maximum allowed experiment runtime (seconds). Processes exceeding this
    # will be force-killed to prevent infinite loops from blocking the agent.
    DEFAULT_MAX_RUNTIME_HOURS = 12

    def __init__(self, poll_interval: int = 900, zero_llm: bool = True,
                 max_runtime_hours: float = None):
        self.poll_interval = poll_interval  # seconds between checks
        # zero_llm is accepted for config compatibility but not stored —
        # the monitor never calls LLMs by design.
        _ = zero_llm
        self.max_runtime = (max_runtime_hours or self.DEFAULT_MAX_RUNTIME_HOURS) * 3600
        self._active_experiments: dict[int, dict] = {}

    def register_experiment(self, pid: int, log_file: str, command: str = "", start_time: float = None):
        """Register an externally-launched experiment for tracking.

        Call this when a process is launched outside of launch_experiment()
        (e.g., via ToolRegistry._exec_launch_experiment) so that
        has_active_experiments() and has_completed_experiments() work correctly.
        """
        if pid and pid not in self._active_experiments:
            self._active_experiments[pid] = {
                "pid": pid,
                "log_file": log_file,
                "start_time": start_time or time.time(),
                "command": command,
                "status": "running",
            }
            logger.info(f"Registered experiment PID={pid} for monitoring")

    def launch_experiment(self, command: str, log_file: str, gpu: Optional[str] = None) -> dict:
        """Launch an experiment via nohup and track its PID.

        Args:
            command: The training command to run
            log_file: Path to redirect stdout/stderr
            gpu: CUDA_VISIBLE_DEVICES value

        Returns:
            dict with pid, log_file, start_time
        """
        env = os.environ.copy()
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)

        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        with open(log_path, "w") as log_f:
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                env=env,
                preexec_fn=os.setsid,  # New process group (survives parent death)
            )

        experiment = {
            "pid": process.pid,
            "log_file": str(log_path),
            "start_time": time.time(),
            "command": command,
            "status": "running",
        }
        self._active_experiments[process.pid] = experiment

        logger.info(f"Launched experiment: PID={process.pid}, cmd={command[:80]}...")
        return experiment

    def wait_for_completion(self, pid: int, log_file: str, notify: bool = True, start_time: float = None) -> dict:
        """Wait for experiment to complete. ZERO LLM calls during wait.

        This is the core cost-saving mechanism. Instead of asking the LLM
        "is training done?", we just check if the process is alive.

        Args:
            start_time: When the experiment was launched (epoch seconds).
                        If None, falls back to _active_experiments or current time.
        """
        logger.info(f"Monitoring PID={pid}, polling every {self.poll_interval}s")

        # Resolve start_time: explicit param > tracked experiment > now
        effective_start = start_time
        if effective_start is None:
            effective_start = self._active_experiments.get(pid, {}).get("start_time", time.time())

        while self._is_process_alive(pid):
            time.sleep(self.poll_interval)

            # Log current status (no LLM involved)
            gpu_info = self._get_gpu_status()
            log_tail = self._tail_file(log_file, lines=5)
            elapsed = time.time() - effective_start

            logger.info(
                f"PID={pid} alive | elapsed={elapsed/3600:.1f}h | "
                f"GPU={gpu_info.get('utilization', 'N/A')} | "
                f"last_log: {log_tail[-1] if log_tail else 'N/A'}"
            )

            # Hard timeout: kill process if it exceeds max allowed runtime
            if elapsed > self.max_runtime:
                logger.warning(
                    f"PID={pid} exceeded max runtime "
                    f"({elapsed/3600:.1f}h > {self.max_runtime/3600:.1f}h) — killing"
                )
                self._kill_process(pid)
                break

        # Experiment finished
        elapsed = time.time() - effective_start
        log_tail = self._tail_file(log_file, lines=50)

        if pid in self._active_experiments:
            self._active_experiments[pid]["status"] = "completed"

        result = {
            "pid": pid,
            "status": "completed",
            "elapsed_hours": elapsed / 3600,
            "log_tail": "\n".join(log_tail),
            "metrics": self._extract_metrics(log_tail),
        }

        logger.info(f"Experiment PID={pid} completed after {result['elapsed_hours']:.1f}h")

        if notify:
            self._notify_completion(result)

        return result

    def has_completed_experiments(self) -> bool:
        """Check if any tracked experiment has finished."""
        for pid, exp in list(self._active_experiments.items()):
            if exp["status"] == "running" and not self._is_process_alive(pid):
                exp["status"] = "completed"
                return True
        self._cleanup_stale_experiments()
        return False

    def has_active_experiments(self) -> bool:
        """Check if any experiment is currently running."""
        for pid, exp in self._active_experiments.items():
            if exp["status"] == "running" and self._is_process_alive(pid):
                return True
        return False

    def _is_process_alive(self, pid: int) -> bool:
        """Check if process is still running (zero cost)."""
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    def _cleanup_stale_experiments(self):
        """Remove completed/failed experiments older than 24h to prevent memory leak.

        Called automatically from has_completed_experiments().
        """
        now = time.time()
        stale_pids = [
            pid for pid, exp in self._active_experiments.items()
            if exp.get("status") in ("completed", "failed")
            and now - exp.get("start_time", 0) > 86400  # 24 hours
        ]
        for pid in stale_pids:
            del self._active_experiments[pid]
        if stale_pids:
            logger.debug(f"Cleaned up {len(stale_pids)} stale experiment(s) from tracker")

    def _kill_process(self, pid: int) -> bool:
        """Force-kill a process and its entire process group.

        Uses SIGTERM first, then SIGKILL to ensure ALL child processes
        (including nohup'd training scripts) are terminated.
        """
        # Try graceful termination of the entire process group first
        try:
            os.killpg(pid, signal.SIGTERM)
            logger.info(f"Sent SIGTERM to process group PGID={pid}")
            time.sleep(2)  # Grace period for cleanup
        except OSError:
            pass

        # Force kill the entire process group
        try:
            os.killpg(pid, signal.SIGKILL)
            logger.info(f"Sent SIGKILL to process group PGID={pid}")
        except OSError:
            pass

        # Fallback: kill just the process
        try:
            os.kill(pid, signal.SIGKILL)
            logger.info(f"Killed process PID={pid}")
            return True
        except OSError:
            return False

    def _get_gpu_status(self) -> dict:
        """Get GPU utilization via nvidia-smi."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                gpus = []
                for line in lines:
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 3:
                        gpus.append({
                            "utilization": f"{parts[0]}%",
                            "memory": f"{parts[1]}MB/{parts[2]}MB",
                        })
                return {"gpus": gpus, "utilization": gpus[0]["utilization"] if gpus else "N/A"}
        except Exception:
            pass
        return {"gpus": [], "utilization": "N/A"}

    def _tail_file(self, filepath: str, lines: int = 50) -> list[str]:
        """Read last N lines of a file (zero cost, memory-efficient)."""
        from collections import deque
        try:
            with open(filepath, "r") as f:
                return list(deque(f, maxlen=lines))
        except Exception:
            return []

    def _extract_metrics(self, log_lines: list[str]) -> dict:
        """Try to extract common metrics from training logs.

        Phase 1: Uses training_log_parser for metric extraction (single source
        of truth), replacing the duplicated regex patterns that were inline
        here. Keeps monitor-specific handling (epoch/step/percentage) that
        the shared parser doesn't cover.
        """
        import re
        from .training_log_parser import extract_metrics

        log_text = "\n".join(log_lines)

        # Use shared parser for val_MAE / rmse / accuracy etc.
        metrics = {}
        parsed = extract_metrics(log_text)
        # extract_metrics returns lowercase keys with float values.
        # Keep as float — do NOT str(), that breaks falsification gate
        # comparisons and fact milestone formatting (Reform v21 fix).
        for k, v in parsed.items():
            metrics[k] = v

        # Monitor-specific patterns not in the shared parser
        for line in reversed(log_lines):
            for pattern, key in [
                (r"loss[:\s]+([0-9.]+)", "loss"),
                (r"acc(?:uracy)?[:\s]+([0-9.]+%?)", "accuracy"),
                (r"FGD[:\s]+([0-9.]+)", "FGD"),
                (r"FID[:\s]+([0-9.]+)", "FID"),
                (r"epoch[:\s]+(\d+)", "epoch"),
                (r"step[:\s]+(\d+)", "step"),
            ]:
                if key not in metrics:
                    match = re.search(pattern, line, re.IGNORECASE)
                    if match:
                        value = match.group(1)
                        if value.endswith('%'):
                            try:
                                value = str(float(value[:-1]) / 100.0)
                            except ValueError:
                                pass
                        metrics[key] = value

        # Fallback: generic key=value pattern for any numeric metric
        if not metrics or len(metrics) < 2:
            for line in reversed(log_lines[-20:]):
                generic_matches = re.findall(r'(\w+)[:\s=]+([0-9.]+)', line)
                for key, value in generic_matches:
                    if key not in metrics and key.lower() not in ('time', 'pid', 'count'):
                        metrics[key] = value

        return metrics

    def _notify_completion(self, result: dict):
        """Send notification when experiment completes."""
        logger.info(
            f"EXPERIMENT COMPLETE | PID={result['pid']} | "
            f"Time={result['elapsed_hours']:.1f}h | "
            f"Metrics={result.get('metrics', {})}"
        )
