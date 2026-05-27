"""
AutoResearcher — Python API for external tool integration.

Allows CodeBuddy, Cursor, Claude Code, and any Python tool to invoke
the autonomous research agent programmatically.

Usage:
    from api import AutoResearcher

    # Create researcher for a project
    researcher = AutoResearcher("/path/to/project")

    # Run one cycle
    result = researcher.run_one_cycle()

    # Run N cycles
    results = researcher.run_n_cycles(5)

    # Get project status
    status = researcher.get_status()

    # Start daemon (non-blocking background loop)
    researcher.start_daemon(gpu=0, max_cycles=10)
    researcher.stop_daemon()
"""

import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger("autoresearcher.api")

# Ensure core is importable
_REPO_DIR = Path(__file__).parent.resolve()
if str(_REPO_DIR) not in sys.path:
    sys.path.insert(0, str(_REPO_DIR))


class AutoResearcher:
    """Programmatic interface to the AutoResearcher agent.

    This class can be used by any AI coding tool (CodeBuddy, Cursor,
    Claude Code, etc.) to integrate autonomous ML experimentation
    into their workflow.
    """

    def __init__(self, project_dir: str, config_path: str = "config.yaml",
                 workspace: str = None):
        """Initialize the researcher.

        Args:
            project_dir: Path to the project directory (must contain PROJECT_BRIEF.md).
            config_path: Config file name relative to project_dir.
            workspace: Optional workspace path (defaults to project_dir).
        """
        self.project_dir = Path(project_dir).resolve()
        self.config_path = config_path
        self.workspace = Path(workspace).resolve() if workspace else self.project_dir
        self._daemon_process: Optional[subprocess.Popen] = None
        self._daemon_log = None

        if not (self.project_dir / "PROJECT_BRIEF.md").exists():
            raise FileNotFoundError(
                f"PROJECT_BRIEF.md not found in {self.project_dir}. "
                f"Create one to describe your research goal."
            )

    # ── Synchronous single-cycle API ──────────────────────────────

    def run_one_cycle(self) -> dict:
        """Run exactly one THINK→EXECUTE→VERIFY→REFLECT cycle.

        Returns:
            dict with keys: cycle, action, experiment_launched, metrics,
                            milestone, dead_end, errors
        """
        from core.loop import ResearchLoop

        config = self._load_config()
        config.setdefault("agent", {})["max_cycles"] = 1

        loop = ResearchLoop(config=config, project_dir=str(self.project_dir))
        loop.run()

        return self._extract_last_result()

    def run_n_cycles(self, n: int = 5) -> list[dict]:
        """Run N cycles and return results.

        Args:
            n: Number of cycles to run.

        Returns:
            List of result dicts (one per cycle).
        """
        from core.loop import ResearchLoop

        config = self._load_config()
        config.setdefault("agent", {})["max_cycles"] = n

        loop = ResearchLoop(config=config, project_dir=str(self.project_dir))
        loop.run()

        return self._get_all_results(limit=n)

    # ── Daemon (background process) API ───────────────────────────

    def start_daemon(self, gpu: str = None, max_cycles: int = -1) -> int:
        """Start the agent as a background daemon process.

        Args:
            gpu: GPU device(s) to use (e.g. "0" or "0,1").
            max_cycles: Max cycles (-1 for unlimited).

        Returns:
            PID of the daemon process.
        """
        if self._daemon_process and self._daemon_process.poll() is None:
            raise RuntimeError(f"Daemon already running (PID {self._daemon_process.pid})")

        cmd = [sys.executable, "-m", "core.loop", "--project", str(self.project_dir)]
        if gpu:
            cmd.extend(["--gpu", gpu])
        if max_cycles > 0:
            cmd.extend(["--max-cycles", str(max_cycles)])

        env = os.environ.copy()
        env["PYTHONPATH"] = str(_REPO_DIR)

        self._daemon_log = open(self.project_dir / "autoresearcher.log", "a")
        self._daemon_process = subprocess.Popen(
            cmd,
            cwd=str(_REPO_DIR),
            env=env,
            stdout=self._daemon_log,
            stderr=self._daemon_log,
        )

        logger.info(f"Daemon started: PID {self._daemon_process.pid}")
        return self._daemon_process.pid

    def stop_daemon(self):
        """Stop the background daemon process."""
        if self._daemon_process and self._daemon_process.poll() is None:
            self._daemon_process.terminate()
            try:
                self._daemon_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._daemon_process.kill()
            logger.info("Daemon stopped")
        if self._daemon_log and not self._daemon_log.closed:
            self._daemon_log.close()

    def is_daemon_running(self) -> bool:
        """Check if daemon is running."""
        return self._daemon_process is not None and self._daemon_process.poll() is None

    # ── Status / Query API ────────────────────────────────────────

    def get_status(self) -> dict:
        """Get current project status.

        Returns:
            dict with: cycle, best_metrics, dead_ends, active_problems,
                       experiment_running, last_action
        """
        result = {
            "project": str(self.project_dir),
            "cycle": 0,
            "best_metrics": {},
            "dead_ends": [],
            "active_problems": [],
            "experiment_running": False,
            "last_action": None,
        }

        # Read cycle counter
        counter_path = self.project_dir / ".cycle_counter"
        if counter_path.exists():
            try:
                result["cycle"] = int(counter_path.read_text().strip())
            except ValueError:
                pass

        # Read memory log
        from core.memory import MemoryManager
        try:
            mm = MemoryManager(project_dir=self.project_dir, workspace=self.workspace)
            history = mm.get_experiment_history(limit=5)
            if history:
                last = history[0]
                result["last_action"] = last.get("action", "")
                try:
                    metrics = json.loads(last.get("metrics_json", "{}"))
                    result["best_metrics"] = metrics
                except (json.JSONDecodeError, TypeError):
                    pass
        except Exception:
            pass

        # Check running process
        result["experiment_running"] = self._check_training_running()

        return result

    def get_experiment_history(self, limit: int = 20) -> list[dict]:
        """Get experiment history from SQLite.

        Args:
            limit: Max number of records to return.

        Returns:
            List of experiment records.
        """
        from core.memory import MemoryManager
        mm = MemoryManager(project_dir=self.project_dir, workspace=self.workspace)
        return mm.get_experiment_history(limit=limit)

    def get_code_review_lessons(self, severity: str = None,
                                 category: str = None,
                                 limit: int = 30) -> list[dict]:
        """Get learned code review lessons.

        Args:
            severity: Filter by minimum severity ("HIGH", "MEDIUM", "LOW").
            category: Filter by category.
            limit: Max results.

        Returns:
            List of lesson dicts.
        """
        from core.memory import MemoryManager
        mm = MemoryManager(project_dir=self.project_dir, workspace=self.workspace)
        return mm.get_code_review_lessons(severity=severity, category=category, limit=limit)

    # ── Internal helpers ──────────────────────────────────────────

    def _load_config(self) -> dict:
        """Load YAML config from project dir."""
        import yaml
        config_file = self.project_dir / self.config_path
        if config_file.exists():
            with open(config_file) as f:
                return yaml.safe_load(f) or {}
        return {}

    def _extract_last_result(self) -> dict:
        """Extract the last cycle result from SQLite."""
        try:
            from core.memory import MemoryManager
            mm = MemoryManager(project_dir=self.project_dir, workspace=self.workspace)
            history = mm.get_experiment_history(limit=1)
            if history:
                return dict(history[0])
        except Exception:
            pass
        return {"cycle": 0, "action": "unknown"}

    def _get_all_results(self, limit: int = 5) -> list[dict]:
        """Get recent results from SQLite."""
        try:
            from core.memory import MemoryManager
            mm = MemoryManager(project_dir=self.project_dir, workspace=self.workspace)
            return mm.get_experiment_history(limit=limit)
        except Exception:
            return []

    def _check_training_running(self) -> bool:
        """Check if a training process is currently running."""
        try:
            state_file = self.project_dir / "state.json"
            if state_file.exists():
                with open(state_file) as f:
                    state = json.load(f)
                pid = state.get("pid")
                if pid:
                    os.kill(pid, 0)  # raises if process doesn't exist
                    return True
        except (ProcessLookupError, PermissionError, json.JSONDecodeError, FileNotFoundError):
            pass
        return False


# ── CLI entry point for direct tool invocation ────────────────────

def cli():
    """Command-line interface for tool integration.

    Usage:
        python api.py status --project /path/to/project
        python api.py run --project /path/to/project --cycles 1
        python api.py start --project /path/to/project --gpu 0
        python api.py stop
        python api.py lessons --project /path/to/project --severity HIGH
    """
    import argparse

    parser = argparse.ArgumentParser(description="AutoResearcher CLI")
    sub = parser.add_subparsers(dest="command")

    # status
    p = sub.add_parser("status", help="Get project status")
    p.add_argument("--project", required=True, help="Project directory")

    # run
    p = sub.add_parser("run", help="Run N cycles synchronously")
    p.add_argument("--project", required=True)
    p.add_argument("--cycles", type=int, default=1)

    # start daemon
    p = sub.add_parser("start", help="Start background daemon")
    p.add_argument("--project", required=True)
    p.add_argument("--gpu", default=None)
    p.add_argument("--max-cycles", type=int, default=-1)

    # stop daemon
    p = sub.add_parser("stop", help="Stop background daemon")

    # lessons
    p = sub.add_parser("lessons", help="Show learned code review lessons")
    p.add_argument("--project", required=True)
    p.add_argument("--severity", default=None, choices=["HIGH", "MEDIUM", "LOW"])
    p.add_argument("--limit", type=int, default=20)

    args = parser.parse_args()

    if args.command == "status":
        r = AutoResearcher(args.project)
        print(json.dumps(r.get_status(), indent=2, ensure_ascii=False))

    elif args.command == "run":
        r = AutoResearcher(args.project)
        results = r.run_n_cycles(args.cycles)
        print(json.dumps(results, indent=2, ensure_ascii=False, default=str))

    elif args.command == "start":
        r = AutoResearcher(args.project)
        pid = r.start_daemon(gpu=args.gpu, max_cycles=args.max_cycles)
        print(f"Daemon started: PID {pid}")

    elif args.command == "stop":
        # Find and stop running daemon
        result = subprocess.run(
            ["pgrep", "-f", "core.loop.*--project"],
            capture_output=True, text=True
        )
        for pid_str in result.stdout.strip().split("\n"):
            if pid_str.strip():
                os.kill(int(pid_str.strip()), 15)
                print(f"Stopped PID {pid_str.strip()}")

    elif args.command == "lessons":
        r = AutoResearcher(args.project)
        lessons = r.get_code_review_lessons(severity=args.severity, limit=args.limit)
        print(json.dumps(lessons, indent=2, ensure_ascii=False))

    else:
        parser.print_help()


if __name__ == "__main__":
    cli()
