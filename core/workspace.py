"""Shared workspace and runtime-state helpers."""
from __future__ import annotations

import fcntl
import logging
import os
import time
from pathlib import Path

logger = logging.getLogger("autoresearcher.workspace")


class WorkspaceLock:
    """Non-blocking process lock held for one ResearchLoop run."""

    def __init__(self, workspace: Path):
        self.workspace = Path(workspace).resolve()
        self.path = self.workspace / ".autoresearcher.lock"
        self._fh = None

    def acquire(self):
        self.workspace.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a+")
        try:
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._fh.seek(0)
            owner = self._fh.read().strip()
            self._fh.close()
            self._fh = None
            raise RuntimeError(
                f"Workspace is already locked: {self.workspace}. Owner: {owner or 'unknown'}"
            ) from exc
        self._fh.seek(0)
        self._fh.truncate()
        self._fh.write(f"pid={os.getpid()} started_at={time.time()} workspace={self.workspace}\n")
        self._fh.flush()
        os.fsync(self._fh.fileno())
        return self

    def release(self):
        if self._fh is None:
            return
        try:
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
        finally:
            self._fh.close()
            self._fh = None

    def __enter__(self):
        return self.acquire()

    def __exit__(self, exc_type, exc, tb):
        self.release()
        return False


def resolve_workspace(project_dir: Path | str, config: dict | None = None,
                      explicit_override: Path | str | None = None) -> Path:
    """Resolve the effective workspace shared by API, loop and tools.

    Precedence: explicit override > config.project.workspace > project_dir.
    Relative paths are resolved against project_dir; absolute paths keep their
    absolute meaning but are logged when they point outside the project.
    """
    project = Path(project_dir).resolve()
    config = config or {}
    raw = explicit_override
    if raw is None:
        raw = (config.get("project", {}) or {}).get("workspace")
    if raw in (None, ""):
        return project

    path = Path(raw).expanduser()
    resolved = path.resolve() if path.is_absolute() else (project / path).resolve()
    try:
        resolved.relative_to(project)
    except ValueError:
        logger.warning(
            "Workspace is outside project_dir: workspace=%s project=%s", resolved, project
        )
    return resolved


def atomic_write_text(path: Path, text: str):
    """Atomically replace a text file with fsync for crash resistance."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    try:
        dir_fd = os.open(str(path.parent), os.O_DIRECTORY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except OSError:
        pass
