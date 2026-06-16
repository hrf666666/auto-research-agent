"""Deterministic garbage collector (Phase 2, P1: safety in the tool).

Replaces the old _auto_code_cleanup which dispatched a code agent (consuming
LLM quota) to clean up files. This GC is pure Python — no LLM involved.

Rules:
  - temp_patterns: files matching these in tools/ are archived (not deleted)
  - output_dirs: when outputs/ exceeds max_output_dirs, oldest non-best archived
  - protected: models/, datasets/, scripts/, PROJECT_BRIEF.md, config.yaml never touched
  - archive_dir: archived files go to archive/temp/ (recoverable, not deleted)
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

logger = logging.getLogger("autoresearcher.gc")


class GarbageCollector:
    """Deterministic cleanup — no LLM, no ambiguity."""

    TEMP_PATTERNS = ("debug_", "diag_", "_check_", "dryrun_", "dry_run_", "test_output_")

    PROTECTED_DIRS = {"models", "datasets", "scripts", "outputs", "logs", "workspace", "tools"}
    PROTECTED_FILES = {"PROJECT_BRIEF.md", "config.yaml", "PERSISTENT_CONSTRAINTS.md",
                       "DATASET_MANIFEST.json", "MEMORY_LOG.md", "experiment_history.db"}

    def __init__(self, project_dir: Path, max_output_dirs: int = 10):
        self.project_dir = project_dir
        self.archive_dir = project_dir / "archive" / "temp"
        self.max_output_dirs = max_output_dirs

    def run(self) -> dict:
        """Run all cleanup rules. Returns a summary of what was archived."""
        summary = {"temp_files_archived": 0, "output_dirs_archived": 0, "errors": []}

        try:
            self.archive_dir.mkdir(parents=True, exist_ok=True)
            summary["temp_files_archived"] = self._archive_temp_files()
            summary["output_dirs_archived"] = self._archive_old_outputs()
        except Exception as e:
            summary["errors"].append(str(e))
            logger.warning(f"GC error: {e}")

        if summary["temp_files_archived"] or summary["output_dirs_archived"]:
            logger.info(f"GC: archived {summary['temp_files_archived']} temp files, "
                        f"{summary['output_dirs_archived']} output dirs")
        return summary

    def _archive_temp_files(self) -> int:
        """Archive diagnostic/debug/dry-run scripts from tools/ and workspace root."""
        count = 0
        # Scan tools/ directory for temp files
        tools_dir = self.project_dir / "tools"
        if tools_dir.exists():
            for f in tools_dir.glob("*.py"):
                if self._is_temp_file(f.name) and not self._is_protected(f):
                    self._archive_file(f)
                    count += 1

        # Scan workspace root for stray temp files
        ws = self.project_dir / "workspace"
        if ws.exists():
            for f in ws.glob("*.py"):
                if self._is_temp_file(f.name) and not self._is_protected(f):
                    self._archive_file(f)
                    count += 1

        return count

    def _archive_old_outputs(self) -> int:
        """Archive oldest output directories when exceeding max_output_dirs."""
        outputs_dir = self.project_dir / "outputs"
        if not outputs_dir.exists():
            return 0

        dirs = sorted(
            [d for d in outputs_dir.iterdir() if d.is_dir()],
            key=lambda d: d.stat().st_mtime
        )

        if len(dirs) <= self.max_output_dirs:
            return 0

        # Archive oldest dirs that don't contain best_model.pth
        to_archive = []
        for d in dirs:
            if len(dirs) - len(to_archive) <= self.max_output_dirs:
                break
            if not (d / "best_model.pth").exists():
                to_archive.append(d)

        count = 0
        for d in to_archive:
            try:
                dest = self.archive_dir / d.name
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.move(str(d), str(dest))
                count += 1
            except Exception as e:
                logger.debug(f"GC: could not archive {d.name}: {e}")

        return count

    def _is_temp_file(self, name: str) -> bool:
        """Check if a filename matches temp patterns."""
        return any(name.startswith(p) for p in self.TEMP_PATTERNS)

    def _is_protected(self, path: Path) -> bool:
        """Check if a path is in the protected list."""
        if path.name in self.PROTECTED_FILES:
            return True
        try:
            rel = path.relative_to(self.project_dir)
            return rel.parts[0] in self.PROTECTED_DIRS and rel.parts[0] != "tools"
        except ValueError:
            return True

    def _archive_file(self, path: Path):
        """Move a file to the archive directory (recoverable)."""
        dest = self.archive_dir / path.name
        if dest.exists():
            dest.unlink()  # overwrite previous archive
        shutil.move(str(path), str(dest))
