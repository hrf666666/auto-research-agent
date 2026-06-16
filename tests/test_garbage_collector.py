"""Tests for Phase 2: deterministic garbage collector."""
from __future__ import annotations

from pathlib import Path

import pytest

from core.garbage_collector import GarbageCollector


@pytest.fixture
def project(tmp_path):
    """Create a realistic project structure with temp files."""
    (tmp_path / "tools").mkdir()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "models").mkdir()
    (tmp_path / "datasets").mkdir()
    (tmp_path / "outputs").mkdir()
    (tmp_path / "tools" / "debug_shapes.py").write_text("x=1")
    (tmp_path / "tools" / "diag_model.py").write_text("x=1")
    (tmp_path / "tools" / "_check_grad.py").write_text("x=1")
    (tmp_path / "tools" / "helper.py").write_text("x=1")  # not temp
    (tmp_path / "tools" / "dryrun_test.py").write_text("x=1")
    (tmp_path / "scripts" / "train_model.py").write_text("x=1")  # protected
    (tmp_path / "models" / "net.py").write_text("x=1")  # protected
    (tmp_path / "PROJECT_BRIEF.md").write_text("brief")  # protected
    return tmp_path


class TestGarbageCollector:

    def test_archives_temp_files(self, project):
        gc = GarbageCollector(project)
        summary = gc.run()
        assert summary["temp_files_archived"] >= 3  # debug_ diag_ _check_ dryrun_

    def test_does_not_archive_non_temp(self, project):
        gc = GarbageCollector(project)
        gc.run()
        # helper.py is not a temp file
        assert (project / "tools" / "helper.py").exists()

    def test_does_not_touch_protected(self, project):
        gc = GarbageCollector(project)
        gc.run()
        assert (project / "scripts" / "train_model.py").exists()
        assert (project / "models" / "net.py").exists()
        assert (project / "PROJECT_BRIEF.md").exists()

    def test_archive_is_recoverable(self, project):
        gc = GarbageCollector(project)
        gc.run()
        # Archived files should be in archive/temp/
        archived = list(gc.archive_dir.glob("*.py"))
        assert len(archived) >= 3
        # Content preserved
        for f in archived:
            assert f.read_text() == "x=1"

    def test_old_output_dirs_archived(self, project):
        # Create 15 output dirs, only 2 with best_model.pth
        import time
        for i in range(15):
            d = project / "outputs" / f"exp_{i}"
            d.mkdir()
            (d / "log.txt").write_text(f"exp {i}")
            if i >= 13:  # last 2 have best_model
                (d / "best_model.pth").write_text("model")
            time.sleep(0.01)  # ensure different mtimes

        gc = GarbageCollector(project, max_output_dirs=10)
        summary = gc.run()
        assert summary["output_dirs_archived"] >= 3  # 15 - 10 - 2(protected) = 3

    def test_empty_project_no_error(self, tmp_path):
        gc = GarbageCollector(tmp_path)
        summary = gc.run()
        assert summary["temp_files_archived"] == 0
        assert summary["output_dirs_archived"] == 0
        assert summary["errors"] == []
