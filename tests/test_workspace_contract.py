"""Workspace and API cycle contract tests."""
from __future__ import annotations

import json

from api import AutoResearcher
from core.loop import ResearchLoop
from core.workspace import resolve_workspace


def test_resolve_workspace_relative_to_project(tmp_path):
    project = tmp_path / "proj"
    project.mkdir()
    assert resolve_workspace(project, {"project": {"workspace": "workspace"}}) == project / "workspace"


def test_api_uses_config_workspace_for_status(tmp_path):
    project = tmp_path / "proj"
    project.mkdir()
    (project / "PROJECT_BRIEF.md").write_text("# brief")
    (project / "config.yaml").write_text("project:\n  workspace: workspace\n")
    workspace = project / "workspace"
    workspace.mkdir()
    (workspace / ".cycle_counter").write_text("7")
    (workspace / "state.json").write_text(json.dumps({"pid": 999999}))

    status = AutoResearcher(str(project)).get_status()

    assert status["workspace"] == str(workspace.resolve())
    assert status["cycle"] == 7


def test_research_loop_passes_config_to_tools(tmp_path):
    project = tmp_path / "proj"
    project.mkdir()
    (project / "PROJECT_BRIEF.md").write_text("# brief")
    config = {
        "project": {"workspace": "workspace"},
        "safety": {"mandatory_dry_run": True},
        "agent": {"max_cycles": 1},
    }

    loop = ResearchLoop(config=config, project_dir=str(project))
    try:
        assert loop.workspace == (project / "workspace").resolve()
        assert loop.tools._mandatory_dry_run is True
    finally:
        loop.tools.shutdown()
