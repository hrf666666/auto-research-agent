"""
Install AutoResearcher skills into AI coding tools.

Supported targets:
    --claude-code    Install into ~/.claude/commands/ (Claude Code)
    --codebuddy      Install into .codebuddy/commands/ (CodeBuddy)
    --cursor         Install into .cursor/rules/ (Cursor)
    --all            Install into all supported tools (default)

One-command setup:
    python install.py              # install to all tools
    python install.py --claude-code
    python install.py --codebuddy
    python install.py --cursor
    python install.py --uninstall  # remove from all
"""

import os
import shutil
import sys
from pathlib import Path
from typing import Optional


REPO_DIR = Path(__file__).parent.resolve()
SKILLS_SOURCE = REPO_DIR / "skills"
CORE_SOURCE = REPO_DIR / "core"
GPU_SOURCE = REPO_DIR / "gpu"

# ── Target directory layouts ──────────────────────────────────────────

TARGETS = {
    "claude-code": {
        "label": "Claude Code",
        "skill_dir": Path.home() / ".claude" / "commands",
        "core_dir": Path.home() / ".claude" / "deep-researcher",
        "file_ext": ".md",
        "prefix": "",          # no prefix on command names
    },
    "codebuddy": {
        "label": "CodeBuddy",
        "skill_dir": Path.home() / ".codebuddy" / "commands",
        "core_dir": Path.home() / ".codebuddy" / "deep-researcher",
        "file_ext": ".md",
        "prefix": "",
    },
    "cursor": {
        "label": "Cursor",
        "skill_dir": Path.home() / ".cursor" / "rules",
        "core_dir": Path.home() / ".cursor" / "deep-researcher",
        "file_ext": ".mdc",    # Cursor rule format
        "prefix": "autoresearcher-",  # avoid naming conflicts
    },
}


def _install_skills(target_key: str) -> int:
    """Install skill files to a single target. Returns count installed."""
    t = TARGETS[target_key]
    dest_dir = t["skill_dir"]
    dest_dir.mkdir(parents=True, exist_ok=True)

    installed = 0
    for skill_dir in sorted(SKILLS_SOURCE.iterdir()):
        if skill_dir.is_dir():
            skill_file = skill_dir / "SKILL.md"
            if skill_file.exists():
                dest = dest_dir / f"{t['prefix']}{skill_dir.name}{t['file_ext']}"
                shutil.copy2(skill_file, dest)
                installed += 1
    return installed


def _install_core(target_key: str):
    """Install core Python modules to target."""
    t = TARGETS[target_key]
    core_dest = t["core_dir"] / "core"
    core_dest.mkdir(parents=True, exist_ok=True)
    if CORE_SOURCE.exists():
        for py_file in CORE_SOURCE.glob("*.py"):
            shutil.copy2(py_file, core_dest / py_file.name)

    gpu_dest = t["core_dir"] / "gpu"
    gpu_dest.mkdir(parents=True, exist_ok=True)
    if GPU_SOURCE.exists():
        for py_file in GPU_SOURCE.glob("*.py"):
            shutil.copy2(py_file, gpu_dest / py_file.name)

    # Copy default config
    config_src = REPO_DIR / "config.yaml"
    config_dest = t["core_dir"] / "config.yaml"
    if config_src.exists() and not config_dest.exists():
        shutil.copy2(config_src, config_dest)


def _install_target(target_key: str):
    """Full install for one target."""
    t = TARGETS[target_key]
    count = _install_skills(target_key)
    _install_core(target_key)
    print(f"  [{t['label']}] {count} skills → {t['skill_dir']}")


def install(targets: Optional[list[str]] = None):
    """Install skills to specified targets (or all)."""
    print()
    print("  AutoResearcher — Skill Installer")
    print("  " + "=" * 40)
    print()

    if not targets:
        targets = list(TARGETS.keys())

    total = 0
    for key in targets:
        if key in TARGETS:
            _install_target(key)
            total += 1
        else:
            print(f"  [WARN] Unknown target: {key}")

    print()
    print(f"  Done! Installed to {total} tool(s).")
    print()
    print("  Available commands:")
    print("  ─────────────────────────────────────")
    print("    /auto-experiment     Launch 24/7 experiment loop")
    print("    /code-review         Architecture & code review")
    print("    /code-cleanup        Remove obsolete code")
    print("    /experiment-status   Check experiment progress")
    print("    /gpu-monitor         GPU status & availability")
    print("    /idea-validation     Validate research ideas")
    print("    /model-architect     Design model architecture")
    print("    /paper-research      Research papers for project")
    print("    /paper-analyze       Deep paper analysis")
    print("    /dataset-understanding  Analyze dataset structure")
    print("    /error-handler       Handle experiment errors")
    print("    /progress-report     Generate progress report")
    print()
    print("  Quick start:")
    print("    1. Create a project with PROJECT_BRIEF.md")
    print("    2. Run: /auto-experiment --project <path> --gpu 0")
    print()


def uninstall(targets: Optional[list[str]] = None):
    """Remove installed skills."""
    if not targets:
        targets = list(TARGETS.keys())

    for key in targets:
        if key not in TARGETS:
            continue
        t = TARGETS[key]

        # Remove skill files
        removed = 0
        if t["skill_dir"].exists():
            for skill_dir in sorted(SKILLS_SOURCE.iterdir()):
                if skill_dir.is_dir():
                    dest = t["skill_dir"] / f"{t['prefix']}{skill_dir.name}{t['file_ext']}"
                    if dest.exists():
                        dest.unlink()
                        removed += 1

        # Remove core dir
        if t["core_dir"].exists():
            shutil.rmtree(t["core_dir"])

        print(f"  [{t['label']}] Removed {removed} skills.")


if __name__ == "__main__":
    args = sys.argv[1:]

    if "--uninstall" in args:
        uninstall()
    else:
        selected = []
        for key in TARGETS:
            if f"--{key}" in args:
                selected.append(key)
        if "--all" in args or not selected:
            selected = None  # all
        install(selected)
