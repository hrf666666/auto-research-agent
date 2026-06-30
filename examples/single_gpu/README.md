# Single GPU Setup Guide

This guide shows how to set up AutoResearcher for a single-GPU research project.

## 1. Create Your Project

```bash
mkdir my_project && cd my_project
```

## 2. Write a Project Brief

Create `PROJECT_BRIEF.md` describing your research:

```markdown
# Goal
Improve depth estimation accuracy on [dataset]. Target: val_MAE < 0.15.

# Codebase
- models/: PyTorch model definitions (nn.Module subclasses)
- datasets/: Dataset loaders (torch Dataset subclasses)
- data/: Training/validation data

# Constraints
- Single GPU (RTX 3090, 24GB)
- Max 50 epochs per experiment

# Success Criteria
val_MAE on validation set < 0.15
```

## 3. Create Config

Copy `config.yaml` from the repo root and adjust:

```bash
cp /path/to/auto_research_agent/config.yaml .
# Edit project.name, workspace (use "." for current dir), goals.metrics
```

## 4. Add Your Data and Code

```bash
mkdir -p data/ models/ datasets/ scripts/
# Place your dataset under data/, model code under models/, loaders under datasets/
```

## 5. Launch

```bash
# Run 10 cycles synchronously (recommended for first run)
python api.py run --project . --cycles 10

# Or run as a background daemon (24/7)
python api.py start --project . --gpu 0 --max-cycles -1
python api.py stop   # to stop
```

## 6. Monitor

- `workspace/MEMORY_LOG.md` — human-readable progress (milestones, decisions, dead ends)
- `autoresearcher.log` — detailed logs (VERIFY results, provider failovers)
- `workspace/state.json` — current cycle/status/metrics snapshot
- `python api.py status --project .` — JSON status

## Tips

- Start with `--cycles 3` to verify everything works before a long run
- Keep `PROJECT_BRIEF.md` concise (under 3000 chars)
- Use `DIRECTIVE.md` to redirect the agent if it goes off track
- Set `GLM_CODING_PLAN_API_KEY` (or `ALI_TOKEN_PLAN_API_KEY`) before launching
