# Auto Research Agent

> An autonomous AI research agent that conducts ML experiments end-to-end —
> from idea and dataset to validated results — at a PhD researcher's level.

[English](README.md) | [中文](docs/README_CN.md)

---

## What It Does

Give it a **research brief** (your hypothesis + target metrics) and a **dataset**.
The agent will:

1. **Understand the data** — scan dataset structure, generate a manifest, identify quality issues
2. **Survey existing approaches** — search papers, analyze transferable ideas across domains
3. **Design experiments** — formulate falsifiable hypotheses, plan model architectures
4. **Implement & train** — write code, run dry-runs, launch training on GPU
5. **Verify results** — 12-layer automated verification (not just "did it run")
6. **Reflect & iterate** — record structured outcomes, learn from failures, adjust direction
7. **Stop when done** — automatically halts when target metrics are achieved

All autonomously. No human intervention between "here's my idea" and "here are your results."

---

## Quick Start

```bash
# Install dependencies (Python 3.11+)
pip install -r requirements.txt

# Configure your LLM provider
export GLM_CODING_PLAN_API_KEY="your-key-here"

# Run the agent on your project
python -m core.loop --project /path/to/your/project
```

Your project directory needs:
- `PROJECT_BRIEF.md` — your research hypothesis, target metrics, methodology notes
- A dataset (the agent will auto-discover and analyze it)
- `config.yaml` — agent configuration (copy from this repo and edit)

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                     run() cycle                       │
│                                                       │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐ │
│  │  THINK  │→│ EXECUTE │→│ VERIFY  │→│ REFLECT │ │
│  │ (plan)  │  │(code+  │  │(12-layer│  │(learn+ │ │
│  │         │  │ train)  │  │ check)  │  │ record)│ │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘ │
│       ↕             ↕            ↕            ↕      │
│  ┌─────────────────────────────────────────────────┐ │
│  │              Memory (SQLite + MEMORY_LOG)        │ │
│  │  experiment records · causal chains · lessons    │ │
│  │  goal progress · dead ends · pareto frontier     │ │
│  └─────────────────────────────────────────────────┘ │
│       ↕             ↕            ↕            ↕      │
│  ┌─────────────────────────────────────────────────┐ │
│  │              Tool Layer (self-enforcing)         │ │
│  │  write_file · launch_experiment · analyze_model  │ │
│  │  search_papers · check_feasibility · GC          │ │
│  └─────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────┘
```

**Design principles:**
- **System** = hard constraints (safety, lifecycle, tools, memory)
- **Prompt** = research methodology guidance (how to think, not what to do)
- **LLM** = the PhD brain (design, implement, judge, iterate)

See [docs/architecture.md](docs/architecture.md) for full module documentation
and [version history](docs/architecture.md#36-v17--systemic-architecture-fixes-5-root-causes-97-tests).

---

## Key Features

- **Provider failover**: GLM-5.x → Qwen fallback with quota-aware cooldown
- **Anti-deception**: tool-trace-based verification (LLM can't fake results)
- **Goal tracking**: auto-stops when target metrics are met
- **IdeaScout** (optional): cross-domain idea discovery via [research-idea-scout](https://github.com/YangyangQu/research-idea-scout)
- **Structured memory**: quantitative results written deterministically (not lost to LLM paraphrase)
- **134 automated tests** covering dispatch, safety, memory, enforcement, and knowledge loops

---

## Configuration

See [config.yaml](config.yaml) for all options. Key sections:

```yaml
goals:               # Target metrics (auto-stop when achieved)
  metrics:
    - key: "val_MAE"
      target: 0.20
      direction: "lower"

safety:              # Tool-level safety contracts
  naming: { forbidden_root_py: true }
  garbage_collection: { max_output_dirs: 10 }

idea_scout:          # Cross-domain idea discovery (disabled by default)
  enabled: false
```

---

## License

MIT
