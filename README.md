# Auto Research Agent

> An autonomous AI research agent that conducts ML experiments end-to-end —
> from idea and dataset to validated results — at a PhD researcher's level.

[English](README.md) | [中文](docs/README_CN.md)

---

## What It Does

Give it a **research brief** (your hypothesis + target metrics) and a **dataset**.
The agent will autonomously: understand the data, survey approaches, design
experiments, implement & train models, verify results, reflect & iterate.

All autonomous. No human intervention between "here's my idea" and "here are your results."

---

## Quick Start

```bash
pip install -r requirements.txt
export GLM_CODING_PLAN_API_KEY="your-key-here"
python -m core.loop --project /path/to/your/project --max-cycles 10
```

Your project needs: `PROJECT_BRIEF.md`, a dataset, and `config.yaml`.

---

## Architecture

```
THINK → EXECUTE → VERIFY → REFLECT → repeat (until goal met or max cycles)

Memory (SQLite + MEMORY_LOG)     ← structured experiment records
Tool Layer (self-enforcing)      ← write_file, launch_experiment, GC, analyze_model
```

**Design: System = hard constraints. Prompt = methodology. LLM = PhD brain.**

See [docs/architecture.md](docs/architecture.md) for full documentation.

---

## Key Features

- Provider failover (GLM → Qwen) with quota-aware cooldown
- Anti-deception tool-trace verification
- Deterministic garbage collection (no LLM cost)
- Tool-level naming enforcement
- 99 automated tests

## License

MIT
