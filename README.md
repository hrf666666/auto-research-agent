# Auto Research Agent

> An autonomous AI research agent that conducts ML experiments end-to-end —
> from idea and dataset to validated results — at a PhD researcher's level.

[English](README.md) | [中文](docs/README_CN.md)

---

## What It Does

Give it a **research brief** (your hypothesis + target metrics) and a **dataset**.
The agent autonomously: understands the data, surveys approaches, designs
experiments, implements & trains models, verifies results, reflects & iterates.

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
THINK → EXECUTE → VERIFY → REFLECT → repeat

LLM (Leader) decides what to do based on:
  - PROJECT_BRIEF.md (research goals)
  - MEMORY_LOG.md (experiment history)
  - query_memory tool (causal chains, dead ends, best metrics)
  - domain knowledge (method properties, data constraints)

System provides:
  - Tool layer with built-in safety (naming, dead-end checks, GC)
  - 12-layer VERIFY (objective result validation)
  - Provider failover (GLM → Qwen) with quota cooldown
  - Structured memory (quantitative results written deterministically)
```

**Design principles:**
- System = hard constraints (safety, lifecycle, tools, memory)
- Prompt = research methodology (how to think, not what to do)
- LLM = PhD brain (design, implement, judge, iterate)

See [docs/architecture.md](docs/architecture.md) for full documentation.

---

## Key Features

- **query_memory tool**: LLM actively queries experiment history
- **Provider failover**: GLM → Qwen with quota-aware cooldown
- **Anti-deception**: tool-trace verification (LLM can't fake results)
- **Deterministic GC**: no LLM cost, archives temp files each cycle
- **Tool-level safety**: naming enforcement, dead-end checks, dry-run gate
- **99 automated tests**

---

## Configuration

See [config.yaml](config.yaml). Key sections:

```yaml
goals:               # Target metrics
  metrics:
    - key: "val_MAE"
      target: 0.20
      direction: "lower"

safety:              # Tool-level safety contracts
  naming: { forbidden_root_py: true }
  garbage_collection: { max_output_dirs: 10 }
```

---

## License

MIT
