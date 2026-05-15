# Architecture

> Detailed architecture documentation for Deep Researcher Agent.

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Deep Researcher Agent                      │
│                                                              │
│  ┌─────────────┐                                             │
│  │ config.yaml │──→ Configuration for all components         │
│  └─────────────┘                                             │
│                                                              │
│  ┌──────────── Core Loop (loop.py) ────────────────────┐     │
│  │                                                      │     │
│  │  ┌───────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐ │     │
│  │  │ THINK │→│ EXECUTE │→│ VERIFY  │→│ REFLECT │→↻  │     │
│  │  └───┬───┘  └────┬────┘  └────┬────┘  └────┬────┘ │     │
│  │      │             │              │                 │     │
│  │      ↓             ↓              ↓                 │     │
│  │  ┌───────────────────────────────────────┐          │     │
│  │  │        Agent Dispatcher (agents.py)   │          │     │
│  │  │                                       │          │     │
│  │  │  Leader ──→ Idea / Code / Writing /  │          │     │
│  │  │            Researcher                 │          │     │
│  │  │  (w/ Reasoning Principles injection)  │          │     │
│  │  └───────────────────────────────────────┘          │     │
│  │      │             │              │                 │     │
│  │      ↓             ↓              ↓                 │     │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────┐ │     │
│  │  │ Memory   │ │ Monitor  │ │  Tools   │ │Verifier│ │     │
│  │  │ Manager  │ │ (Zero$)  │ │ Registry │ │(CHECK) │ │     │
│  │  └──────────┘ └──────────┘ └────┬─────┘ └────────┘ │     │
│  │                                   │                    │     │
│  │          ┌────────────────────────┼──────────┐       │     │
│  │          │    ToolRegistry (v8)   │          │       │     │
│  │          │  ┌─────────────────────┼───────┐  │       │     │
│  │          │  │MCPClientMixin       │       │  │       │     │
│  │          │  │(mcp_client.py)      │       │  │       │     │
│  │          │  └─────────────────────┼───────┘  │       │     │
│  │          │  ┌─────────────────────┼───────┐  │       │     │
│  │          │  │ModelAnalyzerMixin   │       │  │       │     │
│  │          │  │(model_analyzer.py)  │       │  │       │     │
│  │          │  └─────────────────────┼───────┘  │       │     │
│  │          └────────────────────────┼──────────┘       │     │
│  │                                   │                    │     │
│  │                            ┌──────┴──────┐            │     │
│  │                            │   Vision    │            │     │
│  │                            │  Analyzer   │            │     │
│  │                            │(MCP+API)   │            │     │
│  │                            │SSE+stdio   │            │     │
│  │                            └─────────────┘            │     │
│  │                                                      │     │
│  │  ┌──────────── Research Intelligence (v8) ──────────┐│     │
│  │  │ DomainKnowledgeMixin (domain_knowledge.py)       ││     │
│  │  │ Idea Guardian (every 5 cycles)                    ││     │
│  │  │ Direction Circuit Breaker                         ││     │
│  │  │ Data Scarcity Awareness                           ││     │
│  │  └──────────────────────────────────────────────────┘│     │
│  │                                                      │     │
│  │  ┌──────────── Audit & Safety ──────────────┐       │     │
│  │  │ Experiment Auditor → 3-Level Escalation  │       │     │
│  │  │ Error-handler skill │ Code-cleanup skill │       │     │
│  │  └──────────────────────────────────────────┘       │     │
│  └──────────────────────────────────────────────────────┘     │
│                                                              │
│  ┌──────────── GPU Layer ──────────────────────────────┐     │
│  │  detect.py  │  keeper.py (standalone utility)       │     │
│  └─────────────────────────────────────────────────────┘     │
│                                                              │
│  ┌──────────── Skills Layer ───────────────────────────┐     │
│  │  daily-papers │ paper-analyze │ conf-search │ report │     │
│  │  REASONING_PRINCIPLES.md │ error-handler │ auditor  │     │
│  └─────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Core Loop (`core/loop.py`)

The main orchestrator. Runs the THINK → EXECUTE → VERIFY → REFLECT cycle indefinitely.

**Key design decisions:**
- **Signal handling**: SIGTERM/SIGINT trigger graceful shutdown
- **Cycle counter**: Persisted to `.cycle_counter` file (survives restarts)
- **Smart cooldown**: Polls every N seconds instead of fixed sleep
- **Directive consumption**: Human directives are archived after reading (no re-reads)
- **Error backoff**: Doubles cooldown after errors to prevent burn loops
- **VERIFY phase**: Between EXECUTE and REFLECT, reverse-engineers whether each module actually worked

### 2. Agent Dispatcher (`core/agents.py`)

**Leader-Worker pattern** where:
- Leader persists conversation within a cycle (for coherent multi-step reasoning)
- Workers are stateless (each dispatch is independent)
- Only one worker runs at a time

**Anti-Deception Architecture:**
Every worker dispatch returns a **ToolTrace** alongside the LLM's text response. The ToolTrace records every tool call the LLM made, including the actual system-returned results. Key facts (PIDs, log file paths, exit codes) are extracted from tool results — never from LLM narrative text.

```
┌─────────────────────────────────────────────────────────────┐
│               Anti-Deception Data Flow                       │
│                                                               │
│  LLM says: "I launched PID=12345, dry-run passed"           │
│       │                                                       │
│       ├──── Tool Trace says: launch_experiment NOT called    │
│       │     → FABRICATION DETECTED                           │
│       │     → experiment_launched = False                    │
│       │     → deception_detected = True                      │
│       │                                                       │
│  LLM says: "dry-run OK, launching training"                 │
│       │                                                       │
│       ├──── Tool Trace says: launch_experiment returned      │
│       │     {"pid": 67890, "log_file": "logs/exp.log"}      │
│       │     → VERIFIED: pid=67890 from tool result           │
│       │     → dry-run check: no "dry" command in trace       │
│       │     → dry_run_performed = False (warning)            │
│                                                               │
│  VERIFY then checks:                                          │
│  - Is PID 67890 a real process?                              │
│  - Does log_file exist?                                      │
│  - Did dry-run actually happen?                              │
└─────────────────────────────────────────────────────────────┘
```

**Key classes:**
- `ToolTrace`: Records all tool calls + results from one LLM session
- `ToolCallRecord`: Immutable record of one tool call
- `extract_launch_facts()`: Gets PID/log_file from actual `launch_experiment` results
- `extract_shell_facts()`: Gets exit codes from actual `run_shell` results

**Worker Types:**
| Worker | Tools | Role |
|--------|-------|------|
| Idea Agent | `read_files`, `web_search`, `web_fetch`, `search_papers` | Literature research, hypothesis generation |
| Code Agent | `read_files`, `write_file`, `run_shell`, `launch_experiment`, `log_memory` | Experiment implementation |
| Writing Agent | `read_files`, `write_file`, `run_shell` | Reporting, documentation |
| Researcher Agent | `web_search`, `web_fetch`, `search_papers`, `get_paper` | Deep literature search for breakthrough methods |

**Code Agent Turn Budget:**
- `max_turns`: 25 (reduced from 40 to prevent endless exploration)
- Turn budget reminder injected into every tool result: shows current turn / max turns
- At 60% budget: "WARNING: Stop exploring and focus on PRIMARY task"
- At 80% budget: "CRITICAL: Must call launch_experiment NOW or report failure"
- Consecutive `list_files` limited to 3 calls — prevents directory browsing loops

**Session Statistics Injection:**
The Leader agent context includes SQLite session statistics:
- Total cycles completed
- Experiments launched count and launch rate
- Dead ends accumulated
- Recent failure patterns (last 3)

This enables the Leader to make informed decisions based on session history rather than only the current cycle's data.

**Tiered Model Strategy:**
Tasks are categorized by complexity. `STRONG_MODEL_TASKS` (think, reflect, idea, researcher) use the strong model for better reasoning. Routine tasks (code, writing) use the fast model to save tokens. Configurable via `"model": "auto"` in config.

**Provider Failover:**
When a provider fails, the system automatically falls back to the next available provider with health tracking and cooldown periods. Supports: `anthropic`, `openai`, `ali_token_plan`, `glm_token_plan`.

**Why this works:**
- Leader sees the full picture without re-reading everything each step
- Workers are cheap (no accumulated context)
- Switching workers costs nothing (previous worker's context is gone)
- **Facts come from system execution, not LLM claims**

### 3. Memory Manager (`core/memory.py`)

**Two tiers with automatic compaction:**

- **Tier 1 (Brief)**: Human-written, frozen. The "constitution" of the project.
- **Tier 2 (Log)**: Agent-written, rolling. Milestones and decisions.

**Compaction rules:**
1. Milestones: Drop oldest when section exceeds 1,200 chars
2. Decisions: Keep only last 15 entries
3. Total log: Hard cap at 4,000 chars (summarize old entries first, then trim if still too large)
4. Compression: Old entries are summarized into a single `[Historical N entries: ...]` line before deletion, preserving knowledge

### 4. Experiment Monitor (`core/monitor.py`)

**The zero-cost innovation.** During training:
- `os.kill(pid, 0)` — is process alive? (zero cost)
- `nvidia-smi` — GPU utilization (zero cost)
- File tail read — last log lines (zero cost)

No LLM API calls until training completes.

### 5. Tool Registry (`core/tools.py` + Mixins)

**v8 Architecture**: `ToolRegistry` inherits from two mixin classes:

```
ToolRegistry (core/tools.py, 1667 lines)
  ├── MCPClientMixin (core/mcp_client.py, 724 lines)
  │     MCP transport (SSE + stdio), service detection, vision tools
  └── ModelAnalyzerMixin (core/model_analyzer.py, 2282 lines)
        9-layer AST analysis, runtime probes, diagnostics, ablation design
```

**Why mixins instead of a monolith**: The original `tools.py` was 4,714 lines — too large for effective maintenance. The mixin pattern preserves single-inheritance semantics while separating concerns. Each mixin is independently testable and can be reused by other classes.

**Per-agent minimal tool sets** reduce token overhead:
- Each tool definition is ~200 tokens in the API call
- 15 tools = 3,000 extra tokens per call
- 4 tools = 800 extra tokens per call
- Over 100 API calls/day, that's 220K tokens saved

**Security:** Shell commands are validated with regex-based safety checks (`_validate_command`). Shell operators (`cd`, `&&`, `|`, `> /dev/null`) are supported via `shell=True`, while dangerous operations (sudo, rm -rf /, dd to device) are blocked.

**File write protection:** The `_exec_write_file` tool enforces:
- Protected files: `state.json`, `MEMORY_LOG.md`, `PROJECT_BRIEF.md`, `config.yaml`, etc.
- Protected directories: `models/`, `datasets/`, `data/`, `scripts/` — with exceptions:
  - `scripts/*.py` is allowed (experiment scripts)
  - `datasets/__init__.py` and `datasets/unified_lf_dataset.py` are allowed (dataset registration)
- Synthetic data detection: Writing to `scripts/*.py` triggers pattern scanning for random noise (`np.random.rand`, `torch.rand`, `SyntheticLF`, `RandomDataset`). Returns a warning to the agent, and VERIFY will block the experiment.

#### Model Analysis Tools (v5)

Two complementary tools for model architecture understanding:

**`analyze_model`** — Static multi-layer architecture analysis:
```
Layer 1: Surface analysis (parameter counts, channel ratios)
Layer 2: Data flow graph (input → processing → fusion → output)
Layer 3: Information bottleneck detection (compression > 8:1)
Layer 4: Gradient path analysis (dead branches, skip connections)
Layer 5: Structural soundness score (0-10)
Layer 6: Domain assumption detection (EPI→Lambertian, FFT→stability, etc.)
Layer 7: Data feasibility + GPU memory estimate
Layer 8: Result-to-architecture diagnosis (when metrics provided)
```

**`probe_model`** — Runtime tensor diagnostics:
```
1. Instantiates model, runs forward+backward with dummy data
2. Captures per-module activation statistics (mean/std/dead_ratio)
3. Captures per-module gradient norms
4. Computes gradient balance (max/min ratio, warns > 100x imbalance)
5. Tests input sensitivity (random vs uniform vs near-zero)
6. Optionally loads trained checkpoint for post-training diagnosis
```

| Tool | Type | Best For |
|------|------|----------|
| `analyze_model` | Static (AST) | Before training: catch design flaws |
| `probe_model` | Runtime (PyTorch) | After training: diagnose why model failed |

#### Research Intelligence Tools (v6→v7)

**v7: Idea-Architecture Alignment** — Ensures models faithfully implement research ideas:
```
analyze_model() → Layer 9: idea_architecture_alignment
  1. Parse PROJECT_BRIEF.md → extract 10 key idea components
  2. Map idea components to model branches/modules
  3. Per-branch channel allocation vs idea importance
     - Flag: KEY INNOVATION branch < 15% of fusion → "under-represented"
  4. Structural gap detection (skip connections, multi-scale, attention)
     - Only flags patterns relevant to the specific idea
  5. Decoder adequacy (depth, skip connections, domain-awareness)
  6. Alignment score (0-10) + specific improvement suggestions

Example output (AngularFreqDepthNetV2):
  Score: 3/10 — "core idea components missing or under-represented"
  CRITICAL: fft_branch gets only 11% of fusion channels (32/288)
  MISSING: skip_connection_residual, multi_scale_processing
  SUGGESTION: INCREASE fft_branch from 32 to ~72 channels
```

**v6: Research Intelligence Tools**

Three new capabilities for PhD-level scientific reasoning:

**`generate_diagnostic`** — Targeted diagnostic script generation:
```
Input:  Natural language question ("Is FFT branch dead for Non-Lambertian?")
Output: Generated + executed Python diagnostic script

Four diagnostic types (auto-detected from question):
- domain_analysis:  Tests different input patterns (smooth/high_freq/specular/constant)
- branch_analysis:  Compares branch activations + pairwise cosine similarity
- gradient_analysis: Checks gradient flow + identifies bottleneck layers
- attention_analysis: Analyzes attention weight distribution
```

**`design_ablation`** — Systematic ablation experiment planning:
```
1. AST parse model to identify all components
2. Group by category: backbone, branch, head, fusion, normalization
3. Generate ablation experiments:
   - Component removal (each branch, normalization layers)
   - Freeze (backbone, specific branches)
   - Single-branch model
   - Fusion replacement (learned → mean)
4. Priority rank by expected information value
```

**Pareto Frontier + Causal Chain + VOI** (system-level, not tools):
```
pareto_matrix:    method × domain → best MAE (avoids repeating suboptimal methods)
causal_chain:     design_decision → architectural_property → metric (tracks causation)
experiment_value: hypothesis → expected_improvement × prior_probability (calibrates judgment)
```

### 6. GPU Utilities (`gpu/`)

- **detect.py**: Auto-detect GPUs, check availability, reserve last GPU
- **keeper.py**: Keep cloud instances alive with minimal GPU activity

### 7. Reasoning Principles System (`skills/REASONING_PRINCIPLES.md`)

A mandatory behavioral framework injected into every agent dispatch to reduce common LLM reasoning mistakes. Five principles (plus Verify-first) guide the THINK→EXECUTE→VERIFY→REFLECT cycle:

| Principle | THINK | EXECUTE | VERIFY | REFLECT |
|-----------|-------|---------|--------|---------|
| **Think Before Acting** | State assumptions, present alternatives | Investigate before implementing | — | Don't rationalize failures |
| **Simplicity First** | Pick simplest hypothesis to test | Change ONE variable, minimal code | — | — |
| **Surgical Changes** | — | Only touch relevant files | — | — |
| **Goal-Driven Execution** | Define concrete success criteria | Verify at each step | Check outputs vs criteria | Check if criteria were met |
| **Verify-First** | — | — | Module outputs → are they real? | Address VERIFY failures before judging |
| **Anti-Deception** | — | Don't trust claims, trust tool traces | Cross-verify PID/paths against tool trace | If fabrication detected, mark UNRELIABLE |

**Injection mechanism:**
- `_REASONING_REMINDER` constant in `agents.py` is prepended to every Leader dispatch
- Code agent prompt (`agents/code_agent.md`) includes full principles in system prompt
- Leader agent prompt (`agents/leader.md`) includes expanded Decision Framework with mandatory assumption/criteria steps

### 8. Experiment Verifier (`core/verifier.py`)

**The VERIFY phase: reverse-engineering whether each module actually worked.**

Unlike the old auditor which checked static properties (code text matching, dataset registration), VERIFY checks **runtime behavior** — did the dataset loader actually load? Did the model produce valid loss? Did training make progress?

```
┌──────────────────────────────────────────────────────────────────┐
│                     VERIFY Phase Pipeline                         │
│                                                                    │
│  Step 1: Artifact Discovery                                       │
│  ├── Scan workspace for output files, logs, checkpoints           │
│  └── Classify by module (dataset, model, training, evaluation)   │
│                                                                    │
│  Step 2: Module-Level Verification                                │
│  ├── Dataset: files exist? shape/dtype valid? not all zeros?      │
│  ├── Model: checkpoint exists? weights valid? forward pass OK?    │
│  ├── Training: loss not NaN/Inf? decreasing? GPU was used?        │
│  └── Evaluation: metrics file exists? values in expected range?   │
│                                                                    │
│  Step 3: Behavioral Cross-Checks                                  │
│  ├── Plan vs reality: did EXECUTE implement what THINK planned?   │
│  ├── Loss dynamics: is the loss curve physically plausible?       │
│  └── Metric consistency: do reported metrics match training logs? │
│                                                                    │
│  Step 4: Diagnosis Output                                         │
│  ├── PASS: module functioned correctly                            │
│  ├── FAIL: module did not produce expected output (with detail)   │
│  └── SKIP: module not applicable this cycle                       │
│                                                                    │
│  Step 5 (v5): Model Structural Soundness (Layer 9)               │
│  ├── Dead modules: __init__ assigns but forward() never uses     │
│  └── Fusion warnings: 3+ branches concatenated without balance   │
│                                                                    │
│  Step 6 (v6): Training Curve Analysis                             │
│  ├── Overfitting: loss rises after minimum                        │
│  ├── Oscillation: direction change ratio > 15%                    │
│  ├── Convergence speed: < 5% total decrease = under-capacity     │
│  └── Plateau: loss flat (< 0.1% change) for extended period      │
└──────────────────────────────────────────────────────────────────┘
```

**9 verification layers (+ training curve diagnostics):**
1. Execution verification (did it run?)
2. Output artifact verification (did it produce files?)
3. Module functionality verification (did each module work?)
4. Data integrity verification (is data valid?)
5. Metric consistency verification (do metrics match logs?)
6. Configuration consistency verification (checkpoint/config match?)
7. System health verification (OOM, disk full?)
8. Dataset quality verification (validation splits statistically meaningful?)
9. **Model structural soundness** (v5 — dead modules, fusion balance)

**Training curve diagnostics** (v6, within Layer 3 loss function check):
- Overfitting detection (loss rises > 5% above minimum in latter half)
- Oscillation detection (direction change ratio > 15%)
- Convergence speed classification (very_slow / fast_early_plateau / normal)
- Plateau detection (< 0.1% variation over extended window)

**Key design:**
- **Zero LLM cost**: All checks are file/system-based (no API calls)
- **Structured diagnosis**: `VerifyReport` with per-module `VerifyCheck` objects
- **Actionable**: Each failure includes a detail string explaining *what* went wrong
- **Composable**: New module checkers can be added via `ExperimentVerifier` subclass
- **Anti-Deception**: Cross-verifies tool trace against LLM claims:
  - `llm_fabrication`: LLM claimed experiment launched but `launch_experiment` was never called
  - `pid_trace_mismatch`: PID in LLM text differs from PID in tool trace
  - `dry_run_skipped`: Experiment launched without mandatory dry-run
  - `dry_run_failed`: Dry-run failed but experiment was launched anyway

### 9. Audit Escalation System (`core/loop.py`)

A 4-level escalation system that prevents the agent from looping indefinitely on repeated errors. Now powered by VERIFY failure detection:

```
┌──────────────────────────────────────────────────────────────────┐
│                     Audit Escalation Flow                         │
│                                                                    │
│  VERIFY Phase (every cycle)                                       │
│       │                                                            │
│       ▼                                                            │
│  ┌─────────────────┐                                               │
│  │ Same issue again?│──→ No ──→ Reset counter                     │
│  └────────┬────────┘                                               │
│           │ Yes                                                     │
│           ▼                                                        │
│  ┌─────────────────────────────────────────────────┐               │
│  │ Count ≥ 3?  (L1) ──→ Inject DIRECTIVE.md        │               │
│  │ Count ≥ 6?  (L2) ──→ Force error-handler skill  │               │
│  │ Count ≥ 9?  (L3) ──→ Pause + AGENT_STUCK.md     │               │
│  │ Count ≥ 12? (L4) ──→ Mark unfixable dead_end    │               │
│  └─────────────────────────────────────────────────┘               │
└──────────────────────────────────────────────────────────────────┘
```

**L4 (unfixable)** is new: after 4x the threshold attempts, the issue is logged as a dead_end and the counter is reset. The agent stops trying to fix this automatically and moves on to alternative approaches.

**VERIFY checks include:**
1. Dataset: files exist, shape/dtype valid, not all zeros/ones
2. Model: checkpoint exists, weights contain no NaN/Inf, forward pass produces valid output
3. Training: loss is not NaN/Inf, loss decreased over time, GPU was actually used
4. Evaluation: metrics file exists, values in expected range, match training logs
5. Plan vs reality: THINK plan items implemented by EXECUTE
6. Loss dynamics: loss curve is physically plausible
7. Runtime data fingerprint: loads one sample from dataset, checks spatial correlation and constant values to verify training used real data (not random noise)

**Error-handler skill** (`skills/error-handler/SKILL.md`) provides a 7-step diagnostic workflow:
1. Identify error → 2. Read relevant files → 3. Diagnose root cause → 4. Validate fix → 5. Apply minimal fix → 6. Re-validate → 7. Cleanup

### 10. Token Plan Provider Support (`core/agents.py`)

Cost-optimized LLM providers using OpenAI-compatible protocol:

```yaml
# config.yaml
agent:
  provider: "ali_token_plan"    # Alternative to "anthropic" / "openai"
  model: "qwen3.6-plus"         # See TOKEN_PLAN_PROVIDERS for all models
```

**Available models:**
| Provider | Models | Best For |
|----------|--------|----------|
| `ali_token_plan` | qwen3.6-plus, deepseek-v3.2, glm-5, MiniMax-M2.5 | Cost-optimized daily experiments |
| `ali_token_plan` | qwen-image-2.0, wan2.7-image | Image generation tasks |
| `glm_token_plan` | glm-5.1, glm-5-turbo | Zhipu GLM Coding Plan |

**Architecture:** All token plan providers use the OpenAI-compatible API (`_call_token_plan`), which shares the same tool-execution loop as `_call_openai`. Automatic failover between providers with health tracking and cooldown.

### 11. No-Progress Paper Research Fallback (`core/loop.py`)

When the agent detects repeated cycles with no progress on the same experimental plan, it automatically redirects effort to paper research:

```
┌──────────────────────────────────────────────────────────────────┐
│                     No-Progress Detection                         │
│                                                                    │
│  After each REFLECT:                                              │
│       │                                                            │
│       ▼                                                            │
│  ┌──────────────────────┐                                          │
│  │ Same plan repeated   │──→ No ──→ Continue normal cycle          │
│  │ N times with no      │                                          │
│  │ metric improvement?  │──→ Yes ──→ Dispatch Researcher Agent     │
│  └──────────────────────┘            to search for new methods     │
│                                      via papers/arXiv              │
│                                      ↓                             │
│                                      New ideas feed back into      │
│                                      next THINK phase              │
└──────────────────────────────────────────────────────────────────┘
```

This prevents the agent from looping endlessly on a plateau. Instead of repeating the same failed approach, it seeks breakthrough methods from the literature.

### 12. Dataset Understanding System (`core/loop.py` + `core/verifier.py`)

On the first cycle (or when `DATASET_MANIFEST.json` is missing), the agent performs a mandatory scan of the `data/` directory:

- Validates file existence and structure
- Checks data shapes, dtypes, and value ranges
- Produces a structured `DATASET_MANIFEST.json` for future reference
- Feeds dataset understanding into the THINK phase

This ensures the agent knows what data it's working with before making experimental plans.

### 13. Visual Analysis & Vision MCP (`core/visual_analyzer.py` + `core/tools.py`)

When training results are consistently poor (>= 5 consecutive no-progress cycles), the agent triggers a **visual analysis pipeline**:

```
Inference Pipeline:
1. Find best checkpoint → run inference on validation scenes
2. Collect output images (depth maps, predictions)
3. Send to multimodal LLM for visual diagnosis
4. Parse into structured findings for REFLECT phase
```

#### MCP Transport Architecture

The system uses **two MCP transport types** depending on the service:

| Service | Transport | How it works |
|---------|-----------|-------------|
| web_search_prime | SSE (remote) | `GET /sse` → endpoint event → `POST /message` (dual-connection) |
| web_reader | SSE (remote) | Same as above |
| zread | SSE (remote) | Same as above |
| zai-mcp-server | stdio (local) | `npx @z_ai/mcp-server` subprocess, stdin/stdout JSON-RPC |

**SSE dual-connection protocol** (GLM platform services):
- Connection 1: `GET /sse` — long-lived SSE stream, background reader thread collects responses
- Connection 2: `POST /message?sessionId=...` — send JSON-RPC requests (returns empty HTTP 202)
- All responses arrive asynchronously on the SSE stream

**stdio transport** (zai-mcp-server vision):
- Spawned as `npx -y @z_ai/mcp-server` with `Z_AI_API_KEY` + `Z_AI_MODE=ZHIPU` env vars
- JSON-RPC messages sent via stdin, responses read from stdout (newline-delimited)
- **Requires Node.js 18+ and npx** — if absent, the degradation chain skips MCP entirely

#### Vision Degradation Chain

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. MCP zai-mcp-server (stdio, requires Node.js + npx)          │
│    Tools: analyze_image, diagnose_error_screenshot,             │
│    analyze_data_visualization, understand_technical_diagram,    │
│    extract_text_from_screenshot, ui_diff_check, ui_to_artifact, │
│    analyze_video                                                 │
│    Images: local file paths (subprocess reads filesystem)       │
├─────────────────────────────────────────────────────────────────┤
│ 2. Direct API — GLM Coding Plan (GLM_CODING_PLAN_API_KEY)       │
│    Models: glm-5v-turbo → glm-4.6v                              │
│    Endpoint: open.bigmodel.cn/api/coding/paas/v4                │
│    Images: base64 data URLs                                      │
├─────────────────────────────────────────────────────────────────┤
│ 3. Direct API — Ali Token Plan (ALI_TOKEN_PLAN_API_KEY)          │
│    Models: qwen3.6-plus                                          │
│    Endpoint: token-plan.cn-beijing.maas.aliyuncs.com             │
│    Images: base64 data URLs                                      │
├─────────────────────────────────────────────────────────────────┤
│ 4. Direct API — Ali DashScope (ALI_API_KEY)                      │
│    Models: qwen3.6-plus → qwen3.5-plus                          │
│    Endpoint: dashscope.aliyuncs.com                              │
│    Images: base64 data URLs                                      │
└─────────────────────────────────────────────────────────────────┘

Note: If Node.js is not installed, step 1 is skipped entirely.
      Steps 2-4 work without any Node.js dependency.
```

**MCP Vision Tools** (via `@z_ai/mcp-server`, stdio transport):
- `analyze_image`: General-purpose image understanding
- `diagnose_error_screenshot`: Parse error popups/stack traces
- `extract_text_from_screenshot`: OCR from screenshots
- `analyze_data_visualization`: Charts → trends/anomalies
- `understand_technical_diagram`: Architecture/flow diagrams
- `ui_diff_check`, `ui_to_artifact`, `analyze_video`

**Agent-facing tools**:
- `analyze_image` (researcher agent): Routes `analysis_type` to specialized MCP tools, with API fallback
- `diagnose_error` (code agent): Error screenshot diagnosis with MCP + API fallback

**Parameter compatibility**: Internal code uses `image_path`, auto-remapped to `image_source` for zai-mcp-server. Tool name `image_analysis` auto-remapped to `analyze_image`.

**Node.js requirement**: Vision MCP requires Node.js 18+ with npx. Install via `apt install nodejs` (Ubuntu) or `brew install node` (macOS). Without Node.js, vision analysis degrades gracefully to direct multimodal API calls.

### 14. Enhanced REFLECT Phase (`core/agents.py`)

The REFLECT phase has been enhanced for deep cross-validation:

| Feature | Value |
|---------|-------|
| Tools available | `read_file`, `list_files` |
| Max turns | 20 (vs 10 for other phases) |
| Max tokens | 16384 (vs 4096 previously) |
| Cross-validation | Read model code + data manifest + training logs |

The Leader receives explicit prompting to use these tools for **Step 3.5: Visual + Code Cross-Validation**:
1. Read model source code to verify architecture matches visual findings
2. Read `DATASET_MANIFEST.json` to check data splits
3. Read training logs to verify loss curves
4. List output files to check what artifacts were produced

### 15. Auto Code-Cleanup v2 (`core/loop.py`)

Automatic cleanup triggers with 6 conditions:

| # | Condition | Threshold |
|---|-----------|-----------|
| 1 | Root .py files | > 15 |
| 2 | Unarchived log files | > 10 |
| 3 | Output experiment dirs | > 10 |
| 4 | Archived experiments | > 20 |
| 5 | scripts/ .py files | > 10 (naming pollution) |
| 6 | Experiment failed | Immediate trigger |

Cleanup steps: outputs/ (delete dry-runs, keep only best_checkpoint.pt) → archive/ (delete .pt files, remove dirs without SUMMARY.md) → scripts/ (delete stale/diagnostic scripts) → root .py files.

### 16. Script Naming Convention (`agents/code_agent.md`)

Enforced naming rules for experiment scripts:

| Category | Pattern | Example |
|----------|---------|---------|
| Training | `train_{model}.py` | `train_v12.py`, `train_dcbn.py` |
| Evaluation | `eval_{target}.py` | `eval_per_domain.py` |
| Diagnostic | `diagnose_{target}.py` | `diagnose_gt_stats.py` |
| One-time | `_{name}.py` (delete after use) | `_check_shapes.py` |
| Dry-run output | `dry_*` or `dryrun_*` | Auto-cleaned |

**Forbidden**: `*_v2.py`, `*_fix.py`, `*_new.py`, `test_*.py`, `debug_*.py`

### 17. Output Quality Awareness (`core/loop.py`)

The agent now tracks per-domain metrics and detects when experiments produce "successful bad results":

- `_best_domain_metrics`: Tracks best metric per domain (dynamically from `domain_keys`).
- `_quality_alert_streak`: Counts consecutive cycles with domain degradation (>10% worse than best).
- When streak >= 2: Forces paper research with hypothesis validation prompt.

### 18. Strategic Abandonment (`core/loop.py`)

Research direction stagnation detection prevents the agent from stuck in the same approach:

- `_extract_direction_signature()`: Extracts methodology keywords from task description (edge/loss/pretrain/epi/angular/etc.) with hyphen/underscore normalization.
- `_direction_stagnation_count`: Counts cycles without improvement in the same direction.
- When count >= 3: Forces paper research for fundamentally different approaches.

### 19. Infrastructure Degradation (`core/loop.py`)

After 3 consecutive infrastructure failures (API timeout, process crash):

- VERIFY no longer blocks experiments — runs in non-blocking mode.
- Infrastructure issues are logged but don't halt progress.
- Counter resets when no new infra failures occur.

### 20. Enhanced REFLECT Prompts (`core/loop.py`)

Two new prompts are injected into the REFLECT context:

- **Cross-Domain Analysis Prompt**: Forces the Leader to analyze WHY different domains perform differently, identify violated method assumptions, and estimate the current approach's ceiling.
- **Hypothesis Validation Prompt**: When quality degrades repeatedly, the agent must state the core assumption, identify which domain violates it, and propose a new method.

### 21. Configuration Consistency Verification (`core/verifier.py`)

New VERIFY Layer 6 checks:

- Checkpoint mismatch detection: Scans training log tail (last 50KB) for "size mismatch" or "missing key" errors.
- Hardcoded parameter detection: Flags `num_views` hardcoded values that may not match actual data grid sizes.

### 22. Dynamic Domain Knowledge (`core/domain_knowledge.py`) — v8

Replaces all hardcoded domain-specific logic with dynamic extraction:

- **`DomainKnowledgeMixin`** is inherited by `ResearchLoop` (same pattern as `MCPClientMixin` for `ToolRegistry`)
- **`METHOD_PROPERTIES`**: Generic method database (7 entries: EPI, FFT, attention, ResNet, sigmoid, Conv3D, contrastive) with scientific knowledge only. Each entry has `patterns` (for detection), `assumption`, `violated_when`, `failure_symptoms`, and `alternatives`. Adding a new method here automatically enables detection for all projects.
- **`_infer_domain_compatibility()`**: Extracts domain names from PROJECT_BRIEF text, then uses method violation conditions to determine strong/weak domains. No hardcoded "EPI is strong for Lambertian" logic.
- **`_extract_data_constraints()`**: Detects data scarcity (`< 10 training samples`) and data imbalance from brief text patterns.
- **`_detect_implemented_methods()`**: Scans `models/` directory to find which methods are actually in the codebase.
- **Cross-project reusability**: Zero hardcoded project references. The same module works for any domain.

### 23. Idea Guardian & Direction Circuit Breaker — v8

Two mechanisms to prevent research direction drift:

**Idea Guardian** (every 5 cycles):
- Injects `idea_guardian_check` context into THINK phase
- Forces Leader to: (1) re-read PROJECT_BRIEF phase goals, (2) rate core idea implementation / phase completion / data-first verification (each 0-10), (3) propose course correction if any score < 5
- Prevents the common failure mode of spending 20+ cycles on incremental tuning while ignoring the core research idea

**Direction Circuit Breaker**:
- When `_direction_stagnation_count >= _direction_change_threshold`, injects `direction_circuit_breaker` context
- Forces Leader to stop and re-read PROJECT_BRIEF, record current direction as dead end, propose fundamentally different approach
- Prevents infinite loops on the same failed direction

**Data Scarcity Awareness**:
- When `data_constraints` detects < 10 training samples for a domain, injects `data_scarcity_warning`
- Leader is instructed: "DO NOT propose architecture changes — the model CANNOT learn domain-specific features from < 10 samples"
- Hard wall prevents wasted GPU hours on impossible tasks

### 24. Data Analysis Experiments — v8

Not every experiment requires model training. v8 explicitly supports data analysis experiments:

- **Code Agent workflow**: New section in `code_agent.md` describes data analysis experiment patterns
- Uses `run_shell` (NOT `launch_experiment`) since no training is involved
- Designed for Phase 1 verification (verify data supports the idea before building models)
- Scripts prefixed with `_` and deleted after use (e.g., `_phase1_fft_analysis.py`)
- Example: "Load 5 scenes from each domain, compute angular FFT spectra, plot histograms, report KL divergence"

### 25. Memory Configuration Injection — v8

`MemoryManager.__init__` now accepts optional parameters:
- `method_keywords: dict` — custom method→keyword mapping (default: generic method vocabulary)
- `domain_keys: list` — custom domain metric keys (default: inferred from `DATASET_MANIFEST.json`)
- `get_method_domain_effect_matrix()` and related queries use instance variables instead of hardcoded values
- Enables cross-domain reuse without code changes

### 26. Forward Design Pipeline (`core/idea_planner.py`) — v9

**IdeaPlanner** generates a PhD-level architecture plan from `PROJECT_BRIEF.md` before any model code is written. The 9-phase pipeline:

1. **Idea Formalization** — Extracts hypothesis, innovations, assumptions, success criteria from PROJECT_BRIEF
2. **Module Decomposition** — Breaks idea into independent functional modules using `IDEA_PATTERNS` (6 generic architectural patterns)
3. **Capacity Planning** — Channel counts, parameter budgets (light/medium/heavy tiers)
4. **Fusion Strategy** — Optimal combination method (attention_weighted, gated, concat, etc.)
5. **Integration Plan** — Data flow, skip connections, normalization choices
6. **Verification Plan** — Per-module and overall verification checkpoints
7. **Risk Assessment** — Data scarcity, branch imbalance, overfitting probability
8. **Implementation Order** — Which module to build first, second, etc.
9. **Alignment Score** — Overall plan quality rating (0-10)

**Knowledge organization design**:
- `_extract_innovations()`: Vocabulary-to-concept mapping (natural language → abstract innovation categories). This is a knowledge organization method — adding new term-category pairs extends coverage without changing logic.
- `_extract_physical_assumptions()`: Pattern-based assumption extraction from text
- `_extract_target_domains()`: Generic CV domain vocabulary (Lambertian, outdoor, indoor, specular, etc.)
- `_INNOVATION_TO_PATTERN`: Maps innovation keywords to `IDEA_PATTERNS` keys for architecture pattern selection

**Pipeline integration**:
- Registered as `plan_model` tool in ToolRegistry (available to Leader and Code Agent)
- Auto-invoked in THINK phase (cycle ≤ 1) to generate `architecture_plan` context
- Code Agent uses `implementation_order` as step-by-step build guide

### 27. Post-Experiment Evaluation (`core/experiment_evaluator.py`) — v9

Three classes for structured post-experiment analysis:

**ExperimentEvaluator**:
- Compares planned success criteria against actual results
- 5-type failure diagnosis: `architecture`, `data`, `training`, `alignment`, `capacity`
- Generates priority-sorted iteration guidance
- Injected into REFLECT phase as `experiment_evaluation` + `iteration_guidance_prompt` context

**IndependentProbe** (third-party verification):
- Loads model checkpoint, runs forward pass on random input
- Detects output anomalies: collapsed output (std < 1e-5), NaN/Inf, range mismatch
- Integrated as VERIFY Layer 10 (`_verify_independent_probe`)
- Avoids "self-evaluation" by independently probing the model's actual behavior

**IterationGuidance** + **FailureDiagnosis**:
- Root cause chain with severity levels
- Actionable next-step recommendations sorted by priority
- Failure type → correct response mapping table in leader.md

### 28. Domain-Agnostic Hardcoding Cleanup — v9

Systematic removal of all project-specific hardcoding across 10 files. Design principle: **only knowledge organization methods may be hardcoded; project-specific data/scenarios must be dynamic**.

**Agent prompts** (`agents/*.md`):
- All project-specific examples replaced with generic equivalents
- `UnifiedLFDataset` → "the project's real dataset class"
- `EPI assumes Lambertian` → "check `domain_knowledge.critical_assumptions`"
- `Non-Lambertian MAE=0.335` → "domain A metric=X vs domain B metric=Y"
- `81 angular views` / `9x9` → generic feature extraction language
- Zero remaining project-specific identifiers (HCInew, EPINet, etc.)

**Dynamic discovery patterns** (replacing hardcoded references):

| Before (hardcoded) | After (dynamic) |
|---|---|
| `{"HCInew": "Lambertian", ...}` dict | `DATASET_MANIFEST.json` `type` field + heuristic fallback |
| `UnifiedLFDataset` class reference | AST-based scanning of `datasets/` for Dataset subclasses |
| `AngularAwareDepthNet` class reference | AST-based scanning of `models/` for `nn.Module` subclasses |
| `[1, 81, 3, 64, 64]` default shape | `_infer_input_shape()` reading Conv3d/Conv2d `in_channels` |
| `["MAE_Lambertian", "MAE_Non_Lambertian", "MAE_Mixed"]` | `_infer_domain_keys()` reading manifest types |
| `{"epi": ["epi", "epinet", "epipolar"], ...}` | `_default_method_keywords()` with generic CV vocabulary |

**What remains as acceptable "hardcoding"** (knowledge organization methods):
- `METHOD_PROPERTIES` in `domain_knowledge.py`: Scientific method properties (EPI/FFT/attention assumptions) — generic CV knowledge
- `IDEA_PATTERNS` in `idea_planner.py`: 6 reusable architectural patterns — generic structural templates
- `_extract_innovations()` vocabulary: Natural language → concept category mapping — extensible retrieval vocabulary
- `_extract_domain_names()` patterns: Common CV domain vocabulary (Lambertian, outdoor, indoor) — generic scientific terms

### 29. Knowledge Organization Architecture — v9

The agent's knowledge is organized in 6 layers, from static to dynamic:

```
Layer 1: Static Prompts (agents/*.md)
  → Generic workflow guidance (THINK/REFLECT checklists, tool usage)
  → Zero project-specific references after v9 cleanup

Layer 2: Persistent Memory (MEMORY_LOG.md + SQLite)
  → Accumulated experimental results, dead ends, milestones
  → Pareto frontier, hypothesis calibration, causal history

Layer 3: Domain Knowledge (METHOD_PROPERTIES in domain_knowledge.py)
  → Generic scientific method properties: assumptions, failure symptoms, alternatives
  → 7 entries: EPI, FFT, attention, ResNet, sigmoid, Conv3D, contrastive

Layer 4: Architecture Patterns (IDEA_PATTERNS in idea_planner.py)
  → 6 reusable architectural patterns for plan generation
  → Pattern matching from innovation keywords

Layer 5: Runtime Context Injection (48 keys in loop.py _think() + _reflect())
  → Dynamic knowledge from current project state:
    architecture_plan, experiment_evaluation, domain_compatibility,
    data_constraints, training_curve_analysis, pareto_frontier, etc.
  → Constraint engine outputs (v10):
    plan_compliance_warning, quick_benchmark_warning,
    adaptive_thresholds, implementation_progress

Layer 6: Tool Chain Knowledge
  → domain_knowledge.py: Method-property-based analysis framework
  → idea_planner.py: 9-phase forward design pipeline
  → experiment_evaluator.py: Post-experiment diagnosis framework
  → model_analyzer.py: 9-layer AST structural analysis
  → constraint_engine.py: LLM behavior control (v10)
```

All layers except Layer 2 (project-specific memory) are fully domain-agnostic after v9.

### 30. Constraint Engine (`core/constraint_engine.py`) — v10

**LLM Behavior Control Layer**: Prevents hallucination, metric fabrication, and corner-cutting through 6 hard verifiable constraint mechanisms:

1. **PlannerChecker**: Scans implementation AST vs architecture plan. Detects missing modules, stub patterns (pass, NotImplementedError, hardcoded returns), and Code Agent freelancing. Generates `PlanComplianceReport` with compliance score and fabrication risk rating.

2. **StrategyConstraintEngine**: Learns constraint rules from SQLite history. Three rule sources:
   - Hypothesis calibration → confidence constraints (accuracy < 30%: must cite evidence)
   - Dead ends → forbidden approaches (failed 3+ times: FORBIDDEN)
   - Pareto frontier → dominated method elimination
   Rules persist in `STRATEGY_RULES.json`, checked after THINK dispatch.

3. **QuickBenchmark**: Loads checkpoint, runs forward pass on random input with dynamic shape inference. Compares output statistics vs reported metrics. Flags discrepancy > 20%. Runs as subprocess (120s timeout), never blocks main loop.

4. **AdaptiveThresholds**: Calibrates diagnostic thresholds from project's historical metric range. E.g., if metric range is [0.05, 0.35], `domain_gap_critical` = 0.24 instead of default 0.30.

5. **ImplementationTracker**: Persistent JSON tracking of planned module status across cycles. Prevents "pretending to be done" by injecting pending module lists into THINK context.

6. **ContextPruner**: 4-tier priority system that trims context to 20 keys max before LLM dispatch. Prevents information overload from masking critical constraints.

**Integration points**:
- `_think()`: StrategyConstraintEngine check, AdaptiveThresholds injection, ImplementationTracker prompt, ContextPruner pruning
- `_reflect()`: PlannerChecker compliance, QuickBenchmark verification, ImplementationTracker update, StrategyEngine rule generation, ContextPruner pruning
- `_think()` (plan generation): ImplementationTracker.update_from_plan()

### 31. Simulation Sandbox (`core/simulation_sandbox.py`) — v11

**Pre-Training Model Validation & A/B Evaluation**: A 5-layer evaluation system that answers whether model modifications are actually useful.

**Layer 0 — Feasibility Check** (PRE-VERIFY):
Runs the model in a subprocess: instantiate → forward → backward. Checks shape correctness, GPU memory estimation, crash detection. Blocks training if model cannot run.

**Layer 1 — Design Comparison** (REFLECT):
AST-based structural A/B comparison of before/after models. Tracks: parameter delta, new/removed modules, information bottleneck ratios (compress ratio), module parameter share percentages.

**Layer 2a — Reference Evaluation** (REFLECT):
Runs inference on 5-10 validation samples with both before and after models. Computes: per-sample MAE delta, parameter efficiency (MAE per 1K params), domain-specific breakdown.

**Layer 2b — Internal Behavior** (REFLECT):
Reference-free module-level analysis: activation dead ratio, gradient health (norm vs backbone baseline), parameter utilization (weight std), module contribution via ablation, data flow shape tracing.

**Layer 3 — Synthesis Judgment** (REFLECT):
Combines Layers 1+2a+2b into comprehensive verdict: effective/partial/ineffective/harmful. Checks project intent alignment by matching modification keywords against PROJECT_BRIEF core goals.

**Layer 4 — Scaling Guidance** (REFLECT):
Identifies scalable modules (active + healthy gradients + good parameter utilization), remaining bottlenecks, GPU memory budget headroom, and estimated max batch size after scaling.

**Key data structures**:
- `FeasibilityReport`: Layer 0 output (feasible, params, shapes, GPU memory)
- `DesignComparison`: Layer 1 output (A/B params, new modules, bottlenecks)
- `ReferenceEvaluation`: Layer 2a output (MAE delta, param efficiency)
- `InternalBehaviorReport`: Layer 2b output (dead/active modules, gradient balance)
- `SynthesisJudgment`: Layer 3 output (verdict, effective/ineffective modules, alignment)
- `ScalingGuidance`: Layer 4 output (scalable modules, GPU budget)
- `SandboxReport`: Full pipeline output combining all layers

**Integration points**:
- `_pre_verify()`: Layer 0 feasibility + model snapshot save
- `_reflect()`: Layers 1-4 full evaluation + verdict caching
- `_think()` (next cycle): Reads cached verdict as `sandbox_design_guidance`
