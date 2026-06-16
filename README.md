<h1 align="center">Auto Research Agent</h1>




<p align="center">
  <a href="README.md">English</a> |
  <a href="docs/README_CN.md">中文</a> |
</p>

> **Attribution**: This project was originally derived from [auto-deep-researcher-24x7](https://github.com/Xiangyue-Zhang/auto-deep-researcher-24x7).

---

## Recent Updates

**2026-05-27 (v16.1) — Runtime-Grounded Gate Overhaul & Dead Module Cleanup**

*Fixes the core problem: Phase Gate v1 never triggered (0/11 cycles), FORBIDDEN rules never activated (0/11 cycles), and PRE-EXECUTE "3-consecutive downgrade" leaked 4 training runs past the gate.*

### Problem Solved
v16 added Phase Gate, FORBIDDEN hard gate, and scope prefix injection — but none worked in production:
1. **Phase Gate v1 matched task text** (LLM generates abstract descriptions like "frequency analysis" that don't contain blocked_pattern keywords like `train`, `epoch`, `Conv`) → **0 triggers in 11 cycles**
2. **FORBIDDEN rules didn't exist** — `STRATEGY_RULES.json` was never created, auto-generated rules never have FORBIDDEN priority, and `generate_rules_from_history()` overwrote human rules → **0 triggers in 11 cycles**
3. **PRE-EXECUTE 3-consecutive downgrade** — after 3 HARD blocks, gate auto-downgraded to SOFT regardless of phase status → **4 training runs leaked through**
4. **Metrics pipeline incomplete** — `monitor._extract_metrics()` only extracted epoch/loss, no accuracy/AUC → Phase Status never auto-updated → Phase 1 stuck at PARTIAL forever
5. **4 dead modules** consumed ~793 lines but scored ≤3/10 effectiveness in runtime analysis
6. **Context overload** — 20+ keys diluted critical constraints, scope prefix injection ignored by LLM

### v16.1 Changes (6 fixes)

| # | Fix | Files | What Changed |
|---|-----|-------|-------------|
| 1 | **Phase Gate v2** | `loop.py` | `_check_phase_blocked()` rewritten: scans **actual code files** (model file + training script) instead of matching task text. Uses `_extract_model_path_from_task()` + `_find_training_script_content()` to get real code, then `re.search()` against blocked_patterns. |
| 2 | **FORBIDDEN rules** | `constraint_engine.py`, `STRATEGY_RULES.json` | Created human-authored FORBIDDEN rules (no training before Phase 1 validated, no skip-validation, no architecture switch). Fixed `generate_rules_from_history()` to preserve `source=human` rules instead of overwriting. |
| 3 | **Metrics pipeline** | `monitor.py` | Extended `_extract_metrics()` with AUC, FGD, FID, val_MAE patterns + generic key=value fallback. Enables Phase Status auto-update from experiment results. |
| 4 | **Phase-aware downgrade** | `loop.py` | PRE-EXECUTE CODE REVIEW: 3-consecutive downgrade now checks Phase Status — if phase is NOT VALIDATED, no downgrade (cap streak at 2). Only downgrades when phase is VALIDATED. |
| 5 | **Dead module removal** | `loop.py`, `constraint_engine.py` | Removed PlannerChecker (2/10), QuickBenchmark (1/10), AdaptiveThresholds (3/10), ImplementationTracker (2/10). Replaced with hardcoded thresholds. ~793 lines deleted. |
| 6 | **Context engineering** | `loop.py`, `constraint_engine.py`, `PERSISTENT_CONSTRAINTS.md` | Removed scope_prefix injection (ineffective), sandbox_design_guidance (context bloat). Added `PERSISTENT_CONSTRAINTS.md` loading in `_think()`. ContextPruner MAX_KEYS 20→14. Updated TIER lists. |

### New Files
- `<project>/workspace/STRATEGY_RULES.json` — 3 human FORBIDDEN rules (Phase 1 no-training, no-skip-validation, no-architecture-switch)
- `<project>/PERSISTENT_CONSTRAINTS.md` — Project-level hard constraints injected every THINK cycle

### Module Size After v16.1

| Module | v15.5 | v16.1 |
|--------|-------|-------|
| `core/loop.py` | ~4,350 | ~4,802 |
| `core/constraint_engine.py` | 1,164 | 422 |
| `core/monitor.py` | ~300 | 323 |
| **Change** | | **constraint_engine -742 lines (-64%)** |

---

**2026-05-25 (v15.5) — Research ROADMAP: Module-Level State Machine & Phase-Gated Research**

*Prevents premature model training and enforces structured theory verification before committing GPU resources.*

### Problem Solved
Agent skipped theory verification and jumped directly to model training, wasting GPU hours on architectures whose underlying assumptions were never validated. Module verification used a hard "exactly 3 methods" requirement that was too rigid — some directions need more methods to properly disprove, while the dual deviation counter between ROADMAP and ResearchLoop could desync, causing stale state.

### New Module (v15)
- **`core/research_roadmap.py`** — Module-level state machine tracking research phases per module:
  `theory_verification → module_design → module_validation → integrated → dead_end`

### New Mechanisms (v15)

| # | Mechanism | Phase | What It Does |
|---|-----------|-------|-------------|
| 1 | **Research ROADMAP** | THINK | Tracks per-module research phase. Each module must pass through theory_verification → module_design → module_validation before being marked integrated. |
| 2 | **Phase-Gated Research** | THINK → EXECUTE | Code-level enforcement prevents premature model training during theory_verification phase. Only allows analysis/research tasks. |
| 3 | **3-Strike Hard Gate** | THINK → EXECUTE | Deviation from ROADMAP phase: 1st = warning, 2nd = stronger warning, 3rd = force override to `paper_research`. Dual counter sync between `ResearchRoadmap._deviation_count` and `ResearchLoop._phase_violation_count`. |
| 4 | **Circuit Breaker Priority** | THINK | ROADMAP overrides direction/architecture circuit breakers during theory_verification. |
| 5 | **Flexible Method Verification** | VERIFY | 3-6 methods required per module (min 3, max 7) before dead_end marking. Not a hard 3 — supports directions needing deeper analysis. |
| 6 | **Assumption-Embedded Methods** | THINK | `_suggest_verification_methods` embeds assumption text in every method description, adds cross-assumption consistency check when ≥2 assumptions exist. |

### v15.5 Hardening (8 Code Review Fixes)

| # | File | Fix |
|---|------|-----|
| 1 | `loop.py` + `research_roadmap.py` | `check_alignment()` no longer returns `should_force_paper_research`; loop.py solely controls 3-strike enforcement via `_phase_violation_count`. Sync-resets `roadmap._deviation_count` on hard gate trigger. |
| 2 | `research_roadmap.py` + `loop.py` | New public properties `active_module_names` and `is_theory_verification_phase` replace private `_get_active_modules()` access. |
| 3 | `research_roadmap.py` | `_is_training_task` rewritten with strong_indicators (e.g. "train the", "training", "epoch") and weak_indicators (e.g. "loss function", "loss curve") to avoid false positives on "information loss", "loss of detail". |
| 4 | `research_roadmap.py` | `_is_task_related` improved with sub-token matching via `re.split(r"[_\s]+")`, dynamic threshold `max(2, len(words)//3)`, lower word length threshold (>3 instead of >4). |
| 5 | `research_roadmap.py` | `MODULE_DESIGN` milestone handling: keywords "designed", "implemented", "coded", "built" advance to `MODULE_VALIDATION`. |
| 6 | `idea_planner.py` | `_suggest_verification_methods` embeds assumption text in every method description, adds cross-assumption consistency check (method 4) when ≥2 assumptions exist. |
| 7 | `research_roadmap.py` | Markdown parser regex changed from `\[\\w+\]` to `\[\\w_-+\]`, added evidence extraction from next line. |
| 8 | `context_keys.py` | Removed orphan `roadmap_alignment_warning` ContextKey (registered but never used in context injection flow). |

### Key Changes
- `core/research_roadmap.py`: New ROADMAP module (~500 lines)
- `core/loop.py`: `_enforce_roadmap_alignment()`, ROADMAP-aware direction circuit breaker
- `core/idea_planner.py`: Cross-assumption consistency check in method suggestions
- `core/context_keys.py`: Cleanup of orphan key
- `agents/leader.md`: v15 rules for ROADMAP awareness and phase constraints

### Module Size After v15.5

| Module | v14 | v15.5 |
|--------|-----|-------|
| `core/loop.py` | ~4,150 | ~4,350 |
| `core/research_roadmap.py` | — | ~500 |
| `core/idea_planner.py` | 1,059 | ~1,070 |
| `core/context_keys.py` | 194 | 193 |
| **Total** | **~21,750** | **~22,470** |

**2026-05-20 (v14) — Strategic Architecture Intelligence**

*Prevents the #1 failure mode: spending dozens of cycles patching a fundamentally wrong architecture.*

### Problem Solved
In a 64-cycle run, the agent spent ALL 64 cycles patching EPINet (baseline from PROJECT_BRIEF) without ever researching alternative architectures. Only 12.5% of cycles did paper_research, and none led to architecture switching. Root causes: (1) no architecture survey before commitment — agent blindly used PROJECT_BRIEF's baseline from cycle 1; (2) dead ends were recorded individually but never clustered into "EPINet architecture is the bottleneck"; (3) direction signatures were too fine-grained — "EPINet+edge_loss" and "EPINet+angular_conv" counted as different directions; (4) paper_research reset all stagnation counters, masking persistent architectural problems.

### New Mechanisms (v14)
1. **Architecture Survey Gate** — Forces a 3+ candidate architecture survey in cycles 1-2 before committing to any baseline. Outputs `ARCHITECTURE_SURVEY.md`.
2. **Architecture-Level Direction Signature** — Detects the underlying architecture (EPINet, U-Net, Transformer, etc.) regardless of direction keywords. Architecture stagnation accumulates across all directions on the same architecture.
3. **Dead End Synthesis Engine** — Clusters dead ends by architecture and auto-detects when 5+ dead ends trace to the same architecture → `[ARCHITECTURE BOTTLENECK]` warning.
4. **Architecture Switch Enforcer** — When architecture stagnation reaches threshold (5 cycles), forces `architecture_switch` action that mandates switching to a fundamentally different architecture (not a variant).

### Key Changes
- `core/loop.py`: Added `_architecture_stagnation_count`, `_extract_architecture_name()`, `_analyze_architecture_dead_ends()`, `architecture_switch` action handler, architecture survey gate injection, architecture circuit breaker.
- `core/domain_knowledge.py`: Added `_synthesize_architecture_dead_ends()` with architecture-level dead end clustering.
- `agents/leader.md`: Added "Architecture Intelligence Rules" section with 4 mandatory rules.

### Problem Solved
The agent repeated the same mistakes across cycles: (1) the PRE-EXECUTE Code Review's regex checks repeatedly triggered on the same false positive (matching keywords in comments/docstrings, not actual code), creating an infinite HARD GATE loop that blocked ALL training for 9+ consecutive cycles; (2) verified architectural flaws (dead ends, module failures) were recorded as text in MEMORY_LOG.md but never systematically extracted into reusable lessons; (3) API timeout of 300s caused the agent to hang for 25+ minutes on a single LLM call; (4) `reflect_result` fields containing `null` (Python `None`) crashed SQLite recording with `'NoneType' object is not subscriptable`.

### Root Cause Analysis
1. **Regex false positives**: Check 1 (`routing without aux_loss`) matched `auxiliary` in docstrings/comments, not distinguishing real code from documentation. Each cycle blocked training → code agent "fixed" → next cycle same regex triggered → dead loop.
2. **No institutional memory**: Dead ends and module failures were logged but never extracted into structured, searchable lessons that could be injected into future THINK phases.
3. **API timeout too long**: 300s timeout meant a single hung request could waste 5+ minutes. Combined with retries, one bad API call could stall the agent for 15+ minutes.
4. **NoneType in SQLite**: LLM JSON responses with `null` values (e.g. `"milestone": null`) caused `dict.get("milestone", "")` to return `None` (key exists), then `None[:500]` crashed.

### New Mechanisms

| # | Mechanism | Phase | What It Does |
|---|-----------|-------|-------------|
| 1 | **Code Review Lessons Knowledge Base** | Memory | New `code_review_lessons` SQLite table with pattern-based dedup, severity ranking (HIGH > MEDIUM > LOW), hit count tracking, and keyword-based relevance search. 4 new methods in `MemoryManager`. |
| 2 | **Post-Reflect Lesson Extraction** | REFLECT → Memory | After each REFLECT, automatically extracts lessons from: (a) VERIFY failures (per-check), (b) dead-end decisions (keyword-based pattern), (c) module failures (keyword-based pattern), (d) LLM-based semantic analysis when other methods don't find specific lessons. |
| 3 | **THINK Phase Lesson Injection** | THINK | Loads relevant code review lessons (keyword-matched against latest model file) and injects as `relevant_code_review_lessons` context. Uses mtime-based caching to avoid re-reading unchanged model files. |
| 4 | **HARD GATE Dead-Loop Detection** | THINK → EXECUTE (gate) | Tracks `_hard_gate_consecutive_blocks` counter. After 2 consecutive HARD blocks on the same issue, auto-downgrades to SOFT GATE (warning only, doesn't block training). Resets when code review passes cleanly. |
| 5 | **Comment-Aware Regex Checks** | THINK → EXECUTE (gate) | New `_strip_comments_and_strings()` static method removes comments, docstrings, and string literals before regex matching. Prevents false positives from documentation keywords. |
| 6 | **Enhanced LOW VOI Guidance** | THINK | LOW VOI experiments now strongly suggest switching to `paper_research` to find new approaches, instead of just warning. |
| 7 | **`code_review` Tool** | EXECUTE | New tool registered for Code Agent: structural code analysis (Conv2d channel asymmetry, routing supervision, 1×1 conv router). |
| 8 | **API Timeout Reduction** | EXECUTE | Timeout reduced from 300s to 120s to prevent long hangs on unresponsive API calls. |
| 9 | **Two-Phase Code Review** *(v12.3)* | THINK → EXECUTE (gate) | Phase 1: Zero-LLM regex checks (routing supervision, channel asymmetry, 1×1 conv router). Phase 2: LLM semantic review using cheap fast model (only runs when Phase 1 has no HIGH issues). |
| 10 | **Log Fallback Parser** *(v12.3)* | VERIFY | `_parse_log_text_to_json()` — when `training_log.json` doesn't exist, extracts routing weights, aux losses, and per-domain MAE from raw stdout/log text as fallback. |
| 11 | **Dynamic Domain Discovery** *(v12.3)* | VERIFY | Per-domain baseline regression no longer uses hardcoded domain names. Parses any `DOMAIN_NAME: MAE=X.XXX` or `MAE_DOMAIN: X.XXX` pattern dynamically. |
| 12 | **Ordered Gate Pipeline** *(v12.4)* | THINK → EXECUTE | Three gates execute in priority order: (1) PRE-VERIFY (critical preconditions), (2) CODE REVIEW (architectural defects), (3) FALSIFIABILITY (hypothesis quality, soft gate). A hard-gate firing causes subsequent gates to skip entirely, preventing gate conflicts. |

### Key Design Decisions
- **Pattern-based dedup**: Lessons use auto-generated patterns (first significant words from content) instead of fixed strings, preserving distinctiveness of different dead ends/failures.
- **Severity ranking in Python**: SQLite `MAX()` doesn't work for TEXT severity (lexicographic order is wrong: HIGH < LOW < MEDIUM). Severity upgrade comparison done in Python with explicit rank mapping.
- **IN clause for severity filter**: `get_code_review_lessons(severity="MEDIUM")` uses `IN ('HIGH', 'MEDIUM')` instead of integer comparison against TEXT column.
- **Mtime caching for model file**: Only re-reads model file when mtime changes, avoiding redundant I/O every THINK cycle.
- **Knowledge closed loop**: Failure → Extract → Store → Inject → Avoid repeat. The system learns from its own mistakes across cycles.

### Bug Fixes
- **`MAX(severity, ?)` on TEXT column** (CRITICAL): SQLite lexicographic comparison made HIGH < LOW < MEDIUM, completely inverting severity upgrade logic. Fixed with Python-side comparison.
- **Integer vs TEXT comparison in severity filter** (CRITICAL): `severity >= 1` compared int against TEXT column. Fixed with `IN (...)` clause.
- **`NoneType` subscriptable in SQLite recording** (HIGH): `reflect_result.get("milestone", "")` returned `None` when LLM set `milestone: null`. Fixed with `(x or "")[:500]` pattern.
- **`search_relevant_lessons` full table scan** (MEDIUM): Added `LIMIT 100` to prevent loading entire table into Python memory.
- **Fixed lesson patterns**: Dead-end and module-failure lessons used fixed patterns (`"dead_end_approach"`, `"module_failure"`), causing all distinct failures to merge into one lesson. Now uses content-derived keywords.

### v13.1 Hardening (Code Review Fixes)
- **`_strip_comments_and_strings` cross-line bug**: Multi-line `"""..."""` docstrings were processed line-by-line, failing to remove cross-line content. Now removes multi-line strings on full content first, then processes lines.
- **`gate` keyword false positive**: Regex check #1 matched `gate` in any context (e.g. `torch.sigmoid` gate patterns). Changed to `gate_weight|gate_network|gating` for precision.
- **`search_relevant_lessons` scoring**: High `hit_count` lessons with no keyword match were returned. Now requires at least one keyword match; `hit_count` used as tiebreaker (capped at 10).
- **`record_code_review_lesson` double SQL query**: UPDATE path did separate SELECT + UPDATE. Merged into single query (SELECT includes `severity`).
- **`_llm_extract_lesson` JSON response**: Tool-based APIs may return JSON-wrapped responses. Added JSON parse fallback before structured `|` parsing.
- **Training script check independent of model**: `_pre_execute_code_review` returned early when model file unavailable, skipping training script checks. Now runs training script checks even without model code.
- **Gate 2 double `if` cleanup**: `if not warnings: reset` + `if warnings: process` merged into `if/else`.

### v13.2 — Multi-Tool Skill Integration + Python API
- **Multi-tool installer**: `install.py` now supports `--claude-code`, `--codebuddy`, `--cursor`, `--all` (default). Installs skill commands to each tool's native directory.
- **Python API** (`api.py`): New `AutoResearcher` class for programmatic access — `run_one_cycle()`, `run_n_cycles()`, `start_daemon()`, `stop_daemon()`, `get_status()`, `get_code_review_lessons()`.
- **CLI API** (`python api.py {status|run|start|stop|lessons}`): Command-line interface for tool integration and scripting.
- **Cursor support**: Skills install as `.mdc` rule files to `~/.cursor/rules/`.
- **CodeBuddy support**: Skills install as `.md` commands to `~/.codebuddy/commands/`.

### Context Keys Added
- THINK: `relevant_code_review_lessons` (auto-matched from knowledge base)
- EXECUTE: `code_review` tool for Code Agent
- REFLECT → Memory: automatic lesson extraction pipeline

### Flow Diagram
```
REFLECT (cycle ends)
  → [NEW] Post-Reflect Lesson Extraction
    → Extract from VERIFY failures (per-check)
    → Extract from dead-end decisions (keyword pattern)
    → Extract from module failures (keyword pattern)
    → [NEW] LLM semantic analysis (fallback)

THINK (next cycle)
  → [NEW] Load relevant code_review_lessons (keyword-matched)
  → [NEW] Inject as context for Leader
  → Plan experiment

PRE-EXECUTE Code Review
  → [NEW] Strip comments/strings before regex matching
  → [NEW] HARD GATE dead-loop detection (auto-downgrade after 2)
```

### Module Size After v13

| Module | v12.1 | v12.2 | v13 | v13.2 | v14 | v15.5 |
|--------|-------|-------|-----|-------|-----|-------|
| `core/loop.py` | 2,992 | 3,248 | ~3,904 | 3,919 | ~4,150 | ~4,350 |
| `core/memory.py` | 963 | 963 | ~1,130 | 1,134 | 1,134 | 1,134 |
| `core/agents.py` | 1,357 | 1,357 | ~1,411 | 1,411 | 1,411 | 1,411 |
| `core/tools.py` | 1,739 | 1,739 | ~1,884 | 1,884 | 1,884 | 1,884 |
| `api.py` | — | — | — | 364 | 364 | 364 |
| `core/domain_knowledge.py` | — | — | — | — | ~660 | ~660 |
| `core/research_roadmap.py` | — | — | — | — | — | ~500 |
| Other modules | — | — | unchanged | unchanged | unchanged | unchanged |
| **Total** | **~19,071** | **~19,734** | **~20,850** | **~21,220** | **~21,750** | **~22,470** |

**2026-05-18 (v12.2) — Pre-Execute Code Review + Training Architecture Verification**

### Problem Solved
Agent-generated model code (e.g. MaterialDualCueNet) contained multiple architectural flaws (routing weights never differentiating, aux loss too low, input channel information asymmetry) that were NOT detected before training. The agent wasted GPU hours and LLM tokens training architecturally flawed models, then the REFLECT phase couldn't diagnose WHY the model failed because v12's analysis reflection only covered data analysis experiments, not training experiments.

### Root Cause Analysis
The agent had two blind spots:
1. **No code review before training**: Code agent writes model → immediately trains → VERIFY only checks if training ran, not if the architecture design is sound.
2. **No training architecture reflection**: v12's `analysis_reflection_prompt` only triggers for analysis experiments (`experiment_launched=False`). Training experiments get no specialized reflection about routing convergence, aux loss effectiveness, or per-domain regression.

### New Mechanisms

| # | Mechanism | Phase | What It Does |
|---|-----------|-------|-------------|
| 1 | **Pre-Execute Code Review** | THINK → EXECUTE (gate) | Zero-LLM structural review of model code BEFORE training. Checks: routing without aux supervision, input channel asymmetry (>5x ratio), 1×1 conv router (no spatial context), weighted fusion without skip connection. Issues injected as mandatory-fix preamble in task. |
| 2 | **Layer 12: Training Architecture Verification** | VERIFY | Post-training checks: (a) routing weight differentiation — all domains within 5% = FAIL, (b) aux loss convergence — flat across epochs = FAIL, (c) per-domain regression vs baseline — >20% degradation = WARN. |
| 3 | **Training Architecture Reflection Prompt** | REFLECT | When VERIFY Layer 12 detects issues, injects specialized reflection forcing the Leader to evaluate routing convergence, aux loss effectiveness, per-domain regression, and correct failure categorization. Only triggers for training experiments (`experiment_launched=True`). |
| 4 | **Structured Metrics Injection** | REFLECT | Parses `training_log.json` for per-domain MAE trends (epoch-by-epoch) and aux loss trends. Flags domains getting WORSE. Replaces raw-log regex parsing with structured JSON parsing. |

### Key Design Decisions
- **Pre-Execute Review is a GATE, not a BLOCK**: Issues are injected as warnings in the task preamble, not hard blocks. This prevents false positives from stopping legitimate experiments. The Code agent sees the warnings and can choose to fix or proceed.
- **Layer 12 uses `training_log.json`**: Structured JSON parsing (not regex on raw log text) for reliable metric extraction. Falls back to regex parsing of log text when JSON unavailable.
- **Baseline metrics from MEMORY_LOG.md**: Per-domain regression compares against the most recently recorded baseline in MEMORY_LOG.md. No extra state file needed.
- **`_pre_execute_code_review` is zero-LLM**: All checks are regex/AST-based, no LLM calls. Adds ~50ms latency per cycle.

### Context Keys Added
- THINK → EXECUTE: code review warnings injected into task (pre-execute gate)
- VERIFY: Layer 12 `routing_differentiation`, `aux_loss_convergence`, `domain_regression` checks
- REFLECT: `training_architecture_reflection_prompt`, `per_domain_mae_trend`, `aux_loss_trend`

### Flow Diagram
```
THINK (plan experiment)
  → Pre-Verify (data, imports)
  → [NEW] Pre-Execute Code Review (model architecture)
    → Issues found? → Inject as warnings in task
  → EXECUTE (Code agent writes + trains model)
  → VERIFY (Layer 1-11 as before)
  → [NEW] VERIFY Layer 12 (training architecture checks)
  → [NEW] VERIFY Layer 13 (aux loss convergence)
  → REFLECT
    → [NEW] training_architecture_reflection_prompt (if Layer 12 issues)
    → [NEW] per_domain_mae_trend + aux_loss_trend
```

### Module Size After v12.2

| Module | v12.1 | v12.2 | v13 |
|--------|-------|-------|-----|
| `core/loop.py` | 2,992 | 3,248 | ~3,904 |
| `core/verifier.py` | 2,210 | 2,537 | 2,537 |
| Other modules | — | unchanged | see above |
| **Total** | **~19,071** | **~19,734** | **~20,850** |

**2026-05-18 (v12.1) — API Quota Exhaustion: REFLECT Degraded Fallback & Clean Shutdown**

### Problem Solved
When API quota is exhausted during the REFLECT phase, the entire cycle's results (EXECUTE + VERIFY) were lost. The agent entered a 600-second backoff loop that would never recover (retrying won't restore quota).

### Changes
- **REFLECT Degraded Fallback** (`_degraded_reflect`): When `dispatch_leader()` fails with `insufficient_quota` / `All providers failed` / `429`, a rule-based fallback generates a basic reflection from VERIFY reports, training logs, and analysis outputs. The output follows the same JSON schema as Leader REFLECT, so downstream code (`_record_cycle_outcome`, `_update_state`) works unchanged.
- **Degraded Reflect Pending Note**: Writes `.degraded_reflect_pending` JSON file. The next cycle's THINK phase reads this and injects a `degraded_reflect_pending` context, prompting the Leader to revisit the incomplete cycle before planning new work.
- **Clean Shutdown on Quota Error**: Outer `except` block now detects quota errors and breaks the main loop cleanly (no 600s backoff for unrecoverable errors).
- **Context Key Added**: THINK: `degraded_reflect_pending` (injected when previous cycle had degraded REFLECT).

**2026-05-18 (v12) — Exploratory Analysis Mode: Preventing False-Negative Dead Ends**

### Problem Solved
Agent concluded "angular frequency material classification is infeasible" after using only 1 method (FFT energy ratios, 5 features, Cohen's d max 0.47). But with 5 methods and 17 features, Cohen's d reached 2.06, AUC 0.969 — the direction was perfectly viable. The agent's analysis was too narrow, producing a false-negative dead end.

### New Mechanisms

| # | Mechanism | Phase | What It Does |
|---|-----------|-------|-------------|
| 1 | **Exploratory Analysis Mode** | VERIFY (Layer 11) | Scans analysis output for method families (FFT, statistical, view-consistency, spatial, frequency-decomposition). Requires ≥ 3 independent methods before concluding a direction is infeasible. |
| 2 | **Feature Completeness Report** | VERIFY | Checks that the analysis covers ≥ 4 feature families. Generates structured JSON report with `feature_families_tested`, `methods_with_strong_signal`, `recommendation`. |
| 3 | **Failure Category System** | REFLECT + Memory | Dead ends now carry a structured `failure_category`: `hypothesis_wrong`, `implementation_bug`, `insufficient_experiment`, or `method_inadequacy`. Stored in SQLite `failure_category` column (auto-migrated). |
| 4 | **Method Inadequacy Re-awakening** | THINK | When dead ends exist with `failure_category='method_inadequacy'`, the Leader receives a prompt encouraging retry with broader analysis instead of abandoning the direction. |
| 5 | **Analysis Reflection Prompt** | REFLECT | When the experiment was a data analysis (not training), injects specialized reflection prompts forcing the Leader to evaluate method coverage, feature completeness, and correct failure categorization. |

### Key Design Decisions
- **`failure_category` column**: Auto-migrated via `ALTER TABLE` in `log_dead_end()`. Pre-v12 databases continue to work (empty string default).
- **`get_dead_ends_by_category()`**: Structured retrieval of dead ends by category, enabling the re-awakening mechanism.
- **`get_method_inadequacy_count()`**: Quick count query for THINK phase injection.
- **Layer 11 `_verify_analysis_coverage()`**: Pattern-based method detection in analysis output files. If only 1 method family found, generates a warning that the Leader MUST categorize as `method_inadequacy` (not `hypothesis_wrong`).

### Context Keys Added
- THINK: `method_inadequacy_retry_prompt`, `degraded_reflect_pending` (v12.1)
- VERIFY: Layer 11 analysis coverage checks
- REFLECT: `analysis_reflection_prompt`, `method_inadequacy_history`

### Module Size After v12

| Module | v11 | v12 | v12.1 | v13 |
|--------|-----|-----|-------|-----|
| `core/loop.py` | ~2,660 | ~2,860 | ~2,992 | ~3,904 |
| `core/memory.py` | 852 | ~960 | ~963 | ~1,130 |
| `core/verifier.py` | 2,102 | ~2,200 | 2,210 | 2,537 |
| `core/tools.py` | 1,739 | 1,739 | 1,739 | 1,884 |
| `core/model_analyzer.py` | 2,323 | 2,323 | 2,323 | 2,323 |
| `core/simulation_sandbox.py` | ~600 | ~600 | 1,518 | 1,518 |
| `core/constraint_engine.py` | ~580 | ~580 | 1,164 | 1,164 |
| `core/agents.py` | — | — | 1,357 | ~1,411 |
| Other modules | — | — | 7,317 | unchanged |
| **Total** | **~17,204** | **~17,500** | **~19,071** | **~20,850** |

**2026-05-13 (v11) — Simulation Sandbox: Pre-Training Model Validation & A/B Evaluation**

### Problem Solved
Agent修改模型后直接训练，无法回答：
1. **修改有没有用？** — 加了模块但模块可能根本没激活
2. **和修改前比怎样？** — 参数量翻了倍但指标没变
3. **能不能继续放大？** — 不知道瓶颈在哪，盲目扩展导致OOM
4. **是否背离项目初衷？** — PROJECT_BRIEF说要解决X，修改却在优化Y

### New Module
- **`core/simulation_sandbox.py`** (~600 lines): 5层评价体系

| Layer | Name | When | What |
|-------|------|------|------|
| 0 | Feasibility Check | PRE-VERIFY | 模型能不能跑？shape对不对？GPU够不够？ |
| 1 | Design Comparison | REFLECT | A/B结构对比：参数变化、模块占比、信息瓶颈 |
| 2a | Reference Evaluation | REFLECT | vs GT指标 + vs 修改前指标 + 参数效率 |
| 2b | Internal Behavior | REFLECT | 模块活性/贡献度/梯度健康/参数利用率 |
| 3 | Synthesis Judgment | REFLECT | 综合判定：有效/部分/无效/有害 + 项目初衷对齐 |
| 4 | Scaling Guidance | REFLECT | 可扩展模块/瓶颈/显存预算/规模扩展建议 |

### Key Design Decisions
- **Model snapshots**: 每次PRE-VERIFY自动保存当前模型快照到`model_snapshots/`，下一cycle的A/B对比用
- **Verdict caching**: 评价结果缓存到`_sandbox_last_verdict.json`，下一cycle的THINK阶段读取作为设计指引
- **Subprocess isolation**: 所有模型运行都在subprocess中，主循环不受影响
- **Format as prompt**: `format_report_prompt()` 将结构化数据转换为LLM可理解的中文prompt

### Context Keys Added
- THINK: `sandbox_design_guidance` (from previous cycle's scaling guidance)
- REFLECT: `sandbox_evaluation` (full 5-layer report)

**2026-05-13 (v10) — Constraint Engine: LLM Behavior Control**

### Problem Solved
LLMs as agent brains have three critical weaknesses that no amount of prompt engineering can fix:
- **Hallucination**: Claims to have done things it didn't actually do
- **Metric Fabrication**: Reports great metrics while the model outputs garbage
- **Corner-cutting**: Skips difficult modules, uses stub/empty implementations

v10 introduces **hard verifiable constraints** — every check is machine-verifiable, not relying on LLM self-reporting.

### New Module
- **`core/constraint_engine.py`** (~580 lines → 422 lines in v16.1): Originally 6 constraint mechanisms, now 2 after v16.1 dead module removal:

| # | Mechanism | Phase | LLM Problem Addressed |
|---|-----------|-------|-----------------------|
| 1 | **StrategyConstraintEngine** | THINK | Repeating failed approaches, ignoring historical lessons |
| 2 | **ContextPruner** | THINK+REFLECT | Information overload causing LLM confusion (14-key limit in v16.1) |

**Removed in v16.1** (dead modules, scored ≤3/10 in runtime analysis):
| # | Mechanism | Reason for Removal |
|---|-----------|-------------------|
| ~~PlannerChecker~~ | AST compliance check never useful — PlannerChecker only produced warnings (score 2/10) |
| ~~QuickBenchmark~~ | Conditions too strict, never triggered (score 1/10) |
| ~~AdaptiveThresholds~~ | Insufficient data for calibration, always fell back to defaults (score 3/10) |
| ~~ImplementationTracker~~ | Overlapped with research_roadmap functionality (score 2/10) |

### How It Works

**StrategyConstraintEngine**: Reads SQLite history (hypothesis calibration, dead ends, Pareto frontier) and generates executable constraint rules. Example: "edge loss failed 5 times → FORBIDDEN", "hypothesis accuracy < 30% → must cite evidence before proposing experiments". Rules persist in `STRATEGY_RULES.json`. Violations are checked after THINK dispatch and injected back into memory. v16.1: Fixed `generate_rules_from_history()` to preserve human-authored rules (`source=human`).

**ContextPruner**: 4-tier priority system (always > situational > conditional > rare). Trims context dict to **14 keys max** (reduced from 20 in v16.1) before dispatching to LLM. Ensures critical constraints (`persistent_constraints` added in v16.1) aren't drowned out by low-priority information.

### Context Key Registry Update (v16.1)
- THINK keys: 21 → 18 (removed `adaptive_thresholds`, `implementation_progress`, `sandbox_design_guidance`; added `persistent_constraints`)
- REFLECT keys: 27 → 24 (removed `plan_compliance_warning`, `quick_benchmark_warning`, `implementation_progress`)
- Total: ~42 registered context keys with validation

### Bug Fixes (v9→v10)
- **`memory.py` dead code bug (CRITICAL)**: `__init__` initialization code (`mkdir`, `_init_log()`, `_init_db()`) was unreachable after `return []` in `_infer_domain_keys()` — persistent memory system silently failed
- **`experiment_evaluator.py` hardcoded input shapes**: `IndependentProbe` had `(1, 81, 3, 64, 64)` — replaced with dynamic inference from state_dict
- **`loop.py` `_last_architecture_plan` never assigned**: Architecture plan generated but thrown away each cycle, so `_reflect()` always re-generated
- **`verifier.py`/`loop.py` private attribute access**: `VerifyReport._independent_assessment` → formal `independent_assessment` dataclass field
- **13 silent `except Exception: pass` blocks** in `loop.py` → replaced with `logger.warning/debug`
- **`idea_planner.py` dependency chains**: Secondary pattern modules incorrectly depended on previous module instead of backbone

### Module Size After v10

| Module | v8 | v9 | v10 | v11 | v12.1 | v12.2 | v13 |
|--------|-----|-----|------|------|-------|-------|-----|
| `core/tools.py` | 1,667 | 1,740 | 1,739 | 1,739 | 1,739 | 1,739 | 1,884 |
| `core/loop.py` | 2,268 | 2,355 | ~2,450 | ~2,660 | 2,992 | 3,248 | ~3,904 |
| `core/verifier.py` | 2,010 | 2,099 | 2,102 | 2,102 | 2,210 | 2,537 | 2,537 |
| `core/model_analyzer.py` | 2,282 | 2,323 | 2,323 | 2,323 | 2,323 | 2,323 | 2,323 |
| `core/memory.py` | 831 | 852 | 852 | 852 | 963 | 963 | ~1,130 |
| `core/visual_analyzer.py` | 853 | 853 | 848 | 848 | 848 | 848 | 848 |
| `core/idea_planner.py` | (new) | 1,053 | 1,053 | 1,053 | 1,059 | 1,059 | 1,059 |
| `core/experiment_evaluator.py` | (new) | 786 | 784 | 784 | 800 | 800 | 800 |
| `core/constraint_engine.py` | — | — | ~580 | ~580 | 1,164 | 1,164 | 1,164 |
| `core/context_keys.py` | — | — | 194 | 194 | 194 | 194 | 194 |
| `core/simulation_sandbox.py` | — | — | — | ~600 | 1,518 | 1,518 | 1,518 |
| `core/agents.py` | — | — | — | — | 1,357 | 1,357 | ~1,411 |
| `core/domain_knowledge.py` | 559 | 559 | 559 | 559 | 565 | 565 | 565 |
| `core/mcp_client.py` | 724 | 724 | 724 | 724 | 724 | 724 | 724 |
| **Total** | **~13,307** | **~15,244** | **~16,604** | **~17,204** | **~19,071** | **~19,734** | **~20,850** |

**2026-05-13 (v9) — Forward Design Pipeline, Post-Experiment Evaluation & Domain-Agnostic Cleanup**

### New Modules
- **`core/idea_planner.py`** (1053 lines): `IdeaPlanner` — 9-phase forward design pipeline that generates PhD-level architecture plans from `PROJECT_BRIEF.md` before any model code is written:
  - Idea Formalization → Module Decomposition → Capacity Planning → Fusion Strategy → Integration Plan → Verification Plan → Risk Assessment → Implementation Order → Alignment Score
  - `IDEA_PATTERNS` database: 6 generic architectural patterns (multi_branch_fusion, frequency_domain_analysis, domain_adaptive, component_aware, attention_mechanism, progressive_refinement)
  - Vocabulary-to-concept mapping for innovation detection (natural language terms → abstract concept categories)
- **`core/experiment_evaluator.py`** (786 lines): Three post-experiment analysis classes:
  - `ExperimentEvaluator`: Plan-vs-result comparison, 5-type failure diagnosis (architecture/data/training/alignment/capacity), priority-sorted iteration guidance
  - `IndependentProbe`: Third-party verification using model checkpoint + forward pass on random input to detect output anomalies (collapsed, NaN, range mismatch)
  - `IterationGuidance` + `FailureDiagnosis`: Structured analysis with root cause chain and actionable next-step recommendations

### Pipeline Integration
- **`plan_model` tool** registered in ToolRegistry — Leader and Code Agent can invoke architecture planning
- **THINK phase** (cycle ≤ 1): Auto-injects `architecture_plan` + `architecture_plan_summary` context from IdeaPlanner
- **REFLECT phase**: Auto-injects `experiment_evaluation` + `iteration_guidance_prompt` + `independent_assessment_warning` from ExperimentEvaluator
- **VERIFY phase**: New Layer 10 (`_verify_independent_probe`) runs IndependentProbe as third-party assessment
- **Agent prompts** (`leader.md`, `code_agent.md`): Added plan_model usage guide, post-evaluation response table, independent assessment handling

### Domain-Agnostic Hardcoding Cleanup
Systematic removal of all project-specific hardcoding across 10 files. The principle: **only knowledge organization methods may be hardcoded; project-specific data/scenarios must be dynamic**.

- **Agent prompts** (`agents/*.md`): All project-specific examples generalized (EPI/Lambertian/HCInew/UnifiedLFDataset/81-angular-views → generic domain/method/dataset language)
- **`core/loop.py`**: Cross-domain analysis from hardcoded metric names → dynamic `self.memory.domain_keys` + pattern scanning; domain breakdown prompts → generic
- **`core/verifier.py`**: Dataset→domain mapping from hardcoded dict → `DATASET_MANIFEST.json` `type` field + heuristic fallback; class discovery from hardcoded list → AST-based dynamic scanning
- **`core/model_analyzer.py`**: Default input shape from `[1,81,3,64,64]` → `_infer_input_shape()` reading Conv3d/Conv2d `in_channels`; assumption descriptions → generic scientific knowledge
- **`core/memory.py`**: Default keywords/domains from depth-estimation-specific → `_default_method_keywords()` / `_infer_domain_keys()` reading manifest
- **`core/visual_analyzer.py`**: Removed inference script hardcoding; domain gap examples → generic
- **`core/idea_planner.py`**: Added docstrings explaining vocabulary-to-concept mapping design principle; fixed duplicate domain entry

### Knowledge Organization Architecture (6 Layers)
1. **Static prompts** (`agents/*.md`): Generic workflow guidance, now zero project-specific references
2. **Persistent memory** (`MEMORY_LOG.md` + SQLite): Accumulated experimental results and dead ends
3. **Domain knowledge** (`METHOD_PROPERTIES`): Generic scientific method properties (assumptions, failure symptoms, alternatives)
4. **Architecture patterns** (`IDEA_PATTERNS`): 6 reusable architectural patterns for plan generation
5. **Runtime context injection** (27 keys in `loop.py`): Dynamic knowledge from project state (manifest, metrics, plan, evaluation)
6. **Tool chain knowledge**: `domain_knowledge.py`, `idea_planner.py`, `experiment_evaluator.py` as organizational methods

### Module Size After v9

| Module | v8 | v9 |
|--------|-----|-----|
| `core/tools.py` | 1,667 | 1,740 |
| `core/loop.py` | 2,268 | 2,355 |
| `core/verifier.py` | 2,010 | 2,099 |
| `core/model_analyzer.py` | 2,282 | 2,323 |
| `core/memory.py` | 831 | 852 |
| `core/visual_analyzer.py` | 853 | 853 |
| `core/idea_planner.py` | (new) | 1,053 |
| `core/experiment_evaluator.py` | (new) | 786 |
| `core/domain_knowledge.py` | 559 | 559 |
| `core/mcp_client.py` | 724 | 724 |
| **Total** | **~13,307** | **~~15,244** |

**2026-05-13 (v8) — Architectural Refactoring & Research Intelligence**

This version addresses fundamental limitations identified through a comprehensive capability assessment. Key changes fall into two categories: code architecture refactoring and research intelligence improvements.

### Code Architecture Refactoring
- **`core/tools.py` split** (4714 → 1667 lines): Extracted ~3000 lines into two new mixin modules:
  - `core/mcp_client.py` (724 lines): `MCPClientMixin` — all MCP transport logic (SSE + stdio), service detection, vision tools, image analysis
  - `core/model_analyzer.py` (2282 lines): `ModelAnalyzerMixin` — all 9 layers of AST analysis, data flow tracing, bottleneck detection, gradient path analysis, structural soundness scoring, domain assumption detection, idea-architecture alignment, decoder adequacy, runtime model probe, diagnostic script generation, ablation experiment design
- **`core/loop.py` split** (2550 → 2268 lines): Extracted into:
  - `core/domain_knowledge.py` (559 lines): `DomainKnowledgeMixin` — domain knowledge injection and cross-experiment meta-pattern analysis
- **`ToolRegistry` now inherits `MCPClientMixin` + `ModelAnalyzerMixin`**; `ResearchLoop` now inherits `DomainKnowledgeMixin`
- **`visual_analyzer.py` duplicate MCP code removed**: `_call_zai_direct_spawn` (132 lines) deleted — always uses `ToolRegistry`'s MCP session instead of spawning a separate subprocess
- **`memory.py` domain keywords configurable**: `MemoryManager.__init__` accepts `method_keywords` and `domain_keys` parameters for cross-domain reuse
- **`gpu/keeper.py` labeled**: Docstring clearly marks as standalone utility not used by core research loop
- **Bug fixes**: `_find_last_assistant_text` handles tool_calls-only messages, `_tail_file` returns `list` not `deque`, `_get_gpu_status` always returns `"gpus"` key

### Research Intelligence Improvements
- **Dynamic domain knowledge** (replaces hardcoded logic): `domain_knowledge.py` no longer contains project-specific hardcoded domain mappings. Instead:
  - `METHOD_PROPERTIES` database: 7 generic method entries (EPI, FFT, attention, ResNet, sigmoid, Conv3D, contrastive) with scientific knowledge only (assumptions, failure symptoms, alternatives)
  - `_infer_domain_compatibility()`: Dynamically infers method-domain compatibility from PROJECT_BRIEF text using violation condition matching
  - `_extract_data_constraints()`: Detects data scarcity (< 10 training samples) and data imbalance from brief text
  - `_detect_implemented_methods()`: Scans `models/` directory to detect which methods are actually implemented in code
  - `_extract_domain_names()`: Generic domain name extraction (Lambertian, Non-Lambertian, Mixed, outdoor, indoor, etc.)
- **Idea Guardian** (leader.md + loop.py): Every 5 cycles, forces the Leader to re-read PROJECT_BRIEF phase goals, rate progress on core idea implementation (0-10), phase completion (0-10), and data-first verification (0-10). Any score < 5 triggers a mandatory course correction instead of another training run
- **Direction Circuit Breaker** (loop.py): When `_direction_stagnation_count` exceeds threshold, injects a `direction_circuit_breaker` context forcing the Leader to re-evaluate the research direction and record the current direction as a dead end
- **Data Analysis Experiment support** (code_agent.md): New section explicitly supports non-training experiments — data feasibility verification, Phase 1 analysis, assumption validation. Uses `run_shell` not `launch_experiment`
- **Data Scarcity Awareness** (leader.md + loop.py): When `data_constraints` detects < 10 training samples for a domain, the Leader is instructed NOT to propose architecture changes (hard wall) and to focus on data augmentation, transfer learning, or accepting the limitation

### Module Size After Refactoring

| Module | Before | After |
|--------|--------|-------|
| `core/tools.py` | 4,714 | 1,667 |
| `core/loop.py` | 2,550 | 2,268 |
| `core/visual_analyzer.py` | 985 | 853 |
| `core/mcp_client.py` | (new) | 724 |
| `core/model_analyzer.py` | (new) | 2,282 |
| `core/domain_knowledge.py` | (new) | 559 |
| **Total** | **~13,000** | **~13,307** |

**2026-05-12 (v7) — Idea-Architecture Alignment Analysis**
- **Layer 9: Idea-Architecture Alignment** (new in `analyze_model`): Compares model architecture against `PROJECT_BRIEF.md` to verify the core research idea is faithfully implemented. This catches a critical class of errors where the agent builds a model that loosely follows the idea but structurally under-represents key innovations.
  - **Idea Component Detection**: Parses PROJECT_BRIEF for 10 key patterns (angular frequency, dual mask, EPI, BRDF, component-aware depth, non-Lambertian handling, attention, adaptive fusion, etc.) and maps them to model branches.
  - **Channel Allocation Analysis**: Computes per-branch channel ratios and flags when a KEY INNOVATION branch gets < 15% of fusion channels (the "FFT is only 9%" problem). Provides specific improvement suggestions (e.g., "INCREASE fft_branch from 32 to ~72 channels").
  - **Structural Gap Detection**: Checks for missing architectural patterns implied by the idea (skip connections, multi-scale processing, attention for multi-branch fusion). Only flags patterns relevant to the specific idea.
  - **Decoder Adequacy Analysis**: Evaluates decoder depth, skip connections, and domain-awareness. Warns when a shallow decoder must handle large fusion inputs or when the idea requires domain prediction but the model lacks it.
  - **Alignment Score (0-10)**: Deducts points for missing idea components (-3 for high-importance), under-represented branches (-2), missing structural patterns (-1). Provides an overall assessment with specific action guidance.
- **Enhanced Branch Detection**: `_extract_branch_info` now traces into custom class definitions (e.g., `AngularFFTBranch`) to find output channel counts, rather than only supporting direct `nn.Conv2d`/`nn.Sequential` calls. Also matches `center_*` pattern.
- **Leader Prompt v7**: Added "Idea-Architecture Alignment" section requiring `analyze_model` after model creation/modification. Provides action guidance based on alignment score (< 5: major revision, 5-7: address top findings, ≥ 8: focus on training).
- **Code Agent Prompt v7**: Updated Step 1.5 with idea-alignment guidance — must check `idea_architecture_alignment` section before coding, verify key innovation branches have sufficient channel allocation.

**2026-05-12 (v6) — PhD-Level Research Capabilities**
- **Training Curve Analysis**: VERIFY now performs rich training curve diagnostics beyond simple "loss decreasing?": detects overfitting (loss rises after minimum), oscillation (direction change ratio), convergence speed (too slow = under-capacity), and plateau detection (flat loss for extended periods). Results injected into REFLECT for the Leader.
- **Pareto Frontier Tracking**: New `pareto_matrix` SQLite table tracks method×domain MAE across all experiments. Automatically identifies Pareto-optimal methods (best for at least one domain) and dominated methods (never best). Injected into THINK so the Leader avoids repeating suboptimal methods.
- **`generate_diagnostic` Tool** (new): Generates targeted diagnostic scripts based on natural language questions. Unlike `probe_model` (random data), this creates scripts that answer specific questions like "Is the FFT branch dead for Non-Lambertian inputs?" or "What do EPI slopes look like on specular surfaces?" Four diagnostic types: domain_analysis, branch_analysis, gradient_analysis, attention_analysis.
- **`design_ablation` Tool** (new): AST-based systematic ablation experiment design. Identifies all model components (branches, backbone, heads, fusion, normalization), groups by functional category, and generates prioritized ablation experiments (component removal, freeze, single-branch, fusion replacement) with estimated information value. Available to Code agent.
- **Experiment Value of Information (VOI)**: Before running experiments, the system estimates the value of information (expected_improvement × success_probability). Low-VOI experiments get a warning. Tracks calibration (how often hypotheses are actually correct) to improve future estimates.
- **Structured Meta-Pattern Learning**: `_build_cross_experiment_insights` now uses structured SQLite queries (`get_method_domain_effect_matrix`, `get_stuck_domains_structured`) instead of fragile regex on markdown text. Falls back to regex only if database is empty. Detects dominant methods, hypothesis accuracy trends, and calibration feedback.
- **Adaptive Experiment Granularity (Pilot Experiments)**: High-risk hypotheses (prior probability < 0.4) automatically get a pilot experiment recommendation (2-3 epochs). Low-value experiments get warnings. Injected into task descriptions.
- **Causal Chain Tracking**: New `causal_chain` SQLite table records design_decision → architectural_property → metric_affected links. THINK records expected effects, REFLECT updates with actual effects. History injected into THINK so the Leader avoids repeating failed causal chains.
- **New SQLite Tables**: `pareto_matrix`, `causal_chain`, `experiment_value` — all with indexes for efficient queries.
- **Leader Prompt v6**: Added sections for Pareto frontier awareness, experiment value estimation, pilot experiments, causal chain tracking, and training curve analysis interpretation.

**2026-05-11 (v5) — Deep Model Analysis & Runtime Diagnostics**
- **`analyze_model` Deep Architecture Analysis**: Upgraded from surface-level AST analysis to 8-layer deep analysis: (1) parameter counts, (2) data flow graph, (3) information bottleneck detection (compression ratio > 8:1), (4) gradient path analysis, (5) structural soundness scoring (0-10), (6) domain assumption detection (EPI→Lambertian, FFT→frequency stability, etc.), (7) data feasibility + GPU memory, (8) result-to-architecture diagnosis. Available to Leader, Code, and Researcher agents.
- **`probe_model` Runtime Diagnostics** (new tool): Instantiates the model, runs forward+backward with dummy data, captures REAL tensor statistics — activation mean/std/dead_ratio per module, gradient norms per module, gradient balance analysis (detects > 100x imbalance), input sensitivity testing (detects output collapse), and parameter distribution. Can optionally load trained checkpoint for post-training diagnosis. This closes the gap between static analysis and real model behavior.
- **Result→Architecture Feedback Loop**: When domain gap > 0.10 (e.g., Non-Lambertian MAE 2x worse than Lambertian), REFLECT now injects a mandatory reasoning chain: identify the worst domain → state the architectural assumption → verify if the assumption holds → diagnose root cause → propose fix. Explicitly forbids incremental tuning for structural problems.
- **Hypothesis Pre-Validation (THINK Step 6)**: Before any architectural change, the Leader must state the core assumption, check if it holds in the data, pre-validate via `analyze_model`, and define the minimum experiment.
- **Code Agent Structural Analysis (Step 1.5 & 1.6)**: Code agent must run `analyze_model` before modifying architectures (check structural score, bottlenecks, dead branches). After failed training, must run `probe_model` to get runtime evidence (dead activations, gradient-dead branches, output collapse).
- **VERIFY Layer 9 — Model Structural Soundness**: New verification layer detects dead modules (declared in `__init__` but unused in `forward()`) and multi-branch fusion warnings.
- **Bug Fixes**: `import ast` missing in verifier.py (Layer 9 was silent NameError), `_check_model_fusion_balance` was no-op (now produces warnings), WORKER_CONFIGS tool lists synced with actual `get_tools_for()`, NaN values filtered in domain gap calculation, duplicate Step 3.5 in leader.md renamed to Step 3.6.

**2026-05-11 (v4) — Agent Capability Upgrade**
- **Output Quality Awareness**: Agent now tracks per-domain metrics and detects when experiments produce "successful bad results" (e.g., Non-Lambertian MAE degrades from 0.30 to 0.41). Two consecutive quality degradations force paper research with hypothesis validation.
- **Forced Visual Analysis**: When any domain MAE exceeds 0.35, visual analysis is triggered immediately regardless of streak count. The agent MUST visually inspect its own predictions.
- **Strategic Abandonment**: Research direction stagnation detection — same direction for 3+ cycles without improvement forces paper research for fundamentally different approaches.
- **Infrastructure Degradation**: After 3 consecutive infrastructure failures (API timeout, process crash), VERIFY no longer blocks experiments. Infrastructure issues are logged but don't halt progress.
- **Audit Death Loop Protection**: After 5 consecutive audit directives, the agent auto-clears the directive and continues with real work. Infrastructure issues are logged as known limitations.
- **Cross-Domain Analysis Prompt**: REFLECT phase now forces the Leader to analyze WHY different domains perform differently, identify violated method assumptions, and estimate the current approach's ceiling.
- **Hypothesis Validation Prompt**: When quality degrades repeatedly, the agent must state the core assumption of its method, identify which domain violates it, and propose a method that doesn't rely on the violated assumption.
- **Configuration Consistency Verification**: New VERIFY layer checks for checkpoint mismatches (size mismatch / missing keys) and hardcoded parameters that may not match actual data.
- **Direction Signature Extraction**: Research direction is extracted from task descriptions using methodology keywords (edge/loss/pretrain/epi/angular/etc.) with hyphen/underscore normalization.
- **Dataset Quality Verification** (new Layer 8): Agent now detects when validation splits are statistically unreliable — flags domains with < 3 validation scenes, warns when metrics are based on single scenes (fluctuations = noise, not signal), and records dataset issues for REFLECT to address. Prevents the agent from making decisions based on noisy metrics.
- **Model Architect Skill** (`skills/model-architect/SKILL.md` + `analyze_model` tool): Agent can now analyze model architecture before training — extracts parameter counts, branch channel ratios, GPU memory estimates, data-to-parameter ratios, and physical reasonableness checks. Detects design flaws like branches with < 10% fusion channels (will be gradient-drowned) and severe overfitting risk (< 10 samples per K params).

**2026-05-08 (v3)**
- **MCP SSE Dual-Connection Protocol**: Fixed all GLM platform MCP services (web_search_prime, web_reader, zread) — switched from broken `POST /mcp` to correct `GET /sse` + `POST /message` dual-connection SSE transport. Background reader thread collects JSON-RPC responses from the persistent SSE stream.
- **MCP stdio Transport for Vision**: `zai-mcp-server` runs as a **local npx subprocess** (stdio transport), not a remote SSE service. Requires Node.js 18+ and npx. Communicates via stdin/stdout JSON-RPC. If Node.js is absent, the vision degradation chain skips MCP and falls through to direct API calls.
- **4 MCP Services Verified Working**: web_search_prime ✅, web_reader ✅, zread ✅, zai-mcp-server (stdio) ✅
- **Vision Degradation Chain** (updated):
  ```
  MCP zai-mcp-server (stdio, requires Node.js)
    → Direct API: glm-5v-turbo → glm-4.6v (GLM Coding Plan)
    → Direct API: qwen3.6-plus (Ali Token Plan)
    → Direct API: qwen3.6-plus → qwen3.5-plus (Ali DashScope)
  ```
- **Parameter Compatibility**: Auto-remaps `image_path` → `image_source` and `image_analysis` → `analyze_image` for zai-mcp-server compatibility.
- **Node.js Requirement**: Vision MCP requires `npx` (Node.js 18+). Without it, vision analysis still works via direct API fallback but without specialized MCP tools.

**2026-05-07 (v2)**
- **Vision MCP Full Integration**: `@z_ai/mcp-server` 8 vision tools integrated into `core/tools.py`. Agent-facing tools: `analyze_image` (researcher) and `diagnose_error` (code agent).
- **Three-Provider Vision Fallback**: Strict separation of 3 API providers (GLM Coding Plan / Ali Token Plan / Ali DashScope), each with its own key and endpoint.
- **REFLECT Phase Deep Analysis**: Leader gets `read_file` + `list_files` tools during REFLECT (20 max turns, 16384 max tokens) for cross-validation — reading model code, data manifests, and training logs alongside visual analysis findings.
- **Lazy MCP Detection**: MCP service availability detected lazily on first access (property-based) instead of blocking `__init__`, preventing startup delays.
- **Auto Code-Cleanup v2**: New conditions for scripts/ naming pollution (> 10 files) and archive bloat (> 20 entries). Cleanup now targets .pt checkpoints as #1 disk priority, deletes final_checkpoint.pt, dry-run outputs, and archive dirs without SUMMARY.md.
- **Script Naming Convention**: Code agent now enforces `train_{model_short_name}.py` naming — no incremental suffixes (`_v2`, `_fix`). One-time diagnostics use `_` prefix and must be deleted after use.
- **Bug Fix**: API fallback chain includes ALI_TOKEN_PLAN_API_KEY provider. `AngularAwareDepthModelV12` now exported from `models/__init__.py`.

**2026-05-07 (v1)**
- **Visual Analysis Module**: New `core/visual_analyzer.py` — when training results are consistently poor (>= 5 consecutive cycles), automatically runs inference, generates prediction images, and sends them to a multimodal LLM for visual failure diagnosis. Catches problems invisible to numeric metrics alone (e.g., uniform depth maps, domain collapse, structural failures).
- **MCP Multimodal Integration**: Supports `@z_ai/mcp-server` for vision-capable analysis when primary LLM (GLM-5.1) lacks multimodal input. Fallback to GLM-5V-Turbo → GLM-4.6V direct API calls if MCP unavailable.
- **Visual diagnosis injected into REFLECT phase**: Leader agent now receives structured visual findings (diagnosis categories, severity, recommended actions) alongside VERIFY report, enabling data-informed experiment planning.

**2026-04-28**
- **Code Agent turn budget tightened**: `max_turns` reduced from 40→25 to prevent endless exploration. Turn budget reminder injected into tool results — agent gets explicit "CRITICAL: Almost out of turns" warnings at 80% budget.
- **Consecutive `list_files` rate-limit**: Max 3 consecutive `list_files` calls before forced stop. Prevents the agent from repeatedly browsing directories instead of doing actual work.
- **`run_shell` security regex fix**: Device redirect pattern now correctly allows `/dev/null`, `/dev/zero`, `/dev/tty`, `/dev/fd/`, `/dev/stdin|stdout|stderr` while blocking unsafe device writes.
- **`write_file` datasets registration allowlist**: Agents can now write to `datasets/__init__.py` and `datasets/unified_lf_dataset.py` (previously blocked). Other dataset files remain protected.
- **Synthetic data detection in training scripts**: When writing to `scripts/*.py`, the tool scans for random noise patterns (`np.random.rand`, `torch.rand`, `SyntheticLF`, `RandomDataset`) and returns a warning. VERIFY will block experiments using synthetic data.
- **AUDIT escalation Level 4 (unfixable)**: New `mark_unfixable` level (count ≥ threshold×4) records the issue as a dead_end and stops escalating — preventing infinite retry on truly unsolvable problems.
- **Code-Cleanup trigger tightened**: No-progress streak threshold raised from 2→4 cycles before triggering code-cleanup. Reduces false positives.
- **Memory log budget doubled**: `log_max` increased from 2,000→4,000 chars. Old entries are now **compressed** (summarized) instead of deleted — preserving knowledge while fitting the budget. New `get_log_summary()` method for LLM context.
- **Session statistics injection**: Leader context now includes SQLite session stats (total cycles, experiments launched, launch rate, dead ends count) and recent failure patterns — enabling better decision-making.
- **Runtime data fingerprint verification**: VERIFY phase now runs a quick Python check to load one dataset sample and verify it's not random noise (checks spatial correlation and constant values). The most reliable defense against LLMs secretly swapping in synthetic data.

**2026-04-24**
- **Anti-Deception Architecture**: `ToolTrace` and `ToolCallRecord` now record every tool call's actual system-returned result. Key facts (PIDs, log files, exit codes) are extracted from tool execution results — never from LLM narrative text. VERIFY cross-checks tool trace against LLM claims to detect fabrication.
- **Multi-provider Token Plan support**: Added `glm_token_plan` (Zhipu GLM) alongside `ali_token_plan`. Automatic failover between providers with health tracking and cooldown.
- **Tiered model strategy**: `STRONG_MODEL_TASKS` (think/reflect/idea/researcher) use strong model, routine tasks (code/writing) use fast model. Configurable via `"model": "auto"`.
- **Researcher Agent**: New specialized agent for deep literature search, equipped with `web_search`, `web_fetch`, `search_papers`, `get_paper` tools.
- **`get_paper` tool**: Fetch paper details by Semantic Scholar ID or arXiv ID (e.g., `"arXiv:2401.12345"`).
- **Auto code-cleanup trigger**: Automatically dispatches code-cleanup when root `.py` files exceed 15, logs pile up, or experiment fails.
- **Dataset understanding**: First-cycle mandatory scan validates `data/` directory and produces `DATASET_MANIFEST.json`.
- **No-progress paper research fallback**: After repeated cycles with no progress on the same plan, agent automatically redirects to paper research to seek breakthrough methods.
- **Memory compaction fix**: MEMORY_LOG now trims `active_problems` and `dead_ends` sections when over budget (previously only trimmed milestones and decisions).
- **`★` major event prefix**: `log_major_event()` uses ★ prefix for paper research breakthroughs; `_parse_log()` and Obsidian exporter now correctly handle this prefix.
- **Bug fixes**: `urllib.parse` import in `_exec_web_search`/`_exec_get_paper`, `_consume_directive` rename collision (uuid suffix), `_dataset_manifest_exists` validates JSON content.

**2026-04-23**
- **BREAKING**: Upgraded to **THINK → EXECUTE → VERIFY → REFLECT** four-phase pipeline. The new VERIFY phase reverse-engineers whether each module (dataset, model, training, evaluation) actually worked — before REFLECT draws conclusions.
- Added **Experiment Verifier** (`core/verifier.py`) — zero-LLM-cost module-level verification with structured diagnosis. Checks: dataset validity, model checkpoint sanity, training loss dynamics, metric consistency.
- Added **Reasoning Principles** system (`skills/REASONING_PRINCIPLES.md`) — 6 mandatory guidelines (Think Before Acting, Simplicity First, Surgical Changes, Goal-Driven Execution, Honesty, Verify-First) injected into every THINK/REFLECT dispatch to reduce common LLM reasoning mistakes.
- Added **3-level Audit Escalation** system — repeated VERIFY failures are auto-detected and escalated: L1 targeted fix → L2 force error-handler skill → L3 pause for human intervention.
- Added **error-handler skill** (`skills/error-handler/SKILL.md`) — 7-step diagnostic workflow for systematic root cause analysis.
- Added **Token Plan provider** support (ali_token_plan) — cost-optimized subscription for code agents with models like qwen3.6-plus, deepseek-v3.2, glm-5.
- Fixed `run_shell`/`launch_experiment` to use `shell=True` with regex-based safety validation — shell operators (`cd`, `&&`, `|`, `> /dev/null`) now work correctly.

**2026-04-09**
- Reduced token growth by resetting leader context between cycles.
- Added a lightweight fallback to avoid repeated no-progress loops.
- Hardened tool execution against path traversal and shell injection.

**2026-04-08**
- Added progress tracking exports for experiment monitoring.
- Supports optional Obsidian sync for a live dashboard plus daily notes.
- If no Obsidian vault is configured, progress falls back to project-local text files under `workspace/progress_tracking/`.

## Start In 3 Steps

If you only want the shortest path to a working experiment loop, do this:

1. Create a project folder with one file: `PROJECT_BRIEF.md`
2. Run `/auto-experiment --project /path/to/project --gpu 0`
3. Check progress with `/experiment-status` or optional Obsidian/local text notes

Prefer AI-guided setup? Open `AI_GUIDE.md` in Claude / ChatGPT / Codex and let the assistant walk you through it.

## What You Actually Need

| Requirement | Required | Notes |
|-------------|----------|-------|
| Python 3.10+ | Yes | Runtime |
| 1+ NVIDIA GPU | Yes | For training |
| API key | Yes | Anthropic or OpenAI |
| `PROJECT_BRIEF.md` | Yes | Main control file |
| Project `config.yaml` | Optional | Only if you want to override defaults |
| Obsidian vault | Optional | If absent, notes fall back to local text files |

## Minimum Working Example

The smallest project you can launch looks like this:

```text
my-first-experiment/
├── PROJECT_BRIEF.md
└── workspace/                  # auto-created
```

Minimal `PROJECT_BRIEF.md`:

```md
# Goal
Train a ResNet-50 on CIFAR-100 to reach 80%+ accuracy.

# Codebase
Create the training code from scratch in PyTorch.

# What to Try
- Start with a basic ResNet-50 baseline.
- If accuracy < 75%, improve optimization and schedule.
- If accuracy is 75-80%, try augmentation.
- If accuracy > 80%, stop and report.

# Constraints
- Use GPU 0 only
- Max 100 epochs per run
```

That is enough to start. Everything else is optional refinement.

## What This Project Is Good At

This project is for people who already know what experiment they want to run, but do not want to babysit the loop:

- edit code
- launch training
- monitor runs
- parse logs
- decide the next variation
- keep going while you sleep

It is not trying to replace the researcher. It is trying to take over the repetitive experiment-ops layer.

## Why It Feels Different From A Simple Script

- It does not just launch one run. It keeps iterating.
- It does not just monitor. It reflects and decides the next step.
- It stays cheap because training-time monitoring makes zero LLM calls.
- It stays controllable because the human can override direction at any cycle.
- It now supports persistent progress notes in Obsidian or local text files.

## How You Stay In Control

You control the research direction through three files:

- `PROJECT_BRIEF.md`: stable goal, constraints, allowed search space
- `HUMAN_DIRECTIVE.md`: temporary redirect for the next cycle
- `workspace/MEMORY_LOG.md`: rolling memory of results and decisions

Common control patterns:

```md
# Keep the search narrow
- Only tune augmentation.
- Do not change the backbone.
- Keep training budget fixed.
```

```md
# Make the agent stop exploring a weak direction
- If gain stays below 0.3 points for 3 runs, stop this branch.
- Return to the last trusted baseline and try a different idea.
```

```md
# Force result verification
- If a result looks unusually strong, rerun with the same seed and one new seed.
- Do not claim improvement until both reproduce.
```

## How You See Progress

You should never have to guess what the agent is doing.

- `/experiment-status` shows current goal, best result, cycle count, running status, and recent decisions
- `/progress-report` generates a structured summary
- `/obsidian-sync` refreshes persistent notes manually
- `workspace/progress_tracking/` stores local text notes when no Obsidian vault is configured

If you want a dashboard outside the terminal:

```yaml
obsidian:
  enabled: true
  vault_path: "~/Documents/MyObsidianVault"   # Optional
  auto_append_daily: true
```

If `vault_path` is empty, the same information is saved locally:

```text
workspace/progress_tracking/Dashboard.txt
workspace/progress_tracking/Daily/YYYY-MM-DD.txt
```

---

## 💛 A Note on Why We Built This — and How We Hope You'll Use It

> **Our hope is simple: science stays pure, and the human stays in the loop.**

We built this framework for one reason — to take the *repetitive, mechanical* parts of running deep learning experiments off the researcher's plate (launching jobs, watching GPUs, parsing logs, sweeping hyperparameters) so that more of your time can go into **the part that actually matters: thinking**.

If you're here because you want to spend less time babysitting training runs and more time reading, reasoning, and chasing your own ideas — welcome. That's exactly who we built this for.

**A gentle thought we'd love every user to share with us:**

The agent is happy to run the experiments. But please let the *ideas*, the *interpretation*, and the *scientific judgment* remain yours. We don't see automation and academic integrity as being in tension — quite the opposite. The hours this tool gives back are meant to be reinvested in **deeper thinking**, not in skipping it.

So we'd kindly ask that this project not be used to fabricate results, to generate "research" with no human in the loop, or to shortcut the parts of science that depend on a human actually understanding what they're doing. That isn't the future we want to help build — and we don't think it's the one most of you want either.

> **Science should stay pure. The agent can run the experiments — but the ideas, the interpretation, and the responsibility belong to the human.**
>
> **学术应当保持纯粹。** Agent 可以替你跑实验，但 idea、判断与责任，请留给人来承担。我们真心希望每一位使用者都能 **human in the loop 地去思考**，把这个工具省下来的时间，投入到真正属于你自己的研究方向里。


We trust the people who pick up this tool to take that seriously — and we built it because we believe most of you already do. Thank you for being one of them. 💛

---

## The Core Idea

You design the experiment. The agent handles the repetitive loop.

**Deep Researcher Agent**:

1. **Thinks** — Reads your project brief, analyzes previous results, plans the next experiment
2. **Executes** — Modifies code/configs, runs a dry-run, launches training on GPU
3. **Verifies** — Reverse-engineers whether each module actually worked (dataset loaded? loss valid? training progressed?)
4. **Visual Analyzes** *(when stuck)* — Runs inference, sends prediction images to multimodal LLM, diagnoses WHY model fails (catches problems numeric metrics miss)
5. **Reflects** — Parses results, compares with baselines, decides what to try next (informed by VERIFY + visual diagnosis)
6. **Repeats** — 24/7, without human intervention

```
You sleep 8 hours     → Agent runs 3 experiment cycles
You go on vacation    → Agent explores 50+ hyperparameter configs  
You write your paper  → Agent already has the results table ready
```

---

## Battle-Tested Results

> Not benchmarks. Real results from months of 24/7 autonomous operation across research projects.

| Metric | Result |
|--------|--------|
| Autonomous experiment cycles completed | 500+ |
| Best single-project improvement | 52% over baseline (across 200+ auto-run experiments) |
| Concurrent projects managed | 4 projects across 4 GPU servers |
| Longest continuous autonomous operation | 30+ days without human intervention |
| Average LLM cost per 24h cycle | ~$0.08 |

---

## Key Innovation: Zero-Cost Monitoring

The #1 concern with running LLM agents 24/7: **cost**.

Most agent frameworks call the LLM every few minutes to "check progress". That's $50+/day.

Experiment Agent **sleeps** during training — zero API calls. It only wakes the LLM when training finishes.

```
                    LLM Active              Zero Cost              Zero Cost           LLM Active
                  ┌────────────┐    ┌─────────────────────┐    ┌──────────────┐    ┌────────────┐
                  │   THINK    │    │   TRAIN & MONITOR    │    │   VERIFY     │    │  REFLECT   │
                  │ (5-10 min) │    │   (hours/days)       │    │  (1-2 min)   │    │ (5-10 min) │
                  │            │    │                      │    │              │    │            │
                  │ • Analyze  │    │ • kill -0 $PID       │    │ • File check │    │ • Parse    │
                  │ • Plan     │    │ • nvidia-smi         │    │ • Loss valid │    │   logs     │
                  │ • Code     │    │ • tail log           │    │ • Ckpt sane  │    │ • Compare  │
                  │            │    │                      │    │              │    │ • Decide   │
                  │  ~$0.05    │    │      $0.00           │    │   $0.00      │    │  ~$0.03    │
                  └────────────┘    └─────────────────────┘    └──────────────┘    └────────────┘
```

**24-hour cycle with 8 hours of training: ~$0.08 in LLM calls.**

---

## Architecture

### The THINK → EXECUTE → VERIFY → REFLECT Loop

```
┌──────────────────────────────────────────────────────────────────────────┐
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐ │
│  │  THINK   │→│ EXECUTE  │→│  VERIFY  │→│VISUAL    │→│ REFLECT  │──┐    │
│  │          │  │          │  │          │  │ANALYZE*  │  │          │  │    │
│  │ Analyze  │  │ Dry-run  │  │ Module   │  │          │  │ Evaluate │  │    │
│  │ Plan     │  │ Launch   │  │ checks   │  │ Inference │  │ Compare  │  │    │
│  │ Decide   │  │ Monitor  │  │ Diagnose │  │ Vision    │  │ Decide   │  │    │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  └──────────┘  │    │
│       ↑                                                    │            │    │
│       └────────────────────────────────────────────────────┘            │    │
│                    ↻ 24/7 Loop                                        │
└──────────────────────────────────────────────────────────────────────────┘

* VISUAL ANALYZE: Only triggers when training is consistently poor (>=5 cycles).
  Runs inference → sends prediction images to multimodal LLM → diagnoses WHY model fails.
```

### Leader-Worker Agent System

Only ONE worker runs at a time. Others idle at zero cost.

```
              ┌───────────────┐
              │    Leader     │  Persistent conversation
              │   (Planner)   │  within each cycle
              └──┬──┬──┬──┬──┘
                 │  │  │  │
         ┌───────┘  │  │  └───────┐
         ↓          ↓  ↓          ↓
   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────┐
   │   Idea   │ │   Code   │ │ Writing  │ │ Researcher │
   │  Agent   │ │  Agent   │ │  Agent   │ │   Agent    │
   │ (4 tools)│ │ (5 tools)│ │ (3 tools)│ │  (4 tools) │
   └──────────┘ └──────────┘ └──────────┘ └────────────┘
```

### Two-Tier Memory (Constant Size Forever)

```
┌─────────────────────────────────────────┐
│ Tier 1: PROJECT_BRIEF.md               │
│ • Frozen project reference              │
│ • Max 3,000 chars                       │
├─────────────────────────────────────────┤
│ Tier 2: MEMORY_LOG.md                   │
│ • Key Results (auto-compact at 1,200ch) │
│ • Recent Decisions (rolling last 15)    │
│ • Max 4,000 chars                       │
├─────────────────────────────────────────┤
│ Total: ~7K chars / ~2,000 tokens        │
│ SAME whether running 1 day or 6 months  │
└─────────────────────────────────────────┘
```

### Cost Control Strategies (8 Total)

| # | Strategy | Savings |
|---|----------|---------|
| 1 | Zero-LLM monitoring during training | 90%+ of runtime is free |
| 2 | Two-Tier memory with auto-compaction | Fixed context window |
| 3 | Leader conversation persists within cycle | Brief sent once per cycle |
| 4 | Anthropic prompt caching | System/tools cached |
| 5 | Per-agent minimal tool sets (3-5 tools) | Less schema overhead |
| 6 | Slim system prompts | Fewer input tokens |
| 7 | State trimmed before sending | No bloat |
| 8 | Single worker at a time | No parallel LLM costs |

### Reasoning Principles System

Every THINK and REFLECT dispatch includes a **mandatory reasoning checklist**:

1. **Assumptions** — What am I assuming? Write them out.
2. **Alternatives** — Is there a simpler way? Am I changing too many variables?
3. **Success criteria** — Concrete, measurable — not "improve" but "MAE < 0.35".
4. **Surgical** — Every code change must trace to this experiment's hypothesis.
5. **Honesty** — If results don't meet criteria, say so — don't spin.
6. **Verify-first** — If VERIFY found module failures, address those BEFORE judging the experiment.

These are defined in `skills/REASONING_PRINCIPLES.md` and injected into the Leader agent's context at every dispatch via `_REASONING_REMINDER` in `agents.py`. The Code agent also has these principles in its system prompt.

### 4-Level Audit Escalation

When the experiment auditor detects the same issue repeatedly:

| Level | Trigger | Action |
|-------|---------|--------|
| L1 | Issue appears ≥ 3 cycles | Inject targeted fix directive into next THINK |
| L2 | Issue appears ≥ 6 cycles | Force error-handler skill execution |
| L3 | Issue appears ≥ 9 cycles | Pause agent, write `AGENT_STUCK.md` for human |
| L4 | Issue appears ≥ 12 cycles | Mark as unfixable dead_end, stop escalating |

This prevents the agent from looping indefinitely on trivial errors (e.g., dataset key mismatches).

---

<a name="quickstart"></a>
## Getting Started (Step by Step)

> **Complete beginner?** Follow every step below. You'll go from zero to a running experiment agent in ~10 minutes.
>
> **Prefer AI-guided setup?** Open [`AI_GUIDE.md`](AI_GUIDE.md) in Claude Code, ChatGPT, or Codex — the AI will walk you through everything interactively.

### Step 0: What You Need

| Requirement | Why | How to Check |
|-------------|-----|-------------|
| Python 3.10+ | Runtime | `python3 --version` |
| [Claude Code](https://claude.ai/claude-code) | The AI backbone | `claude --version` |
| 1+ NVIDIA GPU | For training | `nvidia-smi` |
| Anthropic API key | LLM calls | `echo $ANTHROPIC_API_KEY` |

Don't have an API key? Get one at [console.anthropic.com](https://console.anthropic.com/) and set it:
```bash
export ANTHROPIC_API_KEY="sk-ant-xxxxx"
# Add to ~/.bashrc or ~/.zshrc to make it permanent
```

### Step 1: Install

```bash
# Clone the repo
git clone https://github.com/hrf666666/auto_research_agent.git
cd auto_research_agent

# Install Python dependencies
pip install -r requirements.txt

# Install 8 slash commands into Claude Code
python install.py

# Verify everything works
python -m core.loop --check
```

You should see:
```
  Deep Researcher Agent — Installer
  ========================================

    ✓ /auto-experiment
    ✓ /experiment-status
    ✓ /gpu-monitor
    ✓ /daily-papers
    ✓ /paper-analyze
    ✓ /conf-search
    ✓ /progress-report

    ✓ /obsidian-sync

  Done! 8 skills installed.
```

### Step 2: Create Your First Project

Let's say you want to train a ResNet on CIFAR-100. Create a project folder with a `PROJECT_BRIEF.md`:

```bash
mkdir ~/my-first-experiment
cd ~/my-first-experiment
```

Now write the brief — **this is the most important file**. It tells the agent what you want:

```bash
cat > PROJECT_BRIEF.md << 'EOF'
# Goal
Train a ResNet-50 on CIFAR-100 to reach 80%+ test accuracy.

# Codebase
The agent should create the training code from scratch using PyTorch.
- Use torchvision for the dataset (auto-download)
- Save checkpoints to ./checkpoints/
- Log metrics to ./logs/

# What to Try
- Start with a basic ResNet-50, lr=0.1, SGD, 100 epochs
- If accuracy < 75%, try cosine annealing + warmup
- If accuracy 75-80%, try adding mixup or cutout augmentation
- If accuracy > 80%, the goal is reached

# Constraints
- Use GPU 0 only
- Max 100 epochs per run
- Batch size 128

# Current Status
No experiments run yet. Starting from scratch.
EOF
```

**Tips for writing a good brief:**
- Be specific about the goal (metric + target value)
- Tell it where the code/data is (or say "create from scratch")
- List constraints (which GPU, max epochs, etc.)
- Give it a decision tree ("if X, try Y") — this guides the agent like you would guide a junior student

### Step 3: Launch the Agent

**Option A: Through Claude Code (recommended)**

Open Claude Code and type:
```
/auto-experiment --project ~/my-first-experiment --gpu 0
```

**Option B: Through Python directly**

```bash
python -m core.loop \
  --project ~/my-first-experiment \
  --gpu 0 \
  --max-cycles 5    # Stop after 5 cycles (remove for unlimited)
```

### Step 4: Watch What Happens

The agent will now do everything automatically. Here's what each cycle looks like:

```
=== Cycle 1 ===

[THINK] Reading PROJECT_BRIEF.md...
        Goal: ResNet-50 on CIFAR-100, target 80%+
        No previous experiments. Starting with baseline.
        Plan: Basic ResNet-50, lr=0.1, SGD with momentum, 100 epochs.

[EXECUTE] Creating train.py...
          Creating config.yaml...
          Running dry-run (2 steps)... ✓ No errors
          Launching training: nohup python train.py --config config.yaml
          PID: 12345, Log: logs/exp001.log

[MONITOR] Training in progress... (zero LLM cost)
          15:00 — PID alive, GPU 98%, Epoch 12/100, loss=2.34
          15:15 — PID alive, GPU 97%, Epoch 25/100, loss=1.87
          15:30 — PID alive, GPU 98%, Epoch 38/100, loss=1.54
          ...
          17:45 — PID alive, GPU 97%, Epoch 100/100, loss=0.82
          18:00 — PID terminated. Training complete.

[VERIFY] Checking module outputs...
         Dataset: ✓ loaded 50K images, shape [3,32,32], no zeros
         Model: ✓ checkpoint 97MB, no NaN in weights
         Training: ✓ loss 2.34→0.82 (decreasing), GPU used
         Evaluation: ✓ test accuracy 76.3% in [0,100] range
         All modules verified. No issues found.

[REFLECT] Parsing logs... test accuracy = 76.3%
          Result: 76.3% — below 80% target
          Brief says: "If < 75%, try cosine annealing"
          76.3% > 75%, so try augmentation instead.
          Decision: Add mixup augmentation, keep lr=0.1 + cosine
          Milestone logged: "Exp001: ResNet-50 baseline, 76.3%"

=== Cycle 2 ===

[THINK] Best so far: 76.3% (Exp001)
        Plan: Add mixup (alpha=0.2) + cosine annealing schedule
        ...
```

### Step 5: Check Progress Anytime

While the agent is running, you can check on it:

```bash
# In Claude Code:
/experiment-status --project ~/my-first-experiment

# Or check GPU usage:
/gpu-monitor
```

You'll see something like:
```
# Experiment Status — my-first-experiment

## Goal
ResNet-50 on CIFAR-100 → 80%+ accuracy

## Progress
- Cycles completed: 3
- Current best: 79.1% (Exp003: ResNet-50 + mixup + cosine)
- Status: TRAINING (PID 12389, GPU 0, running 1.5h)

## Key Results
[04-07 15:00] Exp001: ResNet-50 baseline, 76.3%
[04-07 18:30] Exp002: + cosine annealing, 77.8%
[04-07 22:00] Exp003: + mixup α=0.2, 79.1%   ← best

## Current Training
Epoch 67/100 | loss: 0.71 | acc: 79.4%
```

### Step 5.5: Save Progress to Obsidian or Local Text

Enable progress export in your project `config.yaml`:

```yaml
obsidian:
  enabled: true
  vault_path: "~/Documents/MyObsidianVault"   # Optional
  project_subdir: "DeepResearcher/{project_name}"
  auto_append_daily: true
```

If `vault_path` is set, the agent writes:

```text
DeepResearcher/my-first-experiment/Dashboard.md
DeepResearcher/my-first-experiment/Daily/YYYY-MM-DD.md
```

If `vault_path` is empty, it falls back to project-local files:

```text
workspace/progress_tracking/Dashboard.txt
workspace/progress_tracking/Daily/YYYY-MM-DD.txt
```

Manual refresh:

```bash
/obsidian-sync --project ~/my-first-experiment
# or
python -m core.obsidian --project ~/my-first-experiment
```

### Step 6: Intervene If Needed

Want to change direction? Three ways, from anywhere:

```bash
# Way 1: Drop a directive file (agent reads it next cycle)
echo "Stop trying ResNet. Switch to ViT-B/16, start with lr=1e-3" \
  > ~/my-first-experiment/workspace/HUMAN_DIRECTIVE.md

# Way 2: Command-line flag
python -m core.loop --project ~/my-first-experiment \
  --directive "Try label smoothing 0.1"

# Way 3: Edit memory directly (for permanent changes)
vim ~/my-first-experiment/workspace/MEMORY_LOG.md
```

## Human-in-the-Loop Playbook

Use the agent as an operator, not a replacement researcher.

```text
Human decides:
- goal
- constraints
- forbidden directions
- when to pivot

Agent executes:
- code edits
- runs
- monitoring
- summaries
```

Write stable rules in `PROJECT_BRIEF.md`, and temporary steering in `HUMAN_DIRECTIVE.md`.

```md
# HUMAN_DIRECTIVE.md
- Do not change the dataset.
- Try label smoothing 0.1 before changing the backbone.
- Stop this direction if gain stays below 0.3 for 3 runs.
- Compare against the last trusted baseline, not just the latest run.
```

Case 1: Safer ablation

```md
- Only change augmentation.
- Keep model, optimizer, and training budget fixed.
- Report a clean comparison table after each run.
```

Case 2: Deliberate pivot

```md
- Current ResNet line is saturated.
- Switch to ViT-B/16 only if the last 3 runs plateau.
- Before switching, write a short rationale.
```

Case 3: Suspicious result

```md
- Accuracy jumped unexpectedly.
- Re-run with the same seed and one new seed.
- Do not claim improvement until both runs reproduce.
```

Rule of thumb: let the agent handle repetition, but keep direction, interpretation, and responsibility human.

### What a Good PROJECT_BRIEF.md Looks Like

The brief is your main lever. Here are examples for different scenarios:

<details>
<summary><b>Example: Fine-tuning a pretrained model</b></summary>

```markdown
# Goal
Fine-tune ViT-B/16 (pretrained on ImageNet-21K) on Oxford Flowers-102.
Target: 95%+ test accuracy.

# Codebase
- Training script: finetune.py (already exists)
- Config: configs/vit_flowers.yaml
- Data: /data/flowers102/ (already downloaded)
- Pretrained weights: /models/vit-b16-21k.pth

# What to Try
1. First: freeze backbone, train classifier head only (10 epochs, lr=1e-2)
2. Then: unfreeze all, fine-tune end-to-end (30 epochs, lr=1e-4)
3. If stuck below 93%: try layer-wise lr decay (0.65)
4. If above 94%: try test-time augmentation

# Constraints
- GPU 0, batch size 64
- Save best checkpoint based on val accuracy
```
</details>

<details>
<summary><b>Example: Hyperparameter search</b></summary>

```markdown
# Goal
Find the best hyperparameters for our GAN on CelebA-HQ 256x256.
Target: FID < 15.

# Codebase
- train_gan.py, configs/celeba_gan.yaml
- Data: /data/celeba_hq_256/
- Evaluation: eval_fid.py --real_dir /data/celeba_hq_256/val

# Search Space
- Learning rate: [1e-4, 2e-4, 5e-4]
- Beta1: [0.0, 0.5]
- Discriminator steps per generator step: [1, 2, 5]
- Spectral norm: [yes, no]

# Strategy
Start with lr=2e-4, beta1=0.0, d_steps=1, spectral_norm=yes (baseline).
Change ONE variable at a time. Run each for 50K steps.
Always evaluate FID after training.

# Constraints
- GPU 0-1 (can use both)
- Max 50K steps per run (~4 hours)
```
</details>

<details>
<summary><b>Example: Debugging a training issue</b></summary>

```markdown
# Goal
Figure out why our transformer model diverges after epoch 20.
Currently: loss explodes from 0.5 to NaN around epoch 20-25.

# Codebase
- train_transformer.py, model/transformer.py
- Config: configs/base.yaml
- Logs from failed runs: logs/failed_run_001.log, logs/failed_run_002.log

# What to Investigate
1. Check gradient norms — add gradient clipping (max_norm=1.0)
2. Try lower learning rate (current: 1e-3, try: 1e-4, 5e-5)
3. Check if it's a specific layer — add per-layer gradient logging
4. Try warmup (1000 steps) if not already present
5. Check data — are there any NaN/Inf in the dataset?

# Constraints
- GPU 0, run each test for 30 epochs (enough to see if it diverges)
- Log gradient norms every 100 steps
```
</details>

### FAQ

<details>
<summary><b>Q: How much does it cost to run?</b></summary>

About $0.08 per 24-hour cycle (if training takes 8 hours). The secret: zero LLM calls during training. You only pay for the THINK and REFLECT phases (~10 min each).
</details>

<details>
<summary><b>Q: Can it modify my existing code?</b></summary>

Yes. The Code Agent can read, write, and modify any file in your project. It will make changes, dry-run to verify, then launch training. It won't touch protected files (PROJECT_BRIEF.md, MEMORY_LOG.md).
</details>

<details>
<summary><b>Q: What if the agent goes in a wrong direction?</b></summary>

Drop a directive: `echo "Stop. Go back to the ResNet approach" > workspace/HUMAN_DIRECTIVE.md`. The agent reads it next cycle with highest priority.
</details>

<details>
<summary><b>Q: Can I run multiple projects at the same time?</b></summary>

Yes. Launch separate agent instances in different terminals/tmux sessions, each pointing to a different project and GPU.
</details>

<details>
<summary><b>Q: What happens if training crashes?</b></summary>

The monitor detects the process died, captures the error log, and passes it to REFLECT. The agent will analyze the crash, fix the code, and retry.
</details>

<details>
<summary><b>Q: Can I use it with PyTorch / TensorFlow / JAX?</b></summary>

Yes. The agent works with any training framework. It just launches shell commands and reads log files — it doesn't care what framework produces them.
</details>

---

## One-Click Install (AI Coding Tool Skills)

All features are packaged as slash commands for **Claude Code**, **CodeBuddy**, and **Cursor**. **One command to install:**

```bash
# Install to all supported tools
python install.py

# Install to specific tool
python install.py --claude-code
python install.py --codebuddy
python install.py --cursor
```

### Supported Tools

| Tool | Install Target | Skill Format |
|------|---------------|--------------|
| **Claude Code** | `~/.claude/commands/` | `.md` slash commands |
| **CodeBuddy** | `~/.codebuddy/commands/` | `.md` slash commands |
| **Cursor** | `~/.cursor/rules/` | `.mdc` rule files |

After installation, you get **8 slash commands** in Claude Code:

### Core Skills

| Command | What It Does |
|---------|-------------|
| `/auto-experiment` | Launch the 24/7 autonomous THINK→EXECUTE→VERIFY→REFLECT experiment loop |
| `/experiment-status` | Check running experiments: progress, metrics, cycle count, GPU usage |
| `/gpu-monitor` | Quick GPU status: free/busy, memory, utilization, running processes |

### Research Skills

| Command | What It Does |
|---------|-------------|
| `/daily-papers` | Daily arXiv recommendations with automatic dedup |
| `/paper-analyze 2312.12345` | Deep paper analysis + extract real figures from arXiv source |
| `/conf-search --venue CVPR2025 --query "motion"` | Search CVPR/NeurIPS/ICML/ICLR/AAAI/ECCV... |
| `/progress-report` | Generate structured progress report with metrics |
| `/obsidian-sync` | Refresh Obsidian or local progress notes |

### Usage Example

```bash
# Step 1: Install skills (one time)
python install.py

# Step 2: In Claude Code, launch an experiment loop
/auto-experiment --project /path/to/my_project --gpu 0

# Step 3: Check how it's going
/experiment-status --project /path/to/my_project

# Step 4: Check GPU resources
/gpu-monitor

# Step 5: Read papers while the agent trains for you
/daily-papers --topics "vision transformer, image classification"
```

### Uninstall

```bash
python install.py --uninstall
```

### Python API (for tool integration)

```python
from api import AutoResearcher

# Initialize for a project
r = AutoResearcher("/path/to/project")

# Run one cycle synchronously
result = r.run_one_cycle()

# Get project status
status = r.get_status()

# Start background daemon
pid = r.start_daemon(gpu="0", max_cycles=10)
r.stop_daemon()

# Query learned code review lessons
lessons = r.get_code_review_lessons(severity="HIGH")
```

### CLI API

```bash
python api.py status --project /path/to/project
python api.py run --project /path/to/project --cycles 3
python api.py start --project /path/to/project --gpu 0
python api.py stop
python api.py lessons --project /path/to/project --severity HIGH
```

---

## Supported LLM Providers

Works with **both Anthropic and OpenAI** out of the box. Pick your provider:

| Tier | Anthropic (Claude) | OpenAI (Codex/GPT) | Best For |
|------|-------------------|-------------------|----------|
| **Fast** | `claude-sonnet-4-6` | `codex-5.3` | Daily experiments, iteration |
| **Strongest** | `claude-opus-4-6` | `gpt-5.4` | Complex reasoning, architecture decisions |

Switch provider in `config.yaml`:
```yaml
agent:
  provider: "openai"       # or "anthropic"
  model: "codex-5.3"       # or "claude-sonnet-4-6"
```

Or set via environment:
```bash
# For Anthropic
export ANTHROPIC_API_KEY="sk-ant-xxxxx"

# For OpenAI
export OPENAI_API_KEY="sk-xxxxx"
```

---

## Configuration

```yaml
# config.yaml
project:
  name: "my-research"
  brief: "PROJECT_BRIEF.md"

agent:
  provider: "anthropic"           # "anthropic" or "openai"
  model: "claude-sonnet-4-6"      # See model table above
  max_cycles: -1                  # -1 = run forever
  max_steps_per_cycle: 3          # Max worker dispatches per cycle
  cooldown_interval: 300          # Smart cooldown polling (seconds)

memory:
  brief_max_chars: 3000           # Tier 1 cap
  log_max_chars: 4000             # Tier 2 cap
  milestone_max_chars: 1200       # Key results cap
  max_recent_entries: 15          # Rolling decision count

gpu:
  auto_detect: true
  reserve_last: true              # Reserve last GPU for keep-alive

monitor:
  poll_interval: 900              # Check every 15 min during training
  zero_llm: true                  # No LLM during monitoring

experiment:
  mandatory_dry_run: true         # Always dry-run before real training
  max_parallel: 1                 # Concurrent experiments

# Visual Analysis Module — Inference + Multimodal Diagnosis
visual_analysis:
  enabled: true                   # Master switch for visual analysis
  trigger_threshold: 5            # Trigger after N consecutive poor cycles
  mcp_timeout: 60                # MCP server timeout (seconds)
  max_images: 6                  # Max images per analysis session

# MCP Services — Two Transport Types
# Type 1: SSE (GLM Platform) — web_search, web_reader, zread
#   GET /sse → receive endpoint event → POST /message (dual-connection)
# Type 2: stdio (Local npx) — zai-mcp-server (vision tools)
#   Requires Node.js 18+ and npx
#   If Node.js absent, vision falls through to direct API calls
mcp_services:
  sse:
    - name: web_search_prime
      url: https://open.bigmodel.cn/api/mcp/web_search_prime/sse
    - name: web_reader
      url: https://open.bigmodel.cn/api/mcp/web_reader/sse
    - name: zread
      url: https://open.bigmodel.cn/api/mcp/zread/sse
  stdio:
    - name: zai_vision
      command: npx
      args: ["-y", "@z_ai/mcp-server"]
      env:
        Z_AI_API_KEY: ${GLM_CODING_PLAN_API_KEY}  # Same key as GLM
        Z_AI_MODE: ZHIPU
```

---

---

## Project Structure

```
auto_research_agent/
├── core/                    # Autonomous experiment loop engine
│   ├── loop.py              # THINK → EXECUTE → VERIFY → VISUAL → REFLECT cycle (v16.1: Phase Gate v2 code-scan + phase-aware downgrade)
│   ├── memory.py            # Two-Tier constant-size memory (v12: failure_category system)
│   ├── monitor.py           # Zero-LLM experiment monitoring
│   ├── agents.py            # Leader-Worker agent dispatch
│   ├── tools.py             # Minimal per-agent tool registry
│   ├── verifier.py          # Module-level result verification (v13: unchanged from v12.2)
│   ├── visual_analyzer.py   # Inference + multimodal visual diagnosis
│   ├── simulation_sandbox.py # Pre-training model validation & A/B evaluation
│   ├── constraint_engine.py # LLM behavior control (v16.1: StrategyEngine + ContextPruner, 4 dead modules removed)
│   ├── idea_planner.py      # 9-phase forward design pipeline
│   ├── experiment_evaluator.py # Post-experiment evaluation & failure diagnosis
│   ├── domain_knowledge.py  # Dynamic domain knowledge injection
│   ├── context_keys.py      # Context key registry with validation
│   ├── research_roadmap.py  # Module-level research state machine (v15)
│   └── mcp_client.py        # MCP transport (SSE + stdio)
├── skills/                  # Skill commands (python install.py --all)
├── api.py                   # Python API for tool integration
│   ├── auto-experiment/     # 24/7 autonomous experiment loop
│   ├── experiment-status/   # Check experiment progress
│   ├── gpu-monitor/         # GPU status & availability
│   ├── daily-papers/        # Daily arXiv recommendations
│   ├── paper-analyze/       # Deep paper analysis + figure extraction
│   ├── conf-search/         # Conference paper search
│   ├── progress-report/     # Progress report generation
│   ├── error-handler/       # 7-step diagnostic workflow for systematic fixes
│   ├── experiment-auditor/  # Post-cycle audit for shortcuts & hallucinations
│   ├── code-cleanup/        # Automatic code hygiene & log archiving
│   ├── dataset-understanding/ # Data directory validation & manifest generation
│   ├── idea-validation/     # Hypothesis validation before committing GPU time
│   ├── hands-off-issue-handling/ # Autonomous issue resolution protocol
│   └── REASONING_PRINCIPLES.md # Mandatory reasoning guidelines for all agents
├── agents/                  # Agent prompt definitions
│   ├── leader.md            # Central decision-maker
│   ├── idea_agent.md        # Literature & hypothesis
│   ├── code_agent.md        # Experiment execution
│   └── writing_agent.md     # Reporting & writing
├── gpu/                     # GPU utilities
│   ├── detect.py            # Detection & monitoring
│   └── keeper.py            # Cloud instance keep-alive
├── examples/                # Ready-to-run demos
├── docs/                    # Docs + translations (CN/JP)
├── install.py               # Multi-tool skill installer (Claude Code / CodeBuddy / Cursor)
├── config.yaml              # Default configuration
└── requirements.txt         # Dependencies
```

---

## Contributing

Areas where we'd love help:
- More cloud GPU platforms (AWS, GCP, Lambda Labs, RunPod)
- Experiment tracker integration (W&B, MLflow, TensorBoard)
- New research skills (visualization, result comparison)
- Metric extraction for more training frameworks

See [CONTRIBUTING.md](CONTRIBUTING.md).

---




## v17 — Systemic Architecture Fixes

A function-level code review found **5 structural diseases** (not 40 isolated bugs). Each had one root cause and one systemic fix. **97 tests** (up from 0) protect every fix.

| Fix | Disease | Impact | Measured improvement |
|-----|---------|--------|---------------------|
| **A. Context Schema** | 37/48 context keys silently dropped — domain knowledge, architecture plans, training analysis all computed but never reached the LLM | `context_keys.py` became the single source of truth (inject + prune + serialize all read one registry); `_format_leader_input` 170→8 lines | Prompt sections 13→21; keys reaching LLM 19%→98% |
| **B. Advisory→Enforced** | Audit + constraint engine detected problems but couldn't change behavior (LLM ignored text directives) | Audit: per-signature counter, 2 strikes→forced fix, 3→pause. Constraint: dead-end 5+ now sets `forbidden` (hard gate reachable) | FORBIDDEN gate reachable; audit escalations have enforcement teeth |
| **C. Shared Primitives** | 15+ AST-scan copies + 4 loss-parse copies with 3 divergent thresholds | `training_log_parser.py` + `model_structure_scanner.py` | Also fixes "0 params" bug (kwargs now handled) |
| **D. Dead Code** | ~200 lines of never-called methods | Deleted 7 dead methods + dead config fields | — |
| **E. loop.py split** | 5018-line god-object | Deferred (0 dead methods, pure maintainability, needs runtime baseline) | — |

**IdeaScout integration**: `core/idea_scout_bridge.py` adds cross-domain idea discovery to the `paper_research` phase. Finds papers from other fields whose core ideas can transfer to your task (not just keyword-similar papers). Auto-generates a research profile from `PROJECT_BRIEF.md`. Disabled by default (`idea_scout.enabled: false`).

See `docs/architecture.md §36` for full details.

---

## v18 — Architecture Reform (principle-driven, 4 phases)

A 25-dimension system audit found **8 structural root causes**. Reform follows 5 principles: safety is a tool property (not a layer), information is queried (not injected), LLM works autonomously, memory replaces enforcement, prompt defines methodology (not control). **134 tests** (up from 0).

| Phase | What changed | Measured effect |
|-------|-------------|-----------------|
| **1. Knowledge loop** | 89 lessons + 102 causal chains + experiment values were stored but 0% consumed. Fixed consumption (SQL schema, verified filter, models/ fallback). System writes quantitative metrics to MEMORY_LOG. Goal tracking + auto-stop on achievement. | Knowledge injection: 0% → >80% |
| **2. Tool safety** | write_file enforces naming conventions (root .py/train_*/debug_*). Deterministic GC (no LLM, no quota). Dry-run gate. | tools/ files: 117 → auto-archived |
| **3. Remove enforcement** | Deleted 245 lines of 0-trigger research-decision methods (stagnation/architecture-switch). Disabled 4 circuit breakers. | Enforcement layers: 7 → 3 |
| **4. Generalization** | Metric keys config-driven (not hardcoded val_MAE). Prompt trimmed. | Domain-agnostic metrics |

New modules: `signal_arbiter.py`, `garbage_collector.py`, `training_log_parser.py`, `model_structure_scanner.py`, `idea_scout_bridge.py`. Context key registry (`context_keys.py`) is the single source of truth for injection/pruning/serialization.

See `docs/architecture.md §37` for full details.

---

## License

Apache 2.0 — see [LICENSE](LICENSE).

---

<p align="center">
  <strong><i>"Experiments run through the night. Results arrive at dawn."</i></strong>
</p>
