# Function-Level Code Review & Refactoring Plan

> **Scope**: Every function in every `core/` module evaluated — does it work, is it called,
> should it stay/go/split. Based on grep-verified caller analysis across the entire `core/` tree,
> not on reading alone. Findings cross-checked against the live run log.
>
> **Verdict categories**: KEEP (works + used) / DEAD (0 callers, delete) / BROKEN (called but
> can't work correctly, fix) / BLOAT (works but oversized, split) / DORMANT (works but unreachable
> in production config) / SILENT-DROP (output computed but never reaches consumer)

---

## Executive Summary

| Module | Lines | Funcs | KEEP | DEAD | BROKEN | Key finding |
|--------|-------|-------|------|------|--------|-------------|
| loop.py | 5018 | 66 | 66 | 0 | 0 | God-object; 0 dead but ~2400 lines extractable into 5 modules |
| verifier.py | 2679 | 47 | 44 | 1 | ~7 | 1 dead property; 15+ silent-swallow `except:pass` sites |
| experiment_evaluator.py | 800 | 15 | 14 | 0 | ~2 | Silent-pass defaults make probe failures look like passes |
| simulation_sandbox.py | 1518 | 36 | 24 | 4 (prod) | 1 | 4 layers never run in production (no `val_data_dir`); PFM loader bug |
| agents.py | 2274 | 39 | 36 | 2 | 1 | 2 dead methods; GLM guard returns error-string not raise |
| tools.py | 1906 | 49 | 49 | 0 | 0 | `WORKER_CONFIGS["tools"]` is dead divergent documentation |
| memory.py | 1200 | 46 | 39 | 7 | 0 | 7 dead methods + dead `roadmap_history` table + dead `in_llm_context` column |
| idea_planner.py | 1113 | 32 | 32 | 0 | 0 | Capacity/fusion/verification plan never reaches LLM (P0) |
| domain_knowledge.py | 650 | 8 | 8 | 0 | 0 | **Entire 650-line output silently dropped** (P0) |
| model_analyzer.py | 2323 | 37 | — | — | 1 | `_estimate_params_from_ast` ignores kwargs → returns 0 |

**Total dead code to delete**: ~10 methods + 1 table + 1 column ≈ 200 lines.
**Total extractable from loop.py**: ~2400 lines into 5 new modules.
**Most impactful bug**: P0 — 650+ lines of domain knowledge + idea planning computed every cycle, never reaching the Leader LLM.

---

## 🔴 P0 — The Silent-Drop Bug (highest impact)

**Where**: `agents.py:1855-2025` (`_format_leader_input`)
**What**: This method serializes the `context` dict into the Leader's prompt. It reads ~16 keys
(`brief`, `memory_log`, `cycle`, `session_stats`, `experiment_result`, `verify_diagnosis`, etc.)
but **does NOT read**:
- `domain_knowledge` (650 lines of DomainKnowledgeMixin output)
- `data_constraints`
- `cross_experiment_insights`
- `architecture_plan` / `architecture_plan_summary` (IdeaPlanner output)
- `pareto_frontier`
- `causal_history`
- `hypothesis_calibration`
- `data_scarcity_warning`

**Why this matters**: These keys are built every cycle, stored in `context`, survive
`ContextPruner` pruning (they're TIER_2/TIER_3), then **discarded at serialization time**. The
entire domain-knowledge subsystem and idea-planning subsystem produce output that never reaches
the LLM. This is why the agent "ignores" domain knowledge — it literally never sees it.

**Fix**: Add serialization blocks in `_format_leader_input` for the missing keys. Each block:
```python
if context.get("domain_knowledge"):
    parts.append(f"### Domain Knowledge\n{context['domain_knowledge']}")
```

---

## loop.py — Split Plan (the core ask)

**Current**: 5018 lines, 66 methods, 1 god-class. **0 dead methods** (every method has ≥1 caller).
The problem isn't dead code — it's that 8 unrelated responsibility clusters share one namespace
and one `self` blob.

### What stays in loop.py (~2600 lines, the orchestration skeleton)

Cluster A — Orchestration core: `run`, `_think`, `_execute`, `_verify`, `_reflect`,
`_record_cycle_outcome`, `_execute_paper_research`, `_execute_idea_scout`, `_monitor_experiment`,
`_pre_verify`, `_handle_signal`, `_load_state`, `_update_state`, `_load_cycle_counter`,
`_save_cycle_counter`, `_consume_directive`, `_smart_cooldown`, `_setup_file_logging`.

### What gets extracted (~2400 lines → 5 new modules)

| Cluster | New module | Lines | Methods | Extraction difficulty |
|---------|-----------|-------|---------|----------------------|
| **C. Code review** | `core/code_review.py` → `CodeReviewer` | 835 | 13 | Medium (shares 2 helpers with F) |
| **F. Phase gates** | `core/phase_gates.py` → `PhaseGate` | 449 | 7 | Medium (shares 2 helpers with C) |
| **G. Metrics/analysis** | `core/metrics_analysis.py` → `MetricsAnalyzer` | 364 | 4 | Easy (narrow interface) |
| **D. Audit/escalation** | `core/audit_escalation.py` → `AuditEscalator` | 232 | 5 | Easy (cleanest interface) |
| **E. Stagnation** | `core/stagnation.py` → `StagnationTracker` | 266 | 5 | Easy |

### Recommended extraction order (by difficulty, lowest first)

1. **D (Audit) first** — narrowest `self.*` interface. The one cross-cutting side effect
   (`_handle_escalated_issues` sets `self._running = False`) becomes a return value
   `("pause",)` that `run()` acts on.
2. **G (Metrics)** — only needs `self._best_metric_ever`, `self.cycle_count`, `self.memory`.
3. **E (Stagnation)** — owns ~10 counter attributes; these move INTO `StagnationTracker`.
4. **F (Phase gates)** — needs `self.roadmap`, `self.workspace`. Shares
   `_extract_model_path_from_task` with C — move that to a shared util.
5. **C (Code review) last** — biggest (835 lines) but also most self-contained. Needs
   `self.dispatcher`, `self.memory`, `self.project_dir`.

### Cluster B (Context building) — NOT directly extractable

There is **no `_build_context` method**. Context injection (~660 lines, ~80 keys) is **inline
in `_think`** (L843-1196) and **inline in `_reflect`** (L2379-2776). Before extracting, these
inline blocks must be refactored into `_build_think_context()` / `_build_reflect_context()`
methods first. This is a prerequisite, not a one-shot move.

### Shared helper extraction (cross-cluster)

Two methods are shared between clusters C and F:
- `_extract_model_path_from_task` (L1519-1543) — used by code review + phase gate
- `_find_training_script_content` (L1940-1975) — used by code review + phase gate

Move these to `core/text_utils.py` or keep in `loop.py` as shared utilities.

---

## Stubs & No-Ops audit (Task 3 results)

**No pure no-ops found.** All three flagged methods do real work:
- `_run_dataset_understanding` — genuinely dispatches the code agent + checks manifest creation
- `_estimate_experiment_value` — computes real VOI + mutates `think_result` with warnings
- `_check_audit_escalation` — writes `DIRECTIVE.md`, sets `_running=False`, records dead-ends

**However**: `_estimate_experiment_value`'s LOW-VOI warning threshold (`voi < 0.005 AND prior < 0.3`)
is tuned so tightly (prior floored at 0.1, expected_improvement defaults to 0.05) that `voi` floor
is ~0.005 — the warning sits right at its trigger boundary and may rarely fire. Not dead, but
effectively dormant.

---

## Cross-cutting duplication (the consolidation targets)

### 1. AST-walking for `nn.Module` — **15+ copies across 4 files**

`ast.walk` → `ClassDef` → `base.attr == "Module"` → collect `self.X` assigns is copy-pasted in:
- `verifier.py` (6 sites: L277, L301, L1681, L1823, L1889, +2)
- `experiment_evaluator.py` (1 site: L402)
- `simulation_sandbox.py` (1 site: L964)
- `model_analyzer.py` (7+ sites)

**→ Create `core/model_structure_scanner.py`** with:
- `find_nn_module_classes(tree) -> list[ClassDef]`
- `collect_init_self_assigns(class_node) -> dict`
- `estimate_params_from_ast(class_node) -> int` (fixes the kwargs bug too)
- `find_fusion_calls(class_node) -> list[Call]`

### 2. Loss parsing — **4 copies, 3 divergent thresholds**

`re.findall(r"loss[=:\s]+([0-9.]+)")` at:
- `verifier.py:820` (stagnation), `verifier.py:1059` (decrease: `*0.99`), `experiment_evaluator.py:311` (decrease: `*0.99`, diverge: `*2.0`), `loop.py:4851` (10% windows)

**→ Create `core/training_log_parser.py`** with:
- `parse_loss_series(log_text) -> list[float]`
- `classify_loss_trend(floats) -> TrendResult` (single threshold set)
- `analyze_curve(floats) -> CurveReport`
- `load_training_log(project_dir, execute_result) -> dict` (cached)

### 3. Subprocess probe-script builders — **4 near-duplicate copies**

`experiment_evaluator.py:665`, `simulation_sandbox.py:1058/1194`, `model_analyzer.py:1429` all
build 50-170 line inline Python scripts that: find nn.Module → load checkpoint → forward →
collect stats.

**→ Consolidate into templated scripts** under `core/sandbox_scripts/` loaded via `string.Template`.

---

## Per-module dead-code deletion list (safe to delete now)

| File | Line(s) | What | Action |
|------|---------|------|--------|
| `agents.py` | 818-840 | `_resolve_model_for_provider` (0 callers) | DELETE |
| `agents.py` | 2112-2128 | `_extract_first_decision_json` tail after `return None` (unreachable) | DELETE |
| `agents.py` | 1048 | GLM base_url guard returns error-string | → `raise RuntimeError` |
| `memory.py` | 313 | `get_metric_trend` (0 callers) | DELETE |
| `memory.py` | 750 | `get_full_context` (0 callers) | DELETE |
| `memory.py` | 827 | `resolve_active_problem` (0 callers) | DELETE |
| `memory.py` | 1118 | `get_log_summary` (0 callers) | DELETE |
| `memory.py` | 1155 | `log_roadmap_update` (0 callers, aspirational docstring) | DELETE |
| `memory.py` | 1181 | `get_roadmap_history` (0 callers) | DELETE |
| `memory.py` | 224-238 | `roadmap_history` table DDL (never written/read) | DELETE DDL |
| `memory.py` | 295,803 | `in_llm_context` column (written, never read) | DROP column |
| `verifier.py` | 61 | `VerifyReport.critical_failures` property (0 callers) | DELETE |
| `simulation_sandbox.py` | 1434 | PFM loader bug (`dtype=np.float32 if scale > 0 else np.float32` — both branches identical) | FIX |
| `agents.py` | 441-459 | `WORKER_CONFIGS[*]["tools"]` (never read, divergent from `get_tools_for`) | DELETE field |

---

## Silent-swallow `except: pass` sites (verification blind spots)

15+ sites in `verifier.py` silently drop verification checks on error, making failures invisible:
Lines 285, 309, 933, 1004, 1053, 1311, 1412, 1441, 1464, 1569, 1690, 1757, 2340, 2541, 2629.

**Pattern**: `except Exception: pass` or `except (...Exception): pass` where `Exception` makes
the specific types unreachable. Each silently drops a check → the verify report looks clean when
it shouldn't.

**Fix**: Replace each with `except (SpecificError,): logger.debug(...)` and drop `Exception`
from tuples. Never bare `except Exception: pass` in a verifier.

---

## Sandbox dead-in-production layers

4 methods in `simulation_sandbox.py` are correct code that **never executes** because
`loop.py:2670` calls `full_evaluation()` without `val_data_dir` / `gt_dir` / `input_shape`:
- `evaluate_vs_reference` (L433) — Layer 2a, the headline A/B comparison
- `_find_val_samples` (L921)
- `_run_inference` (L943)
- `_build_inference_script` (L1367)

**Two options**: (a) wire up the missing args (discover `val_data_dir` from
`DATASET_MANIFEST.json`, infer `input_shape` from the model's first conv); (b) delete the 4
methods and stop advertising "5-layer evaluation."

---

## The audit/architecture question: "meaningful but unused"

> "有的模块函数是有意义的，比如审计，但是如果一直用不上，也是一种浪费，该如何修改？"

The audit escalation system (cluster D, 232 lines) is **functional but wasteful** in its current
form: it fires, writes `DIRECTIVE.md`, the agent ignores it, it escalates to L2/L3, the death-loop
guard clears it, and it re-accumulates from zero. The work is done but the signal is wasted.

**Architectural fix**: Don't delete it — change its **consumption model**. Currently audit findings
flow: `VERIFY → audit → DIRECTIVE.md → next THINK reads it → agent ignores`. The break is at the
"agent ignores" step. Two options:

1. **Make audit findings structurally enforced** (like Phase 4's `_enforce_launch_after_failure`):
   after 2 cycles of the same audit issue, rewrite `think_result["action"]` to a targeted fix task.
   After 3, `pause_human`. This replaces the advisory DIRECTIVE.md covert-channel with real teeth.

2. **Fold audit into the existing Phase 4 counter**: `_consecutive_failed_launches` already
   forces action. Generalize it to `_consecutive_audit_issues[signature]` with the same
   2-strike-forced-fix / 3-strike-pause policy. One enforcement mechanism, not two.

The principle: **advisory systems that the LLM can ignore are wasted compute.** Either give them
enforcement power (rewrite action) or delete them. The current "inject text and hope" model is
the worst of both worlds — it costs cycles but changes nothing.
