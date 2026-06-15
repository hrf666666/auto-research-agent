# Architecture Review — Systemic Diagnosis & Redesign

> **Method**: 6 parallel module-level reviews (dispatch, orchestration, tools, verification,
> memory/intelligence, infra) → every HIGH/MED finding verified against the actual source and
> the live run log (`autoresearcher.log` from the depth-estimation project, 2026-06-15).
> False positives removed; one scale overstatement corrected.
>
> **Stance**: This is *not* a patch list. It identifies the few **cross-cutting structural
> diseases** that produce the dozens of surface symptoms, and proposes an elegant redesign per
> disease. Fix the disease, the symptoms disappear together.

---

## TL;DR — The whole codebase has **four diseases**, not forty bugs

| # | Disease | One-line | Live-log proof |
|---|---------|----------|----------------|
| **D1** | **String-coupling everywhere** | Phases, launches, gates, directions, lessons are all detected/enforced by regex & keyword-sniffing of free LLM text. There is no structured event the system can trust. | "0 gate triggers in 11 cycles"; launch-invisible because script name ≠ regex; roadmap "detected then ignored". |
| **D2** | **Detection without enforcement** | A huge fraction of "safety" features (Idea Guardian, Direction Circuit Breaker, Data Scarcity "hard wall", Constraint Engine FORBIDDEN, ROADMAP deviation #1, VERIFY criticals) only *log* or *inject a prompt string*. None alter `think_result["action"]`. | "ROADMAP DEVIATION … task=📊 PILOT RECOMMENDED" → experiment ran anyway; "constraint violation" → proceeded. |
| **D3** | **State scattered & incoherent** | Cycle state lives across ~31 in-memory fields + 5 JSON files + a `DIRECTIVE.md` filesystem covert-channel + SQLite, with no single schema and no synchronization. Counters self-reset or dead-end. | Audit death-loop auto-clears then re-accumulates forever; hard-gate "streak capped at 2" loops; architecture counter in-memory → reset on restart → re-does the switch. |
| **D4** | **Resilience-by-exhaustion** | Every external dependency (provider quota, MCP, Semantic Scholar, DuckDuckGo) fails via serial cascade with no backoff/circuit-breaker/cooldown; mid-generation timeouts throw away completed work and restart from zero. | 144× quota-429 errors; 88 failovers; 24 timeouts; MCP 17× + web_reader 7× timeouts; glm-5.2 wrote all deliverables, timed out on summary, glm-5.1 restarted from scratch. |

Everything below is a *manifestation* of one of these four. The redesign section collapses all of them.

---

## PART 1 — Verified findings (by disease)

### D1 — String-coupling (detection/enforcement by text-sniffing)

| Sev | Finding | Location | Verified |
|-----|---------|----------|----------|
| HIGH | **Launch detection is a regex on shell text** (`\bpython\b…\btrain(_v\d+)?\.py\b`); a renamed script, `python -m`, torchrun, or the naming convention in doc §16 (which *forbids* `train_v2.py`) makes a real launch invisible → `experiment_launched=False` → cycle scores as no-progress. This is the **single biggest reason a "successful" cycle produces no signal.** | `agents.py:1960-1972`, `verifier.py:516-525` | ✅ regex confirmed; live log: "EXECUTE did not launch" ×3 |
| HIGH | **`_estimate_params_from_ast` only handles `Conv2d`/`Linear`/`BN`/`GroupNorm` with all positional `ast.Constant` args.** Returns 0 for any model using kwargs, custom modules, Conv3d, or builders. The 9-layer analysis then runs on `total_params=0` and emits *plausible-looking garbage* (worse than an error). Duplicated & divergent copy in `simulation_sandbox.py:949-1023`. | `model_analyzer.py:172-194` | ✅ confirmed; live log: "0 params detected" |
| HIGH | **ROADMAP phase-violation detector is text-keyword based** (`research_roadmap.py:589-609`), the exact failure mode v16.1 documented and fixed *for the phase gate* but **not** for the roadmap alignment check. Abstract task wording ("validate angular decomposition") passes as "aligned". | `research_roadmap.py:355-364` calls text-based `_is_training_task` | ✅ confirmed |
| HIGH | **Code-review regex runs on a half-correct comment/string stripper** (`split('#')[0]`) that corrupts code containing `#` inside strings → both false-positive and false-negative HIGH blocks feed the hard gate. | `loop.py:1467-1484` ("doesn't handle '#' inside strings" admitted in comment) | ✅ confirmed |
| MED | `_strip_comments_and_strings` + `_regex_code_review_*` is the engine of the hard gate — the gate's correctness is bounded by a regex mini-parser. | `loop.py:1486-1650` | ✅ |
| MED | Dead-end rule matching is substring-keyword intersection on a 16-word hardcoded list → trivially bypassed by rephrasing. | `constraint_engine.py:282-286, 201-210` | ✅ |
| MED | Lesson retrieval is substring-match of `pattern` (e.g. "multi branch fusion") inside code text → KB fills but is effectively write-only. | `memory.py:966-967`, `loop.py:2107,2126` | ✅ |
| MED | Pareto rule matching splits the human-readable description by commas. | `constraint_engine.py:297` | ✅ |
| LOW | `_extract_innovations` embeds pattern strings into a new regex (latent footgun). | `idea_planner.py:392-397` | ✅ |

**Root pattern:** the system never records a *structured event* ("experiment launched", "phase transitioned", "module verified"). It tries to *re-derive* the event by reading free text afterward. Every such re-derivation is a new regex in a new file that drifts from every other one.

---

### D2 — Detection without enforcement

| Sev | Finding | Location | Verified |
|-----|---------|----------|----------|
| HIGH | **Constraint Engine "FORBIDDEN" hard-gate is unreachable.** Auto-generated rules only ever set priority `"high"`/`"medium"` (never `"forbidden"`); `has_forbidden_violation` searches for the literal `[CONSTRAINT:FORBIDDEN]`. So the engine *always* logs and *never* blocks — unless a human-authored `STRATEGY_RULES.json` (absent from repo) exists. The log line "StrategyConstraintEngine: 1 constraint violation(s)" is, in practice, the engine doing nothing. | `constraint_engine.py:118-188, 303-314`; `loop.py:1163-1185` | ✅ confirmed: auto priorities are high/medium only |
| HIGH | **`_enforce_roadmap_alignment` on deviation #1 only APPENDS correction text to `task` and returns the original `experiment` action unchanged.** The `_roadmap_alignment_warning` key it sets is **never read anywhere** (single grep hit = the write). So off-roadmap experiments run with a scolding footnote. | `loop.py:3423-3431`; `_roadmap_alignment_warning` read-count = 0 | ✅ confirmed dead key |
| HIGH | **Idea Guardian, Direction Circuit Breaker, Data Scarcity "hard wall" are all pure prompt-string injection** into a context dict. None rewrite `action`, none re-prompt the leader, none block. The doc calls them "circuit breaker"/"hard wall" — the code does string concatenation. | `loop.py:867-885, 896-914` | ✅ confirmed |
| HIGH | **VERIFY's critical/high diagnoses are advisory-only.** After `verify_report = self._verify(...)`, the loop proceeds unconditionally; only REFLECT (an LLM call) consumes the diagnoses. The "THINK planned but EXECUTE did not launch" HIGH diagnosis is logged 3× with no corrective action. | `loop.py:609`; `verifier.py:516-525` | ✅ live log ×3 |
| MED | Direction-stagnation counter self-resets the cycle the breaker "fires" (`_direction_stagnation_count=0`), so a stuck agent loops with the breaker firing every N cycles and never escalating. | `loop.py:3762-3773` | ✅ |
| MED | `context_keys.validate_context` warnings emitted at `logger.debug` and wrapped in `try/except: pass` — schema unenforced. Registry drifted: 5 ghost keys (removed modules), 12+ untracked keys. | `loop.py:1205`; `context_keys.py:82-165` | ✅ |

**Root pattern:** the codebase separates *detection* from *enforcement* (a sound instinct) but then implements enforcement as either (a) appending text to a prompt, or (b) nothing. A "gate" that cannot change `action` is not a gate.

---

### D3 — State scattered & incoherent

| Sev | Finding | Location | Verified |
|-----|---------|----------|----------|
| HIGH | **Audit death-loop is structural.** After 5 consecutive directives, the guard clears the directive, resets `_consecutive_audit_directives=0`, and `pop`s the issue signature → the identical issue re-accumulates from zero next cycle → infinite 5-cycle oscillation. The reset-on-fire defeats the detector. | `loop.py:4096-4117` | ✅ confirmed; live log "DEATH LOOP … AUTO-CLEARING" |
| HIGH | **Hard-gate streak is a dead-end, not a backoff.** `# cap at 2 to avoid overflow` clamps `_hard_gate_consecutive_blocks=2` in non-VALIDATED phase. The downgrade path requires VALIDATED. So in Phase 1 it blocks at streak=2 forever — never escalates, never yields. | `loop.py:483-498` | ✅ confirmed (no real overflow risk — just a counter mistreatment) |
| HIGH | **6 independent gate mechanisms with uncoordinated counters stack** (Phase Gate v2, Strategy/FORBIDDEN, Roadmap deviation, no-progress fallback, Pre-VERIFY, Pre-EXECUTE code review). Each has its own streak + its own rewrite of `think_result["action"]`. No single object sees "blocked N times for reason R." | `loop.py:270-289, 424-575, 1158-1185, 3370-3437, 3439-3592` | ✅ |
| HIGH | **Architecture-switch state is in-memory only** → resets on restart → switch re-fires → re-does all work. Stagnation only resets if the new arch name matches a hardcoded pattern dict. | `loop.py:154`, `_ARCHITECTURE_PATTERNS:2977` | ✅ |
| MED | `DIRECTIVE.md` is a filesystem covert-channel: audit escalator + SMART CIRCUIT BREAKER both write the same path; no type/version/dedup. | `loop.py:4161, 742` | ✅ |
| MED | "Summarize" compaction is truncation: keeps last 3 entries verbatim, replaces older with first-60-chars slice (usually a timestamp). Knowledge destroyed, not compressed, despite SQLite holding the full data. | `memory.py:1093-1116` | ✅ |
| MED | `failure_category` column added by runtime `ALTER TABLE` in the write path; category queries `except Exception → return []` until first categorized dead-end → silent false negatives. | `memory.py:801-816, 391-393` | ✅ |
| MED | `ContextPruner` 14-key cap is phase-blind; REFLECT's 6 TIER_2 diagnostic keys + 5 always-include leaves 3 slots → critical diagnostics silently dropped (only `logger.debug`). | `constraint_engine.py:383-422` | ✅ |

**Root pattern:** there is no `CycleState`/`SessionState` object. State is a cloud of `self._*` fields (≈31), JSON files read 3-5× per cycle, and a markdown file abused as a message bus. Every counter invents its own reset rule, and most reset rules are wrong.

---

### D4 — Resilience by exhaustion

| Sev | Finding | Location | Verified |
|-----|---------|----------|----------|
| HIGH | **429 quota-exhaustion is classified transient.** `_is_permanent_error` explicitly excludes 429. But the live error is *"5-hour account-wide quota reached"* — shared by the whole `GLM_CODING_PLAN_API_KEY`. So glm-5.2→5.1→5→turbo→4.7→4.6 all 429 identically (6 wasted calls × every dispatch), then provider-failover repeats. **144× quota-429 messages in one run.** | `agents.py:266-272` | ✅ live log 144× |
| HIGH | **Provider cooldown is too weak to matter.** Triggers at 3 consecutive failures, lasts 300s; but the quota window is 5h. And the "last resort re-add primary" defeats cooldown entirely. | `agents.py:711-718, 742-744` | ✅ |
| HIGH | **On timeout/failover the entire dispatch restarts from zero.** `_call_openai_compatible` rebuilds `api_messages` from `system+messages` each call (line 872). A mid-generation timeout throws away N completed tool turns; the next model re-reads & rewrites everything. | `agents.py:555-596, 870-876` | ✅ confirmed; live log: glm-5.2 wrote deliverables, timed out, glm-5.1 restarted |
| HIGH | **No backoff/circuit-breaker/rate-limit anywhere in the tool layer.** Grep for `retry|backoff|circuit|cooldown|429` in mcp_client.py + search tools = **0 hits**. Every search is a serial cascade: MCP(20s)+SemanticScholar(15s)+DuckDuckGo(15s) ≈ 50s blocking per call, re-attempted every cycle, with zero memoization. | `mcp_client.py`, `tools.py:988-1336` | ✅ live log: MCP 17×, web_reader 7×, DDG 10×, SS 4× timeouts |
| HIGH | **MCP SSE reader thread + socket leak on session eviction.** Reader thread started (daemon, handle discarded) holding the HTTP socket; eviction `.pop()`s without setting `stop_event`. Every MCP timeout leaks a thread + fd. | `mcp_client.py:214-219, 337, 355` | ✅ |
| HIGH (latent) | **`generate_diagnostic` raises `NameError`** — `_build_diagnostic_script` references `shape_str` (lines 1953/2007/2060/2088) which is only assigned in `_build_probe_script`. File parses fine; crashes at runtime when the tool is invoked. Tool is registered & reachable. | `model_analyzer.py:1842, 1953` | ✅ confirmed reachable + undefined |
| HIGH | **Inference subprocess leaks GPU-holding children on timeout** (`subprocess.run` timeout kills only direct child; DataLoader workers orphaned, hold GPU memory). | `visual_analyzer.py:327-347` | ✅ |
| HIGH | **`read_file` crashes on directories** — `.exists()` is True for dirs, `.read_text()` raises `IsADirectoryError`, except catches only `UnicodeDecodeError`. | `tools.py:960-968` | ✅ live log "[Errno 21] Is a directory" |
| MED | MCP `_mcp_call_tool` returns `None` for all 6 failure modes → permanent 401 retried forever (re-init loop). | `mcp_client.py:335-375` | ✅ |
| MED | Timeout applied to both POST and response-poll sequentially → ~2× wall time. | `mcp_client.py:248-252, 340-344` | ✅ |

**Root pattern:** the system treats every external failure as "try the next thing immediately, harder." There is no concept of "this provider is down until time T" or "this work is partially done, resume it." Resilience = exhaustion.

---

### Cross-cutting (not a disease, but consequence of all four)

| Sev | Finding | Verified |
|-----|---------|----------|
| HIGH | **God-object `loop.py` (4802 lines)** doing orchestration + code-review + audit + phase-gates + progress + cleanup + dataset-understanding + metrics. `DomainKnowledgeMixin` proves the decomposition pattern was understood then abandoned. *(Note: sub-agent claimed "78 attrs / 360-line context"; actual measurement is ~31 `self._*` in `__init__` and 24 `context[` injections in `_think`. The god-object diagnosis stands; the specific counts were overstated ~2×.)* | ✅ |
| MED | **`STRONG_MODEL_TASKS` secretly includes `"code"`** (line 247) contradicting doc/config ("code→fast"). The most token-heavy agent runs on the most expensive thinking model → burns quota faster → accelerates D4. | ✅ confirmed |
| MED | **glm-5.2 is usable in practice** (14 live calls succeed) yet config/code comments say it returns 403 and is "reserved." The failover chains actually route to it. Doc/config lie about the model. | ✅ live log 14× |
| MED | Domain hardcoding survived the "agnostic v9" cleanup: `_degraded_reflect` hardcodes Lambertian/Non-Lambertian/Urban; 5 overlapping keyword dictionaries (3 near-identical `epi` lists that already drifted); MAE hardcoded in `memory.get_summary_stats`. | ✅ |
| MED | Sandbox "5-layer" design collapses in production: Layer 2a never runs (`val_data_dir` never passed), Layer 1 needs snapshots that rarely exist, Layer 0 gated on a default `[1,3,64,64]` shape wrong for depth-estimation → whole sandbox returns empty. | ✅ |
| LOW/MED | Dead code: `_resolve_model_for_provider` (uncalled), `_extract_first_decision_json` tail after `return None` (unreachable), `_extract_mcp_text` in visual_analyzer, dead `dead_branches` loop, residual `return json.dumps({"error":...})` contract violation at `agents.py:847-859`. | ✅ |

---

## PART 2 — Systematic redesign (the elegant fix)

The four diseases have **one shared enabling cause**: the system has no **typed event spine** and no **typed state spine**. Everything is free text flowing between free fields. The redesign installs those two spines. Below, each disease maps to one structural change.

### Architectural target

```
                         ┌──────────────── typed EVENT BUS (in-process) ──────────────┐
   THINK/EXECUTE/        │  ExperimentLaunched | PhaseTransition | ModuleVerified       │
   VERIFY/REFLECT  ─────▶│  DeviationDetected  | GateDecision    | ProviderQuotaExhausted│
   (thin orchestrator)   │  (immutable dataclasses; emitted ONCE at the source of truth)│
                         └───────────────┬──────────────────────────────┬──────────────┘
                                         │                              │
                          ┌──────────────▼──────────┐      ┌────────────▼──────────────┐
                          │   SessionState (typed,  │      │   Subscribers (each a     │
                          │   persisted, atomic)    │      │   single-responsibility   │
                          │  - CycleContext         │      │   collaborator, NOT a     │
                          │  - per-signature gates  │      │   mixin/god-object):      │
                          │  - counters that don't  │      │  CodeReviewer, AuditEsc,  │
                          │    self-reset           │      │  PhaseGate, GatePipeline,  │
                          └─────────────────────────┘      │  ProviderRouter, ...       │
                                                           └───────────────────────────┘
```

**`loop.py` target size: ~600 lines** — only the phase skeleton + event emission + state read/write. Everything currently bolted into `ResearchLoop` becomes a subscriber/collaborator.

---

### Fix for D1 (string-coupling) — **emit structured events at the source of truth**

> *Stop re-deriving truth from text. Record it once, typed.*

1. **`ExperimentManifest` event.** `launch_experiment` is the *only* sanctioned launch path. It writes a structured `experiment_manifest.json` (pid, script, log, shape, seed) **and** emits an `ExperimentLaunched` event. EXECUTE's return contract is: a manifest, or an explicit `LaunchAborted(reason)`. **Delete the regex launch-sniffing in `agents.py` entirely** — `experiment_launched` becomes `event was emitted`, not "text matched a pattern." This kills the launch-invisibility bug, the no-progress miscounting, and the silent-EXECUTE-failure symptom in one move.

2. **`ModelIntrospection` service.** Replace all AST-source param estimation (`_estimate_params_from_ast` + the divergent sandbox copy) with **one** runtime probe: import the model in a subprocess and `sum(p.numel())`. The probe script already exists and works (`model_analyzer.py:1630`). Source-AST becomes a *fallback* flagged `low_confidence=True`, never the primary number. One `PyTorchASTInspector` (channels, branches, dead-modules) shared by verifier + sandbox + analyzer. **Kills "0 params detected" and the 3-way AST duplication.**

3. **`tokenize`-based code stripping.** Replace the `split('#')` mini-parser with stdlib `tokenize.tokenize` (exact). The regex code-review then operates on real tokens, not corrupted text. Kills hard-gate false positives/negatives.

4. **Phase detection by code-scan, shared.** v16.1 already built a code-scanning predicate for `_check_phase_blocked`. Route the ROADMAP alignment check through the *same* predicate instead of its own text-keyword `_is_training_task`. One source of truth for "is this a training task."

5. **Lesson retrieval by trigger-tokens, not prose.** Store the concrete grep/regex that fired (e.g. `torch.cat`, the file path) as a `trigger_keywords` column; match against that. Makes the SQLite KB actually retrievable.

---

### Fix for D2 (detection without enforcement) — **gates return a `GateDecision`, not a log line**

> *A gate that can't change `action` is documentation, not a gate. Make the decision typed and route every detector through one pipeline.*

1. **Unify all 6 gates into one `GatePipeline`** with a single ordered decision type:
   ```python
   class GateDecision:
       verdict: Literal["allow", "modify_task", "redirect_action", "pause_human"]
       new_action: Optional[str]      # for redirect_action
       task_patch: Optional[str]      # for modify_task
       reason: str
       signature: str                 # stable key for counter bookkeeping
   ```
   Every detector (constraint engine, roadmap, phase gate, code review, no-progress) returns a `GateDecision`. The pipeline applies them in defined order and keeps **one** `BlockedState` keyed by `signature`. No more 6 independent counters stacking.

2. **Enforcement is monotonic.** Per-signature escalation: block → modify → redirect → pause_human, **never resetting to block on the same signature within a session**. This kills the audit death-loop (D3 below too), the hard-gate streak-cap-2 dead-end, and the direction-breaker oscillation in one design.

3. **Make the "advisory" features honest or real.** Either:
   - give Idea Guardian / Direction Circuit Breaker / Data Scarcity the same `GateDecision` power (they can `redirect_action`), **or**
   - relabel them `*_prompt` and delete the "circuit breaker / hard wall" language from docs.

4. **ROADMAP deviation #1 must `redirect_action`** (to `paper_research`) or loop the corrected result back through the leader *before* dispatch — not append a footnote to a task that then executes. Delete the dead `_roadmap_alignment_warning` key.

5. **Constraint Engine FORBIDDEN**: dead-end rules with `count >= N` set `priority="forbidden"`; `has_forbidden_violation` reads the `StrategyRule.priority` field directly (not a string-match of rendered text). Then the engine is an enforcer, not a logger.

6. **VERIFY gets teeth**: a VERIFY `critical` produces a `GateDecision(redirect_action)` consumed next cycle (or a typed directive on `SessionState`, see D3), instead of decorating the REFLECT prompt.

---

### Fix for D3 (scattered state) — **one `SessionState` object, atomically persisted**

> *One place for cross-cycle truth. Counters that mean what they say.*

1. **`SessionState` dataclass** (persisted atomically to one store, e.g. SQLite or a single `state.json` with temp-then-rename) holding: `CycleContext`, all stagnation/blocked counters, `best_metric`, architecture signature, phase status, and **`next_cycle_directive` (typed)**. Replace the `DIRECTIVE.md` covert channel + 5 JSON files + 31 `self._*` fields.

2. **Counters reset only on genuine state change.** Direction-stagnation resets when the *signature* changes (validated), not when the breaker fires. Architecture counter persists across restart. Audit escalation level is monotonic per signature.

3. **Honest memory compaction.** "Summarize" either (a) becomes real summarization by aggregating the SQLite `metrics_json`/`dead_end` fields the DB already holds, or (b) is renamed `_truncate_old_entries` and the "preserving knowledge" claim is deleted from the doc.

4. **`failure_category` in the DDL.** Add it to `CREATE TABLE memory_entries` (NOT NULL DEFAULT ''); drop the runtime `ALTER TABLE`.

5. **`context_keys` enforced.** Delete the 5 ghost keys, register the 12+ untracked, mark critical keys `required=True`, log validation at WARNING, and add a CI test (AST-walk `loop.py` for `context[...]` assignments ↔ registry). `ContextPruner` becomes phase-aware (THINK vs REFLECT budgets).

6. **`ResearchLoop` slims to ~600 lines**: orchestrator only. Extract `CodeReviewer`, `AuditEscalator`, `PhaseGate`, `GatePipeline`, `ProgressTracker`, `ContextBuilder` as collaborators (compose, don't mixin). `DomainKnowledgeMixin` either becomes a composed `DomainKnowledgeBuilder` or is folded in — the "mixin with one consumer" indirection is removed.

---

### Fix for D4 (resilience by exhaustion) — **a typed `ProviderRouter` + `ServiceHealth`, and resumable dispatch**

> *Failures are classified, not retried blindly. Work is resumable, not disposable.*

1. **`ProviderRouter` with a proper error taxonomy.** `_is_permanent_error` splits 429 into:
   - **per-minute rate-limit** (transient → short backoff + next model), vs.
   - **quota/window-exhausted** (permanent *for that provider key until window reset*). Detected from the structured error body/code (e.g. `1308`/"5小时上限"), not just HTTP status. A quota-429 marks the **entire provider** unavailable until the announced reset time (parsed from the message), skips its whole model chain, and is not re-queued. **Kills the 144-call cascade.**

2. **Resumable dispatch.** `_call_openai_compatible` carries the in-progress `api_messages` forward across same-provider retries; a mid-stream timeout returns a **best-effort partial result** (accumulated `stream_content`/`tool_calls`, flagged `partial`) instead of raising. Only empty/early failures raise. **Kills "wrote everything, timed out, redid everything."**

3. **`ServiceHealth` for external tools.** Per-backend record (last-failure, failure-count, retry-after); exponential backoff (skip 30s/60s/120s after K failures); honor HTTP 429 `Retry-After`; cache successful query→result for the session. Race healthy backends concurrently, first-good-enough wins, total wall-cap ~20s. **Kills the MCP/DDG/SemanticScholar timeout storms.**

4. **MCP lifecycle fixed.** Store reader-thread handle + `stop_event` in the session; `_destroy_session(name)` sets event + closes socket + `join(timeout=2)` before `pop`, called from eviction + `shutdown_mcp`. Drain `proc.stderr` on a daemon thread; `proc.wait()` after every kill. **Kills the thread/fd leak and stdio deadlock.**

5. **Trivial correctness fixes (1-3 lines each, but real):**
   - `read_file`: add `if file_path.is_dir(): return {"error":..., "suggestion":"use list_files"}`; broaden except to `OSError`.
   - `generate_diagnostic`: define `shape_str` (or pass `input_shape`) so it stops crashing.
   - `_run_inference`: use `start_new_session=True` + `os.killpg` on timeout; don't collect outputs after a timeout.
   - Remove `"code"` from `STRONG_MODEL_TASKS` (or make tier membership config-driven) so the code agent uses the fast model as documented.
   - Correct the glm-5.2 config/comments to reflect that it works (or pin routing intentionally and make the comment accurate).

---

## PART 3 — Priority & sequencing

The diseases are coupled; fix in this order to get compounding benefit:

**Phase 1 — Stop the bleeding (D4, highest ROI, most contained):**
`ProviderRouter` error taxonomy + quota-cooldown → resumable dispatch → `ServiceHealth` → MCP lifecycle → the 5 trivial fixes. *Expected: eliminates the 144×429 cascade, the rewrite-on-timeout, the search timeout storms. Agent stops burning quota and redoing work.*

**Phase 2 — Make truth trustworthy (D1):** `ExperimentManifest` event → runtime model probe → `tokenize` stripping → shared phase predicate → lesson trigger-tokens. *Expected: cycles stop scoring as false no-progress; analyze_model stops lying; the hard gate stops false-blocking.*

**Phase 3 — Make safety real (D2):** `GatePipeline` + `GateDecision` + monotonic escalation; roadmap/VERIFY/constraint enforcement; relabel-or-enforce advisory features. *Expected: deviations actually redirect; violations actually block; no more death-loops.*

**Phase 4 — Consolidate (D3):** `SessionState` + atomic persistence; `ResearchLoop` slim-down + collaborator extraction; honest memory compaction; enforced `context_keys`. *Expected: restart-safe; readable; one place to reason about state.*

Phases 1 and 2 alone resolve ~80% of the live-log symptoms and require no full rewrite. Phases 3-4 are the structural payoff that prevents the diseases from recurring.

---

## What was *not* a real problem (corrected / deprioritized)

- **`generate_diagnostic` NameError** is real but **latent** — the tool is registered; it crashes only when actually invoked. Real bug, MED not HIGH-in-practice.
- **God-object scale numbers** ("78 attrs / 360-line context") were **overstated ~2×** — actual is ~31 `self._*` in `__init__`, 24 `context[` injections in `_think`. The god-object diagnosis is still correct; the counts aren't.
- **`_tail_file` "not memory-efficient"** — it *is* RAM-bounded (`deque(maxlen=...)`); it's disk-I/O-heavy, not memory-heavy. LOW, and acceptable at 900s poll interval.
- **Monitor "zero-cost" claim** — holds (no LLM calls). The nvidia-smi fork-per-poll is negligible at 900s. The real monitor risks are PID-reuse and PID/PGID conflation (MED), not cost.
