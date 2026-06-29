# Data Contract — SQLite Schema & Data-Flow Truth

> **Authoritative** source of truth for every SQLite table in the system.
> The L3 contract test (`tests/test_db_read_write_contract.py`) reads this file
> to determine which tables are allowed to break the read/write-pairing rule,
> and why. If you add a table or change a table's write/read contract, update
> **this file** in the same change.

Last updated: 2026-06-29 (L1 dead_end migration + L2 orphan cleanup).

---

## How to read this

Each table has a status:

| Status | Meaning |
|---|---|
| `LIVE` | Write path and read path both active; data flows end-to-end. |
| `KNOWN-BROKEN` | Code path exists but data does not flow (a bug). Root cause documented. Tracked for separate fix. The L3 test **must not** fail on this — it's explicitly listed as an exemption. |
| `INTENTIONAL` | By design only written or only read (e.g. a write-only audit log, or a read-only reference table). Listed as an exemption with rationale. |

The L3 test asserts: every `CREATE TABLE` has at least one `INSERT` writer AND
one `SELECT` reader in `core/`, **unless** the table is marked `KNOWN-BROKEN` or
`INTENTIONAL` here. Exemptions require a root-cause note.

---

## Tables (single DB: `experiment_history.db`)

All tables live in one SQLite file: `<workspace>/experiment_history.db`. DDL is
created by `core/memory.py:_init_db` (except `experiment_facts`, created by
`core/fact_scanner.py:_ensure_table`). There is no migrations framework; DDL is
`CREATE TABLE IF NOT EXISTS` only.

### 1. `experiments` — per-cycle structured experiment record

| Field | Value |
|---|---|
| DDL | `core/memory.py` `_init_db` |
| Writer | `record_cycle_outcome` (INSERT, 1 row per cycle) |
| Readers | `get_experiment_history`, `get_summary_stats`, `get_method_domain_effect_matrix` |
| Status | `LIVE` |

**Note:** the `dead_end` column was removed (2026-06-29). dead_end tracking
migrated to `memory_entries` (see below). The `get_method_domain_effect_matrix`
`success_rate`/`dead_ends` fields are kept as baseline placeholders (1.0 / 0)
since they no longer have a data source; downstream consumers (`api.py`,
`tools.py`) only pass them through and never branch on them.

### 2. `memory_entries` — append-only memory stream

| Field | Value |
|---|---|
| DDL | `core/memory.py` `_init_db` (+ runtime `ALTER TABLE ADD COLUMN failure_category`) |
| Writer | `_record_memory_entry` (milestone/decision/active_problem/major_event), `log_dead_end` (dead_end) |
| Readers | `get_dead_ends_full`, `get_dead_ends_by_category`, `get_method_inadequacy_count`, `get_summary_stats`, B9 gate `check_dead_end_signature` |
| Status | `LIVE` |

**Source of truth for dead_end.** dead_end entries are rows with
`entry_type='dead_end'`; `content` carries a tagged string
(`[category] [timestamp] <text>`). `failure_category` is added by runtime ALTER
(not in DDL) — see `core/memory.py` `log_dead_end`. `get_dead_ends_full()`
returns `list[str]` of the `content` column.

### 3. `causal_chain` — causal attribution history

| Field | Value |
|---|---|
| DDL | `core/memory.py` `_init_db` |
| Writer | `record_causal_chain_entry` (called from REFLECT when `causal_link` present) |
| Readers | `get_causal_history` (used in THINK context) |
| Status | `LIVE` |

### 4. `code_review_lessons` — reusable code lessons

| Field | Value |
|---|---|
| DDL | `core/memory.py` `_init_db` |
| Writer | `record_code_review_lesson` (called from REFLECT when `lesson` present) |
| Readers | `get_code_review_lessons`, `search_relevant_lessons` (exposed via memory tool) |
| Status | `LIVE` |

### 5. `pareto_matrix` — method × domain MAE frontier

| Field | Value |
|---|---|
| DDL | `core/memory.py` `_init_db` |
| Writer | `record_pareto_entry` (loop.py, gated on `domain_metrics` non-empty) |
| Readers | `get_pareto_matrix` → `get_pareto_frontier` (consumed by `constraint_engine`) |
| Status | `KNOWN-BROKEN` |

**Root cause:** `training_log_parser.py:100` forces metric keys to lowercase
(`.lower()`), so `final_metrics` keys are `mae_lambertian` (lowercase). But
`loop.py:1251` filters with `key.startswith("MAE_")` (uppercase), which never
matches → `domain_metrics` is always empty → `record_pareto_entry` never fires.
The Pareto frontier rule in `constraint_engine` is silently never generated.
Consumer (`constraint_engine.py:166`) tolerates the empty matrix (skips rule
generation, no crash). **Fix:** normalize the case (uppercase `MAE_` in the
filter, or lowercase `domain_keys`) — out of scope for the current cleanup.

### 6. `experiment_value` — hypothesis calibration

| Field | Value |
|---|---|
| DDL | `core/memory.py` `_init_db` |
| Writer | `record_experiment_value` (loop.py, gated on `hypothesis` non-empty AND `current_metric is not None`) |
| Readers | `get_experiment_calibration`, `get_low_value_experiments` (consumed by `constraint_engine`, `domain_knowledge`) |
| Status | `KNOWN-BROKEN` |

**Root cause:** (a) `hypothesis` is a soft dependency — the THINK schema only
validates the `action` key, so `think_result.get("hypothesis","")` is empty
whenever the LLM omits it, skipping the write. (b) Secondary bug:
`constraint_engine.py:95` reads `calibration.get("accuracy_rate", 0.5)` but
`get_experiment_calibration` returns the key as `accuracy` — a key-name
mismatch. Consumers tolerate the empty result (`total < 3` → skip rule, no
crash). **Fix:** strengthen the THINK schema/parse to require `hypothesis` for
experiment actions, and fix the `accuracy_rate`→`accuracy` key name — out of
scope for the current cleanup.

### 7. `experiment_facts` — scanned-from-disk experiment facts

| Field | Value |
|---|---|
| DDL | `core/fact_scanner.py` `_ensure_table` |
| Writer | `scan_single` (INSERT OR REPLACE) via `scan_all` |
| Readers | `get_facts_for_output_dir`, `get_all_facts`, `run_all_gates` (control gate, metrics_json) |
| Status | `KNOWN-BROKEN` |

**Root cause:** `scan_all` is gated on `<workspace>/outputs/` existing. In
environments without an `outputs/` dir, `scan_all` early-returns and
`experiment_facts` is never populated (the table may not even be created,
depending on db_path resolution). When the table is missing, consumers
catch `sqlite3.Error` and degrade silently. **Note:** this is environmental,
not a logic bug — it activates once `outputs/` exists. Tracked separately to
verify `MemoryManager.db_path` resolution matches the on-disk DB.

### 8. `roadmap_history` — REMOVED (2026-06-29)

Was a v15 Research Roadmap history table. DDL existed but had **zero** writers
and **zero** readers (pure zombie). Deleted in L2 cleanup. Listed here for
archival; the L3 test asserts it does **not** reappear.

---

## Column-level invariants (enforced by L3 test)

Beyond table-level read/write pairing, the L3 test checks column consistency:

- Every column referenced in a `SELECT col FROM experiments` (or any table)
  must exist in that table's DDL — otherwise the query raises
  `sqlite3.OperationalError` at runtime.
- Every column in an `INSERT INTO table (cols)` must exist in DDL.
- The migration of `dead_end` out of `experiments` is the canonical example:
  removing the column required removing all `SELECT ... dead_end FROM
  experiments` sites in lockstep. The L3 column-level test prevents such a
  drift from ever silently reappearing.

## Non-SQL "fields" (exempt from column checks)

Some dict keys returned to callers are computed in Python, not SQL columns.
These are exempt from column-level checks (they never appear in SQL):

- `method_effect_matrix` `success_rate`, `dead_ends` (computed in
  `get_method_domain_effect_matrix`, now baseline 1.0 / 0).

---

## Change log

- **2026-06-29 (L1+L2 cleanup):**
  - Removed `dead_end` column from `experiments`; dead_end source of truth is
    now `memory_entries`. B9 gate migrated. REFLECT schema (`agents/leader.md`)
    now prompts for `dead_end`/`active_problem`.
  - Removed `roadmap_history` table (zombie).
  - Removed `get_recent_failures` (0 callers).
  - Marked `pareto_matrix`, `experiment_value` as `KNOWN-BROKEN` with root
    causes (fix deferred).
