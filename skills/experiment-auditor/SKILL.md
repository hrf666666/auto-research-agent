---
name: experiment-auditor
description: "Mandatory execution audit for every experiment cycle — verifies that declared plans are actually implemented, no shortcuts taken, and code changes match task requirements. Prevents LLM hallucination of task completion."
---

# /experiment-auditor

**This skill is MANDATORY at the end of every REFLECT phase.** It catches cases where the LLM agent
"claims" to have completed a task but actually took a shortcut, bypassed the unified codebase, or
implemented something different from what was planned.

## When to Trigger

**Mandatory** — after every REFLECT phase, before the next THINK phase:
```
REFLECT phase completes → /experiment-auditor
```

## Core Principle

**Trust, but verify.** The LLM agent tends to:
1. **Hallucinate completion** — claim "I added UrbanLF support" but only created an inline loader
2. **Take shortcuts** — bypass `unified_lf_dataset.py` with per-experiment scripts
3. **Claim analysis** — report "I verified the results" without actually checking
4. **Overgeneralize** — apply one fix and claim all similar issues are resolved

This skill provides a deterministic checklist that catches all of the above.

## Audit Checklist

Run ALL of the following checks after every REFLECT phase. Each check must PASS or be explicitly ACKNOWLEDGED with a reason.

### Check 1: Plan vs Implementation Alignment

**What:** Verify that every task declared in the THINK phase plan was actually implemented.

**How:**
1. Extract tasks from the THINK phase output (look for "plan", "will", "going to", "task")
2. For each declared task, check the actual code changes:
   - Did the agent modify the correct file?
   - Did the modification match the described approach?
   - Or did the agent create a workaround/bypass instead?
3. **Red flags that indicate a shortcut:**
   - New standalone scripts in `scripts/` that duplicate functionality of existing modules
   - Inline data loaders in training scripts instead of using `unified_lf_dataset.py`
   - Claims of "testing" without actual test execution logs
   - Claims of "fixed" without showing the diff or verification

**Output:** For each task: `PASS` | `SHORTCUT` | `NOT_DONE` | `PARTIAL`

### Check 2: No Orphan Scripts (Inline Workarounds)

**What:** Detect scripts that bypass the unified codebase.

**How:**
1. List all `.py` files in `scripts/` directory
2. For each script, check if it contains:
   - Direct `np.load()` / `Image.open()` / `h5py.File()` for data loading
   - WITHOUT importing from `datasets.unified_lf_dataset` or `RealLightFieldDataset`
3. Any script that loads data without using the unified dataset is an **ORPHAN**
4. Check if orphan scripts are still referenced (should be migrated or deleted)

**Output:** List of orphan scripts, or `PASS` if none found

### Check 3: Dataset Registration Completeness

**What:** Verify all datasets in `data/` are handled by `unified_lf_dataset.py` and match `DATASET_MANIFEST.json`.

**How:**
1. Read `workspace/DATASET_MANIFEST.json` — this is the canonical source of truth
2. Read `datasets/unified_lf_dataset.py` — verify it handles every dataset listed in manifest
3. Cross-reference: every dataset in manifest must have a corresponding loader function
4. Verify `_collect_scenes()` covers all directories listed in manifest
5. Verify excluded datasets (HCI-Old, Urban-Real) are documented with reasons
6. **Quick smoke test**: if possible, verify `from datasets import UnifiedLFDataset` works

**Output:** Missing registrations or mismatches, or `PASS`

### Check 4: Lambertian/Non-Lambertian Classification

**What:** Verify dataset type classification is correct.

**How:**
1. Read `DATASET_SOURCES` — check `r_default` values:
   - Lambertian datasets (HCInew, Wanner_HCI): `r_default >= 0.8`
   - Non-Lambertian datasets: `r_default <= 0.3`
   - Mixed datasets (UrbanLF-Syn): `0.3 < r_default < 0.8`
2. Verify against domain knowledge:
   - HCInew = Lambertian (synthetic, diffuse-only scenes)
   - Wanner_HCI = Lambertian (standard LF benchmark)
   - HCI-Old = has specular/diffuse decomposition, r computed from ratio
   - Non-lamertian_zhenglong = Non-Lambertian (controlled reflectance)
   - UrbanLF-Syn = Mixed (urban scenes with specular/shadow)

**Output:** Misclassifications, or `PASS`

### Check 5: Training Log Integrity

**What:** Verify training actually ran and produced valid metrics.

**How:**
1. Find the latest log file referenced in REFLECT output
2. Check log contains:
   - At least 1 completed epoch with loss value
   - Validation metrics (MAE, MSE, or accuracy)
   - No unhandled exceptions or premature termination
3. Verify the metrics reported in REFLECT match the actual log values
4. **Catch:** Agent reporting "val_MAE=0.05" but log shows "val_MAE=0.50"

**Output:** Metric match/mismatch, or `PASS`

### Check 6: Memory Log Consistency

**What:** Verify MEMORY_LOG.md doesn't contain contradictions.

**How:**
1. Check `## Dead Ends` — are any of these methods being used in current code?
2. Check `## Active Problems` — are resolved problems still listed?
3. Check `## Key Results` — are the "best" metrics actually the best?
4. Check for duplicate or stale entries that waste context space

**Output:** Contradictions/stale entries, or `PASS`

## Audit Report Format

After running all checks, produce a structured report:

```markdown
# Experiment Audit — Cycle {N}

## Summary: {PASS / SHORTCUTS_FOUND / FAILURES_FOUND}

### Check 1: Plan vs Implementation
| Task | Status | Evidence |
|------|--------|----------|
| {task description} | PASS/SHORTCUT/NOT_DONE | {file modified or reason} |

### Check 2: Orphan Scripts
- {list of orphan scripts} or "None found"

### Check 3: Dataset Registration
- {missing registrations} or "All datasets registered"

### Check 4: Classification
- {misclassifications} or "All classifications correct"

### Check 5: Training Log
- {metric mismatches} or "Metrics verified"

### Check 6: Memory Consistency
- {contradictions} or "No contradictions"

## Required Actions
- [ ] {action item for each SHORTCUT or FAILURE}
```

## Integration with auto-experiment Loop

This skill should be integrated into the REFLECT phase:

```
REFLECT phase:
  1. Parse training results (existing)
  2. Update MEMORY_LOG.md (existing)
  3. >>> RUN /experiment-auditor <<<  (NEW)
  4. If auditor finds SHORTCUTS or FAILURES:
     - Do NOT proceed to next experiment
     - Fix the issues first
     - Only proceed when auditor returns all PASS
  5. Decide: iterate / pivot / report
```

## Consequences of Failed Audit

If the auditor finds issues, the agent MUST:
1. **SHORTCUT**: Refactor orphan scripts into unified codebase before next experiment
2. **NOT_DONE**: Actually implement the missing task
3. **PARTIAL**: Complete the remaining portion
4. **Metric mismatch**: Re-read logs and correct MEMORY_LOG.md

The agent should NOT start a new experiment until all audit issues are resolved.
This prevents compounding errors where each cycle builds on faulty previous work.
