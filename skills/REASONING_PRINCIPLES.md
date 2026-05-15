# Reasoning Principles for Autonomous Research Agents

> Behavioral guidelines to reduce common LLM reasoning mistakes in the THINK→EXECUTE→REFLECT cycle.
> **Tradeoff:** These guidelines bias toward caution and clarity over speed. For trivial decisions, use judgment.

---

## 1. Think Before Acting

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before planning or implementing an experiment:
- State your assumptions explicitly. If uncertain, investigate first.
- If multiple interpretations exist, present them — don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Investigate.

**In the THINK phase:** Before deciding on the next experiment, explicitly write:
1. What assumption am I making?
2. What evidence supports/contradicts it?
3. What's the simplest way to test it?

---

## 2. Simplicity First

**Minimum changes that solve the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior researcher say this is overcomplicated?" If yes, simplify.

**In experiment design:** Prefer changing ONE variable at a time. Don't change architecture, loss function, and data augmentation simultaneously — you won't know what helped.

---

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it — don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

**The test:** Every changed line should trace directly to the experiment's hypothesis.

---

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write a test that runs 2 steps and checks output shape"
- "Fix the bug" → "Reproduce the error in a dry-run, then make it pass"
- "Improve accuracy" → "Target: metric X < threshold Y; verify in logs after training"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

**Strong success criteria** let you loop independently. **Weak criteria** ("make it work") require constant clarification.

**In the REFLECT phase:** Always check — did we actually meet the success criteria from THINK? If not, the experiment is inconclusive at best.

---

## How to Apply These Principles

| Phase | Key Principle | Action |
|-------|--------------|--------|
| THINK | Think Before Acting | State assumptions, present alternatives, pick simplest test |
| THINK | Goal-Driven | Define concrete success criteria before executing |
| EXECUTE | Simplicity First | Change ONE variable, minimal code, no over-engineering |
| EXECUTE | Surgical Changes | Only touch files relevant to the hypothesis |
| REFLECT | Goal-Driven | Check if success criteria were met, be honest about failures |
| REFLECT | Think Before Acting | Don't rationalize — if it didn't work, say why honestly |

---

**These guidelines are working if:** fewer unnecessary code changes in diffs, fewer rewrites due to overcomplication, clearer hypotheses in THINK outputs, and honest REFLECTION that admits failures rather than spinning them.
