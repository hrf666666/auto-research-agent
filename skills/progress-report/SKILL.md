---
name: progress-report
description: "Generate structured research progress reports"
---

# /progress-report

Generate a structured progress report for the current research project.

## Behavior

1. Read the project's MEMORY_LOG.md for milestones and decisions
2. Check recent experiment logs in workspace
3. Compile results into a structured report

## Output Format

```markdown
# Progress Report — YYYY-MM-DD

## Current Status
- Best result: [metric]
- Total experiments: [N]
- Current direction: [description]

## Experiment History
| # | Config | Metric | Date | Notes |
|---|--------|--------|------|-------|

## Dead Ends (What NOT to Retry)
| Method | Why Failed | Lesson |
|--------|-----------|--------|
| ... | ... | ... |

## Active Problems
| Problem | Severity | Status | Notes |
|---------|----------|--------|-------|

## Key Insights
- What we learned
- What works / doesn't work

## Next Steps
1. Planned experiments (must address Active Problems)
2. Open questions

## Blockers
- Any issues or risks
```
