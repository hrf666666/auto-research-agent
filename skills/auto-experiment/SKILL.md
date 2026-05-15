---
name: auto-experiment
description: "Launch an autonomous THINK→EXECUTE→REFLECT experiment loop on a GPU project"
argument-hint: "[--project <path>] [--gpu <id>] [--max-cycles <n>]"
---

# /auto-experiment

Launch an autonomous experiment agent that runs your deep learning experiments 24/7.

## What This Does

This skill starts a **THINK → EXECUTE → REFLECT** loop that:
1. Reads your `PROJECT_BRIEF.md` to understand the research goal
2. Analyzes previous results in `MEMORY_LOG.md`
3. Plans the next experiment (hypothesis + success criteria)
4. Implements code changes and runs a **mandatory dry-run**
5. Launches GPU training via `nohup` (tracks PID)
6. **Monitors at zero LLM cost** (only `kill -0 PID` + `tail log` + `nvidia-smi`)
7. Wakes up when training finishes to analyze results
8. Updates memory and decides: iterate, pivot, or report
9. Repeats

## Usage

```
/auto-experiment
/auto-experiment --project /path/to/my_project --gpu 0
/auto-experiment --project . --max-cycles 5
```

## Prerequisites

The project directory must contain:

### `PROJECT_BRIEF.md` (required)
A frozen reference describing your research goal. Example:

```markdown
# Goal
Train a ViT-B/16 on ImageNet to reach 78%+ top-1 accuracy.

# Codebase
- Training: train.py
- Config: configs/vit_base.yaml
- Data: /data/imagenet/

# Constraints
- GPU 0-3 available (use DDP)
- Max 90 epochs per run
- Report val accuracy after each run

# Current Best
- ResNet-50 baseline: 76.1%
```

### `config.yaml` (optional)
Override default agent settings:

```yaml
agent:
  model: "claude-sonnet-4-6"
  max_cycles: -1          # -1 = unlimited
  max_steps_per_cycle: 3  # max sub-agent dispatches per cycle
  cooldown_interval: 300  # 5 min smart polling

memory:
  brief_max_chars: 3000
  log_max_chars: 2000

monitor:
  poll_interval: 900      # check every 15 min during training
  zero_llm: true

experiment:
  mandatory_dry_run: true
```

## Workflow Details

### Phase 0: DATASET UNDERSTANDING (first cycle only)
- **Automatically triggered** by the pipeline on cycle 1 (or when DATASET_MANIFEST.json is missing)
- Dispatches code agent to scan `data/` directory thoroughly
- Validates GT formats, scene counts, split assignments for every dataset
- Produces `workspace/DATASET_MANIFEST.json` as single source of truth
- Reads skill instructions from `skills/dataset-understanding/SKILL.md`
- **This must complete before any THINK phase on first run**

### Phase 1: THINK
- Read `PROJECT_BRIEF.md` (frozen, max 3000 chars)
- Read `workspace/DATASET_MANIFEST.json` if it exists (dataset ground truth)
- Read `MEMORY_LOG.md` — **必须解析以下四个章节**：
  1. `## Key Results` — 当前最佳结果
  2. `## Dead Ends` — **已知的失败方法（最高优先级）**，绝对不能重复
  3. `## Active Problems` — 当前未解决的技术问题
  4. `## Recent Decisions` — 推理过程
- Check for `HUMAN_DIRECTIVE.md` (highest priority, auto-archived after reading)
- Analyze: 当前最佳结果是什么？Dead Ends 排除了哪些方向？Active Problems 中哪些需要优先解决？下一个实验应该针对哪个 Active Problem？
- **如果新方案与 Dead Ends 中的方法本质相同，必须换方向或给出充分理由**
- Output: experiment plan with hypothesis, success criteria, and which Active Problem it targets

### Phase 2: EXECUTE
- Dispatch to Code Agent (5 tools: `run_shell`, `launch_experiment`, `write_file`, `read_file`, `list_files`)
- Code Agent implements changes
- **Mandatory dry-run** (2-step verify, abort if fails)
- Launch training via `nohup`, capture PID
- Enter zero-cost monitoring loop:
  - `kill -0 $PID` — is process alive?
  - `nvidia-smi` — GPU utilization
  - `tail -50 logfile` — latest training output
  - **Zero LLM API calls during this phase**

### Phase 3: REFLECT
- Parse training logs for metrics (loss, accuracy, FGD, FID, etc.)
- Compare against previous best
- **如果实验失败或部分失败 → 必须更新 MEMORY_LOG.md**：
  - 实验结果 → `## Key Results` 表
  - 失败原因 → `## Dead Ends` 表（避免重复）
  - 未解决的问题 → `## Active Problems` 表
  - 推理过程 → `## Recent Decisions` 表
- **调用 `code-cleanup` 归档本次实验的 raw logs**
- **调用 `/experiment-auditor` 执行强制审计**（防止走捷径/幻觉完成任务）：
  - Check 1: THINK 阶段声明的计划是否在代码中实际实现
  - Check 2: 是否存在绕过 unified_lf_dataset.py 的孤儿脚本
  - Check 3: data/ 下所有有深度GT的数据集是否都已注册
  - Check 4: Lambertian/Non-Lambertian 分类是否正确
  - Check 5: 报告的指标是否与训练日志一致
  - Check 6: MEMORY_LOG.md 是否存在矛盾/过时条目
- **审计未通过时，必须先修复问题再开始下一轮实验**
- Decide: try another config / pivot direction / generate report
- **关键原则**：每个失败的实验都要提炼出"为什么失败"，不能只记录指标

### Human Override (anytime)
```bash
# Drop a directive file — agent reads it next cycle with highest priority
echo "Try learning rate 1e-5 with cosine schedule" > workspace/HUMAN_DIRECTIVE.md
```

## Memory System

Two-Tier, constant size (~5K chars / ~1500 tokens), no matter how long the agent runs:

| Tier | File | Content | Cap |
|------|------|---------|-----|
| 1 | `PROJECT_BRIEF.md` | Frozen project reference | 3,000 chars |
| 2 | `MEMORY_LOG.md` | Key Results + Dead Ends + Active Problems + Recent Decisions | 2,000 chars |

**Auto-compaction rules:**
- Key Results: oldest rows dropped when table > 3,000 chars
- Dead Ends: rows **只增不减**（永远保留，避免重复踩坑）
- Active Problems: resolved rows → 移到 Dead Ends
- Recent Decisions: only last 15 entries kept
- Total log hard-capped at 3,000 chars

## Cost

| Phase | Duration | LLM Cost |
|-------|----------|----------|
| THINK | 5-10 min | ~$0.05 |
| EXECUTE (training) | hours/days | **$0.00** |
| REFLECT | 5-10 min | ~$0.03 |
| **24h cycle total** | | **~$0.08** |

## Example Output

After a few cycles, your `workspace/MEMORY_LOG.md` will look like:

```markdown
# Memory Log

## Key Results
| Exp | Model | Config | Metric | Date | Notes |
|-----|-------|--------|--------|------|-------|
| Exp001 | ResNet-50 | lr=0.1 | acc=76.1% | 04-07 | baseline |
| Exp002 | ViT-B/16 | lr=1e-3 | acc=74.8% | 04-07 | lr太高，负优化 |
| Exp003 | ViT-B/16 | lr=3e-4 + cosine | acc=77.9% | 04-08 | new best |
| Exp004 | ViT-B/16 | lr=3e-4 + cosine + mixup | acc=78.3% | 04-08 | ✅ 达标 |

## Dead Ends
| Method | Why Failed | Lesson |
|--------|-----------|---------|
| ViT lr=1e-3 | 初始学习率过高，训练发散 | lr > 5e-4 不适合 ViT |
| ResNet-50 baseline | acc=76.1%，低于 ViT 潜力上限 | 切换 ViT 方向正确 |

## Active Problems
| Problem | Severity | Status | Notes |
|---------|----------|--------|-------|
| 跨域泛化能力不足 | P1 | 进行中 | 正在尝试 mixup |
| GPU 显存不足 batch=16 | P2 | 已降级 | batch=8 继续实验 |

## Recent Decisions
[04-07 22:15] ViT lr=1e-3 失败 → 排除高 lr 方向，降低到 3e-4
[04-08 06:00] cosine schedule 有效 → 保持，加入 mixup 进一步提升
[04-08 14:45] 目标达成，生成最终报告
```
