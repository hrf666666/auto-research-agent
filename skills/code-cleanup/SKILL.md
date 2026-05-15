---
name: code-cleanup
description: "Aggressively removes obsolete code to keep the codebase lean and navigable"
argument-hint: "[--project <path>] [--dry-run] [--confirm]"
---

# /code-cleanup

**核心理念**：删除是常态，归档是例外。如果一个文件没人用、没checkpoint关联、不是文档，就删。

## 核心原则（按优先级）

| 优先级 | 原则 | 说明 |
|--------|------|------|
| 1 | 无人引用 → **删除** | 没被 import、没被 shell 调用、没被文档引用 |
| 2 | 功能被覆盖 → **删除** | 有更新的同类脚本，旧的就是垃圾 |
| 3 | 孤立 checkpoint → **删除** | 对应网络已不存在，无法加载，留着无意义 |
| 4 | 纯文档/.bak → **删除** | git history 能恢复，不需要本地备份 |
| 5 | 根目录散落脚本 → **移动或删除** | 保留在 scripts/，不在根目录 |

---

## 清理对象分类与动作

### A. 孤立脚本（Orphaned Scripts）→ **删除**

**定义**：没有任何引用，也没有 checkpoint 关联。

**识别**：
```bash
# 扫描根目录所有 .py
for f in *.py; do
  imported=$(grep -r "import $f" --include="*.py" . | wc -l)
  called=$(grep -r "$f" --include="*.sh" . | wc -l)
  docs=$(grep -r "$f" --include="*.md" . | wc -l)
  echo "$f: import=$imported call=$called doc=$docs"
done
```

**动作**：
- `import=0 && call=0 && doc=0` → **删除**（不做任何保留）
- `import=0 && call=0 && doc>0` → 确认文档是否还有价值，有价值则移动到 scripts/，否则删除

### A2. scripts/ 目录清理 → **删除不符合规范的脚本**

**命名规范（强制性）**：
- 训练脚本: `train_{model_short_name}.py`（如 `train_v12.py`, `train_dcbn.py`）
- 评估脚本: `eval_{target}.py`（如 `eval_per_domain.py`）
- 诊断脚本: `diagnose_{target}.py`（如 `diagnose_gt_stats.py`）
- 一次性诊断: `_` 前缀（如 `_check_shapes.py`），**用后立即删除**
- 通用工具: `{verb}_{noun}.py`（如 `dry_run.py`）

**必须删除的 scripts/**：
1. **增量版本脚本**（`*_v2.py`, `*_v3.py`, `*_fix.py`, `*_new.py`）— 保留最新版，删除旧版
2. **一次性诊断脚本**（`test_*.py`, `diagnose_forward.py`, `audit_*.py`）— 已执行完毕的
3. **功能被覆盖的脚本**（如 `train_balanced.py` 被 `train_domain_balanced.py` 覆盖，后者又被 `train_v11.py` 覆盖）
4. **引用旧模型的脚本**（对应模型类已不存在或已被新模型替代）
5. **旧 baseline 脚本**（如使用 `AngularAwareDepthNet` 而非最新的 `AngularAwareDepthModelV12`）

**识别命令**：
```bash
# 检查 scripts/ 中哪些模型类已过时
for f in scripts/train_*.py; do
  model=$(grep -oP 'from models\.\w+ import \K\w+' "$f")
  echo "$f → $model"
done
# 检查哪些脚本从未被任何文档或shell引用
for f in scripts/*.py; do
  refs=$(grep -rl "$(basename $f)" --include="*.md" --include="*.sh" . 2>/dev/null | wc -l)
  echo "$f: refs=$refs"
done
```

**保留规则**：
- 当前主力训练脚本（使用最新模型）
- 最近的实验变体（如上一个 cycle 的实验脚本，可能需要复现）
- 通用工具脚本（`dry_run.py`, `eval_*.py`, `inference_*.py`）
- 每种模型架构最多保留 **1个** 训练脚本（合并/覆盖旧版本）

### B. 功能重复脚本 → **删除旧版**

**识别**：
```bash
# 检查多个 train_*.py 是否有相同功能
# 保留：最新 / 参与过成功实验 / 被 run_*.sh 调用
# 删除：旧版本 / 未参与实验的变体
```

**判断标准**（互斥，优先级从高到低）：
1. 被 `run_*.sh` 调用 → **保留**
2. 产生过 checkpoint 且 val_MAE < baseline → **归档**
3. 既无调用也无 checkpoint → **删除**

### C. .bak / _old / _backup 文件 → **删除**

**理由**：git history 能恢复每个版本，不需要本地备份。

**动作**：
- 全部 **删除**，不做例外
- 删除前 git tag 一次（防止极端情况）：`git tag cleanup-$(date +%Y%m%d)-$filename $filename`

### D. 孤立 Checkpoint → **删除**

**定义**：对应的训练脚本已不存在，或 checkpoint 的 architecture_version 与当前代码不兼容，或模型类在代码库中找不到。

**判断逻辑**：
```bash
# 对每个 .pt checkpoint
for ckpt in checkpoints/*.pt; do
  # 1. 能否加载 metadata？
  metadata=$(torch.load($ckpt, map_location="cpu")["metadata"])
  model_class=${metadata["model_class"]}

  # 2. 该 model_class 在代码库中是否存在？
  if ! grep -r "class $model_class" --include="*.py" . > /dev/null; then
    echo "$ckpt → ORPHANED (class not found)"
  fi

  # 3. 该 model_class 对应的 .py 文件是否还存在？
  #    如果网络文件被删除，checkpoint 没有任何意义
done
```

**动作**：
- 对应的**训练脚本已删除** → **删除 checkpoint**（无法加载）
- architecture_version 不兼容 → **删除 checkpoint**（无法加载）
- model_class 在代码库中找不到 → **删除 checkpoint**（无法加载）

**禁止**：
- ❌ 孤立的 checkpoint 归档到 `archive/`
- ❌ checkpoint 有科研价值所以保留

**唯一例外**（需要用户明确授权）：外部发布的 benchmark checkpoint（如 MPI-INF/6d-lightfield benchmark），此类 checkpoint 与项目代码无关，可保留。

### E. 根目录散落脚本 → **删除**（不是移动，不是归档）

**允许留在根目录的文件**（白名单）：
```
PROJECT_BRIEF.md
config.yaml
requirements.txt
README.md
.gitignore
dry_run_check.py
```

**其余所有根目录 .py → 直接删除**，不要移动到 `scripts/` 或 `archive/`。

**理由**：
- 废弃脚本无人引用，移动到别处只会占空间
- 有用的逻辑已经总结在 `archive/experiments/*/SUMMARY.md` 中
- 如需恢复，git history 可以找回
- `archive/root_scripts/` 中堆积的废弃脚本也应一并删除

**禁止**：
- ❌ 把废弃脚本移动到 `archive/root_scripts/` — 这不是归档，是垃圾堆积
- ❌ 保留"以防万一"的脚本 — git 是备份

### E2. archive/root_scripts/ 清理 → **删除全部**

`archive/root_scripts/` 目录是之前错误归档策略的产物。里面的脚本：
- 没有任何引用
- 对应的实验结论已在 `archive/experiments/*/SUMMARY.md` 中
- 占用空间且增加项目复杂度

**动作**：
```bash
# 直接删除整个目录
rm -rf archive/root_scripts/
```

### F. 废弃实验记录（Raw Logs + 孤立实验结果）→ **总结后删除**

**问题**：logs/ 和 outputs/ 中大量 raw log 和图片没人整理，时间长了变成噪音。

**闭环流程**：

```
raw logs → 提取关键信息 → 写入归档报告 → 删除 raw logs → 更新 MEMORY_LOG.md
```

**Step 1: 扫描废弃实验记录**
```bash
# 扫描 logs/ outputs/ 中未被整理的实验
find ./logs -name "*.log" | sort
find ./outputs -name "*.json" -o -name "*.png" | sort

# 检查哪些已有归档记录
grep -r "archive/experiments/" workspace/MEMORY_LOG.md
```

**Step 2: 判断是否已归档**
```
logs/experiment_20260420.log
└─ MEMORY_LOG.md 中有对应记录?
    └─ 是 → SKIP（已归档）
    └─ 否 → 候选整理
```

**Step 3: 写入归档报告**

每个实验结果归档到 `archive/experiments/{experiment_id}/`：

```
archive/experiments/exp_angular_aware_v1_20260420/
├── SUMMARY.md       ← 必填，总结报告
├── metrics.json     ← 原始指标数据
├── depth_viz/       ← 可选，关键可视化
└── raw_log.txt      ← 可选保留一行 tail
```

**SUMMARY.md 格式**：
```markdown
# Experiment Archive — {experiment_id}

## 基本信息
| 字段 | 值 |
|------|-----|
| 实验时间 | YYYY-MM-DD HH:MM |
| 训练脚本 | train_angular_aware.py |
| 模型类 | AngularAwareDepth |
| 架构版本 | v0.9 |
| Checkpoint | checkpoints/angular_v1_ep20.pt |

## 模型结构
```python
# 关键架构描述
AngularAwareDepth:
  - Branch 0: AngularStatistics (12 channels)
  - Branch 1: DepthCore (hidden=32)
  - Branch 2: SpecularPeakTracker (未实现)
  - Branch 3: MaterialDecoder (未实现)
  - 融合方式: weighted_sum
```

## 测试结果
| 指标 | 值 | 说明 |
|------|-----|------|
| val_MAE | 0.443 | 远超目标 0.1 |
| r_corr | -0.39 | 负相关，方向错误 |
| conf | 0.52 | 置信度偏低 |

## 发现的问题
1. **Bug 1（致命）**：r 相关性为负 — 模型学习方向与物理定义相反
2. **Bug 2（致命）**：验证集包含 constant-depth 场景（pyramids），指标不可信
3. **Bug 3**：无学习率调度，训练后半段不稳定
4. **Bug 4**：无早停机制，epoch 1 后持续过拟合

## 根因分析
- r 为负的根因：angular_physical_loss 中 r_gt_proxy 符号与数据集生成公式不一致
- val_MAE 高的根因：Bug 1 + Bug 2 叠加

## 后续修改思路
1. 修复 r_gt_proxy 符号定义（Phase 1 最高优先级）
2. 排除 constant-depth 场景重做验证
3. 添加 CosineAnnealingLR + early stopping
4. 扩大训练数据（加入 HCInew additional）

## 结论
❌ **不可用** — 需修复 Bug 1-4 后重新训练
```

**Step 4: 删除 Raw Logs**
```bash
# 已归档的 raw logs → 删除
rm logs/experiment_20260420.log
rm outputs/depth_viz/exp_20260420_*/  # 目录删除

# 保留必要的 checkpoint（如果模型还在）
cp checkpoints/angular_v1_ep20.pt archive/experiments/exp_angular_aware_v1_20260420/
```

**Step 5: 更新 MEMORY_LOG.md**
```
## Experiment History
| ID | Date | Model | val_MAE | Status | Archive |
|----|------|-------|---------|--------|---------|
| exp_angular_aware_v1 | 2026-04-20 | AngularAwareDepth v0.9 | 0.443 | ❌ 失败 | archive/experiments/exp_angular_aware_v1_20260420/ |
```

**判断标准**：
- 有对应 MEMORY_LOG 记录且有归档路径 → **跳过**
- 无记录且无价值（重复实验/调参试跑） → **直接删除**
- 无记录但有参考价值 → **归档后删除 raw**

---

## 操作流程

### Step 1: 扫描（Dry-run，无破坏性）

```bash
# 根目录脚本
find . -maxdepth 1 -name "*.py" | sort

# .bak 文件
find . -name "*.bak" -o -name "*_old*" -o -name "*_backup*" -o -name "*_deprecated*"

# 孤立 checkpoint
find ./checkpoints -name "*.pt"
```

### Step 2: 建立引用矩阵

```bash
# 对每个根目录 .py，检查引用情况
for f in *.py; do
  imported=$(grep -rl "import.*${f%.py}" --include="*.py" . 2>/dev/null | wc -l)
  called=$(grep -rl "$f" --include="*.sh" . 2>/dev/null | wc -l)
  docs=$(grep -rl "$f" --include="*.md" . 2>/dev/null | wc -l)
  ckpt=$(ls checkpoints/*${f%.py}*.pt 2>/dev/null | wc -l)
  echo "$f | $imported | $called | $docs | $ckpt"
done
```

输出格式：
```
File                    | Imported | Called | In Docs | Checkpoint | Action
train_mini.py           | 0        | 1      | 1       | 0          | KEEP
train_physics_05.py     | 0        | 0      | 0       | 0          | DELETE
train_4d.py.bak         | 0        | 0      | 0       | 0          | DELETE
dual_mask_ckpt.pt       | -        | -      | -       | 1          | ARCHIVE
```

### Step 3: 输出清理报告

```markdown
# Code Cleanup Report — YYYY-MM-DD

## Summary
| Action   | Count |
|----------|-------|
| DELETE   | N     |
| ARCHIVE  | M     |
| MOVE     | K     |
| KEEP     | J     |

## DELETE (Files + Orphaned Checkpoints)
| File | Reason |
|------|--------|
| train_physics_05.py | 孤立，无引用无checkpoint |
| train_4d.py.bak | .bak，git可恢复 |
| eval_mini_run.py | 功能被 scripts/eval_mini.py 覆盖 |
| dual_mask_ckpt_ep5.pt | 对应网络类已不存在，无法加载 |
| ... | |

## ARCHIVE (Experiment Results)
| Experiment ID | Status | Archive Path |
|---------------|--------|-------------|
| exp_angular_aware_v1 | ❌ 失败 | archive/experiments/exp_angular_aware_v1_20260420/ |
| ... | | |

## MOVE (Relocate to scripts/)
| From | To |
|------|-----|
| quick_diag.py | scripts/diag_quick.py |
| final_eval.py | scripts/eval_final.py |
| ... | |

## KEEP (Files)
| File | Reason |
|------|--------|
| train_r_supervised_consolidated.py | 被 run_train.sh 调用 |
| ... | |

## Execution Plan
1. [ ] 扫描 logs/ 和 outputs/ 中的废弃实验记录
2. [ ] 对有参考价值的实验写入 SUMMARY.md 到 archive/experiments/
3. [ ] 删除 raw logs 和孤立实验结果
4. [ ] 更新 MEMORY_LOG.md 中的 Experiment History
5. [ ] DELETE N 个孤立文件
6. [ ] DELETE 孤立 checkpoint
7. [ ] MOVE M 个文件到 scripts/

Dry-run preview:
```bash
# EXPERIMENT CLEANUP:
mkdir -p archive/experiments/exp_angular_aware_v1_20260420/
# ... write SUMMARY.md ...
rm logs/experiment_20260420.log

# CODE CLEANUP:
rm train_physics_05.py
rm train_4d.py.bak
rm checkpoints/dual_mask_ckpt_ep5.pt
mv quick_diag.py scripts/
...
```
```

### Step 4: 执行（必须带 `--confirm`）

```
/code-cleanup --project ~/depth_estimation_unify_theory        # Dry-run
/code-cleanup --project ~/depth_estimation_unify_theory --confirm  # 执行
```

**执行前必须**：
1. 确认 DELETE 列表中没有仍在使用的文件
2. 确认 ARCHIVE 列表中的 checkpoint 确实不可用
3. 对每个要删除的 .bak 文件，先打 git tag

---

## 与其他 Skill 的关系

### 与 code-review 的集成
```
code-review 发现架构变更 → code-cleanup 清理旧架构脚本
```

### 与 idea-validation 的集成
```
idea-validation 通过 → code-cleanup 先清旧版再实现新功能
```

### 与 auto-experiment 的集成
```
REFLECT 阶段 → 发现根目录脚本 > 15 个 → 触发 code-cleanup
```

---

## 触发条件

1. 根目录 `.py` 文件超过 15 个
2. 大版本切换（v1 → v2）
3. logs/ 或 outputs/ 中有超过 10 个未归档的实验记录
4. 实验失败或遇到重大 Bug 后（**必须立即触发**，防止 raw logs 堆积）
5. 用户明确要求：`/code-cleanup --project <path> --confirm`

**实验失败后的强制归档流程**：
```
experiment 失败
  → code-cleanup 立即介入
  → 写入 SUMMARY.md（含模型/指标/问题/思路）
  → 删除 raw logs
  → 更新 MEMORY_LOG.md
  → agent 继续修复
```
