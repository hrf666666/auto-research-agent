# AutoResearcher 架构改革：状态驱动结构化方案

## 核心原则

**不对抗 LLM 的自然倾向，而是改变问题的结构，让 LLM 的自然倾向为你工作。**

当前系统的根本问题：所有模块都是"开放式提问 + 事后纠正"模式。
LLM 被问"下一步做什么？"，自由发挥后由各种 gate 兜底。
这导致：(1) 上下文膨胀 (2) gate 防不胜防 (3) token 浪费在循环阻止上

改为"状态驱动结构化提问"：**让 LLM 看到紧缩的当前状态和精确的差距，
问题本身限定答案空间，LLM 自然给出正确答案。**

---

## 模块级分析与改造方案

### 模块 1: THINK 阶段

**位置**: `loop.py:751-1161`

**当前问题**:
- context 注入 20+ 个键（brief、memory_log、session_stats、domain_knowledge、dataset_manifest...）
- LLM 收到海量文本，关键约束被淹没
- `_format_leader_input()` 将所有 context 平铺注入 user message
- 输出是自由 JSON，无结构化验证

**改造: Phase-Gated Structured THINK**

```
当前 (开放式):
  context = {brief, memory_log, session_stats, domain_knowledge, ...20+ keys}
  prompt = "分析研究现状，决定下一步行动"
  → LLM 自由发挥 → 可能偏离 → gate 兜底

改造后 (状态驱动):
  PHASE_FOCUS = {
      "phase": "phase_1",
      "goal": "AUC ≥ 0.90, accuracy ≥ 0.85",
      "current": {"auc": 0.557, "accuracy": 0.7648},
      "gap": {"auc": -0.343, "accuracy": -0.0852},
      "focus_methods": ["FFT特征工程", "频域分析", "特征选择", "统计分析"],
      "blocked_methods": ["train", "model.*archit", "fine.?tun", "neural.*net"]
  }
  prompt = "Phase 1 目标是 X，当前差距 Y。聚焦以下方法缩小差距。"
  → LLM 自然给出聚焦方案 → 几乎不偏离
```

**实现方式**:
1. 新文件 `PHASE_STATUS.json`：紧凑的阶段状态（~10行 JSON）
2. `_think()` 开头从 PHASE_STATUS 构建 `phase_focus` 注入 context
3. 将 20+ 个 context 键精简为核心 5 个：`phase_focus`, `recent_memory`(最近3条), `brief_summary`(精简版), `verify_feedback`, `directive`
4. 保留硬门作为安全网（仅 blocked_patterns 检查，1道门足矣）

**改动量**: loop.py ~50行修改

---

### 模块 2: EXECUTE 阶段

**位置**: `loop.py:1163-1199`

**当前问题**:
- `task_description` 是纯文本，LLM 可自由解读
- Code agent 有 `run_shell`，可执行任意命令
- max_turns=40，最多 40 次工具调用
- Worker 无状态，看不到阶段约束

**改造: Scoped Task Template**

```
当前:
  task = "实现 Angular FFT 分类器并验证在多个材料上的效果"
  → Code agent 自由发挥，可能：建模型、写训练循环、调超参...

改造后:
  task = f"""
  ## 执行范围
  阶段: phase_1 (Angular FFT Material Classification)
  允许操作: 数据分析, 特征提取, 统计验证, 可视化
  禁止操作: model.train(), optimizer.step(), nn.Module构建
  
  ## 具体任务
  {original_task}
  
  ## 约束
  - 如果你的代码包含 nn.Module 或 .train() 或 optimizer，立即停止
  - 只允许 numpy/scipy/sklearn 操作
  """
```

**实现方式**:
1. `PHASE_STATUS.json` 的 `allowed`/`blocked` 直接注入 task 前缀
2. 不改 Worker agent 本身，只改 task 描述
3. Code agent 的 system prompt 已经要求"遵循用户指令"，自然遵守

**改动量**: loop.py ~20行

---

### 模块 3: REFLECT 阶段

**位置**: `loop.py:2195-2320+`

**当前问题**:
- context 注入 verify_report、fabrication、dataset_quality、visual_analysis、domain_maes...
- LLM 被海量诊断信息淹没，可能得出错误结论
- 反思是开放式的"分析结果"，没有结构化引导
- 反思结论不与阶段目标对比

**改造: Gap-Closing Reflection**

```
当前:
  context = {brief, memory_log, experiment_result, verify_report, 
             fabrication, dataset_quality, visual_analysis, domain_maes...}
  prompt = "分析实验结果"

改造后:
  context = {
      "phase_focus": PHASE_STATUS,  # 复用同一个状态
      "experiment_delta": {"auc": "+0.02", "accuracy": "-0.01"},  # 与上次的差异
      "gap_remaining": {"auc": -0.323, "accuracy": -0.0952},      # 距目标还差多少
      "verify_critical": [只包含 severity=critical 的验证失败],   # 精简
      "recent_memory": [最近3条],                                   # 精简
  }
  prompt = "Phase 1 目标 AUC ≥ 0.90。本次实验 AUC 从 0.557 提升到 0.577。
            差距从 0.343 缩小到 0.323。分析为什么提升有限，提出下一步改进方向。"
```

**关键改进**:
1. 反思自动与阶段目标对比，而不是自由评论
2. context 精简到 5 个核心键
3. verify_report 只注入 critical 级别，其他存日志不浪费 token
4. 反思输出必须包含 `gap_assessment` 和 `next_focus`

**改动量**: loop.py ~40行修改

---

### 模块 4: _enforce_roadmap_alignment

**位置**: `loop.py:3089-3160`

**当前问题**:
- 基于 keyword matching 判断是否偏移（`_is_task_related`）
- 3次偏移才强制纠正，前两次只是"注入警告"
- keyword matching 误报/漏报率高
- 与 THINK 改造后的 phase_focus 功能重叠

**改造: 合并到 Phase Gate**

```
当前:
  _enforce_roadmap_alignment → keyword match → 3次警告 → 强制

改造后:
  如果 THINK 已经用了 phase_focus 结构化提问，roadmap alignment 的职责
  被 phase_focus 的 blocked_methods 检查完全覆盖。
  
  这个方法可以精简为：只做 blocked_patterns 检查（1道门），
  去掉复杂的 keyword matching 和渐进式警告。
```

**实现方式**:
- 保留方法但大幅简化
- 用 PHASE_STATUS 的 blocked_methods 替代 keyword matching
- 去掉 3 次渐进式警告，改为单次 hard block
- 如果 THINK 阶段已被 phase_focus 约束，此门几乎不触发

**改动量**: loop.py ~30行修改（净减少代码）

---

### 模块 5: StrategyConstraintEngine (约束引擎)

**位置**: `constraint_engine.py:298+`

**当前问题**:
- `check_constraints()` 返回违规列表，但调用方只作为 `_constraint_warnings` 注入
- `generate_rules_from_history()` 从历史中学习规则，但这些规则是"软建议"
- FORBIDDEN 级别规则和 AVOID 级别规则处理方式相同
- 规则数量随时间增长，增加 context 膨胀

**改造: Binary Gate + Rule Freeze**

```
当前:
  violations = check_constraints(result, memory)  # 返回列表
  result["_constraint_warnings"] = get_constraint_prompt(violations)  # 软注入

改造后:
  violations = check_constraints(result, memory)
  if has_forbidden(violations):
      # 硬门: 直接阻止
      result["action"] = "data_analysis"
      result["task"] = f"⛔ BLOCKED: {blocked_msg}\n请聚焦: {phase_focus['focus_methods']}"
  # 非 FORBIDDEN 的不注入（减少 context 膨胀）
```

**关键改进**:
1. FORBIDDEN → 硬阻止 + 重定向到当前阶段允许的方向
2. 非 FORBIDDEN 规则不注入 context（减少噪音）
3. 规则数量冻结：超过 20 条时停止生成新规则

**改动量**: constraint_engine.py ~15行 + loop.py ~10行

---

### 模块 6: HUMAN_DIRECTIVE 消费

**位置**: `loop.py:4062-4098`

**当前问题**:
- `rename()` 归档指令文件，只消费一次
- 用户设置的 ⛔ STOP 只影响 1 个 cycle
- 没有持久约束机制

**改造: 与 PHASE_STATUS 统一**

```
当前:
  _consume_directive() → rename() → 一次性

改造后:
  不改 directive 机制（保持一次性灵活性）
  但将"持久约束"移到 PHASE_STATUS.json 的 blocked_methods 中
  和 PERSISTENT_CONSTRAINTS.md（每 cycle 注入 context）
```

这个改造已经在方案中，无需额外修改。

---

## 数据流总览（改造后）

```
PHASE_STATUS.json (唯一的真相源)
    │
    ├──→ _think(): 构建 phase_focus 注入 context (5行核心状态)
    │         │
    │         ▼
    │    LLM 结构化思考 (方向被问题限定)
    │         │
    │         ▼
    │    Phase Gate: 检查 blocked_patterns (1道门)
    │         │
    │         ▼
    │    _execute(): task 注入 scope 约束
    │         │
    │         ▼
    │    Code/Research Agent 执行 (方向被 task 限定)
    │         │
    │         ▼
    │    _verify(): 只检查 critical (精简)
    │         │
    │         ▼
    │    _reflect(): 自动与阶段目标对比差距
    │         │
    │         ▼
    │    更新 PHASE_STATUS.json (结果→目标对比)
    │
    └──→ 循环
```

## 改动量估算

| 模块 | 改动类型 | 行数 |
|------|---------|------|
| PHASE_STATUS.json | 新增 | ~15行 |
| _think() phase_focus 注入 | 修改 | ~30行 |
| _think() context 精简 | 修改 | ~20行 |
| _execute() scope 注入 | 修改 | ~15行 |
| _reflect() gap-closing | 修改 | ~40行 |
| _enforce_roadmap_alignment 简化 | 修改 | ~30行 |
| constraint_engine FORBIDDEN 硬门 | 修改 | ~15行 |
| loop.py FORBIDDEN 处理 | 修改 | ~10行 |
| PHASE_STATUS 更新逻辑 | 新增 | ~25行 |
| **总计** | | **~200行** |

## 设计原则检查

- ✅ **面多加水水多加面禁忌**: 只改 2 个文件 + 1 个新 JSON，不新增模块
- ✅ **利用现有架构**: 在 `_think()`, `_execute()`, `_reflect()` 内部修改，不新建类
- ✅ **单一真相源**: PHASE_STATUS.json 是唯一的阶段状态来源
- ✅ **上下文极小**: 每个 phase 注入 5-7 行核心状态
- ✅ **约束行动也引导思考**: 问题结构本身限定答案空间
- ✅ **保留硬门作为安全网**: blocked_patterns 单门检查
