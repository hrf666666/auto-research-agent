---
name: hands-off-issue-handling
description: "Passive skill — guides how the autonomous agent handles, classifies, and records problems during THINK/EXECUTE/REFLECT phases"
user-invocation: false
---

# Hands-Off Issue Handling

本 skill 是**被动框架**，不通过 slash command 调用。它定义了在 auto-experiment
循环中遇到各类问题时的处理原则、分类标准和记录格式。

**核心职责**：在 agent 发现问题时，提供"什么该记、什么不该记、记到哪里"的统一规则。

---

## 关键原则

| 原则 | 说明 |
|------|------|
| 能自主解决的不记录 | 代码 bug → 直接修，不记录 |
| 3 尝试后仍失败才归类 | 代码逻辑问题必须解决，环境问题才记录 |
| 统一写到 MEMORY_LOG.md | 不新建 issue 文件，保持框架 memory 体系单一 |
| 阻塞性→通知，否则继续 | P0 问题通过 HUMAN_DIRECTIVE.md 干预 |

---

## 问题分类与处理流程

### Type A：代码逻辑 / 实现问题（自主解决，不记录）

**判断标准**：运行时报错、行为不符合预期、逻辑错误。
**处理**：立即修复，不写入任何记录文件。
**示例**：ImportError、shape mismatch、loss 不收敛。

### Type B：实验规划问题（记录到 MEMORY_LOG.md）

**判断标准**：任务规划文档 (plan) 与实际代码实现之间存在歧义，
需要 agent 自行假设后才能继续。
**处理**：将假设记录到 MEMORY_LOG.md，继续执行。
**记录格式**（追加到 MEMORY_LOG.md 的 Recent Decisions）：

```
[YYYY-MM-DD HH:MM] Plan Ambiguity: [描述]. Assumption: [所做的假设]
```

### Type C：规格文档问题（记录到 MEMORY_LOG.md）

**判断标准**：PROJECT_BRIEF.md 或上层规格中有模糊/矛盾之处，
无法通过阅读解决，必须假设。
**处理**：将假设记录到 MEMORY_LOG.md，通知用户。
**记录格式**：

```
[YYYY-MM-DD HH:MM] Spec Issue: [描述]. Assumption: [所做的假设]
```

### Type D：环境问题（3 尝试后记录到 MEMORY_LOG.md + 降级处理）

**判断标准**：尝试 3 种不同方向的方案仍失败，且不是代码逻辑问题。
**处理**：
1. 将所有尝试记录到 MEMORY_LOG.md
2. 降级处理（例如：GPU OOM → 减小 batch size；缺少依赖 → 记录但不阻塞）
3. 如果 P0 级别（完全阻塞），写 `workspace/HUMAN_DIRECTIVE.md` 请求用户介入

**记录格式**（追加到 MEMORY_LOG.md）：

```
[YYYY-MM-DD HH:MM] Env Issue: [描述]
  Attempt 1: [方案A] → [失败原因]
  Attempt 2: [方案B] → [失败原因]
  Attempt 3: [方案C] → [失败原因]
  Assumption: [降级处理方案或 None]
  Impact: P0/P1/P2
```

---

## P0 / P1 / P2 优先级处理

| 优先级 | 定义 | 处理方式 |
|--------|------|---------|
| **P0** | 完全阻塞，3 种尝试均失败，无法降级 | 写 `workspace/HUMAN_DIRECTIVE.md` 请求人工干预 |
| **P1** | 影响后续实验方向决策 | 记录到 MEMORY_LOG.md，在下次 REFLECT 时讨论 |
| **P2** | 已知限制，不影响当前执行 | 记录到 MEMORY_LOG.md，继续工作 |

---

## 与框架 phase 的对应关系

### THINK 阶段（dispatch_leader）
- 读取 PROJECT_BRIEF.md 时发现歧义 → **Type C**
- 读取 MEMORY_LOG.md 时发现已有 Assumption → 遵循其假设，不重复记录

### EXECUTE 阶段（Code Agent）
- 代码实现与 plan 不符 → **Type B**（假设+记录）
- 运行时报错 → **Type A**（直接修复，不记录）
- 遇到环境问题 → **Type D**（3 尝试后记录）

### REFLECT 阶段（dispatch_leader reflect）
- 解析实验结果时发现方向性问题 → **Type B**（记录决策到 MEMORY_LOG.md）
- 检查是否有 P0 问题需要干预 → 如有，写 HUMAN_DIRECTIVE.md

---

## 禁止记录的内容

以下信息**禁止**写入 MEMORY_LOG.md 或任何文件：

- ❌ "no issue found" — 没有任何意义
- ❌ "Notes:" — 与真正的问题无关
- ❌ 同一问题的重复记录（查重后跳过）
- ❌ 违反 spec/plan 本身的问题（必须立即修复，不是记录）
- ❌ 已有代码中的 bug（必须立即修复，不属于任何 issue 类型）

---

## 与其他 Skill 的关系

| Skill | 关系 |
|-------|------|
| `auto-experiment` | 主框架，本 skill 的调用方 |
| `progress-report` | 引用 MEMORY_LOG.md 中的 Blockers；本 skill 的记录为其提供输入 |
| `experiment-status` | 引用 MEMORY_LOG.md；Type D 中的 P0 可能触发指令干预 |
| `paper-analyze` | 无直接关联 |
| `conf-search` | 无直接关联 |

本 skill **不创建任何新文件**，所有问题都追加到 `workspace/MEMORY_LOG.md`
的对应章节中，保持 memory 体系单一。
