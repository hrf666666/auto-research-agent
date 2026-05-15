---
name: idea-validation
description: "Validates proposed ideas technically before implementation, and searches literature when experiments fail"
argument-hint: "<idea_text> or [--query <search_query>] or [--validate-proposal <file_path>]"
---

# /idea-validation

对提出的研究 idea 做技术验证，或在实验失败时联网查阅文献。

## 两种工作模式

### 模式 A：Idea 技术验证（`--validate-proposal`）

在实现任何新想法之前，必须先验证其技术可行性。避免 agent 花大量时间实现一个物理上不可行或与现有代码不兼容的想法。

### 模式 B：文献调研（`--query` 或实验失败触发）

当实验方法行不通时，搜索相关文献。如果文献可下载，保存到 `workspace/ref_paper/`。

---

## 模式 A：Idea 技术验证流程

### Step 1: 读取上下文（按顺序）

在验证任何 idea 之前，必须依次读取：

```
1. PROJECT_BRIEF.md        ← 项目的核心目标和当前状态
2. MEMORY_LOG.md           ← 读取 ## Dead Ends 表（新！idea 绝对不能与死胡同方法相同）
3. working/plan-issues.md  ← 当前已知的问题（idea 可能与已知问题冲突）
4. working/env-issues.md   ← 已知的环境限制
```

**关键**：如果 idea 的方法与 `## Dead Ends` 中的条目本质相同，验证**必须 FAIL**，除非能给出充足理由说明为什么这次会不同。

### Step 2: 验证清单（每条必须给出明确结论：PASS / FAIL / WARN）

#### V1. 合理性（Physical Plausibility）

- [ ] **物理原理是否自洽？**
  - 检查 idea 是否违反了基本物理约束（如光场渲染方程、视差几何）
  - 检查角域/空域的假设是否与数据吻合（如：假设 81 视角角方差低 → 暗含朗伯假设）
  - **FAIL 如果**：idea 要求"所有像素共享同一个 r 值"，但物理上不同材质角方差不同

- [ ] **与当前数据集是否匹配？**
  - idea 依赖的 GT 标签是否在数据集中存在？（如：per-pixel roughness label）
  - idea 假设的视角数量是否与数据集一致？（HCI4D=81视角，Lytro=其他配置）
  - **FAIL 如果**：idea 需要 per-pixel material label，但数据集没有提供

- [ ] **超参数是否有物理意义？**
  - 超参数的范围是否与物理量对应？（如：`r ∈ [0.1, 10]` 对应粗糙度 a/λ）
  - **WARN 如果**：超参数是 magic number，无物理对应

#### V2. 实现可行性（Implementation Feasibility）

- [ ] **Idea 是否与现有代码架构兼容？**
  - 检查 `models/` 中现有模型类的输入输出格式
  - 检查 `datasets/` 中现有 dataloader 的数据格式
  - **FAIL 如果**：idea 需要改变数据格式，但 pipeline 无法适配

- [ ] **算力是否满足？**
  - 估算显存需求：输入分辨率 × batch_size × 模型参数量
  - 当前 GPU（RTX 3090 24GB）是否能承载？
  - **FAIL 如果**：单张输入 926×926 + 81视角 + batch=1 → 估算 > 24GB

- [ ] **训练时间是否合理？**
  - 估算每个 epoch 的迭代次数和每步耗时
  - 是否能在可接受的时间内完成训练？
  - **WARN 如果**：单个 epoch 预计 > 2小时

#### V3. 与现有方案的关系（Novelty Check）

- [ ] **Idea 与当前方案的核心区别是什么？**
  - 读懂当前方案（`PROJECT_BRIEF.md` 中的 "Current Problems"）的局限性
  - 新 idea 是否直接针对这些局限性？
  - **WARN 如果**：新 idea 只是增量改进而非解决根本问题

- [ ] **是否与之前失败的方案重复？**
  - 检查 `MEMORY_LOG.md` → `## Dead Ends` 表（最高优先级）
  - 检查 `workspace/ref_paper/` 中的历史尝试
  - **FAIL 如果**：新 idea 本质上是之前尝试过且失败的方法，且无充分理由

### Step 3: 输出验证报告

```markdown
# Idea Validation Report — YYYY-MM-DD

## Idea Summary
[一句话描述 idea 的核心思想]

## Context
- Validated against: PROJECT_BRIEF.md (dated XXX)
- Known issues checked: working/plan-issues.md, working/env-issues.md
- Historical attempts checked: MEMORY_LOG.md, workspace/ref_paper/

## Validation Results

### V1. Physical Plausibility
| Check | Result | Notes |
|-------|--------|-------|
| 物理原理自洽 | PASS/FAIL | ... |
| 数据集匹配 | PASS/FAIL | ... |
| 超参数物理意义 | PASS/FAIL | ... |

### V2. Implementation Feasibility
| Check | Result | Notes |
|-------|--------|-------|
| 代码架构兼容 | PASS/FAIL | ... |
| 显存需求 | PASS/FAIL | ... |
| 训练时间 | PASS/WARN | ... |

### V3. Novelty Check
| Check | Result | Notes |
|-------|--------|-------|
| 针对现有局限 | PASS/FAIL | ... |
| 避免重复失败 | PASS/FAIL | ...（对比 MEMORY_LOG Dead Ends） |

## Overall Verdict

**[APPROVE / REJECT / NEEDS_REVISION]**

### If APPROVE:
- Next steps for implementation
- Key risks to monitor

### If REJECT:
- Specific reasons for rejection
- Suggestions for how to salvage or redirect

### If NEEDS_REVISION:
- Specific gaps that need to be filled
- Additional validation needed (e.g., literature search)
```

### 验证结论的处理

- **APPROVE**：将 idea 转化为具体的 implementation plan，更新 `PLAN.md`
- **REJECT**：记录拒绝原因到 `MEMORY_LOG.md`，并提出替代方向
- **NEEDS_REVISION**：如果 gap 在文献中 → 进入模式 B 做文献调研；如果 gap 在数据集 → 建议收集/生成新数据

---

## 模式 B：文献调研流程

### Step 1: 确定调研触发条件

文献调研在以下任一条件下触发：
1. **实验失败**：当前方法无法达到目标（如 val MAE > 0.1）
2. **Idea 被标记为 NEEDS_REVISION**：需要补充物理/算法依据
3. **用户明确要求**：用户提供了参考文献或引用了某篇论文
4. **探索新方向**：项目需要引入新的技术路线

### Step 2: 优先读取用户提供的参考文献

**优先级顺序（依次尝试，找到即停）：**

```
① 用户直接提供的 PDF/链接
   → 检查 workspace/ref_paper/ 下是否有对应文件
   → 如果有，直接读取分析

② 用户在 PROJECT_BRIEF / 聊天中引用的文献
   → 从引用中提取 arXiv ID 或 DOI
   → 尝试下载到 workspace/ref_paper/

③ workspace/ref_paper/ 目录下已有的文献
   → 扫描是否有未分析的 PDF
   → 按修改时间排序，最新的优先分析
```

### Step 3: 联网搜索（仅在用户文献不足时）

如果 Step 2 没有找到足够的相关文献，使用 `web_search` 工具搜索：

**搜索策略（按优先级）：**

```
优先级 1: 直接搜 "paper title" 或 arXiv ID
优先级 2: 按关键词搜: "light field depth estimation non-Lambertian"
优先级 3: 按关键词搜: "光场深度估计 非朗伯"
优先级 4: 按方法搜: "epipolar plane image depth specular"
优先级 5: 泛化搜: "light field depth estimation survey 2023 2024"
```

**每轮搜索最多取前 10 篇**，优先选择：
- 有 PDF 下载链接的
- 来自 CVPR/ICCV/ECCV/TPAMI/TMM 等顶会顶刊
- 发表时间近 5 年内的

### Step 4: 下载并保存文献

对于每篇找到的文献：

```
① 尝试 arXiv PDF 直链: https://arxiv.org/pdf/{arxiv_id}.pdf
② 如果是 IEEE/ACM: 尝试 DOI → 出版社 PDF
③ 保存到: workspace/ref_paper/{第一作者}{年份}_{arxiv_id}.pdf
④ 记录到: workspace/ref_paper/INDEX.md (文献索引)
```

**INDEX.md 格式：**

```markdown
# Reference Papers Index

## {日期} — 调研原因: [实验失败 / Idea需要Revision / 用户提供]

| # | 论文标题 | 作者 | 年份 | 来源 | 本地路径 | 关键发现 |
|---|---------|------|------|------|---------|---------|
| 1 | ... | ... | 2024 | arXiv | ref_paper/xxx.pdf | ... |
```

### Step 5: 文献分析（对最相关的 1-3 篇做深度分析）

使用 `paper-analyze` skill 分析最相关的文献：
- 与当前问题最相关的
- 提供了可借鉴方法/损失函数/模型架构的
- 揭示了当前方法根本性问题的

**分析输出**：写入 `workspace/ref_paper/ANALYSIS_{date}.md`

### Step 6: 输出调研报告

```markdown
# Literature Survey Report — YYYY-MM-DD

## Trigger
[实验失败: val MAE=0.738 on HCI4D / Idea需要Revision / 用户提供]

## User References (优先)
| 文献 | 路径 | 关键发现 |
|------|------|---------|
| ... | ref_paper/xxx.pdf | ... |

## Web Search Results
| 论文 | 来源 | 下载状态 | 与本项目的相关性 |
|------|------|---------|---------------|
| ... | arXiv | ✓ saved | 高: 可借鉴XXX方法 |

## Key Insights from Literature
1. [来自文献1]: ...
2. [来自文献2]: ...

## Recommendations
- [具体的算法改进建议，基于文献]
- [下一步实验计划]
```

---

## 与其他 Skill 的关系

### 与 hands-off-issue-handling 的集成

当 idea 验证发现物理上不可行时：
- **不属于 hands-off-issue-handling**：这是 idea 设计阶段的问题，不是执行阶段的问题
- **处理**：直接在验证报告中标记为 REJECT/NEEDS_REVISION，记录到 MEMORY_LOG.md

当文献调研发现新的已知问题时：
- **属于 hands-off-issue-handling**：如果文献揭示了一个当前未记录的系统性问题
- **处理**：记录到 `working/plan-issues.md`，在下次 plan 更新时解决

### 与 paper-analyze 的集成

```
idea-validation (模式B)
  → 发现需要深度分析的文献
  → 调用 paper-analyze
  → 分析结果写入 workspace/ref_paper/ANALYSIS_*.md
```

### 与 auto-experiment 的集成

```
auto-experiment loop:
  EXECUTE phase:
    执行新 idea 的实验
    如果失败 → 触发 idea-validation 模式B
      → 搜索文献
      → 如果找到有效方法 → 更新 PLAN.md
      → 如果无结果 → 记录到 plan-issues.md
```

---

## 硬性规则

1. **验证必须在实现之前**：任何新 idea 在写代码之前必须经过模式 A 验证
2. **文献必须保存本地**：联网找到的文献必须下载到 `workspace/ref_paper/`，不能只读摘要
3. **INDEX.md 必须维护**：每次新增文献必须更新索引
4. **验证报告必须留存**：将验证报告保存到 `workspace/idea_validation_reports/`
5. **搜索记录必须完整**：web search 的关键词、结果数量、选择理由都要记录
