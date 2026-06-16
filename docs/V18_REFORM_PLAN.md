# v18 架构改革方案（最终版）

## 设计原则（5 条，每条改动必须服务其中至少一条）

```
P1. 安全是工具的契约，不是一个组件
    → write_file 自带路径校验，launch_experiment 自带 GPU 检查
    → 不需要 SafetyGuard 在上层检查

P2. 信息是被查询的，不是被注入的
    → 记忆提供 query 接口，LLM 自主决定看什么
    → 不需要 50 个 context 注入点替 LLM 预判

P3. LLM 在认知循环里自主工作，系统不干预（安全除外）
    → observe → decide → act → learn
    → 系统不覆盖 LLM 的决策

P4. 记忆是核心——记忆有效时 enforcement 自然消亡
    → 修记忆而不是加 enforcement
    → 89 条教训 0 次使用 → 修消费端，不加强制执行

P5. prompt 定义研究方法论框架，不定义步骤
    → "先假设再验证" 是框架
    → "连续 2 次没 launch 就强制重写" 不是框架，是控制
```

## 改革策略：渗透式演化，不是推倒重来

不从零重写。每一步让系统**更接近 P1-P5**，同时保持可运行。
核心判据：每步交付后，系统的某一部分从"系统控制"变成"LLM 自主"或"工具内置"。

---

## 阶段一：修记忆（服务 P4）

**问题**：89 条教训 0 次使用、60% 实验记录为空、定量结果变成定性文本。
**本质**：记忆的产出端在工作，消费端断裂。修消费端不需要改架构。

### 1.1 闭合知识查询

当前 `search_relevant_lessons`（loop.py:1177）和 `get_causal_history`（loop.py:1145）有调用但运行时跳过——因为 `models/` 目录不存在、`verified` 过滤为空。

改法：
- `get_causal_history`：有 verified 就给 verified，没有就给全部（让 LLM 自己判断可信度）
- `search_relevant_lessons`：`models/` 不存在时用 task 文本搜索，不跳过
- `experiment_value`：注入低价值方向列表

### 1.2 结构化实验记录

当前定量 metric（val_MAE=0.184）在 SQLite 是结构化的，但传到 MEMORY_LOG.md 变成 LLM 自由文本。

改法：
- monitor 提取 metric 后，系统**直接写入** MEMORY_LOG.md 一行结构化记录：
  `[Cycle N] method=X val_MAE=0.184 status=success target=0.20 gap=-0.016`
- 这行不经过 LLM——系统确定性地写入
- LLM 的 milestone 文本仍然保留（定性判断），但定量 anchor 总是在

### 1.3 目标进度

当前没有目标达成检测。agent 不知道自己离目标多近。

改法：
- 从 PROJECT_BRIEF 解析目标（`val_MAE < 0.20`）
- 从 SQLite 查最好成绩
- 注入 `goal_progress` context key（系统计算，不需要 LLM）
- 全部目标达成 → 停止循环

### 1.4 启用已建未用的共享模块

`training_log_parser.py` 和 `model_structure_scanner.py`（Fix C 建的）当前未被调用。
- `training_log_parser`：接入 monitor 的 metric 提取 + verifier 的 loss 解析（替代 4 份重复正则）
- `model_structure_scanner`：接入 `analyze_model` 工具（替代 `_estimate_params_from_ast` 的 kwargs bug）

**阶段一验收**：
- 日志出现非空的知识注入
- MEMORY_LOG 每条实验记录有定量 metric
- THINK prompt 有 Goal Progress
- 达成目标时 agent 停止

---

## 阶段二：工具内聚安全（服务 P1）

**问题**：安全约束散落在 run()、constraint_engine、phase gate 等多处。
**本质**：安全应该是工具的内置属性，不是一个独立层。

### 2.1 write_file 内聚命名+路径安全

当前命名规则只在 code_agent.md 的 prompt 建议。tools/ 有 117 个文件。

改法：
- `write_file` 增加路径校验（工具的契约，不是上层检查）：
  - `train_*.py` → 只允许 `scripts/`
  - 根目录 `.py` → 拒绝
  - protected files → 拒绝
- 违规返回明确 error（让 LLM 知道为什么被拒、怎么改正）

### 2.2 launch_experiment 内聚 dry-run+资源检查

当前 dry-run 前置是 prompt 建议 + 60% budget gate。

改法：
- `launch_experiment` 内置 dry-run 检查（如果 config `mandatory_dry_run: true`）：
  检查同一个脚本是否刚跑过 dry-run（通过 experiment_manifest 的 timestamp）
  如果没有 → 返回 "dry-run required first" error
- GPU 资源检查内置（检查 nvidia-smi 可用量）

### 2.3 确定性垃圾回收

当前 `_auto_code_cleanup` dispatch code agent（消耗配额）。

改法：
- 新建 `core/garbage_collector.py`：纯 Python 确定性操作
- 每 cycle 结束扫描 `debug_*`/`diag_*`/`dryrun_*` → 归档
- `outputs/` 超 10 目录 → 最旧归档
- `_auto_code_cleanup` 的 LLM dispatch 路径删除

**阶段二验收**：
- write_file 拒绝违规路径
- launch_experiment 强制 dry-run
- tools/ 临时文件 < 10
- 不再 dispatch code agent 做清理

---

## 阶段三：LLM 自主认知循环（服务 P3、P5）

**问题**：THINK/EXECUTE/VERIFY/REFLECT 固定流水线 + 36 个 enforcement 分支。
**本质**：LLM 应该在认知循环里自主工作，系统不干预。

这是最大的一步。分两个子阶段降低风险。

### 3.1 先：去掉研究决策类 enforcement（低风险）

删除这些**研究决策**（不是安全）的 enforcement：
- 9 个停滞计数器 + `_apply_no_progress_fallback`（~400 行）→ 0 次触发
- `_enforce_audit_findings` 独立方法 → 合并进 launch/audit 计数器
- architecture_switch 机制（~200 行）→ 研究决策
- idea_guardian / direction_circuit_breaker（~100 行）→ 变 prompt 方法论指导

**保留**的安全相关 enforcement：
- `_check_phase_blocked`（代码扫描，防止过早训练）
- `_enforce_launch_after_failure`（行为约束）
- `constraint_engine` FORBIDDEN（已证明失败的方案）
- pause_human（安全阀）

**prompt 调整**（服务 P5）：
在 leader.md 加一段方法论指导（替代被删的 enforcement）：
```markdown
## Research Methodology
- If multiple consecutive experiments show no improvement, consider 
  surveying alternative approaches before continuing.
- Use your experiment history to avoid repeating failed directions.
- Verify results honestly before concluding success.
```
这不是 enforcement（"你必须X"），是方法论（"如果Y，考虑Z"）。

### 3.2 后：简化 run() 为认知循环

将 run() 从 4923 行简化为 ~300 行的编排循环：

```python
def run(self):
    self._init_goal_and_memory()
    while self._running and not self._goal_achieved():
        self.cycle_count += 1
        try:
            # LLM 自主决策
            think_result = self._think()       # observe + decide
            self._execute(think_result)         # act（含安全检查在工具层）
            
            # 实验完成时验证+反思
            if self._has_completed_experiment():
                verify = self._verify(...)      # 系统客观验证
                reflect = self._reflect(verify) # LLM 学习
                self._record(reflect, verify)   # 结构化记忆
            
            self._goal_check()                  # 目标检查
            self._garbage_collect()             # GC
        except SafetyViolation as e:
            logger.warning(f"Safety: {e}")
            # 不覆盖决策，让 LLM 在下个 cycle 自然调整
```

**关键变化**：
- run() 不再修改 think_result（P3）
- 安全检查在工具层（P1），不在 run()
- VERIFY/REFLECT 只在有完成的实验时运行（不是每 cycle 固定）
- 目标达成自动停止

### 3.3 插件变工具

| 插件 | 变成 | 注册给 | 一句话 prompt 指导 |
|------|------|--------|-------------------|
| simulation_sandbox | `check_feasibility` 工具 | code | "训练前可用 check_feasibility 验证前向" |
| idea_planner | `plan_architecture` 工具 | leader | "设计新架构时可用 plan_architecture" |
| visual_analyzer | `analyze_predictions` 工具 | leader | "metric 停滞时可用 analyze_predictions 看预测" |
| experiment_evaluator | 修 bug，保留在 VERIFY | 系统 | — |
| domain_knowledge | 背景注入（保留） | 系统 | — |
| obsidian | 可选，不进核心 | — | — |

**副作用处理**：
- prompt 里工具指导只写一行（P5），不写 enforcement
- 工具的 JSON schema 告诉 LLM 它存在，LLM 自己决定调用
- 如果 LLM 不用某个工具 → 说明它不需要，不强求

**阶段三验收**：
- run() < 500 行
- LLM 的决策不被系统修改（除安全退回）
- 4/6 插件作为工具可用
- leader.md < 400 行

---

## 阶段四：非阻塞 + 通用化（服务 P1、P5）

### 4.1 训练异步化

- launch_experiment 返回 PID 后不阻塞
- 每 cycle 检查实验状态
- 训练期间 LLM 可做其他事
- config `parallel_experiments` 控制（GPU 限制在 launch_experiment 内聚检查）

### 4.2 去领域硬编码

- metric key 从 config/manifest 读取
- verifier 阈值从 config 读取
- domain_knowledge METHOD_PROPERTIES 从外部 YAML

### 4.3 prompt 最终精简

- leader.md：去掉被删机制的规则，保留方法论框架，~300 行
- code_agent.md：命名规则移到工具层后不重复，~200 行

**阶段四验收**：
- 训练期间 agent 能做其他事
- 领域硬编码 < 3 处
- leader.md < 300 行

---

## 已有组件去留

| 组件 | 去留 | 理由（服务哪条原则）|
|------|------|-------------------|
| context_keys.py | 保留+改造 | P2：记忆查询的序列化基础 |
| signal_arbiter.py | **合并进工具层** | P1：安全是工具属性，不是独立仲裁层 |
| training_log_parser.py | 启用 | P4：结构化记忆的基础 |
| model_structure_scanner.py | 启用 | P1：analyze_model 工具的实现 |
| idea_scout_bridge.py | 保留为工具 | P3：LLM 自主调用的工具 |
| 9 停滞计数器 | 删除 | P3：研究决策 |
| _apply_no_progress_fallback | 删除 | P3：研究决策 |
| audit escalation 5 方法 | 简化+合并 | P1：保留安全 enforcement，删 advisory |
| architecture_switch | 删除 | P3：研究决策 |
| idea_guardian | 变 prompt 指导 | P5：方法论，不是 enforcement |
| ContextPruner | 被记忆查询取代 | P2：LLM 自主查询不需要预注入+裁剪 |

---

## 总效果

| 指标 | 当前 | 阶段一 | 阶段二 | 阶段三 | 阶段四 |
|------|------|--------|--------|--------|--------|
| 总行数 | 22833 | ~22900 | ~22800 | ~21200 | ~20500 |
| loop.py | 5179 | 5179 | 5100 | ~2500 | ~2500 |
| self._ 状态 | 101 | 95 | 90 | 35 | 30 |
| 知识消费率 | 0% | >80% | >80% | >80% | >80% |
| 目标追踪 | 无 | 有 | 有 | 有 | 有 |
| 训练阻塞 | 是 | 是 | 是 | 是 | 否 |
| 安全/研究分层 | 否 | 否 | 工具内聚 | 工具内聚 | 工具内聚 |
| 插件作工具 | 0/6 | 0/6 | 0/6 | 4/6 | 4/6 |
| 命名/GC | prompt | prompt | 系统强制 | 系统强制 | 系统强制 |
| leader.md | 673 | 673 | 673 | ~400 | ~300 |
| enforcement 层数 | 7 | 7 | 5 | 2 | 2 |

**最终**：代码 ~20500 行（-10%），loop.py 2500 行（-52%），状态变量 30 个（-70%），enforcement 从 7 层降到 2 层（安全+pause_human）。

---

## 每阶段的风险与回退

| 阶段 | 风险 | 回退 |
|------|------|------|
| 一 | 知识注入质量低 | context budget 截断；config 开关 |
| 二 | 工具内聚检查误拒 | 返回明确 error，LLM 可调整重试 |
| 三 | LLM 决策质量下降 | prompt 方法论指导补充；可逐步删 enforcement |
| 四 | 非阻塞状态不一致 | config non_blocking: false 回退同步 |
