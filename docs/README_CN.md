<h1 align="center">Auto Research Agent</h1>
<h3 align="center">24/7 全自主深度学习实验 Agent</h3>



<p align="center">
  <a href="../README.md">English</a> |
  <a href="README_CN.md">中文</a> |
</p>

> **来源声明**：本项目最初来源于 [auto-deep-researcher-24x7](https://github.com/Xiangyue-Zhang/auto-deep-researcher-24x7)。

---

> **第一次来？别慌。** 你不用读完整个README，只需要做一件事：
>
> 1. 打开 `AI_GUIDE.md`，丢给 **Claude / ChatGPT / Codex**
> 2. AI 会一步步带你装好、配好、跑起第一个实验
> 3. 就这样。不焦虑，我们慢慢来。
>
> *想先了解原理？继续往下看。*

---

## 💛 写在最前：我们造它的初衷，以及希望大家怎么用它

> **我们的愿望很朴素：让学术保持纯粹，让人始终留在循环里。**

我们做这个框架，只有一个目的——把跑深度学习实验里那些**机械、重复**的环节（起任务、看 GPU、读日志、扫超参）从研究者身上拿掉，让大家把省下来的时间，留给**真正重要的事：思考**。

如果你来到这里，是因为想少花点时间盯训练，多花点时间读论文、想 idea、追自己的研究方向——欢迎你。这个工具就是为你做的。

**有一件事，想轻轻地拜托每一位使用者：**

Agent 很乐意替你把实验跑完，但请把 *idea*、*结果的解读* 和 *科学判断* 留给自己。我们不觉得"自动化"和"学术诚信"是对立的——恰恰相反，这个工具帮你省下来的时间，本意是让你**投入到更深的思考里**，而不是让你跳过思考本身。

所以我们想善意地请求大家：不要用这个项目去伪造结果、不要用它去"生成"完全没有人类参与的所谓研究、也不要用它去绕开那些真正需要一个人去理解、去判断的科研环节。那不是我们想帮忙建造的未来，我们也相信，那同样不是大多数你们想要的未来。

> **学术应当保持纯粹。Agent 可以替你跑实验，但 idea、判断与责任，请留给人来承担。**
>
> **Science should stay pure.** The agent can run the experiments — but the ideas, the interpretation, and the responsibility belong to the human. We genuinely hope every user will keep a **human in the loop for thinking**, and make their own real contribution in their own research direction.
>


我们相信每一个愿意拿起这个工具的人，都会认真对待这件事——也正是因为相信你们当中的大多数本来就是这样的人，我们才愿意把它开源出来。谢谢你成为其中之一。💛

---

## 痛点

现有的 AI 研究工具帮你**写论文**。Deep Researcher Agent 帮你**做实验**。

| 现有工具 | Deep Researcher Agent |
|:-:|:-:|
| 帮你写论文 | **自主运行实验** |
| 整理你的笔记 | **分析结果并迭代** |
| 帮你搜论文 | **形成假设并验证** |
| 你问它才动 | **24/7 不间断工作** |

```
你睡 8 小时     → Agent 跑了 3 轮实验
你出去度假      → Agent 探索了 50+ 组超参配置
你在写论文      → Agent 已经把 results table 准备好了
```

### 实战验证的成果

> 这不是 benchmark，是真实项目中数月 24/7 自主运行的成果。

| 指标 | 数据 |
|------|------|
| 自主完成的实验循环 | 500+ 轮 |
| 单项目最佳指标提升 | 比基线提升 52%（200+ 次自动实验）|
| 同时管理的项目数 | 4 个项目，4 台 GPU 服务器 |
| 最长连续运行时间 | 30+ 天无需人工干预 |
| 24 小时平均 LLM 成本 | ~¥0.55 |

---

## 最近更新

**2026-05-25 (v15.5) — Research ROADMAP：模块级状态机与阶段门控研究**

*防止过早训练模型，强制在提交 GPU 资源前进行结构化理论验证。*

### 解决的核心问题
Agent 跳过理论验证直接进行模型训练，在架构假设从未被验证的情况下浪费 GPU 时间。模块验证使用硬性"恰好 3 种方法"的要求过于僵化——某些方向需要更多方法才能正确证伪，而 ROADMAP 与 ResearchLoop 之间的双偏差计数器可能不同步，导致状态过期。

### 新模块 (v15)
- **`core/research_roadmap.py`** — 模块级状态机，追踪每个模块的研究阶段：
  `theory_verification → module_design → module_validation → integrated → dead_end`

### 新增机制 (v15)

| # | 机制 | 阶段 | 功能 |
|---|------|------|------|
| 1 | **Research ROADMAP** | THINK | 追踪每个模块的研究阶段。每个模块必须经过 theory_verification → module_design → module_validation 才能标记为 integrated。 |
| 2 | **阶段门控研究** | THINK → EXECUTE | 代码级强制：theory_verification 阶段禁止模型训练，仅允许分析/研究任务。 |
| 3 | **三击硬门控** | THINK → EXECUTE | 偏离 ROADMAP 阶段：第 1 次 = 警告，第 2 次 = 强警告，第 3 次 = 强制切换到 `paper_research`。`ResearchRoadmap._deviation_count` 和 `ResearchLoop._phase_violation_count` 双计数器同步。 |
| 4 | **断路器优先级** | THINK | theory_verification 期间 ROADMAP 覆盖方向/架构断路器。 |
| 5 | **灵活方法验证** | VERIFY | 每个模块要求 3-6 种方法（最少 3，最多 7）才能标记 dead_end。不是硬性 3 种——支持需要更深入分析的方向。 |
| 6 | **假设嵌入方法** | THINK | `_suggest_verification_methods` 在每个方法描述中嵌入假设文本，当存在 ≥2 个假设时添加跨假设一致性检查。 |

### v15.5 加固修复（8 项 Code Review 修复）

| # | 文件 | 修复内容 |
|---|------|----------|
| 1 | `loop.py` + `research_roadmap.py` | `check_alignment()` 不再返回 `should_force_paper_research`；loop.py 通过 `_phase_violation_count` 独立控制三击强制机制。硬门控触发时同步重置 `roadmap._deviation_count`。 |
| 2 | `research_roadmap.py` + `loop.py` | 新增公开属性 `active_module_names` 和 `is_theory_verification_phase`，替代私有 `_get_active_modules()` 访问。 |
| 3 | `research_roadmap.py` | `_is_training_task` 重写：使用强指示器（如"train the"、"training"、"epoch"）和弱指示器（如"loss function"、"loss curve"），避免"information loss"、"loss of detail"等误报。 |
| 4 | `research_roadmap.py` | `_is_task_related` 改进：通过 `re.split(r"[_\s]+")` 子词元匹配，动态阈值 `max(2, len(words)//3)`，降低词长度阈值（>3 替代 >4）。 |
| 5 | `research_roadmap.py` | `MODULE_DESIGN` 里程碑处理：关键词"designed"、"implemented"、"coded"、"built"推进到 `MODULE_VALIDATION`。 |
| 6 | `idea_planner.py` | `_suggest_verification_methods` 在每个方法描述中嵌入假设文本，当 ≥2 个假设时添加跨假设一致性检查（方法 4）。 |
| 7 | `research_roadmap.py` | Markdown 解析器正则从 `\[\\w+\]` 改为 `\[\\w_-+\]`，添加下一行证据提取。 |
| 8 | `context_keys.py` | 移除孤立的 `roadmap_alignment_warning` ContextKey（已注册但从未在上下文注入流程中使用）。 |

### 关键改动
- `core/research_roadmap.py`：新 ROADMAP 模块（~500 行）
- `core/loop.py`：`_enforce_roadmap_alignment()`，ROADMAP 感知的方向断路器
- `core/idea_planner.py`：方法建议中的跨假设一致性检查
- `core/context_keys.py`：清理孤立键
- `agents/leader.md`：v15 规则（ROADMAP 感知和阶段约束）

### 模块大小

| 模块 | v14 | v15.5 |
|------|-----|-------|
| `core/loop.py` | ~4,150 | ~4,350 |
| `core/research_roadmap.py` | — | ~500 |
| `core/idea_planner.py` | 1,059 | ~1,070 |
| `core/context_keys.py` | 194 | 193 |
| **总计** | **~21,750** | **~22,470** |

---

**2026-05-20 (v14) — 战略架构智能**

*解决 #1 失败模式：在错误的架构上浪费数十个 cycle 修修补补。*

### 解决的核心问题
在一次 64 cycle 的运行中，Agent 花费所有 64 个 cycle 修补 EPINet（PROJECT_BRIEF 中的 baseline），从未调研替代架构。仅 12.5% 的 cycle 做了 paper_research，且没有一个导致架构切换。根因：(1) 无架构调研 — Agent 从第 1 个 cycle 就盲目使用 PROJECT_BRIEF 推荐的 baseline；(2) Dead End 只记录为个体条目，从未聚类为"EPINet 架构本身就是瓶颈"；(3) 方向签名过于细粒度 — "EPINet+edge_loss"和"EPINet+angular_conv"被识别为不同方向；(4) paper_research 重置所有停滞计数器，掩盖了持续的架构问题。

### 新增机制 (v14)
1. **架构调研门控** — 在 cycle 1-2 强制调研 3+ 候选架构后才允许选择 baseline。输出 `ARCHITECTURE_SURVEY.md`。
2. **架构级方向签名** — 检测底层架构（EPINet、U-Net、Transformer 等），而非仅检测方向关键词。架构停滞在同一架构的所有方向间累积。
3. **Dead End 聚合引擎** — 按架构聚类 dead end，当 5+ dead end 追溯到同一架构时自动生成 `[ARCHITECTURE BOTTLENECK]` 警告。
4. **架构切换执行器** — 当架构停滞达到阈值（5 cycles），强制 `architecture_switch` 动作，要求切换到完全不同的架构。

### 关键改动
- `core/loop.py`：新增 `_architecture_stagnation_count`、`_extract_architecture_name()`、`_analyze_architecture_dead_ends()`、`architecture_switch` 动作处理、架构调研门控注入、架构断路器。
- `core/domain_knowledge.py`：新增 `_synthesize_architecture_dead_ends()` 架构级 dead end 聚类分析。
- `agents/leader.md`：新增"架构智能规则"章节，4 条强制规则。

### 模块大小

| 模块 | v13.2 | v14 |
|------|-------|-----|
| `core/loop.py` | 3,919 | ~4,150 |
| `core/domain_knowledge.py` | — | ~660 |
| **总计** | **~21,220** | **~21,750** |

---

**2026-05-19 (v13.2) — Code Review 知识库 + 多工具集成**

*包含 v12.3–v13.1 变更 + v13.2（多工具安装器、Python API、Cursor/CodeBuddy 支持）。*

### 解决的核心问题
Agent 在多个 cycle 中重复犯相同的错误：(1) PRE-EXECUTE Code Review 的正则检查反复匹配注释/文档字符串中的关键词，造成无限 HARD GATE 死循环，连续阻止训练 9+ 个 cycle；(2) 已验证的架构缺陷（死胡同、模块失败）仅记录为文本，未系统提取为可复用的教训；(3) API 超时 300 秒导致单次调用挂起 25+ 分钟；(4) `reflect_result` 中的 `null` 值导致 SQLite 记录崩溃。

### 新增机制

| # | 机制 | 阶段 | 功能 |
|---|------|------|------|
| 1 | **Code Review Lessons 知识库** | Memory | 新增 `code_review_lessons` SQLite 表，支持模式去重、严重度排名、命中计数跟踪、关键词相关性搜索 |
| 2 | **Post-Reflect 教训提取** | REFLECT → Memory | 自动从 VERIFY 失败、死胡同、模块失败、LLM 语义分析中提取教训 |
| 3 | **THINK 阶段教训注入** | THINK | 自动加载与当前模型代码相关的教训，注入 Leader 上下文 |
| 4 | **HARD GATE 死循环检测** | Gate | 连续 2 次 HARD 阻止后自动降级为 SOFT GATE |
| 5 | **注释感知正则检查** | Gate | 新增 `_strip_comments_and_strings()` 方法，正则检查仅匹配真实代码 |
| 6 | **增强 LOW VOI 引导** | THINK | 强烈建议切换到 `paper_research` 寻找新方向 |
| 7 | **`code_review` 工具** | EXECUTE | Code Agent 新增结构化代码分析工具 |
| 8 | **API 超时优化** | EXECUTE | 超时从 300 秒降至 120 秒 |
| 9 | **两阶段代码审查** *(v12.3)* | Gate | Phase 1: 零 LLM 正则检查；Phase 2: LLM 语义审查（仅在 Phase 1 无 HIGH 问题时运行） |
| 10 | **日志回退解析器** *(v12.3)* | VERIFY | `training_log.json` 不存在时，从原始日志文本提取 routing weights、aux loss、per-domain MAE |
| 11 | **动态域名发现** *(v12.3)* | VERIFY | 不再硬编码域名，动态解析任何 `DOMAIN: MAE=X.XXX` 模式 |
| 12 | **有序 Gate Pipeline** *(v12.4)* | THINK→EXECUTE | 三个 Gate 按优先级顺序执行：PRE-VERIFY → CODE REVIEW → FALSIFIABILITY。Hard-gate 触发后跳过后续 Gate |

### Bug 修复
- **`MAX(severity, ?)` 对 TEXT 列无效** (严重): SQLite 字典序使 HIGH < LOW < MEDIUM，严重度升级逻辑完全反了。改为 Python 侧比较。
- **整数与 TEXT 比较** (严重): severity 过滤用整数比较 TEXT 列。改为 `IN (...)` 子句。
- **`NoneType` 下标错误** (HIGH): LLM 返回 `null` 值导致 SQLite 崩溃。改为 `(x or "")[:500]` 模式。
- **全表加载性能问题** (MEDIUM): `search_relevant_lessons` 添加 `LIMIT 100`。
- **固定 pattern 合并** (LOW): 死胡同和模块失败改为内容关键词提取 pattern。

### v13.1 加固修复
- **跨行字符串处理 bug**: `_strip_comments_and_strings` 逐行处理无法移除跨行 `"""..."""`。改为先在全内容上移除多行字符串，再逐行处理。
- **`gate` 关键词误报**: 正则 Check 1 匹配所有 `gate`（如 `torch.sigmoid`）。改为 `gate_weight|gate_network|gating`。
- **搜索评分优化**: 高 `hit_count` 的无关 lesson 不再返回。要求至少一个关键词匹配。
- **SQL 查询合并**: `record_code_review_lesson` 的 UPDATE 路径从两次查询合并为一次。
- **LLM JSON 响应处理**: `_llm_extract_lesson` 添加 JSON 解析回退。
- **训练脚本独立检查**: 模型文件不可用时仍执行训练脚本检查。
- **Gate 2 代码清理**: 双重 `if` 改为 `if/else`。

### v13.2 — 多工具 Skill 集成 + Python API
- **多工具安装器**: `install.py` 支持 `--claude-code`、`--codebuddy`、`--cursor`、`--all`，一键安装 skill 到各工具原生目录。
- **Python API** (`api.py`): `AutoResearcher` 类提供 `run_one_cycle()`、`start_daemon()`、`get_status()`、`get_code_review_lessons()` 等编程接口。
- **CLI API**: `python api.py {status|run|start|stop|lessons}` 命令行接口。
- **Cursor 支持**: Skill 安装为 `.mdc` 规则文件到 `~/.cursor/rules/`。
- **CodeBuddy 支持**: Skill 安装为 `.md` 命令到 `~/.codebuddy/commands/`。

### 知识闭环

```
失败 → 提取教训 → 存储到 SQLite → 关键词匹配注入 THINK → 避免重复犯错
```

### 模块大小

| 模块 | v12.2 | v13 | v13.2 | v14 | v15.5 |
|------|-------|-----|-------|-----|-------|
| `core/loop.py` | 3,248 | ~3,904 | 3,919 | ~4,150 | ~4,350 |
| `core/memory.py` | 963 | ~1,130 | 1,134 | 1,134 | 1,134 |
| `core/agents.py` | 1,357 | ~1,411 | 1,411 | 1,411 | 1,411 |
| `core/tools.py` | 1,739 | ~1,884 | 1,884 | 1,884 | 1,884 |
| `core/domain_knowledge.py` | — | — | — | ~660 | ~660 |
| `core/research_roadmap.py` | — | — | — | — | ~500 |
| **Total** | **~19,734** | **~20,850** | **~21,220** | **~21,750** | **~22,470** |

**2026-05-18 (v12.2) — Pre-Execute 代码审查 + 训练架构验证**

### 解决的核心问题
Agent 生成的模型代码（如 MaterialDualCueNet）包含多个架构缺陷（routing weights 永远不分化、aux loss 太低、输入通道信息不对称），但训练前没有被检测到。Agent 在架构有缺陷的模型上浪费了 GPU 时间和 LLM token，而 REFLECT 阶段无法诊断模型为什么失败——因为 v12 的分析反思只覆盖数据分析实验，不覆盖训练实验。

### 根因分析
Agent 存在两个盲区：
1. **训练前无代码审查**：Code agent 写完模型 → 立即训练 → VERIFY 只检查训练是否执行了，不检查架构设计是否合理。
2. **无训练架构反思**：v12 的 `analysis_reflection_prompt` 只在分析实验（`experiment_launched=False`）时触发。训练实验没有专门的反思机制来评估 routing 收敛、aux loss 有效性或 per-domain 回退。

### 新增机制

| # | 机制 | 触发阶段 | 功能 |
|---|------|---------|------|
| 1 | **Pre-Execute 代码审查** | THINK → EXECUTE（门控） | 训练前对模型代码做零 LLM 结构审查。检查：routing 无辅助监督、输入通道不对称（>5x 比率）、1×1 conv router（无空间上下文）、加权融合无 skip 连接。问题以强制修复前缀注入任务描述。 |
| 2 | **Layer 12: 训练架构验证** | VERIFY | 训练后检查：(a) routing weights 分化度——所有 domain 差异 <5% = FAIL，(b) aux loss 收敛——跨 epoch 不变 = FAIL，(c) per-domain 回退——比基线退化 >20% = WARN。 |
| 3 | **训练架构反思提示** | REFLECT | 当 VERIFY Layer 12 检测到问题时，注入专门的反思提示，强制 Leader 评估 routing 收敛、aux loss 有效性、per-domain 回退，并正确分类失败。仅在训练实验（`experiment_launched=True`）时触发。 |
| 4 | **结构化指标注入** | REFLECT | 解析 `training_log.json` 的 per-domain MAE 趋势（逐 epoch）和 aux loss 趋势。标记正在变差的 domain。用结构化 JSON 解析替代原始日志正则解析。 |

### 关键设计决策
- **Pre-Execute 审查是门控，不是阻塞**：问题作为警告注入任务前缀，不是硬阻塞。避免误报阻止合法实验。Code agent 看到警告后可选择修复或继续。
- **Layer 12 使用 `training_log.json`**：结构化 JSON 解析（不是原始日志文本的正则）确保可靠提取指标。JSON 不可用时回退到正则解析。
- **基线指标来自 MEMORY_LOG.md**：Per-domain 回退与 MEMORY_LOG.md 中最近记录的基线比较，无需额外状态文件。
- **`_pre_execute_code_review` 零 LLM 调用**：所有检查基于 regex/AST，无 LLM 调用。每 cycle 增加约 50ms 延迟。

### 新增上下文键
- THINK → EXECUTE: code review 警告注入 task（pre-execute 门控）
- VERIFY: Layer 12 `routing_differentiation`、`aux_loss_convergence`、`domain_regression` 检查
- REFLECT: `training_architecture_reflection_prompt`、`per_domain_mae_trend`、`aux_loss_trend`

### 流程图
```
THINK (规划实验)
  → Pre-Verify (数据、导入)
  → [新增] Pre-Execute 代码审查 (模型架构)
    → 发现问题? → 作为警告注入任务
  → EXECUTE (Code agent 写代码 + 训练模型)
  → VERIFY (Layer 1-11 不变)
  → [新增] VERIFY Layer 12 (训练架构检查)
  → [新增] VERIFY Layer 13 (aux loss 收敛)
  → REFLECT
    → [新增] training_architecture_reflection_prompt (当 Layer 12 有问题时)
    → [新增] per_domain_mae_trend + aux_loss_trend
```

### 模块大小

| 模块 | v12.1 | v12.2 |
|------|-------|-------|
| `core/loop.py` | 2,992 | 3,248 |
| `core/verifier.py` | 2,210 | 2,537 |
| **总计** | **~19,071** | **~19,734** |

**2026-05-18 (v12.1) — API 配额耗尽时 REFLECT 降级回退与干净停止**

### 解决的核心问题
API 配额耗尽时 REFLECT 阶段失败，导致整个 cycle 的 EXECUTE + VERIFY 成果丢失。Agent 进入 600 秒退避循环，且重试也无法恢复配额。

### 改动
- **REFLECT 降级回退**（`_degraded_reflect`）：当 `dispatch_leader()` 因 `insufficient_quota` / `All providers failed` / `429` 失败时，基于规则的降级 REFLECT 从 VERIFY 报告、训练日志和分析输出中生成基本反思结果。输出格式与 Leader REFLECT 的 JSON schema 完全兼容，下游代码（`_record_cycle_outcome`、`_update_state`）无需修改。
- **降级反思待处理标记**：写入 `.degraded_reflect_pending` JSON 文件。下一 cycle 的 THINK 阶段读取该文件并注入 `degraded_reflect_pending` 上下文，提示 Leader 在规划新工作前先回顾未完成的 cycle。
- **配额错误时干净停止**：外层 `except` 块现在能检测配额错误并直接退出主循环（不再对不可恢复的错误执行 600s 退避）。
- **新增上下文键**：THINK: `degraded_reflect_pending`（上一个 cycle 存在降级 REFLECT 时注入）。

**2026-05-18 (v12) — 探索性分析模式：防止假阴性死胡同**

### 解决的核心问题
Agent 仅用 1 种方法（FFT 能量比，5 个特征，Cohen's d 最高 0.47）就断定"角度频率材料分类不可行"。但用 5 种方法 17 个特征重新分析后，Cohen's d 达到 2.06，AUC 0.969——方向完全可行。Agent 的分析过于狭窄，产生了假阴性死胡同。

### 新增机制

| # | 机制 | 触发阶段 | 功能 |
|---|------|---------|------|
| 1 | **探索性分析模式** | VERIFY (Layer 11) | 扫描分析输出中的方法族（FFT、统计、视角一致性、空间、频率分解）。在断定方向不可行前要求至少 3 种独立方法。 |
| 2 | **特征完整性报告** | VERIFY | 检查分析是否覆盖至少 4 个特征族。生成结构化 JSON 报告，包含 `feature_families_tested`、`methods_with_strong_signal`、`recommendation`。 |
| 3 | **失败分类系统** | REFLECT + Memory | 死胡同现在携带结构化的 `failure_category`：`hypothesis_wrong`、`implementation_bug`、`insufficient_experiment` 或 `method_inadequacy`。存储在 SQLite `failure_category` 列（自动迁移）。 |
| 4 | **方法不足再唤醒** | THINK | 当存在 `failure_category='method_inadequacy'` 的死胡同时，Leader 收到提示，鼓励用更广泛的分析重试而非放弃方向。 |
| 5 | **分析实验反思提示** | REFLECT | 当实验为数据分析（非训练）时，注入专门的反思提示，强制 Leader 评估方法覆盖度、特征完整性，并正确分类失败。 |

### 关键设计决策
- **`failure_category` 列**：通过 `log_dead_end()` 中的 `ALTER TABLE` 自动迁移。v12 之前的数据库继续正常工作（空字符串默认值）。
- **`get_dead_ends_by_category()`**：按类别结构化检索死胡同，支持再唤醒机制。
- **`get_method_inadequacy_count()`**：THINK 阶段注入的快速计数查询。
- **Layer 11 `_verify_analysis_coverage()`**：基于模式的分析输出方法检测。如果只找到 1 个方法族，生成警告要求 Leader 必须分类为 `method_inadequacy`（而非 `hypothesis_wrong`）。

### 新增上下文键
- THINK: `method_inadequacy_retry_prompt`、`degraded_reflect_pending`（v12.1）
- VERIFY: Layer 11 分析覆盖度检查
- REFLECT: `analysis_reflection_prompt`、`method_inadequacy_history`

### 模块大小

| 模块 | v11 | v12 | v12.1 |
|------|-----|-----|-------|
| `core/loop.py` | ~2,660 | ~2,860 | 2,992 |
| `core/memory.py` | 852 | ~960 | 963 |
| `core/verifier.py` | 2,102 | ~2,200 | 2,210 |
| `core/simulation_sandbox.py` | ~600 | ~600 | 1,518 |
| `core/constraint_engine.py` | ~580 | ~580 | 1,164 |
| `core/agents.py` | — | — | 1,357 |
| **总计** | **~17,204** | **~17,500** | **~19,071** |

**2026-05-14 (v11.1) — 三轮 Code Review 全面修复 + 硬编码参数提取**

### 解决的核心问题
v1-v11 从未做过系统性 code review，积累了多个静默 bug 和大量硬编码参数，影响跨项目复用和健壮性。

### 修复的 Bug（9 个）
| # | 严重度 | 模块 | 问题 |
|---|--------|------|------|
| 1 | BUG | `constraint_engine.py` | `get_dead_ends(limit=20)` 方法不存在 → `get_dead_ends_full()[:20]` |
| 2 | BUG | `memory.py` | `get_summary_stats()` 缺少 `best_metric`/`worst_metric` → AdaptiveThresholds 无法校准 |
| 3 | BUG | `simulation_sandbox.py` | `_resolve_model()` 危险 fallback → 找不到模型时静默返回最近的 .py 文件 |
| 4 | BUG | `loop.py` | 重复 `import math` |
| 5 | BUG | `loop.py` | `_cooldown_after_error` 时间计算错误：sleep 不足60秒也加60 |
| 6 | BUG | `loop.py` | `_quality_alert_streak` 无条件递增（不管有没有质量退化） |
| 7 | BUG | `simulation_sandbox.py` | Hook 双重注册（`_build_behavior_script` 注册两次 forward/backward hook） |
| 8 | BUG | `simulation_sandbox.py` | Conv3d 参数估算用 `k*k` 而非 `k*k*kd` |
| 9 | BUG | `simulation_sandbox.py` | Layer 2a 推理脚本不计算 MAE（只收集统计量，vs GT 始终为空） |

### 硬编码参数提取到 `config.yaml`
新增 `sandbox` 配置节，所有子进程超时、GPU 内存预算、默认输入 shape 从配置读取：

```yaml
sandbox:
  gpu_memory_mb: 24000            # 目标 GPU 显存预算 (MB)，0=自动检测
  default_input_shape: [1, 3, 64, 64]  # 输入 shape fallback
  subprocess_timeout: 120         # 子进程超时 (秒)
  inference_timeout: 120          # 推理超时 (秒)
  feasibility_timeout: 90         # 可行性检查超时 (秒)
```

### 阈值动态化
- `loop.py` 视觉分析触发阈值 → AdaptiveThresholds 的 `severe_degradation`
- `loop.py` 改进判定阈值 → AdaptiveThresholds 的 `improvement_threshold`
- `experiment_evaluator.py` gap 阈值 → 从 AdaptiveThresholds 获取
- `verifier.py` 振荡/过拟合阈值 → 从 AdaptiveThresholds 获取
- `domain_knowledge.py` stuck domain 阈值 → 从 AdaptiveThresholds 获取
- 各模块 `1e-8` epsilon → 提取为模块级 `_EPS` 常量

### 原子写入修复
- `state.json` 写入 → 先写 `.tmp` 再 `rename()`，防止 crash 时损坏
- `MEMORY_LOG.md` 写入 → 同上

### 默认 Provider 更新
- `loop.py` 默认 provider 从 `"anthropic"` → `"glm_token_plan"`（与 config.yaml 一致）
- `loop.py` 默认 model 从 `"claude-sonnet-4-6"` → `"auto"`（tier 路由）

**2026-05-13 (v11) — 仿真沙盒：训练前模型验证与A/B评价**

### 解决的核心问题
Agent修改模型后直接训练，无法回答四个关键问题：
1. **修改有没有用？** — 加了模块但可能根本没激活
2. **和修改前比怎样？** — 参数量翻倍但指标没变
3. **能不能继续放大？** — 不知道瓶颈在哪，盲目扩展导致OOM
4. **是否背离项目初衷？** — PROJECT_BRIEF说要解决X，修改却在优化Y

### 新模块
- **`core/simulation_sandbox.py`**（~600行）：5层评价体系

| 层 | 名称 | 时机 | 内容 |
|----|------|------|------|
| 0 | 可行性检查 | PRE-VERIFY | 模型能不能跑？shape对不对？GPU够不够？ |
| 1 | 设计对比 | REFLECT | A/B结构对比：参数变化、模块占比、信息瓶颈 |
| 2a | 有参考评价 | REFLECT | vs GT指标 + vs 修改前指标 + 参数效率 |
| 2b | 无参考评价 | REFLECT | 模块活性/贡献度/梯度健康/参数利用率 |
| 3 | 综合判定 | REFLECT | 有效/部分/无效/有害 + 项目初衷对齐 |
| 4 | 扩展指引 | REFLECT | 可扩展模块/瓶颈/显存预算/规模扩展建议 |

### 关键设计
- **模型快照**：PRE-VERIFY自动保存当前模型到`model_snapshots/`，下一cycle的A/B对比用
- **判定缓存**：评价结果缓存到`_sandbox_last_verdict.json`，下一cycle的THINK读取作为设计指引
- **子进程隔离**：所有模型运行在subprocess中，主循环不受影响
- **双维度评价**：有参考（vs GT）+ 无参考（内部行为），综合判定模块是否真正有效

### 启动方式
```bash
source ~/.bashrc  # 确保 GLM_CODING_PLAN_API_KEY 已加载
python -m core.loop --project /path/to/project --config config.yaml
```

**config.yaml 关键配置**：
```yaml
agent:
  provider: "glm_token_plan"    # 默认 provider，auto-failover 到阿里
  model: "auto"                 # auto = 按任务 tier 选强/快模型

sandbox:
  gpu_memory_mb: 24000          # 你的 GPU 显存 (MB)，按实际调整
  default_input_shape: [1, 3, 64, 64]  # 项目输入 shape

monitor:
  max_runtime_hours: 12         # 训练最大时长
```

**2026-05-13 (v10) — 约束引擎：LLM行为控制层**

### 解决的核心问题
LLM作为Agent大脑有三个致命弱点，纯靠prompt无法解决：
- **幻觉**：声称做了实际没做的事
- **虚报指标**：报告漂亮的指标但模型输出是垃圾
- **投机取巧**：跳过困难模块，用空函数/stub糊弄

v10引入**可验证的硬约束**——每个检查都是机器可验证的，不依赖LLM自觉遵守。

### 新模块
- **`core/constraint_engine.py`**（~580行）：6大约束机制

| # | 机制 | 触发阶段 | 解决的LLM问题 |
|---|------|---------|-------------|
| 1 | **PlannerChecker** | REFLECT | Code Agent自由发挥——实现了和计划不同的架构 |
| 2 | **StrategyConstraintEngine** | THINK | 重复失败方向，无视历史教训 |
| 3 | **QuickBenchmark** | REFLECT | 指标虚报——报告的指标与实际计算不符 |
| 4 | **AdaptiveThresholds** | THINK | 固定阈值导致不同metric尺度下误诊 |
| 5 | **ImplementationTracker** | THINK+REFLECT | "假装完成"——悄悄跳过计划模块 |
| 6 | **ContextPruner** | THINK+REFLECT | 信息过载导致LLM困惑（30+键→前20） |

### 工作原理
- **PlannerChecker**：扫描 `models/*.py` 的AST，模糊匹配计划模块名，检测stub模式（`pass`、`NotImplementedError`、硬编码返回值、`forward()`不足5行）。生成合规报告和伪造风险评级。
- **StrategyConstraintEngine**：从SQLite历史（假设校准、死胡同、Pareto前沿）自动生成约束规则。如"edge loss失败5次→禁止再提"、"假设准确率<30%→必须提供证据"。规则持久化到 `STRATEGY_RULES.json`。
- **QuickBenchmark**：加载模型checkpoint，随机输入推理，对比输出统计量 vs 报告指标。差异>20%标记为异常。以子进程运行（120s超时），不阻塞主循环。
- **AdaptiveThresholds**：从SQLite读取metric范围，自动校准所有诊断阈值（如 `domain_gap_critical = range * 0.8`）。
- **ImplementationTracker**：跨cycle持久跟踪模块状态（`pending → implemented → verified`），强制Leader完成待定模块。
- **ContextPruner**：4级优先级裁剪（始终>情境>条件>罕见），限制20个key以内。

### 上下文注册更新
- THINK键：19 → 21（新增 `adaptive_thresholds`、`implementation_progress`）
- REFLECT键：24 → 27（新增 `plan_compliance_warning`、`quick_benchmark_warning`、`implementation_progress`）
- 总计：48个注册context key，带运行时验证

### 修复的关键Bug（v9→v10）
- **`memory.py`死代码（关键）**：`__init__`初始化代码在 `_infer_domain_keys()` 的 `return []` 之后不可达——持久化记忆系统静默失效
- **`experiment_evaluator.py`硬编码输入形状**：`IndependentProbe` 的 `(1, 81, 3, 64, 64)` → 从state_dict动态推断
- **`loop.py` `_last_architecture_plan` 从未赋值**：架构计划生成后被丢弃，每次reflect都重新生成
- **`verifier.py`/`loop.py` 私有属性访问**：`VerifyReport._independent_assessment` → 正式的dataclass字段
- **`loop.py` 13个静默 `except Exception: pass`** → 替换为 `logger.warning/debug`
- **`idea_planner.py` 依赖链错误**：次级模式模块错误地依赖前一个模块而非主干

**2026-05-13 (v9) — 前向设计流水线、实验后评估与领域无关化清理**

### 新模块
- **`core/idea_planner.py`**（1053 行）：`IdeaPlanner` — 9 阶段前向设计流水线，在编写模型代码前从 `PROJECT_BRIEF.md` 生成博士级架构规划：
  - 想法形式化 → 模块分解 → 容量规划 → 融合策略 → 集成规划 → 验证规划 → 风险评估 → 实现顺序 → 对齐评分
  - `IDEA_PATTERNS` 数据库：6 种通用架构模式（多分支融合、频域分析、域自适应、分量感知、注意力机制、渐进精化）
  - 词汇→概念映射：将自然语言术语映射到抽象概念类别（知识组织方法，非项目特定逻辑）
- **`core/experiment_evaluator.py`**（786 行）：三个实验后分析类：
  - `ExperimentEvaluator`：计划-结果对比、5 类失败诊断（架构/数据/训练/对齐/容量）、优先级排序的迭代指导
  - `IndependentProbe`：第三方验证——使用模型检查点 + 随机输入前向传播，检测输出异常（坍塌、NaN、范围不匹配）
  - `IterationGuidance` + `FailureDiagnosis`：结构化分析，含根因链和可操作的下一步建议

### 流水线集成
- **`plan_model` 工具**注册到 ToolRegistry — Leader 和 Code Agent 可调用架构规划
- **THINK 阶段**（cycle ≤ 1）：自动注入 `architecture_plan` + `architecture_plan_summary` 上下文
- **REFLECT 阶段**：自动注入 `experiment_evaluation` + `iteration_guidance_prompt` + `independent_assessment_warning`
- **VERIFY 阶段**：新增 Layer 10（`_verify_independent_probe`）运行第三方评估
- **Agent 提示**（`leader.md`、`code_agent.md`）：添加 plan_model 使用指南、后评估响应表、独立评估处理

### 领域无关化硬编码清理
系统性清除 10 个文件中所有项目特定硬编码。原则：**只有知识组织方法可以硬编码；项目特定的数据/场景必须是动态的**。

- **Agent 提示**（`agents/*.md`）：所有项目特定示例泛化（EPI/Lambertian/HCInew/UnifiedLFDataset/81角视图 → 通用领域/方法/数据集语言）
- **`core/loop.py`**：跨域分析从硬编码指标名 → 动态 `self.memory.domain_keys` + 模式扫描；域分解提示 → 通用
- **`core/verifier.py`**：数据集→域映射从硬编码字典 → `DATASET_MANIFEST.json` `type` 字段 + 启发式回退；类发现从硬编码列表 → AST 动态扫描
- **`core/model_analyzer.py`**：默认输入形状从 `[1,81,3,64,64]` → `_infer_input_shape()` 读取 Conv3d/Conv2d `in_channels`
- **`core/memory.py`**：默认关键词/域从深度估计特定 → `_default_method_keywords()` / `_infer_domain_keys()` 读取 manifest
- **`core/visual_analyzer.py`**：移除推理脚本硬编码；域差距示例 → 通用
- **`core/idea_planner.py`**：添加 docstring 说明词汇→概念映射设计原则；修复重复域条目

### 知识组织架构（6 层）
1. **静态提示**（`agents/*.md`）：通用工作流指导，现在零项目特定引用
2. **持久化记忆**（`MEMORY_LOG.md` + SQLite）：累积的实验结果和死胡同
3. **领域知识**（`METHOD_PROPERTIES`）：通用科学方法属性（假设、失效症状、替代方案）
4. **架构模式**（`IDEA_PATTERNS`）：6 种可复用架构模式用于规划生成
5. **运行时上下文注入**（loop.py 中 27 个键）：从项目状态动态获取知识（manifest、指标、规划、评估）
6. **工具链知识**：`domain_knowledge.py`、`idea_planner.py`、`experiment_evaluator.py` 作为组织方法

**2026-05-13 (v8) — 架构重构与研究智能增强**

本版本通过全面的能力评估识别出根本性限制，从代码架构重构和研究智能两个维度进行改进。

### 代码架构重构
- **`core/tools.py` 拆分**（4714 → 1667 行）：约 3000 行代码提取到两个新的 mixin 模块：
  - `core/mcp_client.py`（724 行）：`MCPClientMixin` — 所有 MCP 传输逻辑（SSE + stdio）、服务检测、视觉工具、图像分析
  - `core/model_analyzer.py`（2282 行）：`ModelAnalyzerMixin` — 全部 9 层 AST 分析、数据流追踪、瓶颈检测、梯度路径分析、结构完整性评分、域假设检测、Idea-架构对齐、Decoder 充分性分析、运行时模型探针、诊断脚本生成、消融实验设计
- **`core/loop.py` 拆分**（2550 → 2268 行）：提取到：
  - `core/domain_knowledge.py`（559 行）：`DomainKnowledgeMixin` — 领域知识注入和跨实验元模式分析
- **`ToolRegistry` 现在继承 `MCPClientMixin` + `ModelAnalyzerMixin`**；`ResearchLoop` 现在继承 `DomainKnowledgeMixin`
- **`visual_analyzer.py` 重复 MCP 代码移除**：删除 `_call_zai_direct_spawn`（132 行）— 始终通过 `ToolRegistry` 的 MCP session 调用
- **`memory.py` 域关键词可配置**：`MemoryManager.__init__` 接受 `method_keywords` 和 `domain_keys` 参数，支持跨领域复用
- **Bug 修复**：`_find_last_assistant_text` 处理仅含 tool_calls 的消息，`_tail_file` 返回 `list` 而非 `deque`，`_get_gpu_status` 始终返回 `"gpus"` 键

### 研究智能增强
- **动态领域知识**（替代硬编码逻辑）：`domain_knowledge.py` 不再包含项目特定的硬编码域映射，而是：
  - `METHOD_PROPERTIES` 数据库：7 个通用方法条目（EPI、FFT、注意力、ResNet、sigmoid、Conv3D、对比学习），仅包含科学知识（假设、失效症状、替代方案）
  - `_infer_domain_compatibility()`：从 PROJECT_BRIEF 文本动态推断方法-域兼容性
  - `_extract_data_constraints()`：从 brief 文本检测数据稀缺性（< 10 训练样本）和数据不平衡
  - `_detect_implemented_methods()`：扫描 `models/` 目录检测实际实现的方法
  - `_extract_domain_names()`：通用域名称提取
- **Idea 守护者**（leader.md + loop.py）：每 5 个 cycle 强制 Leader 重新阅读 PROJECT_BRIEF 阶段目标，对核心 idea 实现度、阶段完成度、数据优先验证进行评分（0-10）。任何分数 < 5 触发强制纠偏
- **方向熔断机制**（loop.py）：当 `_direction_stagnation_count` 超过阈值时，注入 `direction_circuit_breaker` 上下文，强制 Leader 重新评估研究方向
- **数据分析实验支持**（code_agent.md）：新增章节明确支持非训练实验 — 数据可行性验证、Phase 1 分析、假设验证
- **数据稀缺感知**（leader.md + loop.py）：当某域训练样本 < 10 时，指示 Leader 不要提出架构变更（硬约束），转向数据增强、迁移学习或接受限制

**2026-05-12 (v7) — Idea-架构对齐分析**
- **Layer 9: Idea-架构对齐分析**（`analyze_model` 新增）：将模型架构与 `PROJECT_BRIEF.md` 进行对比，验证核心研究 idea 是否被忠实实现。捕获 agent 构建的模型仅松散遵循 idea 但结构性低估关键创新的错误。
  - **Idea 组件检测**：解析 PROJECT_BRIEF 中的 10 个关键模式（角度频率、双掩码、EPI、BRDF、分量感知深度、非朗伯处理、注意力、自适应融合等）并映射到模型分支。
  - **通道分配分析**：计算各分支通道占比，当关键创新分支在融合通道中占比 < 15% 时发出警告（"FFT 仅占 9%" 问题）。提供具体改进建议（如"将 fft_branch 从 32 增加到 ~72 通道"）。
  - **结构缺口检测**：检查 idea 隐含但模型缺失的架构模式（跳跃连接、多尺度处理、多分支融合注意力）。仅标记与特定 idea 相关的模式。
  - **Decoder 充分性分析**：评估解码器深度、跳跃连接和领域感知能力。当浅层解码器需要处理大融合输入或 idea 需要领域预测但模型缺失时发出警告。
  - **对齐评分（0-10）**：对缺失 idea 组件（-3）、代表性不足的分支（-2）、缺失的结构模式（-1）扣分。提供总体评估和具体行动指导。
- **增强分支检测**：`_extract_branch_info` 现在能追踪自定义类定义（如 `AngularFFTBranch`）查找输出通道数，而非仅支持直接的 `nn.Conv2d`/`nn.Sequential` 调用。
- **Leader Prompt v7**：新增 "Idea-Architecture Alignment" 章节，要求模型创建/修改后必须调用 `analyze_model`。提供基于对齐评分的行动指导（< 5: 大幅修订，5-7: 处理关键发现，≥ 8: 关注训练）。
- **Code Agent Prompt v7**：更新 Step 1.5 添加 idea-对齐指导——编码前必须检查 `idea_architecture_alignment` 部分，验证关键创新分支有足够通道分配。

**2026-05-12 (v6) — 博士级科研能力增强**
- **训练曲线分析**：VERIFY 现在执行丰富的训练曲线诊断，不仅检查"loss 是否下降"：检测过拟合（loss 在最低点后上升）、振荡（方向变化比）、收敛速度（过慢 = 模型容量不足）、平台检测（loss 长期持平）。结果注入 REFLECT 供 Leader 使用。
- **帕累托前沿追踪**：新增 `pareto_matrix` SQLite 表追踪方法×领域 MAE 矩阵。自动识别帕累托最优方法（至少在一个领域最优）和被支配方法（从未最优）。注入 THINK 阶段，避免重复次优方法。
- **`generate_diagnostic` 工具**（新增）：根据自然语言问题生成针对性诊断脚本。与 `probe_model`（随机数据）不同，它创建脚本回答具体问题，如"FFT 分支在 Non-Lambertian 输入上是否死掉了？"支持 4 种诊断类型：领域分析、分支分析、梯度分析、注意力分析。
- **`design_ablation` 工具**（新增）：基于 AST 的系统化消融实验设计。识别所有模型组件（分支、骨干网络、预测头、融合层、归一化层），按功能分类，生成优先排序的消融实验（组件移除、冻结、单分支、融合替换）及信息价值估计。Code agent 可用。
- **实验信息价值（VOI）估计**：实验运行前，系统估计信息价值（预期改进 × 成功概率）。低 VOI 实验发出警告。追踪校准数据（假设实际正确率）以改进未来估计。
- **结构化元模式学习**：`_build_cross_experiment_insights` 现在使用结构化 SQLite 查询而非脆弱的正则表达式匹配 markdown 文本。仅在数据库为空时回退到正则。检测支配性方法、假设准确率趋势和校准反馈。
- **自适应实验粒度（Pilot 实验）**：高风险假设（先验概率 < 0.4）自动推荐 pilot 实验（2-3 epoch）。低价值实验发出警告。注入任务描述中。
- **因果链追踪**：新增 `causal_chain` SQLite 表记录设计决策→架构属性→受影响指标的因果关系。THINK 记录预期效果，REFLECT 更新实际效果。历史注入 THINK 避免重复失败的因果链。
- **新增 SQLite 表**：`pareto_matrix`、`causal_chain`、`experiment_value` — 均带索引支持高效查询。
- **Leader Prompt v6**：新增帕累托前沿感知、实验价值估计、Pilot 实验、因果链追踪和训练曲线分析解读等章节。

**2026-05-11 (v5) — 深度模型分析与运行时诊断**
- **`analyze_model` 深度架构分析**：从表面 AST 分析升级为 8 层深度分析：(1) 参数计数，(2) 数据流图，(3) 信息瓶颈检测（压缩比 > 8:1），(4) 梯度路径分析，(5) 结构合理性评分（0-10），(6) 领域假设检测（EPI→Lambertian、FFT→频率稳定性等），(7) 数据可行性+GPU显存估算，(8) 结果→架构诊断。Leader、Code、Researcher 均可使用。
- **`probe_model` 运行时诊断**（新工具）：实例化模型，用 dummy 数据跑 forward+backward，抓取真实 tensor 统计——每个模块的激活 mean/std/dead_ratio、梯度范数、梯度平衡分析（检测 > 100x 不平衡）、输入敏感性测试（检测输出崩溃）、参数分布。可选加载训练好的 checkpoint 做训练后诊断。弥补了静态分析与真实模型行为之间的差距。
- **结果→架构反馈循环**：当域间差距 > 0.10（如 Non-Lambertian MAE 比 Lambertian 差 2 倍），REFLECT 注入强制推理链：识别最差域→陈述架构假设→验证假设是否成立→诊断根因→提出修复。明确禁止用调参解决结构性问题。
- **假设预验证（THINK Step 6）**：任何架构变更前，Leader 必须陈述核心假设、检查假设是否在数据中成立、通过 `analyze_model` 预验证、定义最小实验。
- **Code Agent 结构分析（Step 1.5 & 1.6）**：Code agent 修改架构前必须运行 `analyze_model`（检查结构评分、瓶颈、死分支）；训练失败后必须运行 `probe_model` 获取运行时证据（死激活、梯度死分支、输出崩溃）。
- **VERIFY Layer 9 — 模型结构合理性**：新验证层检测死模块（`__init__` 中声明但 `forward` 中未使用）和多分支融合警告。
- **Bug 修复**：verifier.py 缺少 `import ast`（Layer 9 静默 NameError），`_check_model_fusion_balance` 为空操作（已补充警告），WORKER_CONFIGS 工具列表与 `get_tools_for()` 同步，NaN 值在域差距计算中过滤，leader.md 重复 Step 3.5 重命名为 Step 3.6。

**2026-05-11 (v4) — Agent 能力升级**
- **输出质量感知**：Agent 现在追踪每个域的指标，检测"成功产出坏结果"的情况（如 Non-Lambertian MAE 从 0.30 恶化到 0.41）。连续 2 次质量降级强制论文研究并要求假设验证。
- **强制视觉分析**：当任何域 MAE 超过 0.35 时，无论连续周期数如何，立即触发视觉分析。Agent 必须"看"自己的预测输出。
- **战略性放弃**：研究方向停滞检测 — 同一方向连续 3 轮无改善，强制论文研究寻找根本不同的方法。
- **基础设施降级**：连续 3 次基础设施失败（API 超时、进程崩溃）后，VERIFY 不再阻塞实验。基础设施问题被记录但不阻止进度。
- **审计死循环防护**：连续 5 次审计指令后自动清除 DIRECTIVE.md，让 Agent 继续做实验。
- **跨域对比分析**：REFLECT 阶段强制要求 Leader 分析不同域性能差异的根本原因，识别被违反的方法假设，评估当前方法天花板。
- **假设验证**：质量持续降级时，Agent 必须声明当前方法的核心假设、识别哪个域违反了该假设、提出不依赖该假设的新方法。
- **配置一致性验证**：新增 VERIFY 层检查 checkpoint 不匹配和硬编码参数。
- **研究方向签名提取**：从任务描述中提取方法论关键词（edge/loss/pretrain/epi 等），支持连字符和下划线归一化。

**2026-05-08 (v3)**
- **MCP SSE 双连接协议修复**：修复所有 GLM 平台 MCP 服务（web_search_prime、web_reader、zread）— 从错误的 `POST /mcp` 改为正确的 `GET /sse` + `POST /message` 双连接 SSE 传输。后台读取线程从持久 SSE 流中收集 JSON-RPC 响应。
- **MCP stdio 传输用于视觉分析**：`zai-mcp-server` 以 **本地 npx 子进程**方式运行（stdio 传输），不是远程 SSE 服务。需要 Node.js 18+ 和 npx。通过 stdin/stdout JSON-RPC 通信。
- **4 个 MCP 服务全部验证通过**：web_search_prime ✅、web_reader ✅、zread ✅、zai-mcp-server (stdio) ✅
- **视觉降级链**（更新）：
  ```
  MCP zai-mcp-server (stdio, 需要 Node.js)
    → 直连 API: glm-5v-turbo → glm-4.6v (GLM Coding Plan)
    → 直连 API: qwen3.6-plus (阿里 Token Plan)
    → 直连 API: qwen3.6-plus → qwen3.5-plus (阿里 DashScope)
  ```
- **Node.js 依赖说明**：视觉 MCP 需要 `npx`（Node.js 18+）。没有 Node.js 时，视觉分析仍然通过直连 API fallback 工作，只是没有 MCP 专用工具。

**2026-05-07 (v2)**
- **视觉 MCP 完整集成**：`@z_ai/mcp-server` 8 个视觉工具集成到 `core/tools.py`。Agent 工具：`analyze_image`（研究 agent）和 `diagnose_error`（代码 agent）。
- **三供应商视觉降级**：3 个 API 供应商严格隔离（GLM Coding Plan / 阿里 Token Plan / 阿里 DashScope），各自独立 key 和 endpoint。
- **REFLECT 阶段深度分析**：Leader 在 REFLECT 期间获得 `read_file` + `list_files` 工具（20 轮上限，16384 token 上限），支持交叉验证。
- **脚本命名规范**：代码 agent 强制 `train_{model_short_name}.py` 命名，禁止增量后缀（`_v2`、`_fix`）。
- **自动代码清理 v2**：新增 scripts/ 命名污染（>10 文件）和 archive 膨胀（>20 条目）触发条件。

**2026-04-28**
- **Code Agent 轮次预算收紧**：`max_turns` 从 40 降到 25，防止无限探索。工具结果中注入轮次预算提醒 — 在 80% 预算时发出 "CRITICAL: 几乎用完轮次" 警告。
- **连续 `list_files` 限速**：最多连续 3 次 `list_files` 调用后强制停止，防止 Agent 反复浏览目录而不干正事。
- **`run_shell` 安全正则修复**：设备重定向模式现在正确允许 `/dev/null`、`/dev/zero`、`/dev/tty` 等，同时阻止不安全设备写入。
- **`write_file` datasets 注册白名单**：Agent 现在可以写 `datasets/__init__.py` 和 `datasets/unified_lf_dataset.py`（之前被阻止），其他 datasets 文件仍受保护。
- **训练脚本合成数据检测**：写入 `scripts/*.py` 时自动扫描随机噪声模式（`np.random.rand`、`torch.rand`、`SyntheticLF`、`RandomDataset`），返回警告。VERIFY 会阻止使用合成数据的实验。
- **AUDIT 升级 Level 4（不可修复）**：新增 `mark_unfixable` 级别（出现 ≥ 阈值×4 次），将问题记录为 dead_end 并停止升级 — 防止在真正无法解决的问题上无限重试。
- **Code-Cleanup 触发条件收紧**：无进展连续周期阈值从 2 提高到 4，减少误触发。
- **Memory 日志预算翻倍**：`log_max` 从 2,000 增加到 4,000 字符。旧条目现在被**压缩**（摘要化）而非删除 — 在满足预算的同时保留知识。新增 `get_log_summary()` 方法。
- **会话统计注入**：Leader 上下文现在包含 SQLite 会话统计（总周期数、已启动实验数、启动率、dead ends 数量）和近期失败模式 — 支持更好的决策。
- **运行时数据指纹校验**：VERIFY 阶段现在运行快速 Python 检查，加载一个数据集样本并验证不是随机噪声（检查空间相关性和常量值）。这是防御 LLM 暗中替换合成数据的最可靠手段。

**2026-04-24**
- **重大更新**：升级为 **THINK → EXECUTE → VERIFY → REFLECT** 四阶段流水线。新增 VERIFY 阶段在 REFLECT 之前反向验证每个模块（数据集、模型、训练、评估）是否真正工作正常。
- 新增 **实验验证器**（`core/verifier.py`）— 零 LLM 成本的模块级验证，带结构化诊断。检查：数据集有效性、模型 checkpoint 完整性、训练 loss 动态、指标一致性。
- 新增 **反欺骗架构**（Anti-Deception）— `ToolTrace` 记录每次工具调用的实际系统返回值，从执行结果中提取关键事实（PID、log_file、exit codes），而非信任 LLM 文本。
- 新增 **Token Plan 多供应商** 支持 — 阿里 Token Plan（qwen3.6-plus, deepseek-v3.2）和智谱 GLM Coding Plan（glm-5.1, glm-5-turbo），支持自动 failover。
- 新增 **分层模型策略** — think/reflect/idea/researcher 任务使用 strong model，code/writing 任务使用 fast model，节省 token。
- 新增 **Researcher Agent** — 专门执行深度文献搜索的 agent，配备 web_search + web_fetch + paper 工具。
- 新增 `get_paper` 工具 — 支持通过 Semantic Scholar ID 或 arXiv ID 获取论文详情。
- 新增 **自动代码清理** 触发机制 — 当根目录 .py 文件过多或日志堆积时自动触发清理。
- 新增 **数据集理解** 首轮强制扫描 — 首个 cycle 或 manifest 缺失时自动验证 data/ 目录。
- 修复 `urllib.parse` 未在 `_exec_web_search` 和 `_exec_get_paper` 中局部导入的运行时错误。
- 修复 `_consume_directive` 文件重命名碰撞 — 使用 uuid 后缀避免快速循环时的冲突。
- 修复 MEMORY_LOG 压缩不清理 active_problems 和 dead_ends 的问题。
- 修复 `_dataset_manifest_exists` 只检查文件存在不验证内容的问题。
- 修复 `_parse_log` 不处理 `★` 前缀（重大事件标记）的问题。

**2026-04-23**
- 新增**推理原则系统**（`skills/REASONING_PRINCIPLES.md`）— 6 条强制准则注入每次 THINK/REFLECT 调度，减少常见 LLM 推理错误
- 新增**3 级审计升级机制** — 自动检测重复实验问题并升级处理：L1 针对修复 → L2 强制 error-handler skill → L3 暂停等人工介入
- 新增 **error-handler skill**（`skills/error-handler/SKILL.md`）— 7 步诊断流程，系统化根因分析
- 修复 `run_shell`/`launch_experiment` 切换到 `shell=True` + 正则安全校验 — Shell 操作符（`cd`、`&&`、`|`、`> /dev/null`）现在正常工作

---

## 核心创新：零成本监控

24/7 跑 LLM Agent 很贵？Deep Researcher Agent 不会：

```
          LLM 活跃           零成本              零成本           LLM 活跃
        ┌──────────┐    ┌──────────────┐    ┌──────────┐    ┌──────────┐
        │  THINK   │    │ 训练 & 监控   │    │  VERIFY  │    │ REFLECT  │
        │ (5-10分) │    │  (数小时/天)  │    │ (1-2分)  │    │ (5-10分) │
        │          │    │              │    │          │    │          │
        │ • 分析   │    │ • 进程活着？  │    │ • 文件检查│    │ • 解析   │
        │ • 规划   │    │ • GPU利用率？ │    │ • Loss有效│    │   日志   │
        │ • 写代码 │    │ • 读日志尾部  │    │ • Ckpt正常│    │ • 对比   │
        │          │    │              │    │          │    │ • 决策   │
        │  ¥0.35   │    │   ¥0.00     │    │  ¥0.00   │    │  ¥0.20   │
        └──────────┘    └──────────────┘    └──────────┘    └──────────┘
                               ↑
                        零 LLM API 调用
                        只做文件读取 +
                        进程存活检查
```

**训练 8 小时的一个完整周期，LLM 成本约 ¥0.55，而不是 ¥350+。**

---

## 架构设计

### THINK → EXECUTE → VERIFY → REFLECT 循环

```
┌──────────────────────────────────────────────────────────────┐
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │  THINK   │→ │ EXECUTE  │→ │  VERIFY  │→ │ REFLECT  │─┐ │
│  │          │  │          │  │          │  │          │ │ │
│  │ 分析现状 │  │ Dry-run  │  │ 模块验证 │  │ 评估结果 │ │ │
│  │ 制定计划 │  │ 启动训练 │  │ 诊断故障 │  │ 对比基线 │ │ │
│  │ 做出决策 │  │ 零成本监控│  │ 零LLM成本│  │ 更新记忆 │ │ │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘ │ │
│       ↑                                                   │ │
│       └───────────────────────────────────────────────────┘ │
│                   ↻ 24/7 循环                               │
└──────────────────────────────────────────────────────────────┘
```

### Leader-Worker Agent 架构

```
              ┌───────────────┐
              │    Leader     │  周期内保持对话历史
              │   (决策者)    │  跨周期清空
              └──┬──┬──┬──┬──┘
                 │  │  │  │
         ┌───────┘  │  │  └───────┐
         ↓          ↓  ↓          ↓
   ┌──────────┐┌──────────┐┌──────────┐┌────────────┐
   │   Idea   ││   Code   ││ Writing  ││ Researcher │
   │  Agent   ││  Agent   ││  Agent   ││   Agent    │
   │ (4工具)  ││ (5工具)  ││ (3工具)  ││  (4工具)   │
   └──────────┘└──────────┘└──────────┘└────────────┘
   
   同一时间只有一个 Worker 活跃
   其他 Worker 零 token 消耗
```

### 两层记忆系统（恒定大小）

```
┌─────────────────────────────────────┐
│ 第一层: PROJECT_BRIEF.md            │
│ • 冻结的项目参考（Agent 不可修改）  │
│ • 上限 3,000 字符                   │
├─────────────────────────────────────┤
│ 第二层: MEMORY_LOG.md               │
│ • 关键成果（自动压缩到 1,200 字符）│
│ • 最近决策（只保留最近 15 条）      │
│ • 上限 4,000 字符                   │
├─────────────────────────────────────┤
│ 总计: ~5,000 字符 (~1,500 tokens)   │
│ 无论运行多久，内存大小恒定不变      │
└─────────────────────────────────────┘
```

### 推理原则系统

每次 THINK 和 REFLECT 调度都包含一个**强制推理清单**：

1. **假设** — 我在假设什么？写出来。
2. **替代方案** — 有没有更简单的方式？是不是改了太多变量？
3. **成功标准** — 具体可量化 — 不是"提升"而是"MAE < 0.35"。
4. **精准** — 每行代码改动必须追溯到实验假设。
5. **诚实** — 如果结果没达标，照实说——不粉饰。
6. **验证优先** — 如果 VERIFY 发现模块故障，在评判实验前先解决它们。

推理原则定义在 `skills/REASONING_PRINCIPLES.md`，通过 `agents.py` 中的 `_REASONING_REMINDER` 注入到每次 Leader 调度中。Code Agent 的系统提示词也包含这些原则。

### 4 级审计升级

当实验审计器重复检测到同一问题时：

| 级别 | 触发条件 | 动作 |
|------|---------|------|
| L1 | 问题出现 ≥ 3 轮 | 向下一轮 THINK 注入针对性修复指令 |
| L2 | 问题出现 ≥ 6 轮 | 强制执行 error-handler skill |
| L3 | 问题出现 ≥ 9 轮 | 暂停 Agent，写 `AGENT_STUCK.md` 等人工介入 |
| L4 | 问题出现 ≥ 12 轮 | 标记为不可修复的 dead_end，停止升级 |

这防止了 Agent 在低级错误（如数据集 key 不匹配）上无限循环。

---

## 手把手教程（从零开始）

> **完全不会？** 跟着下面每一步走，10分钟从零到跑起来。
>
> **想让AI带你装？** 把 `AI_GUIDE.md` 丢给 Claude / ChatGPT / Codex，AI会交互式地一步步教你。

### 第 0 步：检查环境

| 需要什么 | 怎么检查 |
|---------|---------|
| Python 3.10+ | `python3 --version` |
| [Claude Code](https://claude.ai/claude-code) 或命令行 | `claude --version` |
| NVIDIA GPU (至少1块) | `nvidia-smi` |
| LLM API Key (至少一个) | 见下方 |

**推荐：GLM Coding Plan**（默认 provider，性价比最高）：
```bash
# 智谱 GLM Coding Plan — 强模型 glm-5.1，快模型 glm-5
# 失败自动 failover 到阿里 Token Plan (qwen3.6-plus)
export GLM_CODING_PLAN_API_KEY="your-key-here"
# 加到 ~/.bashrc 永久生效
echo 'export GLM_CODING_PLAN_API_KEY="your-key"' >> ~/.bashrc
```

**备选：Anthropic Claude**（如需使用，修改 config.yaml 中 `provider: "anthropic"`）：
```bash
export ANTHROPIC_API_KEY="sk-ant-xxxxx"
# 加到 ~/.bashrc 永久生效
```

### 第 1 步：安装

```bash
# 克隆仓库
git clone https://github.com/hrf666666/auto_research_agent.git
cd auto_research_agent

# 安装依赖
pip install -r requirements.txt

# 安装 8 个 Claude Code 斜杠命令
python install.py

# 验证
python -m core.loop --check
```

你会看到：
```
  Deep Researcher Agent — Installer
  ========================================

    ✓ /auto-experiment
    ✓ /experiment-status
    ✓ /gpu-monitor
    ✓ /daily-papers
    ✓ /paper-analyze
    ✓ /conf-search
    ✓ /progress-report
    ✓ /obsidian-sync

  Done! 8 skills installed.
```

### 第 2 步：创建你的第一个项目

假设你想在 CIFAR-100 上训练 ResNet。先建一个项目文件夹：

```bash
mkdir ~/my-first-experiment
cd ~/my-first-experiment
```

然后写 `PROJECT_BRIEF.md` —— **这是最重要的文件**，告诉 Agent 你想干嘛：

```bash
cat > PROJECT_BRIEF.md << 'EOF'
# 目标
在 CIFAR-100 上训练 ResNet-50，测试准确率 >80%。

# 代码
Agent 从零开始写 PyTorch 训练代码。
- 用 torchvision 加载数据集（自动下载）
- 模型存到 ./checkpoints/
- 日志写到 ./logs/

# 尝试方向
- 先试基础 ResNet-50，lr=0.1，SGD，100 epochs
- 如果准确率 <75%，加 cosine annealing + warmup
- 如果 75-80%，加 mixup 或 cutout 数据增强
- 如果 >80%，目标达成

# 约束
- 只用 GPU 0
- 每次最多 100 epochs
- Batch size 128

# 当前状态
还没跑过任何实验，从零开始。
EOF
```

**写好 Brief 的技巧：**
- 目标要具体（指标 + 目标值）
- 说清楚代码/数据在哪（或者"从零写"）
- 列出约束（哪块GPU、最多多少epochs）
- 给决策树（"如果X，就试Y"）—— 像指导研究生一样指导它

### 第 3 步：启动 Agent

**方法 A：在 Claude Code 里（推荐）**

打开 Claude Code 输入：
```
/auto-experiment --project ~/my-first-experiment --gpu 0
```

**方法 B：命令行**

```bash
source ~/.bashrc  # 确保 API key 已加载
python -m core.loop \
  --project ~/my-first-experiment \
  --gpu 0 \
  --max-cycles 5    # 跑5轮就停（去掉则无限跑）
```

### 第 4 步：看它干活

Agent 现在全自动了。每一轮大概是这样的：

```
=== 第 1 轮 ===

[THINK] 读取 PROJECT_BRIEF.md...
        目标：ResNet-50 CIFAR-100，>80%
        没有历史实验，从 baseline 开始
        计划：ResNet-50, lr=0.1, SGD + momentum, 100 epochs

[EXECUTE] 创建 train.py...
          创建 config.yaml...
          Dry-run (跑2步验证)... ✓ 没报错
          启动训练：nohup python train.py --config config.yaml
          PID: 12345，日志: logs/exp001.log

[MONITOR] 训练中...（零 LLM 成本）
          15:00 — 进程活着，GPU 98%，Epoch 12/100，loss=2.34
          15:15 — 进程活着，GPU 97%，Epoch 25/100，loss=1.87
          ...
          18:00 — 进程结束，训练完成

[VERIFY] 检查模块输出...
        数据集: ✓ 加载了 50K 图像，形状 [3,32,32]，无异常值
        模型: ✓ checkpoint 97MB，权重无 NaN
        训练: ✓ loss 2.34→0.82（持续下降），GPU 已使用
        评估: ✓ 测试准确率 76.3%，在 [0,100] 范围内
        所有模块验证通过，未发现问题。

[REFLECT] 解析日志... 测试准确率 = 76.3%
          76.3% < 80% 目标
          Brief 说 75-80% 应该加数据增强
          决策：下一轮加 mixup (alpha=0.2) + cosine annealing
          记录里程碑："Exp001: ResNet-50 baseline, 76.3%"

=== 第 2 轮 ===

[THINK] 当前最佳：76.3%（Exp001）
        计划：加 mixup + cosine annealing
        ...
```

### 第 5 步：随时查看进度

Agent 跑着的时候，你可以随时看：

```bash
# 在 Claude Code 里：
/experiment-status --project ~/my-first-experiment

# 或者看 GPU：
/gpu-monitor
```

会看到：
```
# 实验状态 — my-first-experiment

## 目标

## Human-in-the-Loop 实操指南

不要把 Agent 当成替代研究者的按钮，而要把它当成你来掌舵的实验操作员。

```text
人来决定：
- 目标
- 约束
- 禁止方向
- 什么时候转向

Agent 来执行：
- 改代码
- 跑实验
- 做监控
- 写总结
```

把稳定规则写进 `PROJECT_BRIEF.md`，把临时指令写进 `HUMAN_DIRECTIVE.md`。

```md
# HUMAN_DIRECTIVE.md
- 不要改数据集。
- 先试 label smoothing 0.1，再考虑换 backbone。
- 如果连续 3 次增益低于 0.3 个点，就停止这个方向。
- 和上一个可信 baseline 对比，不要只和最近一次结果对比。
```

Case 1：做干净 ablation

```md
- 只允许改 augmentation。
- 模型、优化器、训练预算保持不变。
- 每轮结束后输出一张干净的对比表。
```

Case 2：有意识地转方向

```md
- 当前 ResNet 路线已经接近平台期。
- 只有最近 3 次都没有明显提升，才切到 ViT-B/16。
- 切换前先写一段简短理由。
```

Case 3：结果可疑时

```md
- 这次准确率提升异常大。
- 用相同 seed 重跑一次，再换一个新 seed 再跑一次。
- 两次都复现之前，不要宣称改进成立。
```

一句话：重复劳动交给 Agent，方向、解释和责任留给人。

---
ResNet-50 on CIFAR-100 → 80%+

## 进度
- 已完成轮数：3
- 当前最佳：79.1%（Exp003: ResNet-50 + mixup + cosine）
- 状态：训练中（PID 12389, GPU 0, 已跑 1.5h）

## 关键结果
[04-07 15:00] Exp001: ResNet-50 baseline, 76.3%
[04-07 18:30] Exp002: + cosine annealing, 77.8%
[04-07 22:00] Exp003: + mixup α=0.2, 79.1%   ← 最佳
```

### 第 6 步：随时介入

想换方向？三种方式：

```bash
# 方式 1：放一个文件（Agent 下一轮自动读取）
echo "别试 ResNet 了，换 ViT-B/16，lr=1e-3" \
  > ~/my-first-experiment/workspace/HUMAN_DIRECTIVE.md

# 方式 2：命令行
python -m core.loop --project ~/my-first-experiment \
  --directive "加 label smoothing 0.1"

# 方式 3：直接改记忆文件
vim ~/my-first-experiment/workspace/MEMORY_LOG.md
```

### PROJECT_BRIEF.md 示例

Brief 是你最主要的控制方式。不同场景的写法：

<details>
<summary><b>示例：微调预训练模型</b></summary>

```markdown
# 目标
用 ImageNet-21K 预训练的 ViT-B/16 在 Flowers-102 上微调。
目标：测试准确率 >95%。

# 代码
- finetune.py（已有）
- 配置：configs/vit_flowers.yaml
- 数据：/data/flowers102/（已下载）
- 预训练权重：/models/vit-b16-21k.pth

# 尝试方向
1. 先冻结backbone，只训 classifier head（10 epochs, lr=1e-2）
2. 然后全部解冻微调（30 epochs, lr=1e-4）
3. 如果低于 93%：试 layer-wise lr decay (0.65)
4. 如果高于 94%：试 test-time augmentation

# 约束
- GPU 0，batch size 64
- 按 val accuracy 保存最佳 checkpoint
```
</details>

<details>
<summary><b>示例：超参搜索</b></summary>

```markdown
# 目标
给 GAN 找最佳超参，在 CelebA-HQ 256x256 上。
目标：FID < 15。

# 代码
- train_gan.py, configs/celeba_gan.yaml
- 数据：/data/celeba_hq_256/
- 评估：eval_fid.py --real_dir /data/celeba_hq_256/val

# 搜索空间
- Learning rate: [1e-4, 2e-4, 5e-4]
- Beta1: [0.0, 0.5]
- D steps / G step: [1, 2, 5]
- Spectral norm: [是, 否]

# 策略
从 lr=2e-4, beta1=0.0, d_steps=1, spectral_norm=是 开始。
每次只改一个变量。每组跑 50K steps。

# 约束
- GPU 0-1
- 每次最多 50K steps（约4小时）
```
</details>

<details>
<summary><b>示例：排查训练问题</b></summary>

```markdown
# 目标
排查 transformer 模型为什么 epoch 20 后 loss 爆炸。
现状：loss 在 epoch 20-25 从 0.5 飙到 NaN。

# 代码
- train_transformer.py, model/transformer.py
- configs/base.yaml
- 失败日志：logs/failed_run_001.log

# 排查方向
1. 查梯度 — 加 gradient clipping (max_norm=1.0)
2. 降学习率（现在 1e-3，试 1e-4, 5e-5）
3. 查具体哪层 — 加逐层梯度日志
4. 加 warmup (1000 steps)
5. 查数据 — 有没有 NaN/Inf

# 约束
- GPU 0，每次跑 30 epochs 就够
- 每 100 steps 记录梯度范数
```
</details>

### 常见问题

<details>
<summary><b>Q：跑一天花多少钱？</b></summary>

大约 $0.08（¥0.55）。秘密：训练期间零 LLM 调用。只有 THINK 和 REFLECT 阶段（各 ~10 分钟）才花钱。
</details>

<details>
<summary><b>Q：它能改我的代码吗？</b></summary>

能。Code Agent 可以读、写、修改项目里的任何文件。它会改完之后先 dry-run 验证，没问题再启动训练。不会动受保护的文件（PROJECT_BRIEF.md、MEMORY_LOG.md）。
</details>

<details>
<summary><b>Q：Agent 跑偏了怎么办？</b></summary>

放一个指令文件：`echo "停下。回到 ResNet 方案" > workspace/HUMAN_DIRECTIVE.md`。Agent 下一轮以最高优先级读取。
</details>

<details>
<summary><b>Q：能同时跑多个项目吗？</b></summary>

能。在不同终端/tmux session 里启动多个 Agent 实例，每个指向不同项目和 GPU。
</details>

<details>
<summary><b>Q：训练崩了怎么办？</b></summary>

Monitor 检测到进程挂了，会抓取错误日志交给 REFLECT。Agent 会分析崩溃原因、修复代码、重试。
</details>

<details>
<summary><b>Q：支持 PyTorch / TensorFlow / JAX 吗？</b></summary>

都支持。Agent 本质上是启动 shell 命令 + 读日志文件，不关心你用什么框架。
</details>

---

## 一键安装（Claude Code Skills）

所有功能打包为 Claude Code 斜杠命令，**一行安装：**

```bash
python install.py
```

安装后获得 **8 个斜杠命令**：

### 核心技能

| 命令 | 功能 |
|------|------|
| `/auto-experiment` | 启动 24/7 自主 THINK→EXECUTE→VERIFY→REFLECT 实验循环 |
| `/experiment-status` | 查看实验进度：指标、周期数、GPU 状态 |
| `/gpu-monitor` | GPU 快速检查：空闲/占用、显存、利用率 |

### 研究技能

| 命令 | 功能 |
|------|------|
| `/daily-papers` | arXiv 每日推荐，自动去重 |
| `/paper-analyze <id>` | 深度论文分析 + 从 arXiv 源码提取原图 |
| `/conf-search --venue CVPR2025` | 搜索 CVPR/NeurIPS/ICML/ICLR/AAAI... |
| `/progress-report` | 生成结构化实验进度报告 |
| `/obsidian-sync` | 刷新 Obsidian 或本地进度笔记 |

### 使用示例

```bash
# 安装（一次性）
python install.py

# 在 Claude Code 中启动实验循环
/auto-experiment --project /path/to/project --gpu 0

# 查看实验进度
/experiment-status

# 查看 GPU 状态
/gpu-monitor

# Agent 训练的时候你看论文
/daily-papers --topics "vision transformer"
```

### 卸载

```bash
python install.py --uninstall
```

---

---


#
## 协议

Apache 2.0 — 详见 [LICENSE](../LICENSE)

---

<p align="center">
  <strong><i>"实验通宵运行，结果黎明到来。"</i></strong>
</p>
