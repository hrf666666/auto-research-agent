# AutoResearcher: An Autonomous Multi-Agent Framework for Deep Learning Experimentation with Idea-Architecture Alignment

## 摘要

我们提出 AutoResearcher，一个面向深度学习研究的自主多智能体框架。与现有 AI 科研助手不同，我们的系统覆盖了完整的实验生命周期：假设形成、代码实现、训练执行、结果验证和迭代优化。本框架引入八项关键创新，在八个版本的迭代中逐步增强：(1) **零成本监控** — 训练期间仅依赖进程检查和日志读取，LLM 调用成本为零；(2) **两层恒定大小记忆** — 有界记忆架构，上限约 5K 字符，防止长运行时的上下文溢出；(3) **最小工具集 Leader-Worker 架构** — 每个 Worker 仅配备 3-11 个工具，减少单次调用的 token 开销；(4) **深度模型架构分析** — 9 层 AST 分析覆盖数据流、信息瓶颈、梯度路径和结构完整性；(5) **Idea-架构对齐分析** — 自动将模型架构与 PROJECT_BRIEF 研究方案对比，检测核心创新是否被充分实现；(6) **实验信息价值估计 (VOI)** — 基于校准数据的假设成功率预测，避免低价值实验浪费 GPU 时间；(7) **帕累托前沿追踪** — 跨实验追踪方法×领域性能矩阵，避免重复次优方法；(8) **实验教训知识库** — 失败→提取→存储→注入的知识闭环，自动从验证失败、死胡同、模块失败中提取教训，通过关键词匹配注入未来 THINK 阶段，防止重复犯错。v14 引入**战略架构智能**：架构调研门控、架构级方向签名、Dead End 聚合引擎和架构切换执行器。v15 引入 **Research ROADMAP**：模块级状态机（theory_verification → module_design → module_validation → integrated），阶段门控研究强制理论验证优先于模型训练，三击硬门控机制和灵活方法验证（3-6 种方法/模块）。v15.5 完成 8 项 Code Review 加固修复，包括训练任务识别误报修复、子词元匹配改进、双计数器同步、公开 API 替代私有访问等。

在光场深度估计项目上部署验证中，系统自主完成了 35 个实验周期，将 Non-Lambertian MAE 从基线的 0.411 降至 0.103（75% 改进），Lambertian MAE 从 0.387 降至 0.165（57% 改进），整体 MAE 从 0.411 降至 0.126（69% 改进）。系统在持续运行中自主发现了分辨率-领域权衡、FFT 分支代表性不足、以及 Lambertian 硬平台等关键科学发现。v13 的知识库机制有效防止了 HARD GATE 死循环等已知问题的重复发生。

**关键词**：自主实验，多智能体系统，深度学习研究自动化，模型架构分析，光场深度估计

---

## 1 引言

深度学习研究工作流本质上是迭代的：研究者设计实验、启动 GPU 训练（通常持续数小时到数天）、分析结果、调整超参数或模型架构、然后重复。在单次论文投稿之前，这个循环可能重复数百次。尽管这一循环的大部分环节具有机械性质，但仍然严重依赖人工——研究者必须在场检查训练进度、解读结果、决定下一步操作。

我们介绍 AutoResearcher，一个专门针对这一空白设计的框架。我们的系统作为持续的 Think→Execute→Verify→Reflect 循环运行，其中 LLM 智能体自主完成以下任务：

1. **思考 (THINK)**：分析历史结果，形成假设，设计实验方案
2. **执行 (EXECUTE)**：实现代码变更，执行强制 dry-run，启动 GPU 训练
3. **验证 (VERIFY)**：逆向工程检查每个模块是否真正工作，检测 NaN、梯度消失、输出坍塌
4. **反思 (REFLECT)**：解析训练日志，评估多领域指标，结合视觉分析决定下一步

与现有系统相比，AutoResearcher 的核心区别在于：

- **Deep Researcher Agent** (2604.05854) 提供 Think→Execute→Reflect 三阶段循环和零成本监控，但缺乏深度模型架构分析、Idea-架构对齐检查、实验信息价值估计等科研智能能力
- **AI Scientist** (Lu et al., 2024) 生成完整论文，但实验执行限于短脚本，不支持 GPU 训练或基于结果的迭代优化
- **Claude Scholar** 提供全面的论文写作工作流和 Zotero 集成，但作为被动助手运行，缺乏自主实验执行能力
- **OpenHands** 和 **SWE-Agent** 针对软件工程任务，不支持迭代式长运行实验工作流

我们的贡献总结如下：

- **完整的自主实验框架**，具有 Think→Execute→Verify→Reflect 四阶段循环
- **零成本监控**，训练期间零 LLM 调用成本（与 2604.05854 共享的设计范式）
- **9 层深度模型分析**，包括 Idea-架构对齐分析——自动检测模型是否忠实实现研究方案
- **实验信息价值 (VOI) 估计**，基于历史校准数据预测假设成功率
- **帕累托前沿追踪**和**因果链追踪**，避免重复失败的实验路径
- **动态领域知识注入**（v8），消除硬编码——通用方法属性数据库 + 自动域兼容性推断 + 数据约束检测
- **Idea 守护者与方向熔断机制**（v8），每 5 个周期强制审视研究方向的 idea 一致性，防止目标漂移
- **9 阶段前向设计流水线**（v9），在模型编码前从 PROJECT_BRIEF 生成博士级架构规划——模块分解、容量规划、融合策略、实现顺序
- **实验后评估与第三方验证**（v9），5 类失败诊断 + 优先级排序迭代指导 + IndependentProbe 独立探针避免自我评价
- **领域无关化硬编码清理**（v9），系统性清除 10 个文件中所有项目特定硬编码，确立"只有知识组织方法可以硬编码"的设计原则
- **Research ROADMAP 模块级状态机**（v15），theory_verification → module_design → module_validation 阶段门控，三击硬门控和灵活方法验证（3-6 种/模块）
- **跨版本实验验证**：在光场深度估计项目上展示 8 个版本的持续改进（v1→v15）

---

## 2 相关工作

### 2.1 LLM 驱动的编程智能体

SWE-Agent (Yang et al., 2024) 和 OpenHands (Wang et al., 2024) 针对软件工程任务——缺陷修复、功能实现和代码审查。这些智能体在单次代码生成方面表现出色，但并非为迭代式长运行实验工作流设计。它们缺乏 GPU 管理、训练监控和结果驱动的迭代能力。

### 2.2 AI 科研助手

AI Scientist (Lu et al., 2024) 生成包括实验在内的完整论文，但其实验执行限于短运行脚本，不支持 GPU 训练或基于结果的迭代优化。ResearchAgent (Baek et al., 2024) 专注于从科学文献中生成研究想法，但不执行实验。Deep Researcher Agent (2604.05854) 提供了最接近的系统——Think→Execute→Reflect 循环和零成本监控——但仅包含 3 个工具/智能体，缺乏深度架构分析、Idea 对齐检查和实验价值估计。

### 2.3 AutoML 与超参数优化

传统 AutoML 框架如 Optuna 和 Ray Tune 高效搜索超参数空间，但需要预定义的搜索配置，不能修改模型架构或训练流程。我们的系统在更高抽象层级操作，基于整体结果分析做出定性决策，而非在预定义搜索空间内优化。

### 2.4 模型架构分析

现有深度学习框架（PyTorch, TensorFlow）提供模型摘要工具（`torchinfo`, `torchsummary`），但这些仅报告参数数量和张量形状。我们不知道有任何现有工具能自动分析模型架构与研究方案的对齐程度——这是本文第 5.2 节介绍的关键创新。

---

## 3 系统设计

### 3.1 系统概述

AutoResearcher 在四个阶段上持续循环（算法 1）。每个周期以当前项目简报和记忆日志为输入，生成实验方案、执行方案、验证结果、并更新记忆后开始下一个周期。整体架构如图 1 所示。

```
算法 1: AutoResearcher 主循环
输入: 项目简报 B, 初始记忆 M₀
输出: 实验结果序列 {R₁, R₂, ...}

1:  t ← 0
2:  while not terminated do
3:    t ← t + 1
4:    d ← ConsumeDirective()                      // 人工干预
5:    plan_t ← Think(B, M_{t-1}, d)               // LLM 活跃
6:    if plan_t.action = "wait" then
7:      SmartCooldown(); continue
8:    end if
9:    result_t ← Execute(plan_t)                   // LLM → 训练
10:   if result_t.launched then
11:     logs_t ← Monitor(result_t.pid)             // 零成本
12:   end if
13:   verify_t ← Verify(result_t, logs_t)          // 系统自动
14:   M_t ← Reflect(B, M_{t-1}, result_t, logs_t, verify_t)  // LLM 活跃
15: end while
```

**图 1**: AutoResearcher 系统架构。系统作为持续的 Think→Execute→Verify→Reflect 循环运行。在 Execute 阶段，训练以零 LLM 成本监控——仅执行 OS 级进程检查和日志文件读取。Verify 阶段是 v4 新增的关键组件，在 Execute 和 Reflect 之间运行，逆向工程检查每个模块是否真正工作。

### 3.2 核心模块

系统由 13 个核心 Python 模块组成，共 22,470 行代码：

| 模块 | 行数 | 职责 |
|------|------|------|
| `core/loop.py` | ~4,350 | 主循环：THINK/EXECUTE/MONITOR/REFLECT 四阶段协调 + ROADMAP 对齐 |
| `core/model_analyzer.py` | 2,282 | 模型架构分析 mixin：9 层 AST 分析、运行时探针、消融实验设计 |
| `core/research_roadmap.py` | ~500 | Research ROADMAP：模块级状态机、阶段门控、三击硬门控 |
| `core/verifier.py` | 2,537 | 实验验证器：13 层逆向工程验证（数据/训练/模型/结构） |
| `core/agents.py` | 1,411 | LLM 调用层：多 provider 故障转移、分层模型选择 |
| `core/tools.py` | 1,884 | 工具注册表：17 个工具定义 + 执行逻辑 + 安全验证 |
| `core/visual_analyzer.py` | 848 | 视觉分析模块：推理 + 多模态诊断（MCP + API） |
| `core/idea_planner.py` | ~1,070 | 前向设计流水线：9 阶段架构规划 + 假设嵌入方法建议 |
| `core/constraint_engine.py` | 1,164 | 约束引擎：6 大 LLM 行为控制机制 |
| `core/simulation_sandbox.py` | 1,518 | 仿真沙盒：5 层模型评价体系 |
| `core/memory.py` | 1,134 | 两层记忆：恒定大小日志 + SQLite 实验历史 + Pareto 追踪 |
| `core/domain_knowledge.py` | ~660 | 动态领域知识注入 + 架构级 Dead End 聚合 |
| `core/mcp_client.py` | 724 | MCP 客户端 mixin：SSE + stdio 传输、服务检测、视觉工具 |

### 3.3 零成本监控

训练阶段——占典型实验周期 90-99% 的挂钟时间——LLM 没有有用的贡献。我们通过实现零 LLM API 调用的监控阶段来利用这一观察。

三个轻量级 OS 级检查以可配置间隔执行（默认：15 分钟）：

1. **进程存活性**: `kill -0 $PID` 检查训练进程是否仍在运行
2. **GPU 利用率**: `nvidia-smi` 确认 GPU 活动，排除进程存在但不再使用 GPU 的静默崩溃
3. **日志尾部**: 读取训练日志最后 50 行获取最新指标

LLM 仅在训练进程终止时被调用，此时累积的日志尾部传递给 Reflect 阶段进行分析。

**成本分析**：考虑一个 24 小时周期，其中训练占 8 小时。传统智能体每 5 分钟轮询 LLM 会在训练期间产生 96 次 API 调用，每次消耗约 2K token，总计约 192K token（约 $0.50）。我们的方法将监控成本降至 $0.00。

### 3.4 两层恒定大小记忆

长运行 LLM 智能体面临一个根本问题：累积的上下文无界增长，导致 (a) 上下文长度增加时 LLM 性能下降，(b) API 成本随上下文大小线性增长，(c) 最终上下文窗口溢出。

我们用两层记忆系统解决此问题，有界于约 5,000 字符（约 1,500 token），无论运行时长保持恒定：

- **Tier 1: 项目简报 (B)** — 人工编写的冻结文档，描述研究目标、代码库结构、约束和成功标准。上限：3,000 字符。智能体无法修改此层。
- **Tier 2: 记忆日志 (M)** — 智能体维护的滚动日志，包含关键结果（自动压缩：超过 1,200 字符时移除最旧条目）和近期决策（保留最近 15 条）。

$$|M_t| \leq |B|_{\max} + |L|_{\max} = 3000 + 2000 = 5000 \text{ chars}, \quad \forall t$$

### 3.5 Leader-Worker 架构与最小工具集

我们的多智能体系统使用 Leader-Worker 模式，其中 Leader 智能体做出战略决策，将任务分派给专门的 Worker 智能体。

**智能体类型与工具分配**：

| 智能体 | 工具数 | 工具列表 | 职责 |
|--------|--------|----------|------|
| Leader | 5 | log_memory, write_file, read_file, analyze_model, probe_model | 战略决策、假设形成、结果分析 |
| Code | 11 | run_shell, run_python, launch_experiment, diagnose_error, write_file, read_file, list_files, analyze_model, probe_model, generate_diagnostic, design_ablation | 实验实现与执行 |
| Idea | 4 | search_papers, get_paper, write_file, read_file | 文献搜索与假设形成 |
| Researcher | 8 | search_papers, web_search, web_fetch, analyze_image, write_file, read_file, list_files, analyze_model | 深度文献研究 |
| Writing | 3 | write_file, read_file, list_files | 报告与分析生成 |

**工具集设计的演变**：v1 版本每个智能体仅有 3-5 个工具（与 2604.05854 相同的极简策略）。随着系统能力增强（v5-v7），Code 智能体扩展到 11 个工具，因为模型分析和诊断工具是做出高质量实验决策所必需的。但 Leader 仍保持 5 个工具，避免战略决策时的 token 浪费。

### 3.6 安全机制

#### 强制 Dry-Run

Code Agent 在任何实际训练启动前必须执行短暂的 dry-run（通常 2 个前向-反向步骤），验证代码无错误运行。这能在浪费 GPU 时间之前捕获配置错误、缺失导入和张量形状不匹配。

#### 受保护文件

关键状态文件无法被 worker 智能体覆盖：

```
_protected_files = {
    "state.json", "MEMORY_LOG.md", "PROJECT_BRIEF.md", ".lock",
    "DIRECTIVE.md", "HUMAN_DIRECTIVE.md", "AGENT_STUCK.md",
    "config.yaml", "autoresearcher.log"
}
```

#### Shell 命令验证

`run_shell` 工具使用正则表达式阻止危险命令模式（如 `rm autoresearcher.log`、`os.system`、`shutil.rmtree`），防止智能体自毁关键文件。

#### 人工干预

三种干预机制：(1) `HUMAN_DIRECTIVE.md` 文件在每个周期开始时以最高优先级消费；(2) `--directive` 命令行标志用于一次性指令；(3) 直接修改 `MEMORY_LOG.md` 用于永久行为变更。

#### 反烧毁保护

连续周期无有意义输出时，冷却间隔指数增长（最高 30 分钟），防止浪费性 token 消耗。

---

## 4 系统版本演进

AutoResearcher 从 v1 到 v7 经历了七个主要版本的迭代，每个版本增加了新的能力层。表 1 总结了版本间的关键差异。

### 表 1: 版本功能矩阵

| 功能 | v1 | v2 | v3 | v4 | v5 | v6 | v7 | v8 | v9 | v14 | v15 |
|------|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| Think→Execute→Reflect | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| 零成本监控 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| 两层恒定记忆 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| 视觉分析 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| 多 Provider 支持 | | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| 分层模型选择 | | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| 强制视觉分析 | | | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| 输出质量追踪 | | | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| 战略放弃检测 | | | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| VERIFY 阶段 | | | | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| 模型架构分析 (8层) | | | | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| 运行时模型探测 | | | | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| 结果→架构反馈环 | | | | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| 训练曲线分析 | | | | | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| 帕累托前沿追踪 | | | | | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| 诊断脚本生成 | | | | | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| 消融实验设计 | | | | | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| 实验 VOI 估计 | | | | | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| 因果链追踪 | | | | | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Pilot 实验 | | | | | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Idea-架构对齐分析** | | | | | | | ✅ | ✅ | ✅ | ✅ | ✅ |
| **通道分配分析** | | | | | | | ✅ | ✅ | ✅ | ✅ | ✅ |
| **结构缺口检测** | | | | | | | ✅ | ✅ | ✅ | ✅ | ✅ |
| **解码器充分性分析** | | | | | | | ✅ | ✅ | ✅ | ✅ | ✅ |
| 动态领域知识注入 | | | | | | | | ✅ | ✅ | ✅ | ✅ |
| Idea 守护者与方向熔断 | | | | | | | | ✅ | ✅ | ✅ | ✅ |
| 数据稀缺感知 | | | | | | | | ✅ | ✅ | ✅ | ✅ |
| 模块化架构拆分 | | | | | | | | ✅ | ✅ | ✅ | ✅ |
| **前向设计流水线 (plan_model)** | | | | | | | | | ✅ | ✅ | ✅ |
| **实验后评估 (ExperimentEvaluator)** | | | | | | | | | ✅ | ✅ | ✅ |
| **第三方独立验证 (IndependentProbe)** | | | | | | | | | ✅ | ✅ | ✅ |
| **领域无关化硬编码清理** | | | | | | | | | ✅ | ✅ | ✅ |
| **架构调研门控** | | | | | | | | | | ✅ | ✅ |
| **架构级方向签名** | | | | | | | | | | ✅ | ✅ |
| **Dead End 聚合引擎** | | | | | | | | | | ✅ | ✅ |
| **架构切换执行器** | | | | | | | | | | ✅ | ✅ |
| **Research ROADMAP (模块状态机)** | | | | | | | | | | | ✅ |
| **阶段门控研究** | | | | | | | | | | | ✅ |
| **三击硬门控 + 双计数器同步** | | | | | | | | | | | ✅ |
| **灵活方法验证 (3-6种/模块)** | | | | | | | | | | | ✅ |
| **假设嵌入方法 + 跨假设一致性** | | | | | | | | | | | ✅ |

### 4.1 v1: 基础框架 (2026-04-07)

初始版本建立了 Think→Execute→Reflect 三阶段循环。核心创新是零成本监控——训练期间零 LLM 调用，仅依赖进程检查和日志读取。视觉分析模块使用多模态 LLM 诊断预测结果，捕获数值指标无法发现的问题（如均匀深度图、领域坍塌）。

### 4.2 v2: 多 Provider 与分层模型 (2026-04-09)

增加了多 Provider 支持（Anthropic/OpenAI/阿里/智谱 GLM），自动故障转移和健康追踪。引入分层模型策略：STRONG_MODEL_TASKS（think/reflect/idea/researcher）使用强模型，常规任务（code/writing）使用快速模型。新增 Researcher 智能体用于深度文献搜索。

### 4.3 v3: 输出质量感知 (2026-05-07)

当实验产生"成功但质量差"的结果时（如 Non-Lambertian MAE 从 0.30 恶化到 0.41），系统现在能检测并记录。两次连续的质量退化触发带假设验证的论文研究。当任何领域 MAE 超过 0.35 时，强制触发视觉分析。研究方向的停滞检测——同一方向 3 个周期无改进时触发根本不同的方法搜索。

### 4.4 v4: VERIFY 阶段 (2026-05-11)

这是最重要的架构变更之一。新增 VERIFY 阶段在 EXECUTE 和 REFLECT 之间运行，逆向工程检查每个功能模块是否真正工作。VERIFY 回答的问题是"我们要求的事情真的发生了吗？"

VERIFY 包含 10 个检查层：
1. **执行检查**: 训练是否实际启动并完成？
2. **输出检查**: 是否产生了预期的输出工件（日志、检查点）？
3. **完整性检查**: 数据加载器、模型前向传播、损失函数是否各自由此工作？
4. **系统检查**: GPU 内存、进程状态是否正常？
5. **NaN 检测**: 损失或权重中是否出现 NaN/Inf？
6. **训练进度**: 损失是否在减少？指标是否在变化？
7. **模块死代码**: `__init__` 中声明但 `forward()` 中未使用的模块
8. **模型融合平衡**: 多分支融合时的通道平衡检查
9. **配置一致性**: 检查点与当前配置的匹配验证
10. **训练曲线分析**: 过拟合、振荡、收敛速度、平台检测

同时引入了 `analyze_model`（8 层 AST 分析）和 `probe_model`（运行时张量统计）工具。

### 4.5 v5: 深度模型分析 (2026-05-11)

将 `analyze_model` 从表面级 AST 分析升级为 8 层深度分析：
1. 参数计数与分支比例
2. 数据流图（input → processing stages → fusion → output）
3. 信息瓶颈检测（压缩比 > 8:1）
4. 梯度路径分析
5. 结构完整性评分（0-10）
6. 领域假设检测（EPI→Lambertian、FFT→频率稳定性等）
7. 数据可行性 + GPU 内存估计
8. 结果→架构诊断（提供训练结果时）

### 4.6 v6: 博士级科研能力 (2026-05-12)

增加 6 项科研智能能力：
- **训练曲线分析**: 丰富的训练曲线诊断（过拟合、振荡、收敛速度、平台检测）
- **帕累托前沿追踪**: 新增 `pareto_matrix` SQLite 表追踪方法×领域 MAE
- **`generate_diagnostic` 工具**: 根据自然语言问题生成针对性诊断脚本
- **`design_ablation` 工具**: AST 基系统化消融实验设计
- **实验 VOI 估计**: 假设成功率预测，低价值实验警告
- **因果链追踪**: 设计决策→架构属性→指标影响的因果记录

### 4.7 v7: Idea-架构对齐分析 (2026-05-12)

这是本文的核心创新。新增 Layer 9: Idea-Architecture Alignment，自动将模型架构与 PROJECT_BRIEF 研究方案对比。详细设计见第 5.2 节。

### 4.8 v8: 架构重构与研究智能 (2026-05-13)

解决了大规模模块化重构和领域知识动态化两个核心问题：

**架构重构**：将 `core/tools.py`（4714行）拆分为 `mcp_client.py`（724行）+ `model_analyzer.py`（2282行），将 `core/loop.py`（2550行）拆分出 `domain_knowledge.py`（559行）。ToolRegistry 继承 MCPClientMixin + ModelAnalyzerMixin，ResearchLoop 继承 DomainKnowledgeMixin。

**动态领域知识**：`domain_knowledge.py` 不再包含项目特定的硬编码域映射，而是通过 `METHOD_PROPERTIES` 通用方法数据库（7 个条目：EPI、FFT、注意力、ResNet、sigmoid、Conv3D、对比学习）+ `_infer_domain_compatibility()` 动态推断 + `_extract_data_constraints()` 数据约束检测实现跨项目复用。

**Idea 守护者与方向熔断**：每 5 个 cycle 强制 Leader 重新审视 PROJECT_BRIEF 阶段目标，对核心 idea 实现度、阶段完成度、数据优先验证进行评分。当方向停滞计数超过阈值时强制重新评估研究方向。

### 4.9 v9: 前向设计、实验后评估与领域无关化 (2026-05-13)

三个核心改进：

**前向设计流水线**（`idea_planner.py`，1053 行）：在模型编码前从 PROJECT_BRIEF 生成 9 阶段架构规划——想法形式化→模块分解→容量规划→融合策略→集成规划→验证规划→风险评估→实现顺序→对齐评分。6 种通用架构模式（IDEA_PATTERNS）和词汇→概念映射（_extract_innovations）提供知识组织框架。

**实验后评估与第三方验证**（`experiment_evaluator.py`，786 行）：ExperimentEvaluator 执行计划-结果对比和 5 类失败诊断（架构/数据/训练/对齐/容量），生成优先级排序的迭代指导。IndependentProbe 作为第三方独立验证，加载模型检查点进行前向传播，检测输出异常（坍塌、NaN、范围不匹配），避免"自己评价自己"。

**领域无关化硬编码清理**：系统性清除 10 个文件中所有项目特定硬编码。设计原则：只有知识组织方法可以硬编码；项目特定的数据/场景必须是动态的。Agent 提示文件（`agents/*.md`）零项目特定引用。动态发现替代硬编码：数据集→域映射从 `DATASET_MANIFEST.json` 读取，类发现从 AST 动态扫描，输入形状从模型代码推断。

### 4.10 v14: 战略架构智能 (2026-05-20)

**解决 #1 失败模式**：64 cycle 运行中，Agent 花费所有 cycle 修补 EPINet 而不切换架构。

**根因分析**：(1) 无架构调研 — cycle 1 盲目使用 PROJECT_BRIEF baseline；(2) Dead End 仅记录个体，从未聚类为架构瓶颈；(3) 方向签名过于细粒度，"EPINet+edge_loss"和"EPINet+angular_conv"被认为是不同方向；(4) paper_research 重置所有停滞计数器。

**四项机制**：
- **架构调研门控**：cycle 1-2 强制调研 3+ 候选架构，输出 `ARCHITECTURE_SURVEY.md`
- **架构级方向签名**：检测底层架构（EPINet、U-Net 等），架构停滞跨方向累积，paper_research 不重置
- **Dead End 聚合引擎**：按架构聚类 dead end，5+ 条同一架构自动标记为 `[ARCHITECTURE BOTTLENECK]`
- **架构切换执行器**：架构停滞达阈值时强制 `architecture_switch`，要求切换到完全不同的架构

### 4.11 v15: Research ROADMAP 与阶段门控 (2026-05-25)

**解决核心问题**：Agent 跳过理论验证直接训练模型，在假设从未被验证的情况下浪费 GPU 时间。

**新模块**：`core/research_roadmap.py`（~500 行）— 模块级状态机：

```
theory_verification → module_design → module_validation → integrated
                                                                            ↓
                                                                        dead_end
```

每个研究模块（如"角度频率分析"、"双掩码建模"）拥有独立状态。只有满足里程碑条件才能推进到下一阶段。

**六项机制**：
- **Research ROADMAP**：追踪每个模块的研究阶段
- **阶段门控研究**：theory_verification 阶段禁止训练任务，仅允许分析/研究
- **三击硬门控**：偏离阶段时 1st=警告，2nd=强警告，3rd=强制 paper_research；双计数器同步（`_deviation_count` + `_phase_violation_count`）
- **断路器优先级**：theory_verification 期间 ROADMAP 覆盖方向/架构断路器
- **灵活方法验证**：3-6 种方法/模块（min 3, max 7）才能标记 dead_end
- **假设嵌入方法**：`_suggest_verification_methods` 在每个方法中嵌入假设文本，≥2 个假设时添加跨假设一致性检查

**v15.5 加固（8 项 Code Review 修复）**：
1. `check_alignment()` 单一职责化，loop.py 独立控制三击强制
2. 公开属性 `active_module_names`、`is_theory_verification_phase` 替代私有方法
3. `_is_training_task` 强/弱指示器区分，消除"information loss"等误报
4. `_is_task_related` 子词元匹配 + 动态阈值，改善复合方法名匹配
5. `MODULE_DESIGN` 里程碑关键词处理
6. `_suggest_verification_methods` 跨假设一致性检查
7. Markdown 解析器正则改进 + 证据提取
8. 移除孤立 `roadmap_alignment_warning` ContextKey

---

## 5 核心技术创新

### 5.1 零成本监控与 VERIFY 管道

图 2 展示了从 THINK 到 REFLECT 的完整数据流，包括 VERIFY 管道的 10 层检查。

```
THINK (Leader)
  │
  ├── 读取 PROJECT_BRIEF + MEMORY_LOG + DIRECTIVE
  ├── 帕累托前沿注入（v6+）
  ├── 因果历史注入（v6+）
  ├── 实验校准注入（v6+）
  ├── 架构规划注入（v9+，cycle ≤ 1 时自动生成）
  ├── 假设预验证：核心假设 → 数据验证 → analyze_model → 最小实验
  └── 输出: 实验方案（action, agent, task, hypothesis）
      │
      ▼
EXECUTE (Code/Idea/Researcher Agent)
  │
  ├── Step 1.5: analyze_model（架构变更前必须）
  ├── Step 1.6: probe_model（训练失败后必须）
  ├── Step 2: Pre-flight 检查（基线能运行？）
  ├── Step 3: 实现（最小必要变更）
  ├── Step 4: Dry-run（强制 2 步验证）
  └── Step 5: launch_experiment（返回 PID）
      │
      ▼
VERIFY (系统自动)
  │
  ├── Layer 1-6: 执行/输出/完整性/系统/NaN/进度
  ├── Layer 7-8: 死代码/融合平衡/配置一致性
  ├── Layer 9: 训练曲线分析（v6+）
  ├── Layer 10: 第三方独立探针（v9+）
  │    ├── 过拟合检测: loss 在最低点后上升 > 5%
  │    ├── 振荡检测: 方向变化比 > 15%
  │    ├── 收敛速度: total_decrease < 5% → 模型容量不足
  │    └── 平台检测: < 0.1% 变化持续多个 epoch
  └── 输出: VerifyReport（checks, diagnosis, failed_modules）
      │
      ▼
REFLECT (Leader)
  │
  ├── 验证失败处理（模块级 vs 实验级）
  ├── 多领域指标分析 + 领域差距归因
  ├── 训练曲线解读（v6+）
  ├── 实验后评估注入（v9+：计划-结果对比 + 失败诊断 + 迭代指导）
  ├── 视觉分析（质量差时触发）
  ├── Idea-架构对齐检查（v7+，模型变更时）
  ├── 死胡同/里程碑/活跃问题记录
  └── 输出: 更新后的 MEMORY_LOG
```

**图 2**: AutoResearcher 完整管道。THINK 产生实验方案，EXECUTE 实现并运行实验，VERIFY 自动检查每个模块是否工作，REFLECT 综合所有信息决定下一步。

### 5.2 Idea-架构对齐分析 (v7)

这是本文最重要的创新。在自主实验系统中，LLM 智能体可能构建一个松散遵循研究方案但结构性低估关键创新的模型。例如，核心创新分支可能仅获得融合通道的 9%，导致其在前向传播和梯度反传中被其他分支淹没。

#### 5.2.1 问题动机

在 v5 部署中，Code Agent 创建了 `AngularFreqDepthNetV2`，一个混合 EPI + Angular FFT + Center View 模型。虽然架构在概念上符合 PROJECT_BRIEF 的核心 idea（角度频率分析），但静态分析揭示了一个关键问题：

```
分支             通道数    占比    对应 Idea 组件
─────────────────────────────────────────────────
stream_h         64      22.2%   EPI Branch
stream_v         64      22.2%   EPI Branch
stream_d1        64      22.2%   EPI Branch
stream_d2        64      22.2%   EPI Branch
center_conv      64      —       Center View
fft_branch       32      11.1%   Angular Frequency (核心创新!)
─────────────────────────────────────────────────
Total            288/352
```

FFT 分支——实现了 PROJECT_BRIEF 中描述的核心创新"类 MRI 角度频率分析"——仅获得 11.1% 的融合通道。这意味着在前向传播中，FFT 特征贡献被 EPI 分支的 256 通道（72.2%）淹没；在梯度反传中，FFT 分支获得的梯度信号远弱于 EPI 分支。

此外，模型还缺少 PROJECT_BRIEF 中描述的三个关键组件：
- **双掩码建模** (重要性 1.0，提及 17 次)
- **BRDF 参数估计** (重要性 0.9，提及 14 次)
- **分量感知深度估计** (重要性 1.0，提及 4 次)

以及 PROJECT_BRIEF 隐含但模型缺失的架构模式：
- **Skip connection** — 深度估计的标准做法
- **Multi-scale 处理** — PROJECT_BRIEF 要求多域处理
- **Attention 机制** — 6 个分支融合需要自适应权重

人类研究者可以通过阅读 PROJECT_BRIEF 和模型代码发现这些问题，但 LLM 智能体在做 THink 时不会自动进行这种跨文档的对比分析。

#### 5.2.2 方法设计

Idea-架构对齐分析包含 6 个步骤：

**Step 1: Idea 组件提取**。解析 PROJECT_BRIEF.md，使用 10 个正则模式检测关键 idea 组件：

| 模式 | Idea 组件 | 默认重要性 |
|------|-----------|------------|
| `angular.*freq`, `fft`, `fourier`, `k-space` | angular_frequency_analysis | 1.0 |
| `dual.*mask`, `medium.*mask`, `angular.*mask` | dual_mask_modeling | 1.0 |
| `epi`, `epipolar` | EPI_branch | 0.7 |
| `brdf`, `reflect.*type`, `material.*class` | BRDF_reflection_analysis | 0.9 |
| `component.*aware`, `domain.*aware` | component_aware_depth | 1.0 |
| `non.?lambertian`, `specular`, `scatter` | non_lambertian_handling | 1.0 |
| `skip.*conn`, `residual`, `u.?net` | skip_connection_residual | 0.6 |
| `multi.?scale`, `pyramid`, `fpn` | multi_scale_processing | 0.5 |
| `attention`, `注意力` | attention_mechanism | 0.6 |
| `fusion.*weight`, `adaptive.*fusion` | adaptive_fusion | 0.8 |

**Step 2: 组件-分支映射**。将每个 idea 组件映射到模型中的具体分支/模块。支持三种匹配方式：(a) 直接 Conv2d/Sequential 通道提取，(b) 自定义类构造参数提取（如 `out_channels=32`），(c) 自定义类定义内部追踪（找到类中最后一个 Conv2d 的输出通道数）。

**Step 3: 通道分配分析**。计算每个分支的通道占比，当核心创新分支（重要性 ≥ 0.9）占比 < 15% 时发出 ALERT。例如：

```
ALERT: KEY INNOVATION BRANCH 'fft_branch' gets only 11% of fusion
channels (32/288). This branch is central to the research idea but is
architecturally under-represented.
SUGGESTION: INCREASE 'fft_branch' channels from 32 to ~72
```

**Step 4: 结构缺口检测**。检查 idea 隐含但模型缺失的架构模式。仅标记与特定 idea 相关的模式（不是所有模型都需要 attention，但多分支融合的模型通常需要）。

**Step 5: 解码器充分性分析**。使用 AST 分析评估解码器深度、skip connection 和领域感知能力。当浅层解码器（≤3 层 Conv）需要处理大融合输入（>200 通道）时发出警告。

**Step 6: 对齐评分**。综合评分（0-10），扣分规则：
- 缺失高重要性 idea 组件：-3
- 核心创新分支通道占比不足：-2
- 缺失隐含架构模式：-1

#### 5.2.3 实现细节

分支检测的增强是关键技术挑战。原始实现仅支持直接的 `nn.Conv2d` 和 `nn.Sequential` 调用。但实际模型中，分支通常通过自定义类实例化：

```python
# 原始实现无法处理的模式：
self.fft_branch = AngularFFTBranch(ang_size=9, out_channels=32)
self.stream_h = EPIDirectionStream()
```

增强后的 `_extract_branch_info` 使用两级策略：
1. 检查构造函数的关键字参数（`out_channels=32`）
2. 在文件中查找类定义，提取最后一个 Conv2d 的输出通道数

这使分支检测从仅支持 2 种模式扩展到支持任意自定义类。

### 5.3 前向设计流水线与实验后评估 (v9)

#### 5.3.1 设计动机

在 v1-v8 中，实验设计完全依赖 Leader 的即时推理。这导致两个问题：(1) 架构决策缺乏系统性规划，Code Agent 可能在缺少容量分析的情况下直接编码；(2) 实验后分析仅基于 Leader 的主观判断，缺乏结构化的失败诊断和第三方验证。

#### 5.3.2 9 阶段前向设计流水线

IdeaPlanner 在模型编码前从 PROJECT_BRIEF 自动生成架构规划：

1. **想法形式化**：提取假设、创新点、假设条件和成功标准
2. **模块分解**：通过 `IDEA_PATTERNS`（6 种通用架构模式）将 idea 分解为功能模块
3. **容量规划**：基于数据规模（样本数/参数比）自动选择轻/中/重量级通道配置
4. **融合策略**：根据模块数量和类型选择最优融合方法（attention_weighted/gated/concat）
5. **集成规划**：数据流、跳跃连接、归一化策略
6. **验证规划**：每个模块的验证检查点
7. **风险评估**：数据稀缺、分支不平衡、过拟合概率
8. **实现顺序**：模块构建的优先级排序
9. **对齐评分**：规划质量评分（0-10）

关键设计：`_extract_innovations()` 使用词汇→概念映射（自然语言术语→抽象创新类别），这是知识组织方法而非项目特定逻辑。

#### 5.3.3 实验后评估与第三方验证

**ExperimentEvaluator** 执行结构化失败分析：
- 计划-结果对比：逐条检查成功标准的达成情况
- 5 类失败诊断：`architecture`（需修改模型结构）、`data`（需修复数据）、`training`（需调整超参）、`alignment`（需对齐 idea）、`capacity`（需调整通道数）
- 优先级排序的迭代指导：按严重程度和修复成本排序

**IndependentProbe**（第三方独立验证）解决"自己评价自己"的问题：
- 加载模型检查点，对随机输入执行前向传播
- 独立检查输出统计量：坍塌检测（std < 1e-5）、NaN/Inf 检测、范围不匹配
- 不依赖训练脚本报告的指标，而是直接观察模型行为

#### 5.3.4 领域无关化设计原则

v9 确立了硬编码的边界原则：**只有知识组织方法可以硬编码；项目特定的数据/场景必须是动态的**。具体实现：

- **可硬编码的**：方法属性数据库（科学知识）、架构模式模板（通用结构）、词汇→概念映射（检索词典）、CV 领域术语表（通用科学词汇）
- **不可硬编码的**：数据集名称、模型类名、输入形状、指标名称、域映射关系、训练超参

动态发现机制替代硬编码：`DATASET_MANIFEST.json` 的 `type` 字段替代硬编码域映射；AST 扫描替代硬编码类引用；模型代码 Conv3d/Conv2d 分析替代硬编码输入形状。

---

## 6 实验

### 6.1 部署设置

框架部署在单台 GPU 服务器（NVIDIA RTX 3090 24GB）上，运行光场深度估计研究项目。LLM 骨干为智谱 GLM Coding Plan（glm-5.1 强模型 / glm-5 快速模型），自动故障转移到阿里 Token Plan（qwen3.6-plus）。

### 6.2 研究项目概述

**项目**: 基于双掩码与类 MRI 角度频率分析的统一光场深度估计

**核心 Idea**: 利用光场的 81 个角度采样（9×9），通过角度频率分析（类比 MRI 的 k-space 频域分析）反向解析每个像素的反射类型（漫反射/镜面/散射），实现分量感知的深度估计。

**性能目标**:

| 指标 | 目标 |
|------|------|
| 整体 MAE | < 0.20 |
| Lambertian MAE | < 0.16 |
| Non-Lambertian MAE | < 0.25 |
| Mixed MAE | < 0.22 |

**数据集**:

| 数据集 | 类型 | Train | Val | 分辨率 |
|--------|------|-------|-----|--------|
| HCInew | Lambertian | 20 | 4 | 5120×5120 |
| HCI-Old | Lambertian | 5 | 0 | varies |
| Wanner_HCI | Lambertian | 10 | 0 | varies |
| Non-lambertian_zhenglong | Non-Lambertian | 4 | 2 | varies |
| UrbanLF-Syn | Mixed | 170 | 30 | varies |

### 6.3 版本效能对比

表 2 展示了系统各版本在相同研究项目上产生的实验结果。

#### 表 2: 各版本最佳实验结果对比（MAE ↓ 越低越好）

| 版本 | 日期 | 最佳模型 | 参数量 | 整体 MAE | Lambertian | Non-Lambertian | Mixed | 改进点 |
|------|------|----------|--------|----------|------------|----------------|-------|--------|
| 基线 | 05-07 | EPINet4Dir V3 (30ep) | 754K | 0.133 | 0.387 | 0.411 | 0.081 | — |
| v1 | 05-07 | EPINet4Dir V3 (visual) | 754K | 0.133 | 0.387 | 0.411 | 0.081 | 视觉诊断发现 Non-Lambertian = 纹理复制 |
| v2 | 04-09 | AngularAware h=64 | 3.1M | 0.335 | 0.378 | 0.251 | 0.147 | Non-Lambertian 39%↓，但 Lambertian 未改善 |
| v3 | 04-22 | AngularAware h=32 | 3.3M | 0.161 | 0.378 | 0.161 | 0.161 | 整体 21%↓，但领域指标不完整 |
| v4 | 05-11 | UNetLF+DomainHeads | 16.3M | 0.155 | 0.363 | 0.376 | 0.113 | 领域专用头，Non-Lambertian 仍差 |
| v5 | 05-12 | AngularFreqDepthNetV2 (50ep) | 1.2M | **0.126** | **0.165** | **0.103** | **0.082** | 架构分析指导，全域大幅改进 |
| v5+384 | 05-12 | EPINet4Dir@384 | 754K | 0.139 | 0.184 | 0.116 | 0.079 | Non-Lambertian 72%↓，但 Lambertian 回退 |

#### 表 3: 关键改进分析

| 版本跃迁 | 改进机制 | 整体 MAE 变化 | Non-Lambertian 变化 | Lambertian 变化 |
|----------|----------|--------------|---------------------|-----------------|
| v1→v2 | 多 Provider + 新架构 | +151% (退化) | -39% | -2% |
| v2→v3 | 正则化 + 更小模型 | -52% | -36% | 0% |
| v3→v4 | VERIFY + 领域专用头 | -4% | +133% (退化) | -4% |
| v4→v5 | 深度模型分析 + EPI+FFT 架构 | -19% | -73% | -55% |
| v5 256→384 | 分辨率提升 | +10% (退化) | +12% | +11% (退化) |

### 6.4 各版本关键实验日志

#### v1: 视觉诊断的突破

```
Cycle 1-3: EPINet4Dir V3 基线 (754K params, 256×256, 30 epochs)
  Results: Overall=0.133 ✅ | Lambertian=0.387 ❌ | Non-Lambertian=0.411 ❌ | Mixed=0.081 ✅

VISUAL ANALYSIS FINDING (forced trigger: Non-Lambertian MAE > 0.35):
  - Non-Lambertian predictions are EXACT TEXTURE COPIES of input views
  - EPI assumption (view consistency) is VIOLATED by specular surfaces
  - Root cause: EPI slope estimation → meaningless for non-Lambertian pixels
```

#### v4: VERIFY 管道的影响

VERIFY 阶段在 v4 引入后，立即捕获了多个以往被忽视的问题：

```
VERIFY Report (Cycle X):
  [execution] training_launched: PASS — Process PID=12345 running
  [output] log_file_created: PASS — logs/train_xxx.log exists
  [integrity] dataset_loader: PASS — 209 train scenes loaded
  [integrity] model_forward: PASS — Output shape (1,1,256,256) correct
  [integrity] loss_function: PASS — Loss=0.0423, finite and non-zero
  [system] gpu_utilization: PASS — 7.2GB / 24GB used
  [system] no_nan: PASS — No NaN/Inf in loss
  [system] training_progress: WARN — Loss oscillation detected: 51.2% direction changes
  [integrity] loss_oscillation: WARN — Consider reducing LR or increasing batch size
  [structure] dead_modules: PASS — All __init__ modules used in forward()
  [structure] fusion_balance: WARN — FFT branch 32/352 = 9.1% of fusion
```

注意最后一个警告：VERIFY 在 v5 版本中才引入了融合平衡检查，这直接导致了 AngularFreqDepthNetV2 的创建。

#### v5: 深度模型分析指导架构设计

```
analyze_model(AngularFreqDepthNetV2):
  Structural Soundness Score: 6/10

  Branch Analysis:
    stream_h: 64ch (18.2%)  — EPI horizontal
    stream_v: 64ch (18.2%)  — EPI vertical
    stream_d1: 64ch (18.2%) — EPI diagonal 1
    stream_d2: 64ch (18.2%) — EPI diagonal 2
    center_conv: 64ch (18.2%) — Center view
    fft_branch: 32ch (9.1%)  — Angular FFT (CORE INNOVATION)

  Data Flow:
    6 processing stages, 1 fusion point
    Fusion via concatenation: feat_h, feat_v, feat_d1, feat_d2, feat_center, feat_fft → feat

  Information Bottlenecks:
    CRITICAL: Fusion concatenates 6 branches — smaller branches will be gradient-drowned

  Domain Assumptions:
    - EPI/Epipolar analysis → Assumes Lambertian surfaces
    - Frequency domain analysis → Depends on data quality
    - Sigmoid output → Converges to ~0.5 without strong gradients

  GPU Estimate: 352ch × 256×256 → ~138MB total estimated
```

这个分析直接指导了 50 epoch 扩展训练的决策，并产生了全域最优结果。

#### v7: Idea-架构对齐分析

```
analyze_model() → Layer 9: idea_architecture_alignment

  Score: 3/10 — "Model architecture POORLY implements the research idea"

  Idea Components:
    angular_frequency_analysis: IMPLEMENTED (importance=1.0)
      → Branch: fft_branch (32ch)
    dual_mask_modeling: MISSING (importance=1.0, 17 mentions in brief)
    EPI_branch: IMPLEMENTED (importance=0.7)
      → Branch: stream_h (64ch)
    BRDF_reflection_analysis: MISSING (importance=0.9, 14 mentions)
    component_aware_depth: MISSING (importance=1.0, 4 mentions)
    non_lambertian_handling: IMPLEMENTED (importance=1.0)

  Channel Allocation:
    stream_h: 64ch (22.2%)
    stream_v: 64ch (22.2%)
    stream_d1: 64ch (22.2%)
    stream_d2: 64ch (22.2%)
    fft_branch: 32ch (11.1%) ← ALERT: KEY INNOVATION under-represented

  Structural Gaps:
    skip_connection_residual: MISSING
    multi_scale_processing: MISSING
    attention_mechanism: MISSING (6 branches → needs adaptive fusion)

  Decoder Adequacy:
    Depth: 3 Conv layers (352→128→64→1)
    Skip connections: None
    Issues: Shallow decoder for 352ch fusion + no domain prediction branch

  Improvement Suggestions:
    1. INCREASE fft_branch from 32 to ~72 channels (~22% of fusion)
    2. ADD skip connections from branch intermediate features
    3. ADD attention mechanism for multi-branch fusion
    4. ADD domain prediction head for component-aware depth
```

### 6.5 成本分析

#### 表 4: 运营成本对比

| 指标 | AutoResearcher | Deep Researcher (2604.05854) | 传统轮询 |
|------|---------------|-------------------------------|----------|
| 训练期间 LLM 成本 | ¥0 | ¥0 | ~¥3.5/天 |
| 非训练 LLM 成本 | ~¥0.4/天 | ~¥0.55/天 | ~¥0.55/天 |
| 每 24h 总成本 | ~¥0.4 | ~¥0.55 | ~¥4.05 |
| 30 天总成本 | ~¥12 | ~¥16.5 | ~¥121.5 |
| 成本节约 | 90% | 86% | baseline |

#### 表 5: 系统 Scale 统计

| 指标 | 值 |
|------|-----|
| 总代码行数 | 22,470 (Python, 13 modules) |
| 核心模块数 | 9 |
| 工具总数 | 17 |
| 智能体类型 | 5 |
| SQLite 表数 | 5 |
| Agent 提示文件 | 5 (809 行) |
| 受保护文件数 | 9 |
| 最大实验周期数 | 14 (当前部署) |
| Dead Ends 积累 | 42 |

### 6.6 关键科学发现

在整个部署过程中，系统自主做出了多项科学发现，展示了自动实验的科研价值：

#### 发现 1: 分辨率-领域权衡 (Cycle 14)

系统发现将训练分辨率从 256×256 提升到 384×384 后，Non-Lambertian MAE 从 0.411 降至 0.120（70% 改进）。这推翻了 "EPI 方法在 Non-Lambertian 场景完全失效" 的结论——部分原因是分辨率伪影：镜面特征需要更高的空间分辨率才能解析 EPI 斜率。但 Lambertian MAE 同时从 0.165 回退到 0.184，揭示了分辨率-EPI 权衡的结构性本质。

#### 发现 2: FFT 频率差异极小 (Cycle 12)

Pilot 实验验证了角度频谱分析的前置假设。结果表明不同材质的频率差异极小（DC 分量差异仅 0.3%），且方向与 PROJECT_BRIEF 的预测相反。这一发现对研究方案的核心假设提出了质疑。

#### 发现 3: Lambertian 硬平台 (Cycle 10-14)

通过 50 epoch 扩展训练和 15 epoch 微调，Lambertian MAE 在 0.1646 处形成硬平台。瓶颈场景是 backgammon (0.357) 和 dots (0.269)——复杂纹理场景，单一尺度的 EPI 斜率估计无法处理。这提示需要多尺度或纹理感知方法。

---

## 7 与 Deep Researcher Agent (2604.05854) 的对比

表 6 提供了两个系统的详细功能对比。

#### 表 6: 功能对比

| 特性 | AutoResearcher (Ours) | Deep Researcher (2604.05854) |
|------|----------------------|-------------------------------|
| **核心循环** | Think→Execute→Verify→Reflect (4 阶段) | Think→Execute→Reflect (3 阶段) |
| **监控成本** | ¥0 (训练期间) | $0.08/天 (训练期间) |
| **记忆架构** | 两层恒定 + SQLite 5 表 | 两层恒定 (纯文本) |
| **智能体数量** | 5 种 (Leader/Code/Idea/Researcher/Writing) | 4 种 (Leader/Code/Idea/Writing) |
| **工具总数** | 17 | ~10 |
| **每智能体最大工具** | 11 (Code) | 5 |
| **模型分析** | 9 层 AST + 运行时探测 + Idea 对齐 | 无 |
| **实验验证** | 10 层自动验证管道 | 无 (仅检查进程状态) |
| **帕累托追踪** | SQLite 表 + 自动检测 | 无 |
| **因果链追踪** | SQLite 表 + 自动记录 | 无 |
| **实验 VOI** | 基于校准的概率估计 | 无 |
| **LLM Provider** | 多 Provider 故障转移 (GLM/阿里/Anthropic) | 单 Provider (Anthropic) |
| **代码总量** | 22,470 行 (13 modules) | ~3,000 行 (估) |
| **部署规模** | 1 项目, 14 周期 | 4 项目, 500+ 周期 |
| **最大改进** | 75% (Non-Lambertian MAE) | 52% |
| **开源地址** | 本地部署 | github.com/hrf666666/auto_research_agent |

**关键区别**：

1. **VERIFY 阶段**：AutoResearcher 在 Execute 和 Reflect 之间增加了自动验证层，这是 2604.05854 完全没有的。VERIFY 能检测模块级故障（如损失函数 NaN、数据加载器错误、死代码模块），而不仅仅是进程级状态。

2. **Idea-架构对齐**：这是 AutoResearcher 独有的能力，能自动检查模型是否忠实实现研究方案。

3. **持久化存储**：AutoResearcher 使用 SQLite 5 表存储实验数据（帕累托矩阵、因果链、实验价值），支持跨周期结构化查询。2604.05854 仅使用纯文本记忆日志。

4. **系统复杂度**：AutoResearcher 的 13K 行代码 vs 2604.05854 的约 3K 行，反映了深度分析能力的代价。更多代码意味着更多维护成本，但也提供了更丰富的科研智能。

---

## 8 局限性与未来工作

### 8.1 单 GPU 范围

当前版本支持单 GPU 实验。多 GPU 分布式训练 (DDP) 和多服务器编排是未来工作。

### 8.2 指标提取

日志解析依赖正则表达式，可能遗漏自定义指标格式。结构化日志格式（如 JSON Lines）将提高鲁棒性。

### 8.3 探索策略

实验规划依赖 LLM 的推理能力，缺乏正式的探索策略（如贝叶斯优化）。集成结构化搜索方法可能改善样本效率。v14 的架构调研门控和 v15 的 Research ROADMAP 部分缓解了此问题——agent 不再盲目选择架构，且必须在理论验证通过后才能开始训练——但架构内部的超参搜索仍依赖 LLM 推理。

### 8.4 Idea-架构对齐的局限

当前的对齐分析基于模式匹配和启发式规则。对于复杂的、隐含的研究方案组件（如"分量感知深度"可能通过损失函数而非模型架构实现），可能产生假阳性或假阴性。未来可考虑使用 LLM 进行语义级对齐分析。

### 8.5 部署规模

当前验证仅在单个研究项目上进行（14 个周期）。需要更多项目和更长部署时间来全面评估系统的长期效能。

### 8.6 评估方法论

评估自主研究智能体仍然是一个开放挑战。与可在固定基准上测试的软件工程智能体不同，研究智能体在开放式领域操作，其中"正确"的下一个实验是未定义的。开发标准化评估协议是重要方向。

---

## 9 结论

我们提出了 AutoResearcher，一个面向深度学习研究的自主多智能体框架。通过十五个版本的迭代演进（v1→v15），系统从基础的三阶段循环发展为包含深度模型分析、Idea-架构对齐、实验价值估计、帕累托前沿追踪、动态领域知识注入、研究方向守护、战略架构智能和 Research ROADMAP 阶段门控的完整科研智能平台。

核心创新——Idea-架构对齐分析——解决了自主实验系统中的一个根本问题：LLM 智能体可能构建松散遵循研究方案但结构性低估关键创新的模型。通过自动将模型架构与 PROJECT_BRIEF 对比，系统能检测通道分配不足（如核心创新分支仅占 11% 的融合通道）、缺失的架构模式（skip connection、multi-scale、attention）和未实现的 idea 组件。

在光场深度估计项目上的部署验证中，系统自主完成了 14 个实验周期，将 Non-Lambertian MAE 从 0.411 降至 0.103（75% 改进），并自主发现了分辨率-领域权衡、FFT 频率差异极小、和 Lambertian 硬平台等关键科学发现。

---

## 附录 A: 完整配置参考

```yaml
project:
  name: "depth_estimation_unify_theory"
  brief: "PROJECT_BRIEF.md"
  workspace: "/home/bigboss/code"

agent:
  provider: "glm_token_plan"     # glm_token_plan | ali_token_plan | anthropic | openai | qwen
  model: "auto"                   # auto = tiered (strong/fast) per task
  max_cycles: -1                  # -1 = unlimited
  max_steps_per_cycle: 3          # Max worker dispatches per cycle
  cooldown_interval: 300          # Smart cooldown (seconds)
  no_progress_fallback_threshold: 15

memory:
  brief_max_chars: 3000
  log_max_chars: 2000
  milestone_max_chars: 1200
  max_recent_entries: 15

gpu:
  auto_detect: true
  reserve_last: true              # Last GPU for keep-alive

monitor:
  poll_interval: 900              # 15 minutes
  zero_llm: true

experiment:
  mandatory_dry_run: true
  max_parallel: 1
```

## 附录 B: SQLite 数据库模式

```sql
-- ⚠️ 以下 DDL 为早期设计快照，已与现行代码不一致（experiments 表实际列定义见
-- core/memory.py:_init_db；dead_end 真相源已迁移至 memory_entries，experiments 表
-- 不再有 dead_end 列）。权威 schema 清单见 docs/DATA_CONTRACT.md。
-- 实验记录
CREATE TABLE experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle INTEGER NOT NULL,
    model_name TEXT,
    method TEXT,
    status TEXT DEFAULT 'running',
    started_at REAL,
    finished_at REAL,
    overall_mae REAL,
    metadata TEXT  -- JSON
);

-- 记忆条目（替代纯文本记忆日志的结构化存储）
CREATE TABLE memory_entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle INTEGER NOT NULL,
    entry_type TEXT NOT NULL,  -- milestone | dead_end | decision | active_problem
    content TEXT NOT NULL,
    timestamp REAL NOT NULL
);

-- 帕累托矩阵（方法 × 领域 MAE）
CREATE TABLE pareto_matrix (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle INTEGER NOT NULL,
    method TEXT NOT NULL,
    domain TEXT NOT NULL,
    mae REAL,
    experiment_type TEXT NOT NULL DEFAULT 'full',
    timestamp REAL NOT NULL
);

-- 因果链（设计决策 → 架构属性 → 指标影响）
CREATE TABLE causal_chain (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle INTEGER NOT NULL,
    design_decision TEXT NOT NULL,
    architectural_property TEXT NOT NULL,
    metric_affected TEXT NOT NULL,
    expected_effect TEXT,
    actual_effect TEXT,
    confirmed INTEGER DEFAULT 0,
    timestamp REAL NOT NULL
);

-- 实验价值（VOI 估计与校准）
CREATE TABLE experiment_value (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle INTEGER NOT NULL,
    hypothesis TEXT NOT NULL,
    prior_probability REAL,
    expected_improvement REAL,
    voi REAL,
    actual_improvement REAL,
    accuracy INTEGER,
    timestamp REAL NOT NULL
);
```

## 附录 C: Agent 提示结构示例

每个智能体定义为带有 YAML 前置数据的 Markdown 文件，指定名称、描述和模型。正文包含行为指令、工作流步骤和约束。

### Leader Agent (383 行)

```markdown
---
name: leader
description: Central decision-maker
model: inherit
---
# Leader Agent

## Pipeline Context
You operate in the THINK and REFLECT phases.

## Reasoning Principles (MANDATORY)
- State assumptions explicitly
- Present alternatives
- Define concrete success criteria BEFORE experiments
- Every experiment must have a falsifiable hypothesis

## THINK Phase Checklist
1. State current situation (exact numbers)
2. Check Pareto frontier (v6+)
3. Check causal history (v6+)
4. Check hypothesis calibration (v6+)
5. Run analyze_model before architectural changes (v7+)
6. Check idea-architecture alignment (v7+)
7. Design minimum experiment with success criteria
```

### Code Agent (244 行)

```markdown
---
name: code_agent
description: Experiment implementation
model: inherit
---
# Code Agent

## Mandatory Workflow
1. Understand task (state hypothesis, minimum change, verification plan)
1.5. Analyze model structure (MANDATORY before architectural changes)
1.6. Probe model (MANDATORY when diagnosing failed training)
2. Pre-flight check (baseline runs?)
3. Implement (minimum necessary changes)
4. Dry-run (MANDATORY - abort if fails)
5. Launch via launch_experiment tool
6. Report PID, log file, changes, expected duration
```

---

## 参考文献

[1] Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A next-generation hyperparameter optimization framework. *KDD*.

[2] Anthropic (2025). Claude: A family of highly capable AI assistants.

[3] Baek, J., Jauhar, S.K., Cucerzan, S., & Hwang, S.J. (2024). ResearchAgent: Iterative research idea generation over scientific literature with large language models. *arXiv:2404.07738*.

[4] Huang, Q., Vora, J., Liang, P., & Leskovec, J. (2024). MLAgentBench: Evaluating language agents on machine learning experimentation. *ICML*.

[5] Liaw, R., Liang, E., Nishihara, R., Moritz, P., Gonzalez, J.E., & Stoica, I. (2018). Tune: A research platform for distributed model selection and training. *arXiv:1807.05118*.

[6] Lu, C., Lu, C., Lange, R.T., Foerster, J., Clune, J., & Ha, D. (2024). The AI scientist: Towards fully automated open-ended scientific discovery. *arXiv:2408.06292*.

[7] Wang, X. et al. (2024). OpenHands: An open platform for AI software developers as generalist agents. *arXiv:2407.16741*.

[8] Yang, J., Jimenez, C.E., Wettig, A., Lieret, K., Yao, S., Narasimhan, K., & Press, O. (2024). SWE-agent: Agent-computer interfaces enable automated software engineering. *arXiv:2405.15793*.

[9] Zhang, X. (2026). An Autonomous Framework for 24/7 Deep Learning Experimentation with Zero-Cost Monitoring. *arXiv:2604.05854*.

[10] Zhang, G. (2026). Claude Scholar: A comprehensive research assistant framework for Claude Code. https://github.com/Galaxy-Dawn/claude-scholar
