# Auto Research Agent（自动研究智能体）

> 一个全自动 AI 研究智能体——从 idea 和数据集到验证结果，端到端自主完成，
> 达到博士生研究员的水平。

[English](../README.md) | [中文](README_CN.md)

---

## 它做什么

给它一份**研究 brief**（你的假设 + 目标指标）和一个**数据集**，它会：

1. **理解数据** — 扫描数据集结构、生成 manifest、识别质量问题
2. **调研现有方法** — 搜索论文、分析跨领域可迁移的 idea
3. **设计实验** — 提出可证伪的假设、规划模型架构
4. **实现与训练** — 写代码、跑 dry-run、在 GPU 上启动训练
5. **验证结果** — 12 层自动验证（不只是"跑没跑通"）
6. **反思与迭代** — 记录结构化结果、从失败中学习、调整方向
7. **达成目标即停止** — 目标指标全部达成后自动停止

全程自主。从"这是我的 idea"到"这是你的结果"，无需人工干预。

---

## 快速开始

```bash
# 安装依赖（Python 3.11+）
pip install -r requirements.txt

# 配置 LLM provider
export GLM_CODING_PLAN_API_KEY="your-key-here"

# 在你的项目上运行
python -m core.loop --project /path/to/your/project
```

你的项目目录需要：
- `PROJECT_BRIEF.md` — 研究假设、目标指标、方法论笔记
- 数据集（智能体会自动发现并分析）
- `config.yaml` — 智能体配置（从本仓库复制并修改）

---

## 架构

```
┌──────────────────────────────────────────────────────┐
│                     run() 循环                        │
│                                                       │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐ │
│  │  THINK  │→│ EXECUTE │→│ VERIFY  │→│ REFLECT │ │
│  │ (规划)  │  │(编码+  │  │(12层   │  │(学习+ │ │
│  │         │  │ 训练)  │  │ 检查)  │  │ 记录) │ │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘ │
│       ↕             ↕            ↕            ↕      │
│  ┌─────────────────────────────────────────────────┐ │
│  │           记忆系统（SQLite + MEMORY_LOG）         │ │
│  │  实验记录 · 因果链 · 教训 · 目标进度 · 死胡同     │ │
│  └─────────────────────────────────────────────────┘ │
│       ↕             ↕            ↕            ↕      │
│  ┌─────────────────────────────────────────────────┐ │
│  │           工具层（自带安全约束）                   │ │
│  │  write_file · launch_experiment · analyze_model  │ │
│  │  search_papers · check_feasibility · GC          │ │
│  └─────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────┘
```

**设计原则：**
- **系统** = 硬约束（安全、生命周期、工具、记忆）
- **指引文件** = 研究方法论指导（怎么思考，不是做什么）
- **LLM** = 博士生的大脑（设计、实现、判断、迭代）

详见 [architecture.md](architecture.md) 获取完整模块文档和
[版本变迁历史](architecture.md#36-v17--systemic-architecture-fixes-5-root-causes-97-tests)。

---

## 核心特性

- **Provider 故障转移**：GLM-5.x → Qwen 自动切换，配额感知冷却
- **反欺骗**：基于工具痕迹的验证（LLM 无法伪造结果）
- **目标追踪**：目标指标达成后自动停止
- **IdeaScout**（可选）：通过 [research-idea-scout](https://github.com/YangyangQu/research-idea-scout) 发现跨领域可迁移的 idea
- **结构化记忆**：定量结果确定性写入（不因 LLM 改写而丢失）
- **134 个自动化测试**，覆盖调度、安全、记忆、强制执行和知识闭环

---

## 配置

见 [config.yaml](../config.yaml) 获取所有选项。关键配置段：

```yaml
goals:               # 目标指标（达成后自动停止）
  metrics:
    - key: "val_MAE"
      target: 0.20
      direction: "lower"

safety:              # 工具层安全约束
  naming: { forbidden_root_py: true }
  garbage_collection: { max_output_dirs: 10 }

idea_scout:          # 跨领域 idea 发现（默认关闭）
  enabled: false
```

---

## 许可证

MIT
