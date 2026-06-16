# Auto Research Agent（自动研究智能体）

> 一个全自动 AI 研究智能体——从 idea 和数据集到验证结果，端到端自主完成。

[English](../README.md) | [中文](README_CN.md)

---

## 它做什么

给它一份**研究 brief** 和一个**数据集**，它会自主完成：理解数据、调研方法、
设计实验、实现与训练模型、验证结果、反思与迭代。

全程自主。从"这是我的 idea"到"这是你的结果"，无需人工干预。

---

## 快速开始

```bash
pip install -r requirements.txt
export GLM_CODING_PLAN_API_KEY="your-key-here"
python -m core.loop --project /path/to/your/project --max-cycles 10
```

你的项目需要：`PROJECT_BRIEF.md`、数据集、`config.yaml`。

---

## 架构

```
THINK → EXECUTE → VERIFY → REFLECT → 循环（直到达成目标或达到最大周期数）

记忆系统（SQLite + MEMORY_LOG）  ← 结构化实验记录
工具层（自带安全约束）            ← write_file, launch_experiment, GC, analyze_model
```

**设计原则：系统 = 硬约束。指引 = 方法论。LLM = 博士生大脑。**

详见 [architecture.md](architecture.md)。

---

## 核心特性

- Provider 故障转移（GLM → Qwen），配额感知冷却
- 反欺骗工具痕迹验证
- 确定性垃圾回收（不消耗 LLM 配额）
- 工具层命名规范强制
- 99 个自动化测试

## 许可证

MIT
