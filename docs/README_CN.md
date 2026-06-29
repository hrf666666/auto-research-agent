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
THINK → EXECUTE → VERIFY → REFLECT → 循环

LLM（Leader）基于以下信息做决策：
  - PROJECT_BRIEF.md（研究目标）
  - MEMORY_LOG.md（实验历史）
  - query_memory 工具（因果链、死胡同、最佳指标）
  - 域知识（方法属性、数据约束）

系统提供：
  - 自带安全约束的工具层（命名、死胡同检查、GC）
  - 方法论门控（证伪、对照覆盖、死路签名、规范一致性）
  - 12 层 VERIFY（客观结果验证）
  - Provider 故障转移（GLM → Qwen），配额感知冷却
  - 结构化记忆（定量结果确定性写入）
  - 数据库读写契约测试（孤儿表/列在上线前即被拦截）
```

**设计原则：**
- 系统 = 硬约束（安全、生命周期、工具、记忆、方法论）
- 指引 = 研究方法论（怎么思考，不是做什么）
- LLM = 博士生大脑（设计、实现、判断、迭代）
- 每类数据只有一个真相源（例如 dead_end 只存 `memory_entries`，不在多表重复 — 见 [DATA_CONTRACT.md](DATA_CONTRACT.md)）

详见 [architecture.md](architecture.md)。

---

## 核心特性

- **query_memory 工具**：LLM 主动查询实验历史
- **方法论门控**：证伪 + 对照覆盖 + 死路签名 + 规范一致性（Phase 3/4 改革）
- **Provider 故障转移**：GLM → Qwen，配额感知冷却
- **反欺骗**：基于工具痕迹的验证（LLM 无法伪造结果）
- **确定性 GC**：不消耗配额，每周期归档临时文件
- **工具层安全**：命名规范、死胡同检查、dry-run 门控
- **dead_end 反馈回路**：被证伪的方向会被记录，重试时告警（真相源单一在 `memory_entries`）
- **数据库读写契约**：测试断言每张 SQLite 表都有写者与读者，每个 SQL 列引用都与 schema 一致 — 孤儿表/列会让构建失败
- **255+ 自动化测试**

---

## 配置

见 [config.yaml](../config.yaml)。

---

## 许可证

MIT
