# 阶段四冻结清单

> 服务原则：P1（安全）、P5（prompt 精简）、P5（代码量降）
> 按风险从低到高执行

## 改动项

### 1. 去领域硬编码（低风险）
- memory.py: metric key 从 config/DATASET_MANIFEST 读取
- loop.py: val_MAE 提取从 config goals.metrics 读取 key
### 2. prompt 精简（低风险）
- leader.md: 删被禁用机制的规则，保留方法论框架
- code_agent.md: 命名规则移到工具层后不重复
### 3. loop.py 提取 code_review（中风险）
- 12 个 code_review 方法移到 core/code_review.py
### 4. 非阻塞执行（高风险，最后做）
- monitor 非阻塞化
- 实验注册表

## 文件边界
| 文件 | 改动项 |
|------|--------|
| core/memory.py | 1 |
| core/loop.py | 1, 3, 4 |
| agents/leader.md, code_agent.md | 2 |
| core/code_review.py（新建）| 3 |
| core/monitor.py | 4 |
