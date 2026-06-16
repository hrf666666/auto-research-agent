# 阶段一冻结清单（最终版）

> 改动只做清单上的事。遇到新发现记录到 discovered_during_execution.md，不做。

## 改动项（7 项，含盲点应对）

### 1. causal_history 消费闭合

**文件**：core/loop.py:1145-1148
**改什么**：去掉 `verified` 过滤。有 verified 给 verified，没有给全部，标注状态。
**盲点应对**（#3）：给全部因果链，LLM 自己判断可信度。
**配套 prompt**（#1）：leader.md THINK 阶段加："查看 Causal History 部分，了解之前的因果假设及其验证状态。未验证的假设谨慎参考。"
**tier**（#2）：context_keys.py 里 causal_history 已是 tier=2。
**测试**：test_knowledge_consumption.py — models/ 不存在时 causal_history 非空注入。
**不许碰**：memory.py 的 get_causal_history 本身（不改存储，只改消费）。

### 2. code_review_lessons 消费闭合

**文件**：core/loop.py:1165-1185
**改什么**：models/ 不存在时用 task 文本做关键词搜索，不跳过。
**盲点应对**（#7）：确认 search_relevant_lessons 接受 task 文本作为输入。
**配套 prompt**（#1）：leader.md THINK 阶段加："查看 Code Review Lessons 部分，避免重复之前的代码错误。"
**tier**（#2）：code_review_lessons 已是 tier=2。
**测试**：test_knowledge_consumption.py — models/ 不存在时 lessons 非空注入。
**不许碰**：memory.py 的 search_relevant_lessons 本身。

### 3. experiment_value 低价值方向注入

**文件**：core/loop.py（_think 的 context 构建里新增注入点）
**改什么**：从 experiment_value 表查 voi < 0.01 的方向，注入 context。
**tier**（#2）：context_keys.py 注册 experiment_value_warn（新 key，tier=3）。
**测试**：test_knowledge_consumption.py — 有低价值记录时注入。
**不许碰**：memory.py 的 experiment_value 表结构。

### 4. 结构化实验记录

**文件**：core/loop.py（_record_cycle_outcome）、core/memory.py（log_structured_result 新方法）
**改什么**：monitor 提取的 metric 系统直接写入 MEMORY_LOG.md 一行 `[Cycle N] metric=val status=target gap=`。
**盲点应对**（#4）：行格式 `[Cycle N]` 开头，和 `_parse_log` 的 `[` 前缀匹配；compaction 保留最新。
**测试**：test_structured_memory.py — 实验完成后 MEMORY_LOG 有定量行。
**不许碰**：MEMORY_LOG.md 的现有 section 结构（Key Results / Dead Ends / Decisions）。

### 5. 目标进度追踪

**文件**：core/loop.py（新增 _check_goal_progress + _goal_achieved）、core/context_keys.py（注册 goal_progress）
**改什么**：从 brief 正则提取目标 + 从 SQLite 查最好成绩 + 计算差距 + 注入 context + 全部达成则停止。
**盲点应对**（#5）：正则 `(?:val_)?(\w+)\s*(?:<|>|<=|>=)\s*([0-9.]+)` 提取；失败时从 config goals 读。
**配套 prompt**（#1）：leader.md 加："查看 Goal Progress 部分，了解当前成绩与目标的差距。优先攻克未达成的子目标。"
**tier**（#2）：goal_progress = tier=1（每 cycle 必须看到）。
**测试**：test_goal_tracking.py — 解析目标正确 + 注入 context + 达成时返回 True。
**不许碰**：PROJECT_BRIEF.md 本身。

### 6. training_log_parser 接入

**文件**：core/monitor.py（_extract_metrics 用 parser 替代内置正则）、core/verifier.py（loss 解析用 parser）
**改什么**：monitor 和 verifier 调 training_log_parser 的 parse_loss_series/classify_loss_trend 替代自己的正则。
**盲点应对**（#6）：接入时删除 monitor/verifier 里的旧正则，不并存。
**测试**：扩展现有 test_shared_primitives.py — monitor 提取结果和 parser 一致。
**不许碰**：training_log_parser.py 本身（不改，只接入）。

### 7. model_structure_scanner 接入 analyze_model

**文件**：core/model_analyzer.py（_estimate_params_from_ast 用 scanner 替代）
**改什么**：model_analyzer 的参数估计调用 scanner.estimate_params 替代自己的版本（修 kwargs bug）。
**盲点应对**（#7）：先验证 scanner 的返回字段和 model_analyzer 的消费者兼容。
**测试**：扩展现有 test_shared_primitives.py — kwargs 风格的 Conv2d 返回非零参数。
**不许碰**：model_structure_scanner.py 本身（不改，只接入）。

---

## 文件边界汇总

| 文件 | 改动项 | 允许的改动 |
|------|--------|-----------|
| core/loop.py | 1,2,3,4,5 | 条件修复 + 新增注入点 + 新增 _check_goal_progress |
| core/context_keys.py | 3,5 | 注册 experiment_value_warn + goal_progress |
| core/memory.py | 4 | 新增 log_structured_result 方法 |
| core/monitor.py | 6 | _extract_metrics 接入 parser |
| core/verifier.py | 6 | loss 解析接入 parser |
| core/model_analyzer.py | 7 | _estimate_params_from_ast 接入 scanner |
| agents/leader.md | 1,2,5 | 加 3 句方法论指导（causal history, lessons, goal progress）|
| **不许碰** | — | agents.py, tools.py, run() enforcement 链, constraint_engine, 任何插件 |

## 验收标准

1. test_knowledge_consumption.py：causal_history + lessons + experiment_value 在 models/ 不存在时非空注入
2. test_structured_memory.py：实验完成后 MEMORY_LOG 有定量 metric 行
3. test_goal_tracking.py：目标解析正确 + goal_progress 注入 + 达成时停止
4. test_shared_primitives.py 扩展：monitor + verifier 使用 parser；scanner 修 kwargs bug
5. 全量 97+ 测试通过
6. THINK prompt diff（改前 vs 改后）确认新增了 causal_history/lessons/goal_progress 内容
