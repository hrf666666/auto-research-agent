# 阶段三冻结清单

> 服务原则：P3（LLM 自主）、P5（prompt 定义框架）
> 最大的一步：删研究决策 enforcement + 简化 run()

## 策略：渐进式，先删 0 次触发的，再简化有触发的

### 1. 删除 0 次触发的研究决策 enforcement

**删除**（数据证明 0 次触发）：
- `_apply_no_progress_fallback`（159行）
- `_extract_direction_signature`（~43行）
- `_extract_architecture_name`（~17行）
- `_analyze_architecture_dead_ends`（~48行）
- 9 个停滞计数器的初始化（`_no_progress_streak` 等）
- 7 个 circuit breaker context 注入（`idea_guardian_check` 等）

**保留**：
- `_check_phase_blocked`（安全约束，0 次但防止过早训练）
- `_enforce_roadmap_alignment`（偏离检测，1 次触发）
- `_enforce_launch_after_failure`（5 次触发，有效）
- `constraint_engine` FORBIDDEN 检查

### 2. run() enforcement 链简化

当前链：`_arbitrate_cycle → phase_gate → roadmap → launch_enforce → audit_enforce`
改为：`_arbitrate_cycle → phase_gate（安全）→ roadmap（偏离）`
launch/audit enforcement 已被 arbiter 内部调用，不在 run() 重复。

### 3. idea_guardian 等变 prompt 方法论

leader.md 加一段方法论指导（替代被删的 circuit breaker）。

### 4. 插件变工具（sandbox → check_feasibility）

提取 sandbox 的 check_feasibility 作为一个工具注册给 code agent。

## 文件边界

| 文件 | 改动 | 不许碰 |
|------|------|--------|
| core/loop.py | 删方法 + 简化 run() + 删 context 注入 | agents.py |
| agents/leader.md | 加方法论段 | code_agent.md |
| core/tools.py | 注册 check_feasibility 工具 | write_file/launch_experiment |
| **不许碰** | — | memory.py, verifier.py, monitor.py, constraint_engine.py |

## 验收标准
1. loop.py 行数减少 > 400 行
2. 全量 134+ 测试通过
3. 删除的方法不被任何残留代码引用
4. run() enforcement 从 7 层降到 3 层
