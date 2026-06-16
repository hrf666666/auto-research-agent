# 阶段二冻结清单

> 服务原则：P1（安全是工具的契约，不是一个组件）
> 盲点：#8（workspace 路径）、#9（dry-run 状态）、#10（GC 保护列表）、#11（error 后 LLM 重试）

## 改动项（4 项）

### 1. write_file 内聚命名+路径安全

**文件**：core/tools.py（`_exec_write_file` + `_resolve_workspace_path`）
**改什么**：
- write_file 在路径解析通过后，增加命名规范校验：
  - `.py` 文件在根目录 → 拒绝（"Python files must go in scripts/ or tools/"）
  - `train_*.py` 不在 `scripts/` → 拒绝（"Training scripts must be in scripts/"）
  - `debug_*`/`diag_*`/`_check_*` → 只允许在 `tools/`
- 违规返回明确 error（包含规则说明 + 建议路径）
**盲点应对**（#8）：workspace 路径基于 `_resolve_workspace_path` 已解析的 Path，不硬编码子目录名
**盲点应对**（#11）：error 信息包含"请改用 X 路径"，不是泛化错误
**测试**：test_tool_safety.py — 各类违规路径被拒绝
**不许碰**：agents.py, loop.py, run() 结构

### 2. launch_experiment 内聚 dry-run 前置检查

**文件**：core/tools.py（`_exec_launch_experiment`）
**改什么**：
- 检查 experiment_manifest.json 是否存在且最近的记录里有同脚本的 dry-run
- 如果 config `mandatory_dry_run: true`（默认 false）且没有 dry-run 记录 → 拒绝
- 默认关闭（不破坏现有行为），config 开启
**盲点应对**（#9）：experiment_manifest.json 已确认存在
**测试**：test_tool_safety.py — 无 dry-run 时被拒绝
**不许碰**：monitor.py, agents.py

### 3. 确定性垃圾回收

**文件**：core/garbage_collector.py（新建）、core/loop.py（调用点）
**改什么**：
- GC 纯 Python 确定性操作：
  - 扫描 `debug_*`/`diag_*`/`_check_*`/`dryrun_*`/`dry_run_*` → 归档到 `archive/temp/`
  - `outputs/` 超 10 个子目录 → 最旧（非 best_model）归档
- loop.py 每 cycle 结束调用 `gc.run()`（不 dispatch code agent）
- 删除 `_auto_code_cleanup` 的 LLM dispatch 路径
**盲点应对**（#10）：保护列表 = models/ datasets/ scripts/ PROJECT_BRIEF.md config.yaml；GC 只归档不删除
**测试**：test_garbage_collector.py — 临时文件被归档；保护文件不动
**不许碰**：memory.py, agents.py, verifier.py

### 4. config.yaml 新增 safety 配置段

**文件**：config.yaml
**改什么**：新增 `safety` 配置段（命名规则路径、GC 阈值、dry-run 开关、保护列表）
**不许碰**：其他 config 段

---

## 文件边界

| 文件 | 改动项 | 允许的改动 |
|------|--------|-----------|
| core/tools.py | 1, 2 | write_file/launch_experiment 内聚校验 |
| core/garbage_collector.py | 3 | 新建 |
| core/loop.py | 3 | 调用 gc.run() + 删 _auto_code_cleanup LLM 路径 |
| config.yaml | 4 | 新增 safety 段 |
| **不许碰** | — | agents.py, memory.py, verifier.py, monitor.py, context_keys.py, run() enforcement 链 |

## 验收标准
1. test_tool_safety.py：write_file 拒绝根目录 .py / train_*.py 不在 scripts/
2. test_tool_safety.py：launch_experiment dry-run 检查（config 开启时）
3. test_garbage_collector.py：临时文件归档 + 保护文件不动
4. 全量 121+ 测试通过
