---
name: code-review
description: "Code review for ML projects — validates architecture consistency, checkpoint compatibility, and code-version alignment"
argument-hint: "[--project <path>] [--focus <arch|inference|tests>]"
---

# /code-review

Code review skill for machine learning projects. Validates that **code changes don't break
existing checkpoints, training-inference alignment is maintained, and model architecture is
version-consistent**.

## When to Trigger

**Mandatory** — before every dry-run and before committing:
```
Before dry-run → /code-review --focus inference
Before commit  → /code-review --focus arch
```

Also triggered automatically when:
- A new checkpoint is saved (validate it can be reloaded)
- Model architecture is modified
- Inference script is modified

## Focus Areas

### `--focus arch` — Architecture Version Consistency

**Goal**: 确保每次模型架构改动都被追踪，不会出现"训练一个版本、推理另一个版本"。

检查清单：
1. **Checkpoint 是否有元数据记录架构版本？** 每次保存 checkpoint 必须包含：
   ```python
   # checkpoint["metadata"] 必须包含：
   {
       "architecture_version": "v1.2",  # 与代码中的 VERSION 同步
       "model_class": "DualMaskDepthModel",
       "commit_hash": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True).stdout.decode().strip(),
       "training_script": "train.py",
       "date": datetime.now().isoformat(),
   }
   ```

2. **推理时加载 checkpoint 的代码是否验证了版本兼容性？**
   ```python
   # 推理脚本加载 checkpoint 时必须验证：
   assert checkpoint["metadata"]["architecture_version"] == current_version, \
       f"Checkpoint v{ckpt_ver} != Code v{current_version}. Run migration first."
   ```

3. **新增的 checkpoint 是否能被当前的模型类加载？**
   ```bash
   # 测试加载最近 N 个 checkpoint：
   python -c "from models.dual_mask import DualMaskDepthModel; \
       m = DualMaskDepthModel(); \
       ckpt = torch.load('checkpoints/latest.pt'); \
       m.load_state_dict(ckpt['model_state_dict'])"
   ```

4. **`dual_mask.py` 和 `dual_mask.py.bak` 的区别是否有文档记录？**
   - 如果存在 `.bak` 文件，必须在文件头注释中写清楚备份原因和时间
   - 理想情况：用 git branch/tag 而非 .bak 文件管理版本

### `--focus inference` — Training-Inference Alignment

**Goal**: 确保训练产物（模型、配置）可以正确地在推理阶段被加载和使用。

检查清单：
1. **推理脚本是否独立可运行？**（不依赖训练时的 import 环境）
   ```bash
   # 推理脚本必须能在干净环境中运行：
   python -c "exec(open('scripts/inference.py').read())"  # 不应报 ImportError
   ```

2. **模型类的定义是否与 checkpoint 中的 key 完全匹配？**
   ```python
   # 加载 checkpoint 后必须验证：
   model_keys = set(model.state_dict().keys())
   ckpt_keys = set(ckpt["model_state_dict"].keys())
   missing_in_model = ckpt_keys - model_keys  # checkpoint 有但模型没有
   extra_in_model = model_keys - ckpt_keys    # 模型有但 checkpoint 没有

   if missing_in_model or extra_in_model:
       print(f"KEY MISMATCH: missing={missing_in_model}, extra={extra_in_model}")
       assert False, "Architecture mismatch"
   ```

3. **训练时的 normalization 是否与推理时一致？**
   ```python
   # 检查数据集的 normalize 方式：
   #   ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
   # 如果训练用但推理没用 → 必须报告为 P0 问题
   ```

4. **损失函数改动是否影响了 checkpoint 的可解释性？**
   - 如果 loss 从 MSE 改成了 L1，旧的 checkpoint 的 loss 值就不可比了
   - 这种情况必须在 checkpoint metadata 中记录

### `--focus tests` — Test Coverage for ML Components

**Goal**: 确保关键 ML 组件有测试覆盖，特别是模型加载和推理。

检查清单：
1. **每个模型类是否有加载测试？**
   ```python
   # tests/test_model_loading.py
   def test_checkpoint_loading():
       for ckpt_path in Path("checkpoints/").glob("*.pt"):
           model = DualMaskDepthModel()
           ckpt = torch.load(ckpt_path)
           state = model.state_dict()
           # 检查所有 key 是否兼容
           assert set(state.keys()) == set(ckpt["model_state_dict"].keys()), \
               f"Key mismatch in {ckpt_path.name}"
   ```

2. **是否有版本迁移测试？**
   ```python
   # tests/test_migration.py
   def test_v1_to_v2_migration():
       ckpt_v1 = load_checkpoint_v1("checkpoints/v1_checkpoint.pt")
       ckpt_v2 = migrate_to_v2(ckpt_v1)
       model = DualMaskDepthModel(version="v2")
       model.load_state_dict(ckpt_v2)
   ```

3. **dry-run 是否覆盖了模型前向传播？**
   - 当前的 dry-run 只检查 `import` 和配置加载
   - 应该增加一步：运行一个 dummy forward pass

## Review Output Format

```markdown
# Code Review — YYYY-MM-DD

## Project: [name]

## Focus: [arch | inference | tests]

### Issues Found

#### [P0] — Blocking (must fix before dry-run / commit)
- [具体问题描述]
- **Fix**: [建议的修复方式]

#### [P1] — Warning (should fix before next experiment)
- [具体问题描述]
- **Fix**: [建议的修复方式]

#### [P2] — Suggestion (nice to have)
- [具体问题描述]

### Checkpoints Reviewed
| File | Architecture Version | Loadable | Notes |
|------|---------------------|----------|-------|
| dual_mask_log_space_r_ep20.pt | v1.1 | ✓ | 最新最佳 |
| ... | | | |

### Test Coverage
| Component | Covered | Test File |
|-----------|---------|-----------|
| Model loading | ✓/✗ | ... |
| Inference pipeline | ✓/✗ | ... |
| Version migration | ✓/✗ | ... |

### Recommendations
- [架构版本管理建议]
- [checkpoint 归档策略建议]
```

## P0 Issues (Block dry-run/commit)

以下情况必须报告为 P0：
1. 新 checkpoint 无法被当前模型类加载（key mismatch）
2. 推理脚本缺少版本验证，新旧 checkpoint 混用
3. `.bak` 文件存在但无文档说明备份原因
4. 训练和推理使用了不同的 normalization 参数

## 与 hands-off-issue-handling 的关系

当 code-review 发现 P0 问题时：
- **Type**: 属于 hands-off-issue-handling 中的"代码逻辑问题"（Type A）
- **处理**: 立即修复，不记录到 MEMORY_LOG.md
- **如果涉及版本兼容性**：属于 Type B（需要记录 Assumption）

## 与 auto-experiment 的集成

```
auto-experiment loop:
  EXECUTE phase:
    Code Agent modifies model → code-review --focus arch
    Code Agent saves checkpoint → code-review --focus inference
    Dry-run → code-review --focus tests
  REFLECT phase:
    发现 checkpoint 加载问题 → code-review --focus inference
```

## 集成建议（写入 core/tools.py）

在 `launch_experiment` 工具的 `_exec_launch_experiment` 方法中，
**dry-run 之后、真正 nohup 之前**，插入自动化检查：

```python
def _exec_launch_experiment(self, command: str, log_file: str, gpu: str = None) -> str:
    # ... 解析命令等前置步骤 ...

    # 1. Dry-run (现有逻辑)
    dry_run_ok = self._run_dry_run(argv)
    if not dry_run_ok:
        return json.dumps({"error": "Dry-run failed, aborting launch"})

    # 2. NEW: Code-review automated checks (P0 only)
    p0_issues = self._run_code_review_p0_checks(self.workspace)
    if p0_issues:
        return json.dumps({
            "error": "P0 code-review issues found",
            "issues": p0_issues,
        })

    # 3. 真正的 nohup launch
    with open(log_path, "w") as f:
        proc = subprocess.Popen(...)
    return json.dumps({"pid": proc.pid, "log_file": str(log_path), "status": "launched"})
```

**自动化 P0 检查包括**：
- `torch.load(checkpoint)` 是否报错
- `model.load_state_dict(ckpt)` 是否报错（key mismatch 检测）
- 推理脚本 `python -c "exec(open(script).read())"` 是否报 ImportError
- 检测是否存在未标注的 `.bak` 文件

这样 agent 每次 launch 实验前都会自动过一遍 P0 检查，**不需要每次手动调用 code-review skill**。
