# Reform v21 — 可证伪的改革实验设计

> **这份文档是实验设计，不是实现记录。**
> 它定义：基线（X）→ 改革动作 → 预期成效（Y）→ 10-cycle 验证协议。
> 改革完成后跑 10 个 cycle，用预期值反向检验。**偏离即问题，问题即深挖。**
>
> 制定日期：2026-06-24
> 基线来源：depth_estimation_unify_theory 项目 6月23日 V30 运行的真实数据 + SQLite 现状

---

## PART 0：前置技术验证结果（Phase 0 已完成）

**执行日期：2026-06-24。用 `tools/probe_schema_conformance.py` 实测 GLM-5.1 对 schema 强制的服从度。**

### 实测数据

| 策略 | tool_choice | 工具 schema | N | (a) 调用工具 | (b) 字段齐全合法 | (c) 裸文本 |
|---|---|---|---|---|---|---|
| A（现状基线） | auto | 旧 dummy {response:string} | 10 | 6/10 | **0/10** | 4/10（其中 4/4 含可解析 JSON） |
| B（Phase2 默认路线） | **required** | decision schema（5 字段） | 10 | 1/10 | **1/10 = 10%** | 9/10（全是 ```` ```json ```` 代码块） |

### 三个决定性发现

**发现 1：`tool_choice="required"` 对 GLM-5.1 基本无效（b=10%）。**
模型在 required 下仍 9/10 走文本通道输出，无视 tool_call 要求。**Phase 2 的"schema 强制"路线被否决。** SDK 层支持（zai completions.py:60）≠ 模型层服从。

**发现 2：但模型输出的文本内容是对的。**
策略 A 的 4 次裸文本 100% 含可解析 JSON；策略 B 的 9 次裸文本全是 ```` ```json {...} ```` 格式。模型理解了任务、产出了正确字段，只是**拒绝走 tool_call 通道，偏要走文本通道**。现有解析器 `_extract_first_decision_json`（brace-walking）能从这些 ```` ```json ```` 块里提取 JSON（3/3 验证通过）。

**发现 3（最重要，推翻了根因 A）：12 次 unparseable 全是 REFLECT 阶段，不是 THINK。**
查真实日志：12 次 Leader response unparseable 里，11 次在 6月17日（全 REFLECT），1 次在 6月23日（V30 的 REFLECT）。**0 次是 THINK。** THINK 阶段的失败是 API 层面（401/超时，看 traceback），不是格式问题。

### 根因 A 的修正

**原诊断（错误）**：leader 12 次输出 Markdown 不输出 JSON → 解析失败 → wait。归咎于"模型不服从格式指令"。

**修正后（正确）**：
- **THINK 阶段**：leader 其实大多能输出可解析的 JSON（probe 证明 GLM 愿意输出 JSON，解析器也能提取）。失败是 API 层面的（401/超时），不是格式问题。
- **REFLECT 阶段**：这才是真正的问题。REFLECT 要求 leader 把开放式反思压缩成 JSON（milestone/decision/causal_link），但模型自然倾向于写散文。12 次 unparseable 全发生在这里。
- **为什么 probe 和真实日志矛盾**：probe 测的是"做一个决策并输出 JSON"（THINK 类任务），模型配合；真实日志的失败是 REFLECT 类任务（"反思结果并输出 JSON"），模型不配合。**任务类型决定了模型是否愿意走 JSON 格式。**

### Phase 2 路线决策

**不走 schema 强制路线**（b=10% 证伪）。改为：

**Phase 2 重新定位：REFLECT 的"事实/叙事分离"。**
- REFLECT 失败的根因不是"格式"，而是"要求 LLM 同时做两件本质不同的事"：① 提取事实（milestone 达成没、metrics 是多少）② 产出叙事（因果归因、教训）。事实部分应该是确定性的（Phase 1 的 fact_scanner 已经能做），叙事部分才是 LLM 的活。
- **修正方向**：不让 REFLECT 的 LLM 对"事实"负责（事实由 fact_scanner 确定），只让 LLM 对"叙事"负责。即使 LLM 输出散文，事实已经由 fact_scanner 记录了——REFLECT 失败不再导致失忆。这和 Phase 1 的方向一致，只是分工更清晰。
- THINK 阶段：保持现有解析器（brace-walking 已能提取 ```` ```json ````），加上"解析失败→带反馈重试"即可。不需要 schema 强制。

**这个修正实际上简化了改革**：Phase 1（fact_scanner）比预想的更重要——它不只治 B1（失忆），还治 B2/B3/B4（milestone/dead_end/causal 空），因为这些都该是事实层的确定性产出，不该依赖 REFLECT 的 LLM。



这一轮改革的全部设计，建立在三轮求证后确立的四条地基上。**任何与这四条冲突的方案，都不准入。**

### 地基 1：enforcement 的有效性 = LLM 能否绕过
- v17 用"文本注入 context"做 enforcement → 运行数据证明无效（circuit breaker 0 触发、code_review 92% 被忽略）→ v18 删了。
- v18 保留的 launch_experiment FORBIDDEN（工具层 return error）→ 有效，LLM 绕不过。
- **结论：任何新 enforcement，第一个问题是"在哪个 LLM 绕不过的环节拦截"，不是"怎么提示 LLM"。文本注入一律不算 enforcement。**

### 地基 2：事实层可硬拦，解释层不可硬拦
- 对抗论证 + 独立验证后收敛的边界：
  - **事实层**（真相能在动作源头结构化记录，无需事后从文本推导）：可硬拦。如"实验是否真启动""loss 是否 NaN""success_criteria 谓词是否满足""session 内是否缺 control run"。
  - **解释层**（需要语义理解）：不可硬拦。如"control 结果说明方法有效还是没训够""threshold 是否合理""死路的边界条件"。
- **ARCHITECTURE_REVIEW D1 的 root pattern 原文**："the system never records a structured event. It tries to re-derive the event by reading free text afterward." → **一切新机制必须在源头记录结构化事件，禁止用 regex 事后推导真相。**

### 地基 3：认知闭环不能是单点
- V30 案例证明：一次 REFLECT LLM 调用失败 → milestone/dead_end/causal 全空 → 下个 cycle 失忆 → 重复实验。
- **结论：任何"必须存活"的信息（事实层），必须有 LLM 之外的确定性来源。LLM 只负责解读，不负责记录事实。**

### 地基 4：顺序由依赖关系硬性决定，不由"哪个急"决定
- v18 的教训：没划清边界就动，从"错误干预"摆到"完全不干预"。
- **每一步的输入必须来自更底层，绝不来自 LLM 自由文本。顺序反了 = 重蹈 D1（string-coupling）。**

---

## PART 1：基线（X）—— 改革前的真实状态

数据来源：`depth_estimation_unify_theory/experiment_history.db`（14 条记录）+ `autoresearcher.log`（6月23日运行）+ outputs 目录。

### 基线指标表

| # | 指标 | 基线值（X） | 数据来源 | 含义 |
|---|------|-----------|---------|------|
| B1 | **完整训练结果进入持久记忆率** | **0%**（6月23日 V30 50-epoch 完整结果，SQLite 里 0 条） | `SELECT * FROM experiments WHERE timestamp LIKE '2026-06-23%'` → 0 行 | 一次完整实验跑完后，其结果是否能被系统记住 |
| B2 | **milestone 字段非空率** | **0/14 = 0%** | SQLite experiments.milestone 全空 | 实验成果的叙事是否留存 |
| B3 | **dead_end 字段非空率** | **0/14 = 0%** | SQLite experiments.dead_end 全空 | 死路是否被记录（Phase 1 已判死但字段空）。**注：已迁移** — dead_end 的真相源现为 `memory_entries` 表（`entry_type='dead_end'`），`experiments.dead_end` 列已删除（见 `docs/DATA_CONTRACT.md`）；此基线为迁移前的历史测量 |
| B4 | **causal_chain 表行数** | **0** | `SELECT COUNT(*) FROM causal_chain` = 0 | 因果归因是否留存 |
| B5 | **experiment_value 表行数** | **1** | `SELECT COUNT(*) FROM experiment_value` = 1 | 假设校准是否在运作 |
| B6 | **success_criteria 被机器求值次数** | **0**（11/14 有该字段，但 0 处代码比较它） | grep 全库无谓词求值器 | 可证伪性是否真的被执行 |
| B7 | **leader 决策解析失败后的恢复方式** | **wait（被动空转）** | agents.py:1748 默认 wait，无重试无反馈 | 解析失败是阻断还是被吞 |
| B8 | **重复实验检测能力** | **无**（V30 出结果后重跑了内容相同的 V30） | outputs 下有 v30_energy_cost_volume + cost_volume_v30 两个近似实验 | 系统能否识别"已做过" |
| B9 | **重复死路重试** | **存在**（constraint_engine 16 词关键词，可被改写绕过） | constraint_engine.py:200-207 | 已证伪方向是否会被重试 |
| B10 | **实现偏离规范的检测** | **无**（V30 实现 sigmoid 软门控，规范要求硬 trimmed mean，无人发现） | ROADMAP 规范 vs 实际代码 | spec 是否被执行链传递 |

### 基线的核心解读

> **系统现在的状态：能跑训练，但跑完即忘。事实（metrics）有 35% 进库，解读（milestone/dead_end/causal）0% 进库。一次完整 50-epoch 训练可以从系统记忆里彻底消失。系统既不记得做过什么（B1/B2），也不知道什么不该再做（B8/B9），也无法判定结果是否达到目标（B6）。**

---

## PART 2：改革动作（按依赖顺序，Phase 0 → 4）

每个 Phase 遵循：①只引入一个结构变化 ②该变化能独立验证 ③系统始终可运行。
**顺序不可调换**——每个 Phase 的输入依赖前一个 Phase 的输出。

### Phase 0：前置技术验证 ✅ 已完成（2026-06-24）

**结果见本文件 PART 0 节。结论：`tool_choice="required"` 对 GLM-5.1 无效（b=10%），schema 强制路线被否决。同时推翻了根因 A（12 次 unparseable 全是 REFLECT 不是 THINK）。Phase 2 改为"REFLECT 事实/叙事分离"方向。**

---

### Phase 1：确定性事实脊柱（治 B1/B2/B3/B4 的根）

**目标**：系统的永久记忆里，"每次实验发生了什么"（事实层）由**扫描磁盘文件**确定性重建，不依赖任何 agent 进程存活，不依赖任何 LLM 调用。

**⚠️ 再思考修正（推翻了初稿设计）**：
- **初稿错误**：把 fact 记录放在 monitor 回调（进程退出即触发）。
- **为什么错**：monitor 在 agent 进程内。V30 失忆的真正原因不是 REFLECT 失败，而是**系统重启杀了整个 agent 进程**，monitor 回调根本没机会执行。把记录放在 agent 内的回调，和 REFLECT 一样会随 agent 死亡而失效——重蹈单点覆辙。
- **证据**：6月23日 V30 的 `train.log`（1000 行，50 epoch 全在）和 `experiment_manifest.json` 在系统重启后**仍然存活**——文件杀不死，但 agent 进程会死。事实一直在磁盘上，只是没人去读。

**修正后的核心动作**：
1. 新增 `core/fact_scanner.py`：**启动时（每个 session 第一个 cycle、以及 runner 每次重启后）扫描** `outputs/*/experiment_manifest.json` + 对应 `train.log`，用 `training_log_parser.py` 提取 `{output_dir, command, per_epoch_loss, per_domain_metrics, best_metric, best_epoch, loss_trend, log_exists, log_complete}`，写入 SQLite 新表 `experiment_facts`。
2. **幂等键 = `output_dir`**（manifest 的 pid 会复用，不能当唯一键）。扫描时 `INSERT OR IGNORE`（已存在的 output_dir 不重写），保证重复扫描不产生重复行。
3. **复用现有扫描**：`tools.py:988` 已有 `outputs_dir.glob("*/experiment_manifest.json")` 的逻辑——不新建扫描，找到它、把它的结果接到 fact 表写入。
4. **不依赖 REFLECT，不依赖 monitor 回调，不依赖 agent 存活**：扫描只依赖文件在不在。文件在 = 事实在。

**为什么这个最先（在 Phase 0 之后）**：它是 Phase 2-4 的地基。Phase 2 的降级、Phase 3 的谓词求值、Phase 4 的死路匹配，全都依赖"系统能确定性地知道每次实验发生了什么"。没有这个，后面一切又是从 LLM 文本推导（重蹈 D1）。而且它**最独立**——不碰 leader、不碰 verify，副作用最小。

**验证不变量（Phase 1 完成的判定标准）**：
- [ ] 对当前已有的 outputs（含 V30 的 v30_energy_cost_volume），扫描后 `experiment_facts` 表有对应记录，含 best_metric=0.144、loss_trend
- [ ] **模拟 agent 崩溃**：手动 `kill -9` agent 进程，重启后扫描，V30 的事实**仍然进入** fact 表（因为靠扫文件，不靠 agent 内存）
- [ ] 幂等性：扫描两次，`experiment_facts` 行数不变
- [ ] 一个 output_dir 没有 train.log（如 dryrun）→ 记录 `log_exists=false`，不崩溃

---

### Phase 2：REFLECT 事实/叙事分离 + THINK 解析加固（治 B7/B2/B3/B4）

**⚠️ Phase 0 结果重定向**：原计划是"schema 强制 THINK 输出"，但 Phase 0 证明 (a) `tool_choice=required` 对 GLM 无效，(b) 12 次 unparseable 全是 REFLECT 不是 THINK。因此 Phase 2 重定向为两件事：加固 THINK 的文本解析（小改），重构 REFLECT 的事实/叙事分工（核心改）。

**核心动作**：

**2a. THINK 解析加固（轻量，因为 THINK 本来就基本能用）**：
- 现有 `_extract_first_decision_json`（brace-walking）已能从 ```` ```json ```` 块提取 JSON（Phase 0 验证 3/3）。保持。
- 解析失败 → 带反馈重试 1 次（agents.py:1748）："你的输出无法解析为决策 JSON，请输出 ```` ```json ```` 包裹的 JSON，含 action/task/hypothesis/success_criteria 字段"。重试仍失败才 wait。
- **不做 schema 强制**（Phase 0 已证无效）。

**2b. REFLECT 事实/叙事分离（核心改）**：
- **问题本质**：REFLECT 要求 LLM 同时做事实提取（milestone 达成没、metrics 多少）和叙事产出（因果归因、教训）。LLM 在叙事任务上倾向输出散文，导致整个 JSON unparseable，连带事实也丢了。
- **修正**：把 REFLECT 的产出拆成两层：
  - **事实层（确定性，不依赖 LLM）**：`fact_scanner`（Phase 1）在实验完成时已记录 metrics/loss_trend/criteria_met。这些是"发生了什么"，REFLECT 不再对它负责。
  - **叙事层（LLM 负责，但容错）**：REFLECT 的 LLM 只产出"解读"（为什么、教训、下一步建议），用**自由文本**，不强制 JSON。叙事写进 SQLite 的一个**可空字段**（`narrative TEXT`），失败就空着——不影响事实。
- **后果**：REFLECT 失败 → 叙事空，但事实（metrics/criteria_met）已在 fact_scanner 里。**不再失忆。** milestone/dead_end 改为从 fact_scanner 的 `criteria_met` 派生（达标=milestone，连续未达标+claim_type=causal=候选 dead_end），不依赖 LLM 文本。
- **新字段 claim_type**：仍需 LLM 在 THINK 时输出（claim_type 是 THINK 决策的一部分，probe 证明 THINK 类任务模型愿意输出 JSON）。用于 Phase 3 的 control-coverage gate。

**为什么在 Phase 1 之后**：2b 的"事实层"完全依赖 fact_scanner（Phase 1）。没有 Phase 1，REFLECT 失败时事实无处可依。

**验证不变量**：
- [ ] THINK 解析失败触发 1 次重试（日志可见 "retrying think with feedback"），而非直接 wait
- [ ] REFLECT 输出散文（非 JSON）时，系统不崩溃，`narrative` 字段记录文本，事实字段（来自 fact_scanner）仍完整
- [ ] milestone 从 `criteria_met` 派生：一个 success_criteria 满足的实验，即使 REFLECT 失败，milestone 也非空
- [ ] claim_type 在 THINK 输出里出现（作为 JSON 字段，经 brace-walking 提取）

### Phase 3：事实层方法论门禁（治 B6/B8/B9，注意边界）

**目标**：把"事实层"的方法论约束做成硬拦截。**严格只做地基 2 允许的部分——事实层硬拦，解释层绝不碰。**

**核心动作**（三个 gate，全是事实层）：

1. **Falsification gate（可证伪判定）**：
   - `success_criteria`（Phase 2 已结构化）解析成谓词 `(metric, op, threshold)`，与 `experiment_facts`（Phase 1）的 metrics 确定性比较。
   - 输出 `criteria_met: bool`，写进 `experiment_facts`。
   - **这是事实层**（谓词求值是数学运算，无语义）。
   - **边界**：threshold 是否合理（解释层）不判定。不可解析的 criteria → 标记 `unparseable`，不假装通过。

2. **Control-coverage gate（对照覆盖检查）**：
   - 当 `claim_type == "causal"` 时，查 `experiment_facts`：session 内是否存在同 method、减去该组件的 control run。这是确定性 SQL 查询。
   - 缺 control → 标记 `uncontrolled_inconclusive`（**不是**"方法无效"，**不是**"必须重做"——只是标记，让 LLM 知道这个结论的强度）。
   - **边界**：control 结果说明什么（解释层）不判定。只判定"有没有"，不判定"够不够好"。

3. **Dead-end signature gate（死路精确匹配）**：
   - 用结构化签名 `(method, key_config, dataset)` 替代 constraint_engine 的 16 词关键词。
   - method 来自 Phase 2 的结构化字段；key_config 从 think_result 的结构化 config 段提取。
   - 命中历史死路签名 → 在 **THINK 阶段**注入警告（不是 launch 时才查——这是 v18 删掉的 constraint_engine 的问题：从不在决策路径调用）。
   - **边界**：死路的边界条件（哪个 config 下成立）不预判。新方法首次出现无历史 → 不拦（诚实承认"无数据"）。

**为什么在 Phase 2 之后**：三个 gate 全依赖 Phase 2 的结构化字段（success_criteria、claim_type、method）。没有 Phase 2，它们又得从自由文本推导（重蹈 D1）。

**验证不变量**：
- [ ] 跑一个 `success_criteria: "val_MAE < 0.20"`、实际 0.15 的实验，`experiment_facts.criteria_met == true`（确定性求值，无 LLM）
- [ ] `claim_type: causal` 且 session 无 control run → 标记 `uncontrolled_inconclusive`，但不改变 action（不越界到解释层）
- [ ] 死路签名命中 → THINK context 里有结构化警告（不是文本注入 prompt，是 fact_recorder 写入的 fact 字段）

### Phase 4：门禁执行权 + 状态统一（治 B10 + 固化前三步）

**目标**：把前三个 Phase 的成果固化成不可绕过的脊柱，并补上 spec 遵从检查（B10）。

**核心动作**：
1. **Spec-conformance gate**（B10 的回应）：code agent 写出的模型/脚本，在 VERIFY 阶段对照 ROADMAP 的**结构化 spec**（不是自由文本）检查关键签名（如"聚合层是否含 sort 操作"对应硬 trimmed mean）。spec 从 ROADMAP 提取成机器可读的签名清单。偏离 → 标记 `spec_deviation`（事实层：有没有偏离，是确定的；为什么偏离留给 LLM）。
2. **fact-based action gating**：把"事实层 gate 的结果"接入 loop 的决策路径。例如 `criteria_met=false 且 claim_type=causal 且 uncontrolled` → 不允许该 cycle 标记为 progress（事实层硬拦：你不能把一个未达标准、无对照的实验记为"进步"）。
3. **state.json 与 experiment_facts 对齐**：state.json 的 `last_metrics` 从 `experiment_facts` 读，不再从 REFLECT 残留读。

**为什么最后**：它依赖前三个 Phase 都已就位（需要 facts、结构化字段、gate 结果）。这时才有条件做"基于事实的 action gating"——否则 gating 的输入又是 LLM 文本。

**验证不变量**：
- [ ] 实现偏离 spec（如用 sigmoid 替代 sort）→ `spec_deviation` 标记为 true
- [ ] 未达标 + 无对照的实验 → `made_progress` 被系统强制为 false（不依赖 LLM 判断）
- [ ] state.json 的 last_metrics 与 experiment_facts 一致

---

## PART 3：预期成效（Y）—— 拿着答案问问题

改革完成后跑 10 个 cycle。**每个指标有明确的预期值。偏离 > 容差 → 触发深挖。**

> ⚠️ **再思考修正**：初稿的 M1/M5/M6 设了不合理的容差。再思考发现：(1) M1 的"100% 触发"假设了回调机制不会失败，但 V30 失忆正是进程被杀导致回调失效——改成扫描后，容差应基于"文件存在率"而非"触发率"；(2) M5/M6 容差设 0 把"新方法首次出现的诚实 miss"误当成失败——必须区分"无数据 miss"（系统不知道，合理）和"该拦没拦"（系统知道但没拦，bug）。

| # | 指标 | 基线 X | 预期 Y（10 cycle 后） | 容差 | 偏离时的归因方向 |
|---|------|--------|---------------------|------|----------------|
| M1 | **磁盘上已有实验**进入 experiment_facts 的比率 | 0% (B1) | **100%**（扫描覆盖所有有 manifest 的 output_dir） | 0（文件在就一定能扫，确定性操作） | 若 <100%：扫描的 glob 模式漏了某些目录，或 manifest 路径不规范。查 fact_scanner 的 glob + 哪些 output_dir 被漏 |
| M1b | **本 session 新跑实验**进入 experiment_facts 的比率 | 0% (B1) | **100%**（实验完成 + 下个 cycle 启动扫描即记录） | 允许"实验在跑、agent 在等"的 transient 窗口不计；只算"进程已退出"的实验 | 若 <100%：扫描没在 cycle 开始时跑，或进程退出检测有问题。**红旗**——这说明扫描触发点设计错 |
| M2 | milestone 字段非空率（含 LLM 解读） | 0% (B2) | **≥60%** | ±15% | 若 <45%：REFLECT 仍频繁失败 → 查 Phase 2 重试是否生效；若 >80%：抽查 milestone 内容质量，可能 LLM 敷衍填字段 |
| M3 | success_criteria 被机器求值的比率（分母=有可解析 criteria 的实验） | 0% (B6) | **100%**（fact_scanner 在 Phase 1/2 提取后求值，不依赖 REFLECT） | 0（谓词求值是数学运算，不该失败） | 若 <100%：谓词解析器有 bug，或 metric 命名对不齐（求证发现的 4 套打架的 key）。查解析器 + metric 归一化 |
| M3b | criteria **可解析率**（分母=所有有 criteria 的实验） | 未知 | **≥80%** | — | 若 <80%：leader 在写定性 criteria。**区分**：这是 leader prompt 问题（可改 prompt）还是真实科研困难（无法消除），看 unparseable 的 criteria 长什么样 |
| M4 | leader 解析失败**直接 wait** 的次数（未触发重试） | 12次/天 (B7，但全是 REFLECT) | **THINK 失败 ≤2次/10cycle** | — | Phase 0 证明 THINK 基本能输出 JSON，失败应少见。REFLECT 失败不再计入 M4（因为 2b 让 REFLECT 容错，失败不导致 wait）。若 THINK 失败 >2：查 API 层（401/超时）非格式问题 |
| M5 | **重复实验**（同 output_dir 或同签名重跑）发生次数 | 存在 (B8) | **0**（针对**历史已有签名**的重复） | 仅允许"全新方法/全新 config"的重跑不计 | 若 ≥1 且该签名在 experiment_facts 里**已有记录**：gate 没在 THINK 生效（D2 复发）→ 红旗。若该签名**无历史记录**（新方法）：不算失败，记录为"新方法首次出现" |
| M6 | **已证伪方向**被重试次数 | 存在 (B9) | **0**（针对**有 dead_end 记录的签名**） | 仅允许"死路边界条件不同"的重试不计（如换了 dataset） | 若 ≥1 且该签名 dead_end 非空：死路 gate 没生效 → 红旗。若 dead_end 为空（Phase 1 前历史没记录）：不算失败，但说明历史死路数据不足，需补录 |
| M7 | claim_type=causal 且无 control 的实验 → 被标记 `uncontrolled` 的比率 | 未知 | **100%**（标记率） | 0（标记是事实层 SQL 查询，确定性） | 若 <100%：claim_type 字段没被正确填写（Phase 2 schema 没强制），或 control 查询逻辑错。查 claim_type 实际取值分布 |
| M8 | 每次 experiment 有 spec_deviation 检查记录 | 0% (B10) | **100%**（有/无偏离都记录） | 0 | 若 <100%：spec 提取失败或 gate 没接入 VERIFY |

### 关于 M3/M3b 的关键区分（再思考新增）

求证时发现 config.yaml 只有一个全局目标 `val_MAE < 0.20`，但真实目标是分层的（Lambertian<0.16 等）。这意味着：
- **M3（可解析的 criteria 都被求值）是系统职责**，应 100%。
- **M3b（criteria 可解析率）是 leader 职责 + 科研本质困难**。leader 可能写"验证机制是否有效"这种定性目标——这是真实的科研表达，不该强求全部可解析。M3b 的预期因此只设 80%，留出"定性目标"的空间。

**这个区分本身就是"事实层 vs 解释层"边界在指标上的体现**：求值（M3）是事实层，100%；可解析性（M3b）部分依赖 leader 质量，留余量。

### 预期的"反直觉"信号（这些偏离可能是好事，要区分）

| 信号 | 可能含义 | 如何区分好坏 |
|------|---------|------------|
| M2（milestone非空率）反而下降 | Phase 2 让 leader 更诚实，不再用空 milestone 填充 | 抽查 milestone 内容：有实质 > 有字段 |
| M7 标记率很高，但实验没增加 control | 系统在标记但 LLM 选择不做 control（合理，control 很贵） | 这是预期内的——gate 只标记不强制，属于解释层留给 LLM |
| M3b 的 unparseable 集中在某些类型实验 | leader 对定性目标写不出定量 criteria | 这是真实的科研困难，不是系统 bug——记录下来作为 prompt 改进输入 |
| M5 出现"重复"但全都是新方法 | leader 在探索新方向，每个都是首次 | 看这些实验的签名是否真在 experiment_facts 里无历史记录 |

### 最关键的"红旗"（出现任何一个立即停止，深挖）

- 🔴 **M1 < 100%**：事实脊柱断了。扫描漏了文件，或 glob 模式错。这是地基，确定性操作不允许失败。
- 🔴 **M5 ≥ 1 且该签名在 experiment_facts 里已有记录**：enforcement 又退回"只 log 不 enforce"（D2 病复发）。**注意**：必须是"已有记录"才算红旗——新方法的 miss 不是。
- 🔴 **10 cycle 后 causal_chain 表仍为 0 或 experiment_value 仍 ≤1**：LLM 没产出这些，或产出没被记录——Phase 1-2 的结构化字段没回流到 REFLECT 的输入。
- 🔴 **M3 < 100%（可解析的没全求值）**：谓词解析器或 metric 对齐有 bug。这是事实层，不该失败。

---

## PART 4：10-Cycle 验证协议

### 运行方式
1. 改革完成（4 个 Phase 全部实现 + 不变量全部通过）后，在 `depth_estimation_unify_theory` 项目上跑连续 10 个 cycle。
2. 每个 cycle 结束后自动采集 M1-M8 指标（写一个 `verify_reform.py` 脚本，从 SQLite + 日志提取）。
3. 10 cycle 结束后出一份对照报告：基线 X vs 实测 Y，逐指标判定 PASS/FAIL/INVESTIGATE。

### 偏离处理流程（"拿着答案问问题"）
对每个未达预期的指标，按以下顺序归因（**禁止跳步**）：

1. **是不是确定性管道坏了？**（M1/M3/M5/M6 这类应该是确定性的）→ 查 fact_recorder / gate 的触发日志，找断点。
2. **是不是 LLM 没产出结构化字段？**（M2/M7）→ 查 leader 输出的原始 JSON，看字段是否齐全。
3. **是不是 enforcement 没硬拦？**（M5/M6 的"命中未拦"）→ 查 gate 结果是否真的接入了 loop 的 action 决策。
4. **是不是边界画错了？**（M3 的 unparseable 高）→ 这可能是真实的科研困难，不是 bug。区分"系统问题"和"科研问题"。

### 终止条件
- 10 cycle 全跑完且无红旗 → 改革成功，归档。
- 出现红旗 → 立即停止，深挖根因，修复后重跑（不累计计数，从 0 重来）。
- M1-M3 全 PASS 但 M5/M6 FAIL → 特定 gate 的设计问题，不推翻整体，定向修复。

---

## PART 5：诚实的风险声明（含再思考修正）

这份计划是基于求证的推断，**且经过一轮"再思考"推翻了初稿的三个想当然**。以下风险分两层：第一层是再思考已经发现的、已修正的；第二层是仍然存在的、可能在 10-cycle 时暴露的。

### 第一层：再思考已发现并修正的（记录为教训，防复发）

1. **【已修正】初稿把 fact 记录放在 monitor 回调（agent 进程内）。** 再思考发现 V30 失忆的根因是系统重启杀了整个 agent，回调根本没机会执行——和 REFLECT 一样是单点。**修正为磁盘扫描**，只依赖文件存在。教训：任何"必须存活"的信息，不能依赖 agent 进程内的任何机制。
2. **【已修正】初稿假设 `tool_choice="required"` 能强制 leader 输出。** 再思考查证：SDK 层支持（zai completions.py:60），但模型层服从度未知，且 required 只强制调用不强制填字段。**修正为插入 Phase 0 前置验证**，实现路线由实测结果决定。教训：技术前提不能假设，必须前置实测。
3. **【已修正】初稿 M5/M6 容差设 0。** 再思考发现新方法首次出现必然 miss（无历史签名），把"诚实 miss"误当失败。**修正为区分"已有记录却没拦"（红旗）和"新方法首次出现"（正常）**。教训：指标必须区分"系统该知道的"和"系统不可能知道的"。

### 第二层：仍然存在的风险（10-cycle 时最可能偏离的点）

4. **Phase 0 的实测结果可能迫使 Phase 2 降级。** 若 GLM 对 schema 强制的服从度 (b) < 35%，整个 Phase 2-4 的"结构化字段"基础就塌了，M3/M4/M7 的预期全部要重设。这是最大的未知，也是 Phase 0 存在的全部理由。
5. **Phase 3 的死路签名粒度是经验参数。** 太粗误杀（把可救方向判死），太细漏匹配。M5/M6 会暴露这个。这本身需要根据 10-cycle 数据迭代调参——是预期的迭代，不是失败，但要防止"调参变成永远调不准"（那说明签名方法本身有问题，要回到"死路判定能否代码化"的求证）。
6. **Phase 4 的 spec 提取依赖 ROADMAP 的可机器解析性。** 当前 ROADMAP 是给人看的 Markdown，转成结构化 spec 需人工/半自动转换。转换不完整 → spec_deviation gate 漏检。这不是致命的（漏检只是少了一道防护），但会让 M8 的 100% 达不成。
7. **M2（milestone）本质上依赖 LLM 质量。** 即使 Phase 2 让字段齐全，milestone 内容好不好仍是 LLM 的事。M2 预期设 60% 而非 100%，留了这个余量。
8. **"解释层不可硬拦"的边界可能在某些场景下被证明画错了。** 求证时是基于对抗论证收敛的，但收敛不等于绝对正确。如果 10-cycle 显示某个"解释层判断"反复导致错误决策，可能需要重新论证那条边界——这是最深的潜在修正。

### 我诚实承认的元风险

**这份计划本身也可能有第四、第五个想当然，是我再思考也没发现的。** 对抗这个的唯一方式就是：10-cycle 验证时，对每个偏离都保持"先归因、再判断是 bug 还是设计缺陷"的纪律，而不是急于调整指标来让数字好看。**如果发现偏离就放宽预期，那就违背了"拿着答案问问题"的初衷——预期是用来暴露问题的，不是用来被满足的。**

**这些风险本身就是"拿着答案问问题"的一部分——它们是预先标注的、最可能偏离的点，也是最可能产生新认知的点。**
