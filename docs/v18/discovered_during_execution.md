# 改革执行期间发现的清单外问题

> 记录但不做。留到方案复审时决定归入哪个阶段。

## 阶段二发现

### D2-1: GC 归档后 LLM 找不到文件
- **问题**: GC 把 `_check_grad.py` 归档到 archive/temp/，下个 cycle LLM 调 read_file 找不到
- **影响**: 低（GC 只归档临时文件，LLM 通常不需要跨 cycle 复用）
- **解法**: GC 归档时记录文件列表，注入 context 让 LLM 知道哪些被归档了
- **归入**: 考虑归入阶段三（ResearchAdvisor 注入归档通知）

### D2-2: archive/temp/ 无上限管理
- **问题**: GC 只往 archive/temp/ 写，不清理它，长期运行会积累
- **影响**: 低（不在工作区，不影响 agent 行为）
- **解法**: archive/temp/ 超过 50 个文件时，删除最旧的
- **归入**: 考虑归入阶段四（通用化/精简）
