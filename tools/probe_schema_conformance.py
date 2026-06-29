#!/usr/bin/env python3
"""
Phase 0 probe: 实测 GLM (glm-5.1, zai.ZhipuAiClient) 对 tool_choice 强制的服从度。

测试三种策略，每种 N 次，统计:
  (a) 模型调用了工具的次数（无论字段是否齐全）
  (b) 调用工具且填满所有 required 字段 + 字段值合法 的次数  ← 真正的成功
  (c) 输出裸文本（Markdown/散文，无工具调用）的次数

三种策略:
  STRATEGY A: tool_choice="auto" + dummy respond工具(现状基线)
  STRATEGY B: tool_choice="required" + decision schema工具(Phase 2 默认路线)
  STRATEGY C: tool_choice="required" + decision schema + 失败重试1次(带反馈)

判定 Phase 2 路线(基于 STRATEGY B 的 b 比例):
  b >= 70% → 走 schema 强制路线
  b 在 35-70% → schema 强制 + 重试补充
  b < 35% → schema 路线失败, 考虑 response_format=json_object 降级
"""
import os
import sys
import json
import time
import re
import signal

class CallTimeout(Exception):
    pass

def _timeout_handler(signum, frame):
    raise CallTimeout("single call exceeded wall-clock limit")

# 确保能 import 项目模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

N_PER_STRATEGY = 10  # 每策略测 10 次 (单次~25s, 10次~4-5min, 可在前台跑完)

# ── 复刻真实 leader 场景 ──
SYSTEM_PROMPT = """You are the strategic leader of an autonomous depth-estimation research agent.

## OUTPUT FORMAT — CRITICAL
Your response MUST be a JSON object on the FIRST line. No markdown, no headers, no preamble.

The JSON must contain these fields:
- action: "experiment" | "paper_research" | "wait" | "analyze"
- agent: "code" | "paper_researcher" (omit if action != experiment/paper_research)
- task: concrete description of what to do
- hypothesis: the falsifiable claim this tests
- success_criteria: concrete measurable criterion, e.g. "val_MAE < 0.15"
- claim_type: "causal" | "correlational" | "null"
"""

# 模拟一个真实的 THINK 输入: 刚跑完一个实验, 要决定下一步
USER_INPUT = """## Current State
- Cycle 3 of 10
- Last experiment: V30 Energy-Guided Cost Volume, 50 epochs, completed
- Results: best val_MAE = 0.1440 (epoch 38), but Lambertian_MAE = 0.3641 (target < 0.16, NOT met)
- Overall MAE met target (< 0.20), Lambertian far from target
- No ablation/control run exists (energy_guided was always on)

## Memory
- Phase 1 (material classification) confirmed dead end (AUC ~0.557)
- Cost volume direction is the locked research direction

## What should the next experiment be? Output your decision as JSON."""


def get_client_and_model():
    """复刻 agents.py 的 GLM 客户端构造."""
    from zai import ZhipuAiClient
    api_key = os.environ.get("GLM_CODING_PLAN_API_KEY")
    if not api_key:
        print("ERROR: GLM_CODING_PLAN_API_KEY not set", file=sys.stderr)
        sys.exit(1)
    # Coding plan endpoint (must contain /coding/)
    base_url = "https://open.bigmodel.cn/api/coding/paas/v4"
    client = ZhipuAiClient(api_key=api_key, base_url=base_url, timeout=120.0, max_retries=1)
    model = "glm-5.1"  # strong model for think tier
    return client, model


def call_once(client, model, tool_choice, tools, messages, feedback=None):
    """发起一次调用, 返回 (tool_called, fields_ok, raw_text, error)."""
    api_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in messages:
        api_messages.append(m)
    if feedback:
        api_messages.append({"role": "assistant", "content": feedback.get("prev", "")})
        api_messages.append({"role": "user", "content": feedback["msg"]})

    create_kwargs = dict(
        model=model,
        max_tokens=2048,
        messages=api_messages,
        tools=tools,
        tool_choice=tool_choice,
        stream=True,
        tool_stream=True,
    )
    # thinking enabled (think tier, like real leader)
    create_kwargs["thinking"] = {"type": "enabled"}

    content_parts = []
    tool_calls = {}  # idx -> {name, arguments}
    finish_reason = None
    try:
        # wall-clock 保护: 流式读取不受 socket timeout 保护, 用 SIGALRM 兜底
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(90)  # 单次调用最多 90s (正常 ~25s, 留余量)
        resp = client.chat.completions.create(**create_kwargs)
        for chunk in resp:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            dc = getattr(delta, "content", None)
            if dc:
                content_parts.append(dc)
            tcs = getattr(delta, "tool_calls", None)
            if tcs:
                for tc in tcs:
                    idx = getattr(tc, "index", 0) or 0
                    if idx not in tool_calls:
                        tool_calls[idx] = {"name": "", "arguments": ""}
                    fn = getattr(tc, "function", None)
                    if fn:
                        if getattr(fn, "name", None):
                            tool_calls[idx]["name"] += fn.name
                        if getattr(fn, "arguments", None):
                            tool_calls[idx]["arguments"] += fn.arguments
            fr = getattr(chunk.choices[0], "finish_reason", None)
            if fr:
                finish_reason = fr
        signal.alarm(0)  # 取消 alarm
    except CallTimeout:
        return False, False, "", "TIMEOUT(>90s, likely stream hung)"
    except Exception as e:
        signal.alarm(0)
        return False, False, "", f"API_ERROR: {type(e).__name__}: {str(e)[:200]}"

    raw_text = "".join(content_parts)

    # 解析工具调用
    tool_called = len(tool_calls) > 0
    tool_args_str = ""
    if tool_called:
        # 取第一个工具调用
        first = list(tool_calls.values())[0]
        tool_args_str = first["arguments"]

    # 判定字段是否齐全且合法
    fields_ok = False
    if tool_called and tool_args_str:
        try:
            args = json.loads(tool_args_str)
            required = ["action", "task", "hypothesis", "success_criteria", "claim_type"]
            if all(k in args and args[k] not in (None, "", []) for k in required):
                # action 必须合法
                if args["action"] in ("experiment", "paper_research", "wait", "analyze"):
                    # claim_type 必须合法
                    if args["claim_type"] in ("causal", "correlational", "null"):
                        fields_ok = True
        except (json.JSONDecodeError, KeyError):
            fields_ok = False

    return tool_called, fields_ok, raw_text, None


def run_strategy(label, client, model, tool_choice, tools, with_retry=False):
    """跑一个策略 N 次, 返回统计."""
    print(f"\n{'='*60}")
    print(f"STRATEGY {label}: tool_choice={tool_choice!r}, retry={with_retry}")
    print(f"{'='*60}")
    a = b = c = errs = 0  # 调用工具 / 字段齐全 / 裸文本 / 错误
    details = []
    messages = [{"role": "user", "content": USER_INPUT}]

    for i in range(N_PER_STRATEGY):
        tool_called, fields_ok, raw, err = call_once(
            client, model, tool_choice, tools, messages
        )
        if err:
            errs += 1
            status = "ERR"
        elif tool_called and fields_ok:
            b += 1; a += 1; status = "OK(fields)"
        elif tool_called and not fields_ok:
            a += 1; status = "TOOL(bad_fields)"
            # 重试一次
            if with_retry:
                fb = {"prev": raw[:300], "msg": "Your tool call had missing/invalid fields. Output a complete respond call with action, task, hypothesis, success_criteria, claim_type."}
                tc2, fo2, raw2, err2 = call_once(client, model, tool_choice, tools, messages, feedback=fb)
                if err2:
                    status += "→retry_ERR"
                elif fo2:
                    b += 1; status += "→retry_OK"
                else:
                    status += "→retry_FAIL"
        else:
            c += 1; status = "RAW_TEXT"
        details.append((i+1, status, raw[:80] if raw else "(tool call)"))
        print(f"  [{i+1:2d}/{N_PER_STRATEGY}] {status}", flush=True)
        time.sleep(0.5)  # 避免限流

    print(f"\n--- {label} 汇总 (n={N_PER_STRATEGY}, errs={errs}) ---")
    valid = N_PER_STRATEGY - errs
    print(f"  (a) 调用工具: {a}/{valid} ({100*a//valid if valid else 0}%)")
    print(f"  (b) 字段齐全合法: {b}/{valid} ({100*b//valid if valid else 0}%)  ← 决定Phase2路线")
    print(f"  (c) 裸文本: {c}/{valid} ({100*c//valid if valid else 0}%)")
    return {"a": a, "b": b, "c": c, "errs": errs, "valid": valid, "details": details}


def main():
    print("Phase 0 Probe: GLM schema-conformance test")
    print(f"Model: glm-5.1 | N per strategy: {N_PER_STRATEGY}")
    client, model = get_client_and_model()

    # 工具定义 (OpenAI function 格式, 复刻 agents.py:1067-1074)
    dummy_tool_old = [{
        "type": "function", "function": {
            "name": "respond",
            "description": "Return your analysis and decision as structured JSON.",
            "parameters": {"type": "object", "properties": {"response": {"type": "string", "description": "Your full response text"}}, "required": ["response"]}
        }
    }]
    decision_tool = [{
        "type": "function", "function": {
            "name": "respond",
            "description": "Output your research decision. You MUST call this tool with all fields.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["experiment", "paper_research", "wait", "analyze"]},
                    "task": {"type": "string", "description": "Concrete description of what to do"},
                    "hypothesis": {"type": "string", "description": "The falsifiable claim this tests"},
                    "success_criteria": {"type": "string", "description": "e.g. 'val_MAE < 0.15'"},
                    "claim_type": {"type": "string", "enum": ["causal", "correlational", "null"]},
                },
                "required": ["action", "task", "hypothesis", "success_criteria", "claim_type"],
            }
        }
    }]

    results = {}
    # A: 现状基线 (auto + 旧dummy)
    results["A"] = run_strategy("A(baseline)", client, model, "auto", dummy_tool_old)
    # B: Phase2默认 (required + decision schema)
    results["B"] = run_strategy("B(schema+required)", client, model, "required", decision_tool)
    # C: B + 重试
    results["C"] = run_strategy("C(schema+required+retry)", client, model, "required", decision_tool, with_retry=True)

    # ── 判定 Phase 2 路线 ──
    print("\n" + "="*60)
    print("PHASE 2 ROUTE DECISION (based on Strategy B 的 b%)")
    print("="*60)
    b_pct = results["B"]["b"] / results["B"]["valid"] if results["B"]["valid"] else 0
    c_b = results["C"]["b"] / results["C"]["valid"] if results["C"]["valid"] else 0
    print(f"Strategy B fields-ok rate: {results['B']['b']}/{results['B']['valid']} = {b_pct:.0%}")
    print(f"Strategy C (with retry) fields-ok rate: {results['C']['b']}/{results['C']['valid']} = {c_b:.0%}")
    if b_pct >= 0.70:
        print("→ DECISION: 走 schema 强制路线 (b>=70%). Phase 2 用 tool_choice=required + decision schema.")
    elif b_pct >= 0.35:
        print(f"→ DECISION: schema 强制部分有效 ({b_pct:.0%}), 叠加重试. 重试提升到 {c_b:.0%}.")
        if c_b >= 0.70:
            print("  重试后达标. Phase 2 = schema + 1次带反馈重试.")
        else:
            print(f"  重试后仍 {c_b:.0%} < 70%. Phase 2 = schema + 重试, 但 M3/M4 预期需留大余量.")
    else:
        print(f"→ DECISION: schema 强制失败 (b={b_pct:.0%}<35%). Phase 2 降级为 response_format=json_object + 后处理.")
        print("  M3/M4/M7 预期需重新评估——结构化字段质量无保障.")

    # 存结果
    out = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": model,
        "n_per_strategy": N_PER_STRATEGY,
        "strategy_A_baseline": {k: v for k, v in results["A"].items() if k != "details"},
        "strategy_B_schema_required": {k: v for k, v in results["B"].items() if k != "details"},
        "strategy_C_schema_required_retry": {k: v for k, v in results["C"].items() if k != "details"},
        "phase2_route_b_pct": round(b_pct, 3),
        "phase2_route_c_pct": round(c_b, 3),
    }
    outpath = os.path.join(os.path.dirname(__file__), "..", "docs", "phase0_probe_result.json")
    with open(outpath, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n结果已存: {outpath}")


if __name__ == "__main__":
    main()
