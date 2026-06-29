"""Tests for the Reform v21 root-cause fix: task-aware leader response parser.

ROOT CAUSE (proven by black-box probe, Reform v21 angle-3):
  The leader response parser demanded an ``action`` key in EVERY parsed JSON,
  but REFLECT's schema (milestone/decision/dead_end/active_problem/causal_link/
  lesson) contains NO ``action`` key. So every valid REFLECT JSON was rejected
  as "Unparseable", REFLECT was reported as "100% failing", and the wrong root
  cause ("LLM writes prose") was logged. Black-box testing proved GLM returns
  valid REFLECT JSON 3/3 — the parser was the bug.

FIX: the parser now selects the acceptance key by task:
  - think   → must contain ``action``
  - reflect → must contain ``milestone`` or ``decision``

These tests (a) pin the fix, (b) guard against regression to the old hard
``action`` requirement, (c) verify that when REFLECT succeeds, the fields that
were previously ALWAYS dropped (dead_end / causal_link / lesson) actually reach
the memory layer.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Real GLM REFLECT outputs (captured by the angle-3 black-box probe).
# All are valid JSON, all lack an ``action`` key, all were REJECTED by the old
# parser. They are the canonical regression fixtures.
# ─────────────────────────────────────────────────────────────────────────────
REAL_REFLECT_OUTPUTS = [
    '{"milestone": "Depth model achieved val_MAE 0.149, beating target <0.20", "decision": "Proceed to ablations", "dead_end": null, "active_problem": null, "causal_link": "Steady loss 2.1->0.31 shows pipeline well-configured", "lesson": "2400/600 split baseline established"}',
    '{"milestone": "V30 cost volume trained 50 epochs, val_MAE=0.149", "decision": "Run Lambertian ablation next", "dead_end": null, "active_problem": "Lambertian MAE 0.364 far from 0.16 target", "causal_link": null, "lesson": null}',
    '{"milestone": "baseline established", "decision": "explore augmentation", "dead_end": null, "active_problem": null, "causal_link": null, "lesson": null}',
]
FENCED_REFLECT = "```json\n" + REAL_REFLECT_OUTPUTS[0] + "\n```"
THINK_OUTPUT = '{"action": "experiment", "task": "train v30", "hypothesis": "X", "success_criteria": "val_MAE<0.2", "claim_type": "causal"}'


# ─────────────────────────────────────────────────────────────────────────────
# Parser-level: schema selection by task
# ─────────────────────────────────────────────────────────────────────────────
class TestParserSchemaSelection:
    """The parser must accept REFLECT JSON (no action key) for task=reflect."""

    def test_reflect_outputs_accepted(self):
        from core.agents import AgentDispatcher

        for i, s in enumerate(REAL_REFLECT_OUTPUTS, 1):
            r = AgentDispatcher._extract_first_decision_json(s, task="reflect")
            assert r is not None, f"REFLECT output {i} rejected (regression to old bug)"
            assert "milestone" in r or "decision" in r

    def test_fenced_reflect_accepted(self):
        from core.agents import AgentDispatcher

        r = AgentDispatcher._extract_first_decision_json(FENCED_REFLECT, task="reflect")
        assert r is not None
        assert "milestone" in r

    def test_think_output_still_accepted(self):
        """Regression guard: think parsing must not break."""
        from core.agents import AgentDispatcher

        r = AgentDispatcher._extract_first_decision_json(THINK_OUTPUT, task="think")
        assert r is not None
        assert r.get("action") == "experiment"

    def test_think_json_rejected_by_reflect_parser(self):
        """Schema isolation: a pure think JSON (action but no milestone/decision)
        must NOT be accepted when parsing as reflect."""
        from core.agents import AgentDispatcher

        r = AgentDispatcher._extract_first_decision_json(THINK_OUTPUT, task="reflect")
        assert r is None, "reflect parser must not accept think-schema JSON"

    def test_reflect_json_rejected_by_think_parser(self):
        """Schema isolation: a pure reflect JSON (milestone but no action) must
        NOT be accepted when parsing as think (preserves THINK safety: a
        confused reflect output must never be turned into an experiment launch)."""
        from core.agents import AgentDispatcher

        r = AgentDispatcher._extract_first_decision_json(REAL_REFLECT_OUTPUTS[0], task="think")
        assert r is None, "think parser must not accept reflect-schema JSON"

    def test_parse_leader_response_reflect_returns_fields(self):
        """_parse_leader_response(task='reflect') returns the full reflect dict,
        NOT the wait/default shell. This is the core fix: REFLECT no longer
        reports failure on valid output."""
        from core.agents import AgentDispatcher

        # Build a minimal dispatcher to call the instance method.
        d = AgentDispatcher.__new__(AgentDispatcher)
        r = d._parse_leader_response(REAL_REFLECT_OUTPUTS[0], task="reflect")
        assert r.get("milestone") == "Depth model achieved val_MAE 0.149, beating target <0.20"
        assert "causal_link" in r
        assert "lesson" in r
        # Crucially NOT the old default-wait shell
        assert r.get("action") != "wait"
        assert "Unparseable" not in r.get("reason", "")


# ─────────────────────────────────────────────────────────────────────────────
# Loop-level: REFLECT success → dead_end / causal_link / lesson reach memory
# ─────────────────────────────────────────────────────────────────────────────
@pytest.fixture
def reflect_loop(tmp_path):
    """A ResearchLoop shell wired to mock memory + dispatcher, for testing
    that REFLECT's cognitive fields (dead_end/causal_link/lesson) are persisted
    when REFLECT succeeds."""
    from core.loop import ResearchLoop

    loop = ResearchLoop.__new__(ResearchLoop)
    loop.memory = MagicMock()
    loop.memory.get_fact_for_output_dir.return_value = None
    loop.dispatcher = MagicMock()
    loop.context_pruner = MagicMock()
    loop.context_pruner.prune.side_effect = lambda ctx, tier: ctx
    loop.workspace = tmp_path
    loop.cycle_count = 7
    loop.memory.get_brief.return_value = ""
    loop.memory.get_log.return_value = ""
    return loop


class TestReflectSuccessPersistsCognitiveFields:
    """When REFLECT returns a rich result (the NEW normal after the parser fix),
    dead_end / causal_link / lesson must be persisted — they were ALWAYS dropped
    before because REFLECT 'failed' (parser bug) and the fallback shell had them
    as None/empty."""

    def test_dead_end_persisted(self, reflect_loop):
        loop = reflect_loop
        loop.dispatcher.dispatch_leader.return_value = {
            "milestone": "ablation done",
            "decision": "next",
            "dead_end": "FFT aggregation diverges on sparse Lambertian",
            "active_problem": None,
            "causal_link": None,
            "lesson": None,
        }
        execute_result = {"log_file": "/proj/outputs/exp/train.log"}
        loop._reflect(execute_result, verify_report=None)

        loop.memory.log_dead_end.assert_called_once_with(
            "FFT aggregation diverges on sparse Lambertian"
        )

    def test_causal_link_persisted(self, reflect_loop):
        """The field that fed causal_chain table (which was stuck at 0)."""
        loop = reflect_loop
        link = "cost_volume aggregation caused Non-Lambertian MAE to drop because energy weighting down-weights specularity"
        loop.dispatcher.dispatch_leader.return_value = {
            "milestone": "done",
            "decision": "next",
            "dead_end": None,
            "active_problem": None,
            "causal_link": link,
            "lesson": None,
        }
        execute_result = {"log_file": "/proj/outputs/exp/train.log"}
        loop._reflect(execute_result, verify_report=None)

        loop.memory.record_causal_chain_entry.assert_called_once_with(
            cycle=7, design_decision=link
        )

    def test_lesson_persisted(self, reflect_loop):
        loop = reflect_loop
        loop.dispatcher.dispatch_leader.return_value = {
            "milestone": "done",
            "decision": "next",
            "dead_end": None,
            "active_problem": None,
            "causal_link": None,
            "lesson": "always normalize cost volume before aggregation",
        }
        execute_result = {"log_file": "/proj/outputs/exp/train.log"}
        loop._reflect(execute_result, verify_report=None)

        loop.memory.record_code_review_lesson.assert_called_once()
        call_kwargs = loop.memory.record_code_review_lesson.call_args
        assert call_kwargs[1]["description"] == "always normalize cost volume before aggregation"

    def test_all_fields_persisted_together(self, reflect_loop):
        """A fully-populated REFLECT result persists every field in one pass."""
        loop = reflect_loop
        loop.dispatcher.dispatch_leader.return_value = {
            "milestone": "m", "decision": "d",
            "dead_end": "de", "active_problem": "ap",
            "causal_link": "cl", "lesson": "ln",
        }
        loop._reflect({"log_file": "/proj/outputs/exp/train.log"}, verify_report=None)

        loop.memory.log_milestone.assert_called_once()
        loop.memory.log_decision.assert_called_once()
        loop.memory.log_dead_end.assert_called_once()
        loop.memory.log_active_problem.assert_called_once()
        loop.memory.record_causal_chain_entry.assert_called_once()
        loop.memory.record_code_review_lesson.assert_called_once()

    def test_no_fallback_when_reflect_succeeds(self, reflect_loop):
        """When REFLECT returns a real milestone, the fact-spine fallback must
        NOT fire (no _milestone_source marker). Regression guard: we must not
        accidentally trigger the fallback on success."""
        loop = reflect_loop
        loop.dispatcher.dispatch_leader.return_value = {
            "milestone": "real LLM milestone",
            "decision": "real decision",
            "dead_end": None, "active_problem": None,
            "causal_link": None, "lesson": None,
        }
        result = loop._reflect({"log_file": "/proj/outputs/exp/train.log"}, verify_report=None)

        assert result["milestone"] == "real LLM milestone"
        assert result.get("_milestone_source") != "fact_spine_fallback"
