"""Tests for the register-driven Context Schema (core/context_keys.py).

These tests enforce the SINGLE SOURCE OF TRUTH contract: every key that
loop.py injects MUST have a serializer in the registry, and every serializer
MUST produce output when given a non-empty value. This prevents the class of
bug where 37 of 48 context keys were computed but silently dropped.

The key invariant: if a subsystem computes a context value, the registry
MUST serialize it into the prompt. No silent drops.
"""
from __future__ import annotations

import pytest

from core.context_keys import (
    serialize_context, THINK_KEY_NAMES, REFLECT_KEY_NAMES, ALL_KEY_NAMES,
    get_keys_for_phase, ContextKey,
)


class TestRegistryCompleteness:
    """The registry must cover every key that subsystems inject."""

    def test_think_keys_nonempty(self):
        assert len(THINK_KEY_NAMES) >= 20, "THINK_KEYS should have at least 20 keys"


    def test_no_ghost_keys_from_removed_modules(self):
        """Keys from removed modules (v16.1 cleanup) must not linger."""
        ghosts = {"adaptive_thresholds", "implementation_progress",
                  "sandbox_design_guidance", "plan_compliance_warning",
                  "quick_benchmark_warning"}
        actual = ALL_KEY_NAMES
        lingering = ghosts & actual
        assert not lingering, f"Ghost keys from removed modules still in registry: {lingering}"

    def test_every_key_has_serializer(self):
        """Every registered key MUST have a serializer — no key should be
        silently unserializable (the root cause of the 37-key drop bug)."""
        for key in THINK_KEY_NAMES | REFLECT_KEY_NAMES:
            all_keys = list(get_keys_for_phase("think")) + list(get_keys_for_phase("reflect"))
            ck = next((k for k in all_keys if k.name == key), None)
            assert ck is not None, f"Key '{key}' not found in registry"
            # serializer can be None only for legacy keys being phased out;
            # flag them explicitly so they're visible.
            if ck.serializer is None:
                pytest.fail(f"Key '{key}' has no serializer — it will be silently dropped")


class TestSerialization:
    """The serialize_context function must include every non-empty key."""


    def test_empty_values_are_skipped(self):
        """Empty/None values must not produce empty sections."""
        ctx = {"domain_knowledge": "", "cycle": None, "brief": "keep this"}
        prompt = serialize_context(ctx, "think")
        assert "keep this" in prompt
        assert "Domain Knowledge" not in prompt  # empty → skipped

    def test_think_and_reflect_use_different_key_sets(self):
        """THINK and REFLECT must not serialize each other's phase-specific keys."""
        ctx = {"experiment_result": {"x": 1}, "verify_diagnosis": ["d"]}
        think_prompt = serialize_context(ctx, "think")
        reflect_prompt = serialize_context(ctx, "reflect")
        # experiment_result is REFLECT-only
        assert "Experiment Result" in reflect_prompt
        assert "Experiment Result" not in think_prompt

    def test_truncation_works(self):
        """Long values must be truncated to prevent context overflow."""
        long_text = "A" * 10000
        ctx = {"domain_knowledge": long_text}
        prompt = serialize_context(ctx, "think")
        assert "truncated" in prompt
        assert len(prompt) < 10000  # truncated well below the raw size


class TestIntegrationWithFormatLeaderInput:
    """The registry must be wired into AgentDispatcher._format_leader_input."""

    def test_format_leader_input_includes_domain_knowledge(self):
        """End-to-end: domain_knowledge injected by loop.py must reach the prompt."""
        from core.agents import AgentDispatcher
        d = AgentDispatcher(model="auto", provider="glm_token_plan", tools=None)
        ctx = {
            "brief": "test", "cycle": 1,
            "domain_knowledge": "IMPORTANT DOMAIN INSIGHT",
        }
        prompt = d._format_leader_input("think", ctx)
        assert "IMPORTANT DOMAIN INSIGHT" in prompt

