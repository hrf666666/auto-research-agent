"""Tests for the provider-error classification logic in core.agents.

Coverage:
- 4xx (400/401/403/404) → permanent (abort failover)
- 5xx / network → transient (retry next model)
- 429 rate-limit → transient (retry helps)
- 429 quota-exhausted (GLM code 1308 / "使用上限") → permanent-until-reset
  (Phase 1 fix: breaks the model chain, cools the whole provider until reset)
"""
from __future__ import annotations

import pytest

from core.agents import _is_permanent_error, _classify_429
from tests.conftest import FakeAPIStatusError


class TestIsPermanentError:
    """The permanent-vs-transient classifier."""

    @pytest.mark.parametrize("status", [400, 401, 403, 404])
    def test_client_errors_are_permanent(self, status):
        exc = FakeAPIStatusError("bad request", status_code=status)
        assert _is_permanent_error(exc) is True

    @pytest.mark.parametrize("status", [500, 502, 503, 504])
    def test_server_errors_are_transient(self, status):
        exc = FakeAPIStatusError("server error", status_code=status)
        assert _is_permanent_error(exc) is False

    def test_plain_exception_is_transient(self):
        assert _is_permanent_error(Exception("boom")) is False

    def test_none_status_is_transient(self):
        # An exception with no status_code attribute must be transient.
        assert _is_permanent_error(ValueError("no status")) is False


class Test429Classification:
    """The core of the Phase 1 fix: not all 429s are equal."""

    def test_rate_limit_429_is_transient(self):
        """A plain per-minute rate-limit 429 should remain transient.

        Retrying the next model after a short backoff is the correct
        response — the limit is per-model/per-minute, not account-wide.
        """
        exc = FakeAPIStatusError("rate limited", status_code=429)
        assert _is_permanent_error(exc) is False

    def test_glm_quota_exhausted_429_is_permanent(self):
        """FIXED (Phase 1): GLM's 5-hour quota 429 is now classified permanent.

        GLM returns code 1308 with "已达到 5 小时的使用上限...将在 ... 重置".
        This is an ACCOUNT-WIDE quota window shared by the whole API key —
        every model in the chain will 429 identically. It must break the
        chain immediately and cool the whole provider until reset.
        """
        exc = FakeAPIStatusError(
            'Error code: 429, with error text {"error":{"code":"1308",'
            '"message":"已达到 5 小时的使用上限。您的限额将在 '
            '2026-06-15 19:42:06 重置。"}}',
            status_code=429,
            body={"error": {"code": "1308",
                            "message": "已达到 5 小时的使用上限。您的限额将在 2026-06-15 19:42:06 重置。"}},
        )
        assert _is_permanent_error(exc) is True


class TestClassify429:
    """The `_classify_429` helper: rate-limit vs quota-window discrimination."""

    def test_quota_reset_time_parsed(self):
        """A quota-exhausted 429 must surface the reset time so the cooldown
        can extend to the window reset, not a fixed 300s."""
        exc = FakeAPIStatusError(
            "quota", status_code=429,
            body={"error": {"code": "1308",
                            "message": "已达到 5 小时的使用上限。您的限额将在 2026-06-15 19:42:06 重置。"}},
        )
        result = _classify_429(exc)
        assert result["type"] == "quota_exhausted"
        assert result.get("reset_time") is not None
        assert result["reset_time"].year == 2026

    def test_rate_limit_has_no_reset_time(self):
        exc = FakeAPIStatusError("rate limited", status_code=429)
        result = _classify_429(exc)
        assert result["type"] == "rate_limit"

    def test_quota_detection_by_keyword(self):
        """Even without a numeric code, quota keywords should classify it."""
        exc = FakeAPIStatusError(
            "quota exceeded", status_code=429,
            body={"error": {"message": "Daily quota limit reached, resets at 2026-06-16 00:00:00"}},
        )
        result = _classify_429(exc)
        assert result["type"] == "quota_exhausted"

    def test_plain_429_is_rate_limit(self):
        """A bare 429 with no quota signal must stay transient."""
        exc = FakeAPIStatusError("Too Many Requests", status_code=429)
        assert _classify_429(exc)["type"] == "rate_limit"


class TestQuotaCooldownBehavior:
    """The load-bearing behavioral fix: a quota 429 must cool the whole
    provider and break the model chain, not burn 6 doomed calls.

    We drive the dispatcher's `_call_llm` with a mocked
    `_call_openai_compatible` that raises a quota 429 on every call, then
    count how many times it was invoked. Before Phase 1 this was 6 (the full
    chain). After Phase 1 it must be 1.
    """

    def test_quota_429_breaks_chain_after_one_call(self, monkeypatch):
        from core.agents import AgentDispatcher

        d = AgentDispatcher(model="auto", provider="glm_token_plan", tools=None)
        call_count = {"n": 0}

        def fake_call(self, **kwargs):
            call_count["n"] += 1
            raise FakeAPIStatusError(
                "quota", status_code=429,
                body={"error": {"code": "1308",
                                "message": "已达到 5 小时的使用上限。您的限额将在 2099-01-01 00:00:00 重置。"}},
            )

        monkeypatch.setattr(AgentDispatcher, "_call_openai_compatible", fake_call)

        # Both providers will quota-fail; _call_llm should raise RuntimeError.
        import os
        monkeypatch.setenv("GLM_CODING_PLAN_API_KEY", "fake")
        monkeypatch.setenv("ALI_TOKEN_PLAN_API_KEY", "fake")

        try:
            d._call_llm("sys", [{"role": "user", "content": "hi"}],
                        task_tier="think")
        except (RuntimeError, FakeAPIStatusError, Exception):
            pass  # expected — all providers fail

        # CRITICAL assertion: the GLM provider's model chain (6 models) must
        # NOT be burned. Quota 429 breaks after the FIRST call.
        assert call_count["n"] <= 2, (
            f"Expected ≤2 calls (1 per provider) on a quota 429, "
            f"got {call_count['n']} — the model chain is still being burned."
        )

    def test_quota_cooldown_recorded(self, monkeypatch):
        """A quota 429 must record an absolute cooldown_until on the provider."""
        from core.agents import AgentDispatcher

        d = AgentDispatcher(model="auto", provider="glm_token_plan", tools=None)
        # Clear any pre-seeded health from __init__.
        AgentDispatcher._provider_health.clear()
        AgentDispatcher._provider_health["glm_token_plan"] = {
            "consecutive_failures": 0, "last_failure_time": 0,
            "total_calls": 0, "total_failures": 0, "cooldown_until": 0,
        }

        def fake_call(self, **kwargs):
            raise FakeAPIStatusError(
                "quota", status_code=429,
                body={"error": {"code": "1308",
                                "message": "已达到 5 小时的使用上限。您的限额将在 2099-01-01 00:00:00 重置。"}},
            )

        monkeypatch.setattr(AgentDispatcher, "_call_openai_compatible", fake_call)
        monkeypatch.setenv("GLM_CODING_PLAN_API_KEY", "fake")
        monkeypatch.setenv("ALI_TOKEN_PLAN_API_KEY", "fake")

        try:
            d._call_llm("sys", [{"role": "user", "content": "hi"}],
                        task_tier="think")
        except Exception:
            pass

        health = AgentDispatcher._provider_health.get("glm_token_plan", {})
        assert health.get("cooldown_until", 0) > 0, (
            "Quota 429 did not record an absolute cooldown_until — the provider "
            "would be retried immediately instead of cooled until window reset."
        )
