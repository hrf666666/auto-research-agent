"""GLM provider contract tests."""
from __future__ import annotations

import pytest

from core.agents import AgentDispatcher
from tests.conftest import FakeAPIStatusError


def test_glm_endpoint_misconfiguration_raises(monkeypatch):
    d = AgentDispatcher(model="auto", provider="glm_token_plan", tools=None)

    with pytest.raises(RuntimeError, match="Coding Plan endpoint misconfigured"):
        d._call_openai_compatible(
            system="sys",
            messages=[{"role": "user", "content": "hi"}],
            base_url="https://open.bigmodel.cn/api/paas/v4",
            api_key="fake",
            model="glm-5",
            provider_label="token_plan[glm_token_plan]",
            task_tier="think",
        )


def test_403_1220_tries_next_model(monkeypatch):
    d = AgentDispatcher(model="auto", provider="glm_token_plan", tools=None)
    AgentDispatcher._provider_health.clear()
    monkeypatch.setenv("GLM_CODING_PLAN_API_KEY", "fake")
    monkeypatch.delenv("ALI_TOKEN_PLAN_API_KEY", raising=False)

    calls = []

    def fake_call(self, **kwargs):
        calls.append(kwargs["model"])
        if len(calls) == 1:
            raise FakeAPIStatusError(
                "model forbidden",
                status_code=403,
                body={"error": {"code": "1220", "message": "model not authorized"}},
            )
        return '{"action":"wait","reason":"ok"}'

    monkeypatch.setattr(AgentDispatcher, "_call_openai_compatible", fake_call)

    text, _trace = d._call_llm("sys", [{"role": "user", "content": "hi"}], task_tier="think")

    assert text == '{"action":"wait","reason":"ok"}'
    assert calls[:2] == ["glm-5.2", "glm-5.1"]
