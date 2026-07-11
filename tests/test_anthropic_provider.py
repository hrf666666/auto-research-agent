"""Provider compatibility tests for optional Anthropic support."""
from __future__ import annotations

from core.agents import AgentDispatcher


def test_find_last_assistant_text_handles_anthropic_blocks():
    messages = [{
        "role": "assistant",
        "content": [
            {"type": "text", "text": "partial answer"},
            {"type": "tool_use", "id": "toolu_1", "name": "read_file", "input": {}},
        ],
    }]

    assert AgentDispatcher._find_last_assistant_text(messages, "fallback") == "partial answer"


def test_anthropic_auto_model_maps_to_real_model():
    d = AgentDispatcher(model="auto", provider="anthropic", tools=None)

    assert d._resolve_anthropic_model() == "claude-sonnet-4-6"


def test_anthropic_explicit_model_preserved():
    d = AgentDispatcher(model="claude-opus-4-6", provider="anthropic", tools=None)

    assert d._resolve_anthropic_model() == "claude-opus-4-6"
