import json
import sys
import types

import pytest

from src import claude_synthesizer as cs
from src.schemas import ClauseInsight

_GOOD_JSON = json.dumps({
    "explanation": "Mixed sentiment: praise followed by a complaint.",
    "confidence_note": "65% means the model is only mildly confident.",
    "suggestion": "Clarify which clause carries the writer's intent.",
})


def _install_fake_anthropic(monkeypatch, create):
    """Wire a fake `anthropic.Anthropic` whose messages.create runs `create`."""
    class _Msg:
        def __init__(self, text):
            self.content = [types.SimpleNamespace(text=text)]

    class _Client:
        def __init__(self, **kw):
            self.messages = types.SimpleNamespace(create=create)

    fake = types.ModuleType("anthropic")
    fake.Anthropic = _Client
    monkeypatch.setitem(sys.modules, "anthropic", fake)
    monkeypatch.setattr(cs.time, "sleep", lambda *_: None)  # no real waiting
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    return _Msg


def test_borderline_band():
    assert cs.is_borderline(0.55) and cs.is_borderline(0.75)
    assert not cs.is_borderline(0.54) and not cs.is_borderline(0.99)


def test_returns_insight_schema(monkeypatch):
    Msg = _install_fake_anthropic(monkeypatch, lambda **kw: Msg(_GOOD_JSON))
    insight = cs.explain("great but slow", "POSITIVE", 0.65)
    assert isinstance(insight, ClauseInsight)
    assert insight.explanation.startswith("Mixed sentiment")


def test_retry_on_failure(monkeypatch):
    calls = {"n": 0}

    def flaky(**kw):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise RuntimeError("transient")
        return Msg(_GOOD_JSON)

    Msg = _install_fake_anthropic(monkeypatch, flaky)
    insight = cs.explain("meh", "POSITIVE", 0.6)
    assert calls["n"] == 3  # failed twice, succeeded on third
    assert insight.explanation.startswith("Mixed sentiment")


def test_full_failure_fallback(monkeypatch):
    calls = {"n": 0}

    def always_fail(**kw):
        calls["n"] += 1
        raise RuntimeError("boom")

    _install_fake_anthropic(monkeypatch, always_fail)
    insight = cs.explain("meh", "POSITIVE", 0.6)
    assert calls["n"] == cs.RETRIES
    assert "unavailable" in insight.explanation.lower()
    assert insight.confidence_note == ""
