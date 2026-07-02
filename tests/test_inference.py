import logging

import pytest

from src.inference import SentimentInferencer, is_edge_case
from src.schemas import InferenceResult


def _fake_inferencer(pipe):
    # Build without downloading the real model; inject a fake pipeline.
    inf = object.__new__(SentimentInferencer)
    inf._pipe = pipe
    return inf


# ── real-model tests (skipped unless transformers + model are available) ─────
def _real():
    transformers = pytest.importorskip("transformers")  # noqa: F841
    return SentimentInferencer()


def test_positive_classification():
    assert _real().analyze("excellent!").label == "POSITIVE"


def test_negative_classification():
    assert _real().analyze("terrible").label == "NEGATIVE"


# ── pure-logic / mocked tests (always run) ───────────────────────────────────
def test_edge_case_flagging():
    assert is_edge_case(0.65) is True
    result = _fake_inferencer(lambda *a, **k: [{"label": "POSITIVE", "score": 0.65}]
                              ).analyze("meh")
    assert result.is_edge_case is True


def test_truncation_warning(caplog):
    long_text = "word " * 500  # > 450 words
    inf = _fake_inferencer(lambda *a, **k: [{"label": "POSITIVE", "score": 0.99}])
    with caplog.at_level(logging.WARNING):
        inf.analyze(long_text)
    assert any("truncates" in r.message for r in caplog.records)


def test_inference_result_bounds():
    InferenceResult(label="POSITIVE", score=0.9, is_edge_case=False,
                    word_count=1, char_count=2)
    with pytest.raises(ValueError):
        InferenceResult(label="MAYBE", score=0.9, is_edge_case=False,
                        word_count=1, char_count=2)
