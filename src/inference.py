"""HF sentiment pipeline, loaded once per SentimentInferencer instance."""
import logging

from . import EDGE_CASE_HIGH, EDGE_CASE_LOW
from .schemas import InferenceResult

MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
WORD_WARN = 450  # HF truncates at 512 tokens; warn before we silently lose text

log = logging.getLogger(__name__)


class SentimentInferencer:
    def __init__(self):
        import torch
        from transformers import pipeline

        # Cached on init — never reload per inference.
        self._pipe = pipeline(
            "text-classification",
            model=MODEL,
            device=0 if torch.cuda.is_available() else -1,
        )

    def analyze(self, text: str) -> InferenceResult:
        word_count = len(text.split())
        if word_count > WORD_WARN:
            log.warning("Text is %d words; HF truncates at 512 tokens.", word_count)

        result = self._pipe(text, truncation=True, max_length=512)[0]
        return InferenceResult(
            label=result["label"],
            score=result["score"],
            is_edge_case=is_edge_case(result["score"]),
            word_count=word_count,
            char_count=len(text),
        )


def is_edge_case(score: float) -> bool:
    return EDGE_CASE_LOW <= score <= EDGE_CASE_HIGH
