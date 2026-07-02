"""Pydantic v2 models."""
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class InferenceResult(BaseModel):
    label: Literal["POSITIVE", "NEGATIVE"]
    score: float = Field(ge=0.0, le=1.0)
    is_edge_case: bool
    word_count: int
    char_count: int


class ClauseInsight(BaseModel):
    explanation: str
    confidence_note: str
    suggestion: str
    triggered_at: datetime
