"""Claude Synthesizer role.

Receives ONLY borderline predictions (55%–75% confidence) and explains, in
plain English, *why* the result is ambiguous. It NEVER reclassifies — the
distilBERT label is authoritative. The role boundary is the SYSTEM_PROMPT
constant below, kept verbatim so it is auditable at a glance.
"""
import json
import logging
import os
import time
from datetime import datetime, timezone

from .inference import is_edge_case  # single source; no duplicated band check
from .schemas import ClauseInsight

is_borderline = is_edge_case  # back-compat alias

MODEL = "claude-sonnet-5"
MAX_TOKENS = 300
RETRIES = 3

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """
You are the Synthesizer component of a sentiment analysis pipeline.
You receive predictions that fall in the ambiguous confidence band
(55%–75%) from a distilBERT classifier. Your job is to explain, in
plain English, why this result might be uncertain — not to reclassify.

Always respond with valid JSON only. No preamble, no markdown fences.
Schema: { "explanation": str, "confidence_note": str, "suggestion": str }

- explanation: 1–2 sentences. What linguistic features make this text
  ambiguous? (e.g., mixed sentiment, sarcasm, domain-specific language)
- confidence_note: 1 sentence. What does a score of X% mean in practice?
- suggestion: 1 sentence. What additional context would resolve ambiguity?
"""


def explain(text: str, label: str, confidence: float) -> ClauseInsight:
    from anthropic import Anthropic  # ponytail: lazy import, keeps import cheap

    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    prompt = f"Text: {text!r}\nLabel: {label}\nConfidence: {confidence:.0%}"

    for attempt in range(RETRIES):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            data = json.loads(response.content[0].text)
            return ClauseInsight(**data, triggered_at=datetime.now(timezone.utc))
        except Exception:
            log.warning("Synthesizer attempt %d/%d failed", attempt + 1, RETRIES,
                        exc_info=True)
            if attempt < RETRIES - 1:
                time.sleep(5 * (2 ** attempt))  # 5s → 10s → 20s

    # All retries exhausted → graceful fallback so the pipeline never crashes.
    return ClauseInsight(
        explanation="Synthesizer unavailable — Claude API call failed.",
        confidence_note="",
        suggestion="",
        triggered_at=datetime.now(timezone.utc),
    )
