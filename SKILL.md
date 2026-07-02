---
name: sentia-d
description: Sentiment analysis with a Claude Synthesizer that explains borderline predictions. Use when classifying text sentiment and surfacing why low-confidence (0.55–0.75) cases are ambiguous. Claude explains, never reclassifies.
---

# SentiA-D

HF sentiment pipeline classifies text. Predictions in the 0.55–0.75 confidence
band are "borderline" and get routed to the Claude Synthesizer, which returns a
plain-English explanation of the ambiguity — it does **not** change the label.

## Flow

1. `inference.classify(texts)` → labels + confidences (`src/inference.py`)
2. Borderline (`EDGE_CASE_LOW ≤ conf ≤ EDGE_CASE_HIGH`) routed to
   `claude_synthesizer.explain(text, label, confidence)` (`src/claude_synthesizer.py`)
3. `metrics` computes accuracy/precision/recall/F1 (`src/metrics.py`)
4. `app/dashboard.py` renders results with the chart palette (see CLAUDE.md)

## Synthesizer contract

Returns JSON `{ explanation, confidence_note, suggestion }`.
Never reclassifies — the HF label is authoritative.
