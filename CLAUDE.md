# SentiA-D

Sentiment analysis with a Claude Synthesizer for borderline cases.

## Claude role: Synthesizer

Claude receives **only borderline predictions** (confidence in the 0.55–0.75
band) and returns a plain-English explanation of *why* the case is ambiguous.

**NEVER reclassifies.** The HF model owns the label; Claude only explains.

```python
EDGE_CASE_LOW  = 0.55
EDGE_CASE_HIGH = 0.75
```

## Chart palette (mirrors grr-analysis-tool)

```python
POSITIVE_COLOR = "#22c55e"  # green-500
NEGATIVE_COLOR = "#ef4444"  # red-500
NEUTRAL_COLOR  = "#94a3b8"  # slate-400
EDGE_COLOR     = "#f59e0b"  # amber-500
```

## Claude API contract

- model: `claude-sonnet-4-6`
- max_tokens: 300
- response_format: JSON `{ explanation, confidence_note, suggestion }`

## Layout

- `src/inference.py` — HF pipeline wrapper
- `src/claude_synthesizer.py` — Synthesizer role
- `src/metrics.py` — classification metrics
- `src/schemas.py` — Pydantic v2 models
- `app/dashboard.py` — Streamlit entry point
