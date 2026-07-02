# SentiA-D

[![CI](https://github.com/LSaiko/SentiA-D/actions/workflows/ci.yml/badge.svg)](https://github.com/LSaiko/SentiA-D/actions/workflows/ci.yml)

Sentiment analysis with a **Claude Synthesizer** for borderline cases.

An HF pipeline classifies text. Predictions in the **0.55–0.75** confidence band
are routed to Claude, which explains *why* the case is ambiguous — it never
changes the label.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # add your ANTHROPIC_API_KEY
```

## Run

```bash
streamlit run app/dashboard.py
pytest -q
```

## Layout

| Path | Role |
|------|------|
| `src/inference.py` | HF pipeline wrapper |
| `src/claude_synthesizer.py` | Synthesizer — explains, never reclassifies |
| `src/metrics.py` | Classification metrics (sklearn) |
| `src/schemas.py` | Pydantic v2 models |
| `app/dashboard.py` | Streamlit UI |

See [CLAUDE.md](CLAUDE.md) for the role boundary, edge-case band, and palette.
