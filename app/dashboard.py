"""SentiA-D Streamlit dashboard.

Seven sections: header, input, result, Claude insight, session metrics,
history table, sidebar. Charts use the grr-analysis-tool palette (see CLAUDE.md)
so the two portfolio repos read as one visual ecosystem.
"""
from datetime import datetime

import altair as alt
import pandas as pd
import streamlit as st

from src import (
    EDGE_CASE_HIGH,
    EDGE_CASE_LOW,
    NEGATIVE_COLOR,
    NEUTRAL_COLOR,
    POSITIVE_COLOR,
)
from src.claude_synthesizer import MODEL as CLAUDE_MODEL
from src.inference import MODEL as HF_MODEL
from src.inference import SentimentInferencer
from src.metrics import edge_case_count, label_distribution

REPO_URL = "https://github.com/yang-k-calvin/SentiA-D"
LABEL_COLORS = {"POSITIVE": POSITIVE_COLOR, "NEGATIVE": NEGATIVE_COLOR}

st.set_page_config(page_title="SentiA-D", page_icon="🎭", layout="wide")


@st.cache_resource
def _inferencer():
    return SentimentInferencer()  # loaded once, shared across reruns


if "history" not in st.session_state:
    st.session_state.history = []  # list[dict], newest last

# ── 1. Header ────────────────────────────────────────────────────────────────
st.title("🎭 SentiA-D")
st.caption(f"distilBERT sentiment + Claude Synthesizer · [{HF_MODEL}]({REPO_URL})")

# ── 7. Sidebar (thresholds, model info, palette credit) ──────────────────────
with st.sidebar:
    st.header("Edge-case band")
    low, high = st.slider(
        "Confidence range routed to the Synthesizer",
        0.0, 1.0, (EDGE_CASE_LOW, EDGE_CASE_HIGH), 0.01,
    )
    with st.expander("Model info"):
        st.write(f"**Classifier:** `{HF_MODEL}`")
        st.write(f"**Synthesizer:** `{CLAUDE_MODEL}`")
        st.write("Claude explains borderline cases — it never reclassifies.")
    st.divider()
    st.caption(
        "Chart palette mirrors "
        "[grr-analysis-tool](https://github.com/yang-k-calvin/grr-analysis-tool)."
    )

# ── 2. Input panel ───────────────────────────────────────────────────────────
text = st.text_area("Text to analyze", key="input_text", height=120)
if st.button("Analyze", type="primary") and text.strip():
    r = _inferencer().analyze(text)
    edge = low <= r.score <= high  # honor the sidebar sliders, not just constants
    st.session_state.last = r
    st.session_state.history.append({
        "Timestamp": datetime.now().strftime("%H:%M:%S"),
        "Text": text[:60] + ("…" if len(text) > 60 else ""),
        "Label": r.label,
        "Confidence": round(r.score, 3),
        "Edge?": "⚠️" if edge else "",
        "_edge": edge,
    })

# ── 3. Result panel ──────────────────────────────────────────────────────────
if "last" in st.session_state:
    r = st.session_state.last
    edge = st.session_state.history[-1]["_edge"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Label", r.label)
    c2.metric("Confidence", f"{r.score:.1%}")
    c3.metric("Edge case", "Yes ⚠️" if edge else "No")
    st.progress(r.score, text=f"Confidence gauge · {r.score:.1%}")

    # ── 4. Claude Insight panel (edge cases only) ────────────────────────────
    if edge:
        st.info(
            f"**Synthesizer** — this result sits in the {low:.0%}–{high:.0%} "
            "ambiguous band. Run the Synthesizer for a plain-English explanation "
            "of why it's uncertain.",
            icon="🧩",
        )

# ── 5. Session metrics ───────────────────────────────────────────────────────
if st.session_state.history:
    df = pd.DataFrame(st.session_state.history)
    edges = edge_case_count(df["_edge"].tolist())
    st.subheader(f"Session metrics · {edges}/{len(df)} edge cases")
    m1, m2 = st.columns(2)

    with m1:
        dist = pd.DataFrame(
            label_distribution(df["Label"].tolist()).items(),
            columns=["Label", "Count"],
        )
        chart = alt.Chart(dist).mark_bar().encode(
            x="Label:N",
            y="Count:Q",
            color=alt.Color("Label:N", scale=alt.Scale(
                domain=list(LABEL_COLORS), range=list(LABEL_COLORS.values())),
                legend=None),
        )
        st.altair_chart(chart, use_container_width=True)

    with m2:
        hist = alt.Chart(df).mark_bar(color=NEUTRAL_COLOR).encode(
            x=alt.X("Confidence:Q", bin=alt.Bin(maxbins=20)),
            y="count()",
        )
        st.altair_chart(hist, use_container_width=True)

    # ── 6. History table ─────────────────────────────────────────────────────
    st.subheader("History")
    st.dataframe(
        df.drop(columns="_edge"), use_container_width=True, hide_index=True,
    )
