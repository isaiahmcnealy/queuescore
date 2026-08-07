"""QueueScore Streamlit app.

Runs end-to-end today against ``DummyScorer`` with baked-in demo data, so
``streamlit run src/app.py`` works with no network and no API key. Four panels:

  1. Pull live queue    — button; live gridstatus pull with cached-snapshot
                          fallback and an offline indicator.
  2. Leaderboard        — sortable table of projects by completion probability.
  3. County map         — plotly Texas map stub.
  4. Project detail     — verdict placeholder + natural-language question box.

Day-of: swap ``DummyScorer`` for the trained ``XGBScorer`` and replace the demo
frame with ``features.build_features(ingest.fetch_ercot_queue())``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Make `from src import ...` work when launched via `streamlit run src/app.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config, explain, ingest  # noqa: E402
from src.features import FEATURE_FRAME_COLUMNS  # noqa: E402
from src.scorer import DummyScorer, ScoreResult  # noqa: E402


# --------------------------------------------------------------------------- #
# Demo data (stand-in until features.build_features is implemented)
# --------------------------------------------------------------------------- #
def _demo_queue() -> pd.DataFrame:
    """A tiny model-ready frame so every panel renders before real data exists."""
    rows = [
        ("ERC-1001", 150.0, "Solar", "Harris", 2022, 640, "large"),
        ("ERC-1002", 20.0, "Battery", "Travis", 2023, 275, "small"),
        ("ERC-1003", 300.0, "Wind", "Nolan", 2021, 1010, "large"),
        ("ERC-1004", 75.0, "Solar", "Pecos", 2023, 190, "medium"),
        ("ERC-1005", 600.0, "Gas", "Bexar", 2020, 1400, "xlarge"),
        ("ERC-1006", 45.0, "Battery", "Webb", 2024, 90, "medium"),
    ]
    return pd.DataFrame(rows, columns=FEATURE_FRAME_COLUMNS)


def _score_to_frame(features: pd.DataFrame, result: ScoreResult) -> pd.DataFrame:
    """Join scores back onto the feature frame for display."""
    out = features.copy()
    out["completion_probability"] = result.probabilities
    out["_attributions"] = result.attributions
    return out.sort_values("completion_probability", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Data loading with offline fallback
# --------------------------------------------------------------------------- #
def _load_queue(pull_live: bool) -> tuple[pd.DataFrame, str]:
    """Return (feature_frame, source_label).

    ``source_label`` is one of: "demo", "live", "snapshot" — drives the offline
    indicator. Any live-pull failure degrades gracefully to snapshot or demo.
    """
    if not pull_live:
        return _demo_queue(), "demo"
    try:
        # Pull (or fall back to cached snapshot) so the offline path is exercised.
        ingest.fetch_ercot_queue(use_cache_on_error=True)
        # NOTE: build_features is a day-of stub returning empty, so we still show
        # demo rows to keep the UI populated. Day-of, replace this with:
        #   return build_features(raw), "live"
        return _demo_queue(), "snapshot"
    except Exception:  # noqa: BLE001
        return _demo_queue(), "demo"


# --------------------------------------------------------------------------- #
# Panels
# --------------------------------------------------------------------------- #
def _county_map_stub(scored: pd.DataFrame) -> go.Figure:
    """Plotly Texas county map STUB.

    Day-of: swap for a real county-level choropleth keyed on Texas FIPS codes.
    Today it plots projects at jittered placeholder coordinates so the panel is
    live and wired to the scored frame.
    """
    fig = go.Figure(
        go.Scattergeo(
            lon=[-99 - i * 0.4 for i in range(len(scored))],
            lat=[31 + i * 0.3 for i in range(len(scored))],
            text=scored["queue_id"] + " · " + (scored["completion_probability"] * 100).round().astype(int).astype(str) + "%",
            marker=dict(
                size=(scored["capacity_mw"] / 10).clip(6, 40),
                color=scored["completion_probability"],
                colorscale="Viridis",
                colorbar=dict(title="P(complete)"),
                cmin=0,
                cmax=1,
            ),
            mode="markers",
        )
    )
    fig.update_geos(scope="usa", center=dict(lon=-99, lat=31), projection_scale=4)
    fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        height=360,
        title="Texas county map (STUB — placeholder coordinates)",
    )
    return fig


def main() -> None:
    st.set_page_config(page_title="QueueScore", layout="wide")
    st.title("QueueScore")
    st.caption("Completion probability for every project in the ERCOT interconnection queue.")

    # Panel 1: pull live queue + offline indicator
    col_btn, col_status = st.columns([1, 3])
    pull_live = col_btn.button("Pull live queue")
    features, source = _load_queue(pull_live)

    badge = {
        "live": "🟢 Live ERCOT data",
        "snapshot": "🟡 Cached snapshot (offline)",
        "demo": "⚪ Demo data (no pull yet)",
    }[source]
    col_status.markdown(f"**Data source:** {badge}")

    scored = _score_to_frame(features, DummyScorer().score(features))

    left, right = st.columns([3, 2])

    # Panel 2: leaderboard
    with left:
        st.subheader("Leaderboard")
        st.dataframe(
            scored.drop(columns=["_attributions"]),
            use_container_width=True,
            hide_index=True,
        )

        # Panel 3: county map
        st.subheader("Geography")
        st.plotly_chart(_county_map_stub(scored), use_container_width=True)

    # Panel 4: project detail + Q&A
    with right:
        st.subheader("Project detail")
        pick = st.selectbox("Project", scored["queue_id"].tolist())
        row = scored[scored["queue_id"] == pick].iloc[0]
        st.metric("Completion probability", f"{row['completion_probability']:.0%}")

        verdict = explain.explain_project(
            queue_id=row["queue_id"],
            generation_type=row["generation_type"],
            capacity_mw=row["capacity_mw"],
            county=row["county"],
            queue_year=int(row["queue_year"]),
            probability=float(row["completion_probability"]),
            attributions=row["_attributions"],
        )
        st.write("**Verdict**")
        st.info(verdict)

        st.write("**Ask a question about the queue**")
        question = st.text_input("e.g. Which solar projects are most likely to complete?")
        if question:
            st.write(explain.answer_question(question, scored.drop(columns=["_attributions"])))

    st.divider()
    st.caption("Training data: LBNL Queued Up (CC BY 4.0)")


if __name__ == "__main__":
    main()
