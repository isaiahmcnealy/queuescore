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

import html
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

# Make `from src import ...` work when launched via `streamlit run src/app.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config, explain, ingest  # noqa: E402
from src.features import FEATURE_FRAME_COLUMNS  # noqa: E402
from src.scorer import DummyScorer, ScoreResult  # noqa: E402


# --------------------------------------------------------------------------- #
# Demo data (stand-in until features.build_features is implemented)
# --------------------------------------------------------------------------- #
def _demo_queue() -> pd.DataFrame:
    """A tiny model-ready frame so every panel renders before real data exists.

    ``lat``/``lon`` are placeholder Houston-area coordinates so the map and the
    satellite site view share one source of truth. Day-of, real coordinates come
    from TCEQ (``aiLatDecCoord``/``aiLongDecCoord``); ERCOT rows without a permit
    match fall back to county centroids.
    """
    rows = [
        ("ERC-1001", 150.0, "Solar", "Harris", 2022, 640, "large", 29.88, -95.65),
        ("ERC-1002", 20.0, "Battery", "Travis", 2023, 275, "small", 29.52, -95.46),
        ("ERC-1003", 300.0, "Wind", "Nolan", 2021, 1010, "large", 30.00, -95.28),
        ("ERC-1004", 75.0, "Solar", "Pecos", 2023, 190, "medium", 29.40, -95.10),
        ("ERC-1005", 600.0, "Gas", "Bexar", 2020, 1400, "xlarge", 30.12, -94.92),
        ("ERC-1006", 45.0, "Battery", "Webb", 2024, 90, "medium", 29.64, -95.82),
    ]
    return pd.DataFrame(rows, columns=[*FEATURE_FRAME_COLUMNS, "lat", "lon"])


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
# Design system — modern earthy (cream + sage mint). Single source of truth for
# Python-drawn visuals; the Streamlit chrome is themed in .streamlit/config.toml
# with the same hexes.
# --------------------------------------------------------------------------- #
# Brand palette — warm cream + sage/olive greens with a terracotta accent.
# Source: Warm Cream #F7F4EB · Sage #96A18C · Deep Olive #4A533C ·
#         Charcoal Forest #2E332A · Soft Terracotta #C88E72
PALETTE = {
    "bg": "#F7F4EB",         # warm cream — page canvas
    "surface": "#E7EBE1",    # pale sage — cards, table header, metric, inputs (tint of sage)
    "sage": "#96A18C",       # mid-tone green — map borders, ramp mid
    "olive": "#4A533C",      # deep olive — primary accent, headings, high probability
    "terracotta": "#C88E72", # warm accent — low end of the probability ramp
    "ink": "#2E332A",        # charcoal forest — text / headings
    "land": "#E4E8DD",       # map land fill (pale sage)
}
# Probability ramp: low = terracotta, mid = sage, high = deep olive.
PROB_COLORSCALE = [[0.0, PALETTE["terracotta"]], [0.5, PALETTE["sage"]], [1.0, PALETTE["olive"]]]

# Fixed height of the right-hand detail card, tuned to the left column
# (leaderboard + map). Keeping it fixed means a long Claude verdict scrolls
# inside the card instead of stretching the page and shoving the map down.
DETAIL_PANEL_HEIGHT = 856

# Targeted CSS for a modern look (pill buttons, rounded cards, tighter rhythm,
# hidden deploy chrome). NOTE: the data-testid / chrome selectors hook Streamlit
# internals and may need a tweak after a Streamlit upgrade — the config.toml
# theme above is the durable layer; this is polish on top.
_CSS = f"""
<style>
  .stApp {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Roboto, sans-serif; }}
  .block-container {{ padding-top: 2.5rem; max-width: 1320px; }}
  h1 {{ letter-spacing: -0.02em; font-weight: 700; }}
  h2, h3 {{ letter-spacing: -0.01em; color: {PALETTE['olive']}; }}
  /* pill buttons */
  .stButton > button {{
    border-radius: 999px; border: 1px solid {PALETTE['olive']};
    color: {PALETTE['olive']}; font-weight: 600; padding: 0.4rem 1.1rem;
  }}
  .stButton > button:hover {{ background: {PALETTE['olive']}; color: {PALETTE['bg']}; border-color: {PALETTE['olive']}; }}
  /* card containers + metric */
  [data-testid="stMetric"] {{ background: {PALETTE['surface']}; padding: 1rem 1.25rem; border-radius: 14px; }}
  div[data-testid="stVerticalBlockBorderWrapper"] {{ border-radius: 16px; }}
  /* thematic verdict callout (replaces the default blue st.info) */
  .verdict-box {{
    background: rgba(74, 83, 60, 0.08); border-left: 4px solid {PALETTE['olive']};
    border-radius: 12px; padding: 0.85rem 1.05rem; color: {PALETTE['ink']};
  }}
  .verdict-box p {{ margin: 0 0 0.6rem 0; line-height: 1.5; }}
  .verdict-box p:last-child {{ margin-bottom: 0; }}
  /* clean demo chrome */
  [data-testid="stToolbar"], #MainMenu, footer {{ visibility: hidden; }}
</style>
"""


def _inject_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def _render_verdict(text: str) -> None:
    """Render the Claude verdict as a thematic callout (escaped; paragraphs kept)."""
    paras = [html.escape(p.strip()) for p in text.split("\n\n") if p.strip()]
    body = "".join(f"<p>{p}</p>" for p in paras)
    st.markdown(f'<div class="verdict-box">{body}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Panels
# --------------------------------------------------------------------------- #
def _overview_map(scored: pd.DataFrame, focus: pd.Series) -> go.Figure:
    """Interactive overview map on a labeled street basemap (keyless Carto tiles).

    Real roads, towns, and water make it obvious where each project sits — no
    API key needed. Centered on the currently selected project (with a
    terracotta halo) so flipping to Site view keeps the same location.
    """
    hover = (
        scored["queue_id"] + " · " + scored["generation_type"]
        + "<br>" + scored["county"] + " County · "
        + scored["capacity_mw"].round(0).astype(int).astype(str) + " MW · "
        + (scored["completion_probability"] * 100).round().astype(int).astype(str) + "%"
    )
    fig = go.Figure()
    # Halo under the selected project so the eye lands on it immediately.
    fig.add_trace(
        go.Scattermap(
            lon=[float(focus["lon"])],
            lat=[float(focus["lat"])],
            mode="markers",
            marker=dict(size=36, color=PALETTE["terracotta"], opacity=0.45),
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scattermap(
            lon=scored["lon"],
            lat=scored["lat"],
            mode="markers",
            text=hover,
            hoverinfo="text",
            marker=dict(
                size=(scored["capacity_mw"] / 18).clip(10, 34),
                color=scored["completion_probability"],
                colorscale=PROB_COLORSCALE,
                cmin=0,
                cmax=1,
                colorbar=dict(title="P(complete)", outlinewidth=0),
            ),
        )
    )
    fig.update_layout(
        map=dict(
            style="carto-positron",  # light, labeled, palette-friendly; keyless
            center=dict(lat=float(focus["lat"]), lon=float(focus["lon"])),
            zoom=8.7,  # metro scale — same "relative location" the Site view zooms into
        ),
        autosize=True,
        margin=dict(l=0, r=0, t=0, b=0),
        height=360,
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=PALETTE["ink"]),
    )
    return fig


def _site_view(row: pd.Series) -> None:
    """Interactive satellite view of one project's site (keyless Google embed).

    Lets the user eyeball the actual land — empty field vs. active construction
    is an origination signal. Uses the classic keyless Google Maps embed
    (``t=k`` = satellite); needs network, so we degrade with a caption offline.
    Day-of upgrade path: Mapbox satellite basemap on the overview map itself,
    and/or Google Static/Embed API with a key for SLA. See ROADMAP.md.
    """
    lat, lon = float(row["lat"]), float(row["lon"])
    maps_url = f"https://www.google.com/maps/@{lat},{lon},900m/data=!3m1!1e3"
    embed = (
        f"https://maps.google.com/maps?q={lat},{lon}&t=k&z=16&output=embed"
    )
    components.html(
        f'<iframe src="{embed}" width="100%" height="340" frameborder="0" '
        f'style="border:0; border-radius:12px;" loading="lazy" '
        f'referrerpolicy="no-referrer-when-downgrade"></iframe>',
        height=348,
    )
    st.caption(
        f"Site view: **{row['queue_id']}** · {row['county']} County · "
        f"[open in Google Maps]({maps_url}) — coordinates are placeholders until "
        f"real TCEQ coords land (demo mode)."
    )


def main() -> None:
    st.set_page_config(page_title="QueueScore", page_icon="🌱", layout="wide")
    _inject_css()
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

    # Shared focus: the project picked in the detail pane (selectbox key
    # "project_pick"; session_state carries it across reruns). Both Geography
    # views center here, so Overview ⇄ Site view stay on the same location.
    pick = st.session_state.get("project_pick", scored["queue_id"].iloc[0])
    focus = scored[scored["queue_id"] == pick].iloc[0]

    left, right = st.columns([3, 2])

    # Panel 2: leaderboard
    with left:
        with st.container(border=True):
            st.subheader("Leaderboard")
            # Lead with the probability bar; keep the table narrow enough that the
            # % label isn't clipped (queue_age_days dropped from the display view).
            display = scored[
                ["queue_id", "completion_probability", "generation_type",
                 "capacity_mw", "county", "size_bucket", "queue_year"]
            ]
            st.dataframe(
                display,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "queue_id": st.column_config.TextColumn("Project"),
                    "completion_probability": st.column_config.ProgressColumn(
                        "P(complete)", format="percent", min_value=0.0, max_value=1.0,
                        width="medium",
                    ),
                    "generation_type": st.column_config.TextColumn("Type"),
                    "capacity_mw": st.column_config.NumberColumn("Capacity", format="%d MW"),
                    "county": st.column_config.TextColumn("County"),
                    "size_bucket": st.column_config.TextColumn("Size"),
                    "queue_year": st.column_config.NumberColumn("Queued", format="%d"),
                },
            )

        # Panel 3: geography — overview map ⇄ per-site satellite view
        with st.container(border=True):
            geo_head, geo_toggle = st.columns([1, 1])
            geo_head.subheader("Geography")
            mode = geo_toggle.segmented_control(
                "View",
                options=["Overview", "Site view"],
                default="Overview",
                label_visibility="collapsed",
            )
            if mode == "Site view":
                _site_view(focus)
            else:
                st.caption(
                    f"All projects · centered on **{focus['queue_id']}** (halo) · "
                    "dot size = capacity, color = P(complete) · demo coordinates"
                )
                st.plotly_chart(_overview_map(scored, focus), width="stretch", config={"scrollZoom": True})

    # Panel 4: project detail + Q&A — fixed height, scrolls internally so a long
    # verdict never pushes the map/leaderboard around.
    with right:
        with st.container(height=DETAIL_PANEL_HEIGHT, border=True):
            st.subheader("Project detail")
            pick = st.selectbox("Project", scored["queue_id"].tolist(), key="project_pick")
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
            _render_verdict(verdict)

            st.write("**Ask a question about the queue**")
            question = st.text_input("e.g. Which solar projects are most likely to complete?")
            if question:
                st.write(explain.answer_question(question, scored.drop(columns=["_attributions"])))

    st.divider()
    st.caption("Training data: LBNL Queued Up (CC BY 4.0)")


if __name__ == "__main__":
    main()
