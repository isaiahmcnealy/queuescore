"""QueueSense — Streamlit app.

Live origination intelligence for Texas power projects. Runs against the REAL
unified record table (ERCOT interconnection queue + TCEQ power-generation air
permits) via ``src.sources.load_all``. Panels:

  1. Refresh + freshness   — per-source "last updated" stamps; refresh pulls live
                             with graceful snapshot fallback (offline-safe).
  2. Records table         — filterable/sortable view of every record.
  3. Geography             — overview map (labeled basemap) ⇄ satellite site view,
                             both centered on the selected record.
  4. Record detail         — the record's story + Claude origination read + Q&A.

Next day-of steps: cross-source matching (M3) links ERCOT⇄TCEQ records into
projects; stage inference (M4) turns stage_signal into funnel labels.
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

from src import explain, resolve  # noqa: E402
from src.sources import load_all  # noqa: E402


# --------------------------------------------------------------------------- #
# Design system — warm cream + sage/olive greens with a terracotta accent.
# Source: Warm Cream #F7F4EB · Sage #96A18C · Deep Olive #4A533C ·
#         Charcoal Forest #2E332A · Soft Terracotta #C88E72
# Mirrored in .streamlit/config.toml so the chrome matches.
# --------------------------------------------------------------------------- #
PALETTE = {
    "bg": "#F7F4EB",
    "surface": "#E7EBE1",
    "sage": "#96A18C",
    "olive": "#4A533C",
    "terracotta": "#C88E72",
    "ink": "#2E332A",
    "land": "#E4E8DD",
}
SOURCE_COLORS = {"ercot": PALETTE["olive"], "tceq": PALETTE["terracotta"]}

# Fixed height of the right-hand detail card; long content scrolls inside it
# instead of stretching the page (keeps the two columns visually aligned).
DETAIL_PANEL_HEIGHT = 900

_CSS = f"""
<style>
  .stApp {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Roboto, sans-serif; }}
  .block-container {{ padding-top: 2.5rem; max-width: 1320px; }}
  h1 {{ letter-spacing: -0.02em; font-weight: 700; }}
  h2, h3 {{ letter-spacing: -0.01em; color: {PALETTE['olive']}; }}
  .stButton > button {{
    border-radius: 999px; border: 1px solid {PALETTE['olive']};
    color: {PALETTE['olive']}; font-weight: 600; padding: 0.4rem 1.1rem;
  }}
  .stButton > button:hover {{ background: {PALETTE['olive']}; color: {PALETTE['bg']}; border-color: {PALETTE['olive']}; }}
  [data-testid="stMetric"] {{ background: {PALETTE['surface']}; padding: 1rem 1.25rem; border-radius: 14px; }}
  div[data-testid="stVerticalBlockBorderWrapper"] {{ border-radius: 16px; }}
  .verdict-box {{
    background: rgba(74, 83, 60, 0.08); border-left: 4px solid {PALETTE['olive']};
    border-radius: 12px; padding: 0.85rem 1.05rem; color: {PALETTE['ink']};
  }}
  .verdict-box p {{ margin: 0 0 0.6rem 0; line-height: 1.5; }}
  .verdict-box p:last-child {{ margin-bottom: 0; }}
  [data-testid="stToolbar"], #MainMenu, footer {{ visibility: hidden; }}
</style>
"""


def _inject_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def _render_verdict(text: str) -> None:
    """Render the Claude read as a thematic callout (escaped; paragraphs kept)."""
    paras = [html.escape(p.strip()) for p in text.split("\n\n") if p.strip()]
    body = "".join(f"<p>{p}</p>" for p in paras)
    st.markdown(f'<div class="verdict-box">{body}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
TEXAS_CENTER = dict(lat=31.0, lon=-99.3)


def _load_records(refresh: bool) -> tuple[pd.DataFrame, dict]:
    try:
        return load_all(refresh=refresh)
    except Exception as exc:  # noqa: BLE001 - first run offline with no snapshots
        st.error(
            "No data available: live pull failed and no snapshots exist yet. "
            f"Run once with network access to seed the cache. ({exc})"
        )
        st.stop()


def _label(row: pd.Series) -> str:
    name = row["project_name"] or row["company"] or "unnamed"
    return f"{row['source_id']} · {name[:38]} [{row['source']}]"


# --------------------------------------------------------------------------- #
# Geography panels
# --------------------------------------------------------------------------- #
def _overview_map(records: pd.DataFrame, focus: pd.Series | None) -> go.Figure:
    """All records with coordinates on a labeled street basemap (keyless Carto).

    Color = source (olive ERCOT / terracotta TCEQ). ERCOT rows carry no
    coordinates until matching attaches permit coords, so today the dots are
    mostly TCEQ — stated honestly in the caption.
    """
    plottable = records[records["lat"].notna() & records["lon"].notna()]
    fig = go.Figure()

    focused = focus is not None and pd.notna(focus.get("lat")) and pd.notna(focus.get("lon"))
    if focused:
        fig.add_trace(
            go.Scattermap(
                lon=[float(focus["lon"])], lat=[float(focus["lat"])],
                mode="markers",
                marker=dict(size=34, color=PALETTE["sage"], opacity=0.5),
                hoverinfo="skip",
            )
        )

    for source, group in plottable.groupby("source"):
        hover = (
            group["source_id"] + " · " + group["project_name"].str.slice(0, 40)
            + "<br>" + group["company"].str.slice(0, 40)
            + "<br>" + group["county"] + " County · " + group["status"]
        )
        fig.add_trace(
            go.Scattermap(
                lon=group["lon"], lat=group["lat"],
                mode="markers",
                name=source.upper(),
                text=hover,
                hoverinfo="text",
                marker=dict(size=10, color=SOURCE_COLORS[source], opacity=0.75),
            )
        )

    center = (
        dict(lat=float(focus["lat"]), lon=float(focus["lon"])) if focused else TEXAS_CENTER
    )
    fig.update_layout(
        map=dict(style="carto-positron", center=center, zoom=8.5 if focused else 4.9),
        autosize=True,
        margin=dict(l=0, r=0, t=0, b=0),
        height=380,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=PALETTE["ink"]),
    )
    return fig


def _site_view(row: pd.Series) -> None:
    """Interactive satellite view of the selected record's site (keyless embed)."""
    if pd.isna(row.get("lat")) or pd.isna(row.get("lon")):
        st.info(
            "No coordinates for this record — ERCOT publishes none. "
            "Cross-source matching (next step) attaches permit coordinates."
        )
        return
    lat, lon = float(row["lat"]), float(row["lon"])
    maps_url = f"https://www.google.com/maps/@{lat},{lon},900m/data=!3m1!1e3"
    embed = f"https://maps.google.com/maps?q={lat},{lon}&t=k&z=16&output=embed"
    components.html(
        f'<iframe src="{embed}" width="100%" height="340" frameborder="0" '
        f'style="border:0; border-radius:12px;" loading="lazy" '
        f'referrerpolicy="no-referrer-when-downgrade"></iframe>',
        height=348,
    )
    st.caption(
        f"Site view: **{row['source_id']}** · {row['county']} County · "
        f"[open in Google Maps]({maps_url})"
    )


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
def main() -> None:
    st.set_page_config(page_title="QueueSense", page_icon="📡", layout="wide")
    _inject_css()
    st.title("QueueSense")
    st.caption(
        "Live origination intelligence for Texas power projects — "
        "ERCOT interconnection queue + TCEQ air permits in one view."
    )

    # Panel 1: refresh + freshness
    col_btn, col_status = st.columns([1, 3])
    refresh = col_btn.button("Refresh data")
    records, fresh = _load_records(refresh=refresh)

    # Cross-source matching: link ERCOT queue entries to their TCEQ permits
    # (match table produced by `python -m src.resolve`; snapshot-cached).
    matches = resolve.load_matches()
    if matches is not None and len(matches):
        records = resolve.link_records(records, matches)
        n_links = len(matches)
    else:
        records = records.assign(match_id="", match_reason="")
        n_links = 0

    stamps = " · ".join(
        f"**{name.upper()}** {ts:%b %d %H:%M}" if ts else f"**{name.upper()}** never"
        for name, ts in fresh.items()
    )
    col_status.markdown(
        f"🟢 {len(records):,} records · 🔗 {n_links} cross-source links · "
        f"last updated: {stamps}"
    )

    # Filters (the on-thesis lens lives here)
    fcol1, fcol2, fcol3 = st.columns([2, 1, 1])
    search = fcol1.text_input(
        "Search", placeholder="Company, project, or county…", label_visibility="collapsed"
    )
    gas_focus = fcol2.toggle("Gas-to-power focus", value=False)
    source_pick = fcol3.multiselect(
        "Sources", ["ercot", "tceq"], default=["ercot", "tceq"], label_visibility="collapsed"
    )

    view = records[records["source"].isin(source_pick or ["ercot", "tceq"])]
    if gas_focus:
        view = view[view["kind"].str.contains("gas|fossil", case=False, na=False)]
    if search:
        needle = search.strip()
        mask = (
            view["company"].str.contains(needle, case=False, na=False)
            | view["project_name"].str.contains(needle, case=False, na=False)
            | view["county"].str.contains(needle, case=False, na=False)
        )
        view = view[mask]
    view = view.reset_index(drop=True)

    # Shared focus: table click and the searchable selectbox both write record_pick.
    labels = view.apply(_label, axis=1).tolist() if len(view) else []
    if labels and st.session_state.get("record_pick") not in labels:
        st.session_state.record_pick = labels[0]

    left, right = st.columns([3, 2])

    # Panel 2: records table (click a row to select)
    with left:
        with st.container(border=True):
            st.subheader("Records")
            st.caption(
                f"{len(view):,} shown — click a row to select it (updates the detail pane). "
                "Same project can appear in both sources until matching links them."
            )
            table = view[
                ["source", "source_id", "project_name", "company", "county",
                 "kind", "status", "capacity_mw", "record_date"]
            ]
            event = st.dataframe(
                table,
                width="stretch",
                hide_index=True,
                height=330,
                on_select="rerun",
                selection_mode="single-row",
                key="records_table",
                column_config={
                    "source": st.column_config.TextColumn("Src"),
                    "source_id": st.column_config.TextColumn("ID"),
                    "project_name": st.column_config.TextColumn("Project"),
                    "company": st.column_config.TextColumn("Company"),
                    "county": st.column_config.TextColumn("County"),
                    "kind": st.column_config.TextColumn("Type"),
                    "status": st.column_config.TextColumn("Status"),
                    "capacity_mw": st.column_config.NumberColumn("MW", format="%.0f"),
                    "record_date": st.column_config.DateColumn("Filed"),
                },
            )
            selected_rows = event.selection.rows if event and event.selection else []
            if selected_rows:
                idx = int(selected_rows[0])
                # Only apply when the clicked row *changes* — otherwise a sticky
                # table selection would overwrite the searchable selectbox on every rerun.
                if (
                    0 <= idx < len(labels)
                    and st.session_state.get("_last_table_sel") != idx
                ):
                    st.session_state._last_table_sel = idx
                    st.session_state.record_pick = labels[idx]

        pick = st.session_state.get("record_pick")
        focus = view.iloc[labels.index(pick)] if pick in labels else None

        with st.container(border=True):
            geo_head, geo_toggle = st.columns([1, 1])
            geo_head.subheader("Geography")
            mode = geo_toggle.segmented_control(
                "View", options=["Overview", "Site view"],
                default="Overview", label_visibility="collapsed",
            )
            if mode == "Site view" and focus is not None:
                _site_view(focus)
            else:
                n_plot = int((view["lat"].notna() & view["lon"].notna()).sum())
                st.caption(
                    f"{n_plot:,} records with coordinates (TCEQ permits carry them; "
                    "ERCOT rows plot after matching). Olive = ERCOT · terracotta = TCEQ."
                )
                st.plotly_chart(
                    _overview_map(view, focus), width="stretch", config={"scrollZoom": True}
                )

    # Panel 4: record detail
    with right:
        with st.container(height=DETAIL_PANEL_HEIGHT, border=True):
            st.subheader("Record detail")
            if not labels:
                st.info("No records match the current filters.")
            else:
                # No widget key on selectbox — index= tracks record_pick so table
                # clicks can move the dropdown. Type in the box to search/filter.
                prev_pick = st.session_state.record_pick
                idx = labels.index(prev_pick) if prev_pick in labels else 0
                choice = st.selectbox(
                    "Record",
                    labels,
                    index=idx,
                    help="Click a table row or type here to search records.",
                )
                st.session_state.record_pick = choice
                if choice != prev_pick:
                    # Selectbox drove the change; clear sticky table idx so the
                    # next row click is applied even if it's the previously highlighted row.
                    st.session_state._last_table_sel = None
                row = view.iloc[labels.index(choice)]
                st.metric("Status", row["status"] or "—")
                cap = f"{row['capacity_mw']:.0f} MW · " if pd.notna(row["capacity_mw"]) else ""
                filed = (
                    f"{pd.to_datetime(row['record_date']):%Y-%m-%d}"
                    if pd.notna(row["record_date"]) else "unknown date"
                )
                st.markdown(
                    f"**{row['project_name'] or '(unnamed project)'}**\n\n"
                    f"{row['company'] or 'unknown company'} · {row['county']} County\n\n"
                    f"{cap}{row['kind']} · filed {filed}\n\n"
                    f"Stage signal: `{row['stage_signal'] or 'n/a'}` · "
                    f"ref `{row['link_id'] or row['source_id']}`"
                )

                # The stitched story: show the matched record from the other source.
                if row.get("match_id"):
                    other = records[
                        (records["source"] != row["source"])
                        & (records["source_id"] == row["match_id"])
                    ]
                    st.write("**🔗 Linked record (cross-source match)**")
                    if len(other):
                        o = other.iloc[0]
                        o_filed = (
                            f"{pd.to_datetime(o['record_date']):%Y-%m-%d}"
                            if pd.notna(o["record_date"]) else "unknown date"
                        )
                        st.markdown(
                            f"`{o['source'].upper()}` **{o['source_id']}** · "
                            f"{o['project_name'] or o['company']}\n\n"
                            f"{o['company']} · {o['status']} · filed {o_filed}\n\n"
                            f"_Why linked:_ {row['match_reason'] or 'name match in same county'}"
                        )

                st.write("**Origination read**")
                _render_verdict(explain.explain_record(row.to_dict()))

                st.write("**Ask about this record**")
                question = st.text_input(
                    "Question",
                    placeholder="e.g. Is this far enough along for an EPC conversation?",
                    label_visibility="collapsed",
                )
                if question:
                    with st.spinner("Asking Claude…"):
                        st.write(explain.answer_record_question(question, row.to_dict()))

    st.divider()
    st.caption(
        "Sources: ERCOT interconnection queue (via gridstatus) · "
        "TCEQ Permit Search · training data: LBNL Queued Up (CC BY 4.0)"
    )


if __name__ == "__main__":
    main()
