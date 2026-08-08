"""Project Radar — Streamlit app.

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

from src import explain  # noqa: E402
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
# Map / legend colors by filing status (ERCOT + TCEQ vocabularies).
STATUS_COLORS: dict[str, str] = {
    "Active": PALETTE["olive"],
    "Completed": PALETTE["sage"],
    "ISSUED PERMIT": PALETTE["terracotta"],
    "NEW APPLICATION": "#2E6B6B",
    "RENEWAL/AMENDMENT": "#8B6914",
}
_STATUS_FALLBACK = PALETTE["ink"]

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
def _focus_point(
    focus: pd.Series | None, records: pd.DataFrame
) -> tuple[float, float, str] | None:
    """Lat/lon to center the map on for the selected record.

    Prefer the record's own coordinates; if missing (typical ERCOT), fall back to
    the mean of other filings in the same county that do have coords.
    """
    if focus is None:
        return None
    if pd.notna(focus.get("lat")) and pd.notna(focus.get("lon")):
        return float(focus["lat"]), float(focus["lon"]), "site"
    county = str(focus.get("county") or "").strip()
    if not county:
        return None
    peers = records[
        (records["county"].str.casefold() == county.casefold())
        & records["lat"].notna()
        & records["lon"].notna()
    ]
    if peers.empty:
        return None
    return (
        float(peers["lat"].mean()),
        float(peers["lon"].mean()),
        f"{county} County (approx)",
    )


def _status_color(status: str) -> str:
    return STATUS_COLORS.get(status, _STATUS_FALLBACK)


def _overview_map(
    records: pd.DataFrame,
    focus: pd.Series | None,
    focus_pt: tuple[float, float, str] | None,
) -> go.Figure:
    """Records with coordinates on a labeled street basemap (keyless Carto).

    Color = status. Selected row zooms the camera (exact coords, or county
    average when the row itself has none — common for ERCOT).
    """
    plottable = records[records["lat"].notna() & records["lon"].notna()].copy()
    fig = go.Figure()

    if focus_pt is not None:
        flat, flon, kind = focus_pt
        fig.add_trace(
            go.Scattermap(
                lon=[flon], lat=[flat],
                mode="markers",
                marker=dict(size=36, color=PALETTE["sage"], opacity=0.45),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scattermap(
                lon=[flon], lat=[flat],
                mode="markers",
                name="Selected",
                text=[
                    f"Selected · {focus.get('source_id', '') if focus is not None else ''} · "
                    f"{focus.get('status', '') if focus is not None else ''} · {kind}"
                ],
                hoverinfo="text",
                marker=dict(size=16, color=PALETTE["olive"], opacity=1.0),
            )
        )

    known = [s for s in STATUS_COLORS if s in set(plottable["status"].dropna())]
    other = sorted(set(plottable["status"].dropna()) - set(known))
    for status in known + other:
        group = plottable[plottable["status"] == status]
        if group.empty:
            continue
        hover = (
            group["source_id"] + " · " + group["project_name"].str.slice(0, 40)
            + "<br>" + group["company"].str.slice(0, 40)
            + "<br>" + group["county"] + " County · " + group["status"]
            + "<br>" + group["source"].str.upper()
        )
        fig.add_trace(
            go.Scattermap(
                lon=group["lon"], lat=group["lat"],
                mode="markers",
                name=str(status),
                text=hover,
                hoverinfo="text",
                marker=dict(size=10, color=_status_color(str(status)), opacity=0.8),
            )
        )

    if focus_pt is not None:
        flat, flon, kind = focus_pt
        center = dict(lat=flat, lon=flon)
        zoom = 10 if kind == "site" else 8.5
        focus_key = (
            f"{focus.get('source')}-{focus.get('source_id')}-{kind}"
            if focus is not None
            else f"pt-{kind}"
        )
    else:
        center = TEXAS_CENTER
        zoom = 4.9
        focus_key = "statewide"

    fig.update_layout(
        map=dict(style="carto-positron", center=center, zoom=zoom),
        autosize=True,
        margin=dict(l=0, r=0, t=0, b=0),
        height=380,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=PALETTE["ink"]),
        uirevision=focus_key,
    )
    return fig


def _site_view(
    row: pd.Series,
    focus_pt: tuple[float, float, str] | None = None,
) -> None:
    """Interactive satellite view of the selected record's site (keyless embed).

    Uses exact coords when present; otherwise the same county approximation as
    the overview map so ERCOT rows still show a useful satellite frame.
    """
    if focus_pt is not None:
        lat, lon, kind = focus_pt
    elif pd.notna(row.get("lat")) and pd.notna(row.get("lon")):
        lat, lon, kind = float(row["lat"]), float(row["lon"]), "site"
    else:
        st.info(
            "No coordinates for this record — ERCOT publishes none, and no "
            "same-county peers are available to approximate a location."
        )
        return

    zoom = 16 if kind == "site" else 11
    maps_url = f"https://www.google.com/maps/@{lat},{lon},900m/data=!3m1!1e3"
    embed = (
        f"https://maps.google.com/maps?q={lat},{lon}&t=k&z={zoom}&output=embed"
    )
    components.html(
        f'<iframe src="{embed}" width="100%" height="340" frameborder="0" '
        f'style="border:0; border-radius:12px;" loading="lazy" '
        f'referrerpolicy="no-referrer-when-downgrade"></iframe>',
        height=348,
    )
    if kind == "site":
        st.caption(
            f"Site view: **{row['source_id']}** · {row['county']} County · "
            f"[open in Google Maps]({maps_url})"
        )
    else:
        st.caption(
            f"Approximate view for **{row['source_id']}** — centered on **{kind}** "
            f"(no site coords on this filing). "
            f"[open in Google Maps]({maps_url})"
        )


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
def main() -> None:
    st.set_page_config(page_title="QueueScore", page_icon="📡", layout="wide")
    _inject_css()
    st.title("QueueScore")
    st.caption(
        "Live origination intelligence for Texas power projects — "
        "ERCOT interconnection queue + TCEQ air permits in one view."
    )

    # Panel 1: refresh + freshness
    col_btn, col_status = st.columns([1, 3])
    refresh = col_btn.button("Refresh data")
    records, fresh = _load_records(refresh=refresh)
    stamps = " · ".join(
        f"**{name.upper()}** {ts:%b %d %H:%M}" if ts else f"**{name.upper()}** never"
        for name, ts in fresh.items()
    )
    col_status.markdown(f"🟢 {len(records):,} records · last updated: {stamps}")

    # Filters — status multiselect: empty = all (chips with X only when narrowed).
    statuses_present = set(records["status"].dropna().astype(str))
    status_options = [s for s in STATUS_COLORS if s in statuses_present] + sorted(
        statuses_present - set(STATUS_COLORS)
    )
    fcol1, fcol2, fcol3, fcol4 = st.columns([2, 1.4, 1, 1])
    search = fcol1.text_input(
        "Search", placeholder="Company, project, or county…", label_visibility="collapsed"
    )
    status_pick = fcol2.multiselect(
        "Status",
        status_options,
        default=[],
        placeholder="All statuses",
        help="Leave empty for all. Add statuses to narrow; X removes a chip.",
        label_visibility="collapsed",
    )
    gas_focus = fcol3.toggle("Gas-to-power focus", value=False)
    source_pick = fcol4.multiselect(
        "Sources", ["ercot", "tceq"], default=["ercot", "tceq"], label_visibility="collapsed"
    )

    view = records[records["source"].isin(source_pick or ["ercot", "tceq"])]
    if status_pick:
        view = view[view["status"].isin(status_pick)]
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
                f"{len(view):,} shown — click a row to select it and zoom the map. "
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
        # Use full `records` so ERCOT rows can borrow a county centroid from TCEQ peers.
        focus_pt = _focus_point(focus, records)

        with st.container(border=True):
            geo_head, geo_toggle = st.columns([1, 1])
            geo_head.subheader("Geography")
            mode = geo_toggle.segmented_control(
                "View", options=["Overview", "Site view"],
                default="Overview", label_visibility="collapsed",
            )
            if mode == "Site view" and focus is not None:
                _site_view(focus, focus_pt)
            else:
                n_plot = int((view["lat"].notna() & view["lon"].notna()).sum())
                if focus is not None and focus_pt is None:
                    st.caption(
                        f"Selected **{focus.get('source_id')}** has no map location "
                        f"(no coords / no peers in {focus.get('county') or 'unknown'} County)."
                    )
                elif focus_pt is not None and focus_pt[2] != "site":
                    st.caption(
                        f"Zoomed to **{focus_pt[2]}** — this filing has no site coords "
                        "(typical for ERCOT). Exact pin when matched to a permit."
                    )
                else:
                    st.caption(
                        f"{n_plot:,} with coordinates · color = status · "
                        "click a table row to zoom to that site."
                    )
                focus_key = (
                    f"{focus['source']}-{focus['source_id']}"
                    if focus is not None else "none"
                )
                st.plotly_chart(
                    _overview_map(view, focus, focus_pt),
                    width="stretch",
                    config={"scrollZoom": True},
                    key=f"overview_map_{focus_key}",
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
                record_key = f"{row['source']}:{row['source_id']}"
                # Drop cached LLM answers when the selected record changes.
                if st.session_state.get("_llm_record_key") != record_key:
                    st.session_state._llm_record_key = record_key
                    st.session_state.pop("_origination_read", None)
                    st.session_state.pop("_record_answer", None)

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

                st.write("**Origination read**")
                if st.button("Generate read", key="btn_origination_read"):
                    with st.spinner("Asking Claude…"):
                        st.session_state._origination_read = explain.explain_record(
                            row.to_dict()
                        )
                if st.session_state.get("_origination_read"):
                    _render_verdict(st.session_state._origination_read)

                st.write("**Ask about this record**")
                with st.form("record_qa_form", clear_on_submit=False):
                    question = st.text_input(
                        "Question",
                        placeholder="e.g. Is this far enough along for an EPC conversation?",
                        label_visibility="collapsed",
                    )
                    asked = st.form_submit_button("Ask")
                if asked:
                    if not question.strip():
                        st.warning("Enter a question first.")
                    else:
                        with st.spinner("Asking Claude…"):
                            st.session_state._record_answer = explain.answer_record_question(
                                question.strip(), row.to_dict()
                            )
                if st.session_state.get("_record_answer"):
                    st.write(st.session_state._record_answer)

    st.divider()
    st.caption(
        "Sources: ERCOT interconnection queue (via gridstatus) · "
        "TCEQ Permit Search · training data: LBNL Queued Up (CC BY 4.0)"
    )


if __name__ == "__main__":
    main()
