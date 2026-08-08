"""QueueScore — Streamlit app.

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

from src import explain, resolve, stage  # noqa: E402
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
DETAIL_PANEL_HEIGHT = 860
MAP_HEIGHT = 480
MAP_HEIGHT_EXPANDED = 560  # when filings drawer is collapsed

_CSS = f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,550;9..144,700&family=Source+Sans+3:wght@400;600;700&display=swap');

  .stApp {{
    font-family: "Source Sans 3", "Segoe UI", sans-serif;
    background:
      radial-gradient(1200px 500px at 10% -10%, rgba(150, 161, 140, 0.22), transparent 55%),
      radial-gradient(900px 420px at 100% 0%, rgba(200, 142, 114, 0.14), transparent 50%),
      {PALETTE['bg']};
  }}
  .block-container {{ padding-top: 1.2rem; padding-bottom: 1.5rem; max-width: 1600px; }}
  h1, .qs-brand {{
    font-family: Fraunces, Georgia, serif !important;
    letter-spacing: -0.03em; font-weight: 700; color: {PALETTE['ink']};
  }}
  h2, h3 {{
    font-family: Fraunces, Georgia, serif !important;
    letter-spacing: -0.02em; color: {PALETTE['olive']};
  }}
  .stButton > button {{
    border-radius: 999px; border: 1px solid {PALETTE['olive']};
    color: {PALETTE['olive']}; font-weight: 600; padding: 0.4rem 1.1rem;
    transition: background 160ms ease, color 160ms ease, transform 160ms ease;
  }}
  .stButton > button:hover {{
    background: {PALETTE['olive']}; color: {PALETTE['bg']}; border-color: {PALETTE['olive']};
    transform: translateY(-1px);
  }}
  [data-testid="stMetric"] {{
    background: {PALETTE['surface']}; padding: 0.75rem 1rem; border-radius: 14px;
    animation: qs-fade-up 420ms ease both;
  }}
  div[data-testid="stVerticalBlockBorderWrapper"] {{
    border-radius: 18px; border-color: rgba(46, 51, 42, 0.08) !important;
    background: rgba(247, 244, 235, 0.72);
    backdrop-filter: blur(6px);
    animation: qs-fade-up 480ms ease both;
  }}
  .verdict-box {{
    background: rgba(74, 83, 60, 0.08); border-left: 4px solid {PALETTE['olive']};
    border-radius: 12px; padding: 0.85rem 1.05rem; color: {PALETTE['ink']};
  }}
  .verdict-box p {{ margin: 0 0 0.6rem 0; line-height: 1.5; }}
  .verdict-box p:last-child {{ margin-bottom: 0; }}

  .qs-hero {{
    display: flex; flex-wrap: wrap; align-items: flex-end; justify-content: space-between;
    gap: 1rem 1.5rem; margin: 0.2rem 0 0.85rem 0;
    animation: qs-fade-up 380ms ease both;
  }}
  .qs-brand {{ font-size: 2.2rem; line-height: 1.05; margin: 0; }}
  .qs-tagline {{
    margin: 0.35rem 0 0 0; max-width: 34rem; color: rgba(46, 51, 42, 0.72);
    font-size: 1.0rem; line-height: 1.35;
  }}
  .qs-pills {{ display: flex; flex-wrap: wrap; gap: 0.45rem; align-items: center; }}
  .qs-pill {{
    display: inline-flex; align-items: center; gap: 0.4rem;
    padding: 0.38rem 0.75rem; border-radius: 999px;
    background: rgba(231, 235, 225, 0.95); color: {PALETTE['ink']};
    font-size: 0.86rem; font-weight: 600; border: 1px solid rgba(74, 83, 60, 0.12);
  }}
  .qs-pill strong {{ font-variant-numeric: tabular-nums; }}
  .qs-pill-live {{
    background: rgba(74, 83, 60, 0.12); border-color: rgba(74, 83, 60, 0.22);
  }}
  .qs-dot {{
    width: 0.55rem; height: 0.55rem; border-radius: 50%;
    background: {PALETTE['olive']};
    box-shadow: 0 0 0 0 rgba(74, 83, 60, 0.55);
    animation: qs-pulse 1.8s ease-out infinite;
  }}
  .qs-pill-link {{
    background: rgba(200, 142, 114, 0.16); border-color: rgba(200, 142, 114, 0.35);
  }}
  .qs-link-card {{
    margin: 0.35rem 0; padding: 0.85rem 0.95rem;
    border-radius: 14px; border: 1px solid rgba(200, 142, 114, 0.35);
    background: linear-gradient(135deg, rgba(200, 142, 114, 0.14), rgba(150, 161, 140, 0.1));
    animation: qs-fade-up 360ms ease both;
  }}
  .qs-link-card .qs-kicker {{
    font-size: 0.72rem; letter-spacing: 0.08em; text-transform: uppercase;
    font-weight: 700; color: {PALETTE['terracotta']}; margin: 0 0 0.35rem 0;
  }}

  /* Story rail — chat-like beats on the right */
  .qs-rail-label {{
    font-size: 0.72rem; letter-spacing: 0.1em; text-transform: uppercase;
    font-weight: 700; color: rgba(46, 51, 42, 0.45); margin: 0 0 0.65rem 0;
  }}
  .qs-msg {{
    margin: 0 0 0.65rem 0; padding: 0.8rem 0.95rem; border-radius: 16px 16px 16px 6px;
    background: rgba(231, 235, 225, 0.9); color: {PALETTE['ink']};
    animation: qs-fade-up 320ms ease both;
    line-height: 1.45;
  }}
  .qs-msg p {{ margin: 0 0 0.35rem 0; }}
  .qs-msg p:last-child {{ margin-bottom: 0; }}
  .qs-msg-stage {{
    background: rgba(74, 83, 60, 0.12); border: 1px solid rgba(74, 83, 60, 0.18);
  }}
  .qs-msg-meta {{ font-size: 0.88rem; opacity: 0.78; }}
  .qs-drawer-bar {{
    display: flex; align-items: center; justify-content: space-between;
    gap: 0.75rem; margin: 0.35rem 0 0.15rem 0;
    padding-top: 0.55rem; border-top: 1px solid rgba(46, 51, 42, 0.08);
  }}
  .qs-drawer-bar .qs-drawer-title {{
    font-size: 0.8rem; font-weight: 700; letter-spacing: 0.04em;
    text-transform: uppercase; color: rgba(46, 51, 42, 0.55); margin: 0;
  }}

  [data-testid="stPopover"] {{ display: flex; justify-content: flex-end; }}
  [data-testid="stPopover"] > button {{ padding: 0.15rem 0.55rem; }}
  [data-testid="stToolbar"], #MainMenu, footer,
  header[data-testid="stHeader"] {{ visibility: hidden; height: 0; }}
  div[data-testid="stDecoration"] {{ display: none; }}

  @keyframes qs-pulse {{
    0% {{ box-shadow: 0 0 0 0 rgba(74, 83, 60, 0.45); }}
    70% {{ box-shadow: 0 0 0 10px rgba(74, 83, 60, 0); }}
    100% {{ box-shadow: 0 0 0 0 rgba(74, 83, 60, 0); }}
  }}
  @keyframes qs-fade-up {{
    from {{ opacity: 0; transform: translateY(8px); }}
    to {{ opacity: 1; transform: translateY(0); }}
  }}
</style>
"""


def _inject_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def _hero(n_records: int, n_links: int, stamps: str) -> None:
    """Brand + live pulse strip — one composition above the map."""
    st.markdown(
        f"""
        <div class="qs-hero">
          <div>
            <p class="qs-brand">QueueScore</p>
            <p class="qs-tagline">
              Live Texas power origination — ERCOT queue + TCEQ permits,
              stitched when we can prove it's the same project.
            </p>
          </div>
          <div class="qs-pills">
            <span class="qs-pill qs-pill-live"><span class="qs-dot"></span> Live</span>
            <span class="qs-pill"><strong>{n_records:,}</strong> filings</span>
            <span class="qs-pill qs-pill-link"><strong>{n_links}</strong> stitched</span>
            <span class="qs-pill">{stamps}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _schema_help() -> None:
    """ℹ️ popover: what a row is, where fields come from, and the stage ladder."""
    with st.popover("ℹ️", help="About this data", use_container_width=False):
        st.markdown(
            """
**One row = one public filing, not one project.** The same project can appear
once per source until matching links them (🔗).

**Sources**

| Src | What the filing is |
|---|---|
| `ercot` | ERCOT interconnection queue entry — a request to connect a plant to the grid |
| `tceq` | TCEQ air permit (Air New Source Review) — permission to emit; gas plants need one |

**Columns**

| Column | Meaning |
|---|---|
| ID | The filing's number in its source (ERCOT queue ID / TCEQ permit #) |
| Project | Project or site name as filed |
| Company | Who filed it (ERCOT "Interconnecting Entity" / TCEQ permit holder) |
| County | Texas county — also the key used for cross-source matching |
| Type | Generation type (ERCOT) or industry description (TCEQ) |
| Stage | **Computed by QueueScore** — funnel position inferred from the signals below |
| Status | The source's own status word (Active/Completed · NEW APPLICATION/ISSUED PERMIT) |
| MW | Plant size — ERCOT only; permits don't state capacity |
| Filed | Date it entered the queue / permit date |

**Computed, not in the sources**

- **Stage + confidence + evidence** — read from the ERCOT study phase, the
  grid-agreement (IA) date, and the permit status. Ladder:
  *Early planning → Engineering studies → Studies complete → Grid agreement
  signed* (ERCOT) · *Permit application filed → Permit issued* (TCEQ).
- **🔗 Cross-source links** — same project found in both sources (name
  similarity + Claude adjudication); every link stores its reason.
- **ERCOT coordinates** — inherited from the matched permit (ERCOT publishes none).
            """
        )


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
    height: int = MAP_HEIGHT,
) -> go.Figure:
    """Statewide map of filings with coordinates (keyless Carto).

    Color = status. Selected filing gets a halo when it has a focus point.
    Camera stays Texas-wide with a mild pitch/bearing by default;
    ``uirevision`` keeps the user's pan/zoom across Streamlit reruns. Point
    ``[source, source_id]`` so map clicks can change the selection.
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
                customdata=[["", ""]],
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
                customdata=[["", ""]],
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
            + "<br><i>click to select</i>"
        )
        custom = group[["source", "source_id"]].astype(str).to_numpy().tolist()
        fig.add_trace(
            go.Scattermap(
                lon=group["lon"], lat=group["lat"],
                mode="markers",
                name=str(status),
                text=hover,
                hoverinfo="text",
                customdata=custom,
                marker=dict(size=10, color=_status_color(str(status)), opacity=0.8),
            )
        )

    # Slightly angled (not full isometric); uirevision keeps user pans.
    fig.update_layout(
        map=dict(
            style="carto-positron",
            center=TEXAS_CENTER,
            zoom=5.0,
            pitch=28,
            bearing=0,
        ),
        autosize=True,
        margin=dict(l=0, r=0, t=0, b=0),
        height=height,
        showlegend=True,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, x=0,
            bgcolor="rgba(247,244,235,0.85)",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=PALETTE["ink"], family="Source Sans 3"),
        uirevision="texas-angled-v3",
        clickmode="event+select",
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

    # Panel 1: refresh + load
    refresh = st.button("Refresh data", type="secondary")
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

    # Stage inference (M4): funnel position + confidence + evidence per record.
    records = stage.annotate_stages(records)

    stamps = " · ".join(
        f"{name.upper()} {ts:%b %d %H:%M}" if ts else f"{name.upper()} never"
        for name, ts in fresh.items()
    )
    _hero(len(records), n_links, html.escape(stamps))

    # Filters — status multiselect: empty = all; stitched lens = matched only.
    statuses_present = set(records["status"].dropna().astype(str))
    status_options = [s for s in STATUS_COLORS if s in statuses_present] + sorted(
        statuses_present - set(STATUS_COLORS)
    )
    fcol1, fcol2, fcol3, fcol4, fcol5 = st.columns([2.2, 1.5, 1.1, 1.0, 1.1])
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
    gas_focus = fcol3.toggle("Gas-to-power", value=False)
    stitched_only = fcol4.toggle(
        "Stitched only",
        value=False,
        help="Show only filings linked across ERCOT ↔ TCEQ.",
        disabled=n_links == 0,
    )
    source_pick = fcol5.multiselect(
        "Sources", ["ercot", "tceq"], default=["ercot", "tceq"], label_visibility="collapsed"
    )

    view = records[records["source"].isin(source_pick or ["ercot", "tceq"])]
    if status_pick:
        view = view[view["status"].isin(status_pick)]
    if gas_focus:
        view = view[view["kind"].str.contains("gas|fossil", case=False, na=False)]
    if stitched_only:
        view = view[view["match_id"].astype(str).str.len() > 0]
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

    pick = st.session_state.get("record_pick")
    focus = view.iloc[labels.index(pick)] if pick in labels else None
    focus_pt = _focus_point(focus, records)

    explore, story = st.columns([7, 3], gap="medium")

    # Left: map + minimizable filings drawer in one surface
    with explore:
        with st.container(border=True):
            geo_head, geo_toggle = st.columns([3, 2], vertical_alignment="center")
            geo_head.markdown("### Explore")
            mode = geo_toggle.segmented_control(
                "View", options=["Overview", "Site view"],
                default="Overview", label_visibility="collapsed",
            )

            show_filings = st.session_state.get("show_filings", True)
            map_h = MAP_HEIGHT if show_filings else MAP_HEIGHT_EXPANDED

            if mode == "Site view" and focus is not None:
                _site_view(focus, focus_pt)
            else:
                n_plot = int((view["lat"].notna() & view["lon"].notna()).sum())
                st.caption(
                    f"{n_plot:,} pins · **click a pin** · scroll to zoom"
                    + (f" · **{n_links} stitched**" if n_links else "")
                )
                map_event = st.plotly_chart(
                    _overview_map(view, focus, focus_pt, height=map_h),
                    width="stretch",
                    config={"scrollZoom": True, "displayModeBar": False},
                    key=f"overview_map_{st.session_state.get('_map_epoch', 0)}",
                    on_select="rerun",
                    selection_mode="points",
                )
                points = []
                if map_event and getattr(map_event, "selection", None):
                    sel = map_event.selection
                    points = (
                        sel.get("points") if hasattr(sel, "get")
                        else getattr(sel, "points", [])
                    ) or []
                if points:
                    cd = points[0].get("customdata")
                    if isinstance(cd, (list, tuple)) and len(cd) >= 2 and cd[0] and cd[1]:
                        src, sid = str(cd[0]), str(cd[1])
                        map_sel = f"{src}:{sid}"
                        if st.session_state.get("_last_map_sel") != map_sel:
                            hit = view[
                                (view["source"].astype(str) == src)
                                & (view["source_id"].astype(str) == sid)
                            ]
                            if len(hit):
                                st.session_state._last_map_sel = map_sel
                                st.session_state.record_pick = _label(hit.iloc[0])
                                st.session_state._last_table_sel = None
                                st.rerun()

            # Filings drawer — same container as the map, collapsible
            d_left, d_right = st.columns([4, 1], vertical_alignment="center")
            d_left.markdown(
                f'<div class="qs-drawer-bar"><p class="qs-drawer-title">'
                f'Filings · {len(view):,}</p></div>',
                unsafe_allow_html=True,
            )
            show_filings = d_right.toggle(
                "List",
                value=st.session_state.get("show_filings", True),
                key="show_filings",
                help="Show or hide the filings table under the map.",
            )
            if show_filings:
                t_head, t_info = st.columns([9, 1], vertical_alignment="center")
                t_head.caption(
                    "Click a row to select · same project can appear twice until stitched"
                )
                with t_info:
                    _schema_help()
                table = view[
                    ["source", "source_id", "project_name", "company", "county",
                     "kind", "stage", "status", "capacity_mw", "record_date"]
                ]
                event = st.dataframe(
                    table,
                    width="stretch",
                    hide_index=True,
                    height=220,
                    on_select="rerun",
                    selection_mode="single-row",
                    key="records_table",
                    column_config={
                        "source": st.column_config.TextColumn("Src", width="small"),
                        "source_id": st.column_config.TextColumn("ID", width="small"),
                        "project_name": st.column_config.TextColumn("Project"),
                        "company": st.column_config.TextColumn("Company"),
                        "county": st.column_config.TextColumn("County", width="small"),
                        "kind": st.column_config.TextColumn("Type", width="small"),
                        "stage": st.column_config.TextColumn("Stage"),
                        "status": st.column_config.TextColumn("Status", width="small"),
                        "capacity_mw": st.column_config.NumberColumn(
                            "MW", format="%.0f", width="small"
                        ),
                        "record_date": st.column_config.DateColumn("Filed", width="small"),
                    },
                )
                selected_rows = event.selection.rows if event and event.selection else []
                if selected_rows:
                    tidx = int(selected_rows[0])
                    if (
                        0 <= tidx < len(labels)
                        and st.session_state.get("_last_table_sel") != tidx
                    ):
                        st.session_state._last_table_sel = tidx
                        st.session_state.record_pick = labels[tidx]
                        st.session_state._last_map_sel = None
                        st.session_state._map_epoch = (
                            st.session_state.get("_map_epoch", 0) + 1
                        )

    # Right: story rail (chat-like beats)
    with story:
        with st.container(height=DETAIL_PANEL_HEIGHT, border=True):
            st.markdown(
                '<p class="qs-rail-label">Project story</p>',
                unsafe_allow_html=True,
            )
            if not labels:
                st.info("No records match the current filters.")
            else:
                prev_pick = st.session_state.record_pick
                sidx = labels.index(prev_pick) if prev_pick in labels else 0
                choice = st.selectbox(
                    "Record",
                    labels,
                    index=sidx,
                    label_visibility="collapsed",
                    help="Type to search, or select from the map / filings list.",
                )
                st.session_state.record_pick = choice
                if choice != prev_pick:
                    st.session_state._last_table_sel = None
                    st.session_state._last_map_sel = None
                    st.session_state._map_epoch = st.session_state.get("_map_epoch", 0) + 1
                row = view.iloc[labels.index(choice)]
                record_key = f"{row['source']}:{row['source_id']}"
                if st.session_state.get("_llm_record_key") != record_key:
                    st.session_state._llm_record_key = record_key
                    st.session_state.pop("_origination_read", None)
                    st.session_state.pop("_record_answer", None)

                conf_icon = {"high": "🟢", "medium": "🟡", "low": "⚪"}.get(
                    row["stage_confidence"], "⚪"
                )
                stage_name = html.escape(str(row["stage"] or "—"))
                stage_ev = html.escape(str(row["stage_evidence"] or ""))
                st.markdown(
                    f"""
                    <div class="qs-msg qs-msg-stage">
                      <p style="font-weight:700;font-size:1.05rem;margin:0 0 0.25rem 0;">
                        {stage_name}
                      </p>
                      <p class="qs-msg-meta">{conf_icon}
                        {html.escape(str(row['stage_confidence']))} confidence
                        — {stage_ev}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                cap = (
                    f"{row['capacity_mw']:.0f} MW · "
                    if pd.notna(row["capacity_mw"]) else ""
                )
                filed = (
                    f"{pd.to_datetime(row['record_date']):%Y-%m-%d}"
                    if pd.notna(row["record_date"]) else "unknown date"
                )
                st.markdown(
                    f"""
                    <div class="qs-msg">
                      <p style="font-weight:700;margin:0 0 0.35rem 0;">
                        {html.escape(str(row['project_name'] or '(unnamed project)'))}
                      </p>
                      <p class="qs-msg-meta">
                        {html.escape(str(row['company'] or 'unknown company'))}
                        · {html.escape(str(row['county'] or ''))} County
                      </p>
                      <p class="qs-msg-meta">
                        {html.escape(cap)}{html.escape(str(row['kind'] or ''))}
                        · {html.escape(str(row['status'] or 'status unknown'))}
                        · filed {filed}
                      </p>
                      <p class="qs-msg-meta" style="margin-top:0.35rem;">
                        `{html.escape(str(row['source']).upper())}`
                        {html.escape(str(row['source_id']))}
                        · signal `{html.escape(str(row['stage_signal'] or 'n/a'))}`
                      </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if row.get("match_id"):
                    other = records[
                        (records["source"] != row["source"])
                        & (records["source_id"] == row["match_id"])
                    ]
                    if len(other):
                        o = other.iloc[0]
                        o_filed = (
                            f"{pd.to_datetime(o['record_date']):%Y-%m-%d}"
                            if pd.notna(o["record_date"]) else "unknown date"
                        )
                        reason = html.escape(
                            str(row["match_reason"] or "name match in same county")
                        )
                        st.markdown(
                            f"""
                            <div class="qs-link-card">
                              <p class="qs-kicker">Stitched across sources</p>
                              <p style="margin:0 0 0.35rem 0;font-weight:700;">
                                {html.escape(str(o['source']).upper())}
                                {html.escape(str(o['source_id']))}
                                · {html.escape(str(o['project_name'] or o['company'] or ''))}
                              </p>
                              <p style="margin:0;opacity:0.85;">
                                {html.escape(str(o['company'] or ''))} ·
                                {html.escape(str(o['status'] or ''))} · filed {o_filed}
                              </p>
                              <p style="margin:0.45rem 0 0 0;font-size:0.9rem;opacity:0.75;">
                                Why linked: {reason}
                              </p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                st.markdown(
                    '<p class="qs-rail-label">Ask</p>',
                    unsafe_allow_html=True,
                )
                if st.button("Generate origination read", key="btn_origination_read"):
                    with st.spinner("Asking Claude…"):
                        st.session_state._origination_read = explain.explain_record(
                            row.to_dict()
                        )
                if st.session_state.get("_origination_read"):
                    _render_verdict(st.session_state._origination_read)

                with st.form("record_qa_form", clear_on_submit=False):
                    question = st.text_input(
                        "Question",
                        placeholder="Ask about this project…",
                        label_visibility="collapsed",
                    )
                    asked = st.form_submit_button("Send")
                if asked:
                    if not question.strip():
                        st.warning("Enter a question first.")
                    else:
                        with st.spinner("Asking Claude…"):
                            st.session_state._record_answer = explain.answer_record_question(
                                question.strip(), row.to_dict()
                            )
                if st.session_state.get("_record_answer"):
                    st.markdown(
                        f'<div class="qs-msg">{html.escape(st.session_state._record_answer)}</div>',
                        unsafe_allow_html=True,
                    )

    st.caption(
        "Sources: ERCOT interconnection queue (via gridstatus) · "
        "TCEQ Permit Search · LBNL Queued Up (CC BY 4.0)"
    )


if __name__ == "__main__":
    main()
