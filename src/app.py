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

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

# Make `from src import ...` work when launched via `streamlit run src/app.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config, explain, resolve, score_model, stage  # noqa: E402
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
DETAIL_PANEL_HEIGHT = 920
MAP_HEIGHT = 480

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
  /* Hero: brand | tight status cluster */
  div[data-testid="stHorizontalBlock"]:has(.qs-hero-mark) {{
    align-items: center !important;
    margin: 0.15rem 0 0.85rem 0 !important;
  }}
  .qs-status-cluster {{
    display: flex; flex-wrap: wrap; align-items: center; justify-content: flex-end;
    gap: 0.4rem;
  }}
  .qs-pill {{
    display: inline-flex; align-items: center; gap: 0.4rem;
    height: 2rem; padding: 0 0.75rem; border-radius: 999px;
    background: rgba(231, 235, 225, 0.95); color: {PALETTE['ink']};
    font-size: 0.82rem; font-weight: 600; border: 1px solid rgba(74, 83, 60, 0.12);
    white-space: nowrap; box-sizing: border-box; line-height: 1;
  }}
  .qs-pill strong {{ font-variant-numeric: tabular-nums; }}
  .qs-pill-live {{
    background: rgba(74, 83, 60, 0.12); border-color: rgba(74, 83, 60, 0.22);
  }}
  .qs-pill-link {{
    background: rgba(200, 142, 114, 0.16); border-color: rgba(200, 142, 114, 0.35);
  }}
  .qs-pill-link-on {{
    background: rgba(200, 142, 114, 0.32); border-color: rgba(200, 142, 114, 0.55);
  }}
  .qs-dot {{
    width: 0.5rem; height: 0.5rem; border-radius: 50%; flex: 0 0 auto;
    background: {PALETTE['olive']};
    box-shadow: 0 0 0 0 rgba(74, 83, 60, 0.55);
    animation: qs-pulse 1.8s ease-out infinite;
  }}
  /* Action pills — identical chrome for Live / filings / stitched / refresh */
  div[data-testid="stHorizontalBlock"]:has(.qs-hero-actions) {{
    align-items: center !important;
    gap: 0.4rem !important;
    margin: 0 !important;
  }}
  div[data-testid="stElementContainer"]:has(.qs-hero-actions) {{
    height: 0 !important; margin: 0 !important; padding: 0 !important;
  }}
  div[data-testid="stElementContainer"]:has(.qs-live-btn)
    [data-testid="stButton"] > button,
  div[data-testid="stElementContainer"]:has(.qs-filings-btn)
    [data-testid="stButton"] > button,
  div[data-testid="stElementContainer"]:has(.qs-stitched-btn)
    [data-testid="stButton"] > button,
  div[data-testid="stElementContainer"]:has(.qs-refresh-btn)
    [data-testid="stButton"] > button {{
    height: 2rem !important;
    min-height: 2rem !important;
    max-height: 2rem !important;
    border-radius: 999px !important;
    padding: 0 0.75rem !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    line-height: 1 !important;
    box-shadow: none !important;
    transform: none !important;
    white-space: nowrap !important;
    display: inline-flex !important;
    align-items: center !important;
    width: 100% !important;
    justify-content: center !important;
    opacity: 1 !important;
    cursor: default !important;
  }}
  div[data-testid="stElementContainer"]:has(.qs-live-btn)
    [data-testid="stButton"] > button {{
    background: rgba(74, 83, 60, 0.12) !important;
    border: 1px solid rgba(74, 83, 60, 0.22) !important;
    color: {PALETTE['ink']} !important;
  }}
  div[data-testid="stElementContainer"]:has(.qs-filings-btn)
    [data-testid="stButton"] > button,
  div[data-testid="stElementContainer"]:has(.qs-refresh-btn)
    [data-testid="stButton"] > button {{
    background: rgba(231, 235, 225, 0.95) !important;
    border: 1px solid rgba(74, 83, 60, 0.12) !important;
    color: {PALETTE['ink']} !important;
  }}
  div[data-testid="stElementContainer"]:has(.qs-stitched-btn)
    [data-testid="stButton"] > button {{
    background: rgba(200, 142, 114, 0.16) !important;
    border: 1px solid rgba(200, 142, 114, 0.35) !important;
    color: {PALETTE['ink']} !important;
    cursor: pointer !important;
  }}
  div[data-testid="stElementContainer"]:has(.qs-stitched-btn)
    [data-testid="stButton"] > button:hover {{
    background: rgba(200, 142, 114, 0.28) !important;
  }}
  div[data-testid="stElementContainer"]:has(.qs-stitched-btn.on)
    [data-testid="stButton"] > button {{
    background: rgba(200, 142, 114, 0.36) !important;
    border-color: rgba(200, 142, 114, 0.6) !important;
  }}
  div[data-testid="stElementContainer"]:has(.qs-refresh-btn)
    [data-testid="stButton"] > button {{
    cursor: pointer !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.01em;
    overflow: hidden !important;
  }}
  div[data-testid="stElementContainer"]:has(.qs-refresh-btn)
    [data-testid="stButton"] > button p,
  div[data-testid="stElementContainer"]:has(.qs-refresh-btn)
    [data-testid="stButton"] > button span {{
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
  }}
  div[data-testid="stElementContainer"]:has(.qs-refresh-btn)
    [data-testid="stButton"] > button:hover {{
    background: rgba(74, 83, 60, 0.12) !important;
  }}
  /* Filter row — top-align; equal control heights */
  div[data-testid="stTextInput"][class*="st-key-filter_search"] input,
  div[data-testid="stMultiSelect"][class*="st-key-filter_status"]
    [data-baseweb="select"] > div,
  div[data-testid="stMultiSelect"][class*="st-key-filter_sources"]
    [data-baseweb="select"] > div {{
    min-height: 2.5rem !important;
  }}
  /* Story rail — shared quiet card language */
  .qs-rail-label {{
    font-size: 0.68rem; letter-spacing: 0.1em; text-transform: uppercase;
    font-weight: 700; color: rgba(46, 51, 42, 0.4); margin: 0.55rem 0 0.35rem 0;
  }}
  .qs-rail-label:first-child {{ margin-top: 0; }}
  .qs-msg, .qs-link-card {{
    margin: 0 0 0.5rem 0; padding: 0.7rem 0.75rem; border-radius: 12px;
    background: rgba(231, 235, 225, 0.72); color: {PALETTE['ink']};
    border: 1px solid rgba(46, 51, 42, 0.06);
    animation: qs-fade-up 320ms ease both;
    line-height: 1.4;
  }}
  .qs-msg p, .qs-link-card p {{ margin: 0 0 0.25rem 0; }}
  .qs-msg p:last-child, .qs-link-card p:last-child {{ margin-bottom: 0; }}
  /* Shared text column: aligns with stage-step labels across cards */
  .qs-msg > p, .qs-link-card > p {{
    padding-left: calc(1.1rem + 0.5rem);
  }}
  .qs-msg-stage {{
    background: rgba(74, 83, 60, 0.07);
  }}
  .qs-link-card {{
    border-left: 3px solid rgba(200, 142, 114, 0.55);
  }}
  .qs-link-card .qs-kicker {{
    font-size: 0.68rem; letter-spacing: 0.08em; text-transform: uppercase;
    font-weight: 700; color: {PALETTE['terracotta']}; margin: 0 0 0.25rem 0;
  }}
  .qs-msg-meta {{ font-size: 0.82rem; opacity: 0.72; }}

  .qs-funnel {{
    display: flex; flex-direction: column; gap: 0; margin: 0;
  }}
  .qs-funnel-step {{
    display: grid; grid-template-columns: 1.1rem 1fr; gap: 0.5rem;
    align-items: start; position: relative; min-height: 1.85rem;
  }}
  .qs-funnel-step:not(:last-child)::before {{
    content: ""; position: absolute; left: 7px; top: 16px; bottom: -2px;
    width: 2px; background: rgba(46, 51, 42, 0.12);
  }}
  .qs-funnel-step.done:not(:last-child)::before {{
    background: {PALETTE['olive']};
  }}
  .qs-funnel-step.current:not(:last-child)::before {{
    background: linear-gradient({PALETTE['olive']}, rgba(46, 51, 42, 0.12));
  }}
  .qs-funnel-dot {{
    width: 1.1rem; height: 1.1rem; border-radius: 50%;
    border: 2px solid rgba(46, 51, 42, 0.18);
    background: {PALETTE['bg']};
    display: flex; align-items: center; justify-content: center;
    font-size: 0.58rem; font-weight: 700; color: rgba(46, 51, 42, 0.4);
    z-index: 1;
  }}
  .qs-funnel-step.done .qs-funnel-dot {{
    background: {PALETTE['olive']}; border-color: {PALETTE['olive']}; color: {PALETTE['bg']};
  }}
  .qs-funnel-step.current .qs-funnel-dot {{
    background: {PALETTE['terracotta']}; border-color: {PALETTE['terracotta']};
    color: {PALETTE['bg']};
    box-shadow: 0 0 0 3px rgba(200, 142, 114, 0.2);
  }}
  .qs-funnel-label {{
    font-size: 0.86rem; font-weight: 600; color: rgba(46, 51, 42, 0.45);
    padding-top: 0.05rem; line-height: 1.25;
  }}
  .qs-funnel-step.done .qs-funnel-label {{ color: rgba(46, 51, 42, 0.72); }}
  .qs-funnel-step.current .qs-funnel-label {{
    color: {PALETTE['ink']}; font-weight: 700;
  }}
  .qs-msg-stage .qs-msg-meta {{
    margin-top: 0.4rem; padding-left: calc(1.1rem + 0.5rem);
  }}

  /* Completion gauge — hover ring + drivers reveal */
  .qs-msg-score {{
    background: rgba(200, 142, 114, 0.1);
    border: 1px solid rgba(200, 142, 114, 0.2);
    padding: 0.85rem 0.75rem 0.7rem;
    text-align: center;
  }}
  .qs-msg-score > p {{ padding-left: 0 !important; }}
  .qs-gauge-wrap {{
    display: flex; flex-direction: column; align-items: center; gap: 0;
    margin: 0;
  }}
  .qs-gauge {{
    position: relative; width: 124px; height: 124px;
    transition: transform 320ms cubic-bezier(0.22, 1, 0.36, 1);
    cursor: help;
  }}
  .qs-msg-score:hover .qs-gauge {{
    transform: scale(1.07) rotate(-3deg);
  }}
  .qs-gauge svg {{ width: 100%; height: 100%; display: block; transform: rotate(-90deg); }}
  .qs-gauge-track {{
    fill: none; stroke: rgba(46, 51, 42, 0.1); stroke-width: 9;
    transition: stroke 280ms ease, stroke-width 280ms ease;
  }}
  .qs-msg-score:hover .qs-gauge-track {{
    stroke: rgba(46, 51, 42, 0.16); stroke-width: 10;
  }}
  .qs-gauge-fill {{
    fill: none; stroke: {PALETTE['terracotta']}; stroke-width: 9;
    stroke-linecap: round;
    transition: stroke-width 280ms ease, filter 280ms ease;
    animation: qs-gauge-draw 900ms cubic-bezier(0.22, 1, 0.36, 1) both;
  }}
  .qs-msg-score:hover .qs-gauge-fill {{
    stroke-width: 12;
    filter: drop-shadow(0 0 7px rgba(200, 142, 114, 0.45));
  }}
  .qs-gauge-fill.tier-top {{ stroke: {PALETTE['olive']}; }}
  .qs-msg-score:hover .qs-gauge-fill.tier-top {{
    filter: drop-shadow(0 0 7px rgba(74, 83, 60, 0.4));
  }}
  .qs-gauge-fill.tier-likely {{ stroke: {PALETTE['sage']}; }}
  .qs-gauge-fill.tier-watch {{ stroke: {PALETTE['terracotta']}; }}
  .qs-gauge-fill.tier-risk {{ stroke: #A65D4A; }}
  .qs-gauge-center {{
    position: absolute; inset: 0; display: flex; flex-direction: column;
    align-items: center; justify-content: center; text-align: center;
    pointer-events: none;
    transition: transform 280ms ease;
  }}
  .qs-msg-score:hover .qs-gauge-center {{ transform: scale(1.05); }}
  .qs-gauge-pct {{
    font-family: Fraunces, Georgia, serif; font-size: 1.7rem; font-weight: 700;
    letter-spacing: -0.03em; line-height: 1; color: {PALETTE['ink']}; margin: 0;
  }}
  .qs-gauge-caption {{
    margin: 0.4rem 0 0 0; font-size: 0.68rem; font-weight: 700;
    letter-spacing: 0.08em; text-transform: uppercase;
    color: rgba(46, 51, 42, 0.5); white-space: nowrap;
  }}
  .qs-drivers-panel {{
    width: 100%; max-height: 0; opacity: 0; overflow: hidden;
    margin-top: 0; text-align: left;
    transition: max-height 320ms ease, opacity 240ms ease, margin-top 240ms ease;
  }}
  .qs-msg-score:hover .qs-drivers-panel {{
    max-height: 14rem; opacity: 1; margin-top: 0.65rem;
  }}
  .qs-drivers-kicker {{
    font-size: 0.68rem; letter-spacing: 0.08em; text-transform: uppercase;
    font-weight: 700; color: rgba(46, 51, 42, 0.45);
    margin: 0 0 0.25rem 0; padding-left: 0 !important; text-align: left;
  }}
  .qs-drivers {{
    margin: 0; padding: 0; list-style: none; font-size: 0.82rem;
  }}
  .qs-drivers li {{
    display: flex; justify-content: space-between; gap: 0.75rem;
    padding: 0.18rem 0; border-bottom: 1px solid rgba(46, 51, 42, 0.06);
  }}
  .qs-drivers li:last-child {{ border-bottom: 0; }}
  .qs-drv-pos {{ color: {PALETTE['olive']}; font-variant-numeric: tabular-nums; }}
  .qs-drv-neg {{ color: {PALETTE['terracotta']}; font-variant-numeric: tabular-nums; }}

  .qs-drawer-bar {{
    display: flex; align-items: center; justify-content: space-between;
    gap: 0.75rem; margin: 0.35rem 0 0.15rem 0;
    padding-top: 0.55rem; border-top: 1px solid rgba(46, 51, 42, 0.08);
  }}
  .qs-drawer-bar .qs-drawer-title {{
    font-size: 0.8rem; font-weight: 700; letter-spacing: 0.04em;
    text-transform: uppercase; color: rgba(46, 51, 42, 0.55); margin: 0;
  }}
  /* Filings header row — chevron label + info on one line */
  div[data-testid="stHorizontalBlock"]:has(.qs-filings-marker) {{
    margin-top: 0.45rem;
    padding-top: 0.55rem;
    border-top: 1px solid rgba(46, 51, 42, 0.08);
    align-items: center;
  }}
  div[data-testid="stElementContainer"]:has(.qs-filings-marker) {{
    height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
  }}
  div[data-testid="stElementContainer"]:has(.qs-filings-marker)
    + div[data-testid="stElementContainer"]
    [data-testid="stButton"] > button {{
    background: transparent !important;
    border: none !important;
    color: rgba(46, 51, 42, 0.55) !important;
    font-size: 0.8rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
    padding: 0.2rem 0.15rem !important;
    box-shadow: none !important;
    transform: none !important;
  }}
  div[data-testid="stElementContainer"]:has(.qs-filings-marker)
    + div[data-testid="stElementContainer"]
    [data-testid="stButton"] > button:hover {{
    color: {PALETTE['ink']} !important;
    background: transparent !important;
  }}
  div[data-testid="stElementContainer"]:has(.qs-filings-marker)
    + div[data-testid="stElementContainer"]
    [data-testid="stButton"] > button svg {{
    fill: {PALETTE['olive']} !important;
  }}
  /* Overlay "Reset search" on the bottom-left of the map */
  div[data-testid="stElementContainer"]:has(.qs-map-reset) {{
    height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    overflow: visible !important;
  }}
  div[data-testid="stElementContainer"]:has(.qs-map-reset)
    + div[data-testid="stElementContainer"] {{
    margin-top: -3.15rem !important;
    margin-bottom: 2.4rem !important;
    margin-left: 0.65rem !important;
    position: relative;
    z-index: 30;
    width: fit-content !important;
  }}
  div[data-testid="stElementContainer"]:has(.qs-map-reset)
    + div[data-testid="stElementContainer"] [data-testid="stButton"] > button {{
    background: rgba(247, 244, 235, 0.94) !important;
    border: 1px solid rgba(74, 83, 60, 0.28) !important;
    color: {PALETTE['olive']} !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    padding: 0.28rem 0.75rem !important;
    box-shadow: 0 1px 4px rgba(46, 51, 42, 0.12);
  }}
  div[data-testid="stElementContainer"]:has(.qs-map-reset)
    + div[data-testid="stElementContainer"] [data-testid="stButton"] > button:hover {{
    background: {PALETTE['olive']} !important;
    color: {PALETTE['bg']} !important;
  }}

  [data-testid="stPopover"] {{ display: flex; justify-content: flex-end; }}
  [data-testid="stPopover"] > button {{
    padding: 0.2rem 0.45rem !important;
    min-height: 1.85rem !important;
    border-radius: 999px !important;
    border: 1px solid rgba(74, 83, 60, 0.28) !important;
    color: {PALETTE['olive']} !important;
    background: rgba(231, 235, 225, 0.7) !important;
  }}
  [data-testid="stPopover"] > button:hover {{
    background: {PALETTE['olive']} !important;
    color: {PALETTE['bg']} !important;
    border-color: {PALETTE['olive']} !important;
  }}
  [data-testid="stPopover"] > button svg {{
    fill: currentColor !important;
  }}
  [data-testid="stToolbar"], #MainMenu, footer,
  header[data-testid="stHeader"] {{ visibility: hidden; height: 0; }}
  div[data-testid="stDecoration"] {{ display: none; }}

  @keyframes qs-pulse {{
    0% {{ box-shadow: 0 0 0 0 rgba(74, 83, 60, 0.45); }}
    70% {{ box-shadow: 0 0 0 10px rgba(74, 83, 60, 0); }}
    100% {{ box-shadow: 0 0 0 0 rgba(74, 83, 60, 0); }}
  }}
  @keyframes qs-gauge-draw {{
    from {{ stroke-dashoffset: 264; }}
  }}
  @keyframes qs-fade-up {{
    from {{ opacity: 0; transform: translateY(8px); }}
    to {{ opacity: 1; transform: translateY(0); }}
  }}
</style>
"""


def _inject_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def _format_freshness(fresh: dict) -> tuple[str, str]:
    """Compact pill label + full detail for the refresh control tooltip."""
    detail_parts = []
    times = []
    for name, ts in fresh.items():
        if ts is not None:
            detail_parts.append(f"{name.upper()} {ts:%b %d %H:%M}")
            times.append(ts)
        else:
            detail_parts.append(f"{name.upper()} never")
    detail = " · ".join(detail_parts) if detail_parts else "No freshness data"
    if not times:
        return "Refresh data", detail
    # Same minute across sources → one clean line; else short per-source times.
    keyed = {(t.year, t.month, t.day, t.hour, t.minute) for t in times}
    if len(keyed) == 1:
        label = f"Updated {times[0]:%b %d · %H:%M}"
    else:
        label = " · ".join(
            f"{name.upper()} {ts:%H:%M}" if ts is not None else f"{name.upper()} —"
            for name, ts in fresh.items()
        )
    return label, detail


def _hero(n_records: int, n_links: int, stamps: str, stamps_help: str) -> bool:
    """Brand left; equal-height status pills right (in line with QueueScore)."""
    stitched_on = bool(st.session_state.get("stitched_only", False))
    if n_links == 0:
        stitched_on = False
        st.session_state.stitched_only = False

    brand, pills = st.columns([1.15, 1], gap="large", vertical_alignment="center")
    with brand:
        st.markdown(
            (
                '<div class="qs-hero-mark"></div>'
                '<p class="qs-brand">QueueScore</p>'
                '<p class="qs-tagline">'
                "Live Texas power origination — ERCOT queue + TCEQ permits"
                "</p>"
            ),
            unsafe_allow_html=True,
        )
    with pills:
        # One row of identical-height controls (buttons share the same chrome).
        st.markdown('<div class="qs-hero-actions"></div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns([0.85, 1.1, 1.15, 2.1], gap="small")
        with c1:
            st.markdown('<div class="qs-live-btn"></div>', unsafe_allow_html=True)
            st.button("Live", key="pill_live", disabled=True)
        with c2:
            st.markdown('<div class="qs-filings-btn"></div>', unsafe_allow_html=True)
            st.button(f"{n_records:,} filings", key="pill_filings", disabled=True)
        with c3:
            on_cls = " on" if stitched_on else ""
            st.markdown(
                f'<div class="qs-stitched-btn{on_cls}"></div>',
                unsafe_allow_html=True,
            )
            if st.button(
                f"{n_links} stitched",
                key="btn_stitched_lens",
                disabled=n_links == 0,
                help="Show only filings linked across ERCOT ↔ TCEQ. Click again to clear.",
            ):
                st.session_state.stitched_only = not stitched_on
                st.rerun()
        with c4:
            st.markdown('<div class="qs-refresh-btn"></div>', unsafe_allow_html=True)
            if st.button(
                stamps,
                key="btn_refresh_stamps",
                icon=":material/refresh:",
                help=f"Last updated — {stamps_help}. Click to pull fresh ERCOT + TCEQ data.",
            ):
                st.session_state._do_refresh = True
                st.rerun()

    return bool(st.session_state.get("stitched_only", False))


def _schema_help() -> None:
    """Info popover: what a row is, where fields come from, and the stage ladder."""
    with st.popover(
        "",
        icon=":material/info:",
        help="About this data",
        use_container_width=False,
        type="tertiary",
    ):
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
    """Render the Claude read as a thematic callout.

    HTML-escaped for safety, then minimal markdown (**bold**) re-applied so the
    brief's section labels render properly.
    """
    import re

    paras = []
    for p in text.split("\n\n"):
        if not p.strip():
            continue
        safe = html.escape(p.strip())
        safe = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", safe)
        paras.append(safe)
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


@st.cache_resource(show_spinner=False)
def _load_score_bundle():
    """Load the ERCOT completion model once per process (or None if missing)."""
    if not config.MODEL_BUNDLE_PATH.exists():
        return None
    try:
        return score_model.load_bundle()
    except Exception as exc:  # noqa: BLE001
        # Common on macOS without `brew install libomp`.
        st.session_state["_score_load_error"] = str(exc)
        return None


@st.cache_data(show_spinner="Scoring ERCOT queue…")
def _score_ercot_snapshot(_fingerprint: str, ercot: pd.DataFrame) -> pd.DataFrame:
    """Score the full ERCOT slice (congestion needs the whole snapshot)."""
    bundle = _load_score_bundle()
    if bundle is None:
        return pd.DataFrame()
    return score_model.score_queue(ercot, bundle)


def _ercot_fingerprint(ercot: pd.DataFrame) -> str:
    cols = ["source_id", "capacity_mw", "kind", "county", "status"]
    return pd.util.hash_pandas_object(ercot[cols], index=False).sum().__format__("x")


def _attach_model_scores(records: pd.DataFrame) -> pd.DataFrame:
    out = records.copy()
    out["completion_probability"] = np.nan
    out["ia_tier"] = ""
    ercot = out[out["source"] == "ercot"]
    if ercot.empty or _load_score_bundle() is None:
        return out
    scored = _score_ercot_snapshot(_ercot_fingerprint(ercot), ercot.reset_index(drop=True))
    if scored.empty:
        return out
    by_id = scored.set_index(scored["q_id"].astype(str))[
        ["completion_probability", "ia_tier"]
    ]
    sid = ercot["source_id"].astype(str)
    out.loc[ercot.index, "completion_probability"] = sid.map(by_id["completion_probability"]).values
    out.loc[ercot.index, "ia_tier"] = sid.map(by_id["ia_tier"]).fillna("").values
    # Keep scored frame for on-demand SHAP drivers.
    st.session_state["_ercot_scored"] = scored
    return out


def _drivers_for(row: pd.Series) -> list[tuple[str, float]]:
    """SHAP drivers for one ERCOT row (cached per source_id in session)."""
    if row.get("source") != "ercot":
        return []
    sid = str(row["source_id"])
    cache = st.session_state.setdefault("_shap_cache", {})
    if sid in cache:
        return cache[sid]
    bundle = _load_score_bundle()
    scored = st.session_state.get("_ercot_scored")
    if bundle is None or scored is None or getattr(scored, "empty", True):
        return []
    hit = scored[scored["q_id"].astype(str) == sid]
    if hit.empty:
        return []
    drivers = score_model.explain_drivers(hit, bundle, top_k=5)[0]["top_drivers"]
    cache[sid] = drivers
    return drivers


def _render_score_card(row: pd.Series) -> None:
    """Compact P(IA) gauge; hover reveals SHAP drivers. ERCOT only."""
    if row.get("source") != "ercot":
        return
    p = row.get("completion_probability")
    if p is None or (isinstance(p, float) and pd.isna(p)):
        err = st.session_state.get("_score_load_error")
        if err:
            st.caption(f"Completion model unavailable ({err[:120]})")
        return

    prob = float(p)
    tier = str(row.get("ia_tier") or score_model.ia_tier(prob))
    pct_label = f"{prob:.0%}"
    c = 2 * 3.14159265 * 42
    offset = c * (1.0 - max(0.0, min(1.0, prob)))
    tier_class = {
        "Top": "tier-top",
        "Likely": "tier-likely",
        "Watch": "tier-watch",
        "At-risk": "tier-risk",
    }.get(tier, "tier-watch")

    drivers_html = ""
    try:
        drivers = _drivers_for(row)
    except Exception:  # noqa: BLE001
        drivers = []
    if drivers:
        items = []
        for feat, val in drivers:
            label = html.escape(score_model.FEATURE_LABELS.get(feat, feat))
            cls = "qs-drv-pos" if val >= 0 else "qs-drv-neg"
            sign = "+" if val >= 0 else ""
            items.append(
                f'<li><span>{label}</span>'
                f'<span class="{cls}">{sign}{val:.2f}</span></li>'
            )
        drivers_html = (
            f'<div class="qs-drivers-panel">'
            f'<p class="qs-drivers-kicker">Why this score</p>'
            f'<ul class="qs-drivers">{"".join(items)}</ul>'
            f'</div>'
        )

    # Keep HTML flush-left — indented lines become markdown code blocks.
    st.markdown(
        (
            f'<div class="qs-msg qs-msg-score">'
            f'<div class="qs-gauge-wrap">'
            f'<div class="qs-gauge">'
            f'<svg viewBox="0 0 100 100" aria-hidden="true">'
            f'<circle class="qs-gauge-track" cx="50" cy="50" r="42"></circle>'
            f'<circle class="qs-gauge-fill {tier_class}" cx="50" cy="50" r="42" '
            f'style="stroke-dasharray:{c:.2f};stroke-dashoffset:{offset:.2f}"></circle>'
            f'</svg>'
            f'<div class="qs-gauge-center">'
            f'<p class="qs-gauge-pct">{html.escape(pct_label)}</p>'
            f'</div></div>'
            f'<p class="qs-gauge-caption">Completion chance</p>'
            f'{drivers_html}'
            f'</div></div>'
        ),
        unsafe_allow_html=True,
    )


def _render_stage_ladder(row: pd.Series) -> None:
    """Visual funnel showing where this filing sits on the stage ladder."""
    source = str(row.get("source") or "")
    try:
        rank = int(row.get("stage_rank", 0))
    except (TypeError, ValueError):
        rank = 0

    if source == "ercot":
        steps = [
            (0, "Early planning"),
            (1, "Engineering studies"),
            (2, "Studies complete"),
            (3, "Grid agreement signed"),
        ]
    else:
        steps = [
            (0, "Status unclear"),
            (1, "Application filed"),
            (2, "Permit issued"),
        ]

    items = []
    last_rank = steps[-1][0]
    for step_rank, label in steps:
        if step_rank < rank:
            state, mark = "done", "✓"
        elif step_rank == rank:
            # Terminal stage (e.g. IA signed) reads as complete, not "step 4".
            state = "current"
            mark = "✓" if step_rank == last_rank else str(step_rank + 1)
        else:
            state, mark = "", str(step_rank + 1)
        items.append(
            f'<div class="qs-funnel-step {state}">'
            f'<div class="qs-funnel-dot">{mark}</div>'
            f'<div class="qs-funnel-label">{html.escape(label)}</div>'
            f'</div>'
        )

    conf = html.escape(str(row.get("stage_confidence") or ""))
    # Keep evidence short — full detail lives in the brief if needed.
    signal = str(row.get("stage_signal") or row.get("status") or "").strip()
    if len(signal) > 42:
        signal = signal[:40] + "…"
    signal = html.escape(signal)
    st.markdown(
        (
            f'<div class="qs-msg qs-msg-stage">'
            f'<div class="qs-funnel">{"".join(items)}</div>'
            f'<p class="qs-msg-meta">'
            f'{conf} confidence'
            f'{f" · {signal}" if signal else ""}</p>'
            f'</div>'
        ),
        unsafe_allow_html=True,
    )


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

    # Panel 1: load (refresh is the hero timestamp pill)
    refresh = bool(st.session_state.pop("_do_refresh", False))
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

    # Completion model: P(reach signed IA) for ERCOT rows (LBNL-trained XGB).
    records = _attach_model_scores(records)

    stamps, stamps_detail = _format_freshness(fresh)
    stitched_only = _hero(len(records), n_links, stamps, stamps_detail)

    # Filters — top-aligned so search matches status dropdown baseline.
    statuses_present = set(records["status"].dropna().astype(str))
    status_options = [s for s in STATUS_COLORS if s in statuses_present] + sorted(
        statuses_present - set(STATUS_COLORS)
    )
    fcol1, fcol2, fcol3, fcol4 = st.columns(
        [2.8, 1.8, 1.15, 1.6], gap="small", vertical_alignment="top"
    )
    search = fcol1.text_input(
        "Search",
        placeholder="Company, project, or county…",
        label_visibility="collapsed",
        key="filter_search",
    )
    status_pick = fcol2.multiselect(
        "Status",
        status_options,
        default=[],
        placeholder="All statuses",
        help="Leave empty for all. Add statuses to narrow; X removes a chip.",
        label_visibility="collapsed",
        key="filter_status",
    )
    gas_focus = fcol3.toggle("Gas-to-power", value=False, key="filter_gas")
    source_pick = fcol4.multiselect(
        "Sources",
        ["ercot", "tceq"],
        default=["ercot", "tceq"],
        label_visibility="collapsed",
        key="filter_sources",
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

    # Shared focus: map pin and filings table write record_pick.
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

            map_h = MAP_HEIGHT
            pin_focus = bool(st.session_state.get("_map_table_focus"))

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
                if pin_focus:
                    st.markdown('<div class="qs-map-reset"></div>', unsafe_allow_html=True)
                    if st.button(
                        "Reset search",
                        key="btn_reset_map_search",
                        help="Clear the map pin filter and show all filings again.",
                    ):
                        st.session_state._map_table_focus = False
                        st.session_state._last_map_sel = None
                        st.rerun()
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
                                st.session_state._map_table_focus = True
                                st.session_state._last_table_sel = None
                                st.rerun()

            # Filings — when a map pin is selected, show only that row.
            map_focus = bool(st.session_state.get("_map_table_focus"))
            if map_focus and focus is not None:
                table_df = view.loc[[focus.name]].reset_index(drop=True)
            else:
                table_df = view
                st.session_state._map_table_focus = False

            filings_label = (
                f"Filings · selected pin"
                if map_focus and len(table_df) == 1
                else f"Filings · {len(table_df):,}"
            )
            if "filings_open" not in st.session_state:
                st.session_state.filings_open = True

            head_l, head_r = st.columns([14, 1], vertical_alignment="center")
            with head_l:
                st.markdown('<div class="qs-filings-marker"></div>', unsafe_allow_html=True)
                chevron = (
                    ":material/expand_more:"
                    if st.session_state.filings_open
                    else ":material/chevron_right:"
                )
                if st.button(
                    filings_label,
                    key="btn_filings_toggle",
                    type="tertiary",
                    icon=chevron,
                    help="Show or hide the filings table",
                ):
                    st.session_state.filings_open = not st.session_state.filings_open
                    st.rerun()
            with head_r:
                _schema_help()

            if st.session_state.filings_open:
                table = table_df[
                    ["project_name", "capacity_mw", "status", "company", "kind",
                     "source", "source_id", "county", "stage",
                     "completion_probability", "record_date"]
                ]
                event = st.dataframe(
                    table,
                    width="stretch",
                    hide_index=True,
                    height=220 if not map_focus else 120,
                    on_select="rerun",
                    selection_mode="single-row",
                    key="records_table",
                    column_config={
                        "project_name": st.column_config.TextColumn("Project"),
                        "capacity_mw": st.column_config.NumberColumn(
                            "MW", format="%.0f", width="small"
                        ),
                        "status": st.column_config.TextColumn("Status", width="small"),
                        "company": st.column_config.TextColumn("Company"),
                        "kind": st.column_config.TextColumn("Type", width="small"),
                        "source": st.column_config.TextColumn("Src", width="small"),
                        "source_id": st.column_config.TextColumn("ID", width="small"),
                        "county": st.column_config.TextColumn("County", width="small"),
                        "stage": st.column_config.TextColumn("Stage"),
                        "completion_probability": st.column_config.NumberColumn(
                            "P(IA)",
                            help="Chance of signed interconnection agreement (ERCOT).",
                            format="%.0%",
                            width="small",
                        ),
                        "record_date": st.column_config.DateColumn("Filed", width="small"),
                    },
                )
                selected_rows = event.selection.rows if event and event.selection else []
                if selected_rows:
                    tidx = int(selected_rows[0])
                    if 0 <= tidx < len(table_df):
                        picked = _label(table_df.iloc[tidx])
                        if st.session_state.get("_last_table_sel") != picked:
                            st.session_state._last_table_sel = picked
                            st.session_state.record_pick = picked
                            # Manual table pick leaves the full list (unless already pin-focused).
                            if not map_focus:
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
                # Selection comes from the map / filings table — no dropdown.
                pick = st.session_state.get("record_pick")
                if pick not in labels:
                    pick = labels[0]
                    st.session_state.record_pick = pick
                row = view.iloc[labels.index(pick)]
                record_key = f"{row['source']}:{row['source_id']}"
                if st.session_state.get("_llm_record_key") != record_key:
                    st.session_state._llm_record_key = record_key
                    st.session_state.pop("_origination_read", None)
                    st.session_state.pop("_record_answer", None)

                # Score (ERCOT) → stage → identity → stitched → ask
                _render_score_card(row)
                _render_stage_ladder(row)

                cap = (
                    f"{row['capacity_mw']:.0f} MW · "
                    if pd.notna(row["capacity_mw"]) else ""
                )
                filed = (
                    f"{pd.to_datetime(row['record_date']):%Y-%m-%d}"
                    if pd.notna(row["record_date"]) else "—"
                )
                st.markdown(
                    (
                        f'<div class="qs-msg">'
                        f'<p style="font-weight:700;margin:0 0 0.2rem 0;">'
                        f'{html.escape(str(row["project_name"] or "(unnamed)"))}</p>'
                        f'<p class="qs-msg-meta">'
                        f'{html.escape(str(row["company"] or "—"))}'
                        f' · {html.escape(str(row["county"] or ""))} County</p>'
                        f'<p class="qs-msg-meta">'
                        f'{html.escape(cap)}{html.escape(str(row["kind"] or ""))}'
                        f' · {html.escape(str(row["status"] or "—"))}'
                        f' · {filed}</p>'
                        f'</div>'
                    ),
                    unsafe_allow_html=True,
                )

                linked_rec = None
                if row.get("match_id"):
                    other = records[
                        (records["source"] != row["source"])
                        & (records["source_id"] == row["match_id"])
                    ]
                    if len(other):
                        o = other.iloc[0]
                        linked_rec = o.to_dict()
                        reason = str(row["match_reason"] or "name match in same county")
                        if len(reason) > 90:
                            reason = reason[:88] + "…"
                        st.markdown(
                            (
                                f'<div class="qs-link-card">'
                                f'<p class="qs-kicker">Stitched</p>'
                                f'<p style="font-weight:700;margin:0 0 0.15rem 0;">'
                                f'{html.escape(str(o["source"]).upper())} '
                                f'{html.escape(str(o["source_id"]))}'
                                f' · {html.escape(str(o["project_name"] or o["company"] or ""))}'
                                f'</p>'
                                f'<p class="qs-msg-meta">'
                                f'{html.escape(str(o["company"] or ""))} · '
                                f'{html.escape(str(o["status"] or ""))}</p>'
                                f'<p class="qs-msg-meta">{html.escape(reason)}</p>'
                                f'</div>'
                            ),
                            unsafe_allow_html=True,
                        )

                st.markdown(
                    '<p class="qs-rail-label">Ask</p>',
                    unsafe_allow_html=True,
                )
                if st.button("Generate brief", key="btn_origination_read"):
                    # Context beyond the record itself: the company's other
                    # filings and county activity — all computed locally.
                    my_company = resolve.normalize_name(row["company"])
                    if my_company:
                        same_co = records[
                            (records["company"].map(resolve.normalize_name) == my_company)
                            & (records["source_id"] != row["source_id"])
                        ]
                    else:
                        same_co = records.iloc[0:0]
                    county_rows = records[
                        (records["county_key"] == row["county_key"])
                        & (records["source_id"] != row["source_id"])
                    ]
                    county_ctx = (
                        f"{len(county_rows)} other filings "
                        f"({county_rows['source'].value_counts().to_dict()}); e.g. "
                        + "; ".join(
                            (county_rows["project_name"].replace("", pd.NA).dropna().head(3))
                        )
                    ) if len(county_rows) else ""
                    with st.spinner("Building brief…"):
                        st.session_state._origination_read = explain.generate_brief(
                            row.to_dict(),
                            linked_rec,
                            same_co.to_dict("records"),
                            county_ctx,
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
