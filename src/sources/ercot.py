"""ERCOT interconnection-queue ingestion for QueueSense.

Wraps ``gridstatus.Ercot().get_interconnection_queue()`` (same live source the
QueueSense skeleton used) but keeps the columns Radar needs that the old
mapping dropped: Project Name, Interconnecting Entity (the company — our match
key), and GIM Study Phase / IA Signed (the stage signals).
"""

from __future__ import annotations

import datetime as dt
import json

import pandas as pd

from src import config
from src.sources import RECORD_COLUMNS

SNAPSHOT_NAME = "ercot_radar.parquet"
META_NAME = "ercot_radar.meta.json"


def _normalize(raw: pd.DataFrame) -> pd.DataFrame:
    """Map the raw 35-column gridstatus frame onto the unified record schema."""
    county_raw = raw["County"].fillna("").astype(str).str.strip()
    # Stage signal: study phase + IA date presence, e.g.
    # "SS Completed, FIS Started, No IA" / "... IA (signed 2020-08-07)"
    ia = pd.to_datetime(raw["IA Signed"], errors="coerce")
    stage = raw["GIM Study Phase"].fillna("").astype(str).str.strip()
    stage = stage.where(ia.isna(), stage + " (IA signed " + ia.dt.strftime("%Y-%m-%d") + ")")

    df = pd.DataFrame(
        {
            "source": "ercot",
            "source_id": raw["Queue ID"].astype(str).str.strip(),
            "project_name": raw["Project Name"].fillna("").astype(str).str.strip(),
            "company": raw["Interconnecting Entity"].fillna("").astype(str).str.strip(),
            "county": county_raw.str.title(),
            "county_key": county_raw.str.upper(),
            # ERCOT publishes no coordinates; filled by a permit match or county
            # centroid downstream. Typed float so concat with TCEQ stays clean.
            "lat": pd.Series(float("nan"), index=raw.index, dtype="float64"),
            "lon": pd.Series(float("nan"), index=raw.index, dtype="float64"),
            "status": raw["Status"].fillna("").astype(str).str.strip(),
            "stage_signal": stage,
            "capacity_mw": pd.to_numeric(raw["Capacity (MW)"], errors="coerce"),
            "kind": raw["Generation Type"].fillna("").astype(str).str.strip(),
            "record_date": pd.to_datetime(raw["Queue Date"], errors="coerce"),
            "link_id": raw["Project Name"].fillna("").astype(str).str.strip(),
        },
        columns=RECORD_COLUMNS,
    )
    return df.drop_duplicates(subset=["source_id"]).reset_index(drop=True)


def fetch_queue(use_cache_on_error: bool = True) -> pd.DataFrame:
    """Pull the live ERCOT queue; fall back to the snapshot offline."""
    try:
        import gridstatus  # lazy so offline runs don't need it

        raw = gridstatus.Ercot().get_interconnection_queue()
        df = _normalize(raw)
        save_snapshot(df)
        return df
    except Exception as exc:  # noqa: BLE001 - any failure -> offline fallback
        if use_cache_on_error and snapshot_path().exists():
            return load_snapshot()
        raise RuntimeError(
            "ERCOT live pull failed and no snapshot exists. "
            "Run once online to seed the cache."
        ) from exc


# --------------------------------------------------------------------------- #
# Snapshot cache + freshness
# --------------------------------------------------------------------------- #
def snapshot_path():
    return config.SNAPSHOT_DIR / SNAPSHOT_NAME


def _meta_path():
    return config.SNAPSHOT_DIR / META_NAME


def save_snapshot(df: pd.DataFrame) -> None:
    config.SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(snapshot_path(), index=False)
    _meta_path().write_text(
        json.dumps({"fetched_at": dt.datetime.now().isoformat(), "rows": len(df)})
    )


def load_snapshot() -> pd.DataFrame:
    return pd.read_parquet(snapshot_path())


def last_updated() -> dt.datetime | None:
    """When the snapshot was last refreshed (None if never)."""
    if not _meta_path().exists():
        return None
    return dt.datetime.fromisoformat(json.loads(_meta_path().read_text())["fetched_at"])
