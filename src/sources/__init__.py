"""Data sources for QueueSense.

Each source module exposes ``fetch_*`` (live pull, saves a snapshot) and
normalizes into the unified record schema below, so everything downstream
(matching, staging, the map) works off one table.

Unified record columns (one row = one filing/record, NOT one project — the
same project appears once per source until matching links them):

| column        | meaning                                                    |
|---------------|------------------------------------------------------------|
| source        | "ercot" or "tceq"                                          |
| source_id     | the record's id in its source (Queue ID / permit number)   |
| project_name  | site/project name as the source spells it                  |
| company       | the company behind it (ERCOT Interconnecting Entity /      |
|               | TCEQ permit holder)                                        |
| county        | Texas county, Title Case for display                       |
| county_key    | county upper-cased — the cross-source join key             |
| lat / lon     | coordinates (TCEQ has real ones; ERCOT is NaN until a      |
|               | permit match or county-centroid fallback fills it)         |
| status        | the source's own status string                             |
| stage_signal  | the raw text used to infer how far along the project is    |
| capacity_mw   | plant size (ERCOT only)                                    |
| kind          | generation type (ERCOT) or NAICS description (TCEQ)        |
| record_date   | when this record entered the source                        |
| link_id       | secondary id for tracing back to the filing (RN number /   |
|               | project name)                                              |
"""

RECORD_COLUMNS: list[str] = [
    "source",
    "source_id",
    "project_name",
    "company",
    "county",
    "county_key",
    "lat",
    "lon",
    "status",
    "stage_signal",
    "capacity_mw",
    "kind",
    "record_date",
    "link_id",
]


def load_all(refresh: bool = False):
    """Return (records, freshness): the combined unified table + per-source stamps.

    ``refresh=True`` pulls both sources live (each falls back to its snapshot on
    failure). ``refresh=False`` serves snapshots when they exist and only goes
    live for a source with no snapshot yet.

    Returns:
        records: one DataFrame in the unified schema, both sources stacked.
        freshness: dict of source -> datetime|None (last successful pull).
    """
    import pandas as pd

    from src.sources import ercot, tceq

    frames = []
    freshness = {}
    for name, mod in (("ercot", ercot), ("tceq", tceq)):
        fetch = mod.fetch_queue if name == "ercot" else mod.fetch_air_permits
        if refresh or not mod.snapshot_path().exists():
            df = fetch(use_cache_on_error=True)
        else:
            df = mod.load_snapshot()
        # Defensive dtype pinning so concat never trips on an all-NA column
        # from an older snapshot (lat/lon/capacity must always be float).
        for col in ("lat", "lon", "capacity_mw"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        frames.append(df)
        freshness[name] = mod.last_updated()
    records = pd.concat(frames, ignore_index=True)[RECORD_COLUMNS]
    return records, freshness
