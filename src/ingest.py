"""Data ingestion: LBNL historical training data + live ERCOT queue.

Two responsibilities:
  * ``load_lbnl`` — parse the LBNL "Queued Up" xlsx into the internal schema
    (stub today; real parsing is day-of once the codebook is open).
  * ``fetch_ercot_queue`` — pull the live ERCOT interconnection queue via
    gridstatus, with a real snapshot cache so the app runs fully offline.

The snapshot cache is implemented for real: the first successful live pull is
saved to ``data/snapshots``; later calls (or any run with no network) fall back
to the cached parquet.
"""

from __future__ import annotations

import pandas as pd

from . import config


# --------------------------------------------------------------------------- #
# LBNL historical training data
# --------------------------------------------------------------------------- #
def load_lbnl(path: str) -> pd.DataFrame:
    """Load the LBNL "Queued Up" dataset and map it to the internal schema.

    Args:
        path: path to the LBNL xlsx under ``data/raw``.

    Returns:
        A frame with columns == ``config.FEATURE_COLUMNS`` plus the outcome
        column used to build the training label.

    Stub behavior: returns an empty, correctly-columned frame. Day-of, this
    reads the xlsx with ``pd.read_excel``, renames via ``config.LBNL_TO_MODEL``,
    and derives the target per ``config.TARGET_DEFINITION``.
    """
    # TODO(day-of): pd.read_excel(path); df.rename(columns=config.LBNL_TO_MODEL);
    # coerce dtypes; derive target. Returning the typed empty frame keeps the
    # training pipeline importable and testable now.
    return pd.DataFrame(columns=config.FEATURE_COLUMNS)


# --------------------------------------------------------------------------- #
# Live ERCOT queue + offline cache
# --------------------------------------------------------------------------- #
def fetch_ercot_queue(use_cache_on_error: bool = True) -> pd.DataFrame:
    """Fetch the live ERCOT interconnection queue, mapped to the internal schema.

    Wraps ``gridstatus.Ercot().get_interconnection_queue()``. On success the
    result is cached via ``save_snapshot``. If the live pull fails (no network,
    gridstatus error) and ``use_cache_on_error`` is True, falls back to the last
    saved snapshot so the app keeps working offline.

    Returns:
        A frame with columns == ``config.FEATURE_COLUMNS``.

    Raises:
        RuntimeError: if the live pull fails and no cached snapshot exists.
    """
    try:
        import gridstatus  # imported lazily so offline runs don't need it

        raw = gridstatus.Ercot().get_interconnection_queue()
        mapped = _map_gridstatus(raw)
        save_snapshot(mapped)
        return mapped
    except Exception as exc:  # noqa: BLE001 - any failure -> offline fallback
        if use_cache_on_error and snapshot_exists():
            return load_snapshot()
        raise RuntimeError(
            "Live ERCOT pull failed and no cached snapshot is available. "
            "Run once with network access to seed the cache."
        ) from exc


def _map_gridstatus(raw: pd.DataFrame) -> pd.DataFrame:
    """Rename gridstatus columns to the internal schema and keep only those."""
    mapped = raw.rename(columns=config.GRIDSTATUS_TO_MODEL)
    keep = [c for c in config.FEATURE_COLUMNS if c in mapped.columns]
    return mapped[keep].copy()


# --------------------------------------------------------------------------- #
# Snapshot cache (implemented for real)
# --------------------------------------------------------------------------- #
def _snapshot_path():
    return config.SNAPSHOT_DIR / config.ERCOT_SNAPSHOT_NAME


def snapshot_exists() -> bool:
    """True if a cached ERCOT snapshot is on disk."""
    return _snapshot_path().exists()


def save_snapshot(df: pd.DataFrame) -> None:
    """Persist ``df`` as the current ERCOT snapshot (parquet)."""
    config.SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(_snapshot_path(), index=False)


def load_snapshot() -> pd.DataFrame:
    """Load the cached ERCOT snapshot.

    Raises:
        FileNotFoundError: if no snapshot has been saved yet.
    """
    path = _snapshot_path()
    if not path.exists():
        raise FileNotFoundError(f"No snapshot at {path}. Run fetch_ercot_queue() online first.")
    return pd.read_parquet(path)
